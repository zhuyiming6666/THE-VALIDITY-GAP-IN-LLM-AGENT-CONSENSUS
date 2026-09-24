"""Recompute debate gains after averaging all repeats within each task.

This offline submission check preserves the archived results and the existing
analysis. It uses the same benchmark scoring, unanimity definition, and
plurality tie-break as recompute_debate.py, while retaining every repetition
instead of selecting the last record for each task.

Run: python3 code/submission_debate_check.py (from v3, or any directory).
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

from collision import task_bootstrap
from recompute_debate import analyse_debate
from recompute_e1 import load_tasks


V3 = Path(__file__).resolve().parent.parent
SOURCE = V3 / "data/interaction/e3_benign.json"
DESTINATION = V3 / "results/submission_debate_check.json"
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260911


def interval(values):
    return task_bootstrap(values, B=BOOTSTRAP_REPLICATES,
                          seed=BOOTSTRAP_SEED, alpha=0.05)


def recompute():
    payload = json.loads(SOURCE.read_text())
    cells = analyse_debate(payload, load_tasks())
    expected_repeats = payload["config"]["repeats"]
    report = {
        "source": str(SOURCE.relative_to(V3)),
        "source_config": payload["config"],
        "method": {
            "rounds": [0, 2],
            "f": 0,
            "unit": "probability; multiply by 100 for percentage points",
            "aggregation": (
                "Average all repeats within each task and round, subtract "
                "round 0 from round 2 within task, then average over tasks."
            ),
            "interval": "95% percentile paired task bootstrap",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "task_order": "lexicographically sorted task IDs",
            "scoring": "Same parser and plurality tie-break as recompute_debate.py",
            "missingness": (
                "Unanimity requires every slot to be parseable. All retained "
                "debates have a non-missing plurality correctness indicator."
            ),
        },
        "by_n": {},
    }
    for n in (3, 5, 7, 10):
        grouped = {}
        for rnd in (0, 2):
            by_task = collections.defaultdict(list)
            for record in cells[(n, rnd, 0)]:
                by_task[record["task_id"]].append(record)
            grouped[rnd] = by_task
        assert set(grouped[0]) == set(grouped[2]), "Rounds have different task sets"
        per_task = []
        for task_id in sorted(grouped[0]):
            row = {"task_id": task_id}
            for rnd in (0, 2):
                records = grouped[rnd][task_id]
                assert len(records) == expected_repeats, (n, rnd, task_id)
                assert all(r["majority_acc"] is not None for r in records)
                row[f"round_{rnd}"] = {
                    "repeats": len(records),
                    "agreement": sum(float(r["agree"]) for r in records) / len(records),
                    "majority_accuracy": sum(float(r["majority_acc"]) for r in records) / len(records),
                }
            row["agreement_gain"] = row["round_2"]["agreement"] - row["round_0"]["agreement"]
            row["accuracy_gain"] = row["round_2"]["majority_accuracy"] - row["round_0"]["majority_accuracy"]
            row["agreement_minus_accuracy_gain"] = row["agreement_gain"] - row["accuracy_gain"]
            per_task.append(row)

        summary = {"n_tasks": len(per_task), "repeats_per_task_per_round": expected_repeats}
        for rnd in (0, 2):
            summary[f"round_{rnd}"] = {
                metric: interval([r[f"round_{rnd}"][metric] for r in per_task])
                for metric in ("agreement", "majority_accuracy")
            }
        for metric in ("agreement_gain", "accuracy_gain", "agreement_minus_accuracy_gain"):
            summary[metric] = interval([r[metric] for r in per_task])
        # Balanced repetitions imply the paired means also reproduce the
        # difference between the full archived round-level outcome rates.
        for metric, field in (("agreement_gain", "agree"), ("accuracy_gain", "majority_acc")):
            full_difference = (
                sum(float(r[field]) for r in cells[(n, 2, 0)]) / len(cells[(n, 2, 0)])
                - sum(float(r[field]) for r in cells[(n, 0, 0)]) / len(cells[(n, 0, 0)])
            )
            assert math.isclose(summary[metric]["estimate"], full_difference, abs_tol=1e-12)
        summary["per_task"] = per_task
        report["by_n"][str(n)] = summary
    return report


if __name__ == "__main__":
    result = recompute()
    DESTINATION.write_text(json.dumps(result, indent=2) + "\n")
    for n, row in result["by_n"].items():
        print(f"n={n}; {row['n_tasks']} tasks; {row['repeats_per_task_per_round']} repeats/task")
        for metric in ("agreement_gain", "accuracy_gain", "agreement_minus_accuracy_gain"):
            v = row[metric]
            print(f"  {metric}: {v['estimate']:.6f} [{v['lo']:.6f}, {v['hi']:.6f}]")
    print(f"Wrote {DESTINATION}")
