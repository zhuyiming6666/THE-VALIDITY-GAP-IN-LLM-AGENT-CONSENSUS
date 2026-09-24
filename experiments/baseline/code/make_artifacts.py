"""Artifacts that other scripts and the paper generator depend on.

Two small analyses that are cleaner as their own step:

1. ``prediction.json`` --- the parameter-free comparison of Section 5.2.  The
   antecedent probability $A(n)$ is estimated from single-agent samples on the
   tasks the debate actually ran, and compared with the observed round-0
   all-agree rate of the groups.  Nothing is fitted.  Aggregation is per task
   first and then over tasks, matching how the debate cells were reported.

2. ``scoring_attribution.json`` --- how much of the accuracy change comes from
   the corrected scorer rather than from the change of task set.  Reported
   per benchmark and overall, under both rules, on both the full record set and
   the common task set, so the two effects can be separated.

Run:  python3 make_artifacts.py
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collision import A_unbiased  # noqa: E402
from recompute_debate import analyse_debate  # noqa: E402
from recompute_e1 import load_records, load_tasks  # noqa: E402
from scoring import correct  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
PACKAGE = HERE.parent
ROOT = PACKAGE
RESULTS = PACKAGE / "results"

MODELS = ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1-mini"]


# --------------------------------------------------------------------------
# 1. parameter-free prediction
# --------------------------------------------------------------------------
def prediction():
    tasks = load_tasks()
    by, _funnel, _t = load_records()
    e3_path = ROOT / "data/interaction/e3_benign.json"
    if not e3_path.exists():
        raise SystemExit(f"missing {e3_path}")
    e3 = json.loads(e3_path.read_text())
    cells = analyse_debate(e3, tasks)

    used = sorted({d["task_id"] for d in e3["raw"]})
    rows = {}
    for n in (3, 5, 7, 10):
        # prediction: per-task A^U(n) on the same tasks the debate ran
        vals = []
        for t in used:
            recs = [r for r in by["gpt-4.1-nano"][t] if r["has_answer"]]
            counts = collections.Counter(r["answer_class"] for r in recs)
            a = A_unbiased(sorted(counts.values(), reverse=True), n)
            if a is not None:
                vals.append(a)
        predicted = sum(vals) / len(vals)

        # observation: per-task agree rate, then mean over tasks
        per = collections.defaultdict(list)
        for x in cells.get((n, 0, 0), []):
            per[x["task_id"]].append(1.0 if x["agree"] else 0.0)
        observed = sum(sum(v) / len(v) for v in per.values()) / len(per)
        sem = collections.defaultdict(list)
        for x in cells.get((n, 0, 0), []):
            sem[x["task_id"]].append(1.0 if x["agree_semantic"] else 0.0)
        observed_sem = sum(sum(v) / len(v) for v in sem.values()) / len(sem)
        rows[str(n)] = {
            "predicted": round(predicted, 6),
            "observed": round(observed, 6),
            "residual": round(observed - predicted, 6),
            "observed_semantic": round(observed_sem, 6),
            "n_tasks": len(vals),
            "bench": sorted({tasks[t]["bench"] for t in used}),
        }
    gaps = [abs(r["residual"]) for r in rows.values()]
    return {
        "model": "gpt-4.1-nano",
        "n_tasks": len(used),
        "aggregation": "per-task rate, then mean over tasks; nothing fitted",
        "rows": rows,
        "max_abs_gap": max(gaps),
        "mean_abs_gap": sum(gaps) / len(gaps),
    }


# --------------------------------------------------------------------------
# 2. scoring attribution
# --------------------------------------------------------------------------
_NUM = r"-?\d+(?:\.\d+)?"


def _v1_norm(s):
    """The v1 normaliser, reproduced verbatim for attribution only."""
    if s is None:
        return None
    s = s.strip()
    s = re.sub(r"^(?:\*\*|`|#)+\s*", "", s)
    s = re.sub(r"^(?:final\s+)?answer\s*[:：]\s*", "", s, flags=re.I)
    s = re.sub(r"(?:\*\*|`)+$", "", s).strip()
    s = s.lower().rstrip(".").replace(",", "").replace("$", "").strip("'\" ")
    m = re.fullmatch(r"([a-d])(?:[.)]\s*.*)?", s)
    if m:
        return m.group(1)
    m = re.fullmatch(rf"({_NUM})(?:\s*(?:%|[a-z][a-z .\-/]*))?", s)
    if m:
        return str(float(m.group(1)))
    return " ".join(s.replace('"', "'").split())


def scoring_attribution():
    tasks = load_tasks()
    e1 = json.loads((RESULTS / "e1_analysis.json").read_text())
    common = set(e1["common_tasks"]["ids"])

    tally = collections.defaultdict(lambda: collections.Counter())
    for rel in ("results/samples.jsonl", "results/samples2.jsonl"):
        path = ROOT / rel
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            m = rec.get("model")
            tid = rec.get("task_id")
            if m not in MODELS or tid not in tasks:
                continue
            bench = tasks[tid]["bench"]
            gold = tasks[tid]["answer"]
            a = rec.get("answer")
            if a is None:
                continue
            old = int(_v1_norm(a) == _v1_norm(gold))
            new = int(correct(a, gold, bench))
            for scope in ("all", "common"):
                if scope == "common" and tid not in common:
                    continue
                t = tally[(scope, m, bench)]
                t["n"] += 1
                t["old"] += old
                t["new"] += new

    def block(scope):
        out = {}
        benches = ["gsm8k", "mmlu", "mbpp"]
        agg = collections.Counter()
        for m in MODELS:
            for b in benches:
                t = tally.get((scope, m, b))
                if not t or not t["n"]:
                    continue
                out.setdefault(b, collections.Counter())
                for k in ("n", "old", "new"):
                    out[b][k] += t[k]
                    agg[k] += t[k]
        res = {}
        for b in benches:
            t = out.get(b)
            if not t or not t["n"]:
                continue
            res[b] = {
                "n": t["n"],
                "v1_accuracy": round(t["old"] / t["n"], 6),
                "v3_accuracy": round(t["new"] / t["n"], 6),
                "delta": round((t["new"] - t["old"]) / t["n"], 6),
            }
        if agg["n"]:
            res["_overall"] = {
                "n": agg["n"],
                "v1_accuracy": round(agg["old"] / agg["n"], 6),
                "v3_accuracy": round(agg["new"] / agg["n"], 6),
                "delta": round((agg["new"] - agg["old"]) / agg["n"], 6),
            }
        return res

    common_block = block("common")
    all_block = block("all")
    return {
        "note": "old = v1 normaliser applied to every benchmark; new = "
                "benchmark-specific scoring in scoring.py",
        "common_tasks": sorted(common),
        "per_bench": {k: v for k, v in common_block.items() if k != "_overall"},
        "all": common_block.get("_overall"),
        "all_tasks_per_bench": {k: v for k, v in all_block.items()
                                if k != "_overall"},
        "all_tasks_overall": all_block.get("_overall"),
    }


def main():
    RESULTS.mkdir(exist_ok=True)

    p = prediction()
    (RESULTS / "prediction.json").write_text(json.dumps(p, indent=1))
    print("=== parameter-free prediction (gpt-4.1-nano) ===")
    print(f"{'n':>3}{'predicted':>11}{'observed':>10}{'residual':>10}"
          f"{'observed sem':>14}{'tasks':>7}")
    for n, r in p["rows"].items():
        print(f"{n:>3}{r['predicted']:>11.4f}{r['observed']:>10.4f}"
              f"{r['residual']:>+10.4f}{r['observed_semantic']:>14.4f}"
              f"{r['n_tasks']:>7}")
    print(f"  max |residual| = {p['max_abs_gap']:.4f}")

    a = scoring_attribution()
    (RESULTS / "scoring_attribution.json").write_text(json.dumps(a, indent=1))
    print("\n=== scoring attribution, common task set ===")
    print(f"{'bench':<8}{'n':>7}{'v1 acc':>9}{'v3 acc':>9}{'delta':>9}")
    for b, v in a["per_bench"].items():
        print(f"{b:<8}{v['n']:>7}{v['v1_accuracy']:>9.4f}{v['v3_accuracy']:>9.4f}"
              f"{v['delta']:>+9.4f}")
    o = a["all"]
    print(f"{'ALL':<8}{o['n']:>7}{o['v1_accuracy']:>9.4f}{o['v3_accuracy']:>9.4f}"
          f"{o['delta']:>+9.4f}")
    print(f"  (all-tasks overall delta, for reference: "
          f"{a['all_tasks_overall']['delta']:+.4f})")

    print(f"\nwrote prediction.json and scoring_attribution.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
