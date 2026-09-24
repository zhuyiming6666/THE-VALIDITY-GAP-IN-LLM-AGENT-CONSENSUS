"""Recompute the offline KEY-prefix sensitivity experiment."""

from collections import Counter
from pathlib import Path
import hashlib
import json
import sys

import numpy as np


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "code"))

from collision import A_unbiased
from recompute_e1 import MODELS, load_records


def main():
    sources = {}

    def read_json(relative_path):
        path = PACKAGE / relative_path
        sources[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    e1 = read_json("results/e1_analysis.json")
    task_ids = e1["common_tasks"]["ids"]
    records_by_model, _, _ = load_records()

    for relative_path in [
        "data/samples.jsonl",
        "data/samples2.jsonl",
        "code/scoring.py",
        "code/recompute_e1.py",
    ]:
        path = PACKAGE / relative_path
        sources[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()

    result = {
        "method": (
            "Normalize KEY with the main scorer, then truncate to the first L "
            "whitespace-delimited words. The same replies and 123 tasks are used "
            "at every L; missing KEY remains None. This is offline lexical "
            "coarsening, not a generation-length intervention."
        ),
        "models": {},
    }

    for model in MODELS:
        rows = []
        lengths = []
        for task_id in task_ids:
            records = [r for r in records_by_model[model][task_id] if r["has_answer"]]
            values = {}
            for length in [5, 10, 20, None]:
                labels = [
                    (
                        r["answer_class"],
                        " ".join(r["semantic_class"][1].split()[:length])
                        if r["has_key"]
                        else None,
                    )
                    for r in records
                ]
                values[str(length)] = A_unbiased(list(Counter(labels).values()), 3)

            assert values["5"] + 1e-14 >= values["10"]
            assert values["10"] >= values["20"] - 1e-14
            assert values["20"] + 1e-14 >= values["None"]
            rows.append({"task": task_id, **values})
            lengths.extend(
                len(r["semantic_class"][1].split()) for r in records if r["has_key"]
            )

        means = {
            length: float(np.mean([row[length] for row in rows]))
            for length in ["5", "10", "20", "None"]
        }
        assert abs(means["None"] - e1["summary_common_tasks"][model]["A_semantic_3"]) < 1e-6
        result["models"][model] = {
            "n_tasks": len(task_ids),
            "usable_keys": len(lengths),
            "over_5": sum(value > 5 for value in lengths),
            "over_10": sum(value > 10 for value in lengths),
            "over_20": sum(value > 20 for value in lengths),
            "mean_A3": means,
            "per_task": rows,
        }

    result["sources_sha256"] = sources
    output = PACKAGE / "results/key_length_sensitivity.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
