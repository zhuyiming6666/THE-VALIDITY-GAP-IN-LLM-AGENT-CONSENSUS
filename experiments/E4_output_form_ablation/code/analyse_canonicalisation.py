"""Offline re-analysis of the output-form ablation (Appendix "Output-form ablation").

Recomputes per-condition accuracy / p_max and the task-paired A-B and B-C contrasts from the
archived claude-opus-5 replies (50 GSM8K tasks x 3 conditions x k=8) and writes
results/x1_canonicalisation.recomputed.json.  Run from the repository root after
`source experiments/env.sh`:
    python experiments/E4_output_form_ablation/code/analyse_canonicalisation.py
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp_canonicalisation as X  # noqa: E402

EXP = HERE.parents[1]
tasks = {json.loads(l)["id"]: json.loads(l) for l in open(EXP / "baseline/data/tasks.jsonl") if l.strip()}
recs = [json.loads(l) for l in open(HERE.parent / "results/x1_canon_opus5.samples.jsonl") if l.strip()]
models = sorted({r["model"] for r in recs})
k = max(r["rep"] for r in recs) + 1
out = {"model": models, "k": k, "n_records": len(recs),
       "per_condition": X.analyse(recs, tasks, k),
       "paired": X.paired_contrasts(recs, tasks)}
(HERE.parent / "results/x1_canonicalisation.recomputed.json").write_text(json.dumps(out, indent=1))
for c, e in out["per_condition"].items():
    print(c, "tasks", e["n_tasks"], "acc %.4f" % e["accuracy"], "pmax %.4f" % e["p_max"])
for c, e in out["paired"].items():
    a = e["accuracy"]; print(c, "diff %.4f [%.4f, %.4f]" % (a["estimate"], a["lo"], a["hi"]))
