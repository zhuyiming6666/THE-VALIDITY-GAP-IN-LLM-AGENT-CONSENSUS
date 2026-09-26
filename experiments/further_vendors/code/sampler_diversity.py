"""Sampler-diversity check for the further-vendor runs (k=16, T=1.0).

Replaces an unarchived interactive probe.  For each model and task it counts distinct reply
openings (first 80 characters of the raw reply) and distinct normalised KEY strings among the
archived replies.  A near-deterministic sampler gives one opening on most tasks.

Run from the repository root:
    python experiments/further_vendors/code/sampler_diversity.py
"""
import collections, json, statistics as st
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
rows = []
for name in ("e1_crossvendor.samples.jsonl", "e1_qwen14b.samples.jsonl"):
    rows += [json.loads(l) for l in open(R / name) if l.strip()]
by = collections.defaultdict(list)
for r in rows:
    by[(r["model"], r["task_id"])].append(r)
out = {"opening_chars": 80}
for m in sorted({k[0] for k in by}):
    ks = [k for k in by if k[0] == m]
    op = [len({(x.get("raw") or "")[:80] for x in by[k]}) for k in ks]
    ky = [len({(x.get("key") or "").strip().lower() for x in by[k]}) for k in ks]
    out[m] = {"tasks": len(ks), "replies_per_task": st.mean(len(by[k]) for k in ks),
              "mean_distinct_openings": st.mean(op), "tasks_single_opening": sum(o == 1 for o in op),
              "mean_distinct_keys": st.mean(ky)}
    print(m, out[m])
json.dump(out, open(R / "sampler_diversity.json", "w"), indent=1)
