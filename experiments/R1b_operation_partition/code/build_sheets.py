"""Build the R1b partition-annotation materials (no API calls).

Design: 24 model-task cells from the 123-task common set, 8 per benchmark. Within each benchmark the
tasks are stratified by answer-grain modal mass (four quartile bins, two tasks per bin) and the three
main models are assigned in rotation, so every model contributes 8 cells. For each cell, K=12 replies
are drawn at random from that model's replies with a parseable answer (the same pool that defines
the answer-grain plug-in law). Annotators see the question and, for each reply, its normalised final
answer and its KEY; they assign every reply to an operation group. The resulting partitions give
r_max, A(h) and designated-mode coverage at the human decisive-operation grain directly.

    python experiments/R1b_operation_partition/code/build_sheets.py
Writes annotation/tasks.tsv, annotation/annotator_{A,B,C}.tsv (identical blank sheets),
annotation/admin/key.json (hidden: model, source lines, selection data).
"""
import csv, json, random, sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]
OUT = EXP / "R1b_operation_partition" / "annotation"
sys.path.insert(0, str(EXP / "E1_human_calibration" / "code"))
from annotation_normalization import answer_class, norm_key  # noqa: E402

SEED, K, PER_BENCH = 20260925, 12, 8
MODELS = ("gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1-mini")
BENCH = lambda t: {"gsm": "gsm8k", "mmlu": "mmlu", "mbpp": "mbpp"}[t.split("_")[0]]
E1 = json.load(open(EXP / "baseline/results/e1_analysis.json"))
common = E1["common_tasks"]["ids"]
tasks = {json.loads(l)["id"]: json.loads(l) for l in open(EXP / "baseline/data/tasks.jsonl") if l.strip()}

recs = {}
for f in ("samples.jsonl", "samples2.jsonl"):
    for i, l in enumerate(open(EXP / "baseline/data" / f), 1):
        if l.strip():
            r = json.loads(l)
            if answer_class(r.get("answer"), BENCH(r["task_id"])) is not None:
                recs.setdefault((r["model"], r["task_id"]), []).append((f, i, r))
pmax = {(m, r["id"]): max(r["counts_verdict"]) / sum(r["counts_verdict"])
        for m in MODELS for r in E1["per_task"][m] if r["id"] in common}

rng = random.Random(SEED)
cells = []
for b in ("gsm8k", "mmlu", "mbpp"):
    ids = sorted(t for t in common if BENCH(t) == b)
    rng.shuffle(ids)
    used = set()
    for j in range(PER_BENCH):
        m = MODELS[(len(cells)) % 3]
        cand = sorted((t for t in ids if t not in used and len(recs.get((m, t), [])) >= K), key=lambda t: pmax[(m, t)])
        q = j // 2                                   # quartile bin 0..3, two tasks per bin
        lo, hi = q * len(cand) // 4, (q + 1) * len(cand) // 4
        t = rng.choice(cand[lo:hi])
        used.add(t)
        cells.append((m, t))

rows, key = [], {"seed": SEED, "K": K, "cells": {}}
for n, (m, t) in enumerate(cells, 1):
    tid = f"T{n:02d}"
    pick = rng.sample(recs[(m, t)], K)
    items = []
    for j, (f, line, r) in enumerate(pick, 1):
        iid = f"{tid}-R{j:02d}"
        a = answer_class(r.get("answer"), BENCH(t))
        items.append({"item": iid, "file": f, "line": line, "answer_class": repr(a),
                      "key_norm": norm_key(r.get("key"))})
        rows.append({"task": tid, "item": iid, "answer": (r.get("answer") or "").strip(),
                     "key": (r.get("key") or "").strip(), "group": "", "vague": "", "note": ""})
    key["cells"][tid] = {"model": m, "task_id": t, "bench": BENCH(t), "answer_pmax_full": pmax[(m, t)],
                         "n_pool": len(recs[(m, t)]), "items": items}

OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "tasks.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["task", "bench", "question"])
    for tid, c in key["cells"].items():
        w.writerow([tid, c["bench"], " ".join(tasks[c["task_id"]]["question"].split())])
for a in "ABC":
    with open(OUT / f"annotator_{a}.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader(); w.writerows(rows)
(OUT / "admin").mkdir(exist_ok=True)
(OUT / "admin" / "key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
print(len(cells), "cells,", len(rows), "items per annotator")
for tid, c in key["cells"].items():
    print(tid, c["bench"], c["model"], c["task_id"], f"pmax={c['answer_pmax_full']:.2f}")
