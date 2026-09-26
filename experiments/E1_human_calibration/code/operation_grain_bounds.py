"""Bounds on activation and certification at the human decisive-operation grain (no API calls).

At n = 3f+1 the designated-mode coverage under a plug-in law r is exactly r_max^h with h = 2f+1
(Remark "threshold degeneracy").  The human operation relation is not measured as a partition, but
if it is one and it refines the answer partition, then for every (model, task) cell

    A2 <= r_max <= p_max          (A2 = sum_c r(c)^2, p_max = answer-grain modal mass)
    => A2^h <= r_max^h <= p_max^(h-2) * A2 .

Both sides are linear or convex in A2, the pairwise human agreement probability of two replies
from the same model on the same task, which the stratified calibration sample estimates:

    upper:  E[r_max^h] <= E[p_max^(h-2) * A2]          (no homogeneity or independence assumption)
    lower:  E[r_max^h] >= E[A2^h] >= (E[A2])^h          (Jensen)

A2 is the plug-in (with-replacement) collision probability, A2 = 1/k + (1 - 1/k) * A2_distinct, so the
bounds refer to the same plug-in coverage as the certification tables.  Unresolved human pairs are
counted SAME for the upper bound and DIFFERENT for the lower bound.  Cell means are Hajek estimates
with weight w_i / N_cell (w_i = stratum weight, N_cell = number of same-model frame pairs in the cell),
so every (model, task) cell of the 123-task common set counts equally, as in the paper.
Intervals: within-stratum pair bootstrap, 2,000 resamples, seed 20260924.

The same estimator applied to answer equality is checked against the exact answer-grain values.

    python experiments/E1_human_calibration/code/operation_grain_bounds.py
Writes E1_human_calibration/results/operation_grain_bounds.json.
"""
import collections, json, math
from pathlib import Path
import numpy as np

EXP = Path(__file__).resolve().parents[2]
RES = EXP / "E1_human_calibration" / "results"
S = json.load(open(RES / "annotation_scoring.json"))
key = json.load(open(EXP / "E1_human_calibration/annotation/admin/pairs_key.json"))["pairs"]
E1 = json.load(open(EXP / "baseline/results/e1_analysis.json"))
common = set(E1["common_tasks"]["ids"])
MODELS = ("gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1-mini")

# answer-grain plug-in law per (model, task)
cell = {}
for m in MODELS:
    for r in E1["per_task"][m]:
        if r["id"] in common:
            cnt = r["counts_verdict"]; k = sum(cnt)
            cell[(m, r["id"])] = {"k": k, "pmax": max(cnt) / k, "A2": sum((c / k) ** 2 for c in cnt)}

# number of same-model frame pairs per cell (all stored replies of that model on that task)
nrec = collections.Counter()
for f, m in (("samples.jsonl", None), ("samples2.jsonl", None)):
    for l in open(EXP / "baseline/data" / f):
        if l.strip():
            r = json.loads(l); nrec[(r["model"], r["task_id"])] += 1
Ncell = {c: nrec[c] * (nrec[c] - 1) / 2 for c in cell}

rows = []
for p in S["per_pair"]:
    kk = key[p["pair_id"]]
    if not p["same_model_pair"] or p["task_id"] not in common:
        continue
    c = (kk["A_model"], p["task_id"])
    rows.append({"stratum": p["stratum"], "w": p["weight"] / Ncell[c], "cell": c,
                 "lo": p["joint"] is True, "hi": p["joint"] is not False,
                 "ans": bool(p["instruments"]["answer_only"])})
strata = collections.defaultdict(list)
for i, r in enumerate(rows):
    strata[r["stratum"]].append(i)


def estimate(idx, field, h):
    w = np.array([rows[i]["w"] for i in idx])
    k = np.array([cell[rows[i]["cell"]]["k"] for i in idx], float)
    pm = np.array([cell[rows[i]["cell"]]["pmax"] for i in idx])
    same = np.array([rows[i][field] for i in idx], float)
    a2 = 1 / k + (1 - 1 / k) * same                     # per-pair unbiased term for the plug-in A2
    mean_a2 = (w * a2).sum() / w.sum()
    upper = (w * pm ** (h - 2) * a2).sum() / w.sum()
    return mean_a2, upper


H = (3, 5, 7)
out = {"n_pairs": len(rows), "n_cells_with_pairs": len({r["cell"] for r in rows}),
       "n_cells": len(cell), "assumptions": [
           "the human operation relation is an equivalence relation (a partition)",
           "it refines the answer partition (human-equivalent => same normalised answer)",
           "pairs in a cell are exchangeable draws of that model's replies on that task"],
       "refinement_violations": sum(1 for p in S["per_pair"] if p["joint"] is True and not p["instruments"]["answer_only"]),
       "results": {}}
rng = np.random.default_rng(20260924)
boot_idx = [np.concatenate([rng.choice(v, len(v), replace=True) for v in strata.values()]) for _ in range(2000)]
allidx = np.arange(len(rows))
for grain, lo_f, hi_f in (("human_operation", "lo", "hi"), ("answer_equality_check", "ans", "ans")):
    res = {}
    for h in H:
        a2_lo, _ = estimate(allidx, lo_f, h)
        _, up = estimate(allidx, hi_f, h)
        bl = [estimate(b, lo_f, h)[0] ** h for b in boot_idx]
        bu = [estimate(b, hi_f, h)[1] for b in boot_idx]
        res[f"h={h}"] = {"lower": a2_lo ** h, "lower_ci95_low": float(np.quantile(bl, 0.025)),
                         "upper": up, "upper_ci95_high": float(np.quantile(bu, 0.975)),
                         "mean_A2_used_for_lower": a2_lo}
    out["results"][grain] = res
exact = {h: float(np.mean([v["pmax"] ** h for v in cell.values()])) for h in H}
exact_a2 = float(np.mean([v["A2"] for v in cell.values()]))
out["answer_grain_exact"] = {f"h={h}": exact[h] for h in H} | {"mean_A2": exact_a2}
q = {}
for f in ("lo", "hi"):
    sub = [r for r in rows if r["ans"]]
    w = np.array([r["stratum"] and 1.0 for r in sub]) * np.array([r["w"] for r in sub])
    q[f] = float((w * np.array([r[f] for r in sub], float)).sum() / w.sum())
out["homogeneous_model"] = {"q_same_answer_same_model": q,
                            **{f"h={h}": [exact[h] * q['lo'] ** ((h) / 2), exact[h] * q['hi'] ** ((h) / 2)] for h in H}}
(RES / "operation_grain_bounds.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "assumptions"}, indent=1))
