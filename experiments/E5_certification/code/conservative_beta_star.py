"""Conservative resilience thresholds (no API calls).

For each model-task cell and grain, the top-two margin Delta is replaced by its 5% posterior quantile
under the Dirichlet posterior used in Appendix B (observed classes plus one unseen class, total prior
mass 1), with the designated mode fixed at the empirical mode; beta*_cons = Delta_05 / (2 + Delta_05),
floored at 0. 4,000 posterior draws per cell, seed 20260924.

    python experiments/E5_certification/code/conservative_beta_star.py
Writes E5_certification/results/conservative_beta_star.json.
"""
import json
from pathlib import Path
import numpy as np

EXP = Path(__file__).resolve().parents[2]
E1 = json.load(open(EXP / "baseline/results/e1_analysis.json"))
common = set(E1["common_tasks"]["ids"])
rng = np.random.default_rng(20260924)
out = {}
for g, key in (("answer", "counts_verdict"), ("lexical", "counts_semantic")):
    plug, cons = [], []
    for m in ("gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1-mini"):
        for r in E1["per_task"][m]:
            if r["id"] not in common:
                continue
            c = np.array(sorted(r[key], reverse=True), float); q = len(c) + 1
            alpha = np.append(c + 1.0 / q, 1.0 / q)
            P = rng.dirichlet(alpha, size=4000)
            d = P[:, 0] - P[:, 1:].max(axis=1)
            d05 = max(0.0, float(np.quantile(d, 0.05)))
            dp = (c[0] - (c[1] if len(c) > 1 else 0)) / c.sum()
            plug.append(dp / (2 + dp)); cons.append(d05 / (2 + d05))
    plug, cons = np.array(plug), np.array(cons)
    out[g] = {"n_cells": int(len(cons)), "mean_plugin": float(plug.mean()), "mean_conservative": float(cons.mean()),
              "share_conservative_gt_0.2": float((cons > 0.2).mean()), "share_conservative_gt_0.3": float((cons > 0.3).mean()),
              "max_conservative": float(cons.max())}
(EXP / "E5_certification/results/conservative_beta_star.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
