"""Theorem (dispersion-limited resilience): numbers and figure (no API calls).

For every model-task cell of the 123-task common set and each grain, the plug-in law gives the
top-two margin Delta and the resilience threshold beta* = Delta / (2 + Delta).  With f = floor(beta n)
and h = n - f, the designated-mode coverage P_n = Pr[H_mode - max_d H_d > 2f] (the best any sound rule
can do) is computed by Monte Carlo (numpy multinomial, 4,000 draws per cell) for n in NS; the theorem
predicts P_n -> 1 for cells with beta < beta* and -> 0 for cells with beta > beta*, so the task-mean
coverage tends to the share of cells with beta* > beta.  Tied modes (Delta = 0) have beta* = 0.

    python experiments/E5_certification/code/resilience_threshold.py
Writes E5_certification/results/resilience_threshold.json and figures/fig_resilience.pdf.
"""
import json, os
from pathlib import Path
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-grain")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parents[2]
E1 = json.load(open(EXP / "baseline/results/e1_analysis.json"))
common = set(E1["common_tasks"]["ids"])
MODELS = ("gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1-mini")
NS = (10, 20, 40, 80, 160, 320)
BETAS = (0.1, 0.2, 0.3)
rng = np.random.default_rng(20260924)
DRAWS = 4000

laws = {"answer": [], "lexical": []}
for m in MODELS:
    for r in E1["per_task"][m]:
        if r["id"] in common:
            for g, key in (("answer", "counts_verdict"), ("lexical", "counts_semantic")):
                c = np.array(sorted(r[key], reverse=True), float)
                laws[g].append(c / c.sum())


def coverage(p, n, beta):
    f = int(np.floor(beta * n)); h = n - f
    if len(p) == 1:
        return 1.0 if h > 2 * f else 0.0
    H = rng.multinomial(h, p, size=DRAWS)
    return float(((H[:, 0] - H[:, 1:].max(axis=1)) > 2 * f).mean())


out = {"draws": DRAWS, "ns": NS, "betas": BETAS, "grains": {}}
for g, P in laws.items():
    delta = np.array([p[0] - (p[1] if len(p) > 1 else 0.0) for p in P])
    bstar = delta / (2 + delta)
    res = {"n_cells": len(P), "mean_delta": float(delta.mean()), "mean_beta_star": float(bstar.mean()),
           "median_beta_star": float(np.median(bstar)),
           "share_beta_star_ge": {str(b): float((bstar > b).mean()) for b in (0.1, 0.2, 0.25, 0.3, 1 / 3 - 1e-9)},
           "share_deterministic": float((delta == 1).mean()), "coverage": {}}
    for beta in BETAS:
        res["coverage"][str(beta)] = {"limit_share": float((bstar > beta).mean()),
                                      **{str(n): float(np.mean([coverage(p, n, beta) for p in P])) for n in NS}}
    out["grains"][g] = res
(EXP / "E5_certification/results/resilience_threshold.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

# ---- figure: (a) sharp threshold on one law, (b) distribution of beta* across cells
ILL = np.array([0.8, 0.15, 0.05]); ILL_BSTAR = (ILL[0] - ILL[1]) / (2 + ILL[0] - ILL[1])
NS_ILL = (10, 20, 40, 80, 160, 320, 640)
ill = {str(b): [coverage(ILL, n, b) for n in NS_ILL] for b in (0.15, 0.22, 0.27, 0.32)}
BGRID = np.linspace(0, 0.34, 69)
surv = {g: [float((np.array([(q[0] - (q[1] if len(q) > 1 else 0)) / (2 + q[0] - (q[1] if len(q) > 1 else 0)) for q in P]) > b).mean()) for b in BGRID]
        for g, P in laws.items()}
BPTS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
mc160 = {g: [float(np.mean([coverage(q, 160, b) for q in P])) for b in BPTS] for g, P in laws.items()}
out["illustration"] = {"law": ILL.tolist(), "beta_star": float(ILL_BSTAR), "ns": NS_ILL, "coverage": ill}
out["mc_n160"] = {"betas": BPTS, **mc160}
(EXP / "E5_certification/results/resilience_threshold.json").write_text(json.dumps(out, indent=1))

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "legend.fontsize": 6.5, "pdf.fonttype": 42})
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5), layout="constrained")
cols = ["#2f6690", "#6aa0c7", "#d59a8a", "#b34f36"]
for c, (b, v) in zip(cols, ill.items()):
    ax[0].plot(NS_ILL, v, "o-", ms=3, color=c, label=f"$\\beta={b}$")
ax[0].set(xscale="log", xticks=NS_ILL, xticklabels=[str(n) for n in NS_ILL], ylim=(-0.03, 1.03),
          xlabel="group size $n$  ($f=\\lfloor\\beta n\\rfloor$)", ylabel="designated-mode coverage",
          title=f"(a) $p=(0.8,0.15,0.05)$, $\\beta^*={ILL_BSTAR:.3f}$")
ax[0].minorticks_off(); ax[0].legend(loc="center left", fontsize=6.5)
for g, c, lab in (("answer", "#2f6690", "answer"), ("lexical", "#b34f36", "answer+KEY")):
    ax[1].plot(BGRID, surv[g], color=c, lw=1.4, label=f"{lab}: share with $\\beta^*>\\beta$")
    ax[1].plot(BPTS, mc160[g], "o", ms=3.5, color=c, mfc="white", label=f"{lab}: coverage at $n=160$")
ax[1].axvline(1 / 3, color="0.4", ls="--", lw=0.8); ax[1].text(0.328, 0.55, "classical $1/3$", rotation=90, ha="right", fontsize=6.5, color="0.3")
ax[1].set(xlim=(0, 0.345), ylim=(-0.03, 1.03), xlabel="fault fraction $\\beta=f/n$", ylabel="share of model--task cells",
          title="(b) measured laws: tolerable fault fraction")
ax[1].legend(loc="center left", fontsize=6)
figdir = EXP.parent / "figures"; figdir.mkdir(exist_ok=True)
fig.savefig(figdir / "fig_resilience.pdf"); print("saved", figdir / "fig_resilience.pdf")
