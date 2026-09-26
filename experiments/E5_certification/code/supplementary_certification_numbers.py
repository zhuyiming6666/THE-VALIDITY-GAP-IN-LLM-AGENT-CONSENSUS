"""Supplementary numbers used in the text (no API calls, no new data).

1. Hoeffding bound vs exact coverage.  Corollary (Hoeffding) evaluated on the measured plug-in laws
   (n=10, h=10-f, m-1 = support size) against the exact designated-mode coverage, task-equal means.
2. Dirichlet prior sensitivity at f=3.  At n=3f+1 the event is "all h honest labels on the designated
   mode", so under a Dirichlet posterior the coverage is the posterior mean of p(c*)^h, available in
   closed form: prod_{i<h} (a*+i)/(A+i). Prior mass s is spread evenly over the observed classes plus
   one unseen class (the paper uses s=1).
3. Dependence rho from round-0 debates.  Grid least squares of the task-mean round-0 unanimity at each n on the
   U-statistic prediction plus the increment A_rho(n) - A_0(n) (plug-in law, dependence model of the
   Appendix), pooled over n in {3,5,7,10}; 95% task-bootstrap interval (1,000 resamples).

    python experiments/E5_certification/code/supplementary_certification_numbers.py
Writes E5_certification/results/supplementary_certification_numbers.json.
"""
import json, math
from math import comb
from pathlib import Path
import numpy as np

EXP = Path(__file__).resolve().parents[2]
E1 = json.load(open(EXP / "baseline/results/e1_analysis.json"))
common = set(E1["common_tasks"]["ids"])
MODELS = ("gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1-mini")
out = {}

# ---------------------------------------------------------------- 1. Hoeffding vs exact
def law(counts):
    k = sum(counts); return sorted((c / k for c in counts), reverse=True)

res = {}
for grain, key in (("answer", "counts_verdict"), ("lexical", "counts_semantic")):
    res[grain] = {}
    for f in range(4):
        h = 10 - f; bound, exact = [], []
        for m in MODELS:
            for r in E1["per_task"][m]:
                if r["id"] not in common: continue
                p = law(r[key]); d = p[0] - (p[1] if len(p) > 1 else 0.0); sup = len(p)
                bound.append(max(0.0, 1 - sup * math.exp(-(h * d - 2 * f) ** 2 / (2 * h))) if h * d > 2 * f else 0.0)
                if f == 3:
                    exact.append(p[0] ** h)
        res[grain][f"f={f}"] = {"hoeffding_mean": float(np.mean(bound))}
        if exact:
            res[grain][f"f={f}"]["exact_mean"] = float(np.mean(exact))
out["hoeffding_vs_exact"] = res

# ---------------------------------------------------------------- 2. Dirichlet prior sensitivity (f=3)
def post_mean_pow(counts, s, h):
    q = len(counts) + 1
    a = [c + s / q for c in counts] + [s / q]
    star = max(range(len(counts)), key=lambda i: counts[i])
    A = sum(a)
    return math.prod((a[star] + i) / (A + i) for i in range(h))

dir_res = {}
for grain, key in (("answer", "counts_verdict"), ("lexical", "counts_semantic")):
    dir_res[grain] = {}
    for s in (0.1, 1.0, 10.0):
        v = [post_mean_pow(r[key], s, 7) for m in MODELS for r in E1["per_task"][m] if r["id"] in common]
        dir_res[grain][f"s={s}"] = float(np.mean(v))
out["dirichlet_prior_sensitivity_f3"] = dir_res

# ---------------------------------------------------------------- 3. rho from round-0 debates
deb = json.load(open(EXP / "baseline/results/submission_debate_check.json"))["by_n"]
per_task = {r["id"]: r for r in E1["per_task"]["gpt-4.1-nano"]}
NS = (3, 5, 7, 10)


def A_rho(p, n, rho):
    return sum(ps * sum((rho * (c == s) + (1 - rho) * pc) ** n for c, pc in enumerate(p)) for s, ps in enumerate(p))


GRID = np.round(np.arange(0, 0.5001, 0.0025), 4)
rows = []  # (task, n, observed, U-statistic prediction, increment A_rho - A_0 on the plug-in law per grid value)
for n in NS:
    for t in deb[str(n)]["per_task"]:
        cnt = per_task[t["task_id"]]["counts_verdict"]; k = sum(cnt)
        u = sum(comb(c, n) for c in cnt) / comb(k, n)
        p = [c / k for c in cnt]
        a0 = A_rho(p, n, 0.0)
        rows.append((t["task_id"], n, t["round_0"]["agreement"], u, np.array([A_rho(p, n, r) - a0 for r in GRID])))
tasks = sorted({r[0] for r in rows})


def fit(sel):
    # fit the task-mean unanimity at each n (the quantity of Table 2); dA/drho = 0 at rho = 0, so use a grid
    sse = np.zeros(len(GRID))
    for n in NS:
        s = [x for x in sel if x[1] == n]
        obs = np.mean([x[2] for x in s]); pred = np.mean([x[3] for x in s]) + np.mean(np.stack([x[4] for x in s]), axis=0)
        sse += (obs - pred) ** 2
    return float(GRID[int(np.argmin(sse))])


rho = fit(rows)
rng = np.random.default_rng(20260924)
by_task = {t: [x for x in rows if x[0] == t] for t in tasks}
boot = [fit([x for t in rng.choice(tasks, len(tasks)) for x in by_task[t]]) for _ in range(1000)]
out["dependence_rho"] = {"estimate": rho, "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                         "n_tasks": len(tasks), "ns": NS, "grid_step": 0.0025,
                         "note": "grid least squares of task-mean round-0 unanimity at each n on the task-mean U-statistic prediction "
                                 "plus the plug-in increment A_rho(n)-A_0(n); dA_rho/drho vanishes at rho=0"}
(EXP / "E5_certification/results/supplementary_certification_numbers.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
