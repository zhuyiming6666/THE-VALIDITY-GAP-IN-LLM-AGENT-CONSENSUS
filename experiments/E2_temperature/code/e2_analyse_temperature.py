"""E2: temperature sweep analysis on the 123-task common set (gpt-4.1-nano).

Run from the repository root:
    python experiments/E2_temperature/code/e2_analyse_temperature.py

Inputs: E2_temperature/results/temp_T0.3.samples.jsonl and temp_T0.7.samples.jsonl (k=16) and
the stored T=1.0 replies (baseline/data/samples.jsonl, all replies). Per temperature and
partition: U-statistic A(3), A(10), mean p_max, mean top-two margin, single-sample accuracy,
designated-mode coverage at n=10 for f=1,2,3 and at n=13, f=3 (Monte Carlo, plug-in law).

The first part reproduces the archived e2_temperature_analysis.json exactly (same RNG stream).
The second part is a sample-size control: the T=1.0 replies are subsampled to k=16 per task
(20 random subsamples, averaged) so that plug-in quantities (p_max, coverage) are compared at
the same k as the T=0.3/0.7 runs.  Writes e2_temperature_analysis.json,
e2_temperature_k16_control.json and fig_temperature.pdf.
"""
import sys, json, collections, random, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments"
sys.path.insert(0, str(EXP / "E1_human_calibration" / "code"))
from annotation_normalization import answer_class, semantic_class, correct  # noqa: E402
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-grain")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = EXP / "E2_temperature" / "results"
BENCH = lambda tid: {"gsm": "gsm8k", "mmlu": "mmlu", "mbpp": "mbpp"}[tid.split("_")[0]]
common = json.load(open(EXP / "baseline/results/common_tasks_123.json"))
gold = {json.loads(l)["id"]: json.loads(l)["answer"] for l in open(EXP / "baseline/data/tasks.jsonl") if l.strip()}
rng = random.Random(20260919)


def load(path, model="gpt-4.1-nano"):
    by = collections.defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        if r.get("model", model) == model and r["task_id"] in set(common): by[r["task_id"]].append(r)
    return by


def falling(a, h):
    p = 1
    for i in range(h): p *= (a - i)
    return p


def A_U(cls, h):
    k = len(cls); c = collections.Counter(cls)
    return sum(falling(v, h) for v in c.values()) / falling(k, h) if k >= h else None


def coverage(cls, f, h, draws=5000):
    c = collections.Counter(cls); labels = list(c); probs = [c[x] / len(cls) for x in labels]
    star = max(range(len(labels)), key=lambda i: (probs[i], -i)); ok = 0
    for _ in range(draws):
        hist = collections.Counter(rng.choices(range(len(labels)), probs, k=h))
        ok += hist[star] - max([hist[i] for i in range(len(labels)) if i != star] + [0]) > 2 * f
    return ok / draws


def mean(v): v = [x for x in v if x is not None]; return sum(v) / len(v)


def classes(by, t, part):
    b = BENCH(t); cls = []
    for r in by[t]:
        a = answer_class(r.get("answer"), b)
        if a is None: continue
        cls.append(a if part == "answer" else semantic_class(r.get("answer"), r.get("key"), b))
    return cls


def analyse(by):
    out = {}
    for part in ("answer", "semantic"):
        rows = {"A3": [], "A10": [], "pmax": [], "margin": [], "cov10_f1": [], "cov10_f2": [], "cov10_f3": [], "cov13_f3": []}
        for t in common:
            cls = classes(by, t, part)
            if len(cls) < 10: continue
            c = collections.Counter(cls); p = sorted((v / len(cls) for v in c.values()), reverse=True)
            rows["A3"].append(A_U(cls, 3)); rows["A10"].append(A_U(cls, 10)); rows["pmax"].append(p[0]); rows["margin"].append(p[0] - (p[1] if len(p) > 1 else 0))
            for f in (1, 2, 3): rows[f"cov10_f{f}"].append(coverage(cls, f, 10 - f))
            rows["cov13_f3"].append(coverage(cls, 3, 10))
        out[part] = {k: mean(v) for k, v in rows.items()}
    acc = []
    for t in common:
        va = [correct(r.get("answer"), gold[t], BENCH(t)) for r in by[t] if answer_class(r.get("answer"), BENCH(t)) is not None]
        if va: acc.append(sum(bool(x) for x in va) / len(va))
    out["accuracy"] = mean(acc); out["n_tasks"] = len(acc)
    return out


# ---- Part 1: archived analysis (unchanged computation and RNG order) ----
by10 = load(EXP / "baseline/data/samples.jsonl")
res = {"1.0": analyse(by10)}
for T in ("0.3", "0.7"):
    res[T] = analyse(load(OUT / f"temp_T{T}.samples.jsonl"))
json.dump(res, open(OUT / "e2_temperature_analysis.json", "w"), indent=1)
Ts = sorted(res, key=float)
print(f"{'T':>4}{'acc':>7} | {'ans A3':>7}{'ans A10':>8}{'pmax':>6}{'cov f1':>7}{'cov f2':>7}{'cov f3':>7}{'n13 f3':>7} | {'KEY A3':>7}{'KEY A10':>9}{'pmax':>6}{'cov f1':>7}{'cov f3':>7}")
for T in Ts:
    a, s = res[T]["answer"], res[T]["semantic"]
    print(f"{T:>4}{res[T]['accuracy']:>7.3f} | {a['A3']:>7.3f}{a['A10']:>8.3f}{a['pmax']:>6.3f}{a['cov10_f1']:>7.3f}{a['cov10_f2']:>7.3f}{a['cov10_f3']:>7.3f}{a['cov13_f3']:>7.3f} | {s['A3']:>7.3f}{s['A10']:>9.5f}{s['pmax']:>6.3f}{s['cov10_f1']:>7.3f}{s['cov10_f3']:>7.3f}")

# ---- Part 2: k=16 sample-size control for T=1.0 ----
nrng = np.random.default_rng(20260924)


def cov_np(cls, f, h, draws=5000):
    c = collections.Counter(cls); probs = np.array([v / len(cls) for v in c.values()])
    star = int(np.argmax(probs))  # first maximal class, as in coverage()
    H = nrng.multinomial(h, probs, size=draws)
    lead = H[:, star] - (np.delete(H, star, axis=1).max(axis=1) if len(probs) > 1 else 0)
    return float((lead > 2 * f).mean())


REPS, K = 20, 16
ctl = {"k": K, "subsamples": REPS, "seed": 20260924, "answer": {}, "semantic": {}}
for part in ("answer", "semantic"):
    acc = collections.defaultdict(list)
    for t in common:
        full = classes(by10, t, part)
        if len(full) < K: continue
        vals = collections.defaultdict(list)
        for _ in range(REPS):
            cls = [full[i] for i in nrng.choice(len(full), K, replace=False)]
            c = collections.Counter(cls); p = sorted((v / K for v in c.values()), reverse=True)
            vals["A3"].append(A_U(cls, 3)); vals["A10"].append(A_U(cls, 10)); vals["pmax"].append(p[0])
            vals["cov10_f1"].append(cov_np(cls, 1, 9)); vals["cov10_f3"].append(cov_np(cls, 3, 7))
            vals["cov13_f3"].append(cov_np(cls, 3, 10))
        for k2, v in vals.items(): acc[k2].append(sum(v) / len(v))
    ctl[part] = {k2: mean(v) for k2, v in acc.items()}
    ctl[part]["n_tasks"] = len(acc["A3"])
json.dump(ctl, open(OUT / "e2_temperature_k16_control.json", "w"), indent=1)
a, s = ctl["answer"], ctl["semantic"]
print(f"1.0@k16 | ans A3 {a['A3']:.3f} A10 {a['A10']:.3f} pmax {a['pmax']:.3f} cov f1 {a['cov10_f1']:.3f} f3 {a['cov10_f3']:.3f} n13 {a['cov13_f3']:.3f}"
      f" | KEY A3 {s['A3']:.3f} A10 {s['A10']:.5f} pmax {s['pmax']:.3f} cov f1 {s['cov10_f1']:.3f} f3 {s['cov10_f3']:.3f}")

# ---- Figure: same f for both grains; accuracy on a non-exaggerated axis ----
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "legend.fontsize": 6.5, "pdf.fonttype": 42})
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6), layout="constrained"); x = [float(T) for T in Ts]
ax[0].plot(x, [res[T]["answer"]["A3"] for T in Ts], "o-", ms=3, color="#2f6690", label="answer $\\hat A_U(3)$")
ax[0].plot(x, [res[T]["answer"]["cov10_f1"] for T in Ts], "s--", ms=3, color="#2f6690", label="answer coverage, $f=1$")
ax[0].plot(x, [res[T]["semantic"]["A3"] for T in Ts], "o-", ms=3, color="#b34f36", label="answer+KEY $\\hat A_U(3)$")
ax[0].plot(x, [res[T]["semantic"]["cov10_f1"] for T in Ts], "s--", ms=3, color="#b34f36", label="answer+KEY coverage, $f=1$")
ax[0].set(title="(a) Activation and coverage", xlabel="Temperature", ylim=(-0.03, 1.03), xticks=x); ax[0].legend()
ax[1].plot(x, [res[T]["accuracy"] for T in Ts], "o-", ms=3, color="#444444")
ax[1].set(title="(b) Single-sample accuracy", xlabel="Temperature", xticks=x, ylim=(0.5, 1.0))
fig.savefig(OUT / "fig_temperature.pdf"); print("saved", OUT / "fig_temperature.pdf")
