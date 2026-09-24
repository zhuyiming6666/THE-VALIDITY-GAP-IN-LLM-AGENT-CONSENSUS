"""E2: temperature sweep analysis on the 123-task common set (gpt-4.1-nano).
Inputs: results/v11/temp_T0.3.samples.jsonl, temp_T0.7.samples.jsonl (k=16) and the stored
T=1.0 replies (results/samples.jsonl, all replies). Per temperature and partition: U-statistic
A(3), A(10), mean p_max, mean top-two margin, single-sample accuracy, designated-mode coverage at
n=10 for f=1,2,3 and at n=13, f=3 (Monte Carlo, plug-in law). Writes JSON and a two-panel figure.
"""
import sys, json, collections, random, os
sys.path.insert(0, "v4/code")
from annotation_normalization import answer_class, semantic_class, correct
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-v12")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BENCH = lambda tid: {"gsm": "gsm8k", "mmlu": "mmlu", "mbpp": "mbpp"}[tid.split("_")[0]]
common = json.load(open("results/v11/common_tasks_123.json"))
gold = {json.loads(l)["id"]: json.loads(l)["answer"] for l in open("data/tasks.jsonl") if l.strip()}
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

def analyse(by):
    out = {}
    for part in ("answer", "semantic"):
        rows = {"A3": [], "A10": [], "pmax": [], "margin": [], "cov10_f1": [], "cov10_f2": [], "cov10_f3": [], "cov13_f3": []}
        for t in common:
            b = BENCH(t); cls = []
            for r in by[t]:
                a = answer_class(r.get("answer"), b)
                if a is None: continue
                cls.append(a if part == "answer" else semantic_class(r.get("answer"), r.get("key"), b))
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

res = {"1.0": analyse(load("results/samples.jsonl"))}
for T in ("0.3", "0.7"):
    p = f"results/v11/temp_T{T}.samples.jsonl"
    if os.path.exists(p): res[T] = analyse(load(p))
json.dump(res, open("results/v11/e2_temperature_analysis.json", "w"), indent=1)
Ts = sorted(res, key=float)
print(f"{'T':>4}{'acc':>7} | {'ans A3':>7}{'ans A10':>8}{'pmax':>6}{'cov f1':>7}{'cov f2':>7}{'cov f3':>7}{'n13 f3':>7} | {'KEY A3':>7}{'KEY A10':>9}{'pmax':>6}{'cov f1':>7}{'cov f3':>7}")
for T in Ts:
    a, s = res[T]["answer"], res[T]["semantic"]
    print(f"{T:>4}{res[T]['accuracy']:>7.3f} | {a['A3']:>7.3f}{a['A10']:>8.3f}{a['pmax']:>6.3f}{a['cov10_f1']:>7.3f}{a['cov10_f2']:>7.3f}{a['cov10_f3']:>7.3f}{a['cov13_f3']:>7.3f} | {s['A3']:>7.3f}{s['A10']:>9.5f}{s['pmax']:>6.3f}{s['cov10_f1']:>7.3f}{s['cov10_f3']:>7.3f}")
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "legend.fontsize": 6.5, "pdf.fonttype": 42})
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6), layout="constrained"); x = [float(T) for T in Ts]
ax[0].plot(x, [res[T]["answer"]["A3"] for T in Ts], "o-", ms=3, label="answer $\\hat A_U(3)$")
ax[0].plot(x, [res[T]["answer"]["cov10_f3"] for T in Ts], "s--", ms=3, label="answer coverage, $f=3$")
ax[0].plot(x, [res[T]["semantic"]["A3"] for T in Ts], "o-", ms=3, label="answer+KEY $\\hat A_U(3)$")
ax[0].plot(x, [res[T]["semantic"]["cov10_f1"] for T in Ts], "s--", ms=3, label="answer+KEY coverage, $f=1$")
ax[0].set(title="(a) Activation and coverage", xlabel="Temperature", ylim=(-0.03, 1.03), xticks=x); ax[0].legend()
ax[1].plot(x, [res[T]["accuracy"] for T in Ts], "o-", ms=3, color="#b34f36")
ax[1].set(title="(b) Single-sample accuracy", xlabel="Temperature", xticks=x)
fig.savefig("v6/fig/v12temp.pdf"); print("saved v6/fig/v12temp.pdf")
