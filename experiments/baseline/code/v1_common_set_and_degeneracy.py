"""Rebuild the 123-task common set with the frozen v4 normaliser and run check V1:
by Remark 7, designated-mode coverage at n=10, f=3 (h=7) must equal E_task[p(c*)^7].
Also recomputes the f=0..3 designated-mode coverage by Monte Carlo (30,000 draws) for
comparison with Table 3 (answer 0.970/0.948/0.897/0.843; answer+KEY 0.225/0.069/0.018/0.004),
and the U-statistic A(3) headline values (nano answer 0.855, answer+KEY 0.0125).
"""
import sys, json, collections, random, math
sys.path.insert(0, "v4/code")
from annotation_normalization import answer_class, semantic_class

MODELS = ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1-mini"]
BENCH = lambda tid: {"gsm": "gsm8k", "mmlu": "mmlu", "mbpp": "mbpp"}[tid.split("_")[0]]

recs = collections.defaultdict(lambda: collections.defaultdict(list))  # model -> task -> records
for f in ("results/samples.jsonl", "results/samples2.jsonl"):
    for l in open(f):
        r = json.loads(l); recs[r["model"]][r["task_id"]].append(r)

def classes(model, tid, partition):
    b = BENCH(tid); out = []
    for r in recs[model][tid]:
        a = answer_class(r.get("answer"), b)
        if a is None: continue
        out.append(a if partition == "answer" else semantic_class(r.get("answer"), r.get("key"), b))
    return out

tasks = sorted({t for m in MODELS for t in recs[m]})
common = [t for t in tasks if all(len(classes(m, t, "answer")) >= 10 for m in MODELS)]
print("common tasks:", len(common), "(paper: 123)")
json.dump(common, open("results/v11/common_tasks_123.json", "w"))

def falling(a, h):
    p = 1
    for i in range(h): p *= (a - i)
    return p

def A_U(cls, h):
    k = len(cls); c = collections.Counter(cls)
    return sum(falling(v, h) for v in c.values()) / falling(k, h) if k >= h else None

for part in ("answer", "semantic"):
    vals = [A_U(classes("gpt-4.1-nano", t, part), 3) for t in common]
    vals = [v for v in vals if v is not None]
    print(f"nano A_U(3) {part}: {sum(vals)/len(vals):.4f}")

rng = random.Random(20260919)
def coverage(cls, f, h, draws=10000):
    c = collections.Counter(cls); labels = list(c); probs = [c[x] / len(cls) for x in labels]
    star = max(range(len(labels)), key=lambda i: (probs[i], -i))  # deterministic mode
    ok = 0
    for _ in range(draws):
        hist = collections.Counter(rng.choices(range(len(labels)), probs, k=h))
        lead = hist[star] - max([hist[i] for i in range(len(labels)) if i != star] + [0])
        ok += lead > 2 * f
    return ok / draws, probs[star] ** h

print(f"{'partition':<10}{'f':>3}{'MC coverage':>13}{'E[p*^h] (f=3 only)':>20}")
for part in ("answer", "semantic"):
    for f in (0, 1, 2, 3):
        h = 10 - f; mc = []; an = []
        for t in common:
            per_model = [coverage(classes(m, t, part), f, h) for m in MODELS]
            mc.append(sum(x[0] for x in per_model) / 3); an.append(sum(x[1] for x in per_model) / 3)
        line = f"{part:<10}{f:>3}{sum(mc)/len(mc):>13.3f}"
        if f == 3: line += f"{sum(an)/len(an):>20.3f}"
        print(line)
