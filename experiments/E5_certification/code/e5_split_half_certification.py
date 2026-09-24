"""E5: undersampling sensitivity of the certification coverage (Table 3), zero API.
Two re-estimates per (model, task, partition), n=10, f in 0..3, h=10-f:
  split-half : designated mode from half A of the stored replies; the event (honest lead > 2f
               on an h-subset) evaluated by exact enumeration / subsampling over h-subsets of
               half B, i.e. on fresh draws from the true law (U-statistic-style, unbiased).
  dirichlet  : law drawn from Dirichlet(counts + 1/(m+1)) over observed classes plus one unseen
               class; histogram drawn from that law (posterior predictive under a Jeffreys-like prior).
Plug-in (Table 3) is recomputed alongside for reference. Tasks: the 123 common set.
"""
import sys, json, collections, random, itertools, math
from pathlib import Path
EXPERIMENTS = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS / "baseline/code"))
from scoring import answer_class, semantic_class

MODELS = ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1-mini"]
BENCH = lambda tid: {"gsm": "gsm8k", "mmlu": "mmlu", "mbpp": "mbpp"}[tid.split("_")[0]]
rng = random.Random(20260919)
common = json.load(open(EXPERIMENTS / "baseline/results/common_tasks_123.json"))
recs = collections.defaultdict(lambda: collections.defaultdict(list))
for f in (EXPERIMENTS / "baseline/data/samples.jsonl",
          EXPERIMENTS / "baseline/data/samples2.jsonl"):
    for l in open(f):
        r = json.loads(l); recs[r["model"]][r["task_id"]].append(r)

def classes(model, tid, part):
    b = BENCH(tid); out = []
    for r in recs[model][tid]:
        a = answer_class(r.get("answer"), b)
        if a is None: continue
        out.append(a if part == "answer" else semantic_class(r.get("answer"), r.get("key"), b))
    return out

def lead_ok(hist, star, f):
    rival = max([v for k, v in hist.items() if k != star] + [0])
    return hist.get(star, 0) - rival > 2 * f

def plugin(cls, f, h, draws=4000):
    c = collections.Counter(cls); labels = list(c); probs = [c[x] / len(cls) for x in labels]
    star = max(labels, key=lambda x: (c[x], -labels.index(x)))
    return sum(lead_ok(collections.Counter(rng.choices(labels, probs, k=h)), star, f) for _ in range(draws)) / draws

def split_half(cls, f, h, max_subsets=3000):
    cls = list(cls); rng.shuffle(cls); k = len(cls) // 2
    A, B = cls[:k], cls[k:]
    if len(B) < h or not A: return None
    cA = collections.Counter(A); star = max(cA, key=lambda x: (cA[x], -list(cA).index(x)))
    if math.comb(len(B), h) <= max_subsets:
        subs = itertools.combinations(range(len(B)), h); n = math.comb(len(B), h)
        return sum(lead_ok(collections.Counter(B[i] for i in idx), star, f) for idx in subs) / n
    return sum(lead_ok(collections.Counter(rng.sample(B, h)), star, f) for _ in range(max_subsets)) / max_subsets

def dirichlet(cls, f, h, draws=4000):
    c = collections.Counter(cls); labels = list(c) + ["__unseen__"]; m = len(labels)
    alpha = [c.get(x, 0) + 1.0 / m for x in labels]
    star = max(c, key=lambda x: (c[x], -list(c).index(x)))
    ok = 0
    for _ in range(draws):
        g = [rng.gammavariate(a, 1.0) for a in alpha]; s = sum(g); p = [x / s for x in g]
        ok += lead_ok(collections.Counter(rng.choices(labels, p, k=h)), star, f)
    return ok / draws

out = {}
print(f"{'partition':<9}{'f':>3}{'plug-in':>9}{'split-half':>12}{'dirichlet':>11}{'n tasks(sh)':>12}")
for part in ("answer", "semantic"):
    for f in (0, 1, 2, 3):
        h = 10 - f; P = []; S = []; D = []
        for t in common:
            pm, sm, dm = [], [], []
            for m in MODELS:
                cls = classes(m, t, part)
                pm.append(plugin(cls, f, h)); dm.append(dirichlet(cls, f, h))
                sh = split_half(cls, f, h)
                if sh is not None: sm.append(sh)
            P.append(sum(pm) / 3); D.append(sum(dm) / 3)
            if len(sm) == 3: S.append(sum(sm) / 3)
        row = {"plugin": sum(P) / len(P), "split_half": sum(S) / len(S), "dirichlet": sum(D) / len(D), "n_tasks_split_half": len(S)}
        out[f"{part}|f={f}"] = row
        print(f"{part:<9}{f:>3}{row['plugin']:>9.3f}{row['split_half']:>12.3f}{row['dirichlet']:>11.3f}{len(S):>12}")
json.dump(out, open(PACKAGE / "results/e5_certification_sensitivity.json", "w"), indent=1)
