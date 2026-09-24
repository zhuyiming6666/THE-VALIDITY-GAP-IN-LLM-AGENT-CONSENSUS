"""E3: parameter-free prediction of round-0 unanimity for a mixed group, and scoring of the
observed debate run. Usage:
  python exp/e3_predict_mixture.py results/v11/e3_mixed.json
Prediction: A = sum_c prod_m [n_{m,c}]_{k_m} / [N_m]_{k_m} (unbiased under independence),
where k_m is the number of slots held by model m, from the stored single-agent replies of
the same 50 GSM8K tasks. Homogeneous nano predictions are printed as a consistency check
against Table 1 (0.873 / 0.822 / 0.786 / 0.748).
"""
import sys, json, collections
from pathlib import Path
EXPERIMENTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENTS / "baseline/code"))
from scoring import answer_class

MODELS = ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1-mini"]
recs = collections.defaultdict(lambda: collections.defaultdict(list))
for f in (EXPERIMENTS / "baseline/data/samples.jsonl",
          EXPERIMENTS / "baseline/data/samples2.jsonl"):
    for l in open(f):
        r = json.loads(l); recs[r["model"]][r["task_id"]].append(r)

def falling(a, h):
    p = 1
    for i in range(h): p *= (a - i)
    return p

def counts(model, tid):
    c = collections.Counter(answer_class(r["answer"], "gsm8k") for r in recs[model][tid])
    c.pop(None, None); return c

def predict(tid, slots):  # slots: dict model -> k
    cs = {m: counts(m, tid) for m in slots}
    labels = set().union(*[set(c) for c in cs.values()])
    tot = 0.0
    for lab in labels:
        p = 1.0
        for m, k in slots.items():
            N = sum(cs[m].values())
            if N < k: return None
            p *= falling(cs[m].get(lab, 0), k) / falling(N, k)
        tot += p
    return tot

tasks = [json.loads(l) for l in open(EXPERIMENTS / "baseline/data/tasks.jsonl") if l.strip()][:50]
tids = [t["id"] for t in tasks]
assert all(t.startswith("gsm") for t in tids), "first 50 tasks must be GSM8K"

def mean(v): v = [x for x in v if x is not None]; return sum(v) / len(v)
print("homogeneous nano check:", {n: round(mean([predict(t, {"gpt-4.1-nano": n}) for t in tids]), 3) for n in (3, 5, 7, 10)})
MIX = {"gpt-4.1-nano": 3, "gpt-4o-mini": 3, "gpt-4.1-mini": 4}
pred_mix = mean([predict(t, MIX) for t in tids])
print(f"mixed 3/3/4 prediction A(10) = {pred_mix:.3f}")
for m in MODELS:
    print(f"  homogeneous {m} A(10) = {mean([predict(t, {m: 10}) for t in tids]):.3f}")

if len(sys.argv) > 1:
    d = json.load(open(sys.argv[1])); raw = d["raw"]
    gold = {t["id"]: answer_class(t["answer"], "gsm8k") for t in tasks}
    N = una = maj_ok = 0; una_by_model = collections.Counter()
    for rec in raw:
        ans = [answer_class(a[0], "gsm8k") for a in rec["rounds"][0]]
        if any(a is None for a in ans): continue
        N += 1; una += len(set(ans)) == 1
        maj_ok += collections.Counter(ans).most_common(1)[0][0] == gold[rec["task_id"]]
    obs = una / N
    print(f"observed round-0 unanimity (mixed) = {obs:.3f}  N={N}  residual = {obs - pred_mix:+.3f}  majority acc = {maj_ok/N:.3f}")
    json.dump({"prediction": pred_mix, "observed": obs, "N": N, "residual": obs - pred_mix, "majority_acc": maj_ok / N,
               "slots": MIX, "homogeneous_predictions": {m: mean([predict(t, {m: 10}) for t in tids]) for m in MODELS}},
              open(sys.argv[1].replace(".json", "_analysis.json"), "w"), indent=1)
