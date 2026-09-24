"""E6: analyse the nano forced-attacker run with the frozen v4 normaliser.
Usage: python exp/e6_analyse_forced.py results/v11/e6_forced_nano.json [results/v3/e4_byz.json]
Reports per (f, round): all-participant unanimity, honest-only unanimity, honest-majority
accuracy, target adoption (attacker slots equal to the designated wrong value), and, if the
archived prompted run is given, the same cells for the prompted attacker on the same tasks.
"""
import sys, json, collections
from pathlib import Path
EXPERIMENTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENTS / "baseline/code"))
from scoring import answer_class

def cells(raw, label):
    agg = collections.defaultdict(collections.Counter)
    for rec in raw:
        f = rec["f"]; gold = answer_class(rec["gold"], "gsm8k")
        wrong = answer_class(rec.get("wrong", ""), "gsm8k") if rec.get("wrong") else None
        for rnd, state in enumerate(rec["rounds"]):
            ans = [answer_class(a[0], "gsm8k") for a in state]
            if any(a is None for a in ans[f:]): continue
            c = agg[(f, rnd)]; c["N"] += 1
            c["all_unan"] += len(set(ans)) == 1
            hon = ans[f:]; c["hon_unan"] += len(set(hon)) == 1
            c["hon_maj_ok"] += collections.Counter(hon).most_common(1)[0][0] == gold
            if f:
                c["slots"] += f
                c["target"] += sum(1 for a in ans[:f] if wrong is not None and a == wrong)
                c["dissent"] += sum(1 for a in ans[:f] if a is None or a not in hon)
    out = {}
    print(f"[{label}] {'f':>2}{'rnd':>4}{'N':>5}{'all_unan':>10}{'hon_unan':>10}{'hon_maj':>9}{'target/slots':>14}{'dissent/slots':>15}")
    for (f, rnd), c in sorted(agg.items()):
        N = c["N"]; row = dict(N=N, all_unanimity=c["all_unan"]/N, honest_unanimity=c["hon_unan"]/N,
                               honest_majority_acc=c["hon_maj_ok"]/N,
                               target_adoption=(c["target"]/c["slots"]) if c["slots"] else None,
                               dissent=(c["dissent"]/c["slots"]) if c["slots"] else None)
        out[f"{f}|{rnd}"] = row
        t = f"{c['target']}/{c['slots']}" if c["slots"] else "--"; d = f"{c['dissent']}/{c['slots']}" if c["slots"] else "--"
        print(f"{'':<{len(label)+3}}{f:>2}{rnd:>4}{N:>5}{row['all_unanimity']:>10.3f}{row['honest_unanimity']:>10.3f}{row['honest_majority_acc']:>9.3f}{t:>14}{d:>15}")
    return out

d = json.load(open(sys.argv[1]))
res = {"forced": cells(d["raw"], "forced nano"), "usage": d.get("usage")}
if len(sys.argv) > 2:
    p = json.load(open(sys.argv[2]))
    for rec in p["raw"]:  # archived prompted run has no stored wrong value; recompute it
        g = rec["gold"].strip()
        try:
            v = float(g); rec["wrong"] = str(int(v * 2)) if v == int(v) else str(round(v * 2, 2))
        except ValueError: rec["wrong"] = None
    res["prompted"] = cells(p["raw"], "prompted nano (archived)")
json.dump(res, open(sys.argv[1].replace(".json", "_analysis.json"), "w"), indent=1)
