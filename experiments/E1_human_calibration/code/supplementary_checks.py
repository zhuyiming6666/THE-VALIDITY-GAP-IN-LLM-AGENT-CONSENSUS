"""Supplementary calibration and instrument statistics reported in the paper.

Previously these numbers existed only as audit outputs outside the repository.  This script
regenerates them from files in the repository.

Run from the repository root:
    python experiments/E1_human_calibration/code/supplementary_checks.py

Outputs (E1_human_calibration/results/):
  instrument_archived_intervals.json  800-pair instrument frame (baseline/data/instrument2.json):
                                      SAME rates overall and conditional on archived answer equality,
                                      task-cluster percentile bootstrap (5,000 resamples, seed 20260915).
  annotation_sensitivity.json         weighted lexical false rejection on same-/cross-model pairs and
                                      with each annotator dropped (remaining two must agree).
  pairwise_collision.json             weighted pairwise SAME rates on the calibration frame (Table a2),
                                      human operation-axis ranges, and the instrument conditional
                                      SAME rates on all pairs and on resolved pairs only.
  sampling_design_check.json          strata populations recomputed from the stored replies and the
                                      sqrt allocation with a 25-pair floor.
"""
import collections, itertools, json, math, sys
from pathlib import Path
import numpy as np

EXP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotation_normalization import answer_class, norm_key  # noqa: E402

RES = EXP / "E1_human_calibration" / "results"
S = json.load(open(RES / "annotation_scoring.json"))
I = {p["pair_id"]: p for p in json.load(open(RES / "e1_calibration_instruments.json"))["per_pair"]}
PP = S["per_pair"]
for p in PP:
    p["ins"] = I[p["pair_id"]]


def wmean(sub, f):
    w = sum(p["weight"] for p in sub)
    return sum(p["weight"] * f(p) for p in sub) / w


# ---------------------------------------------------------------- 800-pair instrument frame
rows = json.load(open(EXP / "baseline/data/instrument2.json"))["results"]["_pairs"]
tasks = sorted({r["task"] for r in rows})
ix = np.random.default_rng(20260915).integers(len(tasks), size=(5000, len(tasks)))
intervals = {}
for ins in ("lexical", "embedding", "judge", "ref"):
    intervals[ins] = {}
    for cond in ("all", "same", "different"):
        keep = lambda r: cond == "all" or r["same_answer"] == (cond == "same")
        cells = np.array([[sum(r[ins] for r in rows if r["task"] == t and keep(r)),
                           sum(1 for r in rows if r["task"] == t and keep(r))] for t in tasks])
        draws = cells[ix].sum(axis=1)
        draws = draws[draws[:, 1] > 0]
        tot = cells.sum(0)
        intervals[ins][cond] = {"estimate": tot[0] / tot[1],
                                "ci95": list(np.quantile(draws[:, 0] / draws[:, 1], [0.025, 0.975])),
                                "n": int(tot[1])}
intervals["_note"] = "judge = gpt-4.1-nano, ref = gpt-4.1-mini judge; answer equality uses archived labels"
json.dump(intervals, open(RES / "instrument_archived_intervals.json", "w"), indent=2)

# ---------------------------------------------------------------- annotator / pair-type sensitivity


def lexical_fnr(sub, drop=None):
    num = den = 0.0
    for p in sub:
        if drop:
            vs = [v for k, v in p["votes"].items() if k != drop]
            a = vs[0]["answer"] if vs[0]["answer"] == vs[1]["answer"] else None
            m = vs[0]["method"] if vs[0]["method"] == vs[1]["method"] else None
        else:
            a, m = p["majority"]["answer"], p["majority"]["method"]
        if a not in ("SAME", "DIFF") or m not in ("SAME", "DIFF"):
            continue
        if a == "SAME" and m == "SAME":
            den += p["weight"]; num += p["weight"] * (not p["instruments"]["lexical"])
    return num / den


sens = {"all": lexical_fnr(PP),
        "same_model": lexical_fnr([p for p in PP if p["same_model_pair"]]),
        "cross_model": lexical_fnr([p for p in PP if not p["same_model_pair"]])}
for a in "ABC":
    sens[f"drop_{a}"] = lexical_fnr(PP, a)
json.dump({k: {"FNR": v} for k, v in sens.items()}, open(RES / "annotation_sensitivity.json", "w"), indent=2)

# ---------------------------------------------------------------- Table a2 and conditional rates
subsets = {"all": PP, "same-model": [p for p in PP if p["same_model_pair"]],
           "cross-model": [p for p in PP if not p["same_model_pair"]]}
for b in ("gsm8k", "mbpp", "mmlu"):
    subsets[b] = [p for p in PP if p["bench"] == b]
a2 = {}
for name, sub in subsets.items():
    lo = wmean(sub, lambda p: p["joint"] is True)
    un = wmean(sub, lambda p: p["joint"] is None)
    a2[name] = {"pairs": len(sub),
                "human_answer_same": wmean(sub, lambda p: p["majority"]["answer"] == "SAME"),
                "answer_equality_instrument": wmean(sub, lambda p: p["instruments"]["answer_only"]),
                "lexical": wmean(sub, lambda p: p["instruments"]["lexical"]),
                "human_answer_operation": [lo, lo + un], "unresolved": un}
cond = {}
for lab in ("SAME", "DIFF"):
    sub = [p for p in PP if p["majority"]["answer"] == lab]
    cond[f"human_operation_same_given_answer_{lab}"] = [
        wmean(sub, lambda p: p["majority"]["method"] == "SAME"),
        wmean(sub, lambda p: p["majority"]["method"] != "DIFF")]
inst = ("lexical", "embedding", "judge_nano", "judge_mini", "judge_nano_cot", "judge_nano_fewshot", "answer_only")
resolved = [p for p in PP if p["joint"] is not None]
for frame, pool in (("all_pairs", PP), ("resolved_pairs", resolved)):
    for lab in ("SAME", "DIFF"):
        sub = [p for p in pool if p["majority"]["answer"] == lab]
        cond[f"{frame}_given_human_answer_{lab}"] = {"n": len(sub), **{j: wmean(sub, lambda p, j=j: p["ins"][j]) for j in inst}}
agree = {j: sum(p["ins"][j] == p["ins"]["answer_only"] for p in resolved) for j in ("judge_nano", "judge_mini")}
split = [p for p in resolved if p["instruments"]["answer_only"] and p["joint"] is False]
json.dump({"table_a2": a2, "conditional": cond, "judge_vs_answer_equality_resolved": agree,
           "n_resolved": len(resolved),
           "answer_equal_but_human_split": {"n": len(split),
                                            "accepted_by_both_judges": sum(p["ins"]["judge_nano"] and p["ins"]["judge_mini"] for p in split)}},
          open(RES / "pairwise_collision.json", "w"), indent=2)

# ---------------------------------------------------------------- sampling frame and allocation
BENCH = lambda tid: {"gsm": "gsm8k", "mmlu": "mmlu", "mbpp": "mbpp"}[tid.split("_")[0]]
recs = collections.defaultdict(list)
for f in ("samples.jsonl", "samples2.jsonl"):
    for l in open(EXP / "baseline/data" / f):
        if l.strip():
            r = json.loads(l); recs[r["task_id"]].append(r)
pop = collections.Counter()
for t, rs in recs.items():
    b = BENCH(t)
    lab = [(answer_class(r.get("answer"), b), norm_key(r.get("key"))) for r in rs]
    for (a1, k1), (a2_, k2) in itertools.combinations(lab, 2):
        if k1 is None or k2 is None: pop["S4"] += 1
        elif a1 != a2_: pop["S2"] += 1
        elif k1 == k2: pop["S3"] += 1
        else: pop["S1"] += 1
recorded = {k.split("_")[0]: v["population"] for k, v in
            json.load(open(EXP / "E1_human_calibration/annotation/admin/pairs_key.json"))["meta"]["strata"].items()}
use = recorded  # allocation is defined on the recorded populations
root = {s: math.sqrt(n) for s, n in use.items()}
raw = {s: 25 + (300 - 25 * len(use)) * root[s] / sum(root.values()) for s in use}
alloc = {s: int(v) for s, v in raw.items()}
for s in sorted(use, key=lambda s: raw[s] - int(raw[s]), reverse=True)[:300 - sum(alloc.values())]:
    alloc[s] += 1
json.dump({"recomputed_population": dict(pop), "recomputed_total": sum(pop.values()),
           "recorded_population": recorded, "recorded_total": sum(recorded.values()),
           "allocation_from_recorded": alloc},
          open(RES / "sampling_design_check.json", "w"), indent=2)

print("intervals lexical all", intervals["lexical"]["all"])
print("sensitivity", {k: round(v, 3) for k, v in sens.items()})
for k, v in a2.items():
    print("a2", k, v["pairs"], round(v["human_answer_same"], 3), round(v["answer_equality_instrument"], 3),
          round(v["lexical"], 3), [round(x, 3) for x in v["human_answer_operation"]], round(v["unresolved"], 3))
print("judge agreement with answer equality", agree, "of", len(resolved))
print("population recomputed", dict(pop), sum(pop.values()), "recorded", recorded, "alloc", alloc)
