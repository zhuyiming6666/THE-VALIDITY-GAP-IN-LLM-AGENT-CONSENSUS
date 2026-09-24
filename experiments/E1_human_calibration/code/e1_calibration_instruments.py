"""E1: score the LLM judges and the embedding rule on the 300 human-calibrated pairs.

Inputs
  v4/annotation/pairs_blind.tsv                 pair texts (question, A/B answer+KEY)
  v7/audit/annotation_scoring.rechecked.json    human joint reference, weights, strata
Instruments (identical prompt / threshold to exp/instrument_error.py)
  judge_nano, judge_mini            : JUDGE system prompt, temperature 0, "SAME" in reply
  judge_nano_cot, judge_nano_fewshot: prompt variants (adjustment 4 of the plan)
  embedding                         : cosine(text-embedding-3-small on "answer || key") > 0.90
Reference: resolved joint majority (both axes SAME) -> positive; unresolved pairs excluded
from error denominators, exactly as in the calibration scorer.
"""
import argparse, csv, json, math, os, random, threading, collections, pathlib, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instrument_error import JUDGE

COT = ("Two agents answered the same question. First, in one sentence, compare the decisive "
       "step each agent used. Second, compare their final answers. Then output a final line "
       "containing exactly one word: SAME if they express the same solution (same final answer "
       "reached by the same decisive step, wording differences ignored), otherwise DIFFERENT.")
FEWSHOT = JUDGE + ("\n\nExamples:\n"
    "Agent 1 -- ANSWER: 72 | KEY: add April and May sales\n"
    "Agent 2 -- ANSWER: 72 | KEY: 48 + 24 = 72\n-> SAME (same step, one in words, one as arithmetic)\n"
    "Agent 1 -- ANSWER: 3 | KEY: use the quadratic formula\n"
    "Agent 2 -- ANSWER: 3 | KEY: factor the polynomial\n-> DIFFERENT (same answer, different decisive step)\n"
    "Agent 1 -- ANSWER: 15 | KEY: multiply hours by rate\n"
    "Agent 2 -- ANSWER: 18 | KEY: multiply hours by rate\n-> DIFFERENT (different final answer)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind", default="v4/annotation/pairs_blind.tsv")
    ap.add_argument("--ref", default="v7/audit/annotation_scoring.rechecked.json")
    ap.add_argument("--out", default="results/v11/e1_calibration_instruments.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--tau", type=float, default=0.90)
    args = ap.parse_args()

    from dotenv import load_dotenv; load_dotenv(".env")
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=5)

    rows = list(csv.DictReader(open(args.blind, encoding="utf-8"), delimiter="\t"))
    ref = json.load(open(args.ref))
    per = {p["pair_id"]: p for p in ref["per_pair"]}
    assert len(rows) == 300 and all(r["pair_id"] in per for r in rows)

    lock = threading.Lock(); usage = collections.Counter()

    def ask(model, system, r, max_tokens):
        u = (f"Question:\n{r['question']}\n\n"
             f"Agent 1 -- ANSWER: {r['A_answer']} | KEY: {r['A_key']}\n"
             f"Agent 2 -- ANSWER: {r['B_answer']} | KEY: {r['B_key']}")
        try:
            resp = client.chat.completions.create(model=model, temperature=0, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": u}])
            txt = (resp.choices[0].message.content or "")
            with lock:
                usage["in"] += resp.usage.prompt_tokens; usage["out"] += resp.usage.completion_tokens
            if max_tokens > 10:  # CoT: decide on the last line
                last = [l for l in txt.strip().splitlines() if l.strip()][-1].upper()
                return ("SAME" in last) and ("DIFFERENT" not in last)
            return "SAME" in txt.upper()
        except Exception as e:
            with lock: usage["err"] += 1
            return None

    def embed(texts):
        out = []
        for i in range(0, len(texts), 128):
            resp = client.embeddings.create(model="text-embedding-3-small", input=texts[i:i+128])
            out += [d.embedding for d in resp.data]
            with lock: usage["embed_in"] += resp.usage.prompt_tokens
        return out

    def cos(u, v):
        d = sum(x*y for x, y in zip(u, v))
        return d / (math.sqrt(sum(x*x for x in u)) * math.sqrt(sum(y*y for y in v)) + 1e-12)

    txts = [f"{r[s+'_answer']} || {r[s+'_key']}" for r in rows for s in ("A", "B")]
    E = embed(txts)
    preds = {"embedding": [cos(E[2*i], E[2*i+1]) > args.tau for i in range(len(rows))]}
    cosines = [cos(E[2*i], E[2*i+1]) for i in range(len(rows))]
    with ThreadPoolExecutor(args.workers) as ex:
        preds["judge_nano"] = list(ex.map(lambda r: ask("gpt-4.1-nano", JUDGE, r, 5), rows))
        preds["judge_mini"] = list(ex.map(lambda r: ask("gpt-4.1-mini", JUDGE, r, 5), rows))
        preds["judge_nano_cot"] = list(ex.map(lambda r: ask("gpt-4.1-nano", COT, r, 160), rows))
        preds["judge_nano_fewshot"] = list(ex.map(lambda r: ask("gpt-4.1-nano", FEWSHOT, r, 5), rows))
    # frozen instruments from the key
    preds["lexical"] = [per[r["pair_id"]]["instruments"]["lexical"] for r in rows]
    preds["answer_only"] = [per[r["pair_id"]]["instruments"]["answer_only"] for r in rows]

    def score(idx, pred):
        """weighted confusion on resolved pairs among idx; returns dict"""
        W = TP = FP = FN = TN = 0.0; n = tp = fp = fn = tn = 0; cov_w = tot_w = 0.0
        for i in idx:
            p = per[rows[i]["pair_id"]]; w = p["weight"]; tot_w += w
            if pred[i] is None or p["joint"] is None: continue
            cov_w += w; n += 1
            if pred[i] and p["joint"]: TP += w; tp += 1
            elif pred[i] and not p["joint"]: FP += w; fp += 1
            elif (not pred[i]) and p["joint"]: FN += w; fn += 1
            else: TN += w; tn += 1
        def r(a, b): return a / b if b > 0 else None
        return {"n_resolved": n, "counts": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
                "error": r(FP + FN, TP + FP + FN + TN), "false_accept": r(FP, FP + TN),
                "false_reject": r(FN, FN + TP), "precision": r(TP, TP + FP),
                "coverage": r(cov_w, tot_w),
                "unweighted": {"error": r(fp + fn, n), "false_accept": r(fp, fp + tn), "false_reject": r(fn, fn + tp)}}

    def cond_rates(idx, pred):
        """instrument SAME rate given human answer-majority SAME / DIFF (weighted)"""
        out = {}
        for lab in ("SAME", "DIFF"):
            num = den = 0.0
            for i in idx:
                p = per[rows[i]["pair_id"]]
                if p["majority"]["answer"] != lab or pred[i] is None: continue
                den += p["weight"]; num += p["weight"] * bool(pred[i])
            out[f"SAME_given_answer_{lab}"] = num / den if den else None
        return out

    rng = random.Random(args.seed)
    strata = collections.defaultdict(list)
    for i, r in enumerate(rows): strata[per[r["pair_id"]]["stratum"]].append(i)

    def boot(pred, stat):
        vals = []
        for _ in range(args.boot):
            idx = [j for s, ids in strata.items() for j in rng.choices(ids, k=len(ids))]
            v = score(idx, pred)[stat]
            if v is not None: vals.append(v)
        vals.sort()
        return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]] if vals else None

    all_idx = list(range(len(rows)))
    same_idx = [i for i in all_idx if per[rows[i]["pair_id"]]["same_model_pair"]]
    report = {"config": vars(args), "usage": dict(usage), "instruments": {}}
    for name, pred in preds.items():
        s = score(all_idx, pred)
        s["ci95"] = {k: boot(pred, k) for k in ("error", "false_accept", "false_reject")}
        s["same_model"] = score(same_idx, pred)
        s["by_bench"] = {b: score([i for i in all_idx if per[rows[i]["pair_id"]]["bench"] == b], pred)
                         for b in sorted({per[r["pair_id"]]["bench"] for r in rows})}
        s["conditional"] = cond_rates(all_idx, pred)
        s["marginal_SAME_weighted"] = sum(per[rows[i]["pair_id"]]["weight"] * bool(pred[i]) for i in all_idx if pred[i] is not None) / \
                                      sum(per[rows[i]["pair_id"]]["weight"] for i in all_idx if pred[i] is not None)
        s["n_none"] = sum(p is None for p in pred)
        report["instruments"][name] = s
    report["per_pair"] = [{"pair_id": r["pair_id"], "cosine": round(cosines[i], 4),
                           **{k: preds[k][i] for k in preds}} for i, r in enumerate(rows)]
    report["usage"] = dict(usage)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)

    print(f"{'instrument':<20}{'n':>5}{'error':>8}{'FA':>8}{'FR':>8}{'prec':>8}{'S|ansS':>9}{'S|ansD':>9}{'none':>6}")
    for name, s in report["instruments"].items():
        c = s["conditional"]
        f = lambda v: f"{v:8.3f}" if v is not None else f"{'--':>8}"
        g = lambda v: f"{v:9.3f}" if v is not None else f"{'--':>9}"
        print(f"{name:<20}{s['n_resolved']:>5}{f(s['error'])}{f(s['false_accept'])}{f(s['false_reject'])}{f(s['precision'])}"
              f"{g(c['SAME_given_answer_SAME'])}{g(c['SAME_given_answer_DIFF'])}{s['n_none']:>6}")
    print("usage:", dict(usage))


if __name__ == "__main__":
    main()
