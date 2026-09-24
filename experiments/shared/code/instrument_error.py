"""E2: measure the equivalence-instrument error eps of Definition 3.4.

Theorem 5.3 takes eps as input; this measures it for three instruments on the
same honest outputs used in E1, at the SEMANTIC granularity (answer + decisive
step), where no canonical form exists.

Instruments
  lexical    : normalised string equality on (answer, key)      -- training-free
  embedding  : cosine(text-embedding-3-small) > tau
  judge      : an LLM asked whether two rationales are equivalent

Reference relation: a stronger adjudicator model.  This is a limitation, not a
ground truth, and is reported as such: eps is measured RELATIVE to the best
available adjudicator, so the true eps of every instrument is at least as large.

Usage:
  .venv/bin/python exp/instrument_error.py --pairs 600
"""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import norm, parse
import argparse, json, os, re, random, collections, threading, pathlib
from concurrent.futures import ThreadPoolExecutor

JUDGE = ("Two agents answered the same question. Decide whether they express the "
         "SAME solution: the same final answer reached by the same decisive step. "
         "Differences in wording do not matter. Reply with exactly one word: "
         "SAME or DIFFERENT.")




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="results/samples.jsonl",
                    help="jsonl of {task_id,question,answer,key} produced by --dump")
    ap.add_argument("--pairs", type=int, default=600)
    ap.add_argument("--judge-model", default="gpt-4.1-nano")
    ap.add_argument("--ref-model", default="gpt-4.1-mini")
    ap.add_argument("--embed-model", default="text-embedding-3-small")
    ap.add_argument("--tau", type=float, default=0.90)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="results/instrument.json")
    args = ap.parse_args()

    from dotenv import load_dotenv; load_dotenv(".env")
    from openai import OpenAI
    kw = {"api_key": os.environ["OPENAI_API_KEY"], "max_retries": 5}
    if os.getenv("OPENAI_BASE_URL"): kw["base_url"] = os.environ["OPENAI_BASE_URL"]
    client = OpenAI(**kw)

    recs = [json.loads(l) for l in open(args.samples) if l.strip()]
    by_task = collections.defaultdict(list)
    for r in recs: by_task[r["task_id"]].append(r)

    rng = random.Random(0)
    pairs = []
    tids = [t for t, v in by_task.items() if len(v) >= 2]
    while len(pairs) < args.pairs and tids:
        t = rng.choice(tids)
        a, b = rng.sample(by_task[t], 2)
        pairs.append((t, a, b))

    lock = threading.Lock(); usage = collections.Counter()

    def ask(model, a, b):
        u = (f"Question:\n{a['question']}\n\n"
             f"Agent 1 -- ANSWER: {a['answer']} | KEY: {a['key']}\n"
             f"Agent 2 -- ANSWER: {b['answer']} | KEY: {b['key']}")
        try:
            r = client.chat.completions.create(
                model=model, temperature=0, max_tokens=5,
                messages=[{"role": "system", "content": JUDGE},
                          {"role": "user", "content": u}])
            with lock:
                usage["in"] += r.usage.prompt_tokens; usage["out"] += r.usage.completion_tokens
            return "SAME" in (r.choices[0].message.content or "").upper()
        except Exception:
            with lock: usage["err"] += 1
            return None

    def embed(texts):
        out = []
        for i in range(0, len(texts), 128):
            r = client.embeddings.create(model=args.embed_model, input=texts[i:i+128])
            out += [d.embedding for d in r.data]
        return out

    # --- instruments ---
    lex = [norm(a["answer"]) == norm(b["answer"]) and norm(a["key"]) == norm(b["key"])
           for _, a, b in pairs]

    txts = [f"{x['answer']} || {x['key']}" for _, a, b in pairs for x in (a, b)]
    E = embed(txts)
    import math
    def cos(u, v):
        d = sum(x*y for x, y in zip(u, v))
        return d / (math.sqrt(sum(x*x for x in u)) * math.sqrt(sum(y*y for y in v)) + 1e-12)
    emb = [cos(E[2*i], E[2*i+1]) > args.tau for i in range(len(pairs))]

    with ThreadPoolExecutor(args.workers) as ex:
        jud = list(ex.map(lambda p: ask(args.judge_model, p[1], p[2]), pairs))
        ref = list(ex.map(lambda p: ask(args.ref_model, p[1], p[2]), pairs))

    res = {}
    for name, pred in (("lexical", lex), ("embedding", emb), ("judge", jud)):
        ok = [(p, r) for p, r in zip(pred, ref) if r is not None and p is not None]
        n = len(ok)
        err = sum(p != r for p, r in ok) / n
        fp = sum(p and not r for p, r in ok) / max(sum(not r for _, r in ok), 1)
        fn = sum((not p) and r for p, r in ok) / max(sum(r for _, r in ok), 1)
        res[name] = {"n": n, "eps": round(err, 4), "fp": round(fp, 4), "fn": round(fn, 4)}

    res["_pairs"] = [{"task": t, "a": [a["answer"], a["key"]], "b": [b["answer"], b["key"]],
                      "same_answer": norm(a["answer"]) == norm(b["answer"]),
                      "lexical": l, "embedding": e, "judge": j, "ref": r}
                     for (t, a, b), l, e, j, r in zip(pairs, lex, emb, jud, ref)]
    res["_ref_same_rate"] = round(sum(bool(r) for r in ref if r is not None)
                                  / max(sum(r is not None for r in ref), 1), 4)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": vars(args), "usage": dict(usage), "results": res},
              open(args.out, "w"), indent=2)
    print(f"{'instrument':<12}{'n':>6}{'eps':>9}{'fp':>9}{'fn':>9}")
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"{k:<12}{v['n']:>6}{v['eps']:>9.3f}{v['fp']:>9.3f}{v['fn']:>9.3f}")
    print("reference SAME-rate:", res["_ref_same_rate"], "| usage:", dict(usage))


if __name__ == "__main__":
    main()
