"""E1: measure p_max -- the mass of the largest semantic equivalence class in an
honest LLM agent's output distribution.  Premise of T1''' (i):

    P[classical validity antecedent holds]  <=  p_max^(n-f)   ->  0

Two equivalence granularities are reported:
  * answer-level   : verdict only (what majority voting sees)
  * semantic-level : (answer, decisive-step) pair  (the semantic core)

Usage:
  .venv/bin/python exp/measure_pmax.py --models gpt-4.1-nano,gpt-4o-mini --k 30
"""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import norm, parse
import argparse, json, os, re, collections, statistics, pathlib, threading
from concurrent.futures import ThreadPoolExecutor

SCHEMA_TIGHT = ("Answer the question. Reply with EXACTLY two lines and nothing else:\n"
                "ANSWER: <final answer, minimal form>\n"
                "KEY: <the single decisive step, <=10 words>")

SCHEMA_COT = ("Solve the question. Think step by step first, then end your reply with "
              "EXACTLY these two lines and nothing after them:\n"
              "ANSWER: <final answer, minimal form>\n"
              "KEY: <the single decisive step, <=10 words>")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gpt-4.1-nano")
    ap.add_argument("--taskfile", default="data/tasks.jsonl")
    ap.add_argument("--tasks", type=int, default=150)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--cot", action="store_true", help="allow reasoning before the schema")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default="results/pmax.json")
    ap.add_argument("--dump", default="", help="raw samples jsonl (default: <out>.samples.jsonl; always written)")
    ap.add_argument("--env", default=".env", help="dotenv file with the provider credentials")
    ap.add_argument("--key-var", default="OPENAI_API_KEY")
    ap.add_argument("--url-var", default="OPENAI_BASE_URL")
    ap.add_argument("--no-think", action="store_true", help="send enable_thinking=false (Qwen3 hybrid models)")
    args = ap.parse_args()

    from dotenv import load_dotenv; load_dotenv(args.env)
    from openai import OpenAI
    kw = {"api_key": os.environ[args.key_var], "max_retries": 5, "timeout": 120}
    if os.getenv(args.url_var): kw["base_url"] = os.environ[args.url_var]
    client = OpenAI(**kw)
    extra = {"extra_body": {"enable_thinking": False}} if args.no_think else {}

    SCHEMA = SCHEMA_COT if args.cot else SCHEMA_TIGHT
    if not args.dump:
        args.dump = args.out.rsplit(".", 1)[0] + ".samples.jsonl"
    tasks = [json.loads(l) for l in open(args.taskfile) if l.strip()][:args.tasks]
    usage = collections.Counter(); lock = threading.Lock(); done = [0]
    total = len(tasks) * args.k * len(args.models.split(","))

    def one(model, q):
        try:
            r = client.chat.completions.create(
                model=model, temperature=args.temp, max_tokens=(700 if args.cot else 200),
                messages=[{"role": "system", "content": SCHEMA},
                          {"role": "user", "content": q}], **extra)
            with lock:
                usage["in"] += r.usage.prompt_tokens; usage["out"] += r.usage.completion_tokens
                done[0] += 1
                if done[0] % 500 == 0: print(f"  ..{done[0]}/{total}", flush=True)
            txt = r.choices[0].message.content
            return parse(txt) + (txt,)
        except Exception as e:
            with lock: usage["err"] += 1; done[0] += 1
            return (None, None, None)

    all_rows = {}
    for model in args.models.split(","):
        print(f"\n=== {model} ===", flush=True)
        jobs = [(t, _) for t in tasks for _ in range(args.k)]
        with ThreadPoolExecutor(args.workers) as ex:
            res = list(ex.map(lambda j: one(model, j[0]["question"]), jobs))
        rows = []
        for i, t in enumerate(tasks):
            chunk = res[i*args.k:(i+1)*args.k]
            ans = [norm(a) for a, _, _ in chunk]; keys = [norm(k) for _, k, _ in chunk]
            va = [a for a in ans if a is not None]
            if not va: continue
            ca = collections.Counter(va)
            cs = collections.Counter(z for z in zip(ans, keys) if z[0] is not None)
            rows.append({"id": t["id"], "bench": t["bench"], "gold": norm(t["answer"]),
                         "counts_answer": sorted(ca.values(), reverse=True),
                         "counts_semantic": sorted(cs.values(), reverse=True),
                         "n_valid": len(va),
                         "p_max_answer": ca.most_common(1)[0][1] / len(va),
                         "support_answer": len(ca),
                         "p_max_semantic": cs.most_common(1)[0][1] / len(va),
                         "support_semantic": len(cs),
                         "acc": sum(a == norm(t["answer"]) for a in va) / len(va)})
        all_rows[model] = rows
        if args.dump:
            with open(args.dump, "a") as fh:
                for i, t in enumerate(tasks):
                    for a, k, txt in res[i*args.k:(i+1)*args.k]:
                        fh.write(json.dumps({"model": model, "task_id": t["id"],
                                             "question": t["question"], "schema": "cot" if args.cot else "tight",
                                             "answer": a, "key": k, "raw": txt}, ensure_ascii=False) + "\n")

    keys = ("p_max_answer", "p_max_semantic", "support_answer", "support_semantic", "acc")
    summary = {m: {k: round(statistics.mean(r[k] for r in rs), 4) for k in keys}
               | {"n_tasks": len(rs)} for m, rs in all_rows.items()}
    def agg(rs, k):
        v = [r[k] for r in rs]
        return round(statistics.mean(v), 4) if v else None
    per_bench = {m: {b: ({k: agg([r for r in rs if r["bench"] == b], k) for k in keys}
                         | {"n": sum(r["bench"] == b for r in rs)})
                     for b in ("gsm8k", "mmlu", "mbpp")} for m, rs in all_rows.items()}

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": vars(args), "usage": dict(usage), "summary": summary,
               "per_bench": per_bench, "rows": all_rows}, open(args.out, "w"), indent=2)

    print("\n" + "=" * 78)
    print(f"{'model':<16}{'p_max_ans':>11}{'p_max_sem':>11}{'supp_sem':>10}{'acc':>8}")
    for m, s in summary.items():
        print(f"{m:<16}{s['p_max_answer']:>11.3f}{s['p_max_semantic']:>11.3f}"
              f"{s['support_semantic']:>10.2f}{s['acc']:>8.3f}")
    print("\nT1''' (i)  P[all-honest-agree] = p_max^n     [f=0]")
    print(f"{'model / level':<26}" + "".join(f"{f'n={n}':>10}" for n in (3, 5, 7, 10, 20)))
    for m, s in summary.items():
        for lvl in ("answer", "semantic"):
            p = s[f"p_max_{lvl}"]
            print(f"{m + ' / ' + lvl:<26}" + "".join(f"{p**n:>10.4f}" for n in (3,5,7,10,20)))
    print("\nusage:", dict(usage))

if __name__ == "__main__":
    main()
