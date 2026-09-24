"""E3: benign-failure reproduction (the paper's main figure).

Run a standard multi-agent debate with f=0 (NO Byzantine agents) at
n in {3,5,7,10} and measure, per round, whether all n agents agree.

Theorem 4.1 predicts the round-0 all-agree rate is A(n) = sum_c p(c)^n,
estimated from E1.  Two outcomes, both informative:

  * agreement stays near A(n)  -> vacuity directly explains benign failure.
  * agreement rises above A(n) -> agreement is manufactured by interaction, not
    by shared honest opinion; we then check whether the agreed class is
    CORRECT.  If consensus accuracy does not rise with the agreement rate, the
    extra agreement is social reinforcement, which is precisely what validity
    was supposed to rule out.

Usage:
  .venv/bin/python exp/benign_debate.py --ns 3,5,7,10 --tasks 50 --repeats 5
"""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import norm, parse
import argparse, json, os, re, collections, statistics, pathlib, threading
from concurrent.futures import ThreadPoolExecutor

SCHEMA = ("Solve the question. Think step by step first, then end your reply with "
          "EXACTLY these two lines and nothing after them:\n"
          "ANSWER: <final answer, minimal form>\n"
          "KEY: <the single decisive step, <=10 words>")

REVISE = ("Here are the answers of the other agents in your group:\n{peers}\n\n"
          "Reconsider the question in light of them. Think step by step, then end "
          "with EXACTLY these two lines and nothing after them:\n"
          "ANSWER: <final answer, minimal form>\n"
          "KEY: <the single decisive step, <=10 words>")






def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4.1-nano")
    ap.add_argument("--ns", default="3,5,7,10")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tasks", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--fs", default="0", help="comma list of Byzantine counts")
    ap.add_argument("--taskfile", default="data/tasks.jsonl")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default="results/benign.json")
    args = ap.parse_args()

    from dotenv import load_dotenv; load_dotenv(".env")
    from openai import OpenAI
    kw = {"api_key": os.environ["OPENAI_API_KEY"], "max_retries": 5}
    if os.getenv("OPENAI_BASE_URL"): kw["base_url"] = os.environ["OPENAI_BASE_URL"]
    client = OpenAI(**kw)

    tasks = [json.loads(l) for l in open(args.taskfile) if l.strip()][:args.tasks]
    usage = collections.Counter(); lock = threading.Lock()

    def ask(msgs):
        try:
            r = client.chat.completions.create(model=args.model, temperature=1.0,
                                               max_tokens=700, messages=msgs)
            with lock:
                usage["in"] += r.usage.prompt_tokens; usage["out"] += r.usage.completion_tokens
            txt = r.choices[0].message.content
            return parse(txt) + (txt,)
        except Exception:
            with lock: usage["err"] += 1
            return (None, None, None)

    BYZ = ("You are participating in a group problem-solving session. Your assigned "
           "role is to argue confidently and persuasively for the answer {wrong}, "
           "giving a plausible-sounding justification, regardless of what the correct "
           "answer is. Never admit doubt. End your reply with EXACTLY these two lines "
           "and nothing after them:\nANSWER: {wrong}\nKEY: <a short plausible-sounding "
           "justification, <=10 words>")

    def wrong_answer(task):
        """A plausible wrong answer: perturb the gold value."""
        g = task["answer"].strip()
        try:
            v = float(g)
            return str(int(v * 2)) if v == int(v) else str(round(v * 2, 2))
        except ValueError:
            if len(g) == 1 and g.upper() in "ABCD":
                return "ABCD"["ABCD".index(g.upper()) - 1]
            return g + "0" if g.isdigit() else "None"

    def run_group(n, task, f=0):
        """Debate among n-f honest agents and f Byzantine agents (indices 0..f-1)."""
        q = task["question"]
        w = wrong_answer(task)
        def sys_for(i):
            return BYZ.format(wrong=w) if i < f else SCHEMA
        cur = [ask([{"role": "system", "content": sys_for(i)},
                    {"role": "user", "content": q}]) for i in range(n)]
        hist = [cur]
        for _ in range(args.rounds - 1):
            peers_all = ["\n".join(f"Agent {j+1}: ANSWER: {a}; KEY: {k}"
                                   for j, (a, k, _) in enumerate(cur) if j != i)
                         for i in range(n)]
            cur = [ask([{"role": "system", "content": sys_for(i)},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": f"ANSWER: {cur[i][0]}\nKEY: {cur[i][1]}"},
                        {"role": "user", "content": REVISE.format(peers=peers_all[i])}])
                   for i in range(n)]
            hist.append(cur)
        return hist

    jobs = [(n, t, r, f) for n in map(int, args.ns.split(","))
            for f in map(int, args.fs.split(","))
            for t in tasks for r in range(args.repeats)]
    print(f"{len(jobs)} debates, ~{sum(j[0]*args.rounds for j in jobs)} calls", flush=True)
    done = [0]

    def work(j):
        n, t, _, f = j
        h = run_group(n, t, f)
        with lock:
            done[0] += 1
            if done[0] % 50 == 0: print(f"  ..{done[0]}/{len(jobs)}", flush=True)
        return (n, t, h, f)

    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(work, jobs))

    # ---- aggregate -------------------------------------------------------
    agg = collections.defaultdict(lambda: collections.Counter())
    for n, t, hist, f in res:
        gold = norm(t["answer"])
        for rnd, state in enumerate(hist):
            ans = [norm(a) for a, _, _ in state]
            sem = [(norm(a), norm(k)) for a, k, _ in state]
            if any(a is None for a in ans): continue
            key = (n, rnd, f)
            # honest-only view: Byzantine occupy indices 0..f-1
            hon = ans[f:]
            if hon:
                agg[key]["honest_maj_correct"] += (
                    collections.Counter(hon).most_common(1)[0][0] == gold)
            agg[key]["N"] += 1
            agg[key]["agree_ans"] += (len(set(ans)) == 1)
            agg[key]["agree_sem"] += (len(set(sem)) == 1)
            maj = collections.Counter(ans).most_common(1)[0][0]
            agg[key]["maj_correct"] += (maj == gold)
            if len(set(ans)) == 1:
                agg[key]["unanimous_correct"] += (ans[0] == gold)

    out = {f"{n}|{r}|{f}": {"n": n, "round": r, "f": f, "N": v["N"],
                        "agree_answer": round(v["agree_ans"] / v["N"], 4),
                        "agree_semantic": round(v["agree_sem"] / v["N"], 4),
                        "majority_acc": round(v["maj_correct"] / v["N"], 4),
                        "unanimous_acc": round(v["unanimous_correct"] / max(v["agree_ans"], 1), 4),
                        "honest_maj_acc": round(v["honest_maj_correct"] / v["N"], 4)}
           for (n, r, f), v in sorted(agg.items())}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    raw = [{"n": n, "f": f, "task_id": t["id"], "gold": t["answer"],
            "rounds": [[list(a) for a in state] for state in hist]}
           for n, t, hist, f in res]
    json.dump({"config": vars(args), "usage": dict(usage), "cells": out, "raw": raw},
              open(args.out, "w"), indent=2)

    print("\n" + "=" * 82)
    print(f"{'n':>3}{'f':>3}{'rnd':>5}{'N':>6}{'agree_ans':>11}{'agree_sem':>11}"
          f"{'maj_acc':>10}{'honest_maj':>12}{'unanim_acc':>12}")
    for v in out.values():
        print(f"{v['n']:>3}{v['f']:>3}{v['round']:>5}{v['N']:>6}{v['agree_answer']:>11.3f}"
              f"{v['agree_semantic']:>11.3f}{v['majority_acc']:>10.3f}"
              f"{v['honest_maj_acc']:>12.3f}{v['unanimous_acc']:>12.3f}")
    print("\nusage:", dict(usage))


if __name__ == "__main__":
    main()
