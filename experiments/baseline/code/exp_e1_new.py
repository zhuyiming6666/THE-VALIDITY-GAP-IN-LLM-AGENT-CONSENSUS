"""E1 re-measurement on the v4 model set.

Reproduces the archived E1 protocol *exactly* -- 150 tasks in file order,
``k=30`` samples per task, temperature 1.0, ``max_tokens=700``, and the
free-reasoning schema prompt (``sampling.SYSTEM_FREE``, the same text the
archived run used under ``measure_pmax.py --cot``) -- but drives the models
through :class:`sampling.Sampler` instead of the OpenAI SDK, because the relay
serves Claude models over the Anthropic messages API and has no
``/chat/completions`` route at all.

Only the model list changes relative to the archived run.  That is the point:
E1 is a measurement of each model's output distribution, so it cannot be
carried over from a different model by renaming.

Records are written per model to ``results/raw/e1_<model>.samples.jsonl`` with
the fields ``recompute_e1.py`` consumes (``model``, ``task_id``, ``answer``,
``key``) plus the full raw reply, finish reason, token counts and config hash
that the v1 sampler discarded.

Usage
-----
    python3 exp_e1_new.py --plan                      # count jobs and cost, no calls
    python3 exp_e1_new.py --tasks 2 --k 3 --outdir /tmp/e1_pilot
    python3 exp_e1_new.py                             # the full 150 x 30 x 3 run
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sampling import Sampler, Writer, schema_prompt, provider_for  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
V3 = HERE.parent
ROOT = V3.parent
RESULTS = V3 / "results"

DEFAULT_MODELS = "claude-opus-5,claude-sonnet-5,deepseek-v4-pro"
TASKFILE = ROOT / "data" / "tasks.jsonl"

#: measured per-call prices in USD.  The relay bills every call at ~7,221 input
#: tokens regardless of prompt length (observed: a 10-token probe returned
#: ``input_tokens: 7221``), which is where the flat 0.00709 comes from.
PRICE_PER_CALL = {"relay": 0.00709, "deepseek": 0.000067, "openai": 0.0}

CONDITION = "free_reasoning"


def safe(name: str) -> str:
    return name.replace("/", "_")


def load_tasks(limit):
    tasks = []
    with open(TASKFILE) as fh:
        for line in fh:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks[:limit]


def estimate(models, n_tasks, k):
    rows = []
    total = 0.0
    calls = n_tasks * k
    for m in models:
        p = provider_for(m)
        c = calls * PRICE_PER_CALL.get(p, 0.0)
        total += c
        rows.append((m, p, calls, c))
    return rows, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--tasks", type=int, default=150)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--outdir", default=str(RESULTS / "raw"))
    ap.add_argument("--env", default=None)
    ap.add_argument("--plan", action="store_true",
                    help="print the job count and cost estimate, call nothing")
    ap.add_argument("--fresh", action="store_true",
                    help="rename any existing dump aside and start over")
    ap.add_argument("--max-calls", type=int, default=0,
                    help="hard safety cap on calls issued this invocation")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = load_tasks(args.tasks)
    outdir = pathlib.Path(args.outdir)
    rows, total = estimate(models, len(tasks), args.k)

    print(f"tasks={len(tasks)}  k={args.k}  temp={args.temp}  "
          f"max_tokens={args.max_tokens}  condition={CONDITION}")
    print(f"{'model':<22}{'provider':<10}{'calls':>8}{'est. USD':>10}")
    for m, p, c, cost in rows:
        print(f"  {m:<20}{p:<10}{c:>8}{cost:>10.2f}")
    print(f"  {'TOTAL':<20}{'':<10}{sum(c for _, _, c, _ in rows):>8}{total:>10.2f}")
    if args.plan:
        print("\n(plan only -- no calls issued)")
        return

    cap = args.max_calls or (len(tasks) * args.k * len(models))
    issued = 0
    lock = threading.Lock()
    usage = collections.Counter()
    errored = []

    for model in models:
        out = outdir / f"e1_{safe(model)}.samples.jsonl"
        if args.fresh and out.exists():
            out.rename(out.with_suffix(out.suffix + ".bak"))
        sampler = Sampler(model=model, env=args.env, temperature=args.temp,
                          max_tokens=args.max_tokens)
        system = schema_prompt(free=True)

        def job_key(rec):
            return (rec.get("model"), rec.get("task_id"), rec.get("sample_index"))

        writer = Writer(out, resume=True, job_key=job_key)
        jobs = [(t["id"], i) for t in tasks for i in range(args.k)]
        todo = [(tid, i) for tid, i in jobs
                if (model, tid, i) not in writer.seen]
        print(f"\n[{model}] {out}")
        print(f"  {len(jobs)} jobs, {len(jobs) - len(todo)} already on disk, "
              f"{len(todo)} to run", flush=True)

        done = [0]
        t_start = time.time()

        def work(tid, idx):
            q = next(t["question"] for t in tasks if t["id"] == tid)
            rec = sampler.complete(system, q)
            rec["model"] = model
            rec["task_id"] = tid
            rec["sample_index"] = idx
            rec["condition"] = CONDITION
            rec["config"] = sampler.config
            return rec

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {}
            for tid, idx in todo:
                if issued >= cap:
                    print(f"  hit --max-calls={cap}, stopping early")
                    break
                futs[pool.submit(work, tid, idx)] = (tid, idx)
                issued += 1
            for fut in as_completed(futs):
                tid, idx = futs[fut]
                rec = fut.result()
                if rec.get("error"):
                    errored.append((model, tid, idx, rec["error"]))
                else:
                    writer.write(rec)
                    with lock:
                        usage["in"] += rec.get("tokens_in") or 0
                        usage["out"] += rec.get("tokens_out") or 0
                        usage["ok"] += 1
                done[0] += 1
                if done[0] % 50 == 0 or done[0] == len(futs):
                    el = time.time() - t_start
                    rate = done[0] / el if el else 0
                    eta = (len(futs) - done[0]) / rate / 60 if rate else 0
                    print(f"  {done[0]}/{len(futs)}  {rate:.2f}/s  "
                          f"eta {eta:.1f} min  ok={usage['ok']} "
                          f"err={len(errored)}", flush=True)

        writer.close()
        print(f"  wrote {out}  ({usage['ok']} ok, {len(errored)} failed)", flush=True)

    print(f"\nusage: in={usage['in']} out={usage['out']} ok={usage['ok']} "
          f"errors={len(errored)}")
    if errored:
        print("first failures:")
        for m, tid, idx, err in errored[:5]:
            print(f"  {m} {tid} #{idx}: {err[:160]}")
        p = RESULTS / "e1_new_failures.json"
        p.write_text(json.dumps(errored, indent=1))
        print(f"full failure list: {p}")


if __name__ == "__main__":
    main()
