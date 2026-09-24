"""X1: does constraining the output form hurt, or does removing the reasoning?

The v1 ablation compared ``free reasoning -> answer+KEY schema`` with
``schema only`` and read the accuracy drop as evidence that canonicalisation
hurts.  That comparison changes two things at once --- the output format and
whether the model reasons --- so it cannot attribute the drop.  This experiment
separates them with three conditions that hold the reasoning budget fixed:

  A  free reasoning -> free-form final answer      (no schema required)
  B  free reasoning -> the two-line schema         (format constrained)
  C  schema only, no reasoning                     (the v1 ``tight`` condition)

A vs B isolates the format constraint with reasoning held constant, which is
the comparison the review asked for.  B vs C reproduces the v1 contrast and is
labelled as a reasoning intervention rather than a canonicalisation one.

The answer field is extracted from free-form conditions by the same parser, so
a condition cannot win by having an easier extraction target.

    python3 exp_canonicalisation.py --model gpt-4.1-nano --tasks 50 --k 16 \
        --out ../results/x1_canon.json --dump ../results/x1_canon.samples.jsonl
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collision import A_unbiased, task_bootstrap  # noqa: E402
from sampling import (SYSTEM_FREE, SYSTEM_FREE_FORMAL, SYSTEM_SCHEMA_ONLY,  # noqa: E402
                      Sampler, Writer, load_env, stratified_tasks, write_json)
from scoring import answer_class, correct  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# The three conditions must differ in exactly one respect each, otherwise the
# ablation repeats the v1 confound it exists to remove.
#
#   A  free reasoning -> free-form final line      (no KEY asked for)
#   B  free reasoning -> final line + KEY line     (format extended by one field)
#   C  no reasoning   -> the two-line schema       (the v1 "tight" condition)
#
# A versus B changes only how much of the output form is pinned down, with the
# reasoning budget held fixed; A and B use the same answer-line wording so the
# answer is extracted identically.  B versus C reproduces the v1 contrast and
# is reported as a reasoning intervention, not as canonicalisation.
ANSWER_LINE = ("FINAL ANSWER: <answer>")

SYSTEM_A = (
    "Solve the question. Think step by step first. Then state your final "
    "answer on its own last line in exactly the form 'FINAL ANSWER: <answer>' "
    "and put nothing after it."
)
SYSTEM_B = (
    "Solve the question. Think step by step first. Then end your reply with "
    "EXACTLY these two lines and nothing after them:\n"
    "FINAL ANSWER: <answer, minimal form>\n"
    "KEY: <the single decisive step, <=10 words>"
)
SYSTEM_C = (
    "Reply with EXACTLY two lines and nothing else:\n"
    "FINAL ANSWER: <answer, minimal form>\n"
    "KEY: <the single decisive step, <=10 words>"
)

CONDITIONS = {
    "A_free_reasoning": SYSTEM_A,
    "B_free_reasoning_schema": SYSTEM_B,
    "C_schema_only": SYSTEM_C,
}

#: One token budget for every condition.  The v1 ablation gave the schema-only
#: condition a 200-token cap against 700 for the CoT condition, so a truncation
#: and a reasoning difference were indistinguishable.  Holding the cap fixed
#: makes A-versus-B a comparison of the output form alone, and turns any
#: truncation into a reported parse failure rather than a silent confound.
TOKEN_BUDGET = 700

def _tasks(path, n_per_bench, benches=("gsm8k",)):
    """Tasks to run.

    ``n_per_bench`` is *per benchmark*, so selecting three benchmarks yields
    three times the calls; the default restricts to GSM8K, which is the subset
    the archived E1/E3 experiments used and therefore the one the retained
    records are comparable with.
    """
    rows = [json.loads(l) for l in pathlib.Path(path).read_text().splitlines()
            if l.strip()]
    if benches:
        rows = [r for r in rows if r["bench"] in benches]
    return stratified_tasks(rows, n_per_bench)


def run_one(sampler, task, condition, rep):
    system = CONDITIONS[condition]
    sampler.max_tokens = TOKEN_BUDGET
    out = sampler.complete(system, task["question"])
    raw = out["raw"] or ""
    a, k = out["answer"], out["key"]
    if a is None and raw:
        import re
        m = re.findall(r"(?:FINAL\s+)?ANSWER\s*[:：]\s*(.+)", raw, re.I)
        if m:
            a = m[-1].strip()
    if condition == "A_free_reasoning":
        # Condition A never asks for a KEY, so any "KEY:" the free text happens
        # to contain is noise rather than a field.  Leaving it in would make A
        # look like it had keyed output and would contaminate A-versus-B.
        k = None
    return {
        "task_id": task["id"], "bench": task["bench"], "model": sampler.model,
        "condition": condition, "rep": rep,
        "answer": a, "key": k, "raw": out["raw"],
        "finish_reason": out["finish_reason"],
        "tokens_in": out["tokens_in"], "tokens_out": out["tokens_out"],
        "error": out["error"], "latency_s": round(out["latency_s"], 3),
        "config_hash": sampler.config,
    }


def analyse(records, tasks_by_id, k_expected, min_records=None):
    """Per-condition accuracy, dispersion and antecedent, per task then mean."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in records:
        if r["error"]:
            continue
        cls = answer_class(r["answer"], r["bench"])
        if cls is None:
            continue
        by[(r["condition"], r["model"])][r["task_id"]].append(
            (cls, correct(r["answer"], tasks_by_id[r["task_id"]]["answer"],
                          r["bench"])))

    out = {}
    for (cond, model), per_task in sorted(by.items()):
        rows = []
        floor = min_records if min_records is not None else max(3, k_expected // 2)
        for tid, vals in per_task.items():
            if len(vals) < floor:
                continue
            counts = collections.Counter(c for c, _ in vals)
            rows.append({
                "id": tid, "n": len(vals),
                "counts": sorted(counts.values(), reverse=True),
                "pmax": max(counts.values()) / len(vals),
                "acc": sum(1 for _, ok in vals if ok) / len(vals),
                "support": len(counts),
            })
        if not rows:
            continue
        entry = {
            "n_tasks": len(rows),
            "mean_n": sum(r["n"] for r in rows) / len(rows),
            "accuracy": sum(r["acc"] for r in rows) / len(rows),
            "p_max": sum(r["pmax"] for r in rows) / len(rows),
            "support": sum(r["support"] for r in rows) / len(rows),
        }
        for h in (3, 10):
            vals = [A_unbiased(r["counts"], h) for r in rows]
            vals = [v for v in vals if v is not None]
            entry[f"A_{h}"] = (sum(vals) / len(vals)) if vals else None
            entry[f"A_{h}_n"] = len(vals)
        entry["accuracy_ci"] = task_bootstrap([r["acc"] for r in rows])
        out[f"{cond}|{model}"] = entry
    return out


def paired_contrasts(records, tasks_by_id):
    """Task-paired accuracy and A(h) differences between conditions.

    Pairing on the task is what makes A-vs-B a controlled comparison: the same
    questions, the same model, the same number of samples, differing only in
    whether the output form is constrained.
    """
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in records:
        if r["error"]:
            continue
        cls = answer_class(r["answer"], r["bench"])
        if cls is None:
            continue
        per[r["task_id"]][r["condition"]].append(
            (cls, correct(r["answer"], tasks_by_id[r["task_id"]]["answer"],
                          r["bench"])))
    pairs = [("A_free_reasoning", "B_free_reasoning_schema"),
             ("B_free_reasoning_schema", "C_schema_only")]
    out = {}
    for x, y in pairs:
        acc_pairs, a3_pairs = [], []
        for tid, conds in per.items():
            if x not in conds or y not in conds:
                continue
            ax = sum(1 for _, ok in conds[x] if ok) / len(conds[x])
            ay = sum(1 for _, ok in conds[y] if ok) / len(conds[y])
            acc_pairs.append((ax, ay))
            cx = collections.Counter(c for c, _ in conds[x])
            cy = collections.Counter(c for c, _ in conds[y])
            vx = A_unbiased(sorted(cx.values(), reverse=True), 3)
            vy = A_unbiased(sorted(cy.values(), reverse=True), 3)
            if vx is not None and vy is not None:
                a3_pairs.append((vx, vy))
        if not acc_pairs:
            continue
        out[f"{x}_minus_{y}"] = {
            "accuracy": task_bootstrap([a - b for a, b in acc_pairs]),
            "A_3": task_bootstrap([a - b for a, b in a3_pairs]) if a3_pairs else None,
            "n_tasks": len(acc_pairs),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5",
                help="any model the configured provider serves; see sampling.PROVIDERS")
    ap.add_argument("--tasks", type=int, default=50, help="per benchmark")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="sampling temperature; the config hash records it, so "
                         "runs at different temperatures never share a dump")
    ap.add_argument("--benches", default="gsm8k",
                    help="comma-separated benchmarks; default gsm8k, the subset "
                         "the archived records cover")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--env", default=None)
    ap.add_argument("--out", default=str(HERE.parent / "results/x1_canon.json"))
    ap.add_argument("--dump", default=str(HERE.parent / "results/x1_canon.samples.jsonl"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--plan", action="store_true",
                    help="print the outstanding jobs and exit without calling any model")
    ap.add_argument("--smoke", action="store_true",
                    help="1 task per benchmark, k=2, to validate the plumbing")
    args = ap.parse_args()

    benches = tuple(b for b in args.benches.split(",") if b)
    tasks = _tasks(ROOT / "data/tasks.jsonl", args.tasks, benches)
    k = args.k
    if args.smoke:
        tasks = tasks[:3]
        k = 2
    tasks_by_id = {t["id"]: t for t in tasks}

    KEY_REC = lambda r: (r["condition"], r["task_id"], r["rep"])
    KEY_JOB = lambda j: (j[1], j[0]["id"], j[2])

    if args.plan:
        jobs_all = [(t, c, rep) for t in tasks for c in CONDITIONS
                    for rep in range(k)]
        with Writer(args.dump, resume=True, job_key=KEY_REC) as w:
            recorded = w.load(KEY_REC)
            w.seen = recorded
            todo = w.todo(jobs_all, KEY_JOB)
            print(f"[plan] dump={args.dump} exists={__import__('pathlib').Path(args.dump).exists()} "
                  f"records={len(w.records)} recorded_keys={len(recorded)} "
                  f"seen={len(w.seen)} skipped={w.skipped}")
            print(f"plan: {len(jobs_all)} jobs, {w.skipped} already recorded, "
                  f"{len(todo)} remaining")
            print(f"      estimated cost ${len(todo)*0.00709:.2f} at the "
                  f"measured relay rate")
            if todo:
                ex = todo[:3] + todo[-1:]
                for t, c, rep in ex:
                    print(f"      e.g. {t['id']:<10} {c:<26} rep={rep}")
        return 0

    sampler = Sampler(model=args.model, env=args.env,
                      temperature=args.temperature)
    n_calls = len(tasks) * len(CONDITIONS) * k
    print(f"model={args.model} tasks={len(tasks)} conditions={len(CONDITIONS)} "
          f"k={k} -> {n_calls} calls")

    jobs = [(t, c, rep) for t in tasks for c in CONDITIONS for rep in range(k)]
    records = []
    with Writer(args.dump, resume=args.resume, job_key=KEY_REC) as w:
        w.seen = w.load(KEY_REC)
        jobs = w.todo(jobs, KEY_JOB)
        if w.skipped:
            print(f"  resuming: {w.skipped} jobs already recorded, "
                  f"{len(jobs)} remaining")
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_one, sampler, t, c, rep) for t, c, rep in jobs]
            for i, fut in enumerate(cf.as_completed(futs), 1):
                rec = fut.result()
                records.append(rec)
                w.write(rec)
                if i % 100 == 0 or i == len(futs):
                    print(f"  {i}/{len(futs)}  errors={sampler.usage['errors']}")

    ok = [r for r in records if not r["error"]]
    parsed = [r for r in ok if answer_class(r["answer"], r["bench"]) is not None]
    summary = {
        "config": vars(args) | {"config_hash": sampler.config,
                                "n_tasks": len(tasks), "k": k},
        "usage": dict(sampler.usage),
        "funnel": {
            "attempted": len(records),
            "returned": len(ok),
            "parsed_answer": len(parsed),
            "unparsed": len(ok) - len(parsed),
            "errors": len(records) - len(ok),
        },
        "conditions": {c: CONDITIONS[c][:60] for c in CONDITIONS},
        "token_budget": TOKEN_BUDGET,
        "per_condition": analyse(records, tasks_by_id, k, min_records=max(2, k // 2)),
        "paired": paired_contrasts(records, tasks_by_id),
    }
    write_json(args.out, summary)

    print("\n--- per condition ---")
    print(f"{'condition':<26}{'n':>5}{'acc':>8}{'p_max':>8}{'#cls':>7}"
          f"{'A(3)':>9}{'A(10)':>9}")
    for key, v in sorted(summary["per_condition"].items()):
        cond = key.split("|")[0]
        a3 = f"{v['A_3']:.4f}" if v["A_3"] is not None else "  -   "
        a10 = f"{v['A_10']:.5f}" if v["A_10"] is not None else "  -   "
        print(f"{cond:<26}{v['n_tasks']:>5}{v['accuracy']:>8.3f}"
              f"{v['p_max']:>8.3f}{v['support']:>7.2f}{a3:>9}{a10:>9}")
    print("\n--- paired contrasts (task-paired, 95% CI) ---")
    for name, v in summary["paired"].items():
        acc = v["accuracy"]
        line = (f"  {name:<50} acc {acc['estimate']:+.3f} "
                f"[{acc['lo']:+.3f},{acc['hi']:+.3f}]  n={v['n_tasks']}")
        if v.get("A_3"):
            line += f"   A(3) {v['A_3']['estimate']:+.4f}"
        print(line)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
