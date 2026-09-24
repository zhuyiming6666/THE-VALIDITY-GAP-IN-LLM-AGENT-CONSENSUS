"""X2: an end-to-end test of the external-verification guarantee.

Theorem 4 makes a conditional claim: if every receiver verifies locally and
reliable delivery holds, then class-level agreement with external validity is
achievable even with one honest proposer, at a cost set by the verifier's
per-task acceptance probability and its soundness error.  The v1 paper stated
this result and never ran it.  This script runs it.

Design
------
* Tasks are the MBPP-derived items in ``data/tasks.jsonl``.  Each embeds a
  reference implementation and one concrete call whose return value is the
  task's gold answer.  The agents write a function; the verifier executes it in
  a subprocess, makes that same call, and compares the returned value to the
  gold using the paper's own scoring rule.  The verifier therefore checks a
  property of the submitted program rather than of the reply text.
* Honest agents receive the ordinary code prompt.  Byzantine agents are
  instructed to submit code that is plausible but wrong.  Because we cannot
  assume they comply, the achieved attack is *measured*: we check whether each
  Byzantine submission actually fails the tests.
* Every honest receiver runs the verifier itself on the bytes it received, so
  the ``local verification`` assumption is realised rather than assumed.
* ``alpha_x`` is estimated per task from the honest submissions, which is the
  quantity Theorem 4 needs.  A pooled rate is also reported so the difference
  between the two expressions can be seen.

Metrics per (n, f): class-level agreement among honest receivers, external
correctness, abstention rate, the fraction of rounds with no acceptable output
(the ``no honest success`` term), and the measured Byzantine false-accept rate
(the empirical counterpart of ``eps_fp``).

    python3 exp_verifier_e2e.py --model gpt-4.1-nano --tasks 50 --f 0,1,2,3,4 \
        --out ../results/x2_verifier.json
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import os
import pathlib
import random
import re
import subprocess
import sys
import tempfile
import textwrap
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collision import wilson  # noqa: E402
from sampling import (BYZ_FIXED, SYSTEM_CODE, Sampler, Writer,  # noqa: E402
                      extract_code, stratified_tasks, write_json)

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent

#: A prompted attacker is not a Byzantine one.  Instructed to answer wrongly,
#: the model solves the problem correctly anyway (verified: every accepted
#: "attacker" submission returned the gold value), so a real adversarial
#: submission is constructed here instead of requested.  The honest function is
#: wrapped so its return value is deterministically altered, which is both
#: guaranteed non-compliant and representative of an adversary that simply
#: emits a wrong answer.  ``mutate_code`` reports whether the mutation could be
#: applied; a failure to mutate is recorded rather than silently accepted.
WRONG_PROMPT = None  # unused; kept for reference to the earlier design

MUTATOR = '''
_orig = {fname}
def {fname}(*a, **k):
    v = _orig(*a, **k)
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v + 1
    if isinstance(v, str):
        return v + "!"
    if isinstance(v, tuple):
        return tuple(list(v) + ["!"])
    if isinstance(v, list):
        return list(v) + ["!"]
    if isinstance(v, dict):
        d = dict(v)
        d["__x__"] = 1
        return d
    return "__wrong__"
'''


def mutate_code(code, task):
    """Wrap an honest submission so it returns a wrong value deterministically.

    Returns ``(mutated_code, ok)``.  ``ok`` is False when the function name
    cannot be determined, in which case the caller records a non-compliant
    attacker rather than pretending the attack succeeded.
    """
    if not code:
        return None, False
    m = re.search(r"^\s*def\s+(\w+)\s*\(", code, re.M)
    if not m:
        return None, False
    fname = m.group(1)
    return code + "\n\n" + MUTATOR.format(fname=fname), True


# --------------------------------------------------------------------------
# the verifier
# --------------------------------------------------------------------------
# The MBPP-derived tasks in data/tasks.jsonl are value-prediction items.  Each
# question embeds a reference implementation and one concrete call, and the gold
# answer is that call's return value.  That is enough to build a genuine
# external verifier without any new data: execute the candidate function, make
# the same call in a fresh namespace, and compare the value using the same
# benchmark-specific scoring the paper uses elsewhere.  The verifier therefore
# checks a property of the *program*, not of the reply text.
CALL_RE = re.compile(r"The function is called as:\s*(.+?)\s*\n", re.S)
REF_SPLIT = "Reference implementation:"

#: The honest prompt sees only the problem statement.  It must not see the
#: reference implementation or the specific call, otherwise the task collapses
#: into reading the answer out of the prompt and the verifier has nothing to
#: test.  The stem is the text before the reference block.
TASK_PROMPT = (
    "Write a Python function that solves this problem. Return ONLY the "
    "function definition inside a single fenced code block, with no "
    "explanation before or after it.\n\nProblem:\n{q}"
)


def load_mbpp_tasks(path, n_per_bench=None, seed=20260911):
    rows = []
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("bench") != "mbpp":
            continue
        calls = CALL_RE.findall(r["question"])
        if not calls or REF_SPLIT not in r["question"]:
            continue
        r["_call"] = calls[0]
        # strip the leaked answer material before it can reach the model
        stem = r["question"].split(REF_SPLIT)[0]
        stem = CALL_RE.sub("", stem)
        stem = re.sub(r"What is the exact value returned by that call\?",
                      "Return the value the function would produce for the "
                      "described input.", stem)
        r["_stem"] = stem.strip()
        rows.append(r)
    rows.sort(key=lambda r: r["id"])
    if n_per_bench and n_per_bench < len(rows):
        rows = random.Random(seed).sample(rows, n_per_bench)
        rows.sort(key=lambda r: r["id"])
    return rows


VERIFY_SNIPPET = """
import sys, json
_ns = {{}}
exec(compile(open({cand!r}).read(), {cand!r}, 'exec'), _ns)
_value = eval({call!r}, _ns)
print('RESULT:' + json.dumps(repr(_value)))
"""


def run_verifier(code, task, timeout=10):
    """Execute the candidate and the specified call; compare with the gold.

    Returns ``(accepted, detail)``.  A timeout, a syntax error, a missing
    function, and a wrong value are all non-acceptance, and are reported
    separately so a systematically broken harness cannot look like a correctly
    failing verifier.
    """
    if not code:
        return False, "no_code"
    try:
        with tempfile.TemporaryDirectory() as td:
            cand = pathlib.Path(td) / "candidate.py"
            cand.write_text(code)
            drv = pathlib.Path(td) / "driver.py"
            drv.write_text(VERIFY_SNIPPET.format(cand=str(cand),
                                                 call=task["_call"]))
            proc = subprocess.run([sys.executable, str(drv)],
                                  capture_output=True, text=True,
                                  timeout=timeout, cwd=td)
        if proc.returncode != 0:
            err = (proc.stderr or "").strip().splitlines()
            last = err[-1] if err else f"returncode={proc.returncode}"
            if "SyntaxError" in (proc.stderr or ""):
                return False, "syntax_error"
            if "NameError" in (proc.stderr or ""):
                return False, "missing_function"
            return False, f"error:{last[:80]}"
        marker = [l for l in proc.stdout.splitlines() if l.startswith("RESULT:")]
        if not marker:
            return False, "no_result"
        produced = json.loads(marker[-1][len("RESULT:"):])
        want = str(task["answer"])
        # compare with the paper's benchmark-specific rule, so the verifier and
        # the offline scoring agree on what "the same value" means
        from scoring import gold_key, norm_answer
        ok = norm_answer(produced, "mbpp") == gold_key(want, "mbpp")
        return ok, ("accepted" if ok else "wrong_value")
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as exc:  # noqa: BLE001
        return False, f"harness_error:{type(exc).__name__}"


# --------------------------------------------------------------------------
# one round of the protocol
# --------------------------------------------------------------------------
def run_round(sampler, task, n, f, honest_system):
    """Sample n submissions, verify each locally, and apply the decide rule."""
    def one(i):
        byz = i < f
        system = honest_system
        out = sampler.complete(system, TASK_PROMPT.format(q=task["_stem"]))
        code = extract_code(out["raw"])
        mutated = False
        if byz and code:
            code, mutated = mutate_code(code, task)
        accepted, detail = run_verifier(code, task)
        # for an attacker slot, acceptance now means the verifier accepted a
        # value that was deliberately altered -> a soundness violation
        soundness_violation = bool(byz and accepted and mutated)
        return {
            "slot": i, "byzantine": byz, "raw": out["raw"], "code": code,
            "accepted": accepted, "verify_detail": detail,
            "mutated": mutated,
            "soundness_violation": soundness_violation,
            "error": out["error"], "tokens_out": out["tokens_out"],
        }

    with cf.ThreadPoolExecutor(max_workers=n) as pool:
        slots = list(pool.map(one, range(n)))

    acceptable = [s for s in slots if s["accepted"]]
    honest = [s for s in slots if not s["byzantine"]]
    byz = [s for s in slots if s["byzantine"]]

    # Keep every submission, not just the aggregate.  Without the code there is
    # no way to tell afterwards whether the verifier accepted something genuinely
    # wrong (which would violate the soundness assumption of Theorem 4) or
    # whether the attacker simply complied poorly.  Storing only counters was a
    # defect of the earlier pipeline and it is what made the first f=1 result
    # impossible to explain.
    slot_detail = [{
        "slot": s["slot"], "byzantine": s["byzantine"],
        "accepted": s["accepted"], "verify_detail": s["verify_detail"],
        "code": s["code"], "error": s["error"],
        "mutated": s.get("mutated"),
        "soundness_violation": s.get("soundness_violation"),
    } for s in slots]

    # every honest receiver decides the class of the first accepted output it
    # receives; all accepted outputs form one class by construction of the
    # verifier, so agreement reduces to whether anyone accepted at all
    decision = "accept" if acceptable else "abstain"
    return {
        "task_id": task["id"],
        "n": n, "f": f,
        "n_acceptable": len(acceptable),
        "n_honest_acceptable": sum(1 for s in honest if s["accepted"]),
        "n_byz_acceptable": sum(1 for s in byz if s["accepted"]),
        "n_honest": len(honest), "n_byz": len(byz),
        "decision": decision,
        "agreement": decision == "accept",
        "external_validity": decision == "accept",  # accepted => passes tests
        "abstained": decision == "abstain",
        "byz_compliant": sum(1 for s in byz if not s["accepted"]),
        "honest_alpha_hat": (sum(1 for s in honest if s["accepted"])
                             / len(honest)) if honest else None,
        "errors": sum(1 for s in slots if s["error"]),
        "slots": slot_detail,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5",
                help="any model the configured provider serves; see sampling.PROVIDERS")
    ap.add_argument("--tasks", type=int, default=50)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--f", default="0,1,2,3,4")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--env", default=None)
    ap.add_argument("--out", default=str(HERE.parent / "results/x2_verifier.json"))
    ap.add_argument("--dump", default=str(HERE.parent / "results/x2_verifier.samples.jsonl"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--plan", action="store_true",
                    help="print outstanding rounds and exit; no model calls")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    fs = [int(x) for x in args.f.split(",")]
    tasks = load_mbpp_tasks(ROOT / "data/tasks.jsonl", args.tasks)
    if args.smoke:
        tasks, fs, args.repeats, args.n = tasks[:2], fs[:2], 1, 4
    if not tasks:
        raise SystemExit("no MBPP tasks with an embedded call were found")

    KEY_REC = lambda r: (r["f"], r["rep"], r["task_id"])
    KEY_JOB = lambda j: (j[0], j[1], j[2]["id"])
    if args.plan:
        all_jobs = [(f, rep, t) for f in fs for rep in range(args.repeats) for t in tasks]
        with Writer(args.dump, resume=True, job_key=KEY_REC) as w:
            w.load(KEY_REC)
            w.seen = {KEY_REC(r) for r in w.records}
            todo = w.todo(all_jobs, KEY_JOB)
            print(f"plan: {len(all_jobs)} rounds, {w.skipped} recorded, {len(todo)} remaining")
            print(f"      {len(todo)*args.n} model calls, ${len(todo)*args.n*0.00709:.2f} "
                  f"at the measured rate")
        return 0

    sampler = Sampler(model=args.model, env=args.env, max_tokens=700)
    n_calls = len(tasks) * len(fs) * args.repeats * args.n
    print(f"model={args.model} tasks={len(tasks)} f={fs} repeats={args.repeats} "
          f"n={args.n} -> {n_calls} calls")

    def job_key(r):
        return (r["f"], r["rep"], r["task_id"])

    rounds, verifier_stats = [], collections.Counter()
    with Writer(args.dump, resume=args.resume, job_key=KEY_REC) as w:
        w.seen = w.load(KEY_REC)
        all_jobs = [(f, rep, t) for f in fs for rep in range(args.repeats)
                    for t in tasks]
        all_jobs = w.todo(all_jobs, KEY_JOB)
        pending = {(f, rep, t["id"]) for f, rep, t in all_jobs}
        if w.skipped:
            print(f"  resuming: {w.skipped} rounds already recorded, "
                  f"{len(pending)} remaining")
        # reload what is already on disk so the summary covers the whole run
        if w.seen:
            for line in pathlib.Path(args.dump).read_text().splitlines():
                if line.strip():
                    rounds.append(json.loads(line))
        for f in fs:
            for rep in range(args.repeats):
                for task in tasks:
                    if (f, rep, task["id"]) not in pending:
                        continue
                    rec = run_round(sampler, task, args.n, f, SYSTEM_CODE)
                    rec["rep"] = rep
                    rec["model"] = args.model
                    rec["config_hash"] = sampler.config
                    rounds.append(rec)
                    w.write(rec)
                print(f"  f={f} rep={rep} done  ({len(rounds)} rounds, "
                      f"errors={sampler.usage['errors']})")

    # ---------------------------------------------------------------- metrics
    by_f = collections.defaultdict(list)
    for r in rounds:
        by_f[r["f"]].append(r)

    table = {}
    for f, rs in sorted(by_f.items()):
        n_r = len(rs)
        agree = sum(1 for r in rs if r["agreement"])
        abst = sum(1 for r in rs if r["abstained"])
        byz_slots = sum(r["n_byz"] for r in rs)
        byz_acc = sum(r["n_byz_acceptable"] for r in rs)
        alphas = [r["honest_alpha_hat"] for r in rs
                  if r["honest_alpha_hat"] is not None]
        table[f"f={f}"] = {
            "rounds": n_r,
            "agreement": wilson(agree, n_r),
            "abstention": wilson(abst, n_r),
            "mean_alpha_hat": (sum(alphas) / len(alphas)) if alphas else None,
            "byz_slots": byz_slots,
            "byz_accepted": byz_acc,
            "byz_false_accept_rate": (byz_acc / byz_slots) if byz_slots else None,
            "byz_compliance": (sum(r["byz_compliant"] for r in rs) / byz_slots)
            if byz_slots else None,
        }

    # predicted-vs-observed, using Theorem 4's own expression
    h = args.n - min(fs)
    preds = {}
    for f in sorted(by_f):
        hf = args.n - f
        prod = 1.0
        for r in by_f[f]:
            a = r["honest_alpha_hat"]
            if a is not None:
                prod *= (1 - a) ** hf if False else (1 - a) ** r["n_honest"]
        n_r = len(by_f[f])
        preds[f"f={f}"] = {
            "n_honest": hf,
            "per_task_term": (prod / n_r) if n_r else None,
            "pooled_term": None,
            "observed_abstention": table[f"f={f}"]["abstention"]["estimate"],
        }
    for f in sorted(by_f):
        hf = args.n - f
        alphas = [r["honest_alpha_hat"] for r in by_f[f]
                  if r["honest_alpha_hat"] is not None]
        if alphas:
            mean_alpha = sum(alphas) / len(alphas)
            preds[f"f={f}"]["pooled_term"] = (1 - mean_alpha) ** hf

    summary = {
        "config": vars(args) | {"config_hash": sampler.config},
        "usage": dict(sampler.usage),
        "verifier": "subprocess execution of the candidate, then the task's "
                    "specified call, compared to gold with the paper's own rule",
        "per_f": table,
        "theorem4_terms": preds,
    }
    write_json(args.out, summary)

    print("\n--- protocol outcome by f ---")
    print(f"{'f':>2}{'rounds':>7}{'agreement':>11}{'abstain':>9}{'alpha':>8}"
          f"{'byzAcc':>8}{'byzComply':>10}")
    for k, v in sorted(table.items()):
        def g(x):
            return f"{x:.3f}" if isinstance(x, float) else "  -  "
        print(f"{k.split('=')[1]:>2}{v['rounds']:>7}"
              f"{g(v['agreement']['estimate']):>11}{g(v['abstention']['estimate']):>9}"
              f"{g(v['mean_alpha_hat']):>8}{g(v['byz_false_accept_rate']):>8}"
              f"{g(v['byz_compliance']):>10}")
    print("\n--- Theorem 4 terms: per-task vs pooled abstention ---")
    for k, v in sorted(preds.items()):
        pt = v["per_task_term"]
        pl = v["pooled_term"]
        print(f"  {k:<6} per-task term {pt if pt is None else round(pt,5)}   "
              f"pooled term {pl if pl is None else round(pl,5)}   "
              f"observed {v['observed_abstention']:.3f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
