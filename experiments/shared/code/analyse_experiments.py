"""Turn the X1/X2/X3 dumps into the artifacts the paper reads.

Each experiment writes a JSONL dump of raw records; this module recomputes every
reported quantity from those records, so a number in the paper can always be
traced back to the individual model replies that produced it.  Nothing here
calls a model.

Run after the experiment scripts finish:
    python3 analyse_experiments.py
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[1]
sys.path.insert(0, str(EXPERIMENTS / "baseline/code"))

from collision import A_unbiased, task_bootstrap  # noqa: E402
from scoring import answer_class, correct  # noqa: E402

CONDITIONS = ("A_free_reasoning", "B_free_reasoning_schema", "C_schema_only")


def load_dump(p):
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def tasks_by_id():
    return {r["id"]: r for r in map(
        json.loads, (EXPERIMENTS / "baseline/data/tasks.jsonl").read_text().splitlines())}


# --------------------------------------------------------------------------
# X1  canonicalisation: does constraining the form hurt, or does removing the
#     reasoning hurt?
# --------------------------------------------------------------------------
def analyse_x1(recs, tasks):
    """Per-condition accuracy/dispersion and the two paired contrasts."""
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs:
        if r.get("error") or r.get("answer") is None:
            continue
        tid = r["task_id"]
        if tid not in tasks:
            continue
        bench = tasks[tid]["bench"]
        per[tid][r["condition"]].append(
            (answer_class(r["answer"], bench),
             correct(r["answer"], tasks[tid]["answer"], bench)))

    rows = {}
    for cond in CONDITIONS:
        acc, pmax, supp, a3 = [], [], [], []
        for tid, byc in per.items():
            vals = byc.get(cond)
            if not vals or len(vals) < 2:
                continue
            cls = [c for c, _ in vals]
            counts = collections.Counter(cls)
            acc.append(sum(1 for _, ok in vals if ok) / len(vals))
            pmax.append(max(counts.values()) / len(vals))
            supp.append(len(counts))
            a = A_unbiased(sorted(counts.values(), reverse=True), 3)
            if a is not None:
                a3.append(a)
        if not acc:
            continue
        rows[cond] = {
            "n_tasks": len(acc),
            "accuracy": sum(acc) / len(acc),
            "accuracy_ci": task_bootstrap(acc),
            "p_max": sum(pmax) / len(pmax),
            "support": sum(supp) / len(supp),
            "A_3": (sum(a3) / len(a3)) if a3 else None,
        }

    def paired(x, y):
        acc_pairs, a3_pairs = [], []
        for tid, byc in per.items():
            if x not in byc or y not in byc:
                continue
            ax = sum(1 for _, ok in byc[x] if ok) / len(byc[x])
            ay = sum(1 for _, ok in byc[y] if ok) / len(byc[y])
            acc_pairs.append((ax, ay))
            cx = collections.Counter(c for c, _ in byc[x])
            cy = collections.Counter(c for c, _ in byc[y])
            vx = A_unbiased(sorted(cx.values(), reverse=True), 3)
            vy = A_unbiased(sorted(cy.values(), reverse=True), 3)
            if vx is not None and vy is not None:
                a3_pairs.append((vx, vy))
        out = {"n_tasks": len(acc_pairs)}
        if acc_pairs:
            out["accuracy"] = task_bootstrap([a - b for a, b in acc_pairs])
        if a3_pairs:
            out["A_3"] = task_bootstrap([a - b for a, b in a3_pairs])
        return out

    return {
        "per_condition": rows,
        "paired": {
            "A_free_reasoning_minus_B_free_reasoning_schema":
                paired("A_free_reasoning", "B_free_reasoning_schema"),
            "B_free_reasoning_schema_minus_C_schema_only":
                paired("B_free_reasoning_schema", "C_schema_only"),
        },
        "n_records": len(recs),
        "interpretation": (
            "A - B isolates the output-form constraint with the reasoning "
            "budget held fixed. B - C reproduces the v1 contrast, which "
            "changed the reasoning budget as well."),
    }


# --------------------------------------------------------------------------
# X2  external verifier: agreement and external validity with a real checker
# --------------------------------------------------------------------------
def analyse_x2(recs, tasks):
    """Protocol outcome by fault budget, plus the soundness diagnostic."""
    by_f = collections.defaultdict(list)
    for r in recs:
        by_f[r["f"]].append(r)

    table = {}
    for f, rs in sorted(by_f.items()):
        agree = sum(1 for r in rs if r["agreement"])
        abst = sum(1 for r in rs if r["abstained"])
        slots = [s for r in rs for s in r.get("slots", [])]
        byz = [s for s in slots if s.get("byzantine")]
        hon = [s for s in slots if not s.get("byzantine")]
        alphas = [r["honest_alpha_hat"] for r in rs
                  if r.get("honest_alpha_hat") is not None]
        table[str(f)] = {
            "rounds": len(rs),
            "agreement": agree / len(rs),
            "abstention": abst / len(rs),
            "alpha_hat_mean": (sum(alphas) / len(alphas)) if alphas else None,
            "honest_slots": len(hon),
            "honest_accepted": sum(1 for s in hon if s["accepted"]),
            "byz_slots": len(byz),
            "byz_accepted": sum(1 for s in byz if s["accepted"]),
            "byz_mutated": sum(1 for s in byz if s.get("mutated")),
            # any accepted mutated submission would refute the soundness
            # assumption Theorem 4 rests on
            "soundness_violations": sum(
                1 for s in slots if s.get("soundness_violation")),
            "byz_compliance": (sum(1 for s in byz if s.get("mutated")) / len(byz))
            if byz else None,
        }
    # Per-task acceptance.  Theorem 4 indexes alpha by task; the run shows why
    # that is not a technicality: tasks are solved either always or never, so a
    # pooled rate predicts a failure probability many orders of magnitude below
    # what is observed.
    by_task = collections.defaultdict(list)
    for r in recs:
        by_task[r["task_id"]].append(r)
    per_task = {}
    for tid, rs in sorted(by_task.items()):
        slots = [s for r in rs if r["f"] == 0
                 for s in r.get("slots", []) if not s.get("byzantine")]
        if not slots:
            continue
        per_task[tid] = sum(1 for s in slots if s["accepted"]) / len(slots)
    alphas = list(per_task.values())
    n_t = len(alphas) or 1
    pooled = (sum(alphas) / n_t) if alphas else None

    def _empty(vals):
        return sum(1 for a in vals if a <= 1e-9)

    terms = {}
    for h in sorted({len(r["slots"]) - r["f"] for r in recs}):
        obs, tot = 0, 0
        for r in recs:
            if len(r["slots"]) - r["f"] != h:
                continue
            tot += 1
            obs += int(r["abstained"])
        terms[str(h)] = {
            "observed_abstention": (obs / tot) if tot else None,
            "per_task_term": (sum((1 - a) ** h for a in alphas) / n_t)
            if alphas else None,
            "pooled_term": ((1 - pooled) ** h) if pooled is not None else None,
            "rounds": tot,
        }

    return {
        "n_rounds": len(recs),
        "by_f": table,
        "soundness_holds": all(v["soundness_violations"] == 0
                               for v in table.values()),
        "per_task_alpha": per_task,
        "pooled_alpha": pooled,
        "n_tasks_never_solved": _empty(alphas),
        "n_tasks_always_solved": sum(1 for a in alphas if a >= 1 - 1e-9),
        "n_tasks": n_t,
        "theorem4_terms": terms,
        "task_heterogeneity_note": (
            "The first term of Theorem 4 must be evaluated per task. On this "
            "run the pooled form under-states the observed abstention by "
            "orders of magnitude because acceptance is bimodal across tasks."),
    }


# --------------------------------------------------------------------------
# X3  attackers: a forced Byzantine adversary versus a prompted one
# --------------------------------------------------------------------------
def analyse_x3(recs, tasks):
    cells = collections.defaultdict(list)
    for r in recs:
        for rnd in r.get("history", []):
            if not rnd:
                continue
            cells[(r["strategy"], r["f"], rnd[0]["round"])].append((r, rnd))

    table = {}
    for (strat, f, rnd), items in sorted(cells.items()):
        agree, hon_agree, hon_acc = [], [], []
        byz_slots = byz_oppose = byz_dissent = 0
        for r, rnd_slots in items:
            bench = r["bench"]
            gold = answer_class(tasks[r["task_id"]]["answer"], bench)
            verd = [answer_class(s["answer"], bench) for s in rnd_slots]
            n = len(verd)
            hon = [verd[i] for i in range(n) if i >= f]
            byz = [verd[i] for i in range(n) if i < f]

            def unan(vals):
                seen = {v for v in vals if v is not None}
                return len(seen) == 1 and all(v is not None for v in vals)

            def plurality(vals):
                c = collections.Counter(v for v in vals if v is not None)
                if not c:
                    return None
                top = max(c.values())
                win = sorted(x for x, v in c.items() if v == top)
                return win[0] if len(win) == 1 else None

            hmaj = plurality(hon)
            agree.append(1.0 if unan(verd) else 0.0)
            hon_agree.append(1.0 if unan(hon) else 0.0)
            if hmaj is not None:
                hon_acc.append(1.0 if hmaj == gold else 0.0)
            for b in byz:
                if b is None:
                    continue
                byz_slots += 1
                byz_oppose += int(b != gold)
                byz_dissent += int(hmaj is not None and b != hmaj)

        table[f"{strat}|{f}|{rnd}"] = {
            "strategy": strat, "f": f, "round": rnd, "debates": len(items),
            "agree_all": sum(agree) / len(agree),
            "agree_honest": sum(hon_agree) / len(hon_agree),
            "honest_maj_acc": (sum(hon_acc) / len(hon_acc)) if hon_acc else None,
            "byz_slots": byz_slots,
            # the two compliance measures answer different questions
            "role_compliance": (byz_oppose / byz_slots) if byz_slots else None,
            "held_out_from_majority": (byz_dissent / byz_slots) if byz_slots else None,
        }
    return {"n_debates": len(recs), "cells": table}


def main():
    tasks = tasks_by_id()
    out = {}
    for tag, dump, fn, artefact in (
            ("X1", EXPERIMENTS / "E4_output_form_ablation/results/x1_canon_opus5.samples.jsonl", analyse_x1,
             EXPERIMENTS / "E4_output_form_ablation/results/x1_canonicalisation.json"),
            ("X2", EXPERIMENTS / "X2_external_checker/results/x2_verifier.samples.jsonl", analyse_x2,
             EXPERIMENTS / "X2_external_checker/results/x2_verifier_analysis.json"),
            ("X3", EXPERIMENTS / "E6_attacks/results/x3_attacks.samples.jsonl", analyse_x3,
             EXPERIMENTS / "E6_attacks/results/x3_attacks_analysis.json")):
        recs = load_dump(dump)
        if recs is None:
            print(f"  {tag}: {dump} 尚未生成，跳过")
            continue
        res = fn(recs, tasks)
        artefact.write_text(json.dumps(res, indent=1, default=str))
        print(f"  {tag}: {len(recs)} records -> {artefact.relative_to(EXPERIMENTS)}")
        out[tag] = res

    if "X1" in out:
        print("\n=== X1 ===")
        for k, v in sorted(out["X1"]["per_condition"].items()):
            print(f"  {k:<26} acc={v['accuracy']:.4f}  p_max={v['p_max']:.3f}  "
                  f"#cls={v['support']:.2f}")
        for k, v in out["X1"]["paired"].items():
            a = v.get("accuracy")
            if a:
                print(f"  {k}\n      acc {a['estimate']:+.4f} [{a['lo']:+.4f},{a['hi']:+.4f}]")
    if "X2" in out:
        print(f"\n=== X2 ===  可靠性假设成立: {out['X2']['soundness_holds']}")
        print(f"  {'f':>2}{'轮数':>6}{'同意':>8}{'弃权':>8}{'α̂':>8}{'攻击接受':>10}{'违例':>6}")
        for f, v in sorted(out["X2"]["by_f"].items(), key=lambda kv: int(kv[0])):
            print(f"  {f:>2}{v['rounds']:>6}{v['agreement']:>8.3f}{v['abstention']:>8.3f}"
                  f"{(v['alpha_hat_mean'] or 0):>8.3f}"
                  f"{v['byz_accepted']}/{v['byz_slots']:<8}{v['soundness_violations']:>6}")
    if "X3" in out:
        print("\n=== X3 ===  最后一个轮次")
        print(f"  {'策略':<10}{'f':>2}{'同意':>8}{'诚实同意':>10}{'诚实正确':>10}"
              f"{'服从':>8}{'异议':>8}")
        for k, v in sorted(out["X3"]["cells"].items()):
            if v["round"] != 2:
                continue
            print(f"  {v['strategy']:<10}{v['f']:>2}{v['agree_all']:>8.3f}"
                  f"{v['agree_honest']:>10.3f}"
                  f"{(v['honest_maj_acc'] or 0):>10.3f}"
                  f"{(v['role_compliance'] or 0):>8.3f}"
                  f"{(v['held_out_from_majority'] or 0):>8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
