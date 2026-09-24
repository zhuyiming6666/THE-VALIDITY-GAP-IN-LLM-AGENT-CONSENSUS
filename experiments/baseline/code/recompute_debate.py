"""Recompute the debate experiments from stored per-agent, per-round replies.

Everything here is post-processing of archived transcripts.  No model is
called and no new interaction is generated.

Series
------
E3 benign   n in {3,5,7,10}, f=0, 3 recorded rounds
E4 byzantine n=10, f in {0..4}, the first f agent slots are the attacker

Quantities per (n, round, f) cell
---------------------------------
agree           all n agents share one verdict class
hon_agree       the n-f honest agents share one verdict class
agree_semantic  all n agents share one answer+KEY class
hon_agree_sem    honest agents share one answer+KEY class
majority_acc    plurality of all n verdicts matches gold
honest_maj_acc  plurality of the honest verdicts matches gold
unan_acc        gold rate conditional on all-agent unanimity
parse_fail      slots whose ANSWER could not be parsed
tie             slots where the top two classes had equal counts
byz_gold_rate   share of attacker slots whose verdict was the gold answer

Voting rule: strict plurality, with ties broken deterministically by the
lexicographically smallest class label.  The v1 code used
``Counter.most_common(1)``, whose tie-break depends on first-arrival order and
therefore on where the attacker was placed.  The number of tied cells is
reported so the effect is visible rather than hidden.
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collision import paired_bootstrap, task_bootstrap, wilson  # noqa: E402
from scoring import answer_class, parse_schema, semantic_class, norm_key  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
PACKAGE = HERE.parent
EXPERIMENTS = PACKAGE.parent
RESULTS = PACKAGE / "results"

SOURCE_FILES = [
    PACKAGE / "data/interaction/e3_benign.json",
    EXPERIMENTS / "E6_attacks/data/nano_prompted/e4_byz.json",
]


def _plurality(classes):
    """Strict plurality with a deterministic tie-break; returns (cls, tied)."""
    counts = collections.Counter(c for c in classes if c is not None)
    if not counts:
        return None, False
    top = max(counts.values())
    winners = sorted(c for c, v in counts.items() if v == top)
    return winners[0], len(winners) > 1


def _agent_classes(round_slots, tid, bench, tasks):
    """Turn one round's slots into verdict / semantic classes.

    A slot is the [answer, key, reasoning] triple written by
    ``benign_debate.py``; some older archives store a dict.
    """
    out = []
    for slot in round_slots:
        if isinstance(slot, dict):
            a_raw, k_raw = slot.get("answer"), slot.get("key")
        elif isinstance(slot, (list, tuple)):
            a_raw = slot[0] if len(slot) > 0 else None
            k_raw = slot[1] if len(slot) > 1 else None
        else:
            a_raw = k_raw = None
        if a_raw is None and isinstance(slot, str):
            a_raw, k_raw = parse_schema(slot)
        out.append((answer_class(a_raw, bench), semantic_class(a_raw, k_raw, bench)))
    return out


def _gold_class(tasks, tid):
    t = tasks[tid]
    return answer_class(t["answer"], t["bench"])


def analyse_debate(payload, tasks, byzantine_first_f=True):
    debates = payload.get("raw", [])
    cells = collections.defaultdict(list)

    for deb in debates:
        n = deb["n"]
        f = deb.get("f", 0)
        tid = deb["task_id"]
        bench = tasks[tid]["bench"]
        gold = _gold_class(tasks, tid)
        rounds = deb.get("rounds") or []

        for r, rnd in enumerate(rounds):
            classes = _agent_classes(rnd, tid, bench, tasks)
            verd = [c[0] for c in classes]
            sem = [c[1] for c in classes]
            n_slots = len(verd)
            if n_slots < n:
                continue

            byz_idx = set(range(f)) if byzantine_first_f else set()
            hon_idx = [i for i in range(n) if i not in byz_idx]

            v_all = [verd[i] for i in range(n)]
            v_hon = [verd[i] for i in hon_idx]
            s_all = [sem[i] for i in range(n)]
            s_hon = [sem[i] for i in hon_idx]

            a_all, tie_all = _plurality(v_all)
            a_hon, tie_hon = _plurality(v_hon)
            parse_fail = sum(1 for c in v_all if c is None)

            # unanimity requires every slot to share one non-missing class
            def unanimous(vals):
                seen = {v for v in vals if v is not None}
                return len(seen) == 1 and all(v is not None for v in vals)

            rec = {
                "task_id": tid,
                "agree": unanimous(v_all),
                "hon_agree": unanimous(v_hon),
                "agree_semantic": unanimous(s_all),
                "hon_agree_semantic": unanimous(s_hon),
                "majority_acc": (a_all == gold) if a_all is not None else None,
                "honest_maj_acc": (a_hon == gold) if a_hon is not None else None,
                "unan_acc": (a_all == gold) if (unanimous(v_all) and a_all is not None) else None,
                "tie_all": tie_all,
                "tie_hon": tie_hon,
                "parse_fail": parse_fail,
                "byz_slots": f,
                "byz_gold": sum(1 for i in byz_idx if verd[i] == gold),
            }
            cells[(n, r, f)].append(rec)

    return cells


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def cell_table(cells):
    out = {}
    for (n, r, f), recs in sorted(cells.items()):
        key = f"{n}|{r}|{f}"
        n_valid = len(recs)
        unan = [x for x in recs if x["agree"]]
        byz_slots_total = sum(x["byz_slots"] for x in recs)
        byz_gold_total = sum(x["byz_gold"] for x in recs)
        out[key] = {
            "n": n, "round": r, "f": f,
            "N": n_valid,
            "agree": _mean([1.0 if x["agree"] else 0.0 for x in recs]),
            "hon_agree": _mean([1.0 if x["hon_agree"] else 0.0 for x in recs]),
            "agree_semantic": _mean([1.0 if x["agree_semantic"] else 0.0 for x in recs]),
            "hon_agree_semantic": _mean([1.0 if x["hon_agree_semantic"] else 0.0 for x in recs]),
            "majority_acc": _mean([x["majority_acc"] for x in recs]),
            "honest_maj_acc": _mean([x["honest_maj_acc"] for x in recs]),
            "unan_acc": _mean([x["unan_acc"] for x in recs]),
            "n_unanimous": len(unan),
            "tie_rate": _mean([1.0 if x["tie_all"] else 0.0 for x in recs]),
            "parse_fail_rate": _mean([x["parse_fail"] for x in recs]),
            "byz_slots": byz_slots_total,
            "byz_gold": byz_gold_total,
            "byz_gold_rate": (byz_gold_total / byz_slots_total) if byz_slots_total else None,
        }
    return out


def paired_gains(cells, n, f=0, a_round=2, b_round=0):
    """Task-paired round2 - round0 gains for agreement and majority accuracy.

    The v1 paper compared the two gains by eye.  Pairing on the task lets us
    bootstrap the *difference of differences* and state whether agreement
    really rises faster than accuracy.
    """
    key_a = (n, a_round, f)
    key_b = (n, b_round, f)
    if key_a not in cells or key_b not in cells:
        return None
    A = {x["task_id"]: x for x in cells[key_a]}
    B = {x["task_id"]: x for x in cells[key_b]}
    shared = sorted(set(A) & set(B))
    agree_pairs, acc_pairs, diff_pairs = [], [], []
    for t in shared:
        da = (1.0 if A[t]["agree"] else 0.0) - (1.0 if B[t]["agree"] else 0.0)
        if A[t]["majority_acc"] is None or B[t]["majority_acc"] is None:
            continue
        dc = float(A[t]["majority_acc"]) - float(B[t]["majority_acc"])
        agree_pairs.append((da, 0.0))
        acc_pairs.append((dc, 0.0))
        diff_pairs.append((da, dc))
    res = {"n_shared_tasks": len(shared)}
    if agree_pairs:
        res["agree_gain"] = task_bootstrap([a for a, _ in agree_pairs])
    if acc_pairs:
        res["acc_gain"] = task_bootstrap([c for c, _ in acc_pairs])
    if diff_pairs:
        res["agree_minus_acc"] = paired_bootstrap(diff_pairs)
    return res


def main():
    RESULTS.mkdir(exist_ok=True)
    sys.path.insert(0, str(HERE))
    from recompute_e1 import load_tasks  # noqa: E402
    tasks = load_tasks()

    report = {"sources": {}, "config": {}}
    for path in SOURCE_FILES:
        if not path.exists():
            print(f"missing {path}, skipped")
            continue
        payload = json.loads(path.read_text())
        cells = analyse_debate(payload, tasks)
        table = cell_table(cells)
        tag = path.stem
        report["config"][tag] = payload.get("config", {})
        report["sources"][tag] = table

        print(f"\n=== {path.relative_to(EXPERIMENTS)} ===")
        print(f"{'n':>3}{'rnd':>4}{'f':>3}{'N':>6}{'agree':>8}{'honAgr':>8}"
              f"{'semAgr':>8}{'majAcc':>8}{'honAcc':>8}{'unanAcc':>9}{'tie':>6}{'byzGold':>9}")
        for k in sorted(table, key=lambda s: tuple(int(x) for x in s.split("|"))):
            c = table[k]
            def f6(x):
                return f"{x:.4f}" if isinstance(x, float) else "  -   "
            print(f"{c['n']:>3}{c['round']:>4}{c['f']:>3}{c['N']:>6}"
                  f"{f6(c['agree']):>8}{f6(c['hon_agree']):>8}{f6(c['agree_semantic']):>8}"
                  f"{f6(c['majority_acc']):>8}{f6(c['honest_maj_acc']):>8}"
                  f"{f6(c['unan_acc']):>9}{f6(c['tie_rate']):>6}"
                  f"{f6(c['byz_gold_rate']):>9}")

    # paired gains for the benign series
    e3 = PACKAGE / "data/interaction/e3_benign.json"
    if e3.exists():
        payload = json.loads(e3.read_text())
        cells = analyse_debate(payload, tasks)
        report["paired_gains_e3"] = {
            str(n): paired_gains(cells, n) for n in (3, 5, 7, 10)
        }
        print("\n--- benign debate: round2 - round0, paired over tasks ---")
        for n in (3, 5, 7, 10):
            g = report["paired_gains_e3"][str(n)]
            if not g or "agree_gain" not in g:
                continue
            ag, ac, dm = g["agree_gain"], g.get("acc_gain"), g.get("agree_minus_acc")
            print(f"  n={n:<3} agree {ag['estimate']:+.4f} [{ag['lo']:+.4f},{ag['hi']:+.4f}]"
                  f"   acc {ac['estimate']:+.4f} [{ac['lo']:+.4f},{ac['hi']:+.4f}]"
                  f"   diff {dm['estimate']:+.4f} [{dm['lo']:+.4f},{dm['hi']:+.4f}]")

    (RESULTS / "debate_analysis.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {RESULTS / 'debate_analysis.json'}")


if __name__ == "__main__":
    main()
