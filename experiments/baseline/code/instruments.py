"""Semantic instruments and population heterogeneity.

Two analyses that both feed the paper's third section.

Instruments
-----------
The paper's fine-grained class is ``(normalised answer, normalised KEY)``.  That
is a *lexical* partition, and the question is what it measures.  We compare the
predicates several candidate instruments induce on the same within-task pairs:

  ``answer_only``  final-answer equality
  ``key_exact``    normalised KEY equality
  ``lexical``      answer+KEY equality (the paper's class)
  ``embedding``    cosine similarity of a sentence embedding above a threshold
  ``judge``        an LLM asked whether two replies are "the same solution"
  ``adjudicator``  a stronger LLM asked the same question

``answer_only`` and ``lexical`` are partitions and yield A(h); ``key_exact`` is
a partition of the KEY field alone.  The embedding threshold and the LLM judges
are *pairwise predicates*: neither is transitive, and the archived data
contains no class construction for them, so no A(h) is derived from them.  We
report them as pairwise SAME rates split by whether the final answers agree,
because that split is what shows the two LLM judges answer the verdict question
rather than the semantic one.

Heterogeneity
-------------
Mixed populations are recommended for fault decorrelation.  They also lower
A(h).  The v1 code computed the mixed quantity with a plug-in estimator while
the axis was labelled with the U-statistic symbol; v3 uses the U-statistic
throughout, on a common task set, and reports both comparisons:

  versus the best single model  — the choice a deployer actually faces
  versus the averaged distribution — what Proposition 5 (AM-GM) licenses

Conflating the two is what let the v1 text claim more than the proposition
supports, as the review noted.
"""
from __future__ import annotations

import collections
import itertools
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collision import A_unbiased, falling_factorial, task_bootstrap  # noqa: E402
from scoring import answer_class, norm_key, semantic_class  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
V3 = HERE.parent
ROOT = V3.parent
RESULTS = V3 / "results"

MODELS = ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1-mini"]
SOURCE_FILES = ("results/samples.jsonl", "results/samples2.jsonl")
HS = (3, 6, 9)
PAIR_SAMPLE = 4000          # per model, to bound the pairwise recomputation
SEED = 20260911


# --------------------------------------------------------------------------
# raw records
# --------------------------------------------------------------------------
_TASKS = None
_BY_MODEL = None


def tasks():
    global _TASKS
    if _TASKS is None:
        _TASKS = {r["id"]: r for r in map(
            json.loads, (ROOT / "data" / "tasks.jsonl").read_text().splitlines())}
    return _TASKS


def records_by_model():
    """``by[model][task_id] = [record, ...]`` for the CoT files."""
    global _BY_MODEL
    if _BY_MODEL is None:
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for rel in SOURCE_FILES:
            path = ROOT / rel
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("model") in MODELS:
                    by[rec["model"]][rec["task_id"]].append(rec)
        _BY_MODEL = by
    return _BY_MODEL


# --------------------------------------------------------------------------
# instruments
# --------------------------------------------------------------------------
def _sampled_pairs(recs, limit, seed):
    """Unordered index pairs, deterministically subsampled when too many."""
    idx = list(itertools.combinations(range(len(recs)), 2))
    if limit and len(idx) > limit:
        import random
        idx = random.Random(seed).sample(idx, limit)
    return idx


def instrument_pair_rates(limit=PAIR_SAMPLE):
    """Pairwise SAME rates, recomputed where possible and merged where not."""
    by = records_by_model()
    counters = {k: collections.Counter() for k in
                ("answer_only", "key_exact", "lexical")}
    n_pairs = 0

    for model in MODELS:
        for tid, recs in sorted(by[model].items()):
            bench = tasks()[tid]["bench"]
            for i, j in _sampled_pairs(recs, limit, SEED):
                a, b = recs[i], recs[j]
                sa = answer_class(a.get("answer"), bench)
                sb = answer_class(b.get("answer"), bench)
                if sa is None or sb is None:
                    continue
                ka, kb = norm_key(a.get("key")), norm_key(b.get("key"))
                same_ans = sa == sb
                verdicts = {
                    "answer_only": same_ans,
                    "key_exact": ka is not None and ka == kb,
                    "lexical": semantic_class(a.get("answer"), a.get("key"), bench)
                    == semantic_class(b.get("answer"), b.get("key"), bench),
                }
                n_pairs += 1
                for name, val in verdicts.items():
                    c = counters[name]
                    c["den"] += 1
                    c["same"] += int(val)
                    if same_ans:
                        c["den_sa"] += 1
                        c["same_sa"] += int(val)
                    else:
                        c["den_da"] += 1
                        c["same_da"] += int(val)

    out = {}
    for name, c in counters.items():
        if not c["den"]:
            continue
        out[name] = {
            "n_pairs": c["den"],
            "same_rate": c["same"] / c["den"],
            "same_given_same_answer": (c["same_sa"] / c["den_sa"]) if c["den_sa"] else None,
            "same_given_diff_answer": (c["same_da"] / c["den_da"]) if c["den_da"] else None,
            "n_same_answer": c["den_sa"],
            "n_diff_answer": c["den_da"],
            "is_partition": name in ("answer_only", "key_exact", "lexical"),
            "source": "recomputed from stored CoT replies",
        }

    arch = ROOT / "results" / "instrument2.json"
    if arch.exists():
        try:
            blob = json.loads(arch.read_text())
        except json.JSONDecodeError:
            blob = None
        if isinstance(blob, dict):
            rows = (blob.get("results") or {}).get("_pairs") or blob.get("pairs")
            sub = _archived_rates(rows)
            if sub:
                out.update(sub)
    return {"n_pairs_total": n_pairs, "per_model_pair_cap": limit,
            "seed": SEED, "instruments": out}


def _archived_rates(rows):
    """Rates for the archived embedding / LLM-judge predicates.

    Rows look like ``{"task":..., "a":[answer,key], "b":[...],
    "same_answer":bool, "lexical":bool, "embedding":bool, "judge":bool,
    "ref":bool}``.  ``same_answer`` is the verdict comparison; the booleans are
    the instruments.  No class construction accompanies them.
    """
    if not rows:
        return None
    out = {}
    for field, name in (("lexical", "lexical_archived"),
                        ("embedding", "embedding"),
                        ("judge", "judge"),
                        ("ref", "adjudicator")):
        same = tot = 0
        same_sa = tot_sa = same_da = tot_da = 0
        for p in rows:
            v = p.get(field)
            if v is None:
                continue
            v = bool(v)
            tot += 1
            same += int(v)
            if bool(p.get("same_answer")):
                tot_sa += 1
                same_sa += int(v)
            else:
                tot_da += 1
                same_da += int(v)
        if tot:
            out[name] = {
                "n_pairs": tot,
                "same_rate": same / tot,
                "same_given_same_answer": (same_sa / tot_sa) if tot_sa else None,
                "same_given_diff_answer": (same_da / tot_da) if tot_da else None,
                "n_same_answer": tot_sa,
                "n_diff_answer": tot_da,
                "is_partition": False,
                "source": "archived results/instrument2.json",
            }
    return out or None


# --------------------------------------------------------------------------
# heterogeneity
# --------------------------------------------------------------------------
def _class_counts(tid, bench, lvl):
    """Per-model class counters over *aligned* labels on one task."""
    out = {}
    for m in MODELS:
        recs = records_by_model()[m].get(tid, [])
        if lvl == "verdict":
            vals = [answer_class(r.get("answer"), bench) for r in recs]
            vals = [v for v in vals if v is not None]
        else:
            vals = []
            for r in recs:
                if answer_class(r.get("answer"), bench) is None:
                    continue
                vals.append(semantic_class(r.get("answer"), r.get("key"), bench))
        out[m] = collections.Counter(vals)
    return out


def _single_A(counts, h):
    return A_unbiased(sorted(counts.values(), reverse=True), h)


def _mixture_A(counts_by_model, h_per_model):
    """``sum_c prod_m [n_{m,c}]_{h_m} / [k_m]_{h_m}`` for a split population."""
    allc = set()
    for m in counts_by_model:
        allc |= set(counts_by_model[m])
    total = 0.0
    for c in allc:
        prod = 1.0
        for m, hm in h_per_model.items():
            if hm == 0:
                continue
            k = sum(counts_by_model[m].values())
            if k < hm:
                return None
            prod *= (falling_factorial(counts_by_model[m].get(c, 0), hm)
                     / falling_factorial(k, hm))
        total += prod
    return total if total > 0 else None


def _averaged_counts(counts_by_model):
    """Per-class probability averaged over models, as a count vector."""
    allc = set()
    for m in counts_by_model:
        allc |= set(counts_by_model[m])
    tot = sum(sum(c.values()) for c in counts_by_model.values()) or 1
    out = []
    for c in allc:
        p = sum(counts_by_model[m].get(c, 0) / max(1, sum(counts_by_model[m].values()))
                for m in MODELS) / len(MODELS)
        out.append(p * tot)
    return out


def heterogeneity_analysis(min_samples=None):
    """A^U(h) per single model, for the mixture, and for the averaged law."""
    src = RESULTS / "e1_analysis.json"
    if not src.exists():
        return {}
    common = set(json.loads(src.read_text())["common_tasks"]["ids"])
    need = max(HS) if min_samples is None else min_samples

    out = {}
    for lvl in ("verdict", "semantic"):
        # The accumulator dict must carry a "rel_loss" key from the start: it is
        # filled inside the task loop, and the per-cell means below skip that
        # key.  Building it lazily would both drop the ratio when no task
        # qualified and let a stale key be averaged as if it were a value.
        per_h = {h: {"mixed": [], "best_single": [], "worst_single": [],
                     "averaged_amgm": [], "rel_loss": []} for h in HS}
        used = 0
        for tid in sorted(common):
            bench = tasks()[tid]["bench"]
            counts = _class_counts(tid, bench, lvl)
            if any(sum(c.values()) < need for c in counts.values()):
                continue
            used += 1
            for h in HS:
                if h % len(MODELS):
                    continue
                hpm = {m: h // len(MODELS) for m in MODELS}
                mixed = _mixture_A(counts, hpm)
                singles = {m: _single_A(counts[m], h) for m in MODELS}
                # A^U is undefined when a model has fewer than h records on the
                # task; those tasks are dropped from every column together, so
                # the comparison is not made across different task sets.
                if mixed is None or any(v is None for v in singles.values()):
                    continue
                best = max(singles.values())
                per_h[h]["mixed"].append(mixed)
                per_h[h]["best_single"].append(best)
                per_h[h]["worst_single"].append(min(singles.values()))
                avg = A_unbiased(_averaged_counts(counts), h)
                per_h[h]["averaged_amgm"].append(avg)
                # A relative loss is only meaningful when the reference value
                # is not itself numerically zero; at semantic granularity the
                # best single model can be at the resolution floor, in which
                # case the ratio is reported as undefined rather than as a
                # spurious percentage.
                if best > 1e-9:
                    per_h[h]["rel_loss"].append((best - mixed) / best)
        out[lvl] = {
            "n_tasks": used,
            "min_samples_per_model_per_task": need,
            "cells": {
                f"h={h}": {
                    k: (round(sum(v) / len(v), 6) if v else None)
                    for k, v in d.items() if k != "rel_loss"
                } | {
                    "relative_loss_vs_best_single":
                        round(sum(d["rel_loss"]) / len(d["rel_loss"]), 6)
                        if d["rel_loss"] else None,
                    # How many tasks actually contributed to the ratio.  At
                    # semantic granularity this can be a handful, in which case
                    # the ratio is not a population quantity.  It is also only
                    # the mixed-versus-best-single comparison; Proposition 5
                    # (AM-GM) licenses mixed versus the *averaged* law, which is
                    # the averaged_amgm column, and that comparison holds at
                    # every h.
                    "n_for_relative_loss": len(d["rel_loss"]),
                    "relative_loss_is_stable": len(d["rel_loss"]) >= 20,
                }
                for h, d in per_h.items()
            },
        }
    return out


def main():
    RESULTS.mkdir(exist_ok=True)
    report = {
        "config": {"models": list(MODELS), "h_values": list(HS),
                   "pair_cap_per_model": PAIR_SAMPLE, "seed": SEED,
                   "estimator": "falling-factorial U-statistic throughout"},
        "instruments": instrument_pair_rates(),
        "heterogeneity": heterogeneity_analysis(),
    }
    (RESULTS / "instruments_heterogeneity.json").write_text(
        json.dumps(report, indent=1, default=str))

    ins = report["instruments"]
    print("=== instrument pairwise SAME rates ===")
    print(f"pairs examined (recomputed): {ins['n_pairs_total']}")
    print(f"  {'instrument':<20}{'n':>7}{'SAME':>9}{'P(S|same ans)':>15}"
          f"{'P(S|diff ans)':>15}  partition?")
    for name, v in (ins.get("instruments") or {}).items():
        def f_(x):
            return f"{x:.4f}" if isinstance(x, float) else "  n/a "
        print(f"  {name:<20}{v['n_pairs']:>7}{f_(v['same_rate']):>9}"
              f"{f_(v['same_given_same_answer']):>15}"
              f"{f_(v['same_given_diff_answer']):>15}  {v.get('is_partition')}")

    print("\n=== heterogeneity: mean A^U(h) over tasks ===")
    het = report["heterogeneity"]
    for lvl, block in het.items():
        print(f"-- {lvl} (n={block['n_tasks']} tasks, >={block['min_samples_per_model_per_task']} samples/model) --")
        print(f"{'h':>3}{'best single':>13}{'worst single':>14}{'mixed':>10}"
              f"{'averaged':>10}{'rel loss':>10}")
        for key, v in block["cells"].items():
            rl = v["relative_loss_vs_best_single"]
            rl_s = f"{rl:>10.3f}" if isinstance(rl, float) else f"{'n/a':>10}"
            if isinstance(rl, float) and not v["relative_loss_is_stable"]:
                rl_s += f" ({v['n_for_relative_loss']} tasks)"
            print(f"{key.split('=')[1]:>3}{v['best_single']:>13.4f}"
                  f"{v['worst_single']:>14.4f}{v['mixed']:>10.4f}"
                  f"{v['averaged_amgm']:>10.4f}{rl_s}")
    print(f"\nwrote {RESULTS / 'instruments_heterogeneity.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
