"""Certification coverage on the measured proposal distributions.

Turns Theorem 2 into a number: for each model and each agreement granularity,
how often does the margin rule certify the honest plurality, as a function of
the fault budget ``f``?

Three events are reported, because the v1 paper reported only the first and it
does not answer the question the paper asks:

  ``pairwise``   P(N(c*) > N(c_b) + f) against a *designated* rival.  This is
                 what the v1 threshold table computed.  It is not the
                 certification event: it also holds on draws where a third
                 class wins outright.
  ``plurality``  P(N(c*) > max_{d != c*} N(d) + f), the event the rule actually
                 certifies, computed exactly under a three-group model
                 (c*, its runner-up, and the lumped remainder) and
                 cross-checked by Monte Carlo on the same model.
  ``full``       The same probability when the whole measured alphabet is
                 simulated instead of being lumped.  This is what a deployment
                 would see, and it is *higher* than ``plurality`` because
                 spreading the rival mass over many classes is easier to beat
                 than concentrating it.

Two adversaries appear, and the distinction matters.  ``adaptive`` places all
f votes on whichever rival currently leads, after seeing the honest draw;
``fixed`` commits to the runner-up of the honest distribution beforehand.
Coverage under the adaptive adversary is the headline number.

Reading: this is the coverage of a *specified certificate* under a *specified
per-task label distribution*.  It is not the end-to-end fault tolerance of any
networked protocol, and Section 5.5 of the paper says so.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from certify import (certify, coverage_exact_three_class,  # noqa: E402
                     coverage_exact_pairwise, coverage_montecarlo)
from collision import task_bootstrap  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
V3 = HERE.parent
RESULTS = V3 / "results"

HS = (3, 5, 7, 10, 20)
FS = (0, 1, 2, 3, 4, 5)


def _split(counts):
    """(p_top, p_second) of the observed distribution."""
    total = sum(counts)
    if total <= 0 or not counts:
        return None
    ordered = sorted(counts, reverse=True)
    p_top = ordered[0] / total
    p_second = ordered[1] / total if len(ordered) > 1 else 0.0
    return p_top, p_second


def task_exact(counts, h, f, rule):
    """Exact three-group probability that ``rule`` fires on one task."""
    sp = _split(counts)
    if sp is None:
        return None
    p_top, p_second = sp
    if rule == "pairwise":
        return coverage_exact_pairwise(p_top, p_second, h, f)
    return coverage_exact_three_class(p_top, p_second, h, f)


def analyse(per_task, common, models, h=10, f_grid=FS):
    out = {}
    for model in models:
        rows = [r for r in per_task[model] if r["id"] in common]
        out[model] = {}
        for lvl, field in (("verdict", "counts_verdict"),
                           ("semantic", "counts_semantic")):
            lvl_out = {"n_tasks": len(rows)}
            for f in f_grid:
                pw, pl, full_fixed, full_adapt = [], [], [], []
                for r in rows:
                    counts = sorted(r[field], reverse=True)
                    if not counts:
                        continue
                    a = task_exact(counts, h, f, "pairwise")
                    b = task_exact(counts, h, f, "plurality")
                    if a is not None:
                        pw.append(a)
                    if b is not None:
                        pl.append(b)
                    total = sum(counts)
                    dist = {i: c / total for i, c in enumerate(counts)}
                    full_fixed.append(coverage_montecarlo(
                        dist, h, f, rounds=4000, seed=20260911 + f,
                        adaptive_attack=False, target=0)["coverage"])
                    full_adapt.append(coverage_montecarlo(
                        dist, h, f, rounds=4000, seed=20260911 + f,
                        adaptive_attack=True, target=0)["coverage"])
                if not pl:
                    continue
                lvl_out[f"f={f}"] = {
                    "pairwise_exact": round(sum(pw) / len(pw), 6),
                    "plurality_three_group_exact": round(sum(pl) / len(pl), 6),
                    "pairwise_minus_plurality": round(
                        sum(pw) / len(pw) - sum(pl) / len(pl), 6),
                    "full_alphabet_fixed_mc": round(
                        sum(full_fixed) / len(full_fixed), 6),
                    "full_alphabet_adaptive_mc": round(
                        sum(full_adapt) / len(full_adapt), 6),
                    "n": len(pl),
                }
            for conf in (0.90, 0.99):
                vals = []
                for r in rows:
                    counts = sorted(r[field], reverse=True)
                    if not counts:
                        continue
                    best = -1
                    for f in range(0, len(counts) + 1):
                        if task_exact(counts, h, f, "plurality") >= conf:
                            best = f
                        else:
                            break
                    vals.append(best)
                if vals:
                    lvl_out[f"f_star_conf{conf}"] = {
                        "mean": round(sum(vals) / len(vals), 4),
                        "frac_ge_0": round(sum(1 for v in vals if v >= 0) / len(vals), 4),
                        "frac_ge_1": round(sum(1 for v in vals if v >= 1) / len(vals), 4),
                        "frac_ge_3": round(sum(1 for v in vals if v >= 3) / len(vals), 4),
                        "classical_n_over_3": 3,
                        "n": len(vals),
                    }
            out[model][lvl] = lvl_out
    return out


def montecarlo_crosscheck(per_task, common, models, h=10, f_grid=(0, 2, 4),
                          rounds=20000):
    """Independent Monte-Carlo confirmation of the exact coverage numbers.

    Uses ``lump_to_three=True`` and a fixed rival so that the simulation and
    the exact computation assume the *same* model.  Simulating the full
    alphabet or an adaptive attacker instead would measure a different
    quantity, which is reported separately in :func:`analyse`.
    """
    out = {}
    for model in models:
        rows = [r for r in per_task[model] if r["id"] in common]
        out[model] = {}
        for lvl, field in (("verdict", "counts_verdict"),
                           ("semantic", "counts_semantic")):
            lvl_out = {}
            for f in f_grid:
                diffs, mcvals = [], []
                for r in rows:
                    counts = sorted(r[field], reverse=True)
                    if not counts:
                        continue
                    total = sum(counts)
                    dist = {i: c / total for i, c in enumerate(counts)}
                    mc = coverage_montecarlo(dist, h, f, rounds=rounds,
                                             seed=20260911 + f,
                                             lump_to_three=True,
                                             adaptive_attack=False, target=0)
                    ex = task_exact(counts, h, f, "plurality")
                    mcvals.append(mc["coverage"])
                    diffs.append(abs(mc["coverage"] - ex))
                if mcvals:
                    lvl_out[f"f={f}"] = {
                        "mc_coverage": round(sum(mcvals) / len(mcvals), 6),
                        "max_abs_diff_vs_exact": round(max(diffs), 6),
                        "mean_abs_diff_vs_exact": round(sum(diffs) / len(diffs), 6),
                        "n": len(mcvals), "rounds": rounds,
                    }
            out[model][lvl] = lvl_out
    return out


def main():
    RESULTS.mkdir(exist_ok=True)
    src = RESULTS / "e1_analysis.json"
    if not src.exists():
        print("run recompute_e1.py first")
        return 1
    data = json.loads(src.read_text())
    per_task = data["per_task"]
    common = set(data["common_tasks"]["ids"])
    models = list(per_task)

    report = {
        "config": {"h": 10, "f_grid": list(FS),
                   "coverage": "exact via three-group lumping; see certify.py",
                   "task_set": f"{len(common)} common tasks"},
        "coverage": analyse(per_task, common, models),
        "montecarlo_crosscheck": montecarlo_crosscheck(per_task, common, models),
    }
    (RESULTS / "threshold_analysis.json").write_text(json.dumps(report, indent=1))

    print(f"h=10 honest agents, {len(common)} common tasks\n")
    for model in models:
        for lvl in ("verdict", "semantic"):
            block = report["coverage"][model][lvl]
            print(f"--- {model} / {lvl}  (n={block['n_tasks']} tasks) ---")
            print(f"{'f':>3}{'pairwise':>11}{'plur(3grp)':>12}{'overstate':>11}"
                  f"{'full fixed':>12}{'full adapt':>12}")
            for f in FS:
                cell = block.get(f"f={f}")
                if not cell:
                    continue
                print(f"{f:>3}{cell['pairwise_exact']:>11.4f}"
                      f"{cell['plurality_three_group_exact']:>12.4f}"
                      f"{cell['pairwise_minus_plurality']:>+11.4f}"
                      f"{cell['full_alphabet_fixed_mc']:>12.4f}"
                      f"{cell['full_alphabet_adaptive_mc']:>12.4f}")
            for conf in (0.90, 0.99):
                st = block.get(f"f_star_conf{conf}")
                if st:
                    print(f"    f* @{conf}: mean={st['mean']:>7.3f}  "
                          f"P(f*>=0)={st['frac_ge_0']:.3f}  "
                          f"P(f*>=1)={st['frac_ge_1']:.3f}  "
                          f"P(f*>=3)={st['frac_ge_3']:.3f}   "
                          f"(classical n/3 = 3)")
            print()

    print("--- Monte-Carlo cross-check of the exact three-group numbers ---")
    for model in models:
        for lvl in ("verdict", "semantic"):
            cs = report["montecarlo_crosscheck"][model][lvl]
            parts = [f"f={k.split('=')[1]}: mean|d|={v['mean_abs_diff_vs_exact']:.4f}"
                     f" max|d|={v['max_abs_diff_vs_exact']:.4f}"
                     for k, v in cs.items()]
            print(f"  {model:<15}{lvl:<9} " + "  ".join(parts))
    print(f"\nwrote {RESULTS / 'threshold_analysis.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
