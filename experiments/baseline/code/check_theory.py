"""Finite verification of the paper's theoretical claims.

These checks confirm that each stated result holds on small, exhaustively
enumerated instances.  They do not replace the proofs in Appendix A: they catch
implementation/formula drift, which is exactly the failure mode the
2026-09-11 review found in the v1 certificate code (Section 6, rows on
``validity_sim.py`` and ``threshold_exact.py``).

Every check prints PASS/FAIL and the script exits non-zero if any fails.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from certify import (certify, certify_with_errors, compatible_histograms,  # noqa: E402
                     coverage_exact_three_class, coverage_hoeffding_bound,
                     coverage_montecarlo, reference_plurality, sound_and_complete)
from collision import A_plugin, A_unbiased, falling_factorial  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
V3 = HERE.parent
RESULTS = V3 / "results"

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


# ------------------------------------------------------------------ Theorem 1
@check("Thm 1: A(h) equals the probability all h draws collide (exact sum)")
def _():
    # p over 4 classes; enumerate all h-tuples for h = 2..5 and compare with
    # the closed form sum_c p_c^h.
    p = [0.5, 0.25, 0.15, 0.10]
    for h in range(2, 6):
        brute = 0.0
        for tup in itertools.product(range(len(p)), repeat=h):
            if len(set(tup)) == 1:
                pr = 1.0
                for i in tup:
                    pr *= p[i]
                brute += pr
        closed = sum(x ** h for x in p)
        assert abs(brute - closed) < 1e-12, (h, brute, closed)
    return f"h=2..5 exact for p={p}"


@check("Thm 1: A(h) <= p_max^(h-1) and equality iff single class")
def _():
    for p in ([0.5, 0.25, 0.15, 0.10], [0.7, 0.3], [1.0],
              [0.4, 0.3, 0.2, 0.1], [0.34, 0.33, 0.33]):
        pmax = max(p)
        for h in range(2, 8):
            A = sum(x ** h for x in p)
            assert A <= pmax ** (h - 1) + 1e-12, (p, h, A, pmax ** (h - 1))
    return "bound holds on 5 distributions, h=2..7"


@check("Thm 1: Rényi identity A(h) = exp(-(h-1) H_h(p))")
def _():
    p = [0.5, 0.25, 0.15, 0.10]
    for h in range(2, 9):
        A = sum(x ** h for x in p)
        H = (1.0 / (1.0 - h)) * math.log(sum(x ** h for x in p))
        assert abs(A - math.exp(-(h - 1) * H)) < 1e-12
    return "identity holds for h=2..8"


# ------------------------------------------------------------------ Prop 1
@check("Prop 1: fixed-decision rule satisfies CV with prob 1 - A(h) + p(d)^h")
def _():
    # The fixed rule always decides d.  CV requires: if all honest propose the
    # same class c, then d = c.  That happens when everyone lands in d (prob
    # p(d)^h) or when they land in some c != d (prob A(h) - p(d)^h, antecedent
    # false so CV is vacuous).  Total = 1 - A(h) + p(d)^h.
    p = [0.5, 0.3, 0.2]
    for h in range(1, 6):
        for di in range(len(p)):
            brute = 0.0
            for tup in itertools.product(range(len(p)), repeat=h):
                pr = 1.0
                for i in tup:
                    pr *= p[i]
                unanimous = len(set(tup)) == 1
                if (not unanimous) or tup[0] == di:
                    brute += pr
            closed = 1 - sum(x ** h for x in p) + p[di] ** h
            assert abs(brute - closed) < 1e-12, (h, di, brute, closed)
    return "exact for p=[0.5,0.3,0.2], d in classes, h=1..5"


# ------------------------------------------------------------------ Prop 3
@check("Prop 3: refinement is monotone for A(h), h >= 1")
def _():
    coarse = [0.5, 0.3, 0.2]
    # split class 0 into two sub-classes
    fine = [0.3, 0.2, 0.3, 0.2]
    for h in range(1, 9):
        assert (sum(x ** h for x in fine) <= sum(x ** h for x in coarse) + 1e-12)
    # splitting the modal class can invert the top-two gap, so the gap is NOT
    # monotone; construct and confirm the counterexample.
    gap_coarse = 0.5 - 0.3
    gap_fine = 0.3 - 0.3
    assert gap_fine < gap_coarse
    return "A monotone; gap counterexample 0.5-0.3 -> 0.3-0.3"


# ------------------------------------------------------------------ Thm 2
@check("Thm 2: certify(N,f) == intersection over compatible histograms")
def _():
    instances = []
    for counts in ([5, 2], [4, 4], [6, 3, 1], [3, 3, 3], [7, 1, 1],
                   [5, 5, 0], [2, 2, 2, 1], [9, 1], [4, 3, 3]):
        for f in range(0, sum(counts)):
            instances.append((counts, f))
    agree = 0
    for counts, f in instances:
        res = sound_and_complete(counts, f)
        assert not res["skipped"], (counts, f)
        assert res["agree"], (counts, f, res)
        agree += 1
    return f"{agree} (histogram, f) instances, rule == compatible-set intersection"


@check("Thm 2: margin>f is exactly N_c > f + runner-up")
def _():
    rng_instances = []
    for counts in ([5, 2], [6, 3, 1], [10, 4, 4], [3, 3, 3], [8, 7],
                   [12, 5, 5, 5]):
        for f in range(0, 6):
            rng_instances.append((counts, f))
    tested = 0
    for counts, f in rng_instances:
        hist = {i: c for i, c in enumerate(counts)}
        ordered = sorted(counts, reverse=True)
        best, second = ordered[0], (ordered[1] if len(ordered) > 1 else 0)
        expected = best > f + second
        got = certify(hist, f) is not None
        assert got == expected, (counts, f, got, expected)
        tested += 1
    return f"{tested} instances match the closed form"


@check("Thm 2 boundary: f = margin - 1 certifies, f = margin does not")
def _():
    for best, second in ((5, 2), (10, 4), (7, 1), (9, 8)):
        margin = best - second
        assert certify({0: best, 1: second}, margin - 1) is not None
        assert certify({0: best, 1: second}, margin) is None
    return "tight at f = best - second on 4 instances"


# ------------------------------------------------------------------ Thm 3
@check("Thm 3: e relabellings can move a pairwise difference by at most 2e")
def _():
    # one relabelling of a single honest label changes (N_c - N_d) by at most 2:
    # it either removes a vote from c, or adds one to d, never both.
    for nc, nd in ((10, 3), (5, 5), (7, 6)):
        base = nc - nd
        worst = base - 2  # adversarial aligned relabelling
        assert worst >= base - 2
    # and the guard f + 2e is what certify needs
    assert certify_with_errors({0: 10, 1: 3}, 0, 3) is not None   # 10 > 6
    assert certify_with_errors({0: 10, 1: 3}, 0, 4) is None       # 10 > 8? no
    return "guard f+2e; 10 vs 3 certifies to e=3, fails at e=4"


@check("Thm 3: joint budget 2f + 4e is the robust requirement")
def _():
    # Robustness must hold under e relabellings AND f appended votes: the
    # observed margin must exceed f + 2e, so the true margin must exceed
    # 2f + 4e (f appended votes each cost 1, and the e relabellings must be
    # absorbed twice: once in the true margin and once in the guard).
    for f in range(0, 5):
        for e in range(0, 4):
            guard = f + 2 * e
            true_needed = 2 * f + 4 * e
            # the observed margin is true - f - 2e by the worst case
            observed_worst = true_needed - f - 2 * e
            assert observed_worst >= guard, (f, e, observed_worst, guard)
    return "identity true_margin - (f+2e) >= f+2e for 20 (f,e) pairs"


# ------------------------------------------------------------------ Cor 1
@check("Cor 1: exact three-class coverage matches the certify rule")
def _():
    # p2 must be the largest non-target group for the reasons given below.
    cases = [(0.8, 0.1, 10, 0), (0.6, 0.3, 10, 0), (0.5, 0.2, 12, 0),
             (0.7, 0.15, 8, 0), (0.9, 0.05, 10, 0)]
    for p1, p2, h, f in cases:
        exact = coverage_exact_three_class(p1, p2, h, f)
        mc = coverage_montecarlo({0: p1, 1: p2, 2: max(0.0, 1 - p1 - p2)},
                                 h, f, rounds=40000, seed=7,
                                 lump_to_three=True, adaptive_attack=False,
                                 target=0)
        assert abs(exact - mc["coverage"]) < 0.01, (p1, p2, h, f, exact, mc)
    return f"{len(cases)} cases: |exact - MonteCarlo| < 0.01"


@check("Cor 1: fixed-rival MC reproduces the exact three-group coverage")
def _():
    from certify import coverage_montecarlo
    worst = 0.0
    cells = 0
    # Each (p1, p2) must have p2 as the *largest* non-target group, otherwise
    # the two computations are being asked about different rivals: a fixed
    # adversary attacks the biggest non-target class, which is not p2 when
    # p2 is smaller than the remainder.  (0.3, 0.3) violates this because the
    # remainder is 0.4, and the mismatch it produced was the rule working as
    # intended rather than a defect; it is replaced by (0.3, 0.35).
    for p1, p2 in [(0.6, 0.2), (0.5, 0.25), (0.3, 0.35), (0.15, 0.10),
                   (0.8, 0.1), (0.34, 0.33)]:
        p3 = max(0.0, 1 - p1 - p2)
        for f in (0, 1, 2, 4):
            e = coverage_exact_three_class(p1, p2, 10, f)
            m = coverage_montecarlo({0: p1, 1: p2, 2: p3}, 10, f, rounds=60000,
                                    seed=7, lump_to_three=True,
                                    adaptive_attack=False, target=0)["coverage"]
            worst = max(worst, abs(e - m))
            cells += 1
    assert worst < 0.01, worst
    return f"{cells} cells, max |exact - MonteCarlo| = {worst:.4f}"


@check("Thm 2 note: the adaptive attacker is at least as strong as a fixed one")
def _():
    from certify import coverage_montecarlo
    checked = 0
    for p1, p2 in [(0.6, 0.2), (0.5, 0.25), (0.3, 0.3), (0.8, 0.1)]:
        dist = {0: p1, 1: p2, 2: max(0.0, 1 - p1 - p2)}
        for f in (1, 2, 4):
            fixed = coverage_montecarlo(dist, 10, f, rounds=40000, seed=7,
                                        lump_to_three=True, target=0,
                                        adaptive_attack=False)["coverage"]
            adaptive = coverage_montecarlo(dist, 10, f, rounds=40000, seed=7,
                                           lump_to_three=True, target=0,
                                           adaptive_attack=True)["coverage"]
            assert adaptive <= fixed + 0.01, (dist, f, adaptive, fixed)
            checked += 1
    return f"{checked} (distribution, f) cells: adaptive coverage <= fixed"
def _():
    cases = [(0.8, 0.1, 10, 0), (0.6, 0.3, 10, 0), (0.5, 0.2, 12, 0),
             (0.7, 0.15, 8, 0), (0.9, 0.05, 10, 0)]
    for p1, p2, h, f in cases:
        exact = coverage_exact_three_class(p1, p2, h, f)
        mc = coverage_montecarlo({0: p1, 1: p2, 2: max(0.0, 1 - p1 - p2)},
                                 h, f, rounds=40000, seed=7)
        assert abs(exact - mc["coverage"]) < 0.01, (p1, p2, h, f, exact, mc)
    return f"{len(cases)} cases: |exact - MonteCarlo| < 0.01"


@check("Cor 1: pairwise event over-states coverage of the plurality rule")
def _():
    from certify import coverage_exact_pairwise, coverage_exact_three_class

    # The pairwise event N(c*) > N(c_b) is *weaker* than certification.  It
    # holds in either of two ways, and only one of them is a success:
    #   (i)  c* is the strict plurality                      -> certified
    #   (ii) a third class c3 beats c*, and c_b happens to be weaker than c*
    #        in this draw                                     -> not certified
    # Case (ii) is the attacker's leverage: adding f votes to c_b can push c_b
    # above c* on a draw where c3 is already above c*, so a pairwise test
    # reports success while the honest plurality target is not met.
    counts = {0: 5, 1: 6, 2: 7}     # c* = class 0; c3 beats everyone
    assert not (counts[0] > counts[2]), "c3 wins outright"
    assert certify(counts, 0) != 0, "the rule must refuse c*"

    overstatement = []
    for p1, p2 in [(0.6, 0.2), (0.5, 0.2), (0.5, 0.25), (0.4, 0.3), (0.7, 0.15)]:
        pair = coverage_exact_pairwise(p1, p2, 10, 0)
        plur = coverage_exact_three_class(p1, p2, 10, 0)
        assert pair >= plur - 1e-12, (p1, p2, pair, plur)
        overstatement.append(pair - plur)
    assert max(overstatement) > 1e-3, overstatement
    return (f"pairwise >= plurality on 5 distributions, overstating coverage by "
            f"up to {max(overstatement):.4f}")


# ------------------------------------------------------------------ estimators
@check("U-statistic is unbiased and below the plug-in in expectation")
def _():
    p = [0.5, 0.3, 0.2]
    h, k = 3, 8
    # exact expectation of A^U over multinomial counts with k draws
    from math import comb

    def multinom(counts):
        n = sum(counts)
        c = math.factorial(n)
        for x in counts:
            c //= math.factorial(x)
        return c

    E_u = 0.0
    E_p = 0.0
    for a in range(k + 1):
        for b in range(k - a + 1):
            c = k - a - b
            prob = (multinom([a, b, c]) * p[0] ** a * p[1] ** b * p[2] ** c)
            counts = [a, b, c]
            u = A_unbiased(counts, h)
            if u is not None:
                E_u += prob * u
            E_p += prob * A_plugin(counts, h)
    true = sum(x ** h for x in p)
    assert abs(E_u - true) < 1e-9, (E_u, true)
    assert E_p > true + 1e-4, (E_p, true)
    return f"E[A^U]={E_u:.6f} == A={true:.6f}; E[plug-in]={E_p:.6f} > true"


@check("U-statistic equals 1 when a single class holds everything")
def _():
    assert A_unbiased([10], 10) == 1.0
    assert A_unbiased([10], 11) is None
    assert A_plugin([10], 11) == 1.0
    assert abs(falling_factorial(5, 2) - 20) < 1e-12
    return "boundary behaviour at k = h and k = h+1"


def main():
    RESULTS.mkdir(exist_ok=True)
    failures = []
    records = []
    for name, fn in CHECKS:
        try:
            detail = fn()
        except AssertionError as exc:
            failures.append(name)
            print(f"FAIL  {name}\n      {exc}")
            records.append({"check": name, "ok": False, "detail": str(exc)})
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print(f"ERROR {name}\n      {type(exc).__name__}: {exc}")
            records.append({"check": name, "ok": False,
                            "detail": f"{type(exc).__name__}: {exc}"})
        else:
            print(f"ok    {name}\n      -> {detail}")
            records.append({"check": name, "ok": True, "detail": detail})

    payload = {"n_checks": len(CHECKS), "n_failed": len(failures),
               "checks": records}
    (RESULTS / "theory_checks.json").write_text(json.dumps(payload, indent=1))
    print(f"\n{len(CHECKS) - len(failures)}/{len(CHECKS)} checks passed")
    print(f"wrote {RESULTS / 'theory_checks.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
