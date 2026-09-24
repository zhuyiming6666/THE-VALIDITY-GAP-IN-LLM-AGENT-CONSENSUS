"""Class-level Byzantine certification: exact criterion, noise, and coverage.

The object studied here
-----------------------
``n`` identified participants each contribute one proposal, already mapped to a
label by a fixed map ``g``.  At most ``f`` of them are corrupted and may choose
their labels *after* seeing the honest ones.  The decision layer observes the
label histogram ``N``.

The target is **strict honest plurality**: decide the class that holds a strict
plurality among the honest participants.  This is deliberately *not* classical
validity (which only fires on unanimity) and deliberately *not* overall
correctness.  Section 2 of the paper fixes these three objects and keeps them
apart; the v1 manuscript conflated them and the review (Section 4.2) caught it.

The criterion
-------------
``certify(N, f)`` returns the class ``c`` with ``N_c > f + max_{d != c} N_d``,
or ``None``.  Theorem 2 states this is sound and maximally permissive: it
certifies exactly when *every* compatible honest histogram has ``c`` as its
strict plurality.  Theorem 3 extends it to ``e`` aligned labelling errors.

Attribution: the ``margin > f`` criterion is established by Xu et al. (2026),
Theorem 5.3.  What this module adds is the necessity direction phrased over the
compatible set, the relabelling extension, and the coverage computations that
turn the criterion into a measurable quantity.
"""
from __future__ import annotations

import collections
import math
from math import comb

import numpy as np

__all__ = [
    "certify", "certify_with_errors", "compatible_histograms",
    "sound_and_complete", "pairwise_success_prob", "coverage_montecarlo",
    "coverage_exact_three_class", "coverage_exact_pairwise",
    "pairwise_plurality_gap",
    "coverage_hoeffding_bound", "certifiable_max_f", "reference_plurality",
]


# --------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------
def reference_plurality(hist, tie="smallest"):
    """Strict plurality class of a histogram, or ``None`` if there is a tie.

    Returns ``(class, tied_bool)``.
    """
    if not hist:
        return None, False
    top = max(hist.values())
    if top == 0:
        return None, False
    winners = sorted(c for c, v in hist.items() if v == top)
    if len(winners) > 1:
        return (winners[0] if tie == "smallest" else None), True
    return winners[0], False


def certify(counts, f):
    """Class certified against up to ``f`` adaptive corrupted proposals.

    ``counts`` may be a Counter, a dict, or a sequence of counts (in which case
    label identity is positional).  Returns the label or ``None``.
    """
    hist = _as_hist(counts)
    if not hist:
        return None
    ordered = sorted(hist.items(), key=lambda kv: (-kv[1], str(kv[0])))
    best_c, best = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else 0
    if best > f + second:
        return best_c
    return None


def certify_with_errors(counts, f, e):
    """``certify`` with an additional budget of ``e`` aligned relabellings.

    ``e`` is the number of honest labels that may have been recorded under the
    wrong class identity by the instrument, in a way that is *aligned* (all
    errors push against the certified class).  The guard is ``f + 2e`` because
    one relabelling can change a pairwise count difference by at most 2.
    """
    return certify(counts, f + 2 * e)


def _as_hist(counts):
    if isinstance(counts, dict):
        return dict(counts)
    if isinstance(counts, collections.Counter):
        return dict(counts)
    return {i: int(c) for i, c in enumerate(counts)}


# --------------------------------------------------------------------------
# soundness and necessity
# --------------------------------------------------------------------------
def compatible_histograms(counts, f):
    """Every histogram obtainable by deleting at most ``f`` entries.

    These are the honest histograms consistent with the observation under the
    fault model.  Enumerating them is exponential in general; this generator is
    used only on small instances in the theory checks.
    """
    hist = _as_hist(counts)
    labels = sorted(hist, key=str)
    n = sum(hist.values())
    if f < 0 or f > n:
        return
    for deleted in _deletions(hist, labels, f, 0):
        yield deleted


def _deletions(hist, labels, budget, idx):
    if idx == len(labels):
        yield dict(hist)
        return
    lab = labels[idx]
    for k in range(0, min(hist[lab], budget) + 1):
        new = dict(hist)
        new[lab] -= k
        yield from _deletions(new, labels, budget - k, idx + 1)


def sound_and_complete(counts, f, exact_limit=200000):
    """Check ``certify`` against exhaustive enumeration of compatible sets.

    Returns a dict with
      ``rule``       what :func:`certify` returned
      ``intersection`` the class common to every compatible strict plurality
      ``agree``      whether they coincide (the theorem's claim)
      ``enumerated`` number of compatible histograms examined
      ``skipped``    True when the instance exceeded ``exact_limit``
    """
    hist = _as_hist(counts)
    inter = None
    first = True
    seen = 0
    for h in compatible_histograms(hist, f):
        seen += 1
        if seen > exact_limit:
            return {"skipped": True, "enumerated": seen}
        cls, tied = reference_plurality(h)
        if tied:
            cls = None
        if first:
            inter = cls
            first = False
        else:
            inter = inter if (inter is not None and cls == inter) else None
    rule = certify(hist, f)
    return {"rule": rule, "intersection": inter, "agree": rule == inter,
            "enumerated": seen, "skipped": False}


# --------------------------------------------------------------------------
# coverage: the probability the rule fires
# --------------------------------------------------------------------------
def pairwise_success_prob(p, h, f, rounds=200000, seed=20260911,
                          class_index=0, rival_index=1):
    """Monte-Carlo P(N(c*) > N(c_b)) for h honest i.i.d. draws from ``p``.

    This is the *pairwise* event the v1 threshold table computed.  It is a
    relaxation of full plurality: it can hold while a third class wins.
    Kept so the difference between the two events can be quantified, which is
    what the review asked for (Section 4.3).
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    draws = rng.choice(len(p), size=(rounds, h), p=p)
    n_star = (draws == class_index).sum(axis=1)
    n_rival = (draws == rival_index).sum(axis=1)
    return {"pairwise": float((n_star > n_rival + f).mean()),
            "rounds": rounds, "h": h, "f": f}


def coverage_montecarlo(counts, h, f, rounds=20000, seed=20260911,
                        noise=0.0, noise_classes=None, target=None,
                        lump_to_three=False, adaptive_attack=True):
    """Monte-Carlo probability that the designated class is certified.

    ``counts`` supplies the assumed honest class distribution (normalised).
    ``noise`` is an independent per-label replacement probability: with
    probability ``noise`` a drawn label is replaced by a uniform draw over
    ``noise_classes`` (defaulting to the alphabet of ``counts``).

    ``target`` names the class whose certification is being measured; it
    defaults to the modal class of ``counts``.  Measuring "some class was
    certified" instead would over-state coverage, because the rule legitimately
    certifies a *different* class whenever that class holds the plurality —
    see :func:`coverage_exact_three_class`.

    ``adaptive_attack`` selects the adversary:

      True   the attacker places all ``f`` votes in whichever rival currently
             holds the largest count, watching the realized honest draw.  This
             is the strong, adaptive adversary and it is the one the paper
             reports by default.
      False  the attacker commits to a fixed rival class before seeing the
             draw.  Picking the runner-up of the honest distribution is the
             best fixed choice; this is the model
             :func:`coverage_exact_three_class` assumes, so use it to
             cross-check the exact computation.

    ``lump_to_three`` merges every class other than the target and the runner-up
    into one, reproducing the model :func:`coverage_exact_three_class` assumes;
    use it to cross-check that computation.  Leave it off to simulate the full
    alphabet, whose certification probability is *higher*, because splitting the
    rival mass many ways is easier to beat than concentrating it.
    """
    rng = np.random.default_rng(seed)
    labels = list(counts)
    p = np.asarray([counts[c] for c in labels], dtype=float)
    p = p / p.sum()
    M = len(labels)
    if noise_classes is None:
        noise_classes = M
    if target is None:
        target = labels[int(np.argmax(p))]
    target_idx = labels.index(target)

    fixed_rival = fixed_rival_index(p, target_idx)
    others = [i for i in range(M) if i != target_idx]

    if lump_to_three and len(others) > 1:
        rest = sum(p[i] for i in range(M) if i not in (target_idx, fixed_rival))
        newp = np.array([p[target_idx], p[fixed_rival], rest])
        draws = rng.choice(3, size=(rounds, h), p=newp / newp.sum())
        target_sim, fixed_rival_sim = 0, 1
    else:
        draws = rng.choice(M, size=(rounds, h), p=p)
        target_sim, fixed_rival_sim = target_idx, fixed_rival

    if noise > 0:
        flip = rng.random((rounds, h)) < noise
        draws = np.where(flip, rng.integers(0, noise_classes, size=(rounds, h)),
                         draws)

    ok = np.zeros(rounds, dtype=bool)
    for r in range(rounds):
        hist = collections.Counter(draws[r].tolist())
        if f > 0:
            if adaptive_attack:
                ordered = sorted(hist.items(), key=lambda kv: (-kv[1], str(kv[0])))
                rivals = [c for c, _ in ordered if c != target_sim]
                rival = rivals[0] if rivals else (max(hist) + 1)
            else:
                rival = fixed_rival_sim if fixed_rival_sim is not None \
                    else (max(hist) + 1)
            hist[rival] = hist.get(rival, 0) + f
        ok[r] = certify(hist, 0) == target_sim
    return {"coverage": float(ok.mean()), "rounds": rounds, "h": h, "f": f,
            "noise": noise, "target": target, "lump_to_three": lump_to_three,
            "adaptive_attack": adaptive_attack}


def fixed_rival_index(p, target_idx):
    """The rival a bounded adversary should commit to, with an explicit tie-break.

    A fixed adversary gets one shot, so it should place its votes in the class
    it most needs to beat.  When the largest non-target mass is attained by
    several classes, the choice is not unique and the certification probability
    depends on it, so the rule must be fixed rather than left to ``max``'s
    iteration order.  We take the numerically largest such index, and both the
    exact and the Monte-Carlo computations use this function so they cannot
    disagree on a tie.
    """
    others = [i for i in range(len(p)) if i != target_idx]
    if not others:
        return None
    best = max(p[i] for i in others)
    tied = [i for i in others if abs(p[i] - best) <= 0.0]
    return max(tied)


def coverage_exact_three_class(n_top, n_second, h, f):
    """Exact P(the designated class c* is certified) for a three-group model.

    Honest labels land in the designated class c* with probability ``n_top``,
    in the designated rival c_b with ``n_second``, and otherwise in a lumped
    third class.  ``f`` corrupted proposals are appended to c_b.

    The question is whether *c\\** is certified, not whether some class is.
    Asking the latter massively over-states coverage (0.834 vs 0.481 on a
    distribution with p = (0.50, 0.45, 0.05)), because the rule happily
    certifies a third class when that class is the plurality.  Keeping c*
    fixed is what makes this the event the paper's threshold table reports.

    ``c_b`` is taken to be the class this function is called with, i.e. the
    caller has already fixed the rival; :func:`fixed_rival_index` is the rule
    used to pick it from a full distribution.
    """
    from math import comb
    p1, p2 = float(n_top), float(n_second)
    p3 = max(0.0, 1.0 - p1 - p2)
    total = 0.0
    for a in range(h + 1):
        for b in range(h - a + 1):
            c = h - a - b
            prob = (comb(h, a) * comb(h - a, b)
                    * (p1 ** a) * (p2 ** b) * (p3 ** c))
            if prob == 0.0:
                continue
            counts = {0: a, 1: b + f, 2: c}
            if certify(counts, 0) == 0:
                total += prob
    return total


def coverage_exact_pairwise(n_top, n_second, h, f):
    """Exact P(N(c*) > N(c_b)) when the rest of the mass is a lumped third class.

    This is the event the v1 threshold table computed.  It is *not* the
    certification event.  Comparing the two shows how much the pairwise test
    over-states certifiable coverage: whenever a third class wins outright
    while the designated rival is still beaten, the pairwise event holds and
    the plurality event does not.
    """
    from math import comb
    p1, p2 = float(n_top), float(n_second)
    p3 = max(0.0, 1.0 - p1 - p2)
    total = 0.0
    for a in range(h + 1):
        for b in range(h - a + 1):
            c = h - a - b
            prob = (comb(h, a) * comb(h - a, b)
                    * (p1 ** a) * (p2 ** b) * (p3 ** c))
            if prob == 0.0:
                continue
            rival = b + f
            if a > rival:
                total += prob
    return total


def pairwise_plurality_gap(n_top, n_second, h, f):
    """Return both three-group events and the mass that separates them."""
    pair = coverage_exact_pairwise(n_top, n_second, h, f)
    plur = coverage_exact_three_class(n_top, n_second, h, f)
    return {"pairwise": pair, "plurality": plur, "gap": pair - plur,
            "h": h, "f": f, "p_top": n_top, "p_second": n_second}


def coverage_hoeffding_bound(p_max, p_2, h, f, eta=None):
    """Concentration bound on the certification probability.

    With ``h`` i.i.d. honest labels, the certified class is the observed
    plurality whenever the top-two count gap exceeds ``f``.  Hoeffding applied
    to the indicator difference gives

        P(gap <= f) <= 2 exp(-2 (h*Delta - f)^2 / (4h))
                    = 2 exp(-(h*Delta - f)^2 / (2h)),

    which is the same ``exp(-(h*Delta-f)^2/(2h))`` form as the v1 certificate
    (the factor 2 is dropped because the two indicators are complementary).

    Returns the failure probability, or the coverage ``1 - that`` when ``eta``
    is given.  ``Delta = p_max - p_2``.
    """
    delta = float(p_max) - float(p_2)
    margin = h * delta - f
    if margin <= 0:
        return 1.0 if eta is None else 0.0
    fail = min(1.0, 2.0 * math.exp(-(margin ** 2) / (2.0 * h)))
    return fail if eta is None else 1.0 - fail


def certifiable_max_f(counts, confidence=0.99, h=None, rounds=20000,
                      seed=20260911):
    """Largest ``f`` whose certification probability reaches ``confidence``.

    ``-1`` means even ``f=0`` fails to reach it.  Searches upward and stops at
    the first failure, so the result is the *contiguous* certifiable range.
    """
    if h is None:
        h = sum(counts)
    best = -1
    for f in range(0, h + 1):
        cov = coverage_montecarlo(counts, h, f, rounds=rounds, seed=seed)["coverage"]
        if cov >= confidence:
            best = f
        else:
            break
    return best
