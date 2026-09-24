"""Collision-rate estimation and interval machinery.

Two estimators appear in this literature and the difference matters:

  plug-in   A^plug(h) = sum_c (n_c/k)^h
            treats a sample as if it could be drawn h times.  Biased *up*.

  U-stat.   A^U(h)    = sum_c [n_c]_h / [k]_h,  [x]_h = x(x-1)...(x-h+1)
            the probability that h samples drawn *without replacement* from
            the k stored ones all land in one class.  Unbiased for A(h) when
            the k records are an i.i.d. draw, and undefined for k < h.

The v1 paper used the U-statistic in its main table but the plug-in in the
heterogeneity analysis while still labelling the axis A^ (h).  v3 uses the
U-statistic everywhere and reports the plug-in only to quantify its bias.

Nothing here is specific to language models.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "falling_factorial", "A_unbiased", "A_plugin", "mean_A_unbiased",
    "mean_A_plugin", "task_bootstrap", "paired_bootstrap", "wilson",
]


def falling_factorial(x, h):
    if h < 0:
        raise ValueError("h must be non-negative")
    out = 1
    for i in range(h):
        out *= (x - i)
        if out == 0:
            return 0
    return out


def A_unbiased(counts, h):
    """sum_c [n_c]_h / [k]_h, or ``None`` when k < h."""
    k = sum(counts)
    if k < h:
        return None
    denom = falling_factorial(k, h)
    if denom == 0:
        return None
    return sum(falling_factorial(int(c), h) for c in counts) / denom


def A_plugin(counts, h):
    """sum_c (n_c/k)^h.  Defined for any k >= 1 but biased upward."""
    k = sum(counts)
    if k <= 0:
        return None
    return sum((c / k) ** h for c in counts)


def _mean(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def mean_A_unbiased(rows, field, h):
    """Mean over tasks of A^U(h); tasks with k < h are dropped.

    The caller is responsible for reporting how many tasks survive, because
    dropping them changes the task population across h and across models.
    """
    return _mean([A_unbiased(r[field], h) for r in rows])


def mean_A_plugin(rows, field, h):
    return _mean([A_plugin(r[field], h) for r in rows])


def mean_A_unbiased_with_n(rows, field, h):
    """As :func:`mean_A_unbiased` but also returns the number of tasks used."""
    vals = [A_unbiased(r[field], h) for r in rows]
    used = [v for v in vals if v is not None]
    if not used:
        return None, 0
    return sum(used) / len(used), len(used)


# --------------------------------------------------------------------------
# intervals
# --------------------------------------------------------------------------
def task_bootstrap(values, B=2000, seed=20260911, alpha=0.05):
    """Percentile bootstrap over tasks for a mean of per-task quantities."""
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return None
    if a.size == 1:
        return {"estimate": float(a[0]), "lo": float(a[0]), "hi": float(a[0]),
                "n": 1, "B": B}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(B, a.size))
    means = a[idx].mean(axis=1)
    return {
        "estimate": float(a.mean()),
        "lo": float(np.quantile(means, alpha / 2)),
        "hi": float(np.quantile(means, 1 - alpha / 2)),
        "n": int(a.size),
        "B": B,
    }


def paired_bootstrap(a_pairs, B=2000, seed=20260911, alpha=0.05):
    """Percentile bootstrap for the mean of a paired difference.

    Used wherever two conditions share the same tasks, so that the task-level
    variation cancels.  ``a_pairs`` is a sequence of ``(x, y)`` differences
    computed per task; unpaired means would inflate the interval.
    """
    d = np.asarray([x - y for x, y in a_pairs], dtype=float)
    if d.size == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(B, d.size))
    means = d[idx].mean(axis=1)
    return {
        "estimate": float(d.mean()),
        "lo": float(np.quantile(means, alpha / 2)),
        "hi": float(np.quantile(means, 1 - alpha / 2)),
        "n": int(d.size),
        "B": B,
        "frac_positive": float((means > 0).mean()),
    }


def wilson(successes, total, z=1.959963984540054):
    """Wilson score interval; used for rates that can sit near 0 or 1."""
    if total == 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return {"estimate": p, "lo": max(0.0, centre - half),
            "hi": min(1.0, centre + half), "n": total}
