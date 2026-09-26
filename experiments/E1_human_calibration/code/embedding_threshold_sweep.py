"""Cosine-threshold sweep for the embedding instrument on the human calibration pairs (no API calls).

Uses the stored cosine similarities (e1_calibration_instruments.json per_pair) and the resolved joint
human reference (annotation_scoring.json). Weighted false acceptance (share of human-DIFF weight judged
SAME) and false rejection (share of human-SAME weight judged DIFF) for every threshold; reports the
threshold 0.9 used in the paper, the equal-error point and the minimum of FA+FR.

    python experiments/E1_human_calibration/code/embedding_threshold_sweep.py
Writes E1_human_calibration/results/embedding_threshold_sweep.json.
"""
import json
from pathlib import Path
import numpy as np

RES = Path(__file__).resolve().parents[1] / "results"
S = {p["pair_id"]: p for p in json.load(open(RES / "annotation_scoring.json"))["per_pair"]}
I = {p["pair_id"]: p for p in json.load(open(RES / "e1_calibration_instruments.json"))["per_pair"]}
rows = [(I[k]["cosine"], S[k]["weight"], S[k]["joint"]) for k in S if S[k]["joint"] is not None]
cos = np.array([r[0] for r in rows]); w = np.array([r[1] for r in rows]); pos = np.array([r[2] for r in rows], bool)


def rates(t):
    same = cos >= t
    fa = (w * (same & ~pos)).sum() / (w * ~pos).sum()
    fr = (w * (~same & pos)).sum() / (w * pos).sum()
    return float(fa), float(fr)


grid = np.round(np.arange(0.50, 1.0001, 0.005), 3)
curve = [(float(t), *rates(t)) for t in grid]
fa9, fr9 = rates(0.9)
eer = min(curve, key=lambda x: abs(x[1] - x[2]))
best = min(curve, key=lambda x: x[1] + x[2])
out = {"n_resolved": int(len(rows)), "at_0.9": {"FA": fa9, "FR": fr9},
       "equal_error": {"threshold": eer[0], "FA": eer[1], "FR": eer[2]},
       "min_sum": {"threshold": best[0], "FA": best[1], "FR": best[2], "sum": best[1] + best[2]},
       "curve": [{"threshold": t, "FA": a, "FR": b} for t, a, b in curve]}
(RES / "embedding_threshold_sweep.json").write_text(json.dumps(out, indent=1))
print({k: v for k, v in out.items() if k != "curve"})
