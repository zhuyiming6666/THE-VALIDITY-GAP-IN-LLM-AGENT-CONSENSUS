"""Offline certification estimates for a fixed total population size.

Read the same empirical per-task distributions as threshold_analysis.py, but
evaluate the reference certificate with its actual fault budget.  This script
does not call a model, alter the source data, or regenerate existing artifacts.

With honest histogram H and at most f appended Byzantine labels, the empirical
population-mode target c is certified for every append allocation exactly when
H[c] - max(H[d] for d != c) > 2*f.  A zero-probability rival is retained so that
an attacker may introduce an unobserved label, including for singleton support.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENTS / "baseline/code"))
from certify import certify


ROOT = Path(__file__).resolve().parents[1]
LEVELS = {"verdict": "counts_verdict", "answer_key_lexical": "counts_semantic"}


def seed_for(base_seed: int, *parts: object) -> int:
    encoded = json.dumps([base_seed, *parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def compositions(total: int, width: int):
    if width == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for tail in compositions(total - first, width - 1):
            yield (first, *tail)


def check_events() -> dict:
    """Check substantive counterexamples and enumerate adversarial allocations."""
    assert certify([6, 5], 1) is None  # H=(6,4), one rival append.
    assert certify([6, 5], 0) == 0  # Ordinary plurality is insufficient.
    assert certify([7, 4], 1) == 0
    assert certify([7, 3], 3) == 0  # Singleton honest support, n=10, f=3.
    assert certify([6, 3], 3) is None
    assert certify([4, 6], 0) == 1  # Any-class coverage differs from target recovery.
    checked = 0
    for h in range(1, 8):
        for honest in compositions(h, 3):
            for f in range(4):
                for target in range(3):
                    gap_event = honest[target] - max(
                        honest[d] for d in range(3) if d != target
                    ) > 2 * f
                    all_certified = all(
                        certify([a + b for a, b in zip(honest, attack)], f)
                        == target
                        for b in range(f + 1)
                        for attack in compositions(b, 3)
                    )
                    assert gap_event == all_certified, (honest, f, target)
                    checked += 1
    return {"passed": True, "exhaustive_histogram_budget_target_cases": checked,
            "counterexamples_checked": 6}


def summary(values: np.ndarray, bootstrap_indices: np.ndarray,
            successes: np.ndarray | None = None, draws: int | None = None) -> dict:
    bootstrap_means = values[bootstrap_indices].mean(axis=1)
    out = {
        "estimate": float(values.mean()),
        "task_bootstrap_ci95": [float(x) for x in np.quantile(
            bootstrap_means, [0.025, 0.975])],
        "n_tasks": int(values.size),
    }
    if successes is not None and draws is not None:
        # Conditional simulation error only; this is not distribution-estimation
        # uncertainty and is distinct from the task-bootstrap interval.
        out["estimated_mc_standard_error_of_task_mean"] = float(
            np.sqrt(np.sum(values * (1 - values) / draws)) / values.size)
        out["successes_over_all_task_draws"] = int(successes.sum())
        out["total_task_draws"] = int(draws * values.size)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=30000)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--source", type=Path,
                        default=EXPERIMENTS / "baseline/results/e1_analysis.json")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results/submission_certification_check.json")
    args = parser.parse_args()
    if args.draws < 20000 or args.bootstrap < 2000:
        parser.error("Use at least 20000 draws and 2000 bootstrap resamples.")

    checks = check_events()
    source_bytes = args.source.read_bytes()
    data = json.loads(source_bytes)
    task_ids = sorted(set(data["common_tasks"]["ids"]))
    models = sorted(data["per_task"])
    bootstrap_seed = seed_for(args.seed, "task_bootstrap")
    bootstrap_indices = np.random.default_rng(bootstrap_seed).integers(
        0, len(task_ids), size=(args.bootstrap, len(task_ids)))
    report = {
        "config": {
            "source": str(args.source.relative_to(EXPERIMENTS)),
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "fixed_total_n": 10,
            "fault_budgets": [0, 1, 2, 3],
            "honest_population": "h = n - f",
            "draws_per_task_model_granularity_budget": args.draws,
            "base_seed": args.seed,
            "rng": "numpy.random.default_rng / PCG64; multinomial(h, p)",
            "numpy_version": np.__version__,
            "seed_derivation": "SHA256(JSON([base_seed,model,level,task_id,f])) first 8 bytes, big endian",
            "task_bootstrap_resamples": args.bootstrap,
            "task_bootstrap_seed": bootstrap_seed,
            "task_bootstrap_interval": "percentile 95%; same task resamples across all conditions",
            "task_set": task_ids,
            "n_tasks": len(task_ids),
            "estimation_model": "Conditional on empirical per-task class frequencies; honest labels are iid draws from those frequencies.",
            "target": "One fixed empirical modal class, chosen as the first entry after stable descending count sort; tied empirical modes are retained and counted.",
            "target_event": "H[c*] - max_{d != c*} H[d] > 2*f; equivalent to certify(H+B,f)=c* for every nonnegative append histogram B with total <= f.",
            "any_class_event_f0": "largest honest count > second-largest honest count; certificate outputs some class, possibly other than c*.",
            "unobserved_labels": "One zero-probability rival appended to every empirical support; adversaries may assign votes to it.",
            "uncertainty_scope": "Task-bootstrap intervals quantify between-task variation conditional on estimated distributions. They do not account for finite per-task proposal-sampling uncertainty, modal selection uncertainty, or semantic-partition uncertainty. Simulation standard error is reported separately.",
            "cross_model_aggregation": "Equal model weights within each task, followed by equal task weights and task bootstrap.",
        },
        "checks": checks,
        "results": {},
        "aggregate_across_models": {},
        "per_task": {},
    }
    for model in models:
        rows = {r["id"]: r for r in data["per_task"][model]}
        report["results"][model] = {}
        report["per_task"][model] = {}
        for level, field in LEVELS.items():
            records = []
            for task_id in task_ids:
                counts = sorted(rows[task_id][field], reverse=True)
                assert counts and min(counts) > 0
                p = np.array([*counts, 0], dtype=float)
                p /= p.sum()
                rec = {"task_id": task_id, "n_observed_labels": int(sum(counts)),
                       "observed_support_size": len(counts),
                       "n_tied_empirical_modes": counts.count(counts[0]),
                       "budgets": {}}
                for f in range(4):
                    draw_seed = seed_for(args.seed, model, level, task_id, f)
                    hist = np.random.default_rng(draw_seed).multinomial(
                        10 - f, p, size=args.draws)
                    target_ok = hist[:, 0] - hist[:, 1:].max(axis=1) > 2 * f
                    cell = {"h": 10 - f, "seed": draw_seed,
                            "target_successes": int(target_ok.sum()),
                            "target_certification_probability": float(target_ok.mean())}
                    if f == 0:
                        top_two = np.partition(hist, -2, axis=1)[:, -2:]
                        any_ok = top_two[:, 1] > top_two[:, 0]
                        cell["any_class_successes"] = int(any_ok.sum())
                        cell["any_class_certification_probability"] = float(any_ok.mean())
                        assert np.all(~target_ok | any_ok)
                    rec["budgets"][str(f)] = cell
                records.append(rec)
            report["per_task"][model][level] = records
            level_summary = {
                "n_tasks_with_tied_empirical_modes": sum(
                    r["n_tied_empirical_modes"] > 1 for r in records),
                "n_tasks_with_singleton_support": sum(
                    r["observed_support_size"] == 1 for r in records),
                "target_certification": {},
            }
            for f in range(4):
                values = np.array([r["budgets"][str(f)][
                    "target_certification_probability"] for r in records])
                successes = np.array([r["budgets"][str(f)][
                    "target_successes"] for r in records])
                level_summary["target_certification"][str(f)] = summary(
                    values, bootstrap_indices, successes, args.draws)
            values = np.array([r["budgets"]["0"][
                "any_class_certification_probability"] for r in records])
            successes = np.array([r["budgets"]["0"]["any_class_successes"] for r in records])
            level_summary["any_class_certification_f0"] = summary(
                values, bootstrap_indices, successes, args.draws)
            report["results"][model][level] = level_summary

    for level in LEVELS:
        aggregate = {"target_certification": {}}
        for f in range(4):
            values = np.array([
                [r["budgets"][str(f)]["target_certification_probability"]
                 for r in report["per_task"][model][level]] for model in models])
            aggregate["target_certification"][str(f)] = summary(
                values.mean(axis=0), bootstrap_indices)
        values = np.array([
            [r["budgets"]["0"]["any_class_certification_probability"]
             for r in report["per_task"][model][level]] for model in models])
        aggregate["any_class_certification_f0"] = summary(
            values.mean(axis=0), bootstrap_indices)
        report["aggregate_across_models"][level] = aggregate

    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"checks": checks, "results": report["results"],
                      "aggregate_across_models": report["aggregate_across_models"]}, indent=2))


if __name__ == "__main__":
    main()
