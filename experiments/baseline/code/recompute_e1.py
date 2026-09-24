"""Build the v3 record set: corrected scoring, common task set, funnel.

Outputs (written to ``../results``):

  e1_per_task.json   per model, per task, per granularity: class counts,
                     p_max, p_2, support, accuracy, n_valid, n_missing
  e1_summary.json    headline table + per-benchmark breakdown + bootstrap CIs
  funnel.json        attempted / returned / parsed / valid per model
  common_tasks.json  the retained task set and why

Two decisions the v1 pipeline made implicitly and v3 makes explicit:

  * A reply whose ANSWER cannot be parsed is *missing*, not its own class.
    Counting parse failures as one shared class manufactures agreement.
  * A reply whose KEY is missing still contributes a verdict observation but
    its answer+KEY class is ``(answer, None)``, and the number of such records
    is reported so the reader can judge the effect.
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collision import (A_plugin, A_unbiased, mean_A_plugin,  # noqa: E402
                       mean_A_unbiased_with_n, task_bootstrap)
from scoring import answer_class, semantic_class, parse_schema, correct  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
PACKAGE = HERE.parent
DATA = PACKAGE / "data"
RESULTS = PACKAGE / "results"

MODELS = ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1-mini"]
SOURCE_FILES = ["samples.jsonl", "samples2.jsonl"]
MIN_PER_TASK = 10          # records required on every model to retain a task
HS = (2, 3, 5, 7, 10, 20)


def load_tasks():
    tasks = {}
    with open(DATA / "tasks.jsonl") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                tasks[rec["id"]] = rec
    return tasks


def load_records():
    """Read the stored replies and attach the v3 classes.

    Returns ``by[model][task_id] = [record, ...]`` plus funnel counters.
    """
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    # Every counter is created for every model up front, so a missing key
    # means "zero" rather than "this model was never seen".
    funnel = {m: collections.Counter({"returned": 0, "unparsed_answer": 0,
                                      "missing_key": 0}) for m in MODELS}
    tasks = load_tasks()

    for rel in SOURCE_FILES:
        path = DATA / rel
        if not path.exists():
            continue
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                model = rec.get("model")
                if model not in funnel:
                    continue
                tid = rec.get("task_id")
                if tid not in tasks:
                    continue
                bench = tasks[tid]["bench"]
                funnel[model]["returned"] += 1

                a_raw, k_raw = rec.get("answer"), rec.get("key")
                if a_raw is None or k_raw is None:
                    # Older records stored the fields separately; the schema
                    # text was not always kept, so re-parse when possible.
                    text = rec.get("raw")
                    if text:
                        pa, pk = parse_schema(text)
                        a_raw = a_raw if a_raw is not None else pa
                        k_raw = k_raw if k_raw is not None else pk

                a_cls = answer_class(a_raw, bench)
                s_cls = semantic_class(a_raw, k_raw, bench)
                if a_cls is None:
                    funnel[model]["unparsed_answer"] += 1
                if s_cls is not None and s_cls[1] is None:
                    funnel[model]["missing_key"] += 1

                by[model][tid].append({
                    "answer_class": a_cls,
                    "semantic_class": s_cls,
                    "correct": bool(correct(a_raw, tasks[tid]["answer"], bench)),
                    "has_answer": a_cls is not None,
                    "has_key": s_cls is not None and s_cls[1] is not None,
                })
    return by, funnel, tasks


def _counts(records, field, keep_missing):
    c = collections.Counter()
    for r in records:
        v = r[field]
        if v is None and not keep_missing:
            continue
        c[v] += 1
    return sorted(c.values(), reverse=True)


def per_task_rows(by, tasks):
    rows = {}
    for model in MODELS:
        rows[model] = []
        for tid in sorted(by[model]):
            recs = by[model][tid]
            va = [r for r in recs if r["has_answer"]]
            if not va:
                continue
            ca = _counts(va, "answer_class", keep_missing=False)
            cs = _counts(va, "semantic_class", keep_missing=False)
            gold_n = len(va)
            rows[model].append({
                "id": tid,
                "bench": tasks[tid]["bench"],
                "n_returned": len(recs),
                "n_valid": gold_n,
                "n_missing_key": sum(1 for r in va if not r["has_key"]),
                "counts_verdict": ca,
                "counts_semantic": cs,
                "p_max_verdict": ca[0] / gold_n,
                "p_max_semantic": cs[0] / gold_n,
                "p2_verdict": (ca[1] / gold_n) if len(ca) > 1 else 0.0,
                "p2_semantic": (cs[1] / gold_n) if len(cs) > 1 else 0.0,
                "support_verdict": len(ca),
                "support_semantic": len(cs),
                "acc": sum(1 for r in va if r["correct"]) / gold_n,
            })
    return rows


def _level_summary(rows, field, h_list=HS):
    out = {}
    n = len(rows)
    if n == 0:
        return out
    out["n_tasks"] = n
    pk = "p_max_verdict" if field == "counts_verdict" else "p_max_semantic"
    p2 = "p2_verdict" if field == "counts_verdict" else "p2_semantic"
    sp = "support_verdict" if field == "counts_verdict" else "support_semantic"
    lvl = "verdict" if field == "counts_verdict" else "semantic"
    out[f"p_max_{lvl}"] = sum(r[pk] for r in rows) / n
    out[f"p_2_{lvl}"] = sum(r[p2] for r in rows) / n
    out[f"support_{lvl}"] = sum(r[sp] for r in rows) / n
    for h in h_list:
        mu, used = mean_A_unbiased_with_n(rows, field, h)
        out[f"A_{lvl}_{h}"] = round(mu, 6) if mu is not None else None
        out[f"A_{lvl}_{h}_n"] = used
        mp = mean_A_plugin(rows, field, h)
        out[f"Aplugin_{lvl}_{h}"] = round(mp, 6) if mp is not None else None
    return out


def summarise(rows):
    out = {}
    for model, rs in rows.items():
        s = {"n_tasks": len(rs),
             "acc": sum(r["acc"] for r in rs) / len(rs) if rs else None,
             "mean_n_valid": sum(r["n_valid"] for r in rs) / len(rs) if rs else None,
             "mean_missing_key": sum(r["n_missing_key"] for r in rs) / len(rs) if rs else None}
        s.update(_level_summary(rs, "counts_verdict"))
        s.update(_level_summary(rs, "counts_semantic"))
        out[model] = {k: (round(v, 6) if isinstance(v, float) else v)
                      for k, v in s.items()}
    return out


def per_benchmark(rows):
    out = {}
    for model, rs in rows.items():
        out[model] = {}
        for bench in ("gsm8k", "mmlu", "mbpp"):
            sub = [r for r in rs if r["bench"] == bench]
            if not sub:
                continue
            b = {"n_tasks": len(sub),
                 "acc": sum(r["acc"] for r in sub) / len(sub)}
            b.update(_level_summary(sub, "counts_verdict"))
            b.update(_level_summary(sub, "counts_semantic"))
            out[model][bench] = {k: (round(v, 6) if isinstance(v, float) else v)
                                 for k, v in b.items()}
    return out


def common_task_set(rows, min_per_task=MIN_PER_TASK):
    """Tasks on which every model has >= ``min_per_task`` valid records.

    Every headline number is computed on this set, so no table silently mixes
    different task populations (the v1 defect the review flagged in 5.3).
    """
    per_model = {m: {r["id"] for r in rs if r["n_valid"] >= min_per_task}
                 for m, rs in rows.items()}
    common = set.intersection(*per_model.values()) if per_model else set()
    return sorted(common), per_model


def bootstrap_block(rows, common):
    """Task-level bootstrap CIs on the retained set for the headline numbers."""
    out = {}
    for model, rs in rows.items():
        sub = [r for r in rs if r["id"] in set(common)]
        if not sub:
            continue
        out[model] = {}
        for lvl, field in (("verdict", "counts_verdict"),
                           ("semantic", "counts_semantic")):
            for h in (3, 10):
                vals = [A_unbiased(r[field], h) for r in sub]
                ci = task_bootstrap([v for v in vals if v is not None])
                if ci:
                    out[model][f"A_{lvl}_{h}"] = {k: round(v, 6) if isinstance(v, float) else v
                                                  for k, v in ci.items()}
        ci = task_bootstrap([r["p_max_verdict"] for r in sub])
        if ci:
            out[model]["p_max_verdict"] = {k: round(v, 6) if isinstance(v, float) else v
                                           for k, v in ci.items()}
        ci = task_bootstrap([r["p_max_semantic"] for r in sub])
        if ci:
            out[model]["p_max_semantic"] = {k: round(v, 6) if isinstance(v, float) else v
                                            for k, v in ci.items()}
        ci = task_bootstrap([r["acc"] for r in sub])
        if ci:
            out[model]["acc"] = {k: round(v, 6) if isinstance(v, float) else v
                                 for k, v in ci.items()}
    return out


def main():
    RESULTS.mkdir(exist_ok=True)
    by, funnel, tasks = load_records()

    # A reply is counted as attempted-complete only when the file holds it;
    # the plan was 150 tasks x 30 samples per model.
    for m in MODELS:
        funnel[m]["planned"] = 150 * 30

    rows = per_task_rows(by, tasks)
    common, per_model = common_task_set(rows)

    # Restrict every headline table to the common set.
    retained = {m: [r for r in rs if r["id"] in set(common)] for m, rs in rows.items()}

    payload = {
        "config": {
            "source_files": SOURCE_FILES,
            "models": MODELS,
            "min_valid_per_task": MIN_PER_TASK,
            "h_values": list(HS),
            "scoring": "benchmark-specific; see scoring.py and tests/test_scoring.py",
        },
        "funnel": {m: dict(funnel[m]) for m in MODELS},
        "common_tasks": {
            "n": len(common),
            "ids": common,
            "per_bench": dict(collections.Counter(tasks[t]["bench"] for t in common)),
            "per_model_available": {m: len(v) for m, v in per_model.items()},
        },
        "summary_all_tasks": summarise(rows),
        "summary_common_tasks": summarise(retained),
        "per_bench_common": per_benchmark(retained),
        "bootstrap_common": bootstrap_block(rows, common),
        "per_task": {m: rows[m] for m in MODELS},
    }

    (RESULTS / "e1_analysis.json").write_text(json.dumps(payload, indent=1))

    print(f"common task set: {len(common)} tasks "
          f"({payload['common_tasks']['per_bench']})")
    print("\n--- funnel (planned 4500 per model) ---")
    for m in MODELS:
        f = funnel[m]
        print(f"  {m:<14} returned={f['returned']:>5}  unparsed_answer={f['unparsed_answer']:>4}"
              f"  missing_key={f['missing_key']:>4}")
    print("\n--- headline, common task set ---")
    hdr = f"{'model':<15}{'lvl':<9}{'p_max':>7}{'p_2':>7}{'#cls':>7}{'acc':>7}"
    for h in (3, 10):
        hdr += f"{'A(' + str(h) + ')':>10}"
    print(hdr)
    for m in MODELS:
        s = payload["summary_common_tasks"][m]
        for lvl in ("verdict", "semantic"):
            print(f"  {m:<15}{lvl:<9}{s[f'p_max_{lvl}']:>7.3f}{s[f'p_2_{lvl}']:>7.3f}"
                  f"{s[f'support_{lvl}']:>7.2f}{s['acc']:>7.3f}"
                  f"{s[f'A_{lvl}_3']:>10.5f}{s[f'A_{lvl}_10']:>10.5f}")
    print(f"\nwrote {RESULTS / 'e1_analysis.json'}")


if __name__ == "__main__":
    main()
