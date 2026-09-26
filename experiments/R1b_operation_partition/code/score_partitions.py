"""Score the R1b partition annotations.

    python experiments/R1b_operation_partition/code/score_partitions.py [--selftest]

For every model-task cell (12 replies) this builds three partitions of the same replies:
answer grain (normalised answer), lexical grain (answer, normalised KEY) and the human
answer+decisive-operation grain (consensus of annotators A/B/C; two replies are linked when at least
two annotators put them in the same group, and classes are connected components; VAGUE replies
are singletons in the main analysis and dropped in the sensitivity analysis; the resulting
operation components are intersected with answer classes to match the paper's refinement).
Pure operation results are retained separately. Reported per grain,
as selected-cell-equal means with descriptive 95% cell-bootstrap intervals: modal mass r_max, U-statistic A(3), and the
plug-in n=3f+1 coverage r_max^h for h=3,5,7 (coverage uses the refinement r_max <= answer p_max).
Inter-annotator agreement: adjusted Rand index and pairwise co-assignment agreement.

--selftest fills the sheets synthetically (groups = answer class, then groups = normalised KEY) and
checks that the human-grain results equal the answer-grain and lexical-grain results respectively.
Writes results/r1b_partition_results.json (or results/r1b_selftest.json).
"""
import collections, csv, itertools, json, sys, io, hashlib
from math import comb
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "annotation"
KEY = json.load(open(ANN / "admin" / "key.json"))
H = (3, 5, 7)


def read(a):
    raw = (ANN / f"annotator_{a}.tsv").read_bytes()
    try:
        decoded = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        decoded = raw.decode("gb18030")
    return list(csv.DictReader(io.StringIO(decoded), delimiter="\t"))


def partition(labels):
    """labels: item -> group label or None (vague). Returns item -> class id."""
    out = {}
    for i, g in labels.items():
        out[i] = ("V", i) if g is None else ("G", g)
    return out


def consensus(parts, items, drop_vague):
    keep = [i for i in items if not (drop_vague and sum(p[i][0] == "V" for p in parts) >= 2)]
    parent = {i: i for i in keep}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in itertools.combinations(keep, 2):
        votes = sum(p[i] == p[j] and p[i][0] == "G" for p in parts)
        if votes >= 2:
            parent[find(i)] = find(j)
    return {i: find(i) for i in keep}


def stats(cls):
    c = collections.Counter(cls.values()); k = sum(c.values())
    pm = max(c.values()) / k
    a3 = sum(comb(v, 3) for v in c.values()) / comb(k, 3) if k >= 3 else float("nan")
    return {"k": k, "r_max": pm, "A3": a3, **{f"cov_h{h}": pm ** h for h in H}}


def ari(p, q, items):
    n = len(items)
    cont = collections.Counter((p[i], q[i]) for i in items)
    a = collections.Counter(p[i] for i in items); b = collections.Counter(q[i] for i in items)
    s = sum(comb(v, 2) for v in cont.values()); sa = sum(comb(v, 2) for v in a.values()); sb = sum(comb(v, 2) for v in b.values())
    exp = sa * sb / comb(n, 2); mx = (sa + sb) / 2
    return 1.0 if mx == exp else (s - exp) / (mx - exp)


def run(sheets):
    by_task = collections.defaultdict(dict)
    for a, rows in sheets.items():
        for r in rows:
            g = r["group"].strip() or None
            if r["vague"].strip().upper() == "Y":
                g = None
            by_task[r["task"]].setdefault(a, {})[r["item"]] = g
    cells, agree = {}, collections.defaultdict(list)
    for tid, c in KEY["cells"].items():
        items = [x["item"] for x in c["items"]]
        ans = {x["item"]: x["answer_class"] for x in c["items"]}
        lex = {x["item"]: (x["answer_class"], x["key_norm"]) for x in c["items"]}
        parts = [partition(by_task[tid][a]) for a in sorted(by_task[tid])]
        for (x, p), (y, q) in itertools.combinations(zip(sorted(by_task[tid]), parts), 2):
            agree["ari"].append(ari(p, q, items))
            agree["pair_agreement"].append(np.mean([(p[i] == p[j]) == (q[i] == q[j]) for i, j in itertools.combinations(items, 2)]))
        op = consensus(parts, items, False)
        op_drop = consensus(parts, items, True)
        joint = {i: (ans[i], op[i]) for i in op}
        joint_drop = {i: (ans[i], op_drop[i]) for i in op_drop}
        pairs = list(itertools.combinations(items, 2))
        bridges = [(i, j) for i, j in pairs if op[i] == op[j] and
                   sum(p[i] == p[j] and p[i][0] == "G" for p in parts) < 2]
        cross_answer = [(i, j) for i, j in pairs if op[i] == op[j] and ans[i] != ans[j]]
        cells[tid] = {"model": c["model"], "task_id": c["task_id"], "bench": c["bench"],
                      "answer": stats(ans), "lexical": stats(lex),
                      "human": stats(joint), "human_drop_vague": stats(joint_drop),
                      "operation_only": stats(op), "operation_only_drop_vague": stats(op_drop),
                      "closure_added_pairs": bridges, "operation_cross_answer_pairs": cross_answer,
                      "per_annotator": {a: stats({i: (ans[i], p[i]) for i in items})
                                        for a, p in zip(sorted(by_task[tid]), parts)},
                      "n_vague_majority": sum(sum(p[i][0] == "V" for p in parts) >= 2 for i in items)}
    rng = np.random.default_rng(20260925)
    tids = list(cells)
    boot = rng.integers(len(tids), size=(2000, len(tids)))
    summary = {}
    for grain in ("answer", "human", "human_drop_vague", "lexical", "operation_only", "operation_only_drop_vague"):
        summary[grain] = {}
        for m in ("r_max", "A3", *[f"cov_h{h}" for h in H]):
            v = np.array([cells[t][grain][m] for t in tids])
            summary[grain][m] = {"mean": float(np.nanmean(v)),
                                 "ci95": [float(x) for x in np.nanquantile(np.nanmean(v[boot], axis=1), [0.025, 0.975])]}
    return {"n_cells": len(cells), "annotators": sorted({a for t in by_task.values() for a in t}),
            "agreement": {k: float(np.mean(v)) for k, v in agree.items()},
            "summary": summary, "cells": cells}


def synthetic(field):
    out = {}
    for a in "ABC":
        rows = read(a)
        for r in rows:
            x = next(x for x in KEY["cells"][r["task"]]["items"] if x["item"] == r["item"])
            r["vague"] = ""
            r["group"] = x["answer_class"] if field == "answer" else repr((x["answer_class"], x["key_norm"]))
        out[a] = rows
    return out


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        res = {}
        for field, target in (("answer", "answer"), ("lexical", "lexical")):
            r = run(synthetic(field))
            ok = all(abs(r["summary"]["human"][m]["mean"] - r["summary"][target][m]["mean"]) < 1e-12 for m in r["summary"]["human"])
            res[field] = {"matches_" + target: ok, "ari": r["agreement"]["ari"], "summary": r["summary"]}
            print(f"selftest groups={field}: human == {target}: {ok}")
        (ROOT / "results" / "r1b_selftest.json").write_text(json.dumps(res, indent=1))
    else:
        sheets = {a: read(a) for a in "ABC"}
        if not all(r["group"].strip() or r["vague"].strip() for s in sheets.values() for r in s):
            sys.exit("sheets are not complete: every row needs a group or vague=Y")
        expected = {(t, x["item"]): x for t, c in KEY["cells"].items() for x in c["items"]}
        for a, rows in sheets.items():
            ids = [(r["task"], r["item"]) for r in rows]
            if len(ids) != len(set(ids)) or set(ids) != set(expected):
                sys.exit(f"{a}: missing, duplicated, or unexpected items")
            for r in rows:
                if r["vague"].strip().upper() not in ("", "Y") or (not r["group"].strip() and r["vague"].strip().upper() != "Y"):
                    sys.exit(f"{a}: invalid annotation at {r['item']}")
        for rows in sheets.values():
            for r, ref in zip(rows, sheets["A"]):
                if any(r[k] != ref[k] for k in ("task", "item", "answer", "key")):
                    sys.exit("immutable fields differ between sheets")
        res = run(sheets)
        assert all(c["human"]["r_max"] <= c["answer"]["r_max"] for c in res["cells"].values())
        res["provenance"] = {a: hashlib.sha256((ANN / f"annotator_{a}.tsv").read_bytes()).hexdigest() for a in sheets}
        res["definition"] = "human = (normalised answer, majority-connected operation component); operation_only retains the original operation grouping"
        (ROOT / "results" / "r1b_partition_results.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({k: v for k, v in res.items() if k != "cells"}, indent=1))
