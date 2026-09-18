"""Offline v4 corrections. Read archived replies; never call models or execute candidates.

The v3 manuscript, scripts and historical artifacts are deliberately left unchanged.
We reuse its tested scorer/record loader/certificate, not its legacy heterogeneity
estimator, archived instrument labels, or threshold_analysis.py.
"""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
import random
import sys

import numpy as np

V4 = Path(__file__).resolve().parents[1]
ROOT = V4.parent
sys.path.insert(0, str(ROOT / 'v3/code'))
from scoring import answer_class, semantic_class, correct
from recompute_e1 import load_records, per_task_rows, common_task_set, summarise, MODELS
from collision import A_unbiased
from certify import certify, certify_with_errors, compatible_histograms
from submission_certification_check import compositions, check_events
from analyse_experiments import analyse_x2

SOURCES = {}


def track(rel):
    path = ROOT / rel
    SOURCES[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read(rel):
    return json.loads(track(rel).read_text())


def lines(rel):
    return [json.loads(s) for s in track(rel).read_text().splitlines() if s.strip()]


def falling(n, h):
    if not isinstance(n, int) or n < 0:
        raise ValueError('U-statistics require nonnegative integer counts, not pseudocounts')
    return math.prod(range(n - h + 1, n + 1)) if n >= h else 0


def allocated_collision(counts, allocation):
    """Independent populations, without replacement within each population.

    A genuine zero collision count is 0.0, never missing. None means k < h.
    """
    sizes = [sum(c.values()) for c in counts]
    if any(k < h for k, h in zip(sizes, allocation)):
        return None
    labels = set().union(*counts)
    return sum(math.prod(falling(c.get(label, 0), h) / falling(k, h)
                         for c, k, h in zip(counts, sizes, allocation))
               for label in labels)


def averaged_law_u(counts, h):
    """Unbiased estimator of sum_c [(p_1(c)+...+p_M(c))/M]^h.

    Expand over multinomial allocations; do not feed fractional pseudocounts
    to an integer-count falling-factorial estimator.
    """
    if any(sum(c.values()) < h for c in counts):
        return None
    m = len(counts)
    return sum(math.factorial(h) / math.prod(math.factorial(a) for a in alloc)
               / m ** h * allocated_collision(counts, alloc)
               for alloc in compositions(h, m))


def summary(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(a):
        return {'estimate': None, 'n_tasks': 0, 'ci95': None}
    ix = np.random.default_rng(20260915).integers(len(a), size=(5000, len(a)))
    return {'estimate': float(a.mean()), 'n_tasks': len(a),
            'ci95': np.quantile(a[ix].mean(axis=1), [.025, .975]).tolist()}


def heterogeneity(by, ids):
    result = {}
    for field in ['answer_class', 'semantic_class']:
        result[field] = {}
        for h in [3, 6, 9]:
            rows = []
            for tid in ids:
                cs = [Counter(r[field] for r in by[m][tid] if r['has_answer']) for m in MODELS]
                singles = {m: A_unbiased(list(c.values()), h) for m, c in zip(MODELS, cs)}
                labels = set().union(*cs)
                rows.append({'task': tid, 'balanced': allocated_collision(cs, (h//3,)*3),
                             'averaged_law_U': averaged_law_u(cs, h),
                             'averaged_law_plugin': sum((sum(c.get(v, 0)/sum(c.values()) for c in cs)/3)**h for v in labels),
                             'per_task_max_U_oracle': max(singles.values()), **singles})
            means = {k: summary([r[k] for r in rows]) for k in rows[0] if k != 'task'}
            fixed = max(MODELS, key=lambda m: means[m]['estimate'])
            result[field][str(h)] = {'per_task': rows, 'summary': means,
                'fixed_highest_mean_model': fixed,
                'balanced_zero_tasks': [r['task'] for r in rows if r['balanced'] == 0],
                'reduction_ratio_of_means_vs_fixed': 1-means['balanced']['estimate']/means[fixed]['estimate'],
                'reduction_ratio_of_means_vs_oracle': 1-means['balanced']['estimate']/means['per_task_max_U_oracle']['estimate']}
    return result


def instruments(tasks):
    archived = read('results/instrument2.json')['results']['_pairs']
    grouped = defaultdict(list)
    for r in lines('results/samples.jsonl'):
        grouped[r['task_id']].append(r)
    tids = [t for t, rows in grouped.items() if len(rows) >= 2]
    rng = random.Random(0)
    rows, changes = [], Counter()
    for p in archived:
        # Reconstruct the original random frame, including duplicate pairs.
        tid = rng.choice(tids)
        a, b = rng.sample(grouped[tid], 2)
        assert tid == p['task'] and [a['answer'], a['key']] == p['a'] and [b['answer'], b['key']] == p['b']
        assert a['model'] == b['model'] == 'gpt-4.1-nano'
        bench = tasks[tid]['bench']
        ac, bc = answer_class(p['a'][0], bench), answer_class(p['b'][0], bench)
        assert ac is not None and bc is not None
        same = ac == bc
        lexical = semantic_class(*p['a'], bench) == semantic_class(*p['b'], bench)
        changes['answer_label_changed'] += same != p['same_answer']
        changes['lexical_label_changed'] += lexical != p['lexical']
        rows.append({'task': tid, 'same_answer': same, 'lexical': lexical,
                     'embedding': bool(p['embedding']), 'judge_nano': bool(p['judge']),
                     'judge_mini': bool(p['ref'])})
    task_ids = sorted({r['task'] for r in rows})
    ix = np.random.default_rng(20260915).integers(len(task_ids), size=(5000, len(task_ids)))
    rates = {}
    for instrument in ['lexical', 'embedding', 'judge_nano', 'judge_mini']:
        rates[instrument] = {}
        for condition in ['all', 'same_answer', 'different_answer']:
            keep = lambda r: condition == 'all' or r['same_answer'] == (condition == 'same_answer')
            cells = np.array([[sum(r[instrument] for r in rows if r['task'] == t and keep(r)),
                               sum(1 for r in rows if r['task'] == t and keep(r))] for t in task_ids])
            totals = cells.sum(axis=0)
            draws = cells[ix].sum(axis=1)
            boot = draws[draws[:, 1] > 0]
            rates[instrument][condition] = {'same': int(totals[0]), 'n_pairs': int(totals[1]),
                'estimate': float(totals[0]/totals[1]),
                'ci95': np.quantile(boot[:, 0]/boot[:, 1], [.025, .975]).tolist()}
    return {'n_pairs': len(rows), 'n_tasks': len(task_ids), 'model': 'gpt-4.1-nano',
            'changes': dict(changes), 'rates': rates, 'per_pair': rows,
            'interval': '5000 task-cluster percentile bootstrap resamples; seed 20260915; pair-weighted ratios; archived frame held fixed',
            'reference': 'Current scorer final-answer equality, NOT human semantic truth',
            'archived_answer_counts': dict(Counter(str(p['same_answer']) for p in archived))}


def attacks(tasks):
    data = lines('v3/results/x3_attacks.samples.jsonl')
    assert Counter(r['strategy'] for r in data) == {'none': 144, 'forced': 144}
    rows = []
    for strategy in ['none', 'forced']:
        for f in [1, 2, 3]:
            trials = [r for r in data if r['strategy'] == strategy and r['f'] == f]
            all_agree, honest_agree, acc, target_hits, attack_slots = [], [], [], 0, 0
            for r in trials:
                task = tasks[r['task_id']]
                labels = [answer_class(s['answer'], task['bench']) for s in r['history'][2]]
                honest = labels[f:] if strategy == 'forced' else labels
                unanimity = lambda a: None not in a and len(set(a)) == 1
                all_agree.append(unanimity(labels)); honest_agree.append(unanimity(honest))
                majority = Counter(c for c in honest if c is not None).most_common(1)[0][0]
                acc.append(majority == answer_class(task['answer'], task['bench']))
                if strategy == 'forced':
                    target = answer_class(r['history'][0][0]['answer'], task['bench'])
                    for rnd in r['history']:
                        for s in rnd[:f]:
                            assert s['finish_reason'] == 'forced'
                            attack_slots += 1
                            target_hits += answer_class(s['answer'], task['bench']) == target
            rows.append({'strategy': strategy, 'nominal_f': f, 'actual_b': f if strategy == 'forced' else 0,
                         'n_trials': len(trials), 'round': 2, 'all_unanimity': float(np.mean(all_agree)),
                         'honest_unanimity': float(np.mean(honest_agree)), 'honest_majority_accuracy': float(np.mean(acc)),
                         'all_rounds_target_compliance': target_hits/attack_slots if attack_slots else None,
                         'all_rounds_attacker_slots': attack_slots})
    return {'n_trials': len(data), 'n_tasks': len({r['task_id'] for r in data}), 'rows': rows,
            'prompted_arm': 'absent; separate nano prompted run is not a matched comparison'}


def tests():
    result = check_events()
    assert allocated_collision([Counter(a=3), Counter(b=3)], (2, 2)) == 0
    assert allocated_collision([Counter(a=1)], (2,)) is None
    try:
        falling(2.5, 2)
        raise AssertionError('fractional counts accepted')
    except ValueError:
        pass
    # Independently enumerate all data sets for two binary populations, k=h=2.
    expected = 0.0
    for draws in itertools.product([0, 1], repeat=4):
        weight = math.prod((.2 if v == 0 else .8) for v in draws[:2]) * math.prod((.7 if v == 0 else .3) for v in draws[2:])
        expected += weight * averaged_law_u([Counter(draws[:2]), Counter(draws[2:])], 2)
    assert abs(expected - (.45**2 + .55**2)) < 1e-12
    cv_cases = 0
    for n in range(1, 10):
        for f in range(n):
            for b in range(f+1):
                if n <= f+2*b:
                    continue
                for target in range(3):
                    for attack in compositions(b, 3):
                        hist = list(attack); hist[target] += n-b
                        assert certify(hist, f) == target
                        cv_cases += 1
    assert certify([4, 6], 3) is None  # true honest strict plurality need not suffice
    relabel_cases = 0
    for n in range(1, 7):
        for counts in compositions(n, 3):
            for f in range(n):
                for e in range(3):
                    intersection = {0, 1, 2}
                    for honest in compatible_histograms(counts, f):
                        labels = tuple(c for c, v in honest.items() for _ in range(v))
                        possible = {labels}
                        for _ in range(e):
                            possible |= {t[:i]+(c,)+t[i+1:] for t in list(possible) for i in range(len(t)) for c in range(3)}
                        for lab in possible:
                            ct = Counter(lab); top = max(ct.values())
                            winners = {c for c, v in ct.items() if v == top}
                            intersection &= winners if len(winners) == 1 else set()
                    got = certify_with_errors(counts, f, e)
                    assert intersection == ({got} if got is not None else set())
                    relabel_cases += 1
    return result | {'cv_cases': cv_cases, 'relabel_cases': relabel_cases,
                     'zero_missing_integer_and_unbiasedness_tests': 'passed'}


def build():
    by, funnel, tasks = load_records()
    for rel in ['data/tasks.jsonl', 'results/samples.jsonl', 'results/samples2.jsonl']:
        track(rel)
    rows = per_task_rows(by, tasks); ids, _ = common_task_set(rows)
    e1 = read('v3/results/e1_analysis.json')
    assert summarise({m: [r for r in rs if r['id'] in ids] for m, rs in rows.items()}) == e1['summary_common_tasks']
    assert len(ids) == 123
    mixture = heterogeneity(by, ids)
    m = mixture['answer_class']['9']['summary']
    assert abs(m['balanced']['estimate'] - .7354818123762579) < 1e-12
    assert abs(m['averaged_law_U']['estimate'] - .741573161917309) < 1e-12
    inst = instruments(tasks)
    assert inst['rates']['lexical']['all']['same'] == 28
    assert inst['rates']['judge_nano']['same_answer']['same'] == 714
    assert inst['rates']['judge_nano']['different_answer']['same'] == 13
    assert inst['rates']['judge_mini']['same_answer']['same'] == 709
    assert inst['rates']['judge_mini']['different_answer']['same'] == 3
    missing = {}
    for model in MODELS:
        valid = [r for t in ids for r in by[model][t] if r['has_answer']]
        sensitivity = {f'{field}_A{h}': summary([A_unbiased(list(Counter(r[field] for r in by[model][t] if r['has_answer'] and r['has_key']).values()), h) for t in ids])
                       for field in ['answer_class', 'semantic_class'] for h in [3, 10]}
        missing[model] = {'planned': 4500, **dict(funnel[model]), 'common_answered': len(valid),
                          'common_missing_key': sum(not r['has_key'] for r in valid),
                          'complete_case_sensitivity': sensitivity}
    checker = analyse_x2(lines('v3/results/x2_verifier.samples.jsonl'), tasks)
    assert checker == read('v3/results/x2_verifier_analysis.json')
    cert = read('v3/results/submission_certification_check.json')
    assert cert['config']['source_sha256'] == SOURCES['v3/results/e1_analysis.json']
    for p in (ROOT/'v3/code').glob('*.py'):
        track(str(p.relative_to(ROOT)))
    out = {'heterogeneity': mixture, 'instruments': inst, 'missingness': missing,
           'attacks': attacks(tasks), 'checker_historical': checker, 'tests': tests(),
           'sources_sha256': SOURCES, 'model_calls': 0, 'stored_candidate_executions': 0}
    out['sources_sha256']['v4/code/revision_analysis.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (V4/'results').mkdir(exist_ok=True)
    (V4/'results/revision_analysis.json').write_text(json.dumps(out, indent=2, ensure_ascii=False)+'\n')
    print('Corrected all 123-task mixture cells, 800 instrument pairs, missingness and attack provenance; tests passed.')


if __name__ == '__main__':
    build()
