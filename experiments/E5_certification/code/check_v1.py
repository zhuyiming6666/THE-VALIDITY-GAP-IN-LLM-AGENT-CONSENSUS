"""V1: reconstruct raw counts and compare exact p(mode)^7 to archived MC."""
import collections
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'llmbft'
OUT = ROOT / 'grain-iclr11/audit'
sys.path.insert(0, str(OLD / 'v3/code'))
from recompute_e1 import load_records, per_task_rows, common_task_set, MODELS
from submission_certification_check import compositions


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    by, funnel, tasks = load_records()
    rows = per_task_rows(by, tasks)
    ids, _ = common_task_set(rows)
    source = OLD / 'v3/results/e1_analysis.json'
    archived = json.loads(source.read_text())
    certfile = OLD / 'v3/results/submission_certification_check.json'
    cert = json.loads(certfile.read_text())
    assert ids == archived['common_tasks']['ids'] and len(ids) == 123
    assert sha(source) == cert['config']['source_sha256']
    assert sha(OLD / 'v3/code/scoring.py') == sha(OLD / 'v5/code/annotation_normalization.py')
    lookup = {m: {r['id']: r for r in rows[m]} for m in MODELS}
    compared = 0
    for model in MODELS:
        old = {r['id']: r for r in archived['per_task'][model]}
        for tid in ids:
            for field in ('counts_verdict', 'counts_semantic'):
                assert lookup[model][tid][field] == old[tid][field], (model, tid, field)
                compared += 1
    checked = 0
    for hist in compositions(7, 8):
        assert (hist[0] - max(hist[1:]) > 6) == (hist[0] == 7)
        checked += 1
    report = {'config': {'n': 10, 'f': 3, 'h': 7, 'task_ids': ids,
        'models': MODELS, 'aggregation': 'equal models within each task, then equal tasks',
        'estimand': 'conditional plug-in probability of certification of a designated empirical mode',
        'not_estimands': ['U-statistic A(7)', 'seventh power of mean modal mass', 'true-law population coverage'],
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [source, certfile,
            OLD/'results/samples.jsonl', OLD/'results/samples2.jsonl', OLD/'data/tasks.jsonl',
            OLD/'v3/code/scoring.py']}},
        'checks': {'raw_count_arrays_match': compared, 'common_tasks': len(ids),
            'bench_counts': dict(collections.Counter(tasks[t]['bench'] for t in ids)),
            'normalizers_identical': True, 'event_histograms_checked': checked},
        'results': {}, 'per_task': {}}
    rng = np.random.default_rng(cert['config']['task_bootstrap_seed'])
    boot = rng.integers(0, len(ids), size=(cert['config']['task_bootstrap_resamples'], len(ids)))
    draws = cert['config']['draws_per_task_model_granularity_budget']
    for level, field in [('verdict', 'counts_verdict'), ('answer_key_lexical', 'counts_semantic')]:
        values = []
        records = []
        per_model = {}
        for model in MODELS:
            exact = np.array([(lookup[model][t][field][0] / sum(lookup[model][t][field]))**7 for t in ids])
            mc = {r['task_id']: r['budgets']['3']['target_certification_probability'] for r in cert['per_task'][model][level]}
            observed = np.array([mc[t] for t in ids])
            se = float(np.sqrt(np.sum(exact*(1-exact)/draws))/len(ids))
            per_model[model] = {'exact': float(exact.mean()), 'archived_mc': float(observed.mean()),
                'difference': float(observed.mean()-exact.mean()), 'mc_standard_error': se}
            values.append(exact)
            for tid, q, sim in zip(ids, exact, observed):
                records.append({'model': model, 'task_id': tid, 'counts': lookup[model][tid][field],
                    'exact': float(q), 'archived_mc': float(sim)})
        vals = np.array(values)
        taskmeans = vals.mean(axis=0)
        mean = float(taskmeans.mean())
        mc = cert['aggregate_across_models'][level]['target_certification']['3']['estimate']
        se = float(np.sqrt(np.sum(vals*(1-vals)/draws))/vals.size)
        delta = mc-mean
        target = '0.843' if level == 'verdict' else '0.004'
        report['results'][level] = {'exact': mean, 'archived_mc': mc, 'difference': delta,
            'mc_standard_error': se, 'difference_in_mc_se': delta/se,
            'same_three_decimal_rounding': f'{mean:.3f}' == f'{mc:.3f}' == target,
            'task_bootstrap_ci95': np.quantile(taskmeans[boot].mean(axis=1), [.025,.975]).tolist(),
            'per_model': per_model}
        report['per_task'][level] = records
    report['passed'] = all(r['same_three_decimal_rounding'] and abs(r['difference_in_mc_se']) < 4 for r in report['results'].values())
    report['limitations'] = ['Only checks exact consistency conditional on empirical distributions; not sample-size bias or semantic validity.',
        'MC comparison uses expected simulation standard error; task-bootstrap intervals quantify a different source of variation.']
    (OUT/'v1_certification_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'passed': report['passed'], 'checks': report['checks'], 'results': report['results']}, ensure_ascii=False, indent=2))
    assert report['passed']


if __name__ == '__main__':
    main()
