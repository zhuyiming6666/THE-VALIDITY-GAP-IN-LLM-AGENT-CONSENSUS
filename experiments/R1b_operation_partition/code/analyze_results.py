"""R1b diagnostics and paired comparisons; run after score_partitions.py."""
import collections
import csv
import itertools
import sys
import json
import numpy as np
import score_partitions as s

res = json.loads((s.ROOT / 'results/r1b_partition_results.json').read_text())
sheets = {a: s.read(a) for a in 'ABC'}
# Validate read-only reply fields against the source lines named in the admin key.
sys.path.insert(0, str(s.ROOT.parent / 'E1_human_calibration' / 'code'))
from annotation_normalization import answer_class, norm_key
source = {name: (s.ROOT.parent / 'baseline' / 'data' / name).read_text().splitlines()
          for name in {x['file'] for c in s.KEY['cells'].values() for x in c['items']}}
index = {r['item']: r for r in sheets['A']}
for c in s.KEY['cells'].values():
    for x in c['items']:
        raw = json.loads(source[x['file']][x['line'] - 1])
        assert raw['model'] == c['model'] and raw['task_id'] == c['task_id']
        assert repr(answer_class(raw.get('answer'), c['bench'])) == x['answer_class']
        assert norm_key(raw.get('key')) == x['key_norm']
        assert index[x['item']]['answer'].strip() == (raw.get('answer') or '').strip()
        assert index[x['item']]['key'].strip() == (raw.get('key') or '').strip()
cells = res['cells']
tids = list(cells)
rng = np.random.default_rng(20260926)
boot = rng.integers(24, size=(10000, 24))
out = {'paired_differences': {}, 'leave_one_out': {}, 'by_benchmark': {},
       'by_model': {}, 'annotators': {}, 'pairwise_agreement': {}}
for left, right in [('answer', 'human'), ('human', 'lexical')]:
    v = np.array([cells[t][left]['cov_h7'] - cells[t][right]['cov_h7'] for t in tids])
    out['paired_differences'][left + '_minus_' + right] = {
        'mean': float(v.mean()), 'ci95': np.quantile(v[boot].mean(axis=1), [.025, .975]).tolist()}
for a in 'ABC':
    r = s.run({b: sheets[b] for b in 'ABC' if b != a})
    out['leave_one_out'][a] = r['summary']['human']['cov_h7']['mean']
    out['annotators'][a] = {
        'rows': len(sheets[a]), 'vague': sum(r['vague'].strip().upper() == 'Y' for r in sheets[a]),
        'unsure_notes': sum('unsure' in r['note'].lower() for r in sheets[a]),
        'joint_cov_h7': float(np.mean([c['per_annotator'][a]['cov_h7'] for c in cells.values()]))}
all_three, joint_agreement = [], []
for tid, key in s.KEY['cells'].items():
    items = [x['item'] for x in key['items']]
    ans = {x['item']: x['answer_class'] for x in key['items']}
    parts = [s.partition({r['item']: None if r['vague'].strip().upper() == 'Y' else r['group'].strip()
                          for r in sheets[a] if r['task'] == tid}) for a in 'ABC']
    all_three.append(s.stats({i: (ans[i], *(p[i] for p in parts)) for i in items})['cov_h7'])
    for (a, p), (b, q) in itertools.combinations(zip('ABC', parts), 2):
        z = out['pairwise_agreement'].setdefault(a+b, {'ari': [], 'pair_agreement': []})
        z['ari'].append(s.ari(p, q, items))
        z['pair_agreement'].append(float(np.mean([(p[i] == p[j]) == (q[i] == q[j])
                                                 for i, j in itertools.combinations(items, 2)])))
        joint_agreement.append(s.ari({i: (ans[i], p[i]) for i in items},
                                    {i: (ans[i], q[i]) for i in items}, items))
out['all_three_intersection_cov_h7'] = float(np.mean(all_three))
out['joint_mean_ari'] = float(np.mean(joint_agreement))
for z in out['pairwise_agreement'].values():
    for m in z: z[m] = float(np.mean(z[m]))
for field, target in [('bench', 'by_benchmark'), ('model', 'by_model')]:
    for label in sorted({c[field] for c in cells.values()}):
        selected = [c for c in cells.values() if c[field] == label]
        out[target][label] = {'n_cells': len(selected), **{
            g: float(np.mean([c[g]['cov_h7'] for c in selected]))
            for g in ('answer', 'human', 'lexical')}}
out['audit'] = {
    'majority_vague': sum(c['n_vague_majority'] for c in cells.values()),
    'closure_added_pairs': sum(len(c['closure_added_pairs']) for c in cells.values()),
    'closure_cells': [t for t, c in cells.items() if c['closure_added_pairs']],
    'operation_cross_answer_pairs': sum(len(c['operation_cross_answer_pairs']) for c in cells.values()),
    'cross_answer_cells': [t for t, c in cells.items() if c['operation_cross_answer_pairs']],
    'full_pool_answer_cov_h7_selected_cells': float(np.mean([
        c['answer_pmax_full'] ** 7 for c in s.KEY['cells'].values()]))}
# Same benchmark allocation (8/8/8) maintained in this descriptive bootstrap.
idx = [[j for j, t in enumerate(tids) if cells[t]['bench'] == b]
       for b in ('gsm8k', 'mmlu', 'mbpp')]
strat = np.concatenate([rng.choice(z, size=(10000, len(z))) for z in idx], axis=1)
v = np.array([cells[t]['human']['cov_h7'] for t in tids])
out['benchmark_stratified_cov_h7_ci95'] = np.quantile(v[strat].mean(axis=1), [.025, .975]).tolist()
out['interval_scope'] = 'Descriptive resampling of the selected cells, conditional on 12 replies and consensus labels; not a probability-sampling CI for all 369 cells.'
(s.ROOT / 'results/r1b_diagnostics.json').write_text(json.dumps(out, indent=2))
with (s.ROOT / 'results/r1b_cells.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['cell', 'benchmark', 'model', 'task_id', 'answer_cov_h7', 'human_cov_h7',
                'lexical_cov_h7', 'human_A3', 'majority_vague', 'closure_added_pairs'])
    for t, c in cells.items():
        w.writerow([t, c['bench'], c['model'], c['task_id'], c['answer']['cov_h7'],
                    c['human']['cov_h7'], c['lexical']['cov_h7'], c['human']['A3'],
                    c['n_vague_majority'], len(c['closure_added_pairs'])])
print(json.dumps(out, indent=2))
