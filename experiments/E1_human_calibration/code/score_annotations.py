#!/usr/bin/env python3
"""Score v4 human calibration; no API calls and no writes to annotation sheets.

Run from any directory: python3 /absolute/path/v4/code/score_annotations.py
--validate-only checks materials and reports missing labels without scoring.
See v4/annotation/admin/README.md for the pre-specified estimands.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

from annotation_normalization import answer_class, norm_key, semantic_class

V4 = Path(__file__).resolve().parents[1]
ANNOTATORS = ('A', 'B', 'C')
LABEL_COLUMNS = ('answer_agree', 'method_agree', 'method_reason', 'method_vague', 'note')
TEXT_COLUMNS = ('pair_id', 'task_id', 'bench', 'question', 'A_answer', 'A_key', 'B_answer', 'B_key')
AXES = {'answer': ('SAME', 'DIFF', 'UNSURE'), 'method': ('SAME', 'DIFF', 'VAGUE', 'UNSURE')}
ALIASES = {'同': 'SAME', '异': 'DIFF', '不确定': 'UNSURE', '含糊': 'VAGUE',
           'SAME': 'SAME', 'DIFF': 'DIFF', 'DIFFERENT': 'DIFF', 'UNSURE': 'UNSURE', 'VAGUE': 'VAGUE'}
REASONS = ('missing', 'uninformative', 'granularity_mismatch', 'domain_uncertainty', 'other')
DECISIVE = ('SAME', 'DIFF')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_sheet(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        if reader.fieldnames != list(TEXT_COLUMNS + LABEL_COLUMNS):
            raise ValueError(f'{path}: columns differ from the v4 blank template')
        rows = list(reader)
    ids = [r['pair_id'] for r in rows]
    if not rows or len(ids) != len(set(ids)) or any(not i for i in ids):
        raise ValueError(f'{path}: missing or duplicate pair IDs')
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError(f'{path}: malformed TSV row')
    return {r['pair_id']: r for r in rows}


def parse_label(row):
    out = {}
    for axis, allowed in AXES.items():
        raw = row[axis + '_agree'].strip()
        value = ALIASES.get(raw.upper()) if raw else None
        if raw and value not in allowed:
            raise ValueError(f'{row["pair_id"]}: invalid {axis} label {raw!r}')
        out[axis] = value
    reason = row['method_reason'].strip().lower()
    side = row['method_vague'].strip().upper()
    side = {'两条': 'BOTH'}.get(side, side)
    method = out['method']
    if side and side not in ('A', 'B', 'BOTH'):
        raise ValueError(f'{row["pair_id"]}: invalid method_vague')
    if reason and reason not in REASONS:
        raise ValueError(f'{row["pair_id"]}: invalid method_reason')
    if method == 'VAGUE' and (reason not in ('missing', 'uninformative') or not side):
        raise ValueError(f'{row["pair_id"]}: VAGUE requires missing/uninformative and A/B/两条')
    if method == 'UNSURE' and reason not in ('granularity_mismatch', 'domain_uncertainty', 'other'):
        raise ValueError(f'{row["pair_id"]}: UNSURE requires a comparison-uncertainty reason')
    if method != 'VAGUE' and side:
        raise ValueError(f'{row["pair_id"]}: method_vague only allowed for VAGUE')
    if method not in ('VAGUE', 'UNSURE') and reason:
        raise ValueError(f'{row["pair_id"]}: method_reason only allowed for VAGUE/UNSURE')
    if reason == 'other' and not row['note'].strip():
        raise ValueError(f'{row["pair_id"]}: reason=other requires note')
    out.update(reason=reason or None, side=side or None)
    return out


def majority(values):
    value, n = collections.Counter(values).most_common(1)[0]
    return value if n >= 2 else None


def instruments(row):
    aa, ba = (answer_class(row[s + '_answer'], row['bench']) for s in ('A', 'B'))
    ak, bk = (norm_key(row[s + '_key']) for s in ('A', 'B'))
    # Match the manuscript: a valid answer with no KEY is (answer, None).
    sa, sb = (semantic_class(row[s + '_answer'], row[s + '_key'], row['bench']) for s in ('A', 'B'))
    return {'lexical': None if sa is None or sb is None else sa == sb,
            'answer_only': None if aa is None or ba is None else aa == ba,
            'key_exact': None if ak is None or bk is None else ak == bk}


def load_materials(directory, key_file):
    directory, key_file = Path(directory), Path(key_file)
    key = json.loads(key_file.read_text())
    master = read_sheet(directory / 'pairs_blind.tsv')
    if set(master) != set(key['pairs']) or len(master) != key['meta']['n_pairs']:
        raise ValueError('Hidden key and blank template pair IDs differ')
    if any(r[c].strip() for r in master.values() for c in LABEL_COLUMNS):
        raise ValueError('pairs_blind.tsv must remain an unlabelled master')
    expected_hash = key['meta'].get('blind_template_sha256')
    if expected_hash and sha(directory / 'pairs_blind.tsv') != expected_hash:
        raise ValueError('Blank template differs from the frozen metadata hash')
    if sha(Path(__file__).with_name('annotation_normalization.py')) != key['meta']['normalizer_sha256']:
        raise ValueError('Frozen normalizer hash differs; explicitly revise the protocol before scoring')
    counts = collections.Counter(k['stratum'] for k in key['pairs'].values())
    strata = key['meta']['strata']
    if set(counts) != set(strata):
        raise ValueError('Stratum identities differ')
    for s, n in counts.items():
        if n != strata[s]['sample'] or strata[s]['population'] < n:
            raise ValueError(f'{s}: inconsistent stratum size')
    if sum(s['population'] for s in strata.values()) != key['meta']['sampling_frame_pairs']:
        raise ValueError('Population totals differ')
    files = {p.name for p in directory.glob('annotator_*.tsv')}
    if files != {f'annotator_{a}.tsv' for a in ANNOTATORS}:
        raise ValueError('Exactly annotator_A.tsv, annotator_B.tsv, annotator_C.tsv are required')
    labels, missing, hashes = {}, {}, {'key': sha(key_file), 'blank_template': sha(directory / 'pairs_blind.tsv'),
        'scorer': sha(__file__), 'normalizer': sha(Path(__file__).with_name('annotation_normalization.py'))}
    for ann in ANNOTATORS:
        path = directory / f'annotator_{ann}.tsv'
        sheet = read_sheet(path)
        if set(sheet) != set(master):
            raise ValueError(f'{path}: pair IDs differ from master')
        for pid, row in sheet.items():
            if any(row[c] != master[pid][c] for c in TEXT_COLUMNS):
                raise ValueError(f'{path}: {pid} source text changed')
        labels[ann] = {pid: parse_label(row) for pid, row in sheet.items()}
        missing[ann] = {axis: sum(v[axis] is None for v in labels[ann].values()) for axis in AXES}
        hashes[ann] = sha(path)
    records = []
    for pid, row in master.items():
        k = key['pairs'][pid]
        if row['task_id'] != k['task_id'] or row['bench'] != k['bench']:
            raise ValueError(f'{pid}: task metadata differs')
        votes = {a: labels[a][pid] for a in ANNOTATORS}
        majors = {axis: majority([v[axis] for v in votes.values()]) for axis in AXES}
        resolved = all(majors[axis] in DECISIVE for axis in AXES)
        s = k['stratum']
        records.append({'pair_id': pid, 'task_id': row['task_id'], 'bench': row['bench'],
            'stratum': s, 'weight': strata[s]['population'] / strata[s]['sample'],
            'same_model_pair': k['same_model_pair'], 'votes': votes, 'majority': majors,
            'reason_majority': majority([v['reason'] for v in votes.values()]),
            'side_majority': majority([v['side'] for v in votes.values()]),
            'joint': all(majors[a] == 'SAME' for a in AXES) if resolved else None,
            'instruments': instruments(row)})
    return records, key['meta'], missing, hashes


def ratio(num, den):
    return num / den if den else None


def mass(rows):
    return sum(r['weight'] for r in rows)


def share(rows, pred):
    return ratio(sum(r['weight'] for r in rows if pred(r)), mass(rows))


def statistics(rows):
    out = {}
    for axis, cats in AXES.items():
        for cat in (*cats, None):
            out[f'{axis}_{cat or "NO_MAJORITY"}_share'] = share(rows, lambda r: r['majority'][axis] == cat)
    for reason in REASONS:
        out[f'reason_{reason}_share'] = share(rows, lambda r: r['reason_majority'] == reason)
    out['uncertain_method_without_reason_majority_share'] = share(rows, lambda r: r['majority']['method'] in ('VAGUE', 'UNSURE') and r['reason_majority'] is None)
    for side in ('A', 'B', 'BOTH'):
        out[f'vague_side_{side}_share'] = share(rows, lambda r: r['majority']['method'] == 'VAGUE' and r['side_majority'] == side)
    joint = [r for r in rows if r['joint'] is not None]
    out['joint_reference_coverage'] = ratio(mass(joint), mass(rows))
    out['joint_same_share_all_pairs'] = share(rows, lambda r: r['joint'] is True)
    out['joint_same_rate_resolved'] = share(joint, lambda r: r['joint'])
    # Identification bounds if unresolved pairs could take either value;
    # these are not confidence intervals or bounds on p_max / A(h).
    out['joint_same_lower_bound'] = out['joint_same_share_all_pairs']
    out['joint_same_upper_bound'] = share(rows, lambda r: r['joint'] is not False)
    for a in (*AXES['answer'], None):
        for m in (*AXES['method'], None):
            out[f'cross_{a or "NO_MAJORITY"}_{m or "NO_MAJORITY"}_share'] = share(
                rows, lambda r: r['majority']['answer'] == a and r['majority']['method'] == m)
    # Primary: lexical vs joint. Controls: answer-only vs answer; key-exact vs method.
    for name, axis in [('lexical', 'joint'), ('answer_only', 'answer'), ('key_exact', 'method')]:
        def reference(r):
            if axis == 'joint':
                return r['joint']
            v = r['majority'][axis]
            return v == 'SAME' if v in DECISIVE else None
        usable = [r for r in rows if reference(r) is not None and r['instruments'][name] is not None]
        pos = [r for r in usable if reference(r)]
        neg = [r for r in usable if not reference(r)]
        pred_pos = [r for r in usable if r['instruments'][name]]
        pred_neg = [r for r in usable if not r['instruments'][name]]
        prefix = f'{name}_vs_{axis}_'
        all_pred_neg = [r for r in rows if r['instruments'][name] is False]
        out[prefix + 'instrument_diff_reference_coverage'] = ratio(mass(pred_neg), mass(all_pred_neg))
        out[prefix + 'coverage'] = ratio(mass(usable), mass(rows))
        out[prefix + 'error_rate'] = share(usable, lambda r: r['instruments'][name] != reference(r))
        out[prefix + 'false_accept_rate'] = share(neg, lambda r: r['instruments'][name])
        out[prefix + 'false_reject_rate'] = share(pos, lambda r: not r['instruments'][name])
        out[prefix + 'precision_same'] = share(pred_pos, reference)
        out[prefix + 'human_same_given_instrument_diff'] = share(pred_neg, reference)
        out[prefix + 'resolved_weight'] = mass(usable)
    s1 = [r for r in rows if r['stratum'] == 'S1_same_answer_lexical_diff']
    s1_resolved = [r for r in s1 if r['joint'] is not None]
    out['S1_joint_reference_coverage'] = ratio(mass(s1_resolved), mass(s1))
    out['S1_joint_same_rate_resolved'] = share(s1_resolved, lambda r: r['joint'])
    return out


def kappa(rows, axis, exclude_unsure=False):
    votes = [[r['votes'][a][axis] for a in ANNOTATORS] for r in rows]
    if exclude_unsure:
        votes = [v for v in votes if 'UNSURE' not in v]
    n = len(votes)
    if not n:
        return {'n_pairs': 0, 'raw_agreement': None, 'kappa': None}
    counts = [collections.Counter(v) for v in votes]
    observed = sum(sum(c * (c-1) for c in cs.values()) / 6 for cs in counts) / n
    margins = collections.Counter(v for vs in votes for v in vs)
    chance = sum((c / (3*n))**2 for c in margins.values())
    return {'n_pairs': n, 'raw_agreement': observed, 'chance_agreement': chance,
            'kappa': (observed - chance) / (1-chance) if chance < 1 else None}


def bootstrap(rows, n, seed):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['stratum']].append(row)
    rng = random.Random(seed)
    values = collections.defaultdict(list)
    point = statistics(rows)
    for _ in range(n):
        draw = [rng.choice(g) for s, g in sorted(groups.items()) for _ in range(len(g))]
        for key, value in statistics(draw).items():
            if value is not None:
                values[key].append(value)
    out = {}
    for key, value in point.items():
        xs = sorted(values[key])
        def pct(q):
            if not xs:
                return None
            at = (len(xs)-1)*q
            lo = int(at)
            return xs[lo] + (xs[min(lo+1,len(xs)-1)]-xs[lo])*(at-lo)
        out[key] = {'estimate': value, 'ci95': [pct(.025), pct(.975)], 'valid_resamples': len(xs)}
    return out


def make_report(rows, meta, hashes, resamples, seed):
    return {'protocol': 'v4-20260916', 'input_sha256': hashes, 'sampling': meta,
        'n_pairs': len(rows), 'n_tasks': len({r['task_id'] for r in rows}),
        'same_model_pairs': sum(r['same_model_pair'] for r in rows),
        'statistics': bootstrap(rows, resamples, seed),
        'agreement': {axis: {'all_categories': kappa(rows, axis),
                            'exclude_any_unsure': kappa(rows, axis, True)} for axis in AXES},
        'n_joint_resolved': sum(r['joint'] is not None for r in rows),
        'n_joint_positive': sum(r['joint'] is True for r in rows),
        'label_counts': {axis: dict(collections.Counter(r['majority'][axis] or 'NO_MAJORITY' for r in rows)) for axis in AXES},
        'reason_counts': dict(collections.Counter(r['reason_majority'] or 'NO_REASON_MAJORITY' for r in rows)),
        'instrument_counts': {name: {'n_defined': sum(r['instruments'][name] is not None for r in rows),
            'n_resolved': sum(r['instruments'][name] is not None and
                (r['joint'] is not None if axis == 'joint' else r['majority'][axis] in DECISIVE) for r in rows)}
            for name, axis in [('lexical', 'joint'), ('answer_only', 'answer'), ('key_exact', 'method')]},
        'per_stratum': {s: {'n_pairs': len(rs), 'label_counts': {axis: dict(collections.Counter(r['majority'][axis] or 'NO_MAJORITY' for r in rs)) for axis in AXES}, 'statistics': statistics(rs)}
            for s in sorted({r['stratum'] for r in rows})
            for rs in [[r for r in rows if r['stratum'] == s]]},
        'bootstrap': {'resamples': resamples, 'seed': seed,
            'design': 'within-stratum pair resampling; conditional on stored finite frame, not task-superpopulation uncertainty'},
        'unavailable_instruments': ['embedding', 'judge_nano', 'judge_mini'],
        'interpretation': [
            'Primary reference: two separate 2/3 majorities, both axes decisive; positive iff both SAME.',
            'Unresolved pairs are excluded from error denominators, never treated as negative; report coverage.',
            'Joint same share among all pairs is resolved-positive mass; lower/upper bounds allow unresolved labels either value.',
            'Fleiss kappa and raw agreement are unweighted descriptions of the stratified annotation sample.',
            'False acceptance = FP/(FP+TN); false rejection = FN/(FN+TP), with stratum weights.',
            'Controls use their own axis: answer_only vs answer; key_exact vs method.',
            'No semantic partition, p_max, A(h), or certification probability is estimated.',
            'Practice labels are synthetic instructional references and never enter this report.'],
        'per_pair': rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--annotation-dir', type=Path, default=V4/'annotation')
    ap.add_argument('--key-file', type=Path)
    ap.add_argument('--out', type=Path, default=V4/'results/annotation_scoring.json')
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20260916)
    ap.add_argument('--validate-only', action='store_true')
    args = ap.parse_args(argv)
    try:
        if args.bootstrap < 1:
            raise ValueError('--bootstrap must be positive')
        rows, meta, missing, hashes = load_materials(args.annotation_dir,
            args.key_file or args.annotation_dir/'admin/pairs_key.json')
        print(json.dumps({'n_pairs':len(rows),'missing_labels':missing},ensure_ascii=False,indent=2))
        if args.validate_only:
            print('Materials valid; no scoring output written.')
            return 0
        if any(n for ann in missing.values() for n in ann.values()):
            raise ValueError('All three independent sheets must be complete; no results written.')
        report = make_report(rows, meta, hashes, args.bootstrap, args.seed)
        if args.out.resolve().is_relative_to(args.annotation_dir.resolve()):
            raise ValueError('Write scoring results outside the annotation input directory')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(f'Wrote {args.out}')
        print('Primary lexical calibration:', json.dumps({k:v for k,v in report['statistics'].items()
            if k.startswith('lexical_vs_joint_')},ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
