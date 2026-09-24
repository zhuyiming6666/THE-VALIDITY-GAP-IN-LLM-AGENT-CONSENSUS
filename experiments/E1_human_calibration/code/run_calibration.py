#!/usr/bin/env python3
"""One-command entry point for the v4 human-calibration experiment.

    python3 v4/code/run_calibration.py                # normalise, validate, score if ready
    python3 v4/code/run_calibration.py --validate-only  # stop after validation

It performs, in order:

  1. ``normalize_annotations.prepare``  -- conform the returned sheets in
     ``annotation/结果/`` to the frozen template and write a change audit;
  2. ``score_annotations.load_materials`` -- run the frozen validator
     (master hash, pair IDs, source-text integrity, label legality, strata);
  3. ``score_annotations.main`` -- only when all three sheets are complete and
     issue-free, write ``results/annotation_scoring.json``.

If anything is pending, nothing is scored: the frozen protocol forbids
reporting partial results, and unresolved labels are never imputed.  A status
file is always written to ``results/calibration_status.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import normalize_annotations
import score_annotations

V4 = Path(__file__).resolve().parents[1]
ANNOTATION = V4 / 'annotation'
KEY_FILE = ANNOTATION / 'admin' / 'pairs_key.json'
OUT = V4 / 'results' / 'annotation_scoring.json'
STATUS = V4 / 'results' / 'calibration_status.json'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20260916)
    args = ap.parse_args(argv)

    audit = normalize_annotations.prepare(verbose=True)

    # The frozen validator enforces every integrity check; reuse it verbatim.
    try:
        rows, meta, missing, hashes = score_annotations.load_materials(ANNOTATION, KEY_FILE)
    except (ValueError, OSError, KeyError) as exc:
        print(f'\nFROZEN VALIDATION FAILED: {exc}', file=sys.stderr)
        return 1

    open_issues = {a: e['issues'] for a, e in audit['annotators'].items() if e['issues']}
    incomplete = {a: m for a, m in missing.items() if any(m.values())}
    unlabelled = [a for a, e in audit['annotators'].items() if e['status'] != 'labelled']

    status = {
        'protocol': meta['protocol_version'],
        'n_pairs': len(rows),
        'annotator_status': {a: audit['annotators'][a]['status'] for a in audit['annotators']},
        'missing_labels': missing,
        'open_issues': open_issues,
        'sources': {a: audit['annotators'][a]['source'] for a in audit['annotators']},
        'input_sha256': hashes,
        'normalization_audit': str(normalize_annotations.AUDIT_PATH.relative_to(V4)),
        'scored': False,
        'scoring_output': None,
    }

    print('\nMissing labels per annotator (answer / method):')
    for a, m in missing.items():
        print(f'  {a}: {m["answer"]} / {m["method"]}')

    if unlabelled or incomplete or open_issues:
        status['blocked_by'] = {
            'unlabelled_sheets': unlabelled,
            'incomplete_sheets': sorted(incomplete),
            'open_issues': sorted(open_issues),
        }
        STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
        print('\nPENDING: the frozen protocol requires three complete, conforming sheets.')
        if incomplete:
            print('  incomplete sheets:', ', '.join(sorted(incomplete)))
        if open_issues:
            for a, issues in sorted(open_issues.items()):
                for issue in issues:
                    print(f'  {a} {issue["pair_id"]}: {issue["issue"]}')
        if args.validate_only:
            print('Materials valid; no scoring output written.')
        print(f'Wrote {STATUS.relative_to(V4)}')
        return 0

    if args.validate_only:
        print('Materials valid and complete; no scoring output written.')
        return 0

    code = score_annotations.main(['--annotation-dir', str(ANNOTATION),
                                   '--key-file', str(KEY_FILE), '--out', str(OUT),
                                   '--bootstrap', str(args.bootstrap),
                                   '--seed', str(args.seed)])
    if code == 0:
        status['scored'] = True
        status['scoring_output'] = str(OUT.relative_to(V4))
        status['scoring_output_sha256'] = normalize_annotations.sha(OUT)
    STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(f'Wrote {STATUS.relative_to(V4)}')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
