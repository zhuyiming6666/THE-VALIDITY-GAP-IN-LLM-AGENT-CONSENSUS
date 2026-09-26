#!/usr/bin/env python3
"""Conform the returned v4 human-annotation sheets to the frozen v4-20260916
template, then report what still blocks scoring.

Why this step exists
--------------------
The three returned sheets are hand-edited spreadsheet exports.  They differ
from the frozen blank template in three ways that ``score_annotations.py`` --
deliberately -- refuses to tolerate:

  1. Encoding.  B and C were saved by Excel as GBK/GB18030 (C is additionally
     a mixed export: four question cells carry UTF-8 dashes inside the GBK
     stream).  The scorer reads UTF-8 only.
  2. Export artefacts in the source columns.  Trailing spaces were added to
     some answers and one superscript was flattened (``side²`` -> ``side2``)
     when Excel round-tripped through GBK.  Annotators were only ever allowed
     to touch the five label columns, so these cells are restored verbatim from
     the frozen master ``pairs_blind.tsv``.
  3. Label vocabulary.  The frozen guide maps reason codes to categories:
     ``missing``/``uninformative`` -> VAGUE, and
     ``granularity_mismatch``/``domain_uncertainty``/``other`` -> UNSURE, with
     ``method_vague`` defined only for VAGUE and reason/side undefined on
     decisive rows.  Annotator A used the pre-2026-09-16 convention, in which
     VAGUE also covered granularity/level mismatch.  Sheet C also contains a
     ``uniformative`` typo.  The rule below re-derives the category from the
     stated reason, exactly as the frozen guide defines it.

The conformance rule (deterministic, applied uniformly to all three sheets)
--------------------------------------------------------------------------
  * unlabelled rows are left blank and counted, never fabricated;
  * decisive rows (answer/method SAME or DIFF) keep their category and lose any
    ``method_reason``/``method_vague`` value;
  * non-decisive rows are re-derived from ``method_reason``: a VAGUE reason
    yields VAGUE with ``method_vague`` retained, an UNSURE reason yields UNSURE
    with ``method_vague`` cleared;
  * a spelling map repairs the single observed reason typo.

Nothing is invented and no decisive judgement is ever reversed: only the
category name that the frozen guide attaches to an already-stated reason is
applied.  Every change is written to a machine-readable audit file, and rows
that remain non-conforming are reported instead of silently repaired.

Public API
----------
``prepare()`` writes conforming sheets into ``v4/annotation/`` (the location the
frozen protocol documents) and returns a status/audit dictionary.
"""
from __future__ import annotations

import csv
import collections
import hashlib
import io
import json
import sys
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
ANNOTATION = V4 / 'annotation'
RETURNED = ANNOTATION / 'results'  # returned annotator sheets (originally annotation/结果)
WORKING = ANNOTATION / '工作文件'
CORRECTIONS_PATH = ANNOTATION / 'manual_corrections.json'
AUDIT_PATH = V4 / 'results' / 'annotation_normalization_audit.json'

ANNOTATORS = ('A', 'B', 'C')
TEXT_COLUMNS = ('pair_id', 'task_id', 'bench', 'question',
                'A_answer', 'A_key', 'B_answer', 'B_key')
LABEL_COLUMNS = ('answer_agree', 'method_agree', 'method_reason',
                 'method_vague', 'note')
COLUMNS = TEXT_COLUMNS + LABEL_COLUMNS

ANSWER_ALIASES = {
    '同': 'SAME', '异': 'DIFF', '不确定': 'UNSURE',
    'SAME': 'SAME', 'DIFF': 'DIFF', 'DIFFERENT': 'DIFF', 'UNSURE': 'UNSURE',
}
METHOD_ALIASES = dict(ANSWER_ALIASES)
METHOD_ALIASES.update({'含糊': 'VAGUE', 'VAGUE': 'VAGUE'})
SIDE_ALIASES = {'A': 'A', 'B': 'B', '两条': 'BOTH', 'BOTH': 'BOTH'}
VAGUE_REASONS = ('missing', 'uninformative')
UNSURE_REASONS = ('granularity_mismatch', 'domain_uncertainty', 'other')
# Export typos observed in the returned sheets; mapped to the frozen vocabulary.
REASON_SPELLING = {'uniformative': 'uninformative'}
DECISIVE = ('SAME', 'DIFF')
DECODINGS = ('utf-8-sig', 'gb18030')
# Files that were never part of the frozen layout and would break the scorer's
# strict ``annotator_*.tsv`` glob.  They are working artefacts, not data.
STRAY_FILES = ('annotator_A_中文.tsv',)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode(raw, path):
    """Decode a returned sheet, tolerating GBK and mixed GBK/UTF-8 exports."""
    for enc in DECODINGS:
        try:
            return raw.decode(enc), enc, 0
        except UnicodeDecodeError:
            continue
    text = raw.decode('gb18030', errors='replace')
    return text, 'gb18030+replace', text.count('\ufffd')


def read_sheet(path):
    raw = Path(path).read_bytes()
    text, encoding, replaced = decode(raw, path)
    reader = csv.DictReader(io.StringIO(text), delimiter='\t')
    if reader.fieldnames != list(COLUMNS):
        raise ValueError(f'{path}: header does not match the frozen template')
    rows = list(reader)
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError(f'{path}: malformed TSV row')
    return rows, encoding, replaced


def canonical(aliases, value):
    raw = (value or '').strip()
    if not raw:
        return None
    return aliases.get(raw.upper())


def has_labels(path):
    try:
        rows, _, _ = read_sheet(path)
    except (OSError, ValueError):
        return False
    return any((r[c] or '').strip() for r in rows for c in LABEL_COLUMNS)


def load_corrections():
    """Coordinator-supplied values for cells a returned sheet left blank.

    Kept in a separate, dated file so the returned sheets stay untouched and
    every human intervention is visible in the audit.
    """
    if not CORRECTIONS_PATH.exists():
        return []
    return json.loads(CORRECTIONS_PATH.read_text(encoding='utf-8')).get('corrections', [])


def locate(annotator):
    """Prefer the returned sheet, then a filled sheet already in annotation/.

    A sheet that carries no labels is only used when nothing better exists, so
    re-running never masks a newly returned file with an older placeholder.
    """
    candidates = [RETURNED / f'annotator_{annotator}.tsv',
                  ANNOTATION / f'annotator_{annotator}.tsv']
    existing = [p for p in candidates if p.exists()]
    for path in existing:
        if has_labels(path):
            return path
    return existing[0] if existing else None


def conform_labels(row):
    """Apply the frozen reason->category mapping to one returned row.

    Returns ``(labels, changes, canonicalisations, issues)``.  ``changes`` holds
    only *semantic* revisions -- a category re-derived from the stated reason, a
    reason/side value dropped where the frozen guide leaves it undefined, or a
    reason spelling repaired.  Translating a sheet's own tokens (``同``,
    ``same``) into the frozen English vocabulary is recorded separately as a
    ``canonicalisation`` so it is never confused with an interpretive change.
    """
    changes, canonicalisations, issues = [], [], []
    pid = row['pair_id']
    if not any((row[c] or '').strip() for c in LABEL_COLUMNS):
        return None, changes, canonicalisations, issues    # unlabelled, not an error

    answer = canonical(ANSWER_ALIASES, row['answer_agree'])
    method = canonical(METHOD_ALIASES, row['method_agree'])
    raw_reason = (row['method_reason'] or '').strip().lower()
    reason = REASON_SPELLING.get(raw_reason, raw_reason)
    side = canonical(SIDE_ALIASES, row['method_vague'])
    note = row['note'] or ''

    def slot(a, m, r, s):
        return f'{a or ""}|{m or ""}|{r or ""}|{s or ""}'

    baseline = slot(answer, method, raw_reason, side)
    canonical_baseline = slot(answer, method, reason, side)
    raw_baseline = '|'.join((row[c] or '').strip() for c in LABEL_COLUMNS[:4])
    if raw_baseline != canonical_baseline:
        canonicalisations.append({'pair_id': pid, 'from': raw_baseline, 'to': canonical_baseline})

    if answer is None:
        issues.append({'pair_id': pid, 'issue': 'answer_agree missing or unrecognised',
                       'value': row['answer_agree']})

    if method in DECISIVE:
        new_method, new_reason, new_side = method, None, None
    elif method in ('VAGUE', 'UNSURE'):
        if reason in VAGUE_REASONS:
            new_method, new_reason, new_side = 'VAGUE', reason, side
            if side is None:
                issues.append({'pair_id': pid,
                               'issue': 'VAGUE requires method_vague in A/B/两条',
                               'value': ''})
        elif reason in UNSURE_REASONS:
            new_method, new_reason, new_side = 'UNSURE', reason, None
        else:
            new_method, new_reason, new_side = method, None, side
            issues.append({'pair_id': pid,
                           'issue': 'non-decisive label has no recognised reason',
                           'value': row['method_reason']})
    else:
        new_method, new_reason, new_side = method, None, side
        issues.append({'pair_id': pid, 'issue': 'method_agree missing or unrecognised',
                       'value': row['method_agree']})

    if new_reason == 'other' and not note.strip():
        issues.append({'pair_id': pid, 'issue': 'reason=other requires a note', 'value': ''})

    after = slot(answer, new_method, new_reason, new_side)
    if after != baseline:
        changes.append({'pair_id': pid,
                        'rule': _change_rule(method, new_method, raw_reason, reason, side, new_side),
                        'before': baseline, 'after': after})
    labels = {'answer_agree': answer or '', 'method_agree': new_method or '',
              'method_reason': new_reason or '', 'method_vague': new_side or '',
              'note': note}
    return labels, changes, canonicalisations, issues


def _change_rule(method, new_method, raw_reason, reason, side, new_side):
    if method in DECISIVE:
        return 'reason/side cleared on decisive row'
    if new_method != method:
        return 'category re-derived from reason'
    if side is not None and new_side is None:
        return 'method_vague cleared for non-VAGUE category'
    if raw_reason != reason:
        return 'reason spelling repaired'
    return 'label adjusted'


def prepare(verbose=True):
    """Normalise the returned sheets into v4/annotation/ and audit the changes."""
    master_rows, _, _ = read_sheet(ANNOTATION / 'pairs_blind.tsv')
    master = {r['pair_id']: r for r in master_rows}
    order = [r['pair_id'] for r in master_rows]
    if len(order) != len(set(order)):
        raise ValueError('pairs_blind.tsv has duplicate pair IDs')

    WORKING.mkdir(exist_ok=True)
    moved = []
    for name in STRAY_FILES:
        stray = ANNOTATION / name
        if stray.exists():
            target = WORKING / name
            stray.replace(target)
            moved.append({'from': f'annotation/{name}', 'to': f'annotation/工作文件/{name}',
                          'reason': 'not part of the frozen layout; breaks the scorer glob'})

    corrections = load_corrections()
    audit = {'protocol': 'v4-20260916',
             'purpose': 'Conformance audit for the returned human-annotation sheets',
             'rule': {
                 'decisive_rows': 'keep category; method_reason and method_vague cleared',
                 'vague_reasons': list(VAGUE_REASONS) + ['-> VAGUE (method_vague retained)'],
                 'unsure_reasons': list(UNSURE_REASONS) + ['-> UNSURE (method_vague cleared)'],
                 'source_columns': 'restored verbatim from pairs_blind.tsv',
                 'reason_spelling_repairs': REASON_SPELLING},
             'manual_corrections': {
                 'file': (str(CORRECTIONS_PATH.relative_to(V4)) if CORRECTIONS_PATH.exists() else None),
                 'sha256': (sha(CORRECTIONS_PATH) if CORRECTIONS_PATH.exists() else None),
                 'n': len(corrections)},
             'files_moved': moved, 'annotators': {}}

    for ann in ANNOTATORS:
        source = locate(ann)
        entry = {'source': None, 'encoding': None, 'decoded_replacement_chars': 0,
                 'source_sha256': None, 'status': 'missing', 'rows': 0,
                 'unlabelled_rows': 0, 'text_repairs': {}, 'label_changes': {},
                 'canonicalisations': {}, 'corrections_applied': [], 'issues': []}
        if source is not None:
            rows, encoding, replaced = read_sheet(source)
            entry.update(source=str(source.relative_to(V4)), encoding=encoding,
                         decoded_replacement_chars=replaced,
                         source_sha256=sha(source), rows=len(rows))
            by_id = {r['pair_id']: r for r in rows}
            if set(by_id) != set(master) or len(by_id) != len(rows):
                raise ValueError(f'{source}: pair IDs differ from the frozen master')
            records, repairs = {}, {}
            changes, cans = [], []
            for pid in order:
                row = by_id[pid]
                for corr in corrections:
                    if corr['annotator'] == ann and corr['pair_id'] == pid:
                        column = corr['column']
                        if column not in LABEL_COLUMNS:
                            raise ValueError(f'{CORRECTIONS_PATH}: bad column {column!r}')
                        entry['corrections_applied'].append(
                            {'pair_id': pid, 'column': column,
                             'from': (row[column] or '').strip(), 'to': corr['value'],
                             'supplied_by': corr.get('supplied_by', 'coordinator'),
                             'date': corr.get('date'), 'reason': corr.get('reason')})
                        row[column] = corr['value']
                for col in TEXT_COLUMNS:
                    if row[col] != master[pid][col]:
                        repairs.setdefault(col, []).append(pid)
                labels, row_changes, row_cans, issues = conform_labels(row)
                if labels is None:
                    labels = {c: '' for c in LABEL_COLUMNS}
                    entry['unlabelled_rows'] += 1
                changes.extend(row_changes)
                cans.extend(row_cans)
                entry['issues'].extend(issues)
                records[pid] = {**{c: master[pid][c] for c in TEXT_COLUMNS}, **labels}
            entry['label_changes'] = _summarise(changes)
            entry['canonicalisations'] = _summarise(cans)
            entry['text_repairs'] = {c: {'n': len(v), 'examples': v[:3]}
                                     for c, v in sorted(repairs.items())}
            _write_sheet(ANNOTATION / f'annotator_{ann}.tsv', order, records)
            entry['status'] = ('unlabelled' if entry['unlabelled_rows'] == entry['rows']
                               else 'labelled')
            entry['staged_sha256'] = sha(ANNOTATION / f'annotator_{ann}.tsv')
        audit['annotators'][ann] = entry

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    if verbose:
        _print_summary(audit)
    return audit


def _summarise(records, default_rule='token canonicalisation'):
    tally = collections.Counter(r.get('rule', default_rule) for r in records)
    return {'n': len(records), 'by_rule': dict(tally), 'examples': records[:5]}


def _write_sheet(path, order, records):
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), delimiter='\t',
                                lineterminator='\r\n')
        writer.writeheader()
        for pid in order:
            writer.writerow(records[pid])


def _print_summary(audit):
    for ann, e in audit['annotators'].items():
        print(f"[{ann}] {e['status']:9s} {e['source'] or '(no source)'}"
              f" | encoding={e['encoding']} | rows={e['rows']}"
              f" | unlabelled={e['unlabelled_rows']}"
              f" | canonicalised={e['canonicalisations'].get('n', 0)}"
              f" | semantic_changes={e['label_changes'].get('n', 0)} {e['label_changes'].get('by_rule', {})}"
              f" | text_repairs={ {k: v['n'] for k, v in e['text_repairs'].items()} }"
              f" | issues={len(e['issues'])}")
        for issue in e['issues']:
            print(f"      ! {issue['pair_id']}: {issue['issue']} (value={issue['value']!r})")
    if audit['files_moved']:
        print('moved:', ', '.join(m['from'] + ' -> ' + m['to'] for m in audit['files_moved']))


def main(argv=None):
    prepare(verbose=True)
    print(f'Wrote {AUDIT_PATH.relative_to(V4)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
