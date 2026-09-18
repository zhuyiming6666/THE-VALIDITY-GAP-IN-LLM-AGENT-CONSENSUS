#!/usr/bin/env python3
"""Build blind distribution ZIPs from the blank master, never from returned labels."""
from pathlib import Path
import hashlib
import json
import zipfile
from score_annotations import V4, read_sheet, LABEL_COLUMNS, instruments


def main():
    root = V4 / 'annotation'
    blank = root / 'pairs_blind.tsv'
    for path in (blank, root/'practice/practice.tsv'):
        rows = read_sheet(path)
        if any(r[c].strip() for r in rows.values() for c in LABEL_COLUMNS):
            raise ValueError(f'{path} must be blank before packaging')
    # Hidden labels reflect the paper's current automatic relation. Preserve
    # old sampling-time verdicts so the change remains auditable.
    key_path = root/'admin/pairs_key.json'
    key = json.loads(key_path.read_text())
    for pid, row in read_sheet(blank).items():
        entry = key['pairs'][pid]
        entry.setdefault('original_sampling_instruments', entry['instruments'])
        entry['instruments'] = instruments(row)
    key['meta']['instrument_policy'] = 'lexical includes (answer,None); key_exact undefined on missing KEY; scorer recomputes all predicates'
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2)+'\n')
    common = ['ANNOTATION_GUIDE.md', 'practice/README.md', 'practice/practice.tsv']
    outputs = []
    for suffix, annotators in [('', 'ABC'), ('_A','A'), ('_B','B'), ('_C','C')]:
        dst = V4 / f'annotation{suffix}.zip'
        with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as z:
            for rel in common:
                z.write(root / rel, 'annotation/' + rel)
            for ann in annotators:
                z.writestr(f'annotation/annotator_{ann}.tsv', blank.read_bytes())
        outputs.append(dst)
    inputs = [p for p in root.rglob('*') if p.is_file() and p.name != 'manifest.json']
    inputs += [Path(__file__), Path(__file__).with_name('score_annotations.py'),
               Path(__file__).with_name('annotation_normalization.py')]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {'protocol':'v4-20260916', 'purpose':'Material fingerprints, not public preregistration or completed protocol freeze',
        'inputs_sha256':{str(p.relative_to(V4)):digest(p) for p in sorted(inputs)},
        'blind_archives_sha256':{p.name:digest(p) for p in outputs},
        'blind_archive_policy':'Guide + blank synthetic practice + blank formal sheet(s); excludes admin, identities, gold and reference labels.'}
    (root/'admin/manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    for p in outputs:
        print(p)


if __name__ == '__main__':
    main()
