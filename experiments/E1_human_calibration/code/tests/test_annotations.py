"""Hand-calculated statistical checks and synthetic end-to-end fixtures only."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import score_annotations as sc


def fixture():
    # FN=3, FP=2, TP=5, TN=10; unresolved=20. Weighted total=40.
    specs = [(3, True, False), (2, False, True), (5, True, True), (10, False, False), (20, None, True)]
    rows = []
    for i, (weight, joint, lexical) in enumerate(specs):
        method = 'VAGUE' if joint is None else ('SAME' if joint else 'DIFF')
        vote = {'answer':'SAME', 'method':method, 'reason':'missing' if joint is None else None,
                'side':'BOTH' if joint is None else None}
        rows.append({'pair_id':f'T{i}', 'task_id':'synthetic', 'bench':'gsm8k',
            'stratum':'S1_same_answer_lexical_diff', 'weight':weight, 'joint':joint,
            'majority':{'answer':'SAME','method':method},
            'reason_majority':vote['reason'], 'side_majority':vote['side'],
            'votes':{a:dict(vote) for a in sc.ANNOTATORS}, 'same_model_pair':True,
            'instruments':{'lexical':lexical,'answer_only':True,'key_exact':lexical if joint is not None else None}})
    return rows


def write_tsv(path, rows):
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=sc.TEXT_COLUMNS+sc.LABEL_COLUMNS,delimiter='\t')
        writer.writeheader(); writer.writerows(rows)


def disk_fixture(root, complete=True):
    (root/'admin').mkdir()
    rows=[]
    for i in range(3):
        rows.append(dict(zip(sc.TEXT_COLUMNS+sc.LABEL_COLUMNS,
            [f'X{i}',f't{i}','gsm8k','Synthetic multiplication task','12','3 * 4','12','multiply three by four','','','','',''])))
    write_tsv(root/'pairs_blind.tsv',rows)
    for ann in sc.ANNOTATORS:
        labelled=[dict(r,answer_agree='同' if complete else '',method_agree='同' if complete else '') for r in rows]
        write_tsv(root/f'annotator_{ann}.tsv',labelled)
    key={'meta':{'n_pairs':3,'sampling_frame_pairs':30,
        'normalizer_sha256':sc.sha(Path(sc.__file__).with_name('annotation_normalization.py')),
        'blind_template_sha256':sc.sha(root/'pairs_blind.tsv'),
        'strata':{'S1_same_answer_lexical_diff':{'sample':3,'population':30}}},
        'pairs':{r['pair_id']:{'task_id':r['task_id'],'bench':'gsm8k','stratum':'S1_same_answer_lexical_diff',
            'same_model_pair':True,'instruments':{'lexical':True}} for r in rows}}
    path=root/'admin/pairs_key.json';path.write_text(json.dumps(key));return path


class CalibrationTests(unittest.TestCase):
    def test_weighted_errors_and_unresolved_mass(self):
        stats=sc.statistics(fixture())
        for key,value in {'lexical_vs_joint_error_rate':.25,'lexical_vs_joint_false_accept_rate':1/6,
            'lexical_vs_joint_false_reject_rate':3/8,'joint_reference_coverage':.5,
            'joint_same_lower_bound':.2,'joint_same_upper_bound':.7,
            'lexical_vs_joint_human_same_given_instrument_diff':3/13}.items():
            self.assertAlmostEqual(stats[key],value,msg=key)

    def test_missing_key_lexical_still_defined(self):
        row={'bench':'gsm8k','A_answer':'12','B_answer':'12','A_key':'','B_key':''}
        self.assertEqual(sc.instruments(row),{'lexical':True,'answer_only':True,'key_exact':None})
        row['B_key']='3 * 4';self.assertIs(sc.instruments(row)['lexical'],False)

    def test_fleiss_hand_calculation(self):
        rows=fixture()[:2]
        for r in rows:
            for v in r['votes'].values():v['method']='SAME'
        rows[1]['votes']['C']['method']='DIFF'
        self.assertAlmostEqual(sc.kappa(rows,'method')['kappa'],-.2)
        rows[1]['votes']['C']['method']='UNSURE'
        self.assertEqual(sc.kappa(rows,'method',True)['n_pairs'],1)
        self.assertIsNone(sc.kappa(rows,'method',True)['kappa'])

    def test_zero_denominators_and_bootstrap_determinism(self):
        rows=fixture()[-1:]
        self.assertIsNone(sc.statistics(rows)['lexical_vs_joint_error_rate'])
        a=sc.bootstrap(fixture(),30,7);b=sc.bootstrap(fixture(),30,7)
        self.assertEqual(a,b)
        for stat in a.values():
            lo,hi=stat['ci95']
            if lo is not None:self.assertLessEqual(lo,hi)

    def test_reason_validation(self):
        row={'pair_id':'T','answer_agree':'same','method_agree':'vague',
            'method_reason':'granularity_mismatch','method_vague':'A','note':''}
        with self.assertRaises(ValueError):sc.parse_label(row)
        row.update(method_agree='UNSURE',method_vague='')
        self.assertEqual(sc.parse_label(row)['method'],'UNSURE')

    def test_end_to_end_ignores_stale_instrument_and_checks_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'annotation';root.mkdir();key=disk_fixture(root)
            out=Path(tmp)/'report.json'
            with contextlib.redirect_stdout(io.StringIO()):
                code=sc.main(['--annotation-dir',str(root),'--out',str(out),'--bootstrap','10'])
            self.assertEqual(code,0)
            report=json.loads(out.read_text())
            self.assertEqual(report['n_joint_resolved'],3)
            self.assertEqual(report['statistics']['lexical_vs_joint_false_reject_rate']['estimate'],1)
            text=(root/'annotator_B.tsv').read_text().replace('3 * 4','4 * 4')
            (root/'annotator_B.tsv').write_text(text)
            with self.assertRaisesRegex(ValueError,'source text changed'):
                sc.load_materials(root,key)

    def test_incomplete_never_writes_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'annotation';root.mkdir();disk_fixture(root,False)
            out=Path(tmp)/'report.json'
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                code=sc.main(['--annotation-dir',str(root),'--out',str(out),'--bootstrap','10'])
            self.assertEqual(code,1);self.assertFalse(out.exists())

    def test_majority_not_unanimity(self):
        self.assertEqual(sc.majority(['SAME','SAME','DIFF']),'SAME')
        self.assertIsNone(sc.majority(['SAME','VAGUE','UNSURE']))


if __name__ == '__main__':
    unittest.main()
