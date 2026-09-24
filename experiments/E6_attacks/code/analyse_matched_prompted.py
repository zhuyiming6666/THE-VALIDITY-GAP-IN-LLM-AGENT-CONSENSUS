"""Audit all matched cells and score missing replies without dropping trials."""
from pathlib import Path
from collections import Counter,defaultdict
import sys,json,hashlib
import numpy as np
V5=Path(__file__).resolve().parents[1];ROOT=V5.parent
sys.path.insert(0,str(ROOT/'baseline/code'))
from scoring import answer_class

def analyse():
 paths=[V5/'results/x3_attacks.samples.jsonl',V5/'results/matched_prompted.samples.jsonl']
 arms=[list(map(json.loads,p.read_text().splitlines())) for p in paths]
 assert len(arms[0])==288 and len(arms[1])==144
 assert all(not s['error'] for r in arms[1] for rnd in r['history'] for s in rnd), 'Transport-incomplete primary trial'
 key=lambda r:(r['task_id'],r['f'],r['rep'])
 reference={key(r) for r in arms[0] if r['strategy']=='forced'}
 assert len(reference)==144 and {key(r) for r in arms[1]}==reference
 assert len({key(r) for r in arms[1]})==len(arms[1])
 tasks={t['id']:t for t in map(json.loads,(ROOT/'baseline/data/tasks.jsonl').read_text().splitlines())}
 forced={key(r):r for r in arms[0] if r['strategy']=='forced'}
 scored=[]
 for r in arms[0]+arms[1]:
  assert r['config_hash']==forced[key(r)]['config_hash']
  bench=tasks[r['task_id']]['bench'];gold=answer_class(tasks[r['task_id']]['answer'],bench)
  target=answer_class(forced[key(r)]['history'][0][0]['answer'],bench)
  assert target!=gold
  for rnd,hist in enumerate(r['history']):
   assert len(hist)==10
   labels=[answer_class(s['answer'],bench) for s in hist]
   b=0 if r['strategy']=='none' else r['f'];hon=labels[b:];att=labels[:b]
   majority=Counter(v for v in hon if v is not None).most_common(1)
   maj=majority[0][0] if majority else None
   scored.append({'task_id':r['task_id'],'f':r['f'],'rep':r['rep'],'arm':r['strategy'],'round':rnd,
    'all_unanimity':int(None not in labels and len(set(labels))==1),
    'honest_unanimity':int(None not in hon and len(set(hon))==1),
    'honest_accuracy':int(maj==gold),
    'attacker_slots':b,'target_hits':sum(v==target for v in att),
    'gold_hits':sum(v==gold for v in att),
    'opposes_gold':sum(v is not None and v!=gold for v in att),
    'differs_honest_majority':sum(v is not None and maj is not None and v!=maj for v in att),
    'missing_attacker':sum(v is None for v in att),'missing_honest':sum(v is None for v in hon),
    'request_errors':sum(s['error'] is not None for s in hist)})
 def boot(vals):
  vals=np.array(vals); rng=np.random.default_rng(20260918)
  draws=vals[rng.integers(len(vals),size=(5000,len(vals)))].mean(axis=1)
  return {'estimate':float(vals.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist(),'tasks':len(vals)}
 cells={}
 for arm in ['none','forced','prompted']:
  for f in [1,2,3]:
   for rnd in range(3):
    rr=[r for r in scored if r['arm']==arm and r['f']==f and r['round']==rnd];assert len(rr)==48
    tids=sorted({r['task_id'] for r in rr});assert len(tids)==24
    out={'trials':len(rr)}
    for field in ['all_unanimity','honest_unanimity','honest_accuracy']:
     out[field]=boot([np.mean([r[field] for r in rr if r['task_id']==t]) for t in tids])
    for field in ['attacker_slots','target_hits','gold_hits','opposes_gold','differs_honest_majority','missing_attacker','missing_honest','request_errors']:
     out[field]=sum(r[field] for r in rr)
    for field in ['target_hits','gold_hits','opposes_gold','differs_honest_majority']:
     out[field+'_rate']=out[field]/out['attacker_slots'] if out['attacker_slots'] else None
    cells[f'{arm}|{f}|{rnd}']=out
 comparisons={}
 for other in ['forced','none']:
  for f in [1,2,3]:
   comparisons[f'prompted-minus-{other}|{f}']={}
   for field in ['all_unanimity','honest_unanimity','honest_accuracy']:
    ds=[]
    for t in sorted({r['task_id'] for r in scored}):
     mean=lambda arm: np.mean([r[field] for r in scored if r['arm']==arm and r['f']==f and r['task_id']==t and r['round']==2])
     ds.append(mean('prompted')-mean(other))
    comparisons[f'prompted-minus-{other}|{f}'][field]=boot(ds)
 failed_path=V5/'results/matched_prompted.transport_failed.jsonl'
 aborted_path=V5/'results/matched_prompted.transport_aborted.jsonl'
 transport_paths=[p for p in [failed_path,aborted_path] if p.exists()]
 failures=list(map(json.loads,failed_path.read_text().splitlines())) if failed_path.exists() else []
 aborted=list(map(json.loads,aborted_path.read_text().splitlines())) if aborted_path.exists() else []
 transport={'completed_failed_attempts':len(failures),'aborted_attempts':len(aborted),'failed_slots_in_completed_attempts':sum(s['error'] is not None for r in failures for rnd in r['history'] for s in rnd),'policy':'Transport-incomplete debates rerun in full; all attempts retained separately. Parsing failures never trigger reruns. Analysis uses one transport-complete debate per preselected task/f/rep cell.'}
 result={'transport_retry_audit':transport,'method':'Same task/f/repetition grid, model request name, temperature, token budget and protocol as archived arms; prompted collected later, so not contemporaneously randomized or paired random seeds. 5000 task-bootstrap resamples, seed 20260918; mean over two repeats within each of 24 tasks. Missing answers preclude unanimity, remain in attacker denominators; majority uses parseable honest labels and archive tie-break, no labels is incorrect. Exact target compliance is distinct from non-gold rate.','sources_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+transport_paths+[ROOT/'baseline/data/tasks.jsonl', ROOT/'baseline/code/scoring.py', Path(__file__)]},'cells':cells,'paired_task_differences_round2':comparisons,'per_trial_round':scored}
 (V5/'results/matched_prompted_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in cells.items() if k.endswith('|2')},indent=2))
if __name__=='__main__':analyse()
