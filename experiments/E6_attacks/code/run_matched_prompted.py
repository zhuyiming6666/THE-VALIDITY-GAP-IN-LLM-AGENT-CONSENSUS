"""Complete the missing prompted arm on the archived task/rep/f grid.
Only transport-incomplete debates are retried in full; model parsing failures
remain valid observations. The runner stops dispatch on a transport failure.
"""
from pathlib import Path
import sys,json,concurrent.futures as cf,hashlib
V5=Path(__file__).resolve().parents[1];ROOT=V5.parent
sys.path.insert(0,str(ROOT/'v3/code'))
from sampling import Sampler
from exp_attacks import run_debate,wrong_answer

class AuditedSampler(Sampler):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.audit=[]
 def complete(self,system,user,**kwargs):
  result=super().complete(system,user,**kwargs)
  self.audit.append(result)
  if result['error']:
   raise RuntimeError(result['error'])
  return result

def main():
 old=ROOT/'v3/results/x3_attacks.samples.jsonl'
 records=list(map(json.loads,old.read_text().splitlines()))
 jobs=[r for r in records if r['strategy']=='forced'];assert len(jobs)==144
 tasks={t['id']:t for t in map(json.loads,(ROOT/'data/tasks.jsonl').read_text().splitlines())}
 out=V5/'results/matched_prompted.samples.jsonl'
 key=lambda r:(r['task_id'],r['f'],r['rep'])
 existing=list(map(json.loads,out.read_text().splitlines())) if out.exists() else []
 assert all(not s['error'] for r in existing for rnd in r['history'] for s in rnd)
 done={key(r) for r in existing}
 def run(r):
  sampler=AuditedSampler(r['model'],max_tokens=700,temperature=1.0,timeout=60)
  assert sampler.config==r['config_hash']
  task=tasks[r['task_id']].copy();target=wrong_answer(task['answer'],task['bench'])
  assert str(target)==str(r['history'][0][0]['answer'])
  meta={k:r[k] for k in ['task_id','bench','model','n','f','rep','config_hash']}|{'strategy':'prompted','target':target,'baseline_sha256':hashlib.sha256(old.read_bytes()).hexdigest()}
  try:
   hist=run_debate(sampler,task,10,r['f'],'prompted',3,task['bench'])
   return meta|{'history':hist,'usage':sampler.usage}
  except RuntimeError as e:
   return meta|{'transport_abort':str(e),'calls_before_abort':sampler.audit,'usage':sampler.usage}
 todo=iter(r for r in jobs if key(r) not in done)
 stopped=False
 with out.open('a') as fh,cf.ThreadPoolExecutor(max_workers=3) as pool:
  pending={pool.submit(run,r) for r in [next(todo,None) for _ in range(3)] if r is not None}
  while pending:
   finished,pending=cf.wait(pending,return_when=cf.FIRST_COMPLETED)
   for f in finished:
    r=f.result()
    if 'transport_abort' in r:
     with (V5/'results/matched_prompted.transport_aborted.jsonl').open('a') as af:af.write(json.dumps(r)+'\n')
     stopped=True;print('Transport failure: stopping dispatch; audit saved.',flush=True)
    else:
     fh.write(json.dumps(r)+'\n');fh.flush();done.add(key(r))
     print(f"Completed {len(done)}/144 debates; latest retries={r['usage']['errors']}",flush=True)
   if not stopped:
    while len(pending)<3:
     r=next(todo,None)
     if r is None:break
     pending.add(pool.submit(run,r))
 if stopped:raise SystemExit('Transport interruption: resume after endpoint recovery.')
if __name__=='__main__':main()
