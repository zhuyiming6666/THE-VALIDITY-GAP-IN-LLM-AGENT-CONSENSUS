"""Complete the missing prompted arm on the exact archived task/rep/f grid.
Resume retains completed debates. Failures remain in transcripts and denominators.
"""
from pathlib import Path
import sys,json,concurrent.futures as cf,hashlib
V5=Path(__file__).resolve().parents[1];ROOT=V5.parent
sys.path.insert(0,str(ROOT/'v3/code'))
from sampling import Sampler
from exp_attacks import run_debate,wrong_answer

def main():
 old=ROOT/'v3/results/x3_attacks.samples.jsonl'
 records=list(map(json.loads,old.read_text().splitlines()))
 jobs=[r for r in records if r['strategy']=='forced'];assert len(jobs)==144
 tasks={t['id']:t for t in map(json.loads,(ROOT/'data/tasks.jsonl').read_text().splitlines())}
 out=V5/'results/matched_prompted.samples.jsonl'
 key=lambda r:(r['task_id'],r['f'],r['rep'])
 done={key(r) for r in map(json.loads,out.read_text().splitlines())} if out.exists() else set()
 def run(r):
  sampler=Sampler(r['model'],max_tokens=700,temperature=1.0,timeout=180)
  assert sampler.config==r['config_hash'],(sampler.config,r['config_hash'])
  task=tasks[r['task_id']].copy()
  target=wrong_answer(task['answer'],task['bench'])
  assert str(target)==str(r['history'][0][0]['answer'])
  hist=run_debate(sampler,task,10,r['f'],'prompted',3,task['bench'])
  return {k:r[k] for k in ['task_id','bench','model','n','f','rep','config_hash']}|{'strategy':'prompted','target':target,'history':hist,'usage':sampler.usage,'baseline_sha256':hashlib.sha256(old.read_bytes()).hexdigest()}
 with out.open('a') as fh,cf.ThreadPoolExecutor(max_workers=6) as pool:
  pending=[pool.submit(run,r) for r in jobs if key(r) not in done]
  for f in cf.as_completed(pending):
   r=f.result();fh.write(json.dumps(r)+'\n');fh.flush();done.add(key(r))
   print(f"Completed {len(done)}/144 debates; latest errors={r['usage']['errors']}",flush=True)
if __name__=='__main__':main()
