"""Reproducible offline KEY truncation sensitivity and Figures 2/3 (2026-09-18)."""
from pathlib import Path
from collections import Counter
import sys,json,hashlib,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/bft-todo-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
V5=Path(__file__).resolve().parents[1]; ROOT=V5.parent
sys.path.insert(0,str(ROOT/'v3/code'))
from recompute_e1 import load_records,MODELS
from collision import A_unbiased
SOURCES={}
def read(rel):
 p=ROOT/rel;SOURCES[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def main():
 e=read('v3/results/e1_analysis.json'); ids=e['common_tasks']['ids'];by,_,_=load_records()
 for rel in ['results/samples.jsonl','results/samples2.jsonl','v3/code/scoring.py','v3/code/recompute_e1.py']:
  SOURCES[rel]=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
 result={'method':'Normalize KEY with the main scorer, then truncate to first L whitespace-delimited words. Same replies and 123 tasks at every L; missing KEY retained as None. Offline lexical coarsening, not generation-length intervention.','models':{}}
 for m in MODELS:
  rows=[];lengths=[]
  for tid in ids:
   rs=[r for r in by[m][tid] if r['has_answer']]; vals={}
   for L in [5,10,20,None]:
    labels=[(r['answer_class'],(' '.join(r['semantic_class'][1].split()[:L]) if r['has_key'] else None)) for r in rs]
    vals[str(L)]=A_unbiased(list(Counter(labels).values()),3)
   assert vals['5']+1e-14>=vals['10']>=vals['20']-1e-14 and vals['20']+1e-14>=vals['None']
   rows.append({'task':tid,**vals})
   lengths.extend(len(r['semantic_class'][1].split()) for r in rs if r['has_key'])
  means={L:float(np.mean([r[L] for r in rows])) for L in ['5','10','20','None']}
  assert abs(means['None']-e['summary_common_tasks'][m]['A_semantic_3'])<1e-6
  result['models'][m]={'n_tasks':len(ids),'usable_keys':len(lengths),'over_5':sum(x>5 for x in lengths),'over_10':sum(x>10 for x in lengths),'over_20':sum(x>20 for x in lengths),'mean_A3':means,'per_task':rows}
 pred=read('v3/results/prediction.json')['rows']; gain=read('v3/results/submission_debate_check.json')['by_n']; dissent=read('v3/results/debate_analysis.json')['sources']['e4_byz']; rev=read('v5/results/revision_analysis.json')
 plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.spines.top':False,'axes.spines.right':False,'legend.fontsize':6.5,'pdf.fonttype':42})
 colors=['#32658c','#b34f36','#39836e'];ns=[3,5,7,10]
 fig,axs=plt.subplots(1,3,figsize=(10,2.9),layout='constrained')
 for m,c in zip(MODELS,colors):
  s=e['summary_common_tasks'][m];hs=[2,3,5,7,10]
  for field,ls in [('verdict','-'),('semantic','--')]:
   axs[0].plot(hs,[s[f'A_{field}_{h}'] for h in hs],ls,color=c,marker='o',ms=3,label=m if field=='verdict' else None)
 axs[0].set(yscale='log',xlabel='Honest group size h',ylabel='Unanimity probability',title='(a) Activation by granularity');axs[0].legend(loc='center left');axs[0].text(.03,.04,'Solid: answer; dashed: answer+KEY',transform=axs[0].transAxes,fontsize=6.5)
 for i,m in enumerate(MODELS):
  s=e['summary_common_tasks'][m]
  axs[1].bar(i-.18,s['p_max_verdict'],.36,color=colors[0],label='Answer' if i==0 else None)
  axs[1].bar(i+.18,s['p_max_semantic'],.36,color=colors[1],label='Answer+KEY' if i==0 else None)
 axs[1].set(xticks=range(3),xticklabels=['4.1-nano','4o-mini','4.1-mini'],ylabel='Mean modal mass',ylim=(0,1.08),title='(b) Concentration (123 tasks)');axs[1].legend()
 for field,style,c in [('predicted','o-',colors[0]),('observed','s--',colors[1])]:
  axs[2].plot(ns,[pred[str(n)][field] for n in ns],style,color=c,label=field.capitalize(),ms=4)
 for n in ns:
  r=pred[str(n)];axs[2].annotate(f"{r['residual']:+.3f}",(n,r['observed']),xytext=(0,10),textcoords='offset points',ha='center',fontsize=7)
 axs[2].set(xticks=ns,ylim=(.70,.94),xlabel='Group size n',ylabel='Round-0 unanimity',title='(c) Nano: prediction and residuals');axs[2].legend(loc='upper right')
 for ext in ['pdf','png']:fig.savefig(V5/f'paper/fig/v5_activation.{ext}',dpi=220)
 plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(10,2.9),layout='constrained')
 for key,offset,c,label in [('agreement_gain',-.12,colors[0],'Agreement'),('accuracy_gain',.12,colors[1],'Accuracy')]:
  vals=[gain[str(n)][key] for n in ns]
  axs[0].errorbar(np.array(ns)+offset,[100*v['estimate'] for v in vals],yerr=[[100*(v['estimate']-v['lo']) for v in vals],[100*(v['hi']-v['estimate']) for v in vals]],fmt='o',color=c,capsize=3,label=label,ms=4)
 axs[0].axhline(0,color='.7',lw=1);axs[0].set(xticks=ns,xlabel='Group size n',ylabel='Gain (percentage points)',title='(a) Nano: interaction gains');axs[0].legend()
 for field,style,c,label in [('hon_agree','o-',colors[0],'Honest only'),('agree','s--',colors[1],'All participants')]:
  axs[1].plot(range(5),[dissent[f'10|0|{f}'][field] for f in range(5)],style,color=c,label=label,ms=4)
 axs[1].set(xticks=range(5),ylim=(-.04,1.06),xlabel='Prompted slots f',ylabel='Round-0 unanimity',title='(b) Nano: prompted dissent');axs[1].legend()
 # Recompute Opus panel from archived messages through the current scorer.
 from scoring import answer_class
 tasks={t['id']:t for t in map(json.loads,(ROOT/'data/tasks.jsonl').read_text().splitlines())}
 raw=ROOT/'v3/results/x3_attacks.samples.jsonl';SOURCES[str(raw.relative_to(ROOT))]=hashlib.sha256(raw.read_bytes()).hexdigest(); records=list(map(json.loads,raw.read_text().splitlines()))
 for strategy,honest,style,c,label in [('none',False,'o-',colors[2],'None: all'),('forced',True,'^--',colors[0],'Forced: honest'),('forced',False,'s-',colors[1],'Forced: all')]:
  ys=[]
  for f in [1,2,3]:
   rr=[r for r in records if r['strategy']==strategy and r['f']==f];assert len(rr)==48
   vv=[]
   for r in rr:
    labels=[answer_class(s['answer'],tasks[r['task_id']]['bench']) for s in r['history'][2]]
    if honest:labels=labels[f:]
    vv.append(None not in labels and len(set(labels))==1)
   ys.append(np.mean(vv))
  axs[2].plot([1,2,3],ys,style,color=c,label=label,ms=4)
 axs[2].set(xticks=[1,2,3],ylim=(-.06,1.09),xlabel='Nominal fault budget f',ylabel='Round-2 unanimity',title='(c) Opus: none / forced');axs[2].legend(loc='center')
 for ext in ['pdf','png']:fig.savefig(V5/f'paper/fig/v5_observability.{ext}',dpi=220)
 plt.close(fig)
 result['sources_sha256']=SOURCES
 (V5/'results/key_length_sensitivity.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({m:{k:v for k,v in d.items() if k!='per_task'} for m,d in result['models'].items()},indent=2))
if __name__=='__main__':main()
