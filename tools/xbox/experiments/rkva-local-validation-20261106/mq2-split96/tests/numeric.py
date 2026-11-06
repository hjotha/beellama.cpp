"""Synthetic split-reduction OUT/LSE test, not KVarN decoding or shader execution.
Actual C++ policy supplies geometry. Reuses preserved FP32 reducer/WHT and exact
existing tolerance gate. Independent oracle is dense float64 softmax. Scores
and values are synthetic post-decode inputs, so decoder correctness is NOT tested.
"""
from pathlib import Path
import json, sys
import numpy as np
import rkva_range_kvarn4_opt_model as km
from rkva_kvarn_serial_prefix_probe import compare
F=np.float32
ROOT=Path(__file__).resolve().parents[1]
geometry=np.loadtxt(ROOT/'results/geometry.csv',delimiter=',',dtype=np.int64)
rng=np.random.default_rng(20261106)
runs=[]; negatives=[]
def oracle(s,v,windows,rotated):
 out=np.zeros((len(windows),256),np.float64);lse=np.full(len(windows),-np.inf)
 for j,w in enumerate(windows):
  if w:
   score=s[j,:w].astype(np.float64);m=score.max();p=np.exp(score-m);d=p.sum()
   out[j]=p@v[:w].astype(np.float64)/d;lse[j]=m+np.log(d)
 if rotated:out=out@hadamard()/16
 return out,lse
def hadamard():
 h=np.ones((1,1),np.float64)
 for _ in range(8):h=np.block([[h,h],[h,-h]])
 return h
def run(s,v,windows,k,rows,rotated,mutate=None):
 parts=[]
 for i in range(k):
  mx=np.full(len(windows),-km.FLT_MAX,F);den=np.zeros(len(windows),F);acc=np.zeros((len(windows),256),F)
  for j,w in enumerate(windows):
   a=i*rows;b=min((i+1)*rows,int(w))
   if b>a:
    x=s[j,a:b];mx[j]=x.max();p=np.exp(x-mx[j]).astype(F);den[j]=p.sum(dtype=F);acc[j]=p@v[a:b]
  parts.append((mx,den,acc))
 if mutate=='omit-final':parts[-1]=(parts[-1][0],np.zeros_like(parts[-1][1]),np.zeros_like(parts[-1][2]))
 if mutate=='nan':parts[-1][2][0,0]=np.nan
 return km.reduce_partials(parts,rotated=rotated)
for n,ak,ar,bk,br in geometry:
 for flavor in ('random','last-dominant'):
  nq=9 if n in (257,24577) else 3
  windows=np.array([n]*nq);windows[0]=0 if n!=0 else n
  if nq>2:windows[1]=max(0,n-1)
  s=rng.normal(0,2,(nq,n)).astype(F);v=rng.normal(0,1,(n,256)).astype(F)
  if n and flavor=='last-dominant':s[:,-1]=30;v[-1]=F(2)
  for rotated in (False,True):
   want=oracle(s,v,windows,rotated)
   a=run(s,v,windows,ak,ar,rotated);b=run(s,v,windows,bk,br,rotated)
   ac=compare(np,*a,*want);bc=compare(np,*b,*want);ab=compare(np,*b,*a)
   assert ac['ok'] and bc['ok'] and ab['ok'],(n,flavor,rotated,ac,bc,ab)
   runs.append(dict(rows=int(n),nq=nq,flavor=flavor,rotated=rotated,baseline=ac,candidate=bc,ab=ab,bit_exact=bool(np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1]))))
   if n==49920 and flavor=='last-dominant':
    bad=run(s,v,windows,bk,br,rotated,'omit-final');r=compare(np,*bad,*want);assert not r['ok'];negatives.append(dict(mutation='omit last populated split',detected=True,rotated=rotated,gate=r))
    # Poison a participating head, not an empty query.
    badout=b[0].copy();badout[-1,0]=np.nan;r=compare(np,badout,b[1],*want);assert not r['ok'];negatives.append(dict(mutation='nonfinite output',detected=True,rotated=rotated,gate=r))
    badlse=b[1].copy();badlse[-1]+=F(.1);r=compare(np,b[0],badlse,*want);assert not r['ok'];negatives.append(dict(mutation='LSE bias +0.1',detected=True,rotated=rotated,gate=r))
result=dict(status='PASS_CPU_SYNTHETIC_REDUCTION_ONLY',seed=20261106,runs=runs,negative_controls=negatives,tolerances={'OUT':'3e-4 + 3e-4 * abs(reference)','LSE':'3e-4 + 3e-5 * abs(reference)'},limits=['No decoded KVarN bytes, QK kernel or serial resident engine execution','No Windows/MSVC/DXIL/PSO/GPU/replay/hardware/performance qualification'])
(ROOT/'results/numeric.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],cases=len(runs),negative_controls=len(negatives),worst_out=max(r['candidate']['out_max_abs'] for r in runs),worst_lse=max(r['candidate']['lse_max_abs'] for r in runs),worst_ab_out=max(r['ab']['out_max_abs'] for r in runs),worst_ab_lse=max(r['ab']['lse_max_abs'] for r in runs),non_bit_exact_cases=sum(not r['bit_exact'] for r in runs))))
