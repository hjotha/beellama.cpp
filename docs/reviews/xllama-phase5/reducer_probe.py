import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/hjotha/xllama-qwen35-xbox/tests')
import rkva_range_kvarn4_opt_model as model

F = np.float32
rng = np.random.default_rng(5005)

def h64(x):
    t = np.array(x, np.float64).reshape(-1, 256).copy()
    lane = np.arange(128)
    for stride in (1,2,4,8,16,32,64):
        a,b = t[:,lane].copy(), t[:,lane^stride].copy()
        c,d = t[:,lane+128].copy(), t[:,(lane^stride)+128].copy()
        upper = (lane&stride)!=0
        t[:,lane] = np.where(upper,b-a,a+b)
        t[:,lane+128] = np.where(upper,d-c,c+d)
    a,b = t[:,:128]/np.sqrt(128.0), t[:,128:]/np.sqrt(128.0)
    return np.concatenate([(a+b)/np.sqrt(2.0),(a-b)/np.sqrt(2.0)],axis=1).reshape(np.shape(x))

def reduce(states, transformed=True):
    valid = [s for s in states if s[1]>0]
    maximum = max([s[0] for s in valid], default=-np.finfo(np.float32).max)
    den = F(0)
    acc = np.zeros(256,np.float32)
    for maximum_s, den_s, numerator in valid:
        w = F(np.exp(F(maximum_s-maximum)))
        den = F(den + F(den_s*w))
        acc = (acc + numerator*w).astype(np.float32)
    inv = F(1)/den if den>0 else F(0)
    output = model.butterfly_wht(acc*inv) if transformed else acc*inv
    lse = F(maximum + F(np.log(den))) if den>0 else F(-np.inf)
    return output, lse

def state(scores, rotated_values, original_tail, is_tail):
    if len(scores)==0:
        return (F(-np.finfo(np.float32).max),F(0),np.full(256,np.nan,np.float32)), np.zeros(256,np.float32)
    maximum=F(np.max(scores)); den=F(0)
    body=np.zeros(256,np.float32); tail=np.zeros(256,np.float32)
    for score,v,t,flag in zip(scores,rotated_values,original_tail,is_tail):
        weight=F(np.exp(F(score-maximum)))
        den=F(den+weight)
        if flag: tail=(tail+weight*t).astype(np.float32)
        else: body=(body+weight*v).astype(np.float32)
    return (maximum,den,(body+model.butterfly_wht(tail)).astype(np.float32)), (model.butterfly_wht(body)+tail).astype(np.float32)

worst_out=worst_lse=worst_large_lse=0.0
case_count=0
for splits in (1,2,3,32,64,128,256):
    for scenario in ('body','tail','mixed','empty-final','score-spread','large-negative','large-positive'):
        n=splits*3
        if scenario=='empty-final': n=max(1,n-4)
        scores=rng.normal(0,2,n).astype(np.float32)
        if scenario=='score-spread': scores=np.linspace(-120,0,n,dtype=np.float32)
        if scenario=='large-negative': scores.fill(F(-2097152))
        if scenario=='large-positive': scores.fill(F(2097152))
        vr=rng.normal(size=(n,256)).astype(np.float32)
        vt=rng.normal(size=(n,256)).astype(np.float32)
        tail=np.zeros(n,bool)
        if scenario=='tail': tail[:]=True
        elif scenario not in ('body',): tail[max(0,n-128):]=True
        original=np.where(tail[:,None],vt.astype(np.float64),h64(vr))
        maximum64=np.max(scores.astype(np.float64))
        weights64=np.exp(scores.astype(np.float64)-maximum64)
        oracle=np.sum(original*weights64[:,None],axis=0)/np.sum(weights64)
        oracle_lse=maximum64+np.log(np.sum(weights64))
        states=[]; direct=None
        for s in range(splits):
            begin=min(3*s,n); end=min(begin+3,n)
            st,direct_num=state(scores[begin:end],vr[begin:end],vt[begin:end],tail[begin:end])
            states.append(st)
            if splits==1: direct=direct_num/st[1]
        out,lse=reduce(states)
        _,lse_without_wht=reduce(states, transformed=False)
        assert np.array_equal(np.asarray(lse).view(np.uint32),np.asarray(lse_without_wht).view(np.uint32))
        if splits==1:
            assert np.max(np.abs(direct.astype(np.float64)-oracle))<3e-5
        error=float(np.max(np.abs(out.astype(np.float64)-oracle)))
        lerror=abs(float(lse)-float(oracle_lse))
        worst_out=max(worst_out,error)
        if scenario.startswith('large'): worst_large_lse=max(worst_large_lse,lerror)
        else: worst_lse=max(worst_lse,lerror)
        assert np.all(np.isfinite(out)) and np.isfinite(lse)
        assert error<3e-5
        assert lerror<(0.13 if scenario.startswith('large') else 3e-5)
        case_count+=1
print(f'Finite reducer: {case_count} cases S=1,2,3,32,64,128,256; max_abs_O={worst_out:.9e}; max_abs_LSE_regular={worst_lse:.9e}; max_abs_LSE_large_offset={worst_large_lse:.9e}')
print('LSE unchanged bitwise between the two reducer branches in the emulation')

zero=np.zeros(256,np.float32)
empty=(F(-np.finfo(np.float32).max),F(0),np.full(256,np.nan,np.float32))
for s in (1,64,256):
    out,lse=reduce([empty]*s)
    assert np.array_equal(out,zero) and np.isneginf(lse)
print('All-empty S=1,64,256: O=0, LSE=-Inf; empty numerator poison not consumed')

poison=(F(0),F(np.nan),np.full(256,np.nan,np.float32))
valid=(F(0),F(1),np.ones(256,np.float32))
out,lse=reduce([valid,poison])
assert np.all(np.isfinite(out)) and lse==0
print('NaN denominator: invalid state silently discarded; surviving OUT/LSE finite')

# Positive denominator summation has no cancellation; vector numerators do.
states=[]
for value in (1e8,1,-1e8):
    numerator=np.zeros(256,np.float32); numerator[0]=F(value)*F(128)
    states.append((F(0),F(128),numerator))
out,lse=reduce(states)
reference=h64(np.eye(256)[0]/3)
assert out[0]==0 and abs(reference[0]-1/48)<1e-15
print(f'F32 cancellation: rotated split means=(1e8,1,-1e8), d_s=128; OUT[0]={out[0]}, exact={reference[0]:.9f}; max_abs_error={np.max(np.abs(out-reference)):.9f}')

with np.errstate(over='ignore',invalid='ignore'):
    score=F(np.finfo(np.float32).max)*F(2)
    bad_den=F(np.exp(F(score-score)))
assert np.isinf(score) and np.isnan(bad_den)
print('Finite extreme F32 operands: score overflow=Inf, exp(Inf-Inf)=NaN')

# One-key dot product achievable with FP8 Q and original-F16 precision tail.
q=F(4096); key=F(np.float16(-8192)); scale=F(1/16)
score=F(F(q*key)*scale)
assert score==F(-2097152)
print(f'FP8/F16 finite score example: Q={q:g}, K={key:g}, scale={scale:g}, score=LSE={score:g}')
