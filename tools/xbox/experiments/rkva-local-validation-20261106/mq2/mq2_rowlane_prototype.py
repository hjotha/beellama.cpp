#!/usr/bin/env python3
"""Local CPU FP32 MQ2 rowlane proposal, NOT shipped HLSL or a GPU emulator.

Extends the preserved nq1 rowlane model to 2-query tiles and local head groups
of six or two. Only full, commonly sealed 128-row groups use row-per-lane;
residual stage/record/exact-tail rows retain per-query classification. QK uses
one nibble load for all tile query/head scores, separate other per slice, and
fused accumulation. PV order is sequential per wave; two wave states merge,
then split states merge with the existing WHT256 output contract. All bytes
are loaded using the preserved bounded resident-plane loaders. Oracle builds
each serial image independently from canonical rows and evaluates FP64.
"""
from pathlib import Path
from output_paths import RESULTS
import argparse,hashlib,json,math,time
import numpy as np
import rkva_kvarn_serial_prefix_model as sp
import rkva_kvarn_serial_prefix_probe as probe
import rkva_decode_rowlane_model as rl
km=sp.km; F=np.float32; ROOT=Path(__file__).resolve().parent


def span(windows, origin, tail, begin, end, alignment=128):
    """Mirror shared-prefix selector, tightened from 8 to 128 for rowlane."""
    active=[w for w in windows if min(w,end)>begin]
    lo=(max(begin,128 if origin==0 else 0)+alignment-1)&~(alignment-1)
    hi=min([min(w,end,w&~127,w-min(w,tail)) for w in active],default=0)
    hi &= ~(alignment-1)
    return (lo,hi) if hi>lo else (begin,begin)


def kind(row, window, origin, tail):
    # Independent simple serial image predicate, not resident active_count.
    if row>=window:return 'none'
    if row>=window-min(window,tail):return 'tail'
    if origin+row<128 or (row//128==(window-1)//128 and window%128):return 'stage'
    return 'record'


def record_scores(P, qr, group, kv, wave, broken_shared_other=False):
    """qs=[tile_queries*local_heads,256], row-per-lane, same bytes as HLSL."""
    pos=wave*64+np.arange(64,dtype=np.uint64)
    score=np.zeros((len(qr),64),F); shared_other=None
    for sl in range(2):
        lo,hi=P.region('rec',group,kv*2+sl)
        dims=np.arange(128)
        sc=P.half_at(lo+8192+dims*2,lo,hi)
        ze=P.half_at(lo+8448+dims*2,lo,hi)
        qs=(qr[:,sl*128:(sl+1)*128]*sc).astype(F)
        zero=np.zeros(256,F);zero[sl*128:(sl+1)*128]=ze
        z=rl.wave_sum(rl.dot4(qr,zero))
        acc=np.zeros_like(score)
        for d in range(128):
            word=P.load(lo+d*64+(pos>>3)*4,lo,hi)
            nib=((word>>((pos&7)*4))&15).astype(F)
            acc=rl.mad(qs[:,d,None],nib,acc)
        other=P.half_at(lo+8704+pos*2,lo,hi)
        if shared_other is None:shared_other=other
        if broken_shared_other:other=shared_other
        score=rl.muladd(acc+z[:,None],other,score)
    return (score*F(sp.SCALE)).astype(F)


def run(res, q, positions, split, local_heads=6, broken_shared_other=False):
    K,V=sp.planes(res); nq=len(q); split_k,rows=split
    windows=[km.causal_limit(res.lay,res.live,int(p)) for p in positions]
    partials=[]; writes=np.zeros((split_k,nq,24),np.int32)
    for si in range(split_k):
        begin,end=si*rows,(si+1)*rows
        pm=np.full((nq,24),-km.FLT_MAX,F);pd=np.zeros((nq,24),F);pa=np.zeros((nq,24,256),F)
        for q0 in range(0,nq,2):
            ids=list(range(q0,min(q0+2,nq)));nw=[windows[j] for j in ids]
            lo,hi=span(nw,res.lay.origin,res.lay.ptt,begin,end)
            for gx in range(4*(6//local_heads)):
                kv,hb=divmod(gx,6//local_heads);hs=slice(kv*6+hb*local_heads,kv*6+(hb+1)*local_heads)
                qo=q[ids,hs].reshape(-1,256);qr=km.butterfly_wht(qo)
                hh=len(qo);states=[];covered=[[] for _ in ids]
                for wave in range(2):
                    mx=np.full(hh,-km.FLT_MAX,F);den=np.zeros(hh,F)
                    body=np.zeros((hh,256),F);tail=np.zeros((hh,256),F)
                    for t0 in range(lo,hi,128):
                        active_w=[w for w in nw if min(w,end)>begin]
                        K.window=V.window=min(active_w)
                        score=record_scores(K,qr,t0//128,kv,wave,broken_shared_other)
                        valid=np.repeat([min(w,end)>begin for w in nw],local_heads)
                        score=np.where(valid[:,None],score,-km.FLT_MAX)
                        nm=np.maximum(mx,score.max(axis=1))
                        ow=np.where(den==0,F(0),np.exp(mx-nm)).astype(F)
                        wt=np.where(valid[:,None],np.exp(score-nm[:,None]),F(0)).astype(F)
                        body*=ow[:,None];den=rl.muladd(den,ow,rl.wave_sum(wt));mx=nm
                        for t in range(64):
                            row=t0+wave*64+t
                            vv=km.opt_body4(V,True,row,kv,min(active_w))
                            body=rl.muladd(wt[:,t,None],vv,body)
                            for j,w in enumerate(nw):
                                if min(w,end)>begin:covered[j].append(row)
                    # Classify residual separately for each serial-prefix query.
                    for j,w in enumerate(nw):
                        K.window=V.window=w
                        ss=slice(j*local_heads,(j+1)*local_heads)
                        qe=min(end,w);tb=w-min(w,res.lay.ptt)
                        for sb,se,exact in ((begin,min(lo,qe,tb),False),(hi,min(qe,tb),False),(max(begin,tb),qe,True)):
                            for row in range(sb+wave,se,2):
                                if exact:
                                    kk,vv,qa=km.opt_tail4(K,row,kv),km.opt_tail4(V,row,kv),qo[ss]
                                else:
                                    kk,vv,qa=km.opt_body4(K,False,row,kv,w),km.opt_body4(V,True,row,kv,w),qr[ss]
                                score=(rl.wave_sum(rl.dot4(qa,kk))*F(sp.SCALE)).astype(F)
                                nm=np.maximum(mx[ss],score)
                                ow=np.where(den[ss]==0,F(0),np.exp(mx[ss]-nm)).astype(F)
                                wt=np.exp(score-nm).astype(F)
                                if exact:
                                    body[ss]*=ow[:,None];tail[ss]=rl.muladd(wt[:,None],vv,tail[ss]*ow[:,None])
                                else:body[ss]=rl.muladd(wt[:,None],vv,body[ss]*ow[:,None])
                                den[ss]=rl.muladd(den[ss],ow,wt);mx[ss]=nm
                                covered[j].append(row)
                    states.append((mx,den,np.concatenate((body,tail),axis=1)))
                for j,w in enumerate(nw):assert sorted(covered[j])==list(range(begin,min(end,w))),('coverage',j,w,begin,end)
                mm,md,both=km.merge_partial_states(states);bm,tm=both[:,:256],both[:,256:]
                acc=km.butterfly_wht(bm)+tm if split_k==1 else bm+km.butterfly_wht(tm)
                for j,qi in enumerate(ids):
                    ss=slice(j*local_heads,(j+1)*local_heads)
                    pm[qi,hs],pd[qi,hs],pa[qi,hs]=mm[ss],md[ss],acc[ss]
                    writes[si,qi,hs]+=1
        partials.append((pm.reshape(-1),pd.reshape(-1),pa.reshape(-1,256)))
    assert np.all(writes==1),'duplicated, missing or phantom partial writes'
    out,lse=km.reduce_partials(partials,rotated=split_k!=1)
    return out.reshape(nq,24,256),lse.reshape(nq,24)


def oracle(canon,res,q,positions):
    out=[];lse=[]
    for qi,pos in enumerate(positions):
        w=km.causal_limit(res.lay,res.live,int(pos))
        k,exact=sp.serial_sources(canon,res.lay,w,False);v,_=sp.serial_sources(canon,res.lay,w,True)
        o,l=km.oracle_attention(q[qi],k,v,exact,w,sp.SCALE);out.append(o);lse.append(l)
    return np.array(out),np.array(lse)


def geometry_tests():
    counts={'cases':0,'query_split_checks':0,'sealed_group_checks':0,'phantom_tiles':0}
    rng=np.random.default_rng(4242)
    for origin in (0,72704):
      for tail in (0,128):
       for live in (0,1,127,128,129,255,256,257,447,448,449,511,512,513,895,896,897,1025,28672):
        for nq in (2,3,5,8,9):
         for rows in (128,211,448,512,896,live or 1):
          # Sequential, non-monotonic, duplicated, empty and clipped windows.
          choices=[list(range(max(0,live-nq+1),live+1))[-nq:]]
          choices.append(rng.integers(0,live+1,nq).tolist())
          choices[0]=[0]*(nq-len(choices[0]))+choices[0]
          for windows in choices:
           for si in range(math.ceil(live/rows)+1):
            begin,end=si*rows,(si+1)*rows
            for q0 in range(0,nq,2):
             real=windows[q0:q0+2];phantom=real+[0]*(2-len(real))
             lo,hi=span(phantom,origin,tail,begin,end)
             if len(real)<2:counts['phantom_tiles']+=1
             for w in real:
              if min(w,end)>begin:
               for t0 in range(lo,hi,128):
                assert t0%128==0 and t0+128<=min(end,w)
                assert all(kind(r,w,origin,tail)=='record' for r in (t0,t0+127))
                counts['sealed_group_checks']+=1
              qe=min(end,w);tb=w-min(w,tail)
              intervals=[(begin,min(lo,qe,tb)),(hi,min(qe,tb)),(max(begin,tb),qe)]
              if min(w,end)>begin:intervals.append((lo,hi))
              lengths=sum(max(0,b-a) for a,b in intervals)
              assert lengths==max(0,qe-begin),(windows,si,intervals)
              valid=sorted((a,b) for a,b in intervals if b>a)
              assert all(valid[k][1]==valid[k+1][0] for k in range(len(valid)-1))
              counts['query_split_checks']+=1
           counts['cases']+=1
    counts['split_geometry_28672_nq2']={}
    for rows in (448,512):
        fast=0
        for si in range(math.ceil(28672/rows)):
            lo,hi=span([28671,28672],72704,128,si*rows,(si+1)*rows)
            fast+=hi-lo
        counts['split_geometry_28672_nq2'][str(rows)]={'splits':math.ceil(28672/rows),'shared_full_group_rows':fast,'first_query_residual_rows':28671-fast,'second_query_residual_rows':28672-fast}
    return counts


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--long',action='store_true');ap.add_argument('--contract',action='store_true');args=ap.parse_args()
    rl.CONTRACT=args.contract
    started=time.monotonic();rng=np.random.default_rng(20261106)
    records=[];neg=[];geo=geometry_tests()
    specs=[]
    for origin in (0,72704):
      for tail in (0,128):
        for live,nq in ((127,3),(128,2),(129,5),(255,9),(256,3),(257,9),(449,3),(513,5),(1025,3)):
          specs.append((origin,tail,live,nq,'boundary'))
    specs += [(72704,128,513,3,'empty_first'),(0,128,129,3,'unordered'),(72704,0,257,3,'unordered'),(72704,128,1025,5,'rewind')]
    if args.long:specs.append((72704,128,28672,3,'long'))
    for origin,tail,live,nq,label in specs:
        cap=max(live+128,2048);canon=sp.Canon(rng,cap);canon.append(live)
        res=sp.Resident(canon,km.Layout(origin,cap,tail,ring=256));res.reset(live-nq);res.append(live)
        if label=='rewind':
            canon.rewind(live-5);res.rewind(live-5);canon.append(5);res.append(live)
        positions=np.arange(origin+live-nq,origin+live,dtype=np.int64)
        if label=='empty_first':positions[0]=origin-1
        if label=='unordered':positions=positions[::-1].copy()
        q=rng.normal(size=(nq,24,256)).astype(F)
        want=oracle(canon,res,q,positions)
        splits=[(1,live)] if live<=129 else [(math.ceil(live/448),448),(math.ceil(live/512),512)]
        if label=='empty_first':splits.append((5,211))
        for sk,sr in splits:
            got=run(res,q,positions,(sk,sr),6)
            result=probe.compare(np,*got,*want);assert result['ok'],(label,live,nq,tail,result)
            rec={'label':label,'origin':origin,'tail':tail,'live':live,'nq':nq,'split_k':sk,'split_rows':sr,'local_heads':6,**result}
            records.append(rec);print(json.dumps(rec),flush=True)
            # Exercise identical MQ2 proposal with the GQA2 x-dispatch layout.
            if label!='long' and live in (129,257,513,1025):
                h2=run(res,q,positions,(sk,sr),2)
                assert np.array_equal(h2[0],got[0]) and np.array_equal(h2[1],got[1]),'GQA2 changes arithmetic/order'
                records.append({**rec,'local_heads':2,'bitwise_equal_to_heads6':True})
        if live==513 and nq==5 and origin==72704 and tail==128:
            bad=run(res,q,positions,splits[0],6,broken_shared_other=True)
            br=probe.compare(np,*bad,*want);assert not br['ok'],'negative control was not detected'
            neg.append({'mutation':'use first-slice other for both K slices','detected':True,**br})
    result={'success':True,'seed':20261106,'contract':args.contract,'cpu_seconds_not_gpu_performance':time.monotonic()-started,'numeric_runs':len(records),'numeric_cases':len(specs),'worst_out_max_abs':max(r['out_max_abs'] for r in records),'worst_lse_max_abs':max(r['lse_max_abs'] for r in records),'geometry':geo,'negative_controls':neg,'runs':records,'limits':['CPU FP32 proposal only; no new MQ2 rowlane HLSL exists','Not DXIL interpretation, compilation, Windows/UWP or GPU execution','GQA2 shader structural checks separately; numerical model uses proposed rowlane algorithm','Synthetic Bee-like records, not model-vocabulary/KLD or recurrent-state parity']}
    path=RESULTS/('mq2_results_contract.json' if args.contract else 'mq2_results.json')
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('runs',)}),flush=True)
if __name__=='__main__':main()
