#!/usr/bin/env python3
"""Synthetic packed KVarN fixtures; no Vulkan execution, no original-source edits.
Physical inputs are shared byte-for-byte between policies. Stage/tail are fully
initialized synthetic slots, NOT a capture of a temporally evolving model cache.
Windows near the active end exercise causal seams; arbitrary windows remain a
physical-descriptor test and do not assert historical-cache consistency.
"""
import argparse, hashlib, json, struct, time
from pathlib import Path
import numpy as np
RECORD=8960; ROW=2048; STAGE=384*ROW

def wht(x):
    x=np.array(x,copy=True)
    for n in (1,2,4,8,16,32,64,128):
        y=x.reshape(*x.shape[:-1],-1,2*n); a=y[...,:n].copy(); b=y[...,n:].copy()
        y[...,:n]=a+b; y[...,n:]=a-b
    return x/16

def policy(active,cap):
    k=max(1,min(cap,(active+255)//256)); rows=((active+k-1)//k+7)//8*8
    return (max(1,(active+rows-1)//rows),rows) if rows else (1,0)

def generate(out,active,windows,caps=(32,64,128),origin=128,tail=128,extra=0,seed=20261030):
    t=time.monotonic();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    assert active>0 and origin%128==0 and tail in (0,128) and 2<=len(windows)<=9
    assert all(0<=w<=active for w in windows)
    rng=np.random.default_rng(seed);nq=len(windows);records=(active+127)//128*8*RECORD
    q=rng.normal(0,.1,(nq,24,256)).astype('<f4')
    np.concatenate([q.ravel(),wht(q).astype('<f4').ravel()]).tofile(out/'b0.bin')
    for binding in (1,2):
        with (out/f'b{binding}.bin').open('wb') as f:
            for _ in range((active+127)//128*8):
                f.write(rng.integers(0,256,8192,dtype=np.uint8).tobytes())
                f.write((rng.uniform(.01,.12,128)*rng.choice([-1,1],128)).astype('<f2').tobytes())
                f.write(rng.uniform(-.5,.5,128).astype('<f2').tobytes())
                f.write((rng.uniform(.5,1.5,128)*rng.choice([-1,1],128)).astype('<f2').tobytes())
            f.write(rng.normal(0,.5,(384+(256 if tail else 0),4,256)).astype('<f2').tobytes())
    np.array([origin+w-1 for w in windows],dtype='<i4').tofile(out/'b3.bin')
    hashes={}
    for i in range(4):
        digest=hashlib.sha256()
        with (out/f'b{i}.bin').open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
        hashes[f'b{i}']=digest.hexdigest()
    specs=[]
    for cap in caps:
        splits,rows=policy(active,cap);splits+=extra;p=out/f'cap{cap}';p.mkdir(exist_ok=True)
        for i in range(4):
            target=p/f'b{i}.bin'
            if not target.exists():target.symlink_to(f'../b{i}.bin')
        (p/'b4.bin').write_bytes(bytes(splits*nq*24*258*4));(p/'b5.bin').write_bytes(bytes(nq*24*257*4))
        (p/'b6.bin').write_bytes(struct.pack('<5If6I',nq,active,24,4,256,.0625,origin,splits,rows,0,records,tail))
        specs.append(dict(cap=cap,split_k=splits,split_rows=rows,path=str(p),dispatch_mq=[4,(nq+1)//2,splits],dispatch_gqa2=[12,(nq+1)//2,splits]))
    meta=dict(active=active,windows=windows,origin=origin,tail=tail,nq=nq,records_bytes=records,seed=seed,hashes=hashes,policies=specs,synthetic=True,elapsed_seconds=time.monotonic()-t)
    (out/'fixture.json').write_text(json.dumps(meta,indent=2));return meta
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--active',type=int,required=True);p.add_argument('--windows',help='comma separated causal counts, default active-1,active');p.add_argument('--caps',default='32,64,128');p.add_argument('--origin',type=int,default=128);p.add_argument('--tail',type=int,default=128);p.add_argument('--extra-empty-splits',type=int,default=0)
    a=p.parse_args();print(json.dumps(generate(a.out,a.active,list(map(int,a.windows.split(','))) if a.windows else [a.active-1,a.active],tuple(map(int,a.caps.split(','))),a.origin,a.tail,a.extra_empty_splits),indent=2))
