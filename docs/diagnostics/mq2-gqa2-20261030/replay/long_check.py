#!/usr/bin/env python3
"""Independent streaming FP64 oracle over actual packed bytes; no Vulkan calls.
Record dequantization is FP32 (the stored representation semantics), all attention
math and explicit Sylvester-Hadamard transforms are FP64. Does not share decoder
or WHT implementation with original test model or fixture generator.
"""
import argparse,json,time,struct,sys
from pathlib import Path
import numpy as np
H=np.array([[1.]],dtype=np.float64)
for _ in range(8):H=np.block([[H,H],[H,-H]])
H/=16

def record(raw,off,value):
    packed=np.asarray(raw[off:off+8192]);quants=np.stack((packed&15,packed>>4),axis=1).reshape(128,128).astype(np.float32)
    axes=np.frombuffer(raw[off+8192:off+8960],dtype='<f2').astype(np.float32).reshape(3,128)
    decoded=(quants*axes[0,:,None]+axes[1,:,None])*axes[2,None,:]
    return decoded if value else decoded.T

def block(raw,meta,start,end,window,value):
    rbytes=meta['records_bytes'];tail_begin=window-min(window,meta['tail']);rows=np.arange(start,end);out=np.empty((end-start,4,256),np.float64)
    group=start//128;global_group=meta['origin']//128+group
    staged=global_group==0 or (group==(window-1)//128 and window%128!=0)
    if staged:
        slot=0 if global_group==0 else 1+((global_group-1)%2)
        stage=np.frombuffer(raw[rbytes:rbytes+384*2048],'<f2').reshape(384,4,256)
        out[:]=stage[slot*128+rows%128]
    else:
        for kv in range(4):
            for half in range(2):
                d=record(raw,(group*8+kv*2+half)*8960,value)
                out[:,kv,half*128:(half+1)*128]=d[rows%128]
    mask=rows>=tail_begin if meta['tail'] else np.zeros(len(rows),bool)
    if mask.any():
        tail=np.frombuffer(raw[rbytes+384*2048:],'<f2').reshape(256,4,256)
        out[mask]=tail[rows[mask]%256]
    return out,mask

def oracle(path,meta=None):
    path=Path(path);meta=meta or json.loads((path/'fixture.json').read_text());nq=meta['nq'];q=np.fromfile(path/'b0.bin','<f4')[:nq*24*256].reshape(nq,24,256).astype(np.float64);qr=q@H
    K=np.memmap(path/'b1.bin',dtype='u1',mode='r');V=np.memmap(path/'b2.bin',dtype='u1',mode='r');outputs=[];lses=[]
    for j,window in enumerate(meta['windows']):
        mx=np.full(24,-np.inf);den=np.zeros(24);num=np.zeros((24,256))
        for start in range(0,window,128):
            end=min(start+128,window);k,mask=block(K,meta,start,end,window,False);v,_=block(V,meta,start,end,window,True)
            if mask.any():v[mask]=v[mask]@H
            for kv in range(4):
                hs=slice(kv*6,(kv+1)*6);scores=k[:,kv]@qr[j,hs].T
                if mask.any():scores[mask]=k[mask,kv]@q[j,hs].T
                scores*=.0625;new=np.maximum(mx[hs],scores.max(axis=0));weights=np.exp(scores-new);old=np.exp(mx[hs]-new)
                num[hs]=num[hs]*old[:,None]+weights.T@v[:,kv];den[hs]=den[hs]*old+weights.sum(axis=0);mx[hs]=new
        outputs.append((num/den[:,None])@H if window else np.zeros((24,256)));lses.append(mx+np.log(den) if window else np.full(24,-np.inf))
    return np.concatenate([np.array(outputs).ravel(),np.array(lses).ravel()])

def check(path,outputs):
    path=Path(path);t=time.monotonic();meta=json.loads((path/'fixture.json').read_text());ref=oracle(path,meta);ref.astype('<f8').tofile(path/'oracle-fp64.bin');result={'oracle_seconds':time.monotonic()-t,'oracle_shape':list(ref.shape),'comparisons':{}}
    n=meta['nq']*24*256
    for f in outputs:
        arr=np.fromfile(f,'<f4');assert arr.shape==ref.shape,(f,arr.shape,ref.shape)
        finite=np.isfinite(ref);result['comparisons'][f]={'finite_expected_match':bool(np.array_equal(np.isfinite(arr),finite)),'O_maxabs':float(np.max(np.abs(arr[:n]-ref[:n]))),'LSE_maxabs_finite':float(np.max(np.abs(arr[n:][finite[n:]]-ref[n:][finite[n:]]))) if finite[n:].any() else 0,'negative_infinity_match':bool(np.array_equal(np.isneginf(arr),np.isneginf(ref)))}
    (path/'oracle-result.json').write_text(json.dumps(result,indent=2));return result
def prepare_reducer(attention,output):
    attention=Path(attention);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    params=struct.unpack('<5If6I',(attention/'b6.bin').read_bytes());nq,nh,sk=params[0],params[2],params[7]
    partials=np.fromfile(attention/'out4.bin','<f4');assert partials.size==sk*nq*nh*258 and np.isfinite(partials).all()
    for i in range(7):
        (output/f'b{i}.bin').write_bytes(partials.tobytes() if i==0 else bytes(nq*nh*257*4) if i==4 else struct.pack('<4I',nq,nh,256,sk) if i==6 else bytes(64))
    print(json.dumps({'reducer_input':str(output),'dispatch':[nh,nq,1],'split_k':sk}))

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--prepare-reducer':
        prepare_reducer(sys.argv[2],sys.argv[3]);sys.exit(0)
    p=argparse.ArgumentParser();p.add_argument('fixture');p.add_argument('outputs',nargs='*');a=p.parse_args();print(json.dumps(check(a.fixture,a.outputs),indent=2))
