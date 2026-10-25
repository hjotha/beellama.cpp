#!/usr/bin/env python3
"""Future native KVarN4 GPU gate. Network access requires explicit --execute.

Do not run while another host owns the console's single RKVA connection.
Fixtures must come from current Bee's llama-kvarn.cpp CPU oracle.
"""
import argparse
import json
import struct
from pathlib import Path
import numpy as np
from rkva_range_probe import Client, ORIGIN, H, HKV, D
from rkva_kvarn_range_fixture import decode, wht


def run(host, port, fixtures):
    raw = (fixtures/'bee-kvarn4-records.bin').read_bytes()
    source = np.fromfile(fixtures/'bee-kvarn4-source.bin', dtype='<f4').reshape(2,128,128)
    kd, vd = decode(raw[:8960]), decode(raw[8960:])
    kr = np.tile(source[0].T, (1,2)).astype(np.float16).astype(np.float32)
    vr = np.tile(source[1], (1,2)).astype(np.float16).astype(np.float32)
    ko, vo = wht(np.tile(source[0].T,(1,2))), wht(np.tile(source[1],(1,2)))
    ko, vo = ko.astype('<f2'), vo.astype('<f2')
    rng = np.random.default_rng(260)
    capacity = 1025
    descriptor = [ORIGIN,capacity,4,1,128,128,256,2,4,4,8960,3,2,128,128,1,1,1]
    client = Client(host,port)
    checks = []
    record_stride = HKV*2*8960
    def plane(rows, rotated, value=False):
        if not len(rows): return b''
        data = (vr if value else kr) if rotated else (vo if value else ko)
        return np.repeat(data[np.asarray(rows)%128,None,:],HKV,axis=1).astype('<f2').tobytes()
    def upload(live, previous=0, reset=False, success=True):
        groups = live//128
        first = 0 if reset else previous//128
        new_groups = groups-first
        header = struct.pack('<8I',live,first,new_groups,int(reset),0 if reset else previous,
                             0 if reset else live-previous,0,0)
        records = raw[:8960]*(HKV*2*new_groups) + raw[8960:]*(HKV*2*new_groups)
        if reset:
            kstage=np.zeros((384,HKV,D),dtype='<f2');vstage=kstage.copy()
            for token in range(live):
                group=(ORIGIN+token)//128;slot=0 if group==0 else 1+(group-1)%2
                kstage[slot*128+token%128]=kr[token%128]
                vstage[slot*128+token%128]=vr[token%128]
            stage=kstage.tobytes()+vstage.tobytes()
            tailrows=np.arange(max(0,live-128),live)
        else:
            rows=np.arange(previous,live)
            stage=plane(rows,True)+plane(rows,True,True)
            tailrows=rows
        payload=header+records+stage+plane(tailrows,False)+plane(tailrows,False,True)
        return client.call(16,payload,session=0,success=success)
    def check(live,name):
        positions=[ORIGIN-1,ORIGIN,ORIGIN+1,ORIGIN+live-1]
        q=rng.normal(0,.1,(len(positions),H,D)).astype(np.float32)
        actual,lse=client.attention(q.ravel().tolist(),positions)
        actual=np.asarray(actual).reshape(len(positions),H,D)
        lse=np.asarray(lse).reshape(len(positions),H)
        k=np.empty((live,D),np.float32);v=np.empty_like(k)
        for token in range(live):
            complete=token//128 < live//128
            k[token]=np.tile(kd[:,token%128],2) if complete else kr[token%128]
            v[token]=np.tile(vd[token%128],2) if complete else vr[token%128]
        k,v=wht(k),wht(v)
        tail=min(128,live)
        k[-tail:]=ko[np.arange(live-tail,live)%128].astype(np.float32)
        v[-tail:]=vo[np.arange(live-tail,live)%128].astype(np.float32)
        maxout=maxlse=0.
        for qi,pos in enumerate(positions):
            limit=max(0,min(live,pos-ORIGIN+1))
            if not limit:
                assert np.all(actual[qi]==0) and np.all(lse[qi]==-np.inf)
                continue
            scores=q[qi].astype(np.float64)@k[:limit].astype(np.float64).T/16
            m=scores.max(axis=1)
            weights=np.exp(scores-m[:,None]);denom=weights.sum(axis=1)
            expected=weights@v[:limit].astype(np.float64)/denom[:,None]
            expected_lse=m+np.log(denom)
            assert np.all(np.isfinite(actual[qi])) and np.all(np.isfinite(lse[qi]))
            maxout=max(maxout,float(np.max(np.abs(actual[qi]-expected))))
            maxlse=max(maxlse,float(np.max(np.abs(lse[qi]-expected_lse))))
        assert maxout<3e-4 and maxlse<3e-4,(name,maxout,maxlse)
        checks.append(dict(name=name,live=live,out_max_abs=maxout,lse_max_abs=maxlse))
    try:
        client.hello(capacity)
        assert client.capabilities&4,'native KVarN4 range capability missing'
        client.call(2,struct.pack('<4I',0,0,capacity,1))
        client.call(15,struct.pack('<18I',*descriptor),session=0)
        bad=descriptor.copy();bad[-1]=0 # delayed records must fail closed
        client.call(15,struct.pack('<18I',*bad),session=0,success=False)
        previous=0
        for live in (1,127,128,129,255,256,257,383,384,385,511,512,513,641,769,1025):
            upload(live,previous,reset=previous==0)
            check(live,'append-'+str(live))
            previous=live
        client.call(7,struct.pack('<i',ORIGIN+513),session=0)
        # A rewind invalidates native stage/tail. Old future snapshots must not be read.
        q=[0.]*(H*D)
        payload=struct.pack('<4Ii',1,0,0,0,ORIGIN+512)+struct.pack('<'+'f'*len(q),*q)
        client.call(14,payload,session=0,nq=1,success=False)
        upload(514,513,success=False)
        upload(513,reset=True);check(513,'rewind-full-resync')
        client.call(6,session=0)
        client.call(14,payload,session=0,nq=1,success=False)
        upload(129,reset=True);check(129,'reset-full-resync')
        client.call(3,session=0)
    finally:
        client.socket.close()
    return dict(success=True,storage='kvarn4-native',checks=checks,timings=client.timings)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true')
    p.add_argument('--host');p.add_argument('--port',type=int,default=50054)
    p.add_argument('--fixtures',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args()
    if not args.execute:
        print('No network access. Pass --execute --host HOST --fixtures DIR after reserving the console.')
    else:
        if not args.host or not args.fixtures:p.error('--execute requires --host and --fixtures')
        result=run(args.host,args.port,args.fixtures)
        text=json.dumps(result,indent=2);print(text)
        if args.output:args.output.write_text(text+'\n')
