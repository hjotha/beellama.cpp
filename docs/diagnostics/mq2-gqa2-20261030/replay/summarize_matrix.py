#!/usr/bin/env python3
import json,hashlib
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
result={'scope':'Synthetic active KV rows; not total model context or original LLM gate','logical_wave':128,'native_subgroup':4,'adapter':'logical128-3barrier.hlsl','worker_threads':4,'cases':{}}
for active in (102400,112640,130048):
 fixture=root/f'long-{active}';meta=json.loads((fixture/'fixture.json').read_text());n=meta['nq']*24*256
 case={'geometry':{k:meta[k] for k in ['active','origin','windows','nq','tail','seed','records_bytes']},'input_sha256':meta['hashes'],'runs':{},'comparisons':{}}
 arrays={};parts={}
 for label in ['mq-cap32-4t','mq-cap64-4t','mq-cap128-4t','gqa-cap64-4t']:
  d=fixture/label
  if not (d/'artifact-sha256.json').exists():case['runs'][label]={'state':'not completed'};continue
  run=json.loads((d/'run.json').read_text());oracle=json.loads((d/'oracle-result.json').read_text());metrics=next(iter(oracle['comparisons'].values()));arr=np.fromfile(d/'reduce/out4.bin','<f4');arrays[label]=arr;parts[label]=np.fromfile(d/'out4.bin','<u4')
  case['runs'][label]={'state':'completed','elapsed_s':run['elapsed_s'],'peak_rss_kib':run['maxrss_kib'],'threads':run.get('worker_threads',1),'oracle':metrics,'hashes':json.loads((d/'artifact-sha256.json').read_text())}
 if 'mq-cap64-4t' in arrays:
  ref=arrays['mq-cap64-4t']
  for label,a in arrays.items():
   if label=='mq-cap64-4t':continue
   case['comparisons'][label+'_vs_mq64']={'O_maxabs':float(np.max(abs(a[:n]-ref[:n]))),'LSE_maxabs':float(np.max(abs(a[n:]-ref[n:]))),'O_bitdiff':int(np.count_nonzero(a[:n].view('<u4')!=ref[:n].view('<u4'))),'LSE_bitdiff':int(np.count_nonzero(a[n:].view('<u4')!=ref[n:].view('<u4')))}
   if label=='gqa-cap64-4t':case['comparisons'][label+'_vs_mq64']['partial_bitdiff']=int(np.count_nonzero(parts[label]!=parts['mq-cap64-4t']))
 result['cases'][str(active)]=case
(root/'matrix-results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
