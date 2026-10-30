#!/usr/bin/env python3
"""Build native-guard and labelled logical-wave diagnostics from explicit repos.
No downloading, source editing, commits, installation, host changes or deployment.
"""
import argparse,subprocess,hashlib,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--mq-repo',type=Path,required=True);p.add_argument('--gqa-repo',type=Path,required=True);p.add_argument('--dxc',type=Path,required=True);p.add_argument('--vulkan-include',type=Path,required=True);a=p.parse_args();root=Path(__file__).resolve().parent;os.chdir(root)
paths={n:r.resolve()/'tools/xbox/shaders/rkva' for n,r in [('mq',a.mq_repo),('gqa',a.gqa_repo)]}
expected={'mq':'e52f25fb5fd50e1d35f05a4ca5923b7011f16ffa7aa77bc15c87af6e074540f2','gqa':'f8f5f6b99785c2fba17d95fcb052433065e306ec4782f165087fb2c4fbcc1083'}
for name,s in paths.items():
 actual=hashlib.sha256((s/'rkva_range_attn_kvarn4_mq.hlsl').read_bytes()).hexdigest()
 if actual!=expected[name]:raise SystemExit(f'{name}: unexpected source SHA256 {actual}; refusing unpinned source')
flags=[str(a.dxc.resolve()),'-spirv','-T','cs_6_0','-fspv-target-env=vulkan1.2','-fvk-t-shift','0','0','-fvk-u-shift','4','0','-fvk-b-shift','6','0']
for name,source in [('mq',paths['gqa']/'rkva_range_attn_kvarn4_mq.hlsl'),('mq_gqa2',paths['gqa']/'rkva_range_attn_kvarn4_mq_gqa2.hlsl'),('mq_cap_source',paths['mq']/'rkva_range_attn_kvarn4_mq.hlsl')]:
 subprocess.run(flags+['-E','CSRangeAttnKvarn4Mq',str(source),'-Fo',name+'.spv'],check=True)
 for suffix,adapter in [('', 'logical128.hlsl'),('-3barrier','logical128-3barrier.hlsl')]:
  target='logical128-'+name+suffix;wrapper=root/(target+'.hlsl');wrapper.write_text(f'#include "{adapter}"\n#include "{source}"\n')
  subprocess.run(flags+['-E','CSRangeAttnKvarn4Mq',str(wrapper),'-Fo',target+'.spv'],check=True)
subprocess.run(flags+['-E','CSRangeReduceOpt','-D','RKVA_REDUCE_WHT=1',str(paths['gqa']/'rkva_range_reduce_opt.hlsl'),'-Fo','reduce.spv'],check=True)
for name in ['smoke','logical128-smoke','logical128-smoke-3barrier']:
 subprocess.run(flags+['-E','main',name+'.hlsl','-Fo',name+'.spv'],check=True)
subprocess.run(['g++','-std=c++17','-O2','-I'+str(a.vulkan_include.resolve()),'replay.cpp','-l:libvulkan.so.1','-o','replay'],check=True)
(root/'portable-build-sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.spv')},indent=2))
