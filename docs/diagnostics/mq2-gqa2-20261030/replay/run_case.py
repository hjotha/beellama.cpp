#!/usr/bin/env python3
"""Explicit software logical128 diagnostic; not native subgroup128/Xbox parity."""
import os,sys,json,struct,subprocess,time,resource,hashlib
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent;os.chdir(root)
fixture,cap,arm,label=sys.argv[1:];src=root/fixture/('cap'+cap);out=root/fixture/label;out.mkdir(exist_ok=False)
for i in range(7):
 if i<4:(out/f'b{i}.bin').symlink_to((src/f'b{i}.bin').resolve())
 else:(out/f'b{i}.bin').write_bytes((src/f'b{i}.bin').read_bytes())
params=struct.unpack('<5If6I',(out/'b6.bin').read_bytes());nq,nh,sk=params[0],params[2],params[7]
threads=int(os.environ.get('REPLAY_THREADS','1'));assert 1<=threads<=4
env=os.environ.copy();env.update(VK_DRIVER_FILES=os.environ.get('VK_DRIVER_FILES',str(root/'../root/usr/share/vulkan/icd.d/lvp_icd.json')),LD_LIBRARY_PATH=os.environ.get('LD_LIBRARY_PATH',str(root/'../root/usr/lib/x86_64-linux-gnu')),MESA_SHADER_CACHE_DISABLE='true',GALLIVM_PERF='nopt',LP_NUM_THREADS=str(threads),LP_NATIVE_VECTOR_WIDTH='128',OPENBLAS_NUM_THREADS='1')
command=['./replay',f'logical128-{arm}.spv','CSRangeAttnKvarn4Mq',str(out),'12' if arm.startswith('mq_gqa2') else '4',str((nq+1)//2),str(sk)]
start=time.monotonic()
with (out/'run.log').open('w') as f:p=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
r={'command':command,'returncode':p.returncode,'elapsed_s':time.monotonic()-start,'maxrss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'native_vector_width':128,'logical_wave':128,'LLVM':'nopt','worker_threads':threads};(out/'run.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
if p.returncode:sys.exit(1)
for i in (0,1,2,3,6):assert (out/f'b{i}.bin').read_bytes()==(out/f'out{i}.bin').read_bytes(),f'input{i} changed'
partials=np.fromfile(out/'out4.bin','<f4');assert np.isfinite(partials).all()
red=out/'reduce'
subprocess.run([sys.executable,'long_check.py','--prepare-reducer',str(out),str(red)],env=env,check=True)
subprocess.run(['./replay','reduce.spv','CSRangeReduceOpt',str(red),str(nh),str(nq),'1'],env=env,check=True)
subprocess.run([sys.executable,'long_check.py',fixture,str(red/'out4.bin')],env=env,check=True)
(out/'oracle-result.json').write_bytes((root/fixture/'oracle-result.json').read_bytes())
(out/'artifact-sha256.json').write_text(json.dumps({str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/'out4.bin',red/'out4.bin',out/'b6.bin']},indent=2))
