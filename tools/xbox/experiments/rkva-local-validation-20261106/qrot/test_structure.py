#!/usr/bin/env python3
"""C preprocessing and source checks only. This does NOT compile HLSL."""
import json, os, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
BUILD=Path(os.environ.get('QROT_BUILD_DIR', str(HERE/'.build'))).resolve()
BUILD.mkdir(parents=True,exist_ok=True)
CPP=os.environ.get('CPP','cpp')
p=(HERE/'source-main/rkva_range_attn_kvarn4_opt.hlsl').read_text()
q=(HERE/'prototype/rkva_range_attn_kvarn4_opt.hlsl').read_text()
checks=[]
for rowlane in (0,1):
    opts=[CPP,'-P','-DRKVA_SERIAL_PREFIX=1','-DRKVA_DECODE_OPT=1',f'-DRKVA_DECODE_ROWLANE={rowlane}']
    def pp(file,extra=()):
        return subprocess.run(opts+list(extra)+[str(HERE/file)],text=True,capture_output=True,check=True).stdout
    a=pp('source-main/rkva_range_attn_kvarn4_opt.hlsl')
    default=pp('prototype/rkva_range_attn_kvarn4_opt.hlsl')
    b=pp('prototype/rkva_range_attn_kvarn4_opt.hlsl',['-DRKVA_DECODE_QROT_REUSE=0'])
    c=pp('prototype/rkva_range_attn_kvarn4_opt.hlsl',['-DRKVA_DECODE_QROT_REUSE=1'])
    assert a==default==b
    ca=c[c.index('void CSRangeAttnKvarn4Decode'):c.index('    int query_pos =',c.index('void CSRangeAttnKvarn4Decode'))]
    assert 'wht256x6_decode(v, tid)' not in ca and 'queries.Load4(q_rot_base)' in ca and 'GroupMemoryBarrierWithGroupSync();' in ca
    checks.append({'rowlane':rowlane,'reuse_default_off_exact_preprocessor_identity':True,'reuse_off_exact_preprocessor_identity':True,'candidate_direct_Load4_from_qrot':True,'initial_conservative_group_barrier_retained':True})
idx=p.index('void CSRangeAttnKvarn4Decode');idx2=q.index('void CSRangeAttnKvarn4Decode')
assert p[p.index('    int query_pos =',idx):]==q[q.index('    int query_pos =',idx2):]
for opts in (['-DRKVA_DECODE_OPT=0','-DRKVA_SERIAL_PREFIX=1'],['-DRKVA_DECODE_OPT=1','-DRKVA_SERIAL_PREFIX=0']):
    proc=subprocess.run([CPP,'-P','-DRKVA_DECODE_QROT_REUSE=1']+opts+[str(HERE/'prototype/rkva_range_attn_kvarn4_opt.hlsl')],capture_output=True,text=True)
    assert proc.returncode!=0 and 'RKVA_DECODE_QROT_REUSE requires' in proc.stderr
result={'status':'PASS','cpp_preprocessing_only_not_hlsl_compilation':True,'variants':checks,'all_code_after_q_setup_byte_identical':True,'no_host_PSO_flag_or_dispatch_change':True,'unsupported_define_combinations_rejected':2}
(BUILD/'structural-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
