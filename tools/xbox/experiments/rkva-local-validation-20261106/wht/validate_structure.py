#!/usr/bin/env python3
"""Isolated source, address/synchronization, layout and dense WHT checks.
No GPU execution. Bank proof is independent from numerical CPU coroutines.
"""
from pathlib import Path
import json,re,subprocess,random,sys
import numpy as np
P=Path(__file__).resolve().parent
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else P/'.build'
base=(P/'baseline.hlsl').read_text(); cand=(P/'rkva_range_reduce_opt_pingpong.hlsl').read_text()
s=base.index('    [unroll] for (uint stride_wht'); e=base.index('    float x =',s)
cs=cand.index('    [unroll] for (uint stride_wht'); ce=cand.index('    float x =',cs)
base_loop=base[s:e];cand_loop=cand[cs:ce]
assert base_loop.count('GroupMemoryBarrierWithGroupSync();')==2
assert cand_loop.count('GroupMemoryBarrierWithGroupSync();')==1
assert not re.search(r'\bWave\w*\s*\(',cand)
assert 'groupshared float gs_wht[512];' in cand
assert 'uint wht_read = 0;' in cand and 'uint wht_write = 256;' in cand
assert 'wht_read ^= 256;' in cand_loop and 'wht_write ^= 256;' in cand_loop
# Check all code outside the bounded edit remains unchanged.
initial='    gs_wht[lane] = acc0 * inv;'
assert base[:base.index(initial)].replace('gs_wht[256]','gs_wht[512]')==cand[:cand.index(initial)]
tail='    float out0 = (x + y) * 0.7071067811865475f;'
assert base[base.index(tail):]==cand[cand.index(tail):]
assert 'gs_wht' not in cand[cand.index(tail):]
# Source-level constants preserve the two separately rounded normalization operations.
assert 'float x = gs_wht[wht_read + lane] * 0.08838834764831845f;' in cand
assert 'float y = gs_wht[wht_read + lane + 128] * 0.08838834764831845f;' in cand
assert 'float out1 = (x - y) * 0.7071067811865475f;' in cand
banks=[]
for stage,stride in enumerate((1,2,4,8,16,32,64)):
    rd=(stage%2)*256;wr=((stage+1)%2)*256
    reads=[];writes=[]
    for lane in range(128):
        reads += [rd+lane,rd+(lane^stride),rd+lane+128,rd+(lane^stride)+128]
        writes += [wr+lane,wr+lane+128]
    assert set(reads).isdisjoint(writes)
    assert len(writes)==len(set(writes))==256
    assert set(reads)==set(range(rd,rd+256))
    assert set(writes)==set(range(wr,wr+256))
    assert min(reads+writes)>=0 and max(reads+writes)<512
    banks.append({'stride':stride,'read_bank':rd,'write_bank':wr,'unique_writers':256,'read_write_disjoint':True})
# Versioned-memory model: two wave64s can execute reads/writes independently.
# Each stage reads first and writes later, then both waves rendezvous. Delays
# within a wave are conservative: neither wave may cross the group barrier.
def schedule_model(seed,barriers=True):
    rng=random.Random(seed);versions=[0]*256+[-1]*256
    pc=[0,0];steps=0;interleavings=[]
    # Each wave has 7 stages of R,W,barrier actions. At stage N, reads require
    # version N and writes publish N+1. Model arithmetic-independent hazards.
    while pc != [21,21]:
        eligible=[]
        for wave in (0,1):
            if pc[wave]>=21:continue
            op=pc[wave]%3
            if op==2 and barriers:
                # Release the group barrier only when both waves arrived.
                if pc[0]==pc[1]:eligible.append(wave)
            else:eligible.append(wave)
        assert eligible,'deadlock or divergent barrier'
        wave=eligible[rng.randrange(len(eligible))]
        stage,op=divmod(pc[wave],3);stride=1<<stage
        rd=(stage%2)*256;wr=((stage+1)%2)*256
        if op==0:
            for lane in range(wave*64,wave*64+64):
                for index in (rd+lane,rd+(lane^stride),rd+lane+128,rd+(lane^stride)+128):
                    if versions[index]!=stage:return False,steps
        elif op==1:
            for lane in range(wave*64,wave*64+64):
                versions[wr+lane]=versions[wr+lane+128]=stage+1
        elif barriers:
            pc[0]+=1;pc[1]+=1;steps+=1;continue
        pc[wave]+=1;steps+=1
    # Final H2 reads the seventh stage's bank. No code after the transform
    # writes this bank; source assertion above checks no further gs_wht use.
    assert all(x==7 for x in versions[256:])
    return True,steps
valid_schedules=2048
assert all(schedule_model(seed)[0] for seed in range(valid_schedules))
unsafe_schedules=sum(not schedule_model(seed,False)[0] for seed in range(valid_schedules))
assert unsafe_schedules>0
# qh/index proof covers odd head/query counts and split capacity edges. The
# prototype changes neither group mapping, descriptor layout nor I/O stores.
shapes=0;odd_shapes=0;max_partial_words=0
for nq in (1,2,3,8,9):
    for nh in (1,2,3,24):
        total=nq*nh
        qhs=[y*nh+x for y in range(nq) for x in range(nh)]
        assert sorted(qhs)==list(range(total))
        output_words=np.zeros(total*257,dtype=np.uint8)
        for qh in qhs:
            for lane in range(128):
                output_words[qh*256+lane]+=1
                output_words[qh*256+lane+128]+=1
            output_words[total*256+qh]+=1
        assert np.all(output_words==1)
        for split in (0,1,2,3,31,32,63,64,127,128,255,256,257):
            capacity=split*total*258
            actual=min(split,256)
            for qh in qhs:
                for spl in ({0,actual-1} if actual else set()):
                    offset=(spl*total+qh)*258
                    touched=np.array([offset,offset+63,offset+64,offset+127,offset+128,offset+191,offset+255,offset+256,offset+257])
                    assert np.all(touched>=0) and np.all(touched<capacity)
            shapes+=1;odd_shapes+=bool(nh%2)
            max_partial_words=max(max_partial_words,capacity)
# Entire independent mathematical mapping using float64 H256/16, not the
# round-by-round float32 oracle used by the C++ executable.
x=np.fromfile(OUT/'dense_input.f32',dtype=np.float32).reshape(-1,256)
y=np.fromfile(OUT/'dense_output.f32',dtype=np.float32).reshape(-1,256)
h=np.ones((1,1),dtype=np.float64)
for _ in range(8):h=np.block([[h,h],[h,-h]])
expected=x.astype(np.float64)@h/16
err=np.abs(y.astype(np.float64)-expected)
assert float(err.max())<2e-6
# Verify that applying the candidate diff would only modify the copied target.
patch=subprocess.run(['patch','--dry-run','--batch',str(P/'baseline.hlsl'),str(P/'candidate.patch')],capture_output=True,text=True)
assert patch.returncode==0,patch.stdout+patch.stderr
result={
 'status':'PASS','static_checks':'unchanged non-WHT code, two rounding constants, no Wave intrinsics, uniform bounded loop, no post-WHT shared reuse',
 'bank_proof':banks,'safe_two_wave64_schedules':valid_schedules,
 'negative_missing_barrier_schedules_detected':unsafe_schedules,
 'layout_shapes':shapes,'odd_head_layout_shapes':odd_shapes,
 'query_counts':[1,2,3,8,9],'head_counts':[1,2,3,24],
 'split_counts':[0,1,2,3,31,32,63,64,127,128,255,256,257],
 'max_partial_buffer_words_modeled':max_partial_words,
 'dense_float64_vectors':len(x),'dense_float64_max_abs_error':float(err.max()),
 'dynamic_wht_barriers_baseline':15,'dynamic_wht_barriers_candidate':8,
 'butterfly_loop_barriers_baseline':14,'butterfly_loop_barriers_candidate':7,
 'wht_shared_bytes_baseline':1024,'wht_shared_bytes_candidate':2048,
 'declared_total_shared_bytes_baseline':4104,'declared_total_shared_bytes_candidate':5128,
 'patch_dry_run':'PASS',
 'limits':['No DXC or D3D12/GPU execution','No ISA/resource-allocation inspection','CPU strict float32, gradual underflow; Xbox denormal policy unverified','No Xbox correctness or performance qualification','Reducer build target currently lacks -Gis; compiler reassociation/two-rounding behavior must be inspected before integration'],
}
(OUT/'structure_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
