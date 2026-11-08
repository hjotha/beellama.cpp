#!/usr/bin/env python3
"""Native MQ2 host oracle, NOT GPU execution or performance qualification.
Extracts actual HLSL scalar loaders/codebook and runs them through bounded C++
buffers against real GGML quantization/dequantization. Simulates MQ2 scheduling,
transforms, masking, split states and output, comparing to dense original-domain
attention. Wave reductions/barriers remain GPU-only qualification requirements.
Usage: python tests/test-rkva-native-mq2.py /path/to/build [--sanitize]
"""
import math, pathlib, random, re, subprocess, sys, tempfile
root = pathlib.Path(__file__).resolve().parents[1]
build = pathlib.Path(sys.argv[1]).resolve()
shader = root / 'tools/xbox/shaders/rkva'
source = (shader/'rkva_range_attn_native_mq2.hlsl').read_text()
# Structural guards complement (but do not replace) the numerical emulation.
assert '[numthreads(128, 1, 1)]' in source
assert 'qi = group.y * 2u + wave' in source
assert source.count('GroupMemoryBarrierWithGroupSync();') == 4
assert 'if (wave == 0u)' in source and 'groupshared float4 mq_k[64]' in source
assert 'groupshared float4 mq_v[64]' in source
assert 'uint end = min(active_count, (split + 1u) * split_rows)' in source
assert 'groupshared float4 mq_query_wht[6 * 128]' in source
assert 'mq_query_wht[g * 128u + (tid ^ stride)]' in source
assert 'WaveReadLaneAt' not in source.replace("WaveReadLaneAt lowering", "wave lowering")

# Model the shared-memory butterfly independently for both waves and all heads.
rng = random.Random(855517)
for _ in range(10):
    original = [[[rng.uniform(-2, 2) for _ in range(4)] for _ in range(128)] for _ in range(6)]
    q = [[[a+b+c+d, a-b+c-d, a+b-c-d, a-b-c+d] for a,b,c,d in head] for head in original]
    for stride in (1, 2, 4, 8, 16, 32):
        old = q
        q = [[[x+y if (tid & stride) == 0 else y-x for x,y in zip(old[g][tid], old[g][tid ^ stride])]
              for tid in range(128)] for g in range(6)]
    for g in range(6):
        for wave in range(2):
            expected = [x for lane in original[g][wave*64:(wave+1)*64] for x in lane]
            for stride in (1, 2, 4, 8, 16, 32, 64, 128):
                for start in range(0, 256, 2*stride):
                    for i in range(stride):
                        a, b = expected[start+i], expected[start+i+stride]
                        expected[start+i], expected[start+i+stride] = a+b, a-b
            actual = [x for lane in q[g][wave*64:(wave+1)*64] for x in lane]
            assert all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-10) for a,b in zip(actual, expected))

def function(s, name):
    m = re.search(r'^(?:uint|uint4|float4) '+name+r'\([^\n]*\) \{',s,re.M)
    assert m, name
    return s[m.start():s.index('\n}',m.end())+2]

def table(s, name):
    p = s.index('static const float '+name)
    return s[p:s.index('};',p)+2]

shim = r'''
#include "ggml.h"
#include <cassert>
#include <cstdint>
#include <cstring>
using uint = uint32_t;
struct uint2 { uint x, y; };
struct uint4 { uint x,y,z,w; uint4(uint a,uint b,uint c,uint d):x(a),y(b),z(c),w(d){} };
struct float2 { float x,y; };
struct float4 {
    float v[4];
    float4(float a,float b,float c,float d):v{a,b,c,d}{}
    float4(uint4 a):v{float(a.x),float(a.y),float(a.z),float(a.w)}{}
    float operator[](size_t i)const{return v[i];}
    float4 operator*(float a)const{return {v[0]*a,v[1]*a,v[2]*a,v[3]*a};}
    float4 operator+(float a)const{return {v[0]+a,v[1]+a,v[2]+a,v[3]+a};}
    float4 operator-(float a)const{return *this + (-a);}
};
struct ByteAddressBuffer {
    const uint8_t* data;size_t size;
    uint Load(uint off) const { assert(off%4==0 && off<=size && size-off>=4); uint x;std::memcpy(&x,data+off,4);return x; }
    uint2 Load2(uint off)const{return {Load(off),Load(off+4)};}
};
inline float f16tof32(uint h){return ggml_fp16_to_fp32(uint16_t(h));}
inline float2 asfloat(uint2 v){float2 f;std::memcpy(&f,&v,8);return f;}
'''
b=(shader/'rkva_decode_quants_group_b.hlsli').read_text()
c=(shader/'rkva_decode_quants_group_c.hlsli').read_text()
s=(shader/'rkva_decode_quants_saw_int4.hlsli').read_text()
parts=[function(b,n) for n in ('quarter_dim','quarter_dim_q4_1','load_q4_1')]
parts += [table(c,'kvalues_iq4nl')]
parts += [function(c,n) for n in ('quarter_dim_iq4_nl','load_iq4_nl')]
parts += [function(s,n) for n in ('quarter_dim_saw_int4','load_saw_int4')]
parts += [table(source,'mq_centroids'),function(source,'mq_indices')]
with tempfile.TemporaryDirectory(prefix='rkva-native-mq2-') as temp:
    h=pathlib.Path(temp)/'loaders.h'; h.write_text(shim+'\n'.join(parts))
    exe=pathlib.Path(temp)/'test'
    flags=['-O2']
    if '--sanitize' in sys.argv: flags=['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer']
    subprocess.run(['g++','-std=c++17',*flags,'-ffp-contract=off','-include',str(h),
        '-I'+str(root/'ggml/include'),'-I'+str(root/'ggml/src'),
        str(root/'tests/test-rkva-native-mq2.cpp'),'-L'+str(build/'bin'),
        '-Wl,-rpath,'+str(build/'bin'),'-Wl,--no-as-needed','-lggml','-lggml-base','-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
