#!/usr/bin/env python3
"""Compile actual HLSL scalar loaders as C++ and compare with real GGML.
This checks loader bytes/table/addressing, NOT shader compilation/wave execution.
Usage: python tests/test-rkva-low4-shader.py build-low4
"""
import pathlib, subprocess, sys, tempfile
root = pathlib.Path(__file__).resolve().parents[1]
build = pathlib.Path(sys.argv[1]).resolve()
shader = root / 'tools/xbox/shaders/rkva'
b = (shader / 'rkva_decode_quants_group_b.hlsli').read_text()
c = (shader / 'rkva_decode_quants_group_c.hlsli').read_text()
def function(source, name):
    import re
    match = re.search(r'^(?:uint|float4) ' + name + r'\([^\n]*\) \{', source, re.M)
    assert match, name
    end = source.index('\n}', match.end()) + 2
    return source[match.start():end]
shim = '''#include "ggml.h"
#include <cstdint>
#include <cassert>
#include <cstring>
using uint = uint32_t;
struct uint2 { uint x, y; };
struct float4 {
    float v[4];
    float4(float a, float b, float c, float d) : v{a,b,c,d} {}
    float operator[](size_t i) const { return v[i]; }
    float4 operator*(float x) const { return {v[0]*x,v[1]*x,v[2]*x,v[3]*x}; }
    float4 operator+(float x) const { return {v[0]+x,v[1]+x,v[2]+x,v[3]+x}; }
};
struct ByteAddressBuffer {
    const uint8_t* data; size_t size;
    uint Load(uint off) const { assert(off % 4 == 0 && off <= size && size-off >= 4); uint x; std::memcpy(&x,data+off,4); return x; }
    uint2 Load2(uint off) const { return {Load(off), Load(off+4)}; }
};
inline float f16tof32(uint h) { return ggml_fp16_to_fp32(uint16_t(h)); }
'''
parts = [function(b,n) for n in ('quarter_dim','quarter_dim_q4_1','load_q4_1')]
start = c.index('static const float kvalues_iq4nl')
parts += [c[start:c.index('};',start)+2]]
parts += [function(c,n) for n in ('quarter_dim_iq4_nl','load_iq4_nl')]
with tempfile.TemporaryDirectory(prefix='rkva-low4-loader-') as tmp:
    header = pathlib.Path(tmp)/'loaders.h'; header.write_text(shim + '\n'.join(parts))
    exe = pathlib.Path(tmp)/'test'
    subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-DRKVA_LOW4_SHADER_HOST=1',
        '-include',str(header),'-I'+str(root/'ggml/include'), '-I'+str(root/'ggml/src'),
        str(root/'tests/test-rkva-low4.cpp'), '-L'+str(build/'bin'), '-Wl,-rpath,'+str(build/'bin'),
        '-Wl,--no-as-needed','-lggml','-lggml-base','-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
