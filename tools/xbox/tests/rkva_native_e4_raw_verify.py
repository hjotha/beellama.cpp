#!/usr/bin/env python3
"""CPU execution of the *actual* HLSL integer helpers against the wire oracle.
Does not execute DXIL or qualify Xbox. Compile DXIL separately with DXC/DXV.
"""
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[3]
shader = (root / 'tools/xbox/shaders/rkva/rkva_range_bulk_e4_raw.hlsl').read_text()
helpers = shader[shader.index('uint e4_round_shift'):shader.index('[numthreads')]
harness = r'''
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include "ggml/src/ggml-remote-attn/rkva-fp8-e4m3.h"
using uint = uint32_t;
HELPERS
uint bits(float v) { uint b; std::memcpy(&b, &v, 4); return b; }
float val(uint b) { float v; std::memcpy(&v, &b, 4); return v; }
size_t count = 0;
void check(uint b) {
    uint8_t out = 0;
    const bool valid = rkva_e4::encode_f32(val(b), out);
    const bool shader_valid = (b & 0x7fffffffu) < 0x7f800000u;
    if (valid != shader_valid || (valid && out != e4_from_f32_bits(b))) {
        std::fprintf(stderr, "mismatch %08x\n", b); std::abort();
    }
    ++count;
}
int main() {
    for (uint c = 0; c < 256; ++c) {
        float decoded;
        if (!rkva_e4::decode_f32(uint8_t(c), decoded)) continue;
        if (bits(decoded) != e4_to_f32_bits(c)) std::abort();
        check(bits(decoded));
    }
    for (uint c = 0; c < 126; ++c) {
        float a, b; rkva_e4::decode_f32(uint8_t(c), a); rkva_e4::decode_f32(uint8_t(c+1), b);
        const uint midpoint = bits((a+b)*0.5f);
        for (int delta = -1; delta <= 1; ++delta) {
            check(midpoint + delta); check((midpoint + delta) | 0x80000000u);
        }
    }
    const uint special[] = {0, 0x80000000u, 1, 0x807fffffu, 0x43dfffffu, 0x43e00000u,
        0x43e00001u, 0x477fe000u, 0x7f7fffffu, 0xff7fffffu, 0x7f800000u,
        0xff800000u, 0x7fc00000u, 0x7f800001u, 0xffc00000u};
    for (uint b : special) check(b);
    uint rng = 0x8194a3u;
    for (int i=0; i<1000000; ++i) { rng ^= rng<<13; rng ^= rng>>17; rng ^= rng<<5; check(rng); }
    std::printf("PASS actual HLSL integer helpers: %zu F32 cases and 254 finite E4 decodes\n", count);
}
'''.replace('HELPERS', helpers)
with tempfile.TemporaryDirectory(prefix='rkva-e4-raw-') as tmp:
    source = Path(tmp) / 'test.cpp'
    source.write_text(harness)
    exe = Path(tmp) / 'test'
    subprocess.run([os.environ.get('CXX', 'g++'), '-std=c++17', '-O2', '-fsanitize=address,undefined',
                    '-fno-omit-frame-pointer', '-I'+str(root), str(source), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
# Structural checks are not a Windows compile or hardware execution.
assert 'f32tof16' not in shader and 'f16tof32' not in shader
assert 'lse == 0xff800000u && (values[j] & 0x7fffffffu) != 0u' in shader
assert '(values[j] & 0x7fffffffu) >= 0x7f800000u' in shader
engine = (root / 'tools/xbox/uwp/rkva/remote-attn-engine.cpp').read_text()
route = (root / 'tools/xbox/uwp/rkva/remote-attn.cpp').read_text()
assert 'gpu_fp8 && attn.wire != wire' in engine
assert 'native E4 GPU: invalid LSE/prefix' in engine
assert 'state.engine->ensure_native_fp8_gpu_pipelines(reply_wire)' in route
assert '(q8_rows ? state.q8_fp8_gpu_enabled : state.q4_fp8_gpu_enabled)' in route
print('PASS structural checks: direct F32, status, wire homogeneity, PSO and existing flag admission')

header = (root / 'tools/xbox/uwp/rkva/remote-attn-engine.h').read_text()
assert '#define RKVA_EXPERIMENTAL_NATIVE_FP8_GPU 0' in header
assert '(!new_native_codec || RKVA_EXPERIMENTAL_NATIVE_FP8_GPU)' in route
assert 'gpu_fp8 && !RKVA_EXPERIMENTAL_NATIVE_FP8_GPU' in engine
print('PASS structural check: new codecs default-off, existing E5 routing preserved')
