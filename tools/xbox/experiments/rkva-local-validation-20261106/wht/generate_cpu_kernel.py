from pathlib import Path
import sys
P = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else P / '.build'
OUT.mkdir(parents=True, exist_ok=True)
baseline=(P/'baseline.hlsl').read_text()
candidate=(P/'rkva_range_reduce_opt_pingpong.hlsl').read_text()
def block(source):
    s=source.index('    gs_wht[lane] = acc0 * inv;')
    e=source.index('\n    invalid_numerator = invalid_numerator || !isfinite(out0)', s)
    return source[s:e].replace('[unroll] ', '').replace('GroupMemoryBarrierWithGroupSync();','co_await std::suspend_always{};')
def kernel(name, body):
    return '''Task %s(State& state, const Vec& input, uint lane) {
    float* gs_wht = state.shared.data();
    float acc0 = input[lane], acc1 = input[lane+128], inv = 1.0f;
%s
    state.output[lane] = out0;
    state.output[lane + 128] = out1;
    co_return;
}
''' % (name,body)
orig=block(baseline)
new=block(candidate)
# Deliberately unsafe same-buffer variant, removing ONLY the pre-write barrier.
bad=orig.replace('        co_await std::suspend_always{};\n        gs_wht[lane] =', '        gs_wht[lane] =')
assert bad != orig
# Deliberately wrong normalization, mathematically equivalent but not bit-exact.
fused=new.replace(' * 0.08838834764831845f','').replace(' * 0.7071067811865475f',' * 0.0625f')
# Deliberately read the stale bank at the end (7 stages is odd).
stale=new.replace('float x = gs_wht[wht_read + lane]', 'float x = gs_wht[lane]').replace('float y = gs_wht[wht_read + lane + 128]', 'float y = gs_wht[lane + 128]')
(OUT/'generated_kernels.inc').write_text('// Generated verbatim arithmetic/control from snapshot/prototype HLSL.\n'+kernel('baseline_kernel',orig)+kernel('candidate_kernel',new)+kernel('unsafe_kernel',bad)+kernel('fused_kernel',fused)+kernel('stale_kernel',stale))
