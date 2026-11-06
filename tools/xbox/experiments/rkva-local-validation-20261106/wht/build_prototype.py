from pathlib import Path
import difflib, hashlib, json, sys
HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / '.build'
OUT.mkdir(parents=True, exist_ok=True)
REL = 'tools/xbox/shaders/rkva/rkva_range_reduce_opt.hlsl'
source_bytes = (HERE / 'baseline.hlsl').read_bytes()
assert hashlib.sha256(source_bytes).hexdigest() == '3513aa4a0dc38597cc0f690e768ad2837873543d1cbfc64a645baa1ec6a586fe', 'Baseline fixture SHA256 mismatch'
src = source_bytes.decode('utf-8')
assert hashlib.sha256(src.encode()).hexdigest() == '3513aa4a0dc38597cc0f690e768ad2837873543d1cbfc64a645baa1ec6a586fe'
old = '''    [unroll] for (uint stride_wht = 1; stride_wht < 128; stride_wht *= 2) {
        float own0 = gs_wht[lane];
        float other0 = gs_wht[lane ^ stride_wht];
        float own1 = gs_wht[lane + 128];
        float other1 = gs_wht[(lane ^ stride_wht) + 128];
        GroupMemoryBarrierWithGroupSync();
        gs_wht[lane] = (lane & stride_wht) ? other0 - own0 : own0 + other0;
        gs_wht[lane + 128] = (lane & stride_wht) ? other1 - own1 : own1 + other1;
        GroupMemoryBarrierWithGroupSync();
    }
    float x = gs_wht[lane] * 0.08838834764831845f;
    float y = gs_wht[lane + 128] * 0.08838834764831845f;
'''
new = '''    // Isolated prototype: ping-pong LDS avoids lane-dependent wave shuffles.
    // Each stage reads one complete bank and writes a disjoint bank. The
    // trailing group barrier protects both the next stage's reads and reuse
    // of the old bank. Keep all 128 lanes participating, including both waves.
    uint wht_read = 0;
    uint wht_write = 256;
    [unroll] for (uint stride_wht = 1; stride_wht < 128; stride_wht *= 2) {
        float own0 = gs_wht[wht_read + lane];
        float other0 = gs_wht[wht_read + (lane ^ stride_wht)];
        float own1 = gs_wht[wht_read + lane + 128];
        float other1 = gs_wht[wht_read + (lane ^ stride_wht) + 128];
        gs_wht[wht_write + lane] = (lane & stride_wht) ? other0 - own0 : own0 + other0;
        gs_wht[wht_write + lane + 128] = (lane & stride_wht) ? other1 - own1 : own1 + other1;
        GroupMemoryBarrierWithGroupSync();
        wht_read ^= 256;
        wht_write ^= 256;
    }
    float x = gs_wht[wht_read + lane] * 0.08838834764831845f;
    float y = gs_wht[wht_read + lane + 128] * 0.08838834764831845f;
'''
assert src.count(old) == 1
candidate = src.replace('groupshared float gs_wht[256];', 'groupshared float gs_wht[512];').replace(old, new)
patch = ''.join(difflib.unified_diff(src.splitlines(True), candidate.splitlines(True), fromfile='a/'+REL, tofile='b/'+REL))
assert candidate.encode() == (HERE / 'rkva_range_reduce_opt_pingpong.hlsl').read_bytes(), 'Candidate differs from regenerated prototype'
assert patch.encode() == (HERE / 'candidate.patch').read_bytes(), 'Patch differs from regenerated prototype'
(OUT / 'rkva_range_reduce_opt_pingpong.hlsl').write_text(candidate)
(OUT / 'candidate.patch').write_text(patch)
manifest = json.loads((HERE / 'source_manifest.json').read_text())
assert manifest['source_sha256'] == hashlib.sha256(source_bytes).hexdigest()
assert manifest['prototype_sha256'] == hashlib.sha256(candidate.encode()).hexdigest()
assert manifest['patch_sha256'] == hashlib.sha256(patch.encode()).hexdigest()
print(json.dumps({'status': 'PASS', 'baseline_sha256': manifest['source_sha256'], 'candidate_sha256': manifest['prototype_sha256'], 'patch_sha256': manifest['patch_sha256']}, indent=2))
