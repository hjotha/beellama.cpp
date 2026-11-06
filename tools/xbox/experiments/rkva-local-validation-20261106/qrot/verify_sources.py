#!/usr/bin/env python3
"""Verify bundled exact fixtures and the derivation of the candidate/CPU wrapper."""
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE / "source_manifest.json").read_text())
omitted = set(manifest["publication"]["not_bundled"])
count = 0
for name, record in manifest.items():
    if not name.startswith("tools/") or name in omitted:
        continue
    path = HERE / "source-main" / Path(name).name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == record["main_sha256"], f"Fixture SHA256 mismatch: {path.name}"
    count += 1
for name, expected in manifest["publication"]["model_fixture_sha256"].items():
    actual = hashlib.sha256((HERE / "models" / name).read_bytes()).hexdigest()
    assert actual == expected, f"Model SHA256 mismatch: {name}"

name = "rkva_range_attn_kvarn4_opt.hlsl"
source = (HERE / "source-main" / name).read_text()
candidate = (HERE / "prototype" / name).read_text()
repo_path = "tools/xbox/shaders/rkva/" + name
patch = "".join(difflib.unified_diff(
    source.splitlines(True), candidate.splitlines(True),
    fromfile="a/" + repo_path, tofile="b/" + repo_path))
assert patch == (HERE / "qrot-reuse-opt-in.patch").read_text(), "Patch/prototype mismatch"
name = "rkva_decode_rowlane.hlsli"
assert (HERE / "prototype" / name).read_bytes() == (HERE / "source-main" / name).read_bytes()

header = (HERE / "source-main/rkva-kvarn-range.h").read_text()
start = header.index("inline void butterfly_wht256(")
end = header.index("\n} // namespace", start)
expected = '#include <cstdint>\n' + header[start:end] + '''
extern "C" void reference_wht(const float* src, float* dst, unsigned heads) {
    for (unsigned h=0; h<heads; ++h) butterfly_wht256(src+256*h,dst+256*h);
}
'''
assert (HERE / "cpu_reference.cpp").read_text() == expected, "CPU extraction mismatch"
print(f"PASS: {count} source SHA256 values, 4 model SHA256 values, patch/prototype and CPU extraction")
