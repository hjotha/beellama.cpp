# RKVA rowlane Wave64 LDS candidate (2026-11-06)

## Result and scope

Source-only, opt-in experiment; **not a qualified optimization**. Six host tests
pass. No DXC, DXIL, GPU, console, Windows build, occupancy measurement, or TPS
benchmark ran. No tool was installed. Preparation made no Xbox or production change, dispatch change, new PSO,
worktree, or generated shader-header change. This branch publishes the inert
experiment only; it does not integrate or enable it.

Only three isolated shader fixtures change. The source base
e64ad7c43ee6f2ae6ced1d1a47ba462d7751594e was preserved. These three fixtures
matched that commit byte for byte before patching. Fixture SHA256 hashes and the
host-gate source SHA256 are recorded in source-manifest.json.

## Candidate contract

`RKVA_EXPERIMENT_ROWLANE_WAVE64_LDS=1` opts in. Undefined or zero preserves
preprocessed baseline output byte for byte. No build script sets it.

Use only the existing selected rowlane entry and its existing format define:

- Q4: rkva_range_attn_q4_opt.hlsl, RKVA_DECODE_ROWLANE=1,
  CSRangeAttnQ4DecodeRowlane.
- Q8: rkva_range_attn_gqa_opt.hlsl, RKVA_DECODE_ROWLANE=1 and RKVA_OPT_STORAGE=2,
  CSRangeAttnQ8DecodeRowlane.
- F16: rkva_range_attn_rowlane_matrix.hlsl, ROWLANE_F16=1,
  CSRangeAttnF16DecodeRowlane.
- SAW: same matrix, ROWLANE_SAW_INT4=1, CSRangeAttnSawInt4DecodeRowlane.

Opt-in compiles 2 LDS wave slots, requires D256/GQA6/128 threads, and rejects
incompatible formats/shapes and unsupported matrix aliases. Generic Q4Opt and
GqaOpt entry functions are omitted ONLY from opted-in translation units, so a
build cannot accidentally select the generic entry against reduced arrays.
In the default translation units, generic entries, wave32 behavior, and four-wave
arrays remain unchanged. Runtime fallback remains exactly the existing host's.

The compile-time define is NOT a hardware wave-size guarantee. It must be paired
with the existing host gate `wave_lane_min_ == 64 && wave_lane_max_ == 64`.
Source: tools/xbox/uwp/rkva/remote-attn-engine.cpp, Q4 around lines 860-873,
SAW 879-894, Q8 1176-1189, F16 1195-1208. These gates guard creation of the selected
rowlane PSOs. The flag must never be applied globally or to other host pipelines.
This patch deliberately adds no shipping build/dispatch integration.

## Audit findings

The original broad hypothesis needs narrowing:

- Q4 source declares 27,072 bytes (24,768 state + 2,304 raw K/V staging); candidate
  declares 13,536 bytes (12,384 state + 1,152 staging).
- Selected Q4 **rowlane** does not call the staging helpers. A compiler should be
  free to eliminate the raw arrays, so the 27,072 figure is not that entry's
  proven allocation. This experiment does NOT optimize non-rowlane Q4Opt.
- Q8/F16/SAW state declarations change 24,768 -> 12,384 bytes, but this is a source
  declaration count, not measured compiled LDS.
- Generic F16/Q8 GqaOpt derives wave count from WaveGetLaneCount and requires four
  slots on wave32. A global shrink would be unsafe; generic entry is excluded
  from opt-in compilation and kept unchanged by default.
- KVarN4 decode already has separate two-wave declarations; it is left unchanged.
- DXC might already remove unused upper ranges or otherwise optimize storage.
  If compiled resource usage is unchanged, this is not an LDS/occupancy win.

## Validation

Run from this directory: `python3 test_lds.py`. Requires existing Python and cpp,
no packages. `test-results.txt` records six passing test methods:

1. Every baseline fixture SHA256 matches the source manifest.
2. Seven relevant default variants, each with undefined and explicit-zero flag,
   compare GNU-preprocessed text byte for byte. This is NOT DXC preprocessing or
   byte-identical DXIL evidence.
3. Four opt-in variants preserve their full entry bodies byte for byte, including
   all arithmetic, broadcasts, barriers, references, and loop bounds. After
   removing changed LDS declarations and deliberately omitted generic entry,
   remaining preprocessed code is whitespace-normalized identical.
4. Invalid entry/format selections, mixed formats and aliases, and incompatible
   matrix shape/array overrides fail preprocessing.
5. Enumerated 128 thread IDs x 6 heads x 4 components cover each of 3,072 state
   indices exactly once for both Q4 quarter mapping and linear mapping. Merge
   slots are 0..11. Q4 raw indices remain within 0..35. A wave32 capacity control
   demonstrates why two slots cannot serve the generic path.
6. Eighty seeded CPU publication/merge-model cases return exactly equal partial
   and normalized/LSE results for four-slot versus two-slot storage, including
   empty states. A one-slot negative control raises bounds failure. This models
   only publication/merge using Python floats, NOT full HLSL FP32, quantization,
   wave scheduling, race freedom, or GPU arithmetic.

`git apply --check ../candidate.patch` passes in the baseline fixture directory.
The candidate directory already has the patch applied; do not apply it twice.
The portable package contains ready-to-test baseline and candidate fixtures,
limited to the three changed shader files and their transitive includes. No
checkout-specific preparation script is required or included. The manifest
covers all included baseline fixtures. Run tests directly after extraction.

## Next evidence required, not performed

First compile separate candidate payloads with the exact production DXC/flags,
compare actual resource usage and disassembly, and confirm the expected entry and
fixed-wave64 PSO gate. If usage is already identical, stop this line of tuning.
Only if there is a resource difference, consider independent console correctness
checks and controlled A/B profiling. Do not combine with WHT ping-pong or Q_rot
variants. A source declaration reduction alone is not grounds to deploy.

Patch SHA256: 08db7db1ee36784161893bb5ca4b9f3eeaa703cb0d0a77520b51db7fbfac06c2.


## Exact local DXC comparison recipe

From this directory, use an already installed official DXC matching the local
production toolchain. The recorded source build uses `cs_6_0` and no extra
compile flags for these four entries. Recheck that against the checkout's
`tools/xbox/scripts/compile-rkva-range-shaders.sh` before running. Record the
DXC executable SHA256 and version, exact command lines and all output hashes.
For each variant in the Candidate contract table, run the equivalent of:

```sh
mkdir -p .build/dxil
# Example: Q4. Repeat with the exact Q8/F16/SAW entries and defines above.
"$DXC" -T cs_6_0 -E CSRangeAttnQ4DecodeRowlane -D RKVA_DECODE_ROWLANE=1 \
  baseline/tools/xbox/shaders/rkva/rkva_range_attn_q4_opt.hlsl \
  -Fo .build/dxil/q4-baseline.dxil -Fc .build/dxil/q4-baseline.asm
"$DXC" -T cs_6_0 -E CSRangeAttnQ4DecodeRowlane -D RKVA_DECODE_ROWLANE=1 \
  -D RKVA_EXPERIMENT_ROWLANE_WAVE64_LDS=1 \
  candidate/tools/xbox/shaders/rkva/rkva_range_attn_q4_opt.hlsl \
  -Fo .build/dxil/q4-candidate.dxil -Fc .build/dxil/q4-candidate.asm
```

Also compile the candidate with the experiment flag absent and explicitly zero
and compare to baseline, including generic/wave32-compatible entries. Inspect
actual groupshared allocations/resource metadata in DXIL and the disassembly,
barriers, resources and entry signatures. Missing or ambiguous resource evidence
is **not qualified**; unchanged compiled LDS means stop the LDS optimization
claim. Do not infer occupancy from source arrays. This recipe is not a report of
executed DXC commands. No generated production header is written by it.
