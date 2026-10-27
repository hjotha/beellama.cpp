# RKVA GQA2 opt-in integration — offline validation

Status: **offline integration and CPU validation complete; stable-branch commit
for review.** No MSVC/UWP build, D3D12 PSO creation, Xbox run, GPU timing, LAN,
SSH or model-quality measurement was performed by this phase. The embedded DXIL
comes from a real DXC artifact produced outside this host.

## Scope

- Clone: `/home/hjotha/beellama-agents/gqa2-prefill` (no worktree; dedicated
  clone).
- Branch: `perf/rkva-gqa2-prefill-20261027`.
- Base: `dbbb9f205` (`xbox: add scoped KVarN4 MQ two-head compile experiment`).
- No merge and no push were performed.

Changed files:

- `tools/xbox/shaders/generated/rkva_range_attn_kvarn4_mq2_gqa2_dxil.h` (new,
  rendered from the supplied DXC artifact)
- `tools/xbox/scripts/render-rkva-dxil-header.py` (new; renderer extracted
  verbatim from `compile-rkva-range-shaders.sh`)
- `tools/xbox/scripts/compile-rkva-range-shaders.sh` (new target; calls the
  extracted renderer)
- `tools/xbox/uwp/rkva/remote-attn-engine.{h,cpp}` (PSO/status, flag arm, gate,
  dispatch, counters, samples, bulk segments)
- `tools/xbox/uwp/rkva/remote-attn.cpp` (LocalState flag read at HELLO)
- `tools/xbox/uwp/rkva/rkva-range-profile.h` (policy/gate/arm, GQA2 identity,
  shared counter classifier, JSON)
- `tools/xbox/tests/rkva_range_profile_cpu.cpp`
- `tools/xbox/tests/test_rkva_mq_gqa2_layout.py`
- `tools/xbox/tests/test_rkva_gqa2_engine_policy.py` (new source contract)
- `tools/xbox/tests/test_rkva_decode_rowlane_profile.py` (updated for the shared
  counter classifier; behavior unchanged)
- `tools/xbox/scripts/rkva-quality-gate.py` and
  `tools/xbox/tests/test_rkva_quality_gate.py` (GQA2 flag ownership)
- `tools/xbox/shaders/generated/rkva_range_attn_kvarn4_mq{2,4}_dxil.h` and
  `rkva_range_attn_kvarn4_mq2_i8_dxil.h` (metadata-only provenance update)
- `docs/rkva-prefill-gqa2.md`, `docs/rkva-final-optimizations.md`, this report

The MQ2/MQ4/INT8 HLSL wrappers were not edited. The three shipped generated
headers received a metadata-only update; their embedded DXIL arrays are
byte-identical (hashes below).

## Supplied artifact verification

Artifact directory:
`/home/hjotha/.codex/artifacts/rkva-priorities-20261026/dxc-gqa2-result/`.

```text
compiler.json: DXC 1.8.2502.11, Windows Kits 10.0.26100.0, cs_6_0 F32
candidate/gqa2.dxil: 47924 bytes
b26dd85cb5512225dcb10333fc53a7a7b52676ccac2d480ed237fdcf92565a99
```

Source identity (`sha256sum`) matched byte-for-byte between the repository and
the candidate directory before rendering:

```text
045bc33453cacf9782dcc0c306122cd58fb56dd00989481be9a8effb8a7d640a  rkva_range_attn_kvarn4_mq_gqa2.hlsl
f8f5f6b99785c2fba17d95fcb052433065e306ec4782f165087fb2c4fbcc1083  rkva_range_attn_kvarn4_mq.hlsl
```

The supplied `gqa2.asm` reports `NumThreads=(128,1,1)` and
`EntryFunctionName: CSRangeAttnKvarn4Mq`. The same artifact directory shows the
default variants are unchanged by real DXC:
`baseline/mq2.dxil` == `candidate/mq2.dxil`,
`baseline/mq4.dxil` == `candidate/mq4.dxil`,
`baseline/i8.dxil` == `candidate/i8.dxil` (`cmp` reported identical).

## Header generation

The renderer was extracted from the existing compile script without changing its
emitted bytes. Re-rendering the three shipped headers from the baseline
artifacts reproduced the committed files exactly:

```bash
python3 tools/xbox/scripts/render-rkva-dxil-header.py \
  --binary .../baseline/mq2.dxil --output /tmp/opencode/gqa2/mq2.h \
  --source .../baseline/rkva_range_attn_kvarn4_mq.hlsl \
  --target attn_kvarn4_mq2
cmp /tmp/opencode/gqa2/mq2.h tools/xbox/shaders/generated/rkva_range_attn_kvarn4_mq2_dxil.h
# mq2 header byte-identical
```

The same check passed for `attn_kvarn4_mq4` and `attn_kvarn4_mq2_i8`
(`i8` with `--profile cs_6_4`).

The candidate header was then rendered from the real blob:

```bash
python3 tools/xbox/scripts/render-rkva-dxil-header.py \
  --binary /home/hjotha/.codex/artifacts/rkva-priorities-20261026/dxc-gqa2-result/candidate/gqa2.dxil \
  --output tools/xbox/shaders/generated/rkva_range_attn_kvarn4_mq2_gqa2_dxil.h \
  --source tools/xbox/shaders/rkva/rkva_range_attn_kvarn4_mq_gqa2.hlsl \
  --target attn_kvarn4_mq2_gqa2
# .../rkva_range_attn_kvarn4_mq2_gqa2_dxil.h 47924 bytes
```

The header records `Target: attn_kvarn4_mq2_gqa2`, wrapper SHA256
`045bc3...`, included kernel SHA256 `f8f5f6...`. The Python suite reconstructs
the 47 924 bytes and rechecks the pinned SHA-256.

The shipped MQ2/MQ4/INT8 headers were re-rendered from the same validated DXC
blobs with the current kernel provenance. Only the metadata block changed; every
embedded DXIL array is byte-identical to the previous headers:

```text
header                                  header SHA-256
rkva_range_attn_kvarn4_mq2_dxil.h       7119a24ed39ef605a1b47a63157daadc5a45a9e2da8a60cc10b1fe86f4b82857
rkva_range_attn_kvarn4_mq4_dxil.h       22e5f166d4c588d8dd635d826d962b67c527b2b25d062bd45aa42b6c35592662
rkva_range_attn_kvarn4_mq2_i8_dxil.h    887cbcc80a2c80db411991c282b09a437dbc38f518548bddd2f4ec536056a205

array payload                           bytes     DXIL SHA-256
mq2                                    118452    9d81c3beba4e36756c89c6bc70c5807cedb990be67fba683fc714ff7a1c5b431
mq4                                    236896    ef2878458bd37b6f702fec19f36e674a836803d300915cf2604bfd801dfb48bb
i8                                     140356    a2b66069596fe3b8111dd19d99b4b65a652b2b2e376f44014015ade13f2c2d3f
```

The Python suite pins both the full headers and the embedded arrays; it fails if
a future regeneration changes the DXIL payload or the default preprocessing
contract. The smaller candidate blob and its reduced LDS are **not** evidence of
a TPS gain: the candidate triples the pass-1 groups and reloads/dequantizes each
K/V row three times.

### Compile-script wiring check (not a compilation)

A fake `DXC` that copies the supplied blob was used only to exercise the new
target path, with no claim that DXC ran on this host:

```text
-T cs_6_0 -E CSRangeAttnKvarn4Mq \
  .../rkva_range_attn_kvarn4_mq_gqa2.hlsl -Fo /tmp/tmp.SmEItnbugp
```

The rendered header SHA-256 was unchanged:
`83ec36606535d716e880dc5bef39a615a60421e5afb41c18efcdd61a6edef71c`.
This proves target/source/profile plumbing and renderer determinism only.

## CPU tests executed

Commands and real results:

```bash
python3 tools/xbox/tests/test_rkva_mq_gqa2_layout.py
# Ran 11 tests ... OK

g++ -std=c++17 -Wall -Wextra -Werror -O1 \
  tools/xbox/tests/rkva_range_profile_cpu.cpp -o /tmp/opencode/gqa2/bin/rkva_range_profile_cpu
/tmp/opencode/gqa2/bin/rkva_range_profile_cpu > /tmp/opencode/gqa2/profile.json
# rkva_range_profile_cpu: ok
python3 -m json.tool /tmp/opencode/gqa2/profile.json > /dev/null
# exit 0

for name in rkva_range_profile_cpu rkva_execution_lifetime_cpu \
            rkva_kvarn_range_cpu rkva_kvarn_bulk_cpu rkva_bulk_fp8_cpu; do
  g++ -std=c++17 -O2 -Wall -Wextra -Werror -pthread \
    tools/xbox/tests/$name.cpp -o /tmp/opencode/gqa2/bin/$name
  /tmp/opencode/gqa2/bin/$name
done
# all exit 0
```

The C++ suite covers engine selection for flag OFF/ON, nq 1/2..9/10, both
tails, fine/balanced/coarse, descriptor revisions 1/2/3, non-KVarN4, MQ4,
head_dim, GQA, zero/other tail, wave-lane variants, missing PSO, opt-out and
INT8 conflict. It also checks the engine plan: `DispatchX = 3 * n_head_kv`,
ineligible calls keep `n_head_kv`, and the reported pass-1 groups are exactly
three times the default for a selected plan. `attend_range()` now uses that
plan for PSO selection and group reporting, so the test covers the same policy
the engine executes. A source contract
(`test_rkva_gqa2_engine_policy.py`) additionally keeps the D3D12 engine on the
tested plan and the HELLO arm/conflict path. Kernel names, bucket keys, counters
(including real bulk classification) and the unchanged default JSON
(`enabled:false`, zero GQA2 calls) are covered.

Python discovery:

```bash
OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s tools/xbox/tests -p 'test_rkva*.py'
# Ran 309 tests ... OK

OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s tools/xbox/tests -p 'test_run_rkva*.py'
# Ran 21 tests ... OK

OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s tools/xbox/tests -p 'test_gpupeak*.py'
# Ran 5 tests ... OK
```

The broad `test_*.py` discovery runs 385 tests and still reports the two known
unrelated errors (`test_make_shape_gguf.py` cannot find
`tools/xbox/scripts/make-shape-gguf.py`; `test_validate_mtp_protocol.py` imports
a missing `scripts.validate_mtp_protocol`); no other failure was reported.

### Quality-gate flag ownership

`rkva-quality-gate.py` now treats `rkva-mq-gqa2.flag` as a managed LocalState
flag. `LocalStateFlags` captures it with INT8 and rowlane, clears it for every
existing INT8/rowlane arm, restores the exact original state in the existing
`finally`/`restore()` path, and rejects `INT8 && GQA2` before mutating the
device. Future GQA2 A/B can use `flags.arm(False, gqa2_on=True)` without
changing the existing `flags.arm(True/False)` callers.

Offline coverage added to `test_rkva_quality_gate.py`:

```text
Ran 86 tests ... OK
```

The new cases prove: GQA2-only arm sets only `rkva-mq-gqa2.flag`; INT8-only arm
clears a stale GQA2 flag; the original GQA2 presence is restored after each arm;
the INT8+GQA2 conflict raises `ValueError` without touching LocalState; and a
legacy two-flag `flags-original.json` is migrated by capturing the current GQA2
presence, so resumable runs started before this change remain valid.

## CPU build with CUDA and Vulkan disabled

Configured in `/tmp/opencode/gqa2/build` with `-DGGML_CUDA=OFF
-DGGML_VULKAN=OFF -DLLAMA_BUILD_XBOX_HOST=ON -DXLLAMA_BUILD_TESTS=ON`, built
with `--target xllama-tests -j2` under `nice -n 19`:

```text
[13/13] Linking CXX executable bin/xllama-tests
[doctest] test cases:   366 |   366 passed | 0 failed | 0 skipped
[doctest] assertions: 12430 | 12430 passed | 0 failed |
[doctest] Status: SUCCESS!
```

`ctest` on the same build:

```text
1/4 Test #1: xllama-tests .....................   Passed
2/4 Test #2: xllama-ceiling-roofline ..........   Passed
3/4 Test #3: xllama-gpupeak-stream-model ......   Passed
4/4 Test #4: xllama-gpupeak-dxil-fresh ........   Passed
100% tests passed out of 4
```

## What still depends on Windows/MSVC and hardware

1. **MSVC/UWP compile**: `remote-attn.cpp` and `remote-attn-engine.cpp` include
   D3D12/C++/WinRT and are not compiled by the host build. The new header/C++
   blocks were reviewed but not compiled with MSVC on this host.
2. **DXC determinism**: the embedded DXIL is the supplied artifact. Recompiling
   the wrapper with the Windows SDK 26100 DXC and comparing the blob hash remains
   a Windows-side check.
3. **PSO creation**: `CreateComputePipelineState` for
   `attn_kvarn4_mq2_gqa2` must succeed on the console; only then does
   `mq2_gqa2.available` become true.
4. **Numerical/hardware qualification**: future GPU replay must compare OUT/LSE
   against both the default MQ2 reference and the baseline, including odd query
   counts and the 128-token/seam/split boundaries, and compare full-vocabulary
   logits plus recurrent state. Only then may pass1 timing/prefill throughput and
   stability be measured on the Xbox. No speedup, TPS delta or bit-exactness is
   claimed here; the smaller blob and reduced LDS do not establish a gain
   because the candidate triples groups and K/V row reloads.

## Non-claims

No GPU, Xbox, Device Portal, LAN, SSH, service, model or Windows build was used.
No merge or push was performed. `rkva-mq-gqa2.flag` remains off by default and is
not part of the accepted optimization set.
