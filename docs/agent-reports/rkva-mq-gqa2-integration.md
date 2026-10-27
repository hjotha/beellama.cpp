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
- `tools/xbox/tests/test_rkva_decode_rowlane_profile.py` (updated for the shared
  counter classifier; behavior unchanged)
- `docs/rkva-prefill-gqa2.md`, `docs/rkva-final-optimizations.md`, this report

The MQ2/MQ4/INT8 HLSL wrappers and generated headers were not edited.

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

The shipped MQ2/MQ4/INT8 headers were **not** regenerated. Their recorded source
hash still names the pre-experiment kernel (`e52f25...`); the real-DXC blobs
proved byte-identical, and the default preprocessing token stream is unchanged.
A Windows-side regeneration may therefore change only that provenance line, but
it must be verified rather than assumed. The Python suite pins the current bytes
so an accidental regeneration fails loudly.

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
# Ran 9 tests ... OK

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

The C++ suite covers eligibility for nq 2..9, every boundary (nq1/nq10, mode
1/3, rev1/2, non-KVarN4, MQ4, head_dim, GQA, zero/other tail, non-wave64,
missing PSO), opt-out, INT8 conflict, `DispatchX = 3 * n_head_kv`, kernel name,
bucket key, counters (including real bulk classification) and the unchanged
default JSON (`enabled:false`, zero GQA2 calls).

Python discovery:

```bash
OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s tools/xbox/tests -p 'test_rkva*.py'
# Ran 301 tests ... OK

OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s tools/xbox/tests -p 'test_run_rkva*.py'
# Ran 21 tests ... OK

OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s tools/xbox/tests -p 'test_gpupeak*.py'
# Ran 5 tests ... OK
```

The broad `test_*.py` discovery runs 377 tests and still reports the two known
unrelated errors (`test_make_shape_gguf.py` cannot find
`tools/xbox/scripts/make-shape-gguf.py`; `test_validate_mtp_protocol.py` imports
a missing `scripts.validate_mtp_protocol`); no other failure was reported.

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
4. **Numerical/hardware qualification**: OUT/LSE parity against the default MQ2
   tile, then pass1 timing/prefill throughput and stability on the Xbox. No
   speedup, TPS delta or bit-exactness is claimed here.

## Non-claims

No GPU, Xbox, Device Portal, LAN, SSH, service, model or Windows build was used.
No merge or push was performed. `rkva-mq-gqa2.flag` remains off by default and is
not part of the accepted optimization set.
