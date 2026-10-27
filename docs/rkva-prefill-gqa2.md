# RKVA GQA2 prefill candidate (`attn_kvarn4_mq2_gqa2`)

Status: **offline integration only.** The DXIL blob below is a real DXC
(Windows SDK 26100 / DXC 1.8) artifact that was compiled outside this host; it
is embedded from that artifact, not produced here. No MSVC/UWP build, D3D12 PSO
creation, Xbox run, GPU measurement or quality result exists for this candidate
yet. It is not a performance claim.

* Wrapper: `tools/xbox/shaders/rkva/rkva_range_attn_kvarn4_mq_gqa2.hlsl`
  (`RKVA_MQ_HEADS_PER_GROUP 2`, `RKVA_MQ_QT 2`, `RKVA_MQ_I8 0`) over
  `rkva_range_attn_kvarn4_mq.hlsl`.
* Generated header:
  `tools/xbox/shaders/generated/rkva_range_attn_kvarn4_mq2_gqa2_dxil.h`
  (47 924 B DXIL, SHA-256
  `b26dd85cb5512225dcb10333fc53a7a7b52676ccac2d480ed237fdcf92565a99`).
* Build target: `tools/xbox/scripts/compile-rkva-range-shaders.sh
  attn_kvarn4_mq2_gqa2` (`cs_6_0`, entry `CSRangeAttnKvarn4Mq`).
* Flag: LocalState file `rkva-mq-gqa2.flag` (default off), read on every RKVA
  HELLO.

## What changes

The shipped 2-query tile keeps six local Q heads per group (24 query heads over
4 KV heads, `RKVA_GQA == 6`). The candidate keeps every global stride, the
128-thread group, the record/stage/tail classification, WHT256, online softmax,
the `[split][query * n_head + head][258]` partial layout, the 12 cbuffer
constants, the four SRVs, the two UAVs and the existing reducer. Only the local
head count changes from six to two, so the host must dispatch three consecutive
x groups per KV head:

```
DispatchX = 3 * n_head_kv
kv_head   = group.x / 3
head_base = 2 * (group.x % 3)
global    = kv_head * 6 + head_base + g   (g = 0, 1)
          == 2 * group.x + g
```

Each global head still writes exactly one partial row. The candidate reloads and
dequantizes each K/V row for three groups instead of one; whether that trade is
profitable is **UNKNOWN** until hardware replay.

## Selection and fail-closed behavior

`RemoteAttnEngine::attend_range()` selects the dedicated PSO only when **all**
of the following hold:

1. `rkva-mq-gqa2.flag` was present at HELLO and the INT8 flag was not.
2. Descriptor revision 3 (serial query prefix), native KVarN4 storage.
3. Mode 2 (fine) selected the 2-query MQ tile.
4. `nq` is 2..9 (nq1 decode is untouched).
5. `head_dim == 256`, GQA 6 (`n_head == 6 * n_head_kv`) and the exact 128-token
   precision tail.
6. Wave lanes are exactly 64 on both min and max.
7. The generated GQA2 PSO exists and was created successfully.

Every other call keeps the qualified default `attn_kvarn4_mq2` kernel. A missing
PSO, an unqualified shape, a failed `CreateComputePipelineState`, or a
non-wave64 device never falls into the candidate.

### Flag conflict with INT8

`rkva-mq-gqa2.flag` and `rkva-mq-i8.flag` are mutually exclusive. If both are
present, **GQA2 stays off** (fail-closed), INT8 keeps its own existing selection,
and the profile reports
`"mq2_gqa2": {"enabled": false, "status": "disabled: conflicts with
rkva-mq-i8.flag (fail-closed)"}`. Remove one flag and send a new HELLO to change
the arm.

The INT8 PSO is checked first in the dispatch path as defense in depth, so an
engine that was mis-armed with both booleans still runs INT8, never GQA2.

## Profile identity

* Bucket: `attention/kvarn4-prefix/optimized-kvarn4_mq2_gqa2/<nq>/<rows>/warm`.
* Counter: `kvarn4_multi_query.mq2_gqa2.calls`, a subset of `mq2.calls`; it is
  incremented only when the GQA2 PSO actually ran, and never as an INT8 call.
* JSON object: `kvarn4_multi_query.mq2_gqa2.{available, enabled, status, calls,
  heads_per_group, dispatch}`.
* `Sample.gqa2` / `BulkSegmentProfile.gqa2` mark the actual activation.
* `Sample.groups` / `BulkSegmentProfile.groups` count the real dispatch:
  `3 * n_head_kv * query_groups * split_k` for this candidate, unchanged
  `n_head_kv * query_groups * split_k` otherwise.

Ordinary and native-bulk samples share `rkva_rprof::count_attention()`, so the
same classification drives both paths.

## Build and offline validation

Generate the header from an already-compiled blob with the same renderer the
compile script uses:

```bash
python3 tools/xbox/scripts/render-rkva-dxil-header.py \
  --binary /path/to/gqa2.dxil \
  --output tools/xbox/shaders/generated/rkva_range_attn_kvarn4_mq2_gqa2_dxil.h \
  --source tools/xbox/shaders/rkva/rkva_range_attn_kvarn4_mq_gqa2.hlsl \
  --target attn_kvarn4_mq2_gqa2 --profile cs_6_0
```

The header records the wrapper hash, the included kernel hash and the target.
Regenerating the shipped MQ2/MQ4/INT8 headers was deliberately avoided; their
bytes are pinned by `tools/xbox/tests/test_rkva_mq_gqa2_layout.py`.

CPU/offline checks (no GPU, Windows, LAN or SSH):

```bash
python3 tools/xbox/tests/test_rkva_mq_gqa2_layout.py
g++ -std=c++17 -Wall -Wextra -Werror -O1 \
  tools/xbox/tests/rkva_range_profile_cpu.cpp -o /tmp/rkva_profile
/tmp/rkva_profile > /tmp/rkva_profile.json
python3 -m json.tool /tmp/rkva_profile.json > /dev/null
```

## Pending qualification

1. Compile the wrapper with the real DXC on the Windows build host and compare
   its DXIL hash/output with the embedded artifact (determinism is not assumed).
2. Compile the UWP target with MSVC and confirm the PSO is created at HELLO.
3. On the Xbox, verify OUT/LSE quality against the default MQ2 kernel before any
   timing comparison, then measure pass1 time and prefill throughput.
4. Keep the flag off in user builds until those results are accepted.
