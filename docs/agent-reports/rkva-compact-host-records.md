# RKVA compact host records (CUDA-only)

Branch `perf/rkva-compact-host-records-20261027`, based on fetched
`/home/hjotha/beellama.cpp` `main` `8bad8e84e67e9fcd43d12f637962b87685e0fecc`
("xbox: validate RAW quality captures with frozen provenance and controls").
No worktree was used (project exception for beellama); the branch was created
directly from `FETCH_HEAD` in `/home/hjotha/beellama-agents/quality-gate` and
all existing branches were preserved. Root files, refs, libraries, models,
configuration and artifacts were not touched.

## Scope

- `ggml/src/ggml-remote-attn/rkva-kvarn-producer.h` (existing; ~26 production
  lines changed plus comments)
- `tests/test-rkva-kvarn-record-storage.cpp` (new; 348 lines)
- `tests/CMakeLists.txt` (one `llama_build_and_test` line for the new target)
- `docs/agent-reports/rkva-compact-host-records.md` (this report)

## Design

The CUDA prepare helper (`ggml_backend_cuda_kvarn_native_append_v1`) already
returns compact closed-group records relative to `prepared_begin_`
(`prepared_rk_`/`prepared_rv_`). The host sink nevertheless allocated
`ceil((origin + capacity) / 128)` global groups, although only groups at or
after `origin / 128` can ever be published:

- `record_base_group_ = cuda_ ? origin_ / 128 : 0`, resolved once in the
  constructor after the CUDA proc lookup (`cuda_` remains immutable).
- Allocation is now `(global_groups - record_base_group_) * physical_heads *
  record_bytes`.
- Only two accesses subtract the base: the CUDA destination copy in `append`
  (`dst = (first_group - record_base_group_) * ...`) and the payload record
  slice (`record_offset = (origin_ / 128 + record_begin -
  record_base_group_) * ...`). The CPU path keeps base `0`, so its canonical
  store view and `ggml_kvarn_store` record tensor still cover the full global
  span.
- `record_span_valid()` rejects the append destination span and the replay span
  before any scratch resize, record copy or payload emission. That makes a
  rejected append atomic and removes any chance of unsigned underflow when the
  compact base is subtracted.

Unchanged on purpose: global positions passed to the CUDA kernel,
`prepared_begin_` and the `src` offset inside `prepared_rk_`/`prepared_rv_`,
stage/tail/rewind/original-F32 behaviour, and the wire protocol
(`live_count`, `record_begin`, `record_groups`, `append_begin/rows`, flags).
`record_begin` remains an absolute group count from `origin`, so a RESET
replay (`record_begin == 0`) reads host buffer index `0`, which now corresponds
to group `origin / 128`. No environment knob, math change or ABI change was
added. Non-multiple-of-128 capacities stay valid because the compact allocation
is exactly `ceil(capacity / 128)` groups.

## Calculated host-record savings

Geometry from the task: P = 102400 prefix tokens (800 groups), Hkv = 4
(8 physical heads), 8960 record bytes, K and V, 16 layers. Bytes per group
across K+V = `8 * 8960 * 2 = 143,360`. These numbers are arithmetic from the
allocation formula, not measured RSS or throughput; no memory or t/s claim is
made here.

| Context (groups) | Global records, 16 layers | Compact records, 16 layers | Saved |
| --- | --- | --- | --- |
| 800 prefix groups (P=102400) | 1,835,008,000 B (1750 MiB) | 0 | 1,835,008,000 B (1750 MiB) |
| C=131072 (1024 groups) | 2,348,810,240 B (2240 MiB) | 513,802,240 B (490 MiB) | 1,835,008,000 B (1750 MiB) |
| C=160000 (1250 groups) | 2,867,200,000 B (2734.375 MiB) | 1,032,192,000 B (984.375 MiB) | 1,835,008,000 B (1750 MiB) |

Fixture geometry (`heads=1`) saves 28,672,000 B per `origin=102400,
capacity=640` CUDA scenario (805 global groups vs 5 compact) and 35,840 B per
`origin=128` scenario (6 vs 5 groups); `origin=0` is unchanged because the base
is zero. The additional `origin=102400, heads=4` scenario saves exactly
114,688,000 B, i.e. one production layer of the P=102400 prefix above.

## New fixture

`tests/test-rkva-kvarn-record-storage.cpp` uses only the public existing
`rkva_kvarn_producer` surface: constructor, `host_bytes()`, `prepare()`,
`append()`, `rewind()`/`can_rewind()`, `rows()`. There is no test-only
constructor, injection knob or production hook. The include is
`#include RKVA_KVARN_PRODUCER_HEADER`, defaulting to the in-tree header and
overridable so root can compile the same source against a frozen old header:

```sh
mkdir -p /tmp/rkva-old
git -C /home/hjotha/beellama.cpp archive 8bad8e84e ggml/src ggml/include include common vendor | tar -x -C /tmp/rkva-old
cd <cuda-build-dir>
c++ $(sed -n 's/^CXX_DEFINES = //p' tests/CMakeFiles/test-rkva-kvarn-record-storage.dir/flags.make) \
    $(sed -n 's/^CXX_INCLUDES = //p' tests/CMakeFiles/test-rkva-kvarn-record-storage.dir/flags.make) \
    $(sed -n 's/^CXX_FLAGS = //p' tests/CMakeFiles/test-rkva-kvarn-record-storage.dir/flags.make) \
    -DRKVA_KVARN_PRODUCER_HEADER='"/tmp/rkva-old/ggml/src/ggml-remote-attn/rkva-kvarn-producer.h"' \
    <repo>/tests/test-rkva-kvarn-record-storage.cpp -o /tmp/test-record-old \
    -Wl,-rpath,$PWD/bin -L$PWD/bin -lllama-common
```

### Modes

- No `GGML_RKVA_GPU_TEST`: forces `GGML_RKVA_NATIVE_PRODUCER=cpu`, never calls
  `ggml_backend_load_all()`, checks the global allocation formula for origins
  0/128/102400 at capacity 513, replays packets against one canonical
  `ggml_kvarn_store` batch (byte-exact global record IDs), and runs the CPU
  packet grid (10 scenarios, 820 packets).
- `GGML_RKVA_GPU_TEST=1`: loads the CUDA backend, sets
  `GGML_RKVA_NATIVE_PRODUCER=cuda`, and runs 15 scenarios / 1230 packets,
  including one production-width (Hkv=4) origin P102400 case.
- `--record DIR`: writes one binary packet per append plus `manifest.txt`
  (name and byte size) and `memory.txt` (geometry and ctor `host_bytes`).
- `--compare DIR`: asserts the manifest set/sizes match, then `std::memcmp`s
  every packet, and compares `memory.txt` geometry while reporting
  `reference_host_bytes`, `new_host_bytes` and `saved_bytes`. Host bytes are
  intentionally excluded from packet equality so old (global) and new
  (compact) runs can be compared.
- `--expect-memory global|compact`: asserts the constructor-time `host_bytes`
  against the explicit formula before any transient buffer, in record and
  compare mode. `compact` can only pass with the CUDA producer; it aborts on
  the CPU path, which proves the CPU global span was not compacted.

Scenario coverage: origins 0/128/P102400, tails 0/128, compact-stage ON/OFF,
`prepare(120)` then eight-row appends across the 128 seal, `prepare(256)`, an
unaligned one-row append, a 127-row append closing several groups, a late RESET
that replays the whole committed prefix and must reproduce the first published
group bytes, `rewind(385)` with 16 eight-row reappends, a rejected
`rewind(384)` (129 rows) that must leave `rows()` unchanged and be followed by
a rewind/reappend sequence identical packet-for-packet, and capacity 513
(non-multiple of 128) including exact full-capacity append. `compact_stage`
follows the production gate (origin >= 128 and tail == 128). The existing
`tests/test-rkva-kvarn-gpu-producer.cpp` 80-packet export and its assertions
are untouched and compile against the new header.

## Local CPU validation actually run

Build (CPU only, outside the repo tree):

```sh
cmake -S . -B /tmp/opencode/qg-cpu-build -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF
cmake --build /tmp/opencode/qg-cpu-build \
  --target test-rkva-kvarn-record-storage test-rkva-kvarn-gpu-producer test-rkva-kvarn-producer -j 8
ctest --test-dir /tmp/opencode/qg-cpu-build -R rkva-kvarn --output-on-failure
```

Result: `100% tests passed out of 3` — `test-rkva-kvarn-producer` 0.32 s,
`test-rkva-kvarn-gpu-producer` 0.01 s (skips without `GGML_RKVA_GPU_TEST`),
`test-rkva-kvarn-record-storage` 0.29 s. The new target compiles without
warnings under the project flags and `ggml-remote-attn.cpp` (the production
caller) rebuilds against the modified header.

Fixture runs:

```sh
/tmp/opencode/qg-cpu-build/bin/test-rkva-kvarn-record-storage
/tmp/opencode/qg-cpu-build/bin/test-rkva-kvarn-record-storage \
  --record /tmp/opencode/kvarn-rec-new2 --expect-memory global
/tmp/opencode/qg-cpu-build/bin/test-rkva-kvarn-record-storage \
  --compare /tmp/opencode/kvarn-rec-new2 --expect-memory global
```

Output: `PASS CPU host records retain the global group span for the canonical
store`, `PASS CPU packets replay the canonical global record bytes`,
`RECORD packets=820 scenarios=10`, `COMPARE byte_exact=1 packets=820
scenarios=10`, all `saved_bytes=0` (CPU is intentionally uncompacted).
Negative control: `--expect-memory compact` on CPU aborts at the expected
`host_bytes` assertion.

Frozen old header cross-check (same test source, base `8bad8e84e` header):
old record → new compare and new record → old compare both print
`COMPARE byte_exact=1 packets=820 scenarios=10`, so the fixture itself and the
CPU wire path are unchanged between headers.

## Pending at root (not run here)

No GPU, model, Xbox, LAN, SSH, systemctl, MSVC or CUDA compilation was used.

1. Old-vs-old determinism: old-header binary `--record DIR_A
   --expect-memory global`, then `--record DIR_B` and `--compare DIR_A`
   (the control required before attributing equality to the change).
2. Old-vs-new bytes: new binary `--compare DIR_A --expect-memory compact`
   (must assert the compact formula and all 1230 packets); optionally
   `--record DIR_C` on the new binary and `--compare DIR_C --expect-memory
   global` with the old binary.
3. Rejected expectations as negative controls: new binary
   `--expect-memory global` and old binary `--expect-memory compact` must both
   abort on the memory assertion while still writing/recording identical
   packets.
4. Real model: same frozen P snapshot/new library, compare logits and process
   RSS on the same model and command. The fixture only qualifies packets and
   the constructor-time host allocation; it is not an RSS or tokens/s result.
5. Full native overflow snapshots remain unsupported; neither the guard nor a
   codec was changed.

Root/Astra should review the constructor base resolution, the two intentional
subtractions, and that both spans are validated before mutation.

Astra found a fixture naming collision before any GPU GOLD was recorded:
Hkv1/Hkv4 at P102400 used the same tag. The tag now includes heads; the CPU
startup check builds the actual GPU scenario list and verifies all15 tags are
unique. Packet names and loaded manifests reject duplicates; record mode refuses
nonempty output directories. Root CPU checks recorded/compared820 packets and
confirmed that nonempty-directory and duplicate-manifest negative controls fail
without overwriting the original manifest. No ambiguous GPU baseline was used.
