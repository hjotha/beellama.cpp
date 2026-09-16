# Unified KV snapshots with bounded payload staging

The upward handoff to a fixed no-MTP child and the opt-in automatic disk cache
now use the same canonical native target sequence file, followed by a bounded
checksummed manifest. A private temporary file is closed, checked, synced, and
renamed over the destination only after the complete snapshot is ready. The
optional regenerate logits are embedded in that same file and bound by the
manifest; new `.meta` or `.logits` files are not created. Failed preflight or
publication leaves the previous valid pathname untouched.

The public model ID remains the route group. Internal short/medium/long
transitions still use the existing RAM prompt-cache snapshots, including draft,
carry and checkpoints. Explicit adaptive slot files retain their version-1
format, model/profile validation, rollback and five-copy RAM reservation.

## Store separation and conversation contract

An upward handoff is associated by `X-Conversation-Id`; requests without that
header cannot be transferred between child processes and fall back to normal
routing/prefill behavior. The public model ID remains the route-group ID on all
children.

For the intended ladder, 32768 -> 56320 -> 97536 stays on the existing adaptive
RAM snapshots. The 97536 -> 104448 target-only transfer and automatic cache
entries use the same configured persistent store, index and LRU. The router
passes `LLAMA_SERVER_ROUTER_STATE_DIR` internally to route children and sends an
internal `route_state_transfer` marker; `route_target_no_mtp` selects the
streaming format only. An explicitly configured child `--slot-save-path` is
never replaced on POSIX, so `--slot-save-auto` remains authoritative for the
same store. Route cleanup drops only the in-memory conversation reference and
temporary files; valid snapshots survive child/router shutdown and are reclaimed
only by the shared retention policy. Without an explicit slot store, POSIX
route members retain the legacy public `/slots` behavior through a private
fallback directory, whose valid snapshots are also not removed by shutdown.

## Explicit q4_0 -> compact conversion

A third long tier (KVarN4/KVarN4, b/ub 64, no MTP, candidate 114688) extends the
same public route group. The 104448 q4 tier now carries `route-max-tokens =
104448`, so larger budgets are routed to the compact tier instead of falling
through an uncapped member.

Crossing that boundary does not repeat the prefill. The saved q4_0/q4_0 route
snapshot is converted into the destination child's native compact layout:

- `llama_state_seq_convert_file` / `llama_state_seq_convert_data` accept a
  canonical native state or the raw `state_seq_get_data` form, verify the
  XXH64 checksum when given, and write a canonical compact state file
  (temporary file, fsync, atomic rename). The source is never modified.
- Parsing is deliberately restricted: one KV stream, a contiguous single
  sequence prefix at positions `0..n-1`, untransposed q4_0 K and V rows with
  the exact per-layer row sizes. Anything else is rejected without output.
- Conversion is CPU-side and bounded: one 128-token group per component, with
  per-group rotated tiles (Hadamard per 128-dim head slice plus the cross-slice
  step), F16 rounding exactly where the runtime stage rounds, Sinkhorn
  normalization and the native record layout. No second full cache is
  materialized in RAM or VRAM.
- The native state is rebuilt completely: completed groups become records, the
  sink and live tail groups become F16 stage rows, the exact-tail payload rows
  are rotated F16 rows for the last `exact_tail_tokens` positions, and the
  metadata mirror is serialized by the destination's own writer so manifest,
  body and tail records match a native save byte for byte in structure.
- Hybrid models keep their recurrent/conv state untouched: the trailing
  recurrent section travels verbatim and its byte size is validated against the
  destination's recurrent layout.
- Publication is transactional. The destination is only mutated by the normal
  streaming restore of the converted file, which requires an empty context and
  clears it on any late failure.

The conversion is lossy. The error introduced by the q4_0 source remains and
requantization can add more; a converted prefix is not equivalent to a native
compact prefill and no quality equality is claimed.

Converted snapshots are persisted in the automatic store under the same
deterministic name scheme, wrapped in the canonical envelope, and record
provenance under the manifest's `converted` member (`converter` version,
`source_path`, `source_checksum`, `source_format`, `target_format`). They are
never confused with native snapshots: later restores reuse the converted file
without converting again, including after a router restart, where the KVarN
child's unknown-representation layout is matched by the automatic index through
`common_prompt_cache_layout_reusable`. `route_slot_state` reports the
conversion in the child log (`route snapshot converted: ... target=...`).

## Restore contract

`llama_state_seq_load_file_streaming` is the common restore mechanism for
canonical route and automatic snapshots. It accepts only an entirely empty
context. Nonempty contexts are rejected without mutation. A shared store lock
and striped reference lock are held from manifest inspection through the native
streaming commit, so a publisher cannot replace the version being read. A late
read/checksum failure or throwing publication callback clears the initially
empty memory before returning zero; the server publishes neither prompt nor
cache hit.

1. The server reads a manifest of at most 64 KiB and checks its checksum, native
   header, token count, positions, file size, source/destination capacity, and
   configured disk/RAM budgets.
2. Model identity is SHA-256 over the actual model shards and effective typed
   metadata overrides, prepared before loading and verified against source
   stamps afterward. The target must match that identity and the effective
   attention/cache/RoPE layout. Batch sizes and context capacity may differ.
3. Index construction and prefix lookup read only the bounded footer/header,
   token IDs and identity/layout metadata. Boundary entries are deduplicated by
   `state_path`; the same 100k-token snapshot is not payload-hashed once per
   boundary. The single candidate selected for reuse is then checked by the
   native streaming loader with its XXH64 checksum before any state commit.
   Direct/full readers retain an optional complete streaming checksum pass.
   All memory publication callbacks remain pending until the selected payload is
   valid.
4. Commit re-reads and verifies each chunk before installing it. Metadata is
   published after all tensor writes succeed. Tokens/positions are checked
   again before publishing the server prompt. Embedded logits, when present,
   are accepted only with matching vocabulary, token count and checksum.

Checksums detect corruption; they do not authenticate files against a writer
with access to the private directory. The router's directory must remain trusted.

The transfer buffer is 8 MiB on save and restore, independent of tensor size.
The restore index and synthesized-row staging each have an 8 MiB logical cap
(up to twice that capacity during vector growth); standard q4 handoff does not
use synthesized-row staging. Context metadata and token vectors scale with the
configured context, not the multi-GB KV payload. The server reserves
`16 MiB + 16*n_ctx` for save and `48 MiB + 512*n_ctx` for restore against both
available host RAM and the global prompt-cache budget. These reservations do
not replace the existing budgets for RAM or explicit adaptive snapshots.

Streaming handoff currently supports one text slot, standard KV without
precision tails, no adapters/control vectors, and a fixed no-MTP destination.
KVarN/compact/tail representations retain their existing APIs. POSIX publication
and model identity are required; conventional Windows route groups keep their
existing behavior.

## Generation and checkpoints

Only evaluated tokens are saved: the final sampled token is usually not yet in
KV. For `93440 + 4096`, the expected evaluated prefix is therefore normally
97535, not 97536. The destination evaluates that last sampled token as part of
the new suffix. Save and restore counts must agree exactly. Automatic caching
also saves a dedicated evaluated prompt branch point before generation. For
managed MTP, the final prompt token is separated into its own decode batch and
the persistent object contains the valid `N-1` target state; restart restores
that prefix, evaluates one real suffix token, and bootstraps draft/carry before
speculation. This avoids rewinding a full recurrent state whose rollback plan
was not persisted. FULL/no-MTP models keep the complete prompt and may bind
last-prompt logits into the same file. An MTP snapshot that is not strictly
shorter than the request is not selected for restore. No one-token shortcut
bypasses the original checkpoint or `!slot.can_speculate()` guard.

An entire restored prefix followed by new tokens can append to recurrent state
without rewinding it. New prompt checkpoints are created by the normal suffix
processing path. A target-only MTP restore marks draft/carry as pending and
must complete the existing bootstrap/reconstruction decode before generation;
it is not reported as a usable hit if that bootstrap fails. The route file does
not carry historical branch checkpoints or draft/carry data beyond the target
state and optional logits; a request diverging inside a recurrent snapshot may
need a cold prefill. Explicit adaptive snapshots still preserve their
historical checkpoints and MTP state. Reverse interchild migration is not
added.

The recurrent parser also restores original metadata on a failed metadata
parse, instead of clearing a live sequence. PLE payload placement now uses the
prepared restore head consistently with the other recurrent tensors.

## Reproducible local tests

Build with moderate parallelism:

```sh
cmake --build build-bench --parallel 2 --target llama-server test-state-file-stream \
  test-save-load-state test-recurrent-state-rollback \
  test-server-prompt-checkpoint test-server-model-identity
```

`test-state-file-stream` is registered with the generated-model fixture and a
build-local `TMPDIR`. It exercises a >24 MiB tensor and real late truncation,
q4 512/b256 -> 1024/b64 after generation, deterministic continuation, corrupted
payload/checksum, truncation, incompatible layout, nonempty destination,
recurrent parse transactions, injected late I/O, retry, and publication cleanup.

```sh
CUDA_VISIBLE_DEVICES= GGML_DISABLE_VULKAN=1 ctest --test-dir build-bench \
  --output-on-failure -R '^(test-state-file-stream|test-server-model-identity|test-server-prompt-checkpoint)$'
```

The HTTP harness needs only the Python standard library. CPU mode creates a
private copy of `qwen35-mtp.gguf` with token pieces (the original generated tensor
fixture has no tokenizer), and overrides its training context for the scaled
matrix. It never downloads fixtures or loads cached model presets. Use a new
work directory on disk, not a full tmpfs:

```sh
TMPDIR=/path/on/disk python3 tools/server/tests/router_state_streaming.py \
  --server build-bench/bin/llama-server \
  --model build-bench/tests/test-models/qwen35-mtp.gguf \
  --work-dir /path/on/disk/cpu-run
```

The matrix uses 256/512/768/1024 with 64 generated tokens in CPU mode, and
32768/56320/97536/104448 with 4096 generated tokens under explicit `--gpu`.
Every migration's limit request must have a cache hit. Repetition and
continuation have independent hit assertions; CPU also compares continuation
against a cold oracle and exercises explicit adaptive files and HTTP rejection
of a nonempty streaming destination. The chain is reconstructed with the actual
returned token IDs before each migration. Repriming after branching probes is
recorded separately and may legitimately prefill when a checkpoint was evicted.

GPU mode uses cp=1, default min-step=8192, 8 threads, b/ub256 -> 64,
q4/q4, and the preset's 200/170 W and decode-memory-clock 10501 governors.
The public ID defaults to the literal model path. Production is never modified
by the matrix itself. An explicitly authorized root wrapper is provided for
an exclusive test window; it checks that production is idle, stops the named
service, runs the matrix on 8091 and restores the original service in `finally`,
including SIGINT/SIGTERM. It verifies health and the unchanged unit afterward:

```sh
sudo python3 tools/server/tests/router_state_gpu.py \
  --service llama-server-root.service --artifacts /path/on/disk/gpu-run -- \
  --server /absolute/path/build-bench/bin/llama-server --model /absolute/path/model.gguf
```

The conversion tier has dedicated harnesses. The CPU ladder uses the generated
qwen35 fixture with scaled contexts and covers routing boundaries, the
wide -> KVarN conversion, provenance, reuse, restart reuse and the earlier
profiles:

```sh
TMPDIR=/path/on/disk python3 tools/server/tests/router_state_convert_cpu.py \
  --server build-bench/bin/llama-server \
  --model build-bench/tests/test-models/qwen35-mtp.gguf \
  --work-dir /path/on/disk/convert-cpu
```

The real-model GPU run drives one conversation up the full ladder (97520 -> 100352 -> 110592),
requires the converted prefix restore (`cache_n` from the imported prefix, no
full prefill), repeats it for reuse, compares against a cold compact prefill,
and restarts the router for disk reuse. It runs through the same exclusive
window wrapper with `--harness router_state_convert_gpu.py --harness-raw`:

```sh
sudo python3 tools/server/tests/router_state_gpu.py \
  --service llama-server-root.service --artifacts /path/on/disk/convert-gpu \
  --preset-source /home/hjotha/qwen-3.8-27b-q4-mixed-20260916.ini \
  --harness router_state_convert_gpu.py --harness-raw -- \
  --server /absolute/path/build-bench/bin/llama-server \
  --model /absolute/path/model.gguf
```

`tests/test-state-convert-q4-kvarn.cpp` is the focused unit test: file and RAM
sources must produce identical bytes, prefixes around 128-token boundaries
(1/127/128/129/255/256/257/300) must restore and continue, an independent
reference re-derives the rotated F16 stage rows, the exact-tail rows and the
K/V record orientation from the raw q4 rows, the converted state must survive a
native save/load roundtrip, and corruption, truncation, unsupported formats,
nonempty destinations and truncated recurrent sections must be rejected without
output or source mutation.

The focused GPU handoff smoke avoids repeating the full 4096-token ladder while
still exercising the large file: `97535 + 1` on tri followed by `100352 + 16`
on wide. It uses the same wrapper with `--handoff-smoke`.

```sh
sudo python3 tools/server/tests/router_state_gpu.py \
  --service llama-server-root.service --artifacts /path/on/disk/gpu-handoff-smoke -- \
  --server /absolute/path/build-bench/bin/llama-server --model /absolute/path/model.gguf \
  --handoff-smoke
```

For store separation, use the optional CPU smoke with `--auto-cache`; for a
real restart hit, use `router_state_auto_restart.py`. The complete scaled CPU
A--C matrix is `router_state_unified_cpu.py`; it uses fresh PIDs, no conversation
header for automatic restore, exact timing invariants, MTP bootstrap evidence,
and cold output controls. The legacy route smoke covers different
source/destination stores and the no-explicit-store fallback:

```sh
TMPDIR=/path/on/disk python3 tools/server/tests/router_state_streaming.py \
  --server build-bench/bin/llama-server \
  --model build-bench/tests/test-models/qwen35-mtp.gguf \
  --work-dir /path/on/disk/auto-cache-cpu --auto-cache

TMPDIR=/path/on/disk python3 tools/server/tests/router_state_auto_restart.py \
  --server build-bench/bin/llama-server \
  --model build-bench/tests/test-models/qwen35-mtp.gguf \
  --work-dir /path/on/disk/auto-restart-cpu

TMPDIR=/path/on/disk python3 tools/server/tests/router_state_unified_cpu.py \
  --server build-bench/bin/llama-server \
  --model build-bench/tests/test-models/qwen35-mtp.gguf \
  --work-dir /path/on/disk/unified-cpu-a-c

python3 tools/server/tests/router_state_legacy.py \
  --server build-bench/bin/llama-server \
  --model build-bench/tests/test-models/qwen35-mtp.gguf \
  --work-dir /path/on/disk/legacy-route-cpu
```

Both harnesses retain commands, logs, JSON results and process-tree RSS/HWM
samples. Test-only fault variables, unset in ordinary operation:
`LLAMA_TEST_STATE_FILE_COMMIT_FAIL_AFTER`, `LLAMA_TEST_ROUTE_STATE_PUBLISH_FAIL`
and `LLAMA_TEST_SNAPSHOT_LOOKUP_TRACE`. The last one emits boundary-entry versus
unique-path counts and confirms that payload validation is deferred to the
selected restore.

## Promotion and rollback record

The reviewed unit remains checked in as
`tools/server/llama-server-root.router-proposed.service`. Its frozen release
copy is `/home/hjotha/releases/beellama-router-kv-streaming-20260916-r1/llama-server-root.service`,
and that copy is now installed at `/etc/systemd/system/llama-server-root.service`.
Promotion completed on 2026-09-16 at `14:10:40 CEST` after the service passed
the idle check, health check, model listing, request/cache check, and journal
inspection.

The release contains an independent `cp -a --reflink=auto` copy of all regular
files and symlinks from `build-bench/bin`, including `llama-server`,
`libllama-server-impl.so`, `libllama.so`, and the ggml CPU, CUDA, Vulkan, base,
common and mtmd libraries. The complete source/staged inventory, base commit,
diff hashes and unit hashes are recorded in
`/home/hjotha/releases/beellama-router-kv-streaming-20260916-r1/RELEASE-MANIFEST.txt`.
The staged `llama-server` hash is
`dc406014df74ba5301cd7ea3f6daf01441bb12d4815591993f9dd7bcb3cf44bf`,
`libllama-server-impl.so` is
`0a72436a3f3be67361f6af9452e37a85d2939ca186445f532f5e2f7d7067f368`, and the
frozen INI is
`/home/hjotha/releases/beellama-router-kv-streaming-20260916-r1/qwen-3.8-27b-q4-mixed-20260916.ini`
with SHA-256
`2bc8f0513c4ca025926f9c7d19fb26eba1884b29408dde7b35d77af8202e5899`.

The release unit uses the frozen INI as the authority for
`slot-save-auto=true` and the persistent
`slot-save-path=/home/hjotha/llama-slot-cache/`; it does not add a conflicting
CLI `--slot-save-path`. Route-transfer limits and cleanup remain private to
the router handoff directory. Output goes to journald (`journalctl -u
llama-server-root.service`); no shared/truncating `--log-file` is configured.
The public model ID remains the literal GGUF path.

The promotion smoke returned `{"status":"ok"}`, `/v1/models` exposed one
model with `context_window=104448`, and the first/repeated request reported
`cache_n=0/1`. The child loaded `qwen-3.8-27b-q4-tri`; its startup indexed
`2687` prefix boundaries from `/home/hjotha/llama-slot-cache/`, and the
production store contained `64` automatic state files after the check.

The exact rollback copy is
`/home/hjotha/router-kv-streaming.KllpY2/promotion-20260916/llama-server-root.service`
with SHA-256
`495e690bfdf692d0af831e3164c13126d2e26672d7f878ab431cb2e763ef53ad`.
The unchanged GPU drop-in is backed up at
`/home/hjotha/router-kv-streaming.KllpY2/promotion-20260916/llama-server-root.service.d/10-gpu-ready.conf` and
matches the live drop-in at SHA-256
`6f9dbc21cc4205460bc33d1ed6b825f0f994762e0760a93b4a265b2f9a757e54`.
An earlier staged smoke was automatically rolled back when its request
assertion omitted returned token IDs; rollback restored the original unit and
health, and the corrected smoke then passed. The durable command/result record
is `/home/hjotha/router-kv-streaming.KllpY2/promotion-20260916/PROMOTION-RESULT.md`.

## Validation record (2026-09-16)

Checkout: `/home/hjotha/beellama.cpp`, explicitly authorized `main`, base
`ec6d1e528`. No commit, push or merge; production promotion completed using
the versioned release above.
Artifacts: `/home/hjotha/router-kv-streaming.KllpY2`.

- `ctest.log`: 3/3 passed (state streaming, model identity, prompt checkpoints),
  with `--fixture-exclude-any generate-models` because local fixtures existed.
- `state-stream.log`: bounded transfer, after-generation deterministic
  continuation and negative/transaction tests passed.
- `cpu-matrix-4/results.json` and `cpu-matrix-4.log`: complete CPU chain passed;
  migration `cache_n` was 255, 511, 767. Each profile's repeat/continuation hit,
  cold oracle, explicit adaptive roundtrip, nonempty HTTP rejection, native
  roundtrip, and return to short passed.
- `recurrent.log`: qwen35 rollback, multi-sequence split replay, sequence
  independence passed with both zero and nonzero cache fill.
- `lifecycle-mtp-cache.log`, `lifecycle-mtp-state.log`: passed, including carry
  rejection tests and short/long/short MTP residency lifecycle.
- `save-load-gemma.log`: complete existing Gemma save/load suite passed.
- `save-load.log`: qwen35-mtp q4 fails existing test 7, on-device scatter blob
  byte equality. `baseline/save-load.log` reproduces the same failure with the
  two changed state implementation sources rebuilt from original `HEAD`, using
  the same flags, model and test command. Host scatter and tests 1–6 pass in both.
  This device-scatter issue is not addressed by the disk handoff change.
- `gpu-final/control.json` and `gpu-final/matrix/results.json`: full GPU ladder
  passed and the original service was restored with `after_state=active`, health
  `ok`, and `unit_unchanged=true`. The exact-limit `32768` request produced
  `4095/4096` with `context_limit`; the `28671+4096` margin produced all 4096.
  The `56320` transition used `cache_n=32766`, the `97536` transition used
  `cache_n=56319`, and the long-tier repeat/continue used `93436`/`93455`.
  The no-MTP transition used `100352+4096`, `cache_n=97535`, and returned the
  public model ID unchanged.
- `gpu-final/matrix/server.log`: the OOM/allocation/recovery scan found no OOM,
  CUDA allocation failure, or recovery event. It did contain the expected
  one-token boundary warning (`failed to find a memory slot for batch of size 1`)
  and two preventive CUDA-graph disables because `13.62 MiB < 18.00 MiB`.
- `gpu-final/matrix/memory.jsonl`: the original full-matrix sampler's global
  partial maximum was `VmRSS=2039120 KiB`, `RssAnon=1594196 KiB` (HWM
  `2173864 KiB`). Its restore segment saw only the router parent after the
  child eviction, so that file is explicitly incomplete for target-child RSS
  and must not be used as a full restore-memory measurement.
- `gpu-handoff-smoke/control.json`, `matrix/results.json`: the focused smoke
  passed `97535+1 -> 100352+16`, saved/restored exactly `97535` tokens, and
  transferred `1957391518` bytes with `cache_n=97535` and `prompt_n=2817`.
- `gpu-handoff-smoke/matrix/memory.jsonl`: the corrected all-TID sampler
  observed a global partial maximum of `VmRSS=1636044 KiB`, `RssAnon=808144 KiB`
  (HWM `1754836 KiB`). During the route-save window the aggregate peak was
  `VmRSS=1279636 KiB`, `RssAnon=661216 KiB`; during route-restore it was
  `VmRSS=1235960 KiB`, `RssAnon=501264 KiB`, with the first post-restore sample
  at `1319060/561000 KiB`. These are process RSS measurements, not the 8 MiB
  transfer-buffer size.
- `legacy-route-cpu-2/results.json`: legacy route save/restore passed with
  different explicit source/destination stores and with no explicit
  `--slot-save-path`; the speculative destination remained `draft-mtp`. The
  new focused run is
  `/home/hjotha/router-kv-snapshots-unificados-20260916/legacy-unified-final-2/results.json`;
  it confirms the route snapshot remains in the source/shared store and is not
  erased at shutdown.
- `auto-cache-cpu-5/results.json` is the pre-unification historical result with
  14 state/meta pairs. New auto-cache writes use the canonical envelope and no
  `.meta` sidecar.
- `/home/hjotha/router-kv-snapshots-unificados-20260916/auto-restart-unified-8/results.json`:
  a new second process restored the dedicated original prompt from DISK
  (`original_restart_cache_n=400`, `original_restart_prompt_n=0`), and a
  separate third process restored that same prompt before continuing with all
  32 generated IDs (`restart_continuation_cache_n=400`); both logs contain an
  explicit `auto-restore` event. Its budget cases prove protected count/byte
  publication rejection leaves the prior set/hashes unchanged, then LRU evicts
  after release.
- `/home/hjotha/router-kv-snapshots-unificados-20260916/state-stream-unified-final`
  and `lock-test-final` are the post-unification C++ streaming and
  interprocess-lock artifacts. The lock test covers live/dead readers,
  `O_CLOEXEC`, invalid store-lock acquisition, saturated candidates and bounded
  64-stripe identities.

The source candidate `/home/hjotha/qwen-3.8-27b-q4-mixed-20260916.ini` and its
frozen release copy enable `slot-save-auto` with
`/home/hjotha/llama-slot-cache/`. The release is now active on port 8090; the
original unit is preserved at the rollback path above and the persistent cache
directory was not replaced, redirected or cleaned by route-state cleanup.

## Unified snapshot WIP CPU checkpoint

The first MTP one-token attempt is intentionally retained as a rejected
artifact at `/home/hjotha/router-kv-snapshots-unificados-20260916/cpu-matrix-unified-final`
with its [review failure record](/home/hjotha/router-kv-snapshots-unificados-20260916/cpu-matrix-unified-final/REVIEW-FAILURE.md).
Its restore event was followed by `cache_n=0` and near-full prompt evaluation;
the restore-event count was not a valid reuse count.

The corrected CPU evidence is
`/home/hjotha/router-kv-snapshots-unificados-20260916/cpu-lookup-final`.
Managed MTP publishes the N-1 state before the final prompt token, and the
fresh-PID original-prompt cases report `cache_n=223,prompt_n=1` at scaled 256
and `cache_n=479,prompt_n=1` at scaled 512, with explicit `auto-restore` and
`target-only MTP bootstrap accepted`. Continuation cases preserve all generated
IDs and compare exactly with a no-cache cold control. The same artifact covers
the scaled `long` and `wide` profiles, the disk-only 256→512→768→1024 chain,
real fallback reasons for corrupt/truncated/model/layout/write failures, and
all-TID PID/RSS evidence.

The native C++ evidence is in
`/home/hjotha/router-kv-snapshots-unificados-20260916/state-final-lookup` and
`cpu-only-state-2`. It includes a non-degenerate cold-versus-N-1 restore
comparison of logits and eight recurrent suffix steps. Physical recurrent
bytes are allowed to differ because deserialization normalizes the recurrent
index; no full-state rewind is used. The lookup trace in
`/home/hjotha/router-kv-snapshots-unificados-20260916/lookup-trace-final`
shows `boundary_entries=400 unique_paths=1` and
`payload_validation=deferred_to_selected_restore`, proving metadata/token-only
lookup with one full validation at the selected restore.

This is WIP evidence only: GPU/D and promotion of the unified infrastructure
remain pending principal review. The phase-1 production release and its
persistent cache are not used by these tests.
