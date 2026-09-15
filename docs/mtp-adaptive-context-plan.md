# Plan: adaptive MTP disable based on a separate draft context size

> **SUPERSEDED — do not implement. See `docs/mtp-router-split-plan.md`.**
>
> The goal below (one process offering the full 97,280-token ceiling while still doing MTP
> under a smaller draft budget) is arithmetically impossible on this model/GPU. Re-deriving
> the numbers from `docs/kv-calibration-findings.md` shows the MTP draft KV is only
> ~1.2 KiB/token out of ~19.3 KiB/token (6.8%); the real cost of MTP is ~358 MiB of fixed
> context/workspace overhead plus ~334 MiB of extra weights in the `-mtp` GGUF. Capping the
> draft context therefore raises the target ceiling from 54,272 to **at most ~57,900** — and
> only by shrinking the draft budget to zero, i.e. by disabling the very speedup the plan
> exists to keep. The remaining ~39,300 tokens are fixed costs that only a process boundary
> can reclaim. Full derivation in section 1 of the router plan.
>
> The code-path analysis below stays on record because it is correct and non-obvious: the
> always-on `common_speculative_process()` shadow decode (which would have turned
> over-budget requests into 500s) and the five static-capability uses of `can_speculate()`.

Status: superseded design, not implemented. Revised after tracing the code paths end to end
(`common/speculative.cpp`, `tools/server/server-context.cpp`, `common/common.cpp`,
`common/arg.cpp`, `src/llama-context.cpp`) plus the calibration data in
`docs/kv-calibration-findings.md` (measured on GOKAYA, `192.168.1.57`).

Revision 2 changes three conclusions of the first draft; they are called out inline as
**[rev2]** so the reasoning is not lost.

## Goal

Run the server with two independent numbers:

```
--ctx-size 97280       # target model context (what the client can actually use)
--mtp-ctx-size 54264   # MTP draft context budget (smaller, bounded by VRAM)
```

Requests whose context stays under `mtp-ctx-size` get MTP speculative decoding.
Requests that grow past that threshold keep running normally, just without MTP,
up to the full `ctx-size`. No process restart, no model reload, no request failure.

This is confirmed useful by the existing calibration data: on the same model/GPU,
MTP-traditional tops out at ~54,272 tokens while non-MTP traditional reaches
97,280. Today, enabling `--mtp` forces every request onto the smaller number, even
though most requests never need more than ~54k tokens. Decoupling the two lets a
single server instance offer the full 97,280 ceiling while still getting the MTP
speedup (measured earlier: ~47 decode tok/s vs ~21 tok/s without it) for the requests
that fit.

## Why this is not a job for the multi-model router

`tools/server/server-models.cpp` manages child *processes*: separate ports, separate
weights loaded from disk, separate PID lifecycle. Reusing it here would mean spawning
a second full server process for the >54k case, duplicating model weights in RAM/VRAM,
adding multi-second process-spawn latency, and it can't help a request that starts
under the threshold and grows past it mid-generation (the router routes at request
start, not mid-stream). MTP today is already just an extra `llama_context` sharing the
target model's weights (`common_speculative_init_result`, no separate model load for the
common case where the target model has a built-in MTP head). The right level for this
feature is inside one server process, one model load, toggled per sequence.

## Root cause of today's coupling — two halves, not one

### (a) The draft context is sized off the target

`common/speculative.cpp:2584`, in `common_speculative_init_result::common_speculative_init_result()`:

```cpp
// the draft context holds as many tokens per sequence as the target context
cparams.n_ctx = llama_n_ctx(ctx_tgt);
```

The draft KV cache is always allocated at the full target size. This is the part the
first draft of this plan identified, and it is real.

### (b) The MTP shadow decode is unconditional **[rev2 — this is the actual blocker]**

`common_speculative_process()` is called on **every** decoded target batch, with no
per-slot and no per-sequence gate:

- `tools/server/server-context.cpp:3919-3931` — `if (spec) { ... common_speculative_process(spec.get(), batch_view); }`
- `tools/server/server-context.cpp:777` — same call from the mtmd image-chunk decode callback.

Inside `common_speculative_impl_draft_mtp::process()` (`common/speculative.cpp:1527-1622`)
that hook re-decodes *the same tokens at the same positions* into `ctx_dft`, for every
sequence present in the batch, including prompt prefill ubatches. It is not gated by
`drafting`, by `can_speculate()`, or by anything else.

Consequences, if only (a) is fixed:

- Prefilling an 80,000-token prompt pushes 80,000 rows into a draft KV cache with room
  for ~54,264 (the unified cache is capacity-based — there is no `pos < n_ctx` check, it
  simply runs out of cells), so `llama_decode(ctx_dft, batch)` returns non-zero
  (`common/speculative.cpp:1607-1613`).
- `process()` returns `false`, and the server does
  `throw std::runtime_error("failed to process speculative batch")`
  (`tools/server/server-context.cpp:3925-3930`, still marked `// TODO: handle error`).
- Net effect: the over-threshold request does not "run without MTP", it takes down the
  whole decode batch — every slot in it — with a 500.

So the feature is *not* "shrink the draft context and gate drafting". It is "shrink the
draft context, gate drafting, **and make the shadow decode skip out-of-budget
sequences**". Item (b) is where the real work is; the flag is trivia.

## Design

### 1. Decouple draft context size from target context size

```cpp
// common/common.h, struct common_params_speculative_draft
int32_t n_ctx = 0; // 0 = mirror the target context size (today's behavior)
```

`int32_t`, not `uint32_t`, to match every other field in that struct
(`common/common.h:326-346`).

In `common/speculative.cpp:2584`:

```cpp
cparams.n_ctx = params.speculative.draft.n_ctx > 0
    ? std::min<uint32_t>(params.speculative.draft.n_ctx, llama_n_ctx(ctx_tgt))
    : llama_n_ctx(ctx_tgt);
```

`n_seq_max` must stay identical between the two contexts, so the per-sequence draft
budget is `llama_n_ctx_seq(ctx_dft)` — the flag is a *total*, like `--ctx-size`, and is
divided per slot the same way. Round it to a multiple of `n_parallel` (or just accept
the truncation of the integer division and log the effective per-slot value).

**No-op case:** when the draft context shares the target's memory module
(`is_mem_shared`, `common/speculative.cpp:1466`, i.e. `llama_get_ctx_other(ctx_dft) == ctx_tgt`)
there is no separate draft KV cache to shrink. `ctx_other` is only honored for
`LLM_ARCH_GEMMA4_ASSISTANT` / `EAGLE3` / `DFLASH` (`src/llama-context.cpp:143-162`), so
for a plain built-in-MTP-head target model (the GSQ-RCO case) it is null and the draft
context does own its KV — the flag works. For the shared-memory archs the flag must be
ignored with a warning, not applied to a shared memory module.

### 2. New CLI flag

```cpp
add_opt(common_arg(
    {"--spec-draft-ctx-size", "--mtp-ctx-size"}, "N",
    "context size reserved for the MTP/draft head; independent from --ctx-size. "
    "requests whose context exceeds this value keep running without MTP "
    "(default: same as --ctx-size)",
    [](common_params & params, int value) {
        params.speculative.draft.n_ctx = value;
    }
).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SPEC_DRAFT_CTX_SIZE"));
```

`--spec-draft-ctx-size` is the primary name (the whole family in `common/arg.cpp:4214-4379`
is `--spec-draft-*`, and the field is shared by MTP/Eagle3/DFlash/DSpark/Simple);
`--mtp-ctx-size` stays as the alias actually typed at the command line.

### 3. Gate the *start of a draft cycle*, not `can_speculate()` **[rev2 — reversed]**

The first draft proposed making `can_speculate()` dynamic. That is unsafe:
`can_speculate()` is used as a *static capability* predicate in five places, and three
of them break if it can flip mid-request.

| Site | What breaks if `can_speculate()` becomes dynamic |
|---|---|
| `server-context.cpp:400` (`reset()`) | A slot that finishes while over budget skips `spec_draft/spec_dists/spec_i_batch/spec_ckpt.clear()` → stale speculative state leaks into the next task on that slot. |
| `server-context.cpp:4074` (verify pass) | A partial-acceptance replay leaves `spec_draft` non-empty (`:4120`). Those draft tokens were already added to the batch **and** appended to `prompt.tokens` by `handle_last_sampled_token()` (`:554-562`). If the flag flips false before verification, they are never verified or rolled back → prompt/KV divergence, i.e. silently corrupted output. |
| `server-context.cpp:4021` | Same window: falls through to plain sampling with a stale `i_batch` while a draft is in flight. |
| `server-context.cpp:4014` | `common_speculative_begin()` is never called for a prompt that starts over budget, so a later resume drafts against a context that was never begun. |
| `server-context.cpp:713` (`/slots` json) | `"speculative"` silently changes meaning from "configured" to "currently active". |

The safe gate is the one the code already has. `get_n_draft_max()`
(`server-context.cpp:504-523`) returns `n_ctx - prompt.n_tokens() - 2`, and
`server-context.cpp:3161-3199` only starts (or continues) a draft cycle when
`n_draft_max > 0`. That is *already* the "no room left to draft" path, already exercised
today whenever a generation approaches `n_ctx`, and it deliberately lets an in-flight
partial draft complete. One line:

```cpp
    // determine the max draft that fits the current slot state
    // note: slot.prompt is not yet expanded with the `id` token sampled above
    //       also, need to leave space for 1 extra token to allow context shifts
    int n_draft_max = std::min(n_ctx, n_ctx_dft) - prompt.n_tokens() - 2;
```

where `n_ctx_dft` is a new per-slot field set at slot init from
`llama_n_ctx_seq(ctx_dft)` (defaulting to `n_ctx` when there is no draft context),
alongside `slot.n_ctx = n_ctx_slot_value` at `server-context.cpp:1354`. The
model-training-context cap in `n_ctx_slot()` (`:4210`) does not apply to the draft
budget.

`can_speculate()` stays `return !!spec;`.

### 4. Per-sequence skip in the shadow decode (the real work)

In `common_speculative_impl_draft_mtp::process()` (`common/speculative.cpp:1527`), the
catch-up decode must exclude sequences whose positions no longer fit `ctx_dft`:

```cpp
const llama_pos pos_budget = llama_n_ctx_seq(ctx_dft) - 1;   // computed once in the ctor
```

- A sequence is out of budget when the batch's position for that sequence exceeds
  `pos_budget`. Compute it per sequence from `batch_in.pos[i_batch_end[seq_id]]` — the
  per-sequence begin/end indices are already built at `:1543-1554`.
- Build the `batch` at `:1563-1567` by skipping tokens of out-of-budget sequences
  instead of copying `batch_in` 1:1. The embedding fill at `:1574-1577` currently relies
  on that 1:1 shape (`memcpy` of `h_tgt` shifted right by one row), so with filtering it
  becomes a per-token copy: for kept token `k` at new index `j`,
  `set_h(j, llama_get_embeddings_nextn_ith(ctx_tgt, k - 1))`, except at the sequence's
  first index where `pending_h[seq_id]` is used — which is exactly what `:1584-1590`
  already special-cases. This loop is the one genuinely fiddly edit in the whole plan;
  everything else is bookkeeping.
- If every sequence in the batch is out of budget, skip the `llama_decode(ctx_dft, ...)`
  entirely and return `true`.
- Do not update `pending_h` / `verify_h` for skipped sequences (`:1624-1639`); they are
  only consumed while drafting, which step 3 has already turned off for those
  sequences.
- Also drop the skipped sequence's leftover draft rows once, on the transition, with
  `llama_memory_seq_rm(mem_dft, seq_id, -1, -1)`, so an idle over-budget sequence isn't
  holding draft KV cells that a still-eligible slot could use.

Doing this inside the impl (rather than adding a flag to
`common_speculative_draft_params` that the server sets) covers both call sites — the
server loop and the mtmd callback at `server-context.cpp:777` — with one guard, and
keeps the budget knowledge next to the context that owns it.

### 5. Mid-generation transition

Falls out of step 3, with nothing extra:

- Every step recomputes `get_n_draft_max()`; once `prompt.n_tokens()` crosses the draft
  budget it returns `<= 0` and no new cycle starts (`server-context.cpp:3163`).
- An in-flight replay draft still completes: the verify pass at `:4073` does not depend
  on `n_draft_max`, so the already-decoded draft tokens are verified and accepted or
  rolled back normally. This is the window the dynamic-`can_speculate()` approach would
  have corrupted.
- The target's KV cache is untouched and keeps growing to `--ctx-size`. Prompt-length
  admission is still governed by the target ceiling (`server-context.cpp:2036`,
  `:3307`), which is the whole point of the feature.
- **No `spec_ckpt` invalidation is needed [rev2 — dropped].** `spec_ckpt` is only ever
  read inside the drafting branch, is refreshed by `update_pos()`/`update_dft()` at the
  start of every fresh cycle (`:3174-3181`), and is cleared by `reset()` between tasks
  (`:404` — which keeps working precisely because `can_speculate()` stayed static). A
  checkpoint captured before the disable is never restored, because the only consumer is
  a cycle that step 3 prevents from starting.

### 5b. Re-enabling after the context shrinks (context shift)

Context shift (`server-context.cpp:3071-3120`, gated by `--ctx-shift`) discards a middle
chunk once the *target* hits its own limit, so `slot.prompt.n_tokens()` drops and
`get_n_draft_max()` goes positive again on its own. Notes:

- This only triggers at the *target* ceiling (97,280), not at the draft budget, so the
  resume path is rare in practice. See "Scope decision" below.
- `slot.mem.seq_rm`/`seq_add` (`common_memory`, `common/common.h:1032-1042`) operate on
  `ctx_tgt` and `ctx_dft` together and run unconditionally. Harmless when `ctx_dft` has
  no rows for the sequence.
- **`begin()` does not re-prefill the draft context [rev2 — corrected].** The first draft
  claimed a resume "costs one extra prefill pass through the draft model over the kept
  prefix". It does not: `common_speculative_impl_draft_mtp::begin()`
  (`common/speculative.cpp:1509-1525`) only compares `pos_max` against the prompt length
  and emits `SPC_WRN("... Drafts may degrade")`. Nothing rebuilds `ctx_dft`.
- What that means: after a disable→resume, `ctx_dft` has holes for the skipped span and
  drafts will be poor until it is naturally re-warmed. That is an *acceptance-rate*
  problem, not a correctness one — every draft token is verified against the target
  (`:4084-4148`), so a stale or holey draft context can only make the sequence slower,
  never wrong. Expect the `SPC_WRN` line in logs on resume; it is informative, not a bug.

### 6. Startup / capacity accounting **[rev2 — mechanism corrected]**

The first draft said to "run the overhead probe twice". Wrong layer. The fitting code is
`common_fit_normal_kv_context()` (`common/common.cpp:1468-1580`), and it splits into two
parts:

- **Analytic KV cost** — `bytes_per_token`, accumulated over the target's attention
  layers and then, when MTP is on, over the appended nextn layers too
  (`:1522-1526`, whose comment literally reads "shares the target context length").
  Everything is then divided once: `ctx = available / bytes_per_token` (`:1566`). *This*
  is what assumes one length for both contexts.
- **Measured compute overhead** — `common_probe_context_overhead()` (`:1381-1441`) builds
  a target context *and* an MTP context at a small `probe_ctx` and measures device
  memory minus the analytic KV bytes. It already covers both contexts in one call and
  does not need to run twice; compute overhead is not proportional to context length in
  the way KV is.

So the change is to split the accumulator and solve for the target context with the MTP
term pinned:

```
bytes_per_token_tgt * ctx + bytes_per_token_mtp * min(ctx, n_ctx_mtp) <= available
```

Also worth stating: this whole path only runs under auto-fit (`--ctx-size 0`,
`common/common.cpp:1784`, `:1830`). With both `--ctx-size` and `--mtp-ctx-size` given
explicitly — the configuration this feature is for — fitting is skipped and section 6 is
irrelevant. That makes it the last thing to implement, not part of the first cut.

### 7. Observability

- `/props`: add the configured draft budget (`n_ctx_dft`, total and per-slot), mirroring
  `n_ctx`.
- `/slots`: keep `"speculative"` meaning "configured" (`server-context.cpp:713`) and add
  a live `"speculative_active"` = `can_speculate() && get_n_draft_max() > 0`, so the
  cutover is visible without grepping logs and without changing an existing field's
  meaning.
- One `SLT_INF` on the false transition, with `prompt.n_tokens()` at that moment.

## Scope decision: what actually ships first

Phase 1 is the feature. Phases 2 and 3 are optional and independently useful.

1. **Phase 1** — draft `n_ctx` field + flag + `get_n_draft_max()` clamp (section 3) +
   per-sequence skip in `process()` (section 4). Without section 4 the feature returns
   500s, so these two land together or not at all.
2. **Phase 2** — observability (section 7). Cheap, and needed to validate Phase 1
   properly.
3. **Phase 3** — auto-fit math (section 6). Only matters for `--ctx-size 0`.

Not in scope: any eager cleanup, state machine, or checkpoint invalidation. The first
draft's section 4/4b work items were artifacts of the dynamic-`can_speculate()` design
and disappear with section 3.

## Files touched

| File | Change |
|---|---|
| `common/common.h` | add `int32_t n_ctx` to `common_params_speculative_draft` |
| `common/arg.cpp` | new `--spec-draft-ctx-size` / `--mtp-ctx-size` flag |
| `common/speculative.cpp` | size the draft context from `draft.n_ctx` (`:2584`); ignore it when `is_mem_shared`; **per-sequence skip in `common_speculative_impl_draft_mtp::process()` (`:1527`)** |
| `tools/server/server-context.cpp` | per-slot `n_ctx_dft`, clamp in `get_n_draft_max()`, `/props` + `/slots` fields, transition log |
| `common/common.cpp` | split `bytes_per_token` in `common_fit_normal_kv_context()` (Phase 3 only) |
| `tools/server/README.md` | document the flag and the degrade-not-fail behavior |

`can_speculate()`, `reset()`, `spec_ckpt`, and the verify pass are explicitly *not*
touched.

This stays inside `tools/server`'s documented in-scope areas ("Model management",
"Memory management") from `tools/server/README-dev.md` — no new subsystem, one existing
code path (the out-of-room-to-draft fallback) made budget-aware instead of
target-context-only.

## Validation plan

Same methodology as `docs/kv-calibration-findings.md` on GOKAYA (`192.168.1.57`):
ISTA GSQ-RCO-mtp GGUF, traditional KV, batch/ubatch 64,
`--ctx-size 97280 --mtp-ctx-size 54264`.

1. **Blocker regression (run this first).** Single request with a prompt of ~80,000
   tokens. Before section 4 this is expected to fail with
   `failed to process speculative batch`; after it, the request must complete normally
   with zero draft tokens counted. If this test passes on a build without section 4, the
   reproduction is wrong — recheck that `spec` is non-null and MTP is really enabled.
2. Under-threshold request (~8,192-token prompt): draft acceptance stats non-zero,
   `speculative_active = true`.
3. Over-threshold request (~80,000-token prompt): succeeds to `max_tokens`, draft stats
   zero, `speculative_active = false`, no OOM, VRAM below the pre-change ceiling.
4. Crossover: one generation starting under 54,264 and generating past it. No crash, no
   corrupted text, draft stats stop advancing at the crossover, one `SLT_INF` logged.
   Diff the output against the same prompt/seed on an all-MTP-off run — the crossover
   must not change the text.
5. Concurrency (`n_parallel >= 2`), the highest-risk test for section 4's filtered
   batch: one slot under and one slot over the threshold generating simultaneously. The
   under-threshold slot's acceptance rate must match its solo run — a mis-indexed
   embedding row in the filtered batch shows up here as a silent acceptance-rate
   collapse, not as a crash.
6. Resume: with `--ctx-shift`, drive a slot past the draft budget, keep generating until
   context shift brings `prompt.n_tokens()` back under it. Draft stats resume; expect the
   `Drafts may degrade` warning and a temporarily low acceptance rate; output must still
   match the all-MTP-off baseline for the same prompt.
7. Throughput: compare against the existing baselines (~47 decode tok/s with MTP at
   54,264; ~21 tok/s without MTP at 97,280) to confirm the added `std::min` and the
   per-sequence filter cost nothing measurable.
8. Regression with the flag unset: identical behavior and identical fitted context to
   today (`draft.n_ctx == 0` path), including `--ctx-size 0` auto-fit.

## Decided defaults

- Flag name: `--spec-draft-ctx-size` primary, `--mtp-ctx-size` alias.
- Unset ⇒ mirror `--ctx-size`, i.e. today's behavior bit for bit. No new hard
  requirement when MTP is enabled.
- Value semantics: total tokens like `--ctx-size`, per-slot = value / `n_parallel`;
  clamped to `--ctx-size`; log the effective per-slot number at startup.
- `> --ctx-size` ⇒ clamp silently (it is already the default meaning).

## Remaining open question

Only one, and it is a scope question rather than a design one: is section 5b (resume
after context shift) worth validating at all in the first cut? It costs no code — it
falls out of the live `get_n_draft_max()` check — but test 6 is by far the most tedious
to run, and the path only triggers when a sequence reaches the *target* 97,280 ceiling
with `--ctx-shift` on. Skipping test 6 in Phase 1 risks only a degraded acceptance rate
in a rare configuration, never a crash or wrong output.
