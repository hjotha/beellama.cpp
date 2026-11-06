# MQ2 fine split96: default-OFF isolated candidate

## Scope and integration

This package proposes only a host split-policy change in
`tools/xbox/uwp/rkva/rkva-range-profile.h`. The compile-time switch
`RKVA_EXPERIMENTAL_MQ2_FINE_SPLIT96` defaults to **0** and accepts only 0 or 1.
At 1 it changes the fine-mode cap from 64 to 96 for `q_tile == 2` only.
Target rows remain 256. MQ4, every other mode, unsupported tile fallback,
prefetch, shader selection and shape/descriptor gates are unchanged.

No main checkout, existing checkout, remote computer or console was modified.
No commit/push or publication was performed by this package's author.
`mq2-fine-split96.patch` is the integration artifact; the candidate header is
also in `include/`. `include/baseline.h` changes only the original namespace
for side-by-side host tests. Provenance identifies the original snapshot.

Before integration, inspect the actual target branch HEAD and diff against the
recorded header. Apply with `git apply --check` first; stop and reconcile if it
fails. Do not replace a newer complete header with this snapshot. Publish only
on the approved experiment branch. Do not enable by default or promote based
on these synthetic results. Use the macro consistently across all translation
units: mixed macro definitions for an inline header violate the ODR.

The selector's inherited arithmetic overflow near UINT32_MAX remains unchanged.
UINT32_MAX returns `{1,0}` in both variants; tests explicitly reproduce it, not
certify it as a valid engine input. A fix belongs in a separate reviewed change.

## Reproduction

Requires Python 3, NumPy, and a C++17 compiler. No network, models, GPU, secrets,
local usernames or absolute machine paths are required. From this directory:

```
./run_tests.sh
python3 verify_manifest.py
```

`CXX` may select GCC or Clang. Assertions must remain enabled. The script builds
separate default, explicit-OFF and ON binaries, rejects macro=2, and records
actual C++ geometry for the Python numerical test. Build products stay in
`.build/` and are not publication artifacts. `results/` contains CPU evidence.
The hash manifest covers source and stored evidence, not executables/caches.

## Results and exact interpretation

- Each of default/OFF/ON passes 37,748,916 C++ policy comparisons: exhaustive
  rows 0..1,048,576 plus four large values through INT32_MAX, six modes including
  invalid values, and six valid/invalid tiles. Default and explicit OFF have
  zero differences. ON changes 1,032,196 targeted cases only.
- Coverage, positive/aligned spans, no empty trailing split, sentinel behavior,
  cap96/cap256 bounds and unchanged behavior through 16,384 rows are asserted.
- Preserved Python model passes 103,680 causal/phantom/tail interval checks.
  Origins are enumerated but do not alter this relative-window geometry model;
  it does not validate origin-dependent descriptor or resident-byte logic.
- 36 synthetic CPU numerical cases use actual C++ geometry, the preserved FP32
  reducer/WHT and unchanged comparison gates against an independently computed
  dense FP64 softmax and Sylvester transform. Both cap64 and cap96 pass.
  Inputs include empty windows, clipped prefixes, partial nq3/nq9 tiles,
  random scores, final-row-dominant scores, and original/rotated values.
- Worst candidate-vs-oracle OUT absolute difference: 1.9073486328125e-6;
  LSE: 6.975323803004585e-7. Worst A/B OUT: 8.940696716308594e-8;
  LSE: 9.5367431640625e-7. Twenty cases are **not bit-exact**.
- All six deliberate numerical mutations are rejected: dropping the final
  populated split, nonfinite output, and +0.1 LSE bias, in both value domains.
  The invalid macro is also rejected at compile time.
- OUT tolerance remains `3e-4 + 3e-4 * abs(reference)`; LSE remains
  `3e-4 + 3e-5 * abs(reference)`. Nonfinite output and finiteness mismatches fail.

Selected actual geometries (splits x rows):

- 16,384: 64 x 256 -> 64 x 256.
- 16,385: 63 x 264 -> 65 x 256.
- 21,760: 64 x 344 -> 85 x 256.
- 24,576: 64 x 384 -> 96 x 256.
- 24,577: 63 x 392 -> 94 x 264 (split count need not be monotonic).
- 49,920: 64 x 784 -> 96 x 520.

**Limits:** numerical inputs are synthetic post-decode scores/values. This tests
split softmax merging, FP32 effects, WHT and OUT/LSE gates, not KVarN byte decode,
QK accumulation, real MQ2 pass1, engine residency, full serial-prefix record
selection, MSVC, DXIL, PSO, replay, model KLD or GPU correctness/performance.
The retained model files contain broader machinery, but importing their reducer
does not mean their full suites ran. No hardware stage is marked PASS.
During test development an extra /16 in the oracle transform was detected by
the one-row rotated test and corrected to an independent normalized transform;
the final full suite was rerun successfully.

## Windows / Xbox qualification handoff

1. Record exact experiment commit, base commit, clean/dirty status and this
   manifest. Run existing repository host policy/regression suites alongside
   these tests. Under MSVC compile `tests/policy.cpp` with `/std:c++17 /EHsc /O2`
   in three separate configurations (absent, `/D...=0`, `/D...=1`). Do not use
   `/DNDEBUG`; confirm invalid value 2 fails. Compare printed geometry.
2. Build isolated baseline and candidate UWP configurations with existing
   repository build scripts and installed SDK. Set the macro to 1 for the
   candidate's C++ compilation only, consistently in all relevant projects/TUs;
   leave baseline absent or 0. Record effective compiler command lines and
   clean/rebuild when changing the macro. Do not conflate an environment setting
   with proof it reached MSVC. Keep package identity and existing signing cert;
   do not create credentials or uninstall/deploy merely to run this CPU suite.
3. This patch changes no HLSL or shader binaries. Nevertheless validate the real
   selected MQ2 PSO, source/DXIL hashes, compile macros, reflection/layout and
   D3D12 debug diagnostics on the actual platform. If shaders are rebuilt, run
   the repository's current DXIL build and validate its artifacts. Confirm the
   unchanged reducer and actual scratch sizing safely handle every selected
   split count. The historical 256-split reservation is not runtime proof.
4. Replay real data at nq2..9; active rows 21,760 and 49,920; thresholds
   16,384/24,576 +/-1; nonmultiples of 8/128; zero and nonzero 128-aligned origins;
   tail0/128; empty, short, full and unordered windows; partial tiles; reset,
   append and rewind. Include unsupported shapes/descriptors, missing pipelines,
   MQ4 and modes reference/balanced/coarse to prove unchanged fallback. Keep
   prefetch and all existing gate settings unchanged.
5. Compare true OUT/LSE against the existing reference and unchanged numerical
   gates; run real decoder/engine and exact model replay parity/KLD checks before
   performance evaluation. Synthetic PASS here cannot override any hardware or
   model failure. Preserve dumps/hashes and report failures, not just averages.
6. For the approved 130k and 159k workloads, use identical model hashes, prompts,
   seeds, sampling, batch/ubatch, context, MTP settings and device thermal state.
   Capture actual active-row geometry; do not equate context length to active
   KV rows. Warm both variants and run repeated interleaved ABBA with enough
   independent repetitions. Report pass1, reduce, readback, attention total,
   full generation latency, medians/tails, memory/traffic and regressions.
   Existing qualification tooling must distinguish synthetic fixtures from
   actual runs. 4070 evidence does not establish Xbox wave/runtime correctness.
7. Promote only after required Windows/DXIL/PSO/replay/model and actual Xbox A/B
   gates pass, with review and authorized publication scope. Otherwise retain
   default OFF and rollback by rebuilding with macro=0. More groups can increase
   reducer work and traffic; no speedup or promotion is promised.
