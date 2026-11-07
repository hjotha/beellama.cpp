# Qualifying the native 4-bit FP8 paths

This is a continuation checklist for the experimental implementation based on
`7310f5d7e56b4ed5565f7f941c91a386633f31b9`. Read the
[wire matrix](rkva-4bit-wire-matrix.md), the
[Q4_1/IQ4_NL experiment](rkva-low4-experiment.md), root `AGENTS.md`, and the
[Xbox build instructions](../tools/xbox/README.md) before changing or testing it.
Use a normal branch in the existing checkout, not a Git worktree. Finish any
current task and preserve its changes before switching branches.

## What the publication establishes

The reviewed implementation adds native-row E4M3FN to Q4_0, SAW-INT4 and
TurboQuant NC, plus gated Q4_1/IQ4_NL native E5M2/E4M3FN paths with bulk,
async overlap and opt-in TCP prefetch. The reviewed 36-file implementation
passed 20 selected CPU/loopback CTests. Independent review reproduced the
three corrected DXIL blobs with official Microsoft DXC 1.9.2609.5 and validated
them with DXV. No blocking finding remained within that review's scope.

Those results do not establish a complete repository build, Windows/UWP
compilation, D3D12 PSO creation, real GPU/Xbox attention parity, model quality,
KLD, or throughput. Three CUDA graph sections were explicitly skipped because
CUDA was unavailable. Loopback attention replies are synthetic. Missing CI
runs or checks must not be reported as passing CI.

## Repeat the portable checks

The commands below use a POSIX shell, installed CMake/Ninja, Python 3 and a
C++ compiler. The scalar-loader script specifically requires `g++` and Linux
shared-library linking. Run from the repository root:

```sh
cmake -S . -B build-rkva-4bit -G Ninja \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Debug -DGGML_NATIVE=OFF
cmake --build build-rkva-4bit --parallel 6 --target \
  test-remote-attn-range test-rkva-native-e4-cpu test-rkva-e4m3-cpu \
  test-rkva-e4m3-policy-cpu test-rkva-kvarn-bulk-cpu \
  test-rkva-turboquant-nc-cpu test-rkva-native-state \
  test-rkva-native-snapshot test-rkva-native-store test-rkva-low4 \
  test-position-split-ranges test-kv-mixed-state test-arg-parser
ctest --test-dir build-rkva-4bit --output-on-failure -R \
  '^(test-remote-attn-range|test-arg-parser|test-position-split-ranges|test-kv-mixed-state|test-rkva-(rowlane-uniform|native-e4-cpu|native-rows-e4-loopback|e4m3-loopback|e4m3-cpu|e4m3-policy-cpu|kvarn-bulk-cpu|turboquant-nc-cpu|turboquant-nc-loopback|tcp-prefetch|native-state|native-snapshot|native-store|low4|low4-loopback|low4-state))$'
python3 tests/test-rkva-low4-shader.py build-rkva-4bit
python3 tests/test-rkva-rowlane-uniform.py
git diff --check
```

Verify that CTest selects all 20 tests; an empty or reduced selection is not
equivalent. The baseline negative test is:

```sh
python3 tests/test-rkva-rowlane-uniform.py \
  --revision 7310f5d7e56b4ed5565f7f941c91a386633f31b9
```

That baseline must fail because it contains the old divergent broadcasts. Do
not use `--revision HEAD` as an expected-failure command after committing the
fix. Shader rebuild instructions are in the experiment document. If source
changes affect the shared template or SAW shader, regenerate the three
headers, verify source/include provenance, run DXV, and review the resulting
binary changes before continuing.

## Qualify on the intended Windows/GPU host

1. Fetch the published branch, verify its commit and compare with current
   `main`. Preserve unrelated work. If resolving conflicts or making fixes,
   repeat the affected checks against the final integration candidate.
2. Build the host with its actual CUDA/toolchain configuration and repeat the
   relevant tests, including the CUDA sections previously skipped. Build the
   Xbox UWP sources using the existing build instructions and certificate when
   signing is authorized. A successful package build is not console testing.
3. Verify both low4 gate configurations. The normal build must keep
   `RKVA_EXPERIMENTAL_LOW4=0` and advertise neither Q4_1 nor IQ4_NL. An explicit
   test build must compile the relevant engine translation unit with
   `RKVA_EXPERIMENTAL_LOW4=1`; this is a preprocessor definition, not a runtime
   environment variable. Confirm the actual compiler command rather than
   assuming an arbitrary MSBuild property enables it. The host also needs
   `GGML_RKVA_EXPERIMENTAL_LOW4=1`, explicit matching format/cache types and
   the existing placement restrictions. Keep production defaults unchanged.
4. With authorized access to the target, check PSO creation, D256/GQA6/Wave64
   eligibility, exact format/wire capabilities and old-peer rejection. Compare
   OUT and LSE with decoded-KV references for every native format and E5/E4,
   covering one-row decode, empty prefixes, causal/partial tiles, split
   boundaries, and nq 1/9/128/256. Use the same reference quantization and
   documented tolerances; a packing/loader test is not attention parity.
5. Exercise multi-layer sync/async/prefetch, append/trim, two-frame FIFO/control
   barriers, disconnects, malformed envelopes and publication after failure.
   Confirm capability/type/gate failures do not silently choose another
   representation. Do not enable unsupported native snapshots.
6. Run model inference and KLD regressions with exact model files, commands,
   prompts, sampling, hardware, revision and matched baseline/candidate
   `-b`/`-ub` recorded. Measure throughput with tracing disabled only after
   correctness passes. Report unrun stages as unqualified, not successful.

Testing does not authorize installation, deployment, uninstalling an app,
changing access/security, or creating credentials. Do not replace or uninstall
an existing Xbox app to run this checklist. Preserve its package identity,
certificate, models and sessions.

## Integration record

When integration into `main` is authorized, review the final diff and test the
actual integration candidate. Record source and destination SHAs, build/test
commands, explicit skips, hardware results, and CI for that exact commit.
Stop on unresolved failures or unavailable required qualification rather than
claiming readiness. Merge/push without force or history deletion, verify the
remote SHA, and keep deployment a separate action.
