# Flash-Next on A100-40G: what the crash was (2026-11-21)

Short version: **the model never had a loader bug. The zip that was published had been
built incrementally in a directory that mixed object files from different tree states.**

Evidence, in order:

1. The published `beellama-a100-moe.zip` (built 2026-11-21 11:16 in an incremental build
   directory) crashed on load of `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0`, 5 out of 5 times, with
   `SIGSEGV` around 12 s, before `load_tensors`:
   - one crash read `0x9300000000`, another wrote into `.text` of `libllama.so`
     (`error 7`), both inside graph construction;
   - the failing store disassembled to `llm_graph_result::set_params()`, a
     `memcpy(this + 0x168, &params, sizeof(llm_graph_params))`.
2. A **clean CPU-only build of the same commit** loads that model and creates the context
   (`sched_reserve: reserve took 18.88 ms`, `ctx != NULL`), no crash.
3. A **clean sm_80 CUDA build of the same commit** also loads it (`model loaded`,
   `/health` 200), on the same A100-40G VM.
4. The incremental build had already shown staleness: its final link failed with an
   undefined `llama_model_loader` constructor until a stale `llama-model-loader.cpp.o`
   was deleted.

How it is prevented from now on:

- **Never ship a zip from an incremental build directory.** The Colab package is built in
  a fresh directory (`cmake -S <src> -B /tmp/colab-clean ...`), with
  `-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined` because the chroot's CUDA has
  libcublas 13.8 and the server link needs to tolerate the shared-lib reference.
- **Smoke test before uploading**: two header files plus `truncate` to the real sizes
  reproduce the load path in 96 MB instead of 66 GB:

      cp first-64MiB.gguf  ...-00001-of-00002.gguf && truncate -s 37623740192 ...
      cp first-32MiB.gguf  ...-00002-of-00002.gguf && truncate -s 28800138432 ...

  A server start with the target flags must reach `model loaded` / `/health` 200 within
  ~30 s. This exact test is what separated the bad zip from the good one.
- The graph guard (`LLAMA_DEBUG_GRAPH_GUARD=1`) and the crash backtrace handler
  (`LLAMA_CRASH_BT=1`, prints `backtrace_symbols_fd`) stay in the tree: they turn a bare
  dmesg line into a phase and a stack.
