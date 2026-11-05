# Local VMM pool experiment (2026-11-05)

This branch preserves the VMM experiment only. Do not merge its deployment into
production: tier transitions and production pressure recovery are not qualified.

The experiment changed no CUDA kernel source. It removed GGML_CUDA_NO_VMM for
only ggml-cuda.cu and linked the CUDA driver in a private copy. The helper script
records input hashes, reproduces the control link, and checks that inputs did not
change. Its paths intentionally match the local experiment on gokaya. The
remaining translation units have no functional use of the VMM conditional and
no conditional shared layouts; this does not replace full release qualification.

Artifacts: /home/hjotha/.codex/artifacts/vmm-single-object-20261105/

At context 20480, KVarN4 target/draft, same model and prompt, the minimum free
VRAM was 119 MiB without VMM and 409 MiB with VMM. Output token IDs matched.
Prefill/decode were 911.10/26.57 versus 905.85/26.55 tokens/s. The VMM variant
completed 31744 prompt +1024 output at context 32768 with 111 MiB minimum free.
47 GDN operator cases and exact T134 rollback passed. CUDA graph output parity
passed for direct execution, one graph, and 32-node chunks.

Production remains NO_VMM by explicit user instruction. These measurements do
not qualify adaptive tier transitions, multiple GPUs or allocation failure.
