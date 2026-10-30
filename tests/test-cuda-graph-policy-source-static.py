#!/usr/bin/env python3
"""Check the pre-submission integration points without a CUDA runtime.

This is a structural regression check, not a CUDA execution or lifetime test.
"""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
context = (root / "src/llama-context.cpp").read_text(encoding="utf-8")
cuda = (root / "ggml/src/ggml-cuda/ggml-cuda.cu").read_text(encoding="utf-8")
ctor = context.split("llama_context::llama_context(", 1)[1].split("llama_context::~llama_context()", 1)[0]
setter = ctor.index('"ggml_backend_cuda_set_graph_min_stable_calls"')
assert ctor.rindex("backends.emplace_back(") < setter, "configure after direct backend creation"
assert setter < ctor.index("sched_reserve();"), "configure before scheduler reservation"
assert "graph_compute(" not in ctor[:setter], "no explicit graph submission before policy"
assert 'tcp_position_split, std::getenv("GGML_CUDA_GRAPH_REMOTE_MIN_STABLE_CALLS")' in ctor
enabled = ctor.index("if (remote_graph_stable_calls > 0)")
preflight = ctor.index("llama_remote_cuda_graph_backends_supported(backends", enabled)
meta = ctor.index("return dev && ggml_backend_dev_is_meta(dev);", preflight)
meta_error = ctor.index('throw std::runtime_error("remote CUDA graph policy does not support meta backends")', meta)
direct_loop = ctor.index("using set_policy_fn = bool (*)(ggml_backend_t, int);", meta_error)
assert enabled < preflight < meta < meta_error < direct_loop < setter, "reject any meta before setting direct CUDA"
assert 'std::strcmp(ggml_backend_reg_name(reg), "CUDA") != 0' in ctor
assert 'if (!set_policy || !set_policy(backend.get(), remote_graph_stable_calls))' in ctor
assert 'if (configured == 0)' in ctor

compute = cuda.split("static enum ggml_status ggml_backend_cuda_graph_compute(", 1)[1]
compute = compute.split("static void ggml_backend_cuda_event_record(", 1)[0]
assert compute.index("graph_policy.freeze_and_get_min_stable_calls(") < compute.index(
    "ggml_cuda_graph_evaluate_and_capture("
), "freeze before any graph evaluation or capture"
assert "graph->warmup_stable_calls >= graph_min_stable_calls" in compute
print("CUDA graph policy integration structure checks passed")
