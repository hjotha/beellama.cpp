from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KVAR_N_CACHE = ROOT / "src/llama-kv-cache-kvarn.cpp"
KVAR_N_TEST = ROOT / "tests/test-kvarn.cpp"
CONTEXT = ROOT / "src/llama-context.cpp"
GRAPH = ROOT / "src/llama-graph.cpp"


def function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for pos in range(brace, len(source)):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[brace + 1 : pos]
    raise AssertionError(f"unterminated function: {signature}")


def main() -> None:
    source = KVAR_N_CACHE.read_text(encoding="utf-8")
    can_remove = function_body(source, "bool llama_kv_cache_kvarn::can_remove(")
    seq_rm_plan = function_body(source, "bool llama_kv_cache_kvarn::seq_rm_plan(")
    seq_rm = function_body(source, "bool llama_kv_cache_kvarn::seq_rm(")

    assert "if (swa)" not in can_remove, (
        "SWA removal must use the KVarN group-safety policy instead of trusting metadata alone"
    )
    assert "if (swa ||" not in seq_rm_plan, (
        "deep SWA rollback must widen to a complete KVarN group boundary"
    )
    for diagnostic in ("seq_pos_min", "seq_pos_max", "earliest_exact", "meta_can_seq_rm"):
        assert diagnostic in seq_rm, f"KVarN removal refusal omits {diagnostic} diagnostics"

    tests = KVAR_N_TEST.read_text(encoding="utf-8")
    assert "llama_kvarn_non_swa_tail_groups(0, 0) + 1" in tests, (
        "eager-workspace coverage must derive the production non-SWA stage depth"
    )

    context = CONTEXT.read_text(encoding="utf-8")
    full_restore = function_body(context, "size_t llama_context::state_read_data(")
    seq_restore = function_body(context, "size_t llama_context::state_seq_read_data(")
    invalidate = "io.on_commit([this]() { gf_res_prev->reset(); });"
    for body, read in (
        (full_restore, "memory->state_read(io);"),
        (seq_restore, "memory->state_read(io, seq_id, flags);"),
    ):
        assert body.count(invalidate) == 1, (
            "FULL restore must invalidate graph placement only through its commit callback"
        )
        assert body.index(read) < body.index(invalidate), (
            "graph invalidation must follow successful memory parsing and its commit callbacks"
        )
        assert body.count("gf_res_prev->reset()") == 1, (
            "failed or cancelled restores must retain the live graph"
        )
        assert "LLM_ARCH_DFLASH" not in body, (
            "FULL graph invalidation must apply to every model architecture"
        )
    partial_guard = function_body(seq_restore, "if (!(flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY))")
    assert invalidate in partial_guard, (
        "PARTIAL_ONLY speculative restores must retain their existing graph-reuse behavior"
    )

    # Resetting the result must actually force the existing scheduler/store-slot
    # rebuild path, rather than merely clearing logits or ordinary input values.
    graph = GRAPH.read_text(encoding="utf-8")
    graph_reset = function_body(graph, "void llm_graph_result::reset(")
    assert "params = {};" in graph_reset and "inputs.clear();" in graph_reset
    process = function_body(context, "llm_graph_result * llama_context::process_ubatch(")
    assert "res->can_reuse(gparams)" in process
    assert "mctx->reset_graph_store_slots();" in process
    assert "ggml_backend_sched_reset(sched.get());" in process


if __name__ == "__main__":
    main()
