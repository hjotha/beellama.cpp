// Additive handoff accessor definitions for llama_kv_cache_kvarn
// (one-way pure->mixed prefix reuse). New translation unit so the existing
// cache implementation is not edited. See llama-kv-cache-kvarn.h.

#include "llama-kv-cache-kvarn.h"

bool llama_kv_cache_kvarn::handoff_layer_view(
        int32_t il, llama_kvarn_handoff_layer_view & out) const {
    if (n_stream != 1 || il < 0) {
        return false;
    }
    const auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return false;
    }
    const layer & l = layers[it->second];
    out.il = l.il;
    out.n_head_kv = l.n_head_kv;
    out.head_dim_k = l.head_dim_k;
    out.head_dim_v = l.head_dim_v;
    out.k_slices = l.k_slices;
    out.v_slices = l.v_slices;
    out.record_dim_k = l.k_slices > 1 || l.head_dim_k != 64 ? 128 : 64;
    out.record_dim_v = l.v_slices > 1 || l.head_dim_v != 64 ? 128 : 64;
    // BASE LEAF tensors only: the materializer rejects op != GGML_OP_NONE,
    // and the *_stream views are op-carrying views. Single-stream contract:
    // the base tensors are 3D with ne[3] == 1.
    out.k_records = l.k_records;
    out.v_records = l.v_records;
    out.k_stage = l.k_stage;
    out.v_stage = l.v_stage;
    out.k_tail = l.k_tail;
    out.v_tail = l.v_tail;
    if (out.k_records == nullptr || out.v_records == nullptr ||
            out.k_stage == nullptr || out.v_stage == nullptr ||
            out.k_records->ne[3] != 1 || out.v_records->ne[3] != 1 ||
            out.k_stage->ne[3] != 1 || out.v_stage->ne[3] != 1) {
        return false; // not the single-stream base layout
    }
    return true;
}