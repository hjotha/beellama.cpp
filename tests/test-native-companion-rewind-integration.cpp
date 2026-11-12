// Isolated CPU test: real hybrid cache and native TCP owner with a fake peer.
// No model forward pass or Xbox is used.
#ifndef _WIN32
#define main remote_range_existing_main
#include "test-remote-attn-range.cpp"
#undef main
#define main hybrid_state_existing_main
#include "test-hybrid-rpc-prefix-state.cpp"
#undef main
#include "../src/llama-rollback.h"

int main() {
    storage_env upload("kvarn4", "GGML_RKVA_UPLOAD_FORMAT");
    storage_env experimental("1", "GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL");
    storage_env companion("1", "GGML_RKVA_NATIVE_COMPANION");
    storage_env bulk("0", "GGML_RKVA_NATIVE_BULK");
    storage_env async("0", "GGML_RKVA_ASYNC_LOCAL_REMOTE");
    storage_env precision("128", "GGML_RKVA_NATIVE_TAIL_TOKENS");
    storage_env producer("cpu", "GGML_RKVA_NATIVE_PRODUCER");
    storage_env no_cuda_fp8("0", "GGML_RKVA_CUDA_FP8");
    ggml_backend_load_all();
    loopback peer(false, true, false, false, false, false, 0, 0, true, 3);
    auto * backend = ggml_backend_remote_attn_init("127.0.0.1", peer.port);
    ggml_remote_attn_geometry geo{};
    geo.n_layer_remote = 1; geo.n_head = 1; geo.n_head_kv = 1;
    geo.head_dim = 256; geo.max_ctx = 256; geo.kq_scale = 0.0625f;
    geo.group_tokens = 128; geo.sinkhorn_iters = 3;
    assert(backend && ggml_backend_remote_attn_set_geometry(backend, &geo));
    assert(ggml_backend_remote_attn_connect(backend));
    assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 256));
    ggml_backend_remote_attn_set_active_session(backend, 0);
    assert(ggml_backend_remote_attn_configure_range(backend, 256, 256, GGML_TYPE_F32));
    assert(ggml_backend_remote_attn_native_kvarn_can_rewind(backend));

    std::vector<float> keys(2 * 256, 0.25f), values(2 * 256, -0.5f);
    assert(ggml_backend_remote_attn_native_kvarn_stage_append(backend, 0, 256, 2,
                                                               keys.data(), values.data()));
    assert(ggml_backend_remote_attn_native_kvarn_can_restore(backend, 257));
    {
        std::lock_guard<std::mutex> lock(peer.mutex);
        assert(peer.native_counts == std::vector<uint32_t>({2}));
    }

    tiny_model model;
    model.hparams.n_embd = 256;
    model.hparams.n_embd_head_k_full = model.hparams.n_embd_head_v_full = 256;
    model.hparams.n_rot_full = 256;
    auto attention = std::make_unique<llama_kv_cache_kvarn>(model, model.hparams,
            llama_kvarn_params_for_type(LLAMA_KVARN_K4V4_G128), false, false,
            512, 1, 256, 256, 1, 0, LLAMA_SWA_TYPE_NONE,
            [](int32_t il) { return il == 0; }, nullptr, 0, GGML_TYPE_F16, 0, 1,
            nullptr, nullptr, nullptr, GGML_TYPE_COUNT, GGML_TYPE_COUNT, false, nullptr, 256, true);
    auto * attn = attention.get();
    assert(attn->native_companion_enabled() && !attn->has_standard_cache());
    attn->set_negotiated_native_rpc_owner(true);
    attn->set_native_kvarn_rewind_capable(true);
    attn->set_native_kvarn_rewind_preflight([backend](llama_seq_id seq, llama_pos pos) {
        ggml_backend_remote_attn_set_active_session(backend, uint32_t(seq));
        return ggml_backend_remote_attn_native_kvarn_can_restore(backend, pos);
    });
    attn->set_native_companion_reset([backend]() {
        return ggml_backend_remote_attn_reset(backend, 0);
    });
    auto recurrent = std::make_unique<llama_memory_recurrent>(model, GGML_TYPE_F32, GGML_TYPE_F32,
            false, 1, 1, 0, [](int32_t il) { return il == 1; });
    llama_memory_hybrid hybrid(model, std::move(attention), std::move(recurrent), 256);
    for (const auto & span : {std::pair<llama_pos, uint32_t>{0, 256}, {256, 2}}) {
        const auto ub = batch(span.first, span.second);
        llama_memory_hybrid_context state(&hybrid, attn->init_kv_batch({ub}), {ub});
        assert(state.get_status() == LLAMA_MEMORY_STATUS_SUCCESS && state.apply());
        state.graph_compute_finish(GGML_STATUS_SUCCESS);
    }
    const auto attn_before = attn->seq_pos_max(0);
    const auto recurrent_before = hybrid.get_mem_recr()->seq_pos_max(0);
    assert(attn_before == 257 && recurrent_before == 257);
    const auto cap = attn->get_seq_rm_capability();
    assert(cap.full_clear && !cap.arbitrary_ranges && cap.suffix_rollback_tokens == 0);

    // No partial operation is allowed: preflight must not send an RPC or mutate
    // the recurrent and attention halves, even with a rewind-capable peer.
    size_t before;
    { std::lock_guard<std::mutex> lock(peer.mutex); before = peer.packet_ops.size(); }
    assert(!hybrid.can_seq_rm(0, 257, -1));
    assert(!hybrid.seq_rm(0, 257, -1));
    assert(attn->seq_pos_max(0) == attn_before && hybrid.get_mem_recr()->seq_pos_max(0) == recurrent_before);
    { std::lock_guard<std::mutex> lock(peer.mutex); assert(peer.packet_ops.size() == before); }

    // Simulate local suffix removal after an acknowledged upload. The same
    // helper used by decode must notify the real peer; a lost rewind ACK
    // poisons the backend and makes the full-reset fallback fail closed.
    peer.drop_native_rewind_ack = true;
    int local_rows = 258;
    int suffix_notifications = 0, full_notifications = 0;
    const auto rollback = llama_rollback_sequence_after_decode_failure(
        [&](llama_seq_id seq, llama_pos p0, llama_pos p1) {
            return llama_rollback_remove_and_notify(
                [&](llama_seq_id, llama_pos first, llama_pos) {
                    local_rows = first < 0 ? 0 : first;
                    return true;
                },
                [&](llama_seq_id id, llama_pos first, llama_pos) {
                    if (first < 0) { ++full_notifications; return ggml_backend_remote_attn_reset(backend, uint32_t(id)); }
                    ++suffix_notifications;
                    return ggml_backend_remote_attn_trim(backend, uint32_t(id), first);
                }, seq, p0, p1);
        }, 0, 257);
    assert(rollback == LLAMA_ROLLBACK_NONE && local_rows == 0 && suffix_notifications == 1 && full_notifications == 1);
    assert(llama_decode_aborted_code(true, ggml_backend_remote_attn_failed(backend)) == -4);
    assert(ggml_backend_remote_attn_failed(backend));
    assert(!ggml_backend_remote_attn_connected(backend));
    assert(!ggml_backend_remote_attn_native_kvarn_has_state(backend));
    assert(!ggml_backend_remote_attn_native_kvarn_can_restore(backend, 257));
    assert(!ggml_backend_remote_attn_trim(backend, 0, 257));
    assert(!ggml_backend_remote_attn_native_kvarn_stage_append(backend, 0, 257, 1,
                                                                keys.data(), values.data()));
    {
        std::lock_guard<std::mutex> lock(peer.mutex);
        assert(peer.native_counts == std::vector<uint32_t>({1}));
        assert(std::count(peer.packet_ops.begin(), peer.packet_ops.end(), RKVA_KVARN_RANGE_REWIND) == 1);
        assert(peer.packet_ops.back() == RKVA_KVARN_RANGE_REWIND);
    }
    assert(!hybrid.seq_rm(0, 257, -1));
    assert(attn->seq_pos_max(0) == attn_before && hybrid.get_mem_recr()->seq_pos_max(0) == recurrent_before);
    bool refused = false;
    try { hybrid.clear(false); } catch (const std::runtime_error &) { refused = true; }
    assert(refused && attn->seq_pos_max(0) == attn_before && hybrid.get_mem_recr()->seq_pos_max(0) == recurrent_before);
    ggml_backend_free(backend);
    std::puts("PASS: hybrid refuses partial rewind; applied peer rewind with lost ACK poisons native owner");
    return 0;
}
#else
int main() { return 0; }
#endif
