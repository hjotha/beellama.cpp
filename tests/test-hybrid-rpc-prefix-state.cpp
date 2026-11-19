// CPU memory integration: no GGUF weights, model forward pass or remote host.
#include "../src/llama-model.h"
#include "../src/llama-kv-cache-kvarn.h"
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-io.h"
#include "../src/llama-native-prefix-validation.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <cassert>
#include <cstring>
#include <cstdio>
#include <random>
#include <stdexcept>

struct tiny_model final : llama_model {
    tiny_model() : llama_model(llama_model_default_params()) {
        arch = LLM_ARCH_QWEN35;
        hparams = {};
        hparams.n_layer_all = 2;
        hparams.rope_type = LLAMA_ROPE_TYPE_IMROPE;
        hparams.n_embd = 128;
        hparams.n_ctx_train = 512;
        hparams.n_embd_head_k_full = hparams.n_embd_head_v_full = 128;
        hparams.n_rot_full = 128;
        hparams.n_head_arr.fill(1);
        hparams.n_head_kv_arr.fill(1);
        hparams.is_recr_impl.fill(false);
        hparams.is_recr_impl[1] = true;
        hparams.ssm_d_conv = 2;
        hparams.ssm_d_inner = hparams.ssm_d_state = hparams.ssm_n_group = 1;
        hparams.causal_attn = true;
        layers.resize(2);
    }
    void load_stats(llama_model_loader &) override {}
    void load_hparams(llama_model_loader &) override {}
    void load_vocab(llama_model_loader &) override {}
    bool load_tensors(llama_model_loader &) override { return false; }
    void load_arch_hparams(llama_model_loader &) override {}
    void load_arch_tensors(llama_model_loader &) override {}
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params &) const override { return {}; }
};

struct writer final : llama_io_write_i {
    std::vector<uint8_t> data;
    void write(const void * src, size_t n) override {
        const auto * p = static_cast<const uint8_t *>(src);
        data.insert(data.end(), p, p + n);
    }
    void write_tensor(ggml_tensor * t, size_t offset, size_t n) override {
        const auto begin = data.size(); data.resize(begin + n);
        ggml_backend_tensor_get(t, data.data() + begin, offset, n);
    }
    size_t n_bytes() override { return data.size(); }
};

struct reader final : llama_io_read_i {
    const std::vector<uint8_t> & data;
    size_t pos = 0;
    std::vector<std::function<void()>> pending;
    explicit reader(const std::vector<uint8_t> & data) : data(data) {}
    void read(void * dst, size_t n) override {
        if (n > data.size() - pos) throw std::runtime_error("truncated test state");
        std::memcpy(dst, data.data() + pos, n); pos += n;
    }
    void stage_tensor_set(ggml_tensor * t, const void * src, size_t offset, size_t n) override {
        const auto * p = static_cast<const uint8_t *>(src);
        std::vector<uint8_t> copy(p, p + n);
        pending.push_back([t, offset, copy = std::move(copy)] { ggml_backend_tensor_set(t, copy.data(), offset, copy.size()); });
    }
    void stage_tensor_clear(ggml_tensor * t, size_t offset, size_t n) override {
        std::vector<uint8_t> zero(n, 0); stage_tensor_set(t, zero.data(), offset, n);
    }
    void read_tensor(ggml_tensor * t, size_t offset, size_t n) override {
        std::vector<uint8_t> copy(n); read(copy.data(), n); stage_tensor_set(t, copy.data(), offset, n);
    }
    void on_commit(std::function<void()> cb) override { pending.push_back(std::move(cb)); }
    void commit() override { for (auto & operation : pending) operation(); pending.clear(); }
    void cancel() override { pending.clear(); }
    size_t n_bytes() override { return pos; }
};

static llama_ubatch batch(llama_pos pos, uint32_t count) {
    llama_ubatch ub {};
    ub.b_equal_seqs = 1; ub.n_tokens = ub.n_seq_tokens = count;
    ub.n_seqs = ub.n_seqs_unq = 1; ub.n_pos = 4;
    auto d = std::make_shared<llama_ubatch::data_t>();
    d->token.assign(count, 1); d->pos.resize(4*count); d->n_seq_id.assign(count, 1);
    d->seq_id.resize(count); d->seq_id_data.assign(count, 0); d->seq_id_unq = {0};
    d->seq_idx.assign(LLAMA_MAX_SEQ, -1); d->seq_idx[0] = 0; d->output.assign(count, 0);
    for (uint32_t i = 0; i < count; ++i) { for (uint32_t c = 0; c < 4; ++c) d->pos[c*count+i] = pos+i; d->seq_id[i] = d->seq_id_data.data() + i; }
    ub.data = d; ub.token = d->token.data(); ub.pos = d->pos.data(); ub.n_seq_id = d->n_seq_id.data();
    ub.seq_id = d->seq_id.data(); ub.seq_id_unq = d->seq_id_unq.data(); ub.seq_idx = d->seq_idx.data(); ub.output = d->output.data();
    return ub;
}

static void test_position_range_presence() {
    tiny_model model;
    auto cpu = [](int32_t) { return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU); };
    llama_kv_cache_kvarn cache(model, model.hparams,
            llama_kvarn_params_for_type(LLAMA_KVARN_K4V4_G128), false, false,
            512, 1, 256, 256, 1, 0, LLAMA_SWA_TYPE_NONE,
            [](int32_t il) { return il == 0; }, nullptr, 0, GGML_TYPE_F16, 0, 1,
            nullptr, nullptr, [](int32_t il) { return il == 0; }, GGML_TYPE_Q4_0,
            GGML_TYPE_Q4_0, false, cpu, 256, true);
    // Mutate the real metadata through its ordinary cell APIs, which also
    // maintain the per-sequence position index used by the fast predicate.
    auto & cells = const_cast<llama_kv_cells &>(cache.get_metadata_cache()->get_cells(0));
    const auto check = [&]() {
        for (auto range : {llama_position_split::range_kind::LOCAL,
                           llama_position_split::range_kind::OVERFLOW_RANGE,
                           llama_position_split::range_kind::INVALID}) {
            std::vector<uint32_t> listed;
            const bool present = cache.position_split_live_cells(range, listed) && !listed.empty();
            assert(cache.position_split_has_live_cells(range) == present);
        }
    };
    const auto add = [&](uint32_t cell, llama_pos pos, llama_seq_id owner) {
        cells.pos_set(cell, pos);
        if (owner >= 0) { cells.seq_add(cell, owner); }
    };
    check();
    add(7, 255, 0); check();
    add(2, 256, 0); check();
    add(511, 600, 1); check(); // unrelated owner must not invalidate sequence 0
    add(9, 512, 0); check(); // invalid later position invalidates BOTH ranges
    cells.seq_rm(9, 0); check();
    add(10, -2, 0); check(); // invalid lower bound, including negative positions
    cells.seq_rm(10, 0); check();
    cells.seq_rm(7, 0); check();
    cells.seq_rm(2, 0); check(); // only a foreign sequence remains

    std::mt19937 rng(72704);
    for (int trial = 0; trial < 512; ++trial) {
        cells.reset();
        const uint32_t count = rng() % 32;
        for (uint32_t i = 0; i < count; ++i) {
            llama_pos pos = llama_pos(rng() % 520) - 2;
            if (pos == -1) { pos = -2; }
            add(i, pos, llama_seq_id(rng() % 3) - 1);
        }
        check();
        for (uint32_t i = 0; i < count; ++i) {
            if (cells.seq_has(i, 0)) { cells.seq_rm(i, 0); }
            check();
        }
    }
}

// Exercise the real builder and hybrid reuse wrapper, rather than duplicating
// their stride policy. An overflow batch reads the local tail as history only;
// a below-P batch also appends its two current rows to that tail.
static void test_hybrid_tail_graph_reuse() {
    for (bool overflow : {false, true}) {
        tiny_model model;
        auto cpu = [](int32_t) { return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU); };
        auto attention = std::make_unique<llama_kv_cache_kvarn>(model, model.hparams,
                llama_kvarn_params_for_type(LLAMA_KVARN_K4V4_G128), false, false,
                512, 1, 256, 256, 1, 0, LLAMA_SWA_TYPE_NONE,
                [](int32_t il) { return il == 0; }, nullptr, 128, GGML_TYPE_F16, 128, 1,
                nullptr, nullptr, [](int32_t il) { return il == 0; }, GGML_TYPE_Q4_0,
                GGML_TYPE_Q4_0, false, cpu, 256, true);
        auto * attn = attention.get();
        auto recurrent = std::make_unique<llama_memory_recurrent>(model, GGML_TYPE_F32, GGML_TYPE_F32,
                false, 1, 1, 1, [](int32_t il) { return il == 1; });
        llama_memory_hybrid hybrid(model, std::move(attention), std::move(recurrent), 256);
        if (overflow) {
            auto prefix = attn->init_kv_batch({batch(0, 256)});
            assert(prefix && prefix->get_status() == LLAMA_MEMORY_STATUS_SUCCESS && prefix->apply());
            prefix->graph_compute_finish(GGML_STATUS_SUCCESS);
        }
        auto ub = batch(overflow ? 256 : 0, 2);
        llama_memory_hybrid_context memory(&hybrid, attn->init_kv_batch({ub}), {ub});
        assert(memory.get_status() == LLAMA_MEMORY_STATUS_SUCCESS && memory.apply());
        llm_graph_result result(4096);
        llm_graph_params params{};
        params.arch = model.arch;
        params.hparams = model.hparams;
        params.cparams.n_ctx = 512;
        params.cparams.n_seq_max = 1;
        params.cparams.n_rs_seq = 1;
        params.cparams.causal_attn = true;
        params.cparams.flash_attn = true;
        params.cparams.kv_position_split_p = 256;
        params.cparams.remote_attn_position_split = true;
        params.ubatch = ub;
        params.n_outputs = 2;
        params.mctx = &memory;
        params.res = &result;
        llm_graph_context graph(params);
        auto * input = graph.build_inp_mem_hybrid();
        assert(input && input->inp_attn && input->inp_rs);
        auto * tail = input->inp_attn->self_tail_read_idxs;
        assert(tail && tail->ne[1] == 2);
        const auto * context = memory.get_attn_kv_context();
        assert(tail->ne[0] == context->get_tail_attention_stride(overflow ? 1 : 2));
        assert(context->get_tail_attention_stride(1) != context->get_tail_attention_stride(2));
        assert(input->can_reuse(params));
        assert(input->can_reuse(params)); // repeated equal-shape target pass

        // A real malformed tail extent must still invalidate reuse.
        const int64_t saved_extent = tail->ne[0];
        ++tail->ne[0];
        assert(!input->can_reuse(params));
        tail->ne[0] = saved_extent;
        assert(input->can_reuse(params));

        auto changed = params;
        changed.ubatch = batch(overflow ? 256 : 0, 1);
        assert(!input->can_reuse(changed));
        assert(input->can_reuse(params));
        memory.graph_compute_finish(GGML_STATUS_SUCCESS);
    }
}

static void test_exact_tail_snapshot_layouts() {
    tiny_model model;
    auto cpu = [](int32_t) { return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU); };
    std::vector<std::unique_ptr<llama_memory_hybrid>> owners;
    std::vector<llama_kv_cache_kvarn *> attention_owners;
    std::vector<std::vector<uint8_t>> saved;
    for (uint32_t tail : {0u, 128u}) {
        auto attention = std::make_unique<llama_kv_cache_kvarn>(model, model.hparams,
                llama_kvarn_params_for_type(LLAMA_KVARN_K4V4_G128), false, false,
                512, 1, 256, 256, 1, 0, LLAMA_SWA_TYPE_NONE,
                [](int32_t il) { return il == 0; }, nullptr, tail, GGML_TYPE_F16, tail, 1,
                nullptr, nullptr, [](int32_t il) { return il == 0; }, GGML_TYPE_Q4_0,
                GGML_TYPE_Q4_0, false, cpu, 256, true);
        attention->set_mixed_state_identity(123, 456);
        auto state = attention->init_kv_batch({batch(0, 128)});
        assert(state && state->get_status() == LLAMA_MEMORY_STATUS_SUCCESS && state->apply());
        state->graph_compute_finish(GGML_STATUS_SUCCESS);
        auto recurrent = std::make_unique<llama_memory_recurrent>(model, GGML_TYPE_F32, GGML_TYPE_F32,
                false, 1, 1, 0, [](int32_t il) { return il == 1; });
        attention_owners.push_back(attention.get());
        owners.push_back(std::make_unique<llama_memory_hybrid>(
                model, std::move(attention), std::move(recurrent), 256));
        writer output;
        owners.back()->state_write(output, 0, 0);
        saved.push_back(std::move(output.data));
    }
    for (size_t destination = 0; destination < owners.size(); ++destination) {
        for (size_t source = 0; source < saved.size(); ++source) {
            writer before; owners[destination]->state_write(before, 0, 0);
            bool restored = false;
            reader input(saved[source]);
            try {
                owners[destination]->state_read(input, 0, 0);
                assert(input.n_bytes() == saved[source].size());
                input.commit(); restored = true;
            } catch (const std::exception &) { input.cancel(); }
            assert(restored == (source == destination));
            writer after; owners[destination]->state_write(after, 0, 0);
            if (!restored) { assert(before.data == after.data); }
            // This codec fixture populates attention metadata only; the
            // recurrent owner intentionally has no model-forward state.
            assert(attention_owners[destination]->seq_pos_min(0) == 0 &&
                    attention_owners[destination]->seq_pos_max(0) == 127);
        }
    }
}

static void test_native_resident_prefix_state() {
#ifndef _WIN32
    // Validate identical save/restore policy, including duplicate positions.
    {
        std::vector<llama_kv_mixed::cell_entry> cells(3);
        for (int i = 0; i < 3; ++i) { cells[i].pos = i; cells[i].seq_ids = {0}; }
        llama_validate_native_prefix(cells, 3, 10, 1);
        const auto rejects = [](const auto & c) {
            try { llama_validate_native_prefix(c, 3, 10, 1); } catch (const std::exception &) { return true; }
            return false;
        };
        auto duplicate = cells; duplicate[2].pos = 1; assert(rejects(duplicate));
        auto gap = cells; gap[1].pos = -1; assert(rejects(gap));
        auto wrong_owner = cells; wrong_owner[1].seq_ids = {1}; assert(rejects(wrong_owner));
    }
    const char * old_storage = std::getenv("GGML_RKVA_RANGE_STORAGE");
    const char * old_experimental = std::getenv("GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL");
    const std::string storage = old_storage ? old_storage : "";
    const std::string experimental = old_experimental ? old_experimental : "";
    setenv("GGML_RKVA_RANGE_STORAGE", "kvarn4", 1);
    setenv("GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL", "1", 1);
    for (bool sparse : {false, true}) {
        tiny_model model;
        auto cpu = [](int32_t) { return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU); };
        auto attention = std::make_unique<llama_kv_cache_kvarn>(model, model.hparams,
                llama_kvarn_params_for_type(LLAMA_KVARN_K4V4_G128), false, false,
                512, 1, 256, 256, 1, 0, LLAMA_SWA_TYPE_NONE,
                [](int32_t il) { return il == 0; }, nullptr, 0, GGML_TYPE_F16, 0, 1,
                nullptr, nullptr, [](int32_t il) { return il == 0; }, GGML_TYPE_Q4_0,
                GGML_TYPE_Q4_0, false, cpu, 256, true);
        auto * attn = attention.get();
        assert(!attn->supports_resident_prefix_restore());
        assert(attn->get_seq_rm_capability().suffix_rollback_tokens == 0);
        bool prefix_resident = true;
        auto recurrent = std::make_unique<llama_memory_recurrent>(model, GGML_TYPE_F32, GGML_TYPE_F32,
                false, 1, 1, 0, [](int32_t il) { return il == 1; });
        llama_memory_hybrid hybrid(model, std::move(attention), std::move(recurrent), 256);
        attn->set_mixed_state_identity(123, 456);
        const auto partial = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        auto append = [&](llama_pos p, uint32_t n) {
            auto state = attn->init_kv_batch({batch(p, n)});
            assert(state && state->get_status() == LLAMA_MEMORY_STATUS_SUCCESS && state->apply());
            state->graph_compute_finish(GGML_STATUS_SUCCESS);
        };
        auto save = [&]() { writer out; hybrid.state_write(out, 0, partial); return out.data; };
        auto restore = [&](const std::vector<uint8_t> & data, bool success) {
            bool restored = false;
            reader in(data);
            try { hybrid.state_read(in, 0, partial); in.commit(); restored = true; }
            catch (const std::exception & error) {
                if (success) std::fprintf(stderr, "native PARTIAL sparse=%d error=%s consumed=%zu/%zu\n", int(sparse), error.what(), in.n_bytes(), data.size());
                in.cancel();
            }
            assert(restored == success);
        };
        // Old-server append-only clients can still restore a fresh purely local
        // PARTIAL frame: no native owner exists and no overflow payload is involved.
        append(0, 2);
        const auto local_anchor = save();
        append(2, 1);
        restore(local_anchor, true);
        assert(attn->seq_pos_max(0) == 1);
        append(2, 2); // restored local metadata must remain appendable
        assert(attn->seq_pos_max(0) == 3);
        restore(local_anchor, true);
        hybrid.clear(false);
        restore(local_anchor, false); // fresh owner epoch rejects an old local reference
        assert(attn->seq_pos_max(0) == -1);
        append(0, 2);
        const auto fresh_local_anchor = save();
        restore(fresh_local_anchor, true);
        append(2, 254);
        append(256, 1); // first native append remains possible without rewind
        assert(attn->seq_pos_max(0) == 256);
        assert(!attn->can_seq_rm(0, 256, -1));
        restore(fresh_local_anchor, false); // an existing native owner needs negotiated rewind
        assert(attn->seq_pos_max(0) == 256);
        hybrid.clear(false);
        attn->set_native_kvarn_rewind_capable(true);
        attn->set_native_kvarn_rewind_preflight([&](llama_pos) { return prefix_resident; });
        assert(attn->supports_resident_prefix_restore());
        if (!sparse) append(0, 256);
        append(256, sparse ? 130 : 1); // sparse: two occupied groups after empty local cells
        const llama_pos anchor_end = sparse ? 386 : 257;

        // Exercise the actual mixed-memory body, including overflow metadata.
        // This is not an RKVK/Xbox roundtrip: the flag is an internal promise
        // that the outer file reader must independently enforce before commit.
        writer native_body;
        native_body.set_native_expected_tokens(uint32_t(anchor_end));
        bool saved = false;
        try { hybrid.state_write(native_body, 0, LLAMA_STATE_SEQ_FLAGS_NATIVE_FULL); saved = true; }
        catch (const std::exception &) {}
        assert(saved == !sparse); // a gap must never be published as a valid snapshot
        if (!sparse) {
            auto dest_attn = std::make_unique<llama_kv_cache_kvarn>(model, model.hparams,
                    llama_kvarn_params_for_type(LLAMA_KVARN_K4V4_G128), false, false,
                    512, 1, 256, 256, 1, 0, LLAMA_SWA_TYPE_NONE,
                    [](int32_t il) { return il == 0; }, nullptr, 0, GGML_TYPE_F16, 0, 1,
                    nullptr, nullptr, [](int32_t il) { return il == 0; }, GGML_TYPE_Q4_0,
                    GGML_TYPE_Q4_0, false, cpu, 256, true);
            dest_attn->set_mixed_state_identity(123, 456);
            auto * dest_cache = dest_attn.get();
            auto dest_recr = std::make_unique<llama_memory_recurrent>(model, GGML_TYPE_F32, GGML_TYPE_F32,
                    false, 1, 1, 0, [](int32_t il) { return il == 1; });
            llama_memory_hybrid dest(model, std::move(dest_attn), std::move(dest_recr), 256);
            int hybrid_clears = 0, cache_clears = 0;
            dest.set_on_clear([&]() { ++hybrid_clears; });
            dest_cache->set_on_clear([&]() { ++cache_clears; });
            auto read_body = [&](const std::vector<uint8_t> & bytes, llama_state_seq_flags flags, bool ok,
                                 uint32_t expected_count = 0) {
                reader in(bytes);
                in.set_native_expected_tokens(expected_count);
                bool accepted = false;
                try {
                    dest.state_read(in, 0, flags);
                    assert(in.n_bytes() == bytes.size());
                    assert(dest_cache->seq_pos_max(0) == -1); // staged, not yet published
                    in.commit();
                    accepted = true;
                } catch (const std::exception & error) {
                    if (ok) std::fprintf(stderr, "native FULL sparse=%d error=%s consumed=%zu/%zu\n",
                            int(sparse), error.what(), in.n_bytes(), bytes.size());
                    in.cancel();
                }
                assert(accepted == ok);
            };
            read_body(native_body.data, 0, false); // ordinary body cannot impersonate native
            assert(dest_cache->seq_pos_max(0) == -1);
            auto truncated = native_body.data;
            truncated.pop_back();
            read_body(truncated, LLAMA_STATE_SEQ_FLAGS_NATIVE_FULL, false, uint32_t(anchor_end));
            assert(dest_cache->seq_pos_max(0) == -1);
            read_body(native_body.data, LLAMA_STATE_SEQ_FLAGS_NATIVE_FULL, false, uint32_t(anchor_end + 1));
            assert(dest_cache->seq_pos_max(0) == -1);
            read_body(native_body.data, LLAMA_STATE_SEQ_FLAGS_NATIVE_FULL, true, uint32_t(anchor_end));
            assert(dest_cache->seq_pos_max(0) == anchor_end - 1);
            dest.clear(false);
            assert(hybrid_clears == 1 && cache_clears == 1);
            dest_cache->clear(false);
            assert(hybrid_clears == 1 && cache_clears == 2);
            assert(!dest_cache->state_seq_can_restore(0, 0)); // prior native owner remains guarded
            assert(dest_cache->state_seq_can_restore(0, LLAMA_STATE_SEQ_FLAGS_NATIVE_FULL));
            read_body(native_body.data, LLAMA_STATE_SEQ_FLAGS_NATIVE_FULL, true, uint32_t(anchor_end));
            assert(dest_cache->seq_pos_max(0) == anchor_end - 1);
        }

        const auto anchor = save();
        prefix_resident = false;
        const auto unchanged = save();
        assert(!attn->can_seq_rm(0, anchor_end - 1, -1));
        restore(anchor, false);
        assert(save() == unchanged); // failed preflight must precede staged writes
        prefix_resident = true;
        assert(attn->state_seq_can_restore(0, partial));
        assert(!attn->state_seq_can_save(0, 0)); // never serialize Q4 as native FULL
        append(anchor_end, 2);
        const auto stale = save();
        assert(!attn->can_seq_rm(0, anchor_end, anchor_end + 1)); // interior edit remains unsupported
        assert(attn->seq_rm(0, anchor_end, -1));
        append(anchor_end, 2); // same cells/positions/token IDs, different owner generation
        restore(stale, false);
        restore(anchor, true);
        restore(anchor, true);
        attn->set_native_kvarn_rewind_capable(false);
        assert(!attn->state_seq_can_restore(0, partial));
    }
    if (old_storage) setenv("GGML_RKVA_RANGE_STORAGE", storage.c_str(), 1); else unsetenv("GGML_RKVA_RANGE_STORAGE");
    if (old_experimental) setenv("GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL", experimental.c_str(), 1); else unsetenv("GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL");
#endif
}

int main() {
    ggml_backend_load_all();
    test_native_resident_prefix_state();
    test_position_range_presence();
    test_hybrid_tail_graph_reuse();
    test_exact_tail_snapshot_layouts();
    tiny_model model;
    auto cpu = [](int32_t) { return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU); };
    auto attention = std::make_unique<llama_kv_cache_kvarn>(model, model.hparams,
            llama_kvarn_params_for_type(LLAMA_KVARN_K4V4_G128), false, false,
            512, 1, 256, 256, 1, 0, LLAMA_SWA_TYPE_NONE,
            [](int32_t il) { return il == 0; }, nullptr, 0, GGML_TYPE_F16, 0, 1,
            nullptr, nullptr, [](int32_t il) { return il == 0; }, GGML_TYPE_Q4_0,
            GGML_TYPE_Q4_0, false, cpu, 256, true);
    attention->set_mixed_state_identity(123, 456);
    auto * attn = attention.get();
    auto recurrent = std::make_unique<llama_memory_recurrent>(model, GGML_TYPE_F32, GGML_TYPE_F32,
            false, 1, 1, 0, [](int32_t il) { return il == 1; });
    auto * rs = recurrent.get();
    llama_memory_hybrid hybrid(model, std::move(attention), std::move(recurrent), 256);
    const auto partial = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    assert(attn->requires_state_for_partial_restore());
    assert(hybrid.requires_state_for_partial_restore());
    auto capture = [&](llama_state_seq_flags flags) { writer out; hybrid.state_write(out, 0, flags); return out.data; };
    auto append = [&](llama_pos pos, uint32_t n) {
        auto state = attn->init_kv_batch({batch(pos, n)});
        assert(state && state->get_status() == LLAMA_MEMORY_STATUS_SUCCESS && state->apply());
        state->graph_compute_finish(GGML_STATUS_SUCCESS);
    };
    auto restore = [&](const std::vector<uint8_t> & saved, llama_state_seq_flags flags, bool success) {
        const auto before = capture(0);
        bool published = false;
        reader input(saved);
        std::string error;
        try {
            hybrid.state_read(input, 0, flags);
            assert(input.n_bytes() == saved.size());
            input.on_commit([&] { published = true; });
            input.commit();
        } catch (const std::exception & e) { error = e.what(); input.cancel(); }
        if (published != success) std::fprintf(stderr, "hybrid restore expected_success=%d published=%d consumed=%zu/%zu error=%s\n", int(success), int(published), input.n_bytes(), saved.size(), error.c_str());
        assert(published == success);
        if (!success) assert(capture(0) == before);
        return error;
    };

    writer legacy; rs->state_write(legacy, 0, partial);
    assert(!restore(legacy.data, partial, false).empty()); // actual old RS-only format
    append(0, 256); append(256, 1);
    const auto anchor = capture(partial); // prefix ending before r=257
    uint32_t magic = 0; std::memcpy(&magic, anchor.data(), sizeof(magic));
    assert(magic == 0x4D4B5351u); // genuine mixed attention envelope, not RS-only
    append(257, 1);
    const auto old_h = capture(partial); // includes the soon-to-be-reused row
    uint32_t old_cell = 0, new_cell = 0;
    assert(attn->cells_at_pos(0, 257, &old_cell, 1) == 1);
    assert(attn->seq_rm(0, 257, -1));
    append(257, 1); // same token and absolute position
    assert(attn->cells_at_pos(0, 257, &new_cell, 1) == 1 && old_cell == new_cell);
    float changed[128]; std::fill(changed, changed + 128, 7.0f);
    std::vector<uint8_t> packed(ggml_row_size(GGML_TYPE_Q4_0, 128));
    ggml_quantize_chunk(GGML_TYPE_Q4_0, changed, packed.data(), 0, 1, 128, nullptr);
    ggml_backend_tensor_set(attn->get_standard_cache()->get_k_storage(0), packed.data(), packed.size(), packed.size());
    assert(restore(old_h, partial, false).find("prefix") != std::string::npos);
    assert(restore(anchor, partial, true).empty());
    assert(restore(anchor, partial, true).empty()); // repeated same-owner replay
    assert(restore(old_h, partial, false).find("prefix") != std::string::npos);
    const auto full = capture(0);
    assert(restore(full, 0, true).empty()); // FULL mode remains intact
    assert(restore(anchor, partial, false).find("prefix") != std::string::npos);
    std::puts("Real hybrid PARTIAL: attention envelope, legacy refusal, stale-H rejection, earlier-anchor replay and FULL roundtrip passed");
}
