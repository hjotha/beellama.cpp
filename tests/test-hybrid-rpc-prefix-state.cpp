// CPU memory integration: no GGUF weights, model forward pass or remote host.
#include "../src/llama-model.h"
#include "../src/llama-kv-cache-kvarn.h"
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-io.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <cassert>
#include <cstring>
#include <cstdio>
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
    llama_ubatch ub;
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

int main() {
    ggml_backend_load_all();
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
