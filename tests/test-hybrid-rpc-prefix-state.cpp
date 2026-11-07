// CPU memory integration: no GGUF weights, model forward pass or remote host.
#include "../src/llama-model.h"
#include "../src/llama-kv-cache-kvarn.h"
#include "../src/llama-kv-mixed-state.h"
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-io.h"
#include "../src/llama-io-file.h"
#include "../src/llama-native-prefix-validation.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <cassert>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#ifndef _WIN32
#include <unistd.h>
#endif

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
    size_t first_tensor_offset = SIZE_MAX;
    void write(const void * src, size_t n) override {
        const auto * p = static_cast<const uint8_t *>(src);
        data.insert(data.end(), p, p + n);
    }
    void write_tensor(ggml_tensor * t, size_t offset, size_t n) override {
        if (first_tensor_offset == SIZE_MAX) first_tensor_offset = data.size();
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

#ifndef _WIN32
struct limited_index_reader final : llama_io_read_i {
    const std::vector<uint8_t> & data;
    size_t pos = 0;
    size_t indexed = 0;
    size_t cap;
    bool published = false;
    std::vector<std::function<void()>> callbacks;
    limited_index_reader(const std::vector<uint8_t> & bytes, size_t limit) : data(bytes), cap(limit) {}
    void read(void * dst, size_t n) override {
        if (n > data.size() - pos) throw std::runtime_error("truncated test state");
        std::memcpy(dst, data.data() + pos, n); pos += n;
    }
    bool has_bounded_tensor_reads() const override { return true; }
    void read_tensor(ggml_tensor *, size_t, size_t n) override {
        if (indexed >= cap) throw std::runtime_error("injected index budget exceeded");
        if (n > data.size() - pos) throw std::runtime_error("truncated test tensor");
        ++indexed; pos += n;
    }
    void stage_tensor_set(ggml_tensor *, const void *, size_t, size_t) override {
        throw std::runtime_error("unexpected materialized tensor staging");
    }
    void stage_tensor_clear(ggml_tensor *, size_t, size_t) override {
        throw std::runtime_error("unexpected tensor clear");
    }
    void on_commit(std::function<void()> callback) override { callbacks.push_back(std::move(callback)); }
    void commit() override { for (auto & callback : callbacks) callback(); published = true; }
    void cancel() override { callbacks.clear(); }
    size_t n_bytes() override { return pos; }
};

// A real standard Q4 body >8 MiB plus an F16 overlay.  This exercises the
// cache parser against the same indexed file reader used by the public API,
// without model inference or a GPU.
static void test_standard_tail_bounded_state() {
    tiny_model model;
    model.hparams.n_layer_all = 4;
    model.hparams.n_ctx_train = 32768;
    model.hparams.is_recr_impl.fill(false);
    model.layers.resize(4);
    auto cpu = [](int32_t) { return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU); };
    const auto make_cache = [&]() {
        return std::make_unique<llama_kv_cache>(model, model.hparams, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0,
                false, false, false, 32768, 1, 256, 0, LLAMA_SWA_TYPE_NONE, nullptr,
                llama_memory_i::layer_filter_cb{}, llama_memory_i::layer_reuse_cb{},
                llama_memory_i::layer_share_cb{}, "bounded_test_", 64, 128, GGML_TYPE_F16,
                128, false, 1, 0, false, cpu);
    };
    auto source = make_cache();
    auto destination = make_cache();
    assert(source->requires_state_for_partial_restore() && source->state_streaming_restore_supported());
    {
        llama_kv_cache multiple(model, model.hparams, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0,
                false, false, false, 512, 2, 256, 0, LLAMA_SWA_TYPE_NONE, nullptr,
                {}, {}, {}, "bounded_multi_", 64, 128, GGML_TYPE_F16, 128, false, 1, 0, false, cpu);
        assert(multiple.requires_state_for_partial_restore() && !multiple.state_streaming_restore_supported());
    }
    auto & cells = const_cast<llama_kv_cells &>(source->get_cells(llama_seq_id(0)));
    for (uint32_t i = 0; i < 32768 - 256; ++i) {
        cells.pos_set(i, llama_pos(i));
        cells.seq_add(i, 0);
    }
    for (uint32_t pos = 32768 - 256; pos < 32768; pos += 64) {
        auto state = source->init_kv_batch({batch(llama_pos(pos), 64)});
        assert(state && state->get_status() == LLAMA_MEMORY_STATUS_SUCCESS && state->apply());
        state->graph_compute_finish(GGML_STATUS_SUCCESS);
    }
    assert(source->seq_pos_max(0) == 32767);
    // Distinct deterministic raw payloads expose swapped body/tail destinations.
    ggml_context * view_ctx = ggml_init({ 32*ggml_tensor_overhead(), nullptr, true });
    assert(view_ctx);
    for (int layer = 0; layer < 4; ++layer) {
        uint32_t component = 0;
        for (ggml_tensor * tensor : { source->get_k_storage(layer), source->get_v_storage(layer),
                                     source->get_k_tail(view_ctx, layer), source->get_v_tail(view_ctx, layer) }) {
            assert(tensor);
            if (tensor->view_src && !tensor->buffer) ggml_backend_view_init(tensor);
            std::vector<uint8_t> bytes(ggml_nbytes(tensor));
            for (size_t i = 0; i < bytes.size(); ++i) {
                const size_t row = i/256;
                bytes[i] = uint8_t(i*29 + row*37 + (row >> 8)*11 + layer*17 + component*53);
            }
            if (component >= 2 && bytes.size() >= 512) {
                assert(!std::equal(bytes.begin(), bytes.begin() + 256, bytes.begin() + 256));
            }
            ggml_backend_tensor_set(tensor, bytes.data(), 0, bytes.size());
            ++component;
        }
    }
    ggml_free(view_ctx);
    writer full;
    source->state_write(full, 0, 0);
    assert(full.data.size() > LLAMA_STATE_FILE_BUFFER_SIZE && full.first_tensor_offset < full.data.size());
    char path[] = "/tmp/beellama-kv-tail-stream-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    close(fd);
    struct cleanup { const char * p; ~cleanup() { std::remove(p); } } unlink_file { path };
    const auto write = [&](const std::vector<uint8_t> & bytes) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
        assert(out.good());
    };
    write(full.data);
    {
        llama_file file(path, "rb");
        llama_io_read_file_stream input(file, full.data.size());
        uint32_t prefix = 0;
        input.read(&prefix, sizeof(prefix));
        llama_io_read_replay replay(input, &prefix, sizeof(prefix));
        assert(replay.has_bounded_tensor_reads());
        bool rejected = false;
        try { replay.read_tensor(destination->get_k_storage(0), 0, 1); }
        catch (const std::runtime_error &) { rejected = true; }
        assert(rejected && input.n_bytes() == sizeof(prefix));
        destination->state_read(replay, 0, 0);
        assert(input.n_bytes() == full.data.size() && input.checksum() == XXH64(full.data.data(), full.data.size(), 0));
        assert(destination->seq_pos_max(0) == -1);
        input.commit();
        assert(destination->seq_pos_max(0) == source->seq_pos_max(0));
    }
    writer restored;
    destination->state_write(restored, 0, 0);
    assert(restored.data == full.data);
    llama_kv_tail_coverage_info expected{}, actual{};
    assert(source->get_kv_tail_coverage(0, 0, expected) && destination->get_kv_tail_coverage(0, 0, actual));
    assert(expected.state == actual.state && expected.requested == actual.requested &&
           expected.exact == actual.exact && expected.degradation_flags == actual.degradation_flags);

    const auto reject = [&](const std::vector<uint8_t> & bytes, size_t expected_consumed) {
        auto empty = make_cache();
        write(bytes);
        const uint64_t checksum = XXH64(bytes.data(), bytes.size(), 0); // checksum of the mutated frame
        {
            std::ifstream disk(path, std::ios::binary);
            std::vector<char> block(1024*1024);
            XXH64_state_t hash; XXH64_reset(&hash, 0);
            while (disk) {
                disk.read(block.data(), block.size());
                XXH64_update(&hash, block.data(), size_t(disk.gcount()));
            }
            assert(disk.eof() && XXH64_digest(&hash) == checksum);
        }
        llama_file file(path, "rb");
        llama_io_read_file_stream input(file, bytes.size());
        bool rejected = false;
        try { empty->state_read(input, 0, 0); } catch (const std::exception &) { rejected = true; }
        assert(rejected && empty->seq_pos_max(0) == -1 && input.n_bytes() == expected_consumed);
    };
    writer body_only;
    source->state_write(body_only, 0, LLAMA_STATE_SEQ_FLAGS_BODY_ONLY);
    assert(body_only.data.size() > LLAMA_STATE_FILE_BUFFER_SIZE);
    reject(body_only.data, body_only.first_tensor_offset); // before touching the payload

    // Current version header: seven words, group length/string, three u64
    // payload lengths, then stream/v_trans/n_pos/saved_n_seq_max/layer_count.
    const auto u32 = [](const std::vector<uint8_t> & data, size_t pos) {
        assert(pos + 4 <= data.size()); uint32_t value; std::memcpy(&value, data.data() + pos, 4); return value;
    };
    const auto set_u32 = [](std::vector<uint8_t> & data, size_t pos, uint32_t value) {
        assert(pos + 4 <= data.size()); std::memcpy(data.data() + pos, &value, 4);
    };
    const size_t group_len_offset = 7*sizeof(uint32_t);
    const uint32_t group_len = u32(full.data, group_len_offset);
    assert(group_len > 0 && group_len < 64);
    const size_t manifest_offset = group_len_offset + sizeof(uint32_t) + group_len + 3*sizeof(uint64_t);
    {
        auto bad = full.data;
        set_u32(bad, group_len_offset, UINT32_MAX);
        reject(bad, group_len_offset + sizeof(uint32_t));
    }
    {
        auto bad = full.data;
        assert(u32(bad, manifest_offset + 3*sizeof(uint32_t)) == 1);
        set_u32(bad, manifest_offset + 3*sizeof(uint32_t), UINT32_MAX);
        reject(bad, manifest_offset + 5*sizeof(uint32_t));
    }
    size_t cursor = manifest_offset;
    assert(u32(full.data, cursor) == 1); cursor += 2*sizeof(uint32_t);
    const uint32_t npos = u32(full.data, cursor); cursor += 2*sizeof(uint32_t);
    const uint32_t nlayer = u32(full.data, cursor); cursor += sizeof(uint32_t);
    cursor += nlayer*(4*sizeof(uint32_t) + 2*sizeof(uint64_t) + sizeof(uint32_t));
    const uint32_t ncell = u32(full.data, cursor); cursor += sizeof(uint32_t);
    const uint32_t nruns = u32(full.data, cursor); cursor += sizeof(uint32_t) + size_t(nruns)*sizeof(uint32_t);
    for (uint32_t i = 0; i < ncell; ++i) {
        cursor += sizeof(uint32_t) + sizeof(uint64_t) + sizeof(llama_pos);
        if (npos > 1) cursor += sizeof(llama_kv_cell_ext);
        const uint32_t nseq = u32(full.data, cursor); cursor += sizeof(uint32_t) + size_t(nseq)*sizeof(llama_seq_id);
    }
    cursor += sizeof(uint64_t); // tail_ordinal
    {
        auto bad = full.data;
        assert(u32(bad, cursor) <= 129);
        set_u32(bad, cursor, UINT32_MAX); // record_count
        reject(bad, cursor + 4*sizeof(uint32_t));
    }
    {
        auto empty = make_cache();
        auto truncated = full.data; truncated.pop_back(); write(truncated);
        llama_file file(path, "rb");
        llama_io_read_file_stream input(file, truncated.size());
        bool rejected = false;
        try { empty->state_read(input, 0, 0); } catch (const std::exception &) { rejected = true; }
        assert(rejected && empty->seq_pos_max(0) == -1);
    }
    {
        auto empty_source = make_cache();
        auto empty_dest = make_cache();
        writer empty_frame; empty_source->state_write(empty_frame, 0, 0);
        write(empty_frame.data);
        llama_file file(path, "rb");
        llama_io_read_file_stream input(file, empty_frame.data.size());
        bool published = false;
        empty_dest->state_read(input, 0, 0);
        input.on_commit([&] { published = true; });
        assert(!published && empty_dest->seq_pos_max(0) == -1 && input.n_bytes() == empty_frame.data.size());
        input.commit();
        assert(published && empty_dest->seq_pos_max(0) == -1);
        writer after; empty_dest->state_write(after, 0, 0);
        assert(after.data == empty_frame.data);
    }
    {
        // Keep one contiguous payload run but remap its eight destination
        // cells to even slots. This exercises splitting a single wire run into
        // fragmented read_tensor operations rather than just multiple runs.
        const auto make_small = [&]() {
            return std::make_unique<llama_kv_cache>(model, model.hparams, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0,
                    false, false, false, 512, 1, 256, 0, LLAMA_SWA_TYPE_NONE, nullptr,
                    llama_memory_i::layer_filter_cb{}, llama_memory_i::layer_reuse_cb{},
                    llama_memory_i::layer_share_cb{}, "bounded_small_", 64, 128, GGML_TYPE_F16,
                    128, false, 1, 0, false, cpu);
        };
        auto contiguous = make_small();
        auto & c = const_cast<llama_kv_cells &>(contiguous->get_cells(llama_seq_id(0)));
        for (uint32_t i = 0; i < 8; ++i) { c.pos_set(i, llama_pos(i)); c.seq_add(i, 0); }
        for (int layer = 0; layer < 4; ++layer) {
            int kind = 0;
            for (ggml_tensor * tensor : { contiguous->get_k_storage(layer), contiguous->get_v_storage(layer) }) {
                const size_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);
                std::vector<uint8_t> row(row_size);
                for (uint32_t i = 0; i < 8; ++i) {
                    for (size_t j = 0; j < row.size(); ++j) row[j] = uint8_t(i*37 + j*13 + layer*19 + kind*53);
                    ggml_backend_tensor_set(tensor, row.data(), size_t(i)*row_size, row.size());
                }
                ++kind;
            }
        }
        writer state; contiguous->state_write(state, 0, 0);
        const size_t length_offset = 7*sizeof(uint32_t);
        const size_t manifest = length_offset + sizeof(uint32_t) + u32(state.data, length_offset) + 3*sizeof(uint64_t);
        const uint32_t pos_dims = u32(state.data, manifest + 2*sizeof(uint32_t));
        const uint32_t layer_count = u32(state.data, manifest + 4*sizeof(uint32_t));
        size_t offset = manifest + 5*sizeof(uint32_t) + layer_count*(5*sizeof(uint32_t) + 2*sizeof(uint64_t));
        assert(u32(state.data, offset) == 8 && u32(state.data, offset + 4) == 1 &&
               u32(state.data, offset + 8) == 8);
        offset += 3*sizeof(uint32_t);
        for (uint32_t i = 0; i < 8; ++i) {
            assert(u32(state.data, offset) == i);
            set_u32(state.data, offset, 2*i);
            offset += sizeof(uint32_t) + sizeof(uint64_t) + sizeof(llama_pos);
            if (pos_dims > 1) offset += sizeof(llama_kv_cell_ext);
            const uint32_t nseq = u32(state.data, offset); offset += sizeof(uint32_t) + nseq*sizeof(llama_seq_id);
        }
        const uint64_t checksum = XXH64(state.data.data(), state.data.size(), 0);
        write(state.data); // the test frame's checksum is recomputed after remapping
        auto remapped = make_small();
        llama_file file(path, "rb");
        llama_io_read_file_stream input(file, state.data.size());
        remapped->state_read(input, 0, 0);
        assert(input.checksum() == checksum && remapped->seq_pos_max(0) == -1);
        input.commit();
        assert(remapped->seq_pos_max(0) == 7);
        for (int layer = 0; layer < 4; ++layer) {
            for (bool key : {true, false}) {
                auto * from = key ? contiguous->get_k_storage(layer) : contiguous->get_v_storage(layer);
                auto * to = key ? remapped->get_k_storage(layer) : remapped->get_v_storage(layer);
                const size_t row_size = ggml_row_size(from->type, from->ne[0]);
                std::vector<uint8_t> a(row_size), b(row_size);
                for (uint32_t i = 0; i < 8; ++i) {
                    ggml_backend_tensor_get(from, a.data(), size_t(i)*row_size, row_size);
                    ggml_backend_tensor_get(to, b.data(), size_t(2*i)*row_size, row_size);
                    assert(a == b);
                }
            }
        }
    }
    {
        // Each saved run maps to a separate destination row. A bounded reader
        // rejecting its third index must fail before any logical publication.
        auto sparse = make_cache();
        auto & sparse_cells = const_cast<llama_kv_cells &>(sparse->get_cells(llama_seq_id(0)));
        for (uint32_t i = 0; i < 8; ++i) {
            sparse_cells.pos_set(2*i, llama_pos(i));
            sparse_cells.seq_add(2*i, 0);
        }
        writer state;
        // Rows are intentionally distinct so a reordered fragmented restore
        // cannot pass by copying identical payloads into the wrong cells.
        for (int layer = 0; layer < 4; ++layer) {
            int kind = 0;
            for (ggml_tensor * tensor : { sparse->get_k_storage(layer), sparse->get_v_storage(layer) }) {
                const size_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);
                std::vector<uint8_t> row(row_size);
                for (uint32_t i = 0; i < 8; ++i) {
                    for (size_t j = 0; j < row.size(); ++j) row[j] = uint8_t(i*37 + j*13 + layer*19 + kind*53);
                    ggml_backend_tensor_set(tensor, row.data(), size_t(2*i)*row_size, row.size());
                }
                ++kind;
            }
        }
        sparse->state_write(state, 0, 0);
        {
            auto sparse_dest = make_cache();
            write(state.data);
            llama_file file(path, "rb");
            llama_io_read_file_stream input(file, state.data.size());
            sparse_dest->state_read(input, 0, 0);
            assert(sparse_dest->seq_pos_max(0) == -1 && input.n_bytes() == state.data.size());
            input.commit();
            assert(sparse_dest->seq_pos_max(0) == 7);
            writer after; sparse_dest->state_write(after, 0, 0);
            assert(after.data == state.data);
        }
        {
            // A legacy reader may still restore one selected sequence from a
            // snapshot whose source capacity was larger than one.
            auto legacy_state = state.data;
            const size_t length_offset = 7*sizeof(uint32_t);
            const uint32_t length = u32(legacy_state, length_offset);
            const size_t manifest = length_offset + sizeof(uint32_t) + length + 3*sizeof(uint64_t);
            set_u32(legacy_state, manifest + 3*sizeof(uint32_t), 2);
            auto legacy_dest = make_cache();
            reader legacy(legacy_state);
            legacy_dest->state_read(legacy, 0, 0);
            legacy.commit();
            assert(legacy_dest->seq_pos_max(0) == 7);
        }
        auto empty = make_cache();
        limited_index_reader input(state.data, 2);
        bool rejected = false;
        try { empty->state_read(input, 0, 0); } catch (const std::runtime_error & error) {
            rejected = std::string(error.what()).find("index budget exceeded") != std::string::npos;
        }
        assert(rejected && input.indexed == 2 && !input.published && empty->seq_pos_max(0) == -1);
        input.cancel();
    }
    std::puts("PASS: standard F16 tail >8 MiB bounded restore, replay, coverage and early malformed-frame refusal");
}
#endif

static void test_standard_overflow_full_state() {
#ifndef _WIN32
    const char * previous = std::getenv("GGML_RKVA_UPLOAD_FORMAT");
    const bool had_previous = previous != nullptr;
    const std::string saved = previous ? previous : "";
    for (const auto format : {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}) {
        setenv("GGML_RKVA_UPLOAD_FORMAT", format == GGML_TYPE_Q4_0 ? "q4_0" : "q8_0", 1);
        tiny_model model;
        auto cpu = [](int32_t) { return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU); };
        auto attention = std::make_unique<llama_kv_cache_kvarn>(model, model.hparams,
                llama_kvarn_params_for_type(LLAMA_KVARN_K4V4_G128), false, false,
                512, 1, 256, 256, 1, 0, LLAMA_SWA_TYPE_NONE,
                [](int32_t il) { return il == 0; }, nullptr, 0, GGML_TYPE_F16, 0, 1,
                nullptr, nullptr, [](int32_t il) { return il == 0; }, format,
                format, false, cpu, 256, true);
        attention->set_mixed_state_identity(123, 456);
        auto * attn = attention.get();
        auto recurrent = std::make_unique<llama_memory_recurrent>(model, GGML_TYPE_F32, GGML_TYPE_F32,
                false, 1, 1, 0, [](int32_t il) { return il == 1; });
        llama_memory_hybrid hybrid(model, std::move(attention), std::move(recurrent), 256);
        const auto append = [&](llama_pos p, uint32_t n) {
            auto state = attn->init_kv_batch({batch(p, n)});
            assert(state && state->get_status() == LLAMA_MEMORY_STATUS_SUCCESS && state->apply());
            state->graph_compute_finish(GGML_STATUS_SUCCESS);
        };
        append(0, 256);
        append(256, 1); // a real occupied overflow cell
        assert(hybrid.state_seq_can_save(0, 0));
        writer full;
        hybrid.state_write(full, 0, 0);
        assert(hybrid.seq_rm(0, 0, -1));
        assert(hybrid.state_seq_can_restore(0, 0));
        assert(!hybrid.state_seq_can_restore(0, LLAMA_STATE_SEQ_FLAGS_NATIVE_FULL));
        reader in(full.data);
        hybrid.state_read(in, 0, 0);
        assert(attn->seq_pos_max(0) == -1);
        in.commit();
        assert(attn->seq_pos_max(0) == 256);
        append(257, 1);
        assert(attn->seq_pos_max(0) == 257);
    }
    if (had_previous) setenv("GGML_RKVA_UPLOAD_FORMAT", saved.c_str(), 1);
    else unsetenv("GGML_RKVA_UPLOAD_FORMAT");
#endif
}

// Native row RPC state is not the Q4 companion, even though that companion is
// used for allocation and metadata. Test the real hybrid reader and lineage.
static void test_native_row_snapshot_lifecycle() {
#ifndef _WIN32
    struct saved_env {
        const char * name;
        bool present;
        std::string value;
        explicit saved_env(const char * n) : name(n), present(std::getenv(n) != nullptr),
            value(present ? std::getenv(n) : "") {}
        ~saved_env() { if (present) setenv(name, value.c_str(), 1); else unsetenv(name); }
    } save_format("GGML_RKVA_UPLOAD_FORMAT"), save_legacy("GGML_RKVA_FORMAT"),
      save_storage("GGML_RKVA_RANGE_STORAGE"), save_enable("GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL");
    unsetenv("GGML_RKVA_FORMAT"); unsetenv("GGML_RKVA_RANGE_STORAGE");
    unsetenv("GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL");
    tiny_model model;
    auto cpu = [](int32_t) { return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU); };
    const auto make_attention = [&](ggml_type companion) {
        auto p = std::make_unique<llama_kv_cache_kvarn>(model, model.hparams,
                llama_kvarn_params_for_type(LLAMA_KVARN_K4V4_G128), false, false,
                512, 1, 256, 256, 1, 0, LLAMA_SWA_TYPE_NONE,
                [](int32_t il) { return il == 0; }, nullptr, 0, GGML_TYPE_F16, 0, 1,
                nullptr, nullptr, [](int32_t il) { return il == 0; }, companion,
                companion, false, cpu, 256, true);
        p->set_mixed_state_identity(123, 456);
        return p;
    };
    const auto make_recurrent = [&]() {
        return std::make_unique<llama_memory_recurrent>(model, GGML_TYPE_F32, GGML_TYPE_F32,
                false, 1, 1, 0, [](int32_t il) { return il == 1; });
    };
    const auto append = [](llama_kv_cache_kvarn * a, llama_pos pos, uint32_t n) {
        auto state = a->init_kv_batch({batch(pos, n)});
        assert(state && state->get_status() == LLAMA_MEMORY_STATUS_SUCCESS && state->apply());
        state->graph_compute_finish(GGML_STATUS_SUCCESS);
    };

    // Create legitimate standard-Q4/Q8 FULL bodies with occupied overflow.
    std::vector<uint8_t> full[2];
    for (int i = 0; i < 2; ++i) {
        setenv("GGML_RKVA_UPLOAD_FORMAT", i ? "q8_0" : "q4_0", 1);
        auto source_attn = make_attention(i ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_0);
        auto * source = source_attn.get();
        llama_memory_hybrid generic(model, std::move(source_attn), make_recurrent(), 256);
        append(source, 0, 256); append(source, 256, 1);
        writer out; generic.state_write(out, 0, 0); full[i] = std::move(out.data);
    }

    struct scenario { const char * name; ggml_type companion; bool negotiated_rows; };
    for (const scenario test : {
            scenario{"saw_int4", GGML_TYPE_Q4_0, false}, {"saw-int4", GGML_TYPE_Q4_0, false},
            {"saw4", GGML_TYPE_Q4_0, false}, {"bdr_int4", GGML_TYPE_Q4_0, false},
            {"turboquant_nc", GGML_TYPE_Q4_0, false}, {"turboquant-nc", GGML_TYPE_Q4_0, false},
            {"tqnc", GGML_TYPE_Q4_0, false}, {"turboquant_4bit_nc", GGML_TYPE_Q4_0, false},
            {"q4_0", GGML_TYPE_Q4_0, true}, {"q8_0", GGML_TYPE_Q8_0, true}}) {
        const char * only = std::getenv("RKVA_SNAPSHOT_TEST_CASE");
        if (only && std::strcmp(only, test.name)) continue;
        if (std::getenv("RKVA_SNAPSHOT_TEST_TRACE")) std::fprintf(stderr, "snapshot scenario=%s\n", test.name);
        setenv("GGML_RKVA_UPLOAD_FORMAT", test.name, 1);
        auto attention = make_attention(test.companion);
        auto * attn = attention.get();
        if (test.negotiated_rows) {
            assert(attn->supports_resident_prefix_restore()); // ordinary generic Q4/Q8 before ACK
            attn->set_negotiated_native_rpc_owner(false); // actual layout 4/5, no snapshot codec
        }
        llama_memory_hybrid hybrid(model, std::move(attention), make_recurrent(), 256);
        assert(!attn->supports_resident_prefix_restore());
        assert(attn->get_seq_rm_capability().suffix_rollback_tokens == 0);
        auto restore = [&](const std::vector<uint8_t> & bytes, llama_state_seq_flags flags, bool expected,
                           const char * expected_error = nullptr) {
            reader in(bytes); bool accepted = false;
            std::string error;
            try { hybrid.state_read(in, 0, flags); in.commit(); accepted = true; }
            catch (const std::exception & e) { error = e.what(); in.cancel(); }
            assert(accepted == expected);
            if (expected_error) assert(error.find(expected_error) != std::string::npos);
        };
        writer empty_before; hybrid.state_write(empty_before, 0, 0);
        const auto & standard_full = full[test.companion == GGML_TYPE_Q8_0 ? 1 : 0];
        restore(standard_full, 0, false, "cannot restore a standard-cache overflow snapshot");
        writer empty_after; hybrid.state_write(empty_after, 0, 0);
        assert(empty_before.data == empty_after.data && attn->seq_pos_max(0) == -1);
        restore(standard_full, LLAMA_STATE_SEQ_FLAGS_NATIVE_FULL, false);
        assert(attn->seq_pos_max(0) == -1);

        append(attn, 0, 256);
        writer local; hybrid.state_write(local, 0, 0);
        append(attn, 256, 1);
        assert(!attn->state_seq_can_save(0, 0));
        assert(!attn->can_seq_rm(0, 256, -1));
        bool full_saved = false;
        try { writer full; hybrid.state_write(full, 0, 0); full_saved = true; } catch (const std::exception &) {}
        assert(!full_saved);
        bool native_saved = false;
        try { writer full; hybrid.state_write(full, 0, LLAMA_STATE_SEQ_FLAGS_NATIVE_FULL); native_saved = true; }
        catch (const std::exception &) {}
        assert(!native_saved);

        attn->set_native_kvarn_rewind_capable(true);
        bool resident = false;
        unsigned preflights = 0;
        attn->set_native_kvarn_rewind_preflight([&](llama_pos) { ++preflights; return resident; });
        const auto partial = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        auto checkpoint = [&]() { writer out; hybrid.state_write(out, 0, partial); return out.data; };
        const auto anchor = checkpoint();
        restore(anchor, partial, false, "native partial restore is outside the retained original prefix window");
        assert(preflights != 0 && attn->seq_pos_max(0) == 256 && checkpoint() == anchor);
        resident = true;
        append(attn, 257, 2);
        const auto stale = checkpoint();
        assert(attn->seq_rm(0, 257, -1));
        append(attn, 257, 2);
        restore(stale, partial, false, "mixed partial snapshot prefix was changed or its lineage expired");
        restore(anchor, partial, true);
        assert(attn->seq_pos_max(0) == 256);
        assert(hybrid.seq_rm(0, 0, -1));
        assert(!hybrid.state_seq_can_restore(0, 0)); // historical owner latch survives clear
        // NATIVE_FULL admission here only asks whether a clean destination can
        // be considered; the actual row FULL body still cannot be consumed.
        assert(hybrid.state_seq_can_restore(0, LLAMA_STATE_SEQ_FLAGS_NATIVE_FULL));
        restore(standard_full, LLAMA_STATE_SEQ_FLAGS_NATIVE_FULL, false);
        restore(local.data, 0, true); // only a local prefix may be re-admitted
        assert(attn->seq_pos_max(0) == 255);
    }
    // Negotiation cannot switch ownership after any live rows have been stored.
    for (ggml_type type : {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}) {
        setenv("GGML_RKVA_UPLOAD_FORMAT", type == GGML_TYPE_Q4_0 ? "q4_0" : "q8_0", 1);
        auto late = make_attention(type);
        append(late.get(), 0, 1);
        bool refused = false;
        try { late->set_negotiated_native_rpc_owner(false); } catch (const std::logic_error &) { refused = true; }
        assert(refused && late->seq_pos_max(0) == 0 && late->supports_resident_prefix_restore());
    }
#endif
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
        attn->set_negotiated_native_rpc_owner(true); // native KVarN records retain their codec path
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
        writer full_local;
        hybrid.state_write(full_local, 0, 0);
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
        assert(!hybrid.state_seq_can_restore(0, LLAMA_STATE_SEQ_FLAGS_NATIVE_FULL));
        assert(hybrid.seq_rm(0, 0, -1)); // same full-range removal as slot erase
        // A full erase preserves the historical latch. The file layer may use
        // NATIVE_FULL only for admission after proving a healthy empty remote
        // owner, while the actual mixed body must still be parsed as flags=0.
        assert(!hybrid.state_seq_can_restore(0, 0));
        assert(hybrid.state_seq_can_restore(0, LLAMA_STATE_SEQ_FLAGS_NATIVE_FULL));
        reader local_in(full_local.data);
        hybrid.state_read(local_in, 0, 0);
        assert(local_in.n_bytes() == full_local.data.size());
        assert(attn->seq_pos_max(0) == -1); // no publication before commit
        local_in.commit();
        assert(attn->seq_pos_max(0) == 1);
        append(2, 1); // restored local prefix remains appendable
        assert(attn->seq_pos_max(0) == 2);
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
            assert(native_body.data.size() >= 16);
            uint64_t frame_bytes = 0;
            std::memcpy(&frame_bytes, native_body.data.data() + 8, sizeof(frame_bytes));
            assert(frame_bytes <= native_body.data.size() - 16);
            const auto frame = llama_kv_mixed::snapshot::parse(native_body.data.data() + 16, size_t(frame_bytes));
            size_t overflow_layers = 0;
            for (const auto & layer : frame.layers()) {
                if (layer.range_id == llama_position_split::kRangeOverflow) {
                    ++overflow_layers;
                    assert(layer.payload_bytes == 0); // native RKVK owns the real records
                }
            }
            assert(overflow_layers != 0);
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

static void test_native_companion_full_body_and_reset() {
#ifndef _WIN32
    const char * names[] = {"GGML_RKVA_UPLOAD_FORMAT", "GGML_RKVA_NATIVE_COMPANION", "GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL"};
    std::vector<std::string> old;
    std::vector<bool> present;
    for (const char * name : names) {
        const char * value = std::getenv(name);
        present.push_back(value != nullptr);
        old.emplace_back(value ? value : "");
    }
    setenv(names[0], "kvarn4", 1);
    setenv(names[1], "1", 1);
    setenv(names[2], "1", 1);
    tiny_model model;
    model.hparams.n_embd = 256;
    model.hparams.n_embd_head_k_full = model.hparams.n_embd_head_v_full = 256;
    model.hparams.n_rot_full = 256;
    auto make_cache = [&]() {
        auto result = std::make_unique<llama_kv_cache_kvarn>(model, model.hparams,
                llama_kvarn_params_for_type(LLAMA_KVARN_K4V4_G128), false, false,
                512, 1, 256, 256, 1, 0, LLAMA_SWA_TYPE_NONE,
                [](int32_t il) { return il == 0; }, nullptr, 0, GGML_TYPE_F16, 0, 1,
                nullptr, nullptr, nullptr, GGML_TYPE_COUNT, GGML_TYPE_COUNT, false, nullptr, 256, true);
        result->set_mixed_state_identity(123, 456);
        result->set_negotiated_native_rpc_owner(true);
        return result;
    };
    auto source = make_cache();
    assert(source->native_companion_enabled() && !source->has_standard_cache());
    for (const auto & span : {std::pair<llama_pos, uint32_t>{0, 256}, {256, 1}}) {
        auto state = source->init_kv_batch({batch(span.first, span.second)});
        assert(state && state->get_status() == LLAMA_MEMORY_STATUS_SUCCESS && state->apply());
        state->graph_compute_finish(GGML_STATUS_SUCCESS);
    }
    const auto full = LLAMA_STATE_SEQ_FLAGS_NATIVE_FULL;
    assert(source->state_seq_can_save(0, full));
    assert(!source->state_seq_can_save(0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
    source->set_native_kvarn_rewind_capable(true);
    source->set_native_kvarn_rewind_preflight([](llama_pos) { return true; });
    assert(!source->can_seq_rm(0, 256, -1));
    writer saved;
    saved.set_native_expected_tokens(257);
    source->state_write(saved, 0, full);
    // Legacy native FULL bodies carried an empty Q4/Q8 companion descriptor;
    // the outer file reader still has to validate the paired RKVK trailer.
    auto cpu = [](int32_t) { return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU); };
    for (const auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}) {
        auto legacy = std::make_unique<llama_kv_cache_kvarn>(model, model.hparams,
                llama_kvarn_params_for_type(LLAMA_KVARN_K4V4_G128), false, false,
                512, 1, 256, 256, 1, 0, LLAMA_SWA_TYPE_NONE,
                [](int32_t il) { return il == 0; }, nullptr, 0, GGML_TYPE_F16, 0, 1,
                nullptr, nullptr, [](int32_t il) { return il == 0; }, type, type, false, cpu, 256, true);
        legacy->set_mixed_state_identity(123, 456);
        legacy->set_negotiated_native_rpc_owner(true);
        for (const auto & span : {std::pair<llama_pos, uint32_t>{0, 256}, {256, 1}}) {
            auto state = legacy->init_kv_batch({batch(span.first, span.second)});
            assert(state && state->get_status() == LLAMA_MEMORY_STATUS_SUCCESS && state->apply());
            state->graph_compute_finish(GGML_STATUS_SUCCESS);
        }
        writer legacy_body;
        legacy_body.set_native_expected_tokens(257);
        legacy->state_write(legacy_body, 0, full);
        const auto original_bytes = legacy_body.data;
        auto adapted = make_cache();
        reader legacy_input(legacy_body.data);
        legacy_input.set_native_expected_tokens(257);
        adapted->state_read(legacy_input, 0, full);
        assert(adapted->seq_pos_max(0) < 0);
        legacy_input.commit();
        assert(adapted->seq_pos_max(0) == 256 && legacy_body.data == original_bytes);
        auto wrong_identity = make_cache();
        wrong_identity->set_mixed_state_identity(123, 999);
        reader rejected(legacy_body.data);
        rejected.set_native_expected_tokens(257);
        bool failed = false;
        try { wrong_identity->state_read(rejected, 0, full); } catch (const std::exception &) { failed = true; }
        assert(failed && wrong_identity->seq_pos_max(0) < 0 && legacy_body.data == original_bytes);
        auto wrong_prefix = make_cache();
        reader wrong_count(legacy_body.data);
        wrong_count.set_native_expected_tokens(258);
        failed = false;
        try { wrong_prefix->state_read(wrong_count, 0, full); } catch (const std::exception &) { failed = true; }
        assert(failed && wrong_prefix->seq_pos_max(0) < 0 && legacy_body.data == original_bytes);
        auto corrupt = legacy_body.data;
        corrupt.pop_back();
        auto truncated = make_cache();
        reader short_input(corrupt);
        short_input.set_native_expected_tokens(257);
        failed = false;
        try { truncated->state_read(short_input, 0, full); } catch (const std::exception &) { failed = true; }
        assert(failed && truncated->seq_pos_max(0) < 0 && legacy_body.data == original_bytes);
    }
    auto destination = make_cache();
    reader input(saved.data);
    input.set_native_expected_tokens(257);
    destination->state_read(input, 0, full);
    assert(destination->seq_pos_max(0) < 0);
    input.commit();
    assert(destination->seq_pos_max(0) == 256);
    destination->set_native_companion_reset([]() { return false; });
    bool refused = false;
    try { destination->clear(false); } catch (const std::runtime_error &) { refused = true; }
    assert(refused && destination->seq_pos_max(0) == 256);
    int resets = 0;
    destination->set_native_companion_reset([&]() { ++resets; return true; });
    destination->clear(false);
    assert(resets == 1 && destination->seq_pos_max(0) < 0);
    destination->clear(false);
    assert(resets == 2 && destination->seq_pos_max(0) < 0);
    auto hybrid_attention = make_cache();
    auto * hybrid_cache = hybrid_attention.get();
    auto state = hybrid_cache->init_kv_batch({batch(0, 1)});
    assert(state && state->get_status() == LLAMA_MEMORY_STATUS_SUCCESS && state->apply());
    state->graph_compute_finish(GGML_STATUS_SUCCESS);
    auto recurrent = std::make_unique<llama_memory_recurrent>(model, GGML_TYPE_F32, GGML_TYPE_F32,
            false, 1, 1, 0, [](int32_t il) { return il == 1; });
    llama_memory_hybrid hybrid(model, std::move(hybrid_attention), std::move(recurrent), 256);
    hybrid_cache->set_native_companion_reset([]() { return false; });
    llama_memory_clear(&hybrid, false);
    assert(hybrid_cache->seq_pos_max(0) == 0);
    refused = false;
    try { hybrid.clear(false); } catch (const std::runtime_error &) { refused = true; }
    assert(refused && hybrid_cache->seq_pos_max(0) == 0);
    int hybrid_resets = 0;
    hybrid_cache->set_native_companion_reset([&]() { ++hybrid_resets; return true; });
    hybrid.clear(false);
    assert(hybrid_resets == 1 && hybrid_cache->seq_pos_max(0) < 0);
    for (size_t i = 0; i < old.size(); ++i) {
        if (present[i]) setenv(names[i], old[i].c_str(), 1); else unsetenv(names[i]);
    }
#endif
}

int main() {
    ggml_backend_load_all();
    test_standard_overflow_full_state();
    test_native_row_snapshot_lifecycle();
    test_native_resident_prefix_state();
    test_native_companion_full_body_and_reset();
    test_position_range_presence();
    test_hybrid_tail_graph_reuse();
    test_exact_tail_snapshot_layouts();
#ifndef _WIN32
    test_standard_tail_bounded_state();
#endif
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
