#pragma once

#include "ggml-backend.h"
#include "ggml-remote-attn.h"
#include "../../ggml/src/ggml-remote-attn/rkva-stage-queue.h"
#include "../../src/llama-batch.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// A private peer is published only after every layer has the same token prefix.
class server_rkva_ahead {
public:
    server_rkva_ahead(const char * host, uint16_t port, const ggml_remote_attn_geometry & geometry,
            uint32_t origin, ggml_type type, const char * upload_format, std::vector<int> layer_ids,
            std::vector<llama_token> tokens, const char * wire_format = nullptr)
        : origin_(origin), layer_ids_(std::move(layer_ids)), tokens_(std::move(tokens)), rows_(layer_ids_.size()) {
        if (tokens_.size() <= origin_ || geometry.n_layer_remote != layer_ids_.size()) return;
        const char * producer = std::getenv("GGML_RKVA_NATIVE_PRODUCER");
        if (producer && std::strcmp(producer, "cpu") != 0 && std::strcmp(producer, "cuda") != 0) return;
        const uint64_t groups = (uint64_t(origin_) + geometry.max_ctx + 127) / 128;
        const uint64_t owner_bytes = groups * geometry.n_head_kv * 2 * 8960 * 2 + (16ull << 20);
        if (available_memory() < owner_bytes * geometry.n_layer_remote + (1024ull << 20)) {
            std::fprintf(stderr, "RKVA ahead: insufficient host memory for the shadow producer\n");
            return;
        }
        backend_ = ggml_backend_remote_attn_init(host, port);
        if (backend_ && !ggml_remote_attn_reserve_stage(backend_)) {
            ggml_backend_free(backend_);
            backend_ = nullptr;
            return;
        }
        if (!backend_ || !ggml_backend_remote_attn_set_upload_format(backend_, upload_format) ||
                !ggml_backend_remote_attn_set_wire_format(backend_, wire_format) ||
                !ggml_backend_remote_attn_set_geometry(backend_, &geometry)) return;
        try {
            queue_ = std::make_unique<rkva_stage_queue>(size_t(geometry.n_head_kv) * geometry.head_dim, 256u << 20,
                [this, geometry, type, owner_bytes](const rkva_stage_queue::frame & frame) {
            if (!stage_) {
                if (!ggml_backend_remote_attn_connect(backend_) ||
                        !ggml_backend_remote_attn_create_range_session(backend_, 0, 0, geometry.max_ctx)) return false;
                ggml_backend_remote_attn_set_active_session(backend_, 0);
                if (!ggml_backend_remote_attn_configure_range(backend_, origin_, geometry.max_ctx, type)) return false;
                stage_ = ggml_backend_remote_attn_kvarn_stage_create(backend_);
                if (!stage_) return false;
            }
            if (frame.first_pos == origin_ && available_memory() < owner_bytes + (512ull << 20)) return false;
            return ggml_backend_remote_attn_native_kvarn_stage_append(backend_, frame.layer,
                    frame.first_pos, frame.count, frame.keys.data(), frame.values.data());
            }, 8ull << 30);
        } catch (...) {
            ggml_remote_attn_release_stage(backend_);
            ggml_backend_free(backend_);
            backend_ = nullptr;
            throw;
        }
    }

    ~server_rkva_ahead() {
        queue_.reset();
        const bool taken = stage_ && ggml_backend_remote_attn_kvarn_stage_taken(stage_);
        ggml_backend_remote_attn_kvarn_stage_free(stage_);
        ggml_remote_attn_release_stage(backend_);
        if (backend_ && !taken) ggml_backend_free(backend_);
    }
    server_rkva_ahead(const server_rkva_ahead &) = delete;
    server_rkva_ahead & operator=(const server_rkva_ahead &) = delete;

    bool valid() const { return backend_ && queue_ && !failed_; }
    uint32_t target() const { return uint32_t(tokens_.size()); }
    const std::vector<llama_token> & tokens() const { return tokens_; }
    ggml_remote_attn_kvarn_stage * stage() const { return stage_; }
    void invalidate() { fail("new request cannot reuse the staged prefix"); }

    bool extend(const std::vector<llama_token> & tokens, uint32_t target) {
        if (!valid() || finished_ || target > tokens.size() || target < tokens_.size() ||
                !std::equal(tokens_.begin(), tokens_.end(), tokens.begin())) return fail("new prompt changed the staged prefix");
        const uint32_t previous = uint32_t(tokens_.size()) - origin_;
        if (!std::all_of(rows_.begin(), rows_.end(), [previous](uint32_t n) { return n == previous; }))
            return fail("previous staged prefix is incomplete");
        try { tokens_.insert(tokens_.end(), tokens.begin() + tokens_.size(), tokens.begin() + target); }
        catch (...) { return fail("cannot extend the staged token lineage"); }
        return true;
    }

    bool ready() {
        if (!valid()) return false;
        if (finished_) return finish_ok_;
        const uint32_t expected = uint32_t(tokens_.size()) - origin_;
        if (!std::all_of(rows_.begin(), rows_.end(), [expected](uint32_t n) { return n == expected; })) return false;
        finished_ = true;
        try { finish_ok_ = queue_->drain() && stage_ && ggml_backend_remote_attn_kvarn_stage_finish(stage_, expected); }
        catch (...) { finish_ok_ = false; }
        std::fprintf(stderr, "RKVA ahead: staged_rows=%u spool_bytes=%llu ready=%d\n", expected,
                (unsigned long long) queue_->spooled_bytes(), int(finish_ok_));
        return finish_ok_;
    }

    static bool capture(void * user, ggml_backend_sched_t sched, ggml_cgraph * graph, const llama_ubatch * ubatch) {
        return static_cast<server_rkva_ahead *>(user)->capture_batch(sched, graph, ubatch);
    }

private:
    static uint64_t available_memory() {
#ifdef __linux__
        std::ifstream input("/proc/meminfo");
        std::string line;
        while (std::getline(input, line)) {
            if (line.compare(0, 13, "MemAvailable:") == 0) {
                std::istringstream value(line.substr(13));
                uint64_t kib = 0;
                if (value >> kib && kib <= UINT64_MAX / 1024) return kib * 1024;
                return 0;
            }
        }
        return 0;
#else
        return std::numeric_limits<uint64_t>::max();
#endif
    }

    bool capture_batch(ggml_backend_sched_t sched, ggml_cgraph * graph, const llama_ubatch * batch) {
        if (!valid()) return false;
        if (!batch || !batch->token || !batch->pos || !batch->n_seq_id || !batch->seq_id || batch->n_seqs_unq != 1) return fail("unsupported batch");
        const uint32_t expected = uint32_t(tokens_.size()) - origin_;
        if (std::all_of(rows_.begin(), rows_.end(), [expected](uint32_t n) { return n == expected; })) {
            for (uint32_t i = 0; i < batch->n_tokens; ++i) {
                if (batch->pos[i] >= int64_t(origin_) && batch->pos[i] < int64_t(tokens_.size()))
                    return fail("staged prefix was rewritten");
            }
            return true;
        }
        uint32_t begin = 0;
        while (begin < batch->n_tokens && batch->pos[begin] < int32_t(origin_)) ++begin;
        uint32_t end = begin;
        while (end < batch->n_tokens && batch->pos[end] < int64_t(tokens_.size())) ++end;
        if (begin == end) return true;
        for (uint32_t i = begin; i < end; ++i) {
            if (batch->pos[i] < 0 || size_t(batch->pos[i]) >= tokens_.size() ||
                    batch->n_seq_id[i] != 1 || batch->seq_id[i][0] != 0 ||
                    batch->token[i] != tokens_[batch->pos[i]]) return fail("token lineage mismatch");
        }
        std::vector<ggml_tensor *> keys(layer_ids_.size()), values(layer_ids_.size());
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * tensor = ggml_graph_node(graph, i);
            if (!tensor || tensor->op != GGML_OP_DUP || tensor->type != GGML_TYPE_F32 ||
                    tensor->ne[0] != 256 || tensor->ne[2] != batch->n_tokens ||
                    tensor->ne[3] != 1 || !ggml_is_contiguous(tensor)) continue;
            const bool key = std::strncmp(tensor->name, "rkva_ahead_k-", 13) == 0;
            if (!key && std::strncmp(tensor->name, "rkva_ahead_v-", 13) != 0) continue;
            int layer = -1, used = 0;
            if (std::sscanf(tensor->name + 13, "%d%n", &layer, &used) != 1 || tensor->name[13 + used]) continue;
            auto it = std::find(layer_ids_.begin(), layer_ids_.end(), layer);
            if (it != layer_ids_.end()) (key ? keys : values)[size_t(it - layer_ids_.begin())] = tensor;
        }
        ggml_backend_sched_synchronize(sched);
        for (size_t layer = 0; layer < layer_ids_.size(); ++layer) {
            auto * k = keys[layer]; auto * v = values[layer];
            if (!k || !v || k->ne[1] != v->ne[1]) return fail("missing captured tensor");
            if (batch->pos[begin] != int64_t(origin_ + rows_[layer])) return fail("non-contiguous prefix");
            for (uint32_t i = begin; i < end; ++i)
                if (batch->pos[i] != int64_t(origin_ + rows_[layer] + i - begin)) return fail("non-contiguous batch");
            const size_t width = size_t(k->ne[0]) * size_t(k->ne[1]);
            std::vector<float> key(size_t(batch->n_tokens) * width), value(key.size());
            ggml_backend_tensor_get(k, key.data(), 0, key.size() * sizeof(float));
            ggml_backend_tensor_get(v, value.data(), 0, value.size() * sizeof(float));
            if (!queue_->submit(uint32_t(layer), uint32_t(batch->pos[begin]),
                    end - begin, key.data() + size_t(begin) * width, value.data() + size_t(begin) * width)) return fail("upload queue refused a frame");
            rows_[layer] += end - begin;
        }
        return true;
    }
    bool fail(const char * reason) { std::fprintf(stderr, "RKVA ahead: %s\n", reason); failed_ = true; return false; }

    uint32_t origin_;
    std::vector<int> layer_ids_;
    std::vector<llama_token> tokens_;
    std::vector<uint32_t> rows_;
    ggml_backend_t backend_ = nullptr;
    ggml_remote_attn_kvarn_stage * stage_ = nullptr;
    std::unique_ptr<rkva_stage_queue> queue_;
    bool failed_ = false, finished_ = false, finish_ok_ = false;
};
