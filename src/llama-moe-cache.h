#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;

// One batch of expert uploads in a scheduler copy callback.
class llama_moe_upload {
public:
    explicit llama_moe_upload(ggml_backend_t backend);
    ~llama_moe_upload();
    llama_moe_upload(const llama_moe_upload &) = delete;
    llama_moe_upload & operator=(const llama_moe_upload &) = delete;

    void set(ggml_tensor * tensor, const void * data, size_t offset, size_t size);

private:
    using upload_fn = void (*)(ggml_backend_t, ggml_tensor *, const void *, size_t, size_t, bool);
    ggml_backend_t backend;
    upload_fn upload = nullptr;
    bool pending = false;
};

// counters for the large-batch fill path of a partial group, see prepare_layer
struct llama_moe_cache_stats {
    size_t large_batches    = 0; // large batches that consulted the cache
    size_t large_hits       = 0; // selected experts that were already resident
    size_t large_fills      = 0; // selected experts admitted to the bank
    size_t large_bypass     = 0; // selected experts that did not fit and were copied directly
    size_t large_fill_bytes = 0; // host-to-device bytes for the bank fills
    size_t large_copy_bytes = 0; // host-to-device bytes for the direct scheduler copy
};

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size);
    ~llama_moe_cache();

    static bool prefill_enabled();

    // LLAMA_MOE_CACHE_PARTIAL_PREFILL=1: large batches fill and reuse the slots of a partial
    // group through the scheduler copy path, even though they cannot use its slot map
    static bool partial_prefill_enabled();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    // LLAMA_MOE_CACHE_PREFILL=1 also admits large batches when all experts of the group fit
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // start a graph compute: prepare_layer plans each layer at most once until the next call
    void begin_batch();

    // for a large batch whose group does not fit, fill the selected experts of w's layer that
    // fit in its quota; returns true if the layer was already prepared for this batch
    bool prepare_layer(ggml_backend_t backend, const ggml_tensor * w,
            const int32_t * ids, size_t n_ids, int64_t n_tokens);

    // account direct host-to-device bytes of the scheduler copy of a cached expert tensor
    void note_copy_upload(size_t bytes);

    llama_moe_cache_stats get_stats() const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;

// unit-test hooks for the partial planner and admission, not used by the cache itself
struct llama_moe_cache_test_batch {
    size_t used   = 0;
    size_t hits   = 0;
    size_t fills  = 0;
    size_t bypass = 0;
    std::vector<int32_t> slots; // [n_expert] after this batch
};

bool llama_moe_cache_test_admit_partial(int64_t n_tokens, int64_t n_expert, size_t n_layers,
        int64_t n_slots, bool enabled);

std::vector<llama_moe_cache_test_batch> llama_moe_cache_test_plan_partial(
        int32_t n_expert, int32_t n_slots, int32_t quota,
        const std::vector<std::vector<int32_t>> & batches);
