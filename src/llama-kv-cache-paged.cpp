#include "llama-kv-cache-paged.h"

#include "llama-impl.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <numeric>

// SNAPKVDBG metrics are parsed by the benchmark harness; only emit when asked.
static bool snapkv_debug_enabled() {
    static const bool enabled = getenv("LLAMA_SNAPKV_DEBUG") != nullptr;
    return enabled;
}

//
// llama_kv_cache_paged
//

llama_kv_cache_paged::llama_kv_cache_paged(uint32_t head_dim,
                                           uint32_t n_head_q,
                                           uint32_t n_heads_kv,
                                           uint32_t block_size,
                                           uint32_t n_layers,
                                           uint32_t n_ubatch,
                                           uint32_t n_seq_max,
                                           const layer_filter_cb & filter_attn) :
    kv_type(GGML_TYPE_F16),
    head_dim(head_dim),
    n_heads_kv(n_heads_kv),
    block_size(block_size),
    n_layers(n_layers),
    n_ubatch(n_ubatch),
    n_seq_max(n_seq_max),
    filter_attn(filter_attn),
    num_gpu_blocks(0),
    num_cpu_blocks(0),
    initial_num_gpu_blocks(0),
    growth_num_gpu_blocks(0),
    gpu_watermark_num_blocks(0),
    block_bytes(0),
    allow_dynamic_spill(false),
    cpu_backend(nullptr),
    snapkv_score_heads(n_head_q > 0 ? n_head_q : n_heads_kv) {}

void llama_kv_cache_paged::init(const std::vector<ggml_backend_t> & layer_backends,
                                const std::vector<ggml_backend_t> & candidate_backends,
                                ggml_backend_t backend_cpu,
                                enum ggml_type type,
                                uint32_t       n_gpu_blocks,
                                uint32_t       n_cpu_blocks,
                                float          watermark,
                                uint32_t       initial_gpu_blocks,
                                uint32_t       growth_gpu_blocks,
                                bool           dynamic_spill) {
    GGML_ASSERT(backend_cpu && "backend_cpu is nullptr");

    GGML_ASSERT(n_gpu_blocks && "n_gpu_blocks need to be greater than 0.");
    GGML_ASSERT(n_cpu_blocks && "n_cpu_blocks need to be greater than 0.");

    LLAMA_LOG_INFO(
        "%s: initializing paged KV cache. n_gpu_blocks=%d, n_cpu_blocks=%d, block_size=%d, watermark=%0.2f, initial_gpu_blocks=%d, growth_gpu_blocks=%d, dynamic_spill=%d\n",
        __func__, n_gpu_blocks, n_cpu_blocks, block_size, watermark, initial_gpu_blocks, growth_gpu_blocks, dynamic_spill);
    num_gpu_blocks = n_gpu_blocks;
    num_cpu_blocks = n_cpu_blocks;
    allow_dynamic_spill = dynamic_spill;
    initial_num_gpu_blocks = dynamic_spill ? std::max((uint32_t) 1, std::min(initial_gpu_blocks, n_gpu_blocks))
                                           : n_gpu_blocks;
    growth_num_gpu_blocks = dynamic_spill ? std::max((uint32_t) 1, std::min(
        growth_gpu_blocks == 0 ? initial_num_gpu_blocks : growth_gpu_blocks, n_gpu_blocks))
                                           : n_gpu_blocks;
    gpu_watermark_num_blocks = std::ceil(num_gpu_blocks * watermark);
    kv_type        = type;
    cpu_backend    = backend_cpu;
    GGML_ASSERT(kv_type == GGML_TYPE_F16 || kv_type == GGML_TYPE_Q4_0 || kv_type == GGML_TYPE_Q8_0);
    GGML_ASSERT(head_dim % ggml_blck_size(kv_type) == 0 && "paged KV head_dim must align to the quantization block size");
    const size_t row_bytes = ggml_row_size(kv_type, head_dim);
    block_bytes = (size_t) 2 * block_size * n_heads_kv * row_bytes;

    LLAMA_LOG_INFO(
        "%s: type=%s, bytes_per_row=%zu, bytes_per_block=%zu\n",
        __func__, ggml_type_name(kv_type), row_bytes, block_bytes);

    kv_gpu_layers.clear();
    kv_gpu_layers.resize(n_layers);
    layer_cpu_indices.assign(n_layers, -1);
    attn_layer_ids.clear();
    kv_cpu_layers.clear();
    spill_backends.clear();
    free_gpu_ids.resize(num_gpu_blocks);
    std::iota(free_gpu_ids.begin(), free_gpu_ids.end(), 0);
    gpu_block_ref_counts.assign(num_gpu_blocks, 0);
    for (ggml_backend_t backend : candidate_backends) {
        if (backend != nullptr && std::find(spill_backends.begin(), spill_backends.end(), backend) == spill_backends.end()) {
            spill_backends.push_back(backend);
        }
    }

    for (uint32_t il = 0; il < n_layers; ++il) {
        if (filter_attn && !filter_attn(il)) {
            continue;
        }

        ggml_backend_t layer_backend = il < layer_backends.size() ? layer_backends[il] : backend_cpu;
        GGML_ASSERT(layer_backend && "paged KV layer backend is nullptr");
        ggml_backend_t candidate_backend = layer_backend;
        for (ggml_backend_t backend : spill_backends) {
            if (backend != layer_backend) {
                candidate_backend = backend;
                break;
            }
        }

        auto & storage = kv_gpu_layers[il];
        storage.original_backend  = layer_backend;
        storage.candidate_backend = candidate_backend;
        storage.backend           = layer_backend;

        GGML_ASSERT(reset_layer_storage(storage, layer_backend, initial_num_gpu_blocks, 0) &&
                    "Failed to allocate paged KV layer buffer");
        LLAMA_LOG_INFO(
            "%s: layer %u backend=%s candidate=%s blocks=%u buffer=%8.2f MiB\n",
            __func__, il,
            ggml_backend_dev_name(ggml_backend_get_device(storage.backend)),
            ggml_backend_dev_name(ggml_backend_get_device(storage.candidate_backend)),
            storage.capacity,
            ggml_backend_buffer_get_size(storage.buf.get()) / 1024.0 / 1024.0);

        layer_cpu_indices[il] = kv_cpu_layers.size();
        attn_layer_ids.push_back(il);
    }

    // Set up CPU context and tensor (for swapping)
    struct ggml_init_params cpu_params;
    cpu_params.mem_size           = ggml_tensor_overhead() * 5 * attn_layer_ids.size();
    cpu_params.mem_buffer         = NULL;
    cpu_params.no_alloc           = true;
    cpu_ctx.reset(ggml_init(cpu_params));
    GGML_ASSERT(cpu_ctx && "Failed to initialize CPU KV cache context");
    for (size_t i = 0; i < attn_layer_ids.size(); ++i) {
        kv_cpu_layers.push_back(create_layer_tensor(cpu_ctx.get(), type, n_cpu_blocks));
    }

    // Allocate on the CPU backend (using pinned memory for faster PCIe transfer)
    cpu_buf.reset(ggml_backend_alloc_ctx_tensors(cpu_ctx.get(), backend_cpu));
    GGML_ASSERT(cpu_buf && "Failed to allocate CPU KV cache buffer");
    ggml_backend_buffer_clear(cpu_buf.get(), 0);
    for (uint32_t i = 0; i < kv_cpu_layers.size(); ++i) {
        GGML_ASSERT(kv_cpu_layers[i]->buffer && "CPU layer tensor has null buffer");
    }

    // Setting up our block accountant
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);
}

bool llama_kv_cache_paged::has_free_gpu_blocks(uint32_t num_requested_blocks, bool ignore_watermark) const {
    const size_t curr_free_gpus = free_gpu_ids.size();
    if (!ignore_watermark && curr_free_gpus < gpu_watermark_num_blocks) {
        return false;
    }
    return ignore_watermark ? num_requested_blocks <= curr_free_gpus
                            : num_requested_blocks <= (curr_free_gpus - gpu_watermark_num_blocks);
}

llama_block_ids llama_kv_cache_paged::preview_gpu_blocks(uint32_t num_blocks) const {
    if (num_blocks > free_gpu_ids.size()) {
        return {};
    }
    return llama_block_ids(free_gpu_ids.begin(), free_gpu_ids.begin() + num_blocks);
}

llama_block_ids llama_kv_cache_paged::checkout_gpu_blocks(uint32_t num_blocks) {
    llama_block_ids new_ids = preview_gpu_blocks(num_blocks);
    if (new_ids.size() != num_blocks) {
        return {};
    }

    free_gpu_ids.erase(free_gpu_ids.begin(), free_gpu_ids.begin() + num_blocks);
    for (const uint32_t id : new_ids) {
        gpu_block_ref_counts[id] += 1;
    }
    return new_ids;
}

void llama_kv_cache_paged::release_gpu_blocks(const llama_block_ids & freed_blocks) {
    for (const uint32_t id : freed_blocks) {
        GGML_ASSERT(id < gpu_block_ref_counts.size());
        GGML_ASSERT(gpu_block_ref_counts[id] > 0);
        gpu_block_ref_counts[id] -= 1;
        if (gpu_block_ref_counts[id] == 0) {
            free_gpu_ids.insert(std::lower_bound(free_gpu_ids.begin(), free_gpu_ids.end(), id), id);
        }
    }
}

uint32_t llama_kv_cache_paged::required_gpu_capacity() const {
    for (uint32_t i = num_gpu_blocks; i > 0; --i) {
        if (gpu_block_ref_counts[i - 1] > 0) {
            return i;
        }
    }
    return 0;
}

struct ggml_tensor * llama_kv_cache_paged::get_cpu_tensor(int layer_idx) const {
    GGML_ASSERT(layer_idx >= 0 && layer_idx < (int32_t) layer_cpu_indices.size());
    GGML_ASSERT(layer_cpu_indices[layer_idx] >= 0 && "layer not in CPU paged KV cache");
    return kv_cpu_layers[layer_cpu_indices[layer_idx]];
}

struct ggml_tensor * llama_kv_cache_paged::create_layer_tensor(struct ggml_context * ctx, enum ggml_type type, uint32_t num_blocks) const {
    return ggml_new_tensor_4d(ctx, type, head_dim, block_size, 2 * n_heads_kv, num_blocks);
}

bool llama_kv_cache_paged::reset_layer_storage(layer_storage & storage,
                                               ggml_backend_t  target_backend,
                                               uint32_t        num_blocks,
                                               size_t          bytes_to_copy) {
    GGML_ASSERT(target_backend && "paged KV target backend is nullptr");
    ggml_backend_t old_backend = storage.backend;
    const uint32_t old_capacity = storage.capacity;

    struct ggml_init_params params = {};
    params.mem_size   = ggml_tensor_overhead() * 5;
    params.mem_buffer = NULL;
    params.no_alloc   = true;

    ggml_context_ptr new_ctx { ggml_init(params) };
    if (!new_ctx) {
        return false;
    }

    struct ggml_tensor * new_tensor = create_layer_tensor(new_ctx.get(), kv_type, num_blocks);
    if (!new_tensor) {
        return false;
    }

    ggml_backend_buffer_ptr new_buf;
    std::vector<uint8_t> staging;
    size_t copy_bytes = 0;

    if (storage.tensor != nullptr && target_backend == storage.backend) {
        copy_bytes = std::min(bytes_to_copy, ggml_nbytes(storage.tensor));
        staging.resize(copy_bytes);
        if (copy_bytes > 0) {
            ggml_backend_synchronize(storage.backend);
            ggml_backend_tensor_get(storage.tensor, staging.data(), 0, copy_bytes);
        }

        // Avoid a transient double allocation: under WDDM it can evict otherwise fitting
        // CUDA buffers to system memory and leave decode permanently bandwidth-bound.
        storage.tensor = nullptr;
        storage.buf.reset();
        storage.ctx.reset();
        new_buf.reset(ggml_backend_alloc_ctx_tensors(new_ctx.get(), target_backend));

        if (!new_buf) {
            new_ctx.reset(ggml_init(params));
            new_tensor = new_ctx ? create_layer_tensor(new_ctx.get(), kv_type, old_capacity) : nullptr;
            new_buf.reset(new_tensor ? ggml_backend_alloc_ctx_tensors(new_ctx.get(), old_backend) : nullptr);
            GGML_ASSERT(new_buf && "failed to restore paged KV after in-place growth failure");
            ggml_backend_buffer_clear(new_buf.get(), 0);
            if (copy_bytes > 0) {
                ggml_backend_tensor_set(new_tensor, staging.data(), 0, copy_bytes);
            }
            storage.ctx      = std::move(new_ctx);
            storage.buf      = std::move(new_buf);
            storage.backend  = old_backend;
            storage.tensor   = new_tensor;
            storage.capacity = old_capacity;
            ++storage_generation;
            return false;
        }
    } else {
        new_buf.reset(ggml_backend_alloc_ctx_tensors(new_ctx.get(), target_backend));
    }
    if (!new_buf) {
        return false;
    }

    ggml_backend_buffer_clear(new_buf.get(), 0);

    if (storage.tensor != nullptr && bytes_to_copy > 0) {
        copy_bytes = std::min({ bytes_to_copy, ggml_nbytes(storage.tensor), ggml_nbytes(new_tensor) });
        if (copy_bytes > 0) {
            ggml_backend_synchronize(storage.backend);
            ggml_backend_synchronize(target_backend);
            staging.resize(copy_bytes);
            ggml_backend_tensor_get(storage.tensor, staging.data(), 0, copy_bytes);
        }
    }
    if (copy_bytes > 0) {
        ggml_backend_tensor_set(new_tensor, staging.data(), 0, copy_bytes);
    }

    storage.ctx      = std::move(new_ctx);
    storage.buf      = std::move(new_buf);
    storage.backend  = target_backend;
    storage.tensor   = new_tensor;
    storage.capacity = num_blocks;
    ++storage_generation;
    if (old_backend != nullptr) {
        LLAMA_LOG_INFO("%s: paged KV moved %s:%u -> %s:%u blocks\n", __func__,
            ggml_backend_dev_name(ggml_backend_get_device(old_backend)), old_capacity,
            ggml_backend_dev_name(ggml_backend_get_device(target_backend)), num_blocks);
    }
    return true;
}

bool llama_kv_cache_paged::ensure_physical_capacity(uint32_t required_blocks, bool rebalance) {
    if (required_blocks == 0) {
        return true;
    }

    const uint32_t growth_step = growth_num_gpu_blocks;
    const uint32_t target_blocks = required_blocks <= initial_num_gpu_blocks ? initial_num_gpu_blocks :
        (uint32_t) std::min((uint64_t) num_gpu_blocks, (uint64_t) initial_num_gpu_blocks +
            ((uint64_t) (required_blocks - initial_num_gpu_blocks) + growth_step - 1) / growth_step * growth_step);
    if (target_blocks <= initial_num_gpu_blocks && !rebalance) {
        return true;
    }

    const uint32_t original_budget_blocks = std::min(num_gpu_blocks,
        initial_num_gpu_blocks * 2);
    const size_t max_original_layers = std::min(attn_layer_ids.size(),
        (size_t) ((uint64_t) original_budget_blocks * attn_layer_ids.size() / target_blocks));

    for (size_t layer_index = 0; layer_index < attn_layer_ids.size(); ++layer_index) {
        const uint32_t il = attn_layer_ids[layer_index];
        auto & storage = kv_gpu_layers[il];
        const bool can_spill = allow_dynamic_spill && storage.candidate_backend != nullptr &&
                               storage.candidate_backend != storage.backend;
        const bool candidate_is_cpu = can_spill &&
            ggml_backend_dev_type(ggml_backend_get_device(storage.candidate_backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
        // CPU spill is a last resort: first try to grow the current accelerator
        // storage and move only as many layers as are needed to make room.  The
        // old pre-spill heuristic is useful for balancing multiple accelerator
        // backends, but applying it to CPU would move every layer at the first
        // growth step and needlessly turn the whole attention pass into a CPU
        // reference operation.
        const bool spill_first = (rebalance || target_blocks > original_budget_blocks) && can_spill &&
                                 !candidate_is_cpu &&
                                 storage.backend == storage.original_backend &&
                                 layer_index >= max_original_layers;
        if (storage.capacity >= target_blocks && !spill_first) {
            continue;
        }

        std::vector<ggml_backend_t> candidates;
        if (spill_first) {
            LLAMA_LOG_INFO("%s: spilling layer %u before growth (%zu/%zu layers remain on original backend)\n",
                __func__, il, max_original_layers, attn_layer_ids.size());
        }
        if (spill_first) {
            candidates.push_back(storage.candidate_backend);
        }
        candidates.push_back(storage.backend);
        if (can_spill && !spill_first) {
            candidates.push_back(storage.candidate_backend);
        }
        for (ggml_backend_t backend : spill_backends) {
            if (backend != nullptr && std::find(candidates.begin(), candidates.end(), backend) == candidates.end()) {
                candidates.push_back(backend);
            }
        }
        if (std::find(candidates.begin(), candidates.end(), storage.original_backend) == candidates.end()) {
            candidates.push_back(storage.original_backend);
        }

        const uint32_t new_capacity = std::max(storage.capacity, target_blocks);
        const size_t bytes_to_copy = (size_t) std::min(required_gpu_capacity(), storage.capacity) * block_bytes;
        bool grown = false;
        for (ggml_backend_t candidate : candidates) {
            if (reset_layer_storage(storage, candidate, new_capacity, bytes_to_copy)) {
                grown = true;
                break;
            }
        }
        if (!grown) {
            LLAMA_LOG_WARN("%s: unable to grow layer %u to %u blocks\n", __func__, il, new_capacity);
            return false;
        }
    }

    return true;
}

void llama_kv_cache_paged::maybe_restore_initial_storage() {
    if (!allow_dynamic_spill) {
        return;
    }

    const uint32_t required_blocks = required_gpu_capacity();
    if (required_blocks > initial_num_gpu_blocks) {
        return;
    }

    for (const uint32_t il : attn_layer_ids) {
        auto & storage = kv_gpu_layers[il];
        if (storage.backend == storage.original_backend && storage.capacity == initial_num_gpu_blocks) {
            continue;
        }

        const size_t bytes_to_copy = std::min(required_blocks, storage.capacity) * block_bytes;
        if (!reset_layer_storage(storage, storage.original_backend, initial_num_gpu_blocks, bytes_to_copy)) {
            LLAMA_LOG_WARN("%s: unable to restore layer %u to its original backend\n", __func__, il);
        }
    }
}

bool llama_kv_cache_paged::allocate(int32_t num_tokens, llama_sequence_group & group) {
    uint32_t curr_block_count     = group.block_table.size();
    uint32_t total_num_tokens     = group.n_prompt + group.n_decoded + num_tokens;
    uint32_t num_requested_blocks = std::ceil((float) total_num_tokens / block_size) - curr_block_count;
    LLAMA_LOG_DEBUG("%s: curr_block_count=%d, total_num_tokens=%d, num_requested_blocks=%d\n", __func__,
                    curr_block_count, total_num_tokens, num_requested_blocks);

    if (num_requested_blocks == 0) {
        if (!allow_dynamic_spill || num_tokens > 4) {
            return true;
        }
        maybe_restore_initial_storage();
        return ensure_physical_capacity(required_gpu_capacity(), /*rebalance=*/true);
    }

    const bool ignore_watermark = !group.block_table.empty() || snapkv_is_strict(group);
    if (!has_free_gpu_blocks(num_requested_blocks, ignore_watermark)) {
        LLAMA_LOG_DEBUG("%s: insufficient GPU blocks. Requested: %d.\n", __func__, num_requested_blocks);
        return false;
    }

    llama_block_ids preview_ids = preview_gpu_blocks(num_requested_blocks);
    GGML_ASSERT(preview_ids.size() == num_requested_blocks);

    uint32_t required_blocks = required_gpu_capacity();
    if (!preview_ids.empty()) {
        required_blocks = std::max(required_blocks, (uint32_t) *std::max_element(preview_ids.begin(), preview_ids.end()) + 1);
    }
    if (!ensure_physical_capacity(required_blocks, allow_dynamic_spill && num_tokens <= 4)) {
        return false;
    }

    llama_block_ids new_ids = checkout_gpu_blocks(num_requested_blocks);
    GGML_ASSERT(new_ids.size() == num_requested_blocks);
    concat_block_ids(group.block_table, new_ids);
    LLAMA_LOG_DEBUG("%s: successfully allocated %d.\n", __func__, num_requested_blocks);
    return true;
}

bool llama_kv_cache_paged::prepare_batch(const llama_batch & batch) {
    if (batch.n_tokens <= 0) {
        return false;
    }

    struct sequence_batch {
        llama_pos previous_max = -1;
        llama_pos last_pos     = -1;
        int32_t   n_tokens     = 0;
        size_t    old_size     = 0;
        bool      initialized  = false;
    };

    std::unordered_map<llama_seq_id, sequence_batch> sequences;
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] != 1) {
            LLAMA_LOG_ERROR("%s: server-side paged KV does not support coupled sequences\n", __func__);
            return false;
        }

        const llama_seq_id seq_id = batch.seq_id[i][0];
        auto [it, inserted] = sequences.try_emplace(seq_id);
        auto & sequence = it->second;
        if (inserted) {
            sequence.previous_max = seq_pos_max(seq_id);
            sequence.last_pos = sequence.previous_max;
        }

        if (batch.pos[i] != sequence.last_pos + 1) {
            LLAMA_LOG_ERROR("%s: non-contiguous sequence positions for seq %d: previous_max=%d current=%d\n",
                            __func__, seq_id, sequence.last_pos, batch.pos[i]);
            return false;
        }
        sequence.last_pos = batch.pos[i];
        ++sequence.n_tokens;
    }

    for (auto & entry : sequences) {
        const llama_seq_id seq_id = entry.first;
        auto & sequence = entry.second;
        auto & group = regular_groups[seq_id];
        group.request_id = seq_id;
        const uint32_t prev_decoded = group.n_decoded;
        const uint32_t prev_n_prompt = group.n_prompt;
        // keep the allocator invariant n_prompt + n_decoded == previous_max + 1:
        // while prefilling n_prompt holds the cached tokens and n_decoded is 0;
        // once decoding starts the roles flip.
        group.n_prompt = sequence.previous_max >= 0 ? sequence.previous_max + 1 : 0;
        group.n_decoded = 0;
        if (sequence.n_tokens == 1 && sequence.previous_max >= 0) {
            group.n_decoded = sequence.previous_max + 1;
            group.n_prompt = 0;
        }
        sequence.old_size = group.block_table.size();
        sequence.initialized = true;

        if (sequence.previous_max < 0 && snapkv_enabled) {
            snapkv_start_prefill(group, sequence.last_pos + 1);
        }

        // First decode after a prefill: run the final SnapKV eviction before
        // allocating decode tokens, including a multi-token MTP batch.
        const auto snapkv_state = snapkv_sequences.find(seq_id);
        const bool snapkv_pending = snapkv_state != snapkv_sequences.end() && snapkv_state->second.pending_final_evict;
        const bool prompt_complete = snapkv_pending &&
            snapkv_state->second.expected_prefill_end >= 0 &&
            sequence.previous_max + 1 >= snapkv_state->second.expected_prefill_end;
        const bool first_decode = snapkv_enabled && sequence.previous_max >= 0 &&
                                  snapkv_pending && prev_decoded == 0 && prev_n_prompt > 0 &&
                                  (prompt_complete ||
                                   ((snapkv_state == snapkv_sequences.end() ||
                                     snapkv_state->second.expected_prefill_end < 0) && sequence.n_tokens == 1));
        if (first_decode) {
            end_prefill(group);
        }

        if (allocate(sequence.n_tokens, group)) {
            continue;
        }

        if (!snapkv_enabled) {
            for (auto & rollback_entry : sequences) {
                const llama_seq_id rollback_seq_id = rollback_entry.first;
                const size_t old_size = rollback_entry.second.old_size;
                if (!rollback_entry.second.initialized) {
                    continue;
                }
                auto group_it = regular_groups.find(rollback_seq_id);
                if (group_it == regular_groups.end() || group_it->second.block_table.size() <= old_size) {
                    continue;
                }
                auto & rollback_group = group_it->second;
                llama_block_ids rollback(rollback_group.block_table.begin() + old_size, rollback_group.block_table.end());
                release_gpu_blocks(rollback);
                rollback_group.block_table.resize(old_size);
                if (old_size == 0) {
                    regular_groups.erase(group_it);
                }
            }
            return false;
        }

        // SnapKV streaming prefill: allocate physical pages up to last_pos,
        // evicting old low-importance pages first when the pool is exhausted.
        if (!ensure_pos_blocks((uint32_t) sequence.last_pos, group)) {
            LLAMA_LOG_ERROR("%s: snapkv streaming prefill could not cover pos %d for seq %d\n",
                            __func__, (int) sequence.last_pos, seq_id);
            return false;
        }
    }

    return true;
}

bool llama_kv_cache_paged::ensure_batch_blocks(const llama_ubatch & ubatch) {
    if (!snapkv_enabled || ubatch.n_tokens == 0 || ubatch.n_seqs_unq != 1) {
        return true;
    }
    llama_pos max_pos = ubatch.pos[0];
    for (uint32_t i = 1; i < ubatch.n_tokens; ++i) {
        max_pos = std::max(max_pos, ubatch.pos[i]);
    }
    auto it = regular_groups.find(ubatch.seq_id_unq[0]);
    if (it == regular_groups.end()) {
        return true;
    }
    return ensure_pos_blocks((uint32_t) max_pos, it->second);
}

void llama_kv_cache_paged::snapkv_schedule_capture(llama_sequence_group & group, llama_pos capture_end) {
    if (snapkv_scores_tensor == nullptr || snapkv_token_scores_tensor == nullptr) {
        return;
    }
    auto & state = snapkv_sequences[group.request_id];
    if (snapkv_score_slots.find(group.request_id) == snapkv_score_slots.end()) {
        std::vector<bool> used(n_seq_max, false);
        for (const auto & entry : snapkv_score_slots) {
            if (entry.second < used.size()) {
                used[entry.second] = true;
            }
        }
        const auto free_slot = std::find(used.begin(), used.end(), false);
        GGML_ASSERT(free_slot != used.end());
        snapkv_score_slots[group.request_id] = (uint32_t) std::distance(used.begin(), free_slot);
    }
    const uint32_t slot = snapkv_score_slots[group.request_id];
    const size_t page_stride  = (size_t) max_logical_blocks * snapkv_score_heads;
    const size_t token_stride = page_stride * block_size;
    std::fill_n(snapkv_scores_host.begin() + (size_t) slot * page_stride, page_stride, 0.0f);
    std::fill_n(snapkv_token_scores_host.begin() + (size_t) slot * token_stride, token_stride, 0.0f);
    ggml_backend_tensor_set(snapkv_scores_tensor, snapkv_scores_host.data() + (size_t) slot * page_stride,
                            (size_t) slot * page_stride * sizeof(float), page_stride * sizeof(float));
    ggml_backend_tensor_set(snapkv_token_scores_tensor, snapkv_token_scores_host.data() + (size_t) slot * token_stride,
                            (size_t) slot * token_stride * sizeof(float), token_stride * sizeof(float));
    snapkv_scores_dirty = false;
    state.capture_until = std::max<decltype(capture_end)>(0, capture_end);
    state.capture_from = state.mode == snapkv_prefill_mode::streaming
        ? 0
        : std::max<decltype(capture_end)>(0, state.capture_until - (llama_pos) snapkv_observation_window);
}

void llama_kv_cache_paged::snapkv_start_prefill(llama_sequence_group & group, llama_pos prefill_end) {
    if (snapkv_scores_tensor == nullptr || snapkv_token_scores_tensor == nullptr) {
        return;
    }
    auto & state = snapkv_sequences[group.request_id];
    if (state.expected_prefill_end < 0) {
        state.expected_prefill_end = prefill_end;
    }
    const uint32_t physical_blocks = num_gpu_blocks;
    const uint32_t expected_blocks = (uint32_t) ((state.expected_prefill_end + block_size - 1) / block_size);
    state.mode = expected_blocks <= physical_blocks ? snapkv_prefill_mode::strict : snapkv_prefill_mode::streaming;
    state.next_capture_pos = 0;
    snapkv_schedule_capture(group, state.expected_prefill_end);
    state.pending_final_evict = true;
    LLAMA_LOG_INFO("%s: seq %d mode=%s expected_prefill_end=%lld physical_pool_blocks=%u\n", __func__,
                   group.request_id,
                   state.mode == snapkv_prefill_mode::strict ? "strict" : "streaming",
                   (long long) state.expected_prefill_end, physical_blocks);
    if (snapkv_debug_enabled()) {
        fprintf(stderr, "SNAPKVDBG mode=%s expected_prefill_end=%lld physical_pool_blocks=%u\n",
                state.mode == snapkv_prefill_mode::strict ? "strict" : "streaming",
                (long long) state.expected_prefill_end, physical_blocks);
    }
}

bool llama_kv_cache_paged::snapkv_is_strict(const llama_sequence_group & group) const {
    const auto it = snapkv_sequences.find(group.request_id);
    return snapkv_enabled && it != snapkv_sequences.end() &&
           it->second.pending_final_evict && it->second.mode == snapkv_prefill_mode::strict;
}

void llama_kv_cache_paged::set_snapkv_prefill_end(llama_seq_id seq_id, llama_pos prefill_end) {
    if (!snapkv_enabled || prefill_end < 0) {
        return;
    }
    auto & state = snapkv_sequences[seq_id];
    state.expected_prefill_end = prefill_end;
    if (snapkv_debug_enabled()) {
        fprintf(stderr, "SNAPKVDBG prefill_hint seq=%d expected_prefill_end=%lld\n",
                seq_id, (long long) prefill_end);
    }
}

void llama_kv_cache_paged::snapkv_update_capture(const llama_ubatch & ubatch) {
    snapkv_capture_active = false;
    snapkv_streaming_active = false;
    if (snapkv_scores_tensor == nullptr || ubatch.n_tokens == 0) {
        return;
    }
    std::fill(snapkv_capture_from_host.begin(), snapkv_capture_from_host.end(), -1);
    std::fill(snapkv_score_slots_host.begin(), snapkv_score_slots_host.end(), -1);
    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        const llama_seq_id seq_id = ubatch.seq_id_unq[s];
        const auto group_it = regular_groups.find(seq_id);
        if (group_it != regular_groups.end() &&
            std::any_of(group_it->second.block_table.begin(), group_it->second.block_table.end(),
                        [](int32_t id) { return id < 0; })) {
            snapkv_streaming_active = true;
        }
        auto it = snapkv_sequences.find(seq_id);
        if (it == snapkv_sequences.end() || !it->second.pending_final_evict) {
            continue;
        }
        const bool streaming = it->second.mode == snapkv_prefill_mode::streaming;
        if (it->second.capture_from < 0 || it->second.capture_until <= it->second.capture_from) {
            continue;
        }
        llama_pos first = std::numeric_limits<llama_pos>::max();
        llama_pos last = -1;
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            if (ubatch.n_seq_id[i] == 1 && ubatch.seq_id[i][0] == seq_id) {
                first = std::min(first, ubatch.pos[i]);
                last = std::max(last, ubatch.pos[i]);
            }
        }
        llama_pos capture_from = it->second.capture_from;
        if (streaming) {
            // Streaming eviction only needs a sparse, incremental score sketch.
            // Score only the final query in each sampled ubatch instead of
            // re-scoring all queries against the complete prefix.
            const llama_pos sample_stride = std::max<llama_pos>(
                (llama_pos) snapkv_observation_window, (llama_pos) block_size * 64);
            if (last < it->second.next_capture_pos) {
                continue;
            }
            capture_from = last;
            it->second.next_capture_pos = last + sample_stride;
        }
        if (first < it->second.capture_until && last >= capture_from) {
            snapkv_capture_from_host[s] = capture_from;
            snapkv_score_slots_host[s] = snapkv_score_slots[seq_id];
            snapkv_capture_active = true;
            snapkv_scores_dirty = true;
        }
    }
    if (!snapkv_capture_active) {
        return;
    }
    ggml_backend_tensor_set(snapkv_capture_from_tensor, snapkv_capture_from_host.data(), 0,
                            snapkv_capture_from_host.size() * sizeof(int32_t));
    ggml_backend_tensor_set(snapkv_score_slots_tensor, snapkv_score_slots_host.data(), 0,
                            snapkv_score_slots_host.size() * sizeof(int32_t));
}

bool llama_kv_cache_paged::build_batch_info(const llama_ubatch & ubatch, llama_paged_batch_info & info) const {
    if (ubatch.n_tokens == 0 || ubatch.n_seqs_unq == 0 || ubatch.n_pos < 1) {
        return false;
    }

    regular_write_slots.resize(ubatch.n_tokens);
    regular_context_lens.assign(ubatch.n_seqs_unq, 0);
    regular_batch_offsets.assign(ubatch.n_seqs_unq, -1);
    regular_batch_lens.assign(ubatch.n_seqs_unq, 0);

    std::unordered_map<llama_seq_id, uint32_t> sequence_indices;
    std::vector<const llama_block_ids *> block_tables(ubatch.n_seqs_unq, nullptr);
    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        const llama_seq_id seq_id = ubatch.seq_id_unq[s];
        const auto it = regular_groups.find(seq_id);
        if (it == regular_groups.end() || it->second.block_table.empty()) {
            return false;
        }
        sequence_indices.emplace(seq_id, s);
        block_tables[s] = &it->second.block_table;
    }
    const uint32_t table_width = max_logical_blocks > 0 ? max_logical_blocks : num_gpu_blocks;
    regular_block_table.assign((size_t) ubatch.n_seqs_unq * table_width, -1);
    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        GGML_ASSERT(block_tables[s]->size() <= table_width && "group block table exceeds logical width");
        std::copy(block_tables[s]->begin(), block_tables[s]->end(),
                  regular_block_table.begin() + s * table_width);
    }

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id[i] != 1) {
            return false;
        }
        const auto sequence_it = sequence_indices.find(ubatch.seq_id[i][0]);
        if (sequence_it == sequence_indices.end()) {
            return false;
        }
        const uint32_t s = sequence_it->second;
        if (regular_batch_offsets[s] < 0) {
            regular_batch_offsets[s] = i;
        } else if ((uint32_t) regular_batch_offsets[s] + regular_batch_lens[s] != i) {
            LLAMA_LOG_ERROR("%s: sequence tokens must be contiguous in a paged ubatch\n", __func__);
            return false;
        }

        // Qwen uses 4-position M-RoPE; the first position plane is the
        // sequential token position used by the paged KV block table.
        const llama_pos pos = ubatch.pos[i];
        const uint32_t block = pos / block_size;
        const auto & block_table = *block_tables[s];
        if (block >= block_table.size() || block_table[block] < 0) {
            LLAMA_LOG_ERROR("%s: cannot write pos %lld: block %u out of range or evicted (table size %zu)\n",
                            __func__, (long long) pos, block, block_table.size());
            return false;
        }
        regular_write_slots[i] = block_table[block] * block_size + (pos % block_size);
        regular_context_lens[s] = std::max(regular_context_lens[s], (int32_t) pos + 1);
        ++regular_batch_lens[s];
    }

    info.n_blocks_per_seq = max_logical_blocks > 0 ? max_logical_blocks : num_gpu_blocks;
    info.n_seq            = ubatch.n_seqs_unq;
    info.n_tokens         = ubatch.n_tokens;
    info.write_slots      = regular_write_slots.data();
    info.block_table      = regular_block_table.data();
    info.context_lens     = regular_context_lens.data();
    info.batch_offsets     = regular_batch_offsets.data();
    info.batch_lens        = regular_batch_lens.data();
    return true;
}

void llama_kv_cache_paged::commit_batch(const llama_ubatch & ubatch) {
    if (ubatch.n_tokens == 0) {
        return;
    }

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id[i] != 1) {
            return;
        }
        auto & range = sequence_positions[ubatch.seq_id[i][0]];
        range.min = range.min < 0 ? ubatch.pos[i] : std::min(range.min, ubatch.pos[i]);
        range.max = std::max(range.max, ubatch.pos[i]);
    }
}

void llama_kv_cache_paged::free_blocks(llama_sequence_group & group) {
    if (group.block_table.empty()) {
        return;
    }

    llama_block_ids blocks_to_free_gpu;
    std::vector<uint32_t> blocks_to_free_cpu;

    for (int32_t block_id : group.block_table) {
        if (block_id < 0) {
            continue;
        }
        if (block_manager.is_gpu((uint32_t) block_id)) {
            blocks_to_free_gpu.push_back(block_id);
        } else {
            blocks_to_free_cpu.push_back(block_id);
        }
    }

    if (!blocks_to_free_gpu.empty()) {
        release_gpu_blocks(blocks_to_free_gpu);
    }
    if (!blocks_to_free_cpu.empty()) {
        block_manager.release_cpu_blocks(blocks_to_free_cpu);
    }

    group.block_table.clear();
    sequence_positions.erase(group.request_id);
    if (required_gpu_capacity() <= initial_num_gpu_blocks) {
        maybe_restore_initial_storage();
    }
}

void llama_kv_cache_paged::do_block_copy(const llama_block_ids & src_ids,
                                         const llama_block_ids & new_ids,
                                         bool                    to_gpu) {
    const uint32_t num_blocks = src_ids.size();
    LLAMA_LOG_DEBUG("%s: num_blocks_size=%d, new_ids_size=%ld\n", __func__, num_blocks, new_ids.size());
    GGML_ASSERT(num_blocks == new_ids.size() && "src_ids and new_ids do not have the same size.");
    GGML_ASSERT(std::all_of(src_ids.begin(), src_ids.end(), [](int32_t id) { return id >= 0; }) &&
                "do_block_copy requires valid block ids");

    GGML_ASSERT(kv_cpu_layers.size() == attn_layer_ids.size() && "CPU and attention layer counts differ.");

    // Buffer on HOST to faciliate block data transfer
    // Note: an optimization would be to use views and async copies. Beware of
    // memory overhead heurisitcs.
    std::vector<uint8_t> staging(block_bytes);

    for (uint32_t il = 0; il < attn_layer_ids.size(); ++il) {
        const int32_t layer_id = attn_layer_ids[il];
        struct ggml_tensor * gpu_main = kv_gpu_layers[layer_id].tensor;
        struct ggml_tensor * cpu_main = get_cpu_tensor(layer_id);
        struct ggml_tensor * src_main = to_gpu ? cpu_main : gpu_main;
        struct ggml_tensor * dst_main = to_gpu ? gpu_main : cpu_main;

        ggml_backend_t src_backend = to_gpu ? cpu_backend : kv_gpu_layers[layer_id].backend;
        ggml_backend_t dst_backend = to_gpu ? kv_gpu_layers[layer_id].backend : cpu_backend;
        ggml_backend_synchronize(src_backend);
        ggml_backend_synchronize(dst_backend);

        for (uint32_t i = 0; i < num_blocks; ++i) {
            const uint32_t src_global = src_ids[i];
            const uint32_t dst_global = new_ids[i];

            // GPU and CPu blocks may differ (usually CPU < GPU)
            // We substract the diffence to calculate where the local starts before we calculate offsets
            const uint32_t src_local = to_gpu ? src_global - num_gpu_blocks : src_global;
            const uint32_t dst_local = to_gpu ? dst_global : dst_global - num_gpu_blocks;

            const size_t src_offset = (size_t) src_local * block_bytes;
            const size_t dst_offset = (size_t) dst_local * block_bytes;

            // Put src tensor into HOST staging buffer
            ggml_backend_tensor_get(src_main, staging.data(), src_offset, block_bytes);
            // Put tensor from HOST staging into dst tensor
            ggml_backend_tensor_set(dst_main, staging.data(), dst_offset, block_bytes);
        }
    }
}

bool llama_kv_cache_paged::swap_in(llama_sequence_group & group) {
    const uint32_t num_blocks = group.block_table.size();
    if (num_blocks == 0) {
        return true;
    }
    if (std::any_of(group.block_table.begin(), group.block_table.end(), [](int32_t id) { return id < 0; })) {
        // evicted pages cannot be swapped back in; the server path never hits this
        LLAMA_LOG_WARN("%s: swap_in with evicted pages not supported, freeing instead\n", __func__);
        free_blocks(group);
        return true;
    }

    // A potential optimization to reduce thrashing is to have a heuristic to check if
    // if we can continue decoding after swap_in.
    if (!has_free_gpu_blocks(num_blocks, /*ignore_watermark=*/true)) {
        return false;
    }

    llama_block_ids preview_ids = preview_gpu_blocks(num_blocks);
    if (preview_ids.size() != num_blocks) {
        return false;
    }

    uint32_t required_blocks = required_gpu_capacity();
    if (!preview_ids.empty()) {
        required_blocks = std::max(required_blocks, (uint32_t) *std::max_element(preview_ids.begin(), preview_ids.end()) + 1);
    }
    if (!ensure_physical_capacity(required_blocks)) {
        return false;
    }

    llama_block_ids new_ids = checkout_gpu_blocks(num_blocks);
    do_block_copy(group.block_table, new_ids, /*to_gpu=*/true);

    free_blocks(group);
    group.block_table = new_ids;
    return true;
}

bool llama_kv_cache_paged::swap_out(llama_sequence_group & group) {
    const uint32_t num_blocks = group.block_table.size();
    if (num_blocks == 0) {
        return true;
    }
    if (std::any_of(group.block_table.begin(), group.block_table.end(), [](int32_t id) { return id < 0; })) {
        LLAMA_LOG_WARN("%s: swap_out with evicted pages not supported\n", __func__);
        return false;
    }

    if (!block_manager.has_free_cpu_blocks(num_blocks)) {
        return false;
    }

    std::vector<uint32_t> cpu_ids = block_manager.checkout_cpu_blocks(num_blocks);
    llama_block_ids new_ids(cpu_ids.begin(), cpu_ids.end());
    do_block_copy(group.block_table, new_ids, /*to_gpu=*/false);

    free_blocks(group);
    group.block_table = new_ids;
    return true;
}

void llama_kv_cache_paged::set_paged_batch_info(const llama_paged_batch_info * info) {
    last_paged_info = info;
}

uint32_t llama_kv_cache_paged::get_num_gpu_blocks() const {
    return num_gpu_blocks;
}

bool llama_kv_cache_paged::reserve(uint32_t n_tokens) {
    const uint32_t requested_blocks = std::min<uint64_t>(
        num_gpu_blocks, ((uint64_t) n_tokens + block_size - 1) / block_size);
    return ensure_physical_capacity(std::max(requested_blocks, required_gpu_capacity()));
}

uint32_t llama_kv_cache_paged::freeze_physical_capacity() {
    if (attn_layer_ids.empty()) {
        return 0;
    }

    uint32_t capacity = UINT32_MAX;
    for (const uint32_t il : attn_layer_ids) {
        capacity = std::min(capacity, kv_gpu_layers[il].capacity);
    }
    if (capacity == 0 || capacity == UINT32_MAX) {
        return 0;
    }

    // Stop clear() from restoring the small calibration pool, then discard the
    // synthetic sequence before rebuilding the free block registries.
    allow_dynamic_spill = false;
    initial_num_gpu_blocks = capacity;
    growth_num_gpu_blocks  = capacity;
    clear(false);

    const float watermark = num_gpu_blocks > 0
        ? (float) gpu_watermark_num_blocks / num_gpu_blocks
        : 0.0f;
    num_gpu_blocks = capacity;
    gpu_watermark_num_blocks = std::ceil(num_gpu_blocks * watermark);
    max_logical_blocks = capacity;
    free_gpu_ids.resize(capacity);
    std::iota(free_gpu_ids.begin(), free_gpu_ids.end(), 0);
    gpu_block_ref_counts.assign(capacity, 0);
    block_manager.init(capacity, num_cpu_blocks, watermark);
    return capacity;
}

void llama_kv_cache_paged::set_max_logical_blocks(uint32_t n_logical_blocks) {
    if (n_logical_blocks == 0) {
        return;
    }
    const bool need_alloc = (snapkv_scores_tensor != nullptr || snapkv_token_scores_tensor != nullptr) &&
                            n_logical_blocks > max_logical_blocks;
    if (need_alloc) {
        // a reused decode graph carries the old accumulator pointer in op_params;
        // reallocating would leave a dangling pointer, so refuse to grow post-alloc
        GGML_ASSERT(false && "snapkv scores buffer must not grow after allocation");
    }
    max_logical_blocks = std::max(max_logical_blocks, n_logical_blocks);
}

void llama_kv_cache_paged::configure_snapkv(bool enabled, uint32_t observation_window, uint32_t recent_tokens,
                                            uint32_t pinned_tokens, float retention, uint32_t budget_blocks) {
    snapkv_enabled            = enabled;
    snapkv_observation_window = observation_window;
    snapkv_recent_tokens      = std::max(recent_tokens, observation_window);
    snapkv_pinned_tokens      = pinned_tokens;
    snapkv_retention          = std::clamp(retention, 0.0f, 1.0f);
    snapkv_budget_blocks      = budget_blocks;
    if (!enabled) {
        return;
    }
    if (max_logical_blocks == 0) {
        max_logical_blocks = num_gpu_blocks;
    }
    if (snapkv_scores_tensor != nullptr && snapkv_token_scores_tensor != nullptr) {
        return;
    }
    ggml_backend_t backend = nullptr;
    for (const uint32_t il : attn_layer_ids) {
        if (kv_gpu_layers[il].backend != nullptr) {
            backend = kv_gpu_layers[il].backend;
            break;
        }
    }
    if (backend == nullptr) {
        LLAMA_LOG_WARN("%s: no attention layer backend for snapkv scores; snapkv disabled\n", __func__);
        snapkv_enabled = false;
        return;
    }
    struct ggml_init_params params = {};
    params.mem_size = ggml_tensor_overhead() * 8;
    params.mem_buffer = NULL;
    params.no_alloc = true;
    snapkv_ctx.reset(ggml_init(params));
    const size_t page_score_width = (size_t) max_logical_blocks * snapkv_score_heads;
    const size_t token_score_width = page_score_width * block_size;
    snapkv_scores_tensor = ggml_new_tensor_2d(snapkv_ctx.get(), GGML_TYPE_F32, page_score_width, n_seq_max);
    snapkv_token_scores_tensor = ggml_new_tensor_2d(snapkv_ctx.get(), GGML_TYPE_F32, token_score_width, n_seq_max);
    snapkv_capture_from_tensor = ggml_new_tensor_1d(snapkv_ctx.get(), GGML_TYPE_I32, n_seq_max);
    snapkv_score_slots_tensor = ggml_new_tensor_1d(snapkv_ctx.get(), GGML_TYPE_I32, n_seq_max);
    snapkv_buf.reset(ggml_backend_alloc_ctx_tensors(snapkv_ctx.get(), backend));
    if (!snapkv_buf) {
        LLAMA_LOG_WARN("%s: failed to allocate snapkv scores buffer; snapkv disabled\n", __func__);
        snapkv_enabled = false;
        return;
    }
    snapkv_backend = backend;
    ggml_backend_buffer_clear(snapkv_buf.get(), 0);
    snapkv_scores_host.assign(page_score_width * n_seq_max, 0.0f);
    snapkv_token_scores_host.assign(token_score_width * n_seq_max, 0.0f);
    snapkv_capture_from_host.assign(n_seq_max, -1);
    snapkv_score_slots_host.assign(n_seq_max, -1);
    LLAMA_LOG_INFO("%s: snapkv enabled: observation_window=%u recent=%u pinned=%u retention=%.2f budget_blocks=%u logical_blocks=%u\n",
        __func__, snapkv_observation_window, snapkv_recent_tokens, pinned_tokens, snapkv_retention, budget_blocks, max_logical_blocks);
}

bool llama_kv_cache_paged::snapkv_sync_scores() {
    if (snapkv_scores_tensor == nullptr) {
        return false;
    }
    if (!snapkv_scores_dirty) {
        return true;
    }
    const auto start = std::chrono::steady_clock::now();
    ggml_backend_synchronize(snapkv_backend);
    const size_t page_stride = (size_t) max_logical_blocks * snapkv_score_heads;
    for (const auto & entry : snapkv_score_slots) {
        const uint32_t slot = entry.second;
        if (slot >= n_seq_max) {
            continue;
        }
        ggml_backend_tensor_get(snapkv_scores_tensor,
                                snapkv_scores_host.data() + (size_t) slot * page_stride,
                                (size_t) slot * page_stride * sizeof(float),
                                page_stride * sizeof(float));
    }
    snapkv_scores_dirty = false;
    const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (snapkv_debug_enabled()) {
        fprintf(stderr, "SNAPKVDBG score_sync_ms=%.3f\n", elapsed);
    }
    return true;
}

float llama_kv_cache_paged::snapkv_score(const llama_sequence_group & group, uint32_t bid) const {
    const auto it = snapkv_score_slots.find(group.request_id);
    if (it == snapkv_score_slots.end() || bid >= max_logical_blocks) {
        return 0.0f;
    }
    const size_t page_stride = (size_t) max_logical_blocks * snapkv_score_heads;
    const size_t base = (size_t) it->second * page_stride + bid;
    float score = 0.0f;
    for (uint32_t head = 0; head < snapkv_score_heads; ++head) {
        score = std::max(score, snapkv_scores_host[base + (size_t) head * max_logical_blocks]);
    }
    return score;
}

uint32_t llama_kv_cache_paged::snapkv_global_target(const llama_sequence_group & group) const {
    const uint32_t own_gpu = (uint32_t) std::count_if(group.block_table.begin(), group.block_table.end(), [this](int32_t id) {
        return id >= 0 && block_manager.is_gpu((uint32_t) id);
    });
    const uint32_t shared = (uint32_t) free_gpu_ids.size() + own_gpu;
    const uint32_t usable = shared > gpu_watermark_num_blocks ? shared - gpu_watermark_num_blocks : 0;
    const uint32_t configured = snapkv_budget_blocks > 0 ? snapkv_budget_blocks : num_gpu_blocks;
    return std::min(configured, usable);
}

// Evict pages with the lowest accumulated attention score until the group's
// block table fits target_blocks. Pinned and recent pages are never evicted.
// Logical positions are preserved: evicted slots become -1 in the block table.
uint32_t llama_kv_cache_paged::snapkv_evict_to_target(llama_sequence_group & group, uint32_t target_blocks) {
    const uint32_t total = group.block_table.size();
    const uint32_t active = (uint32_t) std::count_if(group.block_table.begin(), group.block_table.end(),
        [](int32_t id) { return id >= 0; });
    if (active <= target_blocks || total == 0) {
        return 0;
    }

    const auto start = std::chrono::steady_clock::now();
    snapkv_sync_scores();
    const uint32_t pinned_b = std::min(total, (snapkv_pinned_tokens + block_size - 1) / block_size);
    const uint32_t recent_b = std::min(total, (snapkv_recent_tokens + block_size - 1) / block_size);
    const uint32_t old_start = std::min(total, pinned_b);
    const uint32_t old_end   = total > recent_b ? total - recent_b : 0;

    struct candidate {
        int32_t bid;
        float   score;
    };
    std::vector<candidate> candidates;
    candidates.reserve(total);
    for (uint32_t bid = old_start; bid < old_end; ++bid) {
        if (group.block_table[bid] < 0) {
            continue;
        }
        const float score = snapkv_score(group, bid);
        candidates.push_back({ (int32_t) bid, score });
    }
    if (candidates.empty()) {
        return 0;
    }
    const auto by_score = [](const candidate & a, const candidate & b) {
        return a.score != b.score ? a.score < b.score : a.bid < b.bid;
    };
    const size_t selected = std::min<size_t>(active > target_blocks ? active - target_blocks : 0, candidates.size());
    if (selected == 0) {
        return 0;
    }
    std::partial_sort(candidates.begin(), candidates.begin() + selected, candidates.end(), by_score);

    uint32_t n_evicted = 0;
    uint32_t kept = active;
    for (size_t i = 0; i < selected; ++i) {
        const candidate & cand = candidates[i];
        if (kept <= target_blocks) {
            break;
        }
        const int32_t physical = group.block_table[cand.bid];
        if (physical >= 0) {
            if (block_manager.is_gpu((uint32_t) physical)) {
                release_gpu_blocks(llama_block_ids{ static_cast<int32_t>(physical) });
            } else {
                block_manager.release_cpu_blocks({ (uint32_t) physical });
            }
            group.block_table[cand.bid] = -1;
            ++n_evicted;
            --kept;
        }
    }
    if (snapkv_debug_enabled()) {
        fprintf(stderr, "SNAPKVDBG evicted %u of %u active pages (target %u), kept=%u logical_slots=%u\n",
                n_evicted, active, target_blocks, kept, total);
    }
    const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (snapkv_debug_enabled()) {
        fprintf(stderr, "SNAPKVDBG select_evict_ms=%.3f\n", elapsed);
    }
    return n_evicted;
}

// Free enough old pages so that `num_blocks_needed` new physical blocks can be
// allocated while the prefill is still streaming. Uses the partial attention
// scores accumulated so far; pages outside the pinned/recent windows only.
uint32_t llama_kv_cache_paged::snapkv_progressive_evict(llama_sequence_group & group, uint32_t num_blocks_needed) {
    if (group.block_table.empty()) {
        return 0;
    }
    const auto start = std::chrono::steady_clock::now();
    snapkv_sync_scores();

    llama_pos progress = -1;
    auto it = sequence_positions.find(group.request_id);
    if (it != sequence_positions.end()) {
        progress = it->second.max;
    }
    const llama_pos recent_start = progress >= (llama_pos) snapkv_recent_tokens
                                       ? progress - snapkv_recent_tokens : 0;
    const uint32_t pinned_b = std::min((uint32_t) group.block_table.size(),
                                       (snapkv_pinned_tokens + block_size - 1) / block_size);

    struct candidate {
        int32_t bid;
        float   score;
    };
    std::vector<candidate> candidates;
    for (uint32_t bid = pinned_b; bid < group.block_table.size(); ++bid) {
        if (group.block_table[bid] < 0) {
            continue;
        }
        const llama_pos token_start = (llama_pos) bid * block_size;
        if (token_start >= recent_start) {
            continue;
        }
        const float score = snapkv_score(group, bid);
        candidates.push_back({ (int32_t) bid, score });
    }
    if (candidates.empty()) {
        return 0;
    }
    const auto by_score = [](const candidate & a, const candidate & b) {
        return a.score != b.score ? a.score < b.score : a.bid < b.bid;
    };
    const size_t selected = std::min<size_t>(num_blocks_needed, candidates.size());
    if (selected == 0) {
        return 0;
    }
    std::partial_sort(candidates.begin(), candidates.begin() + selected, candidates.end(), by_score);

    uint32_t n_evicted = 0;
    for (size_t i = 0; i < selected; ++i) {
        const candidate & cand = candidates[i];
        const int32_t physical = group.block_table[cand.bid];
        if (physical >= 0) {
            if (block_manager.is_gpu((uint32_t) physical)) {
                release_gpu_blocks(llama_block_ids{ static_cast<int32_t>(physical) });
            } else {
                block_manager.release_cpu_blocks({ (uint32_t) physical });
            }
            group.block_table[cand.bid] = -1;
            ++n_evicted;
        }
    }
    LLAMA_LOG_DEBUG("%s: progressive evict freed %u pages for seq %d\n", __func__, n_evicted, group.request_id);
    if (snapkv_debug_enabled()) {
        fprintf(stderr, "SNAPKVDBG progressive_evicted %u seq=%d\n", n_evicted, group.request_id);
        const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        fprintf(stderr, "SNAPKVDBG progressive_evict_ms=%.3f\n", elapsed);
    }
    return n_evicted;
}

// Move the highest-scored pages still on CPU back to GPU after the final eviction.
bool llama_kv_cache_paged::snapkv_migrate_cpu_to_gpu(llama_sequence_group & group) {
    const uint32_t num_cpu = (uint32_t) std::count_if(group.block_table.begin(), group.block_table.end(),
        [this](int32_t id) { return id >= 0 && !block_manager.is_gpu((uint32_t) id); });
    if (num_cpu == 0) {
        return true;
    }
    std::vector<int32_t> bids;
    for (uint32_t bid = 0; bid < group.block_table.size(); ++bid) {
        const int32_t id = group.block_table[bid];
        if (id >= 0 && !block_manager.is_gpu((uint32_t) id)) {
            bids.push_back((int32_t) bid);
        }
    }
    const uint32_t max_migrate = std::min(num_cpu, (uint32_t) free_gpu_ids.size());
    if (max_migrate == 0) {
        return true;
    }
    std::partial_sort(bids.begin(), bids.begin() + max_migrate, bids.end(), [this, &group](int32_t a, int32_t b) {
        const float sa = snapkv_score(group, (uint32_t) a);
        const float sb = snapkv_score(group, (uint32_t) b);
        return sa != sb ? sa > sb : a < b;
    });

    // Reserve the whole batch after making sure its physical IDs are backed by
    // storage.  If a large batch cannot grow, retry with a smaller one rather
    // than falling back to one synchronized copy per page.
    uint32_t batch_size = max_migrate;
    llama_block_ids new_ids;
    while (batch_size > 0) {
        const llama_block_ids preview_ids = preview_gpu_blocks(batch_size);
        uint32_t required_blocks = required_gpu_capacity();
        if (!preview_ids.empty()) {
            required_blocks = std::max(required_blocks, (uint32_t) *std::max_element(preview_ids.begin(), preview_ids.end()) + 1);
        }
        if (ensure_physical_capacity(required_blocks)) {
            new_ids = checkout_gpu_blocks(batch_size);
            if (new_ids.size() == batch_size) {
                break;
            }
        }
        batch_size /= 2;
    }
    if (batch_size == 0 || new_ids.size() != batch_size) {
        return true;
    }

    llama_block_ids old_ids;
    std::vector<uint32_t> old_global_ids;
    old_ids.reserve(batch_size);
    old_global_ids.reserve(batch_size);
    for (uint32_t i = 0; i < batch_size; ++i) {
        const int32_t old_id = group.block_table[bids[i]];
        old_ids.push_back(old_id);
        old_global_ids.push_back((uint32_t) old_id);
    }

    do_block_copy(old_ids, new_ids, /*to_gpu=*/true);
    block_manager.release_cpu_blocks(old_global_ids);
    for (uint32_t i = 0; i < batch_size; ++i) {
        group.block_table[bids[i]] = new_ids[i];
    }
    const uint32_t migrated = batch_size;
    if (migrated > 0 && snapkv_debug_enabled()) {
        fprintf(stderr, "SNAPKVDBG migrated %u pages CPU->GPU for seq %d\n", migrated, group.request_id);
    }
    return true;
}

bool llama_kv_cache_paged::ensure_pos_blocks(uint32_t max_pos, llama_sequence_group & group) {
    if (!snapkv_enabled) {
        return true;
    }
    const uint32_t blocks_needed = max_pos / block_size + 1;
    auto state_it = snapkv_sequences.find(group.request_id);
    if (blocks_needed > num_gpu_blocks && state_it != snapkv_sequences.end() &&
        state_it->second.mode == snapkv_prefill_mode::strict) {
        state_it->second.mode = snapkv_prefill_mode::streaming;
        state_it->second.capture_from = 0;
        LLAMA_LOG_INFO("%s: seq %d switched strict -> streaming at %u logical blocks (pool=%u)\n",
                       __func__, group.request_id, blocks_needed, num_gpu_blocks);
    }
    if (blocks_needed <= group.block_table.size()) {
        return true;
    }
    const bool streaming = state_it != snapkv_sequences.end() &&
                           state_it->second.mode == snapkv_prefill_mode::streaming;
    uint32_t to_alloc = blocks_needed - (uint32_t) group.block_table.size();
    while (to_alloc > 0) {
        bool allocated = false;
        if (has_free_gpu_blocks(to_alloc, /*ignore_watermark=*/true)) {
            uint32_t required_blocks = required_gpu_capacity();
            llama_block_ids preview_ids = preview_gpu_blocks(to_alloc);
            if (!preview_ids.empty()) {
                required_blocks = std::max(required_blocks, (uint32_t) *std::max_element(preview_ids.begin(), preview_ids.end()) + 1);
            }
            if (ensure_physical_capacity(required_blocks)) {
                llama_block_ids new_ids = checkout_gpu_blocks(to_alloc);
                if (new_ids.size() == to_alloc) {
                    concat_block_ids(group.block_table, new_ids);
                    allocated = true;
                }
            }
        }
        if (allocated) {
            break;
        }
        if (!streaming) {
            return false;
        }
        LLAMA_LOG_DEBUG("%s: physical growth unavailable; evicting before retry for seq %d\n",
                        __func__, group.request_id);
        const uint32_t observation_blocks = (snapkv_observation_window + block_size - 1) / block_size;
        const uint32_t freed = snapkv_progressive_evict(group, std::max(to_alloc, 2 * observation_blocks));
        if (freed == 0) {
            return false;
        }
    }
    return true;
}

void llama_kv_cache_paged::end_prefill(llama_sequence_group & group) {
    if (!snapkv_enabled) {
        return;
    }
    const uint32_t total = (uint32_t) group.block_table.size();
    const uint32_t active = (uint32_t) std::count_if(group.block_table.begin(), group.block_table.end(),
        [](int32_t id) { return id >= 0; });
    const uint32_t ctx_len = group.n_decoded > 0 ? group.n_decoded : group.n_prompt;
    const uint32_t pinned_b = std::min(total, (snapkv_pinned_tokens + block_size - 1) / block_size);
    const uint32_t recent_b = std::min(total, (snapkv_recent_tokens + block_size - 1) / block_size);
    const uint32_t protected_blocks = std::min(active, pinned_b + recent_b);
    const uint32_t old_blocks = active > protected_blocks ? active - protected_blocks : 0;
    const uint32_t keep_old  = (uint32_t) std::ceil(old_blocks * snapkv_retention);

    uint32_t target = snapkv_global_target(group);
    target = std::min(target, active);
    if (snapkv_budget_blocks == 0) {
        target = std::min(target, protected_blocks + keep_old);
    }
    target = std::max(target, protected_blocks);

    if (total > target) {
        snapkv_evict_to_target(group, target);
    } else {
        snapkv_sync_scores();
    }

    float total_mass = 0.0f;
    float kept_mass  = 0.0f;
    for (uint32_t bid = 0; bid < total; ++bid) {
        const float s = snapkv_score(group, bid);
        total_mass += s;
        if (group.block_table[bid] >= 0) {
            kept_mass += s;
        }
    }
    const uint32_t kept_pages = (uint32_t) std::count_if(group.block_table.begin(), group.block_table.end(),
        [](int32_t id) { return id >= 0; });
    const auto state_it = snapkv_sequences.find(group.request_id);
    const char * mode = state_it != snapkv_sequences.end() && state_it->second.mode == snapkv_prefill_mode::streaming
        ? "streaming" : "strict";
    if (snapkv_debug_enabled()) {
        fprintf(stderr, "SNAPKVDBG mass %.4f pct=%.2f pages_kept=%u total_pages=%u mode=%s target=%u budget_old=%u\n",
                total_mass > 0.0f ? kept_mass / total_mass : 1.0f, total_mass > 0.0f ? 100.0f * kept_mass / total_mass : 100.0f,
                kept_pages, total, mode, target, target > protected_blocks ? target - protected_blocks : 0);
    }

    snapkv_migrate_cpu_to_gpu(group);

    snapkv_capture_active = false;
    snapkv_streaming_active = false;
    snapkv_sequences[group.request_id].pending_final_evict = false;
    if (snapkv_debug_enabled()) {
        fprintf(stderr, "SNAPKVDBG prefill_ended seq=%d ctx_len=%u pages_before=%u pages_after=%u\n",
                group.request_id, ctx_len, total, (uint32_t) group.block_table.size());
    }
}

void llama_kv_cache_paged::concat_block_ids(llama_block_ids &       to_block_table,
                                            const llama_block_ids & from_block_table) {
    to_block_table.insert(to_block_table.end(), from_block_table.begin(), from_block_table.end());
}

// llama_memory_i

llama_memory_context_ptr llama_kv_cache_paged::init_batch(llama_batch_allocr & balloc,
                                                          uint32_t             n_ubatch,
                                                          bool /*embd_all*/) {
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_simple(n_ubatch);
            if (ubatch.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ubatch));
        }

        // Failed to find a suitable split
        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            break;
        }

        const bool scheduler_batch = last_paged_info != nullptr;
        if (!scheduler_batch && !prepare_batch(balloc.get_batch())) {
            break;
        }

        auto ctx = std::make_unique<llama_kv_cache_paged_context>(this, std::move(ubatches));
        if (scheduler_batch) {
            ctx->set_batch_data(*last_paged_info);
        } else {
            ctx->set_batch_data(ctx->get_ubatch());
        }
        return ctx;
    } while (false);

    return std::make_unique<llama_kv_cache_paged_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

// Used by llama_context scheduler to dry-run
llama_memory_context_ptr llama_kv_cache_paged::init_full() {
    LLAMA_LOG_DEBUG("%s: reserving graph for n_ubatch=%d, n_seq_max=%d, num_gpu_blocks=%d\n", __func__, n_ubatch,
                    n_seq_max, num_gpu_blocks);

    // Create a "dummy" ubatch that represents the maximum capacity
    // of the system to let the scheduler reserve enough space for metadata.
    llama_ubatch ubatch = {};
    ubatch.n_tokens     = n_ubatch;   // maximum tokens
    ubatch.n_seqs       = n_seq_max;  // maximum sequences
    ubatch.n_pos        = 1;

    std::vector<llama_ubatch> ubatches = { ubatch };

    auto ctx = std::make_unique<llama_kv_cache_paged_context>(this, ubatches);

    ctx->set_batch_size(n_seq_max);       // maximum possible sequences
    ctx->set_n_tokens(n_ubatch);          // representative token count
    ctx->set_max_blocks(max_logical_blocks > 0 ? max_logical_blocks : num_gpu_blocks);

    return ctx;
}

llama_memory_context_ptr llama_kv_cache_paged::init_update(llama_context * /*lctx*/, bool /*optimize*/) {
    return std::make_unique<llama_kv_cache_paged_context>(LLAMA_MEMORY_STATUS_NO_UPDATE);
}

struct ggml_tensor * llama_kv_cache_paged::get_kv_tensor(int layer_idx) const {
    GGML_ASSERT(layer_idx >= 0 && layer_idx < (int32_t) kv_gpu_layers.size());
    GGML_ASSERT(kv_gpu_layers[layer_idx].tensor != nullptr && "layer not in paged KV cache");
    return kv_gpu_layers[layer_idx].tensor;
}

void llama_kv_cache_paged::clear(bool /*data*/) {
    for (auto & entry : regular_groups) {
        if (!entry.second.block_table.empty()) {
            llama_block_ids released_gpu;
            std::vector<uint32_t> released_cpu;
            for (const int32_t block : entry.second.block_table) {
                if (block < 0) {
                    continue;
                }
                if (block_manager.is_gpu((uint32_t) block)) {
                    released_gpu.push_back(block);
                } else {
                    released_cpu.push_back((uint32_t) block);
                }
            }
            if (!released_gpu.empty()) {
                release_gpu_blocks(released_gpu);
            }
            if (!released_cpu.empty()) {
                block_manager.release_cpu_blocks(released_cpu);
            }
        }
    }
    regular_groups.clear();
    sequence_positions.clear();
    snapkv_sequences.clear();
    snapkv_score_slots.clear();
    maybe_restore_initial_storage();
}

bool llama_kv_cache_paged::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos /*p1*/) {
    auto it = regular_groups.find(seq_id);
    if (it != regular_groups.end()) {
        auto & group = it->second;
        if (p0 <= 0) {
            if (!group.block_table.empty()) {
                free_blocks(group);
            }
            regular_groups.erase(it);
            sequence_positions.erase(seq_id);
            snapkv_sequences.erase(seq_id);
            snapkv_score_slots.erase(seq_id);
            return true;
        }

        const size_t keep_blocks = std::min(group.block_table.size(),
                                            (size_t) (p0 + block_size - 1) / block_size);
        llama_block_ids released(group.block_table.begin() + keep_blocks, group.block_table.end());
        if (!released.empty()) {
            llama_block_ids released_gpu;
            std::vector<uint32_t> released_cpu;
            for (const int32_t block : released) {
                if (block < 0) {
                    continue;
                }
                if (block_manager.is_gpu((uint32_t) block)) {
                    released_gpu.push_back(block);
                } else {
                    released_cpu.push_back((uint32_t) block);
                }
            }
            if (!released_gpu.empty()) {
                release_gpu_blocks(released_gpu);
            }
            if (!released_cpu.empty()) {
                block_manager.release_cpu_blocks(released_cpu);
            }
            group.block_table.resize(keep_blocks);
        }

        auto pos_it = sequence_positions.find(seq_id);
        if (pos_it != sequence_positions.end()) {
            pos_it->second.max = std::min(pos_it->second.max, p0 - 1);
        }
    }
    return true;
}

llama_pos llama_kv_cache_paged::seq_pos_min(llama_seq_id seq_id) const {
    auto it = sequence_positions.find(seq_id);
    return (it != sequence_positions.end()) ? it->second.min : -1;
}

llama_pos llama_kv_cache_paged::seq_pos_max(llama_seq_id seq_id) const {
    auto it = sequence_positions.find(seq_id);
    return (it != sequence_positions.end()) ? it->second.max : -1;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_paged::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> breakdown;

    for (const auto & storage : kv_gpu_layers) {
        if (storage.tensor != nullptr) {
            breakdown[ggml_backend_buffer_get_type(storage.tensor->buffer)] += ggml_nbytes(storage.tensor);
        }
    }
    for (const auto * kv_cpu : kv_cpu_layers) {
        breakdown[ggml_backend_buffer_get_type(kv_cpu->buffer)] += ggml_nbytes(kv_cpu);
    }
    return breakdown;
}

void llama_kv_cache_paged::set_seq_min_pos(llama_seq_id seq_id, llama_pos new_min) {
    sequence_positions[seq_id].min = new_min;
}

void llama_kv_cache_paged::set_seq_max_pos(llama_seq_id seq_id, llama_pos new_max) {
    sequence_positions[seq_id].max = new_max;
}

// llama_kv_cache_paged_context

void llama_kv_cache_paged_context::set_batch_data(const llama_paged_batch_info & info) {
    owned_write_slots.assign(info.write_slots, info.write_slots + info.n_tokens);
    owned_block_table.assign(info.block_table, info.block_table + info.n_seq * info.n_blocks_per_seq);
    owned_context_lens.assign(info.context_lens, info.context_lens + info.n_seq);
    owned_batch_offsets.assign(info.batch_offsets, info.batch_offsets + info.n_seq);
    owned_batch_lens.assign(info.batch_lens, info.batch_lens + info.n_seq);
    paged_write_slots   = owned_write_slots.data();
    paged_block_table   = owned_block_table.data();
    paged_context_lens  = owned_context_lens.data();
    paged_batch_offsets = owned_batch_offsets.data();
    paged_batch_lens    = owned_batch_lens.data();
    n_tokens            = info.n_tokens;
    max_blocks          = info.n_blocks_per_seq;
    batch_size          = info.n_seq;
}

void llama_kv_cache_paged_context::set_batch_data(const llama_ubatch & ubatch) {
    llama_paged_batch_info info = {};
    GGML_ASSERT(manager);
    manager->snapkv_update_capture(ubatch);
    GGML_ASSERT(manager->build_batch_info(ubatch, info));
    set_batch_data(info);
}

bool llama_kv_cache_paged_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    manager->commit_batch(ubatches[i_cur]);
    if (++i_cur >= ubatches.size()) {
        return false;
    }
    // SnapKV streaming prefill: make sure the next ubatch has physical pages,
    // evicting old low-importance pages first if the pool is exhausted.
    if (!manager->ensure_batch_blocks(ubatches[i_cur])) {
        return false;
    }
    set_batch_data(ubatches[i_cur]);
    return true;
}

bool llama_kv_cache_paged_context::apply() {
    // Nothing to do for paged KV cache, return true to allow for execution
    return true;
}

const llama_ubatch & llama_kv_cache_paged_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_cur];
}

struct ggml_tensor * llama_kv_cache_paged_context::get_k(int layer_idx) const {
    GGML_ASSERT(manager && "manager has not been initialized.");
    return manager->get_kv_tensor(layer_idx);
}

struct ggml_tensor * llama_kv_cache_paged_context::get_v(int layer_idx) const {
    GGML_ASSERT(manager && "manager has not been initialized.");
    return manager->get_kv_tensor(layer_idx);
}

int32_t llama_kv_cache_paged_context::get_n_tokens() const {
    return n_tokens;
}

int32_t llama_kv_cache_paged_context::get_batch_size() const {
    return batch_size;
}

int32_t llama_kv_cache_paged_context::get_max_blocks() const {
    return max_blocks;
}

uint64_t llama_kv_cache_paged_context::get_storage_generation() const {
    GGML_ASSERT(manager && "manager has not been initialized.");
    return manager->get_storage_generation();
}

int32_t * llama_kv_cache_paged_context::get_write_slots() const {
    return paged_write_slots;
}

int32_t * llama_kv_cache_paged_context::get_block_table() const {
    return paged_block_table;
}

int32_t * llama_kv_cache_paged_context::get_context_lens() const {
    return paged_context_lens;
}

int32_t * llama_kv_cache_paged_context::get_batch_offsets() const {
    return paged_batch_offsets;
}

int32_t * llama_kv_cache_paged_context::get_batch_lens() const {
    return paged_batch_lens;
}

void llama_kv_cache_paged_context::set_n_tokens(int32_t new_n_tokens) {
    n_tokens = new_n_tokens;
}

void llama_kv_cache_paged_context::set_batch_size(int32_t new_batch_size) {
    batch_size = new_batch_size;
}

void llama_kv_cache_paged_context::set_max_blocks(int32_t new_max_blocks) {
    max_blocks = new_max_blocks;
}
