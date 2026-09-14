#pragma once

#include "llama-batch.h"
#include "llama-block-manager.h"
#include "llama-graph.h"
#include "llama-memory.h"
#include "llama-sequence-group.h"

#include <cmath>

//
// llama_kv_cache_paged
//

class llama_kv_cache_paged : public llama_memory_i {
  public:
    llama_kv_cache_paged(uint32_t head_dim,
                         uint32_t n_head_q,
                         uint32_t n_head_kv,
                         uint32_t block_size,
                         uint32_t n_layers,
                         uint32_t n_ubatch,
                         uint32_t n_seq_max,
                         const llama_memory_i::layer_filter_cb & filter_attn = nullptr);

    void init(const std::vector<ggml_backend_t> & layer_backends,
              const std::vector<ggml_backend_t> & candidate_backends,
              ggml_backend_t backend_cpu,
              enum ggml_type type,
              uint32_t       n_gpu_blocks,
              uint32_t       n_cpu_blocks,
              float          watermark,  // percentage
              uint32_t       initial_gpu_blocks,
              uint32_t       growth_gpu_blocks,
              bool           dynamic_spill);

    void init(const std::vector<ggml_backend_t> & layer_backends,
              ggml_backend_t backend_cpu,
              enum ggml_type type,
              uint32_t       n_gpu_blocks,
              uint32_t       n_cpu_blocks,
              float          watermark) {  // percentage
        init(layer_backends, {}, backend_cpu, type, n_gpu_blocks, n_cpu_blocks, watermark, n_gpu_blocks, n_gpu_blocks, false);
    }

    bool reserve(uint32_t n_tokens) override;
    bool allocate(int32_t num_tokens, llama_sequence_group & group);
    void free_blocks(llama_sequence_group & group);
    bool swap_in(llama_sequence_group & group);
    bool swap_out(llama_sequence_group & group);

    // Prepare the llama-server context path. The scheduler path continues to
    // provide its own llama_paged_batch_info.
    bool prepare_batch(const llama_batch & batch);
    bool build_batch_info(const llama_ubatch & ubatch, llama_paged_batch_info & info) const;
    void commit_batch(const llama_ubatch & ubatch);

    void set_paged_batch_info(const llama_paged_batch_info * info);
    const llama_paged_batch_info * get_paged_batch_info() const { return last_paged_info; }

    uint32_t get_num_gpu_blocks() const;
    uint32_t freeze_physical_capacity();
    uint64_t get_storage_generation() const { return storage_generation; }

    // SnapKV-style selective page retention.
    void configure_snapkv(bool enabled, uint32_t observation_window, uint32_t recent_tokens,
                          uint32_t pinned_tokens, float retention, uint32_t budget_blocks);
    bool is_snapkv_enabled() const { return snapkv_enabled; }
    void set_max_logical_blocks(uint32_t n_logical_blocks);
    uint32_t get_max_logical_blocks() const { return max_logical_blocks; }
    struct ggml_tensor * get_snapkv_scores() const { return snapkv_capture_active ? snapkv_scores_tensor : nullptr; }
    struct ggml_tensor * get_snapkv_token_scores() const { return snapkv_capture_active ? snapkv_token_scores_tensor : nullptr; }
    bool get_snapkv_streaming() const { return snapkv_streaming_active; }
    struct ggml_tensor * get_snapkv_capture_from() const { return snapkv_capture_active ? snapkv_capture_from_tensor : nullptr; }
    struct ggml_tensor * get_snapkv_score_slots() const { return snapkv_capture_active ? snapkv_score_slots_tensor : nullptr; }
    void snapkv_update_capture(const llama_ubatch & ubatch);
    void set_snapkv_prefill_end(llama_seq_id seq_id, llama_pos prefill_end) override;
    bool ensure_pos_blocks(uint32_t max_pos, llama_sequence_group & group);
    bool ensure_batch_blocks(const llama_ubatch & ubatch);
    void end_prefill(llama_sequence_group & group);
    //
    // llama_memory_i
    //
    llama_memory_context_ptr init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) override;

    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    struct ggml_tensor * get_kv_tensor(int layer_idx) const;

    bool get_can_shift() const override { return false; }

    void clear(bool data) override;

    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;

    void seq_cp(llama_seq_id /*seq_id_src*/,
                llama_seq_id /*seq_id_dst*/,
                llama_pos /*p0*/,
                llama_pos /*p1*/) override { /* implement later CoW mechanism */
    }

    void seq_keep(llama_seq_id /*seq_id*/) override {}

    void seq_add(llama_seq_id /*seq_id*/, llama_pos /*p0*/, llama_pos /*p1*/, llama_pos /*shift*/) override {}

    void seq_div(llama_seq_id /*seq_id*/, llama_pos /*p0*/, llama_pos /*p1*/, int /*d*/) override {}

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load
    bool state_partial_retained(std::vector<const llama_memory_i *> & out) const override {
        out.push_back(this);
        return true;
    }

    void state_write(llama_io_write_i & /*io*/,
                     llama_seq_id /*seq_id*/         = -1,
                     llama_state_seq_flags /*flags*/ = 0) const override {}

    void state_read(llama_io_read_i & /*io*/,
                    llama_seq_id /*seq_id*/         = -1,
                    llama_state_seq_flags /*flags*/ = 0) override {}

    //
    // Helpers to llama_memory_i
    //
    void set_seq_min_pos(llama_seq_id seq_id, llama_pos new_min);
    void set_seq_max_pos(llama_seq_id seq_id, llama_pos new_max);

  private:
    struct layer_storage {
        ggml_backend_t original_backend = nullptr;
        ggml_backend_t candidate_backend = nullptr;
        ggml_backend_t backend = nullptr;
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        struct ggml_tensor * tensor = nullptr;
        uint32_t capacity = 0;
    };

    void concat_block_ids(llama_block_ids & to_block_table, const llama_block_ids & from_block_table);
    void do_block_copy(const llama_block_ids & src_ids, const llama_block_ids & new_ids, bool to_gpu);
    bool has_free_gpu_blocks(uint32_t num_requested_blocks, bool ignore_watermark) const;
    llama_block_ids preview_gpu_blocks(uint32_t num_blocks) const;
    llama_block_ids checkout_gpu_blocks(uint32_t num_blocks);
    void release_gpu_blocks(const llama_block_ids & freed_blocks);
    uint32_t required_gpu_capacity() const;
    struct ggml_tensor * get_cpu_tensor(int layer_idx) const;
    struct ggml_tensor * create_layer_tensor(struct ggml_context * ctx, enum ggml_type type, uint32_t num_blocks) const;
    bool reset_layer_storage(layer_storage & storage, ggml_backend_t target_backend, uint32_t num_blocks, size_t bytes_to_copy);
    bool ensure_physical_capacity(uint32_t required_blocks, bool rebalance = false);
    void maybe_restore_initial_storage();
    bool snapkv_sync_scores();
    void snapkv_start_prefill(llama_sequence_group & group, llama_pos prefill_end);
    void snapkv_schedule_capture(llama_sequence_group & group, llama_pos capture_end);
    bool snapkv_is_strict(const llama_sequence_group & group) const;
    uint32_t snapkv_global_target(const llama_sequence_group & group) const;
    float snapkv_score(const llama_sequence_group & group, uint32_t bid) const;
    uint32_t snapkv_evict_to_target(llama_sequence_group & group, uint32_t target_blocks);
    uint32_t snapkv_progressive_evict(llama_sequence_group & group, uint32_t num_blocks_needed);
    bool snapkv_migrate_cpu_to_gpu(llama_sequence_group & group);

    std::vector<layer_storage> kv_gpu_layers;
    std::vector<struct ggml_tensor *> kv_cpu_layers;
    ggml_context_ptr cpu_ctx;
    ggml_backend_buffer_ptr cpu_buf;

    std::vector<int32_t> attn_layer_ids;
    std::vector<int32_t> layer_cpu_indices;
    std::vector<ggml_backend_t> spill_backends;

    enum ggml_type kv_type;

    llama_block_manager block_manager;

    // Non-owning pointer to the batch currently being processed.
    // Lifetime: set by the scheduler at the end of step(), cleared at the
    // start of the next step() (before the batch's paged_* arrays are freed).
    // The ordering in llama_paged_scheduler_impl::clear_batch is load-bearing;
    // do not reorder without updating init_batch's contract.
    const llama_paged_batch_info * last_paged_info = nullptr;

    const uint32_t head_dim;
    const uint32_t n_heads_kv;
    const uint32_t block_size;
    const uint32_t n_layers;
    const uint32_t n_ubatch;
    const uint32_t n_seq_max;

    llama_memory_i::layer_filter_cb filter_attn;

    uint32_t       num_gpu_blocks;
    uint32_t       num_cpu_blocks;
    uint32_t       initial_num_gpu_blocks;
    uint32_t       growth_num_gpu_blocks;
    uint32_t       gpu_watermark_num_blocks;
    size_t         block_bytes;
    bool           allow_dynamic_spill;
    uint64_t       storage_generation = 0;

    ggml_backend_t cpu_backend;

    std::vector<uint32_t> free_gpu_ids;
    std::vector<uint32_t> gpu_block_ref_counts;

    // SnapKV state
    bool     snapkv_enabled = false;
    uint32_t snapkv_observation_window = 0;
    uint32_t snapkv_recent_tokens = 0;
    uint32_t snapkv_pinned_tokens = 0;
    float    snapkv_retention = 1.0f;
    uint32_t snapkv_budget_blocks = 0;
    uint32_t max_logical_blocks = 0;
    const uint32_t snapkv_score_heads;
    enum class snapkv_prefill_mode {
        strict,
        streaming,
    };
    struct snapkv_sequence_state {
        bool pending_final_evict = false;
        snapkv_prefill_mode mode = snapkv_prefill_mode::strict;
        llama_pos expected_prefill_end = -1;
        llama_pos next_capture_pos = 0;
        llama_pos capture_from = -1;
        llama_pos capture_until = -1;
    };
    bool     snapkv_capture_active = false;
    bool     snapkv_streaming_active = false;
    bool     snapkv_scores_dirty = false;
    ggml_context_ptr snapkv_ctx;
    ggml_backend_buffer_ptr snapkv_buf;
    ggml_backend_t snapkv_backend = nullptr;
    struct ggml_tensor * snapkv_scores_tensor = nullptr;
    struct ggml_tensor * snapkv_token_scores_tensor = nullptr;
    struct ggml_tensor * snapkv_capture_from_tensor = nullptr;
    struct ggml_tensor * snapkv_score_slots_tensor = nullptr;
    std::vector<float> snapkv_scores_host;
    // Host-side zero staging for resetting the device token accumulator.
    std::vector<float> snapkv_token_scores_host;
    std::vector<int32_t> snapkv_capture_from_host;
    std::vector<int32_t> snapkv_score_slots_host;
    std::unordered_map<llama_seq_id, snapkv_sequence_state> snapkv_sequences;
    std::unordered_map<llama_seq_id, uint32_t> snapkv_score_slots;

    struct seq_range {
        llama_pos min = -1;
        llama_pos max = -1;
    };

    std::unordered_map<llama_seq_id, seq_range> sequence_positions;
    std::unordered_map<llama_seq_id, llama_sequence_group> regular_groups;

    mutable std::vector<int32_t> regular_write_slots;
    mutable std::vector<int32_t> regular_block_table;
    mutable std::vector<int32_t> regular_context_lens;
    mutable std::vector<int32_t> regular_batch_offsets;
    mutable std::vector<int32_t> regular_batch_lens;
};

class llama_kv_cache_paged_context : public llama_memory_context_i {
  public:
    llama_kv_cache_paged_context(llama_kv_cache_paged * parent, const std::vector<llama_ubatch> & in_ubatch) :
        manager(parent),
        ubatches(in_ubatch) {
        i_cur = 0;
    }

    llama_kv_cache_paged_context(llama_memory_status status) : status(status) {}

    void    set_batch_data(const llama_paged_batch_info & info);
    void    set_batch_data(const llama_ubatch & ubatch);
    int32_t get_n_tokens() const;
    int32_t get_batch_size() const;
    int32_t get_max_blocks() const;
    uint64_t get_storage_generation() const;
    int32_t * get_write_slots() const;
    int32_t * get_block_table() const;
    int32_t * get_context_lens() const;
    int32_t * get_batch_offsets() const;
    int32_t * get_batch_lens() const;

    struct ggml_tensor * get_snapkv_scores() const { return manager ? manager->get_snapkv_scores() : nullptr; }
    struct ggml_tensor * get_snapkv_token_scores() const { return manager ? manager->get_snapkv_token_scores() : nullptr; }
    bool get_snapkv_streaming() const { return manager && manager->get_snapkv_streaming(); }
    struct ggml_tensor * get_snapkv_capture_from() const { return manager ? manager->get_snapkv_capture_from() : nullptr; }
    struct ggml_tensor * get_snapkv_score_slots() const { return manager ? manager->get_snapkv_score_slots() : nullptr; }

    void set_n_tokens(int32_t new_n_tokens);
    void set_batch_size(int32_t new_batch_size);
    void set_max_blocks(int32_t new_max_blocks);

    struct ggml_tensor * get_k(int layer_idx) const;
    struct ggml_tensor * get_v(int layer_idx) const;

    //
    // llama_memory_context_i
    //
    bool                 next() override;
    bool                 apply() override;
    const llama_ubatch & get_ubatch() const override;

    llama_memory_status get_status() const override { return status; }

  private:
    llama_kv_cache_paged * manager;

    //
    // batch processing context
    //
    std::vector<llama_ubatch> ubatches;
    size_t                    i_cur = 0;      // index of ubatch to process

    int32_t * paged_write_slots   = nullptr;  // [n_tokens]
    int32_t * paged_block_table   = nullptr;  // [batch_size, max_blocks]
    int32_t * paged_context_lens  = nullptr;  // [batch_size]
    int32_t * paged_batch_offsets = nullptr;  // [batch_size]
    int32_t * paged_batch_lens    = nullptr;  // [batch_size]

    std::vector<int32_t> owned_write_slots;
    std::vector<int32_t> owned_block_table;
    std::vector<int32_t> owned_context_lens;
    std::vector<int32_t> owned_batch_offsets;
    std::vector<int32_t> owned_batch_lens;

    int32_t n_tokens   = 0;
    int32_t batch_size = 0;
    int32_t max_blocks = 0;

    llama_memory_status status = LLAMA_MEMORY_STATUS_SUCCESS;
};
