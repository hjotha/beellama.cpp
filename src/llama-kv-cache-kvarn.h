#pragma once

#include "llama-kv-mixed-state.h"
#include "llama-kv-cache.h"
#include "llama-kvarn.h"
#include "llama-position-split.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <tuple>
#include <string>
#include <unordered_map>
#include <vector>

struct llama_hparams;
struct llama_model;
struct llama_state_q4_source;
struct llama_state_q4_info;

struct llama_kv_cache_standard_layer_layout {
    uint32_t layer_id = 0;
    ggml_type type_k = GGML_TYPE_COUNT;
    ggml_type type_v = GGML_TYPE_COUNT;
    uint32_t rotation_k = 0;
    uint32_t rotation_v = 0;
    bool v_transposed = false;
    uint32_t head_dim_k = 0;
    uint32_t head_dim_v = 0;
    uint32_t n_head_kv = 0;
    uint64_t row_stride_k = 0;
    uint64_t row_stride_v = 0;
    std::string device;
};

struct llama_kv_cache_kvarn_layer_layout {
    uint32_t layer_id = 0;
    uint16_t kvarn_type = 0;
    uint8_t key_bits = 0;
    uint8_t value_bits = 0;
    uint32_t record_dim_k = 0;
    uint32_t record_dim_v = 0;
    uint32_t head_dim_k = 0;
    uint32_t head_dim_v = 0;
    uint32_t head_slices_k = 0;
    uint32_t head_slices_v = 0;
    uint32_t n_head_kv = 0;
    uint64_t record_stride_k = 0;
    uint64_t record_stride_v = 0;
    ggml_type tail_type = GGML_TYPE_COUNT;
};

bool llama_kvarn_backend_supports_native_ops(ggml_backend_dev_t dev);
bool llama_kvarn_backend_supports_ops(ggml_backend_dev_t dev, int head_dim);
bool llama_kvarn_backend_native_attention_uses_original_v(ggml_backend_dev_t dev);
uint32_t llama_kvarn_backend_native_rotated_max_query_tokens(ggml_backend_dev_t dev);
bool llama_kvarn_backend_mixed_tail_native_preferred(ggml_backend_dev_t dev);

// Writer-side section map of the most recent mixed-state save, one line per
// emitted region. Empty unless F4_STATE_SECTIONS was in the environment for
// that save. Recorded as the bytes are emitted, so it is independent of the
// reader-side walk (llama_mixed_state_section_map).
std::string llama_kv_cache_kvarn_state_write_sections_text();

// Additive handoff accessor (one-way pure->mixed prefix reuse). Exposes the
// authoritative per-layer compressed tensors and geometry for stream 0
// without exposing the private layer struct. Non-virtual; implemented in
// llama-kv-cache-kvarn-handoff.cpp.
struct llama_kvarn_handoff_layer_view {
    uint32_t il = 0;
    uint32_t n_head_kv = 0;
    uint32_t head_dim_k = 0;
    uint32_t head_dim_v = 0;
    uint32_t k_slices = 0;
    uint32_t v_slices = 0;
    uint32_t record_dim_k = 0;
    uint32_t record_dim_v = 0;
    ggml_tensor * k_records = nullptr; // stream 0 (single-stream handoff)
    ggml_tensor * v_records = nullptr;
    ggml_tensor * k_stage = nullptr;
    ggml_tensor * v_stage = nullptr;
    ggml_tensor * k_tail = nullptr;
    ggml_tensor * v_tail = nullptr;
};

struct llama_kvarn_tail_policy {
    uint32_t raw_requested_tokens;
    uint32_t requested_tokens;
    uint32_t effective_tokens;
    uint32_t exact_groups;
    bool native_exact;
};

inline llama_kvarn_tail_policy llama_kvarn_tail_policy_for(
        uint32_t raw_requested_tokens,
        uint32_t effective_window) {
    constexpr uint32_t group_size = 128;
    if (effective_window == 0) {
        return { raw_requested_tokens, 0, 0, 0, true };
    }

    const uint32_t intrinsic = std::min(group_size, effective_window);
    const uint64_t rounded = raw_requested_tokens == 0 ? 0 :
        ((uint64_t(raw_requested_tokens) + group_size - 1u) / group_size) * group_size;
    const uint32_t explicit_tokens = uint32_t(std::min<uint64_t>(rounded, effective_window));
    const uint32_t effective_tokens = std::max(intrinsic, explicit_tokens);
    return {
        raw_requested_tokens,
        effective_tokens,
        effective_tokens,
        (effective_tokens + group_size - 1u) / group_size,
        effective_tokens == effective_window,
    };
}

// Completed groups are committed eagerly. Only the currently incomplete group
// needs persistent quantization workspace, regardless of physical ubatch size.
inline uint32_t llama_kvarn_workspace_groups(uint32_t n_ubatch) {
    GGML_UNUSED(n_ubatch);
    return 1;
}

inline uint32_t llama_kvarn_non_swa_tail_groups(uint32_t n_batch, uint32_t n_ubatch) {
    GGML_UNUSED(n_batch);
    GGML_UNUSED(n_ubatch);

    // Reference KVarN keeps only the currently incomplete 128-token group in
    // F16, and completed groups are quantized before the next group starts.
    // A second transient slot is still required: llama_kvarn_can_remove_range
    // promises a position-independent suffix rollback that reaches back into
    // the previous group, and callers rely on that promise
    // (common_context_seq_rm aborts when a removal it was told to expect is
    // refused). With a single slot both groups alias one stage row range, so
    // re-sealing a partially reopened group sources its surviving prefix from
    // the newer group's rows. Two slots keep the reopened group's own F16
    // source resident for exactly the depth the removal contract advertises.
    //
    // This second slot is rollback insurance only - it does not widen the
    // window served from F16, so it does not move the cache away from the
    // reference's single-incomplete-group semantics. That holds because records
    // are eager: ggml_cuda_fattn_kvarn_group_from_stage() ignores tail_groups on
    // the eager path and serves the sink plus the live group alone. Perplexity
    // is unchanged between depth 1 and 2 (7.1898 either way, Qwen3.6-27B kvarn6
    // at 4K). On the delayed path from_stage() DOES widen with tail_groups, so
    // anything that turns eager records off must revisit this value.
    return 2;
}

class llama_kv_cache_kvarn;

class llama_kv_cache_kvarn_context : public llama_kv_cache_context {
public:
    llama_kv_cache_kvarn_context(
            llama_kv_cache_kvarn * cache,
            llama_memory_context_ptr base,
            llama_context * update_lctx = nullptr,
            std::vector<int32_t> shared_graph_layers = {});

    bool next() override;
    bool apply() override;
    void graph_compute_start() override;
    void graph_compute_finish(ggml_status compute_status) override;
    void graph_compute_complete(ggml_backend_sched_t sched, ggml_status compute_status) override;
    void capture_after_compute(ggml_backend_sched_t sched, ggml_status status) override;

    llama_memory_status get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    uint32_t get_n_kv() const override;
    llama_kv_cache * get_kv() const override;
    const llama_kv_cache::slot_info & current_sinfo() const override;
    const slot_info_vec_t & get_sinfos() const override;
    void get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const override;

    ggml_type type_k() const override;
    ggml_type type_v() const override;

    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const override;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const override;
    ggml_tensor * get_k_tail(ggml_context * ctx, int32_t il) const override;
    ggml_tensor * get_v_tail(ggml_context * ctx, int32_t il) const override;
    uint32_t get_tail_slots() const override;
    ggml_type get_tail_type() const override;
    uint32_t get_tail_tokens() const override;
    uint32_t get_tail_tokens(int32_t il) const override;
    uint32_t get_tail_arena_stride() const override;
    uint32_t get_tail_attention_stride(uint32_t n_query_tokens = 0) const override;
    uint32_t get_tail_body_execution_stride() const override;
    uint32_t get_tail_body_execution_rows(int32_t il) const override;
    bool has_compact_tail() const override;
    bool has_compact_tail(int32_t il) const override;
    bool has_kv_body() const override;
    bool has_kv_body(int32_t il) const override;
    bool has_tail_current(int32_t il) const override;
    ggml_backend_dev_t get_tail_backend(int32_t il) const override;
    llama_kv_tail_storage_kind get_tail_storage_kind() const override;
    uint32_t get_tail_rollback_tokens() const override;
    llama_kv_tail_route get_tail_route(int32_t il) const override;
    const llama_kv_tail_layer_route * get_tail_layer_route(int32_t il) const override;
    bool get_tail_explicit_bias(int32_t il) const override;
    bool can_pack_tail_body(const llama_ubatch & ubatch) const override;
    ggml_tensor * get_k_native(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v_native(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_k_for_attention(ggml_context * ctx, int32_t il, bool native_attention) const;
    ggml_tensor * get_v_for_attention(ggml_context * ctx, int32_t il, bool native_attention) const;
    // F3: recuperação KVarN restrita a uma lista explícita de células
    // (leitura por faixa). Usa os mesmos caminhos native/materialize da
    // leitura completa, mas com n_kv e mat idxs do chamador. O chamador
    // garante que todas as células pertencem à mesma faixa e que o plano de
    // leitura cobre exatamente essas células.
    // F4: range_id is EXPLICIT and names the body being READ, which is not
    // necessarily the body this graph stored. build_attn_position_split always
    // reads the local body (kRangeLocal) even inside an overflow-only ubatch,
    // where the current range is 1; deriving the read key from the current range
    // made the local read depend on which ubatch happened to be executing and
    // only worked because the overflow store writes no slot. A Q4 view must
    // never be reachable through this path: the local body is F16 KVarN stage.
    // F4: marks a split-range tensor for executed-node capture (fixture only,
    // gated by F4_TRACE_CAPTURE inside). Lets llama-graph.cpp mark the tensors
    // build_attn_position_split feeds its two FA nodes.
    void mark_split_tensor(ggml_tensor * t, int32_t il, bool value, uint32_t range_id) const;
    ggml_tensor * get_k_for_range(
            ggml_context * ctx, int32_t il, bool native_attention,
            uint32_t range_id, uint32_t n_range, ggml_tensor * range_mat_idxs) const;
    ggml_tensor * get_v_for_range(
            ggml_context * ctx, int32_t il, bool native_attention,
            uint32_t range_id, uint32_t n_range, ggml_tensor * range_mat_idxs) const;
    bool uses_native_attention(int32_t il) const;
    bool uses_kvarn_layer(int32_t il) const;
    bool uses_standard_layer(int32_t il) const;
    bool has_standard_cache() const;
    bool has_qualified_dflash_mask() const;
    ggml_backend_dev_t native_attention_backend(int32_t il) const;
    bool position_split_enabled() const;
    // Occupancy plan 3.1: "O tail/stage do KVarN continua so na faixa local;
    // o overflow Q4 nao tem tail." True when `il` carries the KVarN exact tail
    // under the current placement; the tail follows the LOCAL owner of the
    // layer, never the presence of the Q4 companion.
    bool tail_on_layer(int32_t il) const;
    // True when a LOCAL tail can be read by the graph being built: no valid
    // ubatch (graph reserve) or an overflow-only ubatch with NO local history
    // means there is nothing local to pair with; an overflow ubatch that DOES
    // have local history is exactly the dual-range builder, which must read
    // the local tail (plan 3.4) even though its current range is overflow.
    bool local_tail_available() const;
    // F3: acesso ao cache proprietário para leitura por faixa (listas de
    // células, máscaras e índices). O grafo usa somente métodos const.
    const llama_kv_cache_kvarn * get_kvarn_cache() const { return cache; }
    bool mixed_tail_native_preferred(int32_t il) const;
    bool native_attention_uses_original_v(int32_t il) const;
    uint32_t native_rotated_max_query_tokens(int32_t il) const;
    bool uses_compact_read_indices() const;
    bool uses_materialization_indices() const;
    // Whether the graph currently being built routes an OVERFLOW ubatch: the
    // only range whose store targets the Q4 companion and whose chain the fused
    // split span may replace. False during graph reserve (no valid ubatch).
    bool current_range_is_overflow() const { return current_position_split_range_id() == llama_position_split::kRangeOverflow; }

    // SWA sliding-window ring: per-cell absolute positions for KVarN reads.
    // Built as a graph input sized [n_kv]; set on the host from cells.pos_get(cell).
    ggml_tensor * build_input_kvarn_rot(ggml_context * ctx, int n_rot) const;
    void set_standard_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_kvarn_rot(ggml_tensor * dst) const;
    ggml_tensor * build_input_kvarn_mat_idxs(ggml_context * ctx) const;
    void set_input_kvarn_mat_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    // F4: set_mat_idxs runs once per graph, before any K/V read. The store
    // slots hold tensors owned by the PREVIOUS graph's ggml_context, and that
    // context is reused across ubatches. Without clearing here, a slot written
    // by an earlier ubatch is still readable by the next one: in P-mode an
    // overflow ubatch left a Q4 view under the local key and the local KVarN
    // read then consumed Q4_0 bytes as if they were the F16 stage. Clearing
    // makes every read resolve against this graph's own stores, falling back
    // to the persistent layer stage.
    void set_mat_idxs(ggml_tensor * idxs) const { mat_idxs = idxs; }
    // F3: índices de leitura KVarN restritos à faixa local (pos<P), com a
    // mesma codificação de stage-slot do caminho completo. O chamador obtém a
    // lista via cache->position_split_live_cells(LOCAL) e dimensiona dst com
    // o tamanho dela; células vazias nunca entram na lista.
    // F4: P-mode graphs drive the local range through this setter instead of
    // set_mat_idxs, so it must reset the store slots too; otherwise the slots
    // written by a previous ubatch survive into this graph.
    void set_input_kvarn_mat_idxs_local(
            ggml_tensor * dst, const std::vector<uint32_t> & local_cells) const {
        set_input_kvarn_mat_idxs_local_impl(dst, local_cells);
    }

    // F4: drop the previous ubatch's K/V store slots. Public because the graph
    // builder calls it at the ubatch boundary even when the range input tensor
    // has no buffer yet (first graph of a slot), where a stale slot would
    // otherwise still be visible to the local read.
    void reset_graph_store_slots() const override {
        if (getenv("LLAMA_KVARN_SLOT_TRACE") != nullptr) {
            std::fprintf(stderr, "kv-slot-reset k=%zu v=%zu\n", stored_k.size(), stored_v.size());
        }
        stored_k.clear();
        stored_v.clear();
    }

    // F4 test seam: exposes the REAL per-graph store slots and the REAL key
    // builder, so a regression can drive the production identity logic instead of
    // re-implementing it. Read/write only; the production path never calls these.
    struct store_slot_probe {
        // THE production key: (mapped_layer_id << 32) | range_id. Shared with the
        // store and the readers so a test cannot drift from production identity.
        static uint64_t range_store_key(uint32_t mapped_layer_id, uint32_t range_id) {
            return (int64_t(mapped_layer_id) << 32) | int64_t(range_id);
        }
        static uint64_t key(uint32_t mapped_layer_id, uint32_t range_id) {
            return range_store_key(mapped_layer_id, range_id);
        }
        static size_t size(const llama_kv_cache_kvarn_context & ctx, bool value) {
            return value ? ctx.stored_v.size() : ctx.stored_k.size();
        }
        static void put(llama_kv_cache_kvarn_context & ctx, uint32_t mapped_layer_id,
                        uint32_t range_id, ggml_tensor * t, bool value) {
            (value ? ctx.stored_v : ctx.stored_k)[key(mapped_layer_id, range_id)] = t;
        }
        static ggml_tensor * get(const llama_kv_cache_kvarn_context & ctx,
                                 uint32_t mapped_layer_id, uint32_t range_id, bool value) {
            const auto & m = value ? ctx.stored_v : ctx.stored_k;
            const auto it = m.find(key(mapped_layer_id, range_id));
            return it == m.end() ? nullptr : it->second;
        }
    };
    friend struct store_slot_probe;

    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const override;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const override;
    ggml_tensor * cpy_k_with_tail(
            ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs,
            ggml_tensor * tail_idxs, int32_t il) const override;
    ggml_tensor * cpy_v_with_tail(
            ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs,
            ggml_tensor * tail_idxs, int32_t il) const override;
    ggml_tensor * cpy_k_tail(
            ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * tail_idxs,
            int32_t il, ggml_tensor * dependency = nullptr) const override;
    ggml_tensor * cpy_v_tail(
            ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * tail_idxs,
            int32_t il, ggml_tensor * dependency = nullptr) const override;

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const override;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const override;
    ggml_tensor * build_input_tail_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const override;
    ggml_tensor * build_input_tail_body_idxs(ggml_context * ctx) const override;
    ggml_tensor * build_input_k_rot(ggml_context * ctx) const override;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const override;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const override;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const override;
    void set_input_tail_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const override;
    void set_input_tail_body_idxs(ggml_tensor * dst) const override;
    void set_input_k_idxs_backend(ggml_tensor * dst, const llama_ubatch * ubatch) const override;
    void set_input_v_idxs_backend(ggml_tensor * dst, const llama_ubatch * ubatch) const override;
    void set_input_k_shift(ggml_tensor * dst) const override;
    void set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const override;
    void set_input_kq_mask_tail(
            ggml_tensor * body, ggml_tensor * exact,
            ggml_tensor * read_idxs, ggml_tensor * body_read_idxs, ggml_tensor * bias_read_idxs,
            const llama_ubatch * ubatch, bool causal_attn) const override;
    void set_input_tail_body_plan(
            ggml_tensor * query_order, ggml_tensor * run_desc,
            ggml_tensor * body_mask, const llama_ubatch * ubatch, bool causal_attn) const override;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const override;
    void set_input_k_rot(ggml_tensor * dst) const override;
    void set_input_v_rot(ggml_tensor * dst) const override;
    void set_input_k_rot_backend(ggml_tensor * dst) const override;
    void set_input_v_rot_backend(ggml_tensor * dst) const override;

private:
    llama_kv_cache_context * base() const;
    int32_t graph_layer_for(int32_t il) const;
    // F2b: safe current-ubatch range for position-split routing. Returns true
    // and sets range when the base has a valid ubatch. Returns false during
    // graph reserve/sizing (no valid ubatch); the caller must fall back to
    // legacy layer routing instead of dereferencing an invalid ubatch.
    bool current_position_split_range(llama_position_split::range_kind & range) const;
    // F4: range id of the store/read slot for the current graph. Outside P-mode
    // this is always the local body (0), which is the historical single-range
    // behaviour; in P-mode an overflow ubatch reads and writes range 1 so the
    // two physical bodies of one layer never share a slot.
    uint32_t current_position_split_range_id() const {
        llama_position_split::range_kind range = llama_position_split::range_kind::LOCAL;
        return current_position_split_range(range)
                ? uint32_t(range == llama_position_split::range_kind::OVERFLOW
                        ? llama_position_split::kRangeOverflow
                        : llama_position_split::kRangeLocal)
                : llama_position_split::kRangeLocal;
    }
    bool uses_shared_live_indices() const;
    const std::vector<int64_t> & compact_read_plan() const;
    std::vector<int64_t> shared_live_indices() const;

    llama_kv_cache_kvarn * cache;
    llama_memory_context_ptr base_ctx;
    std::vector<int32_t> shared_graph_layers;
    mutable std::vector<int64_t> compact_read_plan_cache;
    llama_context * update_lctx;

    // F4: the store slot is keyed by (layer_id, range_id), NOT layer_id alone.
    // In P-mode one full layer owns two physical bodies -- the local KVarN stage
    // (F16) and the overflow Q4 companion -- so a layer-only key let the Q4
    // view overwrite the local stage and the local read then consumed Q4_0
    // bytes as if they were the F16 stage. range_id 0 is the local body.
    void set_input_kvarn_mat_idxs_local_impl(
            ggml_tensor * dst, const std::vector<uint32_t> & local_cells) const;

    mutable std::unordered_map<int64_t, ggml_tensor *> stored_k;
    mutable std::unordered_map<int64_t, ggml_tensor *> stored_v;
    mutable ggml_tensor * mat_idxs = nullptr; // SWA per-cell absolute positions for views/materialization
};

class llama_kv_cache_kvarn : public llama_memory_i {
public:
    llama_kv_cache_kvarn(
            const llama_model & model,
            const llama_hparams & hparams,
            llama_kvarn_params params,
            bool offload,
            bool unified,
            uint32_t kv_size,
            uint32_t n_seq_max,
            uint32_t n_batch,
            uint32_t n_ubatch,
            uint32_t n_pad = 1,
            uint32_t n_swa = 0,
            llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE,
            const layer_filter_cb & filter = nullptr,
            const layer_reuse_cb & reuse = nullptr,
            uint32_t tail_tokens = 0,
            ggml_type tail_type = GGML_TYPE_F16,
            uint32_t tail_tokens_requested = UINT32_MAX,
            uint32_t tail_rollback_tokens = 0,
            const layer_device_cb & device_for_layer = nullptr,
            const layer_device_cb & migration_device_for_layer = nullptr,
            const layer_filter_cb & standard_layer_filter = nullptr,
            ggml_type standard_type_k = GGML_TYPE_COUNT,
            ggml_type standard_type_v = GGML_TYPE_COUNT,
            bool standard_v_trans = false,
            const layer_device_cb & standard_device_for_layer = nullptr,
            uint32_t position_split_p = 0,
            bool rpc_position_split = false);
    ~llama_kv_cache_kvarn() override;

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;
    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    uint32_t get_kv_n_stream() const override;
    uint32_t get_kv_size() const override;
    llama_memory_context_ptr init_kv_batch(const std::vector<llama_ubatch> & ubatches) override;

    bool get_can_shift() const override;
    seq_rm_capability get_seq_rm_capability() const override;

    void clear(bool data) override;
    bool can_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const override;
    bool seq_rm_plan(
            llama_seq_id seq_id, llama_pos p0, llama_pos p1,
            llama_pos & planned_p0, llama_pos & planned_p1) const override;
    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;
    bool seq_rm_cell(llama_seq_id seq_id, uint32_t cell_idx) override;
    int cells_at_pos(llama_seq_id seq_id, llama_pos pos, uint32_t * cell_indices, int n_max) override;
    void seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    GGML_NORETURN void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) override;
    GGML_NORETURN void seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) override;
    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;
    llama_kv_memory_stats kv_memory_stats() const override;
    ggml_type get_kv_tail_type() const override { return exact_tail_type; }
    uint32_t get_kv_tail_group_count() const override { return 1; }
    bool get_kv_tail_coverage(
            uint32_t group_index,
            llama_seq_id seq_id,
            llama_kv_tail_coverage_info & out) const override;
    void reset_kv_tail_planner_timing() override;
    uint64_t get_kv_tail_planner_timing_ns() const override;

    bool supports_prefill_migration() const override { return migration_enabled; }
    bool handoff_prefill_migration(bool to_remote) override;
    void release_prefill_migration_inactive_buffers() noexcept override;
    bool drain_prefill_migration() override;
    void enqueue_prefill_migration(
            const llama_kv_cache::slot_info & sinfo,
            ggml_backend_sched_t sched);

    bool requires_state_for_partial_restore() const override;
    bool state_seq_can_save(llama_seq_id seq_id) const override;
    bool state_seq_can_restore(llama_seq_id seq_id) const override;
    bool state_seq_can_save(llama_seq_id seq_id, llama_state_seq_flags flags) const override;
    bool state_seq_can_restore(llama_seq_id seq_id, llama_state_seq_flags flags) const override;
    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read(llama_io_read_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;
    void state_read_sinfo(
            llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags,
            llama_kv_cache::slot_info_vec_t * sinfos_out,
            const llama_kv_cache::slot_info_vec_t * sinfos_in);

    bool state_streaming_restore_supported() const override;

    // Restricted q4_0/q4_0 -> native KVarN conversion. Only complete
    // untransposed q4 source prefixes for this exact cache shape are accepted
    // (no SWA ring, no exact tail, one stream). Writes a canonical sequence
    // state file for this cache's layout and returns its native byte size.
    bool state_parse_q4(
            llama_state_q4_source & src,
            const llama_hparams & hparams,
            llama_state_q4_info & info,
            std::string & error) override;
    size_t state_convert_q4(
            llama_state_q4_source & src,
            const llama_state_q4_info & info,
            const char * dst_path,
            std::vector<uint8_t> * out_mem = nullptr) override;

    llama_kv_cache * get_metadata_cache() const;
    ggml_tensor * get_materialization_source(int32_t il, bool value) const;
    std::unique_ptr<llama_kv_cache> make_shared_metadata_cache(const llama_model & model_view) const;
    int32_t mapped_layer_id(int32_t il) const;
    bool uses_kvarn_layer(int32_t il) const;
    bool uses_standard_layer(int32_t il) const;
    bool has_kvarn_layers() const { return !layers.empty(); }
    bool has_standard_cache() const { return bool(standard_cache); }

    // Per-LAYER, per-FIELD diagnostic of the LIVE state of one sequence.
    // One line per layer with named fields: layer id, whether the layer lives on
    // the remote backend, the live cell count, the live position range, the
    // payload type/bit widths and a hash over the layer's live payload. Used to
    // compare a PURE source against a destination immediately after an import,
    // per layer and per field, instead of diffing an opaque serialized blob.
    std::string live_layer_table(llama_seq_id seq_id) const;

    // Flip one byte in the addressed span; see llama_kvarn_probe_mutation.
    bool probe_mutation(llama_seq_id seq_id, int layer_index, const char * tensor,
            uint32_t row, bool live);
    bool probe_head_span(llama_seq_id seq_id, int layer_index, uint32_t head,
            uint32_t group_or_slot, const char * tensor, bool live,
            uint8_t * orig_out = nullptr, uint8_t * new_out = nullptr,
            uint32_t * group_used_out = nullptr);
    bool probe_restore_byte(llama_seq_id seq_id, int layer_index, uint32_t head,
            uint32_t group, const char * tensor, uint8_t orig);
    // Remote-overflow probe: flip one byte of the Q4 companion storage for a
    // model attention layer, at (head, storage row). Live targets the first
    // owned cell with pos >= P (the occupied overflow range); inactive targets
    // the first empty cell, which no gather reads. Row addressing reuses the
    // exact view layout the overflow gather uses (absolute cell rows).
    // Returns the target position through pos_used_out (-1 when none).
    bool probe_remote_span(llama_seq_id seq_id, int layer_index, uint32_t head,
            bool value, bool live, uint8_t * orig_out = nullptr,
            uint8_t * new_out = nullptr, uint32_t * row_used_out = nullptr,
            llama_pos * pos_used_out = nullptr);
    bool probe_remote_restore_byte(llama_seq_id seq_id, int layer_index, uint32_t head,
            bool value, uint32_t row, uint8_t orig);

    // Consumer-path probe: read the values the attention graph actually consumes
    // for (layer, head, position) through the KVarN materializer, with no decode.
    bool materialized_probe(llama_seq_id seq_id, int layer_index, uint32_t head,
            bool value, llama_pos pos, std::vector<float> & out,
            std::string * status_out = nullptr, ggml_backend_t backend = nullptr) const;
    // Backend owned by this cache, used by materialized_probe(). Assigned from
    // the cache's own transfer backends; never created on demand.
    ggml_backend_t probe_backend = nullptr;

public:
    // The leaf tensor materialized_probe() will read, for callers that need to
    // inspect its buffer type.
    const ggml_tensor * probe_leaf(int layer_index, bool value) const;
    // A backend that this cache's context already owns and that supports the
    // leaf's buffer type, or nullptr. Never initialises a backend.
    ggml_backend_t resolve_probe_backend(int layer_index, bool value) const;

    // Captured consumer-leaf samples, filled by f4_trace_attention_source()
    // during the real build_attn routing (F4_TRACE_CAPTURE=1). This is the only
    // honest way to compare B/C against the tensor the consumer actually read.
    struct capture_sample {
        bool        valid   = false;
        bool        value   = false;
        int32_t     layer   = -1;
        uint32_t    head    = 0;
        llama_pos   pos     = -1;
        std::string route;
        std::string range;
        std::string name;
        int         type    = -1;
        uint64_t    offset  = 0;
        uint64_t    nbytes  = 0;
        int32_t     shared_il = -1;   // graph/model layer used by build_attn
        int32_t     mapped    = -1;   // KVarN storage id for that model layer
        std::string stored_name;       // persistent store behind the read
        std::string stored_from;       // range slot or materialization source
        int         stored_type = -1;
        std::string records_name;      // I8 record leaf for KVarN reads
        int         records_type = -1;
        std::string backend;           // device/buft that executed the node
        std::string src;               // executed input chain for compute nodes
        uint32_t    mark_seq  = 0;     // scheduler mark for this request
        size_t      fa_hits   = 0;     // FA nodes executed with this leaf input
        uint64_t    gen       = 0;     // scheduler compute generation that produced it
        std::vector<uint8_t>  raw;   // exact bytes read (Q4_0/permuted)
        std::vector<float>    vec;   // dequantized values (F16/F32 only)
    };
    static std::vector<capture_sample> & capture_slot(int32_t layer, bool value, uint32_t head);
    // Indexes the samples captured for (layer, value, head); `out` receives every
    // distinct route the consumer used, newest last. Multiple routes coexist in
    // one decode (split-aware LOCAL and legacy standard_local), so collapsing to
    // one sample hides which source the graph really consumed.
    size_t capture_count(int32_t layer, bool value, uint32_t head) const;
    // Returns the captured sample for (layer, head) after the most recent
    // consumer read, or false when that consumer route never captured it.
    bool captured(int32_t layer, bool value, uint32_t head,
                  llama_pos pos, size_t index, capture_sample & out) const;
    void clear_captures() const;
    static void capture_from_leaf(const char * fn, int32_t il, bool value, const char * route,
                                  llama_position_split::range_kind range, bool have_range,
                                  uint32_t range_id, ggml_tensor * t, int32_t shared_il,
                                  int32_t mapped, llama_pos pos);
    // Scheduler-eval capture for tensors that only exist as executed graph
    // nodes. `eval_note_candidate` marks the returned K/V tensor with a unique
    // fixture-only name; `eval_wants_node` matches that mark on the actually
    // allocated/executed node copy; `eval_capture_node` reads it after the
    // scheduler synchronizes that node. Never invents data and never creates a
    // backend. Only active when F4_TRACE_CAPTURE is set.
    struct eval_mark {
        uint32_t    seq = 0;
        int32_t     il = -1;
        bool        value = false;
        uint32_t    range_id = 0;
        std::string route;
        std::string range;
        std::string orig_name;
        int32_t     shared_il = -1;
        int32_t     mapped = -1;
        std::string stored_name;
        std::string stored_from;
        int         stored_type = -1;
        const ggml_tensor * stored_ptr = nullptr; // persistent/graph store behind the read
        uint64_t    stored_nbytes = 0;
        std::string records_name;
        int         records_type = -1;
        int64_t     head_dim = 0;   // marked tensor ne[0]: consumer head width
        int64_t     n_head = 0;     // marked tensor ne[1]: consumer head count
        llama_pos   pos = 0;
    };
    static bool eval_capture_armed();
    static uint32_t eval_note_candidate(int32_t il, bool value, uint32_t range_id,
                                        const char * route, const char * range,
                                        int32_t shared_il, int32_t mapped,
                                        llama_pos pos, ggml_tensor * t);
    static bool eval_wants_node(struct ggml_tensor * t);
    static bool eval_capture_node(struct ggml_tensor * t);
    static size_t eval_mark_count();
    static size_t eval_hit_count();

    struct pending_capture {
        int32_t il; bool value;
        llama_position_split::range_kind range; bool have_range;
        const ggml_tensor * tensor;
    };
    static std::vector<pending_capture> pending_;
    void capture_after_compute(ggml_backend_sched_t sched, ggml_status status);

public:
    mutable std::map<std::tuple<int32_t, bool, uint32_t>, std::vector<capture_sample>> captures;
    // F2b: fronteira de posição entre a faixa local KVarN4 [0,P) e a faixa
    // overflow Q4 [P,C). P == 0 mantém o cache de faixa única legado.
    bool position_split_enabled() const { return position_split_p_ != 0; }
    uint32_t position_split_p() const { return position_split_p_; }
    uint32_t position_split_local_capacity() const { return kv_size_local; }
    uint32_t position_split_overflow_capacity() const { return kv_size_overflow; }
    // Tabela lógica única de faixas por camada full (Fase 2b): dois
    // descritores por camada, [0,P) CUDA/KVarN4 e [P,C) Vulkan0/Q4, sem
    // lacunas nem sobreposições. Vazio quando o modo está desligado.
    std::vector<llama_position_split::range_desc> position_split_ranges() const;
    // Motivo de rejeição do modo por posição, vazio quando válido.
    const std::string & position_split_error() const { return position_split_error_; }
    // F3: células vivas por faixa para a sequência 0, ordenadas por posição
    // absoluta. Usado para montar índices/máscaras de leitura por faixa.
    // Retorna false sem mutar nada se o modo estiver desligado, a sequência
    // for inválida ou qualquer célula estiver fora do contrato.
    bool position_split_live_cells(
            llama_position_split::range_kind range,
            std::vector<uint32_t> & cells_out) const;
    // Additive handoff accessor: raw standard-cache storage for remote layers
    // (rawstorage token rows; the attention views are permuted).
    llama_kv_cache * get_standard_cache() const { return standard_cache.get(); }
    // Additive handoff accessor: authoritative per-layer compressed tensors.
    bool handoff_layer_view(int32_t il, llama_kvarn_handoff_layer_view & out) const;
    std::vector<llama_kv_cache_standard_layer_layout> standard_layer_layout() const;
    std::vector<llama_kv_cache_kvarn_layer_layout> kvarn_layer_layout() const;
    uint64_t mixed_state_owner_id() const { return state_owner_id; }
    uint64_t mixed_state_owner_epoch() const { return state_owner_epoch; }
    // [TAG_SLOT_SAVE_CHECKPOINTS] Adopt the lineage epoch carried by a durable checkpoint
    // companion. Server restore/handoff flows call this only after the full state that
    // produced those partial snapshots was loaded into this cache; every replay still passes
    // the fine-grained reference/identity validation.
    bool adopt_state_owner_epoch(uint64_t epoch) {
        if (epoch == 0 || epoch == UINT64_MAX) {
            return false;
        }
        state_owner_epoch = epoch;
        return true;
    }
    // [TAG_SLOT_SAVE_CHECKPOINTS] Adopt the owner id carried by a durable
    // checkpoint companion (payloads are owner-bound; the full-state restore
    // recreates the cells but not the process-random owner id). Server
    // restore/handoff flows call this only after loading the state the
    // checkpoints were captured from; every replay still passes the
    // fine-grained cell-reference validation.
    bool adopt_state_owner_id(uint64_t owner_id) {
        if (owner_id == 0) {
            return false;
        }
        state_owner_id = owner_id;
        return true;
    }
    // F4 v2: stable model/RoPE identity stamped into the position_split envelope
    // by the caller that built this cache. A snapshot saved under one identity is
    // refused by a cache with a different one, before any staging happens.
    void set_mixed_state_identity(uint64_t model_id, uint64_t rope_id) {
        state_model_id = model_id;
        state_rope_id  = rope_id;
    }
    uint64_t mixed_state_model_id() const { return state_model_id; }
    uint64_t mixed_state_rope_id() const { return state_rope_id; }
    ggml_tensor * standard_get_k(ggml_context * ctx, int32_t il, uint32_t n_kv,
            const llama_kv_cache::slot_info & sinfo) const;
    ggml_tensor * standard_get_v(ggml_context * ctx, int32_t il, uint32_t n_kv,
            const llama_kv_cache::slot_info & sinfo) const;
    ggml_tensor * standard_cpy_k(ggml_context * ctx, ggml_tensor * current, ggml_tensor * indices,
            int32_t il, const llama_kv_cache::slot_info & sinfo) const;
    ggml_tensor * standard_cpy_v(ggml_context * ctx, ggml_tensor * current, ggml_tensor * indices,
            int32_t il, const llama_kv_cache::slot_info & sinfo) const;
    ggml_tensor * standard_build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * standard_build_input_v_rot(ggml_context * ctx) const;
    void standard_set_input_k_rot(ggml_tensor * tensor) const;
    void standard_set_input_v_rot(ggml_tensor * tensor) const;
    void standard_set_input_k_rot_backend(ggml_tensor * tensor) const;
    void standard_set_input_v_rot_backend(ggml_tensor * tensor) const;
    void standard_cache_set_input_kq_mask(ggml_tensor * tensor,
            const llama_ubatch * ubatch, bool causal_attn) const;
    std::vector<uint32_t> standard_layer_ids() const;
    llama_kv_tail_route get_tail_route(int32_t il) const;
    bool get_tail_explicit_bias(int32_t il) const;
    ggml_type get_tail_type() const { return exact_tail_type; }
    bool has_pending_stream_copies() const;
    bool stream_is_exclusive_for(llama_seq_id seq_id) const;
    bool apply_pending_stream_copies(llama_context * lctx);
    bool is_swa() const { return swa; }
    // A QSA model's index cache mirrors the attention cells row for row, so the
    // compact read plan (which reorders rows by physical record) cannot be used.
    void set_indexer_mirror(bool value) { indexer_mirror = value; }
    bool uses_compact_read_indices() const { return !indexer_mirror && !swa && n_stream == 1 && n_seq_max > 1; }
    bool uses_native_attention(int32_t il) const;
    bool has_qualified_dflash_mask() const;
    ggml_backend_dev_t native_attention_backend(int32_t il) const;
    bool mixed_tail_native_preferred(int32_t il) const;
    bool native_attention_uses_original_v(int32_t il) const;
    uint32_t native_rotated_max_query_tokens(int32_t il) const;

    // Fused position-split span ("peça (c)"): one executor bundle per context
    // (this cache instance). The graph builder acquires it lazily and bakes the
    // returned id into every fused op; ~llama_context releases it before the
    // backends go away. A pair change re-registers (the old bundle is released
    // first). Thread-safety: the graph build of one context is single-threaded.
    uint32_t acquire_fused_span_bundle(ggml_backend_t cuda, ggml_backend_t vulkan) const;
    void release_fused_span_bundle();

    // Reference-faithful staging keeps one incomplete 128-token group lossless.
    // Completed records are committed eagerly, so physical ubatch size does not
    // change the logical exact suffix.
    //   non-SWA tail_groups = 2 rollback-safe transient slots
    //   SWA tail_groups     = 2 physical wrap-safety slots
    //   stage_groups        = tail_groups + 1 for non-SWA, tail_groups for SWA
    // The +1 is only the permanent sink slot for non-SWA. SWA has no sink slot,
    // so both F16 slots are physical quantization workspace. The SWA record ring
    // carries the active ubatch span because early query rows can still attend
    // older window groups after later rows advance the ring.
    uint32_t get_stage_groups() const { return stage_groups; }
    uint32_t get_tail_groups()  const { return tail_groups; }
    // Additive handoff accessors (one-way pure->mixed prefix reuse).
    uint32_t get_exact_tail_tokens() const { return exact_tail_tokens; }
    int32_t params_key_bits() const { return params.key_bits; }
    int32_t params_value_bits() const { return params.value_bits; }
    int32_t params_kvarn_type() const { return (int32_t) params.type; }

    ggml_tensor * store(
            ggml_context * ctx,
            ggml_tensor * current,
            ggml_tensor * indices,
            int32_t il,
            const llama_kv_cache::slot_info & sinfo,
            bool value) const;
    ggml_tensor * view(
            ggml_context * ctx,
            ggml_tensor * stored,
            int32_t il,
            uint32_t n_kv,
            const llama_kv_cache::slot_info & sinfo,
            bool value,
            ggml_tensor * mat_idxs = nullptr) const;
    ggml_tensor * materialize(
            ggml_context * ctx,
            ggml_tensor * stored,
            int32_t il,
            uint32_t n_kv,
            const llama_kv_cache::slot_info & sinfo,
            bool value,
            ggml_tensor * mat_idxs = nullptr,
            bool read_indirect = true) const;
    ggml_tensor * get_tail(ggml_context * ctx, int32_t il, bool value) const;
    ggml_tensor * store_tail(
            ggml_context * ctx, ggml_tensor * current, ggml_tensor * indices,
            int32_t il, bool value, ggml_tensor * dependency = nullptr) const;

private:
    struct layer {
        uint32_t il;
        uint32_t n_head_kv;
        uint32_t head_dim_k;
        uint32_t head_dim_v;
        uint32_t k_slices;
        uint32_t v_slices;
        bool native_attention;
        ggml_backend_dev_t native_attention_owner;
        bool mixed_tail_native;
        bool native_original_v;
        uint32_t native_rotated_max_query_tokens;
        ggml_tensor * k_records;
        ggml_tensor * v_records;
        ggml_tensor * k_stage;
        ggml_tensor * v_stage;
        ggml_tensor * k_tail;
        ggml_tensor * v_tail;
        std::vector<ggml_tensor *> k_records_stream;
        std::vector<ggml_tensor *> v_records_stream;
        std::vector<ggml_tensor *> k_stage_stream;
        std::vector<ggml_tensor *> v_stage_stream;
        ggml_tensor * mirror_k_records = nullptr;
        ggml_tensor * mirror_v_records = nullptr;
        ggml_tensor * mirror_k_stage = nullptr;
        ggml_tensor * mirror_v_stage = nullptr;
        ggml_tensor * mirror_k_tail = nullptr;
        ggml_tensor * mirror_v_tail = nullptr;
        std::vector<ggml_tensor *> mirror_k_records_stream;
        std::vector<ggml_tensor *> mirror_v_records_stream;
        std::vector<ggml_tensor *> mirror_k_stage_stream;
        std::vector<ggml_tensor *> mirror_v_stage_stream;
        bool mirror_native_attention = false;
        bool mirror_mixed_tail_native = false;
        bool mirror_native_original_v = false;
        uint32_t mirror_native_rotated_max_query_tokens = 0;
        ggml_backend_dev_t mirror_dev = nullptr;
    };

    const layer & layer_for(int32_t il) const;
    void state_write_kvarn_body(llama_io_write_i & io, llama_seq_id seq_id,
            llama_state_seq_flags flags) const;
    void state_read_kvarn_body(llama_io_read_i & io, llama_seq_id seq_id,
            llama_state_seq_flags flags, llama_kv_cache::slot_info_vec_t * sinfos_out,
            const llama_kv_cache::slot_info_vec_t * sinfos_in,
            const std::function<void(const llama_kv_cache &,
                    const std::unordered_map<uint32_t, uint32_t> &)> & validate_prepared = {});
    void state_write_mixed(llama_io_write_i & io, llama_seq_id seq_id,
            llama_state_seq_flags flags) const;
    void state_read_mixed(llama_io_read_i & io, llama_seq_id seq_id,
            llama_state_seq_flags flags, llama_kv_cache::slot_info_vec_t * sinfos_out,
            const llama_kv_cache::slot_info_vec_t * sinfos_in, uint32_t magic_pre = 0);

    // F4 §3.6: authorized PURE KVarN frame -> mixed/position-split cache.
    // `io` must present the original byte stream (use llama_io_read_replay when
    // the leading word was already read to classify the frame): the pure reader
    // consumes the metadata block and then the KVRN body marker itself, in the
    // same order the pure writer produced them.
    //
    // Identity, types, domain, positions, exact tail, recurrent state and
    // integrity are validated by the existing pure reader; gate_prepared then
    // runs on the prepared metadata cache, still before the live cache is
    // mutated. Only the local range [0,P) is populated and the overflow range
    // [P,C) stays empty. Nothing is truncated or requantized.
    void state_read_pure_to_mixed_import(llama_io_read_i & io, llama_seq_id seq_id,
            llama_state_seq_flags flags, llama_kv_cache::slot_info_vec_t * sinfos_out,
            const llama_kv_cache::slot_info_vec_t * sinfos_in);
    std::unique_ptr<llama_kv_cache> make_metadata_cache() const;
    bool can_remove(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const;
    void ensure_migration_mirror_storage(layer & layer);
    void release_migration_layer_storage(layer & layer, ggml_backend_dev_t device) noexcept;
    void copy_kvarn_stream(uint32_t stream_src, uint32_t stream_dst);

    struct cache_buffer {
        int32_t migration_layer; // -1 for shared per-device storage
        ggml_backend_buffer_type_t buft;
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buffer;
    };

    const llama_model & model;
    const llama_hparams & hparams;
    const llama_kvarn_params params;
    const uint32_t n_stream;
    const uint32_t n_seq_max;
    // Capacidade lógica total C (tabela única de células/posições).
    const uint32_t kv_size;
    // F2b: capacidades por faixa. Com P == 0 ambas reproduzem kv_size, e o
    // cache segue de faixa única (comportamento legado). Com P != 0, a faixa
    // local KVarN4 tem P células e o intervalo lógico de overflow Q4 é C-P.
    // O companheiro Q4 compartilha a tabela lógica de células do metadados,
    // portanto usa as mesmas coordenadas absolutas; somente as linhas de
    // células de overflow recebem stores.
    const uint32_t kv_size_local;
    const uint32_t kv_size_overflow;
    const uint32_t position_split_p_;
    const bool native_kvarn_rpc_experimental_;
    // Declaration order matters: stage_groups depends on tail_groups, so
    // tail_groups must be declared first (C++ initializes members in order).
    const uint32_t tail_groups;   // non-SWA scheduler span; SWA fixed local tail
    const uint32_t stage_groups;   // F16 stage depth (non-SWA sink + tail; SWA tail only)
    const bool swa;
    bool indexer_mirror = false;   // set when an index-aware QSA cache mirrors this one
    const uint32_t n_groups_per_stream;
    const uint32_t exact_tail_tokens;
    const uint32_t metadata_n_pad;
    const uint32_t metadata_n_swa;
    const llama_swa_type metadata_swa_type;
    const uint32_t metadata_n_ubatch;
    const uint32_t exact_tail_tokens_requested;
    const ggml_type exact_tail_type_requested;
    ggml_type exact_tail_type;
    uint64_t state_owner_id;
    uint64_t state_owner_epoch = 1;
    // F4 v2: model/RoPE identity for the position_split envelope. Zero means
    // "not stamped"; a split cache refuses to save without it.
    uint64_t state_model_id = 0;
    uint64_t state_rope_id  = 0;

    std::unique_ptr<llama_kv_cache> metadata;
    // Optional ordinary Qx payloads for selected remote layers. This cache
    // shares only metadata/cells with `metadata`; its K/V buffers remain owned.
    std::unique_ptr<llama_kv_cache> standard_cache;

    // Fused-span executor bundle identity (see acquire_fused_span_bundle()).
    mutable uint32_t fused_span_bundle_id = 0;
    mutable ggml_backend_t fused_span_bundle_cuda = nullptr;
    mutable ggml_backend_t fused_span_bundle_vulkan = nullptr;
    std::vector<layer> layers;
    std::unordered_map<int32_t, int32_t> map_layer_ids;
    std::vector<cache_buffer> ctxs_bufs;
    llama_kv_cache::stream_copy_info pending_stream_copies;
    const bool migration_enabled;
    std::string position_split_error_;
    ggml_backend_dev_t migration_prefill_dev = nullptr;
    ggml_backend_dev_t migration_remote_dev = nullptr;
    bool migration_remote_active = false;
    mutable bool migration_mirror_stale = false;
    std::vector<llama_kv_tail_layer_route> migration_local_tail_routes;
    std::vector<llama_kv_tail_layer_route> migration_remote_tail_routes;
    struct migration_queue;
    std::unique_ptr<migration_queue> migration_copies;
};
