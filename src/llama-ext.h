#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"
#include "llama-kv-memory-stats.h"

#include <array>
#include <cstdint>
#include <map>
#include <vector>

// Convert a q4_0 sequence state held in RAM straight into a caller-owned
// buffer (no temporary file on the critical path). Returns the converted byte
// count and fills `out`; tokens are copied to `tokens_out`.
LLAMA_API size_t llama_state_seq_convert_data_to_mem(
        llama_context * ctx,
        const uint8_t * src, size_t size, uint64_t src_checksum,
        const llama_token * ram_tokens, size_t ram_n_tokens,
        std::vector<uint8_t> & out,
        llama_token * tokens_out, size_t capacity, size_t * count_out);

struct llama_mtp_weights_info {
    bool managed = false;
    bool resident = false;
    size_t host_bytes = 0;
    size_t allocated_bytes = 0; // backend buffer sizes, not RSS or retained file mappings
    size_t gpu_allocated_bytes = 0;
    size_t tensor_count = 0;
    uint64_t model_instance = 0;
    uint64_t model_load_count = 0;
    uint64_t main_gpu_upload_bytes = 0;
    uint64_t mtp_gpu_upload_bytes = 0; // cumulative, including uploads to failed/retired allocations
    uint64_t mtp_reloads = 0;
    uint64_t backing_hash = 0;
};

enum class llama_mtp_weights_fault : uint8_t { none, allocation, upload };

// Exclusive model owner only, with all contexts/speculation destroyed and no adapters.
// Validation failure leaves residency unchanged. Failed reupload clears optional bindings;
// the main model and host backing survive.
// fault is for deterministic failure-injection checks; production callers use none.
LLAMA_API bool llama_model_mtp_weights_set_resident(
        llama_model * model, bool resident, llama_mtp_weights_fault fault = llama_mtp_weights_fault::none);
// Do not read concurrently with residency changes; publish a copy for HTTP status.
LLAMA_API llama_mtp_weights_info llama_model_mtp_weights_get_info(const llama_model * model);

// Managed MTP target, owner-thread only. Zero means no tracked unmasked NextN decode.
// State imports, failed/new decodes, encode, output reservation and a NextN mode change invalidate the ID.
// IDs are monotonic within a context; a context and its ID must be kept together.
LLAMA_API uint64_t llama_get_nextn_decode_id(const llama_context * ctx);
LLAMA_API bool llama_matches_nextn_decode(const llama_context * ctx, uint64_t id, const llama_batch & batch);

struct llama_prompt_cache_profile {
    uint64_t context_instance;
    llama_context_type ctx_type;
    llama_rope_scaling_type rope_scaling_type;
    float rope_freq_base;
    float rope_freq_scale;
    uint32_t n_ctx_orig_yarn;
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;
    bool causal_attn;
    bool kv_unified;
    bool kv_paged;
    bool flash_attn;
    int32_t nextn_layer_offset;
    ggml_type type_k;
    ggml_type type_v;
    ggml_type type_k_aux;
    ggml_type type_v_aux;
    bool kv_layout_known;
    llama_pos attn_min;
    llama_pos attn_max;
    llama_pos attn_aux_min;
    llama_pos attn_aux_max;
    // Components omitted by PARTIAL_ONLY, in stable implementation order.
    // Four pairs cover the current maximum (DSV4); overflow/unknown refuses reuse.
    bool partial_retained_known;
    uint32_t partial_retained_count;
    std::array<llama_pos, 8> partial_retained_bounds;
};

// Context owner thread only. Allocation size and recurrent rollback capacity do not change KV semantics.
LLAMA_API llama_prompt_cache_profile llama_get_prompt_cache_profile(const llama_context * ctx, llama_seq_id seq_id = -1);

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);
LLAMA_API llama_kv_memory_stats llama_get_kv_memory_stats(const struct llama_context * ctx);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);
