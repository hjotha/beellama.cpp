#include "llama-graph.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-sampler.h"
#include "ggml-remote-attn.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-paged.h"
#include "llama-kv-cache-dsa.h"
#include "llama-kv-cache-dsa-iswa.h"
#include "llama-kv-cache-msa.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-kv-cache-kvarn.h"
#include "llama-kvarn-split-fused.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-hybrid-paged.h"
#include "llama-memory-recurrent.h"
#include "llama-position-split-merge.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_set>
#include <unordered_map>

// dedup helpers

static void llm_flash_attn_ext_set_kvarn_domain(
        ggml_tensor * cur,
        enum ggml_flash_attn_ext_kvarn_domain domain) {
    while (cur != nullptr &&
            (cur->op == GGML_OP_RESHAPE ||
             cur->op == GGML_OP_PERMUTE ||
             cur->op == GGML_OP_TRANSPOSE ||
             cur->op == GGML_OP_VIEW ||
             cur->op == GGML_OP_CONT) &&
            cur->src[0] != nullptr) {
        cur = cur->src[0];
    }

    if (cur != nullptr && cur->op == GGML_OP_FLASH_ATTN_EXT) {
        cur->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_KVARN_DOMAIN] = (int32_t) domain;
    }
}

static ggml_tensor * ggml_kvarn_wht_aux(
        ggml_context * ctx,
        ggml_tensor * cur,
        int64_t       head_width) {
    GGML_ASSERT(head_width == 64 || head_width == 128 || head_width == 256 || head_width == 512);
    if (!ggml_is_contiguous(cur)) {
        cur = ggml_cont(ctx, cur);
    }
    return ggml_kvarn_wht(ctx, cur, (int) head_width);
}

static ggml_tensor * llm_kvarn_rot_for_dim(
        ggml_tensor * rot_64,
        ggml_tensor * rot_128,
        ggml_tensor * rot_256,
        ggml_tensor * rot_512,
        int64_t       head_dim) {
    switch (head_dim) {
        case  64: return rot_64;
        case 128: return rot_128;
        case 256: return rot_256;
        case 512: return rot_512;
        default:  return nullptr;
    }
}

static void llm_kvarn_set_rot_inputs(
        const llama_kv_cache_kvarn_context * kvarn,
        ggml_tensor * rot_64,
        ggml_tensor * rot_128,
        ggml_tensor * rot_256,
        ggml_tensor * rot_512) {
    GGML_ASSERT(kvarn != nullptr);
    if (rot_64 && rot_64->buffer) {
        kvarn->set_input_kvarn_rot(rot_64);
    }
    if (rot_128 && rot_128->buffer) {
        kvarn->set_input_kvarn_rot(rot_128);
    }
    if (rot_256 && rot_256->buffer) {
        kvarn->set_input_kvarn_rot(rot_256);
    }
    if (rot_512 && rot_512->buffer) {
        kvarn->set_input_kvarn_rot(rot_512);
    }
}

static ggml_tensor * build_attn_inp_kq_mask(
        ggml_context * ctx,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    // flash attention requires an f16 mask
    const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kq_mask");

    return res;
}

// Physical-map binding for the packed overflow view, factored out because TWO reuse
// paths must enforce it: llm_graph_input_attn_kv::can_reuse and the hybrid memory
// inputs' can_reuse (which duplicates a subset of the attention checks and does NOT
// call it -- that duplication is exactly how a stale view survived 254 decode steps).
static bool split_packed_map_still_valid(
        const llama_kv_cache_context * attn_mctx,
        bool     packed,
        uint32_t baked_row0,
        uint32_t baked_rows,
        const char * where) {
    if (!packed) {
        return true;
    }
    // The packed overflow view lives in the KVarN attention context. Accept either that
    // context directly or a hybrid wrapper around it.
    const llama_kv_cache_context * kv_attn = nullptr;
    if (const auto * kv = dynamic_cast<const llama_kv_cache_kvarn_context *>(attn_mctx)) {
        kv_attn = kv;
    } else if (const auto * hyb = dynamic_cast<const llama_memory_hybrid_context *>(attn_mctx)) {
        kv_attn = hyb->get_attn_kv_context();
    }
    const auto * kvarn_ctx = dynamic_cast<const llama_kv_cache_kvarn_context *>(kv_attn);
    if (!kvarn_ctx || !kvarn_ctx->position_split_enabled()) {
        return false;
    }
    // baked_row0 is deliberately the absolute/global cell ID. This reuse guard
    // compares metadata cell order, not compact Qx physical rows; only packed
    // view construction calls payload_row_for_cell().
    const auto * kvarn_cache = kvarn_ctx->get_kvarn_cache();
    std::vector<uint32_t> ovf;
    if (!kvarn_cache ||
        !kvarn_cache->position_split_live_cells(
                    llama_position_split::range_kind::OVERFLOW, ovf)) {
        return false;
    }
    // Capacity-stable extent: the live run may grow up to the baked extent while
    // the graph is reused; the mask is refilled every batch and rows past the live
    // run stay masked. Only a changed first row or a fragmented run invalidates.
    const bool same_extent = !ovf.empty() && ovf.front() == baked_row0 &&
        ovf.size() <= size_t(baked_rows);
    bool ascending = same_extent;
    for (size_t i = 1; ascending && i < ovf.size(); ++i) {
        ascending = ovf[i] == ovf[i-1] + 1;
    }
    if (!(same_extent && ascending)) {
        if (std::getenv("F4_ALLOC_TRACE") != nullptr) {
            fprintf(stderr,
                    "[F5OVF] reuse REFUSED (%s): baked row0=%u rows=%u | live row0=%u n=%zu contiguous=%d\n",
                    where, baked_row0, baked_rows, ovf.empty() ? 0u : ovf.front(),
                    ovf.size(), int(ascending));
        }
        return false;
    }
    return true;
}

static bool can_reuse_kq_mask(
        ggml_tensor * kq_mask,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    bool res = true;

    res &= (kq_mask->ne[0] == n_kv);
    res &= (kq_mask->ne[1] == n_tokens/n_stream);
    res &= (kq_mask->ne[2] == 1);
    res &= (kq_mask->ne[3] == n_stream);

    return res;
}

// impl

void llm_graph_input_dspark_ctx::set_input(const llama_ubatch * ubatch) {
    if (correction_prefix_rows > 0) {
        if (!ctx_feat || !dctx || dctx->n_ctx_rows != ctx_feat->ne[1]) {
            throw std::runtime_error("DSpark correction prefix requires staged context inputs");
        }
        for (int64_t row = 0; row < ubatch->n_tokens; ++row) {
            const bool expected = row >= dctx->n_ctx_rows && row < dctx->n_ctx_rows + correction_prefix_rows;
            if (!ubatch->output || (ubatch->output[row] != 0) != expected) {
                throw std::runtime_error("DSpark correction prefix requires matching output rows");
            }
        }
    }
    if (reuse_mask_id >= 0) {
        if (!ubatch->token || !dctx || dctx->n_ctx_rows != ctx_feat->ne[1]) {
            throw std::runtime_error("DSpark mask reuse requires staged token inputs");
        }
        for (int64_t i = dctx->n_ctx_rows + 1; i < ubatch->n_tokens; ++i) {
            if (ubatch->token[i] != reuse_mask_id) {
                throw std::runtime_error("DSpark mask reuse requires an anchor followed by MASK tokens");
            }
        }
    }
    // ignores ubatch entirely (like llm_graph_input_cross_embd): the context
    // feature row count (n_ctx_rows) is independent of the current ubatch's
    // token count and comes purely from the staged llama_dspark_ctx.
    if (ctx_feat && dctx && !dctx->v_ctx_feat.empty()) {
        GGML_ASSERT((int64_t) dctx->v_ctx_feat.size() == ggml_nelements(ctx_feat));
        ggml_backend_tensor_set(ctx_feat, dctx->v_ctx_feat.data(), 0, ggml_nbytes(ctx_feat));
    }
}

void llm_graph_input_dspark_logsnr::set_input(const llama_ubatch * ubatch) {
    // v_feat was precomputed at graph-build time and does not depend on the ubatch
    GGML_UNUSED(ubatch);
    if (feat && !v_feat.empty()) {
        GGML_ASSERT((int64_t) v_feat.size() == ggml_nelements(feat));
        ggml_backend_tensor_set(feat, v_feat.data(), 0, ggml_nbytes(feat));
    }
}

bool llm_graph_input_dspark_logsnr::can_reuse(const llm_graph_params & params) {
    // v_feat is a function of n_tokens and n_seqs_unq, both already checked by
    // llm_graph_params::allow_reuse, and of min/max_log_snr, fixed per model
    return feat && feat->ne[1] == params.ubatch.n_tokens;
}

void llm_graph_input_embd::set_input(const llama_ubatch * ubatch) {
    if (ubatch->token) {
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    }

    if (ubatch->embd && embd && !ubatch->is_mixed()) {
        GGML_ASSERT(n_embd == embd->ne[0]);

        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(embd));
    }

    if (ubatch->is_mixed() && embd) {
        GGML_ASSERT(mixed_tokens && mixed_slots && mixed_embd && "mixed token/embd ubatch is not supported here");

        std::vector<int32_t> ids;
        std::vector<int64_t> slots;
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            if (!ubatch->type[i]) {
                ids.push_back(ubatch->token[i]);
                slots.push_back(i);
            }
        }
        GGML_ASSERT((int64_t) ids.size() == mixed_tokens->ne[0]);
        GGML_ASSERT(n_embd == mixed_embd->ne[0]);

        ggml_backend_tensor_set(mixed_tokens, ids.data(),    0, ggml_nbytes(mixed_tokens));
        ggml_backend_tensor_set(mixed_slots,  slots.data(),  0, ggml_nbytes(mixed_slots));
        ggml_backend_tensor_set(mixed_embd,   ubatch->embd,  0, ggml_nbytes(mixed_embd));
    }

    if (scale_rows) {
        const int64_t n_tokens = ubatch->n_tokens;

        std::vector<float> data(n_tokens);
        for (int64_t i = 0; i < n_tokens; ++i) {
            const bool is_embd = !ubatch->token || (ubatch->is_mixed() && ubatch->type[i]);
            data[i] = is_embd ? 1.0f : scale_tok;
        }
        ggml_backend_tensor_set(scale_rows, data.data(), 0, ggml_nbytes(scale_rows));
    }
}

// number of token rows of the mixed path, a non-mixed ubatch is sized for the worst case
static int64_t llm_graph_n_tok_rows(const llama_ubatch & ubatch) {
    if (!ubatch.is_mixed()) {
        return ubatch.n_tokens;
    }
    int64_t n = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        n += !ubatch.type[i];
    }
    return n;
}

bool llm_graph_input_embd::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   &&   embd->ne[1] == params.ubatch.n_tokens);
    res &= (!mixed_tokens) || mixed_tokens->ne[0] == llm_graph_n_tok_rows(params.ubatch);
    res &= (!mixed_embd)   || mixed_embd->ne[1]   == params.ubatch.n_tokens;
    res &= (!scale_rows) || scale_rows->ne[1] == params.ubatch.n_tokens;

    return res;
}

void llm_graph_input_embd_h::set_input(const llama_ubatch * ubatch) {
    ASSERT_EMBD_OR_TOKEN(*ubatch);

    const int64_t n_tokens = ubatch->n_tokens;

    if (ubatch->token) {
        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    } else {
        // note: mtmd embedding input goes through here
        GGML_ASSERT(ubatch->embd);
        GGML_ASSERT(n_embd == embd->ne[0]);

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(embd));
    }

    GGML_ASSERT(ubatch->embd_state && "this graph requires a state embedding, see llama_batch_ext_set_embd_state()");
    GGML_ASSERT(n_embd_state == h->ne[0]);

    ggml_backend_tensor_set(h, ubatch->embd_state, 0, n_tokens*n_embd_state*ggml_element_size(h));
}

bool llm_graph_input_embd_h::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token)      || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)       || (embd   && embd->ne[1]   == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd_state) || (h      && h->ne[1]      == params.ubatch.n_tokens);

    return res;
}

void llm_graph_input_pos::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && pos) {
        const int64_t n_tokens = ubatch->n_tokens;

        const bool has_embd = ubatch->is_mixed() || ubatch->token == nullptr;
        if (rope_section_order == LLAMA_ROPE_SECTION_ORDER_TYXZ || !has_embd) {
            ggml_backend_tensor_set(pos, ubatch->pos, 0, n_tokens*n_pos_per_embd*ggml_element_size(pos));
            return;
        }

        // input is always [t, y, x, z]
        // token entries are expanded by the batch to [p, p, p, 0]
        GGML_ASSERT(n_pos_per_embd == 4);

        // slot index per section, slots are t = 0, y = 1, x = 2, z = 3
        std::array<int64_t, 4> slot_of_section = { 0, 1, 2, 3 };
        switch (rope_section_order) {
            case LLAMA_ROPE_SECTION_ORDER_TYXZ: break;
            case LLAMA_ROPE_SECTION_ORDER_ZYXT: slot_of_section = { 3, 1, 2, 0 }; break;
            default: GGML_ABORT("unsupported rope section order");
        }

        std::vector<llama_pos> pos_data(n_tokens*n_pos_per_embd);
        for (int64_t i = 0; i < n_tokens; ++i) {
            const bool is_embd = ubatch->is_mixed() ? ubatch->type[i] != 0 : true;
            for (int64_t s = 0; s < 4; ++s) {
                const int64_t slot = is_embd ? slot_of_section[s] : s;
                pos_data[s*n_tokens + i] = ubatch->pos[slot*n_tokens + i];
            }
        }
        ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size()*ggml_element_size(pos));
    }
}

bool llm_graph_input_pos::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= pos->ne[0] == params.ubatch.n_tokens*n_pos_per_embd;

    return res;
}

void llm_graph_input_attn_temp::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && attn_scale) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(f_attn_temp_scale != 0.0f);
        GGML_ASSERT(n_attn_temp_floor_scale != 0);

        std::vector<float> attn_scale_data(n_tokens, 0.0f);
        for (int i = 0; i < n_tokens; ++i) {
            const float pos = ubatch->pos[i];
            attn_scale_data[i] = std::log(
                std::floor((pos + f_attn_temp_offset) / n_attn_temp_floor_scale) + 1.0
            ) * f_attn_temp_scale + 1.0;
        }

        ggml_backend_tensor_set(attn_scale, attn_scale_data.data(), 0, n_tokens*ggml_element_size(attn_scale));
    }
}

void llm_graph_input_pos_bucket::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(pos_bucket->buffer));
        GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

        int32_t * data = (int32_t *) pos_bucket->data;

        for (int j = 0; j < n_tokens; ++j) {
            for (int i = 0; i < n_tokens; ++i) {
                data[j*n_tokens + i] = llama_relative_position_bucket(ubatch->pos[i], ubatch->pos[j], hparams.n_rel_attn_bkts, true);
            }
        }
    }
}

void llm_graph_input_pos_bucket_kv::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        mctx->set_input_pos_bucket(pos_bucket, ubatch);
    }
}

void llm_graph_input_out_ids::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(out_ids);

    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(out_ids->buffer));
    int32_t * data = (int32_t *) out_ids->data;

    if (n_outputs == n_tokens) {
        for (int i = 0; i < n_tokens; ++i) {
            data[i] = i;
        }

        return;
    }

    GGML_ASSERT(ubatch->output);

    int n_outputs = 0;

    for (int i = 0; i < n_tokens; ++i) {
        if (ubatch->output[i]) {
            data[n_outputs++] = i;
        }
    }
}

bool llm_graph_input_out_ids::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= n_outputs == params.n_outputs;

    return res;
}

void llm_graph_input_mean::set_input(const llama_ubatch * ubatch) {
    if (cparams.embeddings   &&
       (cparams.pooling_type == LLAMA_POOLING_TYPE_MEAN ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK )) {

        const int64_t n_tokens     = ubatch->n_tokens;
        const int64_t n_seq_tokens = ubatch->n_seq_tokens;
        const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

        GGML_ASSERT(mean);
        GGML_ASSERT(ggml_backend_buffer_is_host(mean->buffer));

        float * data = (float *) mean->data;
        memset(mean->data, 0, n_tokens*n_seqs_unq*ggml_element_size(mean));

        std::vector<uint64_t> sums(n_seqs_unq, 0);
        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                sums[seq_idx] += ubatch->n_seq_tokens;
            }
        }

        std::vector<float> div(n_seqs_unq, 0.0f);
        for (int s = 0; s < n_seqs_unq; ++s) {
            const uint64_t sum = sums[s];
            if (sum > 0) {
                div[s] = 1.0f/float(sum);
            }
        }

        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                for (int j = 0; j < n_seq_tokens; ++j) {
                    data[seq_idx*n_tokens + i + j] = div[seq_idx];
                }
            }
        }
    }
}

void llm_graph_input_cls::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens     = ubatch->n_tokens;
    const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

    if (cparams.embeddings && (
        cparams.pooling_type == LLAMA_POOLING_TYPE_CLS  ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_LAST
    )) {
        GGML_ASSERT(cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(cls->buffer));

        uint32_t * data = (uint32_t *) cls->data;
        memset(cls->data, 0, n_seqs_unq*ggml_element_size(cls));

        std::vector<int> target_pos(n_seqs_unq, -1);
        std::vector<int> target_row(n_seqs_unq, -1);

        const bool last = (
             cparams.pooling_type == LLAMA_POOLING_TYPE_LAST ||
            (cparams.pooling_type == LLAMA_POOLING_TYPE_RANK && cparams.causal_attn)
        );

        for (int i = 0; i < n_tokens; ++i) {
            const llama_pos pos = ubatch->pos[i];

            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                if (
                    (target_pos[seq_idx] == -1) ||
                    ( last && pos >= target_pos[seq_idx]) ||
                    (!last && pos <  target_pos[seq_idx])
                ) {
                    target_pos[seq_idx] = pos;
                    target_row[seq_idx] = i;
                }
            }
        }

        for (int s = 0; s < n_seqs_unq; ++s) {
            if (target_row[s] >= 0) {
                data[s] = target_row[s];
            }
        }
    }
}

void llm_graph_input_rs::set_history(const llama_memory_recurrent_context * mctx_cur) {
    // Other recurrent architectures may not consume the DeltaNet history input.
    // Unused graph inputs have no allocated backend buffer.
    if (!s_history || !s_history->buffer) {
        return;
    }
    GGML_ASSERT(ggml_backend_buffer_is_host(s_history->buffer));
    const int64_t n_seqs = mctx_cur->get_ubatch().n_seqs;
    const int64_t n_history = s_history->ne[0] / n_seqs;
    int32_t * data = (int32_t *) s_history->data;
    for (int64_t age = 0; age < n_history; ++age) {
        for (int64_t seq = 0; seq < n_seqs; ++seq) {
            data[age*n_seqs + seq] = mctx_cur->s_history(seq, age);
        }
    }
}

bool llm_graph_input_rs::can_reuse_history(const llm_graph_params & params) const {
    const int64_t n_history = std::max<int64_t>(0,
            (int64_t) params.cparams.n_rs_seq + 1 - params.ubatch.n_seq_tokens);
    return s_history ? s_history->ne[0] == n_history*params.ubatch.n_seqs : n_history == 0;
}

void llm_graph_input_rs::set_input_rs(const llama_memory_recurrent_context * mctx_cur, const llama_ubatch * ubatch) {
    set_history(mctx_cur); // s_copy consumes the pending rollback index
    const int64_t n_rs = mctx_cur->get_n_rs();

    if (s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(s_copy->buffer));
        int32_t * data = (int32_t *) s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx_cur->s_copy(i);
        }
    }

    if (s_write_rows) {
        GGML_ASSERT(ggml_backend_buffer_is_host(s_write_rows->buffer));
        int64_t * data = (int64_t *) s_write_rows->data;

        // destination cache row of (snapshot slot s, seq i) = s*mem_size + head + i,
        // matching the strided-cpy destination of the legacy gathered path
        const uint32_t mem_size = mctx_cur->get_size();
        const uint32_t rs_head  = mctx_cur->get_head();
        const int64_t  n_seqs   = ubatch->n_seqs;

        for (int64_t r = 0; r < s_write_rows->ne[0]; ++r) {
            data[r] = (r / n_seqs) * (int64_t) mem_size + (int64_t) rs_head + (r % n_seqs);
        }
    }
}

bool llm_graph_input_rs::can_reuse_rs(const llama_memory_recurrent_context * mctx_cur, const llm_graph_params & params) {
    bool res = true;

    res &= s_copy->ne[0] == mctx_cur->get_n_rs();

    res &= s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= s_copy_extra->ne[0] == mctx_cur->get_n_rs() - params.ubatch.n_seqs;

    if (s_write_rows) {
        res &= s_write_rows->ne[0] ==
            std::min<int64_t>(params.ubatch.n_seq_tokens, s_write_K) * params.ubatch.n_seqs;
    }

    res &= head == mctx_cur->get_head();
    res &= rs_z == mctx_cur->get_rs_z();
    res &= can_reuse_history(params);

    return res;
}

void llm_graph_input_rs::set_input(const llama_ubatch * ubatch) {
    set_input_rs(mctx, ubatch);
}

bool llm_graph_input_rs::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_recurrent_context *>(params.mctx);

    this->mctx = mctx;

    return can_reuse_rs(mctx, params);
}

void llm_graph_input_cross_embd::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (cross_embd && !cross->v_embd.empty()) {
        assert(cross_embd->type == GGML_TYPE_F32);

        ggml_backend_tensor_set(cross_embd, cross->v_embd.data(), 0, ggml_nbytes(cross_embd));
    }
}

template <typename T>
static void print_mask(const T * data, int64_t n_tokens, int64_t n_kv, int64_t n_swa, llama_swa_type swa_type) {
    LLAMA_LOG_DEBUG("%s: === Attention mask ===\n", __func__);
    const char * swa_type_str = "unknown";

    switch (swa_type) {
        case LLAMA_SWA_TYPE_NONE:      swa_type_str = "LLAMA_SWA_TYPE_NONE"; break;
        case LLAMA_SWA_TYPE_STANDARD:  swa_type_str = "LLAMA_SWA_TYPE_STANDARD"; break;
        case LLAMA_SWA_TYPE_CHUNKED:   swa_type_str = "LLAMA_SWA_TYPE_CHUNKED"; break;
        case LLAMA_SWA_TYPE_SYMMETRIC: swa_type_str = "LLAMA_SWA_TYPE_SYMMETRIC"; break;
    };

    LLAMA_LOG_DEBUG("%s: n_swa : %d, n_kv: %d, swa_type: %s\n", __func__, (int)n_swa, (int)n_kv, swa_type_str);
    LLAMA_LOG_DEBUG("%s: '0' = can attend, '∞' = masked\n", __func__);
    LLAMA_LOG_DEBUG("%s: Rows = query tokens, Columns = key/value tokens\n\n", __func__);

    LLAMA_LOG_DEBUG("    ");
    for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
        LLAMA_LOG_DEBUG("%2d", j);
    }
    LLAMA_LOG_DEBUG("\n");

    for (int i = 0; i < std::min((int64_t)20, n_tokens); ++i) {
        LLAMA_LOG_DEBUG(" %2d ", i);
        for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
            float val = llama_cast<float>(data[i * n_kv + j]);
            if (val == -INFINITY) {
                LLAMA_LOG_DEBUG(" ∞");
            } else {
                LLAMA_LOG_DEBUG(" 0");
            }
        }
        LLAMA_LOG_DEBUG("\n");
    }
}

void llm_graph_input_attn_no_cache::set_input(const llama_ubatch * ubatch) {
    const int64_t n_kv     = self_kq_mask->ne[0];
    const int64_t n_tokens = ubatch->n_tokens;

    const auto fill_mask = [&](auto * data, int64_t ne, int n_swa, llama_swa_type swa_type) {
        using T = std::remove_reference_t<decltype(*data)>;
        std::fill(data, data + ne, llama_cast<T>(-INFINITY));

        for (int i1 = 0; i1 < n_tokens; ++i1) {
            const llama_seq_id s1 = ubatch->seq_id[i1][0];
            const llama_pos    p1 = ubatch->pos[i1];

            const uint64_t idst = i1*n_kv;

            for (int i0 = 0; i0 < n_tokens; ++i0) {
                const llama_seq_id s0 = ubatch->seq_id[i0][0];
                const llama_pos p0    = ubatch->pos[i0];

                // mask different sequences
                if (s0 != s1) {
                    continue;
                }

                // mask future tokens
                if (cparams.causal_attn && p0 > p1) {
                    continue;
                }

                // apply SWA if any
                if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                    continue;
                }

                data[idst + i0] = llama_cast<T>(hparams.use_alibi ? -std::abs(p0 - p1) : 0.0f);
            }
        }

        if (debug) {
            print_mask(data, n_tokens, n_kv, n_swa, swa_type);
        }
    };

    GGML_ASSERT(self_kq_mask);
    GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask->buffer));
    if (self_kq_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) self_kq_mask->data, ggml_nelements(self_kq_mask), 0, LLAMA_SWA_TYPE_NONE);
    } else {
        fill_mask((float       *) self_kq_mask->data, ggml_nelements(self_kq_mask), 0, LLAMA_SWA_TYPE_NONE);
    }

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        GGML_ASSERT(self_kq_mask_swa);
        GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask_swa->buffer));
        if (self_kq_mask_swa->type == GGML_TYPE_F16) {
            fill_mask((ggml_fp16_t *) self_kq_mask_swa->data, ggml_nelements(self_kq_mask_swa), hparams.n_swa, hparams.swa_type);
        } else {
            fill_mask((float       *) self_kq_mask_swa->data, ggml_nelements(self_kq_mask_swa), hparams.n_swa, hparams.swa_type);
        }
    }
}

static bool tail_query_plan_has_explicit_tokens(const llama_ubatch & ubatch) {
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id[i] <= 0 || ubatch.seq_id[i] == nullptr) {
            return false;
        }
    }
    return true;
}

static llama_seq_id tail_query_plan_primary_seq(
        const llama_ubatch & ubatch, uint32_t token, bool explicit_tokens) {
    if (explicit_tokens) {
        return ubatch.seq_id[token][0];
    }
    // graph_reserve() deliberately creates a lightweight equal-sequence
    // ubatch: only its synthetic sequence descriptors are initialized. The
    // token layout is still the normal stream-major equal-sequence layout.
    GGML_ASSERT(ubatch.equal_seqs() && ubatch.n_seq_tokens > 0 && ubatch.seq_id_unq != nullptr);
    const uint32_t stream = token/ubatch.n_seq_tokens;
    GGML_ASSERT(stream < ubatch.n_seqs_unq);
    return ubatch.seq_id_unq[stream];
}


// P-mode-aware wrapper over the shared helper in llama-position-split.h.
template <typename Ctx>
static int64_t tail_extent_queries(const Ctx * mctx, const llama_ubatch & ub, int64_t q_max) {
    const auto * kvarn = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx);
    if (!kvarn || !kvarn->position_split_enabled()) {
        return q_max;
    }
    const auto * cache = kvarn->get_kvarn_cache();
    const uint32_t p = cache ? cache->position_split_p() : 0;
    return llama_position_split::tail_extent_queries(kvarn->position_split_enabled(), p,
            ub.pos, ub.n_tokens, q_max);
}

static std::pair<int64_t, int64_t> tail_query_plan_shape(const llama_ubatch & ubatch) {
    const bool explicit_tokens = tail_query_plan_has_explicit_tokens(ubatch);
    std::map<llama_seq_id, int64_t> counts;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        ++counts[tail_query_plan_primary_seq(ubatch, i, explicit_tokens)];
    }
    int64_t q_max = 0;
    for (const auto & entry : counts) {
        q_max = std::max(q_max, entry.second);
    }
    return { q_max, int64_t(counts.size()) };
}

static void set_tail_query_plan(
        ggml_tensor * query_order, ggml_tensor * run_desc, const llama_ubatch & ubatch) {
    if (!query_order || !run_desc || !query_order->buffer || !run_desc->buffer) {
        return;
    }
    GGML_ASSERT(ggml_backend_buffer_is_host(query_order->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(run_desc->buffer));
    GGML_ASSERT(query_order->type == GGML_TYPE_I32 && run_desc->type == GGML_TYPE_I32 && run_desc->ne[0] >= 4);
    const bool explicit_tokens = tail_query_plan_has_explicit_tokens(ubatch);
    GGML_ASSERT(explicit_tokens);
    std::map<llama_seq_id, std::vector<int32_t>> queries;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        queries[tail_query_plan_primary_seq(ubatch, i, explicit_tokens)].push_back(int32_t(i));
    }
    GGML_ASSERT(query_order->ne[1] == int64_t(queries.size()) && run_desc->ne[1] == int64_t(queries.size()));
    auto * order = static_cast<int32_t *>(query_order->data);
    auto * desc = static_cast<int32_t *>(run_desc->data);
    std::fill(order, order + ggml_nelements(query_order), -1);
    std::fill(desc, desc + ggml_nelements(run_desc), -1);
    const int64_t desc_stride = run_desc->ne[0];
    std::vector<std::pair<llama_seq_id, int64_t>> layout;
    layout.reserve(queries.size());
    int64_t active = 0;
    for (const auto & entry : queries) {
        GGML_ASSERT(int64_t(entry.second.size()) <= query_order->ne[0]);
        std::copy(entry.second.begin(), entry.second.end(), order + active*query_order->ne[0]);
        layout.emplace_back(entry.first, int64_t(entry.second.size()));
        ++active;
    }
    for (int64_t ia = 0; ia < int64_t(layout.size()); ++ia) {
        int64_t run = 1;
        while (ia + run < int64_t(layout.size()) &&
                layout[ia + run].first == layout[ia].first + run &&
                layout[ia + run].second == layout[ia].second) {
            ++run;
        }
        desc[desc_stride*ia + 0] = layout[ia].first;
        desc[desc_stride*ia + 1] = int32_t(ia*query_order->ne[0]);
        desc[desc_stride*ia + 2] = int32_t(layout[ia].second);
        desc[desc_stride*ia + 3] = int32_t(run);
    }
}

llm_graph_kv_tail_identity llm_graph_kv_tail_identity::capture(
        const llama_hparams & hparams,
        const llama_kv_cache_context * mctx) {
    GGML_ASSERT(mctx != nullptr);
    llm_graph_kv_tail_identity result;
    result.storage_kind = uint32_t(mctx->get_tail_storage_kind());
    result.exact_type = mctx->get_tail_type();
    result.retention_tokens = mctx->get_tail_tokens();
    result.rollback_tokens = mctx->get_tail_rollback_tokens();
    result.arena_stride = mctx->get_tail_arena_stride();
    result.storage_slots = mctx->get_tail_slots();
    result.compact = mctx->has_compact_tail();
    if (result.storage_kind == LLAMA_KV_TAIL_STORAGE_DISABLED) {
        return result;
    }
    result.has_body.reserve(hparams.n_layer());
    result.has_current.reserve(hparams.n_layer());
    result.body_execution_rows.reserve(hparams.n_layer());
    result.routes.reserve(hparams.n_layer());
    result.explicit_bias.reserve(hparams.n_layer());
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        const auto route = mctx->get_tail_route(int32_t(il));
        result.routes.push_back(route);
        result.has_body.push_back(route == LLAMA_KV_TAIL_ROUTE_NONE || mctx->has_kv_body(int32_t(il)));
        result.has_current.push_back(
                route != LLAMA_KV_TAIL_ROUTE_NONE && mctx->has_tail_current(int32_t(il)));
        result.body_execution_rows.push_back(
                route == LLAMA_KV_TAIL_ROUTE_NONE ? 0 : mctx->get_tail_body_execution_rows(int32_t(il)));
        result.explicit_bias.push_back(
                route != LLAMA_KV_TAIL_ROUTE_NONE && mctx->get_tail_explicit_bias(int32_t(il)));
    }
    return result;
}

bool llm_graph_kv_tail_identity::matches(
        const llama_hparams & hparams,
        const llama_kv_cache_context * mctx) const {
    if (mctx == nullptr || storage_kind != uint32_t(mctx->get_tail_storage_kind()) ||
            exact_type != mctx->get_tail_type() ||
            retention_tokens != mctx->get_tail_tokens() ||
            rollback_tokens != mctx->get_tail_rollback_tokens() ||
            arena_stride != mctx->get_tail_arena_stride() ||
            storage_slots != mctx->get_tail_slots() ||
            compact != mctx->has_compact_tail()) {
        return false;
    }
    if (storage_kind == LLAMA_KV_TAIL_STORAGE_DISABLED) {
        return true;
    }
    if (has_body.size() != hparams.n_layer() ||
            has_current.size() != hparams.n_layer() ||
            body_execution_rows.size() != hparams.n_layer() ||
            routes.size() != hparams.n_layer() || explicit_bias.size() != hparams.n_layer()) {
        return false;
    }
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        const auto route = mctx->get_tail_route(int32_t(il));
        const bool current_has_body = route == LLAMA_KV_TAIL_ROUTE_NONE || mctx->has_kv_body(int32_t(il));
        const bool current_has_current =
                route != LLAMA_KV_TAIL_ROUTE_NONE && mctx->has_tail_current(int32_t(il));
        const uint32_t current_body_rows =
                route == LLAMA_KV_TAIL_ROUTE_NONE ? 0 : mctx->get_tail_body_execution_rows(int32_t(il));
        const bool current_explicit_bias =
                route != LLAMA_KV_TAIL_ROUTE_NONE && mctx->get_tail_explicit_bias(int32_t(il));
        if (has_body[il] != current_has_body || has_current[il] != current_has_current ||
                body_execution_rows[il] != current_body_rows ||
                routes[il] != route || explicit_bias[il] != current_explicit_bias) {
            return false;
        }
    }
    return true;
}

llm_graph_input_attn_kv_iswa::llm_graph_input_attn_kv_iswa(
        const llama_hparams & hparams,
        const llama_cparams & cparams,
        const llama_kv_cache_iswa_context * mctx) :
    hparams(hparams),
    cparams(cparams),
    mctx(mctx),
    base_tail_identity(llm_graph_kv_tail_identity::capture(hparams, mctx->get_base())),
    swa_tail_identity(llm_graph_kv_tail_identity::capture(hparams, mctx->get_swa())) {
}

void llm_graph_input_attn_kv::set_input(const llama_ubatch * ubatch) {
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->set_input_k_idxs(self_k_idxs, ubatch);
    }
    if (self_v_idxs && self_v_idxs->buffer) {
        mctx->set_input_v_idxs(self_v_idxs, ubatch);
    }
    // The tail WRITE index is consumed only by the local tail store. A
    // position-split overflow ubatch stores Q4 overflow keys and never writes
    // the local tail, so nothing references this input, the scheduler prunes it
    // and it arrives here without a buffer -- the same contract already used
    // for self_kq_mask and self_kvarn_mat_idxs above. An allocated input that
    // is not host memory still fails inside ggml_backend_buffer_is_host().
    if (self_tail_idxs && self_tail_idxs->buffer) {
        mctx->set_input_tail_idxs(self_tail_idxs, ubatch);
    }
    set_tail_query_plan(self_tail_query_order, self_tail_run_desc, *ubatch);
    if (self_kvarn_mat_idxs && self_kvarn_mat_idxs->buffer) {
        const auto * kvarn = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx);
        GGML_ASSERT(kvarn != nullptr);
        kvarn->set_input_kvarn_mat_idxs(self_kvarn_mat_idxs, ubatch);
    }

    // the mask is left unallocated when the graph only stores K/V without attending
    // (e.g. DFlash's KV-injection pass)
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
        mctx->set_input_kq_mask_tail(self_kq_mask, self_kq_mask_tail,
                self_tail_read_idxs, self_tail_body_read_idxs, self_tail_bias_read_idxs,
                ubatch, cparams.causal_attn);
        mctx->set_input_tail_body_plan(
                self_tail_query_order, self_tail_run_desc, self_kq_mask,
                ubatch, cparams.causal_attn);
    }

    if (self_kq_mask_standard && self_kq_mask_standard->buffer) {
        const auto * kvarn = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx);
        GGML_ASSERT(kvarn && kvarn->has_standard_cache());
        kvarn->set_standard_input_kq_mask(self_kq_mask_standard, ubatch, cparams.causal_attn);
    }

    // F3 position-split range inputs. Recompute live cells per batch (history
    // grows); sizes must match the built tensors (can_reuse rebuilds on
    // change), otherwise fail closed rather than filling a stale layout.
    // Masks/idxs are filled via backend copies so tensor placement (host or
    // device) never matters.
    if ((self_kq_mask_local && self_kq_mask_local->buffer) ||
            (self_kq_mask_overflow && self_kq_mask_overflow->buffer) ||
            (self_kvarn_mat_idxs_local && self_kvarn_mat_idxs_local->buffer) ||
            (self_overflow_cell_idxs && self_overflow_cell_idxs->buffer) ||
            (self_ovf_view_rows && self_ovf_view_rows->buffer)) {
        const auto * kvarn = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx);
        GGML_ASSERT(kvarn && kvarn->position_split_enabled());
        const auto * kvarn_cache = kvarn->get_kvarn_cache();
        GGML_ASSERT(kvarn_cache != nullptr);
        const auto * standard_cache = kvarn_cache->get_standard_cache();
        if (!standard_cache) {
            throw std::runtime_error("position-split overflow indices require the standard Qx cache");
        }
        const uint32_t payload_origin = standard_cache->get_payload_row_origin();
        const uint32_t payload_size = standard_cache->get_payload_size();
        std::vector<uint32_t> local_cells, overflow_cells;
        GGML_ASSERT(kvarn_cache->position_split_live_cells(
                    llama_position_split::range_kind::LOCAL, local_cells));
        GGML_ASSERT(kvarn_cache->position_split_live_cells(
                    llama_position_split::range_kind::OVERFLOW, overflow_cells));
        split_local_cells = std::move(local_cells);
        split_overflow_cells = std::move(overflow_cells);
        const auto * metadata = kvarn_cache->get_metadata_cache();
        GGML_ASSERT(metadata != nullptr);
        // Range masks: temporal causal rule over absolute positions. P-mode
        // is single-sequence, non-SWA, non-Alibi (validated at cache
        // creation); fail closed otherwise. M-RoPE spatial tie-breaks are
        // unnecessary for text batches with distinct temporal positions.
        if (hparams.use_alibi) {
            throw std::runtime_error("position-split range masks do not support Alibi bias");
        }
        auto fill_range_mask = [&](ggml_tensor * mask, const std::vector<uint32_t> & cells, const char * what) {
            GGML_ASSERT(mask && mask->buffer);
            const int64_t n_cap = mask->ne[0];
            GGML_ASSERT(int64_t(cells.size()) <= n_cap);
            const int64_t n_q = mask->ne[1];
            GGML_ASSERT(int64_t(ubatch->n_tokens) == n_q);
            GGML_ASSERT(ubatch->n_pos >= 1 && ubatch->pos != nullptr);
            std::vector<float> host(size_t(n_cap*n_q), -INFINITY);
            const auto & all_cells = metadata->get_cells(0);
            for (int64_t qi = 0; qi < n_q; ++qi) {
                const llama_pos p_q = ubatch->pos[qi];
                for (size_t ki = 0; ki < cells.size(); ++ki) {
                    const llama_pos p_k = all_cells.pos_get(cells[ki]);
                    const bool visible = !cparams.causal_attn || (p_k <= p_q);
                    host[size_t(qi*n_cap + ki)] = visible ? 0.0f : -INFINITY;
                }
                // Beyond live cells: fully masked (empty).
            }
            if (mask->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> host16(host.size());
                for (size_t i = 0; i < host.size(); ++i) {
                    host16[i] = ggml_fp32_to_fp16(host[i]);
                }
                ggml_backend_tensor_set(mask, host16.data(), 0, host16.size()*sizeof(ggml_fp16_t));
            } else if (mask->type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(mask, host.data(), 0, host.size()*sizeof(float));
            } else {
                throw std::runtime_error(format(
                        "position-split %s mask has unsupported type %s",
                        what, ggml_type_name(mask->type)));
            }
            (void) what;
        };
        if (self_kq_mask_local && self_kq_mask_local->buffer) {
            fill_range_mask(self_kq_mask_local, split_local_cells, "local");
            if (self_kq_mask_tail && self_kq_mask_tail->buffer) {
                // Dual attention consumes the local range mask, so the general
                // self_kq_mask can be pruned. Its allocation must not decide
                // whether the shared exact-tail inputs/descriptor are filled.
                std::vector<int64_t> local_read_cells(size_t(self_kq_mask_local->ne[0]), -1);
                for (size_t i = 0; i < split_local_cells.size(); ++i) {
                    local_read_cells[i] = split_local_cells[i];
                }
                metadata->set_input_kq_mask_tail_mapped(
                        self_kq_mask_local, self_kq_mask_tail,
                        self_tail_read_idxs, self_tail_body_read_idxs, self_tail_bias_read_idxs,
                        ubatch, cparams.causal_attn, local_read_cells);
                metadata->set_input_tail_body_plan(
                        self_tail_query_order, self_tail_run_desc, self_kq_mask_local,
                        ubatch, cparams.causal_attn);
            }
        }
        if (self_kq_mask_overflow && self_kq_mask_overflow->buffer) {
            fill_range_mask(self_kq_mask_overflow, split_overflow_cells, "overflow");
        }
        if (self_kvarn_mat_idxs_local && self_kvarn_mat_idxs_local->buffer) {
            GGML_ASSERT(int64_t(split_local_cells.size()) <= self_kvarn_mat_idxs_local->ne[0]);
            kvarn->set_input_kvarn_mat_idxs_local(self_kvarn_mat_idxs_local, split_local_cells);
        }
        if (self_overflow_cell_idxs && self_overflow_cell_idxs->buffer) {
            GGML_ASSERT(int64_t(split_overflow_cells.size()) <= self_overflow_cell_idxs->ne[0]);
            GGML_ASSERT(self_overflow_cell_idxs->type == GGML_TYPE_I32);
            std::vector<int32_t> host_cells(size_t(self_overflow_cell_idxs->ne[0]),
                    cparams.remote_attn_position_split ? -1 : 0);
            for (size_t i = 0; i < split_overflow_cells.size(); ++i) {
                const uint32_t global_cell = split_overflow_cells[i];
                if (global_cell < kvarn_cache->position_split_p() ||
                        global_cell >= kvarn_cache->get_kv_size()) {
                    throw std::runtime_error(format(
                            "position-split overflow cell %u is outside global range [P=%u,C=%u)",
                            global_cell, kvarn_cache->position_split_p(), kvarn_cache->get_kv_size()));
                }
                // The metadata and range mask stay in global cell coordinates. Qx
                // GET_ROWS consumes the companion's physical payload row instead.
                // The cache owns this mapping because compact v2 payloads start at P
                // while legacy standard rows still start at zero.
                const uint32_t physical_row = standard_cache->payload_row_for_cell(global_cell);
                if (physical_row >= payload_size || physical_row > uint32_t(INT32_MAX)) {
                    throw std::runtime_error(format(
                            "position-split overflow cell %u maps outside Qx payload origin=%u size=%u (row=%u)",
                            global_cell, payload_origin, payload_size, physical_row));
                }
                if (cparams.remote_attn_position_split) {
                    const auto expected = kvarn_cache->position_split_p() + uint32_t(i);
                    if (global_cell != expected || physical_row != i ||
                            metadata->get_cells(0).pos_get(global_cell) != llama_pos(expected)) {
                        throw std::runtime_error("RKVA range requires a contiguous, unshifted single-sequence overflow");
                    }
                    host_cells[i] = int32_t(global_cell);
                } else {
                    host_cells[i] = int32_t(physical_row);
                }
            }
            if (std::getenv("F4_ALLOC_TRACE") != nullptr) {
                // Ground truth for the row-mapping question: what row indices does the
                // gather actually reads (global cell IDs and physical payload rows),
                // and are both mappings contiguous ascending runs?
                const size_t n = split_overflow_cells.size();
                bool global_asc = n > 0;
                bool physical_asc = n > 0;
                for (size_t i = 1; i < n; ++i) {
                    global_asc = global_asc && split_overflow_cells[i] == split_overflow_cells[i-1] + 1;
                    physical_asc = physical_asc && host_cells[i] == host_cells[i-1] + 1;
                }
                const int32_t g0 = n ? int32_t(split_overflow_cells[0]) : -1;
                const int32_t g1 = n > 1 ? int32_t(split_overflow_cells[1]) : -1;
                const int32_t g2 = n > 2 ? int32_t(split_overflow_cells[2]) : -1;
                const int32_t gl = n ? int32_t(split_overflow_cells[n-1]) : -1;
                const int32_t p0 = n ? host_cells[0] : -1;
                const int32_t p1 = n > 1 ? host_cells[1] : -1;
                const int32_t p2 = n > 2 ? host_cells[2] : -1;
                const int32_t pl = n ? host_cells[n-1] : -1;
                fprintf(stderr, "[F5IDX] n=%zu payload_origin=%u payload_size=%u global_first=[%d,%d,%d] global_last=%d physical_first=[%d,%d,%d] physical_last=%d global_ascending=%d physical_ascending=%d\n",
                        n, payload_origin, payload_size, g0, g1, g2, gl, p0, p1, p2, pl,
                        int(global_asc), int(physical_asc));
            }
            ggml_backend_tensor_set(self_overflow_cell_idxs,
                    host_cells.data(), 0, host_cells.size()*sizeof(int32_t));
        }
        if (self_ovf_view_rows && self_ovf_view_rows->buffer) {
            GGML_ASSERT(self_ovf_view_rows->type == GGML_TYPE_I32);
            const int64_t n = self_ovf_view_rows->ne[0];
            std::vector<int32_t> rows;
            rows.resize(size_t(n));
            for (int64_t i = 0; i < n; ++i) {
                rows[size_t(i)] = int32_t(i); // identity over the packed view rows
            }
            ggml_backend_tensor_set(self_ovf_view_rows,
                    rows.data(), 0, rows.size()*sizeof(int32_t));
        }
    }

    if (self_k_rot && self_k_rot->buffer) {
        mctx->set_input_k_rot(self_k_rot);
    }

    if (self_v_rot && self_v_rot->buffer) {
        mctx->set_input_v_rot(self_v_rot);
    }

    if ((self_kvarn_rot_64 && self_kvarn_rot_64->buffer) ||
            (self_kvarn_rot_128 && self_kvarn_rot_128->buffer) ||
            (self_kvarn_rot_256 && self_kvarn_rot_256->buffer) ||
            (self_kvarn_rot_512 && self_kvarn_rot_512->buffer)) {
        const auto * kvarn = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx);
        GGML_ASSERT(kvarn != nullptr);
        llm_kvarn_set_rot_inputs(kvarn,
                self_kvarn_rot_64, self_kvarn_rot_128, self_kvarn_rot_256, self_kvarn_rot_512);
    }
}

bool llm_graph_input_attn_kv::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;
    res &= tail_identity.matches(hparams, mctx);
    const auto [q_max, n_active] = tail_query_plan_shape(params.ubatch);
    res &= self_tail_query_order == nullptr ||
            (self_tail_query_order->ne[0] == q_max && self_tail_query_order->ne[1] == n_active);
    const uint32_t tail_attention_stride = mctx->get_tail_attention_stride(
            uint32_t(tail_extent_queries(mctx, params.ubatch, q_max)));
    const int64_t tail_desc_stride = 6 + tail_attention_stride +
            (mctx->can_pack_tail_body(params.ubatch) ? mctx->get_tail_body_execution_stride() : 0);
    res &= self_tail_run_desc == nullptr ||
            (self_tail_run_desc->ne[0] == tail_desc_stride && self_tail_run_desc->ne[1] == n_active);

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
    res &= self_tail_idxs == nullptr || self_tail_idxs->ne[0] == params.ubatch.n_tokens;
    res &= self_kvarn_mat_idxs == nullptr || self_kvarn_mat_idxs->ne[0] == mctx->get_n_kv();
    res &= self_tail_read_idxs == nullptr ||
            (self_tail_read_idxs->ne[0] == tail_attention_stride &&
             self_tail_read_idxs->ne[1] == params.ubatch.n_tokens);
    res &= self_tail_body_read_idxs == nullptr ||
            (self_tail_body_read_idxs->ne[0] == tail_attention_stride &&
             self_tail_body_read_idxs->ne[1] == params.ubatch.n_tokens);
    res &= self_tail_bias_read_idxs == nullptr ||
            (self_tail_bias_read_idxs->ne[0] == tail_attention_stride &&
             self_tail_bias_read_idxs->ne[1] == params.ubatch.n_tokens);
  //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_kq_mask(self_kq_mask, mctx, params.ubatch, params.cparams);
    res &= self_kq_mask_standard == nullptr ||
            can_reuse_kq_mask(self_kq_mask_standard, mctx, params.ubatch, params.cparams);
    // F3: split range inputs are capacity-sized (P and C-P), so history
    // growth never forces a rebuild. Only the query count must match; refresh
    // the stored cell lists so set_input fills fresh data.
    if (self_kq_mask_local || self_kq_mask_overflow ||
            self_kvarn_mat_idxs_local || self_overflow_cell_idxs) {
        const auto * kvarn = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx);
        bool split_ok = kvarn && kvarn->position_split_enabled();
        std::vector<uint32_t> local_cells, overflow_cells;
        if (split_ok) {
            const auto * kvarn_cache = kvarn->get_kvarn_cache();
            split_ok = kvarn_cache &&
                kvarn_cache->position_split_live_cells(
                    llama_position_split::range_kind::LOCAL, local_cells) &&
                kvarn_cache->position_split_live_cells(
                    llama_position_split::range_kind::OVERFLOW, overflow_cells);
        }
        const int64_t n_q = params.ubatch.n_tokens;
        split_ok = split_ok &&
            (self_kq_mask_local == nullptr || self_kq_mask_local->ne[1] == n_q) &&
            (self_kq_mask_overflow == nullptr || self_kq_mask_overflow->ne[1] == n_q);
        // The graph may carry a packed Q4 view of the overflow storage, which is only
        // valid while the live run keeps the same row, extent and order. An append inside
        // existing padding changes the live extent WITHOUT changing n_kv or the query
        // count, so the size-only checks above cannot see it.
        split_ok = split_ok && split_packed_map_still_valid(
                static_cast<const llama_kv_cache_context *>(
                    dynamic_cast<const llama_memory_context_i *>(mctx)),
                split_ovf_packed, split_ovf_row0, split_ovf_n_rows, "attn");
        if (split_ok) {
            split_local_cells = std::move(local_cells);
            split_overflow_cells = std::move(overflow_cells);
        }
        res &= split_ok;
    }
    res &= self_kq_mask_tail == nullptr ||
            (self_kq_mask_tail->ne[0] == tail_attention_stride &&
             self_kq_mask_tail->ne[1] == self_kq_mask->ne[1] &&
             self_kq_mask_tail->ne[3] == self_kq_mask->ne[3]);

    if (self_kvarn_mat_idxs && self_kvarn_mat_idxs->buffer) {
        const auto * kvarn = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx);
        GGML_ASSERT(kvarn != nullptr);
        kvarn->set_mat_idxs(self_kvarn_mat_idxs);
    }

    return res;
}

void llm_graph_input_attn_kv_paged::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(ubatch != nullptr);

    if (paged_write_slots) {
        ggml_backend_tensor_set(paged_write_slots, mctx->get_write_slots(), 0, ggml_nbytes(paged_write_slots));
        last_n_tokens = paged_write_slots->ne[0];
    }
    if (paged_block_table) {
        ggml_backend_tensor_set(paged_block_table, mctx->get_block_table(), 0, ggml_nbytes(paged_block_table));
    }
    if (paged_context_lens) {
        ggml_backend_tensor_set(paged_context_lens, mctx->get_context_lens(), 0, ggml_nbytes(paged_context_lens));
    }
    if (paged_batch_offsets) {
        ggml_backend_tensor_set(paged_batch_offsets, mctx->get_batch_offsets(), 0, ggml_nbytes(paged_batch_offsets));
    }
    if (paged_batch_lens) {
        ggml_backend_tensor_set(paged_batch_lens, mctx->get_batch_lens(), 0, ggml_nbytes(paged_batch_lens));
    }

    const int32_t * context_lens = mctx->get_context_lens();
    int32_t active_context = mctx->get_max_blocks() * cparams.block_size;
    if (context_lens) {
        active_context = 0;
        for (int32_t i = 0; i < mctx->get_batch_size(); ++i) {
            active_context = std::max(active_context, context_lens[i]);
        }
    }

    // Encode base+1 for the common single-sequence contiguous table. Zero keeps
    // the fully general paged lookup path.
    int32_t contiguous_base = -1;
    const int32_t * block_table = mctx->get_block_table();
    if (mctx->get_batch_size() == 1 && context_lens && context_lens[0] > 0 && block_table) {
        const int32_t n_blocks = (context_lens[0] + cparams.block_size - 1) / cparams.block_size;
        contiguous_base = block_table[0];
        for (int32_t i = 0; contiguous_base >= 0 && i < n_blocks; ++i) {
            if (block_table[i] != contiguous_base + i) {
                contiguous_base = -1;
            }
        }
    }

    const auto tensor_addr = [](const ggml_tensor * tensor) {
        return tensor != nullptr && tensor->data != nullptr ? (uintptr_t) tensor->data : 0;
    };
    const ggml_tensor * snapkv_scores   = mctx->get_snapkv_scores();
    const uintptr_t snapkv_addr         = tensor_addr(snapkv_scores);
    const uintptr_t snapkv_capture_addr = tensor_addr(mctx->get_snapkv_capture_from());
    const uintptr_t snapkv_slots_addr   = tensor_addr(mctx->get_snapkv_score_slots());
    const uintptr_t snapkv_token_addr   = tensor_addr(mctx->get_snapkv_token_scores());

    for (ggml_tensor * node : paged_attn_nodes) {
        GGML_ASSERT(node->op == GGML_OP_PAGED_ATTN);
        int32_t * op_params_i = (int32_t *) ((float *) node->op_params + 1);
        op_params_i[2] = active_context;
        op_params_i[3] = mctx->get_snapkv_streaming() ? 1 : 0;
        op_params_i[4] = (int32_t) (snapkv_addr & 0xffffffffu);
        op_params_i[5] = (int32_t) (snapkv_addr >> 32);
        op_params_i[6] = (int32_t) (snapkv_capture_addr & 0xffffffffu);
        op_params_i[7] = (int32_t) (snapkv_capture_addr >> 32);
        op_params_i[8] = (int32_t) (snapkv_slots_addr & 0xffffffffu);
        op_params_i[9] = (int32_t) (snapkv_slots_addr >> 32);
        op_params_i[10] = (int32_t) (snapkv_token_addr & 0xffffffffu);
        op_params_i[11] = (int32_t) (snapkv_token_addr >> 32);
        op_params_i[12] = snapkv_scores != nullptr && node->src[0]->ne[1] > 0
            ? (int32_t) (snapkv_scores->ne[0] / node->src[0]->ne[1])
            : mctx->get_max_blocks();
        op_params_i[13] = contiguous_base + 1;
    }
}

bool llm_graph_input_attn_kv_paged::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_paged_context *>(params.mctx);
    this->mctx = mctx;

    return storage_generation == mctx->get_storage_generation() &&
           paged_write_slots->ne[0] == (int64_t) params.ubatch.n_tokens &&
           paged_block_table->ne[0] == mctx->get_max_blocks() &&
           paged_block_table->ne[1] == mctx->get_batch_size();
}

void llm_graph_input_mem_hybrid_paged::set_input(const llama_ubatch * ubatch) {
    inp_attn->set_input(ubatch);

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid_paged::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_paged_context *>(params.mctx);
    this->mctx = mctx;

    auto * paged = get_attn();
    const auto * attn = mctx->get_attn();
    paged->mctx = attn;

    bool res = paged->storage_generation == attn->get_storage_generation();
    res &= paged->paged_write_slots->ne[0] == (int64_t) params.ubatch.n_tokens;
    res &= paged->paged_block_table->ne[0] == attn->get_max_blocks();
    res &= paged->paged_block_table->ne[1] == attn->get_batch_size();

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();
    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;
    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

void llm_graph_input_attn_k::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);

    mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
}

bool llm_graph_input_attn_k::can_reuse(const llm_graph_params & params) {
    mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    return can_reuse_impl(params);
}

bool llm_graph_input_attn_k::can_reuse_impl(const llm_graph_params & params) {
    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(self_kq_mask, mctx, params.ubatch, params.cparams);

    return res;
}

llm_graph_input_attn_kv_msa::llm_graph_input_attn_kv_msa(
        const llama_hparams & hparams,
        const llama_cparams & cparams,
        const llama_kv_cache_msa_context * mctx) :
    llm_graph_input_attn_kv(hparams, cparams, mctx->get_base()),
    mctx_msa(mctx) {
}

void llm_graph_input_attn_kv_msa::set_input(const llama_ubatch * ubatch) {
    llm_graph_input_attn_kv::set_input(ubatch);

    if (self_k_idxs_idx) {
        mctx_msa->get_idx()->set_input_k_idxs(self_k_idxs_idx, ubatch);
    }
}

bool llm_graph_input_attn_kv_msa::can_reuse(const llm_graph_params & params) {
    mctx_msa = static_cast<const llama_kv_cache_msa_context *>(params.mctx);

    // the parent class operates on the base cache context
    this->mctx = mctx_msa->get_base();

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
    if (self_k_idxs_idx) {
        res &= self_k_idxs_idx->ne[0] == params.ubatch.n_tokens;
    }

    res &= can_reuse_kq_mask(self_kq_mask, this->mctx, params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_k_dsa::set_input(const llama_ubatch * ubatch) {
    mctx->get_mla()->set_input_k_idxs(self_k_idxs_mla, ubatch);

    mctx->get_mla()->set_input_kq_mask(self_kq_mask_mla, ubatch, cparams.causal_attn);

    mctx->get_lid()->set_input_k_idxs(self_k_idxs_lid, ubatch);

    mctx->get_lid()->set_input_kq_mask(self_kq_mask_lid, ubatch, cparams.causal_attn);

    // left unallocated when the indexer does not use the rotation
    if (self_k_rot_lid && self_k_rot_lid->buffer) {
        mctx->get_lid()->set_input_k_rot(self_k_rot_lid);
    }
}

bool llm_graph_input_attn_k_dsa::can_reuse(const llm_graph_params & params) {
    mctx = static_cast<const llama_kv_cache_dsa_context *>(params.mctx);

    return can_reuse_impl(params);
}

bool llm_graph_input_attn_k_dsa::can_reuse_impl(const llm_graph_params & params) {
    bool res = true;

    res &= self_k_idxs_mla->ne[0] == params.ubatch.n_tokens;
    res &= self_k_idxs_lid->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(self_kq_mask_mla, mctx->get_mla(), params.ubatch, params.cparams);
    res &= can_reuse_kq_mask(self_kq_mask_lid, mctx->get_lid(), params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_k_dsa_iswa::set_input(const llama_ubatch * ubatch) {
    inp_dsa->set_input(ubatch);
    inp_swa->set_input(ubatch);
}

bool llm_graph_input_attn_k_dsa_iswa::can_reuse(const llm_graph_params & params) {
    mctx = static_cast<const llama_kv_cache_dsa_iswa_context *>(params.mctx);

    inp_dsa->mctx = mctx->get_dsa();
    inp_swa->mctx = mctx->get_swa();

    bool res = true;

    res &= inp_dsa->can_reuse_impl(params);
    res &= inp_swa->can_reuse_impl(params);

    return res;
}

void llm_graph_input_attn_kv_iswa::set_input(const llama_ubatch * ubatch) {
    set_tail_query_plan(self_tail_query_order, self_tail_run_desc, *ubatch);
    set_tail_query_plan(self_tail_query_order_swa, self_tail_run_desc_swa, *ubatch);
    const auto set_group = [&](const llama_kv_cache_context * group, bool swa) {
        ggml_tensor * k_idxs = swa ? self_k_idxs_swa : self_k_idxs;
        ggml_tensor * v_idxs = swa ? self_v_idxs_swa : self_v_idxs;
        ggml_tensor * tail_idxs = get_tail_idxs(swa);
        ggml_tensor * mask = swa ? self_kq_mask_swa : self_kq_mask;
        ggml_tensor * kvarn_mat_idxs = swa ? self_kvarn_mat_idxs_swa : self_kvarn_mat_idxs;
        if (k_idxs && k_idxs->buffer) {
            group->set_input_k_idxs(k_idxs, ubatch);
            if (v_idxs) {
                group->set_input_v_idxs(v_idxs, ubatch);
            }
        }
        if (tail_idxs && tail_idxs->buffer) {
            group->set_input_tail_idxs(tail_idxs, ubatch);
        }
        if (kvarn_mat_idxs && kvarn_mat_idxs->buffer) {
            const auto * kvarn = dynamic_cast<const llama_kv_cache_kvarn_context *>(group);
            GGML_ASSERT(kvarn != nullptr);
            kvarn->set_input_kvarn_mat_idxs(kvarn_mat_idxs, ubatch);
        }
        if (mask && mask->buffer) {
            group->set_input_kq_mask(mask, ubatch, cparams.causal_attn);
            group->set_input_kq_mask_tail(mask, get_kq_mask_tail(swa),
                    get_tail_read_idxs(swa), get_tail_body_read_idxs(swa),
                    get_tail_bias_read_idxs(swa), ubatch, cparams.causal_attn);
            group->set_input_tail_body_plan(
                    get_tail_query_order(swa), get_tail_run_desc(swa), mask,
                    ubatch, cparams.causal_attn);
        }
    };
    set_group(mctx->get_base(), false);
    set_group(mctx->get_swa(), true);

    if (self_k_rot && self_k_rot->buffer) {
        mctx->get_base()->set_input_k_rot(self_k_rot);
    }

    if (self_v_rot && self_v_rot->buffer) {
        mctx->get_base()->set_input_v_rot(self_v_rot);
    }

    if ((self_kvarn_rot_64 && self_kvarn_rot_64->buffer) ||
            (self_kvarn_rot_128 && self_kvarn_rot_128->buffer) ||
            (self_kvarn_rot_256 && self_kvarn_rot_256->buffer) ||
            (self_kvarn_rot_512 && self_kvarn_rot_512->buffer)) {
        const auto * kvarn_base = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx->get_base());
        const auto * kvarn_swa  = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx->get_swa());
        GGML_ASSERT(kvarn_base != nullptr || kvarn_swa != nullptr);
        llm_kvarn_set_rot_inputs(kvarn_base ? kvarn_base : kvarn_swa,
                self_kvarn_rot_64, self_kvarn_rot_128, self_kvarn_rot_256, self_kvarn_rot_512);
    }

    if (self_kvarn_mat_idxs_swa && self_kvarn_mat_idxs_swa->buffer) {
        const auto * kvarn_swa = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx->get_swa());
        GGML_ASSERT(kvarn_swa != nullptr);
        kvarn_swa->set_input_kvarn_mat_idxs(self_kvarn_mat_idxs_swa, ubatch);
    }

    if (self_k_rot_swa && self_k_rot_swa->buffer) {
        mctx->get_swa()->set_input_k_rot(self_k_rot_swa);
    }

    if (self_v_rot_swa && self_v_rot_swa->buffer) {
        mctx->get_swa()->set_input_v_rot(self_v_rot_swa);
    }
}

bool llm_graph_input_attn_kv_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;
    res &= base_tail_identity.matches(hparams, mctx->get_base());
    res &= swa_tail_identity.matches(hparams, mctx->get_swa());
    // plain locals instead of structured bindings: the lambda below captures
    // them, which C++17 does not allow for structured bindings
    const auto tail_shape = tail_query_plan_shape(params.ubatch);
    const int64_t q_max    = tail_shape.first;
    const int64_t n_active = tail_shape.second;
    res &= self_tail_query_order == nullptr ||
            (self_tail_query_order->ne[0] == q_max && self_tail_query_order->ne[1] == n_active);
    res &= self_tail_query_order_swa == nullptr ||
            (self_tail_query_order_swa->ne[0] == q_max && self_tail_query_order_swa->ne[1] == n_active);
    const auto reuse_tail = [&](const llama_kv_cache_context * group, bool swa) {
        const uint32_t attention_stride = group->get_tail_attention_stride(uint32_t(q_max));
        ggml_tensor * read_idxs = get_tail_read_idxs(swa);
        ggml_tensor * body_idxs = get_tail_body_read_idxs(swa);
        ggml_tensor * bias_idxs = get_tail_bias_read_idxs(swa);
        ggml_tensor * tail_mask = get_kq_mask_tail(swa);
        ggml_tensor * body_mask = swa ? self_kq_mask_swa : self_kq_mask;
        bool valid = true;
        for (ggml_tensor * indices : { read_idxs, body_idxs, bias_idxs }) {
            valid &= indices == nullptr ||
                    (indices->ne[0] == attention_stride &&
                     indices->ne[1] == params.ubatch.n_tokens);
        }
        valid &= tail_mask == nullptr ||
                (tail_mask->ne[0] == attention_stride &&
                 tail_mask->ne[1] == body_mask->ne[1] && tail_mask->ne[3] == body_mask->ne[3]);
        const int64_t desc_stride = 6 + attention_stride +
                (group->can_pack_tail_body(params.ubatch) ? group->get_tail_body_execution_stride() : 0);
        ggml_tensor * run_desc = get_tail_run_desc(swa);
        valid &= run_desc == nullptr ||
                (run_desc->ne[0] == desc_stride && run_desc->ne[1] == n_active);
        return valid;
    };

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask, mctx->get_base(), params.ubatch, params.cparams);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
      //res &= self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask_swa, mctx->get_swa(), params.ubatch, params.cparams);
    }
    res &= reuse_tail(mctx->get_base(), false);
    res &= reuse_tail(mctx->get_swa(), true);

    // Re-bind the SWA KVarN view indices to the new per-batch context before
    // graph construction asks it for native K/V views.
    if (self_kvarn_mat_idxs_swa && self_kvarn_mat_idxs_swa->buffer) {
        if (const auto * kvarn_swa = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx->get_swa())) {
            kvarn_swa->set_mat_idxs(self_kvarn_mat_idxs_swa);
        }
    }
    if (self_kvarn_mat_idxs && self_kvarn_mat_idxs->buffer) {
        if (const auto * kvarn_base = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx->get_base())) {
            kvarn_base->set_mat_idxs(self_kvarn_mat_idxs);
        }
    }

    return res;
}

void llm_graph_input_attn_k_iswa::set_input(const llama_ubatch * ubatch) {
    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
    }

    // the kq mask guards on its own buffer: shared cells leave idxs unbacked while the mask stays live
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->get_base()->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        mctx->get_swa()->set_input_kq_mask(self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    if (self_k_rot && self_k_rot->buffer) {
        mctx->get_base()->set_input_k_rot(self_k_rot);
    }

    if (self_k_rot_swa && self_k_rot_swa->buffer) {
        mctx->get_swa()->set_input_k_rot(self_k_rot_swa);
    }
}

bool llm_graph_input_attn_k_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask, mctx->get_base(), params.ubatch, params.cparams);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask_swa, mctx->get_swa(), params.ubatch, params.cparams);
    }

    return res;
}

static void dsv4_set_i64(ggml_tensor * dst, const std::vector<int64_t> & src) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*ggml_element_size(dst));
}

static void dsv4_set_i32(ggml_tensor * dst, const std::vector<int32_t> & src) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*ggml_element_size(dst));
}

static void dsv4_set_kq_mask(
        ggml_tensor * dst,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(n_tokens%n_stream == 0);
    GGML_ASSERT(dst->ne[0] == plan.n_kv);
    GGML_ASSERT(dst->ne[1] == (int64_t) n_tokens/n_stream);
    GGML_ASSERT(dst->ne[2] == 1);
    GGML_ASSERT(dst->ne[3] == n_stream);
    GGML_ASSERT((int64_t) plan.n_visible.size() == (int64_t) n_tokens);
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    if (dst->type == GGML_TYPE_F32) {
        float * data = (float *) dst->data;

        for (int64_t i = 0; i < (int64_t) n_tokens; ++i) {
            const int32_t n_visible = plan.n_visible[i];

            for (int64_t j = 0; j < dst->ne[0]; ++j) {
                data[i*dst->ne[0] + j] = j < n_visible ? 0.0f : -INFINITY;
            }
        }
    } else if (dst->type == GGML_TYPE_F16) {
        ggml_fp16_t * data = (ggml_fp16_t *) dst->data;
        const ggml_fp16_t fp16_ninf = llama_cast<ggml_fp16_t>(-INFINITY);
        const ggml_fp16_t fp16_zero = llama_cast<ggml_fp16_t>(0.0f);

        for (int64_t i = 0; i < (int64_t) n_tokens; ++i) {
            const int32_t n_visible = plan.n_visible[i];

            for (int64_t j = 0; j < dst->ne[0]; ++j) {
                data[i*dst->ne[0] + j] = j < n_visible ? fp16_zero : fp16_ninf;
            }
        }
    }
}

static ggml_tensor * dsv4_build_raw_kq_mask(
        ggml_context * ctx,
        const llama_kv_cache_dsv4_raw_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams,
        int64_t n_stream) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;

    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(n_tokens%n_stream == 0);

    const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kq_mask");

    return res;
}

static bool dsv4_can_reuse_raw_kq_mask(
        ggml_tensor * kq_mask,
        const llama_kv_cache_dsv4_raw_context * mctx,
        const llama_ubatch & ubatch,
        int64_t n_stream) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;

    GGML_ASSERT(n_stream > 0);

    bool res = true;

    res &= (kq_mask->ne[0] == n_kv);
    res &= (kq_mask->ne[1] == n_tokens/n_stream);
    res &= (kq_mask->ne[2] == 1);
    res &= (kq_mask->ne[3] == n_stream);

    return res;
}

static std::string dsv4_plan_positions(const std::vector<int32_t> & values) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            ss << ", ";
        }
        ss << values[i];
    }
    ss << "]";
    return ss.str();
}

static bool dsv4_compress_debug() {
    static const bool debug = []() {
        const char * env = getenv("LLAMA_DSV4_COMPRESS_DEBUG");
        return env && atoi(env) > 0;
    }();

    return debug;
}

static void dsv4_set_comp_inputs(
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        const char * name,
        bool debug,
        uint32_t n_tokens,
        int64_t n_stream) {
    dsv4_set_i32(inp.state_pos, plan.state_pos);
    dsv4_set_i32(inp.state_persist_src_idxs, plan.state_persist_src_idxs);
    dsv4_set_i32(inp.state_persist_dst_idxs, plan.state_persist_dst_idxs);
    dsv4_set_i32(inp.state_restore_src_idxs, plan.state_restore_src_idxs);
    dsv4_set_i32(inp.state_restore_dst_idxs, plan.state_restore_dst_idxs);
    dsv4_set_i32(inp.state_snapshot_src_idxs, plan.state_snapshot_src_idxs);
    dsv4_set_i32(inp.state_snapshot_dst_idxs, plan.state_snapshot_dst_idxs);
    dsv4_set_i32(inp.state_read_idxs, plan.state_read_idxs);
    dsv4_set_i64(inp.state_write_idxs, plan.state_write_idxs);
    dsv4_set_i32(inp.state_write_pos, plan.state_write_pos);
    dsv4_set_kq_mask(inp.kq_mask, plan, n_tokens, n_stream);

    if (debug || dsv4_compress_debug()) {
        LLAMA_LOG_INFO("%s: %s n_tokens=%u, n_stream=%d, state_persist_dst=%s, state_write_pos=%s\n",
                __func__, name, n_tokens, (int) n_stream,
                dsv4_plan_positions(plan.state_persist_dst_idxs).c_str(),
                dsv4_plan_positions(plan.state_write_pos).c_str());
    }
}

static bool dsv4_can_reuse_tensor_1d(ggml_tensor * t, int64_t ne0) {
    return (t == nullptr && ne0 == 0) || (t != nullptr && t->ne[0] == ne0);
}

static bool dsv4_can_reuse_kq_mask(
        ggml_tensor * t,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    if (plan.n_kv == 0) {
        return t == nullptr;
    }

    GGML_ASSERT(n_stream > 0);

    return t != nullptr &&
           t->ne[0] == plan.n_kv &&
           t->ne[1] == (int64_t) n_tokens/n_stream &&
           t->ne[2] == 1 &&
           t->ne[3] == n_stream;
}

static bool dsv4_can_reuse_comp_input(
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    bool res = true;
    res &= dsv4_can_reuse_tensor_1d(inp.state_pos, plan.state_pos.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_persist_src_idxs, plan.state_persist_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_persist_dst_idxs, plan.state_persist_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_restore_src_idxs, plan.state_restore_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_restore_dst_idxs, plan.state_restore_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_snapshot_src_idxs, plan.state_snapshot_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_snapshot_dst_idxs, plan.state_snapshot_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_read_idxs, plan.state_read_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_write_idxs, plan.state_write_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_write_pos, plan.state_write_pos.size());
    res &= dsv4_can_reuse_kq_mask(inp.kq_mask, plan, n_tokens, n_stream);

    return res;
}

static ggml_tensor * dsv4_build_input_1d(
        ggml_context * ctx,
        ggml_type type,
        int64_t ne0,
        const std::string & name) {
    if (ne0 == 0) {
        return nullptr;
    }

    ggml_tensor * res = ggml_new_tensor_1d(ctx, type, ne0);
    ggml_set_input(res);
    ggml_set_name(res, name.c_str());

    return res;
}

static void dsv4_build_comp_inputs(
        ggml_context * ctx,
        llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        const char * name,
        const llama_cparams & cparams,
        int64_t n_stream) {
    inp.state_pos = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_pos.size(), std::string("dsv4_") + name + "_state_pos");
    inp.state_persist_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_persist_src_idxs.size(), std::string("dsv4_") + name + "_state_persist_src_idxs");
    inp.state_persist_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_persist_dst_idxs.size(), std::string("dsv4_") + name + "_state_persist_dst_idxs");
    inp.state_restore_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_restore_src_idxs.size(), std::string("dsv4_") + name + "_state_restore_src_idxs");
    inp.state_restore_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_restore_dst_idxs.size(), std::string("dsv4_") + name + "_state_restore_dst_idxs");
    inp.state_snapshot_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_snapshot_src_idxs.size(), std::string("dsv4_") + name + "_state_snapshot_src_idxs");
    inp.state_snapshot_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_snapshot_dst_idxs.size(), std::string("dsv4_") + name + "_state_snapshot_dst_idxs");
    inp.state_read_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_read_idxs.size(), std::string("dsv4_") + name + "_state_read_idxs");
    inp.state_write_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I64, plan.state_write_idxs.size(), std::string("dsv4_") + name + "_state_write_idxs");
    inp.state_write_pos = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_write_pos.size(), std::string("dsv4_") + name + "_state_write_pos");

    if (plan.n_kv > 0) {
        const int64_t n_tokens = (int64_t) plan.n_visible.size();

        GGML_ASSERT(n_stream > 0);
        GGML_ASSERT(n_tokens%n_stream == 0);

        inp.kq_mask = ggml_new_tensor_4d(ctx, (strcmp(name, "lid") != 0 && cparams.flash_attn) || (strcmp(name, "lid") == 0 && cparams.fused_lid) ? GGML_TYPE_F16 : GGML_TYPE_F32, plan.n_kv, n_tokens/n_stream, 1, n_stream);
        ggml_set_input(inp.kq_mask);
        ggml_set_name(inp.kq_mask, (std::string("dsv4_") + name + "_kq_mask").c_str());
    }
}

void llm_graph_input_dsv4_raw::set_input(const llama_ubatch * ubatch) {
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->set_input_k_idxs(self_k_idxs);
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
        mctx->set_input_kq_mask_tail(self_kq_mask, self_kq_mask_tail,
                self_tail_read_idxs, self_tail_body_read_idxs, self_tail_bias_read_idxs,
                ubatch, cparams.causal_attn);
    }

    // The tail WRITE index is consumed only by the local tail store. A
    // position-split overflow ubatch stores Q4 overflow keys and never writes
    // the local tail, so nothing references this input, the scheduler prunes it
    // and it arrives here without a buffer -- the same contract already used
    // for self_kq_mask and self_kvarn_mat_idxs above. An allocated input that
    // is not host memory still fails inside ggml_backend_buffer_is_host().
    if (self_tail_idxs && self_tail_idxs->buffer) {
        mctx->set_input_tail_idxs(self_tail_idxs, ubatch);
    }

    if (self_k_rot) {
        mctx->set_input_k_rot(self_k_rot);
    }
}

void llm_graph_input_dsv4::set_input(const llama_ubatch * ubatch) {
    const auto & plan_csa = mctx->get_csa_plan(*ubatch);
    const auto & plan_hca = mctx->get_hca_plan(*ubatch);
    const auto & plan_lid = mctx->get_lid_plan(*ubatch);
    const int64_t n_stream = plan_csa.n_stream;

    inp_raw->mctx = mctx->get_raw();
    inp_raw->set_input(ubatch);

    dsv4_set_comp_inputs(inp_csa, plan_csa, "csa", debug > 0, ubatch->n_tokens, n_stream);
    dsv4_set_comp_inputs(inp_hca, plan_hca, "hca", debug > 0, ubatch->n_tokens, n_stream);
    dsv4_set_comp_inputs(inp_lid, plan_lid, "lid", debug > 0, ubatch->n_tokens, n_stream);

    if (inp_csa.k_rot && inp_csa.k_rot->buffer) {
        mctx->get_csa()->set_input_k_rot(inp_csa.k_rot);
    }

    if (inp_hca.k_rot && inp_hca.k_rot->buffer) {
        mctx->get_hca()->set_input_k_rot(inp_hca.k_rot);
    }

    if (inp_lid.k_rot && inp_lid.k_rot->buffer) {
        mctx->get_lid()->set_input_k_rot(inp_lid.k_rot);
    }
}

bool llm_graph_input_dsv4::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx);

    this->mctx = mctx;
    inp_raw->mctx = mctx->get_raw();

    bool res = true;

    const auto & plan_csa = mctx->get_csa_plan(params.ubatch);
    const auto & plan_hca = mctx->get_hca_plan(params.ubatch);
    const auto & plan_lid = mctx->get_lid_plan(params.ubatch);
    const int64_t n_stream = plan_csa.n_stream;

    const auto * raw_ctx = mctx->get_raw();
    inp_raw->mctx = raw_ctx;
    const auto [tail_q_max, tail_n_active] = tail_query_plan_shape(params.ubatch);
    GGML_UNUSED(tail_n_active);
    const uint32_t tail_attention_stride = raw_ctx->get_tail_attention_stride(
            uint32_t(tail_extent_queries(raw_ctx, params.ubatch, tail_q_max)));

    if (inp_raw->self_k_idxs && inp_raw->self_k_idxs->buffer) {
        res &= inp_raw->self_k_idxs->ne[0] == raw_ctx->get_n_write();
    }
    res &= inp_raw->self_tail_idxs == nullptr || inp_raw->self_tail_idxs->ne[0] == raw_ctx->get_n_write();
    res &= inp_raw->self_tail_read_idxs == nullptr ||
            (inp_raw->self_tail_read_idxs->ne[0] == tail_attention_stride &&
             inp_raw->self_tail_read_idxs->ne[1] == params.ubatch.n_tokens);
    res &= inp_raw->self_tail_body_read_idxs == nullptr ||
            (inp_raw->self_tail_body_read_idxs->ne[0] == tail_attention_stride &&
             inp_raw->self_tail_body_read_idxs->ne[1] == params.ubatch.n_tokens);
    res &= inp_raw->self_tail_bias_read_idxs == nullptr ||
            (inp_raw->self_tail_bias_read_idxs->ne[0] == tail_attention_stride &&
             inp_raw->self_tail_bias_read_idxs->ne[1] == params.ubatch.n_tokens);
    if (inp_raw->self_kq_mask && inp_raw->self_kq_mask->buffer) {
        res &= dsv4_can_reuse_raw_kq_mask(inp_raw->self_kq_mask, raw_ctx, params.ubatch, n_stream);
    }
    res &= inp_raw->self_kq_mask_tail == nullptr ||
            (inp_raw->self_kq_mask_tail->ne[0] == tail_attention_stride &&
             inp_raw->self_kq_mask_tail->ne[1] == inp_raw->self_kq_mask->ne[1] &&
             inp_raw->self_kq_mask_tail->ne[3] == inp_raw->self_kq_mask->ne[3]);

    res &= dsv4_can_reuse_comp_input(inp_csa, plan_csa, params.ubatch.n_tokens, n_stream);
    res &= dsv4_can_reuse_comp_input(inp_hca, plan_hca, params.ubatch.n_tokens, n_stream);
    res &= dsv4_can_reuse_comp_input(inp_lid, plan_lid, params.ubatch.n_tokens, n_stream);

    return res;
}

void llm_graph_input_attn_cross::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(cross_kq_mask);

    const int64_t n_enc    = cross_kq_mask->ne[0];
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(cross_kq_mask->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    const auto fill_mask = [&](auto * data) {
        using T = std::remove_reference_t<decltype(*data)>;
        for (int i = 0; i < n_tokens; ++i) {
            GGML_ASSERT(!cross->seq_ids_enc.empty() && "llama_encode must be called first");
            for (int j = 0; j < n_enc; ++j) {
                float f = -INFINITY;

                for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                    const llama_seq_id seq_id = ubatch->seq_id[i][s];

                    if (cross->seq_ids_enc[j].find(seq_id) != cross->seq_ids_enc[j].end()) {
                        f = 0.0f;
                    }
                }

                data[i*n_enc + j] = llama_cast<T>(f);
            }
        }
    };

    if (cross_kq_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) cross_kq_mask->data);
    } else {
        fill_mask((float *) cross_kq_mask->data);
    }
}

void llm_graph_input_mem_hybrid::set_input(const llama_ubatch * ubatch) {
    if (inp_attn) {
        inp_attn->mctx = mctx->get_attn();
        inp_attn->set_input(ubatch);
    }

    inp_rs->set_input_rs(mctx->get_recr(), ubatch);
}

bool llm_graph_input_mem_hybrid::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    // inp_attn is null when the full-attention layers are offloaded to the
    // remote RKVA backend (no local attention KV input is built).
    if (inp_attn) {
        inp_attn->mctx = mctx->get_attn();

        const auto [tail_q_max, tail_n_active] = tail_query_plan_shape(params.ubatch);
        res &= inp_attn->self_tail_query_order == nullptr ||
                (inp_attn->self_tail_query_order->ne[0] == tail_q_max &&
                 inp_attn->self_tail_query_order->ne[1] == tail_n_active);
        const uint32_t tail_attention_stride = mctx->get_attn()->get_tail_attention_stride(
                uint32_t(tail_extent_queries(mctx->get_attn(), params.ubatch, tail_q_max)));
        const int64_t tail_desc_stride = 6 + tail_attention_stride +
                (mctx->get_attn()->can_pack_tail_body(params.ubatch) ?
                        mctx->get_attn()->get_tail_body_execution_stride() : 0);
        res &= inp_attn->self_tail_run_desc == nullptr ||
                (inp_attn->self_tail_run_desc->ne[0] == tail_desc_stride &&
                 inp_attn->self_tail_run_desc->ne[1] == tail_n_active);

        res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

        res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);
        // Packed overflow views are baked per graph; this path does not call
        // inp_attn->can_reuse(), so the binding is enforced here as well, against the
        // ATTENTION memory context (never the hybrid one).
        res &= split_packed_map_still_valid(mctx->get_attn(),
                inp_attn->split_ovf_packed, inp_attn->split_ovf_row0,
                inp_attn->split_ovf_n_rows, "hybrid");
        res &= inp_attn->self_tail_read_idxs == nullptr ||
                (inp_attn->self_tail_read_idxs->ne[0] == tail_attention_stride &&
                 inp_attn->self_tail_read_idxs->ne[1] == params.ubatch.n_tokens);
        res &= inp_attn->self_tail_body_read_idxs == nullptr ||
                (inp_attn->self_tail_body_read_idxs->ne[0] == tail_attention_stride &&
                 inp_attn->self_tail_body_read_idxs->ne[1] == params.ubatch.n_tokens);
        res &= inp_attn->self_tail_bias_read_idxs == nullptr ||
                (inp_attn->self_tail_bias_read_idxs->ne[0] == tail_attention_stride &&
                 inp_attn->self_tail_bias_read_idxs->ne[1] == params.ubatch.n_tokens);
        res &= inp_attn->self_kq_mask_tail == nullptr ||
                (inp_attn->self_kq_mask_tail->ne[0] == tail_attention_stride &&
                 inp_attn->self_kq_mask_tail->ne[1] == inp_attn->self_kq_mask->ne[1] &&
                 inp_attn->self_kq_mask_tail->ne[3] == inp_attn->self_kq_mask->ne[3]);
    }

    res &= inp_rs->can_reuse_rs(mctx->get_recr(), params);

    return res;
}

// TODO: Hybrid input classes are a bit redundant.
// Instead of creating a hybrid input, the graph can simply create 2 separate inputs.
// Refactoring is required in the future.
void llm_graph_input_mem_hybrid_k::set_input(const llama_ubatch * ubatch) {
    mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);

    mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);

    inp_rs->set_input_rs(mctx->get_recr(), ubatch);
}

bool llm_graph_input_mem_hybrid_k::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);

    res &= inp_rs->can_reuse_rs(mctx->get_recr(), params);

    return res;
}

void llm_graph_input_mem_hybrid_iswa::set_input(const llama_ubatch * ubatch) {
    inp_attn->mctx = mctx->get_attn();
    inp_attn->set_input(ubatch);

    inp_rs->set_input_rs(mctx->get_recr(), ubatch);
}

bool llm_graph_input_mem_hybrid_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    // plain locals instead of structured bindings: the lambda below captures
    // them, which C++17 does not allow for structured bindings
    const auto tail_shape = tail_query_plan_shape(params.ubatch);
    const int64_t tail_q_max    = tail_shape.first;
    const int64_t tail_n_active = tail_shape.second;
    res &= inp_attn->self_tail_query_order == nullptr ||
            (inp_attn->self_tail_query_order->ne[0] == tail_q_max &&
             inp_attn->self_tail_query_order->ne[1] == tail_n_active);
    res &= inp_attn->self_tail_query_order_swa == nullptr ||
            (inp_attn->self_tail_query_order_swa->ne[0] == tail_q_max &&
             inp_attn->self_tail_query_order_swa->ne[1] == tail_n_active);

    const auto * attn_ctx = mctx->get_attn();
    inp_attn->mctx = attn_ctx;
    const auto reuse_tail = [&](const llama_kv_cache_context * group, bool swa) {
        const uint32_t attention_stride = group->get_tail_attention_stride(uint32_t(tail_q_max));
        bool valid = true;
        for (ggml_tensor * indices : {
                inp_attn->get_tail_read_idxs(swa), inp_attn->get_tail_body_read_idxs(swa),
                inp_attn->get_tail_bias_read_idxs(swa) }) {
            valid &= indices == nullptr ||
                    (indices->ne[0] == attention_stride &&
                     indices->ne[1] == params.ubatch.n_tokens);
        }
        ggml_tensor * tail_mask = inp_attn->get_kq_mask_tail(swa);
        ggml_tensor * body_mask = swa ? inp_attn->self_kq_mask_swa : inp_attn->self_kq_mask;
        valid &= tail_mask == nullptr ||
                (tail_mask->ne[0] == attention_stride &&
                 tail_mask->ne[1] == body_mask->ne[1] && tail_mask->ne[3] == body_mask->ne[3]);
        const int64_t desc_stride = 6 + attention_stride +
                (group->can_pack_tail_body(params.ubatch) ? group->get_tail_body_execution_stride() : 0);
        ggml_tensor * run_desc = inp_attn->get_tail_run_desc(swa);
        valid &= run_desc == nullptr ||
                (run_desc->ne[0] == desc_stride && run_desc->ne[1] == tail_n_active);
        return valid;
    };

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, attn_ctx->get_base(), params.ubatch, params.cparams);

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        res &= inp_attn->self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask_swa, attn_ctx->get_swa(), params.ubatch, params.cparams);
    res &= reuse_tail(attn_ctx->get_base(), false);
    res &= reuse_tail(attn_ctx->get_swa(), true);

    res &= inp_rs->can_reuse_rs(mctx->get_recr(), params);

    return res;
}

void llm_graph_input_sampling::set_input(const llama_ubatch * ubatch) {
    // set the inputs only for the active samplers in the current ubatch
    std::unordered_set<llama_seq_id> active_samplers;
    for (uint32_t i = 0; i < ubatch->n_tokens; i++) {
        if (ubatch->output[i]) {
            llama_seq_id seq_id = ubatch->seq_id[i][0];
            active_samplers.insert(seq_id);
        }
    }

    for (auto seq_id : active_samplers) {
        if (samplers.find(seq_id) == samplers.end()) {
            continue;
        }

        auto & sampler = samplers[seq_id];

        if (sampler->iface->backend_set_input) {
            sampler->iface->backend_set_input(sampler);
        }
    }
}

bool llm_graph_input_sampling::can_reuse(const llm_graph_params & params) {
    if (samplers.size() != params.samplers.size()) {
        return false;
    }

    for (const auto & [seq_id, sampler] : params.samplers) {
        if (samplers[seq_id] != sampler) {
            return false;
        }
    }

    return true;
}

//
// llm_graph_result
//

llm_graph_result::llm_graph_result(int64_t max_nodes) : max_nodes(max_nodes) {
    reset();

    const char * LLAMA_GRAPH_RESULT_DEBUG = getenv("LLAMA_GRAPH_RESULT_DEBUG");
    debug = LLAMA_GRAPH_RESULT_DEBUG ? atoi(LLAMA_GRAPH_RESULT_DEBUG) : 0;
}

int64_t llm_graph_result::get_max_nodes() const {
    return max_nodes;
}

void llm_graph_result::reset() {
    t_inp_tokens  = nullptr;
    t_inp_embd    = nullptr;
    t_logits      = nullptr;
    t_embd        = nullptr;
    t_embd_pooled = nullptr;
    t_h_nextn     = nullptr;
    t_h_capture   = nullptr;

    t_layer_inp.resize(LLAMA_MAX_LAYERS + 1);
    std::fill(t_layer_inp.begin(), t_layer_inp.end(), nullptr);

    t_sampled.clear();
    t_dspark_greedy.clear();
    t_sampled_probs.clear();
    t_sampled_logits.clear();
    t_candidates.clear();

    params = {};

    inputs.clear();
    fused_nodes.clear();
    hdmd_inputs.clear();

    buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));

    ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta.size(),
        /*.mem_buffer =*/ buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ctx_compute.reset(ggml_init(params));

    gf = ggml_new_graph_custom(ctx_compute.get(), max_nodes, false);
}

void llm_graph_result::set_inputs(const llama_ubatch * ubatch) {
    for (auto & input : inputs) {
        input->set_input(ubatch);
    }
}

void llm_graph_result::set_outputs(const llm_graph_params & params) {
    if (t_logits != nullptr) {
        ggml_set_output(t_logits);
    }
    if (t_embd != nullptr) {
        ggml_set_output(t_embd);
    }
    if (t_embd_pooled != nullptr) {
        ggml_set_output(t_embd_pooled);
    }
    if (t_h_capture != nullptr) {
        ggml_set_output(t_h_capture);
    }
    if (t_h_nextn != nullptr) {
        ggml_set_output(t_h_nextn);
    }
    {
        const auto & embeddings_layer_inp = params.cparams.embeddings_layer_inp;
        for (size_t il = 0; il < embeddings_layer_inp.size(); ++il) {
            if (embeddings_layer_inp[il]) {
                GGML_ASSERT(t_layer_inp[il] != nullptr && "layer input tensor is null");
                ggml_set_output(t_layer_inp[il]);
            }
        }
    }
    for (auto * tensor : t_sampled) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
    for (auto * tensor : t_sampled_probs) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
    for (auto * tensor : t_sampled_logits) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
    for (auto * tensor : t_candidates) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
}

bool llm_graph_result::can_reuse(const llm_graph_params & params) {
    if (!this->params.allow_reuse(params)) {
        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: cannot reuse graph due to incompatible graph parameters\n", __func__);
        }

        return false;
    }

    if (debug > 1) {
        LLAMA_LOG_DEBUG("%s: checking compatibility of %d inputs:\n", __func__, (int) inputs.size());
    }

    bool res = true;

    for (auto & input : inputs) {
        const bool cur = input->can_reuse(params);

        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: can_reuse = %d\n", "placeholder", cur);
        }

        res = res && cur;
    }

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: can reuse graph = %d\n", __func__, res);
    }

    return res;
}

llm_graph_input_i * llm_graph_result::add_input(llm_graph_input_ptr input) {
    inputs.emplace_back(std::move(input));
    return inputs.back().get();
}

void llm_graph_result::add_fused_node(llm_graph_fused_node result) {
    fused_nodes.push_back(result);
}

ggml_tensor * llm_graph_result::get_hdmd_input(const ggml_tensor * cur, const ggml_tensor * rot) const {
    const auto it = hdmd_inputs.find({ cur, rot });
    return it == hdmd_inputs.end() ? nullptr : it->second;
}

void llm_graph_result::set_hdmd_input(const ggml_tensor * cur, const ggml_tensor * rot, ggml_tensor * res) {
    hdmd_inputs[{ cur, rot }] = res;
}

void llm_graph_result::set_params(const llm_graph_params & params) {
    this->params = params;
}

//
// llm_graph_context
//

llm_graph_context::llm_graph_context(const llm_graph_params & params) :
    arch             (params.arch),
    hparams          (params.hparams),
    cparams          (params.cparams),
    ubatch           (params.ubatch),
    n_embd           (hparams.n_embd),
    n_layer          (hparams.n_layer()),
    n_layer_nextn    (hparams.n_layer_nextn),
    n_rot            (hparams.n_rot()),
    n_ctx            (cparams.n_ctx),
    n_head           (hparams.n_head()),
    n_head_kv        (hparams.n_head_kv()),
    n_embd_head_k    (hparams.n_embd_head_k()),
    n_embd_k_gqa     (hparams.n_embd_k_gqa()),
    n_embd_head_v    (hparams.n_embd_head_v()),
    n_embd_v_gqa     (hparams.n_embd_v_gqa()),
    n_expert         (hparams.n_expert),
    n_expert_used    (cparams.warmup ? hparams.n_expert : hparams.n_expert_used()),
    freq_base        (cparams.rope_freq_base),
    freq_scale       (cparams.rope_freq_scale),
    ext_factor       (cparams.yarn_ext_factor),
    attn_factor      (cparams.yarn_attn_factor),
    beta_fast        (cparams.yarn_beta_fast),
    beta_slow        (cparams.yarn_beta_slow),
    norm_eps         (hparams.f_norm_eps),
    norm_rms_eps     (hparams.f_norm_rms_eps),
    n_tokens         (ubatch.n_tokens),
    n_outputs        (params.n_outputs),
    n_ctx_orig       (cparams.n_ctx_orig_yarn),
    pooling_type     (cparams.pooling_type),
    rope_type        (hparams.rope_type),
    sched            (params.sched),
    backend_cpu      (params.backend_cpu),
    backend_remote   (params.backend_remote),
    cvec             (params.cvec),
    loras            (params.loras),
    mctx             (params.mctx),
    cross            (params.cross),
    dspark_ctx       (params.dspark_ctx),
    dspark_has_context (params.dspark_has_context),
    dspark_ctx_rows  (params.dspark_ctx_rows),
    dspark_ctx_width (params.dspark_ctx_width),
    moe_cache        (params.moe_cache),
    prec_policy      (params.prec_policy),
    hdmd             (params.hdmd),
    samplers         (params.samplers),
    cb_func          (params.cb),
    res              (params.res),
    ctx0             (res->get_ctx()),
    gf               (res->get_gf()) {
        res->set_params(params);
    }


void llm_graph_context::cb(ggml_tensor * cur, const char * name, int il) const {
    if (cb_func) {
        cb_func(ubatch, cur, name, il);
    }
}



ggml_tensor * llm_graph_context::build_cvec(
         ggml_tensor * cur,
                 int   il) const {
    return cvec->apply_to(ctx0, cur, il);
}

ggml_tensor * llm_graph_context::build_hadamard_inverse_embd(
          ggml_tensor * w,
          ggml_tensor * cur) const {
    if (hdmd) {
        if (const auto it = hdmd->inv.find(w); it != hdmd->inv.end()) {
            cur = llama_mul_mat_hadamard(ctx0, cur, it->second.rot);
            if (it->second.signs) {
                cur = ggml_mul(ctx0, cur, it->second.signs);
            }
        }
    }
    return cur;
}

ggml_tensor * llm_graph_context::build_hadamard_input(
          ggml_tensor * w,
          ggml_tensor * cur) const {
    if (!hdmd) {
        return cur;
    }
    const auto it = hdmd->rot.find(w);
    if (it == hdmd->rot.end()) {
        return cur;
    }
    const auto & t = it->second;
    if (ggml_tensor * x = res->get_hdmd_input(cur, t.rot)) {
        return x;
    }
    ggml_tensor * x = cur;
    if (t.perm_rep > 1) {
        // tiled [hd, nk, rep] -> grouped [hd, rep, nk] feature order
        x = ggml_is_contiguous(x) ? x : ggml_cont(ctx0, x);
        const int64_t ne1 = x->ne[1], ne2 = x->ne[2], ne3 = x->ne[3];
        x = ggml_reshape_4d(ctx0, x, t.perm_hd, t.perm_nk, t.perm_rep, ne1*ne2*ne3);
        x = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 1, 3));
        x = ggml_reshape_4d(ctx0, x, t.perm_hd*t.perm_nk*t.perm_rep, ne1, ne2, ne3);
    }
    if (t.signs) {
        x = ggml_mul(ctx0, x, t.signs);
    }
    x = llama_mul_mat_hadamard(ctx0, x, t.rot);
    res->set_hdmd_input(cur, t.rot, x);
    return x;
}

ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur,
          ggml_tensor * w_s) const {
    cur = build_hadamard_input(w, cur);
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);

    if (prec_policy) {
        prec_policy->apply(res);
    }

    if (w->type == GGML_TYPE_NVFP4) {
        ggml_prec_set_acc(res, GGML_PREC_BF16);
    }

    if (w_s) {
        res = ggml_mul(ctx0, res, w_s);
    }

    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float adapter_scale = lora.second;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

        ggml_tensor * ab_cur = ggml_mul_mat(
                ctx0, lw->b,
                ggml_mul_mat(ctx0, lw->a, cur)
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_lora_mm_id(
          ggml_tensor * w,   // ggml_tensor * as
          ggml_tensor * cur, // ggml_tensor * b
          ggml_tensor * ids,
          ggml_tensor * w_s,
          ggml_tensor * slots) const {
    cur = build_hadamard_input(w, cur);
    // the experts in the MoE cache are selected by their slots
    ggml_tensor * res = slots == nullptr ?
        ggml_mul_mat_id(ctx0, w, cur, ids) :
        ggml_mul_mat_id(ctx0, moe_cache->get_experts(w), cur, slots);

    if (prec_policy) {
        prec_policy->apply(res);
    }

    if (w->type == GGML_TYPE_NVFP4) {
        ggml_prec_set_acc(res, GGML_PREC_BF16);
    }

    if (w_s) {
        const int64_t n_expert = w_s->ne[0];
        const int64_t n_tokens = cur->ne[2];
        ggml_tensor * s = ggml_reshape_3d(ctx0, w_s, 1, n_expert, 1);
        s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
        s = ggml_get_rows(ctx0, s, ids);
        res = ggml_mul(ctx0, res, s);
    }
    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float alpha = lora.first->alpha;
        const float rank  = (float) lw->b->ne[0];
        const float scale = alpha ? lora.second * alpha / rank : lora.second;

        ggml_tensor * ab_cur = ggml_mul_mat_id(
                ctx0, lw->b,
                ggml_mul_mat_id(ctx0, lw->a, cur, ids),
                ids
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_norm(
         ggml_tensor * cur,
         ggml_tensor * mw,
         ggml_tensor * mb,
       llm_norm_type   type,
                 int   il) const {
    switch (type) {
        case LLM_NORM:       cur = ggml_norm    (ctx0, cur, hparams.f_norm_eps);     break;
        case LLM_NORM_RMS:   cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps); break;
        case LLM_NORM_GROUP:
            {
                cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);
                cur = ggml_group_norm(ctx0, cur, hparams.n_norm_groups, hparams.f_norm_group_eps);
                cur = ggml_reshape_2d(ctx0, cur, cur->ne[0],    cur->ne[2]);
            } break;
    }

    if (mw || mb) {
        cb(cur, "norm", il);
    }

    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
        if (mb) {
            cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
    }

    return cur;
}


llm_graph_qkv llm_graph_context::build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head,
                  int64_t   n_head,
                  int64_t   n_head_kv,
                      int   il) const {
    return build_qkv(layer, cur,
            n_embd_head, n_head,
            n_embd_head, n_head_kv,
            n_embd_head, n_head_kv,
            il);
}

llm_graph_qkv llm_graph_context::build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head_q,
                  int64_t   n_head_q,
                  int64_t   n_embd_head_k,
                  int64_t   n_head_k,
                  int64_t   n_embd_head_v,
                  int64_t   n_head_v,
                      int   il,
                     bool   reshape) const {
    const int64_t n_embd_q = n_embd_head_q * n_head_q;
    const int64_t n_embd_k = n_embd_head_k * n_head_k;

    ggml_tensor * Qcur, * Kcur, * Vcur;

    if (layer.wqkv) {
        // fused QKV path
        ggml_tensor * qkv = build_lora_mm(layer.wqkv, cur, layer.wqkv_s);
        cb(qkv, "wqkv", il);
        if (layer.wqkv_b) {
            qkv = ggml_add(ctx0, qkv, layer.wqkv_b);
            cb(qkv, "wqkv_b", il);
        } else if (layer.wq_b && layer.wk_b && layer.wv_b) {
            // Fused weights may coexist with separate Q/K/V biases in legacy or custom GGUFs.
            ggml_tensor * qkv_b = ggml_concat(ctx0, ggml_concat(ctx0, layer.wq_b, layer.wk_b, 0), layer.wv_b, 0);
            qkv = ggml_add(ctx0, qkv, qkv_b);
            cb(qkv, "wqkv_b", il);
        }
        if (reshape && hparams.f_clamp_kqv > 0.0f) {
            qkv = ggml_clamp(ctx0, qkv, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(qkv, "wqkv_clamped", il);
        }
        if (reshape) {
            Qcur = ggml_view_3d(ctx0, qkv, n_embd_head_q, n_head_q, n_tokens,
                ggml_row_size(qkv->type, n_embd_head_q), qkv->nb[1], 0);
            Kcur = ggml_view_3d(ctx0, qkv, n_embd_head_k, n_head_k, n_tokens,
                ggml_row_size(qkv->type, n_embd_head_k), qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q));
            Vcur = ggml_view_3d(ctx0, qkv, n_embd_head_v, n_head_v, n_tokens,
                ggml_row_size(qkv->type, n_embd_head_v), qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q + n_embd_k));
        } else {
            Qcur = ggml_view_2d(ctx0, qkv, n_embd_q, n_tokens, qkv->nb[1], 0);
            Kcur = ggml_view_2d(ctx0, qkv, n_embd_k, n_tokens, qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q));
            Vcur = ggml_view_2d(ctx0, qkv, n_embd_head_v * n_head_v, n_tokens, qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q + n_embd_k));
        }
        if (!reshape) {
            Qcur = ggml_cont(ctx0, Qcur);
            Kcur = ggml_cont(ctx0, Kcur);
            Vcur = ggml_cont(ctx0, Vcur);
        }
    } else {
        // separate Q/K/V path
        Qcur = build_lora_mm(layer.wq, cur, layer.wq_s);
        if (reshape) {
            cb(Qcur, "Qcur", il);
        }
        if (layer.wq_b) {
            Qcur = ggml_add(ctx0, Qcur, layer.wq_b);
            if (reshape) {
                cb(Qcur, "Qcur", il);
            }
        }
        if (reshape && hparams.f_clamp_kqv > 0.0f) {
            Qcur = ggml_clamp(ctx0, Qcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Qcur, "Qcur_clamped", il);
        }
        Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
        if (reshape) {
            cb(Kcur, "Kcur", il);
        }
        if (layer.wk_b) {
            Kcur = ggml_add(ctx0, Kcur, layer.wk_b);
            if (reshape) {
                cb(Kcur, "Kcur", il);
            }
        }
        if (reshape && hparams.f_clamp_kqv > 0.0f) {
            Kcur = ggml_clamp(ctx0, Kcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Kcur, "Kcur_clamped", il);
        }
        Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
        if (reshape) {
            cb(Vcur, "Vcur", il);
        }
        if (layer.wv_b) {
            Vcur = ggml_add(ctx0, Vcur, layer.wv_b);
            if (reshape) {
                cb(Vcur, "Vcur", il);
            }
        }
        if (reshape && hparams.f_clamp_kqv > 0.0f) {
            Vcur = ggml_clamp(ctx0, Vcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Vcur, "Vcur_clamped", il);
        }
        if (reshape) {
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_q, n_head_q, n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_k, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head_v, n_head_v, n_tokens);
        }
    }

    if (reshape) {
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
    }

    return { Qcur, Kcur, Vcur };
}


ggml_tensor * llm_graph_context::build_ffn(
         ggml_tensor * cur,
         ggml_tensor * up,
         ggml_tensor * up_b,
         ggml_tensor * up_s,
         ggml_tensor * gate,
         ggml_tensor * gate_b,
         ggml_tensor * gate_s,
         ggml_tensor * down,
         ggml_tensor * down_b,
         ggml_tensor * down_s,
         ggml_tensor * act_scales,
     llm_ffn_op_type   type_op,
   llm_ffn_gate_type   type_gate,
                 int   il) const {
    // NVFP4 support is currently restricted to
    // 1) LORA absence (*_s would be applied after LORA residual, which is incorrect)
    // 2) bias absense (*_s would be applied after bias addition, which is incorrect)
    // TODO: disambiguate LLM-architectural scales (which use *_s) from NVFP4 scale_2 (which also uses *_s currently)
    auto has_lora = [this](ggml_tensor * w) {
        if (!w) {
            return false;
        }
        for (const auto & lora : *loras) {
            if (lora.first->get_weight(w) != nullptr) {
                return true;
            }
        }
        return false;
    };

    GGML_ASSERT(!up_s   || !up_b   || !up   || up->type   != GGML_TYPE_NVFP4);
    GGML_ASSERT(!gate_s || !gate_b || !gate || gate->type != GGML_TYPE_NVFP4);
    GGML_ASSERT(!down_s || !down_b || !down || down->type != GGML_TYPE_NVFP4);
    GGML_ASSERT(!up_s   || !up   || up->type   != GGML_TYPE_NVFP4 || !has_lora(up));
    GGML_ASSERT(!gate_s || !gate || gate->type != GGML_TYPE_NVFP4 || !has_lora(gate));
    GGML_ASSERT(!down_s || !down || down->type != GGML_TYPE_NVFP4 || !has_lora(down));

    ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (up_s) {
        tmp = ggml_mul(ctx0, tmp, up_s);
        cb(tmp, "ffn_up_s", il);
    }

    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = build_lora_mm(gate, tmp);
                    cb(cur, "ffn_gate", il);
                } break;
            case LLM_FFN_PAR:
                {
                    cur = build_lora_mm(gate, cur);
                    cb(cur, "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }

        if (gate_s) {
            cur = ggml_mul(ctx0, cur, gate_s);
            cb(cur, "ffn_gate_s", il);
        }

    } else {
        cur = tmp;
    }

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate && type_gate == LLM_FFN_PAR) {
                if (il >= 0) {
                    const float limit = hparams.swiglu_clamp_shexp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        if (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_GLM5_NEXT || (arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0)) {
                            cur = ggml_swiglu_clamp(ctx0, cur, tmp, limit);
                        } else {
                            tmp = ggml_clamp(ctx0, tmp, -limit, limit);
                            cb(tmp, "ffn_up_clamped", il);
                            ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                            cb(gate_act, "ffn_silu", il);
                            gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                            cb(gate_act, "ffn_silu_clamped", il);
                            cur = ggml_mul(ctx0, gate_act, tmp);
                        }
                        cb(cur, "ffn_swiglu_limited", il);
                        type_gate = LLM_FFN_SEQ;
                        break;
                    }
                }

                cur = ggml_swiglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_swiglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_geglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_gelu", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx0, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            } break;
        case LLM_FFN_RELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_reglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_reglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);

                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_sqr(relu)", il);
            } break;
        case LLM_FFN_SWIGLU:
            {
                cur = ggml_swiglu(ctx0, cur);
                cb(cur, "ffn_swiglu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            if (gate && type_gate == LLM_FFN_PAR) {
                // same alpha/limit constants as gpt-oss
                const float alpha = 1.702f;
                const float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, tmp, alpha, limit);
                cb(cur, "ffn_swiglu_oai", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                GGML_ABORT("LLM_FFN_SWIGLU_OAI_MOE requires a parallel gate");
            } break;
        case LLM_FFN_GEGLU:
            {
                cur = ggml_geglu(ctx0, cur);
                cb(cur, "ffn_geglu", il);
            } break;
        case LLM_FFN_GEGLU_ERF:
            {
                cur = ggml_geglu_erf(ctx0, cur);
                cb(cur, "ffn_geglu_erf", il);
            } break;
        case LLM_FFN_REGLU:
            {
                cur = ggml_reglu(ctx0, cur);
                cb(cur, "ffn_reglu", il);
            } break;
        case LLM_FFN_SITU:
            GGML_ABORT("not yet supported");
        default:
            GGML_ABORT("fatal error");
    }

    if (gate && type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx0, cur, tmp);
        cb(cur, "ffn_gate_par", il);
    }

    if (down) {
        cur = build_lora_mm(down, cur);
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
            ggml_prec_set_acc(cur, GGML_PREC_F32);
        }
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }

    if (down_s) {
        cur = ggml_mul(ctx0, cur, down_s);
        cb(cur, "ffn_down_s", il);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * up_exps,
         ggml_tensor * gate_exps,
         ggml_tensor * down_exps,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
         llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in) const {
    return build_moe_ffn(
        cur,
        gate_inp,  /* gate_inp_b  */ nullptr,
        up_exps,   /* up_exps_b   */ nullptr,
        gate_exps, /* gate_exps_b */ nullptr,
        down_exps, /* down_exps_b */ nullptr,
        exp_probs_b,
        n_expert,
        n_expert_used,
        type_op,
        norm_w,
        w_scale,
        gating_op,
        il,
        probs_in,
        gate_up_exps,
        /* gate_up_exps_b */ nullptr,
        up_exps_s,
        gate_exps_s,
        down_exps_s,
        selected_experts_in
    );
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * gate_inp_b,
         ggml_tensor * up_exps,
         ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,
         ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,
         ggml_tensor * down_exps_b,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
        llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * gate_up_exps_b,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in) const {
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4; // for llama4, we apply the sigmoid-ed weights before the FFN

    ggml_tensor * logits = nullptr;

    if (probs_in == nullptr) {
        logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
        if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
            ggml_prec_set_acc(logits, GGML_PREC_F32);
        }
        cb(logits, "ffn_moe_logits", il);
    } else {
        logits = probs_in;
    }

    if (gate_inp_b) {
        logits = ggml_add(ctx0, logits, gate_inp_b);
        cb(logits, "ffn_moe_logits_biased", il);
    }

    ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            {
                probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            {
                probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
            {
                probs = logits; // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS:
            {
                probs = ggml_sqrt(ctx0, ggml_softplus(ctx0, logits)); // [n_expert, n_tokens]
            } break;
        default:
            GGML_ABORT("fatal error");
    }
    cb(probs, "ffn_moe_probs", il);

    // add experts selection bias - introduced in DeepSeek V3
    // leave probs unbiased as it's later used to get expert weights
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx0, probs, exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // llama4 doesn't have exp_probs_b, and sigmoid is only used after top_k
    // see: https://github.com/meta-llama/llama-models/blob/699a02993512fb36936b1b0741e13c06790bcf98/models/llama4/moe.py#L183-L198
    if (arch == LLM_ARCH_LLAMA4) {
        selection_probs = logits;
    }

    if (arch == LLM_ARCH_GROVEMOE) {
        selection_probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // select top n_group_used expert groups
    // https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py#L440-L457
    if (hparams.n_expert_groups > 1 && n_tokens > 0) {
        const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;

        // organize experts into n_expert_groups
        ggml_tensor * selection_groups = ggml_reshape_3d(ctx0, selection_probs, n_exp_per_group, hparams.n_expert_groups, n_tokens); // [n_exp_per_group, n_expert_groups, n_tokens]

        ggml_tensor * group_scores = ggml_argsort_top_k(ctx0, selection_groups, 2); // [2, n_expert_groups, n_tokens]
        group_scores = ggml_get_rows(ctx0, ggml_reshape_4d(ctx0, selection_groups, 1, selection_groups->ne[0], selection_groups->ne[1], selection_groups->ne[2]), group_scores); // [1, 2, n_expert_groups, n_tokens]

        // get top n_group_used expert groups
        group_scores = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2], group_scores->ne[3])); // [1, n_expert_groups, n_tokens]
        group_scores = ggml_reshape_2d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2]); // [n_expert_groups, n_tokens]

        ggml_tensor * expert_groups = ggml_argsort_top_k(ctx0, group_scores, hparams.n_group_used); // [n_group_used, n_tokens]
        cb(expert_groups, "ffn_moe_group_topk", il);

        // mask out the other groups
        selection_probs = ggml_get_rows(ctx0, selection_groups, expert_groups); // [n_exp_per_group, n_group_used, n_tokens]
        selection_probs = ggml_set_rows(ctx0, ggml_fill(ctx0, selection_groups, -INFINITY), selection_probs, expert_groups); // [n_exp_per_group, n_expert_groups, n_tokens]
        selection_probs = ggml_reshape_2d(ctx0, selection_probs, n_expert, n_tokens); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_masked", il);
    }

    // select experts
    ggml_tensor * selected_experts = selected_experts_in;
    if (selected_experts == nullptr) {
        selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
        cb(selected_experts->src[0], "ffn_moe_argsort", il);
    }
    cb(selected_experts, "ffn_moe_topk", il);

    if (arch == LLM_ARCH_GROVEMOE && n_expert != hparams.n_expert) {
        // TODO: Use scalar div instead when/if implemented
        ggml_tensor * f_sel = ggml_cast(ctx0, selected_experts, GGML_TYPE_F32);
        selected_experts = ggml_cast(ctx0, ggml_scale(ctx0, f_sel, 1.0f / float(hparams.n_group_experts)), GGML_TYPE_I32);
        probs = ggml_reshape_3d(ctx0, probs, 1, hparams.n_expert, n_tokens);
    } else {
        probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens);
    }

    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);


    if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        weights = ggml_soft_max(ctx0, weights); // [n_expert_used, n_tokens]
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        cb(weights, "ffn_moe_weights_softmax", il);
    }

    if (norm_w) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        // Avoid division by zero, clamp to smallest number representable by F16
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
        cb(weights_sum, "ffn_moe_weights_sum_clamped", il);

        weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
    }
    if (w_scale != 0.0f && w_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    //call early so that topk-moe can be used
    ggml_build_forward_expand(gf, weights);

    // the experts of host-resident layers may be read from the MoE cache
    ggml_tensor * slots = build_moe_cache_slots(selected_experts, up_exps, gate_exps, down_exps, gate_up_exps, il);

    cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

    if (weight_before_ffn) {
        // repeat cur to [n_embd, n_expert_used, n_tokens]
        ggml_tensor * repeated = ggml_repeat_4d(ctx0, cur, n_embd, n_expert_used, n_tokens, 1);
        cur = ggml_mul(ctx0, repeated, weights);
        cb(cur, "ffn_moe_weighted", il);
    }

    ggml_tensor * up = nullptr;
    ggml_tensor * experts = nullptr;

    if (gate_up_exps) {
        // merged gate_up path: one mul_mat_id, then split into gate and up views
        ggml_tensor * gate_up = build_lora_mm_id(gate_up_exps, cur, selected_experts, up_exps_s, slots); // [n_ff*2, n_expert_used, n_tokens]
        cb(gate_up, "ffn_moe_gate_up", il);

        if (up_exps_s) {
            cb(gate_up, "ffn_moe_gate_up_scaled", il);
        }

        if (gate_up_exps_b) {
            gate_up = ggml_add_id(ctx0, gate_up, gate_up_exps_b, selected_experts);
            cb(gate_up, "ffn_moe_gate_up_biased", il);
        }

        const int64_t n_ff = gate_up->ne[0] / 2;
        cur = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
        cb(cur, "ffn_moe_gate", il);
        up  = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
        cb(up, "ffn_moe_up", il);
    } else {
        // separate gate and up path
        up = build_lora_mm_id(up_exps, cur, selected_experts, up_exps_s, slots); // [n_ff, n_expert_used, n_tokens]
        cb(up, "ffn_moe_up", il);

        if (up_exps_s) {
            cb(up, "ffn_moe_up_scaled", il);
        }

        if (up_exps_b) {
            up = ggml_add_id(ctx0, up, up_exps_b, selected_experts);
            cb(up, "ffn_moe_up_biased", il);
        }

        if (gate_exps) {
            cur = build_lora_mm_id(gate_exps, cur, selected_experts, gate_exps_s, slots); // [n_ff, n_expert_used, n_tokens]
            cb(cur, "ffn_moe_gate", il);
        } else {
            cur = up;
        }

        if (gate_exps_s) {
            cb(cur, "ffn_moe_gate_scaled", il);
        }

        if (gate_exps_b) {
            cur = ggml_add_id(ctx0, cur, gate_exps_b, selected_experts);
            cb(cur, "ffn_moe_gate_biased", il);
        }
    }

    const bool has_gate = gate_exps || gate_up_exps;

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate_exps) {
                if (il >= 0) {
                    const float limit = hparams.swiglu_clamp_exp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        if (arch == LLM_ARCH_MAPLE || arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_GLM5_NEXT || (arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0) || arch == LLM_ARCH_HY_V4) {
                            cur = ggml_swiglu_clamp(ctx0, cur, up, limit);
                        } else {
                            up = ggml_clamp(ctx0, up, -limit, limit);
                            cb(up, "ffn_moe_up_clamped", il);
                            ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                            cb(gate_act, "ffn_moe_silu", il);
                            gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                            cb(gate_act, "ffn_moe_silu_clamped", il);
                            cur = ggml_mul(ctx0, gate_act, up);
                        }
                        cb(cur, "ffn_moe_swiglu_limited", il);
                        break;
                    }
                }
            }

            if (has_gate) {
                cur = ggml_swiglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_swiglu", il);
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_moe_silu", il);
            } break;
        case LLM_FFN_SITU:
            {
                // situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)
                GGML_ASSERT(has_gate);
                const float beta = hparams.situ_beta;
                const float lb   = hparams.situ_linear_beta;

                ggml_tensor * act = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/beta)), beta);
                act = ggml_mul(ctx0, act, ggml_sigmoid(ctx0, cur));
                if (lb > 0.0f) {
                    up = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, up, 1.0f/lb)), lb);
                }
                cur = ggml_mul(ctx0, act, up);
                cb(cur, "ffn_moe_situ", il);
            } break;
        case LLM_FFN_GELU:
            if (has_gate) {
                cur = ggml_geglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_geglu", il);
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_moe_gelu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            {
                // TODO: move to hparams?
                constexpr float alpha = 1.702f;
                constexpr float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, up, alpha, limit);
                cb(cur, "ffn_moe_swiglu_oai", il);
            } break;
        case LLM_FFN_RELU:
            if (has_gate) {
                cur = ggml_reglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_reglu", il);
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_moe_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            if (has_gate) {
                // TODO: add support for gated squared relu
                GGML_ABORT("fatal error: gated squared relu not implemented");
            } else {
                cur = ggml_relu(ctx0, cur);
                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_moe_relu_sqr", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    experts = build_lora_mm_id(down_exps, cur, selected_experts, down_exps_s, slots); // [n_embd, n_expert_used, n_tokens]
    if (arch == LLM_ARCH_MISTRAL4) {
        // src1 can exceed F16 range
        ggml_prec_set_src(experts, GGML_PREC_F32, 1);
    }
    cb(experts, "ffn_moe_down", il);

    if (down_exps_s) {
        cb(experts, "ffn_moe_down_scaled", il);
    }

    if (down_exps_b) {
        experts = ggml_add_id(ctx0, experts, down_exps_b, selected_experts);
        cb(experts, "ffn_moe_down_biased", il);
    }

    if (!weight_before_ffn) {
        experts = ggml_mul(ctx0, experts, weights);
        cb(experts, "ffn_moe_weighted", il);
    }

    ggml_build_forward_expand(gf, experts);

    ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };

    assert(n_expert_used > 0);

    // order the views before the adds
    // Use per-layer n_expert_used to bound the graph even during warmup (avoids
    // the large-add-nodes issue for uniform arches; for Puzzle the per-layer
    // value is correct). ref: https://github.com/ggml-org/llama.cpp/pull/14753
    const uint32_t n_expert_used_il = hparams.n_expert_used(il);
    for (uint32_t i = 0; i < n_expert_used_il; ++i) {
        cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);

        ggml_build_forward_expand(gf, cur_experts[i]);
    }

    // aggregate experts
    ggml_tensor * moe_out = cur_experts[0];

    for (uint32_t i = 1; i < n_expert_used_il; ++i) {
        moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);

        ggml_build_forward_expand(gf, moe_out);
    }

    if (n_expert_used_il == 1) {
        // avoid returning a non-contiguous tensor
        moe_out = ggml_cont(ctx0, moe_out);
    }

    cb(moe_out, "ffn_moe_out", il);

    return moe_out;
}

ggml_tensor * llm_graph_context::build_moe_cache_slots(
         ggml_tensor * selected_experts,
         ggml_tensor * up_exps,
         ggml_tensor * gate_exps,
         ggml_tensor * down_exps,
         ggml_tensor * gate_up_exps,
                 int   il) const {
    if (moe_cache == nullptr) {
        return nullptr;
    }

    ggml_tensor * slot_map = moe_cache->get_slot_map(il, selected_experts->ne[1], selected_experts->ne[0]);
    if (slot_map == nullptr) {
        return nullptr;
    }
    for (ggml_tensor * w : { up_exps, gate_exps, down_exps, gate_up_exps }) {
        if (w != nullptr && moe_cache->get_experts(w) == nullptr) {
            return nullptr;
        }
    }

    ggml_tensor * ids = selected_experts;
    if (!ggml_is_contiguous(ids)) {
        ids = ggml_cont(ctx0, ids);
    }
    ids = ggml_reshape_1d(ctx0, ids, ggml_nelements(ids));

    // the slot map is a host weight, so the scheduler starts a new split here and copies it with the copy callback
    // the callback reads the selected experts, uploads the missing ones and updates the slot map
    ggml_tensor * slots = ggml_get_rows(ctx0, slot_map, ids); // [1, n_expert_used*n_tokens]
    if (!ggml_backend_supports_op(moe_cache->backend(il), slots)) {
        return nullptr;
    }
    ggml_backend_sched_set_tensor_backend(sched, slots, moe_cache->backend(il));
    cb(slots, "ffn_moe_slots", il);

    // add the lookup now: the transform of a Hadamard-folded expert reads host weights and would start its split first
    ggml_build_forward_expand(gf, slots);

    return ggml_reshape_2d(ctx0, slots, selected_experts->ne[0], selected_experts->ne[1]); // [n_expert_used, n_tokens]
}

// input embeddings with optional lora
ggml_tensor * llm_graph_context::build_inp_embd(ggml_tensor * tok_embd, float tok_scale) const {
    const int64_t n_embd_inp = hparams.n_embd_inp();
    const int64_t n_embd     = hparams.n_embd;

    assert(n_embd_inp >= n_embd);

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    // mixed path (ubatch.is_mixed()): set_rows the token rows into a copy of the embd rows, with its own inputs as select branches must not share tensors
    // TODO: use inp->tokens and inp->embd once ggml_build_forward_select allows it
    const bool has_mixed = llm_arch_supports_mixed_batch(arch) && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT;

    const int64_t n_tok_rows = has_mixed ? llm_graph_n_tok_rows(ubatch) : 0;

    // we have 3 standard paths to produce the input embeddings for the first layer:
    // - embd0: extract from the token embeddings weight (`tok_embd`) using the input token ids
    // - embd1: pass raw embeddings, skipping the `tok_embd`
    // - embd2: mixed path of both tokens ids + raw embeddings (if supported)
    ggml_tensor * embd0 = nullptr;
    ggml_tensor * embd1 = nullptr;
    ggml_tensor * embd2 = nullptr;

    // construct the input tensors
    {
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        cb(inp->tokens, "inp_tokens", -1);
        ggml_set_input(inp->tokens);
        res->t_inp_tokens = inp->tokens;

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens);
        cb(inp->embd, "inp_embd", -1);
        ggml_set_input(inp->embd);

        if (has_mixed) {
            inp->mixed_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tok_rows);
            cb(inp->mixed_tokens, "inp_mixed_tokens", -1);
            ggml_set_input(inp->mixed_tokens);
        }
    }

    // the embeddings placeholders for the 3 paths
    // we use ggml_build_forward_order to make the GET_ROWS ops stick at the beginning of the compute graph
    // this way the embeddings remain in the host buffer, and the GET_ROWS run before any other computations
    {
        embd0 = ggml_get_rows(ctx0, tok_embd, inp->tokens);
        ggml_build_forward_order(gf, embd0);

        embd1 = inp->embd;

        if (has_mixed) {
            embd2 = ggml_get_rows(ctx0, tok_embd, inp->mixed_tokens);
            ggml_build_forward_order(gf, embd2);
        }
    }

    // helper for extracting token embeddings with lora and padding
    // TODO: when lora is active, this is likely going to cause issues similar to https://github.com/ggml-org/llama.cpp/pull/30160
    //       need to add lora tests and refactor the logic to make the lora GET_ROWS go at the front of the graph
    auto build_tok = [&](ggml_tensor * cur, ggml_tensor * ids) {
        cur = build_hadamard_inverse_embd(tok_embd, cur);

        // apply lora for embedding tokens if needed
        for (const auto & lora : *loras) {
            llama_adapter_lora_weight * lw = lora.first->get_weight(tok_embd);
            if (lw == nullptr) {
                continue;
            }

            const float adapter_scale = lora.second;
            const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

            ggml_tensor * inpL_delta = ggml_scale(ctx0, ggml_mul_mat(
                        ctx0, lw->b, // non-transposed lora_b
                        ggml_get_rows(ctx0, lw->a, ids)
                        ), scale);

            cur = ggml_add(ctx0, cur, inpL_delta);
        }

        if (n_embd_inp != n_embd) {
            cur = ggml_pad(ctx0, cur, hparams.n_embd_inp() - n_embd, 0, 0, 0);
        }

        return cur;
    };

    // select one of the 3 inputs, based on the batch contents
    // ref: https://github.com/ggml-org/llama.cpp/pull/18550
    std::array<ggml_tensor *, 3> inps = {};

    // token embeddings path (ubatch.token != nullptr)
    inps[0] = build_tok(embd0, inp->tokens);

    // vector embeddings path (ubatch.embd != nullptr)
    inps[1] = embd1;

    assert(ggml_are_same_shape (inps[0], inps[1]));
    assert(ggml_are_same_stride(inps[0], inps[1]));

    if (has_mixed) {
        inp->mixed_slots = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_tok_rows);
        cb(inp->mixed_slots, "inp_mixed_slots", -1);
        ggml_set_input(inp->mixed_slots);

        inp->mixed_embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens);
        cb(inp->mixed_embd, "inp_mixed_embd", -1);
        ggml_set_input(inp->mixed_embd);

        // note: set_rows writes into its destination, so it gets a copy of the input
        ggml_tensor * embd_mixed = build_tok(embd2, inp->mixed_tokens);
        inps[2] = ggml_set_rows(ctx0, ggml_dup(ctx0, inp->mixed_embd), embd_mixed, inp->mixed_slots);

        assert(ggml_are_same_shape (inps[0], inps[2]));
        assert(ggml_are_same_stride(inps[0], inps[2]));
    }

    const int idx = ubatch.is_mixed() ? 2 : ubatch.token ? 0 : 1;

    ggml_tensor * cur = ggml_build_forward_select(gf, inps.data(), has_mixed ? 3 : 2, idx);

    if (n_embd_inp != n_embd) {
        cur = ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 0);
    }

    res->t_inp_embd = cur;

    // NOTE: For deepstack models, only apply scale to token inputs (ie text-only input).
    //  Raw embeddings are assumed to be multimodal inputs that should not be scaled.
    const bool scale_tok_only = hparams.f_embedding_scale != 0.0f && hparams.n_deepstack_layers > 0;

    // For Granite architecture
    if (hparams.f_embedding_scale != 0.0f && !scale_tok_only) {
        if (!ggml_is_contiguous(cur)) {
            cur = ggml_cont(ctx0, cur);
        }
        cur = ggml_scale(ctx0, cur, hparams.f_embedding_scale);
    }

    // scale the token rows only, applied after the select so that the graph is the same for any batch contents
    inp->scale_tok = tok_scale*(scale_tok_only ? hparams.f_embedding_scale : 1.0f);
    if (inp->scale_tok != 1.0f) {
        inp->scale_rows = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, ubatch.n_tokens);
        cb(inp->scale_rows, "inp_scale_rows", -1);
        ggml_set_input(inp->scale_rows);

        if (!ggml_is_contiguous(cur)) {
            cur = ggml_cont(ctx0, cur);
        }
        cur = ggml_mul(ctx0, cur, inp->scale_rows);
    }

    cb(cur, "embd", -1);

    res->add_input(std::move(inp));

    // make sure the produced embeddings are immediately materialized in the ggml graph
    // ref: https://github.com/ggml-org/llama.cpp/pull/18599
    ggml_build_forward_expand(gf, cur);

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos() const {
    auto inp = std::make_unique<llm_graph_input_pos>(hparams.n_pos_per_embd(), hparams.rope_section_order);

    auto & cur = inp->pos;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t)n_tokens*hparams.n_pos_per_embd());
    ggml_set_input(cur);
    cb(cur, "inp_pos", -1);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_attn_scale() const {
    auto inp = std::make_unique<llm_graph_input_attn_temp>(hparams.n_attn_temp_floor_scale, hparams.f_attn_temp_scale, hparams.f_attn_temp_offset);

    auto & cur = inp->attn_scale;

    // this need to be 1x1xN for broadcasting
    cur = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
    ggml_set_input(cur);
    cb(cur, "inp_attn_scale", -1);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_out_ids() const {
    // note: when all tokens are output, we could skip this optimization to spare the ggml_get_rows() calls,
    //       but this would make the graph topology depend on the number of output tokens, which can interfere with
    //       features that require constant topology such as pipeline parallelism
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14275#issuecomment-2987424471
    //if (n_outputs < n_tokens) {
    //    return nullptr;
    //}

    auto inp = std::make_unique<llm_graph_input_out_ids>(hparams, cparams, n_outputs);

    auto & cur = inp->out_ids;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
    ggml_set_input(cur);
    ggml_set_name(cur, "out_ids");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_mean() const {
    auto inp = std::make_unique<llm_graph_input_mean>(cparams);

    auto & cur = inp->mean;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, ubatch.n_seqs_unq);
    ggml_set_input(cur);
    ggml_set_name(cur, "mean");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cls() const {
    auto inp = std::make_unique<llm_graph_input_cls>(cparams, arch);

    auto & cur = inp->cls;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_seqs_unq);
    ggml_set_input(cur);
    ggml_set_name(cur, "cls");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cross_embd() const {
    auto inp = std::make_unique<llm_graph_input_cross_embd>(cross);

    auto & cur = inp->cross_embd;

    // if we have the output embeddings from the encoder, use them directly
    // TODO: needs more work to be correct, for now just use the tensor shape
    //if (cross->t_embd) {
    //    cur = ggml_view_tensor(ctx0, cross->t_embd);

    //    return cur;
    //}

    const auto n_embd = !cross->v_embd.empty() ? cross->n_embd : hparams.n_embd_inp();
    const auto n_enc  = !cross->v_embd.empty() ? cross->n_enc  : hparams.n_ctx_train;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_enc);
    ggml_set_input(cur);
    ggml_set_name(cur, "cross_embd");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_enc() const {
    auto inp = std::make_unique<llm_graph_input_pos_bucket>(hparams);

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    ggml_set_input(cur);
    ggml_set_name(cur, "pos_bucket_enc");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_dec() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_pos_bucket_kv>(hparams, mctx_cur);

    const auto n_kv = mctx_cur->get_n_kv();

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_tokens);
    ggml_set_input(cur);
    ggml_set_name(cur, "pos_bucket_dec");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_pos_bias(ggml_tensor * pos_bucket, ggml_tensor * attn_rel_b) const {
    ggml_tensor * pos_bucket_1d = ggml_reshape_1d(ctx0, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1]);
    cb(pos_bucket_1d, "pos_bucket_1d", -1);

    ggml_tensor * pos_bias = ggml_get_rows(ctx0, attn_rel_b, pos_bucket_1d);

    pos_bias = ggml_reshape_3d(ctx0, pos_bias, pos_bias->ne[0], pos_bucket->ne[0], pos_bucket->ne[1]);
    pos_bias = ggml_permute   (ctx0, pos_bias, 2, 0, 1, 3);
    pos_bias = ggml_cont      (ctx0, pos_bias);

    cb(pos_bias, "pos_bias", -1);

    return pos_bias;
}

ggml_tensor * llm_graph_context::build_attn_bias_tail(
        ggml_tensor * kq_b, ggml_tensor * bias_read_idxs, ggml_tensor * kq_mask_tail) const {
    if (!kq_b || !kq_mask_tail) {
        return nullptr;
    }

    GGML_ASSERT(bias_read_idxs && bias_read_idxs->type == GGML_TYPE_I32);
    GGML_ASSERT(bias_read_idxs->ne[0] == kq_mask_tail->ne[0]);
    GGML_ASSERT(bias_read_idxs->ne[1] == kq_mask_tail->ne[1]*kq_mask_tail->ne[3]);
    GGML_ASSERT(kq_b->ne[2] == kq_mask_tail->ne[1] && kq_b->ne[3] == kq_mask_tail->ne[3]);

    // Repack [body, head, query, stream] into vectors of heads indexed by the
    // flattened (body, query, stream) row selected during mask preparation.
    ggml_tensor * bias_rows = ggml_cont(ctx0, ggml_permute(ctx0, kq_b, 1, 0, 2, 3));
    bias_rows = ggml_reshape_2d(ctx0, bias_rows, kq_b->ne[1],
            kq_b->ne[0]*kq_b->ne[2]*kq_b->ne[3]);
    ggml_tensor * flat_idxs = ggml_reshape_1d(ctx0, bias_read_idxs, ggml_nelements(bias_read_idxs));
    ggml_tensor * tail = ggml_get_rows(ctx0, bias_rows, flat_idxs);
    tail = ggml_reshape_4d(ctx0, tail, kq_b->ne[1], kq_mask_tail->ne[0],
            kq_mask_tail->ne[1], kq_mask_tail->ne[3]);
    return ggml_permute(ctx0, tail, 1, 2, 0, 3);
}

ggml_tensor * llm_graph_context::build_attn_mha_paged(
         ggml_tensor * q,               // [n_embd_head, n_head, n_tokens]
         ggml_tensor * k_cur,           // [n_embd_head, n_head_kv, n_tokens]
         ggml_tensor * v_cur,           // [n_embd_head, n_head_kv, n_tokens]
         ggml_tensor * k_cache,         // master K buffer
         ggml_tensor * v_cache,         // master V buffer
         ggml_tensor * block_table,     // [max_blocks, batch_size]
         ggml_tensor * write_slots,     // [n_tokens]
         ggml_tensor * context_lens,    // [batch_size]
         ggml_tensor * batch_offsets,   // [batch_size]
         ggml_tensor * batch_lens,      // [batch_size]
               float   kq_scale,
                 int   block_size,
                 int   max_blocks,
                 int   active_context,
         bool   snapkv_streaming,
         ggml_tensor * snapkv_scores,   // [max_blocks, batch_size] page importance accumulator, nullable
         ggml_tensor * snapkv_token_scores,
         ggml_tensor * snapkv_capture_from,
         ggml_tensor * snapkv_score_slots) const {

    // Paged attention kernel (write) assumes dense layout [n_tokens. n_heads_kv, head_dim].
    // Architectures like (Falcon, GPT-2, etc.) produce KV as views into a fused QKV tensor
    // We force contiguity before passing to kernel.
    // This can be optimized in phase 2.
    k_cur = ggml_cont(ctx0, k_cur);
    v_cur = ggml_cont(ctx0, v_cur);
    q     = ggml_cont(ctx0, q);

    ggml_tensor * cur = ggml_paged_attn(ctx0,
                                        q, k_cur, v_cur, k_cache, v_cache,
                                        block_table, write_slots, context_lens, batch_offsets, batch_lens,
                                        kq_scale, block_size, max_blocks, active_context, snapkv_streaming,
                                        snapkv_scores, snapkv_capture_from, snapkv_score_slots, snapkv_token_scores);
    return cur;
}

ggml_tensor * llm_graph_context::build_attn_mha(
         ggml_tensor * q,
         ggml_tensor * k,
         ggml_tensor * v,
         ggml_tensor * kq_b,
         ggml_tensor * kq_mask,
         ggml_tensor * sinks,
         ggml_tensor * v_mla,
             int64_t   n_kv_max,
               float   kq_scale,
                 int   il,
         ggml_tensor * k_tail,
         ggml_tensor * v_tail,
         ggml_tensor * kq_mask_tail,
         ggml_tensor * kq_b_tail,
         ggml_tensor * tail_read_idxs,
         ggml_tensor * tail_query_order,
         ggml_tensor * tail_run_desc,
         llama_kv_tail_route tail_route,
         enum ggml_flash_attn_ext_kvarn_domain kvarn_domain,
         ggml_tensor * k_tail_current,
         ggml_tensor * v_tail_current,
              uint32_t tail_history_slots,
                   bool tail_bodyless,
        ggml_tensor ** final_attn_op) const {
    const bool v_trans = v->nb[1] > v->nb[2];

    // split the batch into streams if needed
    const auto n_stream = k->ne[3];
    ggml_tensor * q_tail_batched = k_tail && !tail_read_idxs ? ggml_reshape_4d(
            ctx0, q, q->ne[0], q->ne[1], 1, q->ne[2]*q->ne[3]) : nullptr;
    if (q_tail_batched) {
        // Tail records are gathered per query, so the query index is the
        // outer batch axis while GQA heads retain the regular matmul axis.
        q_tail_batched = ggml_permute(ctx0, q_tail_batched, 0, 2, 1, 3);
    }

    // the stream dim steps over one stream's worth of dim-2 (tokens): (ne[2]/n_stream) rows of stride nb[2].
    // q->nb[3]/n_stream only equals that for a contiguous q; nope-only MLA (glm5-next) passes a permuted
    // q_absorbed where nb[3] != ne[2]*nb[2], so using nb[3] read another head's queries for streams s >= 1.
    q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream, q->nb[1], q->nb[2], q->nb[2]*(q->ne[2]/n_stream), 0);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);
    if (k_tail || v_tail || kq_mask_tail) {
        if (!k_tail || !v_tail || !kq_mask_tail) {
            throw std::logic_error(format(
                    "incomplete KV tail graph inputs at layer %d: k=%d v=%d mask=%d route=%d tokens=%u",
                    il, k_tail != nullptr, v_tail != nullptr, kq_mask_tail != nullptr,
                    int(tail_route), tail_history_slots));
        }
        k_tail = ggml_permute(ctx0, k_tail, 0, 2, 1, 3);
        v_tail = ggml_permute(ctx0, v_tail, 0, 2, 1, 3);
    }
    if (k_tail_current || v_tail_current) {
        GGML_ASSERT(k_tail_current && v_tail_current && k_tail);
        k_tail_current = ggml_permute(ctx0, k_tail_current, 0, 2, 1, 3);
        v_tail_current = ggml_permute(ctx0, v_tail_current, 0, 2, 1, 3);
    }

    ggml_tensor * cur;

    const bool use_native_tail = k_tail != nullptr && tail_route == LLAMA_KV_TAIL_ROUTE_NATIVE;
    GGML_ASSERT(!k_tail || tail_route != LLAMA_KV_TAIL_ROUTE_NONE);
    GGML_ASSERT(!tail_read_idxs || use_native_tail);
    const bool use_flash_attn = cparams.flash_attn && kq_b == nullptr && (k_tail == nullptr || use_native_tail);
    if (getenv("F4_TAIL_TRACE")) {
        fprintf(stderr, "F4MHA il=%d flash=%d kq_b=%p k_tail=%p route=%d idx=%p "
                "native=%d use_fa=%d mask=%p n_kv=%lld\n",
                il, (int) cparams.flash_attn, (void *) kq_b, (void *) k_tail,
                (int) tail_route, (void *) tail_read_idxs, (int) use_native_tail,
                (int) use_flash_attn, (void *) kq_mask, (long long) n_kv_max);
    }
    if (use_flash_attn) {
        GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");

        if (v_trans) {
            v = ggml_transpose(ctx0, v);
        }

        // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        }

        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        }

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                                  hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        GGML_ASSERT(cparams.kvarn.window_chunk <= uint32_t(INT32_MAX));
        cur->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_KVARN_WINDOW_CHUNK] =
                static_cast<int32_t>(cparams.kvarn.window_chunk);
        if (kvarn_domain != GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_AUTO) {
            cur->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_KVARN_DOMAIN] = (int32_t) kvarn_domain;
        }
        // Raw FlashAttention output, before any native KV-tail merge rewrites
        // the node. This is the operator boundary the fixture dump compares.
        cb(cur, "fa_out", il);
        if (use_native_tail) {
            GGML_ASSERT(tail_query_order && tail_run_desc);
            cur = k_tail_current ?
                    ggml_kv_tail_attention_merge_segmented(
                            ctx0, cur, k_tail, v_tail, k_tail_current, v_tail_current,
                            kq_mask_tail, tail_query_order, tail_run_desc) :
                    ggml_kv_tail_attention_merge(
                            ctx0, cur, k_tail, v_tail, kq_mask_tail, tail_query_order, tail_run_desc);
            if (tail_bodyless) {
                ggml_flash_attn_ext_set_kv_tail_bodyless(cur);
            }
            if (k_tail_current) {
                ggml_flash_attn_ext_set_kv_tail_history_slots(cur, int32_t(tail_history_slots));
            }
        }
        res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});

        ggml_flash_attn_ext_add_sinks(cur, sinks);
        GGML_ASSERT(n_kv_max >= 0 && n_kv_max <= INT32_MAX);
        ggml_flash_attn_ext_set_n_kv_max(cur, static_cast<int32_t>(n_kv_max));
        ggml_prec_set_acc(cur, GGML_PREC_F32);

        if (final_attn_op) {
            *final_attn_op = cur;
            if (getenv("F4_TAIL_TRACE")) {
                fprintf(stderr, "F4SET il=%d final=%p cur=%p merged=%d\n",
                        il, (void *) final_attn_op, (void *) cur, (int) use_native_tail);
            }
        }

        if (v_mla) {
#if 0
            // v_mla can be applied as a matrix-vector multiplication with broadcasting across dimension 3 == n_tokens.
            // However, the code is optimized for dimensions 0 and 1 being large, so this is inefficient.
            cur = ggml_reshape_4d(ctx0, cur, v_mla->ne[0], 1, n_head, n_tokens);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
#else
            // It's preferable to do the calculation as a matrix-matrix multiplication with n_tokens in dimension 1.
            // The permutations are noops and only change how the tensor data is interpreted.
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
            cb(cur, "fattn_mla", il);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont(ctx0, cur); // Needed because ggml_reshape_2d expects contiguous inputs.
#endif
        }

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    } else {
        GGML_ASSERT(!tail_read_idxs && "indexed KV tails require native backend attention support");
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        ggml_prec_set_acc(kq, GGML_PREC_F32);
        cb(kq, "kq", il);

        const int64_t n_kv_body = kq->ne[0];
        if (k_tail) {
            ggml_tensor * kq_tail = ggml_mul_mat(ctx0, k_tail, q_tail_batched);
            ggml_prec_set_acc(kq_tail, GGML_PREC_F32);
            kq_tail = ggml_reshape_4d(ctx0, kq_tail,
                    kq_tail->ne[0], q->ne[2], q->ne[1], q->ne[3]);
            kq_tail = ggml_permute(ctx0, kq_tail, 0, 2, 1, 3);
            kq = ggml_concat(ctx0, kq, kq_tail, 0);
            kq_mask = ggml_concat(ctx0, kq_mask, kq_mask_tail, 0);
            if (kq_b) {
                GGML_ASSERT(kq_b_tail);
                kq_b = ggml_concat(ctx0, kq_b, kq_b_tail, 0);
            }
        }

        // note: this op tends to require high floating point range
        //       while for some models F16 is enough, for others it is not, so we default to F32 here
        ggml_prec_set_acc(kq, GGML_PREC_F32);

        if (arch == LLM_ARCH_GROK) {
            // need to do the following:
            // multiply by attn_output_multiplier
            // and then :
            // kq = 30 * tanh(kq / 30)
            // before the softmax below

            kq = ggml_tanh(ctx0, ggml_scale(ctx0, kq, hparams.f_attn_out_scale / hparams.f_attn_logit_softcapping));
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled", il);
        }

        if (hparams.attn_soft_cap) {
            kq = ggml_scale(ctx0, kq, 1.0f / hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_1", il);
            kq = ggml_tanh (ctx0, kq);
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_2", il);
        }

        if (kq_b) {
            kq = ggml_add(ctx0, kq, kq_b);
            cb(kq, "kq_plus_kq_b", il);
        }

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        ggml_soft_max_add_sinks(kq, sinks);
        cb(kq, "kq_soft_max", il);

        const bool quant_v_body = v_tail && ggml_is_quantized(v->type);
        if (!v_trans && !quant_v_body) {
            // note: avoid this branch
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
            cb(v, "v_cont", il);
        }

        ggml_tensor * kqv;
        if (v_tail) {
            const int64_t n_kv_tail = kq->ne[0] - n_kv_body;
            ggml_tensor * weights_body = ggml_view_4d(ctx0, kq,
                    n_kv_body, kq->ne[1], kq->ne[2], kq->ne[3],
                    kq->nb[1], kq->nb[2], kq->nb[3], 0);
            ggml_tensor * weights_tail = ggml_view_4d(ctx0, kq,
                    n_kv_tail, kq->ne[1], kq->ne[2], kq->ne[3],
                    kq->nb[1], kq->nb[2], kq->nb[3], n_kv_body*kq->nb[0]);
            v_tail = ggml_cont(ctx0, ggml_transpose(ctx0, v_tail));
            ggml_tensor * body_out = quant_v_body ?
                    ggml_out_prod(ctx0, v, ggml_transpose(ctx0, weights_body)) :
                    ggml_mul_mat(ctx0, v, weights_body);
            weights_tail = ggml_cont(ctx0, ggml_permute(ctx0, weights_tail, 0, 2, 1, 3));
            weights_tail = ggml_reshape_4d(ctx0, weights_tail,
                    weights_tail->ne[0], 1, weights_tail->ne[1],
                    weights_tail->ne[2]*weights_tail->ne[3]);
            ggml_tensor * tail_out = ggml_mul_mat(ctx0, v_tail, weights_tail);
            tail_out = ggml_reshape_4d(ctx0, tail_out,
                    tail_out->ne[0], body_out->ne[2], body_out->ne[1], body_out->ne[3]);
            tail_out = ggml_permute(ctx0, tail_out, 0, 2, 1, 3);
            kqv = ggml_add(ctx0, body_out, tail_out);
        } else {
            kqv = ggml_mul_mat(ctx0, v, kq);
        }
        cb(kqv, "kqv", il);

        // for MLA with the absorption optimization, we need to "decompress" from MQA back to MHA
        if (v_mla) {
            kqv = ggml_mul_mat(ctx0, v_mla, kqv);
            cb(kqv, "kqv_mla", il);
        }

        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);

        // recombine streams
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

        if (!cparams.offload_kqv) {
            // all nodes between the KV store and the attention output are run on the CPU
            ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
        }
    }

    ggml_build_forward_expand(gf, cur);

    return cur;
}

static void validate_native_kvarn_noncausal_operation(
        const llama_kv_cache_context * mctx, int32_t il, ggml_tensor * op) {
    const auto * cache = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx);
    auto * dev = cache ? cache->native_attention_backend(il) : nullptr;
    if (op && op->op == GGML_OP_FLASH_ATTN_EXT) {
        op->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_KVARN_NON_CAUSAL_MASK] = 1;
    }
    if (!dev || !op || op->op != GGML_OP_FLASH_ATTN_EXT ||
            !ggml_backend_dev_supports_op(dev, op)) {
        throw std::runtime_error(format(
                "KVarN non-causal layer %d final native attention operation is unsupported by %s",
                il, dev ? ggml_backend_dev_name(dev) : "unknown"));
    }
}

// `ubatch_writes_local_range` states, from the UBATCH alone (not from what was
// built), whether the tokens this graph stores belong to the local range. The
// plan's route.has_current is a per-layer property meaning "a graph that writes
// the local range carries a current K/V segment"; a position-split OVERFLOW
// ubatch writes Q4 overflow keys instead, so its local tail is read purely as
// HISTORY and must carry no current segment. That is made explicit here rather
// than relaxing the guard: a present or half-present current segment in such a
// graph still fails, as does any mismatch in body/exact types, has_body or
// body_execution_rows.
static void validate_native_tail_operation(
        const llama_kv_cache_context * mctx, int32_t il, const ggml_tensor * op,
        bool ubatch_writes_local_range) {
    const auto * route = mctx->get_tail_layer_route(il);
    if (!route || route->capability.route != LLAMA_KV_TAIL_ROUTE_NATIVE) {
        return;
    }
    if (!op || op->op != GGML_OP_FLASH_ATTN_EXT) {
        throw std::logic_error(format(
                "KV tail layer %d advertised native attention but built no final FlashAttention operation", il));
    }
    const bool has_current = op->src[10] != nullptr || op->src[11] != nullptr;
    const bool expect_current = route->has_current && ubatch_writes_local_range;
    if ((op->src[10] == nullptr) != (op->src[11] == nullptr) ||
            has_current != expect_current) {
        throw std::logic_error(format(
                "KV tail layer %d current-segment descriptor does not match the final operation", il));
    }
    if (!op->src[1] || !op->src[2] || !op->src[5] || !op->src[6] ||
            op->src[1]->type != route->body_type_k || op->src[2]->type != route->body_type_v ||
            op->src[5]->type != route->exact_type_k || op->src[6]->type != route->exact_type_v ||
            route->has_body != mctx->has_kv_body(il) ||
            route->body_execution_rows != mctx->get_tail_body_execution_rows(il)) {
        // Report every compared field: the guard must stay fail-closed, and a
        // message that only says "does not match" leaves the operator to guess
        // which of the six conditions moved.
        throw std::logic_error(format(
                "KV tail layer %d execution descriptor does not match final tensor types or extents: "
                "k_src_type=%d route_body_k=%d v_src_type=%d route_body_v=%d "
                "kt_src_type=%d route_exact_k=%d vt_src_type=%d route_exact_v=%d "
                "has_body_src=%d route_has_body=%d rows_src=%d route_rows=%d",
                il,
                op->src[1] ? (int) op->src[1]->type : -1, (int) route->body_type_k,
                op->src[2] ? (int) op->src[2]->type : -1, (int) route->body_type_v,
                op->src[5] ? (int) op->src[5]->type : -1, (int) route->exact_type_k,
                op->src[6] ? (int) op->src[6]->type : -1, (int) route->exact_type_v,
                (int) mctx->has_kv_body(il), (int) route->has_body,
                (int) mctx->get_tail_body_execution_rows(il),
                (int) route->body_execution_rows));
    }
    auto * dev = route->owner;
    if (!dev || !ggml_backend_dev_supports_op(dev, op)) {
        throw std::runtime_error(format(
                "KV tail layer %d native operation is unsupported by its planned backend %s; "
                "refusing scheduler fallback",
                il, dev ? ggml_backend_dev_name(dev) : "unknown"));
    }
    LLAMA_LOG_DEBUG("KV tail final: layer=%d dev=%s route=native body=%s current=%s "
            "body_type=%s/%s exact_type=%s/%s body_rows=%u supported=1\n",
            il, ggml_backend_dev_name(dev), mctx->has_kv_body(il) ? "present" : "bodyless",
            has_current ? "present" : "absent",
            ggml_type_name(op->src[1]->type), ggml_type_name(op->src[2]->type),
            ggml_type_name(op->src[5]->type), ggml_type_name(op->src[6]->type),
            mctx->get_tail_body_execution_rows(il));
}

llm_graph_input_attn_no_cache * llm_graph_context::build_attn_inp_no_cache() const {
    auto inp = std::make_unique<llm_graph_input_attn_no_cache>(hparams, cparams);

    // flash attention requires an f16 mask
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    // note: there is no KV cache, so the KV values are the tokens of the batch, padded like the KV cache (see llama_kv_cache::get_n_kv)
    const int64_t n_kv = GGML_PAD(n_tokens, 256);

    inp->self_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_kv, n_tokens, 1, 1);
    ggml_set_input(inp->self_kq_mask);
    cb(inp->self_kq_mask, "self_kq_mask", -1);

    inp->self_kq_mask_cnv = inp->self_kq_mask;

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        inp->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, type_mask, n_kv, n_tokens, 1, 1);
        ggml_set_input(inp->self_kq_mask_swa);

        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    } else {
        inp->self_kq_mask_swa     = nullptr;
        inp->self_kq_mask_swa_cnv = nullptr;
    }

    return (llm_graph_input_attn_no_cache *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_no_cache * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    GGML_UNUSED(n_tokens);

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const bool is_swa = hparams.is_swa(il);

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    // [TAG_NO_CACHE_PAD]
    // TODO: if ubatch.equal_seqs() == true, we can split the three tensors below into ubatch.n_seqs_unq streams
    //       but it might not be worth it: https://github.com/ggml-org/llama.cpp/pull/15636
    //assert(!ubatch.equal_seqs() || (k_cur->ne[3] == 1 && k_cur->ne[3] == ubatch.n_seqs_unq));

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    // the KV values and the KQ bias are padded with zeros up to the KV dimension of the mask
    const int64_t n_kv_pad = kq_mask->ne[0] - k->ne[2];

    k = ggml_pad(ctx0, k, 0, 0, n_kv_pad, 0);
    v = ggml_pad(ctx0, v, 0, 0, n_kv_pad, 0);

    if (kq_b) {
        kq_b = ggml_pad(ctx0, kq_b, n_kv_pad, 0, 0, 0);
    }

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static void build_attn_inp_tail(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_kv_cache_context * mctx,
             ggml_tensor * kq_mask,
             ggml_tensor *& tail_idxs,
             ggml_tensor *& tail_read_idxs,
             ggml_tensor *& tail_body_read_idxs,
             ggml_tensor *& tail_bias_read_idxs,
             ggml_tensor *& kq_mask_tail,
             ggml_tensor *& tail_query_order,
             ggml_tensor *& tail_run_desc,
                    bool   pack_sparse_body) {
    if (mctx->get_tail_tokens() == 0) {
        // No exact tail: leave all tail inputs null so set_input skips them.
        // (Previously tail_idxs was built unconditionally, leaving a
        // buffer-less tensor that crashed set_input when tails are disabled,
        // e.g. position-split mode.)
        return;
    }
    tail_idxs = mctx->build_input_tail_idxs(ctx0, ubatch);
    const auto [q_max, n_active] = tail_query_plan_shape(ubatch);
    GGML_ASSERT(q_max > 0 && n_active > 0);
    const uint32_t arena_stride = mctx->get_tail_arena_stride();
    const uint32_t body_execution_stride = mctx->get_tail_body_execution_stride();
    const uint32_t attention_stride = mctx->get_tail_attention_stride(
            uint32_t(tail_extent_queries(mctx, ubatch, q_max)));
    GGML_ASSERT(attention_stride >= mctx->get_tail_tokens());
    GGML_ASSERT(mctx->has_compact_tail() || arena_stride >= attention_stride);
    tail_read_idxs = ggml_new_tensor_2d(
            ctx0, GGML_TYPE_I32, attention_stride, ubatch.n_tokens);
    tail_body_read_idxs = ggml_new_tensor_2d(
            ctx0, GGML_TYPE_I32, attention_stride, ubatch.n_tokens);
    tail_bias_read_idxs = ggml_new_tensor_2d(
            ctx0, GGML_TYPE_I32, attention_stride, ubatch.n_tokens);
    kq_mask_tail = ggml_new_tensor_4d(ctx0, kq_mask->type,
            attention_stride, kq_mask->ne[1], 1, kq_mask->ne[3]);
    for (ggml_tensor * input : {
            tail_read_idxs, tail_body_read_idxs, tail_bias_read_idxs, kq_mask_tail }) {
        ggml_set_input(input);
    }
    if (!tail_query_order) {
        tail_query_order = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, q_max, n_active);
        ggml_set_input(tail_query_order);
    }
    if (!tail_run_desc) {
        const bool sparse_body = pack_sparse_body && mctx->can_pack_tail_body(ubatch);
        GGML_ASSERT(!sparse_body || body_execution_stride >= arena_stride);
        const int64_t desc_stride = sparse_body ?
                int64_t(6 + attention_stride + body_execution_stride) : int64_t(6 + attention_stride);
        tail_run_desc = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, desc_stride, n_active);
        ggml_set_input(tail_run_desc);
        LLAMA_LOG_DEBUG("%s: standard KV tail body route=%s physical-capacity-bound=%u\n",
                __func__, sparse_body ? "sparse-packed" : "ordinary", body_execution_stride);
    }
}

static ggml_tensor * prepare_compact_tail_current(
        ggml_context * ctx0, ggml_tensor * persistent, ggml_tensor * current) {
    GGML_ASSERT(persistent && current);
    GGML_ASSERT(persistent->ne[0] == current->ne[0] &&
            persistent->ne[1] == current->ne[1] &&
            persistent->ne[3] == 1 && current->ne[3] == 1);
    if (current->type != persistent->type) {
        current = ggml_cast(ctx0, current, persistent->type);
    }
    current = ggml_cont(ctx0, current);
    current = ggml_reshape_4d(ctx0, current,
            persistent->ne[0], persistent->ne[1], current->ne[2], 1);
    return current;
}

static void build_empty_kv_body(
        ggml_context * ctx0,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_type body_type_k,
        ggml_type body_type_v,
        ggml_tensor * body_mask_template,
        ggml_tensor * body_bias_template,
        ggml_tensor *& k,
        ggml_tensor *& v,
        ggml_tensor *& body_mask,
        ggml_tensor *& body_bias) {
    GGML_ASSERT(k_cur && v_cur && body_mask_template);
    GGML_ASSERT((body_type_k == GGML_TYPE_F16 || body_type_k == GGML_TYPE_BF16) &&
                (body_type_v == GGML_TYPE_F16 || body_type_v == GGML_TYPE_BF16));
    const int64_t n_stream = body_mask_template->ne[3];
    const auto zero_anchor = [&](ggml_type type, const ggml_tensor * current) {
        ggml_tensor * anchor = ggml_fill(ctx0, ggml_new_tensor_4d(
                ctx0, GGML_TYPE_F16, current->ne[0], current->ne[1], 1, n_stream), 0.0f);
        return type == GGML_TYPE_F16 ? anchor : ggml_cast(ctx0, anchor, type);
    };
    const auto first_row_anchor = [&](ggml_tensor * source) {
        ggml_tensor * row = ggml_view_4d(ctx0, source,
                1, source->ne[1], source->ne[2], source->ne[3],
                source->nb[1], source->nb[2], source->nb[3], 0);
        return ggml_cont(ctx0, row);
    };
    k = zero_anchor(body_type_k, k_cur);
    v = zero_anchor(body_type_v, v_cur);
    // Keep the authoritative planner inputs live even though the body itself
    // is empty. Otherwise graph allocation prunes them and set_input() cannot
    // populate the exact-tail mask and run descriptor derived from them.
    body_mask = ggml_fill(ctx0, first_row_anchor(body_mask_template), -INFINITY);
    body_bias = body_bias_template ?
            ggml_fill(ctx0, first_row_anchor(body_bias_template), 0.0f) : nullptr;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv_paged * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * /*kq_b*/,
        ggml_tensor * /*sinks*/,
        ggml_tensor * /*v_mla*/,
            float     kq_scale,
            int       il) const {
    GGML_ASSERT(inp && "attn input is nullptr.");
    auto * paged_mctx = inp->mctx;
    GGML_ASSERT(paged_mctx && "llama_kv_cache_paged_context is nullptr.");

    const int32_t max_blocks = paged_mctx->get_max_blocks();
    const int32_t * context_lens = paged_mctx->get_context_lens();
    int32_t active_context = max_blocks * cparams.block_size;
    if (context_lens) {
        active_context = 0;
        for (int32_t i = 0; i < paged_mctx->get_batch_size(); ++i) {
            active_context = std::max(active_context, context_lens[i]);
        }
    }
    ggml_tensor * k_physical = paged_mctx->get_k(il); // interleaved KV
    ggml_tensor * v_physical = paged_mctx->get_v(il); // interleaved KV

    ggml_tensor * cur = build_attn_mha_paged(
        q_cur, k_cur, v_cur, k_physical, v_physical,
        inp->paged_block_table,
        inp->paged_write_slots,
        inp->paged_context_lens,
        inp->paged_batch_offsets,
        inp->paged_batch_lens,
        kq_scale, cparams.block_size, max_blocks, active_context, paged_mctx->get_snapkv_streaming(),
        paged_mctx->get_snapkv_scores(), paged_mctx->get_snapkv_token_scores(),
        paged_mctx->get_snapkv_capture_from(), paged_mctx->get_snapkv_score_slots());
    inp->paged_attn_nodes.push_back(cur);
    cb(cur, "kqv_out", il);

    // Reshape to [n_embd_head * n_head, n_tokens] (just a view)
    // note: for some archs (e.g. QWEN35) n_head * n_embd_head != n_embd
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], n_tokens);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_kv_paged> build_attn_inp_kv_paged_impl(
           ggml_context * ctx0,
    const llama_cparams & cparams,
    const llama_hparams & hparams,
    const llama_kv_cache_paged_context * mctx_paged);

static std::unique_ptr<llm_graph_input_attn_kv> build_attn_inp_kv_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_kv>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
        build_attn_inp_tail(ctx0, ubatch, mctx_cur, inp->self_kq_mask,
                inp->self_tail_idxs, inp->self_tail_read_idxs, inp->self_tail_body_read_idxs,
                inp->self_tail_bias_read_idxs, inp->self_kq_mask_tail,
                inp->self_tail_query_order, inp->self_tail_run_desc, true);
    }

    inp->self_k_rot = mctx_cur->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_cur->build_input_v_rot(ctx0);
    if (const auto * kvarn = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx_cur)) {
        if (kvarn->has_standard_cache()) {
            inp->self_kq_mask_standard = ggml_new_tensor_4d(ctx0, inp->self_kq_mask->type,
                    inp->self_kq_mask->ne[0], inp->self_kq_mask->ne[1],
                    inp->self_kq_mask->ne[2], inp->self_kq_mask->ne[3]);
            ggml_set_input(inp->self_kq_mask_standard);
            ggml_set_name(inp->self_kq_mask_standard, "attn_inp_kq_mask_standard_qx");
            inp->self_kq_mask_standard_cnv = inp->self_kq_mask_standard;
        }
        inp->self_kvarn_rot_64  = kvarn->build_input_kvarn_rot(ctx0, 64);
        inp->self_kvarn_rot_128 = kvarn->build_input_kvarn_rot(ctx0, 128);
        inp->self_kvarn_rot_256 = kvarn->build_input_kvarn_rot(ctx0, 256);
        inp->self_kvarn_rot_512 = kvarn->build_input_kvarn_rot(ctx0, 512);
        if (kvarn->uses_materialization_indices()) {
            inp->self_kvarn_mat_idxs = kvarn->build_input_kvarn_mat_idxs(ctx0);
            kvarn->set_mat_idxs(inp->self_kvarn_mat_idxs);
        }
        // F3 position-split range inputs (P-mode only). Sized to the RANGE
        // CAPACITY (P and C-P), never to live counts: history can grow
        // between graph build and set_input (pipelined prepare), so exact
        // sizing races. set_input fills the live prefix and masks the rest
        // as empty (-inf); FA cost scales with capacity, acceptable for
        // validation (a future optimization rebuilds on count change).
        // Both ranges must be non-empty... (see below; empty live lists are
        // fine, tensors still built at capacity).
        if (kvarn->position_split_enabled()) {
            const auto * kvarn_cache = kvarn->get_kvarn_cache();
            if (kvarn_cache) {
                const uint32_t p = kvarn_cache->position_split_p();
                const uint32_t c = kvarn_cache->get_kv_size();
                const int64_t n_q = ubatch.n_tokens;
                const int64_t n_stream = 1;
                inp->self_kq_mask_local = ggml_new_tensor_4d(ctx0,
                        inp->self_kq_mask->type,
                        int64_t(p), n_q, 1, n_stream);
                ggml_set_input(inp->self_kq_mask_local);
                ggml_set_name(inp->self_kq_mask_local, "attn_inp_kq_mask_local");
                inp->self_kq_mask_local_cnv = inp->self_kq_mask_local;
                inp->self_kq_mask_overflow = ggml_new_tensor_4d(ctx0,
                        inp->self_kq_mask->type,
                        int64_t(c - p), n_q, 1, n_stream);
                ggml_set_input(inp->self_kq_mask_overflow);
                ggml_set_name(inp->self_kq_mask_overflow, "attn_inp_kq_mask_overflow");
                inp->self_kq_mask_overflow_cnv = inp->self_kq_mask_overflow;
                inp->self_kvarn_mat_idxs_local = ggml_new_tensor_1d(ctx0,
                        GGML_TYPE_I64, int64_t(p));
                ggml_set_input(inp->self_kvarn_mat_idxs_local);
                ggml_set_name(inp->self_kvarn_mat_idxs_local, "attn_inp_kvarn_mat_idxs_local");
                inp->self_overflow_cell_idxs = ggml_new_tensor_1d(ctx0,
                        GGML_TYPE_I32, int64_t(c - p));
                ggml_set_input(inp->self_overflow_cell_idxs);
                ggml_set_name(inp->self_overflow_cell_idxs, "attn_inp_overflow_cell_idxs");
                // Identity-row input for the packed view read (tiny: one i32 per
                // overflow cell at capacity). The view rows are fed to attention as
                // F16 through the supported GET_ROWS path; see the builder.
                inp->self_ovf_view_rows = ggml_new_tensor_1d(ctx0,
                        GGML_TYPE_I32, int64_t(c - p));
                ggml_set_input(inp->self_ovf_view_rows);
                ggml_set_name(inp->self_ovf_view_rows, "attn_inp_ovf_view_rows");
                // Cell lists refreshed in set_input/can_reuse; leave empty here.
            }
        }
    }

    return inp;
}

llm_graph_input_attn_kv * llm_graph_context::build_attn_inp_kv() const {
    if (auto * mctx_paged = dynamic_cast<const llama_kv_cache_paged_context *>(mctx)) {
        auto inp = build_attn_inp_kv_paged_impl(ctx0, cparams, hparams, mctx_paged);
        return (llm_graph_input_attn_kv *) res->add_input(std::move(inp));
    }

    if (auto * mctx_hybrid = dynamic_cast<const llama_memory_hybrid_paged_context *>(mctx)) {
        auto inp = build_attn_inp_kv_paged_impl(ctx0, cparams, hparams, mctx_hybrid->get_attn());
        return (llm_graph_input_attn_kv *) res->add_input(std::move(inp));
    }

    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_kv *) res->add_input(std::move(inp));
}

void llm_graph_context::build_kv_store(
        const llama_kv_cache_context * mctx_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * k_idxs,
        ggml_tensor * v_idxs,
        ggml_tensor * tail_idxs,
        int32_t il) const {
    GGML_ASSERT(mctx_cur != nullptr);
    GGML_ASSERT(k_cur != nullptr && v_cur != nullptr);
    GGML_ASSERT(k_idxs != nullptr && v_idxs != nullptr);

    const bool has_exact_tail = mctx_cur->get_tail_tokens(il) > 0;
    bool k_tail_scheduled = false;
    bool v_tail_scheduled = false;

    ggml_tensor * k_written = mctx_cur->cpy_k_with_tail(ctx0, k_cur, k_idxs, tail_idxs, il);
    if (k_written == nullptr) {
        k_written = mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il);
        if (k_written != nullptr) {
            ggml_build_forward_expand(gf, k_written);
        }
        if (ggml_tensor * tail_written = mctx_cur->cpy_k_tail(
                    ctx0, k_cur, tail_idxs, il, k_written)) {
            ggml_build_forward_expand(gf, tail_written);
            k_tail_scheduled = true;
        }
    } else {
        ggml_build_forward_expand(gf, k_written);
        k_tail_scheduled = has_exact_tail;
    }

    ggml_tensor * v_written = mctx_cur->cpy_v_with_tail(ctx0, v_cur, v_idxs, tail_idxs, il);
    if (v_written == nullptr) {
        v_written = mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il);
        if (v_written != nullptr) {
            ggml_build_forward_expand(gf, v_written);
        }
        if (ggml_tensor * tail_written = mctx_cur->cpy_v_tail(
                    ctx0, v_cur, tail_idxs, il, v_written)) {
            ggml_build_forward_expand(gf, tail_written);
            v_tail_scheduled = true;
        }
    } else {
        ggml_build_forward_expand(gf, v_written);
        v_tail_scheduled = has_exact_tail;
    }

    GGML_ASSERT(!has_exact_tail || (k_tail_scheduled && v_tail_scheduled));
    LLAMA_LOG_DEBUG("%s: layer %d body K/V and exact-tail K/V scheduled=%s\n",
            __func__, il, has_exact_tail ? "yes" : "not-configured");
}

static bool llm_kvarn_native_attention_for(
        const llama_kv_cache_kvarn_context * cache, llm_arch arch, bool causal,
        bool swa, int il, int head_dim) {
    if (!cache || !cache->uses_native_attention(il)) {
        return false;
    }
    if (causal || arch != LLM_ARCH_DFLASH) {
        return llama_kvarn_native_attention_allowed(causal, arch);
    }
    // Test-only materialized oracle for the same persistent records.
    const char * force_materialized = getenv("LLAMA_KVARN_TEST_MATERIALIZE_NONCAUSAL");
    if (force_materialized && atoi(force_materialized) != 0) {
        return false;
    }
    const bool owned_dflash = cache->has_qualified_dflash_mask();
    return llama_kvarn_native_attention_allowed({
        owned_dflash ? (swa ? LLAMA_KVARN_MASK_DFLASH_SWA : LLAMA_KVARN_MASK_DFLASH_BLOCK) :
            LLAMA_KVARN_MASK_UNSUPPORTED,
        owned_dflash,
        cache->uses_native_attention(il),
        llama_kvarn_backend_supports_non_causal_mask(cache->native_attention_backend(il)),
        head_dim,
    });
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla, // TODO: remove
            float     kq_scale,
            int       il,
        ggml_tensor * kq_mask_override) const {
    GGML_ASSERT(v_mla == nullptr);

    if (auto * paged = dynamic_cast<llm_graph_input_attn_kv_paged *>(inp)) {
        return build_attn(paged, wo, wo_b, wo_s, q_cur, k_cur, v_cur, kq_b, sinks, v_mla, kq_scale, il);
    }

    const auto * mctx_cur = inp->mctx;
    const auto * kvarn_ctx = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx_cur);
    const bool use_kvarn = kvarn_ctx && kvarn_ctx->uses_kvarn_layer(il);
    // Backend preferences choose an implementation inside the final operation;
    // they must not veto direct KVarN attention and force full materialization.
    // validate_native_tail_operation() proves the actual attached-tail shape.
    const bool kvarn_native_attention = llm_kvarn_native_attention_for(
        kvarn_ctx, arch, cparams.causal_attn, false, il, (int) q_cur->ne[0]);
    const auto kvarn_plan = use_kvarn ? llama_kvarn_plan_attention(
        kvarn_native_attention,
        kvarn_ctx->native_attention_uses_original_v(il),
        kvarn_ctx->native_rotated_max_query_tokens(il),
        (uint32_t) q_cur->ne[2],
        (int) q_cur->ne[0]) : llama_kvarn_attention_plan {
            false, GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_AUTO };
    if (use_kvarn && arch == LLM_ARCH_DFLASH && !cparams.causal_attn) {
        LLAMA_LOG_DEBUG("%s: DFlash layer %d KVarN attention route=%s\n", __func__, il,
                kvarn_plan.native_attention ? "native" : "materialized");
    }
    const auto kvarn_domain = kvarn_plan.domain;
    const bool use_kvarn_rotated_domain = use_kvarn &&
        kvarn_domain == GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED;
    const bool use_kvarn_mixed_domain = use_kvarn &&
        kvarn_domain == GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V;
    const bool use_kvarn_q_rot = use_kvarn_rotated_domain || use_kvarn_mixed_domain;

    if (use_kvarn) {
        GGML_ASSERT(llama_kvarn_head_dim_supported((int) q_cur->ne[0]));
        GGML_ASSERT(!inp->self_k_rot || kvarn_ctx->has_standard_cache());
        GGML_ASSERT(!inp->self_v_rot || kvarn_ctx->has_standard_cache());
        GGML_ASSERT(!use_kvarn_q_rot || llm_kvarn_rot_for_dim(
                inp->self_kvarn_rot_64, inp->self_kvarn_rot_128,
                inp->self_kvarn_rot_256, inp->self_kvarn_rot_512, q_cur->ne[0]) != nullptr);
    }

    if (!use_kvarn && inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (!use_kvarn && inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    ggml_tensor * k_tail_written = nullptr;
    ggml_tensor * v_tail_written = nullptr;
    const bool compact_tail = mctx_cur->has_compact_tail(il);

    ggml_tensor * k;
    ggml_tensor * v;

    if (cparams.training) {
        GGML_ASSERT(mctx_cur->get_n_kv() == n_tokens);
        k = k_cur;
        v = v_cur;
    } else {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();
        if (compact_tail) {
            if (k_cur) {
                if (ggml_tensor * written = mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il)) {
                    ggml_build_forward_expand(gf, written);
                }
            }
        } else {
            k_tail_written = mctx_cur->cpy_k_with_tail(ctx0, k_cur, k_idxs, inp->self_tail_idxs, il);
            if (k_tail_written) {
                ggml_build_forward_expand(gf, k_tail_written);
            } else {
                ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
                if ((k_tail_written = mctx_cur->cpy_k_tail(ctx0, k_cur, inp->self_tail_idxs, il))) {
                    ggml_build_forward_expand(gf, k_tail_written);
                }
            }
        }
        if (compact_tail) {
            if (v_cur) {
                if (ggml_tensor * written = mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il)) {
                    ggml_build_forward_expand(gf, written);
                }
            }
        } else {
            v_tail_written = mctx_cur->cpy_v_with_tail(ctx0, v_cur, v_idxs, inp->self_tail_idxs, il);
            if (v_tail_written) {
                ggml_build_forward_expand(gf, v_tail_written);
            } else {
                ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
                if ((v_tail_written = mctx_cur->cpy_v_tail(ctx0, v_cur, inp->self_tail_idxs, il))) {
                    ggml_build_forward_expand(gf, v_tail_written);
                }
            }
        }

        k = use_kvarn ?
        kvarn_ctx->get_k_for_attention(ctx0, il, kvarn_plan.native_attention) :
        mctx_cur->get_k(ctx0, il);
        v = use_kvarn ?
        kvarn_ctx->get_v_for_attention(ctx0, il, kvarn_plan.native_attention) :
        mctx_cur->get_v(ctx0, il);

    }

    ggml_tensor * kq_mask = kq_mask_override ? kq_mask_override :
            (kvarn_ctx && kvarn_ctx->uses_standard_layer(il) && !kvarn_ctx->position_split_enabled() ?
                inp->get_standard_kq_mask() : inp->get_kq_mask());
    // Shared graph input: name it without a layer suffix so a later layer
    // renaming the same tensor cannot strand it outside the dump whitelist.
    cb(kq_mask, "kq_mask", -1);

    ggml_tensor * q = q_cur;
    // The K/V that FlashAttention actually consumes in this layer, as opposed
    // to the cache store they were read from. Named ONLY when the fixture
    // capture is off: capture_from_leaf() renames its candidate to "f4kv_<seq>"
    // and BOTH eval_wants_node() and the FlashAttention linkage key off that
    // prefix, so overwriting it here silently emptied the executed-sample
    // registry (F4SELECT routes=0) and every consumer gate reported
    // unavailable data while the decode itself still succeeded.
    if (getenv("F4_TRACE_CAPTURE") == nullptr || getenv("F4_TRACE_CAPTURE")[0] == '\0') {
        cb(k, "fa_kv_k", il);
        cb(v, "fa_kv_v", il);
    }
    ggml_tensor * kvarn_rot = use_kvarn ? llm_kvarn_rot_for_dim(
            inp->self_kvarn_rot_64, inp->self_kvarn_rot_128,
            inp->self_kvarn_rot_256, inp->self_kvarn_rot_512, q->ne[0]) : nullptr;

    if (use_kvarn_q_rot) {
        GGML_ASSERT(q->type == GGML_TYPE_F32);
        GGML_ASSERT(kvarn_rot != nullptr);
        q = ggml_kvarn_wht_aux(ctx0, q, kvarn_rot->ne[0]);
    }

    // The SET_ROWS results alias the complete persistent shadow tensors and
    // carry the write dependency needed by same-graph attention reads.
    ggml_tensor * k_tail = k_tail_written ? k_tail_written : mctx_cur->get_k_tail(ctx0, il);
    ggml_tensor * v_tail = v_tail_written ? v_tail_written : mctx_cur->get_v_tail(ctx0, il);
    const bool gather_k_tail = k_tail != nullptr;
    const bool gather_v_tail = v_tail != nullptr;
    ggml_tensor * tail_read_idxs = inp->get_tail_read_idxs();
    llama_kv_tail_route tail_route = mctx_cur->get_tail_route(il);
    // Fixture-only: say exactly which of the four gates that decide whether a
    // tail is BUILT differs between configurations. The operator census shows
    // the split graph has zero tail nodes and the pure graph has 48; without
    // this line there is no way to tell whether the plan, the compact-tail
    // capability or the tensor retrieval is what turns the tail off.
    if (getenv("F4_TAIL_TRACE")) {
        fprintf(stderr, "F4TAIL il=%d compact=%d route=%d k_tail=%d v_tail=%d "
                "gather=%d tail_read_idxs=%d kv_tail_tokens=%u split_p=%u\n",
                il, (int) compact_tail, (int) tail_route,
                k_tail != nullptr, v_tail != nullptr, gather_k_tail,
                tail_read_idxs != nullptr, cparams.kv_tail_tokens,
                cparams.kv_position_split_p);
    }
    // A backend query-width fallback still requires the generic tail oracle.
    // Non-causal DFlash and D64 are different: only record-consuming attention
    // is unqualified. Their materialized F16 body can retain the advertised
    // native exact-tail merge, avoiding a full F32 QK matrix.
    if (use_kvarn && tail_route == LLAMA_KV_TAIL_ROUTE_NATIVE &&
            !kvarn_plan.native_attention && q->ne[0] != 64 &&
            (arch != LLM_ARCH_DFLASH || cparams.causal_attn)) {
        tail_route = LLAMA_KV_TAIL_ROUTE_GENERIC;
    }
    if (tail_route != LLAMA_KV_TAIL_ROUTE_NONE) {
        GGML_ASSERT(mctx_cur->get_tail_explicit_bias(il) == (kq_b != nullptr) &&
                "KV tail route bias contract does not match the model graph");
    }
    ggml_tensor * k_tail_current = nullptr;
    ggml_tensor * v_tail_current = nullptr;
    if (compact_tail && k_cur && v_cur) {
        k_tail_current = prepare_compact_tail_current(ctx0, k_tail, k_cur);
        v_tail_current = prepare_compact_tail_current(ctx0, v_tail, v_cur);
        if (tail_route != LLAMA_KV_TAIL_ROUTE_NATIVE) {
            k_tail = ggml_concat(ctx0, k_tail, k_tail_current, 2);
            v_tail = ggml_concat(ctx0, v_tail, v_tail_current, 2);
            k_tail_current = nullptr;
            v_tail_current = nullptr;
        }
    }
    const bool use_indexed_tail = gather_k_tail && gather_v_tail && tail_read_idxs &&
            tail_route == LLAMA_KV_TAIL_ROUTE_NATIVE;
    if (!use_indexed_tail && !use_kvarn && mctx_cur->get_tail_tokens(il) > 0) {
        if (!k_tail) {
            k_tail = mctx_cur->get_k_tail_fallback(ctx0, il, inp->self_tail_body_read_idxs);
        }
        if (!v_tail) {
            v_tail = mctx_cur->get_v_tail_fallback(ctx0, il, inp->self_tail_body_read_idxs);
        }
    }
    if (k_tail && !use_indexed_tail) {
        GGML_ASSERT(v_tail && tail_read_idxs);
        auto gather_tail = [&](ggml_tensor * value) {
            const int64_t n_embd_head = value->ne[0];
            const int64_t n_heads = value->ne[1];
            value = ggml_reshape_2d(ctx0, value, n_embd_head*n_heads, value->ne[2]);
            ggml_tensor * read_idxs = ggml_reshape_1d(ctx0, tail_read_idxs,
                    tail_read_idxs->ne[0]*ubatch.n_tokens);
            value = ggml_get_rows_as(ctx0, value, read_idxs, value->type);
            return ggml_reshape_4d(ctx0, value, n_embd_head, n_heads,
                    tail_read_idxs->ne[0], ubatch.n_tokens);
        };
        if (gather_k_tail) {
            k_tail = gather_tail(k_tail);
        }
        if (gather_v_tail) {
            v_tail = gather_tail(v_tail);
        }
    }
    if (use_kvarn && k_tail) {
        GGML_ASSERT(v_tail && kvarn_rot);
        k_tail = ggml_kvarn_wht_aux(ctx0, k_tail, kvarn_rot->ne[0]);
        // msg57 F5: the KVarN FA un-rotates the V it loads from the records and
        // the stage, and it applies the SAME transform to the attached tail V.
        // The stored tail is in the ORIGINAL domain, so the tail V must be
        // rotated in EVERY domain - not only in the fully rotated one, where it
        // happened to be rotated before. Measured in
        // tests/test-position-split-real-pair (domain ROTATED_K_ORIGINAL_V):
        // rotated tail V -> NRMSE 3.168e-04 / maxO 4.297e-03 (PASS);
        // unrotated tail V -> NRMSE 2.155e-01 / maxO 1.857e+00 (catastrophic).
        v_tail = ggml_kvarn_wht_aux(ctx0, v_tail, kvarn_rot->ne[0]);
        if (k_tail_current) {
            k_tail_current = ggml_kvarn_wht_aux(ctx0, k_tail_current, kvarn_rot->ne[0]);
            v_tail_current = ggml_kvarn_wht_aux(ctx0, v_tail_current, kvarn_rot->ne[0]);
        }
    }
    ggml_tensor * kq_mask_tail = tail_route == LLAMA_KV_TAIL_ROUTE_NONE ? nullptr : inp->get_kq_mask_tail();
    ggml_tensor * kq_b_tail = build_attn_bias_tail(
            kq_b, inp->get_tail_bias_read_idxs(), kq_mask_tail);
    ggml_tensor * body_mask = kq_mask;
    ggml_tensor * body_bias = kq_b;
    if (!mctx_cur->has_kv_body(il)) {
        GGML_ASSERT(compact_tail && !use_kvarn);
        const auto * route = mctx_cur->get_tail_layer_route(il);
        GGML_ASSERT(route);
        build_empty_kv_body(ctx0, k_cur ? k_cur : k_tail, v_cur ? v_cur : v_tail,
                route->body_type_k, route->body_type_v, kq_mask, kq_b,
                k, v, body_mask, body_bias);
    }
    ggml_tensor * final_tail_op = nullptr;
    ggml_tensor * cur = build_attn_mha(q, k, v, body_bias, body_mask, sinks, v_mla, 0, kq_scale, il,
k_tail, v_tail, kq_mask_tail, kq_b_tail,
            use_indexed_tail ? tail_read_idxs : nullptr,
            (use_indexed_tail || use_kvarn) ? inp->get_tail_query_order() : nullptr,
            (use_indexed_tail || use_kvarn) ? inp->get_tail_run_desc() : nullptr,
            tail_route,
            kvarn_domain,
            k_tail_current, v_tail_current,
            mctx_cur->get_tail_slots(), !mctx_cur->has_kv_body(il), &final_tail_op);
    if (use_kvarn) {
        llm_flash_attn_ext_set_kvarn_domain(cur, kvarn_domain);
    }
    if (use_kvarn && arch == LLM_ARCH_DFLASH && !cparams.causal_attn &&
            kvarn_plan.native_attention) {
        validate_native_kvarn_noncausal_operation(mctx_cur, il, final_tail_op);
    }
    if (tail_route == LLAMA_KV_TAIL_ROUTE_NATIVE) {
        validate_native_tail_operation(mctx_cur, il, final_tail_op, true);
    }
    if (compact_tail && k_cur && v_cur) {
        if (ggml_tensor * written = mctx_cur->cpy_k_tail(
                    ctx0, k_cur, inp->self_tail_idxs, il, cur)) {
            ggml_build_forward_expand(gf, written);
        }
        if (ggml_tensor * written = mctx_cur->cpy_v_tail(
                    ctx0, v_cur, inp->self_tail_idxs, il, cur)) {
            ggml_build_forward_expand(gf, written);
        }
    }
    cb(cur, "kqv_out", il);

    if (use_kvarn_rotated_domain) {
        GGML_ASSERT(cur->type == GGML_TYPE_F32);
        GGML_ASSERT(kvarn_rot != nullptr);
        cur = ggml_kvarn_wht_aux(ctx0, cur, kvarn_rot->ne[0]);
    } else if (!use_kvarn && inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    cur = apply_attention_reference_cpu(cur, il);

    if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
            cur = build_lora_mm(wo, cur);
            ggml_prec_set_acc(cur, GGML_PREC_F32);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

bool llm_graph_context::use_position_split_attention(int il) const {
    // mctx may be a hybrid context; unwrap to the KVarN attention context.
    const auto * kvarn_ctx = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx);
    if (!kvarn_ctx) {
        if (const auto * hybrid = dynamic_cast<const llama_memory_hybrid_context *>(mctx)) {
            kvarn_ctx = dynamic_cast<const llama_kv_cache_kvarn_context *>(
                hybrid->get_attn_kv_context());
        }
    }
    if (!kvarn_ctx || !kvarn_ctx->position_split_enabled()) {
        return false;
    }
    if (!kvarn_ctx->uses_kvarn_layer(il)) {
        return false;
    }
    const auto * kvarn_cache = kvarn_ctx->get_kvarn_cache();
    if (!kvarn_cache) {
        return false;
    }
    const uint32_t p = kvarn_cache->position_split_p();
    // Every overflow batch uses range-relative inputs, including an empty
    // local history. Compact Qx rows cannot consume the legacy global mask.
    // (n_kv_global alone is insufficient: the first overflow token sees
    // exactly P local tokens, i.e. n_kv == P, not > P.)
    // Invalid/mixed ubatches (e.g. graph reserve dummies) fall back to legacy.
    if (ubatch.n_pos < 1 || ubatch.n_tokens == 0 || ubatch.pos == nullptr) {
        return false;
    }
    bool local = false;
    bool overflow = false;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const auto kind = llama_position_split::range_for_pos(
                ubatch.pos[i], p, kvarn_cache->get_kv_size()).kind;
        if (kind == llama_position_split::range_kind::INVALID) {
            return false;
        }
        if (kind == llama_position_split::range_kind::LOCAL) {
            local = true;
        } else {
            overflow = true;
        }
    }
    if (local && overflow) {
        return false;
    }
    if (local) {
        return false;
    }
    // Empty and occupied local histories use the same split route. Do not
    // scan/sort the full history here merely to populate a debug message:
    // this predicate is called for every full-attention layer on graph build.
    LLAMA_LOG_DEBUG("%s: il=%d P=%u n_kv=%u n_tokens=%u overflow_batch=%d -> %s\n",
            __func__, il, p, kvarn_ctx->get_n_kv(), ubatch.n_tokens,
            int(overflow),
            overflow ? "split" : "legacy");
    return overflow;
}


ggml_tensor * llm_graph_context::apply_attention_reference_cpu(ggml_tensor * output, int il) const {
    const char * reference_env = std::getenv("LLAMA_OVF_ATTN_REFERENCE_CPU");
    if (!reference_env || std::strcmp(reference_env, "1") != 0) { return output; }
    // Long-context comparisons may build an identical GPU prefix first and
    // start the CPU oracle at a fixed suffix. This isolates attention/merge
    // error on the same persistent KV representation without requantization.
    if (const char * minimum = std::getenv("LLAMA_OVF_ATTN_REFERENCE_CPU_MIN_POS")) {
        const std::string value(minimum);
        if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
            throw std::invalid_argument("invalid CPU attention reference minimum position");
        }
        const uint64_t threshold = std::stoull(value);
        if (threshold > uint64_t(INT32_MAX)) {
            throw std::invalid_argument("CPU attention reference minimum position is out of range");
        }
        bool active = false;
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            active |= ubatch.pos[i] >= int64_t(threshold);
        }
        if (!active) { return output; }
    }
    ggml_backend_t backend_host = backend_cpu;
    if (!backend_host) {
        for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
            auto * backend = ggml_backend_sched_get_backend(sched, i);
            auto * device = ggml_backend_get_device(backend);
            if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) { backend_host = backend; break; }
        }
    }
    if (!backend_host) { throw std::runtime_error("CPU structural attention reference requires a CPU backend"); }
    std::unordered_map<ggml_tensor *, ggml_tensor *> host_tensors;
    std::function<ggml_tensor *(ggml_tensor *)> host_copy;
    host_copy = [&](ggml_tensor * tensor) -> ggml_tensor * {
        if (!tensor) { return nullptr; }
        if (auto it = host_tensors.find(tensor); it != host_tensors.end()) { return it->second; }
        const bool descriptor = tensor->op == GGML_OP_KVARN_VIEW;
        const bool view = tensor->op == GGML_OP_VIEW || tensor->op == GGML_OP_RESHAPE ||
                tensor->op == GGML_OP_PERMUTE || tensor->op == GGML_OP_TRANSPOSE;
        ggml_tensor * result;
        if (descriptor || view) {
            result = ggml_dup_tensor(ctx0, tensor);
            host_tensors.emplace(tensor, result);
            result->op = tensor->op;
            std::memcpy(result->op_params, tensor->op_params, sizeof(tensor->op_params));
            std::memcpy(result->nb, tensor->nb, sizeof(tensor->nb));
            for (int i = 0; i < GGML_MAX_SRC; ++i) { result->src[i] = host_copy(tensor->src[i]); }
            if (view) {
                result->view_src = host_copy(tensor->view_src);
                result->view_offs = tensor->view_offs;
            }
        } else {
            if (!ggml_is_contiguous(tensor)) {
                throw std::runtime_error(format("CPU attention reference cannot copy strided owner %s", tensor->name));
            }
            result = ggml_cont(ctx0, tensor);
            host_tensors.emplace(tensor, result);
        }
        ggml_format_name(result, "%s_ref_host", tensor->name);
        ggml_backend_sched_set_tensor_backend(sched, result, backend_host);
        return result;
    };
    ggml_build_forward_expand(gf, output);
    std::vector<ggml_tensor *> roots;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) { roots.push_back(ggml_graph_node(gf, i)); }
    std::vector<ggml_tensor *> attention_nodes;
    std::unordered_set<ggml_tensor *> visited;
    std::function<void(ggml_tensor *)> discover = [&](ggml_tensor * node) {
        if (!node || !visited.insert(node).second) { return; }
        if (node->op == GGML_OP_FLASH_ATTN_EXT) { attention_nodes.push_back(node); return; }
        for (int i = 0; i < GGML_MAX_SRC; ++i) { discover(node->src[i]); }
    };
    discover(output);
    if (attention_nodes.empty()) {
        throw std::runtime_error(format("CPU structural attention reference found no FA in layer %d", il));
    }
    std::unordered_map<ggml_tensor *, ggml_tensor *> original_domain_outputs;
    const auto is_kvarn_read = [](const ggml_tensor * tensor) {
        while (tensor && (tensor->op == GGML_OP_VIEW || tensor->op == GGML_OP_RESHAPE ||
                tensor->op == GGML_OP_PERMUTE || tensor->op == GGML_OP_TRANSPOSE)) {
            tensor = tensor->src[0];
        }
        return tensor && tensor->op == GGML_OP_KVARN_VIEW;
    };
    for (ggml_tensor * attention : attention_nodes) {
        const int local_domain = attention->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_KVARN_DOMAIN];
        for (int i = 0; i < GGML_MAX_SRC; ++i) { attention->src[i] = host_copy(attention->src[i]); }
        if (is_kvarn_read(attention->src[1])) {
            attention->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_KVARN_DOMAIN] = GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED;
        } else {
            // The ordinary CPU Q4 dot kernel quantizes Q to Q8. That is an
            // additional approximation, not an oracle for the stored Q4 KV.
            // Decode standard K/V to F32 on the host, leaving Q unchanged and
            // selecting the F32 output accumulator instead of half V sums.
            for (int i : {1, 2}) {
                if (attention->src[i]->type != GGML_TYPE_F32) {
                    attention->src[i] = ggml_cast(ctx0, attention->src[i], GGML_TYPE_F32);
                    ggml_backend_sched_set_tensor_backend(sched, attention->src[i], backend_host);
                }
            }
        }
        if (local_domain == GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V) {
            ggml_tensor * derotated = ggml_kvarn_wht(ctx0, attention, int(attention->ne[0]));
            ggml_backend_sched_set_tensor_backend(sched, derotated, backend_host);
            original_domain_outputs.emplace(attention, derotated);
        }
        ggml_backend_sched_set_tensor_backend(sched, attention, backend_host);
        if (!ggml_backend_supports_op(backend_host, attention)) {
            throw std::runtime_error(format("CPU structural attention reference unsupported in layer %d", il));
        }
    }
    for (ggml_tensor * node : roots) {
        if (node->op == GGML_OP_FLASH_ATTN_EXT_LSE) {
            ggml_backend_sched_set_tensor_backend(sched, node, backend_host);
            continue;
        }
        if (original_domain_outputs.count(node)) { continue; }
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            if (auto replacement = original_domain_outputs.find(node->src[i]); replacement != original_domain_outputs.end()) {
                node->src[i] = replacement->second;
            }
        }
        if (auto replacement = original_domain_outputs.find(node->view_src); replacement != original_domain_outputs.end()) {
            node->view_src = replacement->second;
        }
    }
    if (auto replacement = original_domain_outputs.find(output); replacement != original_domain_outputs.end()) {
        output = replacement->second;
    }
    ggml_graph_clear(gf);
    for (ggml_tensor * root : roots) { ggml_build_forward_expand(gf, root); }
    ggml_build_forward_expand(gf, output);
    fprintf(stderr, "%s: layer %d ATTENTION-REFERENCE-CPU %zu FA node(s); native records/exact-tail/overflow staged\n",
            __func__, il, attention_nodes.size());
    return output;
}

// F4 bounded allocation evidence (env-gated; shape/type/op/bytes only, no payloads).
static void f4_tensor_trace(const char * tag, const char * name, ggml_tensor * t) {
    if (t == nullptr || std::getenv("F4_ALLOC_TRACE") == nullptr) {
        return;
    }
    char ne[GGML_MAX_DIMS * 14];
    int off = 0;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        off += snprintf(ne + off, sizeof(ne) - off, "%s%" PRId64, i ? "," : "", t->ne[i]);
    }
    LLAMA_LOG_INFO("F4TRACE %s %-20s op=%-14s type=%-8s ne=[%s] bytes=%zu buf=%s\n",
            tag, name, ggml_op_name(t->op), ggml_type_name(t->type), ne, ggml_nbytes(t),
            t->buffer ? "allocated" : "unallocated");
}

// F3 overflow-mask crop+C memo. The overflow range mask is ONE graph input
// per ubatch (llama-graph.cpp build_attn_inp_kv: self_kq_mask_overflow, filled
// once in set_input) consumed by every KVarN layer, and the crop extent is a
// function of the cache-wide occupancy (n_ovf_occ) or the packed-view stable
// extent (n_ovf_rows), both layer-independent. Without the memo every layer
// re-creates the view+C and re-stages the same bytes host->device; with it,
// the first layer builds the cropped contiguous mask and the rest reuse the
// SAME tensor. Content is bit-identical (ggml_cont is a plain copy, consumers
// are read-only), and the memo lives in llm_graph_context, which is
// constructed per build_graph, so no entry can outlive its ggml_context.
// Memory note (review): the ONE cropped buffer is pinned from the first
// consumer to the LAST layer's FA of the graph (its node stays live for the
// whole build), whereas the replaced per-layer copies could each be recycled
// right after their own FA. Net peak: +1 cropped-mask buffer per graph
// (n_cols * ne[1..3] * type_size bytes, usually a few KiB), NOT one per layer;
// the CPU unit test (tests/test-position-split-mask-memo.cpp) asserts exactly
// one crop CONT node per graph, and F4_ALLOC_TRACE logs one create + N-1
// reuses per graph.
ggml_tensor * llm_graph_context::build_ovf_mask_crop(
        ggml_tensor * mask_src,
        int64_t n_cols) const {
    GGML_ASSERT(mask_src != nullptr);
    GGML_ASSERT(n_cols > 0 && n_cols <= mask_src->ne[0]);
    const auto key = std::make_tuple(mask_src, n_cols,
            mask_src->ne[1], mask_src->ne[2], mask_src->ne[3], mask_src->type);
    const auto it = ovf_mask_crop_memo.find(key);
    if (it != ovf_mask_crop_memo.end()) {
        if (std::getenv("F4_ALLOC_TRACE") != nullptr) {
            // stderr on purpose: the server log callback filters INFO from the
            // model process (see the F5OVF trace). Proof of reuse: one
            // "create" per graph, "reuse" for every following layer.
            fprintf(stderr, "[F4TRACE] ovf_mask_crop_memo reuse src=%p n_cols=%" PRId64 " ne1=%" PRId64 " -> %p\n",
                    (const void *) mask_src, n_cols, mask_src->ne[1], (void *) it->second);
        }
        return it->second;
    }
    // ggml_view_4d(ctx, a, ne0..ne3, nb1, nb2, nb3, offset): the strides come
    // before the offset, and the crop is taken on the tensor the consumer
    // actually reads (self_kq_mask_overflow_cnv, which carries the required
    // mask dtype/conversion), with its REAL strides and offset 0.
    if (mask_src->nb[1] == 0) {
        throw std::runtime_error("position-split overflow mask has a zero row stride");
    }
    ggml_tensor * mask_view = ggml_view_4d(ctx0, mask_src,
            n_cols, mask_src->ne[1], mask_src->ne[2], mask_src->ne[3],
            mask_src->nb[1], mask_src->nb[2], mask_src->nb[3], 0);
    if (mask_view->type != mask_src->type ||
            mask_view->ne[1] != mask_src->ne[1] ||
            mask_view->ne[2] != mask_src->ne[2] ||
            mask_view->ne[3] != mask_src->ne[3] ||
            mask_view->nb[0] != mask_src->nb[0] ||
            mask_view->nb[1] != mask_src->nb[1] ||
            n_cols > mask_src->ne[0]) {
        throw std::runtime_error("position-split overflow mask crop changed the mask contract");
    }
    // The overflow FA requires a CONTIGUOUS mask, and a crop that inherits the
    // capacity row stride is never contiguous. Materialize ONLY the cropped
    // extent: capacity-sized input allocation and fill order are untouched,
    // dtype/conversion are preserved, and the unused trailing rows never reach
    // the compute scratch.
    ggml_tensor * mask_cont = ggml_cont(ctx0, mask_view);
    if (!ggml_is_contiguous(mask_cont) ||
            mask_cont->ne[0] != n_cols ||
            mask_cont->ne[1] != mask_src->ne[1] ||
            mask_cont->type != mask_src->type) {
        throw std::runtime_error(
            "position-split overflow mask is not contiguous at the occupied extent");
    }
    ggml_set_name(mask_cont, "attn_inp_kq_mask_overflow_occ");
    ovf_mask_crop_memo.emplace(key, mask_cont);
    if (std::getenv("F4_ALLOC_TRACE") != nullptr) {
        fprintf(stderr, "[F4TRACE] ovf_mask_crop_memo create src=%p n_cols=%" PRId64 " ne1=%" PRId64 " -> %p\n",
                (const void *) mask_src, n_cols, mask_src->ne[1], (void *) mask_cont);
    }
    return mask_cont;
}

ggml_tensor * llm_graph_context::build_attn_position_split_rpc(
        llm_graph_input_attn_kv * inp, ggml_tensor * q_cur, ggml_tensor * k_cur,
        ggml_tensor * v_cur, float kq_scale, int il, ggml_tensor * positions) const {
    const auto * kvarn_ctx = dynamic_cast<const llama_kv_cache_kvarn_context *>(inp->mctx);
    if (!kvarn_ctx || !kvarn_ctx->position_split_enabled() || !positions ||
            !backend_remote || !backend_cpu || !cparams.causal_attn ||
            hparams.use_alibi || hparams.f_max_alibi_bias != 0.0f || hparams.attn_soft_cap ||
            !inp->self_kq_mask_local_cnv || !inp->self_overflow_cell_idxs) {
        throw std::runtime_error("RKVA positional attention requires causal single-sequence KVarN range inputs");
    }
    const auto * cache = kvarn_ctx->get_kvarn_cache();
    const auto * standard = cache->get_standard_cache();
    if (!standard || standard->get_payload_row_origin() != cache->position_split_p()) {
        throw std::runtime_error("RKVA range requires a compact quantized checkpoint mirror beginning at P");
    }
    ggml_backend_t backend_cuda = nullptr;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        auto * backend = ggml_backend_sched_get_backend(sched, i);
        const auto * name = ggml_backend_dev_name(ggml_backend_get_device(backend));
        if (name && std::strncmp(name, "CUDA", 4) == 0) { backend_cuda = backend; break; }
    }
    if (!backend_cuda) throw std::runtime_error("RKVA position split requires local CUDA attention");
    const uint32_t n_local = uint32_t(inp->self_kq_mask_local->ne[0]);
    auto * mask_local = inp->self_kq_mask_local_cnv;
    // Preserve the Q4_0/Q8_0/F16 checkpoint representation; upload only appended rows.
    auto * k_written = kvarn_ctx->cpy_k(ctx0, k_cur, inp->get_k_idxs(), il);
    auto * v_written = kvarn_ctx->cpy_v(ctx0, v_cur, inp->get_v_idxs(), il);
    if (!k_written || !v_written) throw std::runtime_error("missing RKVA companion stores");
    ggml_backend_sched_set_tensor_backend(sched, k_written, backend_cpu);
    ggml_backend_sched_set_tensor_backend(sched, v_written, backend_cpu);
    ggml_backend_sched_set_tensor_backend(sched, inp->self_overflow_cell_idxs, backend_cpu);
    ggml_backend_sched_set_tensor_backend(sched, inp->self_kvarn_mat_idxs_local, backend_cpu);
    ggml_backend_sched_set_tensor_backend(sched, inp->self_kq_mask_local, backend_cpu);
    ggml_build_forward_expand(gf, k_written);
    ggml_build_forward_expand(gf, v_written);
    int remote_layer = 0;
    for (int l = 0; l < il; ++l) {
        if (hparams.has_kv(l) && !hparams.is_recr(l)) ++remote_layer;
    }
    const bool async_overlap = ggml_backend_remote_attn_async_enabled(backend_remote);
    auto * range = ggml_remote_attn_range(ctx0, q_cur, k_written, v_written,
            inp->self_overflow_cell_idxs, positions, remote_layer,
            cache->position_split_p(), uint32_t(inp->self_overflow_cell_idxs->ne[0]), kq_scale);
    const bool cuda_wire = ggml_backend_remote_attn_cuda_enabled_for_node(range);
    ggml_backend_t wire_backend = cuda_wire ? backend_cuda : backend_cpu;
    if (ggml_backend_remote_attn_native_kvarn_enabled(backend_remote)) {
        // Preserve original projected values as independent graph operands.
        // The Q4/Q8 companion above remains metadata/bookkeeping only for this
        // experimental native owner; none of its quantized bytes are decoded.
        auto * native_k = ggml_cont(ctx0, k_cur);
        auto * native_v = ggml_cont(ctx0, v_cur);
        ggml_remote_attn_range_set_native_sources(range, native_k, native_v);
    }
    ggml_backend_sched_set_tensor_backend(sched, range, wire_backend);
    if (async_overlap) {
        // Submit the ordered remote bulk RPC before the local CUDA attention
        // cluster, then join it at the wait node just before the LSE merge.
        // The local range and the remote range are the same layer/query batch;
        // later layers still depend on the merged output.
        ggml_remote_attn_range_set_async(range);
        ggml_build_forward_expand(gf, range);
    }
    const bool native_attention = kvarn_ctx->uses_native_attention(il) &&
        llama_kvarn_native_attention_allowed(cparams.causal_attn, arch);
    const auto kvarn_plan = llama_kvarn_plan_attention(
        native_attention,
        kvarn_ctx->native_attention_uses_original_v(il),
        kvarn_ctx->native_rotated_max_query_tokens(il),
        (uint32_t) q_cur->ne[2],
        (int) q_cur->ne[0]);
    const auto kvarn_domain = kvarn_plan.domain;
    const bool use_kvarn_rotated_domain =
        kvarn_domain == GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED;
    const bool use_kvarn_mixed_domain =
        kvarn_domain == GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V;
    const bool use_kvarn_q_rot = use_kvarn_rotated_domain || use_kvarn_mixed_domain;
    const bool local_tail_here = kvarn_ctx->tail_on_layer(il);
    ggml_tensor * kvarn_rot = nullptr;
    if (use_kvarn_q_rot || local_tail_here) {
        kvarn_rot = llm_kvarn_rot_for_dim(
            inp->self_kvarn_rot_64, inp->self_kvarn_rot_128,
            inp->self_kvarn_rot_256, inp->self_kvarn_rot_512, q_cur->ne[0]);
        GGML_ASSERT(kvarn_rot != nullptr);
    }
    ggml_tensor * q_local = q_cur;
    if (use_kvarn_q_rot) {
        GGML_ASSERT(q_local->type == GGML_TYPE_F32);
        q_local = ggml_kvarn_wht_aux(ctx0, q_local, kvarn_rot->ne[0]);
    }

    // Local K/V via KVarN native range read (same domain as single-range).
    // EXPLICIT local body: this runs inside overflow-only ubatches too, where
    // the current range is 1. Naming the read range keeps the local lookup
    // independent of which ubatch is executing.
    ggml_tensor * k_local = kvarn_ctx->get_k_for_range(
        ctx0, il, kvarn_plan.native_attention,
        llama_position_split::kRangeLocal, n_local, inp->self_kvarn_mat_idxs_local);
    ggml_tensor * v_local = kvarn_ctx->get_v_for_range(
        ctx0, il, kvarn_plan.native_attention,
        llama_position_split::kRangeLocal, n_local, inp->self_kvarn_mat_idxs_local);
    GGML_ASSERT(k_local != nullptr && v_local != nullptr);

    ggml_tensor * k_tail_local   = nullptr;
    ggml_tensor * v_tail_local   = nullptr;
    ggml_tensor * mask_tail_local = nullptr;
    ggml_tensor * bias_tail_local = nullptr;
    ggml_tensor * tail_idx_local  = nullptr;
    ggml_tensor * tail_order_local = nullptr;
    ggml_tensor * tail_desc_local  = nullptr;
    llama_kv_tail_route route_local = LLAMA_KV_TAIL_ROUTE_NONE;
    if (local_tail_here) {
        route_local = kvarn_ctx->get_tail_route(il);
        k_tail_local = kvarn_ctx->get_k_tail(ctx0, il);
        v_tail_local = kvarn_ctx->get_v_tail(ctx0, il);
        tail_idx_local = inp->get_tail_read_idxs();
        if (route_local == LLAMA_KV_TAIL_ROUTE_NONE) {
            k_tail_local = nullptr;
            v_tail_local = nullptr;
        }
        // Fail closed instead of building a half-wired tail: a present tail
        // without its read indices would silently read the wrong keys.
        if (k_tail_local != nullptr && (v_tail_local == nullptr || tail_idx_local == nullptr)) {
            throw std::logic_error(format(
                    "position-split layer %d: local exact tail present without "
                    "its V or read-index operand", il));
        }
        if (k_tail_local != nullptr) {
            mask_tail_local = inp->get_kq_mask_tail();
            bias_tail_local = build_attn_bias_tail(
                    nullptr, inp->get_tail_bias_read_idxs(), mask_tail_local);
            tail_order_local = inp->get_tail_query_order();
            tail_desc_local  = inp->get_tail_run_desc();
            // Same domain rule as the single-range path: the stored tail is
            // unrotated, the local body records are rotated.
            GGML_ASSERT(kvarn_rot != nullptr);
            // NO current segment here: this builder runs for an OVERFLOW
            // ubatch, which writes Q4 overflow keys, so the local tail is read
            // purely as HISTORY (validate_native_tail_operation(..., false)).
            // Overflow keys must never enter the local tail.
            k_tail_local = ggml_kvarn_wht_aux(ctx0, k_tail_local, kvarn_rot->ne[0]);
            // Same rule as the single-range path above: the FA un-rotates the
            // tail V too, so the stored (original-domain) tail V must be rotated
            // in every domain. See the measurement cited there.
            v_tail_local = ggml_kvarn_wht_aux(ctx0, v_tail_local, kvarn_rot->ne[0]);
            // Same domain rule for the current segment: K is always rotated for
            // the KVarN Q-rot path, V only in the rotated-domain case.
        }
    }

    // Two FlashAttention ops with LSE side outputs. No sinks/bias/softcap in
    // P-mode (fail closed below; plan 3.4 keeps sinks out of the first mode);
    // precision F32 like the single-range path.
    ggml_tensor * fa1 = nullptr;
    ggml_tensor * final_local_tail_op = nullptr;
    // Argument order mirrors build_attn() exactly: slot 5 is the MODEL's KQ
    // bias (must stay null -- a non-null value there silently disables the
    // FlashAttention branch and leaves final_attn_op unset), and the tail bias
    // is slot 14. Passing the tail bias one slot early compiled fine and only
    // failed at the GGML_ASSERT(fa1 != nullptr) below.
    ggml_tensor * o1 = build_attn_mha(q_local, k_local, v_local,
        nullptr, mask_local, nullptr, nullptr, int64_t(n_local),
        kq_scale, il,
        k_tail_local, v_tail_local, mask_tail_local, bias_tail_local,
        tail_idx_local, tail_order_local, tail_desc_local,
        route_local, kvarn_domain,
        nullptr, nullptr, kvarn_ctx->get_tail_slots(),
        !kvarn_ctx->has_kv_body(il), &final_local_tail_op);
    // build_attn_mha returns the final (possibly tail-merged) node through
    // final_local_tail_op; both the merge and tail validator consume it.
    fa1 = final_local_tail_op;
    GGML_ASSERT(fa1 != nullptr);
    if (route_local == LLAMA_KV_TAIL_ROUTE_NATIVE) {
        validate_native_tail_operation(inp->mctx, il, final_local_tail_op, false);
    }
    ggml_tensor * lse1 = ggml_flash_attn_ext_lse_out(ctx0, fa1);

    // Local O back to the original V domain before merging (same rule as the
    // single-range path: the KVarN native output is rotated unless the backend
    // uses the original V domain).
    if (use_kvarn_rotated_domain) {
        GGML_ASSERT(o1->type == GGML_TYPE_F32);
        GGML_ASSERT(kvarn_rot != nullptr);
        o1 = ggml_kvarn_wht_aux(ctx0, o1, kvarn_rot->ne[0]);
    }


    llm_flash_attn_ext_set_kvarn_domain(fa1, kvarn_domain);
    ggml_prec_set_acc(fa1, GGML_PREC_F32);
    ggml_tensor * range_wait = nullptr;
    ggml_tensor * remote_source = range;
    if (async_overlap) {
        // Completion point: blocks only when the LSE merge consumes the remote
        // OUT/LSE. It is emitted after the local CUDA cluster and before merge.
        range_wait = ggml_remote_attn_range_wait(ctx0, range);
        ggml_backend_sched_set_tensor_backend(sched, range_wait, wire_backend);
        remote_source = range_wait;
    }
    const int64_t width = q_cur->ne[0]*q_cur->ne[1];
    auto * o2 = ggml_cont(ctx0, ggml_view_2d(ctx0, remote_source, width, q_cur->ne[2], remote_source->nb[1], 0));
    auto * lse2 = ggml_cont(ctx0, ggml_view_2d(ctx0, remote_source, q_cur->ne[1], q_cur->ne[2],
            remote_source->nb[1], size_t(width)*sizeof(float)));
    ggml_backend_sched_set_tensor_backend(sched, o2, wire_backend);
    ggml_backend_sched_set_tensor_backend(sched, lse2, wire_backend);
    if (std::getenv("F4_NODE_DUMP") != nullptr) {
        cb(q_cur, "diag_query", il);
        cb(fa1, "diag_local_fa", il);
        cb(o1, "diag_local_out", il);
        cb(lse1, "diag_local_lse", il);
        cb(range, "diag_rpc_range", il);
        if (range_wait) cb(range_wait, "diag_rpc_wait", il);
        cb(o2, "diag_remote_out", il);
        cb(lse2, "diag_remote_lse", il);
    }
    ggml_build_forward_expand(gf, o1);
    ggml_build_forward_expand(gf, lse1);
    if (range_wait) ggml_build_forward_expand(gf, range_wait);
    ggml_build_forward_expand(gf, o2);
    ggml_build_forward_expand(gf, lse2);
    const int merge_begin = ggml_graph_n_nodes(gf);
    // Stable merge (validated pattern from test-position-split-merge-graph).
    // O is [D,H,Q,B] after restoring the 4D layout; LSE is [H,Q,B].
    const int64_t n_qh = lse1->ne[0];
    const int64_t n_q = lse1->ne[1];
    const int64_t n_b = lse1->ne[2];
    GGML_ASSERT(lse2->ne[0] == n_qh && lse2->ne[1] == n_q && lse2->ne[2] == n_b);
    const int64_t head_dim = q_cur->ne[0];
    // build_attn_mha returns O flattened as [(D*H), (Q*B)]; restore 4D.
    GGML_ASSERT(o1->ne[0] == head_dim*n_qh && o1->ne[1] == n_q*n_b);
    GGML_ASSERT(o2->ne[0] == head_dim*n_qh && o2->ne[1] == n_q*n_b);
    ggml_tensor * o1_4d = ggml_reshape_4d(ctx0, o1, head_dim, n_qh, n_q, n_b);
    ggml_tensor * o2_4d = ggml_reshape_4d(ctx0, o2, head_dim, n_qh, n_q, n_b);
    // This RPC path admits finite LSE values for both non-empty ranges.
    // Select an input exactly: clipping or adding large LSE values can make
    // both exponential weights underflow/overflow despite finite inputs.
    ggml_tensor * choose1 = ggml_step(ctx0, ggml_sub(ctx0, lse1, lse2));
    ggml_tensor * choose2 = ggml_scale_bias(ctx0, choose1, -1.0f, 1.0f);
    ggml_tensor * mx = ggml_add(ctx0,
            ggml_mul(ctx0, choose1, lse1), ggml_mul(ctx0, choose2, lse2));
    ggml_tensor * e1 = ggml_exp(ctx0, ggml_sub(ctx0, lse1, mx));
    ggml_tensor * e2 = ggml_exp(ctx0, ggml_sub(ctx0, lse2, mx));
    ggml_tensor * w1 = ggml_reshape_4d(ctx0, e1, 1, n_qh, n_q, n_b);
    ggml_tensor * w2 = ggml_reshape_4d(ctx0, e2, 1, n_qh, n_q, n_b);
    ggml_tensor * num = ggml_add(ctx0, ggml_mul(ctx0, o1_4d, w1), ggml_mul(ctx0, o2_4d, w2));
    ggml_tensor * den = ggml_add(ctx0, e1, e2);
    // With finite LSE inputs, mx equals one input and one weight is exactly 1.
    // No epsilon is needed for the positive denominator.
    // Broadcast den as [1,H,Q,B] like the validated merge-graph pattern.
    ggml_tensor * den_4d = ggml_reshape_4d(ctx0, den, 1, n_qh, n_q, n_b);
    ggml_tensor * o_merged_4d = ggml_div(ctx0, num, den_4d);
    ggml_tensor * o_merged = ggml_reshape_2d(ctx0, o_merged_4d,
        o_merged_4d->ne[0]*o_merged_4d->ne[1], o_merged_4d->ne[2]*o_merged_4d->ne[3]);
    // Materialize to a fresh contiguous buffer (not a view) so the returned
    // O always owns an allocated output buffer. A reshape view of the div
    // output was left without a valid buffer by the scheduler in the
    // multi-backend split graph (isolated: standalone mul read OOB on the
    // view while sigmoid was clean; materializing eliminates it). Cheap and
    // permanent.
    o_merged = ggml_cont_2d(ctx0, o_merged, o_merged->ne[0], o_merged->ne[1]);


    ggml_build_forward_expand(gf, o_merged);
    for (int i = merge_begin; i < ggml_graph_n_nodes(gf); ++i) {
        auto * node = ggml_graph_node(gf, i);
        if (!node->view_src && node->op != GGML_OP_NONE) {
            ggml_backend_sched_set_tensor_backend(sched, node, backend_cuda);
        }
    }
    ggml_backend_sched_set_tensor_backend(sched, fa1, backend_cuda);
    ggml_backend_sched_set_tensor_backend(sched, lse1, backend_cuda);
    if (q_local != q_cur) ggml_backend_sched_set_tensor_backend(sched, q_local, backend_cuda);
    if (k_tail_local) ggml_backend_sched_set_tensor_backend(sched, k_tail_local, backend_cuda);
    if (v_tail_local) ggml_backend_sched_set_tensor_backend(sched, v_tail_local, backend_cuda);
    ggml_backend_sched_set_tensor_backend(sched, o_merged, backend_cuda);
    cb(o_merged, "attn_position_split_rpc", il);
    return o_merged;
}

ggml_tensor * llm_graph_context::build_attn_position_split(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        float kq_scale,
        int il,
        ggml_tensor * positions) const {
    if (cparams.remote_attn_position_split) {
        return build_attn_position_split_rpc(inp, q_cur, k_cur, v_cur, kq_scale, il, positions);
    }
    // F3 dual-range attention: local CUDA/KVarN FA + overflow Vulkan/Q4 FA +
    // stable merge with CUDA outputs. Requires use_position_split_attention(il); every
    // deviation fails closed instead of silently falling back.
    const auto * kvarn_ctx = dynamic_cast<const llama_kv_cache_kvarn_context *>(inp->mctx);
    if (!kvarn_ctx || !kvarn_ctx->position_split_enabled() || !kvarn_ctx->uses_kvarn_layer(il)) {
        throw std::runtime_error("position-split attention requires an enabled P-mode KVarN layer");
    }
    const auto * kvarn_cache = kvarn_ctx->get_kvarn_cache();
    if (!kvarn_cache || !kvarn_cache->has_standard_cache()) {
        throw std::runtime_error("position-split attention requires the Q4 overflow companion");
    }
    if (!inp->self_kq_mask_local || !inp->self_kq_mask_overflow ||
            !inp->self_kvarn_mat_idxs_local || !inp->self_overflow_cell_idxs) {
        throw std::runtime_error("position-split attention requires range-split graph inputs");
    }
    const uint32_t n_local = uint32_t(inp->self_kq_mask_local->ne[0]);
    const uint32_t n_overflow = uint32_t(inp->self_kq_mask_overflow->ne[0]);
    if (n_local == 0 || n_overflow == 0) {
        throw std::runtime_error("position-split attention requires non-empty local and overflow histories");
    }
    if (int64_t(n_local) != inp->self_kq_mask_local->ne[0] ||
            int64_t(n_overflow) != inp->self_kq_mask_overflow->ne[0] ||
            int64_t(n_local) != inp->self_kvarn_mat_idxs_local->ne[0] ||
            int64_t(n_overflow) != inp->self_overflow_cell_idxs->ne[0]) {
        throw std::runtime_error("position-split range inputs do not match live histories");
    }

    // F5 (measured): only OCCUPIED overflow rows may enter the materialized
    // gather / dual-FA scratch. The graph inputs stay capacity-sized (they are
    // the reservation and the live cells are filled into their leading rows), but
    // at n_kv = P + 256 the capacity (C - P) was 135x the occupancy, and the
    // capacity-sized gather reserved 606.06 MiB on CUDA and failed cudaMalloc
    // (docs/occupancy-evidence/f5-f5-cross-P96512-L96768b-n96768.log). Slicing a
    // view keeps the capacity allocation, keeps the fill order and the slicing so
    // the unused trailing rows are simply never materialized or masked in.
    const uint32_t p_split = kvarn_cache->position_split_p();
    const uint32_t c_split = kvarn_cache->get_kv_size();
    const uint32_t n_kv_live = std::min<uint32_t>(kvarn_ctx->get_n_kv(), c_split);
    const uint32_t n_ovf_occ = n_kv_live > p_split ? n_kv_live - p_split : 0;
    if (n_ovf_occ == 0 || n_ovf_occ > n_overflow) {
        throw std::runtime_error(
            "position-split overflow occupancy is inconsistent with the live histories");
    }
    ggml_tensor * ovf_idxs = inp->self_overflow_cell_idxs;
    ggml_tensor * mask_overflow = inp->self_kq_mask_overflow_cnv;
    uint32_t n_ovf_used = n_overflow;
    // With the packed view enabled the extent policy is decided below to stay
    // stable across tokens; keep the mask at range capacity here instead of
    // cropping it to the live occupancy (otherwise the capacity view would be
    // wider than a cropped mask and the builder would fail closed).
    static const bool ovf_packed_view_env = [] {
        const char * v = std::getenv("LLAMA_OVF_PACKED_VIEW");
        return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
    }();
    if (n_ovf_occ < n_overflow && !ovf_packed_view_env) {
        ovf_idxs = ggml_view_1d(ctx0, inp->self_overflow_cell_idxs,
                int64_t(n_ovf_occ), 0);
        ggml_set_name(ovf_idxs, "attn_overflow_cell_idxs_occ");
        // Materialize ONLY the occupied rows into a packed tensor of the
        // occupied extent: capacity-sized input allocation and fill order are
        // untouched, dtype/conversion are preserved, and the unused trailing
        // rows never reach the compute scratch. This is the small supported
        // step; the capacity-sized inputs stay as reserved. The crop+CONT is
        // memoized per graph build (build_ovf_mask_crop): every layer crops the
        // same host-filled source to the same occupancy extent, so the
        // host->device staging of the identical mask happens once per graph
        // instead of once per layer.
        mask_overflow = build_ovf_mask_crop(inp->self_kq_mask_overflow_cnv,
                int64_t(n_ovf_occ));
        n_ovf_used = n_ovf_occ;
    }
    if (std::getenv("F4_ALLOC_TRACE") != nullptr) {
        LLAMA_LOG_INFO("F4TRACE counts il=%d n_kv=%u P=%u C=%u n_local_cap=%u n_ovf_cap=%u n_ovf_occ=%u n_ovf_used=%u\n",
                il, kvarn_ctx->get_n_kv(), p_split, c_split, n_local, n_overflow, n_ovf_occ, n_ovf_used);
    }

    static const bool tail_local_cuda_enabled = [] {
        const char * e = std::getenv("LLAMA_OVF_TAIL_LOCAL_CUDA");
        return e != nullptr && e[0] == '1' && e[1] == '\0';
    }();
    // Opt-in relocation of the Q WHT from the Vulkan overflow path to the
    // local CUDA backend, for the grouped decode AND the prefill/unanchored
    // arms (see the q_local pin below).
    // Default OFF: absent, "0", or any other value keeps the current
    // placement byte for byte; only the exact string "1" opts in to the CUDA
    // arm, matching LLAMA_OVF_TAIL_LOCAL_CUDA.
    static const bool ovf_qwht_cuda = [] {
        const char * e = std::getenv("LLAMA_OVF_QWHT_CUDA");
        return e != nullptr && e[0] == '1' && e[1] == '\0';
    }();
    // Grouped single-query decode: construct the range stores now for cache
    // bookkeeping but defer their expansion until after fa1/LSE1, so the
    // store + gather + fa2 land in ONE Vulkan cluster instead of reopening CUDA
    // between them. This stays behind LLAMA_OVF_TAIL_LOCAL_CUDA (which also
    // anchors the tail WHT below): prefill and the unanchored F16-tail arm
    // keep the legacy node order because deferring the stores there removes
    // the early Vulkan-assigned store node that the scheduler's forward expand
    // pass used to anchor the unanchored (Vulkan-leaked) tail WHT nodes; the
    // arm would silently flip those WHT bits, and the wht-parity measurement
    // (docs/occupancy-progress-20261003.md) only covers the anchored arm. Q's
    // F32 WHT stays on Vulkan: moving it to CUDA changes bits at head width
    // 256, so it must remain a separate pre-local split unless
    // LLAMA_OVF_QWHT_CUDA=1, which moves the WHT (and its CONT neighbor) onto
    // the local CUDA backend and removes that pre-local split.
    const bool group_overflow_decode = q_cur->ne[2] == 1 &&
            (!kvarn_ctx->tail_on_layer(il) || tail_local_cuda_enabled);
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_tensor * k_written = kvarn_ctx->cpy_k(ctx0, k_cur, inp->get_k_idxs(), il);
    ggml_tensor * v_written = kvarn_ctx->cpy_v(ctx0, v_cur, inp->get_v_idxs(), il);
    GGML_ASSERT(k_written != nullptr && v_written != nullptr);
    if (!group_overflow_decode) {
        ggml_build_forward_expand(gf, k_written);
        ggml_build_forward_expand(gf, v_written);
    }
    std::vector<ggml_tensor *> overflow_probes;
    const auto expand_overflow_probe = [&](ggml_tensor * node) {
        if (group_overflow_decode) {
            overflow_probes.push_back(node);
        } else {
            ggml_build_forward_expand(gf, node);
        }
    };

    // Use the same capability/domain plan as single-range attention for both
    // decode and prefill. Supported shapes consume native KVarN records;
    // unsupported shapes use the planner's explicit materialization fallback.
    // A separate query-count cap would unnecessarily densify the entire local
    // range and change the graph's memory budget near P.
    const bool native_attention = kvarn_ctx->uses_native_attention(il) &&
        llama_kvarn_native_attention_allowed(cparams.causal_attn, arch);
    const auto kvarn_plan = llama_kvarn_plan_attention(
        native_attention,
        kvarn_ctx->native_attention_uses_original_v(il),
        kvarn_ctx->native_rotated_max_query_tokens(il),
        (uint32_t) q_cur->ne[2],
        (int) q_cur->ne[0]);
    const auto kvarn_domain = kvarn_plan.domain;
    if (std::getenv("F5_ROUTE_TRACE") != nullptr) {
        fprintf(stderr, "[F5ROUTE] il=%d nq=%lld native=%d nativereq=%d domain=%d nrotmax=%u minctx=%u\n",
                il, (long long) q_cur->ne[2], int(kvarn_plan.native_attention),
                int(native_attention), int(kvarn_domain),
                kvarn_ctx->native_rotated_max_query_tokens(il),
                kvarn_ctx->get_kvarn_cache()->position_split_p());
    }
    const bool use_kvarn_rotated_domain =
        kvarn_domain == GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED;
    const bool use_kvarn_mixed_domain =
        kvarn_domain == GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V;
    const bool use_kvarn_q_rot = use_kvarn_rotated_domain || use_kvarn_mixed_domain;
    const bool local_tail_here = kvarn_ctx->tail_on_layer(il);
    ggml_tensor * kvarn_rot = nullptr;
    if (use_kvarn_q_rot || local_tail_here) {
        kvarn_rot = llm_kvarn_rot_for_dim(
            inp->self_kvarn_rot_64, inp->self_kvarn_rot_128,
            inp->self_kvarn_rot_256, inp->self_kvarn_rot_512, q_cur->ne[0]);
        GGML_ASSERT(kvarn_rot != nullptr);
    }
    ggml_tensor * q_local = q_cur;
    if (use_kvarn_q_rot) {
        GGML_ASSERT(q_local->type == GGML_TYPE_F32);
        q_local = ggml_kvarn_wht_aux(ctx0, q_local, kvarn_rot->ne[0]);
    }

    // Local K/V via KVarN native range read (same domain as single-range).
    // EXPLICIT local body: this runs inside overflow-only ubatches too, where
    // the current range is 1. Naming the read range keeps the local lookup
    // independent of which ubatch is executing.
    ggml_tensor * k_local = kvarn_ctx->get_k_for_range(
        ctx0, il, kvarn_plan.native_attention,
        llama_position_split::kRangeLocal, n_local, inp->self_kvarn_mat_idxs_local);
    ggml_tensor * v_local = kvarn_ctx->get_v_for_range(
        ctx0, il, kvarn_plan.native_attention,
        llama_position_split::kRangeLocal, n_local, inp->self_kvarn_mat_idxs_local);
    GGML_ASSERT(k_local != nullptr && v_local != nullptr);

    // Overflow K/V via Q4 gather from the shared companion storage.
    const auto * standard_cache = kvarn_cache->get_standard_cache();
    GGML_ASSERT(standard_cache != nullptr);
    GGML_ASSERT(!standard_cache->v_transposed());
    ggml_tensor * k_storage = standard_cache->get_k_storage(il);
    ggml_tensor * v_storage = standard_cache->get_v_storage(il);
    GGML_ASSERT(k_storage != nullptr && v_storage != nullptr);
    const int64_t n_embd_k = k_storage->ne[0];
    const int64_t n_embd_v = v_storage->ne[0];
    ggml_tensor * k_storage_2d = ggml_reshape_2d(ctx0, k_storage,
        n_embd_k, k_storage->ne[1]*k_storage->ne[2]);
    ggml_tensor * v_storage_2d = ggml_reshape_2d(ctx0, v_storage,
        n_embd_v, v_storage->ne[1]*v_storage->ne[2]);
    const uint32_t payload_origin = standard_cache->get_payload_row_origin();
    const uint32_t payload_size = standard_cache->get_payload_size();
    // Gather the Q4_0 overflow K/V DIRECTLY as F16.
    //
    // ggml_get_rows() always materialises an F32 result for a quantized source
    // (ggml.c: ggml_get_rows -> ggml_get_rows_as(..., GGML_TYPE_F32)), and the
    // flash-attention path casts that F32 tensor to F16 right afterwards. At the
    // campaign scale that is a full F32 write plus a full F32 read of the
    // gathered overflow K and V, per layer, per decoded token -- pure waste,
    // because the values are immediately rounded to F16 anyway.
    //
    // ggml_get_rows_as(..., GGML_TYPE_F16) is the same GGML_OP_GET_ROWS with an
    // F16 destination: it dequantises once and stores F16. Both backends that
    // can host the overflow cache already support it:
    //   ggml-cuda/getrows.cu      case GGML_TYPE_F16 in get_rows_cuda()
    //   ggml-vulkan: dispatch at ggml_vk_op_get_pipeline() (dst->type == F16 ->
    //   pipeline_get_rows[src0->type]) with the Q4_0 pipeline created for it.
    // The downstream `if (k->type == GGML_TYPE_F32) k = ggml_cast(...)` guard
    // simply stops firing, so no consumer changes type unexpectedly.
    // A/B toggle for the destination type above, default ON. LLAMA_OVF_GATHER_F16=0
    // restores the legacy F32 materialisation + F16 cast so both paths can be compared
    // inside the SAME binary and the SAME config.
    static const bool ovf_gather_f16 = [] {
        const char * v = std::getenv("LLAMA_OVF_GATHER_F16");
        return v == nullptr || std::strcmp(v, "0") != 0;
    }();

    // PACKED fast path (structural): when the live overflow cells are ONE contiguous
    // ascending run, the attention can read a zero-copy VIEW of the Q4_0 storage rows
    // instead of gathering them. No GET_ROWS, no F32, no F16 scratch: the Vulkan FA
    // pipeline consumes the packed Q4_0 K/V natively (the FA pipeline state is keyed by
    // k_type/v_type, ggml-vulkan.cpp get_fa_pipeline_state), which is exactly what the
    // production per-layer remote path does with its zero-copy quantized views.
    //
    // FAIL CLOSED. The view replaces the gather only when ALL of these hold:
    //   * the live overflow cell list is available and its size equals n_ovf_used;
    //   * cell ids ascend by exactly 1 (single run -> contiguous rows, block aligned
    //     because every offset is a whole number of Q4_0 rows);
    //   * absolute positions ascend strictly along that run, so the per-range mask
    //     columns (built in set_input from the position-sorted list) keep the same
    //     order as the view's rows. Any deviation -> indexed gather, never a wrong view.
    // Fragmented (wrapped ring, eviction) layouts therefore keep the gather, and the
    // reason is logged instead of being silently converted to another KV format.
    // DEFAULT OFF -- MEASURED INCORRECT, not merely unproven.
    // The numeric gate (full-vocab capture, three deterministic prefixes 3000/3072/3200,
    // packed vs indexed gather, same binary/config) rejected the view:
    //   n3000 KL 2.09e-02 nat/token, ppl ratio 25.2, dLSE 1.607
    //   n3072 KL 2.50e-03, ppl ratio 0.671, dLSE 0.793
    //   n3200 KL 9.27e-02, ppl ratio 28.6, argmax FLIPPED, generated token diverged
    // Cause VERIFIED by reviewer traces and reproduced in the shader source:
    // the packed view carried n_view = live rows (e.g. 640) while the contiguous
    // overflow mask kept the padded n_ovf_used width (e.g. 768). The overflow FA
    // derives its mask stride from the K/V extent
    // (vulkan-shaders/flash_attn_base.glsl:213 `m_stride = KV`;
    //  flash_attn.comp:222 `data_m[(i*Br + r)*m_stride + ...]`), so every query
    // row after the first read the WRONG mask row. Fixed below by cropping and
    // materializing a contiguous mask with exactly the K/V extent and passing
    // that same extent as n_kv_max.
    // Two rejected hypotheses, kept for the record:
    //   * rotation_k/rotation_v is a FEATURE-SPACE Hadamard basis on the stored values
    //     (llama-kv-cache-kvarn.cpp: rotation::hadamard(feature width), applied to the
    //     cached basis), not a logical-cell -> physical-row offset. Both the gather and
    //     a raw view read the same stored basis, so it cannot explain the divergence.
    //   * global cell IDs form one contiguous run; the Qx cache's authoritative
    //     payload_row_for_cell mapping must also form a contiguous physical run.
    //     Compact split payloads begin at P, so global cell row0 and physical view
    //     row0 are intentionally tracked separately below.
    // The quantized Vulkan FA path (k_type_eff == Q4_0 at neq1 == 1, proactive dequant
    // disabled for neq1 < 64) is still gated only by the numeric tests that follow,
    // not by assumption.
    static const bool ovf_packed_view = [] {
        const char * v = std::getenv("LLAMA_OVF_PACKED_VIEW");
        return v != nullptr && std::strcmp(v, "0") != 0;
    }();
    const char * ovf_path = "indexed_gather";
    uint32_t ovf_row0 = 0;
    bool ovf_use_view = false;
    // Layout invariants a view depends on, checked instead of assumed:
    //   * the storage row stride must equal one packed row of its own type
    //     (ggml_row_size), otherwise rows are not addressable as views;
    //   * the live run must lie inside the storage extent.
    const bool k_rows_exact = k_storage->nb[1] == ggml_row_size(k_storage->type, n_embd_k);
    const bool v_rows_exact = v_storage->nb[1] == ggml_row_size(v_storage->type, n_embd_v);
    const uint32_t k_rows = uint32_t(k_storage->ne[1] * k_storage->ne[2]);
    const uint32_t v_rows = uint32_t(v_storage->ne[1] * v_storage->ne[2]);
    if (k_rows != payload_size || v_rows != payload_size) {
        throw std::runtime_error(format(
                "Qx payload row shape mismatch: declared origin=%u size=%u, physical K/V rows=%u/%u",
                payload_origin, payload_size, k_rows, v_rows));
    }
    uint32_t ovf_n_view = 0; // extent the view will actually cover (live rows only)
    uint32_t ovf_physical_row0 = 0;
    if (ovf_packed_view) {
        std::vector<uint32_t> cells;
        ovf_use_view = kvarn_cache->position_split_live_cells(
                    llama_position_split::range_kind::OVERFLOW, cells) && !cells.empty();
        // get_n_kv() can include padding/pending cells, so the LIVE cell count is the
        // authority for the view extent. Any disagreement with the padded n_ovf_used is
        // expected and handled by cropping the mask to the same extent below, instead
        // of leaving mask stride and K/V extent out of sync.
        if (ovf_use_view && cells.size() != size_t(n_ovf_used)) {
            ovf_path = "packed_view_live_extent";
        }
        for (size_t i = 1; ovf_use_view && i < cells.size(); ++i) {
            if (cells[i] != cells[i-1] + 1) {
                ovf_use_view = false;
                ovf_path = "indexed_gather_fragmented";
                break;
            }
        }
        if (ovf_use_view) {
            const auto & meta_cells = kvarn_cache->get_metadata_cache()->get_cells(0);
            llama_pos prev = meta_cells.pos_get(cells.front());
            for (size_t i = 1; i < cells.size(); ++i) {
                const llama_pos cur = meta_cells.pos_get(cells[i]);
                if (!(cur > prev)) {
                    ovf_use_view = false;
                    ovf_path = "indexed_gather_position_order";
                    break;
                }
                prev = cur;
            }
        }
        // `cells` and the masks use global metadata IDs. The Qx companion may
        // store only its owned payload interval, so a global cell ID is not
        // necessarily a physical tensor row (compact v2 starts at P). Keep the
        // absolute row for the graph reuse guard and bake a separate physical
        // row for storage access.
        if (ovf_use_view) {
            const uint32_t global_row0 = cells.front();
            if (global_row0 < p_split || uint64_t(global_row0) + cells.size() > c_split) {
                throw std::runtime_error(format(
                        "packed overflow global cells exceed owned range [P=%u,C=%u)",
                        p_split, c_split));
            }
            ovf_row0 = global_row0; // absolute/global; do not remap this reuse key
            ovf_physical_row0 = standard_cache->payload_row_for_cell(global_row0);
            if (ovf_physical_row0 >= payload_size) {
                throw std::runtime_error(format(
                        "packed overflow row map is outside Qx payload: global=%u physical=%u origin=%u size=%u",
                        global_row0, ovf_physical_row0, payload_origin, payload_size));
            }
            for (size_t i = 1; i < cells.size(); ++i) {
                const uint32_t physical = standard_cache->payload_row_for_cell(cells[i]);
                if (physical != ovf_physical_row0 + i) {
                    ovf_use_view = false;
                    ovf_path = "indexed_gather_noncontiguous_payload_rows";
                    break;
                }
            }
        }
        if (ovf_use_view && !(k_rows_exact && v_rows_exact)) {
            ovf_use_view = false;
            ovf_path = "indexed_gather_row_stride";
        }
        if (ovf_use_view && (uint64_t(ovf_physical_row0) + cells.size() > k_rows ||
                              uint64_t(ovf_physical_row0) + cells.size() > v_rows)) {
            ovf_use_view = false;
            ovf_path = "indexed_gather_extent";
        }
        if (ovf_use_view) {
            ovf_n_view = uint32_t(cells.size());
            ovf_path = "packed_view";
        }
    }
    if (std::getenv("F4_ALLOC_TRACE") != nullptr) {
        // stderr on purpose: the server log callback filters INFO from the model
        // process, which silently swallowed an earlier LLAMA_LOG_INFO version of
        // this line and made the fast path look disengaged.
        fprintf(stderr, "[F5OVF] il=%d path=%s n_ovf_used=%u n_view=%u row0_global=%u row0_physical=%u payload_origin=%u payload_size=%u ktype=%s krows=%u kstride_ok=%d vstride_ok=%d kne=%lld|%lld|%lld|%lld vne=%lld|%lld|%lld|%lld knb1=%zu vnb1=%zu\n",
                il, ovf_path, n_ovf_used, ovf_n_view, ovf_row0, ovf_physical_row0, payload_origin, payload_size,
                ggml_type_name(k_storage->type), k_rows, int(k_rows_exact), int(v_rows_exact),
                (long long) k_storage->ne[0], (long long) k_storage->ne[1], (long long) k_storage->ne[2], (long long) k_storage->ne[3],
                (long long) v_storage->ne[0], (long long) v_storage->ne[1], (long long) v_storage->ne[2], (long long) v_storage->ne[3],
                k_storage->nb[1], v_storage->nb[1]);
    }

    // Extent used for the overflow K/V and mask. Identical to the gather path
    // (n_ovf_used, the padded occupancy) on purpose: the isolated op test
    // tests/test-fa-q4-view.cpp shows that a valid-vs-padded extent difference
    // alone moves single-query O by ~1.4e-4 because split_k changes, which is the
    // magnitude class of the model-level packed residual. Rows past the live run
    // are masked -inf by the range mask, so no inactive row is attended.
    uint32_t n_ovf_rows = n_ovf_used;
    if (ovf_use_view) {
        // Stable-extent policy (measured): using the LIVE extent as the baked size
        // forces a graph rebuild on every decode token (reuse guard refuses), which
        // measured 226 ms/token vs 88 ms/token for the capacity-stable gather path
        // at 36K (docs/occupancy-evidence/profile-prof-q4cont*.log). Bake the live
        // run rounded up to 256 rows instead, capped at the range capacity: the
        // extent stays constant between 256-token growth steps, so graphs are
        // reused; rows past the live run stay masked -inf and are zeroed below.
        // (Baking the WHOLE range capacity was tried and is both wasteful - ~73 MB
        // of tail zeroing per layer - and numerically wrong for the raw Q4 FA:
        // capq4 vs capf16 measured KL 1.89e-3.)
        uint32_t stable = (ovf_n_view + 255u) & ~255u;
        if (stable > n_overflow) { stable = n_overflow; }
        if (stable > 0 && uint64_t(ovf_physical_row0) + stable <= k_rows &&
                uint64_t(ovf_physical_row0) + stable <= v_rows) {
            n_ovf_rows = stable;
            ovf_path = "packed_view_stable256";
        } else if (uint64_t(ovf_physical_row0) + n_ovf_used > k_rows ||
                   uint64_t(ovf_physical_row0) + n_ovf_used > v_rows) {
            ovf_use_view = false;
            ovf_path = "indexed_gather_extent_padded";
        }
    }

    // Raw Q4_0 FA selection. LLAMA_OVF_PACKED_Q4_FA=1 enables every layer;
    // LLAMA_OVF_PACKED_Q4_FA_IL=3,7,11 enables only the listed layers, which is
    // how the first divergent layer is isolated against the F16 reference.
    // Hoisted above the fused-span gate below: the fused executor always reads
    // the packed window through the F16 gather, so a raw-Q4 layer keeps the
    // legacy chain.
    static const bool ovf_q4_fa_all = [] {
        const char * v = std::getenv("LLAMA_OVF_PACKED_Q4_FA");
        return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
    }();
    static const std::string ovf_q4_fa_list = [] {
        const char * v = std::getenv("LLAMA_OVF_PACKED_Q4_FA_IL");
        return v != nullptr ? std::string(v) : std::string();
    }();
    bool ovf_q4_fa = ovf_q4_fa_all;
    if (!ovf_q4_fa && !ovf_q4_fa_list.empty()) {
        size_t a = 0;
        while (a <= ovf_q4_fa_list.size()) {
            const size_t b = ovf_q4_fa_list.find(',', a);
            const std::string tok = ovf_q4_fa_list.substr(
                    a, b == std::string::npos ? std::string::npos : b - a);
            if (!tok.empty() && std::atoi(tok.c_str()) == il) { ovf_q4_fa = true; break; }
            if (b == std::string::npos) { break; }
            a = b + 1;
        }
    }

    // Fused span eligibility ("peça (c)" phase 2, LLAMA_OVF_FUSED_SPAN=1,
    // default OFF -> the legacy chain below stays byte-identical). When
    // eligible, ONE GGML_OP_KVARN_SPLIT_FUSED node replaces the overflow half
    // of this chain: the Q4 store (SET_ROWS), the F16 gather, FA2 + LSE2 (all
    // on Vulkan through the executor's persistent sub-graph) and the LSE merge
    // (on CUDA). The local half above (Q WHT, KVarN FA1 + tail, LSE1, O WHT)
    // stays as ordinary nodes and feeds the merge through the op sources.
    // Eligibility is deliberately narrow; every deviation keeps the legacy
    // chain:
    //   * grouped single-query decode (same arm the consolidated order covers;
    //     prefill and the unanchored F16-tail arm keep their node order),
    //   * an OVERFLOW ubatch: the executor performs the store into the Q4
    //     companion, which is only the store of an overflow ubatch (a local
    //     ubatch writes the local KVarN records and keeps the legacy chain),
    //   * packed overflow row view (a contiguous raw Q4_0 window the executor
    //     can alias without a gather node), with the companion V not
    //     transposed,
    //   * F16 gather readback, no raw-Q4 FA opt-in for this layer,
    //   * contiguous single-query q/k/v rows (the executor stages them as flat
    //     contiguous buffers and its FA has no strided-input contract),
    //   * no alibi bias and no logit softcap (the executor's overflow FA is
    //     built with max_bias/softcap hardcoded to zero, exactly like the
    //     split chain's FA2 for the default hparams),
    //   * no K mean-centering bias on the companion layer (the executor mirrors
    //     cpy_k, which would subtract it),
    //   * no CPU attention-reference oracle (it discovers FA nodes below the
    //     returned output and would find none behind the fused node).
    const bool fused_span_eligible = [] {
        const char * e = std::getenv("LLAMA_OVF_FUSED_SPAN");
        return e != nullptr && e[0] == '1' && e[1] == '\0';
    }() && group_overflow_decode && kvarn_ctx->current_range_is_overflow() &&
        ovf_use_view && ovf_gather_f16 && !ovf_q4_fa &&
        k_storage->type == GGML_TYPE_Q4_0 && v_storage->type == GGML_TYPE_Q4_0 &&
        q_cur->ne[2] == 1 && q_cur->ne[3] == 1 &&
        k_cur->ne[2] == 1 && k_cur->ne[3] == 1 &&
        v_cur->ne[2] == 1 && v_cur->ne[3] == 1 &&
        ggml_is_contiguous(q_cur) && ggml_is_contiguous(k_cur) && ggml_is_contiguous(v_cur) &&
        !standard_cache->v_transposed() &&
        hparams.f_max_alibi_bias == 0.0f && !hparams.attn_soft_cap &&
        !standard_cache->layer_k_bias_present(int32_t(il)) &&
        [] {
            const char * e = std::getenv("LLAMA_OVF_ATTN_REFERENCE_CPU");
            return e == nullptr || e[0] != '1' || e[1] != '\0';
        }();

    // Non-finite stale bytes in the INACTIVE tail of the storage poison the raw
    // Q4 FA even though the range mask is -inf for those rows (measured with
    // FA_Q4_NAN_TAIL in tests/test-fa-q4-view.cpp: O 3.4e38, LSE inf). The gather
    // reference never sees them: it reads a finite duplicated cell-0 row. Zero the
    // tail rows [live, padded) before the graph computes; the graph's own K/V store
    // later writes only the live rows, so the zeros persist for this evaluation.
    static const bool ovf_zero_tail = [] {
        const char * v = std::getenv("LLAMA_OVF_ZERO_TAIL");
        return v == nullptr || std::strcmp(v, "0") != 0;
    }();
    if (ovf_use_view && ovf_zero_tail && n_ovf_rows > ovf_n_view &&
            k_storage->buffer && v_storage->buffer) {
        const size_t k_row = ggml_row_size(k_storage->type, n_embd_k);
        const size_t v_row = ggml_row_size(v_storage->type, n_embd_v);
        const size_t k_off = size_t(ovf_physical_row0 + ovf_n_view) * k_storage->nb[1];
        const size_t v_off = size_t(ovf_physical_row0 + ovf_n_view) * v_storage->nb[1];
        std::vector<uint8_t> zk(k_row * (n_ovf_rows - ovf_n_view), 0);
        std::vector<uint8_t> zv(v_row * (n_ovf_rows - ovf_n_view), 0);
        ggml_backend_tensor_set(k_storage, zk.data(), k_off, zk.size());
        ggml_backend_tensor_set(v_storage, zv.data(), v_off, zv.size());
        if (std::getenv("F4_ALLOC_TRACE") != nullptr) {
            fprintf(stderr, "[F5OVF] il=%d zeroed inactive tail rows=%u bytes=%zu|%zu\n",
                    il, n_ovf_rows - ovf_n_view, zk.size(), zv.size());
        }
    }

    // Record the physical map that this graph bakes in, so can_reuse() can refuse a
    // graph whose view would no longer describe the live rows.
    inp->split_ovf_packed = ovf_use_view;
    inp->split_ovf_row0   = ovf_row0;
    inp->split_ovf_n_view = ovf_n_view;
    inp->split_ovf_n_rows = n_ovf_rows;

    ggml_tensor * k_ovf_gathered = nullptr;
    ggml_tensor * v_ovf_gathered = nullptr;
    // Raw packed row views, kept separately for the fused-span gate below: the
    // executor consumes them directly (aliased into its Vulkan sub-graph)
    // instead of the gathered readback.
    ggml_tensor * k_ovf_raw = nullptr;
    ggml_tensor * v_ovf_raw = nullptr;
    if (ovf_use_view) {
        const uint64_t k_off = uint64_t(ovf_physical_row0) * k_storage->nb[1];
        const uint64_t v_off = uint64_t(ovf_physical_row0) * v_storage->nb[1];
        k_ovf_gathered = ggml_view_2d(ctx0, k_storage, n_embd_k, int64_t(n_ovf_rows), k_storage->nb[1], k_off);
        v_ovf_gathered = ggml_view_2d(ctx0, v_storage, n_embd_v, int64_t(n_ovf_rows), v_storage->nb[1], v_off);
        ggml_set_name(k_ovf_gathered, "attn_k_ovf_packed_view");
        ggml_set_name(v_ovf_gathered, "attn_v_ovf_packed_view");
        k_ovf_raw = k_ovf_gathered;
        v_ovf_raw = v_ovf_gathered;
        // The packed view rows are read back as F16 through the SUPPORTED GET_ROWS
        // path over an in-graph identity index list. Measured: this matches the gather
        // reference (mean KL 8.7e-4 at 3000/3072/3200/3328/3584, all argmax equal),
        // while feeding the raw Q4_0 view to the Vulkan FA diverges (mean KL 4.83,
        // argmax flips) because the quantized FA path and the F16 path do not agree
        // for this layout; Vulkan has no dequant_transpose pipeline for Q4_0
        // (ggml-vulkan.cpp:6044 creates it only for Q8_0), so the F16 scratch path
        // cannot be selected. Keep the raw Q4_0 FA behind an explicit opt-in until
        // it is fixed and gated; bounded by the live view extent, no whole-cache
        // expansion and no format change (storage stays Q4_0).
        // Raw Q4_0 FA selection (hoisted above the fused-span gate).
        if (!ovf_q4_fa) {
            if (inp->self_ovf_view_rows == nullptr) {
                throw std::runtime_error(
                    "packed overflow view requires its identity row input");
            }
            ggml_tensor * rows = ggml_view_1d(ctx0, inp->self_ovf_view_rows,
                    int64_t(n_ovf_rows), 0);
            k_ovf_gathered = ggml_get_rows_as(ctx0, k_ovf_gathered, rows, GGML_TYPE_F16);
            v_ovf_gathered = ggml_get_rows_as(ctx0, v_ovf_gathered, rows, GGML_TYPE_F16);
            ggml_set_name(k_ovf_gathered, "attn_k_ovf_view_f16");
            ggml_set_name(v_ovf_gathered, "attn_v_ovf_view_f16");
            fprintf(stderr, "[F5PROBE] il=%d packed_view_getrows_f16 n=%u (padded extent)\n", il, n_ovf_rows);
        } else {
            static const bool cont_kv = [] {
                const char * v = std::getenv("F5_OVF_CONT_KV");
                return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
            }();
            if (cont_kv) {
                // Candidate fix: materialize the Q4 view with a real copy op. This
                // (a) creates an explicit store->read dependency edge for the
                // scheduler/barrier tracking and (b) gives the FA a tensor with its
                // own buffer. q4->q4 copy is far cheaper than the F16 gather.
                static const bool k_only_cont = [] {
                    const char * v = std::getenv("F5_OVF_CONT_K_ONLY");
                    return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
                }();
                k_ovf_gathered = ggml_cont(ctx0, k_ovf_gathered);
                ggml_set_name(k_ovf_gathered, "attn_k_ovf_cont");
                if (!k_only_cont) {
                    v_ovf_gathered = ggml_cont(ctx0, v_ovf_gathered);
                    ggml_set_name(v_ovf_gathered, "attn_v_ovf_cont");
                }
            }
            if (std::getenv("F5_OVF_EXPAND_VIEWS") != nullptr) {
                // Zero-copy lifetime experiment: make the Q4 views explicit graph
                // outputs so the scheduler cannot alias their buffer with a writer.
                expand_overflow_probe(k_ovf_gathered);
                expand_overflow_probe(v_ovf_gathered);
            }
            if (std::getenv("F5_LAYER_CAPTURE") != nullptr) {
            // DIAGNOSTIC: the SAME view rows read back as F16, expanded into the
            // graph only so the dump can compare the effective K/V values of the
            // raw-Q4 leg against the gather reference leg.
            if (inp->self_ovf_view_rows == nullptr) {
                throw std::runtime_error("packed overflow view requires its identity row input");
            }
            static const bool one_row = [] {
                const char * v = std::getenv("F5_DBG_ONE_ROW");
                return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
            }();
            ggml_tensor * rows = ggml_view_1d(ctx0, inp->self_ovf_view_rows,
                    one_row ? 1 : int64_t(n_ovf_rows), 0);
            ggml_tensor * k_dbg = ggml_get_rows_as(ctx0, k_ovf_gathered, rows, GGML_TYPE_F16);
            ggml_tensor * v_dbg = ggml_get_rows_as(ctx0, v_ovf_gathered, rows, GGML_TYPE_F16);
            static const bool k_only = [] {
                const char * v = std::getenv("F5_DBG_K_ONLY");
                return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
            }();
            char nm[64];
            snprintf(nm, sizeof(nm), "f5k16-%d", il); ggml_set_name(k_dbg, nm);
            snprintf(nm, sizeof(nm), "f5v16-%d", il); ggml_set_name(v_dbg, nm);
            expand_overflow_probe(k_dbg);
            if (!k_only) {
                expand_overflow_probe(v_dbg);
            }
            }
        }
    } else {
        k_ovf_gathered = ovf_gather_f16
            ? ggml_get_rows_as(ctx0, k_storage_2d, ovf_idxs, GGML_TYPE_F16)
            : ggml_get_rows(ctx0, k_storage_2d, ovf_idxs);
        v_ovf_gathered = ovf_gather_f16
            ? ggml_get_rows_as(ctx0, v_storage_2d, ovf_idxs, GGML_TYPE_F16)
            : ggml_get_rows(ctx0, v_storage_2d, ovf_idxs);
    }
    f4_tensor_trace("gather", "k_ovf_gathered", k_ovf_gathered);
    f4_tensor_trace("gather", "v_ovf_gathered", v_ovf_gathered);
    f4_tensor_trace("mask",   "mask_overflow",  mask_overflow);
    f4_tensor_trace("store",  "k_storage",      k_storage);
    f4_tensor_trace("store",  "v_storage",      v_storage);
    const int64_t n_head_kv = hparams.n_head_kv(il);
    const int64_t head_dim_k = hparams.n_embd_head_k(il);
    const int64_t head_dim_v = hparams.n_embd_head_v(il);
    GGML_ASSERT(n_embd_k == head_dim_k * n_head_kv);
    GGML_ASSERT(n_embd_v == head_dim_v * n_head_kv);
    // The overflow FA reads the mask with stride = KV (the K/V extent):
    // vulkan-shaders/flash_attn_base.glsl:213 sets `m_stride = KV`, and
    // flash_attn.comp:222 indexes `data_m[(i*Br + r)*m_stride + ...]`. A mask
    // WIDER than the packed K/V therefore makes query rows past the first read
    // the wrong mask row (live640 with padded768 reproduced exactly this).
    // Crop AND materialize a CONTIGUOUS mask with exactly the extent the K/V
    // carries, and use that same extent as n_kv_max below. The capacity-sized
    // graph input keeps its allocation and fill order; only the compute view
    // shrinks, so no KV format is converted and no padding row is unmasked.
    if (mask_overflow->ne[0] != int64_t(n_ovf_rows)) {
        ggml_tensor * mask_src = mask_overflow;
        if (mask_src->ne[0] < int64_t(n_ovf_rows)) {
            throw std::runtime_error(
                "position-split overflow mask is narrower than the packed K/V extent");
        }
        // Same memoized crop+CONT as the occupied-extent path above: the
        // stable packed-view extent is layer-independent too, so one
        // materialized mask serves every layer of the graph.
        mask_overflow = build_ovf_mask_crop(mask_src, int64_t(n_ovf_rows));
    }
    ggml_tensor * k_overflow = ggml_reshape_4d(ctx0, k_ovf_gathered,
        head_dim_k, n_head_kv, int64_t(n_ovf_rows), 1);
    ggml_tensor * v_overflow = ggml_reshape_4d(ctx0, v_ovf_gathered,
        head_dim_v, n_head_kv, int64_t(n_ovf_rows), 1);
    // F4: mark the overflow K/V the second FA consumes, so executed-node
    // capture covers the dual-range path too (fixture-gated, no-op otherwise).
    kvarn_ctx->mark_split_tensor(k_overflow, il, false, llama_position_split::kRangeOverflow);
    kvarn_ctx->mark_split_tensor(v_overflow, il, true,  llama_position_split::kRangeOverflow);

    // Masks per range (filled in set_input from live cell lists).
    ggml_tensor * mask_local = inp->self_kq_mask_local_cnv;
    GGML_ASSERT(mask_local != nullptr && mask_overflow != nullptr);

    // Local body + exact tail (plan 3.4: the LOCAL LSE must normalize the
    // COMPLETE local output, body + tail KVarN, before it is merged with the
    // Q4 overflow). fa2 stays tail-free (plan 3.1). The operands are the very
    // same inputs the single-range path uses, so the body/tail exactly-once
    // bookkeeping (tail_read_idxs, kq_mask_tail, query order, run descriptor)
    // is shared rather than reimplemented for this builder.
    ggml_tensor * k_tail_local   = nullptr;
    ggml_tensor * v_tail_local   = nullptr;
    ggml_tensor * mask_tail_local = nullptr;
    ggml_tensor * bias_tail_local = nullptr;
    ggml_tensor * tail_idx_local  = nullptr;
    ggml_tensor * tail_order_local = nullptr;
    ggml_tensor * tail_desc_local  = nullptr;
    llama_kv_tail_route route_local = LLAMA_KV_TAIL_ROUTE_NONE;
    if (local_tail_here) {
        route_local = kvarn_ctx->get_tail_route(il);
        k_tail_local = kvarn_ctx->get_k_tail(ctx0, il);
        v_tail_local = kvarn_ctx->get_v_tail(ctx0, il);
        tail_idx_local = inp->get_tail_read_idxs();
        if (route_local == LLAMA_KV_TAIL_ROUTE_NONE) {
            k_tail_local = nullptr;
            v_tail_local = nullptr;
        }
        // Fail closed instead of building a half-wired tail: a present tail
        // without its read indices would silently read the wrong keys.
        if (k_tail_local != nullptr && (v_tail_local == nullptr || tail_idx_local == nullptr)) {
            throw std::logic_error(format(
                    "position-split layer %d: local exact tail present without "
                    "its V or read-index operand", il));
        }
        if (k_tail_local != nullptr) {
            mask_tail_local = inp->get_kq_mask_tail();
            bias_tail_local = build_attn_bias_tail(
                    nullptr, inp->get_tail_bias_read_idxs(), mask_tail_local);
            tail_order_local = inp->get_tail_query_order();
            tail_desc_local  = inp->get_tail_run_desc();
            // Same domain rule as the single-range path: the stored tail is
            // unrotated, the local body records are rotated.
            GGML_ASSERT(kvarn_rot != nullptr);
            // NO current segment here: this builder runs for an OVERFLOW
            // ubatch, which writes Q4 overflow keys, so the local tail is read
            // purely as HISTORY (validate_native_tail_operation(..., false)).
            // Overflow keys must never enter the local tail.
            k_tail_local = ggml_kvarn_wht_aux(ctx0, k_tail_local, kvarn_rot->ne[0]);
            // Same rule as the single-range path above: the FA un-rotates the
            // tail V too, so the stored (original-domain) tail V must be rotated
            // in every domain. See the measurement cited there.
            v_tail_local = ggml_kvarn_wht_aux(ctx0, v_tail_local, kvarn_rot->ne[0]);
            // Same domain rule for the current segment: K is always rotated for
            // the KVarN Q-rot path, V only in the rotated-domain case.
        }
    }

    // Two FlashAttention ops with LSE side outputs. No sinks/bias/softcap in
    // P-mode (fail closed below; plan 3.4 keeps sinks out of the first mode);
    // precision F32 like the single-range path.
    ggml_tensor * fa1 = nullptr;
    ggml_tensor * fa2 = nullptr;
    ggml_tensor * final_local_tail_op = nullptr;
    // Argument order mirrors build_attn() exactly: slot 5 is the MODEL's KQ
    // bias (must stay null -- a non-null value there silently disables the
    // FlashAttention branch and leaves final_attn_op unset), and the tail bias
    // is slot 14. Passing the tail bias one slot early compiled fine and only
    // failed at the GGML_ASSERT(fa1 != nullptr) below.
    ggml_tensor * o1 = build_attn_mha(q_local, k_local, v_local,
        nullptr, mask_local, nullptr, nullptr, int64_t(n_local),
        kq_scale, il,
        k_tail_local, v_tail_local, mask_tail_local, bias_tail_local,
        tail_idx_local, tail_order_local, tail_desc_local,
        route_local, kvarn_domain,
        nullptr, nullptr, kvarn_ctx->get_tail_slots(),
        !kvarn_ctx->has_kv_body(il), &final_local_tail_op);
if (getenv("F4_TAIL_TRACE")) {
    fprintf(stderr, "F4SPLIT il=%d local_tail=%d route_local=%d k_tail=%p v_tail=%p "
            "mask_tail=%p bias_tail=%p idx=%p order=%p desc=%p n_local=%d\n",
            il, (int) local_tail_here, (int) route_local,
            (void *) k_tail_local, (void *) v_tail_local, (void *) mask_tail_local,
            (void *) bias_tail_local, (void *) tail_idx_local,
            (void *) tail_order_local, (void *) tail_desc_local, (int) n_local);
}
// build_attn_mha returns the FINAL (possibly tail-merged) node through
// final_local_tail_op; fa1 is what the merge below consumes and what the
// validator must see.
fa1 = final_local_tail_op;
GGML_ASSERT(fa1 != nullptr);
if (route_local == LLAMA_KV_TAIL_ROUTE_NATIVE) {
    validate_native_tail_operation(inp->mctx, il, final_local_tail_op, false);
}
ggml_tensor * lse1 = ggml_flash_attn_ext_lse_out(ctx0, fa1);

    // Local O back to the original V domain before merging (same rule as the
    // single-range path: the KVarN native output is rotated unless the backend
    // uses the original V domain).
    if (use_kvarn_rotated_domain) {
        GGML_ASSERT(o1->type == GGML_TYPE_F32);
        GGML_ASSERT(kvarn_rot != nullptr);
        o1 = ggml_kvarn_wht_aux(ctx0, o1, kvarn_rot->ne[0]);
    }

    // Split input tensors (masks/idxs) stay on host CPU so set_input can fill
    // them; the scheduler copies to device backends as needed. backend_cpu
    // may be null, so locate a host backend from the scheduler explicitly.
    // Resolved before the fused decision: both arms need the same placement.
    ggml_backend_t backend_host = backend_cpu;
    if (!backend_host) {
        for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
            ggml_backend_t be = ggml_backend_sched_get_backend(sched, i);
            ggml_backend_dev_t dev = be ? ggml_backend_get_device(be) : nullptr;
            if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                backend_host = be;
                break;
            }
        }
    }
    if (!backend_host) {
        throw std::runtime_error("position-split attention requires a host CPU backend for range inputs");
    }
    ggml_backend_sched_set_tensor_backend(sched, inp->self_kq_mask_local, backend_host);
    ggml_backend_sched_set_tensor_backend(sched, inp->self_kq_mask_overflow, backend_host);
    ggml_backend_sched_set_tensor_backend(sched, inp->self_kvarn_mat_idxs_local, backend_host);
    ggml_backend_sched_set_tensor_backend(sched, inp->self_overflow_cell_idxs, backend_host);
    ggml_backend_t backend_vulkan = cparams.local_attn_backend;
    if (!backend_vulkan) {
        throw std::runtime_error("position-split attention requires the Vulkan overflow backend");
    }
    ggml_backend_t backend_cuda = nullptr;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        ggml_backend_t be = ggml_backend_sched_get_backend(sched, i);
        ggml_backend_dev_t dev = be ? ggml_backend_get_device(be) : nullptr;
        const char * name = dev ? ggml_backend_dev_name(dev) : nullptr;
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU &&
                name && std::strncmp(name, "CUDA", 4) == 0) {
            backend_cuda = be;
            break;
        }
    }
    if (!backend_cuda) {
        throw std::runtime_error("position-split attention requires a CUDA backend for local/merge");
    }

    // Fused-span acceptance MUST be decided before the deferred legacy store
    // expansion below: the fused executor performs the companion store itself,
    // so when the span is suppressed its store must not be dropped. Building
    // the node is pure metadata; the backend answers through supports_op (and
    // a test-only force-refuse hook can emulate a refusal).
    static const bool fused_span_force_refuse = [] {
        const char * e = std::getenv("LLAMA_OVF_FUSED_SPAN_FORCE_REFUSE");
        return e != nullptr && e[0] == '1' && e[1] == '\0';
    }();
    ggml_tensor * fused = nullptr;
    if (fused_span_eligible) {
        // Fused-span acceptance MUST be decided before the deferred legacy
        // store expansion below (see the comment above). The executor bundle
        // is per context; acquire it lazily (id baked into the op params).
        const uint32_t bundle_id = kvarn_cache->acquire_fused_span_bundle(backend_cuda, backend_vulkan);
        if (bundle_id == 0) {
            LLAMA_LOG_WARN("%s: layer %d fused split span has no executor bundle; using the legacy chain\n",
                    __func__, il);
        } else {
            // ONE node replacing the overflow half of the chain: store + gather
            // + FA2 + LSE2 (Vulkan, executor sub-graph) and the LSE merge
            // (CUDA), returning the same o_merged the legacy chain produces.
            // The local results (o1 after the domain WHT, lse1) and the raw
            // packed views travel as op sources.
            ggml_tensor * candidate = ggml_kvarn_split_fused(ctx0, q_cur, k_cur, v_cur,
                    o1, lse1, k_ovf_raw, v_ovf_raw, mask_overflow,
                    inp->get_k_idxs(), inp->get_v_idxs(),
                    int(p_split), int(n_kv_live),
                    kvarn_cache->params_key_bits(), kvarn_cache->params_value_bits(),
                    int(n_ovf_rows), kq_scale, il, int(bundle_id));
            if (!fused_span_force_refuse && ggml_backend_supports_op(backend_cuda, candidate)) {
                fused = candidate;
            } else {
                LLAMA_LOG_WARN("%s: layer %d fused split span refused%s; using the legacy chain\n",
                        __func__, il, fused_span_force_refuse ? " by LLAMA_OVF_FUSED_SPAN_FORCE_REFUSE" : " by the CUDA backend");
            }
        }
    }

    if (group_overflow_decode) {
        // build_attn_mha already expanded fa1. Put its side output here, before
        // build_attn_mha expands fa2, rather than reopening CUDA after fa2.
        ggml_build_forward_expand(gf, lse1);
        if (fused == nullptr) {
            // Legacy (and refused-fused) arm: the deferred overflow store and
            // the diagnostic probes must be expanded here, or the current
            // token would never reach the Q4 companion.
            ggml_build_forward_expand(gf, k_written);
            ggml_build_forward_expand(gf, v_written);
            for (ggml_tensor * probe : overflow_probes) {
                ggml_build_forward_expand(gf, probe);
            }
        }
    }

    if (fused != nullptr) {
        // Register the host-side executor (once per CUDA/Vulkan pair); it
        // owns the Vulkan overflow sub-graph and the CUDA merge sub-graph.
        if (!llama_kvarn_split_fused_exec_register(backend_cuda, backend_vulkan)) {
            throw std::runtime_error("position-split fused span: executor registration failed");
        }
        // Placement: the local chain keeps its legacy CUDA pins (the
        // executor reads o1/lse1 there and writes the merged output), the
        // raw overflow views stay on Vulkan (the scheduler copy for this
        // op's src[5..9] is suppressed), host range inputs stay on host.
        ggml_backend_sched_set_tensor_backend(sched, fused, backend_cuda);
        ggml_backend_sched_set_tensor_backend(sched, fa1, backend_cuda);
        ggml_backend_sched_set_tensor_backend(sched, lse1, backend_cuda);
        ggml_backend_sched_set_tensor_backend(sched, o1, backend_cuda);
        // Cropped mask: pin the CONT to CUDA so it stays inside the single
        // local cluster (the executor reads it back from there). When no
        // crop was needed the mask IS the host graph input, which must
        // stay host (set_input writes it there) and is read directly.
        if (mask_overflow != inp->self_kq_mask_overflow_cnv) {
            ggml_backend_sched_set_tensor_backend(sched, mask_overflow, backend_cuda);
        }
        // Same Q-WHT pin as the legacy arm below (see the long comment
        // there): CUDA when opted in and supported, else Vulkan.
        if (use_kvarn_q_rot && ovf_qwht_cuda && ggml_backend_supports_op(backend_cuda, q_local)) {
            ggml_backend_sched_set_tensor_backend(sched, q_local, backend_cuda);
            if (q_local->src[0]->op == GGML_OP_CONT) {
                ggml_backend_sched_set_tensor_backend(sched, q_local->src[0], backend_cuda);
            }
        } else if (use_kvarn_q_rot && (group_overflow_decode || ovf_qwht_cuda) &&
                ggml_backend_supports_op(backend_vulkan, q_local)) {
            ggml_backend_sched_set_tensor_backend(sched, q_local, backend_vulkan);
            if (q_local->src[0]->op == GGML_OP_CONT) {
                ggml_backend_sched_set_tensor_backend(sched, q_local->src[0], backend_vulkan);
            }
        }
        if (tail_local_cuda_enabled && k_tail_local != nullptr) {
            ggml_backend_sched_set_tensor_backend(sched, k_tail_local, backend_cuda);
            if (v_tail_local != nullptr) {
                ggml_backend_sched_set_tensor_backend(sched, v_tail_local, backend_cuda);
            }
        }
        // The final local FA node must carry the KVarN domain or the CUDA
        // KVarN gate refuses the shape (see the legacy comment below).
        llm_flash_attn_ext_set_kvarn_domain(fa1, kvarn_domain);
        ggml_build_forward_expand(gf, fused);
        LLAMA_LOG_INFO("%s: layer %d position-split fused span: local=%u keys (CUDA/KVarN), overflow=%u keys (Vulkan/Q4) + merge in one op\n",
                __func__, il, n_local, n_ovf_rows);
        cb(fused, "attn_position_split_fused", il);
        return fused;
    }

    if (std::getenv("F4_NODE_DUMP") != nullptr) {
        cb(q_cur, "diag_query", il);
        cb(fa1, "diag_local_fa", il);
        cb(o1, "diag_local_out", il);
        cb(lse1, "diag_local_lse", il);
    }
    ggml_tensor * o2 = build_attn_mha(q_cur, k_overflow, v_overflow,
        nullptr, mask_overflow, nullptr, nullptr, int64_t(n_ovf_rows),
        kq_scale, il,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr,
    static_cast<llama_kv_tail_route>(0),
    GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_AUTO,
    nullptr, nullptr, 0, false, &fa2);
GGML_ASSERT(fa2 != nullptr);
ggml_tensor * lse2 = ggml_flash_attn_ext_lse_out(ctx0, fa2);
if (std::getenv("F4_NODE_DUMP") != nullptr) {
    cb(o2, "diag_remote_out", il);
    cb(lse2, "diag_remote_lse", il);
}
ggml_prec_set_acc(fa1, GGML_PREC_F32);
ggml_prec_set_acc(fa2, GGML_PREC_F32);
if (std::getenv("F5_LAYER_CAPTURE") != nullptr) {
    // Unique per-layer names so F4_NODE_DUMP can compare the overflow FA output
    // and its LSE between the raw-Q4 and the F16 reference legs. Only the
    // per-layer tensors are renamed: mask_overflow is the GRAPH-SHARED mask
    // (the memoized crop named attn_inp_kq_mask_overflow_occ by
    // build_ovf_mask_crop, or the attn_inp_kq_mask_overflow graph input when no
    // crop was needed), consumed by every layer. A per-layer rename would keep
    // only the last layer's f5mask-<il> on the shared node and clobber the
    // stable memo name, so its STABLE NAME is the capture selector: list
    // attn_inp_kq_mask_overflow_occ / attn_inp_kq_mask_overflow in
    // F4_NODE_DUMP_ROLES to capture the mask (identical for all layers).
    char nm[64];
    snprintf(nm, sizeof(nm), "f5fa2-%d", il);  ggml_set_name(fa2,  nm);
    snprintf(nm, sizeof(nm), "f5lse2-%d", il); ggml_set_name(lse2, nm);
    snprintf(nm, sizeof(nm), "f5k-%d", il);    ggml_set_name(k_overflow, nm);
    snprintf(nm, sizeof(nm), "f5v-%d", il);    ggml_set_name(v_overflow, nm);
}
    // F4: the KVarN domain must land on the FINAL FA node, which is not
    // necessarily the node build_attn_mha created internally (it may be
    // re-wrapped). The single-range paths already do this after the call
    // (llm_flash_attn_ext_set_kvarn_domain); the split path did not, so the
    // local range kept whatever op_params happened to contain. That value is
    // not a valid domain id, the CUDA KVarN gate then refused the shape, and
    // ggml_cuda_flash_attn_ext aborted the whole process ("unsupported KVarN
    // CUDA FlashAttention route"). Set it explicitly on both range nodes.
    llm_flash_attn_ext_set_kvarn_domain(fa1, kvarn_domain);
    llm_flash_attn_ext_set_kvarn_domain(fa2, GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_AUTO);

    // Stable merge (validated pattern from test-position-split-merge-graph).
    // O is [D,H,Q,B] after restoring the 4D layout; LSE is [H,Q,B].
    const int64_t n_qh = lse1->ne[0];
    const int64_t n_q = lse1->ne[1];
    const int64_t n_b = lse1->ne[2];
    GGML_ASSERT(lse2->ne[0] == n_qh && lse2->ne[1] == n_q && lse2->ne[2] == n_b);
    const int64_t head_dim = q_cur->ne[0];
    // build_attn_mha returns O flattened as [(D*H), (Q*B)]; restore 4D.
    GGML_ASSERT(o1->ne[0] == head_dim*n_qh && o1->ne[1] == n_q*n_b);
    GGML_ASSERT(o2->ne[0] == head_dim*n_qh && o2->ne[1] == n_q*n_b);
    ggml_tensor * o1_4d = ggml_reshape_4d(ctx0, o1, head_dim, n_qh, n_q, n_b);
    ggml_tensor * o2_4d = ggml_reshape_4d(ctx0, o2, head_dim, n_qh, n_q, n_b);
    ggml_tensor * mx = llama_position_split_lse_max(ctx0, lse1, lse2);
    ggml_tensor * e1 = ggml_exp(ctx0, ggml_sub(ctx0, lse1, mx));
    ggml_tensor * e2 = ggml_exp(ctx0, ggml_sub(ctx0, lse2, mx));
    ggml_tensor * w1 = ggml_reshape_4d(ctx0, e1, 1, n_qh, n_q, n_b);
    ggml_tensor * w2 = ggml_reshape_4d(ctx0, e2, 1, n_qh, n_q, n_b);
    ggml_tensor * num = ggml_add(ctx0, ggml_mul(ctx0, o1_4d, w1), ggml_mul(ctx0, o2_4d, w2));
    ggml_tensor * den = ggml_add(ctx0, e1, e2);
    // Both ranges are non-empty here (validated above), so den > 0 always:
    // plain division is exact with no epsilon, and merged LSE is exact.
    // Broadcast den as [1,H,Q,B] like the validated merge-graph pattern.
    ggml_tensor * den_4d = ggml_reshape_4d(ctx0, den, 1, n_qh, n_q, n_b);
    ggml_tensor * o_merged_4d = ggml_div(ctx0, num, den_4d);
    ggml_tensor * lse_merged = ggml_add(ctx0, mx, ggml_log(ctx0, den));
    ggml_tensor * o_merged = ggml_reshape_2d(ctx0, o_merged_4d,
        o_merged_4d->ne[0]*o_merged_4d->ne[1], o_merged_4d->ne[2]*o_merged_4d->ne[3]);
    // Materialize to a fresh contiguous buffer (not a view) so the returned
    // O always owns an allocated output buffer. A reshape view of the div
    // output was left without a valid buffer by the scheduler in the
    // multi-backend split graph (isolated: standalone mul read OOB on the
    // view while sigmoid was clean; materializing eliminates it). Cheap and
    // permanent.
    o_merged = ggml_cont_2d(ctx0, o_merged, o_merged->ne[0], o_merged->ne[1]);

    // Explicitly expand every split FA/LSE/merge node in dependency order.
    // The scheduler must allocate all of them; relying solely on output
    // dependencies left the merged output unallocated in one configuration
    // (isolated via no-fusion run: standalone mul read OOB on o_merged
    // while sigmoid was clean).
    ggml_build_forward_expand(gf, fa1);
    ggml_build_forward_expand(gf, lse1);
    ggml_build_forward_expand(gf, fa2);
    ggml_build_forward_expand(gf, lse2);
    const int merge_begin = ggml_graph_n_nodes(gf);
    ggml_build_forward_expand(gf, o_merged_4d);
    ggml_build_forward_expand(gf, lse_merged);
    ggml_build_forward_expand(gf, o_merged);

    // Backend placement (host inputs, backend_host/vulkan/cuda resolution and
    // the Q-WHT pin moved above, before the fused-span branch): overflow
    // FA/LSE/K/V/mask and merge intermediates on Vulkan; local FA/LSE and
    // final outputs on CUDA. This is still TWO Vulkan steps per rotated-query
    // layer: the Q WHT runs first as its own pre-local split (pinned on Vulkan
    // in the grouped decode; naturally placed onto Vulkan, its legacy backend,
    // in prefill), then this overflow cluster. With LLAMA_OVF_QWHT_CUDA=1 the
    // Q WHT joins the local CUDA cluster in both arms and only this overflow
    // cluster remains (see the pin above). Q itself stays on CUDA; the
    // scheduler copies it to Vulkan for FA2.
    ggml_backend_sched_set_tensor_backend(sched, fa2, backend_vulkan);
    ggml_backend_sched_set_tensor_backend(sched, lse2, backend_vulkan);
    ggml_backend_sched_set_tensor_backend(sched, k_overflow, backend_vulkan);
    ggml_backend_sched_set_tensor_backend(sched, v_overflow, backend_vulkan);
    ggml_backend_sched_set_tensor_backend(sched, mask_overflow, backend_vulkan);
    ggml_backend_sched_set_tensor_backend(sched, k_ovf_gathered, backend_vulkan);
    ggml_backend_sched_set_tensor_backend(sched, v_ovf_gathered, backend_vulkan);
    if (group_overflow_decode) {
        // These intermediates already followed lse2 onto Vulkan in the legacy
        // graph. Make that placement explicit: CUDA expf/logf and Vulkan
        // exp/log are not bit-identical merely because both tensors are F32.
        // This includes the original-domain O WHT; the final output CONT stays
        // CUDA-pinned below with exactly the same arithmetic, and the unread
        // merged LSE follows the scheduler.
        for (int i = merge_begin; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            if (node != o_merged && node != lse_merged && !node->view_src &&
                    ggml_backend_supports_op(backend_vulkan, node)) {
                ggml_backend_sched_set_tensor_backend(sched, node, backend_vulkan);
            }
        }
    }
    // Q-rotation backend pin, shared by the grouped decode AND the prefill /
    // unanchored arms. The pre-consolidation graph ran this F32 WHT on Vulkan
    // as its own pre-local split, and F32 alone does not guarantee identical
    // WHT rounding across CUDA/Vulkan.
    //   LLAMA_OVF_QWHT_CUDA absent/"0" (default): keep today's placement in
    //     both arms byte for byte. Grouped decode preserves the legacy Vulkan
    //     backend even though the overflow stores no longer precede it; the
    //     prefill/unanchored arms keep the WHT unpinned, which reproduces the
    //     same legacy pre-local Vulkan split by scheduler placement.
    //     Unsupported Vulkan widths (64) retain the scheduler's existing
    //     supported-backend fallback.
    //   LLAMA_OVF_QWHT_CUDA=1 (opt-in): pin the Q WHT to the local CUDA
    //     backend (the backend fa1/o_merged use) in both arms, removing the
    //     pre-local Vulkan split so the whole local FA1/LSE1 path runs in one
    //     CUDA cluster. At head width 256 the CUDA WHT bits differ from the
    //     Vulkan ones, so this arm is NOT bit-identical to the default: it is
    //     validated by tolerance (nrmse) in tests/test-position-split-merge-graph
    //     and must not silently become the default. The WHT's CONT input moves
    //     with it (the minimal neighbor needed to keep the transform in one
    //     cluster); when CUDA cannot run the shape, the pin falls back to the
    //     legacy Vulkan placement instead of leaving the WHT to an unpinned
    //     scheduler choice.
    if (use_kvarn_q_rot && ovf_qwht_cuda && ggml_backend_supports_op(backend_cuda, q_local)) {
        ggml_backend_sched_set_tensor_backend(sched, q_local, backend_cuda);
        if (q_local->src[0]->op == GGML_OP_CONT) {
            ggml_backend_sched_set_tensor_backend(sched, q_local->src[0], backend_cuda);
        }
    } else if (use_kvarn_q_rot && (group_overflow_decode || ovf_qwht_cuda) &&
            ggml_backend_supports_op(backend_vulkan, q_local)) {
        ggml_backend_sched_set_tensor_backend(sched, q_local, backend_vulkan);
        if (q_local->src[0]->op == GGML_OP_CONT) {
            ggml_backend_sched_set_tensor_backend(sched, q_local->src[0], backend_vulkan);
        }
    }
    ggml_backend_sched_set_tensor_backend(sched, fa1, backend_cuda);
    ggml_backend_sched_set_tensor_backend(sched, lse1, backend_cuda);
    ggml_backend_sched_set_tensor_backend(sched, o_merged, backend_cuda);
    // lse_merged has no readers: the builder returns o_merged only and the
    // merged LSE is expanded solely as a diagnostic side output, so it is left
    // to the scheduler's placement instead of being CUDA-pinned.
    // Opt-in tail-WHT anchor + grouped decode, armed by the REAL value of
    // LLAMA_OVF_TAIL_LOCAL_CUDA:
    //   unset -> OFF (legacy placement: fa2's Vulkan preset leaks backward
    //            (expand-up) into the tail WHT nodes, ~8 MB/token round trip
    //            of bit-static tails and RADV F16 WHT rounding up to 1 ulp
    //            away from the CPU golden, wht-parity maxabs 1.953e-03)
    //   "0"   -> OFF
    //   "1"   -> ON (k_tail_local/v_tail_local computed on CUDA with fa1;
    //            wht-parity maxabs 6.104e-05) AND the grouped single-query
    //            decode (group_overflow_decode above), which defers the
    //            overflow stores into the Vulkan fa2 cluster
    //   other -> OFF (fail closed, unambiguous)
    // Promoting ON to the default requires an independent reference gate
    // (msg57 item 3); until then this stays an A/B arm only. The env gates
    // the grouped decode in the same arm for the same reason: without the
    // pinned tail, deferring the stores would move the scheduler's anchor for
    // the unanchored tail WHT nodes (see group_overflow_decode).
    // ONLY the WHT compute outputs are anchored here: index/order/desc/mask
    // bookkeeping stays wherever the current placement puts it (some of it
    // is CPU-written and asserts host buffers). If CUDA refuses an op
    // (unsupported shape), set_if_supported falls back safely and staging
    // persists (measurable, no crash). No-ops when the local tail is absent.
    // Other than the grouped decode, the env does not touch fa2/overflow
    // placement.
    if (tail_local_cuda_enabled && k_tail_local != nullptr) {
        ggml_backend_sched_set_tensor_backend(sched, k_tail_local, backend_cuda);
        if (v_tail_local != nullptr) {
            ggml_backend_sched_set_tensor_backend(sched, v_tail_local, backend_cuda);
        }
    }
    o_merged = apply_attention_reference_cpu(o_merged, il);
    cb(o_merged, "attn_position_split_merged", il);
    LLAMA_LOG_INFO("%s: layer %d position-split dual FA: local=%u keys (CUDA/KVarN), overflow=%u keys (Vulkan/Q4), merge outputs on CUDA\n",
            __func__, il, n_local, n_overflow);
    return o_merged;
}

static std::unique_ptr<llm_graph_input_attn_kv_paged> build_attn_inp_kv_paged_impl(
           ggml_context * ctx0,
     const llama_cparams & cparams,
    const llama_hparams & hparams,
    const llama_kv_cache_paged_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_kv_paged>(hparams, cparams, mctx_cur);
    inp->storage_generation = mctx_cur->get_storage_generation();

    const int32_t n_tokens   = mctx_cur->get_n_tokens();
    const int32_t batch_size = mctx_cur->get_batch_size();
    const int32_t max_blocks = mctx_cur->get_max_blocks();

    // Create the GGML descriptors
    inp->paged_write_slots   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    inp->paged_block_table   = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, max_blocks, batch_size);
    inp->paged_context_lens  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, batch_size);
    inp->paged_batch_offsets = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, batch_size);
    inp->paged_batch_lens    = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, batch_size);

    ggml_set_input(inp->paged_write_slots);
    ggml_set_input(inp->paged_block_table);
    ggml_set_input(inp->paged_context_lens);
    ggml_set_input(inp->paged_batch_offsets);
    ggml_set_input(inp->paged_batch_lens);

    return inp;
}

llm_graph_input_attn_kv_paged * llm_graph_context::build_attn_inp_kv_paged() const {
    const auto * mctx_paged = static_cast<const llama_kv_cache_paged_context *>(mctx);

    auto inp = build_attn_inp_kv_paged_impl(ctx0, cparams, hparams, mctx_paged);

    return (llm_graph_input_attn_kv_paged *) res->add_input(std::move(inp));
}

llm_graph_input_i * llm_graph_context::build_attn_inp_kv_auto() const {
    if (cparams.kv_paged) {
        return build_attn_inp_kv_paged();
    }
    return build_attn_inp_kv();
}

static std::unique_ptr<llm_graph_input_attn_k> build_attn_inp_k_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_k>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    return inp;
}

llm_graph_input_attn_k * llm_graph_context::build_attn_inp_k() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_k *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE) {
            // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
            cur = build_lora_mm(wo, cur);
            ggml_prec_set_acc(cur, GGML_PREC_F32);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k_dsa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
        ggml_tensor * top_k,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx->get_mla();

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs_mla();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask_mla();

    // prepare new kq mask - starts filled with -INFINITY
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

    // reshape KQ mask into tensor with rows of size 1:
    // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
    // this will be our source of zero values for unmasking top k mask elements
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    // modify KQ mask by unmasking elements that are in top_k indices
    // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

    // reshape to restore the original shape of KQ mask:
    // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    // combine with the original kq mask
    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask_top_k, sinks, v_mla, top_k->ne[0], kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    auto * k_rot = is_swa ? inp->self_k_rot_swa : inp->self_k_rot;
    auto * v_rot = is_swa ? inp->self_v_rot_swa : inp->self_v_rot;

    const auto * mctx_iswa = inp->mctx;
    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();
    const auto * kvarn_ctx = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx_cur);
    const bool use_kvarn = kvarn_ctx && kvarn_ctx->uses_kvarn_layer(il);
    // Q still has all streams flattened here; build_attn_mha divides its
    // token dimension by the mask's stream count. Route by that per-stream
    // query width, not the batch's total token count (e.g. 30 one-row decodes).
    const auto * route_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();
    GGML_ASSERT(route_mask->ne[3] > 0 && q_cur->ne[2] % route_mask->ne[3] == 0);
    const uint32_t query_tokens_per_stream = uint32_t(q_cur->ne[2] / route_mask->ne[3]);
    const bool kvarn_native_attention = llm_kvarn_native_attention_for(
        kvarn_ctx, arch, cparams.causal_attn, is_swa, il, (int) q_cur->ne[0]) &&
        (!is_swa || route_mask->ne[3] == 1 || query_tokens_per_stream == 1);
    const auto kvarn_plan = use_kvarn ? llama_kvarn_plan_attention(
        kvarn_native_attention,
        kvarn_ctx->native_attention_uses_original_v(il),
        kvarn_ctx->native_rotated_max_query_tokens(il),
        query_tokens_per_stream,
        (int) q_cur->ne[0]) : llama_kvarn_attention_plan {
            false, GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_AUTO };
    if (use_kvarn && arch == LLM_ARCH_DFLASH && !cparams.causal_attn) {
        LLAMA_LOG_DEBUG("%s: DFlash layer %d KVarN attention route=%s\n", __func__, il,
                kvarn_plan.native_attention ? "native" : "materialized");
    }
    const auto kvarn_domain = kvarn_plan.domain;
    const bool use_kvarn_rotated_domain = use_kvarn &&
        kvarn_domain == GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED;
    const bool use_kvarn_mixed_domain = use_kvarn &&
        kvarn_domain == GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_ROTATED_K_ORIGINAL_V;
    const bool use_kvarn_q_rot = use_kvarn_rotated_domain || use_kvarn_mixed_domain;

    if (use_kvarn) {
        GGML_ASSERT(llama_kvarn_head_dim_supported((int) q_cur->ne[0]));
        GGML_ASSERT(k_rot == nullptr || kvarn_ctx->has_standard_cache());
        GGML_ASSERT(v_rot == nullptr || kvarn_ctx->has_standard_cache());
        GGML_ASSERT(!use_kvarn_q_rot || llm_kvarn_rot_for_dim(
                inp->self_kvarn_rot_64, inp->self_kvarn_rot_128,
                inp->self_kvarn_rot_256, inp->self_kvarn_rot_512, q_cur->ne[0]) != nullptr);
    }

    if (!use_kvarn && k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, k_rot);
        if (k_cur) {
            k_cur = llama_mul_mat_hadamard(ctx0, k_cur, k_rot);
        }
    }
    if (!use_kvarn && v_rot) {
        if (v_cur) {
            v_cur = llama_mul_mat_hadamard(ctx0, v_cur, v_rot);
        }
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    if (v_cur) {
        ggml_build_forward_expand(gf, v_cur);
    }

    ggml_tensor * k_tail_written = nullptr;
    ggml_tensor * v_tail_written = nullptr;
    const bool compact_tail = mctx_cur->has_compact_tail(il);

    // whole seq fits into batch in training mode
    const bool use_kv_cur = cparams.training && k_cur && v_cur;
    if (use_kv_cur) {
        GGML_ASSERT(mctx_cur->get_n_kv() == n_tokens);
    }

    const bool store_kv = !use_kv_cur || hparams.n_layer_kv_from_start >= 0;

    // optionally store to KV cache
    if (store_kv && k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        if (compact_tail) {
            if (k_cur) {
                if (ggml_tensor * written = mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il)) {
                    ggml_build_forward_expand(gf, written);
                }
            }
        } else {
            k_tail_written = mctx_cur->cpy_k_with_tail(
                    ctx0, k_cur, k_idxs, inp->get_tail_idxs(is_swa), il);
            if (k_tail_written) {
                ggml_build_forward_expand(gf, k_tail_written);
            } else {
                ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
                if ((k_tail_written = mctx_cur->cpy_k_tail(ctx0, k_cur, inp->get_tail_idxs(is_swa), il))) {
                    ggml_build_forward_expand(gf, k_tail_written);
                }
            }
        }
    }

    if (store_kv && v_cur) {
        const auto & v_idxs = is_swa ? inp->get_v_idxs_swa() : inp->get_v_idxs();

        if (compact_tail) {
            if (v_cur) {
                if (ggml_tensor * written = mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il)) {
                    ggml_build_forward_expand(gf, written);
                }
            }
        } else {
            v_tail_written = mctx_cur->cpy_v_with_tail(
                    ctx0, v_cur, v_idxs, inp->get_tail_idxs(is_swa), il);
            if (v_tail_written) {
                ggml_build_forward_expand(gf, v_tail_written);
            } else {
                ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
                if ((v_tail_written = mctx_cur->cpy_v_tail(ctx0, v_cur, inp->get_tail_idxs(is_swa), il))) {
                    ggml_build_forward_expand(gf, v_tail_written);
                }
            }
        }
    }

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = use_kv_cur ? k_cur : use_kvarn ?
        kvarn_ctx->get_k_for_attention(ctx0, il, kvarn_plan.native_attention) :
        mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = use_kv_cur ? v_cur : use_kvarn ?
        kvarn_ctx->get_v_for_attention(ctx0, il, kvarn_plan.native_attention) :
        mctx_cur->get_v(ctx0, il);
    ggml_tensor * kvarn_rot = use_kvarn ? llm_kvarn_rot_for_dim(
            inp->self_kvarn_rot_64, inp->self_kvarn_rot_128,
            inp->self_kvarn_rot_256, inp->self_kvarn_rot_512, q->ne[0]) : nullptr;

    if (use_kvarn_q_rot) {
        GGML_ASSERT(q->type == GGML_TYPE_F32);
        GGML_ASSERT(kvarn_rot != nullptr);
        q = ggml_kvarn_wht_aux(ctx0, q, kvarn_rot->ne[0]);
    }

    ggml_tensor * k_tail = k_tail_written ? k_tail_written : mctx_cur->get_k_tail(ctx0, il);
    ggml_tensor * v_tail = v_tail_written ? v_tail_written : mctx_cur->get_v_tail(ctx0, il);
    const bool gather_k_tail = k_tail != nullptr;
    const bool gather_v_tail = v_tail != nullptr;
    ggml_tensor * tail_read_idxs = inp->get_tail_read_idxs(is_swa);
    llama_kv_tail_route tail_route = mctx_cur->get_tail_route(il);
    // Keep the iSWA route decision identical to the non-iSWA path above.
    if (use_kvarn && tail_route == LLAMA_KV_TAIL_ROUTE_NATIVE &&
            !kvarn_plan.native_attention && q->ne[0] != 64 &&
            (arch != LLM_ARCH_DFLASH || cparams.causal_attn)) {
        tail_route = LLAMA_KV_TAIL_ROUTE_GENERIC;
    }
    if (tail_route != LLAMA_KV_TAIL_ROUTE_NONE) {
        GGML_ASSERT(mctx_cur->get_tail_explicit_bias(il) == (kq_b != nullptr) &&
                "KV tail route bias contract does not match the model graph");
    }
    ggml_tensor * k_tail_current = nullptr;
    ggml_tensor * v_tail_current = nullptr;
    if (compact_tail && k_cur && v_cur) {
        k_tail_current = prepare_compact_tail_current(ctx0, k_tail, k_cur);
        v_tail_current = prepare_compact_tail_current(ctx0, v_tail, v_cur);
        if (tail_route != LLAMA_KV_TAIL_ROUTE_NATIVE) {
            k_tail = ggml_concat(ctx0, k_tail, k_tail_current, 2);
            v_tail = ggml_concat(ctx0, v_tail, v_tail_current, 2);
            k_tail_current = nullptr;
            v_tail_current = nullptr;
        }
    }
    const bool use_indexed_tail = gather_k_tail && gather_v_tail && tail_read_idxs &&
            tail_route == LLAMA_KV_TAIL_ROUTE_NATIVE;
    if (!use_indexed_tail && !use_kvarn && mctx_cur->get_tail_tokens(il) > 0) {
        if (!k_tail) {
            k_tail = mctx_cur->get_k_tail_fallback(ctx0, il, inp->get_tail_body_read_idxs(is_swa));
        }
        if (!v_tail) {
            v_tail = mctx_cur->get_v_tail_fallback(ctx0, il, inp->get_tail_body_read_idxs(is_swa));
        }
    }
    if (k_tail && !use_indexed_tail) {
        GGML_ASSERT(v_tail && tail_read_idxs);
        const auto gather_tail = [&](ggml_tensor * value) {
            const int64_t n_embd_head = value->ne[0];
            const int64_t n_heads = value->ne[1];
            value = ggml_reshape_2d(ctx0, value, n_embd_head*n_heads, value->ne[2]);
            ggml_tensor * read_idxs = ggml_reshape_1d(ctx0, tail_read_idxs,
                    tail_read_idxs->ne[0]*ubatch.n_tokens);
            value = ggml_get_rows_as(ctx0, value, read_idxs, value->type);
            return ggml_reshape_4d(ctx0, value, n_embd_head, n_heads,
                    tail_read_idxs->ne[0], ubatch.n_tokens);
        };
        if (gather_k_tail) {
            k_tail = gather_tail(k_tail);
        }
        if (gather_v_tail) {
            v_tail = gather_tail(v_tail);
        }
    }
    if (use_kvarn && k_tail) {
        GGML_ASSERT(v_tail && kvarn_rot);
        k_tail = ggml_kvarn_wht_aux(ctx0, k_tail, kvarn_rot->ne[0]);
        if (use_kvarn_rotated_domain) {
            v_tail = ggml_kvarn_wht_aux(ctx0, v_tail, kvarn_rot->ne[0]);
        }
        if (k_tail_current) {
            k_tail_current = ggml_kvarn_wht_aux(ctx0, k_tail_current, kvarn_rot->ne[0]);
            if (use_kvarn_rotated_domain) {
                v_tail_current = ggml_kvarn_wht_aux(ctx0, v_tail_current, kvarn_rot->ne[0]);
            }
        }
    }
    ggml_tensor * kq_mask_tail = tail_route == LLAMA_KV_TAIL_ROUTE_NONE ? nullptr : inp->get_kq_mask_tail(is_swa);
    ggml_tensor * kq_b_tail = build_attn_bias_tail(
            kq_b, inp->get_tail_bias_read_idxs(is_swa), kq_mask_tail);
    ggml_tensor * body_mask = kq_mask;
    ggml_tensor * body_bias = kq_b;
    if (!mctx_cur->has_kv_body(il)) {
        GGML_ASSERT(compact_tail && !use_kvarn && is_swa);
        const auto * route = mctx_cur->get_tail_layer_route(il);
        GGML_ASSERT(route);
        build_empty_kv_body(ctx0, k_cur ? k_cur : k_tail, v_cur ? v_cur : v_tail,
                route->body_type_k, route->body_type_v, kq_mask, kq_b,
                k, v, body_mask, body_bias);
    }
    ggml_tensor * final_tail_op = nullptr;
    ggml_tensor * cur = build_attn_mha(q, k, v, body_bias, body_mask, sinks, v_mla, 0, kq_scale, il,
k_tail, v_tail, kq_mask_tail, kq_b_tail,
            use_indexed_tail ? tail_read_idxs : nullptr,
            (use_indexed_tail || use_kvarn) ? inp->get_tail_query_order(is_swa) : nullptr,
            (use_indexed_tail || use_kvarn) ? inp->get_tail_run_desc(is_swa) : nullptr,
            tail_route,
            kvarn_domain,
            k_tail_current, v_tail_current,
            mctx_cur->get_tail_slots(), !mctx_cur->has_kv_body(il), &final_tail_op);
    if (use_kvarn) {
        llm_flash_attn_ext_set_kvarn_domain(cur, kvarn_domain);
    }
    if (use_kvarn && arch == LLM_ARCH_DFLASH && !cparams.causal_attn &&
            kvarn_plan.native_attention) {
        validate_native_kvarn_noncausal_operation(mctx_cur, il, final_tail_op);
    }
    if (tail_route == LLAMA_KV_TAIL_ROUTE_NATIVE) {
        validate_native_tail_operation(mctx_cur, il, final_tail_op, true);
    }
    if (compact_tail && k_cur && v_cur) {
        if (ggml_tensor * written = mctx_cur->cpy_k_tail(
                    ctx0, k_cur, inp->get_tail_idxs(is_swa), il, cur)) {
            ggml_build_forward_expand(gf, written);
        }
        if (ggml_tensor * written = mctx_cur->cpy_v_tail(
                    ctx0, v_cur, inp->get_tail_idxs(is_swa), il, cur)) {
            ggml_build_forward_expand(gf, written);
        }
    }
    cb(cur, "kqv_out", il);

    if (use_kvarn_rotated_domain) {
        GGML_ASSERT(cur->type == GGML_TYPE_F32);
        GGML_ASSERT(kvarn_rot != nullptr);
        cur = ggml_kvarn_wht_aux(ctx0, cur, kvarn_rot->ne[0]);
    } else if (!use_kvarn && v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, v_rot);
    }

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    auto * k_rot = is_swa ? inp->self_k_rot_swa : inp->self_k_rot;

    if (k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, k_rot);
        if (k_cur) {
            k_cur = llama_mul_mat_hadamard(ctx0, k_cur, k_rot);
        }
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    const auto * mctx_iswa = inp->mctx;
    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();

    // optionally store to KV cache
    if (k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    // MLA-style attention: the cached K is used as V
    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (k_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, k_rot);
    }

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

llm_graph_input_attn_cross * llm_graph_context::build_attn_inp_cross() const {
    auto inp = std::make_unique<llm_graph_input_attn_cross>(cross);

    const int32_t n_enc = !cross->v_embd.empty() ? cross->n_enc : hparams.n_ctx_train;

    // flash attention requires an f16 mask
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    inp->cross_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_enc, n_tokens, 1, 1);
    ggml_set_input(inp->cross_kq_mask);

    inp->cross_kq_mask_cnv = inp->cross_kq_mask;

    return (llm_graph_input_attn_cross *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_cross * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const auto & kq_mask = inp->get_kq_mask_cross();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_k_dsa> build_attn_inp_k_dsa_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_dsa_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_k_dsa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs_mla = mctx_cur->get_mla()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask_mla = build_attn_inp_kq_mask(ctx0, mctx_cur->get_mla(), ubatch, cparams);
        inp->self_kq_mask_mla_cnv = inp->self_kq_mask_mla;
    }

    {
        inp->self_k_idxs_lid = mctx_cur->get_lid()->build_input_k_idxs(ctx0, ubatch);

        // ensure that mask type matches fused lightning indexer use (requires f16 mask)
        auto cparams_copy = cparams;
        cparams_copy.flash_attn = cparams.fused_lid;

        inp->self_kq_mask_lid = build_attn_inp_kq_mask(ctx0, mctx_cur->get_lid(), ubatch, cparams_copy);
        inp->self_kq_mask_lid_cnv = inp->self_kq_mask_lid;

        inp->self_k_rot_lid = mctx_cur->get_lid()->build_input_k_rot(ctx0);
    }

    return inp;
}

llm_graph_input_attn_k_dsa * llm_graph_context::build_attn_inp_k_dsa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsa_context *>(mctx);

    auto inp = build_attn_inp_k_dsa_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_k_dsa *) res->add_input(std::move(inp));
}

llm_graph_input_attn_k_dsa_iswa * llm_graph_context::build_attn_inp_k_dsa_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsa_iswa_context *>(mctx);

    auto inp_dsa = build_attn_inp_k_dsa_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_dsa());

    // build_attn_inp_k_impl rejects SWA caches, so construct the input directly
    auto inp_swa = std::make_unique<llm_graph_input_attn_k>(hparams, cparams, mctx_cur->get_swa());

    inp_swa->self_k_idxs = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);

    inp_swa->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
    inp_swa->self_kq_mask_cnv = inp_swa->self_kq_mask;

    auto inp = std::make_unique<llm_graph_input_attn_k_dsa_iswa>(std::move(inp_dsa), std::move(inp_swa), mctx_cur);

    return (llm_graph_input_attn_k_dsa_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_attn_kv_msa * llm_graph_context::build_attn_inp_kv_msa(bool msa_enabled) const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_msa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_msa>(hparams, cparams, mctx_cur);

    const auto * mctx_base = mctx_cur->get_base();
    const auto * mctx_idx  = mctx_cur->get_idx();

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_base->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_base->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_base, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    inp->self_k_rot = mctx_base->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_base->build_input_v_rot(ctx0);

    if (msa_enabled) {
        inp->self_k_idxs_idx = mctx_idx->build_input_k_idxs(ctx0, ubatch);
    }

    return (llm_graph_input_attn_kv_msa *) res->add_input(std::move(inp));
}

// TODO: maybe separate the inner implementation into a separate function
//       like with the non-sliding window equivalent
//       once sliding-window hybrid caches are a thing.
llm_graph_input_attn_kv_iswa * llm_graph_context::build_attn_inp_kv_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur->get_base(), ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
        build_attn_inp_tail(ctx0, ubatch, mctx_cur->get_base(), inp->self_kq_mask,
                inp->self_tail_idxs, inp->self_tail_read_idxs, inp->self_tail_body_read_idxs,
                inp->self_tail_bias_read_idxs, inp->self_kq_mask_tail,
                inp->self_tail_query_order, inp->self_tail_run_desc, true);
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs_swa = mctx_cur->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
        build_attn_inp_tail(ctx0, ubatch, mctx_cur->get_swa(), inp->self_kq_mask_swa,
                inp->self_tail_idxs_swa, inp->self_tail_read_idxs_swa,
                inp->self_tail_body_read_idxs_swa, inp->self_tail_bias_read_idxs_swa,
                inp->self_kq_mask_tail_swa, inp->self_tail_query_order_swa, inp->self_tail_run_desc_swa, true);
    }

    inp->self_k_rot = mctx_cur->get_base()->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_cur->get_base()->build_input_v_rot(ctx0);
    if (const auto * kvarn_base = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx_cur->get_base())) {
        inp->self_kvarn_rot_64  = kvarn_base->build_input_kvarn_rot(ctx0, 64);
        inp->self_kvarn_rot_128 = kvarn_base->build_input_kvarn_rot(ctx0, 128);
        inp->self_kvarn_rot_256 = kvarn_base->build_input_kvarn_rot(ctx0, 256);
        inp->self_kvarn_rot_512 = kvarn_base->build_input_kvarn_rot(ctx0, 512);
        if (kvarn_base->uses_materialization_indices()) {
            inp->self_kvarn_mat_idxs = kvarn_base->build_input_kvarn_mat_idxs(ctx0);
            kvarn_base->set_mat_idxs(inp->self_kvarn_mat_idxs);
        }
    }

    inp->self_k_rot_swa = mctx_cur->get_swa()->build_input_k_rot(ctx0);
    inp->self_v_rot_swa = mctx_cur->get_swa()->build_input_v_rot(ctx0);
    if (const auto * kvarn_swa = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx_cur->get_swa())) {
        if (inp->self_kvarn_rot_64 == nullptr) {
            inp->self_kvarn_rot_64  = kvarn_swa->build_input_kvarn_rot(ctx0, 64);
            inp->self_kvarn_rot_128 = kvarn_swa->build_input_kvarn_rot(ctx0, 128);
            inp->self_kvarn_rot_256 = kvarn_swa->build_input_kvarn_rot(ctx0, 256);
            inp->self_kvarn_rot_512 = kvarn_swa->build_input_kvarn_rot(ctx0, 512);
        }
        inp->self_kvarn_mat_idxs_swa = kvarn_swa->build_input_kvarn_mat_idxs(ctx0);
        // Native K/V views are built before set_input() populates the tensor.
        kvarn_swa->set_mat_idxs(inp->self_kvarn_mat_idxs_swa);
    }

    return (llm_graph_input_attn_kv_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_attn_k_iswa * llm_graph_context::build_attn_inp_k_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_k_iswa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur->get_base(), ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    }

    inp->self_k_rot = mctx_cur->get_base()->build_input_k_rot(ctx0);

    inp->self_k_rot_swa = mctx_cur->get_swa()->build_input_k_rot(ctx0);

    return (llm_graph_input_attn_k_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_dsv4 * llm_graph_context::build_inp_dsv4() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsv4_context *>(mctx);
    const auto * raw_ctx  = mctx_cur->get_raw();

    auto inp_raw = std::make_unique<llm_graph_input_dsv4_raw>(cparams, raw_ctx);

    const int64_t n_stream = mctx_cur->get_csa_plan(ubatch).n_stream;

    GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "DSV4 expects SWA raw cache");

    inp_raw->self_k_idxs = raw_ctx->build_input_k_idxs(ctx0, ubatch);
    inp_raw->self_tail_idxs = raw_ctx->build_input_tail_idxs(ctx0, ubatch);
    inp_raw->self_kq_mask = dsv4_build_raw_kq_mask(ctx0, raw_ctx, ubatch, cparams, n_stream);
    inp_raw->self_kq_mask_cnv = inp_raw->self_kq_mask;
    if (raw_ctx->get_tail_tokens() > 0) {
        const auto [tail_q_max, tail_n_active] = tail_query_plan_shape(ubatch);
        GGML_UNUSED(tail_n_active);
        const uint32_t attention_stride = raw_ctx->get_tail_attention_stride(
                uint32_t(tail_extent_queries(raw_ctx, ubatch, tail_q_max)));
        inp_raw->self_tail_read_idxs = ggml_new_tensor_2d(
                ctx0, GGML_TYPE_I32, attention_stride, ubatch.n_tokens);
        ggml_set_input(inp_raw->self_tail_read_idxs);
        inp_raw->self_tail_body_read_idxs = ggml_new_tensor_2d(
                ctx0, GGML_TYPE_I32, attention_stride, ubatch.n_tokens);
        ggml_set_input(inp_raw->self_tail_body_read_idxs);
        inp_raw->self_tail_bias_read_idxs = ggml_new_tensor_2d(
                ctx0, GGML_TYPE_I32, attention_stride, ubatch.n_tokens);
        ggml_set_input(inp_raw->self_tail_bias_read_idxs);
        inp_raw->self_kq_mask_tail = ggml_new_tensor_4d(ctx0, inp_raw->self_kq_mask->type,
                attention_stride, inp_raw->self_kq_mask->ne[1], 1, inp_raw->self_kq_mask->ne[3]);
        ggml_set_input(inp_raw->self_kq_mask_tail);
    }

    inp_raw->self_k_rot = raw_ctx->build_input_k_rot(ctx0);
    auto inp = std::make_unique<llm_graph_input_dsv4>(cparams, std::move(inp_raw), mctx_cur);

    dsv4_build_comp_inputs(ctx0, inp->inp_csa, mctx_cur->get_csa_plan(ubatch), "csa", cparams, n_stream);
    dsv4_build_comp_inputs(ctx0, inp->inp_hca, mctx_cur->get_hca_plan(ubatch), "hca", cparams, n_stream);
    dsv4_build_comp_inputs(ctx0, inp->inp_lid, mctx_cur->get_lid_plan(ubatch), "lid", cparams, n_stream);
    inp->inp_csa.k_rot = mctx_cur->get_csa()->build_input_k_rot(ctx0);
    inp->inp_hca.k_rot = mctx_cur->get_hca()->build_input_k_rot(ctx0);
    inp->inp_lid.k_rot = mctx_cur->get_lid()->build_input_k_rot(ctx0);

    return (llm_graph_input_dsv4 *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        ggml_tensor * s,
        ggml_tensor * state_copy,
        ggml_tensor * state_copy_main,
            int32_t   state_size,
            int32_t   n_seqs,
           uint32_t   n_rs,
           uint32_t   rs_head,
           uint32_t   rs_size,
            int32_t   rs_zero,
        const llm_graph_get_rows_fn & get_state_rows) const {

    GGML_UNUSED(rs_size);
    ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, s->ne[1]);

    // Clear a single state which will then be copied to the other cleared states.
    // Note that this is a no-op when the view is zero-sized.
    ggml_tensor * state_zero = ggml_view_1d(ctx0, states, state_size*(rs_zero >= 0), rs_zero*states->nb[1]*(rs_zero >= 0));
    ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));

    // copy states
    // NOTE: assuming the copy destinations are ALL contained between rs_head and rs_head + n_rs
    // one gather of the states i0..n_rs (ubatch states then extra states), sized by n_rs so the reserve covers every split
    // {state_size, rs_size} -> {state_size, n_rs - i0}
    const int64_t i0 = n_rs - state_copy->ne[0];

    ggml_tensor * states_all = ggml_get_rows(ctx0, states, state_copy);

    ggml_tensor * output_states = get_state_rows ?
        get_state_rows(ctx0, states, state_copy_main) :
        ggml_view_2d(ctx0, states_all, state_size, n_seqs, states_all->nb[1], 0);
    ggml_build_forward_expand(gf, output_states);

    // copy extra states which won't be changed further (between n_seqs and n_rs)
    ggml_tensor * states_extra = ggml_view_2d(ctx0, states_all, state_size, n_rs - n_seqs, states_all->nb[1], (n_seqs - i0)*states_all->nb[1]);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0,
            states_extra,
            ggml_view_2d(ctx0, s, state_size, (n_rs - n_seqs), s->nb[1], (rs_head + n_seqs)*s->nb[1])));

    return output_states;
}

static std::unique_ptr<llm_graph_input_rs> build_rs_inp_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_memory_recurrent_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_rs>(mctx_cur);

    const int64_t n_rs   = mctx_cur->get_n_rs();
    const int64_t n_seqs = ubatch.n_seqs;

    inp->s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
    ggml_set_input(inp->s_copy);
    ggml_set_name(inp->s_copy, "rs_s_copy");

    const int64_t n_history = std::max<int64_t>(0,
            (int64_t) mctx_cur->get_n_rs_seq() + 1 - ubatch.n_seq_tokens);
    if (n_history > 0) {
        inp->s_history = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_history*n_seqs);
        ggml_set_input(inp->s_history);
    }

    inp->s_copy_main  = ggml_view_1d(ctx0, inp->s_copy, n_seqs, 0);
    inp->s_copy_tail = ggml_view_1d(ctx0, inp->s_copy, n_rs - 1, inp->s_copy->nb[0]);
    inp->s_copy_extra = ggml_view_1d(ctx0, inp->s_copy, n_rs - n_seqs, n_seqs*inp->s_copy->nb[0]);

    inp->head = mctx_cur->get_head();
    inp->rs_z = mctx_cur->get_rs_z();

    return inp;
}

llm_graph_input_rs * llm_graph_context::build_rs_inp() const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    auto inp = build_rs_inp_impl(ctx0, ubatch, mctx_cur);

    return (llm_graph_input_rs *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        llm_graph_input_rs * inp,
        ggml_tensor * s,
            int32_t   state_size,
            int32_t   n_seqs,
        const llm_graph_get_rows_fn & get_state_rows) const {
    const auto * kv_state = inp->mctx;

    // a custom getter reads the states of the ubatch straight from the cache, so the gather skips the first
    // state: it still holds the n_rs - n_seqs extra states and copies no state of a single sequence ubatch
    ggml_tensor * state_copy = get_state_rows ? inp->s_copy_tail : inp->s_copy;

    return build_rs(s, state_copy, inp->s_copy_main, state_size, n_seqs,
                    kv_state->get_n_rs(), kv_state->get_head(), kv_state->get_size(), kv_state->get_rs_z(),
                    get_state_rows);
}

ggml_tensor * llm_graph_context::build_rs_cache_view(
        llm_graph_input_rs * inp,
        ggml_tensor * s,
            int32_t   state_size,
            int32_t   n_seqs) const {
    const auto * kv_state = inp->mctx;

    const uint32_t n_rs     = kv_state->get_n_rs();
    const uint32_t rs_head  = kv_state->get_head();
    const  int32_t rs_zero  = kv_state->get_rs_z();

    ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, s->ne[1]);

    // same cache hygiene as build_rs, minus the main gather (the consumer reads
    // per-seq rows via inp->s_copy_main directly, inside the GDN op).
    //
    // KNOWN LIMITATION (tracked follow-up): build_rs gathers the main rows
    // BEFORE this extra relocation, so an overlapping main row is read before
    // being overwritten. rows mode defers the main read into the consumer, and
    // s_copy() maps a main row to an arbitrary cache slot (idx*size + src0),
    // which can fall inside the extra destination [rs_head+n_seqs, rs_head+n_rs)
    // during a cache reorder -- so this relocation could clobber a main row the
    // consumer will later read. Not reachable on the current single-sequence
    // decode path, but it is a real multi-sequence hazard; the correct fix is
    // to order the relocation AFTER the GDN read (build_rs's read-before-write
    // ordering), which is a graph-dependency refactor left as follow-up.
    ggml_tensor * state_zero = ggml_view_1d(ctx0, states, state_size*(rs_zero >= 0), rs_zero*states->nb[1]*(rs_zero >= 0));
    ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));

    ggml_tensor * states_extra = ggml_get_rows(ctx0, states, inp->s_copy_extra);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0,
            states_extra,
            ggml_view_2d(ctx0, s, state_size, (n_rs - n_seqs), s->nb[1], (rs_head + n_seqs)*s->nb[1])));

    return states;
}

ggml_tensor * llm_graph_context::build_rs_write_rows(
        llm_graph_input_rs * inp,
            int64_t   K,
            int64_t   n_seq_tokens,
            int64_t   n_seqs) const {
    const int64_t n_write = std::min(n_seq_tokens, K);

    if (inp->s_write_rows == nullptr) {
        inp->s_write_rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_write * n_seqs);
        ggml_set_input(inp->s_write_rows);
        inp->s_write_K = K;
    }

    GGML_ASSERT(inp->s_write_K == K);
    GGML_ASSERT(inp->s_write_rows->ne[0] == n_write * n_seqs);

    return inp->s_write_rows;
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_load(
    llm_graph_input_rs * inp,
    const llama_ubatch & ubatch,
                   int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;

    const int64_t n_seqs  = ubatch.n_seqs;

    ggml_tensor * token_shift_all = mctx_cur->get_r_l(il);

    ggml_tensor * token_shift = build_rs(
            inp, token_shift_all,
            hparams.n_embd_r(), n_seqs);

    token_shift = ggml_reshape_3d(ctx0, token_shift, hparams.n_embd, token_shift_count, n_seqs);

    return token_shift;
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_store(
         ggml_tensor * token_shift,
  const llama_ubatch & ubatch,
                 int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;
    const auto n_embd = hparams.n_embd;

    const int64_t n_seqs = ubatch.n_seqs;

    const auto kv_head = mctx_cur->get_head();

    return ggml_cpy(
        ctx0,
        ggml_view_1d(ctx0, token_shift, n_embd * n_seqs * token_shift_count, 0),
        ggml_view_1d(ctx0, mctx_cur->get_r_l(il), hparams.n_embd_r()*n_seqs, hparams.n_embd_r()*kv_head*ggml_element_size(mctx_cur->get_r_l(il)))
    );
}

llm_graph_input_mem_hybrid * llm_graph_context::build_inp_mem_hybrid() const {
    if (cparams.kv_paged) {
        return (llm_graph_input_mem_hybrid *) build_inp_mem_hybrid_paged();
    }

    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    // When ALL full-attention layers are offloaded to the remote RKVA backend,
    // the local attention KV input (k_idxs/v_idxs/mask/tail) is built but never
    // consumed, so the scheduler leaves those tensors unallocated and their
    // set_input would deref a NULL buffer. Only skip building it when ALL full-attn
    // layers are offloaded.
    int n_full_attn = 0;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (hparams.has_kv(il) && !hparams.is_recr(il)) {
            ++n_full_attn;
        }
    }
    const bool all_remote = (cparams.remote_attn_enabled && !cparams.local_attn_backend &&
                             cparams.remote_attn_layers >= n_full_attn);
    std::unique_ptr<llm_graph_input_attn_kv> inp_attn;
    if (!all_remote) {
        inp_attn = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn_kv_context());
    }

    auto inp = std::make_unique<llm_graph_input_mem_hybrid>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_paged * llm_graph_context::build_inp_mem_hybrid_paged() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_paged_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl(ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_kv_paged_impl(ctx0, cparams, hparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_paged>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_paged *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_k * llm_graph_context::build_inp_mem_hybrid_k() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_k>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_k *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_iswa * llm_graph_context::build_inp_mem_hybrid_iswa() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_iswa_context *>(mctx);

    auto inp_rs = build_rs_inp_impl(ctx0, ubatch, mctx_cur->get_recr());

    // build iswa attention input
    const auto * attn_ctx = mctx_cur->get_attn();

    auto inp_attn = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, attn_ctx);

    {
        inp_attn->self_k_idxs = attn_ctx->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs = attn_ctx->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_kq_mask = build_attn_inp_kq_mask(ctx0, attn_ctx->get_base(), ubatch, cparams);
        inp_attn->self_kq_mask_cnv = inp_attn->self_kq_mask;
        build_attn_inp_tail(ctx0, ubatch, attn_ctx->get_base(), inp_attn->self_kq_mask,
                inp_attn->self_tail_idxs, inp_attn->self_tail_read_idxs,
                inp_attn->self_tail_body_read_idxs, inp_attn->self_tail_bias_read_idxs,
                inp_attn->self_kq_mask_tail, inp_attn->self_tail_query_order, inp_attn->self_tail_run_desc, true);
    }

    {
        inp_attn->self_k_idxs_swa = attn_ctx->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs_swa = attn_ctx->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, attn_ctx->get_swa(), ubatch, cparams);
        inp_attn->self_kq_mask_swa_cnv = inp_attn->self_kq_mask_swa;
        build_attn_inp_tail(ctx0, ubatch, attn_ctx->get_swa(), inp_attn->self_kq_mask_swa,
                inp_attn->self_tail_idxs_swa, inp_attn->self_tail_read_idxs_swa,
                inp_attn->self_tail_body_read_idxs_swa, inp_attn->self_tail_bias_read_idxs_swa,
                inp_attn->self_kq_mask_tail_swa, inp_attn->self_tail_query_order_swa,
                inp_attn->self_tail_run_desc_swa, true);
    }

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_iswa>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_iswa *) res->add_input(std::move(inp));
}

void llm_graph_context::build_dense_out(
    ggml_tensor * dense_2,
    ggml_tensor * dense_2_b,
    ggml_tensor * dense_3) const {
    if (!cparams.embeddings || !(dense_2 || dense_2_b || dense_3)) {
        return;
    }
    ggml_tensor * cur = res->t_embd_pooled != nullptr ? res->t_embd_pooled : res->t_embd;
    GGML_ASSERT(cur != nullptr && "missing t_embd_pooled/t_embd");

    if (dense_2) {
        cur = ggml_mul_mat(ctx0, dense_2, cur);
    }
    if (dense_2_b) {
        cur = ggml_add(ctx0, cur, dense_2_b);
    }
    if (dense_3) {
        cur = ggml_mul_mat(ctx0, dense_3, cur);
    }
    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;
    ggml_build_forward_expand(gf, cur);
}


void llm_graph_context::build_pooling(
        ggml_tensor * cls,
        ggml_tensor * cls_b,
        ggml_tensor * cls_out,
        ggml_tensor * cls_out_b,
        ggml_tensor * cls_norm) const {
    if (!cparams.embeddings) {
        // a graph without logits still computes t_embd, so that it is never empty
        if (res->t_logits == nullptr && res->t_embd != nullptr) {
            ggml_build_forward_expand(gf, res->t_embd);
        }
        return;
    }

    ggml_tensor * inp = res->t_embd;

    //// find result_norm tensor for input
    //for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
    //    inp = ggml_graph_node(gf, i);
    //    if (strcmp(inp->name, "result_norm") == 0 || strcmp(inp->name, "result_embd") == 0) {
    //        break;
    //    }

    //    inp = nullptr;
    //}

    GGML_ASSERT(inp != nullptr && "missing result_norm/result_embd tensor");

    ggml_tensor * cur;

    switch (pooling_type) {
        case LLAMA_POOLING_TYPE_NONE:
            {
                cur = inp;
            } break;
        case LLAMA_POOLING_TYPE_MEAN:
            {
                ggml_tensor * inp_mean = build_inp_mean();
                cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
            } break;
        case LLAMA_POOLING_TYPE_CLS:
        case LLAMA_POOLING_TYPE_LAST:
            {
                ggml_tensor * inp_cls = build_inp_cls();
                cur = ggml_get_rows(ctx0, inp, inp_cls);
            } break;
        case LLAMA_POOLING_TYPE_RANK:
            {
                if (hparams.pooling_type_cls == LLAMA_POOLING_TYPE_MEAN) {
                    // modern bert with classifier_pooling = "mean" builds mean first then applies prediction head and classifier
                    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modular_modernbert.py#L1404-1411
                    ggml_tensor * inp_mean = build_inp_mean();
                    cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
                } else {
                    ggml_tensor * inp_cls = build_inp_cls();
                    cur = ggml_get_rows(ctx0, inp, inp_cls);
                }

                // classification head
                // https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/roberta/modeling_roberta.py#L1566
                if (cls) {
                    cur = ggml_mul_mat(ctx0, cls, cur);
                    if (cls_b) {
                        cur = ggml_add(ctx0, cur, cls_b);
                    }
                    cur = ggml_unary(ctx0, cur, hparams.act_cls);
                    if (cls_norm) {
                        // head norm
                        cur = build_norm(cur, cls_norm, NULL, LLM_NORM, -1);
                    }
                }

                // some models don't have `cls_out`, for example: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
                // https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/blob/cb5347e43979c3084a890e3f99491952603ae1b7/modeling_bert.py#L884-L896
                // Single layer classification head (direct projection)
                // https://github.com/huggingface/transformers/blob/f4fc42216cd56ab6b68270bf80d811614d8d59e4/src/transformers/models/bert/modeling_bert.py#L1476
                if (cls_out) {
                    cur = ggml_mul_mat(ctx0, cls_out, cur);
                    if (cls_out_b) {
                        cur = ggml_add(ctx0, cur, cls_out_b);
                    }
                }

                // softmax for qwen3 reranker
                if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_QWEN3VL) {
                    cur = ggml_soft_max(ctx0, cur);
                }
            } break;
        default:
            {
                GGML_ABORT("unknown pooling type");
            }
    }

    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;

    ggml_build_forward_expand(gf, cur);
}

void llm_graph_context::build_logits() const {
    // a pooled embeddings context outputs one vector per sequence, so its graph skips the LM head
    if (cparams.embeddings && pooling_type != LLAMA_POOLING_TYPE_NONE) {
        res->t_logits = nullptr;
    }

    // the models declare t_logits without expanding it, so the LM head is computed only when it is expanded here
    if (res->t_logits != nullptr) {
        ggml_build_forward_expand(gf, res->t_logits);
    }
}

void llm_graph_context::build_sampling() const {
    if (samplers.empty() || !res->t_logits) {
        return;
    }

    std::array<ggml_tensor *, 2> outs;
    outs[0] = res->t_logits;

    auto inp_sampling = std::make_unique<llm_graph_input_sampling>(samplers);
    res->add_input(std::move(inp_sampling));

    std::map<llama_seq_id, std::vector<uint32_t>> sampling_rows;
    uint32_t n_rows = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.output[i]) {
            sampling_rows[ubatch.seq_id[i][0]].push_back(n_rows++);
        }
    }

    res->t_sampled.resize(n_rows, nullptr);
    res->t_sampled_probs.resize(n_rows, nullptr);
    res->t_sampled_logits.resize(n_rows, nullptr);
    res->t_candidates.resize(n_rows, nullptr);

    // res->t_logits will contain logits for all tokens that want the logits calculated (logits=1 or output=1)
    GGML_ASSERT(res->t_logits != nullptr && "missing t_logits tensor");

    // add a dummy row to keep the single-output graph static regardless of active samplers
    // multi-output graphs can still vary with the number of output rows
    // DSpark reserve graphs can expose fewer draft rows than the output budget.
    const int64_t pad_rows =
        res->t_dspark_greedy.empty() ? 1 : std::max<int64_t>(1, int64_t(n_rows) - res->t_logits->ne[1]);
    ggml_tensor * logits_t = ggml_pad(ctx0, res->t_logits, 0, pad_rows, 0, 0);

    for (const auto & entry : samplers) {
        if (entry.second->iface->backend_reset) {
            entry.second->iface->backend_reset(entry.second);
        }
    }

    static const std::vector<uint32_t> no_rows;

    // every sampler builds n_outputs_max_per_seq chains, like the reserve, so the graph keeps its topology
    // whatever rows the ubatch outputs: a chain without a row works on the first row and is not selected
    for (const auto & [seq_id, sampler] : samplers) {
        const auto it = sampling_rows.find(seq_id);
        const auto & rows = it != sampling_rows.end() ? it->second : no_rows;

        for (uint32_t i = 0; i < cparams.n_outputs_max_per_seq; ++i) {
            const bool     active = i < rows.size();
            const uint32_t row    = active ? rows[i] : 0;
            const int      i_out  = active ? 1       : 0;

            if (active && res->t_dspark_greedy.size() == n_rows && llama_sampler_chain_n(sampler) == 1 &&
                    std::strcmp(llama_sampler_name(llama_sampler_chain_get(sampler, 0)), "greedy") == 0) {
                res->t_sampled[row] = res->t_dspark_greedy[row];
                ggml_build_forward_expand(gf, res->t_sampled[row]);
                continue;
            }

            ggml_tensor * logits_seq = ggml_view_1d(ctx0, logits_t, logits_t->ne[0], row * logits_t->nb[1]);
            ggml_format_name(logits_seq, "logits_seq_%d_%u", seq_id, i);

            struct llama_sampler_data data = {
                /*.logits       =*/ logits_seq,
                /*.probs        =*/ nullptr,
                /*.sampled      =*/ nullptr,
                /*.candidates   =*/ nullptr,
            };

            assert(sampler->iface->backend_apply);
            sampler->iface->backend_apply(sampler, ctx0, gf, &data);

            if (data.sampled != nullptr) {
                if (active) {
                    res->t_sampled[row] = data.sampled;
                }
                outs[1] = data.sampled;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            if (data.probs != nullptr) {
                if (active) {
                    res->t_sampled_probs[row] = data.probs;
                }
                outs[1] = data.probs;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            if (data.logits != nullptr) {
                if (active) {
                    res->t_sampled_logits[row] = data.logits;
                }
                outs[1] = data.logits;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            if (data.candidates != nullptr) {
                if (active) {
                    res->t_candidates[row] = data.candidates;
                }
                outs[1] = data.candidates;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }
        }
    }

    // TODO: Call backend_accept after all samplers have been applied.
    /*
    for (const auto & [seq_id, sampler] : samplers) {
        const auto it = sampling_rows.find(seq_id);
        if (it == sampling_rows.end()) {
            continue;
        }

        for (uint32_t row : it->second) {
            ggml_tensor * selected_token = res->t_sampled[row];
            if (selected_token != nullptr && sampler->iface->backend_accept) {
                sampler->iface->backend_accept(sampler, ctx0, gf, selected_token);
            }
        }
    }
    */
}

int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional) {
    // TODO move to hparams if a T5 variant appears that uses a different value
    const int64_t max_distance = 128;

    if (bidirectional) {
        n_buckets >>= 1;
    }

    const int64_t max_exact = n_buckets >> 1;

    int32_t relative_position = x - y;
    int32_t relative_bucket = 0;

    if (bidirectional) {
        relative_bucket += (relative_position > 0) * n_buckets;
        relative_position = std::abs(relative_position);
    } else {
        relative_position = -std::min<int32_t>(relative_position, 0);
    }

    int32_t relative_position_if_large = floorf(max_exact + logf(1.0 * relative_position / max_exact) * (n_buckets - max_exact) / log(1.0 * max_distance / max_exact));
    relative_position_if_large = std::min<int32_t>(relative_position_if_large, n_buckets - 1);
    relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);

    return relative_bucket;
}
