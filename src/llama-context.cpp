#include "llama-context.h"
#include "llama-cuda-graph-policy.h"

#include "ggml.h"
#include "llama-arch.h"
#include "llama-batch.h"
#include "llama-ext.h"
#include "llama-graph.h"
#include "llama-rollback.h"
#include "llama-impl.h"
#include "llama-position-split.h"
#include "llama-wavefront.h"
#include "llama-io.h"
#include "llama-io-file.h"
#include "llama-native-save-file.h"
#include "llama-state-framing.h"
#include "llama-kv-cache-kvarn.h"
#include "llama-memory-hybrid.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-tail.h"
#include "llama-kv-mixed-mtp-budget.h"
#include "llama-kv-mixed-placement.h"
#include "llama-kv-mixed-state-stream.h"
#include "llama-kv-tail-request.h"
#include "llama-kvarn.h"
#include "llama-kvarn-split-fused.h"
#include "ggml-remote-attn.h"
#include "rkva-native-state.h"
#include "rkva-snapshot-digest.h"
#include "llama-remote-snapshot.h"
#include "ggml-local-split.h"

#include "llama-memory.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include "llama-ext.h"
#include "llama-sampler.h"
#include "llama-state-q4.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct llama_kv_tail_config {
    struct group {
        std::string id;
        std::string role;
        uint32_t lowest_layer;
        uint32_t effective_window;
        std::vector<int32_t> layers;
        uint32_t n_tokens = 0;
        bool assigned = false;
    };

    const llama_model * model = nullptr;
    std::vector<group> groups;
    std::string error;
    bool automatic = false;
};

llama_kv_tail_config * llama_kv_tail_config_init(const llama_model * model) {
    if (!model) {
        return nullptr;
    }
    auto * result = new llama_kv_tail_config;
    result->model = model;

    llama_kv_tail_config::group full { "", "full", UINT32_MAX, model->hparams.n_ctx_train, {} };
    llama_kv_tail_config::group swa  { "", "swa",  UINT32_MAX, model->hparams.n_swa, {} };
    for (uint32_t il = 0; il < model->hparams.n_layer_all; ++il) {
        if (!model->hparams.has_kv(il) || model->hparams.is_recr(il)) {
            continue;
        }
        auto & group = model->hparams.is_swa(il) ? swa : full;
        group.lowest_layer = std::min(group.lowest_layer, il);
        group.layers.push_back(int32_t(il));
    }
    for (auto * group : { &full, &swa }) {
        if (!group->layers.empty()) {
            group->id = group->role + "@l" + std::to_string(group->lowest_layer);
            result->groups.push_back(std::move(*group));
        }
    }
    std::sort(result->groups.begin(), result->groups.end(), [](const auto & a, const auto & b) {
        return a.lowest_layer < b.lowest_layer;
    });
    return result;
}

void llama_kv_tail_config_free(llama_kv_tail_config * config) {
    delete config;
}

int32_t llama_kv_tail_config_group_count(const llama_kv_tail_config * config) {
    return config ? int32_t(config->groups.size()) : -1;
}

bool llama_kv_tail_config_get_group_info(
        const llama_kv_tail_config * config, int32_t group_index, llama_kv_tail_group_info * out) {
    if (!config || !out || group_index < 0 || size_t(group_index) >= config->groups.size()) {
        return false;
    }
    const auto & group = config->groups[size_t(group_index)];
    *out = { group.id.c_str(), group.role.c_str(), group.lowest_layer,
            uint32_t(group.layers.size()), group.effective_window };
    return true;
}

int32_t llama_kv_tail_config_group_layer(
        const llama_kv_tail_config * config, int32_t group_index, int32_t layer_index) {
    if (!config || group_index < 0 || size_t(group_index) >= config->groups.size() ||
            layer_index < 0 || size_t(layer_index) >= config->groups[size_t(group_index)].layers.size()) {
        return -1;
    }
    return config->groups[size_t(group_index)].layers[size_t(layer_index)];
}

bool llama_kv_tail_config_set_auto(llama_kv_tail_config * config) {
    if (!config) {
        return false;
    }
    config->automatic = true;
    config->error.clear();
    for (auto & group : config->groups) {
        group.assigned = false;
        group.n_tokens = 0;
    }
    return true;
}

bool llama_kv_tail_config_set_group(
        llama_kv_tail_config * config, const char * group_id, uint32_t n_tokens) {
    if (!config || !group_id) {
        return false;
    }
    config->automatic = false;
    std::vector<llama_kv_tail_config::group *> matches;
    for (auto & group : config->groups) {
        if (group.id == group_id || group.role == group_id) {
            matches.push_back(&group);
        }
    }
    if (matches.size() != 1) {
        config->error = matches.empty() ? "unknown KV tail group: " + std::string(group_id) :
                "ambiguous KV tail group alias: " + std::string(group_id);
        return false;
    }
    if (matches[0]->assigned) {
        config->error = "duplicate KV tail group assignment: " + matches[0]->id;
        return false;
    }
    matches[0]->n_tokens = n_tokens;
    matches[0]->assigned = true;
    config->error.clear();
    return true;
}

const char * llama_kv_tail_config_last_error(const llama_kv_tail_config * config) {
    return config ? config->error.c_str() : "null KV tail config";
}

bool llama_kv_tail_get_coverage(
        const llama_context * ctx,
        llama_seq_id seq_id,
        uint32_t group_index,
        llama_kv_tail_coverage_info * out) {
    if (!ctx || !out) {
        return false;
    }
    const llama_memory_t memory = ctx->get_memory();
    return memory && group_index < memory->get_kv_tail_group_count() &&
            memory->get_kv_tail_coverage(group_index, seq_id, *out);
}

bool llama_kv_tail_get_coverage_aggregate(
        const llama_context * ctx,
        llama_seq_id seq_id,
        llama_kv_tail_coverage_aggregate * out) {
    if (!ctx || !out) {
        return false;
    }

    *out = {};
    const llama_memory_t memory = ctx->get_memory();
    if (!memory) {
        return false;
    }
    const uint32_t n_groups = memory->get_kv_tail_group_count();
    for (uint32_t group = 0; group < n_groups; ++group) {
        llama_kv_tail_coverage_info info;
        if (!memory->get_kv_tail_coverage(group, seq_id, info)) {
            return false;
        }
        ++out->groups;
        out->requested += info.requested;
        out->exact += info.exact;
        out->degradation_flags |= info.degradation_flags;
        switch (info.state) {
            case LLAMA_KV_TAIL_COVERAGE_COMPLETE: ++out->complete_groups; break;
            case LLAMA_KV_TAIL_COVERAGE_PARTIAL:  ++out->partial_groups;  break;
            case LLAMA_KV_TAIL_COVERAGE_NONE:     ++out->none_groups;     break;
        }
    }
    return true;
}

void llama_kv_tail_planner_timing_reset(llama_context * ctx) {
    if (ctx && ctx->get_memory()) {
        ctx->get_memory()->reset_kv_tail_planner_timing();
    }
}

uint64_t llama_kv_tail_planner_timing_ns(const llama_context * ctx) {
    return ctx && ctx->get_memory() ? ctx->get_memory()->get_kv_tail_planner_timing_ns() : 0;
}

//
// llama_context
//

// check that each folded weight in the graph gets its Hadamard transform, and each latent lookup gets the inverse
// without this check, an arch that skips the transform helpers loads and computes wrong results
static void llama_verify_hadamard_graph(
        ggml_cgraph * gf,
        const llama_hadamard_rotations & rotations,
        const llama_hadamard_rotations & inverses,
        const llama_moe_cache * moe_cache) {
    // the MoE cache gives the matmul a copy of the expert weights, so the copy also needs the transform of its source
    llama_hadamard_rotations forward = rotations;
    for (const auto & [w, t] : rotations) {
        if (const ggml_tensor * cached = moe_cache ? moe_cache->get_experts(w) : nullptr) {
            forward.emplace(cached, t);
        }
    }

    auto unwrap = [](const ggml_tensor * t) {
        while (t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW)) {
            t = t->src[0];
        }
        return t;
    };

    std::map<const ggml_tensor *, bool> lookups; // get_rows results of latent tables

    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        const ggml_tensor * node = ggml_graph_node(gf, i);

        if (node->op == GGML_OP_GET_ROWS && inverses.count(node->src[0])) {
            lookups.emplace(node, false);
            continue;
        }

        if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }

        if (node->op == GGML_OP_MUL_MAT && ((const int32_t *) node->op_params)[1] == GGML_HINT_SRC0_IS_HADAMARD) {
            const auto lk = lookups.find(unwrap(node->src[1]));
            if (lk != lookups.end()) {
                lk->second = true;
            }
            continue;
        }

        const auto it = forward.find(node->src[0]);
        if (it == forward.end()) {
            if (inverses.count(node->src[0])) {
                throw std::runtime_error(format(
                    "Hadamard-latent table '%s' is used as a head without a forward transform", node->src[0]->name));
            }
            continue;
        }
        const ggml_tensor * src = unwrap(node->src[1]);
        const bool transformed = src && src->op == GGML_OP_MUL_MAT &&
            ((const int32_t *) src->op_params)[1] == GGML_HINT_SRC0_IS_HADAMARD &&
            src->src[0] == it->second.rot;
        if (!transformed) {
            throw std::runtime_error(format(
                "Hadamard-folded weight '%s' is consumed without its activation transform; "
                "this graph's matmul path does not support prism.hadamard folding",
                node->src[0]->name));
        }
    }

    for (const auto & [node, ok] : lookups) {
        if (!ok) {
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                const ggml_tensor * n2 = ggml_graph_node(gf, i);
                for (int s = 0; s < GGML_MAX_SRC && n2->src[s]; ++s) {
                    if (unwrap(n2->src[s]) == node) {
                        LLAMA_LOG_WARN("%s: latent lookup '%s' consumed by op=%s name='%s' src%d hint=%d\n",
                                __func__, node->name, ggml_op_name(n2->op), n2->name, s,
                                ((const int32_t *) n2->op_params)[1]);
                    }
                }
            }
            throw std::runtime_error(format(
                "Hadamard-latent table '%s' is read without the inverse transform",
                node->src[0]->name));
        }
    }
}

static llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {
    switch (ctx_type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT: return LLM_GRAPH_TYPE_DEFAULT;
        case LLAMA_CONTEXT_TYPE_MTP    : return LLM_GRAPH_TYPE_DECODER_MTP;
    }
    throw std::runtime_error("Unsupported ctx type");
}

struct llm_fused_op_probe {
    llm_fused_op op;
    const char * name;
    uint32_t n_tokens_per_seq;
};

static const llm_fused_op_probe llm_fused_op_flash_attn_probe = {
    /*.op               =*/ LLM_FUSED_OP_FLASH_ATTN,
    /*.name             =*/ "Flash Attention",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ar_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_AR,
    /*.name             =*/ "fused Gated Delta Net (autoregressive)",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ch_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_CH,
    /*.name             =*/ "fused Gated Delta Net (chunked)",
    /*.n_tokens_per_seq =*/ 16,
};

static const llm_fused_op_probe llm_fused_op_lid_probe = {
    /*.op               =*/ LLM_FUSED_OP_LIGHTNING_INDEXER,
    /*.name             =*/ "Lightning Indexer",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_pre_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_PRE,
    /*.name             =*/ "fused DeepSeek V4 HC pre",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_comb_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_COMB,
    /*.name             =*/ "fused DeepSeek V4 HC comb",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_post_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_POST,
    /*.name             =*/ "fused DeepSeek V4 HC post",
    /*.n_tokens_per_seq =*/ 1,
};

static uint64_t next_context_instance() {
    static std::atomic<uint64_t> next{0};
    const uint64_t id = next.fetch_add(1, std::memory_order_relaxed) + 1;
    GGML_ASSERT(id != 0); // Never recycle an identity, including on counter exhaustion.
    return id;
}

// Whether `dev` provides a NATIVE KV-tail attention kernel.
//
// Only the presence of the entry point is checked, not whether it accepts a
// particular type pair: the per-layer planner in llama-kv-cache.cpp already
// asks that finer question. This answers the coarser one -- can this device
// ever serve a precision tail natively -- so that a backend which implements
// no tail attention at all can be recognised before the cache is built.
static bool kv_tail_device_has_native_attention(ggml_backend_dev_t dev) {
    if (dev == nullptr) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    if (dev && ggml_backend_dev_is_meta(dev)) {
        const size_t count = ggml_backend_meta_device_count(dev);
        for (size_t i = 0; i < count; ++i) {
            if (kv_tail_device_has_native_attention(ggml_backend_meta_device_get(dev, i))) {
                return true;
            }
        }
        return false;
    }
    const auto reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg) {
        return false;
    }
    return ggml_backend_reg_get_proc_address(
                   reg, "ggml_backend_kv_tail_segmented_attention_supported") != nullptr ||
           ggml_backend_reg_get_proc_address(
                   reg, "ggml_backend_kv_tail_attention_supported") != nullptr ||
           ggml_backend_reg_get_proc_address(
                   reg, "ggml_backend_kvarn_tail_attention_supported") != nullptr;
}
// Resolve within the Vulkan registry; device indices from other registries are unrelated.
static ggml_backend_dev_t local_attention_device(const char * endpoint) {
    if (!endpoint) return nullptr;
    const std::string value(endpoint);
    if (value != "local" && value != "vulkan" && value.rfind("vulkan:", 0) != 0 &&
            value.rfind("local:", 0) != 0) return nullptr;
    const auto colon = value.find(':');
    const std::string suffix = colon == std::string::npos ? "0" : value.substr(colon + 1);
    if (suffix.empty() || suffix.find_first_not_of("0123456789") != std::string::npos) {
        throw std::runtime_error("local attention device must be vulkan:<index>");
    }
    const auto index = std::stoul(suffix);
    auto * reg = ggml_backend_reg_by_name("Vulkan");
    if (!reg || index >= ggml_backend_reg_dev_count(reg)) {
        throw std::runtime_error("local attention device unavailable: " + value);
    }
    return ggml_backend_reg_dev_get(reg, index);
}

static bool selected_attention_layer(
        const llama_hparams & hparams, int32_t il, int count,
        llama_context_type ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT) {
    const int32_t layer_begin = ctx_type == LLAMA_CONTEXT_TYPE_MTP ? (int32_t) hparams.n_layer() : 0;
    const int32_t layer_end = ctx_type == LLAMA_CONTEXT_TYPE_MTP ?
            (int32_t) hparams.n_layer_all : (int32_t) hparams.n_layer();
    if (il < layer_begin || il >= layer_end || !hparams.has_kv(il) || hparams.is_recr(il)) {
        return false;
    }
    int index = 0;
    for (int32_t l = layer_begin; l < il; ++l) {
        if (hparams.has_kv(l) && !hparams.is_recr(l)) ++index;
    }
    return index < count;
}

// Planned draft-MTP KV reservation (llama_context_params::mtp_reserve_*): size
// the independent draft cache with the DRAFT representation, the resolved
// target capacity and the MTP tail policy (init forces tail 0 / F16; KVarN
// keeps its intrinsic 128-token tail). Shared by the mixed auto-placement
// planner (P == 0) and the LLAMA_MTP_POSITION_SPLIT=1 fit check (P != 0).
static uint64_t estimate_mtp_reserve_bytes(
        const llama_model & model, const llama_context_params & params, const llama_cparams & cparams) {
    llama_kv_mixed_mtp_input mtp_input = {};
    mtp_input.n_ctx = cparams.n_ctx_seq;
    mtp_input.n_seq_max = cparams.n_seq_max;
    mtp_input.kv_unified = cparams.kv_unified;
    mtp_input.n_batch = cparams.n_batch;
    mtp_input.n_ubatch = cparams.n_ubatch;
    mtp_input.kvarn = params.mtp_reserve_kvarn;
    mtp_input.kvarn_bits = params.mtp_reserve_kvarn_bits;
    mtp_input.type_k = params.mtp_reserve_type_k;
    mtp_input.type_v = params.mtp_reserve_type_v;
    mtp_input.tail_tokens = 0; // MTP init forces tail 0 / F16
    mtp_input.tail_type = GGML_TYPE_F16;
    mtp_input.tail_rollback_tokens = params.mtp_reserve_rollback_tokens;
    const uint32_t mtp_layer_begin = model.hparams.n_layer();
    const uint32_t mtp_layer_end = model.hparams.n_layer_all;
    for (uint32_t il = mtp_layer_begin; il < mtp_layer_end; ++il) {
        if (!model.hparams.has_kv(il) || model.hparams.is_recr(il)) {
            continue;
        }
        llama_kv_mixed_mtp_layer layer = {};
        layer.layer = il;
        layer.head_dim_k = model.hparams.n_embd_head_k(il);
        layer.head_dim_v = model.hparams.n_embd_head_v(il);
        layer.n_head_kv = model.hparams.n_head_kv(il);
        layer.n_embd_k_gqa = model.hparams.n_embd_k_gqa(il);
        layer.n_embd_v_gqa = model.hparams.n_embd_v_gqa(il);
        mtp_input.layers.push_back(layer);
    }
    llama_kv_mixed_mtp_budget mtp_budget = {};
    const auto mtp_status = llama_kv_mixed_mtp_budget_estimate(mtp_input, mtp_budget);
    if (mtp_status != LLAMA_KV_MIXED_OK) {
        throw std::runtime_error(format(
                "mixed KV auto-placement cannot reserve the planned draft-MTP cache: %s",
                mtp_budget.error[0] ? mtp_budget.error :
                        llama_kv_mixed_status_name(mtp_status)));
    }
    return mtp_budget.total_bytes;
}

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    context_instance(next_context_instance()),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd(),
                model.arch == LLM_ARCH_DFLASH,
                llm_arch_supports_mixed_batch(model.arch) && params.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT)) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;
    auto * local_attn_dev = local_attention_device(params.remote_attn_host);
    const char * requested_split_env = getenv("LLAMA_KV_POSITION_SPLIT_P");
    const bool split_requested = params.kv_position_split_p != 0 ||
            (params.ctx_type != LLAMA_CONTEXT_TYPE_MTP && params.ctx_other == nullptr &&
             requested_split_env && requested_split_env[0] && std::strcmp(requested_split_env, "0") != 0);
    const bool tcp_position_split = split_requested &&
            params.remote_attn_host && params.remote_attn_host[0] && !local_attn_dev;
    if (tcp_position_split) {
        local_attn_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!local_attn_dev) {
            throw std::runtime_error("TCP position-split requires a CPU checkpoint mirror backend");
        }
    }
    if (!local_attn_dev && params.kv_position_split_p != 0) {
        try {
            local_attn_dev = local_attention_device("vulkan:0");
        } catch (...) {
            // Vulkan unavailable; position-split will validate during setup
        }
    }

    cparams.rkva_wavefront_tile = llama_wavefront::parse_serving_tile_size(
            std::getenv("GGML_RKVA_WAVEFRONT_TILE"));
    // An adaptive server first creates its local/MTP contexts. A global N256
    // request applies only to the remote target without recurrent rollback;
    // otherwise it must not impose UB512 on the local tier (e.g. XL UB128).
    if ((cparams.rkva_wavefront_tile == 256 || cparams.rkva_wavefront_tile == 512) &&
            (!tcp_position_split || params.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT || params.n_rs_seq != 0)) {
        cparams.rkva_wavefront_tile = 0;
    }
    cparams.rkva_wavefront_rpc_batch = llama_wavefront::parse_serving_rpc_batch(
            std::getenv("GGML_RKVA_WAVEFRONT_RPC_BATCH"));
    if (cparams.rkva_wavefront_rpc_batch != 1) {
        throw std::invalid_argument("wavefront RPC_BATCH=4 is unavailable; use RPC_BATCH=1");
    }
    if (cparams.rkva_wavefront_tile == 256 &&
            ((params.n_ubatch != 512 && params.n_ubatch != 1024) || params.n_batch < params.n_ubatch)) {
        throw std::invalid_argument("wavefront NxN256 requires requested ubatch=512 or 1024 and logical batch>=ubatch");
    }
    if (cparams.rkva_wavefront_tile == 512 &&
            (params.n_ubatch < 1024 || params.n_ubatch > 4096 || params.n_ubatch % 512 != 0 ||
             params.n_batch < params.n_ubatch)) {
        throw std::invalid_argument("wavefront Nx512 requires ubatch a multiple of 512 from 1024 to 4096 and logical batch>=ubatch");
    }
    if (cparams.rkva_wavefront_tile) {
        LLAMA_LOG_WARN("RKVA wavefront experimental: tile=%u; eligible Qwen35 dense all-overflow single-sequence n_rs_seq=0 only; explicit RPC batching policy, no rollback after parent failure\n",
                cparams.rkva_wavefront_tile);
    }

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    // Position-split boundary (plan §3.1/§3.8). Abaixo de P, a fronteira já
    // orienta divisão do ubatch, alocação por faixa, stores locais e
    // remoção/rollback; a leitura única acima de P continua pendente da
    // segunda faixa e do merge (F3). O valor é validado estritamente: múltiplo
    // de 256, dentro do contexto, e nunca migrado silenciosamente.
    cparams.kv_position_split_p = 0;
    // The strict P validation (plan 3.1/3.8) is NOT done here: at this point
    // cparams.n_ctx_seq has not been computed yet, and reading it returned an
    // indeterminate value -- the old log printed C=21979 and C=22091, neither a
    // multiple of 256, and a valid P=512 was rejected with "0 < P < n_ctx" on
    // some runs and accepted on others. It runs further down, right after
    // n_ctx_seq is derived and padded.

    cparams.kv_tail_rollback_tokens = std::max(params.kv_tail_rollback_tokens, params.n_rs_seq);
    cparams.n_rs_seq = params.n_rs_seq;
    if (cparams.n_rs_seq > 0 && !llm_arch_supports_rs_rollback(model.arch)) {
        LLAMA_LOG_DEBUG("%s: n_rs_seq=%u requested but model does not support recurrent partial rollback; clamping to 0\n",
                        __func__, cparams.n_rs_seq);
        cparams.n_rs_seq = 0;
    }

    cparams.n_threads               = params.n_threads;
    cparams.n_threads_batch         = params.n_threads_batch;
    cparams.yarn_ext_factor         = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor        = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast          = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow          = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings              = params.embeddings || params.dflash_split;
    cparams.dflash_split            = params.dflash_split;
    cparams.dflash_selector_only    = params.dflash_selector_only;
    cparams.embeddings_nextn        = params.dflash_selector_only;
    cparams.embeddings_nextn_masked = false;
    cparams.offload_kqv             = params.offload_kqv;
    cparams.offload_rs              = params.offload_kqv && !params.no_offload_rs;
    if (params.no_offload_rs && params.offload_kqv) {
        LLAMA_LOG_INFO("%s: recurrent state cache uses host memory (--no-offload-rs); attention KV remains offloaded\n", __func__);
    }
    cparams.no_perf                 = params.no_perf;
    cparams.warmup                  = false;

    // +1: id n_layer() taps the output of the last layer ("input" of the head)
    cparams.embeddings_layer_inp.resize(hparams.n_layer() + 1, false);
    embd_layer_inp.resize(hparams.n_layer() + 1);

    cparams.ctx_type          = params.ctx_type;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.pooling_type      = params.pooling_type;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;
    if (params.kvarn.window_chunk > uint32_t(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument("KVarN window chunk exceeds the supported range");
    }
    cparams.kvarn             = params.kvarn;
    cparams.remote_attn_enabled = (params.remote_attn_host != nullptr && params.remote_attn_host[0] != '\0');
    cparams.remote_attn_position_split = tcp_position_split;
    cparams.remote_attn_prefill = params.remote_attn_prefill;
    cparams.remote_attn_stats   = (params.remote_attn_stats != 0);
    cparams.remote_attn_cache_type_k = params.remote_attn_cache_type_k;
    cparams.remote_attn_cache_type_v = params.remote_attn_cache_type_v;
    const bool remote_type_k = cparams.remote_attn_cache_type_k != GGML_TYPE_COUNT;
    const bool remote_type_v = cparams.remote_attn_cache_type_v != GGML_TYPE_COUNT;
    if (remote_type_k != remote_type_v) {
        throw std::invalid_argument("mixed remote KV requires both K and V cache types");
    }
    if (remote_type_k) {
        const auto is_qx = [](ggml_type type) {
            return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q5_0 ||
                    type == GGML_TYPE_Q6_0 || type == GGML_TYPE_Q8_0;
        };
        const bool formats_supported =
                (is_qx(cparams.remote_attn_cache_type_k) && is_qx(cparams.remote_attn_cache_type_v)) ||
                (tcp_position_split && cparams.remote_attn_cache_type_k == GGML_TYPE_F16 &&
                 cparams.remote_attn_cache_type_v == GGML_TYPE_F16);
        if (!local_attn_dev || !cparams.remote_attn_enabled || model.arch != LLM_ARCH_QWEN35 ||
                cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP ||
                cparams.kvarn.type == LLAMA_KVARN_TYPE_DISABLED ||
                cparams.n_seq_max != 1 || params.kv_unified || params.kv_paged ||
                !params.offload_kqv || params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED ||
                cparams.remote_attn_prefill != 0 || !formats_supported) {
            throw std::invalid_argument(
                    "per-layer remote KV formats require Qwen35 target context, native Vulkan or TCP position-split, "
                    "KVarN local cache, Q4/Q5/Q6/Q8 or matching TCP position-split F16 remote K/V, "
                    "Flash Attention, KV offload, "
                    "one non-unified slot, and static prefill");
        }
    }
    cparams.remote_attn_layers  = 0;
    cparams.kv_tail_tokens    = std::min(params.kv_tail_tokens, cparams.n_ctx);
    cparams.kv_tail_tokens_swa = std::min(params.kv_tail_tokens,
            std::min(cparams.n_ctx, hparams.n_swa > 0 ? hparams.n_swa : cparams.n_ctx));
    cparams.kv_tail_tokens_requested = params.kv_tail_tokens;
    cparams.kv_tail_tokens_swa_requested = params.kv_tail_tokens;
    if (params.kv_tail_type != GGML_TYPE_COUNT &&
            params.kv_tail_type != GGML_TYPE_F16 && params.kv_tail_type != GGML_TYPE_BF16) {
        throw std::invalid_argument("KV tail type must be F16, BF16, or the cache-family default");
    }
    cparams.kv_tail_type = params.kv_tail_type;
    bool tail_request_resolved = false;
    bool mixed_additional_tail_requested = params.kv_tail_tokens > 0 || params.kv_tail_config != nullptr;
    if (params.kv_tail_request && params.kv_tail_config) {
        throw std::invalid_argument("KV tail request and model-bound config are mutually exclusive");
    }
    if (params.kv_tail_request) {
        std::unique_ptr<llama_kv_tail_config, decltype(&llama_kv_tail_config_free)> manifest(
                llama_kv_tail_config_init(&model), llama_kv_tail_config_free);
        if (!manifest) {
            throw std::runtime_error("failed to build model-bound KV tail group manifest");
        }
        std::vector<llama_kv_tail_request_group> groups;
        groups.reserve(manifest->groups.size());
        for (const auto & group : manifest->groups) {
            groups.push_back({ group.id, group.role,
                    std::min(group.effective_window, cparams.n_ctx),
                    model.arch != LLM_ARCH_DEEPSEEK4 });
        }
        const auto resolution = llama_kv_tail_request_resolve(
                *params.kv_tail_request, groups,
                params.kvarn.type != LLAMA_KVARN_TYPE_DISABLED);
        if (!resolution.valid) {
            throw std::invalid_argument("invalid model-bound KV tail request: " + resolution.error);
        }
        cparams.kv_tail_tokens = 0;
        cparams.kv_tail_tokens_swa = 0;
        cparams.kv_tail_tokens_requested = 0;
        cparams.kv_tail_tokens_swa_requested = 0;
        for (const auto & group : resolution.groups) {
            mixed_additional_tail_requested = mixed_additional_tail_requested || group.raw_requested_tokens > 0;
            if (group.role == "swa") {
                cparams.kv_tail_tokens_swa_requested = group.requested_tokens;
                cparams.kv_tail_tokens_swa = group.effective_tokens;
                cparams.kv_tail_native_exact_swa = group.native_exact;
            } else {
                cparams.kv_tail_tokens_requested = group.requested_tokens;
                cparams.kv_tail_tokens = group.effective_tokens;
                cparams.kv_tail_native_exact = group.native_exact;
            }
            cparams.kv_tail_type = group.exact_type;
            LLAMA_LOG_INFO("KV tail: group=%s raw=%u requested=%u effective=%u type=%s representation=%s\n",
                    group.id.c_str(), group.raw_requested_tokens, group.requested_tokens,
                    group.effective_tokens, ggml_type_name(group.exact_type),
                    group.native_exact ? "native_exact" : "overlay");
        }
        tail_request_resolved = true;
    } else if (params.kv_tail_config) {
        const auto & config = *params.kv_tail_config;
        if (config.model != &model) {
            throw std::invalid_argument("KV tail config was created for a different model");
        }
        cparams.kv_tail_tokens = 0;
        cparams.kv_tail_tokens_swa = 0;
        cparams.kv_tail_tokens_requested = 0;
        cparams.kv_tail_tokens_swa_requested = 0;
        const bool explicit_complete = !std::any_of(config.groups.begin(), config.groups.end(), [](const auto & group) {
            return !group.assigned;
        });
        const bool automatic_standard = params.kvarn.type == LLAMA_KVARN_TYPE_DISABLED &&
                model.arch != LLM_ARCH_DEEPSEEK4;
        std::vector<llama_kv_tail_group_request> requests;
        requests.reserve(config.groups.size());
        for (const auto & group : config.groups) {
            requests.push_back({ group.n_tokens, std::min(group.effective_window, cparams.n_ctx),
                    config.automatic ? automatic_standard : true });
        }
        const auto resolution = llama_kv_tail_resolve_groups(
                config.automatic, explicit_complete, requests);
        if (!resolution.valid) {
            LLAMA_LOG_WARN("%s: incomplete KV tail group configuration; resolving all groups to zero\n", __func__);
        } else {
            for (size_t i = 0; i < config.groups.size(); ++i) {
                const auto & group = config.groups[i];
                const uint32_t requested = config.automatic ? 1024 : group.n_tokens;
                const uint32_t resolved = resolution.tokens[i];
                if (group.role == "swa") {
                    cparams.kv_tail_tokens_swa_requested = requested;
                    cparams.kv_tail_tokens_swa = resolved;
                } else {
                    cparams.kv_tail_tokens_requested = requested;
                    cparams.kv_tail_tokens = resolved;
                }
                LLAMA_LOG_INFO("KV tail: group=%s layers=%u requested=%u effective=%u\n",
                        group.id.c_str(), uint32_t(group.layers.size()), requested, resolved);
            }
        }
    }

    // A KV precision tail is only a win where some device can serve it with NATIVE
    // tail attention. Where none can, planning still SUCCEEDS: every layer falls back
    // to the generic tail route, which llama-kv-cache.cpp itself logs as "catastrophic
    // generic attention". That is not a quality/size trade, it is a large throughput
    // loss for a feature the user asked for expecting the opposite -- and the Metal
    // backend exports no tail-attention entry point at all, so every Metal build pays
    // it in full. Measured on an M1 Max, Qwen3.8-27B IQ4_XS at 100k ctx, k=v=q5_0:
    // prompt 39 -> 112 tok/s and decode 5.8 -> 9.0 tok/s simply by dropping
    // --kv-tail-tokens.
    //
    // So decline the tail here, the same way KVarN declines itself just below when its
    // requirements do not hold. LLAMA_KV_TAIL_ALLOW_GENERIC=1 keeps the old behaviour
    // for anyone who wants the exact tail regardless of what it costs.
    if (cparams.kv_tail_tokens > 0 || cparams.kv_tail_tokens_swa > 0) {
        bool any_native = false;
        if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT && params.ctx_other != nullptr) {
            const auto & shared_cparams = params.ctx_other->get_cparams();
            // Shared Gemma MTP reads the target cache's representation, so preserve
            // the target context's already-resolved tail decision.
            any_native = shared_cparams.kv_tail_tokens > 0 ||
                    shared_cparams.kv_tail_tokens_swa > 0;
        } else {
            const uint32_t layer_begin = cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP ?
                    hparams.n_layer() : 0;
            const uint32_t layer_end = cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP ?
                    hparams.n_layer_all : hparams.n_layer();
            for (uint32_t il = layer_begin; il < layer_end; ++il) {
                if (!hparams.has_kv(il)) {
                    continue;
                }
                const int early_count = params.remote_attn_n_layers <= 0 ? INT32_MAX : params.remote_attn_n_layers;
                auto * kv_dev = local_attn_dev && selected_attention_layer(hparams, il, early_count, params.ctx_type) ?
                        local_attn_dev : model.dev_layer(il);
                if (kv_tail_device_has_native_attention(cparams.offload_kqv ? kv_dev : nullptr)) {
                    any_native = true;
                    break;
                }
            }
        }

        if (!any_native) {
            const uint32_t requested = std::max(cparams.kv_tail_tokens, cparams.kv_tail_tokens_swa);
            const char * allow_generic = getenv("LLAMA_KV_TAIL_ALLOW_GENERIC");
            if (allow_generic && allow_generic[0] != '\0' && strcmp(allow_generic, "0") != 0) {
                LLAMA_LOG_WARN("%s: no device provides native KV tail attention, but "
                        "LLAMA_KV_TAIL_ALLOW_GENERIC is set; keeping the %u-token precision "
                        "tail on the generic route, which is much slower\n", __func__, requested);
            } else {
                LLAMA_LOG_WARN("%s: no device provides native KV tail attention; disabling the "
                        "%u-token KV precision tail. The generic tail route costs several times "
                        "more than plain quantized attention, so it is not enabled by default. "
                        "Set LLAMA_KV_TAIL_ALLOW_GENERIC=1 to keep it anyway.\n",
                        __func__, requested);
                cparams.kv_tail_tokens = 0;
                cparams.kv_tail_tokens_swa = 0;
                cparams.kv_tail_tokens_requested = 0;
                cparams.kv_tail_tokens_swa_requested = 0;
                cparams.kv_tail_native_exact = false;
                cparams.kv_tail_native_exact_swa = false;
                if (params.kvarn.type != LLAMA_KVARN_TYPE_DISABLED) {
                    // Reapply KVarN's zero-request policy below so its intrinsic
                    // exact suffix and native-exact state remain consistent.
                    tail_request_resolved = false;
                }
            }
        }
    }

    if (params.kvarn.type != LLAMA_KVARN_TYPE_DISABLED && !tail_request_resolved) {
        const uint32_t full_window = cparams.n_ctx;
        const uint32_t swa_window = std::min(cparams.n_ctx,
                hparams.n_swa > 0 ? hparams.n_swa : cparams.n_ctx);
        const uint32_t raw_full = cparams.kv_tail_tokens_requested;
        const uint32_t raw_swa = cparams.kv_tail_tokens_swa_requested;
        const auto full_policy = llama_kvarn_tail_policy_for(raw_full, full_window);
        const auto swa_policy = llama_kvarn_tail_policy_for(raw_swa, swa_window);
        cparams.kv_tail_tokens = full_policy.effective_tokens;
        cparams.kv_tail_tokens_swa = swa_policy.effective_tokens;
        cparams.kv_tail_tokens_requested = full_policy.requested_tokens;
        cparams.kv_tail_tokens_swa_requested = swa_policy.requested_tokens;
        cparams.kv_tail_native_exact = full_policy.native_exact;
        cparams.kv_tail_native_exact_swa = swa_policy.native_exact;
        LLAMA_LOG_INFO("KVarN exact tail: group=full raw=%u requested=%u effective=%u window=%u representation=%s\n",
                raw_full, full_policy.requested_tokens, full_policy.effective_tokens, full_window,
                full_policy.native_exact ? "native_exact" : "overlay");
        if (hparams.n_swa > 0) {
            LLAMA_LOG_INFO("KVarN exact tail: group=swa raw=%u requested=%u effective=%u window=%u representation=%s\n",
                    raw_swa, swa_policy.requested_tokens, swa_policy.effective_tokens, swa_window,
                    swa_policy.native_exact ? "native_exact" : "overlay");
        }
    }
    // Opt-in measurement control, resolved before allocation and graph planning.
    // Numeric --kv-tail-tokens=0 retains its ordinary intrinsic-128 semantics.
    // Owned Qwen MTP contexts keep their independent, ordinary cache policy.
    const char * experimental_tail = getenv("LLAMA_KVARN_EXPERIMENTAL_TARGET_TAIL");
    if (experimental_tail && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT) {
        if ((strcmp(experimental_tail, "0") != 0 && strcmp(experimental_tail, "128") != 0) ||
                params.kvarn.type == LLAMA_KVARN_TYPE_DISABLED || model.arch != LLM_ARCH_QWEN35 ||
                hparams.n_swa > 0) {
            throw std::invalid_argument(
                    "LLAMA_KVARN_EXPERIMENTAL_TARGET_TAIL requires 0 or 128 and a non-SWA Qwen35 KVarN target");
        }
        const bool retain_intrinsic = strcmp(experimental_tail, "128") == 0;
        const auto policy = llama_kvarn_tail_policy_for(0, cparams.n_ctx, retain_intrinsic);
        cparams.kv_tail_tokens = policy.effective_tokens;
        cparams.kv_tail_tokens_swa = policy.effective_tokens;
        cparams.kv_tail_tokens_requested = policy.requested_tokens;
        cparams.kv_tail_tokens_swa_requested = policy.requested_tokens;
        cparams.kv_tail_native_exact = policy.native_exact;
        cparams.kv_tail_native_exact_swa = policy.native_exact;
        cparams.kv_tail_type = GGML_TYPE_F16;
        LLAMA_LOG_WARN("EXPERIMENTAL Qwen35 target exact tail: effective=%u type=f16; "
                "KVarN sink and incomplete-group F16 staging remain enabled; owned MTP policy unchanged\n",
                policy.effective_tokens);
    }
    if ((cparams.kv_tail_tokens > 0 || cparams.kv_tail_tokens_swa > 0) &&
            cparams.kv_tail_rollback_tokens == 0) {
        // The common capability probe removes one suffix token. Compact
        // transformer caches retain that row explicitly instead of relying on
        // the active ubatch as accidental rollback storage.
        cparams.kv_tail_rollback_tokens = 1;
    }
    if (remote_type_k && (mixed_additional_tail_requested || hparams.n_swa > 0)) {
        throw std::invalid_argument("mixed remote KV currently requires no additional tail and no SWA layers");
    }

    cparams.ctx_other = nullptr;

    if (cparams.dflash_split &&
            (model.arch != LLM_ARCH_DFLASH || hparams.dflash_selector_rank == 0 ||
             params.ctx_other != nullptr || params.dflash_selector_only ||
             model.dfly_layer_fusion != nullptr ||
             model.dspark_markov_w1 != nullptr ||
             cparams.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        throw std::invalid_argument("dflash_split requires plain DFlash2, no ctx_other, and pooling_type NONE");
    }
    if (cparams.dflash_selector_only &&
            (model.arch != LLM_ARCH_DFLASH || hparams.dflash_selector_rank == 0 ||
             params.ctx_other == nullptr || model.dfly_layer_fusion != nullptr ||
             model.dspark_markov_w1 != nullptr ||
             cparams.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        throw std::invalid_argument("dflash_selector_only requires plain DFlash2, ctx_other, and pooling_type NONE");
    }

    // TODO: more generic
    if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        if (params.ctx_other == nullptr) {
            // TODO: change from runtime_error to llama_exception to avoid printing error message
            throw std::runtime_error("Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)");
        }

        cparams.ctx_other = params.ctx_other;
    }

    if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH || model.arch == LLM_ARCH_QWEN4EXP) {
        if (cparams.dflash_selector_only || model.tok_embd == nullptr || model.output == nullptr) {
            if (params.ctx_other == nullptr && !cparams.dflash_split) {
                throw std::runtime_error(model.arch_name() + " requires ctx_other to be set (this warning is normal during memory fitting)");
            }
            cparams.ctx_other = params.ctx_other;
        }
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        cparams.rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;
    if (remote_type_k && (!cparams.flash_attn || !cparams.offload_kqv)) {
        throw std::invalid_argument("per-layer remote Qx KV requires Flash Attention with KV offload enabled");
    }

    cparams.fused_gdn_ar = true;
    cparams.fused_gdn_ch = true;
    cparams.auto_fgdn    = false;

    cparams.fused_lid = true;
    cparams.auto_flid = false;

    cparams.fused_dsv4_hc_pre  = true;
    cparams.fused_dsv4_hc_comb = true;
    cparams.fused_dsv4_hc_post = true;
    cparams.auto_fhc           = true;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.n_outputs_max = params.n_outputs_max == 0 || llama_model_has_encoder(&model) ? cparams.n_batch : params.n_outputs_max;
    cparams.n_outputs_max_per_seq = params.n_outputs_max_per_seq == 0 ?
            cparams.n_outputs_max : std::min(params.n_outputs_max_per_seq, cparams.n_outputs_max);

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    cparams.op_offload = params.op_offload;
    cparams.kv_unified = params.kv_unified;
    cparams.kv_paged   = params.kv_paged;
    cparams.kv_paged_dynamic = params.kv_paged_dynamic;
    cparams.block_size = params.block_size;
    cparams.n_gpu_blocks = params.n_gpu_blocks;
    cparams.n_gpu_blocks_initial = params.n_gpu_blocks_initial;
    cparams.n_gpu_blocks_growth = params.n_gpu_blocks_growth;
    cparams.n_cpu_blocks = params.n_cpu_blocks;
    cparams.kv_paged_watermark = params.kv_paged_watermark;
    cparams.snapkv_enabled = params.snapkv_enabled;
    cparams.snapkv_observation_window = params.snapkv_observation_window;
    cparams.snapkv_recent_tokens = params.snapkv_recent_tokens;
    cparams.snapkv_pinned_tokens = params.snapkv_pinned_tokens;
    cparams.snapkv_retention = params.snapkv_retention;
    cparams.snapkv_budget_blocks = params.snapkv_budget_blocks;
    cparams.moe_cache_size = params.moe_cache_size;

    // initialized later
    cparams.pipeline_parallel = false;
    cparams.training = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;
        const char * attention_reference = getenv("LLAMA_OVF_ATTN_REFERENCE_CPU");
        if (attention_reference && std::strcmp(attention_reference, "1") == 0 &&
                getenv("LLAMA_OVF_ATTN_REFERENCE_CPU_MIN_POS")) {
            // The reference suffix changes graph placement as positions cross
            // the threshold, even when all tensor dimensions stay unchanged.
            graph_reuse_disable = true;
        }

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    // Position-split boundary (plan 3.1/3.8): multiple of 256, 0 < P < effective
    // n_ctx_seq, never migrated silently. Running it against the EFFECTIVE value
    // makes the verdict deterministic and keeps the two failure classes apart:
    // this throw is an invalid SETTING, while an out-of-memory or backend
    // failure is reported separately by the caller.
    {
        uint64_t value = params.kv_position_split_p;
        std::string requested = std::to_string(value);
        const char * env_p = getenv("LLAMA_KV_POSITION_SPLIT_P");
        // Never resurrect P from the env for an MTP draft (or any auxiliary)
        // context: common_base_params_to_speculative zeroes it on purpose, and
        // an MTP context with P != 0 is rejected in create_memory.
        const bool env_p_fallback = llama_position_split::position_split_env_p_fallback_applies(
                params.ctx_type == LLAMA_CONTEXT_TYPE_MTP || params.ctx_other != nullptr);
        if (value == 0 && env_p_fallback && env_p && env_p[0] != '\0') {
            char * end = nullptr;
            errno = 0;
            value = strtoull(env_p, &end, 10);
            requested = env_p;
            if (errno == ERANGE || !end || *end != '\0' || env_p[0] == '-') {
                throw std::invalid_argument("invalid LLAMA_KV_POSITION_SPLIT_P: " + requested);
            }
            if (value == 0) { throw std::invalid_argument("LLAMA_KV_POSITION_SPLIT_P must be positive"); }
        }
        if (value != 0) {
            // LLAMA_MTP_POSITION_SPLIT=1 (exact string) opts into running the
            // MTP draft over a position-split target. Absent/any other value
            // keeps the fail-closed guard byte for byte.
            const bool mtp_position_split = llama_position_split::mtp_position_split_env_enabled(
                    getenv("LLAMA_MTP_POSITION_SPLIT"));
            if (params.mtp_reserve_enabled && !mtp_position_split) {
                throw std::invalid_argument("position-split KV requires MTP disabled in the active profile");
            }
            if (params.mtp_reserve_enabled) {
                // Same split contracts as the non-MTP path, restated against
                // this context because the split and the MTP reservation now
                // share the target: one sequence, non-unified cache, no SWA.
                if (cparams.n_seq_max != 1) {
                    throw std::invalid_argument(
                            "LLAMA_MTP_POSITION_SPLIT=1 requires a single sequence (n_seq_max=1)");
                }
                if (cparams.kv_unified) {
                    throw std::invalid_argument(
                            "LLAMA_MTP_POSITION_SPLIT=1 requires a non-unified KV cache");
                }
                if (hparams.n_swa > 0 && hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
                    throw std::invalid_argument(
                            "LLAMA_MTP_POSITION_SPLIT=1 does not support SWA/iSWA memory");
                }
                LLAMA_LOG_WARN("%s: LLAMA_MTP_POSITION_SPLIT=1: MTP draft enabled over position-split KV "
                               "(P=%llu); the step that would cross P decodes non-speculatively\n",
                               __func__, (unsigned long long) value);
            }
            if (value > UINT32_MAX || !llama_position_split::is_valid_p(uint32_t(value), cparams.n_ctx_seq)) {
                throw std::runtime_error(
                    "KV position-split P must be a multiple of 256 with 0 < P < n_ctx: "
                    "requested_P=" + requested +
                    " requested_n_ctx=" + std::to_string(params.n_ctx) +
                    " n_seq_max=" + std::to_string(cparams.n_seq_max) +
                    " effective_n_ctx_seq=" + std::to_string(cparams.n_ctx_seq) +
                    " [invalid setting, not an allocation failure]");
            }
            cparams.kv_position_split_p = uint32_t(value);
            LLAMA_LOG_INFO("%s: position-split KV placement enabled with P=%u "
                           "(requested n_ctx=%u effective n_ctx_seq=%u n_seq_max=%u)\n",
                           __func__, cparams.kv_position_split_p, params.n_ctx,
                           cparams.n_ctx_seq, cparams.n_seq_max);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max             = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx                 = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq             = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch               = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch              = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn           = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn            = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified            = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base             = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale            = %g\n",   __func__, cparams.rope_freq_scale);
    LLAMA_LOG_INFO("%s: n_rs_seq              = %u\n",   __func__, cparams.n_rs_seq);
    LLAMA_LOG_INFO("%s: n_outputs_max         = %u\n",   __func__, cparams.n_outputs_max);
    LLAMA_LOG_INFO("%s: n_outputs_max_per_seq = %u\n",   __func__, cparams.n_outputs_max_per_seq);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_INFO("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    // The selector-only graph still needs the target head's device and a CPU backend.
    if (!hparams.vocab_only) {
        // GPU backends
        std::vector<ggml_backend_dev_t> initialized_devices;
        auto add_device_backend = [&](const llama_device & dev) {
            if (std::find(initialized_devices.begin(), initialized_devices.end(), dev.dev) !=
                    initialized_devices.end()) {
                return;
            }
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
            initialized_devices.push_back(dev.dev);
        };
        for (const auto & dev : model.devices) {
            add_device_backend(dev);
        }

        // EAGLE3, DFlash, and assistant contexts borrow target-model tensors.
        // Their scheduler must therefore know the owning target devices in
        // addition to the devices that hold the auxiliary model itself.  This
        // is normally invisible when both models use the same simple backend,
        // but a tensor-split target owns its weights through a distinct meta
        // device.  Adding only missing devices keeps auxiliary layers on their
        // requested device while borrowed projections execute where they are
        // physically resident.
        if (cparams.ctx_other != nullptr) {
            const llama_model * model_other = llama_get_model(cparams.ctx_other);
            GGML_ASSERT(model_other != nullptr);
            for (const auto & dev : model_other->devices) {
                add_device_backend(dev);
            }
        }

        // if ctx_other is provided, add devices from ctx_other model so shared tensors can run on their backends
        if (cparams.ctx_other != nullptr) {
            const auto * model_other = llama_get_model(cparams.ctx_other);
            if (model_other != nullptr) {
                for (const auto & dev : model_other->devices) {
                    bool found = false;
                    for (const auto & b : backends) {
                        if (ggml_backend_get_device(b.get()) == dev.dev) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
                        if (backend == nullptr) {
                            throw std::runtime_error(format("failed to initialize %s backend from ctx_other", ggml_backend_dev_name(dev.dev)));
                        }
                        backends.emplace_back(backend);
                    }
                }
            }
        }

        if (local_attn_dev && !tcp_position_split) {
            if (model.arch != LLM_ARCH_QWEN35 || cparams.kv_paged || !cparams.offload_kqv ||
                    cparams.remote_attn_prefill < 0 || cparams.remote_attn_prefill > 1) {
                throw std::runtime_error(
                    "local attention requires Qwen35, non-paged KV, and --offload-kqv");
            }
            if (params.remote_attn_prefill == 1 &&
                    (cparams.kvarn.type == LLAMA_KVARN_TYPE_DISABLED ||
                     cparams.n_seq_max != 1 || cparams.kv_unified ||
                     hparams.swa_type != LLAMA_SWA_TYPE_NONE || params.ctx_other != nullptr ||
                     params.ctx_type == LLAMA_CONTEXT_TYPE_MTP)) {
                throw std::runtime_error(
                    "prefill migration requires single-stream non-unified KVarN without SWA or draft contexts");
            }
            auto * dev = local_attn_dev;
            const std::string name = ggml_backend_dev_name(dev);
            ggml_backend_t remote_backend = nullptr;
            for (auto & backend : backends) {
                if (ggml_backend_get_device(backend.get()) == dev) {
                    remote_backend = backend.get();
                    break;
                }
            }
            if (!remote_backend) {
                auto * backend = ggml_backend_dev_init(dev, nullptr);
                if (!backend) {
                    throw std::runtime_error("failed to initialize local attention device: " + name);
                }
                backends.emplace_back(backend);
                remote_backend = backend;
            }
            cparams.local_attn_migration = params.remote_attn_prefill == 1;
            if (cparams.local_attn_migration) {
                cparams.local_attn_migration_backend = remote_backend;
                ggml_backend_dev_t prefill_dev = nullptr;
                for (const auto & dev_layer : model.devices) {
                    if (dev_layer.dev != nullptr &&
                            ggml_backend_dev_type(dev_layer.dev) == GGML_BACKEND_DEVICE_TYPE_GPU &&
                            std::strncmp(ggml_backend_dev_name(dev_layer.dev), "CUDA", 4) == 0) {
                        prefill_dev = dev_layer.dev;
                        break;
                    }
                }
                if (prefill_dev == nullptr) {
                    throw std::runtime_error("prefill migration requires CUDA model weights");
                }
                for (auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == prefill_dev) {
                        cparams.local_attn_prefill_backend = backend.get();
                        break;
                    }
                }
                if (cparams.local_attn_prefill_backend == nullptr) {
                    auto * backend = ggml_backend_dev_init(prefill_dev, nullptr);
                    if (backend == nullptr) {
                        throw std::runtime_error("failed to initialize CUDA prefill backend");
                    }
                    backends.emplace_back(backend);
                    cparams.local_attn_prefill_backend = backend;
                }
                cparams.local_attn_backend = cparams.local_attn_prefill_backend;
                LLAMA_LOG_INFO("%s: prefill migration enabled: attention starts on %s and mirrors KVarN to %s\n",
                        __func__, ggml_backend_dev_name(prefill_dev), name.c_str());
            } else {
                cparams.local_attn_backend = remote_backend;
            }
            LLAMA_LOG_INFO("%s: native local attention and KV backend: %s\n", __func__, name.c_str());
        } else if (params.remote_attn_prefill != 0) {
            throw std::runtime_error("--remote-attn-prefill=migrate requires an in-process Vulkan attention backend");
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);
        if (tcp_position_split) {
            cparams.local_attn_backend = backend_cpu;
        }

        // Opt in per CUDA backend owned by this TCP position-split context.
        // Configure before any graph submission; other contexts keep the
        // legacy global setting (or its default).
        const int remote_graph_stable_calls = llama_remote_cuda_graph_min_stable_calls(
                tcp_position_split, std::getenv("GGML_CUDA_GRAPH_REMOTE_MIN_STABLE_CALLS"));
        if (remote_graph_stable_calls < 0) {
            throw std::invalid_argument("GGML_CUDA_GRAPH_REMOTE_MIN_STABLE_CALLS must be a positive integer");
        }
        if (remote_graph_stable_calls > 0) {
            // Preflight the entire set before changing any direct backend. A
            // meta backend owns nested backends that this policy cannot set.
            const bool backends_supported = llama_remote_cuda_graph_backends_supported(backends, [](const auto & backend) {
                auto * dev = ggml_backend_get_device(backend.get());
                return dev && ggml_backend_dev_is_meta(dev);
            });
            if (!backends_supported) {
                throw std::runtime_error("remote CUDA graph policy does not support meta backends");
            }
            using set_policy_fn = bool (*)(ggml_backend_t, int);
            size_t configured = 0;
            for (auto & backend : backends) {
                auto * dev = ggml_backend_get_device(backend.get());
                auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
                if (!reg || std::strcmp(ggml_backend_reg_name(reg), "CUDA") != 0) continue;
                auto * set_policy = (set_policy_fn) ggml_backend_reg_get_proc_address(
                        reg, "ggml_backend_cuda_set_graph_min_stable_calls");
                if (!set_policy || !set_policy(backend.get(), remote_graph_stable_calls)) {
                    throw std::runtime_error("CUDA backend cannot configure the requested remote graph policy");
                }
                ++configured;
            }
            if (configured == 0) {
                throw std::runtime_error("remote CUDA graph policy requested but no CUDA backend is available");
            }
            LLAMA_LOG_INFO("%s: remote position-split CUDA graph minimum stable calls = %d (%zu backends)\n",
                           __func__, remote_graph_stable_calls, configured);
        }

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k    =*/ params.type_k,
            /*.type_v    =*/ params.type_v,
            /*.swa_full  =*/ params.swa_full,
            /*.ctx_type  =*/ cparams.ctx_type,
            /*.kvarn     =*/ cparams.kvarn,
            /*.remote_attn_cache_type_k =*/ cparams.remote_attn_cache_type_k,
            /*.remote_attn_cache_type_v =*/ cparams.remote_attn_cache_type_v,
            /*.kv_tail_tokens =*/ cparams.kv_tail_tokens,
            /*.kv_tail_tokens_swa =*/ cparams.kv_tail_tokens_swa,
            /*.kv_tail_tokens_requested =*/ cparams.kv_tail_tokens_requested,
            /*.kv_tail_tokens_swa_requested =*/ cparams.kv_tail_tokens_swa_requested,
            /*.kv_tail_native_exact =*/ cparams.kv_tail_native_exact,
            /*.kv_tail_native_exact_swa =*/ cparams.kv_tail_native_exact_swa,
            /*.kv_tail_rollback_tokens =*/ cparams.kv_tail_rollback_tokens,
            /*.kv_tail_type   =*/ cparams.kv_tail_type,
            /*.mem_other =*/ llama_get_memory(cparams.ctx_other),
        };

std::vector<ggml_backend_t> layer_backends;
        std::vector<ggml_backend_t> kv_backends;
        if (cparams.kv_paged) {
            layer_backends.resize(model.hparams.n_layer(), backend_cpu);
            for (auto & b : backends) {
                if (ggml_backend_dev_type(ggml_backend_get_device(b.get())) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                    bool in_model = false;
                    for (const auto & dev : model.devices) {
                        if (dev.dev == ggml_backend_get_device(b.get())) {
                            in_model = true;
                            break;
                        }
                    }
                    if (in_model) {
                        kv_backends.push_back(b.get());
                    }
                }
            }
            // Keep host KV spill opt-in: the normal paged-GPU path uses the
            // default one-block bookkeeping pool but must not add the CPU as a
            // candidate unless a real CPU spill capacity was requested.
            if (cparams.kv_paged_dynamic && cparams.n_cpu_blocks > 1) {
                kv_backends.push_back(backend_cpu);
            }
            for (uint32_t il = 0; il < model.hparams.n_layer(); ++il) {
                const auto * layer_dev = model.dev_layer(il);
                for (auto & b : backends) {
                    if (ggml_backend_get_device(b.get()) == layer_dev) {
                        layer_backends[il] = b.get();
                        break;
                    }
                }
            }
        }

        if (cparams.remote_attn_enabled) {
            const uint32_t attention_layer_begin = cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP ?
                    model.hparams.n_layer() : 0;
            const uint32_t attention_layer_end = cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP ?
                    model.hparams.n_layer_all : model.hparams.n_layer();
            int n_full_attn = 0;
            for (uint32_t il = attention_layer_begin; il < attention_layer_end; ++il) {
                if (model.hparams.has_kv(il) && !model.hparams.is_recr(il)) {
                    ++n_full_attn;
                }
            }
            uint64_t mixed_host_ram_cap_bytes = 0;
            if (remote_type_k && n_full_attn > 0 &&
                    params.remote_attn_n_layers == -1 && cparams.kv_position_split_p == 0) {
#if defined(__linux__)
                std::ifstream meminfo("/proc/meminfo");
                if (!meminfo) {
                    throw std::runtime_error("mixed KV placement cannot read /proc/meminfo MemAvailable");
                }
                uint64_t host_mem_available_bytes = 0;
                bool found_mem_available = false;
                std::string memline;
                while (std::getline(meminfo, memline)) {
                    if (memline.compare(0, 13, "MemAvailable:") != 0) {
                        continue;
                    }
                    std::istringstream value(memline.substr(13));
                    uint64_t kib = 0;
                    std::string unit;
                    if (!(value >> kib >> unit) || unit != "kB" || kib > UINT64_MAX/1024u) {
                        throw std::runtime_error("mixed KV placement parsed an invalid MemAvailable value");
                    }
                    host_mem_available_bytes = kib*1024u;
                    found_mem_available = true;
                    break;
                }
                if (!found_mem_available) {
                    throw std::runtime_error("mixed KV placement did not find MemAvailable in /proc/meminfo");
                }
                constexpr uint64_t mixed_host_headroom_bytes = 512ull << 20;
                // Keep N=0 (ordinary CUDA/KVarN placement) valid even when
                // the host is under headroom. A one-byte cap makes every
                // positive remote footprint fail the planner closed.
                mixed_host_ram_cap_bytes = host_mem_available_bytes > mixed_host_headroom_bytes
                        ? host_mem_available_bytes - mixed_host_headroom_bytes
                        : 1;
                LLAMA_LOG_INFO("%s: mixed KV UMA RAM budget: MemAvailable=%.1f MiB, headroom=512.0 MiB, "
                               "cap=%.3f MiB; Vulkan/GTT is not added a second time\n",
                        __func__, host_mem_available_bytes / 1024.0 / 1024.0,
                        mixed_host_ram_cap_bytes / 1024.0 / 1024.0);
#else
                throw std::runtime_error(
                        "mixed KV remote placement requires Linux MemAvailable for a global shared-RAM budget");
#endif
            }
            int n_remote = n_full_attn;
            // Position-split KV (P != 0) keeps BOTH bands on every full-attention
            // layer (local KVarN [0,P) + remote Q4 [P,C)), so the per-layer budget
            // count of the auto policy does not apply: auto resolves exactly like
            // "full" (n_remote stays n_full_attn) and the F3 cache contract still
            // validates explicit counts unchanged.
            if (params.remote_attn_n_layers == -1 && remote_type_k && cparams.kv_position_split_p == 0) {
                ggml_backend_dev_t cuda_dev = nullptr;
                uint32_t cuda_owner_layer = UINT32_MAX;
                for (uint32_t il = attention_layer_begin; il < attention_layer_end; ++il) {
                    if (!model.hparams.has_kv(il) || model.hparams.is_recr(il)) {
                        continue;
                    }
                    ggml_backend_dev_t layer_dev = model.dev_layer(int32_t(il));
                    const char * layer_name = layer_dev ? ggml_backend_dev_name(layer_dev) : nullptr;
                    if (!layer_dev || ggml_backend_dev_type(layer_dev) != GGML_BACKEND_DEVICE_TYPE_GPU ||
                            !layer_name || std::strncmp(layer_name, "CUDA", 4) != 0) {
                        throw std::runtime_error(format(
                                "mixed KV auto-placement requires a CUDA owner for local KVarN; "
                                "model layer %u is on %s",
                                il, layer_name ? layer_name : "an unknown device"));
                    }
                    if (cuda_dev && cuda_dev != layer_dev) {
                        const char * first_name = ggml_backend_dev_name(cuda_dev);
                        throw std::runtime_error(format(
                                "mixed KV auto-placement cannot budget multiple CUDA K/V owners: "
                                "layer %u uses %s and layer %u uses %s",
                                cuda_owner_layer, first_name ? first_name : "CUDA device", il, layer_name));
                    }
                    cuda_dev = layer_dev;
                    cuda_owner_layer = il;
                }
                if (!cuda_dev || !local_attn_dev) {
                    throw std::runtime_error(
                            "mixed KV auto-placement requires a model-layer CUDA owner and native Vulkan budget");
                }
                size_t cuda_free = 0, cuda_total = 0;
                size_t vulkan_free = 0, vulkan_total = 0;
                ggml_backend_dev_memory(cuda_dev, &cuda_free, &cuda_total);
                ggml_backend_dev_memory(local_attn_dev, &vulkan_free, &vulkan_total);
                if ((cuda_free == 0 && cuda_total == 0) ||
                        (vulkan_free == 0 && vulkan_total == 0)) {
                    throw std::runtime_error("mixed KV auto-placement cannot query CUDA/Vulkan free memory");
                }

                llama_kv_mixed_sizing sizing = {};
                sizing.capacity_tokens = cparams.n_ctx_seq;
                sizing.n_seq_max = cparams.n_seq_max;
                sizing.kv_unified = cparams.kv_unified;
                sizing.stage_tail_groups = llama_kvarn_non_swa_tail_groups(
                        cparams.n_batch, cparams.n_ubatch) *
                    (cparams.kv_unified ? std::max(1u, cparams.n_seq_max) : 1u);
                sizing.stage_reserve_groups = 1;
                sizing.tail_exact_tokens = cparams.kv_tail_tokens;
                sizing.tail_rollback_tokens = cparams.kv_tail_rollback_tokens;
                sizing.tail_type = cparams.kv_tail_type == GGML_TYPE_COUNT ?
                        GGML_TYPE_F16 : cparams.kv_tail_type;

                std::vector<llama_kv_mixed_layer_cost> costs;
                costs.reserve(n_full_attn);
                const auto layer_params_for = [&](uint32_t il) {
                    llama_kv_mixed_layer_params layer = {};
                    layer.layer = il;
                    layer.head_dim_k = model.hparams.n_embd_head_k(il);
                    layer.head_dim_v = model.hparams.n_embd_head_v(il);
                    layer.n_head_kv = model.hparams.n_head_kv(il);
                    layer.n_embd_k_gqa = model.hparams.n_embd_k_gqa(il);
                    layer.n_embd_v_gqa = model.hparams.n_embd_v_gqa(il);
                    layer.kvarn_bits_k = cparams.kvarn.key_bits;
                    layer.kvarn_bits_v = cparams.kvarn.value_bits;
                    layer.qx_type_k = cparams.remote_attn_cache_type_k;
                    layer.qx_type_v = cparams.remote_attn_cache_type_v;
                    return layer;
                };
                for (uint32_t il = attention_layer_begin; il < attention_layer_end; ++il) {
                    if (!model.hparams.has_kv(il) || model.hparams.is_recr(il)) {
                        continue;
                    }
                    llama_kv_mixed_layer_cost cost = {};
                    char cost_error[256] = {};
                    const auto cost_status = llama_kv_mixed_estimate_layer_cost(
                            layer_params_for(il), sizing, cost, cost_error, sizeof(cost_error));
                    if (cost_status != LLAMA_KV_MIXED_OK) {
                        throw std::runtime_error(format(
                                "mixed KV auto-placement cannot size layer %u: %s",
                                il, cost_error[0] ? cost_error : llama_kv_mixed_status_name(cost_status)));
                    }
                    costs.push_back(cost);
                }
                if (costs.size() != size_t(n_full_attn)) {
                    throw std::runtime_error("mixed KV auto-placement layer geometry count changed during sizing");
                }

                llama_kv_mixed_budget budget = {};
                budget.cuda_free_bytes = cuda_free;
                uint64_t recurrent_reserve = 0;
                if (cparams.offload_rs) {
                    const uint64_t rs_rows = uint64_t(std::max(1u, cparams.n_seq_max)) *
                            (1u + cparams.n_rs_seq);
                    for (uint32_t il = attention_layer_begin; il < attention_layer_end; ++il) {
                        if (!model.hparams.is_recr(il)) {
                            continue;
                        }
                        recurrent_reserve += uint64_t(model.hparams.n_embd_r() + model.hparams.n_embd_s()) *
                                sizeof(float) * rs_rows;
                        if (model.hparams.ple_conv_state() > 0 && model.hparams.is_ple(il)) {
                            recurrent_reserve += uint64_t(model.hparams.ple_conv_state()) * sizeof(float) * rs_rows;
                        }
                    }
                }
                uint64_t mtp_reserve = 0;
                // Planned draft-MTP KV reservation contract (see
                // llama_context_params::mtp_reserve_*). Reserve the independent
                // draft cache with the DRAFT representation (KVarN bits or
                // standard types), the resolved target capacity and the MTP
                // tail policy (init forces tail 0 / F16; KVarN keeps its
                // intrinsic 128-token tail). Inactive by default, so a model
                // that merely has an MTP head does not phantom-reserve
                // target-format KVarN (which could push an extra layer onto
                // the Radeon at 204800).
                if (params.mtp_reserve_enabled) {
                    mtp_reserve = estimate_mtp_reserve_bytes(model, params, cparams);
                }
                budget.cuda_reserve_bytes = params.remote_attn_cuda_reserve;
                if (recurrent_reserve > UINT64_MAX - budget.cuda_reserve_bytes ||
                        mtp_reserve > UINT64_MAX - budget.cuda_reserve_bytes - recurrent_reserve) {
                    throw std::runtime_error("mixed KV CUDA reserve sizing overflow");
                }
                budget.cuda_reserve_bytes += recurrent_reserve + mtp_reserve;
                budget.vulkan_free_bytes = vulkan_free;
                budget.vulkan_reserve_bytes = params.remote_attn_vulkan_reserve;
                budget.ram_cap_bytes = mixed_host_ram_cap_bytes;
                budget.handoff_chunk_tokens = 1024;
                budget.handoff_concurrency = 1;

                llama_kv_mixed_placement placement = {};
                const auto placement_status = llama_kv_mixed_choose_placement(
                        costs, sizing, budget, placement);
                if (placement_status != LLAMA_KV_MIXED_OK) {
                    throw std::runtime_error(format(
                            "mixed KV auto-placement failed: %s",
                            placement.error[0] ? placement.error : llama_kv_mixed_status_name(placement_status)));
                }
                n_remote = int(placement.n_remote);
                LLAMA_LOG_INFO("%s: mixed KV auto-placement: CUDA owner=%s CUDA=%0.1f/%0.1f MiB reserve=%0.1f MiB "
                               "(RS=%0.1f, MTP=%0.1f), Vulkan=%0.1f/%0.1f MiB reserve=%0.1f MiB, "
                               "host handoff=%0.1f MiB "
                               "chunk=%llu tokens -> Vulkan Qx layers=%d, CUDA KVarN layers=%d\n",
                        __func__, ggml_backend_dev_name(cuda_dev),
                        placement.cuda_used_bytes / 1024.0 / 1024.0,
                        placement.cuda_available_bytes / 1024.0 / 1024.0,
                        budget.cuda_reserve_bytes / 1024.0 / 1024.0,
                        recurrent_reserve / 1024.0 / 1024.0,
                        mtp_reserve / 1024.0 / 1024.0,
                        placement.vulkan_used_bytes / 1024.0 / 1024.0,
                        placement.vulkan_available_bytes / 1024.0 / 1024.0,
                        budget.vulkan_reserve_bytes / 1024.0 / 1024.0,
                        placement.ram_temp_bytes / 1024.0 / 1024.0,
                        (unsigned long long) placement.effective_chunk_tokens,
                        n_remote, n_full_attn - n_remote);
            } else if (params.remote_attn_n_layers == -1 && cparams.kv_position_split_p == 0) {
                size_t cuda_free = 0, cuda_total = 0;
                ggml_backend_dev_t cuda_dev = ggml_backend_dev_by_name("CUDA0");
                if (!cuda_dev) cuda_dev = ggml_backend_dev_by_name("CUDA");
                if (cuda_dev) {
                    ggml_backend_dev_memory(cuda_dev, &cuda_free, &cuda_total);
                }
                const size_t reserve = params.remote_attn_cuda_reserve > 0 ?
                    params.remote_attn_cuda_reserve : (600 * 1024 * 1024);

                // Account for recurrent state (RS cache) that will be allocated on CUDA for hybrid architectures
                size_t recr_bytes = 0;
                const uint32_t n_rs_rows = std::max((uint32_t) 1, cparams.n_seq_max) * (1 + cparams.n_rs_seq);
                for (uint32_t il = attention_layer_begin; il < attention_layer_end; ++il) {
                    if (model.hparams.is_recr(il) && cparams.offload_rs) {
                        recr_bytes += (model.hparams.n_embd_r() + model.hparams.n_embd_s()) * sizeof(float) * n_rs_rows;
                        if (model.hparams.ple_conv_state() > 0 && model.hparams.is_ple(il)) {
                            recr_bytes += model.hparams.ple_conv_state() * sizeof(float) * n_rs_rows;
                        }
                    }
                }

                const size_t total_deduction = reserve + recr_bytes;
                const size_t usable_cuda_kv = (cuda_free > total_deduction) ? (cuda_free - total_deduction) : 0;
                std::vector<size_t> full_attn_layer_bytes;
                full_attn_layer_bytes.reserve(n_full_attn);
                size_t max_layer_bytes = 0;
                const bool use_kvarn = cparams.kvarn.type != LLAMA_KVARN_TYPE_DISABLED;
                const size_t n_streams = cparams.kv_unified ? 1u :
                    std::max((uint32_t) 1, cparams.n_seq_max);
                const uint32_t record_groups =
                    (cparams.n_ctx_seq + KVAR_N_GROUP - 1) / KVAR_N_GROUP;
                const uint32_t stage_tail_groups = use_kvarn ?
                    llama_kvarn_non_swa_tail_groups(cparams.n_batch, cparams.n_ubatch) *
                        (cparams.kv_unified ? std::max((uint32_t) 1, cparams.n_seq_max) : 1u) : 0u;
                const uint32_t stage_groups = stage_tail_groups + (use_kvarn ? 1u : 0u);
                const size_t bits_k = cparams.kvarn.key_bits ? cparams.kvarn.key_bits : 4;
                const size_t bits_v = cparams.kvarn.value_bits ? cparams.kvarn.value_bits : 4;
                const ggml_type exact_tail_type = cparams.kv_tail_type != GGML_TYPE_COUNT ?
                    cparams.kv_tail_type : (use_kvarn ? GGML_TYPE_F16 : GGML_TYPE_BF16);

                for (uint32_t il = attention_layer_begin; il < attention_layer_end; ++il) {
                    if (!model.hparams.has_kv(il) || model.hparams.is_recr(il)) {
                        continue;
                    }
                    const size_t n_head_kv = model.hparams.n_head_kv(il);
                    const size_t k_dim = model.hparams.n_embd_k_gqa(il);
                    const size_t v_dim = model.hparams.n_embd_v_gqa(il);
                    size_t layer_bytes = 0;
                    if (use_kvarn) {
                        llama_kvarn_geometry k_geometry = {};
                        llama_kvarn_geometry v_geometry = {};
                        if (!llama_kvarn_geometry_for(model.hparams.n_embd_head_k(il), k_geometry) ||
                                !llama_kvarn_geometry_for(model.hparams.n_embd_head_v(il), v_geometry)) {
                            throw std::runtime_error(format(
                                "auto-placement cannot size KVarN layer %u", il));
                        }
                        const size_t k_records_per_group =
                            llama_kvarn_make_record_layout(k_geometry.record_dim, bits_k, false).record_bytes *
                            n_head_kv * k_geometry.head_slices;
                        const size_t v_records_per_group =
                            llama_kvarn_make_record_layout(v_geometry.record_dim, bits_v, true).record_bytes *
                            n_head_kv * v_geometry.head_slices;
                        layer_bytes += (k_records_per_group + v_records_per_group) *
                            record_groups * n_streams;
                        layer_bytes += (k_dim + v_dim) * sizeof(ggml_fp16_t) *
                            KVAR_N_GROUP * stage_groups * n_streams;
                    } else {
                        layer_bytes += ggml_row_size(params.type_k, k_dim) * cparams.n_ctx_seq * n_streams;
                        layer_bytes += ggml_row_size(params.type_v, v_dim) * cparams.n_ctx_seq * n_streams;
                    }
                    if (cparams.kv_tail_tokens > 0) {
                        const size_t tail_k_row = ggml_row_size(exact_tail_type, k_dim);
                        const size_t tail_v_row = ggml_row_size(exact_tail_type, v_dim);
                        layer_bytes += (tail_k_row + tail_v_row) * cparams.kv_tail_tokens * n_streams;
                    }
                    full_attn_layer_bytes.push_back(layer_bytes);
                    max_layer_bytes = std::max(max_layer_bytes, layer_bytes);
                }

                // Placement selects a suffix of layers for CUDA (the first N
                // full-attention layers are the remote prefix). Fit exact layer
                // sizes from the end so variable GQA/head geometries are safe.
                int n_local_fit = 0;
                size_t local_bytes = 0;
                for (auto it = full_attn_layer_bytes.rbegin(); it != full_attn_layer_bytes.rend(); ++it) {
                    if (*it > usable_cuda_kv - std::min(usable_cuda_kv, local_bytes)) {
                        break;
                    }
                    local_bytes += *it;
                    ++n_local_fit;
                }
                n_local_fit = std::clamp(n_local_fit, 0, n_full_attn);

                n_remote = n_full_attn - n_local_fit;
                if (cparams.local_attn_migration) {
                    const size_t full_prefill_kv = std::accumulate(
                            full_attn_layer_bytes.begin(), full_attn_layer_bytes.end(), size_t(0));
                    if (full_prefill_kv > usable_cuda_kv) {
                        // CUDA prefill needs the complete target cache resident.
                        // If it cannot fit within the safety budget, preserve the
                        // ordinary auto-placement route rather than failing an
                        // adaptive profile or attempting a partial migration.
                        cparams.local_attn_migration = false;
                        cparams.local_attn_migration_fallback = true;
                        cparams.local_attn_backend = cparams.local_attn_migration_backend;
                        LLAMA_LOG_WARN("%s: automatic prefill migration does not fit CUDA (need %.1f MiB, budget %.1f MiB); "
                                       "falling back to static remote attention with %d layer(s)\n",
                                __func__, full_prefill_kv / 1024.0 / 1024.0,
                                usable_cuda_kv / 1024.0 / 1024.0, n_remote);
                    } else {
                        // The existing auto policy returns zero remote layers when
                        // the complete cache fits. Migration mode uses one layer
                        // as its decode destination while keeping prefill local.
                        n_remote = n_full_attn > 0 ? 1 : 0;
                        LLAMA_LOG_INFO("%s: migration auto-placement keeps full target KV on CUDA (%.1f/%.1f MiB) "
                                       "and selects %d Vulkan decode layer(s)\n",
                                __func__, full_prefill_kv / 1024.0 / 1024.0,
                                usable_cuda_kv / 1024.0 / 1024.0, n_remote);
                    }
                }
                LLAMA_LOG_INFO("%s: auto-placement: CUDA free=%.1f MiB, reserve=%.1f MiB, rs_cache=%.1f MiB, usable=%.1f MiB, "
                               "layer_kv_max=%.1f MiB (records+stage+tail), local_kv=%.1f MiB -> %d local full-attn layers on 4070, %d offloaded to %s\n",
                               __func__, cuda_free / 1024.0 / 1024.0, reserve / 1024.0 / 1024.0,
                               recr_bytes / 1024.0 / 1024.0, usable_cuda_kv / 1024.0 / 1024.0,
                               max_layer_bytes / 1024.0 / 1024.0,
                               local_bytes / 1024.0 / 1024.0,
                               n_local_fit, n_remote, params.remote_attn_host);
            } else if (params.remote_attn_n_layers > 0 && params.remote_attn_n_layers <= n_full_attn) {
                n_remote = params.remote_attn_n_layers;
            }
            if (cparams.kv_position_split_p != 0 && params.remote_attn_n_layers == -1) {
                LLAMA_LOG_INFO("%s: position-split KV forces all %d full-attention layers onto the remote "
                               "backend (auto resolves like \"full\")\n",
                        __func__, n_full_attn);
                n_remote = n_full_attn;
            }
            cparams.remote_attn_layers = n_remote;
            if (remote_type_k) {
                int full_idx = 0;
                for (uint32_t il = attention_layer_begin; il < attention_layer_end; ++il) {
                    if (!model.hparams.has_kv(il) || model.hparams.is_recr(il)) {
                        continue;
                    }
                    if (full_idx >= n_remote) {
                        auto * local_dev = model.dev_layer(il);
                        const char * local_name = local_dev ? ggml_backend_dev_name(local_dev) : nullptr;
                        if (!local_name || ggml_backend_dev_type(local_dev) != GGML_BACKEND_DEVICE_TYPE_GPU ||
                                std::strncmp(local_name, "CUDA", 4) != 0) {
                            throw std::invalid_argument(format(
                                    "mixed KV layer %u remains KVarN but its local model layer is not on CUDA",
                                    il));
                        }
                    }
                    ++full_idx;
                }
            }
            if (local_attn_dev != nullptr) {
                LLAMA_LOG_INFO("%s: local attention placement: context=%s Vulkan-layers=%d/%d mode=%s\n",
                        __func__, cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP ? "MTP" : "target",
                        n_remote, n_full_attn,
                        cparams.local_attn_migration ? "CUDA-prefill/mirrored" : "static-split");
            }
            if (cparams.local_attn_migration && n_remote <= 0) {
                throw std::invalid_argument(
                    "--remote-attn-prefill=migrate requires at least one full-attention layer for Vulkan decode");
            }
        }

        memory.reset(model.create_memory(params_mem, cparams, layer_backends, kv_backends, backend_cpu));
        if (memory) {
            const ggml_type actual_tail_type = memory->get_kv_tail_type();
            if ((cparams.kv_tail_tokens > 0 || cparams.kv_tail_tokens_swa > 0) &&
                    actual_tail_type == GGML_TYPE_COUNT) {
                throw std::runtime_error("KV tail cache did not report its resolved storage type");
            }
            if (actual_tail_type != GGML_TYPE_COUNT) {
                cparams.kv_tail_type = actual_tail_type;
            }
        } else if (cparams.kv_tail_tokens > 0 || cparams.kv_tail_tokens_swa > 0) {
            throw std::runtime_error("KV tail cache requested for a model without cache memory");
        }

        // LLAMA_MTP_POSITION_SPLIT=1 fit gate: the mixed auto-placement planner
        // that normally prices the draft-MTP reservation is bypassed when P != 0
        // (the split keeps both bands on every full layer), so the independent
        // draft context would otherwise discover any shortfall as a raw CUDA
        // OOM in graph_reserve below (observed: target pp buffers). Fail closed
        // here, with real numbers, BEFORE the compute buffers: the measured
        // free CUDA memory must still hold the draft KV reservation plus the
        // remaining target pp compute buffers and the draft context's
        // graph/output overhead. The draft context borrows the target model's
        // weights, so only its cache and compute buffers are new. This runs
        // after the target KV/RS allocation, so those costs are already out.
        if (llama_position_split::mtp_position_split_env_enabled(getenv("LLAMA_MTP_POSITION_SPLIT")) &&
                cparams.kv_position_split_p != 0 && params.mtp_reserve_enabled) {
            ggml_backend_dev_t cuda_dev = nullptr;
            for (uint32_t il = 0; il < model.hparams.n_layer(); ++il) {
                if (!model.hparams.has_kv(il) || model.hparams.is_recr(il)) {
                    continue;
                }
                auto * dev = model.dev_layer(int32_t(il));
                const char * name = dev ? ggml_backend_dev_name(dev) : nullptr;
                if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU ||
                        !name || std::strncmp(name, "CUDA", 4) != 0) {
                    throw std::runtime_error(format(
                            "LLAMA_MTP_POSITION_SPLIT=1: full-attention layer %u is not on CUDA (%s); "
                            "the independent draft-MTP cache cannot be budgeted",
                            il, name ? name : "unknown device"));
                }
                cuda_dev = dev;
                break;
            }
            if (cuda_dev == nullptr) {
                throw std::runtime_error(
                        "LLAMA_MTP_POSITION_SPLIT=1: no CUDA full-attention layer found for the draft-MTP cache");
            }
            // The draft KV cache is allocated on the MTP layers' model devices,
            // not on the target full-attention device. This gate only budgets
            // the single-CUDA-owner case: require every MTP KV layer to sit on
            // exactly the budget device and fail closed otherwise (a
            // cross-device draft is not validated here).
            for (uint32_t il = model.hparams.n_layer(); il < model.hparams.n_layer_all; ++il) {
                if (!model.hparams.has_kv(il) || model.hparams.is_recr(il)) {
                    continue;
                }
                auto * dev = model.dev_layer(int32_t(il));
                const char * name = dev ? ggml_backend_dev_name(dev) : nullptr;
                if (dev != cuda_dev) {
                    throw std::runtime_error(format(
                            "LLAMA_MTP_POSITION_SPLIT=1: MTP KV layer %u is on %s, but the draft-MTP budget "
                            "was computed for the target CUDA device %s; cross-device draft placement is "
                            "not validated [fail-closed]",
                            il, name ? name : "unknown device", ggml_backend_dev_name(cuda_dev)));
                }
            }
            const uint64_t mtp_reserve = estimate_mtp_reserve_bytes(model, params, cparams);
            size_t cuda_free = 0, cuda_total = 0;
            ggml_backend_dev_memory(cuda_dev, &cuda_free, &cuda_total);
            // Overhead beyond the draft KV: the target's remaining pp compute
            // buffers (61 MiB reserved, 126 MiB measured at 256-token ubatches
            // on the 27B/4070) plus the one-layer MTP draft compute graph and
            // output staging. The 2026-10-17 sweep showed configurations that
            // passed a 256 MiB overhead and then died on a runtime compute
            // re-reserve (86 MiB at ub=256, 28 MiB at ub=64) before prefill
            // completed, so account for the measured peak growth (256 + 86)
            // with a small margin instead of accepting ~0 MiB of headroom.
            constexpr uint64_t kMtpSplitOverheadBytes = 352ull << 20;
            const uint64_t needed = mtp_reserve + kMtpSplitOverheadBytes;
            LLAMA_LOG_INFO("%s: LLAMA_MTP_POSITION_SPLIT=1 fit check: draft-MTP reserve=%.1f MiB, "
                           "overhead=%.1f MiB, CUDA free=%.1f MiB\n",
                           __func__, mtp_reserve/1048576.0, kMtpSplitOverheadBytes/1048576.0,
                           double(cuda_free)/1048576.0);
            if (uint64_t(cuda_free) < needed) {
                throw std::runtime_error(format(
                        "LLAMA_MTP_POSITION_SPLIT=1: the draft-MTP cache does not fit CUDA at P=%u "
                        "(C=%u): draft reserve %.1f MiB + overhead %.1f MiB = %.1f MiB needed, "
                        "only %.1f MiB free after the position-split target placement [fail-closed]",
                        cparams.kv_position_split_p, cparams.n_ctx_seq,
                        mtp_reserve/1048576.0, kMtpSplitOverheadBytes/1048576.0, needed/1048576.0,
                        double(cuda_free)/1048576.0));
            }
        }

    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();
        backend_kvarn_workspace_y_size.clear();
        backend_kvarn_workspace_split_k_size.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                const auto & dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev.dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
            backend_kvarn_workspace_y_size.push_back(0);
            backend_kvarn_workspace_split_k_size.push_back(0);
        }

        // Remote KV+attention accelerator (Xbox RKVA). Created after the local
        // backends so it joins backend_ptrs/backend_buft for the scheduler, but
        // it only ever claims GGML_OP_REMOTE_ATTN (which we also pin explicitly).
        const bool is_local_vulkan = (params.remote_attn_host != nullptr &&
            (std::strncmp(params.remote_attn_host, "vulkan", 6) == 0 ||
             std::strcmp(params.remote_attn_host, "local") == 0));

        if (cparams.remote_attn_enabled && (!cparams.local_attn_backend || !is_local_vulkan)) {
            if (model.arch != LLM_ARCH_QWEN35) {
                throw std::runtime_error(format(
                    "%s: --remote-attn currently supports only the qwen35 arch (this model arch id: %d)",
                    __func__, (int) model.arch));
            }
            if (cparams.kvarn.type == LLAMA_KVARN_TYPE_DISABLED &&
                    params.type_k != GGML_TYPE_F16) {
                throw std::runtime_error(format(
                    "%s: --remote-attn requires a KVarN cache type (--cache-type-k/-v kvarnN) "
                    "or the F16 bring-up path (--cache-type-k/-v f16)", __func__));
            }

            const int n_remote = cparams.remote_attn_layers;

            ggml_remote_attn_geometry geo {};
            geo.n_layer_remote = (uint32_t) n_remote;
            geo.n_head         = model.hparams.n_head();
            geo.n_head_kv      = model.hparams.n_head_kv();
            geo.head_dim       = model.hparams.n_embd_head_k();
            geo.max_ctx        = (uint32_t) cparams.n_ctx;
            geo.cache_bits_k   = (uint32_t) cparams.kvarn.key_bits;
            geo.cache_bits_v   = (uint32_t) cparams.kvarn.value_bits;
            geo.group_tokens   = 128;
            geo.sinkhorn_iters = (uint32_t) cparams.kvarn.sinkhorn_iters;
            geo.tail_tokens    = cparams.kv_tail_tokens;
            geo.tail_groups    = 2;
            geo.tail_type      = (cparams.kv_tail_type == GGML_TYPE_BF16) ? 1u : 0u;
            geo.domain         = GGML_REMOTE_ATTN_DOMAIN_AUTO;  // server mirrors the KVarN rotated-domain plan
            geo.has_sinks      = 0;
            geo.swa            = 0;
            geo.kq_scale       = model.hparams.f_attention_scale == 0.0f ?
                                 1.0f / sqrtf((float) geo.head_dim) : model.hparams.f_attention_scale;
            if (tcp_position_split) {
                const char * storage = std::getenv("GGML_RKVA_UPLOAD_FORMAT");
                if (!storage) storage = std::getenv("GGML_RKVA_FORMAT");
                if (!storage) storage = std::getenv("GGML_RKVA_RANGE_STORAGE");
                if (storage && (std::strcmp(storage, "kvarn4") == 0 || std::strcmp(storage, "kvarn") == 0) &&
                        (cparams.kv_tail_tokens > 128 || cparams.kv_tail_type != GGML_TYPE_F16)) {
                    throw std::runtime_error("experimental native Xbox KVarN4 requires an F16 precision tail of at most 128 tokens");
                }
                geo.max_ctx = cparams.n_ctx_seq - cparams.kv_position_split_p;
                // The CPU companion encodes Q4_0, Q8_0 or F16 once. The Xbox
                // receives matching native bytes (or an explicit quantized-source
                // F16 expansion), without WHT.
                geo.cache_bits_k = geo.cache_bits_v = 0;
                geo.tail_tokens = geo.tail_groups = 0;
            }

            const bool is_local_vulkan = (params.remote_attn_host != nullptr &&
                (std::strncmp(params.remote_attn_host, "vulkan", 6) == 0 ||
                 std::strcmp(params.remote_attn_host, "local") == 0));

            if (is_local_vulkan) {
                uint32_t vulkan_dev = 0;
                if (std::strncmp(params.remote_attn_host, "vulkan:", 7) == 0) {
                    vulkan_dev = (uint32_t) std::atoi(params.remote_attn_host + 7);
                }
                backend_remote = ggml_backend_local_split_init(vulkan_dev, 3);
                if (backend_remote == nullptr) {
                    throw std::runtime_error(format(
                        "%s: failed to create local-split backend for %s",
                        __func__, params.remote_attn_host));
                }
                ggml_backend_local_split_set_geometry(backend_remote, &geo);
                if (!ggml_backend_local_split_setup(backend_remote)) {
                    ggml_backend_free(backend_remote);
                    backend_remote = nullptr;
                    throw std::runtime_error(format(
                        "%s: failed to setup local-split Vulkan pipelines for %s",
                        __func__, params.remote_attn_host));
                }
                remote_attn_session = 0;
                if (!ggml_backend_local_split_create_session(
                        backend_remote, remote_attn_session, /*seq_id=*/0, (uint32_t) cparams.n_ctx)) {
                    ggml_backend_free(backend_remote);
                    backend_remote = nullptr;
                    throw std::runtime_error(format("%s: local-split CREATE_SESSION failed", __func__));
                }
                ggml_backend_local_split_set_active_session(backend_remote, remote_attn_session);
                LLAMA_LOG_INFO("%s: local-split attention enabled — %d full-attn layers offloaded to %s\n",
                        __func__, n_remote, params.remote_attn_host);
            } else {
                backend_remote = ggml_backend_remote_attn_init(params.remote_attn_host, params.remote_attn_port);
                if (backend_remote == nullptr) {
                    throw std::runtime_error(format(
                        "%s: failed to create remote-attn backend for %s:%u",
                        __func__, params.remote_attn_host, (unsigned) params.remote_attn_port));
                }
                ggml_backend_remote_attn_set_geometry(backend_remote, &geo);
                if (!ggml_backend_remote_attn_connect(backend_remote)) {
                    ggml_backend_free(backend_remote);
                    backend_remote = nullptr;
                    if (tcp_position_split) {
                        throw std::runtime_error("TCP position-split requires a reachable RKVA range server; refusing Vulkan fallback");
                    }
                    LLAMA_LOG_WARN("%s: remote-attn server %s:%u unreachable, falling back to local vulkan:0\n",
                        __func__, params.remote_attn_host, (unsigned) params.remote_attn_port);
                    backend_remote = ggml_backend_local_split_init(0, 3);
                    if (backend_remote) {
                        ggml_backend_local_split_set_geometry(backend_remote, &geo);
                        if (ggml_backend_local_split_setup(backend_remote)) {
                            remote_attn_session = 0;
                            if (ggml_backend_local_split_create_session(backend_remote, remote_attn_session, 0, (uint32_t) cparams.n_ctx)) {
                                ggml_backend_local_split_set_active_session(backend_remote, remote_attn_session);
                                LLAMA_LOG_INFO("%s: fallback local-split attention enabled on vulkan:0\n", __func__);
                            }
                        }
                    }
                    if (!backend_remote) {
                        throw std::runtime_error(format(
                            "%s: failed to connect to %s:%u and fallback to vulkan:0 also failed",
                            __func__, params.remote_attn_host, (unsigned) params.remote_attn_port));
                    }
                } else {
                    remote_attn_session = 0;
                    const bool created = tcp_position_split ?
                            ggml_backend_remote_attn_create_range_session(
                                backend_remote, remote_attn_session, 0, geo.max_ctx) :
                            ggml_backend_remote_attn_create_session(
                                backend_remote, remote_attn_session, 0, geo.max_ctx);
                    if (!created) {
                        ggml_backend_free(backend_remote);
                        backend_remote = nullptr;
                        throw std::runtime_error(format("%s: remote-attn CREATE_SESSION failed", __func__));
                    }
                    ggml_backend_remote_attn_set_active_session(backend_remote, remote_attn_session);
                    if (tcp_position_split && !ggml_backend_remote_attn_configure_range(
                            backend_remote, cparams.kv_position_split_p, geo.max_ctx,
                            cparams.remote_attn_cache_type_k)) {
                        ggml_backend_free(backend_remote);
                        backend_remote = nullptr;
                        throw std::runtime_error("RKVA server does not support positional OUT/LSE ranges");
                    }
                    if (tcp_position_split) {
                        LLAMA_LOG_INFO("%s: RKVA position split: CUDA [0,%u), Xbox [%u,%u), %d layers; no ATTN RPC below P\n",
                                __func__, cparams.kv_position_split_p, cparams.kv_position_split_p,
                                cparams.n_ctx_seq, n_remote);
                    } else {
                        LLAMA_LOG_INFO("%s: remote attention enabled — %d full-attn layers offloaded to %s:%u\n",
                                __func__, n_remote, params.remote_attn_host, (unsigned) params.remote_attn_port);
                    }
                }
            }
            // PATH B: the remote op is computed on the CPU backend (pinned in
            // qwen35), so the scheduler uses its proven CUDA<->CPU copies and we
            // do NOT register backend_remote with the scheduler (a minimal custom
            // backend in the split/copy machinery corrupted tensor backend ids).
            // backend_remote only owns the RPC connection; ggml_remote_attn_exec
            // (called from the CPU compute) uses it via this active registration.
            // Publish negotiated resident-prefix capability to the real memory
            // owner; the Q4 companion remains bookkeeping, not native payload.
            if (remote_native_kvarn_enabled() && memory) {
                llama_kv_cache_kvarn * native_cache = dynamic_cast<llama_kv_cache_kvarn *>(memory.get());
                if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory.get())) {
                    native_cache = dynamic_cast<llama_kv_cache_kvarn *>(hybrid->get_mem_attn_base());
                }
                if (!native_cache) throw std::runtime_error("native RPC owner is not a KVarN memory cache");
                native_cache->set_native_kvarn_rewind_capable(remote_native_kvarn_can_rewind());
                native_cache->set_native_kvarn_rewind_preflight([this](llama_pos p0) {
                    return backend_remote && ggml_backend_remote_attn_native_kvarn_can_restore(backend_remote, p0);
                });
            }
            // Reject before the first speculative write unless the live
            // server negotiated bounded resident rollback. FULL persistence
            // remains unsupported independently of this capability.
            if (params.mtp_reserve_enabled && remote_native_kvarn_enabled() &&
                    (!remote_native_kvarn_can_rewind() || params.mtp_reserve_rollback_tokens > 128)) {
                ggml_backend_free(backend_remote);
                backend_remote = nullptr;
                throw std::runtime_error("draft-MTP requires negotiated native Xbox KVarN4 bounded rewind (maximum 128 tokens)");
            }
            ggml_remote_attn_set_active(backend_remote);

            // RESET-on-clear: a full local cache clear (warmup, /v1/chat/completions
            // reset, slot reuse) must reset the remote session so both sides stay in
            // lockstep; otherwise the next generation starts over stale records.
            if (memory) {
                if (cparams.remote_attn_position_split) {
                    memory->set_on_seq_rm([this](llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
                        if (!backend_remote) return true;
                        if (remote_native_kvarn_enabled() && p0 <= 0 && p1 < 0) {
                            return ggml_backend_remote_attn_reset(backend_remote, remote_attn_session);
                        }
                        if (seq_id != 0 || p0 < 0 || p1 >= 0) {
                            ggml_backend_remote_attn_invalidate_range(backend_remote);
                            return true;
                        }
                        return ggml_backend_remote_attn_trim(backend_remote, remote_attn_session, p0);
                    });
                }
                memory->set_on_clear([this]() {
                    // Void callback: a refused remote reset poisons the backend
                    // (stays failed, no implicit recovery); the next decode
                    // fails closed instead of assuming this clear succeeded.
                    if (backend_remote != nullptr) {
                        if (ggml_backend_is_local_split(backend_remote)) {
                            ggml_backend_local_split_reset(backend_remote, remote_attn_session);
                        } else {
                            ggml_backend_remote_attn_reset(backend_remote, remote_attn_session);
                        }
                    }
                });
            }
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer_all &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides() &&
            cparams.moe_cache_size == 0; // not supported by the MoE cache

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }

        if (cparams.moe_cache_size > 0) {
            moe_cache = std::make_unique<llama_moe_cache>(model, backend_ptrs, backend_buft, cparams.moe_cache_size);
        }

        sched_reserve();

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}

llama_context::~llama_context() {
    for (const auto & entry : wavefront_decisions) {
        LLAMA_LOG_INFO("RKVA wavefront summary: reason=%s parents=%llu\n", entry.first.c_str(),
                (unsigned long long) entry.second);
    }

    // wait for any pending asynchronous copies into the output buffers before they are freed
    synchronize();
    // Fused position-split span: release THIS context's executor bundle (its
    // per-layer Vulkan/CUDA sub-graphs reference this context's backends and
    // caches) while the backends are still alive. The bundle is owned by the
    // KVarN cache; other contexts' bundles are untouched.
    if (cparams.kv_position_split_p != 0) {
        llama_memory_i * mem = get_memory();
        if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) {
            mem = hybrid->get_mem_attn_base();
        }
        if (auto * kvarn = dynamic_cast<llama_kv_cache_kvarn *>(mem)) {
            kvarn->release_fused_span_bundle();
        }
    }
    // KVarN's transfer worker can hold CUDA producer events and live tensor
    // pointers. Drain it before scheduler/model backends begin destruction.
    if (memory && !memory->drain_prefill_migration()) {
        LLAMA_LOG_WARN("%s: KVarN mirror worker stopped with a stale mirror; active cache remains authoritative\n", __func__);
    }

    // when training, ggml_opt allocates extra buffers through the scheduler, so the sizes no longer match the expectation
    if (!model.hparams.no_alloc && !opt_ctx) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }
    ggml_opt_free(opt_ctx);

    // Release the scheduler before the backends it references.
    if (backend_remote != nullptr) {
        // An aborted graph may have submitted a range without reaching its wait.
        ggml_backend_remote_attn_drain(backend_remote);
        if (cparams.remote_attn_stats) {
            const char * stats = ggml_backend_is_local_split(backend_remote) ?
                ggml_backend_local_split_stats_json(backend_remote) :
                ggml_backend_remote_attn_stats_json(backend_remote);
            LLAMA_LOG_INFO("%s: remote_attn stats %s\n", __func__, stats);
        }
        ggml_remote_attn_set_active(nullptr);
    }
    sched.reset();
    if (backend_remote != nullptr) {
        ggml_backend_free(backend_remote);
        backend_remote = nullptr;
    }
}

void llama_context::resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs) {
    const char * func = __func__;
    auto resolve = [&](const llm_fused_op_probe & probe, bool & enabled) {
        if (!enabled) {
            return;
        }

        const uint32_t n_tokens_probe = probe.n_tokens_per_seq*n_seqs;

        auto * gf = graph_reserve(n_tokens_probe, n_seqs, n_tokens_probe, mctx, true);
        if (!gf) {
            throw std::runtime_error(std::string("failed to reserve graph for ") + probe.name + " check");
        }

        bool device_mismatch = false;
        for (const auto & node : get_gf_res_reserve()->get_fused_nodes()) {
            if (node.op != probe.op) {
                continue;
            }

            GGML_ASSERT(node.il >= 0);

            ggml_backend_t backend_fused = ggml_backend_sched_get_tensor_backend(sched.get(), node.tensor);
            ggml_backend_dev_t device_fused = backend_fused ? ggml_backend_get_device(backend_fused) : nullptr;

            // TODO: make this descriptor-specific; model.dev_layer() preserves the current behavior,
            // but is still wrong for cases like --no-kv-offload.
            ggml_backend_dev_t device_layer = model.dev_layer(node.il);
            if (probe.op == LLM_FUSED_OP_FLASH_ATTN && cparams.local_attn_backend &&
                    selected_attention_layer(model.hparams, node.il, cparams.remote_attn_layers, cparams.ctx_type)) {
                device_layer = ggml_backend_get_device(cparams.local_attn_backend);
            }

            if (device_fused != device_layer) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but %s "
                        "is assigned to device %s (usually due to missing support)\n",
                        func, node.il,
                        device_layer ? ggml_backend_dev_name(device_layer) : "none",
                        probe.name,
                        device_fused ? ggml_backend_dev_name(device_fused) : "none");
                device_mismatch = true;
                break;
            }
        }

        if (device_mismatch) {
            enabled = false;
            LLAMA_LOG_WARN("%s: %s not supported, set to disabled\n", func, probe.name);
        } else {
            enabled = true;
            LLAMA_LOG_INFO("%s: %s enabled\n", func, probe.name);
        }
    };

    if (cparams.auto_fa) {
        resolve(llm_fused_op_flash_attn_probe, cparams.flash_attn);
        cparams.auto_fa = false;
    }

    if (cparams.auto_fgdn) {
        LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", func);
        resolve(llm_fused_op_gdn_ar_probe, cparams.fused_gdn_ar);
        resolve(llm_fused_op_gdn_ch_probe, cparams.fused_gdn_ch);
        cparams.auto_fgdn = false;
    }

    if (cparams.auto_flid) {
        LLAMA_LOG_INFO("%s: resolving fused Lightning Indexer support:\n", func);
        resolve(llm_fused_op_lid_probe, cparams.fused_lid);
        cparams.auto_flid = false;
    }

    if (cparams.auto_fhc) {
        LLAMA_LOG_INFO("%s: resolving fused DeepSeek V4 HC support:\n", func);
        resolve(llm_fused_op_dsv4_hc_pre_probe,  cparams.fused_dsv4_hc_pre);
        resolve(llm_fused_op_dsv4_hc_comb_probe, cparams.fused_dsv4_hc_comb);
        resolve(llm_fused_op_dsv4_hc_post_probe, cparams.fused_dsv4_hc_post);
        cparams.auto_fhc = false;
    }
}

static int llama_graph_n_input_tensors(ggml_cgraph * gf) {
    std::unordered_map<const ggml_tensor *, std::vector<ggml_tensor *>> users;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->flags & GGML_TENSOR_FLAG_INPUT) {
            users[node].push_back(node);
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor * src = node->src[j];
            if (!src) {
                break;
            }
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                users[src].push_back(node);
            }
        }
    }

    for (const auto & [tensor, nodes] : users) {
        GGML_ASSERT(tensor->op == GGML_OP_NONE);
        for (const ggml_tensor * node : nodes) {
            LLAMA_LOG_DEBUG("%s: input tensor '%32s' [%s, ne = { %5" PRId64 ", %5" PRId64 ", %5" PRId64 ", %5" PRId64 " }] is used by node '%s' (%s)\n",
                    __func__, tensor->name, ggml_type_name(tensor->type),
                    tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                    node->name, ggml_op_name(node->op));
        }
    }

    return (int) users.size();
}

void llama_context::sched_new(size_t max_nodes, bool parallel) {
    sched.reset(ggml_backend_sched_new(
            backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, parallel, cparams.op_offload));

    ggml_backend_sched_set_copy_callback(sched.get(), sched_copy_experts, this);
    ggml_backend_sched_set_realloc_callback(sched.get(), sched_graph_realloc, this);
}

// WF256 executes ordinary prefix/fallback graphs at physical N256. Reserving
// a dummy serial parent-width graph here increases their live workspace before any
// eligible wavefront parent exists. Keep the node/capacity budget at the public
// parent size; the scheduler can grow for an actual tiled graph on demand.
// No-allocation fit estimates remain conservative at the requested parent size.
static uint32_t rkva_physical_ubatch(const llama_cparams & cp) {
    return (cp.rkva_wavefront_tile == 256 || cp.rkva_wavefront_tile == 512) ? std::min(cp.n_ubatch, 256u) : cp.n_ubatch;
}

static uint32_t rkva_initial_reserve_tokens(const llama_cparams & cp, bool no_alloc) {
    return std::min(cp.n_ctx, no_alloc ? cp.n_ubatch : rkva_physical_ubatch(cp));
}

void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    synchronize();

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs = cparams.n_seq_max;
    const uint32_t capacity_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);
    const uint32_t n_tokens = rkva_initial_reserve_tokens(cparams, model.hparams.no_alloc);

    const size_t max_nodes = this->graph_max_nodes(capacity_tokens);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    for (auto & res : gf_res_prev) {
        res.reset();
    }
    gf_res_reserve.reset(new llm_graph_result(max_nodes));
    std::fill(backend_kvarn_workspace_y_size.begin(),
            backend_kvarn_workspace_y_size.end(), 0);
    std::fill(backend_kvarn_workspace_split_k_size.begin(),
            backend_kvarn_workspace_split_k_size.end(), 0);
    gf_res_prev_active = nullptr;

    sched_new(max_nodes, cparams.pipeline_parallel);

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

    resolve_fused_ops(mctx.get(), n_seqs);

    // reserve worst-case graph
    int n_splits_pp        = -1;
    int n_nodes_pp         = -1;
    int n_inputs_pp        = -1;
    int n_input_tensors_pp = -1;

    int n_splits_tg        = -1;
    int n_nodes_tg         = -1;
    int n_inputs_tg        = -1;
    int n_input_tensors_tg = -1;

    const uint32_t n_outputs_pp = std::min(n_tokens, cparams.n_outputs_max);

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(),
                model.hparams.no_alloc, model.hparams.no_alloc ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                cparams.pipeline_parallel = false;
                sched_new(max_nodes, false);
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get());
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }
        record_backend_private_workspace(gf);

        n_splits_pp        = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp         = ggml_graph_n_nodes(gf);
        n_inputs_pp        = get_gf_res_reserve()->inputs.size();
        n_input_tensors_pp = this->n_input_tensors;

        graph_has_logits = get_gf_res_reserve()->get_logits() != nullptr;
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    if (cparams.training) {
        // no tg graph for training
        n_splits_tg = n_splits_pp;
        n_nodes_tg  = n_nodes_pp;
    } else {
        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }
        record_backend_private_workspace(gf);

        n_splits_tg        = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg         = ggml_graph_n_nodes(gf);
        n_inputs_tg        = get_gf_res_reserve()->inputs.size();
        n_input_tensors_tg = this->n_input_tensors;
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    {
        // TODO: the worst case graph is not always reached for `n_seqs > 1`
        //       need to implement a more robust mechanism that tries a few different inputs and analyzes the results
        ggml_cgraph * gf = nullptr;
        switch (model.arch) {
            case LLM_ARCH_KIMI_LINEAR:
            case LLM_ARCH_MINIMAX_01:
                // [TAG_RESERVE_DIAG_DECAY]
                // the `inp_diag_decay` tensor size scales with `n_seq_tokens^2` which
                // makes `n_seqs == 1` use more memory for the compute graph compared to `n_seqs > 1`
                gf = graph_reserve(n_tokens, 1,      n_outputs_pp, mctx.get(), model.hparams.no_alloc);
                break;
            default:
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), model.hparams.no_alloc);
        };

        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
        record_backend_private_workspace(gf);
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.hparams.no_alloc) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
        const size_t private_size = backend_kvarn_workspace_y_size[i] +
                backend_kvarn_workspace_split_k_size[i];
        if (private_size > 0) {
            LLAMA_LOG_INFO("%s: %10s KVarN backend workspace = %8.2f MiB "
                    "(y %.2f MiB, split-k %.2f MiB)\n", __func__,
                    ggml_backend_buft_name(buft), private_size / 1024.0 / 1024.0,
                    backend_kvarn_workspace_y_size[i] / 1024.0 / 1024.0,
                    backend_kvarn_workspace_split_k_size[i] / 1024.0 / 1024.0);
        }
    }

    {
        const bool diff = n_nodes_pp != n_nodes_tg || n_splits_pp != n_splits_tg ||
                          n_inputs_pp != n_inputs_tg || n_input_tensors_pp != n_input_tensors_tg;

        const auto val = [diff](int v_pp, int v_tg) -> std::string {
            return diff ? format("%d / %d", v_pp, v_tg) : format("%d", v_pp);
        };

        LLAMA_LOG_INFO("%s: graph%s: nodes = %s, splits = %s, input objects = %s, input tensors = %s\n",
                __func__,
                diff ? format(" (pp bs=%d, tg bs=%d)", n_tokens, n_seqs).c_str() : "",
                val(n_nodes_pp, n_nodes_tg).c_str(),
                val(n_splits_pp, n_splits_tg).c_str(),
                val(n_inputs_pp, n_inputs_tg).c_str(),
                val(n_input_tensors_pp, n_input_tensors_tg).c_str());
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));
    // Keep the reservation pending if any allocation/build step above throws.
    sched_need_reserve = false;
}

int32_t llama_context::prefill_migration_handoff(bool to_remote) {
    if (cparams.local_attn_migration_fallback) {
        return LLAMA_PREFILL_MIGRATION_STATIC_REMOTE;
    }
    if (!cparams.local_attn_migration || !memory ||
            !memory->supports_prefill_migration() ||
            cparams.local_attn_migration_backend == nullptr ||
            cparams.local_attn_prefill_backend == nullptr) {
        return LLAMA_PREFILL_MIGRATION_ERROR;
    }

    const ggml_backend_t next_backend = to_remote
        ? cparams.local_attn_migration_backend
        : cparams.local_attn_prefill_backend;
    const ggml_backend_t current_backend = cparams.local_attn_backend;
    if (current_backend == next_backend) {
        // The server reaches this path for every prompt ubatch. Keep the
        // inactive destination allocated while its asynchronous mirror queue
        // may still target it; release only after an actual owner transition.
        try {
            sched_reserve();
            return LLAMA_PREFILL_MIGRATION_OK;
        } catch (const std::exception & e) {
            LLAMA_LOG_ERROR("%s: graph reservation failed for current KV owner: %s\n", __func__, e.what());
            sched_need_reserve = true;
            return LLAMA_PREFILL_MIGRATION_SCHEDULER_FAILED;
        }
    }

    synchronize();
    // A backend scheduler keeps its last CUDA graph workspace allocated. Free
    // that workspace before allocating the destination KVarN owner; on the
    // remote->CUDA transition it can otherwise consume the VRAM needed to
    // recreate the inactive cache bundle.
    { for (auto & res : gf_res_prev) { if (res) res->reset(); } gf_res_prev_active = nullptr; }
    if (gf_res_reserve) gf_res_reserve->reset();
    sched.reset();
    sched_need_reserve = true;

    if (!memory->handoff_prefill_migration(to_remote)) {
        // The cache owner stayed where it was. Rebuild a scheduler for that
        // owner so the caller can safely continue after a rejected handoff.
        try {
            sched_reserve();
        } catch (const std::exception & e) {
            LLAMA_LOG_ERROR("%s: graph reservation failed after rejected KV handoff: %s\n",
                    __func__, e.what());
            sched_need_reserve = true;
            return LLAMA_PREFILL_MIGRATION_SCHEDULER_FAILED;
        }
        return LLAMA_PREFILL_MIGRATION_OWNER_UNCHANGED;
    }

    cparams.local_attn_backend = next_backend;
    // Drop only the old owner's per-migrated-layer buffers after all prior
    // graph work has synchronized and before reserving workspace for the new
    // attention placement. If reservation fails, the rollback re-allocates
    // this owner and copies its KVarN payload back from the still-live target.
    memory->release_prefill_migration_inactive_buffers();
    try {
        sched_reserve();
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: graph reservation failed after KV handoff: %s\n", __func__, e.what());
        cparams.local_attn_backend = current_backend;
        { for (auto & res : gf_res_prev) { if (res) res->reset(); } gf_res_prev_active = nullptr; }
        if (gf_res_reserve) gf_res_reserve->reset();
        sched.reset();
        sched_need_reserve = true;
        if (!memory->handoff_prefill_migration(!to_remote)) {
            // The transfer failed before swapping cache tensors, so the target
            // remains authoritative. Keep backend selection aligned with it;
            // the caller can continue on that owner or report decode failure.
            cparams.local_attn_backend = next_backend;
            sched_need_reserve = true;
            LLAMA_LOG_ERROR("%s: KV handoff rollback failed; retaining target backend as cache owner\n", __func__);
            return LLAMA_PREFILL_MIGRATION_SCHEDULER_FAILED;
        }
        memory->release_prefill_migration_inactive_buffers();
        try {
            sched_reserve();
        } catch (const std::exception & rollback_error) {
            LLAMA_LOG_ERROR("%s: graph reservation also failed after restoring the previous KV owner: %s\n",
                    __func__, rollback_error.what());
            sched_need_reserve = true;
            return LLAMA_PREFILL_MIGRATION_SCHEDULER_FAILED;
        }
        return LLAMA_PREFILL_MIGRATION_OWNER_UNCHANGED;
    }
    LLAMA_LOG_INFO("%s: KVarN payload owner changed to %s\n",
            __func__, ggml_backend_dev_name(ggml_backend_get_device(next_backend)));
    return LLAMA_PREFILL_MIGRATION_OK;
}

void llama_context::record_backend_private_workspace(ggml_cgraph * gf) {
    using workspace_size_fn = size_t (*)(ggml_backend_dev_t, const ggml_tensor *);
    if (gf == nullptr) {
        return;
    }
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), node);
        const auto found = std::find(backend_ptrs.begin(), backend_ptrs.end(), backend);
        if (found == backend_ptrs.end()) {
            continue;
        }
        const size_t index = size_t(found - backend_ptrs.begin());
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg == nullptr) {
            continue;
        }
        auto * y_fn = (workspace_size_fn) ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_kvarn_workspace_y_size");
        auto * split_fn = (workspace_size_fn) ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_kvarn_workspace_split_k_size");
        if (y_fn) {
            const size_t size = y_fn(dev, node);
            backend_kvarn_workspace_y_size[index] = std::max(
                    backend_kvarn_workspace_y_size[index], size);
            if (size > 0) {
                LLAMA_LOG_DEBUG("%s: backend=%s op=%s KVarN y workspace=%.2f MiB\n",
                        __func__, ggml_backend_dev_name(dev), ggml_op_name(node->op),
                        size / 1024.0 / 1024.0);
            }
        }
        if (split_fn) {
            const size_t size = split_fn(dev, node);
            backend_kvarn_workspace_split_k_size[index] = std::max(
                    backend_kvarn_workspace_split_k_size[index], size);
            if (size > 0) {
                LLAMA_LOG_DEBUG("%s: backend=%s op=%s KVarN split-k workspace=%.2f MiB\n",
                        __func__, ggml_backend_dev_name(dev), ggml_op_name(node->op),
                        size / 1024.0 / 1024.0);
            }
        }
    }
}

void llama_context::synchronize() {
    if (!sched) {
        return;
    }

    // Item 2: within one sampling pass, every output getter (logits/probs/ids/
    // embeddings) re-enters synchronize(). After the first barrier the outputs are
    // host-resident and graph_compute has not submitted anything new, so the extra
    // barriers are redundant. outputs_synced is cleared only at the single async
    // submission point, so this never skips a barrier that real pending work needs.
    // Gated by env so a single binary can be A/B benchmarked with the change isolated.
    static const bool single_sync = [] {
        const char * v = std::getenv("SPEC_OPT_SINGLE_SYNC");
        return v != nullptr && (std::string(v) == "1" || std::string(v) == "true" ||
                                std::string(v) == "yes" || std::string(v) == "on");
    }();
    if (single_sync && outputs_synced) {
        return;
    }

    ggml_backend_sched_synchronize(sched.get());
    outputs_synced = true;

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

void llama_context::set_n_ctx(uint32_t n_ctx) {
    GGML_ASSERT(n_ctx > 0 && n_ctx <= cparams.n_ctx);
    cparams.n_ctx = n_ctx;
    cparams.n_ctx_seq = n_ctx;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::block_size() const {
    return cparams.block_size;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

llama_memory_status llama_context::memory_update(bool optimize) {
    if (!memory) {
        return LLAMA_MEMORY_STATUS_NO_UPDATE;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        if (!mctx) {
            LLAMA_LOG_ERROR("%s: failed to initialize memory update\n", __func__);
            return LLAMA_MEMORY_STATUS_FAILED_PREPARE;
        }
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return LLAMA_MEMORY_STATUS_NO_UPDATE;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return mctx->get_status();
                }
        }

        // reset the previous graph results to make sure that they won't be reused
        // TODO: make mctx->apply() report if a graph reserve is needed, then reset graph results only if the memory module reset the scheduler
        for (auto & res : gf_res_prev) {
            if (res) {
                res->reset();
            }
        }
        gf_res_prev_active = nullptr;

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
            const auto status = mctx->get_status();
            return llama_memory_status_is_fail(status) ? status : LLAMA_MEMORY_STATUS_FAILED_COMPUTE;
        }
        if (mctx->get_status() == LLAMA_MEMORY_STATUS_NO_UPDATE) {
            return LLAMA_MEMORY_STATUS_NO_UPDATE;
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            LLAMA_LOG_ERROR("%s: failed to initialize full memory context\n", __func__);
            return LLAMA_MEMORY_STATUS_FAILED_PREPARE;
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = rkva_initial_reserve_tokens(cparams, model.hparams.no_alloc);

        const uint32_t n_outputs_max = std::min(n_tokens, cparams.n_outputs_max);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_max, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
            return LLAMA_MEMORY_STATUS_FAILED_PREPARE;
        }
    }

    return LLAMA_MEMORY_STATUS_SUCCESS;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return logits.data + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

float * llama_context::get_embeddings_nextn() {
    output_reorder();

    return embd_nextn.data;
}

float * llama_context::get_embeddings_nextn_ith(int32_t i) {
    output_reorder();

    try {
        if (embd_nextn.data == nullptr) {
            throw std::runtime_error("no nextn embeddings");
        }

        const uint32_t n_embd = model.hparams.n_embd_out();

        if (!cparams.embeddings_nextn_masked) {
            // unmasked: nextn rows are stored densely, indexed by raw token position.
            if (i < 0 || (size_t)(i + 1) * n_embd > embd_nextn.size) {
                throw std::runtime_error(format("out of range [0, %zu)", embd_nextn.size / n_embd));
            }
            return embd_nextn.data + (size_t) i * n_embd;
        }

        const int64_t j = output_resolve_row(i);
        return embd_nextn.data + j*n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid nextn embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_layer_inp(uint32_t lid) {
    output_reorder();

    GGML_ASSERT(lid < embd_layer_inp.size() && embd_layer_inp[lid].has_data());

    return embd_layer_inp[lid].data;
}

uint32_t llama_context::get_n_capture() const {
    return cparams.n_capture_layers;
}

float * llama_context::get_embeddings_capture() {
    output_reorder();

    return embd_capture.data;
}

float * llama_context::get_embeddings_capture_ith(int32_t i) {
    output_reorder();

    try {
        if (embd_capture.data == nullptr) {
            throw std::runtime_error("no capture embeddings");
        }

        const uint32_t n_cap  = cparams.n_capture_layers;
        const uint32_t n_embd = model.hparams.n_embd;
        const uint32_t row    = n_cap * n_embd;  // width of one concatenated row

        // capture rows always follow the masked (output-row) layout, mirroring the
        // pre-norm masked path: the buffer holds one row per output position.
        const int64_t j = output_resolve_row(i);
        if (j < 0 || (size_t) (j + 1) * row > embd_capture.size) {
            throw std::runtime_error(format("out of range [0, %zu)", embd_capture.size / row));
        }
        return embd_capture.data + (size_t) j * row;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid capture embeddings id %d, reason: %s\n", __func__, i, err.what());
        return nullptr;
    }
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    if (cparams.dflash_split && !value) {
        LLAMA_LOG_ERROR("%s: DFlash2 split execution requires embeddings output\n", __func__);
        return;
    }
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    // a pooled context computes the logits only without embeddings, so the toggle changes the graph shape
    if (cparams.embeddings != value && cparams.pooling_type != LLAMA_POOLING_TYPE_NONE) {
        sched_need_reserve = true;
    }

    cparams.embeddings = value;
}

void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, masked);

    if (cparams.embeddings_nextn != value || cparams.embeddings_nextn_masked != masked) {
        nextn_decode_id = 0;
        // these flags change the graph shape
        sched_need_reserve = true;
    }

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;
}

uint64_t llama_context::get_nextn_decode_id() const {
    return nextn_decode_id;
}

bool llama_context::matches_nextn_decode(uint64_t id, const llama_batch & batch) const {
    if (!id || id != nextn_decode_id || batch.n_tokens <= 0 ||
            (size_t) batch.n_tokens != nextn_decoded_batch.size() ||
            !batch.token || batch.embd || !batch.pos || !batch.n_seq_id || !batch.seq_id) {
        return false;
    }
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] != 1 || !batch.seq_id[i] || batch.seq_id[i][0] != 0 ||
                nextn_decoded_batch[i].first != batch.token[i] || nextn_decoded_batch[i].second != batch.pos[i]) {
            return false;
        }
    }
    return true;
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid <= model.hparams.n_layer());

    cparams.embeddings_layer_inp[lid] = enable;

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_need_reserve = true;
}

void llama_context::set_nextn_layer_offset(int32_t offset) {
    cparams.nextn_layer_offset = offset;
}

void llama_context::set_capture_layers(const std::vector<int32_t> & layer_ids) {
    const int32_t n_layer = (int32_t) model.hparams.n_layer();
    if (layer_ids.size() > LLAMA_MAX_LAYERS) {
        throw std::invalid_argument("Too many capture layers");
    }
    for (int32_t il : layer_ids) {
        if (il < 0 || il >= n_layer || il >= LLAMA_MAX_LAYERS) {
            throw std::invalid_argument("Capture layer index out of range");
        }
    }
    // reset
    cparams.embeddings_capture = false;
    cparams.n_capture_layers   = 0;
    cparams.capture_layer_idx  = {};

    // enabling/disabling capture adds/removes the t_h_capture node from the
    // graph (see llm_graph_result::set_outputs()), so the scheduler's
    // backend-assignment table -- built against whatever topology was live
    // at the last reserve -- must be re-derived before the next decode.
    // Without this, ggml_backend_sched_get_tensor_backend() on the newly
    // introduced t_h_capture tensor correctly reports "unknown" (nullptr),
    // since the scheduler never split a graph that contained it.
    sched_need_reserve = true;

    if (layer_ids.empty()) {
        return;
    }

    uint32_t n = 0;
    for (int32_t il : layer_ids) {
        cparams.capture_layer_idx[n++] = il;
    }

    cparams.n_capture_layers        = n;
    cparams.embeddings_capture      = n > 0;
    // capture rows reuse the masked output-row layout; force masked extraction on.
    cparams.embeddings_nextn_masked = true;
}

void llama_context::set_dspark_ctx(const float * feat, int64_t n_ctx_rows, int64_t n_embd_cap, const int32_t * pos) {
    if (n_ctx_rows <= 0 || n_embd_cap <= 0 || feat == nullptr) {
        // reset: no staged context (e.g. before the very first drafter round,
        // where the whole prompt still needs to go through as context on the
        // first call, or between unrelated decodes).
        dspark_ctx.n_ctx_rows = 0;
        dspark_ctx.n_embd_cap = 0;
        dspark_ctx.v_ctx_feat.clear();
        dspark_ctx.v_ctx_pos.clear();
        return;
    }

    dspark_ctx.n_ctx_rows = n_ctx_rows;
    dspark_ctx.n_embd_cap = n_embd_cap;

    dspark_ctx.v_ctx_feat.assign(feat, feat + (size_t) n_ctx_rows * (size_t) n_embd_cap);

    dspark_ctx.v_ctx_pos.resize((size_t) n_ctx_rows);
    if (pos != nullptr) {
        std::copy(pos, pos + n_ctx_rows, dspark_ctx.v_ctx_pos.begin());
    } else {
        // caller didn't supply explicit positions: assume a contiguous run
        // ending just before the current staged sequence length. this is a
        // convenience default; callers doing real multi-round decoding should
        // pass explicit positions since the growing-cache bookkeeping (Phase 2)
        // owns the authoritative position numbering.
        std::iota(dspark_ctx.v_ctx_pos.begin(), dspark_ctx.v_ctx_pos.end(), 0);
    }
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.causal_attn == value) {
        return;
    }

    cparams.causal_attn = value;

    // no scheduler reserve needed because graph shapes must not depend on causal_attn, a flip only rebuilds the graph
    //sched_need_reserve = true;
}

bool llama_context::get_causal_attn() const {
    return cparams.causal_attn;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.warmup == value) {
        return;
    }

    cparams.warmup = value;

    // warmups are usually with small batches, so no need to reserve
    //sched_need_reserve = true;
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    if (sampler && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
            warned = true;
        }
        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }
        sampling.samplers.erase(seq_id);
        return false;
    }

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft, cparams.n_outputs_max_per_seq);

        sampling.samplers[seq_id] = sampler;

        sched_need_reserve = true;

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sched_need_reserve = true;

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    sched_need_reserve = true;
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    sched_need_reserve = true;

    return res;
}

// F4 fixture-only scheduler hook. When F4_TRACE_CAPTURE is set, KVarN marks the
// exact K/V tensors returned to build_attn and this callback reads the scheduler
// copies that actually execute, after that node is synchronized. It chains any
// pre-existing cparams callback. Without the env var the scheduler path is
// byte-for-byte the old one.
static ggml_backend_sched_eval_callback g_f4_prev_eval_cb = nullptr;
static void * g_f4_prev_eval_user = nullptr;

// ---------------------------------------------------------------------------
// F4 node dump (fixture-only, READ-ONLY, no graph perturbation).
//
// The dump walks the finished graph AFTER a successful compute and copies the
// whitelisted tensors out of their real scheduler buffers. It never asks the
// scheduler to split or materialize a node, so enabling it cannot change which
// ops are fused, which backend computes what, or the numerical result; that is
// checked separately (dump on vs dump off must produce the same NRMSE).
//
// Arming: F4_NODE_DUMP=<path>. Dumping additionally requires a non-empty tag
// set through llama_f4_set_dump_tag() immediately before the prefill under
// test; an empty tag means "not under test" and nothing is written. Changing
// the tag resets the evaluation counter, so a sample produced during one
// prefill can never be read as a sample from another.
// ---------------------------------------------------------------------------

static std::string g_f4_dump_tag;
static uint32_t    g_f4_dump_eval = 0;

static bool f4_dump_armed() {
    static const bool on = [] {
        const char * e = getenv("F4_NODE_DUMP");
        return e && *e;
    }();
    return on;
}

static const char * f4_dump_path() {
    static const char * p = [] {
        const char * e = getenv("F4_NODE_DUMP");
        return e && *e ? e : nullptr;
    }();
    return p;
}

static const std::set<std::string> & f4_dump_roles() {
    static const std::set<std::string> s = [] {
        std::set<std::string> r;
        const char * e = getenv("F4_NODE_DUMP_ROLES");
        std::string spec = (e && *e)
            ? std::string(e)
            : std::string("l_out,attn_norm,Qcur_full,Kproj,Vproj,Qcur_reshaped,"
                          "Qcur_normed,Kcur_normed,gate_reshaped,Qcur,Kcur,Vcur,"
                          "kq_mask,kq_mask_local,kq_mask_overflow,inp_pos,"
                          "fa_kv_k,fa_kv_v,fa_out,attn_pregate,attn_remote,"
                          "gate_sigmoid,attn_gated,attn_output,attn_residual,"
                          "attn_post_norm,ffn_out,post_ffn");
        size_t a = 0;
        while (a <= spec.size()) {
            const size_t b = spec.find(',', a);
            const std::string tok = spec.substr(a, b == std::string::npos ? spec.size() : b - a);
            if (!tok.empty()) { r.insert(tok); }
            if (b == std::string::npos) { break; }
            a = b + 1;
        }
        return r;
    }();
    return s;
}

static const std::set<int> & f4_dump_layers() {
    static const std::set<int> s = [] {
        std::set<int> r;
        const char * e = getenv("F4_NODE_DUMP_LAYERS");
        std::string spec = (e && *e) ? std::string(e) : std::string("2,3");
        size_t a = 0;
        while (a <= spec.size()) {
            const size_t b = spec.find(',', a);
            const std::string tok = spec.substr(a, b == std::string::npos ? spec.size() : b - a);
            if (!tok.empty()) {
                try { r.insert(std::stoi(tok)); } catch (...) {}
            }
            if (b == std::string::npos) { break; }
            a = b + 1;
        }
        return r;
    }();
    return s;
}

// Names are "<role>-<il>" (see llama_context::graph_get_cb), or bare "<role>"
// for tensors named with il == -1 such as shared graph inputs.
static bool f4_dump_match(const ggml_tensor * t) {
    if (!f4_dump_armed() || g_f4_dump_tag.empty() || !t || !t->name[0]) { return false; }
    std::string n = t->name;
    // The scheduler names a cross-backend operand "<backend>#<name>#<index>".
    // The middle segment is the tensor the graph actually built, so strip the
    // scheduler decoration or the real operand would fall outside the whitelist
    // and only its undecorated twin (which the scheduler may not even fill)
    // would be compared.
    const size_t h1 = n.find('#');
    if (h1 != std::string::npos) {
        const size_t h2 = n.find('#', h1 + 1);
        if (h2 != std::string::npos) { n = n.substr(h1 + 1, h2 - h1 - 1); }
    }
    const size_t dash = n.rfind('-');
    if (dash == std::string::npos) {
        return f4_dump_roles().count(n) > 0;
    }
    if (!f4_dump_roles().count(n.substr(0, dash))) { return false; }
    try {
        return f4_dump_layers().count(std::stoi(n.substr(dash + 1))) > 0;
    } catch (...) {
        return false;
    }
}

// Optional verbatim value dump for a small whitelist of (usually tiny) tensors
// so a hash mismatch can be attributed to specific entries instead of just "the
// bytes differ". Limited by F4_NODE_DUMP_FULL_EVALS to keep the file small.
static const std::set<std::string> & f4_dump_full_names() {
    static const std::set<std::string> s = [] {
        std::set<std::string> r;
        const char * e = getenv("F4_NODE_DUMP_FULL");
        if (!e || !*e) { return r; }
        size_t a = 0;
        const std::string spec(e);
        while (a <= spec.size()) {
            const size_t b = spec.find(',', a);
            const std::string tok = spec.substr(a, b == std::string::npos ? spec.size() : b - a);
            if (!tok.empty()) { r.insert(tok); }
            if (b == std::string::npos) { break; }
            a = b + 1;
        }
        return r;
    }();
    return s;
}

static int f4_dump_full_evals() {
    static const int v = [] {
        const char * e = getenv("F4_NODE_DUMP_FULL_EVALS");
        return (e && *e) ? std::atoi(e) : -1;
    }();
    return v;
}

static bool f4_dump_full_wanted(const ggml_tensor * t) {
    if (f4_dump_full_names().empty()) { return false; }
    const int lim = f4_dump_full_evals();
    if (lim >= 0 && (int) g_f4_dump_eval > lim) { return false; }
    const std::string n = t->name;
    const size_t dash = n.rfind('-');
    const std::string role = dash == std::string::npos ? n : n.substr(0, dash);
    return f4_dump_full_names().count(role) > 0 || f4_dump_full_names().count(n) > 0;
}

static uint64_t f4_fnv1a(const void * data, size_t n, uint64_t h = 14695981039346656037ULL) {
    const auto * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

// Views share their producer's storage: RESHAPE/PERMUTE/TRANSPOSE/VIEW and the
// KVarN leaf views carry no buffer of their own. Resolve to a tensor that does,
// and carry the view's byte offset so the copy covers the bytes the consumer
// actually reads, not the whole backing store. Only view-like ops are stepped
// through: resolving through a real op would hash its INPUT and silently
// mislabel a mismatch as an upstream one.
struct f4_dump_src { const ggml_tensor * t = nullptr; const char * from = "self"; size_t off = 0; };

static f4_dump_src f4_dump_resolve(const ggml_tensor * t) {
    f4_dump_src r;
    const ggml_tensor * s = t;
    for (int hop = 0; hop < 8 && s; ++hop) {
        if (s->buffer) {
            if (s != t && t->data && s->data && t->data >= s->data) {
                r.off = (size_t) ((const char *) t->data - (const char *) s->data);
                r.from = "view_into_src";
            }
            r.t = s;
            return r;
        }
        if (s->view_src && s->view_src != s && s->view_src->buffer) {
            if (t->data && s->view_src->data && t->data >= s->view_src->data) {
                r.off = (size_t) ((const char *) t->data - (const char *) s->view_src->data);
            } else {
                r.off = (size_t) s->view_offs;
            }
            r.t = s->view_src;
            r.from = "view_src";
            return r;
        }
        const bool view_like = s->op == GGML_OP_RESHAPE || s->op == GGML_OP_PERMUTE ||
                               s->op == GGML_OP_TRANSPOSE || s->op == GGML_OP_VIEW ||
                               s->op == GGML_OP_KVARN_VIEW;
        if (view_like && s->src[0] && s->src[0] != s) {
            s = s->src[0]; r.from = "viewlike_src0"; continue;
        }
        break;
    }
    return r;
}

// Hash `len` bytes at `off` inside a resolved source, in 4 MiB chunks. Returns
// 0 only if nothing was read; `status` explains anything else.
static uint64_t f4_hash_span(const ggml_tensor * src, size_t off, size_t len,
                             const char ** status, size_t * hashed) {
    uint64_t hash = 14695981039346656037ULL;
    *hashed = 0;
    if (len == 0) { *status = "empty"; return hash; }
    constexpr size_t CHUNK = 1u << 22;
    std::vector<uint8_t> buf(std::min(CHUNK, len));
    for (size_t o = 0; o < len; o += CHUNK) {
        const size_t n = std::min(CHUNK, len - o);
        try {
            ggml_backend_tensor_get(src, buf.data(), off + o, n);
        } catch (...) {
            *status = "readfailed";
            return hash;
        }
        hash = f4_fnv1a(buf.data(), n, hash);
        *hashed += n;
    }
    return hash;
}

static void f4_dump_tensor(const ggml_tensor * t) {
    const char * path = f4_dump_path();
    if (!path) { return; }
    const f4_dump_src rs = f4_dump_resolve(t);
    const ggml_tensor * src = rs.t;
    // The FNV offset basis on the first call: seeding with 0 would make an
    // all-zero tensor hash to 0 and be indistinguishable from "not hashed".
    uint64_t hash = 14695981039346656037ULL;
    const char * status = "ok";
    char backend[128] = "-";
    size_t hashed = 0;
    if (!src) {
        status = "nobuffer";
    } else {
        auto * bt = ggml_backend_buffer_get_type(src->buffer);
        auto * dev = bt ? ggml_backend_buft_get_device(bt) : nullptr;
        std::snprintf(backend, sizeof(backend), "%s", dev ? ggml_backend_dev_name(dev)
                        : (bt ? ggml_backend_buft_name(bt) : "?"));
        const size_t src_total = ggml_nbytes(src);
        if (rs.off > src_total) {
            status = "viewbeyondstore";
        } else {
            // Copy the view's own span, clamped to what the backing store holds:
            // a KVarN view may declare the padded n_kv width while the store
            // only holds the cells written so far.
            const size_t want = ggml_nbytes(t);
            const size_t len0 = std::min(want, src_total - rs.off);
            if (len0 == 0) {
                status = "empty";
            } else {
                constexpr size_t CHUNK = 1u << 22;
                std::vector<uint8_t> buf(std::min(CHUNK, len0));
                for (size_t off = 0; off < len0; off += CHUNK) {
                    const size_t len = std::min(CHUNK, len0 - off);
                    try {
                        ggml_backend_tensor_get(src, buf.data(), rs.off + off, len);
                    } catch (...) {
                        status = "readfailed";
                        break;
                    }
                    hash = f4_fnv1a(buf.data(), len, hash);
                    hashed += len;
                }
                if (len0 < want) { status = "clamped"; }
            }
        }
    }
    char fa[80] = "-";
    // For FlashAttention also record its operands: a fixture name is a label,
    // not an edge, so without the source names there is no way to tell whether
    // the tensor a fixture calls "kq_mask" is the mask this node reads.
    char fasrc[640] = "-";
    if (t->op == GGML_OP_FLASH_ATTN_EXT) {
        // [0]=scale bits [1]=alibi [2]=softcap [3]=precision [4]=kvarn domain
        // (ggml_flash_attn_ext_set_n_kv_max also writes index 4, so this is
        // whatever was written last).
        std::snprintf(fa, sizeof(fa), "%d,%d,%d,%d,%d", t->op_params[0], t->op_params[1],
                      t->op_params[2], t->op_params[3], t->op_params[4]);
        const char * nm[4] = { "-", "-", "-", "-" };
        for (int i = 0; i < 4 && i < GGML_MAX_SRC; ++i) {
            if (t->src[i]) { nm[i] = t->src[i]->name[0] ? t->src[i]->name : "(unnamed)"; }
        }
        std::snprintf(fasrc, sizeof(fasrc), "%s|%s|%s|%s", nm[0], nm[1], nm[2], nm[3]);
    }
    FILE * f = std::fopen(path, "a");
    if (!f) { return; }
    std::fprintf(f,
        "NODE tag=%s eval=%u name=%s op=%s dtype=%s ne=%lld|%lld|%lld|%lld "
        "nb=%zu,%zu,%zu,%zu bytes=%zu flags=%u src=%s srcname=%s srcdtype=%s srcbytes=%zu "
        "srcoff=%zu hashed=%zu backend=%s hash=%016llx status=%s fa=%s fasrc=%s\n",
        g_f4_dump_tag.c_str(), g_f4_dump_eval, t->name, ggml_op_name(t->op),
        ggml_type_name(t->type),
        (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
        t->nb[0], t->nb[1], t->nb[2], t->nb[3], ggml_nbytes(t), t->flags,
        rs.from, src ? src->name : "-", src ? ggml_type_name(src->type) : "-",
        src ? ggml_nbytes(src) : 0, rs.off, hashed,
        backend, (unsigned long long) hash, status, fa, fasrc);
    // FlashAttention's operands are live RIGHT NOW (this fires immediately
    // after the node computed), so they can be read here even when the graph
    // itself is not in front of us -- which is the only way to reach the
    // scheduler's cross-backend operand copies. This is the comparison that
    // says whether the mask and the K/V actually consumed differ.
    if (t->op == GGML_OP_FLASH_ATTN_EXT) {
        static const bool want_src = [] {
            const char * e = getenv("F4_NODE_DUMP_FA_SRC");
            return e && *e && std::atoi(e) != 0;
        }();
        if (want_src) {
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                ggml_tensor * opnd = t->src[i];
                if (!opnd) { continue; }
                const f4_dump_src rs2 = f4_dump_resolve(opnd);
                const char * st2 = "ok";
                size_t h2 = 0;
                uint64_t hv = 0;
                if (!rs2.t) {
                    st2 = "nobuffer";
                } else {
                    const size_t stot = ggml_nbytes(rs2.t);
                    if (rs2.off > stot) { st2 = "viewbeyondstore"; }
                    else { hv = f4_hash_span(rs2.t, rs2.off, std::min(ggml_nbytes(opnd), stot - rs2.off), &st2, &h2); }
                }
                std::fprintf(f, "FASRC tag=%s eval=%u idx=%d name=%s op=%s dtype=%s "
                                "ne=%lld|%lld|%lld|%lld bytes=%zu src=%s srcbytes=%zu hashed=%zu "
                                "hash=%016llx status=%s\n",
                             g_f4_dump_tag.c_str(), g_f4_dump_eval, i,
                             opnd->name, ggml_op_name(opnd->op), ggml_type_name(opnd->type),
                             (long long) opnd->ne[0], (long long) opnd->ne[1],
                             (long long) opnd->ne[2], (long long) opnd->ne[3],
                             ggml_nbytes(opnd), rs2.from,
                             rs2.t ? ggml_nbytes(rs2.t) : 0, h2,
                             (unsigned long long) hv, st2);
            }
        }
    }
    if (f4_dump_full_wanted(t) && src) {
        const size_t nbyte = std::min(ggml_nbytes(t), (ggml_nbytes(src) > rs.off ? ggml_nbytes(src) - rs.off : size_t(0)));
        if (nbyte > 0 && t->type == GGML_TYPE_F16 && (nbyte % 2) == 0) {
            std::vector<uint8_t> raw(nbyte);
            ggml_backend_tensor_get(src, raw.data(), rs.off, nbyte);
            std::fprintf(f, "DATA tag=%s eval=%u name=%s kind=f16 n=%zu v=",
                         g_f4_dump_tag.c_str(), g_f4_dump_eval, t->name, nbyte / 2);
            const size_t n = nbyte / 2;
            std::vector<float> tmp(n);
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), tmp.data(), (int) n);
            for (size_t i = 0; i < n; ++i) {
                std::fprintf(f, i ? ",%.9g" : "%.9g", (double) tmp[i]);
            }
            std::fprintf(f, "\n");
        } else if (nbyte > 0 && t->type == GGML_TYPE_F32 && (nbyte % 4) == 0) {
            std::vector<uint8_t> raw(nbyte);
            ggml_backend_tensor_get(src, raw.data(), rs.off, nbyte);
            std::fprintf(f, "DATA tag=%s eval=%u name=%s kind=f32 n=%zu v=",
                         g_f4_dump_tag.c_str(), g_f4_dump_eval, t->name, nbyte / 4);
            const size_t n = nbyte / 4;
            for (size_t i = 0; i < n; ++i) {
                float val; std::memcpy(&val, raw.data() + i * 4, 4);
                std::fprintf(f, i ? ",%.9g" : "%.9g", (double) val);
            }
            std::fprintf(f, "\n");
        } else if (nbyte > 0 && t->type == GGML_TYPE_I32 && (nbyte % 4) == 0) {
            std::vector<uint8_t> raw(nbyte);
            ggml_backend_tensor_get(src, raw.data(), rs.off, nbyte);
            std::fprintf(f, "DATA tag=%s eval=%u name=%s kind=i32 n=%zu v=",
                         g_f4_dump_tag.c_str(), g_f4_dump_eval, t->name, nbyte / 4);
            const size_t n = nbyte / 4;
            for (size_t i = 0; i < n; ++i) {
                int32_t val; std::memcpy(&val, raw.data() + i * 4, 4);
                std::fprintf(f, i ? ",%d" : "%d", (int) val);
            }
            std::fprintf(f, "\n");
        }
    }
    std::fclose(f);
}

void llama_f4_set_dump_tag(const char * tag) {
    const std::string next = tag ? tag : "";
    if (next == g_f4_dump_tag) { return; }
    g_f4_dump_tag = next;
    g_f4_dump_eval = 0;
    if (f4_dump_armed()) {
        fprintf(stderr, "F4DUMP tag=%s eval_counter_reset\n",
                g_f4_dump_tag.empty() ? "(off)" : g_f4_dump_tag.c_str());
    }
}

// Called once per finished graph, after compute has synchronized. Only INPUT
// tensors are read here: their memory is written by set_inputs and read by the
// graph, never overwritten by a later node, so a post-compute copy is exact.
// Intermediate results are NOT read here -- the scheduler legitimately reuses a
// buffer region as soon as its producer's live range ends, so a post-compute
// copy of an intermediate node would return whatever the last writer left in
// that region. Those go through the eval callback below, which fires while the
// node's own value is still the live occupant.
// Declared here instead of in ggml.h on purpose: adding them to the public
// header would invalidate every translation unit in the tree, including the
// CUDA template instances, which costs a multi-hour rebuild for two accessors.
// The definitions live in ggml/src/ggml.c and keep the same C linkage.
extern "C" {
int                  ggml_graph_n_leafs(struct ggml_cgraph * cgraph);
struct ggml_tensor * ggml_graph_leaf(struct ggml_cgraph * cgraph, int i);
}

// Advances the per-tag evaluation index once per ubatch, BEFORE that ubatch
// computes, so input copies taken after the graph and intermediate copies taken
// during it are stamped with the same index and cannot be crossed.
static void f4_dump_next_eval() {
    if (!f4_dump_armed()) { return; }
    if (g_f4_dump_tag.empty()) {
        // Server-side auto-tag: the dump otherwise only fires when a test calls
        // llama_f4_set_dump_tag() right before the prefill under test.
        const char * t = getenv("F4_NODE_DUMP_AUTOTAG");
        if (t && *t) { g_f4_dump_tag = t; }
    }
    ++g_f4_dump_eval;
}

// Optional operator census: writes one line per graph node (op, shape) for the
// first F4_NODE_DUMP_OPMAP evaluations. Names are auto-generated and therefore
// not comparable across configurations, but the MULTISET of operators is, and
// it is what says whether an operator exists on one side only -- which is the
// only way to locate an operand set that a named-node dump cannot see.
static void f4_dump_opmap(ggml_cgraph * gf) {
    static const int lim = [] {
        const char * e = getenv("F4_NODE_DUMP_OPMAP");
        return (e && *e) ? std::atoi(e) : -1;
    }();
    if (lim < 0 || (int) g_f4_dump_eval > lim) { return; }
    FILE * f = std::fopen(f4_dump_path(), "a");
    if (!f) { return; }
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        std::fprintf(f, "OP tag=%s eval=%u i=%d op=%s ne=%lld|%lld|%lld|%lld name=%s\n",
                     g_f4_dump_tag.c_str(), g_f4_dump_eval, i, ggml_op_name(t->op),
                     (long long) t->ne[0], (long long) t->ne[1],
                     (long long) t->ne[2], (long long) t->ne[3], t->name);
    }
    std::fclose(f);
}

static void f4_dump_graph(ggml_cgraph * gf) {
    if (!f4_dump_armed() || g_f4_dump_tag.empty() || !gf) { return; }
    f4_dump_opmap(gf);
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        const bool is_input = (t->flags & GGML_TENSOR_FLAG_INPUT) != 0;
        if (is_input && f4_dump_match(t)) { f4_dump_tensor(t); }
    }
    // Graph INPUTS (masks, positions, indices) are LEAVES, not nodes: ggml puts
    // every op==NONE tensor in cgraph->leafs. Walking nodes only would silently
    // skip exactly the mask and position tensors this bisect has to compare.
    for (int i = 0; i < ggml_graph_n_leafs(gf); ++i) {
        ggml_tensor * t = ggml_graph_leaf(gf, i);
        if (f4_dump_match(t)) { f4_dump_tensor(t); }
    }
}

// Intermediate results: asked for so the scheduler evaluates them as their own
// step (their value is then live and uniquely owned when the callback fires),
// captured immediately after that step completes.
static bool f4_dump_wants(const ggml_tensor * t) {
    if (!t || t->op == GGML_OP_NONE) { return false; }
    return f4_dump_match(t);
}

static bool f4_dump_capture(const ggml_tensor * t) {
    if (!f4_dump_wants(t)) { return false; }
    f4_dump_tensor(t);
    return true;
}

static bool f4_sched_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);
    if (ask) {
        if (f4_dump_wants(t)) { return true; }
        if (llama_kv_cache_kvarn::eval_wants_node(t)) { return true; }
        if (g_f4_prev_eval_cb) { return g_f4_prev_eval_cb(t, ask, g_f4_prev_eval_user); }
        return false;
    }
    const bool dumped = f4_dump_capture(t);
    GGML_UNUSED(dumped);
    const bool captured = llama_kv_cache_kvarn::eval_capture_node(t);
    GGML_UNUSED(captured);
    if (g_f4_prev_eval_cb) { return g_f4_prev_eval_cb(t, ask, g_f4_prev_eval_user); }
    return true;
}

static const char * rkva_wavefront_fallback(const llm_graph_params & params, bool input_only = false) {
    const auto & ub = params.ubatch;
    const auto & cp = params.cparams;
    if (!cp.rkva_wavefront_tile) return "disabled";
    if (params.arch != LLM_ARCH_QWEN35 || params.gtype == LLM_GRAPH_TYPE_DECODER_MTP ||
            cp.ctx_type == LLAMA_CONTEXT_TYPE_MTP) return "unsupported_model_or_draft";
    if ((!input_only && !params.mctx) || !ub.pos || !ub.seq_id || !ub.n_seq_id) return "reserve_or_missing_input";
    if (cp.rkva_wavefront_tile == 128 &&
            (ub.n_tokens != 256 || cp.embeddings || cp.embeddings_nextn || cp.embeddings_nextn_masked)) {
        return "wf128_requires_complete_n256_parent";
    }
    if (cp.rkva_wavefront_tile == 256 || cp.rkva_wavefront_tile == 512) {
        if ((cp.rkva_wavefront_tile == 256 && cp.n_ubatch != 512 && cp.n_ubatch != 1024) ||
                (cp.rkva_wavefront_tile == 512 && (cp.n_ubatch < 1024 || cp.n_ubatch > 4096 || cp.n_ubatch % 512 != 0)) ||
                ub.n_tokens != cp.n_ubatch) {
            return "requires_exact_requested_n256_parent";
        }
        if (cp.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT || cp.embeddings || cp.embeddings_nextn ||
                cp.embeddings_nextn_masked || !params.samplers.empty()) return "n256_output_contract";
        if (!ub.token || ub.embd || ub.embd_state || ub.type || ub.decision_order) return "n256_token_input_required";
    }
    if (!cp.causal_attn || params.hparams.swa_type != LLAMA_SWA_TYPE_NONE ||
            params.hparams.use_alibi || params.hparams.f_max_alibi_bias != 0.0f ||
            params.hparams.attn_soft_cap) return "attention_contract";
    if (cp.n_rs_seq != 0) return "recurrent_rollback";
    if (cp.n_capture_layers != 0) return "layer_capture";
    if (std::any_of(cp.embeddings_layer_inp.begin(), cp.embeddings_layer_inp.end(),
                [](bool enabled) { return enabled; })) return "layer_input_capture";
    if (ub.n_tokens <= cp.rkva_wavefront_tile) return "single_tile";
    if (ub.n_seqs != 1 || ub.n_seqs_unq != 1 || ub.n_seq_tokens != ub.n_tokens) return "multiple_sequences";
    if (!cp.remote_attn_position_split || !cp.kv_position_split_p || !params.backend_remote ||
            !ggml_backend_remote_attn_async_enabled(params.backend_remote) ||
            !ggml_backend_remote_attn_native_kvarn_enabled(params.backend_remote)) return "remote_native_async_required";
    if (cp.rkva_wavefront_tile == 512 &&
            !ggml_backend_remote_attn_kvarn_bulk_512_enabled(params.backend_remote)) {
        return "remote_kvarn_bulk_512_required";
    }
    if (cp.rkva_wavefront_rpc_batch == 4 && !ggml_backend_remote_attn_cohort_enabled(params.backend_remote)) {
        return "remote_cohort_capability_required";
    }
    const auto * hybrid = dynamic_cast<const llama_memory_hybrid_context *>(params.mctx);
    const auto * kvarn = hybrid ? dynamic_cast<const llama_kv_cache_kvarn_context *>(hybrid->get_attn()) : nullptr;
    if (!input_only && (!kvarn || !kvarn->position_split_enabled())) return "hybrid_kvarn_required";
    if (params.hparams.n_layer() == 0 || params.hparams.is_recr(params.hparams.n_layer() - 1)) return "trailing_recurrent_layers";
    for (uint32_t i = 0; i < ub.n_tokens; ++i) {
        if (ub.n_seq_id[i] != 1 || !ub.seq_id[i] || !ub.seq_id[0] || ub.seq_id[i][0] != ub.seq_id[0][0]) return "multiple_sequences";
        if (ub.pos[i] < int64_t(cp.kv_position_split_p)) return "not_all_overflow";
        if (i && int64_t(ub.pos[i]) != int64_t(ub.pos[i - 1]) + 1) return "noncontiguous_positions";
    }
    return nullptr;
}

llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    static const bool profile_prep = [] {
        const char * value = std::getenv("LLAMA_GRAPH_PREP_PROFILE");
        return value && std::strcmp(value, "0") != 0;
    }();
    const int64_t prep_start = profile_prep ? ggml_time_us() : 0;
    llama_memory_context_finish_guard memory_finish_guard(mctx);
    if (mctx && memory && backend_remote && cparams.remote_attn_position_split &&
            memory->invalidate_prefix_on_write(ubatch)) {
        ggml_backend_remote_attn_invalidate_range(backend_remote);
    }
    if (mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        memory_finish_guard.finish(ret);
        return nullptr;
    }
    const int64_t prep_applied = profile_prep ? ggml_time_us() : 0;

    auto * res = get_gf_res_prev();
    auto * gf  = res->get_gf();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    const auto gparams = graph_params(res, ubatch, mctx, gtype);
    if (cparams.rkva_wavefront_tile) {
        const char * reason = rkva_wavefront_fallback(gparams);
        const uint64_t count = ++wavefront_decisions[reason ? reason : "tiled_parent"];
        if (count == 1 || count % 64 == 0) {
            LLAMA_LOG_INFO("RKVA wavefront selection: reason=%s count=%llu nq=%u tile=%u\n",
                    reason ? reason : "tiled_parent", (unsigned long long) count,
                    ubatch.n_tokens, gparams.wavefront_tile_size);
        }
    }


    wavefront_parent_active = gparams.wavefront_tile_size != 0;
    try {
    const bool reused = !graph_reuse_disable && gf_res_prev_active == res && res->can_reuse(gparams);
    const int64_t prep_checked = profile_prep ? ggml_time_us() : 0;
    int64_t prep_reset = prep_checked, prep_built = prep_checked, prep_allocated = prep_checked;
    if (reused) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        // with pipeline parallelism, the previous graph_compute_async may still be running
        // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
        // that the previous compute is still reading.
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
        }

        n_reused++;
    } else {
        gf_res_prev_active = nullptr;
        res->reset();

        // F4: drop the K/V store slots at the GRAPH CONSTRUCTION boundary, before
        // any store or reader of this graph runs. The slots hold tensors owned by
        // the ggml_context of the previous graph, and that context is reused
        // across ubatches (and re-created on rebuild/reserve), so a slot written
        // by a dead graph can still be reachable here. Clearing later, in
        // set_inputs, was too late: build_graph above already reads K/V, and the
        // first graph of a slot has no input buffer yet. Doing it here is safe for
        // this graph because nothing of THIS graph is stored yet; dependencies
        // inside the graph are built after this point and are unaffected.
        if (mctx) {
            mctx->reset_graph_store_slots();
        }

        ggml_backend_sched_reset(sched.get());
        const char * f4_eval_env = getenv("F4_TRACE_CAPTURE");
        const char * f4_dump_env = getenv("F4_NODE_DUMP");
        if ((f4_eval_env && *f4_eval_env) || (f4_dump_env && *f4_dump_env)) {
            g_f4_prev_eval_cb = cparams.cb_eval;
            g_f4_prev_eval_user = cparams.cb_eval_user_data;
            ggml_backend_sched_set_eval_callback(sched.get(), f4_sched_eval_cb, nullptr);
        } else {
            ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
        }
        prep_reset = profile_prep ? ggml_time_us() : 0;

        //const auto t_start_us = ggml_time_us();

        gf = model.build_graph(gparams);
        prep_built = profile_prep ? ggml_time_us() : 0;

        //LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            memory_finish_guard.finish(ret);
            return nullptr;
        }
        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            memory_finish_guard.finish(ret);
            return nullptr;
        }
        prep_allocated = profile_prep ? ggml_time_us() : 0;

        gf_res_prev_active = res;
    }

    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }
    if (profile_prep) {
        fprintf(stderr, "[GRAPHPREP] start_us=%lld nq=%u reused=%d apply_ms=%.3f check_ms=%.3f reset_ms=%.3f build_ms=%.3f alloc_ms=%.3f inputs_ms=%.3f\n",
                (long long) prep_start, ubatch.n_tokens, int(reused),
                (prep_applied - prep_start)/1000.0, (prep_checked - prep_applied)/1000.0,
                (prep_reset - prep_checked)/1000.0, (prep_built - prep_reset)/1000.0,
                (prep_allocated - prep_built)/1000.0, (ggml_time_us() - prep_allocated)/1000.0);
    }

    if (mctx) {
        mctx->graph_compute_start();
    }
    f4_dump_next_eval();
    static const bool f5_stage_timing = [] {
        const char * v = getenv("LLAMA_F5_STAGE_TIMING");
        return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
    }();
    const int64_t f5_t_mid = f5_stage_timing ? ggml_time_us() : 0;
    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (f5_stage_timing) {
        fprintf(stderr, "[F5STAGE] n_tokens=%u graph_compute_host_ms=%.3f\n",
                ubatch.n_tokens, (ggml_time_us() - f5_t_mid)/1000.0);
    }
    memory_finish_guard.finish(status);
    if (mctx) {
        mctx->capture_after_compute(sched.get(), status);
        mctx->graph_compute_complete(sched.get(), status);
    }
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    // F4: snapshot the whitelisted tensors of THIS finished graph. Read-only,
    // after synchronization, never touches the scheduler's split decisions.
    f4_dump_graph(res->get_gf());

    ret = GGML_STATUS_SUCCESS;
    wavefront_committed = wavefront_committed || wavefront_parent_active;

    return res;
    } catch (const std::exception & e) {
        if (!wavefront_parent_active && !wavefront_committed) throw;
        LLAMA_LOG_ERROR("RKVA wavefront graph/input failure: %s\n", e.what());
    } catch (...) {
        if (!wavefront_parent_active && !wavefront_committed) throw;
        LLAMA_LOG_ERROR("RKVA wavefront graph/input failure: unknown exception\n");
    }
    // Return through decode's full-context retirement/poison path even for
    // a thrown tile-input validation error before graph_compute() starts.
    ret = GGML_STATUS_FAILED;
    memory_finish_guard.finish(ret);
    return nullptr;
}

static bool dflash2_split_block_fits(const llama_batch_ext & batch, uint32_t n_seq_max, uint32_t block_size) {
    std::array<uint32_t, LLAMA_MAX_SEQ> rows = {};
    for (const auto & token : batch.tokens) {
        if (token.seq_ids.size() != 1) {
            return false;
        }
        const llama_seq_id seq_id = *token.seq_ids.begin();
        if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max || ++rows[seq_id] > block_size) {
            return false;
        }
    }
    return true;
}

int llama_context::encode(const llama_batch_ext & batch_inp) {
    nextn_decode_id = 0;
    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    if (batch_inp.n_embd > 0 && batch_inp.n_embd != (cparams.dflash_selector_only ? hparams.n_embd : hparams.n_embd_inp_enc())) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the encoder input %u\n",
                __func__, batch_inp.n_embd, hparams.n_embd_inp_enc());
        return -1;
    }

    // eagle3/DFlash: features as encoder input, and non-draft paths fall back to model's input dim
    if (cparams.dflash_selector_only && std::any_of(batch_inp.tokens.begin(), batch_inp.tokens.end(), [](const auto & t) { return t.id == LLAMA_TOKEN_NULL || !t.has_embd; })) {
        LLAMA_LOG_ERROR("%s: DFlash2 selector batches require token IDs and normalized hidden rows\n", __func__);
        return -1;
    }
    if (cparams.dflash_selector_only &&
            !dflash2_split_block_fits(batch_inp, cparams.n_seq_max, hparams.dflash_block_size)) {
        LLAMA_LOG_ERROR("%s: DFlash2 selector block exceeds trained block_size=%u (anchor counts as one row)\n",
                __func__, hparams.dflash_block_size);
        return -1;
    }
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we always output all tokens and skip position continuity checks (output_all=true)
    if (!balloc->init(batch_inp, model.vocab, true)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    sched_reserve();

    n_queued_tokens += n_tokens;

    // reserve output buffer
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits  = res->get_logits();
    auto * t_embd    = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();
    auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

    // extract logits
    if (logits.data && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_tokens*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_tokens*n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_tokens*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // extract nextn embeddings (hidden state before the final output norm)
    if (embd_nextn.data && t_h_nextn && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
        GGML_ASSERT(backend_h != nullptr);

        const uint32_t n_embd = hparams.n_embd_out();
        GGML_ASSERT(n_tokens*n_embd <= (int64_t) embd_nextn.size);
        ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data, 0, n_tokens*n_embd*sizeof(float));
    }

    // extract multi-layer capture embeddings (concatenated per position).
    // single bulk copy: t_h_capture is already [n_capture * n_embd, n_tokens].
    if (embd_capture.data && cparams.n_capture_layers > 0 && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        ggml_tensor * t_cap = res->get_h_capture();
        if (!t_cap) {
            LLAMA_LOG_ERROR("%s: target graph did not produce requested capture layers\n", __func__);
            return -1;
        }
        ggml_backend_t backend_c = ggml_backend_sched_get_tensor_backend(sched.get(), t_cap);
        GGML_ASSERT(backend_c != nullptr);
        const uint32_t row = cparams.n_capture_layers * hparams.n_embd;
        GGML_ASSERT(n_tokens * (int64_t) row <= (int64_t) embd_capture.size);
        ggml_backend_tensor_get_async(backend_c, t_cap, embd_capture.data, 0, (size_t) n_tokens * row * sizeof(float));
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

template<typename T>
static void copy_tensor_async_rows(
    const std::vector<ggml_tensor *> & tensors,
    const buffer_view<T> & dst,
    size_t stride,
    uint32_t row_offset,
    ggml_backend_sched_t sched,
    std::vector<uint32_t> * counts = nullptr) {
    if (!dst.has_data()) {
        return;
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        auto * tensor = tensors[i];
        if (tensor == nullptr) {
            continue;
        }

        const uint32_t row = row_offset + i;
        const size_t n_elements = ggml_nelements(tensor);
        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampling tensor must be contiguous for async copy");
        GGML_ASSERT(n_elements <= stride);
        GGML_ASSERT((size_t) row * stride + n_elements <= dst.size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        T * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        if (counts) {
            GGML_ASSERT(row < counts->size());
            (*counts)[row] = n_elements;
        }
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

int llama_context::decode(const llama_batch_ext & batch_inp) {
    if (wavefront_poisoned) {
        LLAMA_LOG_ERROR("RKVA wavefront: refusing reuse of a failed tiled context\n");
        return -3;
    }

    nextn_decode_id = 0;
    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const bool dflash_features = model.arch == LLM_ARCH_DFLASH &&
            std::all_of(batch_inp.tokens.begin(), batch_inp.tokens.end(), [](const auto & t) { return t.id == LLAMA_TOKEN_NULL; });
    const size_t expected_embd = dflash_features ? batch_inp.n_embd_inp_enc : batch_inp.n_embd_inp;
    if (batch_inp.n_embd > 0 && batch_inp.n_embd != expected_embd) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the decoder input %zu\n",
                __func__, batch_inp.n_embd, expected_embd);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();
    if (cparams.dflash_split && std::any_of(batch_inp.tokens.begin(), batch_inp.tokens.end(), [](const auto & t) { return t.id != LLAMA_TOKEN_NULL && !t.has_embd; })) {
        LLAMA_LOG_ERROR("%s: DFlash2 split draft batches require token IDs and embeddings\n", __func__);
        return -1;
    }
    if (cparams.dflash_split &&
            std::any_of(batch_inp.tokens.begin(), batch_inp.tokens.end(), [](const auto & t) { return t.id != LLAMA_TOKEN_NULL; }) &&
            !dflash2_split_block_fits(batch_inp, cparams.n_seq_max, hparams.dflash_block_size)) {
        LLAMA_LOG_ERROR("%s: DFlash2 split block exceeds trained block_size=%u (anchor counts as one row)\n",
                __func__, hparams.dflash_block_size);
        return -1;
    }

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    // non-causal attention needs the whole sequence in one ubatch
    const bool embd_all     = output_all || !cparams.causal_attn;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // TODO: avoid this workaround in the future
    // embedding contexts output every token even when no token is explicitly marked as output
    if (has_samplers) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (const auto & tok : batch_inp.tokens) {
            if (!output_all && !tok.output) {
                continue;
            }

            for (auto seq_id : tok.seq_ids) {
                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    continue;
                }

                seq_output_count[seq_id]++;
                auto sampler = sampling.samplers.find(seq_id);
                if (sampler != sampling.samplers.end() &&
                        seq_output_count[seq_id] > (int32_t) cparams.n_outputs_max_per_seq) {
                    LLAMA_LOG_ERROR("%s: backend sampling supports at most %u outputs per sequence "
                            "(seq_id %d had %d)\n", __func__, cparams.n_outputs_max_per_seq,
                            seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    uint32_t n_nextn_copied = 0;
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    output_swaps.clear();
    embd_batch_idxs.clear();

    sched_reserve();

    bool did_optimize = false;

    // handle any pending shifts/copies
    if (llama_memory_status_is_fail(memory_update(false))) {
        LLAMA_LOG_ERROR("%s: failed to apply pending memory update\n", __func__);
        return -2;
    }

    llama_memory_context_ptr mctx;

    // The requested parent is a scheduling boundary, independent of logical B.
    // Check each internal parent, then retain physical N256 for prefix/crossing/
    // tail inputs. Only plain single-sequence batches may use the refining path.
    uint32_t effective_ubatch = cparams.n_ubatch;
    auto * wavefront_hybrid = dynamic_cast<llama_memory_hybrid *>(memory.get());
    bool refine_wavefront = false;
    if (cparams.rkva_wavefront_tile == 256 || cparams.rkva_wavefront_tile == 512) {
        effective_ubatch = 256;
        const auto & batch = balloc->get_batch();
        const auto * kvarn = wavefront_hybrid ?
                dynamic_cast<const llama_kv_cache_kvarn *>(wavefront_hybrid->get_mem_attn_base()) : nullptr;
        bool single_seq = batch.n_tokens > 0 && batch.n_seq_id && batch.seq_id && batch.seq_id[0];
        for (int32_t i = 0; single_seq && i < batch.n_tokens; ++i) {
            single_seq = batch.n_seq_id[i] == 1 && batch.seq_id[i] &&
                    batch.seq_id[i][0] == batch.seq_id[0][0];
        }
        const bool plain_tokens = std::all_of(batch_inp.tokens.begin(), batch_inp.tokens.end(),
                [](const auto & token) { return !token.has_embd && !token.has_state && token.decision_order == 0; });
        refine_wavefront = plain_tokens && single_seq && kvarn && cparams.n_rs_seq == 0;
        if (refine_wavefront) effective_ubatch = cparams.n_ubatch;
    }
    const auto eligible_parent = [&](const llama_ubatch & ub) {
        const auto candidate = graph_params(nullptr, ub, nullptr, ctx_type_to_graph_type(cparams.ctx_type));
        const char * reason = rkva_wavefront_fallback(candidate, true);
        if (reason && ub.n_tokens > 1) {
            LLAMA_LOG_INFO("RKVA wavefront prefix: requested_ubatch=%u effective_ubatch=256 reason=%s\n",
                    cparams.n_ubatch, reason);
        }
        return reason == nullptr;
    };

    while (true) {
        mctx = refine_wavefront ?
                wavefront_hybrid->init_batch(*balloc, effective_ubatch, embd_all, eligible_parent) :
                memory->init_batch(*balloc, effective_ubatch, embd_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true) == LLAMA_MEMORY_STATUS_SUCCESS) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    mctx.reset();
                    if (grow_dflash_swa()) {
                        continue;
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    // start a new sampling transaction for this logical batch
    for (const auto & entry : sampling.samplers) {
        llama_sampler_backend_begin(entry.second);
    }

    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev  = 0;

    bool has_next_ubatch = false;

    do {
        const auto & ubatch = mctx->get_ubatch();

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        ggml_status status;

        const auto * res = process_ubatch(ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), status);

        if (!res) {
            if (wavefront_parent_active || wavefront_committed) {
                // n_rs_seq==0 commits each tile in place. A partial tiled parent
                // cannot be rolled back to its beginning with seq_rm(). Retire
                // all RPC/DMA and local work before clearing either state arena.
                if (backend_remote) ggml_backend_remote_attn_drain(backend_remote);
                ggml_backend_sched_synchronize(sched.get());
                if (backend_remote) ggml_backend_remote_attn_invalidate_range(backend_remote);
                memory->clear(true);
                wavefront_poisoned = true;
                LLAMA_LOG_ERROR("RKVA wavefront context failed: entire context cleared and poisoned; create/restore a fresh context\n");
                // An abort after in-place recurrent commits is not a KV-space
                // retry. This also covers the legacy tail/decode after tiles.
                return status == GGML_STATUS_ALLOC_FAILED ? -2 : -3;
            }
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            bool remote_rollback_failed = false;
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                const llama_rollback_result rollback = llama_rollback_sequence_after_decode_failure(
                    [&](llama_seq_id seq, llama_pos p0, llama_pos p1) {
                        if (p0 < 0 && p1 < 0) {
                            return memory->seq_rm(seq, p0, p1) && memory->run_on_seq_rm(seq, p0, p1);
                        }
                        return memory->seq_rm(seq, p0, p1);
                    }, s, pos_min[s]);
                if (rollback == LLAMA_ROLLBACK_FULL) {
                    LLAMA_LOG_WARN("%s: suffix removal refused for seq_id=%d; invalidated the whole sequence\n", __func__, s);
                    remote_rollback_failed = true;
                } else if (rollback == LLAMA_ROLLBACK_NONE) {
                    LLAMA_LOG_ERROR("%s: failed to invalidate seq_id=%d after decode failure\n", __func__, s);
                    remote_rollback_failed = true;
                }
            }

            const bool remote_backend_failed = backend_remote != nullptr &&
                    ggml_backend_remote_attn_failed(backend_remote);
            switch (status) {
                // A remote/rollback failure must not be retried as a batch-size
                // issue by the server; report it as a distinct compute error.
                // A failed remote backend is terminal even if local seq_rm
                // succeeded, so it must not fall into the retryable path.
                case GGML_STATUS_ABORTED:      return llama_decode_aborted_code(remote_rollback_failed, remote_backend_failed);
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits  = res->get_logits();
        // Explicit DFlash split injections only update KV. Keep the graph's
        // embedding tensor for pooling, but do not copy unused feature rows
        // back to the host. Noise blocks still return all normalized rows.
        const bool split_injection = cparams.dflash_split && ubatch.embd && !ubatch.token;
        auto * t_embd    = cparams.embeddings && !split_injection ? res->get_embd() : nullptr;
        auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn()  : nullptr;

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract multi-layer capture embeddings, concatenated per output position.
        // capture always uses the masked (output-row) layout, so t_h_capture is
        // [n_capture * n_embd, n_outputs]; copy in one shot per ubatch.
        if (embd_capture.data && cparams.n_capture_layers > 0 && n_outputs > 0 &&
            cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
            ggml_tensor * t_cap = res->get_h_capture();
            if (!t_cap) {
                LLAMA_LOG_ERROR("%s: target graph did not produce requested capture layers\n", __func__);
                return -1;
            }
            ggml_backend_t backend_c = ggml_backend_sched_get_tensor_backend(sched.get(), t_cap);
            GGML_ASSERT(backend_c != nullptr);
            const uint32_t row              = cparams.n_capture_layers * hparams.n_embd;
            float *        embd_capture_out = embd_capture.data + (size_t) n_outputs_prev * row;
            GGML_ASSERT((n_outputs_prev + n_outputs) * (int64_t) row <= (int64_t) embd_capture.size);
            ggml_backend_tensor_get_async(backend_c, t_cap, embd_capture_out, 0,
                                          (size_t) n_outputs * row * sizeof(float));
        }

        // extract logits
        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            float * logits_out = logits.data + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float * embd_out = embd.data + n_outputs_prev*n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        // [TAG_EXTRACT_TARGET_EMBEDDINGS]
        bool extract_all_idxs = extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens);

        // extract nextn embeddings before
        // only meaningful in LLAMA_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
        {
            const bool masked    = cparams.embeddings_nextn_masked;
            const int64_t n_rows = masked ? n_outputs       : (int64_t) ubatch.n_tokens;
            const int64_t offset = masked ? n_outputs_prev  : n_tokens_prev;

            if (embd_nextn.data && t_h_nextn && n_rows > 0 && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
                ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
                GGML_ASSERT(backend_h != nullptr);

                const uint32_t n_embd  = hparams.n_embd_out();
                float * embd_nextn_out = embd_nextn.data + offset*n_embd;

                GGML_ASSERT((offset + n_rows)*n_embd <= (int64_t) embd_nextn.size);
                ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float));
                if (!masked) {
                    n_nextn_copied += n_rows;
                }
                extract_all_idxs = extract_all_idxs || !masked;
            }
        }

        if (extract_all_idxs) {
            GGML_ASSERT(ubatch.data && ubatch.data->batch_idxs.size() == ubatch.n_tokens);
            GGML_ASSERT(embd_batch_idxs.size() == (size_t) n_tokens_prev);
            const auto & batch_idxs = ubatch.data->batch_idxs;
            embd_batch_idxs.insert(embd_batch_idxs.end(), batch_idxs.begin(), batch_idxs.end());
        }

        if (has_samplers) {
            const auto stride = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_rows(res->t_sampled,        sampling.sampled,    1,      n_outputs_prev, sched.get());
            copy_tensor_async_rows(res->t_sampled_logits, sampling.logits,     stride, n_outputs_prev, sched.get(), &sampling.logits_count);
            copy_tensor_async_rows(res->t_sampled_probs,  sampling.probs,      stride, n_outputs_prev, sched.get(), &sampling.probs_count);
            copy_tensor_async_rows(res->t_candidates,     sampling.candidates, stride, n_outputs_prev, sched.get(), &sampling.candidates_count);
        }

        n_outputs_prev += n_outputs;
        n_tokens_prev  += ubatch.n_tokens;

        has_next_ubatch = mctx->next();

        // MTP ubatches update the same KV cache and must complete in order.
        // The final ubatch must also finish before draft state can be inspected or checkpointed.
        if (cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
            synchronize();
        }
    } while (has_next_ubatch);

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    if (model.mtp_weights_info().managed && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT &&
            cparams.embeddings_nextn && !cparams.embeddings_nextn_masked && n_nextn_copied == n_tokens_all &&
            std::all_of(batch_inp.tokens.begin(), batch_inp.tokens.end(), [](const auto & t) { return t.id != LLAMA_TOKEN_NULL && !t.has_embd; }) &&
            nextn_decode_serial != std::numeric_limits<uint64_t>::max()) {
        // Allocation failure only makes this decode ineligible for bootstrap; it must
        // never turn otherwise successful inference into an exception across the C API.
        try {
            nextn_decoded_batch.clear();
            for (const auto & token : batch_inp.tokens) {
                if (token.seq_ids.size() != 1 || *token.seq_ids.begin() != 0) {
                    break;
                }
                nextn_decoded_batch.emplace_back(token.id, token.pos[0]);
            }
            if (nextn_decoded_batch.size() == batch_inp.tokens.size()) {
                nextn_decode_id = ++nextn_decode_serial;
            }
        } catch (const std::bad_alloc &) {
            nextn_decoded_batch.clear();
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    nextn_decode_id = 0;
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd     = hparams.n_embd;
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits = !cparams.dflash_split && !cparams.dflash_selector_only && (!cparams.embeddings || graph_has_logits);
    bool has_embd       = cparams.embeddings;
    bool has_embd_nextn = cparams.embeddings_nextn;
    bool has_embd_capture = cparams.n_capture_layers > 0;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }

    size_t backend_float_count = 0;
    size_t backend_token_count = 0;
    size_t embd_layer_inp_float_count = 0;

    logits.size     = has_logits     ? n_vocab*n_outputs_max     : 0;
    embd.size       = has_embd       ? n_embd_out*n_outputs_max  : 0;
    embd_nextn.size = has_embd_nextn ? n_embd_out*n_outputs_max  : 0;

    if (has_embd_nextn && !cparams.embeddings_nextn_masked) {
        // unmasked: nextn row exists for every token in the batch, not just
        // those flagged via batch.logits[i] -> size by token count instead.
        embd_nextn.size = (size_t) n_embd_out * n_batch;
    }

    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled) {
            embd_layer_inp_float_count += (size_t) n_embd * n_batch;
        }
    }

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    embd_capture.size      = has_embd_capture ? (size_t) cparams.n_capture_layers * hparams.n_embd * n_outputs_max : 0;
    const size_t new_size  = (logits.size + embd.size + embd_nextn.size + embd_capture.size +
                              embd_layer_inp_float_count + backend_float_count) *
                                 sizeof(float) +
                             (backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits.data = nullptr;
            embd.data = nullptr;
            embd_nextn.data = nullptr;
            embd_capture.data = nullptr;
            for (auto & layer_inp : embd_layer_inp) {
                layer_inp = {nullptr, 0};
            }
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output.get(), 0);
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{output_base, logits.size} : buffer_view<float>{nullptr, 0};
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{(float *) (base + offset), embd.size} : buffer_view<float>{nullptr, 0};
    offset += embd.size * sizeof(float);

    embd_nextn = has_embd_nextn ? buffer_view<float>{(float *) (base + offset), embd_nextn.size} : buffer_view<float>{nullptr, 0};
    offset += embd_nextn.size * sizeof(float);

    for (uint32_t il = 0; il < embd_layer_inp.size(); ++il) {
        if (cparams.embeddings_layer_inp[il]) {
            embd_layer_inp[il] = buffer_view<float>{(float *) (base + offset), (size_t) n_embd * n_batch};
            offset += embd_layer_inp[il].size * sizeof(float);
        } else {
            embd_layer_inp[il] = buffer_view<float>{nullptr, 0};
        }
    }

    embd_capture = has_embd_capture ? buffer_view<float>{ (float *) (base + offset), embd_capture.size } :
                                      buffer_view<float>{ nullptr, 0 };
    offset += embd_capture.size * sizeof(float);

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max);

    return n_outputs_max;
}

bool llama_context::extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens) {
    bool extracted = false;
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }
        if (!embd_layer_inp[il].has_data()) {
            GGML_ABORT("output layer input buffer not allocated");
        }
        ggml_tensor * t = res->get_layer_inp((int) il);
        if (!t) {
            GGML_ABORT("layer input tensor not found");
        }

        const size_t nbytes = ggml_nbytes(t);
        const size_t nfloats = nbytes / sizeof(float);
        GGML_ASSERT(n_tokens > 0);
        GGML_ASSERT(nfloats % n_tokens == 0);

        const size_t row_floats = nfloats / n_tokens;
        GGML_ASSERT(row_floats == model.hparams.n_embd);
        const size_t dst_offset = token_offset * row_floats;
        GGML_ASSERT(dst_offset + nfloats <= embd_layer_inp[il].size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), t);
        GGML_ASSERT(backend != nullptr);
        // Tensor-split backends require a zero source offset.
        ggml_backend_tensor_get_async(backend, t, embd_layer_inp[il].data + dst_offset, 0, nbytes);
        extracted = true;
    }
    return extracted;
}

void llama_context::output_reorder() {
    const uint64_t n_vocab     = model.vocab.n_tokens();
    const uint64_t n_embd      = model.hparams.n_embd;
    const uint64_t n_embd_out  = model.hparams.n_embd_out();

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0*n_vocab + k], logits.data[i1*n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd.data[i0*n_embd_out + k], embd.data[i1*n_embd_out + k]);
            }
        }

        if (embd_capture.size > 0) {
            const uint64_t row = (uint64_t) cparams.n_capture_layers * n_embd;
            for (uint64_t k = 0; k < row; k++) {
                std::swap(embd_capture.data[i0 * row + k], embd_capture.data[i1 * row + k]);
            }
        }
        if (embd_nextn.size > 0 && cparams.embeddings_nextn_masked) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd_nextn.data[i0*n_embd_out + k], embd_nextn.data[i1*n_embd_out + k]);
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();

    // [TAG_EXTRACT_TARGET_EMBEDDINGS]
    // Layer inputs and unmasked NextN embeddings contain all token rows, independent of logits selection.
    for (size_t i = 0; i < embd_batch_idxs.size(); ++i) {
        while (embd_batch_idxs[i] != (int32_t) i) {
            const int32_t j = embd_batch_idxs[i];
            GGML_ASSERT(j >= 0 && (size_t) j < embd_batch_idxs.size());
            if (embd_nextn.has_data() && !cparams.embeddings_nextn_masked) {
                for (size_t k = 0; k < n_embd_out; ++k) {
                    std::swap(embd_nextn.data[i*n_embd_out + k], embd_nextn.data[j*n_embd_out + k]);
                }
            }
            for (auto & layer : embd_layer_inp) {
                if (layer.has_data()) {
                    for (size_t k = 0; k < n_embd; ++k) {
                        std::swap(layer.data[i*n_embd + k], layer.data[j*n_embd + k]);
                    }
                }
            }
            std::swap(embd_batch_idxs[i], embd_batch_idxs[j]);
        }
    }
    embd_batch_idxs.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    // A generic precision-tail oracle can add compact-current staging, two
    // indexed gathers, body/tail score branches, and ordered tail writes per
    // attention layer. Keep this feature-only graph allowance independent of
    // the upstream base heuristic. Upstream's post-v0.4.3 attention graph adds
    // more per-layer staging nodes; 64 nodes per layer no longer contains the
    // complete compact route exercised by state restore.
    const uint32_t tail_nodes = (cparams.kv_tail_tokens > 0 || cparams.kv_tail_tokens_swa > 0) ?
            96u*model.hparams.n_layer_all : 0;
    uint32_t res;
    if (model.arch == LLM_ARCH_KIMI_K3 || model.arch == LLM_ARCH_GLM5_NEXT) {
        // the n_tokens*40 budget below is exhausted at ubatch 3840
        res = std::max<uint32_t>(n_tokens * 160, 64u * model.n_tensors()) + tail_nodes;
    } else if (model.arch == LLM_ARCH_HRM_TEXT) {
        // the 128-slot looped graph needs roughly one stack per token budget
        res = std::max<uint32_t>(n_tokens * 80, 64u * model.n_tensors()) + tail_nodes;
    } else if (model.arch == LLM_ARCH_QWEN3NEXT ||
        model.arch == LLM_ARCH_KIMI_LINEAR ||
        model.arch == LLM_ARCH_BAILINGMOE3 ||
        model.arch == LLM_ARCH_QWEN35 ||
        model.arch == LLM_ARCH_QWEN35MOE ||
        model.arch == LLM_ARCH_CLEF ||
        model.arch == LLM_ARCH_QWEN4EXP ||
        model.arch == LLM_ARCH_DEEPSEEK4 ||
        (model.arch == LLM_ARCH_DFLASH && model.hparams.dsv4_hc_mult > 0) ||
        model.arch == LLM_ARCH_NANBEIGE ||
        model.arch == LLM_ARCH_MINIMAX_01 ||
        model.arch == LLM_ARCH_MINIMAX_M3 ||
        model.arch == LLM_ARCH_HY_V4) {
        res = std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors()) + tail_nodes;
    } else if (model.arch == LLM_ARCH_DFLASH && model.hparams.dflash_selector_rank > 0) {
        // DFlash2's convolutions and selector are shape work rather than matmuls,
        // so they cost ~8.6 nodes per tensor against ~5.9 for a plain DFlash draft
        res = std::max<uint32_t>(1024u, 12u*model.n_tensors()) + tail_nodes;
    } else {
        res = std::max<uint32_t>(1024u, 8u*model.n_tensors()) + tail_nodes;
        for (const auto & lora : model.loras) {
            res += lora->get_n_nodes();
        }
    }

    if (model.arch == LLM_ARCH_DFLASH && model.hparams.dflash_selector_rank > 0) {
        const uint32_t selector_tokens = std::min<uint32_t>(
                n_tokens, model.hparams.dflash_block_size * cparams.n_seq_max);
        res += 32*selector_tokens;
    }

    // every sampler builds n_outputs_max_per_seq chains, see llm_graph_context::build_sampling
    for (const auto & [seq_id, sampler] : sampling.samplers) {
        res += llama_sampler_backend_n_nodes(sampler) * cparams.n_outputs_max_per_seq;
    }

    if (cparams.training) {
        res *= 4;
    }
    if (cparams.rkva_wavefront_tile) {
        const uint64_t tiles = (uint64_t(n_tokens) + cparams.rkva_wavefront_tile - 1)/cparams.rkva_wavefront_tile;
        const uint64_t expanded = uint64_t(res)*std::max<uint64_t>(1, tiles) + 64*tiles;
        if (expanded > UINT32_MAX) throw std::runtime_error("wavefront graph node budget exceeds uint32");
        res = uint32_t(expanded);
    }
    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

llm_graph_result * llama_context::get_gf_res_prev() {
    auto & res = gf_res_prev[n_outputs > 0];
    if (!res) {
        res.reset(new llm_graph_result(gf_res_reserve->get_max_nodes()));
    }
    return res.get();
}

// pack sampler outputs into as few sequences as possible before using sequences without samplers
static void ubatch_prepare_reserve(
              llama_ubatch                            & ubatch,
              uint32_t                                  n_outputs,
        const std::map<llama_seq_id, llama_sampler *> & samplers,
              uint32_t                                  n_outputs_max_per_seq) {
    const uint32_t n_seqs       = ubatch.n_seqs;
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t t = 0; t < n_seq_tokens; ++t) {
            const uint32_t i = s * n_seq_tokens + t;
            ubatch.n_seq_id[i] = 1;
            ubatch.seq_id[i] = &ubatch.seq_id_unq[s];
        }
    }

    // sequences with a sampler that fit in this ubatch
    std::vector<uint32_t> sampler_seqs;
    std::vector<bool> has_sampler(n_seqs, false);
    for (const auto & entry : samplers) {
        const llama_seq_id seq_id = entry.first;
        if (seq_id < 0 || (uint32_t) seq_id >= n_seqs) {
            continue;
        }

        sampler_seqs.push_back(seq_id);
        has_sampler[seq_id] = true;
    }

    uint32_t n_outputs_set = 0;

    const uint32_t n_outputs_per_seq = std::min(n_seq_tokens, n_outputs_max_per_seq);
    for (uint32_t s : sampler_seqs) {
        if (n_outputs_set >= n_outputs) {
            break;
        }

        for (uint32_t t = 0; t < n_outputs_per_seq && n_outputs_set < n_outputs; ++t) {
            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }

    // use sequences without samplers for any remaining outputs
    for (uint32_t t = 0; t < n_seq_tokens && n_outputs_set < n_outputs; ++t) {
        for (uint32_t s = 0; s < n_seqs && n_outputs_set < n_outputs; ++s) {
            if (has_sampler[s]) {
                continue;
            }

            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes) {
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse old graphs, so we reset the previous graph results
    for (auto & res : gf_res_prev) {
        if (res) {
            res->reset();
        }
    }
    gf_res_prev_active = nullptr;

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    ubatch_prepare_reserve(ubatch, n_outputs, sampling.samplers, cparams.n_outputs_max_per_seq);

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type));

    res->reset();

    // F4: same boundary rule as process_ubatch. The reserve graph is built with a
    // memory context too, so the K/V store slots must be dropped before it reads
    // them; otherwise a stale slot from a previous ubatch is reachable here too.
    if (mctx) {
        mctx->reset_graph_store_slots();
    }

    auto * gf = model.build_graph(gparams);

    // check the graph before scheduling: cross-backend copies break the producer chain that the check follows
    if (!hadamard_verified && gf && (!model.hdmd.rot.empty() || !model.hdmd.inv.empty())) {
        llama_verify_hadamard_graph(gf, model.hdmd.rot, model.hdmd.inv, moe_cache.get());
        hadamard_verified = true;
    }

    this->n_input_tensors = llama_graph_n_input_tensors(gf);
    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const {
    llm_graph_params result = {
        /*.arch        =*/ model.arch,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ sched.get(),
        /*.backend_cpu =*/ backend_cpu,
        /*.backend_remote =*/ backend_remote,
        /*.cvec        =*/ cvec.get(),
        /*.loras       =*/ loras.get(),
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.dspark_ctx  =*/&dspark_ctx,
        /*.dspark_has_context =*/!dspark_ctx.v_ctx_feat.empty(),
        /*.dspark_ctx_rows =*/dspark_ctx.n_ctx_rows,
        /*.dspark_ctx_width =*/dspark_ctx.n_embd_cap,
        /*.moe_cache   =*/ moe_cache.get(),
        /*.prec_policy =*/ &model.prec_policy,
        /*.hdmd        =*/ model.hdmd.rot.empty() ? nullptr : &model.hdmd,
        /*.samplers    =*/ sampling.samplers,
        /*.n_outputs   =*/ n_outputs,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
    };
    if (!rkva_wavefront_fallback(result)) {
        result.wavefront_tile_size = cparams.rkva_wavefront_tile;
    }
    return result;
}

void llama_context::sched_graph_realloc(ggml_backend_sched_t sched, ggml_cgraph * gf, bool backend_ids_changed, bool graph_size_changed, void * user_data) {
    // reallocating after the backend assignments changed is expected, trace only the unexpected ones
    if (backend_ids_changed) {
        return;
    }

    auto * lctx = static_cast<llama_context *>(user_data);

    if (graph_size_changed) {
        lctx->n_reallocs_diff++;
    } else {
        lctx->n_reallocs_same++;
    }

    LLAMA_LOG_INFO("%s: graph reallocation: nodes = %d (changed = %d), splits = %d, n_same = %d, n_diff = %d\n",
            __func__, ggml_graph_n_nodes(gf), graph_size_changed, ggml_backend_sched_get_n_splits(sched),
            lctx->n_reallocs_same, lctx->n_reallocs_diff);
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    copy_experts.reset();

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        // Finish writes into graph tensors before rollback or buffer reuse.
        ggml_backend_remote_attn_drain(backend_remote);
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }
    // New backend work is now in flight; the next synchronize() must run the barrier.
    outputs_synced = false;

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

bool llama_context::sched_copy_experts(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, void * user_data) {
    auto * lctx = static_cast<llama_context *>(user_data);

    // the slot maps of the MoE cache
    if (lctx->moe_cache && lctx->moe_cache->copy(backend, src, dst, graph)) {
        return true;
    }

    auto & st = lctx->copy_experts;

    // the ids must be computed before the split starts, so only the first node of the split is considered
    if (ggml_graph_n_nodes(graph) == 0) {
        return false;
    }
    const ggml_tensor * node = ggml_graph_node(graph, 0);
    if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] != dst) {
        return false;
    }

    const ggml_tensor * ids = node->src[2];
    if (ggml_nelements(ids) == 0) {
        return true;
    }

    const int64_t n_expert    = src->ne[2];
    const size_t  expert_size = src->nb[2];

    if (ids != st.ids || (int64_t) st.used.size() != n_expert) {
        st.ids_data.resize(ggml_nbytes(ids)/sizeof(int32_t));
        ggml_backend_tensor_get_async(backend, ids, st.ids_data.data(), 0, ggml_nbytes(ids));
        ggml_backend_synchronize(backend);

        st.used.assign(n_expert, false);
        for (int64_t i1 = 0; i1 < ids->ne[1]; i1++) {
            for (int64_t i0 = 0; i0 < ids->ne[0]; i0++) {
                const int32_t id = st.ids_data[i1*ids->nb[1]/sizeof(int32_t) + i0*ids->nb[0]/sizeof(int32_t)];
                GGML_ASSERT(id >= 0 && id < n_expert);
                st.used[id] = true;
            }
        }

        st.ids = ids;
    }

    // group consecutive experts and copy them together
    for (int64_t first = 0; first < n_expert; ) {
        if (!st.used[first]) {
            first++;
            continue;
        }
        int64_t last = first;
        while (last + 1 < n_expert && st.used[last + 1]) {
            last++;
        }

        // the experts in the MoE cache are copied from device memory, the others are uploaded
        int64_t next = first;
        for (int64_t e = first; e <= last && lctx->moe_cache; ) {
            const int64_t n = lctx->moe_cache->copy_experts(backend, src, dst, e, last);
            if (n == 0) {
                e++;
                continue;
            }
            if (next < e) {
                ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + next*expert_size, next*expert_size, (e - next)*expert_size);
            }
            e   += n;
            next = e;
        }

        // copy a bit extra to ensure there are no NaNs in the padding of the last expert, this is necessary for MMQ in the CUDA backend
        const size_t offset  = next*expert_size;
        const size_t padding = last < n_expert - 1 ? std::min<size_t>(expert_size, 512) : 0;
        const size_t size    = (last + 1 - next)*expert_size + padding;
        if (size > 0) {
            ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + offset, offset, size);
        }

        first = last + 1;
    }

    return true;
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        // DFlash2 requires a global vocabulary top-k. Tensor-parallel output
        // logits are vocabulary-axis split, so gather them through the CPU
        // scheduler boundary before selecting candidates.
        if (backend_cpu != nullptr && strcmp(name, "dflash2_logits_global") == 0) {
            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend_cpu);
        }

        // - norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // - force the last op of the layer on the specified backend to avoid running it on the backend of the next layer due to scheduling
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer_all;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0)) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

void llama_context::set_snapkv_prefill_end(llama_seq_id seq_id, llama_pos prefill_end) {
    if (memory) {
        memory->set_snapkv_prefill_end(seq_id, prefill_end);
    }
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy(bool skip_tensors) : skip_tensors(skip_tensors) {}

    bool counts_only() const override { return true; }

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        if (skip_tensors) {
            return;
        }

        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    const bool skip_tensors;

    size_t size_written = 0;
};

class llama_io_write_host : public llama_io_write_i {
public:
    llama_io_write_host(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_write_host() {
        // State layouts often interleave regularly-strided slices from several
        // tensors. Group those slices by source tensor so backends with native
        // 2D transfers can copy each run with one synchronization while keeping
        // the serialized byte layout unchanged.
        std::vector<const write_info *> sorted;
        sorted.reserve(winfos.size());
        for (const auto & winfo : winfos) {
            sorted.push_back(&winfo);
        }
        std::sort(sorted.begin(), sorted.end(), [](const write_info * a, const write_info * b) {
            if (a->tensor != b->tensor) {
                return std::less<ggml_tensor *>{}(a->tensor, b->tensor);
            }
            if (a->size != b->size) {
                return a->size < b->size;
            }
            if (a->offset != b->offset) {
                return a->offset < b->offset;
            }
            return a->ptr < b->ptr;
        });

        size_t i = 0;
        while (i < sorted.size()) {
            size_t group_end = i + 1;
            while (group_end < sorted.size() &&
                    sorted[group_end]->tensor == sorted[i]->tensor &&
                    sorted[group_end]->size == sorted[i]->size) {
                ++group_end;
            }

            while (i < group_end) {
                const write_info & first = *sorted[i];
                size_t run = 1;
                size_t source_stride = 0;
                size_t destination_stride = 0;
                if (i + 1 < group_end &&
                        sorted[i + 1]->offset > first.offset &&
                        sorted[i + 1]->ptr > first.ptr) {
                    source_stride = sorted[i + 1]->offset - first.offset;
                    destination_stride = size_t(sorted[i + 1]->ptr - first.ptr);
                    if (source_stride >= first.size && destination_stride >= first.size) {
                        run = 2;
                        while (i + run < group_end &&
                                sorted[i + run]->offset == first.offset + run*source_stride &&
                                sorted[i + run]->ptr == first.ptr + run*destination_stride) {
                            ++run;
                        }
                    }
                }

                if (run > 1) {
                    ggml_backend_tensor_get_2d(
                            first.tensor, first.ptr, first.offset, first.size,
                            run, source_stride, destination_stride);
                } else {
                    ggml_backend_tensor_get(first.tensor, first.ptr, first.offset, first.size);
                }
                i += run;
            }
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;
};

class llama_io_read_host : public llama_io_read_i {
public:
    llama_io_read_host(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_read_host() {
        cancel();
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        rinfos.push_back({tensor, ptr, {}, size, offset, false});

        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void stage_tensor_set(ggml_tensor * tensor, const void * src, size_t offset, size_t size) override {
        read_info info { tensor, nullptr, {}, size, offset, false };
        info.owned.resize(size);
        memcpy(info.owned.data(), src, size);
        rinfos.push_back(std::move(info));
    }

    void stage_tensor_clear(ggml_tensor * tensor, size_t offset, size_t size) override {
        rinfos.push_back({ tensor, nullptr, {}, size, offset, true });
    }

    void on_commit(std::function<void()> callback) override {
        callbacks.push_back(std::move(callback));
    }

    void commit() override {
        for (const auto & rinfo : rinfos) {
            if (rinfo.clear) {
                ggml_backend_tensor_memset(rinfo.tensor, 0, rinfo.offset, rinfo.size);
            } else {
                const void * src = rinfo.owned.empty() ? rinfo.ptr : rinfo.owned.data();
                ggml_backend_tensor_set(rinfo.tensor, src, rinfo.offset, rinfo.size);
            }
        }
        for (auto & callback : callbacks) {
            callback();
        }
        cancel();
    }

    void cancel() override {
        rinfos.clear();
        callbacks.clear();
    }

    size_t n_bytes() override {
        return size_read;
    }
    size_t remaining() const override {
        return buf_size;
    }
    bool peek(void * dst, size_t size) override {
        if (size > buf_size) {
            return false;
        }
        memcpy(dst, ptr, size);
        return true;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        std::vector<uint8_t> owned;
        size_t size;
        size_t offset;
        bool clear;
    };
    std::vector<read_info> rinfos;
    std::vector<std::function<void()>> callbacks;
};

bool llama_context::grow_dflash_swa() {
    if (model.arch != LLM_ARCH_DFLASH) {
        return false;
    }
    auto * iswa = dynamic_cast<llama_kv_cache_iswa *>(memory.get());
    if (!iswa) {
        return false;
    }
    synchronize();
    ggml_backend_sched_reset(sched.get());
    for (auto & res : gf_res_prev) {
        if (res) {
            res->reset();
        }
    }
    gf_res_prev_active = nullptr;
    try {
        return iswa->grow_swa([](llama_memory_i & source, llama_memory_i & destination) {
            llama_io_write_dummy sizing(false);
            source.state_write(sizing, -1, 0);
            std::vector<uint8_t> state(sizing.n_bytes());
            {
                llama_io_write_host writer(state.data(), state.size());
                source.state_write(writer, -1, 0);
                if (writer.n_bytes() != state.size()) {
                    throw std::runtime_error("DFlash SWA growth state size changed");
                }
            }
            llama_io_read_host reader(state.data(), state.size());
            destination.state_read(reader, -1, 0);
            if (reader.n_bytes() != state.size()) {
                throw std::runtime_error("DFlash SWA growth state was not fully consumed");
            }
            reader.commit();
        });
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: could not grow DFlash SWA cache: %s\n", __func__, err.what());
        return false;
    }
}

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(std::min(size, LLAMA_STATE_FILE_BUFFER_SIZE));
        while (size) {
            const size_t chunk = std::min(size, temp_buffer.size());
            ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, chunk);
            write(temp_buffer.data(), chunk);
            offset += chunk;
            size -= chunk;
        }
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    ~llama_io_read_file() {
        cancel();
    }

    void read(void * dst, size_t size) override {
        check_file_remaining(size);
        if (size != 0 && dst == nullptr) {
            throw std::runtime_error("invalid null destination for sequence state read");
        }
        file->read_raw(dst, size);
        size_read += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        check_tensor_span(tensor, offset, size);
        check_file_remaining(size);
        read_info info { tensor, {}, size, offset, false };
        info.data.resize(size);
        read(info.data.data(), size);
        rinfos.push_back(std::move(info));
    }

    void stage_tensor_set(ggml_tensor * tensor, const void * src, size_t offset, size_t size) override {
        check_tensor_span(tensor, offset, size);
        if (size != 0 && src == nullptr) {
            throw std::runtime_error("invalid null source for sequence state tensor staging");
        }
        read_info info { tensor, {}, size, offset, false };
        info.data.resize(size);
        if (size != 0) {
            memcpy(info.data.data(), src, size);
        }
        rinfos.push_back(std::move(info));
    }

    void stage_tensor_clear(ggml_tensor * tensor, size_t offset, size_t size) override {
        check_tensor_span(tensor, offset, size);
        rinfos.push_back({ tensor, {}, size, offset, true });
    }

    void on_commit(std::function<void()> callback) override {
        callbacks.push_back(std::move(callback));
    }

    void commit() override {
        for (const auto & rinfo : rinfos) {
            if (rinfo.clear) {
                ggml_backend_tensor_memset(rinfo.tensor, 0, rinfo.offset, rinfo.size);
            } else {
                ggml_backend_tensor_set(rinfo.tensor, rinfo.data.data(), rinfo.offset, rinfo.size);
            }
        }
        for (auto & callback : callbacks) {
            callback();
        }
        cancel();
    }

    void cancel() override {
        rinfos.clear();
        callbacks.clear();
    }

    size_t n_bytes() override {
        return size_read;
    }
    size_t remaining() const override {
        const size_t position = file->tell();
        const size_t end = file->size();
        return position <= end ? end - position : 0;
    }
    bool peek(void * dst, size_t size) override {
        const size_t position = file->tell();
        const size_t end = file->size();
        if (position > end || size > end - position) {
            return false;
        }
        file->read_raw(dst, size);
        file->seek(position, SEEK_SET);
        return true;
    }

private:
    void check_file_remaining(size_t size) const {
        const size_t position = file->tell();
        const size_t end = file->size();
        if (position > end || size > end - position) {
            throw std::runtime_error(
                    "truncated sequence state file [class=file_remaining_check"
                    " position=" + std::to_string(position) +
                    " end=" + std::to_string(end) +
                    " requested=" + std::to_string(size) +
                    " available=" + std::to_string(end > position ? end - position : 0) + "]");
        }
    }

    static void check_tensor_span(ggml_tensor * tensor, size_t offset, size_t size) {
        if (!tensor || offset > ggml_nbytes(tensor) || size > ggml_nbytes(tensor) - offset) {
            throw std::runtime_error("sequence state tensor range is outside the destination");
        }
    }

    llama_file * file;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        std::vector<uint8_t> data;
        size_t size;
        size_t offset;
        bool clear;
    };
    std::vector<read_info> rinfos;
    std::vector<std::function<void()>> callbacks;
};

class llama_io_write_device : public llama_io_write_i {
public:
    llama_io_write_device(uint8_t * p, size_t len, llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs)  {
    }

    ~llama_io_write_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += winfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ 2*mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
            mbuf.cpy.reserve(mbuf.n_tensors);
        }

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            GGML_ASSERT(winfo.size % ggml_type_size(winfo.tensor->type) == 0);
            const int64_t n = (winfo.size/ggml_type_size(winfo.tensor->type))*ggml_blck_size(winfo.tensor->type);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d      (mbuf.ctx.get(), winfo.tensor, n, winfo.offset));
            mbuf.cpy.push_back(ggml_new_tensor_1d(mbuf.ctx.get(), winfo.tensor->type, n));
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            auto & mbuf_cur = mbufs[buft];

            bool need_alloc = false;

            need_alloc = need_alloc || (!mbuf_cur.buf);
            need_alloc = need_alloc || (mbuf_cur.org.size() != mbuf.org.size());
            need_alloc = need_alloc || (mbuf_cur.total_size != mbuf.total_size);

            if (!need_alloc) {
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    auto * org0 = mbuf_cur.org[i];
                    auto * org1 = mbuf.org[i];

                    if (!ggml_are_same_shape(org0, org1)) {
                        need_alloc = true;
                        break;
                    }

                    if (org0->view_src != org1->view_src || org0->view_offs != org1->view_offs) {
                        need_alloc = true;
                        break;
                    }
                }
            }

            if (need_alloc) {
                if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                    mbuf_cur = std::move(mbuf);

                    mbuf_cur.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx.get(), buft));

                    LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);
                } else {
                    //LLAMA_LOG_INFO("%s: reallocating tensors in '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);

                    // save the old buffer and allocate the new tensors in it
                    auto buf = std::move(mbuf_cur.buf);

                    mbuf_cur = std::move(mbuf);

                    ggml_tallocr talloc = ggml_tallocr_new(buf.get());

                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_view_init(mbuf_cur.org[i]);
                        ggml_tallocr_alloc(&talloc, mbuf_cur.cpy[i]);
                    }

                    mbuf_cur.buf = std::move(buf);
                }
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.org[i], mbuf_cur.cpy[i]);
            }
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;

    llama_memory_buffers & mbufs;
};

class llama_io_read_device : public llama_io_read_i {
public:
    llama_io_read_device(const uint8_t * p, size_t len, const llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs) {
    }

    ~llama_io_read_device() {
        cancel();
    }

    void commit() override {
        llama_memory_buffers mbufs_new;

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += rinfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
        }

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            GGML_ASSERT(rinfo.size % ggml_type_size(rinfo.tensor->type) == 0);
            const int64_t n = (rinfo.size/ggml_type_size(rinfo.tensor->type))*ggml_blck_size(rinfo.tensor->type);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d(mbuf.ctx.get(), rinfo.tensor, n, rinfo.offset));

            ggml_backend_view_init(mbuf.org.back());
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            const auto & mbuf_cur = mbufs.at(buft);

            if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                GGML_ABORT("%s: memory buffer mismatch\n", __func__);
            }

            if (mbuf_cur.n_tensors == mbuf.n_tensors) {
                // an equal tensor count does not imply the same chunking, e.g. save ranges [2,1] vs restore runs [1,2]
                bool same_chunking = true;
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    if (ggml_nbytes(mbuf_cur.cpy[i]) != ggml_nbytes(mbuf.org[i])) {
                        same_chunking = false;
                        break;
                    }
                }

                if (same_chunking) {
                    // same chunking: copy 1:1 by index
                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_tensor_copy(mbuf_cur.cpy[i], mbuf.org[i]);
                    }
                    continue;
                }
            }

            // different chunking: copy the write-side data (mbuf_cur.cpy) into the read-side targets (mbuf.org)
            // with a byte cursor. Write and read enumerate the same logical data in the same order but may chunk
            // it differently (even with an equal number of tensors), so copy across tensor boundaries rather than
            // 1:1 by index.
            const size_t total = mbuf_cur.total_size;

            ggml_init_params params_scratch = {
                /*.mem_size   =*/ 2*(mbuf_cur.cpy.size() + mbuf.org.size())*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx_scratch = ggml_init(params_scratch);

            size_t src_pos  = 0;
            size_t dst_pos  = 0;
            size_t src_j    = 0;
            size_t dst_i    = 0;
            size_t src_base = 0;
            size_t dst_base = 0;

            while (src_pos < total) {
                const auto & src_t = mbuf_cur.cpy[src_j];
                const auto & dst_t = mbuf.org[dst_i];

                const size_t src_size = ggml_nbytes(src_t);
                const size_t dst_size = ggml_nbytes(dst_t);

                const size_t src_off  = src_pos - src_base;
                const size_t dst_off  = dst_pos - dst_base;

                const size_t n_copy = std::min(src_size - src_off, dst_size - dst_off);

                const size_t   el   = ggml_element_size(src_t);
                const int64_t n_el = (int64_t) (n_copy / el);

                auto * src_v = ggml_view_1d(ctx_scratch, src_t, n_el, src_off);
                ggml_backend_view_init(src_v);
                auto * dst_v = ggml_view_1d(ctx_scratch, dst_t, n_el, dst_off);
                ggml_backend_view_init(dst_v);

                ggml_backend_tensor_copy(src_v, dst_v);

                src_pos += n_copy;
                dst_pos += n_copy;

                if (src_pos - src_base == src_size) {
                    src_base = src_pos;
                    ++src_j;
                }
                if (dst_pos - dst_base == dst_size) {
                    dst_base = dst_pos;
                    ++dst_i;
                }
            }

            GGML_ASSERT(src_pos == total && dst_pos == total);
            // any tensors left unvisited hold no data
            for (size_t i = src_j; i < mbuf_cur.cpy.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf_cur.cpy[i]) == 0);
            }
            for (size_t i = dst_i; i < mbuf.org.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf.org[i]) == 0);
            }

            ggml_free(ctx_scratch);
        }

        GGML_ASSERT(buf_size == 0);

        for (const auto & operation : host_operations) {
            if (operation.clear) {
                ggml_backend_tensor_memset(operation.tensor, 0, operation.offset, operation.size);
            } else {
                ggml_backend_tensor_set(operation.tensor, operation.data.data(), operation.offset, operation.size);
            }
        }
        for (auto & callback : callbacks) {
            callback();
        }
        cancel();
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        rinfos.push_back({tensor, ptr, size, offset});
    }

    void stage_tensor_set(ggml_tensor * tensor, const void * src, size_t offset, size_t size) override {
        host_operation operation { tensor, {}, size, offset, false };
        operation.data.resize(size);
        memcpy(operation.data.data(), src, size);
        host_operations.push_back(std::move(operation));
    }

    void stage_tensor_clear(ggml_tensor * tensor, size_t offset, size_t size) override {
        host_operations.push_back({ tensor, {}, size, offset, true });
    }

    void on_commit(std::function<void()> callback) override {
        callbacks.push_back(std::move(callback));
    }

    void cancel() override {
        rinfos.clear();
        host_operations.clear();
        callbacks.clear();
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;

    struct host_operation {
        ggml_tensor * tensor;
        std::vector<uint8_t> data;
        size_t size;
        size_t offset;
        bool clear;
    };
    std::vector<host_operation> host_operations;
    std::vector<std::function<void()>> callbacks;

    const llama_memory_buffers & mbufs;
};

struct llama_state_seq_restore_plan {
    std::unique_ptr<llama_io_read_i> io;
    size_t bytes = 0;
};

size_t llama_context::state_get_size(llama_state_seq_flags flags) {
    llama_io_write_dummy io(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    try {
        return state_write_data(io, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        LLAMA_LOG_ERROR("%s: on-device full-context state is not supported\n", __func__);
        return 0;
    }
    if (llama_memory_status_is_fail(memory_update(false))) {
        return 0;
    }
    llama_io_write_host io(dst, size);
    try {
        return state_write_data(io, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        LLAMA_LOG_ERROR("%s: on-device full-context state is not supported\n", __func__);
        return 0;
    }
    GGML_UNUSED(flags);
    nextn_decode_id = 0;
    llama_io_read_host io(src, size);
    try {
        const size_t nread = state_read_data(io);
        io.commit();
        return nread;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io.discard();
        return 0;
    }
}

static constexpr uint32_t io_magic = 0xaf143cd8;

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    // A context without sequence memory (e.g. diffusion architectures) still
    // exposes upstream's valid header-only sequence state; the shared-stream
    // guard only applies when a memory to share exists.
    if (seq_id < 0 || uint32_t(seq_id) >= cparams.n_seq_max || (memory && !memory->state_seq_can_save(seq_id, flags))) {
        static std::atomic_flag warned = ATOMIC_FLAG_INIT;
        if (!warned.test_and_set()) {
            LLAMA_LOG_WARN("%s: sequence %d cannot be saved while its physical KV stream is shared\n", __func__, seq_id);
        }
        return 0;
    }
    llama_io_write_dummy io(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    if (seq_id < 0 || uint32_t(seq_id) >= cparams.n_seq_max || (memory && !memory->state_seq_can_save(seq_id, flags))) {
        static std::atomic_flag warned = ATOMIC_FLAG_INIT;
        if (!warned.test_and_set()) {
            LLAMA_LOG_WARN("%s: sequence %d cannot be saved while its physical KV stream is shared\n", __func__, seq_id);
        }
        return 0;
    }
    if (llama_memory_status_is_fail(memory_update(false))) {
        return 0;
    }
    std::unique_ptr<llama_io_write_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        io = std::make_unique<llama_io_write_device>(dst, size, mem_storage[seq_id]);
    } else {
        io = std::make_unique<llama_io_write_host>(dst, size);
    }

    try {
        io->write(&io_magic, sizeof(io_magic));
        io->write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
if (seq_id < 0 || uint32_t(seq_id) >= cparams.n_seq_max || !memory || !memory->state_seq_can_restore(seq_id, flags)) {
        static std::atomic_flag warned = ATOMIC_FLAG_INIT;
        if (!warned.test_and_set()) {
            LLAMA_LOG_WARN("%s: sequence %d cannot be restored while its physical KV stream is shared\n", __func__, seq_id);
        }
        return 0;
    }
    nextn_decode_id = 0;
    try {
        std::unique_ptr<llama_io_read_i> io;
        if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
            // Read the host header first to select the matching device storage.
            llama_io_read_host header(src, size);
            uint32_t magic_read;
            header.read(&magic_read, sizeof(magic_read));
            if (io_magic != magic_read) {
                throw std::runtime_error("wrong sequence state magic");
            }

            llama_seq_id seq_id_read;
            header.read(&seq_id_read, sizeof(seq_id_read));
            if (mem_storage.find(seq_id_read) == mem_storage.end()) {
                throw std::runtime_error("missing on-device sequence state storage");
            }
            io = std::make_unique<llama_io_read_device>(src, size, mem_storage.at(seq_id_read));
        } else {
            io = std::make_unique<llama_io_read_host>(src, size);
        }

        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        const size_t nread = state_seq_read_data(*io, seq_id, flags);
        io->commit();
        return nread;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

llama_state_seq_restore_plan * llama_context::state_seq_prepare_data(
        llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    if (src == nullptr || size == 0 ||
            (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) != 0 ||
            seq_id < 0 || uint32_t(seq_id) >= cparams.n_seq_max ||
            !memory || !memory->state_seq_can_restore(seq_id, flags)) {
        return nullptr;
    }

    try {
        auto plan = std::make_unique<llama_state_seq_restore_plan>();
        plan->io = std::make_unique<llama_io_read_host>(src, size);

        uint32_t magic_read;
        plan->io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        plan->io->read(&seq_id_read, sizeof(seq_id_read));
        GGML_UNUSED(seq_id_read);

        plan->bytes = state_seq_read_data(*plan->io, seq_id, flags);
        if (plan->bytes != size) {
            throw std::runtime_error("sequence state contains trailing or missing bytes");
        }
        return plan.release();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error preparing state: %s\n", __func__, err.what());
        return nullptr;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
        io.commit();
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    if (llama_memory_status_is_fail(memory_update(false))) {
        return false;
    }
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    file.close();
    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out, bool require_native) {
    llama_file file(filepath, "rb");
    if (!n_token_count_out) {
        LLAMA_LOG_ERROR("%s: token count output is required\n", __func__);
        return 0;
    }
    *n_token_count_out = 0;

    constexpr size_t sequence_header_size = 3*sizeof(uint32_t);
    if (file.size() < sequence_header_size) {
        LLAMA_LOG_ERROR("%s: truncated sequence state header\n", __func__);
        return 0;
    }

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    uint32_t loaded_token_count = 0;
    {
        const uint32_t n_token_count = file.read_u32();
        loaded_token_count = n_token_count;
        const size_t token_begin = file.tell();
        const size_t file_size = file.size();
        if (token_begin > file_size ||
                uint64_t(n_token_count) > uint64_t(std::numeric_limits<size_t>::max())/sizeof(llama_token)) {
            LLAMA_LOG_ERROR("%s: invalid token count in sequence state file\n", __func__);
            return 0;
        }
        const size_t remaining = file_size - token_begin;
        const size_t token_bytes = size_t(n_token_count)*sizeof(llama_token);
        if (token_bytes > remaining) {
            LLAMA_LOG_ERROR("%s: token payload is truncated (%zu bytes required, %zu remain)\n",
                    __func__, token_bytes, remaining);
            return 0;
        }

        if (tokens_out == nullptr) {
            *n_token_count_out = n_token_count;
            return file.tell();
        }

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, token_bytes);
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        const size_t state_begin = file.tell();
        const bool native_before = remote_native_kvarn_transient_active();
        llama_io_read_file io(&file);
        size_t nread = 0;
        try {
            nread = state_seq_read_data(io, seq_id, 0, loaded_token_count, require_native);
        } catch (...) {
            abort_pending_native_restore();
            if (!native_before && remote_native_kvarn_transient_active() && backend_remote) {
                ggml_backend_remote_attn_native_kvarn_discard(backend_remote);
                (void) ggml_backend_remote_attn_reset(backend_remote, remote_attn_session);
            }
            throw;
        }
        const size_t state_end = file.tell();
        if (!nread || nread > state_size || state_end < state_begin || state_end - state_begin != nread) {
            abort_pending_native_restore();
            LLAMA_LOG_ERROR("%s: failed to restore sequence state (read %zu of %zu bytes)\n",
                    __func__, nread, state_size);
            return 0;
        }
        try {
            if (native_restore_pending) {
                uint8_t prefix_sha256[32] = {};
                if (!llama_native_save::hash_file_prefix(file, state_end, prefix_sha256) ||
                        !commit_pending_native_restore(prefix_sha256)) {
                    abort_pending_native_restore();
                    throw std::runtime_error("native KVarN4 overflow host prefix does not match the Xbox snapshot");
                }
            }
            io.commit();
        } catch (...) {
            abort_pending_native_restore();
            if (!native_before && remote_native_kvarn_transient_active() && backend_remote) {
                ggml_backend_remote_attn_native_kvarn_discard(backend_remote);
                (void) ggml_backend_remote_attn_reset(backend_remote, remote_attn_session);
            }
            throw;
        }
    }

    *n_token_count_out = loaded_token_count;
    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count,
        llama_state_seq_save_preflight_fn preflight, void * user) {
    // A negotiated native FULL save is the only save path that persists the
    // revision-3 owner state. The legacy guard deliberately refuses a FULL
    // bookkeeping snapshot once native rows exist; allow the selective path
    // only when the native disk-snapshot capability was negotiated.
    const bool native_full_ok = remote_native_kvarn_transient_active() &&
            remote_native_kvarn_snapshot_supported();
    const bool can_save = seq_id >= 0 && uint32_t(seq_id) < cparams.n_seq_max && memory &&
            (memory->state_seq_can_save(seq_id, 0) ||
             (native_full_ok && memory->state_seq_can_save(seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)));
    if (!can_save || n_token_count > UINT32_MAX ||
            (n_token_count && !tokens)) { return 0; }
    if (llama_memory_status_is_fail(memory_update(false))) {
        return 0;
    }
    const bool native_active = remote_native_kvarn_transient_active();
    if (native_active && (!remote_native_kvarn_snapshot_supported() ||
            n_token_count <= cparams.kv_position_split_p ||
            n_token_count > cparams.n_ctx_seq)) {
        LLAMA_LOG_ERROR("%s: native KVarN4 overflow FULL save requires the negotiated revision-3 disk snapshot capability\n", __func__);
        return 0;
    }
    // Native FULL saves never truncate the last good slot before the full
    // object and its Xbox snapshot are ready. Standard saves keep their
    // historical direct-write behavior.
    const bool stage_file = native_active || preflight;
    const std::string temporary_path = stage_file ?
        llama_native_save::temporary_path(filepath) : std::string();
    llama_native_save::remove_guard guard{temporary_path};
    const char * write_path = stage_file ? temporary_path.c_str() : filepath;
    llama_file file(write_path, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    if (native_active) io.set_native_expected_tokens(uint32_t(n_token_count));
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    file.close();
    // For native saves this is the last point before the Xbox SAVE operation.
    // The caller can now measure and admit the exact staged host file while
    // the temporary file is still protected by its cleanup guard.
    if (preflight && !preflight(user, write_path, res)) {
        throw std::runtime_error("sequence state save preflight rejected");
    }
    if (native_active) {
        // Direct slot saves bypass server_route_state_save. Pair them with the
        // Xbox snapshot here so a published host file is never left without
        // its GPU state. The key binds the complete host prefix through the
        // RKVK trailer; the server may append SCKP afterwards but that is not
        // part of the key. The staging file is already closed.
        uint8_t native_key[32] = {};
        {
            llama_file input(write_path, "rb");
            if (!llama_native_save::hash_file_prefix(input, input.size(), native_key))
                throw std::runtime_error("native KVarN4 overflow host prefix hash failed");
        }
        const uint32_t live_rows = uint32_t(n_token_count - cparams.kv_position_split_p);
        if (!ggml_backend_remote_attn_native_kvarn_remote_save(backend_remote, native_key, live_rows)) {
            throw std::runtime_error("native KVarN4 overflow Xbox snapshot save failed");
        }
    }
    if (stage_file) {
        if (!llama_native_save::publish(temporary_path, filepath)) {
            throw std::runtime_error("sequence state host snapshot publication failed");
        }
        guard.released = true;
    }
    return res;
}

namespace {

// Bounded-window view over a file region holding a sequence state stream.
class llama_state_q4_file_source final : public llama_state_q4_source {
public:
    llama_state_q4_file_source(llama_file * file, uint64_t begin, uint64_t end)
        : file(file), begin(begin), end(end) {}

    uint64_t size() const override { return end - begin; }
    uint64_t tell() const override { return uint64_t(file->tell()) - begin; }

    void seek(uint64_t offset) override {
        if (offset > end - begin) { throw std::runtime_error("truncated sequence state"); }
        file->seek(begin + offset, SEEK_SET);
    }

    void read_raw(void * dst, size_t size) override {
        const uint64_t pos = uint64_t(file->tell());
        if (pos < begin || pos > end || size > end - pos) {
            throw std::runtime_error("truncated sequence state");
        }
        file->read_raw(dst, size);
    }

private:
    llama_file * file;
    uint64_t begin;
    uint64_t end;
};

class llama_state_q4_memory_source final : public llama_state_q4_source {
public:
    llama_state_q4_memory_source(const uint8_t * data, size_t size) : data(data), len(size) {}

    uint64_t size() const override { return len; }
    uint64_t tell() const override { return pos; }

    void seek(uint64_t offset) override {
        if (offset > len) { throw std::runtime_error("truncated sequence state"); }
        pos = offset;
    }

    void read_raw(void * dst, size_t size) override {
        if (size > len - pos) { throw std::runtime_error("truncated sequence state"); }
        std::memcpy(dst, data + pos, size);
        pos += size;
    }

private:
    const uint8_t * data;
    uint64_t len;
    uint64_t pos = 0;
};

uint64_t llama_state_q4_checksum(const uint8_t * data, size_t size) {
    XXH64_state_t hash;
    XXH64_reset(&hash, 0);
    XXH64_update(&hash, data, size);
    return XXH64_digest(&hash);
}

constexpr uint32_t llama_state_q4_io_magic = 0xaf143cd8;

// Reads the outer header shared by file and in-memory sequence states. Leaves
// the source positioned at the start of the memory-specific body.
bool llama_state_q4_read_outer_header(llama_state_q4_source & src, llama_state_q4_info & info,
        const llama_token * ram_tokens, size_t ram_n_tokens, uint64_t max_tokens,
        std::string & error) {
    try {
        if (max_tokens == 0 || max_tokens > SIZE_MAX / sizeof(llama_token)) {
            throw std::runtime_error("invalid sequence state token budget");
        }

        uint32_t magic = 0;
        src.read_raw(&magic, sizeof(magic));
        if (magic == LLAMA_STATE_SEQ_MAGIC) {
            uint32_t version = 0;
            uint32_t n_tokens = 0;
            src.read_raw(&version, sizeof(version));
            src.read_raw(&n_tokens, sizeof(n_tokens));
            if (version != LLAMA_STATE_SEQ_VERSION) {
                throw std::runtime_error("unsupported sequence state version");
            }
            if (n_tokens == 0 || uint64_t(n_tokens) > max_tokens) {
                throw std::runtime_error("sequence state token count exceeds the destination budget");
            }
            const uint64_t token_bytes = uint64_t(n_tokens) * sizeof(llama_token);
            const uint64_t pos = src.tell();
            if (pos > src.size() || token_bytes > src.size() - pos) {
                throw std::runtime_error("sequence state token header is truncated");
            }
            info.from_ram = false;
            info.n_tokens = n_tokens;
            info.tokens.resize(n_tokens);
            if (n_tokens) {
                src.read_raw(info.tokens.data(), size_t(n_tokens) * sizeof(llama_token));
            }
            return true;
        }
        if (magic == llama_state_q4_io_magic) {
            int32_t seq_id = 0;
            src.read_raw(&seq_id, sizeof(seq_id));
            if (seq_id < 0 || ram_n_tokens == 0 || ram_tokens == nullptr ||
                    ram_n_tokens > max_tokens || ram_n_tokens > UINT32_MAX) {
                throw std::runtime_error("invalid in-memory sequence state token metadata");
            }
            info.from_ram = true;
            info.n_tokens = uint32_t(ram_n_tokens);
            info.tokens.assign(ram_tokens, ram_tokens + ram_n_tokens);
            return true;
        }
        throw std::runtime_error("source is not a sequence state stream");
    } catch (const std::exception & err) {
        error = err.what();
        return false;
    }
}

} // namespace

size_t llama_context::state_seq_convert_seq_stream(
        llama_state_q4_source & src, const llama_state_q4_info & info,
        const char * dst_filepath,
        llama_token * tokens_out, size_t capacity, size_t * count_out,
        std::vector<uint8_t> * out_mem) {
    if (!count_out) { return 0; }
    *count_out = 0;
    if (!tokens_out || !memory || (!dst_filepath && !out_mem) || info.n_tokens == 0) { return 0; }
    if (info.n_tokens > capacity || info.n_tokens > uint64_t(cparams.n_ctx_seq) ||
            info.tokens.size() != info.n_tokens) {
        LLAMA_LOG_ERROR("%s: converted token count exceeds the caller capacity\n", __func__);
        return 0;
    }
    try {
        const size_t written = memory->state_convert_q4(src, info, dst_filepath, out_mem);
        if (written == 0) {
            LLAMA_LOG_ERROR("%s: destination memory has no q4 conversion path\n", __func__);
            return 0;
        }
        std::copy(info.tokens.begin(), info.tokens.end(), tokens_out);
        *count_out = info.tokens.size();
        return written;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_convert_file(
        const char * src_filepath, size_t src_offset, size_t src_size, uint64_t src_checksum,
        const char * dst_filepath,
        llama_token * tokens_out, size_t capacity, size_t * count_out, int32_t rotation_k, int32_t rotation_v) {
    if (!count_out) { return 0; }
    *count_out = 0;
    if (!src_filepath || !dst_filepath) { return 0; }
    try {
        llama_file file(src_filepath, "rb");
        if (src_size == 0 || src_offset > file.size() || src_size > file.size() - src_offset) {
            throw std::runtime_error("invalid sequence state source range");
        }
        // Conversion reads the same bounded q4 rows in a non-linear order
        // (records, stage and exact tail). A private, non-populated mmap keeps
        // the source out of a second user-space buffer while avoiding a
        // seek/fread pair for every 128-token group. Pages are faulted in only
        // as the converter touches them, so this does not materialize the KV
        // payload in RAM. Fall back to the checked file source when mmap is
        // unavailable or rejected by the OS.
        std::unique_ptr<llama_mmap> mapped;
        std::unique_ptr<llama_state_q4_source> source;
        bool source_is_mapped = false;
        if (llama_mmap::SUPPORTED) {
            try {
                mapped = std::make_unique<llama_mmap>(&file, 0, false);
                if (mapped->addr() != nullptr && src_offset <= mapped->size() &&
                        src_size <= mapped->size() - src_offset) {
                    source = std::make_unique<llama_state_q4_memory_source>(
                            static_cast<const uint8_t *>(mapped->addr()) + src_offset, src_size);
                    source_is_mapped = true;
                }
            } catch (const std::exception & error) {
                LLAMA_LOG_DEBUG("%s: q4 source mmap unavailable, using file reads: %s\n",
                        __func__, error.what());
                mapped.reset();
            }
        }
        if (!source) {
            source = std::make_unique<llama_state_q4_file_source>(&file, src_offset, src_offset + src_size);
        }
        LLAMA_LOG_DEBUG("%s: q4 conversion source=%s bytes=%zu\n", __func__,
                source_is_mapped ? "mmap" : "file", src_size);
        if (src_checksum != 0) {
            std::vector<uint8_t> buffer(LLAMA_STATE_FILE_BUFFER_SIZE);
            XXH64_state_t hash;
            XXH64_reset(&hash, 0);
            source->seek(0);
            for (size_t left = src_size; left;) {
                const size_t count = std::min(left, buffer.size());
                source->read_raw(buffer.data(), count);
                XXH64_update(&hash, buffer.data(), count);
                left -= count;
            }
            if (XXH64_digest(&hash) != src_checksum) {
                throw std::runtime_error("sequence state source checksum mismatch");
            }
        }
        source->seek(0);
        llama_state_q4_info info;
        info.rotation_k = rotation_k;
        info.rotation_v = rotation_v;
        std::string error;
        if (!llama_state_q4_read_outer_header(*source, info, nullptr, 0, cparams.n_ctx_seq, error) ||
                !memory->state_parse_q4(*source, model.hparams, info, error)) {
            throw std::runtime_error("unsupported sequence state source: " + error);
        }
        return state_seq_convert_seq_stream(*source, info, dst_filepath, tokens_out, capacity, count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_convert_data(
        const uint8_t * src, size_t size, uint64_t src_checksum,
        const llama_token * ram_tokens, size_t ram_n_tokens,
        const char * dst_filepath,
        llama_token * tokens_out, size_t capacity, size_t * count_out, int32_t rotation_k, int32_t rotation_v) {
    if (!count_out) { return 0; }
    *count_out = 0;
    if (!src || size == 0 || !dst_filepath) { return 0; }
    try {
        if (src_checksum != 0 && llama_state_q4_checksum(src, size) != src_checksum) {
            throw std::runtime_error("sequence state source checksum mismatch");
        }
        llama_state_q4_memory_source source(src, size);
        llama_state_q4_info info;
        info.rotation_k = rotation_k;
        info.rotation_v = rotation_v;
        std::string error;
        if (!llama_state_q4_read_outer_header(source, info, ram_tokens, ram_n_tokens,
                    cparams.n_ctx_seq, error) ||
                !memory->state_parse_q4(source, model.hparams, info, error)) {
            throw std::runtime_error("unsupported sequence state source: " + error);
        }
        return state_seq_convert_seq_stream(source, info, dst_filepath, tokens_out, capacity, count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_convert_data_to_mem(
        const uint8_t * src, size_t size, uint64_t src_checksum,
        const llama_token * ram_tokens, size_t ram_n_tokens,
        std::vector<uint8_t> & out,
        llama_token * tokens_out, size_t capacity, size_t * count_out) {
    if (!count_out) { return 0; }
    *count_out = 0;
    if (!src || size == 0) { return 0; }
    try {
        if (src_checksum != 0 && llama_state_q4_checksum(src, size) != src_checksum) {
            throw std::runtime_error("sequence state source checksum mismatch");
        }
        llama_state_q4_memory_source source(src, size);
        llama_state_q4_info info;
        std::string error;
        if (!llama_state_q4_read_outer_header(source, info, ram_tokens, ram_n_tokens,
                    cparams.n_ctx_seq, error) ||
                !memory->state_parse_q4(source, model.hparams, info, error)) {
            throw std::runtime_error("unsupported sequence state source: " + error);
        }
        return state_seq_convert_seq_stream(source, info, nullptr, tokens_out, capacity, count_out, &out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, err.what());
        return 0;
    }
}

bool llama_context::remote_disk_snapshot_enabled() const {
    return cparams.remote_attn_position_split && cparams.n_seq_max == 1 && backend_remote &&
           ggml_backend_remote_attn_disk_snapshot_enabled(backend_remote);
}

bool llama_context::remote_native_kvarn_enabled() const {
    return backend_remote && ggml_backend_remote_attn_native_kvarn_enabled(backend_remote);
}

bool llama_context::remote_native_kvarn_records_enabled() const {
    return backend_remote && ggml_backend_remote_attn_native_kvarn_records_enabled(backend_remote);
}

bool llama_context::remote_native_kvarn_can_rewind() const {
    return backend_remote && ggml_backend_remote_attn_native_kvarn_can_rewind(backend_remote);
}

bool llama_context::remote_native_kvarn_transient_active() const {
    return backend_remote && ggml_backend_remote_attn_native_kvarn_has_state(backend_remote);
}

bool llama_context::remote_native_kvarn_snapshot_supported() const {
    return cparams.remote_attn_position_split && cparams.n_seq_max == 1 &&
           cparams.n_rs_seq == 0 && backend_remote &&
           ggml_backend_remote_attn_native_kvarn_snapshot_supported(backend_remote);
}

bool llama_context::remote_disk_snapshot(llama_seq_id seq_id, const uint8_t key[32],
                                         size_t evaluated_tokens, bool restore) {
    // Native revision-3 state uses the paired owner trailer and native opcode
    // envelope, never the legacy range disk snapshot.
    if (!remote_disk_snapshot_enabled() || remote_native_kvarn_enabled() || seq_id != 0 ||
            !memory || !key || evaluated_tokens <= cparams.kv_position_split_p ||
            evaluated_tokens > cparams.n_ctx_seq ||
            memory->seq_pos_min(seq_id) != 0 ||
            memory->seq_pos_max(seq_id) != int64_t(evaluated_tokens) - 1) {
        return false;
    }
    synchronize();
    return ggml_backend_remote_attn_disk_snapshot(backend_remote, key,
            uint32_t(evaluated_tokens - cparams.kv_position_split_p), restore);
}

bool llama_remote_disk_snapshot_enabled(const llama_context * ctx) {
    return ctx && ctx->remote_disk_snapshot_enabled();
}

bool llama_remote_native_kvarn_enabled(const llama_context * ctx) {
    return ctx && ctx->remote_native_kvarn_enabled();
}

bool llama_remote_native_kvarn_can_rewind(const llama_context * ctx) {
    return ctx && ctx->remote_native_kvarn_can_rewind();
}

bool llama_remote_native_kvarn_transient_active(const llama_context * ctx) {
    return ctx && ctx->remote_native_kvarn_transient_active();
}

bool llama_remote_native_kvarn_snapshot_supported(const llama_context * ctx) {
    return ctx && ctx->remote_native_kvarn_snapshot_supported();
}

void llama_state_seq_set_native_restore_reservation(llama_context * ctx,
        bool (*reserve)(void *, size_t), void * user) {
    if (ctx) {
        ctx->set_native_restore_reservation(reserve, user);
    }
}

bool llama_state_seq_restore_was_native(const llama_context * ctx) {
    return ctx && ctx->restore_was_native();
}

bool llama_state_seq_remote_disk_snapshot_save(
        llama_context * ctx, llama_seq_id seq_id, const uint8_t key[32], size_t evaluated_tokens) {
    return ctx && ctx->remote_disk_snapshot(seq_id, key, evaluated_tokens, false);
}

size_t llama_context::state_seq_load_file_streaming(llama_seq_id seq_id, const char * filepath,
        llama_token * tokens_out, size_t capacity, size_t * count_out, size_t state_size, uint64_t checksum,
        bool require_native) {
    if (!count_out) { return 0; }
    *count_out = 0;
    // After a successful remote RESET the logical destination and the native
    // owner can both be empty while the memory's historical overflow latch
    // remains set. Admit an ordinary local-prefix file through the empty-owner
    // gate; keep flags=0 during parsing so the mixed reader continues to reject
    // overflow cells and the generic reader continues to reject RKVK trailers.
    // enabled() also requires a connected, non-poisoned backend.
    const bool can_restore_clean_native_prefix = !require_native &&
            remote_native_kvarn_records_enabled() && !remote_native_kvarn_transient_active() && memory &&
            memory->state_seq_can_restore(seq_id, LLAMA_STATE_SEQ_FLAGS_NATIVE_FULL);
    const bool can_restore_native = require_native && remote_native_kvarn_snapshot_supported() &&
            !remote_native_kvarn_transient_active() && memory &&
            memory->state_seq_can_restore(seq_id, LLAMA_STATE_SEQ_FLAGS_NATIVE_FULL);
    if (!tokens_out || !memory || seq_id < 0 || uint32_t(seq_id) >= cparams.n_seq_max ||
            (!can_restore_native && !can_restore_clean_native_prefix &&
             !memory->state_seq_can_restore(seq_id, 0))) { return 0; }
    // These representations may materialize their own payloads while parsing;
    // do not claim a bounded streaming contract for them.
    if (!memory->state_streaming_restore_supported()) {
        LLAMA_LOG_ERROR("%s: compact/precision-tail memory requires the legacy restore API\n", __func__);
        return 0;
    }
    // Streaming commit performs fallible I/O. Preserve the existing APIs'
    // nonempty-destination contract and require an entirely empty context here.
    for (uint32_t seq = 0; seq < cparams.n_seq_max; ++seq) {
        if (memory->seq_pos_max(seq) >= 0) {
            LLAMA_LOG_ERROR("%s: streaming restore requires an empty context\n", __func__);
            return 0;
        }
    }
    try {
        const bool remote_disk = remote_disk_snapshot_enabled();
        const bool native_remote = remote_native_kvarn_enabled();
        // Only the legacy range accelerator keys by the complete host file.
        // Native revision-3 snapshots key by the host prefix through the end
        // of the RKVK frame; a server SCKP append is excluded.
        const bool legacy_remote_disk = remote_disk && !native_remote;
        rkva_snapshot::Sha256 remote_hash;
        rkva_snapshot::Digest remote_key{};
        llama_file file(filepath, "rb");
        if (state_size > file.size() || state_size < 3*sizeof(uint32_t)) {
            throw std::runtime_error("invalid sequence state file size");
        }
        // Verify integrity before handing any metadata to the native parser.
        // The parse pass hashes again and commit checks every indexed chunk.
        {
            std::vector<uint8_t> buffer(LLAMA_STATE_FILE_BUFFER_SIZE);
            XXH64_state_t hash;
            XXH64_reset(&hash, 0);
            for (size_t left = state_size; left;) {
                const size_t count = std::min(left, buffer.size());
                file.read_raw(buffer.data(), count);
                XXH64_update(&hash, buffer.data(), count);
                if (legacy_remote_disk) remote_hash.update(buffer.data(), count);
                left -= count;
            }
            if (XXH64_digest(&hash) != checksum) {
                throw std::runtime_error("sequence state file checksum mismatch");
            }
        }
        if (legacy_remote_disk) {
            remote_key = remote_hash.finish();
        }
        file.seek(0, SEEK_SET);
        llama_io_read_file_stream io(file, state_size);
        uint32_t header[3];
        io.read(header, sizeof(header));
        if (header[0] != LLAMA_STATE_SEQ_MAGIC || header[1] != LLAMA_STATE_SEQ_VERSION ||
                header[2] > capacity || header[2] > (state_size - sizeof(header))/sizeof(llama_token)) {
            throw std::runtime_error("invalid sequence state file header");
        }
        if (can_restore_clean_native_prefix && header[2] > cparams.kv_position_split_p) {
            throw std::runtime_error("native KVarN4 generic restore requires a local-only token prefix");
        }
        io.read(tokens_out, size_t(header[2])*sizeof(llama_token));
        state_seq_read_data(io, seq_id, 0, header[2], require_native);
        const size_t state_end = size_t(file.tell());
        if (native_restore_pending) {
            uint8_t prefix_sha256[32] = {};
            if (!llama_native_save::hash_file_prefix(file, state_end, prefix_sha256) ||
                    !commit_pending_native_restore(prefix_sha256)) {
                abort_pending_native_restore();
                throw std::runtime_error("native KVarN4 overflow host prefix does not match the Xbox snapshot");
            }
        }
        if (io.n_bytes() != state_size || io.checksum() != checksum) {
            // Which half failed matters: logical extent vs content digest.
            throw std::runtime_error(
                    std::string("sequence state file integrity check failed [")
                    + (io.n_bytes() != state_size
                        ? "extent read=" + std::to_string(io.n_bytes()) +
                          " expected=" + std::to_string(state_size)
                        : std::string("digest read=")) + (io.n_bytes() != state_size
                        ? "" : std::to_string(io.checksum())) +
                    (io.n_bytes() != state_size
                        ? "" : " expected=" + std::to_string(checksum)) + "]");
        }
        io.commit();
        if (memory->seq_pos_max(seq_id) >= int64_t(cparams.n_ctx_seq)) {
            throw std::runtime_error("sequence state position exceeds context");
        }
        if (legacy_remote_disk) {
            // CPU/CUDA parsing committed and invalidated the RPC epoch before
            // this optimization. A miss retains full-upload fallback. Native
            // overflow was restored by the trailer path above; it has no
            // bookkeeping fallback.
            (void) remote_disk_snapshot(seq_id, remote_key.data(), header[2], true);
        }
        *count_out = header[2];
        return state_size;
    } catch (const std::exception & error) {
        // Also covers a throwing publication callback. Tensor bytes become
        // unreachable, so a partial restore can never be advertised as a hit.
        abort_pending_native_restore();
        // clear invokes the context's remote-reset hook exactly once; a
        // refused reset poisons the remote backend rather than reusing it.
        memory->clear(false);
        LLAMA_LOG_ERROR("%s: %s; empty destination cleared\n", __func__, error.what());
        return 0;
    }
}

size_t llama_context::state_write_data(llama_io_write_i & io, llama_state_seq_flags flags) {
    if (!(flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) && remote_native_kvarn_transient_active()) {
        throw std::runtime_error("native KVarN4 overflow non-sequence FULL snapshots are unsupported; use the sequence state file API");
    }
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io, -1, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    if (remote_native_kvarn_transient_active()) {
        throw std::runtime_error("native KVarN4 overflow non-sequence restore is unsupported; use the sequence state file API");
    }
    nextn_decode_id = 0;
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);

        // A full restore can move cache placement backwards without changing
        // tensor shapes. Rebuild its cached views and construction-time tail
        // initialization on the next decode, only after the restore commits.
        io.on_commit([this]() {
            for (auto & res : gf_res_prev) { if (res) res->reset(); }
            gf_res_prev_active = nullptr;
            if (cparams.remote_attn_position_split && backend_remote) {
                ggml_backend_remote_attn_invalidate_range(backend_remote);
            }
        });
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags,
        uint8_t native_sha256[32], bool * native_sha256_valid) {
    if (native_sha256_valid) *native_sha256_valid = false;
    const bool partial = (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != 0;
    const bool native_active = remote_native_kvarn_transient_active();
    if (!partial && native_active && !remote_native_kvarn_snapshot_supported()) {
        throw std::runtime_error("native KVarN4 overflow FULL snapshots require the negotiated revision-3 disk snapshot capability; refusing Q4 bookkeeping state");
    }
    if (memory) {
        // The native FULL export appends the RKVK trailer after this body. Mark
        // the memory write so the native-row guard does not reject the local
        // body that the trailer completes.
        const llama_state_seq_flags memory_flags = (!partial && native_active)
            ? llama_state_seq_flags(flags | LLAMA_STATE_SEQ_FLAGS_NATIVE_FULL) : flags;
        memory->state_write(io, seq_id, memory_flags);
    }
    if (!partial && native_active) {
        if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
            throw std::runtime_error("native KVarN4 overflow FULL snapshots require a host file, not on-device state");
        }
        struct sink_adapter final : public rkva_native_state::Sink {
            explicit sink_adapter(llama_io_write_i & io) : io(io) {}
            bool write(const void * data, size_t bytes) override {
                try {
                    io.write(data, bytes);
                    return true;
                } catch (...) {
                    return false;
                }
            }
            llama_io_write_i & io;
        } sink(io);
        if (!ggml_backend_remote_attn_native_kvarn_state_write(backend_remote,
                    [](void * user, const void * data, size_t bytes) {
                        return static_cast<sink_adapter *>(user)->write(data, bytes);
                    }, &sink, native_sha256)) {
            throw std::runtime_error("native KVarN4 overflow owner-state serialization failed");
        }
        if (native_sha256_valid) *native_sha256_valid = true;
    }

    return io.n_bytes();
}

void llama_context::set_native_restore_reservation(bool (*reserve)(void *, size_t), void * user) {
    native_restore_reserve = reserve;
    native_restore_reserve_user = user;
}

bool llama_context::restore_was_native() const {
    return native_restore_committed;
}

void llama_context::abort_pending_native_restore() {
    if (native_restore_pending && backend_remote) {
        ggml_backend_remote_attn_native_kvarn_state_abort(backend_remote);
    }
    native_restore_pending = false;
    native_restore_committed = false;
    native_restore_rows = 0;
    native_restore_saved_generation = 0;
}

bool llama_context::commit_pending_native_restore(const uint8_t prefix_sha256[32]) {
    if (!native_restore_pending) {
        return true;
    }
    if (!prefix_sha256 || !backend_remote) {
        abort_pending_native_restore();
        return false;
    }
    uint64_t new_generation = 0;
    if (!ggml_backend_remote_attn_native_kvarn_remote_restore(backend_remote, prefix_sha256,
                native_restore_rows, native_restore_saved_generation, &new_generation) ||
            !ggml_backend_remote_attn_native_kvarn_state_commit(backend_remote, new_generation)) {
        abort_pending_native_restore();
        return false;
    }
    native_restore_pending = false;
    native_restore_committed = true;
    return true;
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags,
        uint32_t evaluated_tokens, bool require_native) {
    const bool partial = (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != 0;
    // This bit is an internal promise made by the file reader, never a way
    // for callers of the in-memory restore APIs to bypass overflow guards.
    if (flags & LLAMA_STATE_SEQ_FLAGS_NATIVE_FULL) {
        throw std::invalid_argument("native FULL restore flag is reserved for the sequence file reader");
    }
    const bool native_file_body = !partial && evaluated_tokens > cparams.kv_position_split_p &&
            remote_native_kvarn_records_enabled();
    if (remote_native_kvarn_transient_active() && (!partial || !remote_native_kvarn_can_rewind())) {
        throw std::runtime_error("native KVarN4 overflow non-sequence restore is unsupported; use the sequence state file API");
    }
    nextn_decode_id = 0;
    abort_pending_native_restore();
    if (memory) {
        // The memory reader stages writes but cannot see the RKVK trailer,
        // which follows its frame. Only the file reader provides an evaluated
        // token count; allow that candidate body and REQUIRE a verified native
        // trailer below, before any staged writes can commit.
        const llama_state_seq_flags memory_flags = native_file_body
                ? llama_state_seq_flags(flags | LLAMA_STATE_SEQ_FLAGS_NATIVE_FULL) : flags;
        if (native_file_body) io.set_native_expected_tokens(evaluated_tokens);
        memory->state_read(io, seq_id, memory_flags);

        if (!partial) {
            // A native FULL state is an optional trailer after the memory
            // frame. Peek the magic; a replayed prefix lets the native parser
            // consume the complete set without teaching the memory reader
            // about speculative reads.
            const size_t remaining = io.remaining();
            bool native_trailer = false;
            if (remaining) {
                // Peek only the optional native trailer. Any other appendix
                // (notably the server SCKP checkpoint block) is opaque to the
                // llama state payload and must remain unconsumed.
                uint8_t prefix[4] = {};
                const bool peeked = io.peek(prefix, sizeof(prefix));
                if (peeked) {
                    const uint32_t magic = uint32_t(prefix[0]) | (uint32_t(prefix[1]) << 8) |
                                           (uint32_t(prefix[2]) << 16) | (uint32_t(prefix[3]) << 24);
                    if (magic == rkva_native_state::kMagic) {
                        // Generic/automatic readers do not reserve the native
                        // restore peak or enforce native owner admission. Even
                        // an explicitly saved native file may be discovered by
                        // the automatic index, so fail before reading its body.
                        if (!require_native) {
                            throw std::runtime_error("native KVarN4 overflow restore requires explicit native streaming");
                        }
                        uint8_t native_header[rkva_native_state::kHeaderBytes];
                        rkva_native_state::SetHeader header;
                        std::string header_error;
                        if (remaining < sizeof(native_header) ||
                                !io.peek(native_header, sizeof(native_header)) ||
                                !rkva_native_state::decode_header(native_header, header, &header_error)) {
                            throw std::runtime_error("native KVarN4 overflow trailer header is invalid");
                        }
                        const uint64_t peak = rkva_native_state::restore_peak_bytes(header);
                        if (peak == UINT64_MAX || peak > SIZE_MAX ||
                                (native_restore_reserve &&
                                 !native_restore_reserve(native_restore_reserve_user, size_t(peak)))) {
                            throw std::runtime_error("native KVarN4 overflow restore RAM reservation rejected");
                        }
                        io.read(prefix, sizeof(prefix));
                        native_trailer = true;
                        llama_io_read_replay replay(io, prefix, sizeof(prefix));
                        if (!remote_native_kvarn_snapshot_supported()) {
                            ggml_backend_remote_attn_native_kvarn_state_abort(backend_remote);
                            throw std::runtime_error("native KVarN4 overflow snapshot is not supported by the negotiated server");
                        }
                        uint32_t live_rows = 0;
                        uint64_t saved_generation = 0;
                        uint8_t state_key[32] = {};
                        if (!ggml_backend_remote_attn_native_kvarn_state_read(backend_remote,
                                    [](void * user, void * data, size_t bytes) {
                                        try {
                                            static_cast<llama_io_read_replay *>(user)->read(data, bytes);
                                            return true;
                                        } catch (...) {
                                            return false;
                                        }
                                    }, &replay, &live_rows, &saved_generation, state_key)) {
                            throw std::runtime_error("native KVarN4 overflow owner-state trailer is invalid");
                        }
                        if (!evaluated_tokens || evaluated_tokens > cparams.n_ctx_seq ||
                                evaluated_tokens <= cparams.kv_position_split_p ||
                                uint64_t(evaluated_tokens - cparams.kv_position_split_p) != live_rows) {
                            ggml_backend_remote_attn_native_kvarn_state_abort(backend_remote);
                            throw std::runtime_error("native KVarN4 overflow trailer rows do not match the evaluated prefix");
                        }
                        (void) state_key;
                        native_restore_pending = true;
                        native_restore_rows = live_rows;
                        native_restore_saved_generation = saved_generation;
                        if (io.remaining()) {
                            // Only the server's SCKP checkpoint appendix is
                            // recognized after RKVK. Unknown native trailing
                            // bytes fail closed; legacy non-native trailers
                            // remain opaque for compatibility.
                            if (io.remaining() < sizeof(uint32_t)) {
                                abort_pending_native_restore();
                                throw std::runtime_error("truncated native snapshot appendix");
                            }
                            uint8_t tail[4] = {};
                            if (!io.peek(tail, sizeof(tail))) {
                                abort_pending_native_restore();
                                throw std::runtime_error("native snapshot appendix cannot be inspected");
                            }
                            if (llama_state_framing::classify(tail, sizeof(tail)) !=
                                    llama_state_framing::trailing::slot_checkpoint) {
                                abort_pending_native_restore();
                                throw std::runtime_error("unrecognized native snapshot trailing section");
                            }
                        }
                    }
                } else if (evaluated_tokens > cparams.kv_position_split_p &&
                           remote_native_kvarn_records_enabled()) {
                    throw std::runtime_error("native KVarN4 overflow state reader cannot inspect the optional trailer");
                }
            }
            if ((require_native || native_file_body) && !native_trailer) {
                throw std::runtime_error("native streaming slot is not a native KVarN4 snapshot");
            }
            if (!native_trailer && remote_native_kvarn_records_enabled() &&
                    ((evaluated_tokens && evaluated_tokens > cparams.kv_position_split_p) ||
                     memory->seq_pos_max(seq_id) >= int64_t(cparams.kv_position_split_p))) {
                throw std::runtime_error("native KVarN4 overflow restore requires a native state trailer; refusing Q4 bookkeeping state");
            }
        }

        if (partial &&
                cparams.remote_attn_position_split && backend_remote && seq_id == 0) {
            // Mixed partial restore already checked owner identity/generation
            // and resident cell references. Publish a remote suffix trim only
            // after the entire memory restore commits successfully.
            io.on_commit([this, seq_id]() {
                if (!memory->supports_resident_prefix_restore()) {
                    ggml_backend_remote_attn_invalidate_range(backend_remote);
                    return;
                }
                if (remote_native_kvarn_enabled() &&
                        !ggml_backend_remote_attn_native_kvarn_can_restore(backend_remote,
                            memory->seq_pos_max(seq_id) + 1)) {
                    ggml_backend_remote_attn_invalidate_range(backend_remote);
                    throw std::runtime_error("native KV partial restore lacks matching live prefix or retained original rows");
                }
                if (!ggml_backend_remote_attn_trim(backend_remote, remote_attn_session,
                            memory->seq_pos_max(seq_id) + 1)) {
                    throw std::runtime_error("remote KV partial-restore trim failed");
                }
            });
        }
        if (!partial) {
            // Register after memory's callbacks so failed/prepared restores do
            // not invalidate the live graph. Partial speculative checkpoints
            // retain their existing graph-reuse behavior.
            io.on_commit([this]() {
                for (auto & res : gf_res_prev) { if (res) res->reset(); }
            gf_res_prev_active = nullptr;
                // A committed native FULL restore has already installed owner
                // rows and a fresh epoch together with the remote snapshot.
                // Invalidating here would make the first native append reject
                // the restored owner. Standard FULL restores still resync.
                if (!native_restore_committed && cparams.remote_attn_position_split && backend_remote) {
                    ggml_backend_remote_attn_invalidate_range(backend_remote);
                }
            });
        }
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    data.n_reallocs_same = n_reallocs_same;
    data.n_reallocs_diff = n_reallocs_diff;

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;

    // note: the reallocation counters are not reset - they track the lifetime totals for this context
}

llama_memory_breakdown llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (moe_cache) {
        for (const auto & [buft, size] : moe_cache->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (model.hparams.no_alloc) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i] +
                    backend_kvarn_workspace_y_size[i] +
                    backend_kvarn_workspace_split_k_size[i];
        }
    } else {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend) +
                    backend_kvarn_workspace_y_size[i] +
                    backend_kvarn_workspace_split_k_size[i];
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    if (cparams.flash_attn) {
        LLAMA_LOG_INFO("%s: disabling flash attention, FLASH_ATTN_EXT has no backward pass\n", __func__);
        cparams.flash_attn = false;
    }

    // gradients cannot flow through the KV cache, so the attention reads the K and V of the current ubatch directly
    if (n_ubatch == cparams.n_ctx) {
        cparams.training = true;
    } else {
        LLAMA_LOG_WARN("%s: n_ubatch (%u) != n_ctx (%u), the K and V projections will not receive gradients\n", __func__, n_ubatch, cparams.n_ctx);
    }

    // the training graph is different, need to reserve again
    sched_need_reserve = true;
    sched_reserve();

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        // TODO: use llama_batch_ext here
        {
            llama_batch_compat compat(this, batch);
            if (!balloc->init(*compat.batch_ext, model.vocab, true)) {
                LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
                return;
            }
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = get_gf_res_prev();

            const auto gparams = graph_params(res, ubatch, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));

            // the optimizer graph is allocated outside sched, so the next decode must rebuild
            gf_res_prev_active = nullptr;
            res->reset();

            // F4: model-sizing also builds a graph with a memory context; apply
            // the same store-slot boundary so a sizing pass cannot observe a
            // slot left behind by a real ubatch.
            if (mctx) {
                mctx->reset_graph_store_slots();
            }

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            copy_experts.reset();
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_rs_seq                    =*/ 0,
        /*.kv_tail_rollback_tokens     =*/ 0,
        /*.n_outputs_max               =*/ 0,
        /*.n_outputs_max_per_seq       =*/ 1,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.ctx_type                    =*/ LLAMA_CONTEXT_TYPE_DEFAULT,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.kvarn                       =*/ llama_kvarn_default_params(),
        /*.path_kv_mean_center         =*/ nullptr,
        /*.remote_attn_host            =*/ nullptr,
        /*.remote_attn_port            =*/ 0,
        /*.remote_attn_prefill         =*/ 0,
        /*.remote_attn_stats           =*/ 0,
        /*.remote_attn_n_layers        =*/ -1,
        /*.remote_attn_cuda_reserve    =*/ 350 * 1024 * 1024,
        /*.remote_attn_vulkan_reserve =*/ 512 * 1024 * 1024,
        /*.remote_attn_cache_type_k   =*/ GGML_TYPE_COUNT,
        /*.remote_attn_cache_type_v   =*/ GGML_TYPE_COUNT,
        /*.moe_cache_size              =*/ 0,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.kv_paged                    =*/ false,
        /*.kv_paged_dynamic            =*/ false,
        /*.block_size                  =*/ 16,
        /*.n_gpu_blocks                =*/ 0,
        /*.n_gpu_blocks_initial        =*/ 0,
        /*.n_gpu_blocks_growth         =*/ 0,
        /*.n_cpu_blocks                =*/ 0,
        /*.kv_paged_watermark          =*/ 0.05f,
        /*.snapkv_enabled              =*/ false,
        /*.snapkv_observation_window   =*/ 0,
        /*.snapkv_recent_tokens        =*/ 0,
        /*.snapkv_pinned_tokens        =*/ 0,
        /*.snapkv_retention            =*/ 1.0f,
        /*.snapkv_budget_blocks        =*/ 0,
        /*.sampler                     =*/ nullptr,
        /*.n_sampler                   =*/ 0,
        /*.ctx_other                   =*/ nullptr,
        /*.kv_tail_tokens              =*/ 0,
        /*.kv_tail_type                =*/ GGML_TYPE_COUNT,
        /*.kv_tail_config              =*/ nullptr,
        /*.kv_tail_request             =*/ nullptr,
        /*.dflash_split                =*/ false,
        /*.dflash_selector_only        =*/ false,
        /*.no_offload_rs               =*/ false,
        /*.mtp_reserve_enabled         =*/ false,
        /*.mtp_reserve_kvarn           =*/ LLAMA_KVARN_TYPE_DISABLED,
        /*.mtp_reserve_kvarn_bits      =*/ 0,
        /*.mtp_reserve_type_k          =*/ GGML_TYPE_F16,
        /*.mtp_reserve_type_v          =*/ GGML_TYPE_F16,
        /*.mtp_reserve_rollback_tokens =*/ 0,
        /*.kv_position_split_p       =*/ 0,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.kvarn.type != LLAMA_KVARN_TYPE_DISABLED) {
const llama_kvarn_context_route route = llama_kvarn_context_route_for({
            params.ctx_type,
            model->arch,
            params.ctx_other != nullptr,
            model->dspark_markov_w1 != nullptr,
            model->hparams.dflash_selector_top_k > 0,
        });
        if (route != LLAMA_KVARN_CONTEXT_ROUTE_OWNED) {
            const std::string reason = route == LLAMA_KVARN_CONTEXT_ROUTE_SHARED_TARGET
                ? "this MTP topology shares target K/V and has no independent draft KV representation; "
                  "configure target --cache-type-k/v kvarn* instead"
                : format("context type %d for architecture %s is not an audited draft-owned KVarN route; "
                         "choose an ordinary draft cache type",
                         int(params.ctx_type), llm_arch_name(model->arch));
            if (params.kvarn.fail_if_unsupported) {
                LLAMA_LOG_ERROR("%s: cannot enable %s: %s\n",
                        __func__, llama_kvarn_type_name(params.kvarn.type), reason.c_str());
                return nullptr;
            }
            LLAMA_LOG_WARN("%s: cannot enable %s: %s; falling back to the normal KV cache\n",
                    __func__, llama_kvarn_type_name(params.kvarn.type), reason.c_str());
            params.kvarn = llama_kvarn_default_params();
        } else {
            const uint32_t layer_begin = params.ctx_type == LLAMA_CONTEXT_TYPE_MTP
                ? model->hparams.n_layer()
                : 0;
            const uint32_t layer_end = params.ctx_type == LLAMA_CONTEXT_TYPE_MTP
                ? model->hparams.n_layer_all
                : model->hparams.n_layer();
            uint32_t cached_layer_count = 0;
            bool head_dims_supported = true;
            bool backend_ops_supported = true;
            for (uint32_t il = layer_begin; il < layer_end; ++il) {
                if (!model->hparams.has_kv(il) || model->hparams.is_recr(il)) {
                    continue;
                }

                ++cached_layer_count;
                head_dims_supported = head_dims_supported &&
                    llama_kvarn_head_dim_supported(model->hparams.n_embd_head_k(il)) &&
                    llama_kvarn_head_dim_supported(model->hparams.n_embd_head_v(il));

                ggml_backend_dev_t requested_local = nullptr;
                try {
                    requested_local = local_attention_device(params.remote_attn_host);
                } catch (const std::exception & e) {
                    LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
                    return nullptr;
                }
                const int early_count = params.remote_attn_n_layers <= 0 ? INT32_MAX : params.remote_attn_n_layers;
                auto * kv_dev = requested_local && selected_attention_layer(
                        model->hparams, il, early_count, params.ctx_type) ?
                        requested_local : model->dev_layer(il);
                auto * kvarn_dev = params.offload_kqv ? kv_dev : nullptr;
                backend_ops_supported = backend_ops_supported &&
                    llama_kvarn_backend_supports_ops(kvarn_dev, model->hparams.n_embd_head_k(il)) &&
                    llama_kvarn_backend_supports_ops(kvarn_dev, model->hparams.n_embd_head_v(il));
            }

            const bool causal_attn =
                params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED
                    ? model->hparams.causal_attn
                    : params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
            const bool owned_dflash =
                model->arch == LLM_ARCH_DFLASH &&
                route == LLAMA_KVARN_CONTEXT_ROUTE_OWNED;
            const bool attention_supported =
(causal_attn || owned_dflash) &&
                cached_layer_count > 0 &&
                !model->hparams.is_mla() &&
                !llm_arch_is_recurrent(model->arch) &&
                model->arch != LLM_ARCH_DEEPSEEK32;
            const llama_kvarn_runtime_requirements requirements = {
                /*.attention_supported      =*/ attention_supported,
                /*.head_dims_supported      =*/ head_dims_supported,
                /*.backend_ops_supported    =*/ backend_ops_supported,
                /*.n_seq_max                =*/ std::max(1u, params.n_seq_max),
                /*.kv_unified               =*/ params.kv_unified,
            };

            if (const char * reason = llama_kvarn_validate_runtime(params.kvarn, requirements)) {
                if (params.kvarn.fail_if_unsupported) {
                    LLAMA_LOG_ERROR("%s: cannot enable %s: %s\n",
                            __func__, llama_kvarn_type_name(params.kvarn.type), reason);
                    return nullptr;
                }

                LLAMA_LOG_WARN("%s: cannot enable %s: %s; falling back to the normal KV cache\n",
                        __func__, llama_kvarn_type_name(params.kvarn.type), reason);
                params.kvarn = llama_kvarn_default_params();
            } else {
                if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
                    LLAMA_LOG_WARN("%s: KVarN requires Flash Attention; enabling it\n", __func__);
                    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
                }

const char * context_label = params.ctx_type == LLAMA_CONTEXT_TYPE_MTP
                    ? "draft MTP"
                    : owned_dflash ? "draft DFlash" : "target";
                LLAMA_LOG_INFO("%s: enabling structured KVarN cache type %s for %s layers [%u, %u)\n",
                        __func__, llama_kvarn_type_name(params.kvarn.type),
                        context_label, layer_begin, layer_end);
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (model->split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
        if (model->get_split_state_ud.n_devices == 1) {
            LLAMA_LOG_WARN("%s: SPLIT_MODE_TENSOR being used for a single device is not recommended\n", __func__);
        }
    }

    if ((model->hparams.is_mla() || model->arch == LLM_ARCH_DEEPSEEK4) && params.type_k != params.type_v) {
        LLAMA_LOG_ERROR("%s: model does not support different K (%s) and V (%s) cache types\n", __func__, ggml_type_name(params.type_k), ggml_type_name(params.type_v));
        return nullptr;
    }

    if (params.kv_paged && params.type_k != params.type_v) {
        LLAMA_LOG_ERROR("%s: paged KV cache requires the same K (%s) and V (%s) cache type\n",
                        __func__, ggml_type_name(params.type_k), ggml_type_name(params.type_v));
        return nullptr;
    }

    if (params.kv_paged && params.type_k != GGML_TYPE_F16 && params.type_k != GGML_TYPE_Q4_0 && params.type_k != GGML_TYPE_Q8_0) {
        LLAMA_LOG_ERROR("%s: paged KV cache supports only F16, Q4_0, and Q8_0 (got %s)\n",
                        __func__, ggml_type_name(params.type_k));
        return nullptr;
    }

    if (params.kv_paged_dynamic &&
        (params.n_gpu_blocks_initial == 0 || params.n_gpu_blocks_initial > params.n_gpu_blocks)) {
        LLAMA_LOG_ERROR("%s: dynamic paged KV requires 0 < n_gpu_blocks_initial <= n_gpu_blocks\n", __func__);
        return nullptr;
    }

    if (params.kv_paged_dynamic && params.n_gpu_blocks_growth > params.n_gpu_blocks) {
        LLAMA_LOG_ERROR("%s: dynamic paged KV requires n_gpu_blocks_growth <= n_gpu_blocks\n", __func__);
        return nullptr;
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for quantized V cache\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
            LLAMA_LOG_ERROR("%s: quantized V cache requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_k(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_v(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (params.path_kv_mean_center != nullptr && params.type_k != GGML_TYPE_Q4_0) {
        LLAMA_LOG_ERROR("%s: path_kv_mean_center requires the K cache type to be Q4_0 (got %s)\n",
                __func__, ggml_type_name(params.type_k));
        return nullptr;
    }


    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    // router_layer >= 0 means n_layer_nextn is repurposed for a router layer, not real MTP
    if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
        (model->hparams.n_layer_nextn == 0 || model->hparams.router_layer >= 0)) {
        LLAMA_LOG_WARN("%s: context type MTP requested but model doesn't contain MTP layers\n", __func__);
        return nullptr;
    }

    try {
        auto * ctx = new llama_context(*model, params);
        const auto & cparams = ctx->get_cparams();

        if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN && cparams.rope_freq_scale != model->hparams.rope_freq_scale_train) {
            LLAMA_LOG_INFO("%s: custom YaRN scaling detected, re-adjusting n_ctx_train(%u)...\n", __func__, model->hparams.n_ctx_train);
            model->hparams.n_ctx_train = cparams.n_ctx_orig_yarn / cparams.rope_freq_scale;
            LLAMA_LOG_INFO("%s: n_ctx_train adjusted to %u\n", __func__, model->hparams.n_ctx_train);
        }

        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

llama_context_rope_params llama_get_context_rope_params(const llama_context * ctx) {
    const auto & p = ctx->get_cparams();
    return { static_cast<int32_t>(p.rope_scaling_type), p.n_ctx_orig_yarn,
             p.rope_freq_base, p.rope_freq_scale, p.yarn_ext_factor,
             p.yarn_attn_factor, p.yarn_beta_fast, p.yarn_beta_slow };
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_ubatch_physical(const llama_context * ctx) {
    return rkva_physical_ubatch(ctx->get_cparams());
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

uint32_t llama_n_rs_seq(const llama_context * ctx) {
    return ctx->get_cparams().n_rs_seq;
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

bool llama_get_causal_attn(const llama_context * ctx) {
    return ctx->get_causal_attn();
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

int32_t llama_context_prefill_migration_handoff(llama_context * ctx, bool to_remote) {
    if (ctx == nullptr) return LLAMA_PREFILL_MIGRATION_ERROR;
    try {
        return ctx->prefill_migration_handoff(to_remote);
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return LLAMA_PREFILL_MIGRATION_ERROR;
    }
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

extern "C" LLAMA_API void llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked) {
    ctx->set_embeddings_nextn(value, masked);
}

uint64_t llama_get_nextn_decode_id(const llama_context * ctx) {
    return ctx ? ctx->get_nextn_decode_id() : 0;
}

bool llama_matches_nextn_decode(const llama_context * ctx, uint64_t id, const llama_batch & batch) {
    return ctx && ctx->matches_nextn_decode(id, batch);
}

void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool value) {
    ctx->set_embeddings_layer_inp(lid, value);
}

void llama_set_nextn_layer_offset(llama_context * ctx, int32_t offset) {
    ctx->set_nextn_layer_offset(offset);
}

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    return ctx->get_memory();
}

float * llama_get_embeddings_nextn(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn();
}

extern "C" LLAMA_API float * llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn_ith(i);
}

float * llama_get_embeddings_layer_inp(llama_context * ctx, uint32_t lid) {
    ctx->synchronize();

    return ctx->get_embeddings_layer_inp(lid);
}

// multi-layer hidden-state tap C API (staging) -------------------------------

void llama_set_capture_layers(llama_context * ctx, const int32_t * layer_ids, size_t n_layers) {
    std::vector<int32_t> ids;
    ids.reserve(n_layers);
    for (size_t i = 0; i < n_layers; ++i) {
        ids.push_back(layer_ids[i]);
    }
    ctx->set_capture_layers(ids);
}

uint32_t llama_get_n_capture(llama_context * ctx) {
    return ctx->get_n_capture();
}

float * llama_get_embeddings_capture(llama_context * ctx) {
    ctx->synchronize();
    return ctx->get_embeddings_capture();
}

float * llama_get_embeddings_capture_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();
    return ctx->get_embeddings_capture_ith(i);
}

// dspark drafter target-context staging C API --------------------------------

void llama_set_dspark_ctx(llama_context * ctx,
                          const float *   feat,
                          int64_t         n_ctx_rows,
                          int64_t         n_embd_cap,
                          const int32_t * pos) {
    ctx->set_dspark_ctx(feat, n_ctx_rows, n_embd_cap, pos);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

bool llama_memory_reserve(llama_memory_t mem, uint32_t n_tokens) {
    if (!mem) {
        return true;
    }

    return mem->reserve(n_tokens);
}

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_can_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    return mem == nullptr || mem->can_seq_rm(seq_id, p0, p1);
}

llama_memory_seq_rm_capability llama_memory_get_seq_rm_capability(llama_memory_t mem) {
    if (!mem) {
        return { false, false, 0 };
    }
    const auto capability = mem->get_seq_rm_capability();
    return {
        capability.full_clear,
        capability.arbitrary_ranges,
        capability.suffix_rollback_tokens,
    };
}

bool llama_memory_seq_rm_plan(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos * planned_p0,
             llama_pos * planned_p1) {
    if (planned_p0 == nullptr || planned_p1 == nullptr) {
        return false;
    }
    if (mem == nullptr) {
        *planned_p0 = p0;
        *planned_p1 = p1;
        return true;
    }
    return mem->seq_rm_plan(seq_id, p0, p1, *planned_p0, *planned_p1);
}

bool llama_memory_state_seq_can_save(llama_memory_t mem, llama_seq_id seq_id) {
    return mem != nullptr && mem->state_seq_can_save(seq_id);
}

bool llama_memory_state_seq_can_restore(llama_memory_t mem, llama_seq_id seq_id) {
    return mem != nullptr && mem->state_seq_can_restore(seq_id);
}

bool llama_memory_state_seq_can_save_ext(
        llama_memory_t mem, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return mem != nullptr && mem->state_seq_can_save(seq_id, flags);
}

bool llama_memory_state_seq_can_restore_ext(
        llama_memory_t mem, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return mem != nullptr && mem->state_seq_can_restore(seq_id, flags);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1) && mem->run_on_seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return llama_state_get_size_ext(ctx, 0);
}

size_t llama_state_get_size_ext(llama_context * ctx, llama_state_seq_flags flags) {
    return ctx->state_get_size(flags);
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    return llama_state_get_data_ext(ctx, dst, size, 0);
}

size_t llama_state_get_data_ext(
        llama_context * ctx, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size, flags);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    return llama_state_set_data_ext(ctx, src, size, 0);
}

size_t llama_state_set_data_ext(
        llama_context * ctx, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_set_data(src, size, flags);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}
size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

llama_state_seq_restore_plan * llama_state_seq_prepare_data_ext(
        llama_context * ctx,
        const uint8_t * src,
        size_t size,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) {
    if (ctx == nullptr) {
        return nullptr;
    }

    ctx->synchronize();
    return ctx->state_seq_prepare_data(seq_id, src, size, flags);
}

size_t llama_state_seq_restore_plan_commit(llama_state_seq_restore_plan * plan) {
    if (plan == nullptr || !plan->io) {
        return 0;
    }
    plan->io->commit();
    plan->io.reset();
    return plan->bytes;
}

void llama_state_seq_restore_plan_free(llama_state_seq_restore_plan * plan) {
    delete plan;
}

size_t llama_state_seq_load_file_streaming(llama_context * ctx, const char * filepath, llama_seq_id seq_id,
        llama_token * tokens_out, size_t capacity, size_t * count_out, size_t state_size, uint64_t checksum) {
    if (!ctx || !filepath) { return 0; }
    ctx->synchronize();
    return ctx->state_seq_load_file_streaming(seq_id, filepath, tokens_out, capacity, count_out,
            state_size, checksum, false);
}

size_t llama_state_seq_load_file_streaming_native(llama_context * ctx, const char * filepath,
        llama_seq_id seq_id, llama_token * tokens_out, size_t capacity, size_t * count_out,
        size_t state_size, uint64_t checksum) {
    if (!ctx || !filepath) { return 0; }
    ctx->synchronize();
    return ctx->state_seq_load_file_streaming(seq_id, filepath, tokens_out, capacity, count_out,
            state_size, checksum, true);
}

size_t llama_state_seq_convert_file(llama_context * ctx, const char * src_filepath, size_t src_offset, size_t src_size,
        uint64_t src_checksum, const char * dst_filepath, llama_token * tokens_out, size_t capacity, size_t * count_out) {
    if (!ctx || !src_filepath || !dst_filepath) { return 0; }
    return ctx->state_seq_convert_file(src_filepath, src_offset, src_size, src_checksum, dst_filepath,
            tokens_out, capacity, count_out);
}

size_t llama_state_seq_convert_data(llama_context * ctx, const uint8_t * src, size_t size, uint64_t src_checksum,
        const llama_token * ram_tokens, size_t ram_n_tokens,
        const char * dst_filepath, llama_token * tokens_out, size_t capacity, size_t * count_out) {
    if (!ctx || !src || !dst_filepath) { return 0; }
    return ctx->state_seq_convert_data(src, size, src_checksum, ram_tokens, ram_n_tokens,
            dst_filepath, tokens_out, capacity, count_out);
}

size_t llama_state_seq_convert_data_to_mem(llama_context * ctx, const uint8_t * src, size_t size, uint64_t src_checksum,
        const llama_token * ram_tokens, size_t ram_n_tokens,
        std::vector<uint8_t> & out, llama_token * tokens_out, size_t capacity, size_t * count_out) {
    if (!ctx || !src) { return 0; }
    return ctx->state_seq_convert_data_to_mem(src, size, src_checksum, ram_tokens, ram_n_tokens,
            out, tokens_out, capacity, count_out);
}

std::string llama_kvarn_live_layer_table(llama_context * ctx, llama_seq_id seq_id) {
    if (!ctx) { return {}; }
    llama_memory_i * mem = ctx->get_memory();
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) {
        mem = hybrid->get_mem_attn_base();
    }
    if (auto * kvarn = dynamic_cast<llama_kv_cache_kvarn *>(mem)) {
        return kvarn->live_layer_table(seq_id);
    }
    if (auto * sc = dynamic_cast<llama_kv_cache *>(mem)) {
        return sc->live_layer_table(seq_id);
    }
    return {};
}

namespace {

// A sequence blob is [4B io magic][4B seq_id] and, for a mixed/split cache, a
// 16-byte envelope [4B "QSKM"][4B version][8B frame_size] before the KMS
// frame. Locate the frame by ITS OWN magic instead of trusting a remembered
// envelope, so a layout change can never silently shift every reported offset.
bool mixed_frame_offset(const uint8_t * blob, size_t size, size_t * out) {
    if (blob == nullptr || size < 4) {
        return false;
    }
    const auto rd32 = [&](size_t o) -> uint32_t {
        return uint32_t(blob[o]) | (uint32_t(blob[o + 1]) << 8) |
               (uint32_t(blob[o + 2]) << 16) | (uint32_t(blob[o + 3]) << 24);
    };
    constexpr uint32_t KMS1 = 0x31534D4Bu;
    constexpr uint32_t KMS2 = 0x32534D4Bu;
    for (size_t cand : { size_t(24), size_t(8), size_t(0) }) {
        if (cand + 4 > size) {
            continue;
        }
        const uint32_t m = rd32(cand);
        if (m == KMS1 || m == KMS2) {
            *out = cand;
            return true;
        }
    }
    return false;
}

} // namespace

std::string llama_mixed_state_section_map(const uint8_t * seq_blob, size_t size) {
    size_t off = 0;
    if (!mixed_frame_offset(seq_blob, size, &off)) {
        return {};
    }
    try {
        const auto secs = llama_kv_mixed::stream_describe(seq_blob + off, size - off);
        std::string out = "frame_offset=" + std::to_string(off) +
                          " sections=" + std::to_string(secs.size()) + "\n";
        for (auto s : secs) {
            s.start += off;
            s.end   += off;
            out += "  " + llama_kv_mixed::stream_section_text(s) + "\n";
        }
        return out;
    } catch (const std::exception & e) {
        return std::string("describe failed: ") + e.what() + "\n";
    }
}

bool llama_mixed_state_locate(const uint8_t * seq_blob, size_t size, size_t offset,
                              std::string * out_text) {
    size_t off = 0;
    if (!mixed_frame_offset(seq_blob, size, &off) || offset < off) {
        return false;
    }
    try {
        const auto secs = llama_kv_mixed::stream_describe(seq_blob + off, size - off);
        for (auto s : secs) {
            if (offset >= off + s.start && offset < off + s.end) {
                if (out_text) {
                    s.start += off;
                    s.end   += off;
                    *out_text = llama_kv_mixed::stream_section_text(s);
                }
                return true;
            }
        }
    } catch (const std::exception &) {
        return false;
    }
    return false;
}

bool llama_mixed_state_post_frame(const uint8_t * seq_blob, size_t size,
        size_t * out_start, size_t * out_end) {
    size_t off = 0;
    if (!mixed_frame_offset(seq_blob, size, &off)) {
        return false;
    }
    try {
        const auto secs = llama_kv_mixed::stream_describe(seq_blob + off, size - off);
        // stream_describe already reports whatever follows the frame as a
        // "trailing" section, so the frame end is the end of the last section
        // that is NOT trailing. Requiring the frame to end strictly before the
        // blob end would make this function fail on every blob that has one.
        size_t frame_end = 0;
        bool have_frame = false;
        for (auto s : secs) {
            if (strcmp(s.kind, "trailing") == 0) {
                continue;
            }
            frame_end = std::max<size_t>(frame_end, off + s.end);
            have_frame = true;
        }
        if (!have_frame || frame_end >= size) {
            return false;
        }
        if (out_start) { *out_start = frame_end; }
        if (out_end)   { *out_end   = size; }
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

std::string llama_recurrent_state_diff(llama_context * ctx_a, llama_context * ctx_b,
        llama_seq_id seq_id) {
    auto get = [](llama_context * c) -> llama_memory_recurrent * {
        if (!c) {
            return nullptr;
        }
        auto * hyb = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(c));
        return hyb ? hyb->get_mem_recr() : nullptr;
    };
    llama_memory_recurrent * ra = get(ctx_a);
    llama_memory_recurrent * rb = get(ctx_b);
    if (!ra || !rb) {
        return "recurrent-diff: one side has no hybrid/DeltaNet memory\n";
    }
    std::string out = "recurrent-diff: seq=" + std::to_string(seq_id) +
                      " cells_a=" + std::to_string(ra->size) +
                      " cells_b=" + std::to_string(rb->size) +
                      " n_rs_seq_a=" + std::to_string(ra->n_rs_seq) +
                      " n_rs_seq_b=" + std::to_string(rb->n_rs_seq) +
                      " layer_slots_a=" + std::to_string(ra->r_l.size()) +
                      " layer_slots_b=" + std::to_string(rb->r_l.size()) + "\n";
    const size_t n_layer = std::min(ra->r_l.size(), rb->r_l.size());
    size_t compared = 0, diverging = 0;
    double worst_abs = 0.0;
    int worst_layer = -1;
    const char * worst_name = "-";
    std::vector<uint8_t> ba, bb;
    for (size_t il = 0; il < n_layer; ++il) {
        const std::pair<const char *, ggml_tensor *> t_a[] = {
            { "r", ra->r_l[il] }, { "s", ra->s_l[il] },
            { "p", il < ra->p_l.size() ? ra->p_l[il] : nullptr },
        };
        const std::pair<const char *, ggml_tensor *> t_b[] = {
            { "r", rb->r_l[il] }, { "s", rb->s_l[il] },
            { "p", il < rb->p_l.size() ? rb->p_l[il] : nullptr },
        };
        if (ra->r_l[il] == nullptr && rb->r_l[il] == nullptr) {
            continue;
        }
        for (int k = 0; k < 3; ++k) {
            const char * name = t_a[k].first;
            ggml_tensor * ta = t_a[k].second;
            ggml_tensor * tb = t_b[k].second;
            ++compared;
            if ((ta == nullptr) != (tb == nullptr)) {
                out += "  layer=" + std::to_string(il) + " " + name +
                       ": PRESENCE MISMATCH a=" + (ta ? "yes" : "no") +
                       " b=" + (tb ? "yes" : "no") + "\n";
                ++diverging;
                continue;
            }
            if (ta == nullptr) {
                continue;
            }
            const size_t na = ggml_nbytes(ta);
            const size_t nb = ggml_nbytes(tb);
            if (na != nb) {
                out += "  layer=" + std::to_string(il) + " " + name +
                       ": SIZE MISMATCH a=" + std::to_string(na) +
                       " b=" + std::to_string(nb) + "\n";
                ++diverging;
                continue;
            }
            if (na == 0) {
                continue;
            }
            constexpr size_t CHUNK = 1u << 20;
            ba.resize(std::min(CHUNK, na));
            bb.resize(std::min(CHUNK, nb));
            size_t nd = 0;
            size_t first = SIZE_MAX;
            double max_abs = 0.0;
            for (size_t off = 0; off < na; off += CHUNK) {
                const size_t len = std::min(CHUNK, na - off);
                ggml_backend_tensor_get(ta, ba.data(), off, len);
                ggml_backend_tensor_get(tb, bb.data(), off, len);
                for (size_t i = 0; i < len; ++i) {
                    if (ba[i] != bb[i]) {
                        if (first == SIZE_MAX) { first = off + i; }
                        ++nd;
                    }
                }
                if (ta->type == GGML_TYPE_F32) {
                    const float * fa = reinterpret_cast<const float *>(ba.data());
                    const float * fb = reinterpret_cast<const float *>(bb.data());
                    const size_t nf = len / sizeof(float);
                    for (size_t i = 0; i < nf; ++i) {
                        max_abs = std::max(max_abs, std::fabs((double) fa[i] - (double) fb[i]));
                    }
                }
            }
            if (nd == 0) {
                continue;
            }
            ++diverging;
            if (max_abs > worst_abs) {
                worst_abs   = max_abs;
                worst_layer = (int) il;
                worst_name  = name;
            }
            out += "  layer=" + std::to_string(il) + " " + name +
                   " bytes=" + std::to_string(na) +
                   " ndiff=" + std::to_string(nd) +
                   " first=" + std::to_string(first) +
                   " max_abs=" + std::to_string(max_abs) +
                   " shape=" + std::to_string(ta->ne[0]) + "x" + std::to_string(ta->ne[1]) +
                   "x" + std::to_string(ta->ne[2]) +
                   " type=" + std::string(ggml_type_name(ta->type)) + "\n";
        }
    }
    out += "recurrent-diff: compared_tensors=" + std::to_string(compared) +
           " diverging=" + std::to_string(diverging) +
           " worst=layer" + std::to_string(worst_layer) + "." + worst_name +
           " max_abs=" + std::to_string(worst_abs) + "\n";
    return out;
}
std::string llama_mixed_state_write_section_map(void) {
    return llama_kv_cache_kvarn_state_write_sections_text();
}

std::string llama_mixed_state_section_diff(
        const uint8_t * a, size_t na, const uint8_t * b, size_t nb) {
    size_t oa = 0;
    size_t ob = 0;
    if (!mixed_frame_offset(a, na, &oa) || !mixed_frame_offset(b, nb, &ob)) {
        return "section-diff: one of the blobs carries no mixed frame\n";
    }
    std::vector<llama_kv_mixed::stream_section> sa;
    std::vector<llama_kv_mixed::stream_section> sb;
    try {
        sa = llama_kv_mixed::stream_describe(a + oa, na - oa);
        sb = llama_kv_mixed::stream_describe(b + ob, nb - ob);
    } catch (const std::exception & e) {
        return std::string("section-diff: describe failed: ") + e.what() + "\n";
    }
    std::string out = "section-diff: a_sections=" + std::to_string(sa.size()) +
                      " b_sections=" + std::to_string(sb.size()) +
                      " frame_a=" + std::to_string(oa) + " frame_b=" + std::to_string(ob) + "\n";
    const size_t n = std::min(sa.size(), sb.size());
    size_t diverging_sections = 0;
    for (size_t i = 0; i < n; ++i) {
        const auto & x = sa[i];
        const auto & y = sb[i];
        const bool same_shape =
                strcmp(x.kind, y.kind) == 0 && strcmp(x.field, y.field) == 0 &&
                x.layer_id == y.layer_id && x.range_id == y.range_id &&
                (x.end - x.start) == (y.end - y.start);
        const size_t len = size_t(std::min<uint64_t>(x.end - x.start, y.end - y.start));
        size_t first_local = len;
        size_t ndiff = 0;
        for (size_t k = 0; k < len; ++k) {
            if (a[oa + x.start + k] != b[ob + y.start + k]) {
                if (first_local == len) { first_local = k; }
                ++ndiff;
            }
        }
        const bool equal = same_shape && ndiff == 0 &&
                           (x.end - x.start) == (y.end - y.start);
        if (equal) {
            continue;
        }
        ++diverging_sections;
        std::string line = "  sec#" + std::to_string(i) + " [" + std::to_string(x.start) +
                "," + std::to_string(x.end) + ") " + x.kind + "/" + x.field;
        if (x.layer_id != UINT32_MAX) {
            line += " layer=" + std::to_string(x.layer_id);
        }
        if (x.range_id != UINT32_MAX) {
            line += " stream=" + std::to_string(x.range_id);
        }
        line += " bytes=" + std::to_string(x.end - x.start) + "/" + std::to_string(y.end - y.start);
        if (!same_shape) {
            line += " STRUCTURE MISMATCH";
        }
        line += " ndiff=" + std::to_string(ndiff);
        if (first_local != len) {
            line += " first_local=" + std::to_string(first_local);
            line += " abs=" + std::to_string(oa + x.start + first_local);
        }
        if (x.checksum != 0 || y.checksum != 0) {
            char buf[64];
            snprintf(buf, sizeof(buf), " checksum=%016llx/%016llx",
                     (unsigned long long) x.checksum, (unsigned long long) y.checksum);
            line += buf;
        }
        if (x.k_stride != 0 || x.v_stride != 0) {
            line += " stride=" + std::to_string(x.k_stride) + "/" + std::to_string(x.v_stride);
        }
        out += line + "\n";
    }
    out += "section-diff: diverging_sections=" + std::to_string(diverging_sections) +
           " of " + std::to_string(n) + "\n";
    if (sa.size() != sb.size()) {
        out += "section-diff: SECTION COUNT MISMATCH\n";
    }
    return out;
}

bool llama_kvarn_probe_mutation(llama_context * ctx, llama_seq_id seq_id, int layer_index,
        const char * tensor, uint32_t row, bool live) {
    if (!ctx) { return false; }
    llama_memory_i * mem = ctx->get_memory();
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) {
        mem = hybrid->get_mem_attn_base();
    }
    if (auto * kvarn = dynamic_cast<llama_kv_cache_kvarn *>(mem)) {
        return kvarn->probe_mutation(seq_id, layer_index, tensor, row, live);
    }
    if (auto * sc = dynamic_cast<llama_kv_cache *>(mem)) {
        return sc->probe_mutation(seq_id, layer_index, tensor, row, live);
    }
    return false;
}

// [TAG_SLOT_SAVE_CHECKPOINTS] Durable-checkpoint lineage epoch access for the server save/
// restore flow. Mirrors the navigation used by the KVarN probe helpers above.
static llama_kv_cache_kvarn * llama_kvarn_mixed_cache(llama_context * ctx) {
    if (!ctx) { return nullptr; }
    llama_memory_i * mem = ctx->get_memory();
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) {
        mem = hybrid->get_mem_attn_base();
    }
    return dynamic_cast<llama_kv_cache_kvarn *>(mem);
}

uint64_t llama_kvarn_state_owner_epoch(llama_context * ctx) {
    auto * kvarn = llama_kvarn_mixed_cache(ctx);
    return kvarn ? kvarn->mixed_state_owner_epoch() : 0;
}

bool llama_kvarn_adopt_state_owner_epoch(llama_context * ctx, uint64_t epoch) {
    auto * kvarn = llama_kvarn_mixed_cache(ctx);
    return kvarn ? kvarn->adopt_state_owner_epoch(epoch) : false;
}

uint64_t llama_kvarn_state_owner_id(llama_context * ctx) {
    auto * kvarn = llama_kvarn_mixed_cache(ctx);
    return kvarn ? kvarn->mixed_state_owner_id() : 0;
}

bool llama_kvarn_adopt_state_owner_id(llama_context * ctx, uint64_t owner_id) {
    auto * kvarn = llama_kvarn_mixed_cache(ctx);
    return kvarn ? kvarn->adopt_state_owner_id(owner_id) : false;
}

bool llama_kvarn_probe_head_span(llama_context * ctx, llama_seq_id seq_id, int layer_index,
        uint32_t head, uint32_t group_or_slot, const char * tensor, bool live,
        uint8_t * orig_out, uint8_t * new_out, uint32_t * group_used_out) {
    if (!ctx) { return false; }
    llama_memory_i * mem = ctx->get_memory();
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) {
        mem = hybrid->get_mem_attn_base();
    }
    if (auto * kvarn = dynamic_cast<llama_kv_cache_kvarn *>(mem)) {
        return kvarn->probe_head_span(seq_id, layer_index, head, group_or_slot, tensor, live, orig_out, new_out, group_used_out);
    }
    if (auto * sc = dynamic_cast<llama_kv_cache *>(mem)) {
        return sc->probe_mutation(seq_id, layer_index, tensor, 0, live);
    }
    return false;
}

void llama_kvarn_capture_clear(llama_context * ctx) {
    if (!ctx) { return; }
    llama_memory_i * mem = ctx->get_memory();
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) { mem = hybrid->get_mem_attn_base(); }
    if (auto * kvarn = dynamic_cast<llama_kv_cache_kvarn *>(mem)) { kvarn->clear_captures(); }
}

size_t llama_kvarn_capture_routes(llama_context * ctx, int layer, bool value, uint32_t head) {
    if (!ctx) { return 0; }
    llama_memory_i * mem = ctx->get_memory();
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) { mem = hybrid->get_mem_attn_base(); }
    auto * kvarn = dynamic_cast<llama_kv_cache_kvarn *>(mem);
    return kvarn ? kvarn->capture_count(layer, value, head) : 0;
}

bool llama_kvarn_captured_head(llama_context * ctx, int layer, bool value,
        uint32_t head, llama_pos pos, size_t index, float * out, size_t n_out, char * info, size_t info_len) {
    if (!ctx || !out || n_out == 0) { return false; }
    llama_memory_i * mem = ctx->get_memory();
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) { mem = hybrid->get_mem_attn_base(); }
    auto * kvarn = dynamic_cast<llama_kv_cache_kvarn *>(mem);
    if (!kvarn) { return false; }
    llama_kv_cache_kvarn::capture_sample cs;
    if (!kvarn->captured(layer, value, head, pos, index, cs)) {
        if (info && info_len) {
            snprintf(info, info_len, "absent layer=%d kv=%s head=%u index=%zu of %zu",
                    layer, value ? "V" : "K", head, index, kvarn->capture_count(layer, value, head));
        }
        return false;
    }
    if (info && info_len) {
        snprintf(info, info_len,
                "layer=%d kv=%s head=%u pos=%lld route=%s range=%s name=%s type=%d offset=%llu nbytes=%llu elem=%zu shared=%d mapped=%d stored=%s:%s/%d records=%s/%d backend=%s fa=%zu seq=%u gen=%llu",
                cs.layer, value ? "V" : "K", cs.head, (long long) cs.pos, cs.route.c_str(),
                cs.range.c_str(), cs.name.c_str(), cs.type,
                (unsigned long long) cs.offset, (unsigned long long) cs.nbytes, cs.vec.size(),
                cs.shared_il, cs.mapped,
                cs.stored_name.c_str(), cs.stored_from.c_str(), cs.stored_type,
                cs.records_name.c_str(), cs.records_type,
                cs.backend.c_str(), cs.fa_hits, cs.mark_seq, (unsigned long long) cs.gen);
    }
    if (!cs.vec.empty()) {
        const size_t n = std::min(n_out, cs.vec.size());
        for (size_t i = 0; i < n; ++i) { out[i] = cs.vec[i]; }
        if (n < n_out) { for (size_t i = n; i < n_out; ++i) { out[i] = NAN; } }
        return true;
    }
    if (!cs.raw.empty()) {
        const size_t n = std::min(n_out, cs.raw.size());
        for (size_t i = 0; i < n; ++i) { out[i] = (float) cs.raw[i]; }
        if (n < n_out) { for (size_t i = n; i < n_out; ++i) { out[i] = NAN; } }
        return true;
    }
    return false;
}

ggml_backend_t llama_context::backend_supporting_buft(ggml_backend_buffer_type_t bt) const {
    if (!bt) { return nullptr; }
    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        if (ggml_backend_supports_buft(backend_ptrs[i], bt)) {
            return backend_ptrs[i];
        }
    }
    return nullptr;
}

bool llama_kvarn_materialized_probe(llama_context * ctx, llama_seq_id seq_id, int layer_index,
        uint32_t head, bool value, llama_pos pos, std::vector<float> * out) {
    if (!ctx || !out) { return false; }
    llama_memory_i * mem = ctx->get_memory();
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) {
        mem = hybrid->get_mem_attn_base();
    }
    auto * kvarn = dynamic_cast<llama_kv_cache_kvarn *>(mem);
    if (!kvarn) { return false; }
    // The consumer must run on a backend this context ALREADY owns, chosen by
    // asking which of the context's backends supports the leaf's buffer type.
    // Never initialise a backend here: an extra instance corrupts teardown.
    const ggml_tensor * leaf = kvarn->probe_leaf(layer_index, value);
    ggml_backend_t be = nullptr;
    if (leaf && leaf->buffer) {
        be = ctx->backend_supporting_buft(ggml_backend_buffer_get_type(leaf->buffer));
    }
    std::string st;
    const bool ok = kvarn->materialized_probe(seq_id, layer_index, head, value, pos, *out, &st, be);
    if (!ok) {
        LLAMA_LOG_WARN("%s: materialized_probe unavailable: status=%s\n", __func__, st.c_str());
    }
    return ok;
}

bool llama_kvarn_probe_remote_span(llama_context * ctx, llama_seq_id seq_id, int layer_index,
        uint32_t head, bool value, bool live,
        uint8_t * orig_out, uint8_t * new_out, uint32_t * row_used_out,
        llama_pos * pos_used_out) {
    if (!ctx) { return false; }
    llama_memory_i * mem = ctx->get_memory();
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) { mem = hybrid->get_mem_attn_base(); }
    if (auto * kvarn = dynamic_cast<llama_kv_cache_kvarn *>(mem)) {
        return kvarn->probe_remote_span(seq_id, layer_index, head, value, live,
                orig_out, new_out, row_used_out, pos_used_out);
    }
    return false;
}

bool llama_kvarn_probe_remote_restore(llama_context * ctx, llama_seq_id seq_id, int layer_index,
        uint32_t head, bool value, uint32_t row, uint8_t orig) {
    if (!ctx) { return false; }
    llama_memory_i * mem = ctx->get_memory();
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) { mem = hybrid->get_mem_attn_base(); }
    if (auto * kvarn = dynamic_cast<llama_kv_cache_kvarn *>(mem)) {
        return kvarn->probe_remote_restore_byte(seq_id, layer_index, head, value, row, orig);
    }
    return false;
}

bool llama_kvarn_probe_restore_byte(llama_context * ctx, llama_seq_id seq_id, int layer_index,
        uint32_t head, uint32_t group, const char * tensor, uint8_t orig) {
    if (!ctx) { return false; }
    llama_memory_i * mem = ctx->get_memory();
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) {
        mem = hybrid->get_mem_attn_base();
    }
    if (auto * kvarn = dynamic_cast<llama_kv_cache_kvarn *>(mem)) {
        return kvarn->probe_restore_byte(seq_id, layer_index, head, group, tensor, orig);
    }
    return false;
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    return llama_state_seq_save_file_preflight(ctx, filepath, seq_id, tokens, n_token_count, nullptr, nullptr);
}

size_t llama_state_seq_save_file_preflight(llama_context * ctx, const char * filepath, llama_seq_id seq_id,
        const llama_token * tokens, size_t n_token_count, llama_state_seq_save_preflight_fn preflight, void * user) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count, preflight, user);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity,
                n_token_count_out, false);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file_native(llama_context * ctx, const char * filepath,
        llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity,
        size_t * n_token_count_out) {
    if (!ctx) {
        return 0;
    }
    ctx->synchronize();
    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity,
                n_token_count_out, true);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading native sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

// compat: llama_batch -> llama_batch_ext -> encode/decode

int llama_context::encode(const llama_batch & batch_inp) {
    llama_batch_compat compat(this, batch_inp, cparams.dflash_selector_only ? model.hparams.n_embd : model.hparams.n_embd_inp_enc());
    return encode(*compat.batch_ext);
}

int llama_context::decode(const llama_batch & batch_inp) {
    const size_t width = cparams.dflash_split && !batch_inp.token ? model.hparams.n_embd_inp_enc() : 0;
    llama_batch_compat compat(this, batch_inp, width);
    return decode(*compat.batch_ext);
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

void llama_set_snapkv_prefill_end(
        llama_context * ctx,
        llama_seq_id   seq_id,
        llama_pos      prefill_end) {
    if (ctx) {
        ctx->set_snapkv_prefill_end(seq_id, prefill_end);
    }
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
    LLAMA_LOG_INFO("%s:   graph reallocs = %10d    / %5d        (same size) / (changed size)\n", __func__, data.n_reallocs_same, data.n_reallocs_diff);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}

int32_t llama_process(llama_context * ctx, llama_process_type type, llama_batch_ext * batch) {
    switch (type) {
        case LLAMA_PROCESS_TYPE_ENCODE: return ctx->encode(*batch);
        case LLAMA_PROCESS_TYPE_DECODE: return ctx->decode(*batch);
    }
    return -1;
}

//
// ext
//

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    return ctx->memory_breakdown();
}

llama_kv_memory_stats llama_get_kv_memory_stats(const struct llama_context * ctx) {
    llama_memory_t memory = ctx ? ctx->get_memory() : nullptr;
    return memory ? memory->kv_memory_stats() : llama_kv_memory_stats{};
}

extern "C" LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx) {
    return ctx->get_cparams().ctx_other;
}

size_t llama_state_seq_convert_data_rotated(llama_context * ctx, const uint8_t * src, size_t size,
    uint64_t checksum, const llama_token * tokens, size_t n_tokens, const char * dst,
    llama_token * out, size_t capacity, size_t * count, int32_t rotation_k, int32_t rotation_v) {
    if (!ctx || !src || !dst) { return 0; }
    return ctx->state_seq_convert_data(src, size, checksum, tokens, n_tokens, dst,
                                      out, capacity, count, rotation_k, rotation_v);
}
size_t llama_state_seq_convert_file_rotated(llama_context * ctx, const char * src, size_t offset,
    size_t size, uint64_t checksum, const char * dst, llama_token * out,
    size_t capacity, size_t * count, int32_t rotation_k, int32_t rotation_v) {
    if (!ctx || !src || !dst) { return 0; }
    return ctx->state_seq_convert_file(src, offset, size, checksum, dst,
                                      out, capacity, count, rotation_k, rotation_v);
}
