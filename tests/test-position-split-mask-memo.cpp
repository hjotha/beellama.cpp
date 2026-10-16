// F3 overflow-mask crop memo (GPU-free): the position-split overflow mask is
// ONE host-filled graph input shared by every KVarN layer of a graph, and the
// crop extent (occupancy or packed-view stable extent) is layer-independent.
// build_ovf_mask_crop must therefore materialize the cropped contiguous mask
// exactly ONCE per (source, extent) per graph build and hand the same tensor
// to all layers, instead of re-creating view+CONT per layer (16x host->device
// re-staging of identical bytes).
//
// This test drives the REAL llm_graph_context::build_ovf_mask_crop with a
// minimal hand-built context (no model, no backend, no GPU):
//   * 16 calls with the same source/extent return the same pointer, every
//     layer expands it like the FA consumer does, and the graph keeps ONE
//     cropping VIEW + ONE CONT node (the per-graph peak: one cropped-mask
//     node/buffer, not one per layer);
//   * a different extent materializes a different tensor;
//   * a different source tensor gets its own entry;
//   * the materialized tensor keeps the crop contract (extent, rows, dtype,
//     contiguity, stable diagnostic name) and reads the shared source through
//     its own view;
//   * a fresh llm_graph_context (per-build lifetime) never inherits entries
//     from another context.
//
// The two production call sites (build_attn_position_split: the occupancy crop
// and the packed-view crop) need the real KVarN builder (model, paged cache,
// scheduler) and cannot be driven here; the per-context isolation check below
// is the wiring contract they rely on (an entry from another, already freed
// ggml_context must never be handed out).

#include "../src/llama-graph.h"

#include "ggml.h"

#include <cstdio>
#include <cstring>

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond) do { \
    ++g_checks; \
    if (!(cond)) { \
        ++g_failures; \
        std::printf("  FAIL line %d: %s\n", __LINE__, #cond); \
    } \
} while (0)

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    llm_graph_result res(4096);
    llm_graph_params params{};

    // Minimal hparams/cparams so the llm_graph_context constructor can query
    // the per-layer getters without aborting (they index layer arrays).
    params.arch = LLM_ARCH_LLAMA;
    params.res  = &res;
    params.cparams.warmup = true; // avoids the per-layer n_expert_used() table
    params.hparams.n_layer_all      = 1;
    params.hparams.n_head_arr[0]    = 4;
    params.hparams.n_head_kv_arr[0] = 4;
    params.hparams.n_embd_head_k_full = 64;
    params.hparams.n_embd_head_v_full = 64;
    params.hparams.n_rot_full         = 64;

    llm_graph_context ctx(params);
    CHECK(ctx.ctx0 != nullptr);
    CHECK(res.get_gf() != nullptr);

    const int64_t n_cap  = 1024;
    const int64_t n_q    = 8;
    const int64_t n_occ  = 256;
    const int64_t n_stb  = 512;

    ggml_tensor * src = ggml_new_tensor_4d(ctx.ctx0, GGML_TYPE_F16, n_cap, n_q, 1, 1);

    // Graph-level counters. Defined before the layer loop: every layer below
    // expands the mask returned by "build_ovf_mask_crop", exactly like the
    // overflow FA consumer does in build_attn_position_split.
    const auto count_named = [&](ggml_op op, const char * name) {
        int n = 0;
        for (int i = 0; i < ggml_graph_n_nodes(res.get_gf()); ++i) {
            ggml_tensor * node = ggml_graph_node(res.get_gf(), i);
            if (node->op == op && std::strcmp(node->name, name) == 0) {
                ++n;
            }
        }
        return n;
    };
    const auto count_views_of = [&](const ggml_tensor * t) {
        int n = 0;
        for (int i = 0; i < ggml_graph_n_nodes(res.get_gf()); ++i) {
            ggml_tensor * node = ggml_graph_node(res.get_gf(), i);
            if (node->op == GGML_OP_VIEW && node->view_src == t) {
                ++n;
            }
        }
        return n;
    };

    // 16 "layers" of one graph build, same source and extent. EVERY returned
    // tensor is expanded inside the loop (production: the layer's FA consumes
    // it), so the graph must stay at one cropping VIEW + one CONT for all 16.
    ggml_tensor * first = nullptr;
    for (int il = 0; il < 16; ++il) {
        ggml_tensor * t = ctx.build_ovf_mask_crop(src, n_occ);
        if (il == 0) {
            first = t;
        }
        CHECK(t == first);
        ggml_build_forward_expand(res.get_gf(), t);
    }
    CHECK(first != nullptr && first->op == GGML_OP_CONT);
    CHECK(first->ne[0] == n_occ && first->ne[1] == n_q && first->ne[2] == 1 && first->ne[3] == 1);
    CHECK(first->type == GGML_TYPE_F16);
    CHECK(ggml_is_contiguous(first));
    // Stable diagnostic name: the shared mask is never renamed per layer (the
    // F5_LAYER_CAPTURE block keeps it as the capture selector).
    CHECK(std::strcmp(first->name, "attn_inp_kq_mask_overflow_occ") == 0);
    CHECK(first->src[0] != nullptr && first->src[0]->view_src == src);
    // Peak, visible on CPU (what F4_ALLOC_TRACE reports for a real graph):
    // 16 expansions leave exactly ONE mask crop node, hence one cropped-mask
    // buffer held for the whole graph, not one per layer.
    CHECK(count_named(GGML_OP_CONT, "attn_inp_kq_mask_overflow_occ") == 1);
    CHECK(count_views_of(src) == 1);

    // Same source, different extent: distinct materialization.
    ggml_tensor * other_extent = ctx.build_ovf_mask_crop(src, n_stb);
    CHECK(other_extent != first);
    CHECK(other_extent->ne[0] == n_stb && other_extent->ne[1] == n_q);
    CHECK(ggml_is_contiguous(other_extent));
    // ... and the first extent still memoizes to the first tensor.
    CHECK(ctx.build_ovf_mask_crop(src, n_occ) == first);

    // Different source tensor: its own entry (no cross-source aliasing).
    ggml_tensor * src2 = ggml_new_tensor_4d(ctx.ctx0, GGML_TYPE_F16, n_cap, n_q, 1, 1);
    ggml_tensor * second_src = ctx.build_ovf_mask_crop(src2, n_occ);
    CHECK(second_src != first);
    CHECK(second_src->src[0] != nullptr && second_src->src[0]->view_src == src2);

    ggml_build_forward_expand(res.get_gf(), other_extent);
    CHECK(count_named(GGML_OP_CONT, "attn_inp_kq_mask_overflow_occ") == 2);
    CHECK(count_views_of(src) == 2);

    ggml_build_forward_expand(res.get_gf(), second_src);
    CHECK(count_named(GGML_OP_CONT, "attn_inp_kq_mask_overflow_occ") == 3);
    CHECK(count_views_of(src2) == 1);

    // Memo lifetime wiring: the real call sites live in
    // llm_graph_context::build_attn_position_split and need the full KVarN
    // builder (model, paged cache, scheduler), so they cannot be driven here.
    // The contract they depend on is that the memo is per llm_graph_context
    // (per build_graph) and never shared across builds: the same key in a
    // fresh context must materialize a fresh tensor, not hand out one owned
    // by another (already freed) ggml_context.
    llm_graph_result res2(4096);
    llm_graph_params params2 = params;
    params2.res = &res2;
    llm_graph_context ctx2(params2);
    CHECK(ctx2.ctx0 != nullptr && ctx2.ctx0 != ctx.ctx0);
    ggml_tensor * fresh = ctx2.build_ovf_mask_crop(src, n_occ);
    CHECK(fresh != first);
    CHECK(fresh->op == GGML_OP_CONT);
    CHECK(fresh->ne[0] == n_occ && fresh->ne[1] == n_q);
    CHECK(std::strcmp(fresh->name, "attn_inp_kq_mask_overflow_occ") == 0);
    CHECK(fresh->src[0] != nullptr && fresh->src[0]->view_src == src);
    // ... and ctx's own entry is untouched.
    CHECK(ctx.build_ovf_mask_crop(src, n_occ) == first);

    std::printf("position-split-mask-memo: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
