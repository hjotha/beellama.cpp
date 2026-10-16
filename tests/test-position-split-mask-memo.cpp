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
//   * 16 calls with the same source/extent return the same pointer and add a
//     single GGML_OP_CONT node (and a single cropping VIEW) to the graph;
//   * a different extent materializes a different tensor;
//   * a different source tensor gets its own entry;
//   * the materialized tensor keeps the crop contract (extent, rows, dtype,
//     contiguity) and reads the shared source through its own view.

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

    // 16 "layers" of one graph build, same source and extent.
    ggml_tensor * first = nullptr;
    for (int il = 0; il < 16; ++il) {
        ggml_tensor * t = ctx.build_ovf_mask_crop(src, n_occ);
        if (il == 0) {
            first = t;
        }
        CHECK(t == first);
    }
    CHECK(first != nullptr && first->op == GGML_OP_CONT);
    CHECK(first->ne[0] == n_occ && first->ne[1] == n_q && first->ne[2] == 1 && first->ne[3] == 1);
    CHECK(first->type == GGML_TYPE_F16);
    CHECK(ggml_is_contiguous(first));
    CHECK(std::strcmp(first->name, "attn_inp_kq_mask_overflow_occ") == 0);
    CHECK(first->src[0] != nullptr && first->src[0]->view_src == src);

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

    // Graph-level count: one CONT node per distinct (source, extent), even
    // after 16 consumer "layers" reused it. Expanding the memoized tensor is
    // exactly what the production builder does through its FA consumer.
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

    ggml_build_forward_expand(res.get_gf(), first);
    CHECK(count_named(GGML_OP_CONT, "attn_inp_kq_mask_overflow_occ") == 1);
    CHECK(count_views_of(src) == 1);

    ggml_build_forward_expand(res.get_gf(), other_extent);
    CHECK(count_named(GGML_OP_CONT, "attn_inp_kq_mask_overflow_occ") == 2);
    CHECK(count_views_of(src) == 2);

    ggml_build_forward_expand(res.get_gf(), second_src);
    CHECK(count_named(GGML_OP_CONT, "attn_inp_kq_mask_overflow_occ") == 3);
    CHECK(count_views_of(src2) == 1);

    std::printf("position-split-mask-memo: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
