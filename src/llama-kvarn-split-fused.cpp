// Executor for GGML_OP_KVARN_SPLIT_FUSED ("peça (c)" phase 2).
//
// The CUDA dispatcher (ggml/src/ggml-cuda/ggml-cuda.cu,
// ggml_cuda_op_kvarn_split_fused) only synchronizes its stream and forwards
// the node to the executor registered through
// "ggml_backend_cuda_set_kvarn_split_fused_exec". This file is that executor:
// per fused op (one per split decode layer and token) it stages host buffers,
// runs the overflow half of the two-range chain on Vulkan as a persistent
// per-layer ggml graph --
//
//   SET_ROWS  current K/V into the Q4_0 companion storage (the store the
//             legacy chain emits as k_written/v_written)
//   GET_ROWS  identity rows over the packed overflow window -> F16
//   FA2+LSE2  FlashAttention over the overflow range (AUTO domain, F32 acc,
//             n_kv_max = n_ovf_rows), exactly the node build_attn_mha() builds
//
// -- then reads O2/LSE2 back to host and completes the span with a private
// CUDA merge sub-graph that mirrors the legacy merge arithmetic of
// src/llama-graph.cpp (clamp / abs-max / exp / weighted add / div / cont),
// writing the op's o_merged output directly.
//
// Why the overflow K/V stays on Vulkan: src[5]/src[6] are the raw packed row
// views over the Q4 companion storage (Vulkan device memory, position-split
// requires it). The scheduler copy for those sources is suppressed for this op
// (ggml-backend.cpp), so the executor binds them by aliasing the storage
// tensor into its own Vulkan sub-graph; nothing is copied device-to-device.
//
// Fail closed: every contract violation (missing backend, missing view_src,
// wrong dtype/shape, failed submission) aborts via GGML_ABORT with a message.
// There is no silent fallback at execution time; the builder keeps the legacy
// chain whenever the op is not eligible.
//
// Caching policy (review follow-up): the persistent sub-graphs are keyed by
// the LAYER ID only (map il -> executor). The llama graph is rebuilt regularly
// (token counts, cache extents, verification ubatches), which hands fresh
// tensor objects to the same logical operands; a pointer-keyed cache
// multiplied executors per rebuild until the sanity cap aborted.
//   * There is exactly one executor slot per layer id; it is never duplicated
//     and old resources are freed before any rebuild (no orphan accumulation).
//   * The slot is kept while the geometry (shapes, extents, scale, index
//     dtypes) AND the backing memory (tensor data pointers and buffer objects)
//     are unchanged.
//   * Any change - including a graph rebuild that reallocates buffers, which
//     may hand out the SAME tensor addresses over new VkBuffers - rebuilds the
//     two sub-graphs in place. In-place pointer repair (rebinding the alias
//     leaves) was tried and rejected: after an outer rebuild the Vulkan kernel
//     path kept resolving the gather through the previous backing set and
//     produced a deterministically wrong span (the iface readback of the same
//     tensor was correct, the kernel read zeros), so the conservative rebuild
//     is used. The sub-graph contexts are sized for their tensor metadata
//     only, so a rebuild is cheap.

#include "llama-kvarn-split-fused.h"

#include "ggml-alloc.h"
#include "llama-impl.h"

#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

//
// Registration contract, mirrored from ggml/src/ggml-cuda/ggml-cuda.cu. The
// setter is exposed as the "ggml_backend_cuda_set_kvarn_split_fused_exec"
// backend procedure.
//
typedef bool (*kvarn_split_fused_exec_fn)(void * user_data, struct ggml_tensor * dst);
typedef void (*kvarn_split_fused_set_fn)(kvarn_split_fused_exec_fn fn, void * user_data);

[[noreturn]] static void exec_fail(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char msg[512];
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);
    GGML_ABORT("kvarn split fused: %s", msg);
}

namespace {

// The op params are public slots (ggml_kvarn_split_fused_op_param in ggml.h);
// the typed getters are ggml-internal, so the reads are mirrored here.
int32_t op_param_i32(const ggml_tensor * op, int slot) {
    int32_t value = 0;
    std::memcpy(&value, &op->op_params[slot], sizeof(value));
    return value;
}

float op_param_f32(const ggml_tensor * op, int slot) {
    float value = 0.0f;
    std::memcpy(&value, &op->op_params[slot], sizeof(value));
    return value;
}

struct exec_state;

// Backing memory of the outer operands (tensor addresses AND buffer objects).
// The llama graph may hand a rebuilt graph the SAME tensor addresses while the
// underlying buffers are new (freed/reused arenas), so tensor pointers alone
// are not a sufficient identity for the persistent sub-graphs.
struct layer_binding {
    const void * data[11]   = {}; // src[0..9] + dst
    const void * buffer[11] = {};

    bool operator==(const layer_binding & o) const {
        for (int i = 0; i < 11; ++i) {
            if (data[i] != o.data[i] || buffer[i] != o.buffer[i]) {
                return false;
            }
        }
        return true;
    }
};

// Geometry of one fused span, derived from the outer operands and the op
// params. This is the stable per-layer contract: it does not change when the
// llama graph is rebuilt with fresh tensor objects of the same shapes.
struct layer_geometry {
    int64_t D     = 0;
    int64_t Hq    = 0;
    int64_t Hkv_k = 0;
    int64_t Hkv_v = 0;
    int64_t Q     = 0;
    int64_t B     = 0;
    int64_t n_tok = 0;
    int32_t n_ovf_rows = 0;
    float   kq_scale   = 0.0f;
    int32_t idxs_type  = 0;
    int64_t k_view_ne[2] = {}; // packed overflow window extents
    int64_t v_view_ne[2] = {};
    int64_t k_view_nb1 = 0;    // packed row strides (aliases bake them)
    int64_t v_view_nb1 = 0;
    int64_t k_store_rows = 0;  // companion storage row count (2D)
    int64_t v_store_rows = 0;
    int64_t k_store_nb1 = 0;
    int64_t v_store_nb1 = 0;
    int64_t mask_ne[3]  = {};  // mask ne[1..3]

    bool operator==(const layer_geometry & o) const {
        return D == o.D && Hq == o.Hq && Hkv_k == o.Hkv_k && Hkv_v == o.Hkv_v &&
            Q == o.Q && B == o.B && n_tok == o.n_tok && n_ovf_rows == o.n_ovf_rows &&
            kq_scale == o.kq_scale && idxs_type == o.idxs_type &&
            k_view_ne[0] == o.k_view_ne[0] && k_view_ne[1] == o.k_view_ne[1] &&
            v_view_ne[0] == o.v_view_ne[0] && v_view_ne[1] == o.v_view_ne[1] &&
            k_view_nb1 == o.k_view_nb1 && v_view_nb1 == o.v_view_nb1 &&
            k_store_rows == o.k_store_rows && v_store_rows == o.v_store_rows &&
            k_store_nb1 == o.k_store_nb1 && v_store_nb1 == o.v_store_nb1 &&
            mask_ne[0] == o.mask_ne[0] && mask_ne[1] == o.mask_ne[1] && mask_ne[2] == o.mask_ne[2];
    }
};

struct layer_exec {
    exec_state * state = nullptr;
    int il = -1;

    // Stable per-layer geometry and backing set; a mismatch on either rebuilds
    // the sub-graphs IN PLACE (same slot, old resources freed first).
    layer_geometry geom;
    layer_binding  binding;
    bool initialized = false;

    // Current outer binding (this call).
    ggml_tensor * src[10] = {};
    ggml_tensor * dst = nullptr;

    // Vulkan sub-graph (one per layer slot).
    ggml_context * vk_ctx     = nullptr;
    ggml_gallocr_t vk_gallocr = nullptr;
    ggml_cgraph *  vk_graph   = nullptr;
    ggml_tensor *  vk_q       = nullptr;
    ggml_tensor *  vk_kcur    = nullptr;
    ggml_tensor *  vk_vcur    = nullptr;
    ggml_tensor *  vk_mask    = nullptr;
    ggml_tensor *  vk_kidxs   = nullptr;
    ggml_tensor *  vk_vidxs   = nullptr;
    ggml_tensor *  vk_rows    = nullptr;
    ggml_tensor *  vk_o2      = nullptr;
    ggml_tensor *  vk_lse2    = nullptr;
    // Foreign operands as op-NONE aliases: raw packed window views and
    // companion storages, plus the SET_ROWS results that are views over the
    // storage aliases. They are created at build time from the current outer
    // tensors; a binding change rebuilds the sub-graphs (see get_layer).
    ggml_tensor *  vk_k_view_alias  = nullptr;
    ggml_tensor *  vk_v_view_alias  = nullptr;
    ggml_tensor *  vk_k_store_alias = nullptr;
    ggml_tensor *  vk_v_store_alias = nullptr;
    ggml_tensor *  vk_store_k       = nullptr;
    ggml_tensor *  vk_store_v       = nullptr;

    // CUDA merge sub-graph (one per layer slot).
    ggml_context * cu_ctx     = nullptr;
    ggml_gallocr_t cu_gallocr = nullptr;
    ggml_cgraph *  cu_graph   = nullptr;
    ggml_tensor *  cu_o2      = nullptr;
    ggml_tensor *  cu_lse2    = nullptr;

    // Host staging.
    std::vector<uint8_t> host_q;
    std::vector<uint8_t> host_k;
    std::vector<uint8_t> host_v;
    std::vector<uint8_t> host_mask;
    std::vector<uint8_t> host_kidxs;
    std::vector<uint8_t> host_vidxs;
    std::vector<float>   host_o2;
    std::vector<float>   host_lse2;

    void reset_subgraphs() {
        if (cu_gallocr) { ggml_gallocr_free(cu_gallocr); }
        if (cu_ctx)     { ggml_free(cu_ctx); }
        if (vk_gallocr) { ggml_gallocr_free(vk_gallocr); }
        if (vk_ctx)     { ggml_free(vk_ctx); }

        vk_ctx = nullptr; vk_gallocr = nullptr; vk_graph = nullptr;
        vk_q = nullptr; vk_kcur = nullptr; vk_vcur = nullptr; vk_mask = nullptr;
        vk_kidxs = nullptr; vk_vidxs = nullptr; vk_rows = nullptr;
        vk_o2 = nullptr; vk_lse2 = nullptr;
        vk_k_view_alias = nullptr; vk_v_view_alias = nullptr;
        vk_k_store_alias = nullptr; vk_v_store_alias = nullptr;
        vk_store_k = nullptr; vk_store_v = nullptr;

        cu_ctx = nullptr; cu_gallocr = nullptr; cu_graph = nullptr;
        cu_o2 = nullptr; cu_lse2 = nullptr;

        initialized = false;
    }

    ~layer_exec() { reset_subgraphs(); }
};

// One executor bundle per registering context (llama_kv_cache_kvarn): the
// bundle owns the per-layer sub-graphs and the backend pair. Bundles live in
// the immortal dispatcher map below and are referenced with shared_ptr, so an
// unregister of bundle A can never free state another context or an in-flight
// callback is using, and two contexts sharing the CUDA backend route by the
// bundle id baked into the op params (never by "the last registered pair").
struct exec_state {
    ggml_backend_t cuda   = nullptr;
    ggml_backend_t vulkan = nullptr;
    std::mutex mutex;
    // One executor per split layer, keyed by the stable layer id.
    std::unordered_map<int, std::unique_ptr<layer_exec>> layers;
};

// Immortal indirection object installed in the CUDA callback slot. Its
// address never changes and it outlives every bundle, so the slot only ever
// holds (exec_entry, &dispatcher): no callback/user pairing hazard is possible
// on the CUDA side, and bundle lifetime is refcounted on this side.
struct exec_dispatcher {
    std::mutex mutex;
    uint32_t next_id = 1;
    std::unordered_map<uint32_t, std::shared_ptr<exec_state>> bundles;
    kvarn_split_fused_set_fn set_fn = nullptr; // resolved CUDA-side slot setter
};

exec_dispatcher g_dispatcher;

// Diagnostic stages while LLAMA_OVF_FUSED_SPAN_STAGE is set; the executor is
// synchronous and otherwise silent on the hot path.
bool exec_stage_enabled() {
    static const bool enabled = [] {
        const char * v = std::getenv("LLAMA_OVF_FUSED_SPAN_STAGE");
        return v != nullptr && *v != '\0' && *v != '0';
    }();
    return enabled;
}

void exec_stage(const layer_exec & L, const char * where) {
    if (exec_stage_enabled()) {
        fprintf(stderr, "[FUSEDSPAN] il=%d %s\n", L.il, where);
    }
}

// Test/diagnostic dump of the span's internal Vulkan leg: LLAMA_OVF_FUSED_SPAN_DUMP
// names a file that receives, per call, one record { il, o2[], lse2[] } as raw
// binaries. Used by test-kvarn-split-fused to compare the REAL executor O2/LSE2
// (not a legacy proxy) against the reference leg and the FP64 oracle.
void exec_dump_leg(const layer_exec & L) {
    const char * path = std::getenv("LLAMA_OVF_FUSED_SPAN_DUMP");
    if (path == nullptr || *path == '\0') {
        return;
    }
    FILE * f = std::fopen(path, "ab");
    if (!f) {
        return;
    }
    const int32_t il = L.il;
    const uint64_t n_o2 = L.host_o2.size();
    const uint64_t n_lse2 = L.host_lse2.size();
    std::fwrite(&il, sizeof(il), 1, f);
    std::fwrite(&n_o2, sizeof(n_o2), 1, f);
    std::fwrite(&n_lse2, sizeof(n_lse2), 1, f);
    if (n_o2) {
        std::fwrite(L.host_o2.data(), sizeof(float), n_o2, f);
    }
    if (n_lse2) {
        std::fwrite(L.host_lse2.data(), sizeof(float), n_lse2, f);
    }
    std::fclose(f);
}

// External-input alias: an op-NONE leaf over an already allocated outer
// tensor. The nested sub-graphs must NOT reference the outer producer chain:
// ggml_build_forward_expand walks every source, so a raw outer node would drag
// its whole producer chain into the sub-graph, and re-adding the fused op node
// itself would re-enter this executor while its lock is held (deadlock).
// gallocr treats tensors with data set as external, so only the alias leaf is
// visible to the sub-graph and the data/buffer pointers are shared verbatim.
ggml_tensor * alias_leaf(ggml_context * ctx, ggml_tensor * outer, const char * what) {
    if (!outer || !outer->data || !outer->buffer) {
        exec_fail("alias_leaf(%s): outer tensor is not allocated", what);
    }
    if (!ggml_is_contiguous(outer)) {
        exec_fail("alias_leaf(%s): outer tensor is not contiguous", what);
    }
    ggml_tensor * t = ggml_new_tensor(ctx, outer->type, ggml_n_dims(outer), outer->ne);
    t->data   = outer->data;
    t->buffer = outer->buffer;
    ggml_format_name(t, "%s (fused alias)", outer->name);
    return t;
}

// Same, with an explicit 2D shape whose contiguous rows must match the outer
// row stride byte for byte (packed Q4_0 windows and companion storages).
ggml_tensor * alias_leaf_2d(ggml_context * ctx, const ggml_tensor * outer,
        int64_t ne0, int64_t ne1, const char * what) {
    if (!outer || !outer->data || !outer->buffer) {
        exec_fail("alias_leaf_2d(%s): outer tensor is not allocated", what);
    }
    if (outer->nb[1] != ggml_row_size(outer->type, ne0)) {
        exec_fail("alias_leaf_2d(%s): outer row stride does not match the packed row", what);
    }
    ggml_tensor * t = ggml_new_tensor_2d(ctx, outer->type, ne0, ne1);
    t->data   = outer->data;
    t->buffer = outer->buffer;
    ggml_format_name(t, "%s (fused alias)", outer->name);
    return t;
}

// -------------------------------------------------------------- sub-builds --

void build_vulkan_graph(layer_exec & L, int32_t n_ovf_rows) {
    exec_state & S = *L.state;

    const ggml_tensor * q      = L.src[0];
    const ggml_tensor * k_cur  = L.src[1];
    const ggml_tensor * v_cur  = L.src[2];
    const ggml_tensor * k_view = L.src[5];
    const ggml_tensor * v_view = L.src[6];
    const ggml_tensor * mask   = L.src[7];

    const ggml_tensor * k_storage = k_view->view_src;
    const ggml_tensor * v_storage = v_view->view_src;
    if (!k_storage || !v_storage) {
        exec_fail("packed overflow views lost their companion storage (il=%d)", L.il);
    }

    const int64_t D     = q->ne[0];
    const int64_t Hq    = q->ne[1];
    const int64_t n_tok = q->ne[2]*q->ne[3];
    const int64_t Hkv_k = k_cur->ne[1];
    const int64_t Hkv_v = v_cur->ne[1];
    const int64_t Dkv_k = D*Hkv_k;
    const int64_t Dkv_v = D*Hkv_v;
    const int64_t n_ovf = n_ovf_rows;
    const int64_t k_store_rows = k_storage->ne[1]*k_storage->ne[2];
    const int64_t v_store_rows = v_storage->ne[1]*v_storage->ne[2];

    ggml_init_params params = {
        /* .mem_size   = */ 4u*1024u*1024u,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    L.vk_ctx = ggml_init(params);
    if (!L.vk_ctx) {
        exec_fail("Vulkan sub-graph context allocation failed (il=%d)", L.il);
    }

    L.vk_q     = ggml_new_tensor_4d(L.vk_ctx, GGML_TYPE_F32, D, Hq, q->ne[2], q->ne[3]);
    L.vk_kcur  = ggml_new_tensor_4d(L.vk_ctx, GGML_TYPE_F32, D, Hkv_k, k_cur->ne[2], k_cur->ne[3]);
    L.vk_vcur  = ggml_new_tensor_4d(L.vk_ctx, GGML_TYPE_F32, D, Hkv_v, v_cur->ne[2], v_cur->ne[3]);
    L.vk_mask  = ggml_new_tensor_4d(L.vk_ctx, GGML_TYPE_F16, n_ovf, mask->ne[1], mask->ne[2], mask->ne[3]);
    L.vk_kidxs = ggml_new_tensor_1d(L.vk_ctx, L.src[8]->type, n_tok);
    L.vk_vidxs = ggml_new_tensor_1d(L.vk_ctx, L.src[9]->type, n_tok);
    L.vk_rows  = ggml_new_tensor_1d(L.vk_ctx, GGML_TYPE_I32, n_ovf);

    // Foreign operands: op-NONE aliases over the outer packed window views and
    // the companion storages (get_layer rebuilds these sub-graphs when the
    // outer binding changes; nothing is re-pointed in place).
    L.vk_k_view_alias  = alias_leaf_2d(L.vk_ctx, k_view,  Dkv_k, n_ovf, "k_ovf view");
    L.vk_v_view_alias  = alias_leaf_2d(L.vk_ctx, v_view,  Dkv_v, n_ovf, "v_ovf view");
    L.vk_k_store_alias = alias_leaf_2d(L.vk_ctx, k_storage, Dkv_k, k_store_rows, "k_ovf storage");
    L.vk_v_store_alias = alias_leaf_2d(L.vk_ctx, v_storage, Dkv_v, v_store_rows, "v_ovf storage");

    // Store: the ops llama_kv_cache::cpy_k/cpy_v emit for the Q4 companion
    // (flatten heads, SET_ROWS with the payload-row index list over the full
    // storage). The unmodified index lists are used directly.
    ggml_tensor * kcur_2d = ggml_view_2d(L.vk_ctx, L.vk_kcur,
            Dkv_k, k_cur->ne[2], L.vk_kcur->nb[2], 0);
    ggml_tensor * vcur_2d = ggml_view_2d(L.vk_ctx, L.vk_vcur,
            Dkv_v, v_cur->ne[2], L.vk_vcur->nb[2], 0);
    L.vk_store_k = ggml_set_rows(L.vk_ctx, L.vk_k_store_alias, kcur_2d, L.vk_kidxs);
    L.vk_store_v = ggml_set_rows(L.vk_ctx, L.vk_v_store_alias, vcur_2d, L.vk_vidxs);
    ggml_set_name(L.vk_store_k, "kvarn_split_fused_store_k");
    ggml_set_name(L.vk_store_v, "kvarn_split_fused_store_v");

    // Gather: the F16 readback over the packed window with identity rows,
    // mirrored from build_attn_position_split's packed-view read.
    ggml_tensor * k16 = ggml_get_rows_as(L.vk_ctx, L.vk_k_view_alias, L.vk_rows, GGML_TYPE_F16);
    ggml_tensor * v16 = ggml_get_rows_as(L.vk_ctx, L.vk_v_view_alias, L.vk_rows, GGML_TYPE_F16);
    ggml_tensor * k4  = ggml_reshape_4d(L.vk_ctx, k16, D, Hkv_k, n_ovf, 1);
    ggml_tensor * v4  = ggml_reshape_4d(L.vk_ctx, v16, D, Hkv_v, n_ovf, 1);
    ggml_tensor * kp  = ggml_permute(L.vk_ctx, k4, 0, 2, 1, 3);
    ggml_tensor * vp  = ggml_permute(L.vk_ctx, v4, 0, 2, 1, 3);

    // Query exactly like build_attn_mha: view (n_stream == 1) then permute.
    ggml_tensor * q_view = ggml_view_4d(L.vk_ctx, L.vk_q,
            D, Hq, q->ne[2], q->ne[3],
            L.vk_q->nb[1], L.vk_q->nb[2], L.vk_q->nb[3], 0);
    ggml_tensor * qp = ggml_permute(L.vk_ctx, q_view, 0, 2, 1, 3);

    ggml_tensor * fa2 = ggml_flash_attn_ext(L.vk_ctx, qp, kp, vp, L.vk_mask,
            L.geom.kq_scale, 0.0f, 0.0f);
    ggml_prec_set_acc(fa2, GGML_PREC_F32);
    // Same n_kv_max the legacy split FA carries; domain stays AUTO (0), as
    // build_attn_position_split sets it for the overflow range.
    ggml_flash_attn_ext_set_n_kv_max(fa2, int32_t(n_ovf));
    L.vk_o2   = fa2;
    L.vk_lse2 = ggml_flash_attn_ext_lse_out(L.vk_ctx, fa2);

    L.vk_graph = ggml_new_graph_custom(L.vk_ctx, 128, false);
    // Order matters on a single backend: the stores must precede the gather.
    ggml_build_forward_expand(L.vk_graph, L.vk_store_k);
    ggml_build_forward_expand(L.vk_graph, L.vk_store_v);
    ggml_build_forward_expand(L.vk_graph, fa2);
    ggml_build_forward_expand(L.vk_graph, L.vk_lse2);

    L.vk_gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(S.vulkan));
    if (!L.vk_gallocr || !ggml_gallocr_reserve(L.vk_gallocr, L.vk_graph) ||
            !ggml_gallocr_alloc_graph(L.vk_gallocr, L.vk_graph)) {
        exec_fail("Vulkan sub-graph allocation failed (il=%d)", L.il);
    }

    // Static staging content: the identity row list for the gather.
    std::vector<int32_t> rows;
    rows.reserve(size_t(n_ovf));
    for (int64_t i = 0; i < n_ovf; ++i) {
        rows.push_back(int32_t(i));
    }
    ggml_backend_tensor_set(L.vk_rows, rows.data(), 0, rows.size()*sizeof(int32_t));
}

// CUDA merge sub-graph: the exact op sequence of the legacy merge in
// build_attn_position_split (clamp/abs-max/exp/weighted add/div/cont), with
// O2/LSE2 read back from the Vulkan leg. The final cpy writes dst.
void build_cuda_merge_graph(layer_exec & L) {
    exec_state & S = *L.state;

    const ggml_tensor * q       = L.src[0];
    const ggml_tensor * o_local = L.src[3];
    // The merge ops take mutable tensors; the executor never writes them.
    ggml_tensor * lse = const_cast<ggml_tensor *>(L.src[4]);

    const int64_t D     = q->ne[0];
    const int64_t Hq    = q->ne[1];
    const int64_t Q     = q->ne[2];
    const int64_t B     = q->ne[3];
    const int64_t n_tok = Q*B;

    ggml_init_params params = {
        /* .mem_size   = */ 2u*1024u*1024u,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    L.cu_ctx = ggml_init(params);
    if (!L.cu_ctx) {
        exec_fail("CUDA merge sub-graph context allocation failed (il=%d)", L.il);
    }

    // FA output layout is [D, H_q, Q, B] (v->ne[0], q->ne[2], q->ne[1], B) and
    // LSE2 is [H_q, Q, B]; the merge restore reshape is the identity here.
    L.cu_o2   = ggml_new_tensor_4d(L.cu_ctx, GGML_TYPE_F32, D, Hq, Q, B);
    L.cu_lse2 = ggml_new_tensor_3d(L.cu_ctx, GGML_TYPE_F32, Hq, Q, B);

    // Outer operands enter as alias leaves: the sub-graph must not walk into
    // their producer chain (see alias_leaf).
    ggml_tensor * o1_alias  = alias_leaf(L.cu_ctx, const_cast<ggml_tensor *>(o_local), "o_local");
    ggml_tensor * lse_alias = alias_leaf(L.cu_ctx, lse, "lse_local");
    ggml_tensor * dst_alias = alias_leaf(L.cu_ctx, L.dst, "dst");

    // Merge: byte-for-byte the arithmetic of the legacy build (5739-5776).
    ggml_tensor * o1_4d = ggml_reshape_4d(L.cu_ctx, o1_alias, D, Hq, Q, B);
    ggml_tensor * o2_4d = ggml_reshape_4d(L.cu_ctx, L.cu_o2, D, Hq, Q, B);
    ggml_tensor * c1 = ggml_clamp(L.cu_ctx, lse_alias, -1e6f, 1e30f);
    ggml_tensor * c2 = ggml_clamp(L.cu_ctx, L.cu_lse2, -1e6f, 1e30f);
    ggml_tensor * mx = ggml_scale(L.cu_ctx,
            ggml_add(L.cu_ctx, ggml_add(L.cu_ctx, c1, c2), ggml_abs(L.cu_ctx, ggml_sub(L.cu_ctx, c1, c2))),
            0.5f);
    ggml_tensor * e1 = ggml_exp(L.cu_ctx, ggml_sub(L.cu_ctx, lse_alias, mx));
    ggml_tensor * e2 = ggml_exp(L.cu_ctx, ggml_sub(L.cu_ctx, L.cu_lse2, mx));
    ggml_tensor * w1 = ggml_reshape_4d(L.cu_ctx, e1, 1, Hq, Q, B);
    ggml_tensor * w2 = ggml_reshape_4d(L.cu_ctx, e2, 1, Hq, Q, B);
    ggml_tensor * num = ggml_add(L.cu_ctx, ggml_mul(L.cu_ctx, o1_4d, w1), ggml_mul(L.cu_ctx, o2_4d, w2));
    ggml_tensor * den = ggml_add(L.cu_ctx, e1, e2);
    ggml_tensor * den_4d = ggml_reshape_4d(L.cu_ctx, den, 1, Hq, Q, B);
    ggml_tensor * o_4d = ggml_div(L.cu_ctx, num, den_4d);
    ggml_tensor * o_2d = ggml_reshape_2d(L.cu_ctx, o_4d, D*Hq, n_tok);
    // Fresh contiguous output, same ownership as the legacy o_merged, then a
    // copy into the (aliased) op output.
    ggml_tensor * o_cont = ggml_cont_2d(L.cu_ctx, o_2d, D*Hq, n_tok);

    if (L.dst->ne[0] != D*Hq || L.dst->ne[1] != n_tok) {
        exec_fail("output shape mismatch (il=%d): dst [%" PRId64 ",%" PRId64 "] vs [%" PRId64 ",%" PRId64 "]",
                L.il, (int64_t) L.dst->ne[0], (int64_t) L.dst->ne[1], D*Hq, n_tok);
    }
    ggml_tensor * cu_copy = ggml_cpy(L.cu_ctx, o_cont, dst_alias);

    L.cu_graph = ggml_new_graph_custom(L.cu_ctx, 128, false);
    // ggml_new_graph_custom() leaves uid == 0, and
    // ggml_cuda_graph_check_compability() refuses capture for uid == 0: this
    // graph is executed nested inside the outer CUDA compute, where stream
    // capture must not start. Do not assign a uid here.
    ggml_build_forward_expand(L.cu_graph, cu_copy);

    L.cu_gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(S.cuda));
    if (!L.cu_gallocr || !ggml_gallocr_reserve(L.cu_gallocr, L.cu_graph) ||
            !ggml_gallocr_alloc_graph(L.cu_gallocr, L.cu_graph)) {
        exec_fail("CUDA merge sub-graph allocation failed (il=%d)", L.il);
    }
}

// ------------------------------------------------------------ validation ----

// Backing memory of the current outer binding: tensor addresses and buffer
// objects for src[0..9] and the op output.
layer_binding binding_of(ggml_tensor * const * src, ggml_tensor * dst) {
    layer_binding b;
    for (int i = 0; i < 10; ++i) {
        b.data[i]   = src[i] ? src[i]->data : nullptr;
        b.buffer[i] = src[i] ? src[i]->buffer : nullptr;
    }
    b.data[10]   = dst ? dst->data : nullptr;
    b.buffer[10] = dst ? dst->buffer : nullptr;
    return b;
}

// Geometry of the current outer binding + op params (see the caching policy
// at the top of the file). A mismatch against the cached geometry rebuilds
// the sub-graphs in place.
layer_geometry geometry_of(ggml_tensor * const * src, ggml_tensor * dst,
        int32_t n_ovf_rows, float kq_scale) {
    const ggml_tensor * q      = src[0];
    const ggml_tensor * k_cur  = src[1];
    const ggml_tensor * v_cur  = src[2];
    const ggml_tensor * k_view = src[5];
    const ggml_tensor * v_view = src[6];
    const ggml_tensor * mask   = src[7];

    if (!k_view->view_src || !v_view->view_src) {
        exec_fail("packed overflow views lost their companion storage (il geometry)");
    }
    const ggml_tensor * k_storage = k_view->view_src;
    const ggml_tensor * v_storage = v_view->view_src;

    layer_geometry g;
    g.D         = q->ne[0];
    g.Hq        = q->ne[1];
    g.Hkv_k     = k_cur->ne[1];
    g.Hkv_v     = v_cur->ne[1];
    g.Q         = q->ne[2];
    g.B         = q->ne[3];
    g.n_tok     = g.Q*g.B;
    g.n_ovf_rows = n_ovf_rows;
    g.kq_scale   = kq_scale;
    g.idxs_type  = int32_t(src[8]->type);
    g.k_view_ne[0] = k_view->ne[0]; g.k_view_ne[1] = k_view->ne[1];
    g.v_view_ne[0] = v_view->ne[0]; g.v_view_ne[1] = v_view->ne[1];
    g.k_view_nb1 = k_view->nb[1];
    g.v_view_nb1 = v_view->nb[1];
    g.k_store_rows = k_storage->ne[1]*k_storage->ne[2];
    g.v_store_rows = v_storage->ne[1]*v_storage->ne[2];
    g.k_store_nb1 = k_storage->nb[1];
    g.v_store_nb1 = v_storage->nb[1];
    g.mask_ne[0] = mask->ne[1]; g.mask_ne[1] = mask->ne[2]; g.mask_ne[2] = mask->ne[3];
    if (!dst) {
        exec_fail("fused span has no output tensor");
    }
    return g;
}

layer_exec & get_layer(exec_state & S, ggml_tensor * const * src, ggml_tensor * dst,
        int32_t il, int32_t n_ovf_rows, float kq_scale) {
    if (il < 0 || il >= 4096) {
        exec_fail("layer id out of range: %d", il);
    }

    auto it = S.layers.find(il);
    if (it == S.layers.end()) {
        auto entry = std::make_unique<layer_exec>();
        entry->state = &S;
        entry->il = il;
        it = S.layers.emplace(il, std::move(entry)).first;
    }
    layer_exec & L = *it->second;

    // Record the current binding. The sub-graphs are reused only while the
    // geometry AND the backing memory (tensor addresses and buffer objects)
    // are unchanged; any change rebuilds them in place. In-place pointer
    // repair of the aliases was tried and rejected: after an outer rebuild the
    // Vulkan kernel path kept resolving the gather through the previous
    // backing set (deterministically wrong output), so this slot always
    // rebuilds instead of rebinding. See the file header.
    L.dst = dst;
    for (int i = 0; i < 10; ++i) {
        L.src[i] = src[i];
    }

    const layer_geometry geom = geometry_of(L.src, L.dst, n_ovf_rows, kq_scale);
    const layer_binding  bind = binding_of(L.src, L.dst);
    if (!L.initialized || !(L.geom == geom) || !(L.binding == bind)) {
        if (L.initialized && exec_stage_enabled()) {
            fprintf(stderr, "[FUSEDSPAN] il=%d %s -> rebuilding sub-graphs\n", il,
                    (L.geom == geom) ? "rebinding (new backing)" : "geometry changed");
        }
        L.reset_subgraphs();
        L.geom = geom;
        L.binding = bind;

        // Host staging sizes; the device copies in run_layer are the only
        // transfers.
        L.host_q.assign(ggml_nbytes(L.src[0]), 0);
        L.host_k.assign(ggml_nbytes(L.src[1]), 0);
        L.host_v.assign(ggml_nbytes(L.src[2]), 0);
        L.host_mask.assign(ggml_nbytes(L.src[7]), 0);
        L.host_kidxs.assign(ggml_nbytes(L.src[8]), 0);
        L.host_vidxs.assign(ggml_nbytes(L.src[9]), 0);

        build_vulkan_graph(L, n_ovf_rows);
        build_cuda_merge_graph(L);

        L.host_o2.assign(size_t(ggml_nelements(L.vk_o2)), 0.0f);
        L.host_lse2.assign(size_t(ggml_nelements(L.vk_lse2)), 0.0f);

        L.initialized = true;
    }

    return L;
}

// ---------------------------------------------------------------- runtime ---

void run_layer(exec_state & S, layer_exec & L) {
    exec_stage(L, "enter");
    // 0. Defense in depth: the pre-local Vulkan split (Q WHT) may still be in
    //    flight on the scheduler side; drain both backends before touching any
    //    operand. The CUDA dispatch already synchronized its own stream.
    ggml_backend_synchronize(S.vulkan);
    // 1. Device -> host staging for the operands Vulkan has to see. The CUDA
    //    side already synchronized its stream before entering the executor and
    //    every buffer-sourced read below is itself synchronous.
    ggml_backend_tensor_get(L.src[0], L.host_q.data(), 0, L.host_q.size());
    ggml_backend_tensor_get(L.src[1], L.host_k.data(), 0, L.host_k.size());
    ggml_backend_tensor_get(L.src[2], L.host_v.data(), 0, L.host_v.size());
    ggml_backend_tensor_get(L.src[7], L.host_mask.data(), 0, L.host_mask.size());
    ggml_backend_tensor_get(L.src[8], L.host_kidxs.data(), 0, L.host_kidxs.size());
    ggml_backend_tensor_get(L.src[9], L.host_vidxs.data(), 0, L.host_vidxs.size());

    // 2. Host -> Vulkan staging. src[5]/src[6] and the storage stay in place:
    //    the sub-graph reads them through aliased views.
    ggml_backend_tensor_set(L.vk_q,     L.host_q.data(),     0, L.host_q.size());
    ggml_backend_tensor_set(L.vk_kcur,  L.host_k.data(),     0, L.host_k.size());
    ggml_backend_tensor_set(L.vk_vcur,  L.host_v.data(),     0, L.host_v.size());
    ggml_backend_tensor_set(L.vk_mask,  L.host_mask.data(),  0, L.host_mask.size());
    ggml_backend_tensor_set(L.vk_kidxs, L.host_kidxs.data(), 0, L.host_kidxs.size());
    ggml_backend_tensor_set(L.vk_vidxs, L.host_vidxs.data(), 0, L.host_vidxs.size());

    exec_stage(L, "staged");
    // 3. Store + gather + FA2 + LSE2 on Vulkan (synchronous submission).
    if (ggml_backend_graph_compute(S.vulkan, L.vk_graph) != GGML_STATUS_SUCCESS) {
        exec_fail("Vulkan sub-graph compute failed (il=%d)", L.il);
    }
    exec_stage(L, "vulkan-done");

    // 4. Read O2/LSE2 back and stage them for the CUDA merge.
    ggml_backend_tensor_get(L.vk_o2,   L.host_o2.data(),   0, L.host_o2.size()*sizeof(float));
    ggml_backend_tensor_get(L.vk_lse2, L.host_lse2.data(), 0, L.host_lse2.size()*sizeof(float));
    exec_dump_leg(L);
    ggml_backend_tensor_set(L.cu_o2,   L.host_o2.data(),   0, L.host_o2.size()*sizeof(float));
    ggml_backend_tensor_set(L.cu_lse2, L.host_lse2.data(), 0, L.host_lse2.size()*sizeof(float));
    exec_stage(L, "readback-done");

    // 5. Merge on CUDA and write the op output. This re-enters the CUDA
    //    backend (a nested ggml graph compute on the same stream); the outer
    //    split cannot be CUDA-graph captured precisely because of this.
    if (ggml_backend_graph_compute(S.cuda, L.cu_graph) != GGML_STATUS_SUCCESS) {
        exec_fail("CUDA merge sub-graph compute failed (il=%d)", L.il);
    }
    exec_stage(L, "cuda-done");
}

bool exec_entry(void * user_data, ggml_tensor * dst) {
    if (user_data != &g_dispatcher) {
        exec_fail("executor received an unexpected user pointer");
    }
    if (!dst || dst->op != GGML_OP_KVARN_SPLIT_FUSED) {
        exec_fail("executor received a non fused-split node");
    }

    // Resolve the registering context's bundle by the id baked into the op by
    // the graph builder. The shared_ptr copy keeps the bundle alive for this
    // call even if the context unregisters concurrently.
    const int32_t bundle_id = op_param_i32(dst, GGML_KVARN_SPLIT_FUSED_PARAM_BUNDLE_ID);
    std::shared_ptr<exec_state> S;
    {
        std::lock_guard<std::mutex> lock(g_dispatcher.mutex);
        const auto it = g_dispatcher.bundles.find(uint32_t(bundle_id));
        if (it != g_dispatcher.bundles.end()) {
            S = it->second;
        }
    }
    if (!S) {
        exec_fail("no executor bundle %d registered (destroyed context or missing registration)", bundle_id);
    }
    if (!S->cuda || !S->vulkan) {
        exec_fail("executor has no backends");
    }

    ggml_tensor * src[10];
    for (int i = 0; i < 10; ++i) {
        src[i] = dst->src[i];
        if (!src[i]) {
            exec_fail("missing source %d", i);
        }
    }

    const int     n_ovf_rows = op_param_i32(dst, GGML_KVARN_SPLIT_FUSED_PARAM_N_OVF_ROWS);
    const float   kq_scale   = op_param_f32(dst, GGML_KVARN_SPLIT_FUSED_PARAM_KQ_SCALE);
    const int32_t il         = op_param_i32(dst, GGML_KVARN_SPLIT_FUSED_PARAM_LAYER);
    if (n_ovf_rows <= 0 || !(kq_scale > 0.0f) || il < 0) {
        exec_fail("invalid op params: il=%d n_ovf_rows=%d kq_scale=%f", il, n_ovf_rows, kq_scale);
    }

    // The local results and the output must be readable/writable by the CUDA
    // merge sub-graph: the builder pins them to this backend.
    const ggml_backend_buffer_type_t cuda_buft = ggml_backend_get_default_buffer_type(S->cuda);
    for (int i : {3, 4}) {
        if (!src[i]->buffer || ggml_backend_buffer_get_type(src[i]->buffer) != cuda_buft) {
            exec_fail("src[%d] is not on the registered CUDA backend (il=%d)", i, il);
        }
    }
    if (!dst->buffer || ggml_backend_buffer_get_type(dst->buffer) != cuda_buft) {
        exec_fail("fused output is not on the registered CUDA backend (il=%d)", il);
    }

    std::lock_guard<std::mutex> lock(S->mutex);
    layer_exec & L = get_layer(*S, src, dst, il, n_ovf_rows, kq_scale);
    run_layer(*S, L);
    return true;
}

} // namespace

// ------------------------------------------------------------ registration --

uint32_t llama_kvarn_split_fused_exec_register(ggml_backend_t cuda, ggml_backend_t vulkan) {
    if (!cuda || !vulkan) {
        return 0;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(cuda);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg) {
        return 0;
    }
    auto * set_fn = (kvarn_split_fused_set_fn)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_set_kvarn_split_fused_exec");
    if (!set_fn) {
        return 0;
    }

    std::lock_guard<std::mutex> lock(g_dispatcher.mutex);
    if (g_dispatcher.next_id == 0) {
        return 0; // exhausted after 2^32 registrations
    }
    const uint32_t id = g_dispatcher.next_id++;
    auto bundle = std::make_shared<exec_state>();
    bundle->cuda = cuda;
    bundle->vulkan = vulkan;
    g_dispatcher.bundles.emplace(id, std::move(bundle));
    g_dispatcher.set_fn = set_fn;

    // The slot always points at the immortal dispatcher; installing it again
    // is harmless (idempotent pair). Done under the dispatcher lock so a
    // concurrent removal cannot clear a slot we just installed.
    set_fn(exec_entry, &g_dispatcher);
    return id;
}

void llama_kvarn_split_fused_exec_unregister(uint32_t id) {
    if (id == 0) {
        return;
    }
    bool empty = false;
    {
        std::lock_guard<std::mutex> lock(g_dispatcher.mutex);
        g_dispatcher.bundles.erase(id);
        empty = g_dispatcher.bundles.empty();
    }
    // Clearing the slot is done OUTSIDE the dispatcher lock to keep a single
    // lock order (registration takes dispatcher -> CUDA-slot; a dispatch grabs
    // the CUDA slot and then, inside the callback, the dispatcher).
    if (empty && g_dispatcher.set_fn) {
        g_dispatcher.set_fn(nullptr, nullptr);
    }
}

bool llama_kvarn_split_fused_exec_ready(uint32_t id) {
    std::lock_guard<std::mutex> lock(g_dispatcher.mutex);
    return g_dispatcher.bundles.count(id) != 0;
}
