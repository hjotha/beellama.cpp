#include "llama-kv-kvarn-materializer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace llama_kv_qx {

const char * mat_status_name(mat_status st) {
    switch (st) {
        case mat_status::ok:                  return "ok";
        case mat_status::invalid_args:        return "invalid_args";
        case mat_status::invalid_geometry:    return "invalid_geometry";
        case mat_status::invalid_indices:     return "invalid_indices";
        case mat_status::unsupported_backend: return "unsupported_backend";
        case mat_status::workspace_overflow:  return "workspace_overflow";
        case mat_status::compute_failure:     return "compute_failure";
        case mat_status::non_finite_input:    return "non_finite_input";
    }
    return "unknown";
}

namespace {

constexpr uint32_t KVAR_N_GROUP = 128;
constexpr size_t   MAX_GRAPH_ENTRIES = 8;

bool valid_kvarn_bits(int bits) {
    return bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8;
}

size_t record_bytes(uint32_t record_dim, int bits, bool value) {
    const size_t rows = value ? 128 : record_dim;
    const size_t cols = value ? record_dim : 128;
    return (rows * cols * size_t(bits) + 7) / 8 +
        (2 * rows + cols) * sizeof(ggml_fp16_t);
}

bool hadamard_n(float * values, uint32_t width) {
    if (values == nullptr || (width != 64 && width != KVAR_N_GROUP)) {
        return false;
    }
    for (uint32_t stride = 1; stride < width; stride *= 2) {
        for (uint32_t base = 0; base < width; base += 2 * stride) {
            for (uint32_t i = 0; i < stride; ++i) {
                const float a = values[base + i];
                const float b = values[base + stride + i];
                values[base + i] = a + b;
                values[base + stride + i] = a - b;
            }
        }
    }
    const float scale = width == 64 ? 0.125f : 0.08838834764831845f;
    for (uint32_t i = 0; i < width; ++i) {
        values[i] *= scale;
    }
    return true;
}

// Canonical KVarN head transform (self-inverse): WHT-128 per slice, then
// inter-slice mixing. Rotates an ORIGINAL-domain head row into the rotated
// source domain and back.
bool wht_head(float * values, uint32_t head_dim, uint32_t record_dim) {
    if (values == nullptr) {
        return false;
    }
    if (record_dim == 64) {
        return head_dim == 64 && hadamard_n(values, 64);
    }
    if (record_dim != KVAR_N_GROUP || head_dim % KVAR_N_GROUP != 0) {
        return false;
    }
    const uint32_t slices = head_dim / KVAR_N_GROUP;
    if (slices < 1 || slices > 4) {
        return false;
    }
    for (uint32_t s = 0; s < slices; ++s) {
        if (!hadamard_n(values + s * KVAR_N_GROUP, KVAR_N_GROUP)) {
            return false;
        }
    }
    if (slices == 1) {
        return true;
    }
    const float scale = slices == 2 ? 0.7071067811865475f : 0.5f;
    for (uint32_t d = 0; d < KVAR_N_GROUP; ++d) {
        float x[4] = {};
        for (uint32_t s = 0; s < slices; ++s) {
            x[s] = values[s * KVAR_N_GROUP + d];
        }
        for (uint32_t stride = 1; stride < slices; stride <<= 1) {
            for (uint32_t base = 0; base < slices; base += 2 * stride) {
                for (uint32_t i = 0; i < stride; ++i) {
                    const float a = x[base + i];
                    const float b = x[base + stride + i];
                    x[base + i] = a + b;
                    x[base + stride + i] = a - b;
                }
            }
        }
        for (uint32_t s = 0; s < slices; ++s) {
            values[s * KVAR_N_GROUP + d] = x[s] * scale;
        }
    }
    return true;
}

} // namespace

materializer::materializer(const materializer_config & cfg, const materializer_source & src)
        : cfg_(cfg), src_(src) {
    last_ = validate_source();
}

materializer::~materializer() {
    for (auto & g : graphs_) {
        if (g.buf) {
            ggml_backend_buffer_free(g.buf);
        }
        if (g.ctx) {
            ggml_free(g.ctx);
        }
    }
}

mat_status materializer::validate_source() const {
    if (src_.records == nullptr || src_.stage == nullptr || src_.backend == nullptr) {
        return mat_status::invalid_args;
    }
    // Leaves only: submitting the graph through op-carrying tensors (e.g. a
    // store-op view or a cache *_stream view) would re-execute old producer
    // writes or read through a live op chain.
    if (src_.records->op != GGML_OP_NONE || src_.stage->op != GGML_OP_NONE) {
        return mat_status::invalid_args;
    }
    // Backend-safety: the compute kernel (e.g. ggml_cuda_kvarn_materialize)
    // requires contiguous tensors and dereferences device pointers directly.
    // Reject strided/unallocated leaves and buffers the backend cannot use.
    // The buffer-type compatibility check uses the native backend device API
    // (ggml_backend_supports_buft); it is not based on backend names, so
    // pinned/shared host buffers that the device genuinely accepts keep
    // working.
    for (const ggml_tensor * t : { src_.records, src_.stage }) {
        if (t->buffer == nullptr || !ggml_is_contiguous(t)) {
            return mat_status::invalid_geometry;
        }
        if (!ggml_backend_supports_buft(src_.backend,
                ggml_backend_buffer_get_type(t->buffer))) {
            return mat_status::invalid_geometry;
        }
    }
    if (cfg_.head_dim != 64 && cfg_.head_dim != 128 &&
            cfg_.head_dim != 256 && cfg_.head_dim != 512) {
        return mat_status::invalid_geometry;
    }
    if (cfg_.n_heads == 0 || cfg_.n_heads > 4096) {
        return mat_status::invalid_geometry;
    }
    if (cfg_.record_dim != 64 && cfg_.record_dim != KVAR_N_GROUP) {
        return mat_status::invalid_geometry;
    }
    if (cfg_.record_dim == 64 && cfg_.head_dim != 64) {
        return mat_status::invalid_geometry;
    }
    if (cfg_.record_dim == KVAR_N_GROUP &&
            (cfg_.head_dim % KVAR_N_GROUP != 0 || cfg_.head_dim / KVAR_N_GROUP > 4)) {
        return mat_status::invalid_geometry;
    }
    if (cfg_.head_dim / cfg_.record_dim != cfg_.head_slices) {
        return mat_status::invalid_geometry;
    }
    if (!valid_kvarn_bits(cfg_.bits)) {
        return mat_status::invalid_geometry;
    }
    if (cfg_.stage_groups < 2) {
        return mat_status::invalid_geometry;
    }
    if (cfg_.max_window == 0 || cfg_.max_window > LLAMA_KV_MATERIALIZER_MAX_WINDOW) {
        return mat_status::invalid_geometry;
    }

    const uint64_t phys_heads = uint64_t(cfg_.n_heads) * cfg_.head_slices;
    if (src_.stage->type != GGML_TYPE_F16 || src_.records->type != GGML_TYPE_I8) {
        return mat_status::invalid_geometry;
    }
    if (src_.stage->ne[0] != int64_t(cfg_.record_dim) ||
            src_.stage->ne[1] != int64_t(phys_heads) ||
            src_.records->ne[1] != int64_t(phys_heads)) {
        return mat_status::invalid_geometry;
    }
    if (src_.records->ne[0] != int64_t(record_bytes(cfg_.record_dim, cfg_.bits, cfg_.value))) {
        return mat_status::invalid_geometry;
    }
    // Single stream: the stage must be exactly 128*stage_groups rows deep.
    if (src_.stage->ne[2] != int64_t(KVAR_N_GROUP) * cfg_.stage_groups) {
        return mat_status::invalid_geometry;
    }
    if (src_.records->ne[2] <= 0) {
        return mat_status::invalid_geometry;
    }
    if (src_.records->ne[3] != 1 || src_.stage->ne[3] != 1) {
        return mat_status::invalid_geometry;
    }
    return mat_status::ok;
}

mat_status materializer::set_prefix(const int64_t * indices, uint32_t n_tokens,
        exact_token_fn exact, void * exact_user) {
    // Any failure invalidates the installed map: a stale prefix must never
    // be served after a failed set_prefix.
    const auto invalidate = [this]() {
        prefix_.clear();
        n_tokens_ = 0;
        exact_ = nullptr;
        exact_user_ = nullptr;
        exact_scratch_.clear();
    };
    try {
        last_ = validate_source();
        if (last_ != mat_status::ok) {
            invalidate();
            return last_;
        }
        if (indices == nullptr || n_tokens == 0) {
            last_ = mat_status::invalid_args;
            invalidate();
            return last_;
        }
        const uint64_t groups_per_stream = uint64_t(src_.records->ne[2]);
        // Physical cell space of the cache: records groups x 128 cells
        // (single stream). The owner's map may reference sparse/high
        // physical cells (e.g. a compact read plan after rollback), so the
        // bound is PHYSICAL capacity -- not the logical token count.
        const uint64_t physical_capacity = groups_per_stream * KVAR_N_GROUP;

        // Eager validation of the whole effective map before any backend compute.
        for (uint32_t i = 0; i < n_tokens; ++i) {
            const int64_t enc = indices[i];
            if (enc == -1) {
                continue; // hole (fixed-shape padding); materializes as zeros
            }
            uint32_t cell = 0;
            if (enc < -1) {
                // Explicit F16 stage-slot encoding (llama_kvarn_encode_stage_cell).
                const uint64_t payload = uint64_t(-(enc + 2));
                const uint32_t packed_slot = uint32_t(payload >> 32u);
                if (packed_slot == 0 || int64_t(packed_slot - 1) >= cfg_.stage_groups) {
                    last_ = mat_status::invalid_indices;
                    invalidate();
                    return last_;
                }
                cell = uint32_t(payload);
            } else {
                // Plain absolute physical position (sealed-record cell).
                // Store-cell encodings (high word) are write-side and invalid
                // for reads.
                if (uint64_t(enc) >> 32u != 0) {
                    last_ = mat_status::invalid_indices;
                    invalidate();
                    return last_;
                }
                cell = uint32_t(enc);
            }
            if (uint64_t(cell) >= physical_capacity) {
                last_ = mat_status::invalid_indices;
                invalidate();
                return last_;
            }
        }

        prefix_.assign(indices, indices + n_tokens);
        n_tokens_ = n_tokens;
        exact_ = exact;
        exact_user_ = exact_user;
        exact_scratch_.assign(size_t(cfg_.head_dim) * cfg_.n_heads, 0.0f);
        last_ = mat_status::ok;
        return last_;
    } catch (const std::exception &) {
        // The interface is status-based and never throws.
        invalidate();
        last_ = mat_status::workspace_overflow;
        return last_;
    }
}

mat_status materializer::ensure_graph(uint32_t n_window) {
    for (const auto & g : graphs_) {
        if (g.n_window == n_window) {
            return mat_status::ok;
        }
    }
    if (graphs_.size() >= MAX_GRAPH_ENTRIES) {
        return mat_status::workspace_overflow;
    }

    ggml_init_params params = { 512 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return mat_status::workspace_overflow;
    }

    graph_entry g;
    g.n_window = n_window;
    g.ctx = ctx;
    g.indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_window);
    ggml_tensor * mat = ggml_kvarn_materialize(ctx, src_.records, src_.stage, g.indices,
            n_window, 0, 1, cfg_.bits, cfg_.value, cfg_.stage_groups);
    mat->op_params[4] = 1;  // rotated source domain (production default)
    mat->op_params[5] = (int32_t) cfg_.head_slices;
    mat->op_params[10] = 1; // read_indirect: the owner's map is the authority
    ggml_tensor * cast = ggml_cast(ctx, mat, GGML_TYPE_F32);
    g.out = ggml_reshape_4d(ctx, cast, cfg_.head_dim, cfg_.n_heads, n_window, 1);
    // Preflight backend support BEFORE any allocation or compute: an
    // unsupported node must surface as unsupported_backend, not abort.
    if (!ggml_backend_supports_op(src_.backend, mat) ||
            !ggml_backend_supports_op(src_.backend, cast) ||
            !ggml_backend_supports_op(src_.backend, g.out)) {
        ggml_free(ctx);
        return mat_status::unsupported_backend;
    }
    g.graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(g.graph, g.out);
    g.buf = ggml_backend_alloc_ctx_tensors(ctx, src_.backend);
    if (g.buf == nullptr || g.indices == nullptr || g.out == nullptr ||
            g.graph == nullptr || g.indices->buffer == nullptr) {
        if (g.buf) {
            ggml_backend_buffer_free(g.buf);
        }
        ggml_free(ctx);
        return mat_status::workspace_overflow;
    }
    graphs_.push_back(g);
    return mat_status::ok;
}

bool materializer::serve(void * user, const block_request & req) {
    (void) user;
    // The interface is status-based and never throws; allocation failures
    // inside a window are reported instead of propagating.
    try {
        last_ = validate_source();
        if (last_ != mat_status::ok) {
            return false;
        }
        if (req.out_f32 == nullptr || req.n_tokens == 0 ||
                req.n_tokens > cfg_.max_window) {
            last_ = mat_status::invalid_args;
            return false;
        }
        if (req.head_dim != cfg_.head_dim || req.n_heads != cfg_.n_heads ||
                req.value != cfg_.value) {
            last_ = mat_status::invalid_args;
            return false;
        }
        if (prefix_.empty() || uint64_t(req.t0) + req.n_tokens > n_tokens_) {
            last_ = mat_status::invalid_args;
            return false;
        }
        if (exact_ != nullptr && exact_scratch_.size() != size_t(cfg_.head_dim) * cfg_.n_heads) {
            last_ = mat_status::invalid_args;
            return false;
        }

    last_ = ensure_graph(req.n_tokens);
    if (last_ != mat_status::ok) {
        return false;
    }
    graph_entry * g = nullptr;
    for (auto & e : graphs_) {
        if (e.n_window == req.n_tokens) {
            g = &e;
            break;
        }
    }
    if (g == nullptr) {
        last_ = mat_status::workspace_overflow;
        return false;
    }

    win_indices_.resize(req.n_tokens);
    for (uint32_t c = 0; c < req.n_tokens; ++c) {
        win_indices_[c] = prefix_[size_t(req.t0) + c];
    }
    ggml_backend_tensor_set(g->indices, win_indices_.data(), 0,
            win_indices_.size() * sizeof(win_indices_[0]));

    if (ggml_backend_graph_compute(src_.backend, g->graph) != GGML_STATUS_SUCCESS) {
        last_ = mat_status::compute_failure;
        return false;
    }
    ggml_backend_tensor_get(g->out, req.out_f32, 0,
            size_t(req.n_tokens) * cfg_.head_dim * cfg_.n_heads * sizeof(float));

    // Optional exact-row overlay: original-domain rows rotated into the
    // declared source domain, consistent with the materialized body.
    if (exact_ != nullptr) {
        const uint32_t row_floats = cfg_.head_dim * cfg_.n_heads;
        for (uint32_t t = 0; t < req.n_tokens; ++t) {
            if (!exact_(exact_user_, req.t0 + t, exact_scratch_.data())) {
                continue;
            }
            float * dst = req.out_f32 + size_t(t) * row_floats;
            for (uint32_t h = 0; h < cfg_.n_heads; ++h) {
                std::memcpy(dst + size_t(h) * cfg_.head_dim,
                        exact_scratch_.data() + size_t(h) * cfg_.head_dim,
                        cfg_.head_dim * sizeof(float));
                if (!wht_head(dst + size_t(h) * cfg_.head_dim,
                        cfg_.head_dim, cfg_.record_dim)) {
                    last_ = mat_status::invalid_geometry;
                    return false;
                }
            }
        }
    }

    // Finite check before returning the window to the converter.
    const uint64_t n_values = uint64_t(req.n_tokens) * cfg_.head_dim * cfg_.n_heads;
    for (uint64_t i = 0; i < n_values; ++i) {
        if (!std::isfinite(req.out_f32[i])) {
            last_ = mat_status::non_finite_input;
            return false;
        }
    }

    last_ = mat_status::ok;
        return true;
    } catch (const std::exception &) {
        last_ = mat_status::workspace_overflow;
        return false;
    }
}

size_t materializer::graph_count() const {
    return graphs_.size();
}

bool materializer_read_block(void * user, const block_request & req) {
    materializer * m = static_cast<materializer *>(user);
    if (m == nullptr) {
        return false;
    }
    return m->serve(nullptr, req);
}

} // namespace llama_kv_qx