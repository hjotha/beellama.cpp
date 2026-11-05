#pragma once

#include "ggml.h"

// A freed recurrent slot may contain non-finite values after a failed graph.
// Multiplication by zero leaves NaNs/Infs as NaN, so overwrite the row instead.
static inline void llama_build_recurrent_zero(
        ggml_context * ctx, ggml_cgraph * graph, ggml_tensor * states,
        int32_t state_size, int32_t rs_zero) {
    GGML_ASSERT(state_size >= 0);
    // Keep the same graph topology for the full-capacity reserve and for a
    // live batch without a row to clear. The empty view is a no-op, while a
    // real row is overwritten (including NaNs) rather than multiplied by zero.
    const bool has_row = rs_zero >= 0 && state_size > 0;
    ggml_tensor * row = ggml_view_1d(ctx, states, has_row ? state_size : 0,
                                    has_row ? rs_zero*states->nb[1] : 0);
    ggml_build_forward_expand(graph, ggml_fill_inplace(ctx, row, 0.0f));
}
