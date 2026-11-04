#pragma once

#include "ggml.h"

// A freed recurrent slot may contain non-finite values after a failed graph.
// Multiplication by zero leaves NaNs/Infs as NaN, so overwrite the row instead.
static inline void llama_build_recurrent_zero(
        ggml_context * ctx, ggml_cgraph * graph, ggml_tensor * states,
        int32_t state_size, int32_t rs_zero) {
    GGML_ASSERT(state_size >= 0);
    if (rs_zero < 0 || state_size == 0) {
        return;
    }

    ggml_tensor * row = ggml_view_1d(ctx, states, state_size, rs_zero*states->nb[1]);
    ggml_build_forward_expand(graph, ggml_fill_inplace(ctx, row, 0.0f));
}
