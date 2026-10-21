#pragma once

#include "ggml.h"

#include <limits>

// Non-RPC position-split merge: select a finite shift without changing any
// finite LSE. A masked range may publish -inf, so only the selector's inputs
// are bounded to F32's finite limits. The weights must still use the original
// LSEs: -inf then has zero weight, and NaN/+inf remain visible to consumers.
// Both-empty rows are outside this path's admitted non-empty merge contract.
inline ggml_tensor * llama_position_split_lse_max(
        ggml_context * ctx, ggml_tensor * lse1, ggml_tensor * lse2) {
    const float bound = std::numeric_limits<float>::max();
    ggml_tensor * c1 = ggml_clamp(ctx, lse1, -bound, bound);
    ggml_tensor * c2 = ggml_clamp(ctx, lse2, -bound, bound);
    // Adding the LSEs or using (a+b+abs(a-b))/2 can overflow for finite
    // inputs. Subtraction may overflow here, but STEP still selects the
    // larger input exactly, and the two products cannot overflow.
    ggml_tensor * choose1 = ggml_step(ctx, ggml_sub(ctx, c1, c2));
    ggml_tensor * choose2 = ggml_scale_bias(ctx, choose1, -1.0f, 1.0f);
    return ggml_add(ctx, ggml_mul(ctx, choose1, c1), ggml_mul(ctx, choose2, c2));
}
