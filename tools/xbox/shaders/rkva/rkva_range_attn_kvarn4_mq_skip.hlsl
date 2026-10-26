// Opt-in specialization of the native Bee KVarN4 multi-query prefill tile
// (rkva_range_attn_kvarn4_mq.hlsl) that elides the online-softmax rescale when
// the running maximum does not grow.
//
// The elision is numerically exact, not approximate: with half_max <=
// maximum[i], next_max == maximum[i] and the original
//     old_weight = exp(maximum[i] - next_max) = exp(0) = 1.0
// makes `body[i] *= old_weight` and `denom[i] * old_weight + sum` exact
// identities. The specialization removes only those multiplies, the exp(0)
// and the per-half maximum recomputation; the row order, split ranges, K/V
// decode, exponentials, accumulation order and partial layout are unchanged.
//
// Separate PSO target `attn_kvarn4_mq2_skip`; RKVA_MQ_RESCALE_SKIP remains 0
// for every existing target, whose DXIL is byte-identical.
#define RKVA_MQ_RESCALE_SKIP 1
#include "rkva_range_attn_kvarn4_mq.hlsl"
