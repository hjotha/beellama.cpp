// Compile-only two-local-head specialization of the native Bee KVarN4 multi-query
// prefill tile (rkva_range_attn_kvarn4_mq.hlsl).
//
// Global shape is unchanged: 24 query heads over 4 KV heads (RKVA_GQA == 6),
// the same head_dim 256, 128 threads, query/split group axes, record/stage/tail
// classification, WHT256, softmax and [split][query * n_head + head][258]
// partial layout. Only the per-group local head count changes (6 -> 2), with
// this proposed dispatch:
//
//     DispatchX = 3 * n_head_kv
//     kv_head   = group.x / 3
//     head_base = 2 * (group.x % 3)
//     global    = kv_head * 6 + head_base + g   (g = 0, 1)
//
// Each global head is still written exactly once. Three groups now reload and
// dequantize the same K/V rows per KV head; whether that trade is profitable is
// UNKNOWN until the root compiles this PSO with the target DXC and inspects the
// IR/liveness (not physical VGPR counts) and, if warranted, replays it on
// hardware. Numerical bit-exactness is NOT claimed here: the F32 expression
// order is preserved per head, but only a real compile/replay can prove it.
//
// Not wired into the compile script, engine, or activation in phase 1. The
// wrapper exists so the root can add a target later without touching the
// shipped kernel's defaults.
#define RKVA_MQ_HEADS_PER_GROUP 2
#define RKVA_MQ_QT 2
#define RKVA_MQ_I8 0
#include "rkva_range_attn_kvarn4_mq.hlsl"
