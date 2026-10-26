// Opt-in INT8 specialization of the native Bee KVarN4 multi-query prefill tile
// (rkva_range_attn_kvarn4_mq.hlsl): the fast shared record tile scores K with
// `dot4add_i8packed` (SM 6.4) instead of a 4-dim/lane F32 dot.
//
// This variant is NOT bit-exact with `attn_kvarn4_mq2`. It approximates only the
// record-row score, by quantizing the WHT-rotated query to int8 once per
// (query, head, 128-row group) after folding the record's per-dim column scale
// into it. The K nibbles stay exactly as stored (they are already 4-bit int8
// lanes), the per-dim zero term is kept in F32, and the DC part of the Q8
// rounding error is cancelled exactly through a precomputed constant. Stage
// rows, the ORIGINAL-F16 tail, the softmax, the WHT, the split ranges and the
// partial layout are unchanged, so the host contract does not move.
//
// Quality gate: tools/xbox/tests/rkva_mq_i8_quality.py (recipe 0.06-0.08x of
// the score error the shipped 4-bit KVarN4 cache already introduces).
//
// Separate PSO target `attn_kvarn4_mq2_i8`; RKVA_MQ_I8 remains 0 for every
// existing target, whose DXIL is byte-identical. Opted in through the
// LocalState flag file `rkva-mq-i8.flag` (default off).
// The record scores use WavePrefixSum and constant-index row boundaries.
// This is the sole INT8 reduction; the runtime flag remains default off.
#define RKVA_MQ_I8 1
#include "rkva_range_attn_kvarn4_mq.hlsl"
