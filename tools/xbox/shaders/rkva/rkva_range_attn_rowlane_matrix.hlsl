// RKVA v2 Wave64 rowlane decode attention matrix for Xbox Series X (RDNA2).
// Implements dedicated Wave64 rowlane kernels across all requested formats:
// - F16
// - Q4_1
// - Q5_0 / KVarN5
// - Q5_1
// - Q6_0 / KVarN6
// - Q6_1
// - IQ4_NL
// - KVarN2 (Q2_0S)
// - KVarN3 (Q3_0)
// - KVarN8 (Q8_0)
//
// Each group owns (kv_head, query, split) and processes all RKVA_GQA query heads.
// In Wave64 decode, 64 lanes map to 64 quarter-units (covering 256 dims).
// QK Phase: Each lane evaluates the full 256-dim dot product for its row using WaveReadLaneAt.
// Online Softmax: WaveActiveMax & WaveActiveSum across the wave.
// PV Phase: Each lane accumulates its assigned 4 dimensions across valid rows of the tile.
// Wave states merge once via groupshared memory at group boundary.

#include "ggml_float.hlsli"
#include "rkva_decode_quants_group_a.hlsli"
#include "rkva_decode_quants_group_b.hlsli"
#include "rkva_decode_quants_group_c.hlsli"

#ifndef RKVA_GQA
#define RKVA_GQA 6
#endif

#ifndef RKVA_DIM
#define RKVA_DIM 256
#endif

#ifndef RKVA_THREADS
#define RKVA_THREADS 128
#endif

#ifndef RKVA_MAX_WAVES
#define RKVA_MAX_WAVES 4
#endif

#ifndef RKVA_QUARTERS
#define RKVA_QUARTERS 64
#endif

cbuffer Params : register(b0) {
    uint n_query;
    uint active_count;
    uint n_head;
    uint n_head_kv;
    uint head_dim;
    float kq_scale;
    uint origin;
    uint split_k;
    uint split_rows;
    uint direct_output;
};

ByteAddressBuffer queries : register(t0);
ByteAddressBuffer keys : register(t1);
ByteAddressBuffer values : register(t2);
ByteAddressBuffer positions : register(t3);
RWByteAddressBuffer partials : register(u0);
RWByteAddressBuffer output : register(u1);

groupshared float gs_max[RKVA_MAX_WAVES * RKVA_GQA];
groupshared float gs_den[RKVA_MAX_WAVES * RKVA_GQA];
groupshared float gs_acc[RKVA_MAX_WAVES * RKVA_GQA * RKVA_DIM];

#if ROWLANE_F16
#define ROWLANE_ENTRY CSRangeAttnF16DecodeRowlane
#define ROWLANE_ROW_HEAD 512
#define ROWLANE_LOAD load_f16
#define ROWLANE_QUARTER_DIM quarter_dim_f16
#include "rkva_decode_rowlane_template.hlsli"
#endif

#if ROWLANE_Q4_1
#define ROWLANE_ENTRY CSRangeAttnQ4_1DecodeRowlane
#define ROWLANE_ROW_HEAD RKVA_ROW_HEAD_Q4_1
#define ROWLANE_LOAD load_q4_1
#define ROWLANE_QUARTER_DIM quarter_dim_q4_1
#include "rkva_decode_rowlane_template.hlsli"
#endif

#if ROWLANE_Q5_0 || ROWLANE_KVARN5
#define ROWLANE_ENTRY CSRangeAttnQ5_0DecodeRowlane
#define ROWLANE_ROW_HEAD 176
#define ROWLANE_LOAD load_q5_0
#define ROWLANE_QUARTER_DIM quarter_dim_q5_0
#include "rkva_decode_rowlane_template.hlsli"
#endif

#if ROWLANE_Q5_1
#define ROWLANE_ENTRY CSRangeAttnQ5_1DecodeRowlane
#define ROWLANE_ROW_HEAD RKVA_ROW_HEAD_Q5_1
#define ROWLANE_LOAD load_q5_1
#define ROWLANE_QUARTER_DIM quarter_dim_q5_1
#include "rkva_decode_rowlane_template.hlsli"
#endif

#if ROWLANE_Q6_0 || ROWLANE_KVARN6
#define ROWLANE_ENTRY CSRangeAttnQ6_0DecodeRowlane
#define ROWLANE_ROW_HEAD 208
#define ROWLANE_LOAD load_q6_0
#define ROWLANE_QUARTER_DIM quarter_dim_q6_0
#include "rkva_decode_rowlane_template.hlsli"
#endif

#if ROWLANE_Q6_1
#define ROWLANE_ENTRY CSRangeAttnQ6_1DecodeRowlane
#define ROWLANE_ROW_HEAD RKVA_ROW_HEAD_Q6_1
#define ROWLANE_LOAD load_q6_1
#define ROWLANE_QUARTER_DIM quarter_dim_q6_1
#include "rkva_decode_rowlane_template.hlsli"
#endif

#if ROWLANE_IQ4_NL
#define ROWLANE_ENTRY CSRangeAttnIQ4_NLDecodeRowlane
#define ROWLANE_ROW_HEAD RKVA_ROW_HEAD_IQ4_NL
#define ROWLANE_LOAD load_iq4_nl
#define ROWLANE_QUARTER_DIM quarter_dim_iq4_nl
#include "rkva_decode_rowlane_template.hlsli"
#endif

#if ROWLANE_KVARN2 || ROWLANE_Q2_0S
#define ROWLANE_ENTRY CSRangeAttnKvarn2DecodeRowlane
#define ROWLANE_ROW_HEAD RKVA_ROW_HEAD_KVARN2
#define ROWLANE_LOAD load_kvarn2
#define ROWLANE_QUARTER_DIM quarter_dim_kvarn2
#include "rkva_decode_rowlane_template.hlsli"
#endif

#if ROWLANE_KVARN3 || ROWLANE_Q3_0
#define ROWLANE_ENTRY CSRangeAttnKvarn3DecodeRowlane
#define ROWLANE_ROW_HEAD RKVA_ROW_HEAD_KVARN3
#define ROWLANE_LOAD load_kvarn3
#define ROWLANE_QUARTER_DIM quarter_dim_kvarn3
#include "rkva_decode_rowlane_template.hlsli"
#endif

#if ROWLANE_KVARN8 || ROWLANE_Q8_0
#define ROWLANE_ENTRY CSRangeAttnKvarn8DecodeRowlane
#define ROWLANE_ROW_HEAD RKVA_ROW_HEAD_KVARN8
#define ROWLANE_LOAD load_kvarn8
#define ROWLANE_QUARTER_DIM quarter_dim_kvarn8
#include "rkva_decode_rowlane_template.hlsli"
#endif

