#pragma once
// RKVA v2 range-attention kernel selection and in-RAM profiling aggregation.
// WinRT/D3D12-free so the policy and the JSON report are host-testable.
// See docs/rkva-range-profiling.md for the schema and timing semantics.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace rkva_rprof {

// Range kernel modes. The wire/API name stays the legacy "q4_kernel_mode":
// mode 0 is the reference shader of every storage and the default. Modes 1..3
// select the optimized GQA decode kernel of the session's storage when it is
// qualified and its pipeline exists (Q4_0 since 261; F16, Q8_0 and native
// KVarN4 since the future 263 candidate), with the split policies below.
// Unsupported calls and missing pipelines keep the reference kernel.
enum : uint32_t {
    kModeReference = 0,
    kModeOptBalanced = 1,
    kModeOptFine = 2,
    kModeOptCoarse = 3,
    kModeCount = 4,
};

// Shape the optimized Q4 kernel is specialized for (Qwen35 full attention).
constexpr uint32_t kOptHeadDim = 256;
constexpr uint32_t kOptGqa = 6; // Q heads per KV head
constexpr uint32_t kOptMaxSplits = 256;
// Eight proposed MTP tokens plus the verification anchor. Each query gets
// independent workgroups; larger prefill batches retain the reference path.
constexpr uint32_t kOptMaxQueries = 9;
constexpr uint32_t kOptMinWaveLanes = 32;
constexpr uint32_t kOptMaxWaveLanes = 128;
constexpr uint32_t kReferenceMaxSplits = 32;

inline bool valid_mode(uint32_t mode) {
    return mode < kModeCount;
}

inline const char* mode_name(uint32_t mode) {
    switch (mode) {
    case kModeReference:
        return "reference";
    case kModeOptBalanced:
        return "q4_gqa_balanced";
    case kModeOptFine:
        return "q4_gqa_fine";
    case kModeOptCoarse:
        return "q4_gqa_coarse";
    default:
        return "invalid";
    }
}

enum Storage : uint32_t {
    kStorageF16 = 0,
    kStorageQ4 = 1,
    kStorageQ8 = 2,
    kStorageKvarn4 = 3,
    kStorageCount = 4,
};

// Bucket/report kernel name. Q4_0 keeps the 261 names byte for byte.
inline const char* kernel_name(uint32_t storage, uint32_t mode) {
    if (mode == kModeReference || !valid_mode(mode)) {
        return mode_name(mode);
    }
    static const char* names[kStorageCount][3] = {
        {"f16_gqa_balanced", "f16_gqa_fine", "f16_gqa_coarse"},
        {"q4_gqa_balanced", "q4_gqa_fine", "q4_gqa_coarse"},
        {"q8_gqa_balanced", "q8_gqa_fine", "q8_gqa_coarse"},
        {"kvarn4_gqa_balanced", "kvarn4_gqa_fine", "kvarn4_gqa_coarse"},
    };
    return storage < kStorageCount ? names[storage][mode - 1] : "invalid";
}

struct SplitPolicy {
    uint32_t target_rows;
    uint32_t max_splits;
};

inline SplitPolicy split_policy(uint32_t mode) {
    switch (mode) {
    case kModeOptFine:
        return {32, 256};
    case kModeOptCoarse:
        return {512, 64};
    default:
        return {128, 128};
    }
}

// Reference live259 policy: one split per 256 live rows, at most 32.
inline uint32_t reference_split_k(uint32_t active_count) {
    return std::max(1u, std::min(kReferenceMaxSplits, (active_count + 255) / 256));
}

inline uint32_t optimized_split_k(uint32_t active_count, uint32_t mode) {
    const SplitPolicy p = split_policy(mode);
    const uint32_t wanted = (active_count + p.target_rows - 1) / p.target_rows;
    return std::max(1u, std::min(std::min(p.max_splits, kOptMaxSplits), wanted));
}

inline uint32_t split_rows(uint32_t active_count, uint32_t split_k) {
    return split_k ? (active_count + split_k - 1) / split_k : 0;
}

// Q4_0 multi-query tile kernel (rkva_range_attn_q4_mq.hlsl) for MTP verification
// batches: one group shares each K/V row dequantization across kQ4MqTile...
// queries x 6 GQA heads. Only optimized Q4_0 calls with nq 2..kOptMaxQueries use
// it; nq1 keeps the per-query kernel and its split policy. Legacy mode mapping:
//   mode 1 (balanced) -> 4-query tile, balanced split rounded to 64-row tiles
//   mode 2 (fine)     -> 2-query tile, fine split rounded to 64-row tiles
//   mode 3 (coarse)   -> per-query kernel (unchanged 264 behavior, A/B baseline)
constexpr uint32_t kQ4MqRows = 64; // rows per tile = threads per group
constexpr uint32_t kQ4MqTileMax = 4;

inline uint32_t q4_mq_tile(uint32_t mode, uint32_t n_tokens) {
    if (n_tokens < 2 || n_tokens > kOptMaxQueries) {
        return 0;
    }
    return mode == kModeOptBalanced ? 4u : mode == kModeOptFine ? 2u : 0u;
}

inline const char* q4_mq_kernel_name(uint32_t q_tile) {
    return q_tile == 4 ? "q4_mq4_balanced" : q_tile == 2 ? "q4_mq2_fine" : "invalid";
}

struct Q4MqSplit {
    uint32_t split_k;
    uint32_t split_rows; // multiple of kQ4MqRows (0 when active_count == 0)
};

// Mode split policy with split rows rounded up to whole 64-row tiles, then the
// split count recomputed so every split is non-empty (never above the mode's).
inline Q4MqSplit q4_mq_split(uint32_t active_count, uint32_t mode) {
    const uint32_t k0 = optimized_split_k(active_count, mode);
    uint32_t rows = split_rows(active_count, k0);
    rows = (rows + kQ4MqRows - 1) / kQ4MqRows * kQ4MqRows;
    const uint32_t k = rows ? (active_count + rows - 1) / rows : 1;
    return {std::max(1u, k), rows};
}

// Per-call gate. Every unsupported call keeps the reference kernel.
inline bool optimized_q4_supported(uint32_t mode, uint32_t storage, uint32_t n_tokens,
                                   uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim) {
    return mode != kModeReference && valid_mode(mode) && storage == 1 && n_tokens >= 1 &&
           n_tokens <= kOptMaxQueries && head_dim == kOptHeadDim && n_head_kv > 0 &&
           n_head == kOptGqa * n_head_kv;
}

inline bool optimized_shape_supported(uint32_t mode, uint32_t n_tokens, uint32_t n_head,
                                      uint32_t n_head_kv, uint32_t head_dim) {
    return mode != kModeReference && valid_mode(mode) && n_tokens >= 1 &&
           n_tokens <= kOptMaxQueries && head_dim == kOptHeadDim && n_head_kv > 0 &&
           n_head == kOptGqa * n_head_kv;
}

// F16 and Q8_0 share one GQA kernel source (RKVA_OPT_STORAGE specialization)
// and the Q4 split policy.
inline bool optimized_format_supported(uint32_t mode, uint32_t storage, uint32_t n_tokens,
                                       uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim) {
    return (storage == kStorageF16 || storage == kStorageQ8) &&
           optimized_shape_supported(mode, n_tokens, n_head, n_head_kv, head_dim);
}

// F16 / Q8_0 multi-query tile kernel (rkva_range_attn_fmt_mq.hlsl, 267): the
// Q4_0 multi-query algorithm with the storage's own K/V decode. Same legacy
// mode mapping and split as Q4_0 (mode 1 -> 4-query tile, mode 2 -> 2-query
// tile, mode 3 -> per-query optimized kernel as the in-build baseline), only for
// optimized F16/Q8_0 calls with nq 2..kOptMaxQueries; nq1 is unchanged.
inline uint32_t format_mq_tile(uint32_t mode, uint32_t storage, uint32_t n_tokens) {
    return storage == kStorageF16 || storage == kStorageQ8 ? q4_mq_tile(mode, n_tokens) : 0u;
}

inline const char* format_mq_kernel_name(uint32_t storage, uint32_t q_tile) {
    if (q_tile != 2 && q_tile != 4) {
        return "invalid";
    }
    if (storage == kStorageF16) {
        return q_tile == 4 ? "f16_mq4_balanced" : "f16_mq2_fine";
    }
    if (storage == kStorageQ8) {
        return q_tile == 4 ? "q8_mq4_balanced" : "q8_mq2_fine";
    }
    return "invalid";
}

// Native KVarN4 descriptor revision-3 (serial prefix) multi-query tile kernel
// (rkva_range_attn_kvarn4_mq.hlsl): 2 or 4 queries x 6 GQA heads share the K/V
// record/stage/tail loads and decode of rows every query reads with the same
// representation; each query keeps its own serial-prefix windows. Mode 2 (fine
// split, the qualified default) selects the 2-query tile (kvarn4_mq2_fine);
// mode 1 (balanced split) opts into the 4-query tile (kvarn4_mq4_balanced) for
// nq 2..kOptMaxQueries when its PSO is present (falling back to per-query
// kvarn4_gqa_balanced otherwise); mode 3 (coarse), nq1, mode 0 and descriptor
// revisions 1/2 are unchanged.
constexpr uint32_t kKvarn4MqTile = 2;
constexpr uint32_t kKvarn4MqTileMax = 4;
constexpr uint32_t kKvarn4MqWaveLanes = 64; // one 4-dim unit per lane

// Zero exact tails retain the qualified per-query arithmetic. The shared tiles
// require a 128-token original-F16 tail; raw FP64 parity alone does not
// establish the same speculative cache trajectory without that tail.
inline uint32_t kvarn4_mq_tile(uint32_t mode, uint32_t n_tokens, uint32_t precision_tail_tokens) {
    if (precision_tail_tokens != 128 || n_tokens < 2 || n_tokens > kOptMaxQueries) {
        return 0u;
    }
    return mode == kModeOptBalanced ? kKvarn4MqTileMax : mode == kModeOptFine ? kKvarn4MqTile : 0u;
}

inline const char* kvarn4_mq_kernel_name(uint32_t q_tile) {
    return q_tile == kKvarn4MqTileMax ? "kvarn4_mq4_balanced"
           : q_tile == kKvarn4MqTile  ? "kvarn4_mq2_fine"
                                      : "invalid";
}

// Storage-qualified multi-query kernel name (Q4_0 keeps its 266 names).
inline const char* mq_kernel_name(uint32_t storage, uint32_t q_tile) {
    if (storage == kStorageKvarn4) {
        return kvarn4_mq_kernel_name(q_tile);
    }
    return storage == kStorageQ4 ? q4_mq_kernel_name(q_tile)
                                 : format_mq_kernel_name(storage, q_tile);
}

// Native KVarN4 descriptor v1 shapes the optimized kernel reads directly:
// 128-aligned origin and a 0 or 128 token precision tail.
constexpr uint32_t kKvarnGroup = 128;
constexpr uint32_t kKvarnTile = 8;

inline bool optimized_kvarn4_supported(uint32_t mode, uint32_t storage, uint32_t n_tokens,
                                       uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim,
                                       uint32_t origin, uint32_t precision_tail_tokens) {
    return storage == kStorageKvarn4 && origin % kKvarnGroup == 0 &&
           (precision_tail_tokens == 0 || precision_tail_tokens == kKvarnGroup) &&
           optimized_shape_supported(mode, n_tokens, n_head, n_head_kv, head_dim);
}

struct KvarnSplit {
    uint32_t split_k;
    uint32_t split_rows; // multiple of kKvarnTile
};

// Mode split policy with split rows rounded up to whole 8-token record tiles,
// then the split count recomputed so every split is non-empty.
inline KvarnSplit kvarn4_split(uint32_t active_count, uint32_t mode) {
    const uint32_t k0 = optimized_split_k(active_count, mode);
    uint32_t rows = split_rows(active_count, k0);
    rows = (rows + kKvarnTile - 1) / kKvarnTile * kKvarnTile;
    const uint32_t k = rows ? (active_count + rows - 1) / rows : 1;
    return {std::max(1u, k), rows};
}

// Native KVarN4 wave64 decode kernel specialization (nq == 1):
// Requires wave lanes == 64 (wave64 RDNA2 APU), opt modes 1/2.
// Halves LDS to 12 KiB, ping-pongs WHT256 (9 barriers), eliminates wave32
// second-unit registers, and hoists group metadata across 128-token record groups.
constexpr uint32_t kKvarn4DecodeWaveLanes = 64;

inline bool kvarn4_decode_opt(uint32_t mode, uint32_t wave_lane_min) {
    return (mode == kModeOptBalanced || mode == kModeOptFine) &&
           wave_lane_min >= kKvarn4DecodeWaveLanes;
}

inline const char* kvarn4_decode_kernel_name(uint32_t mode) {
    switch (mode) {
    case kModeOptBalanced:
        return "kvarn4_decode_balanced";
    case kModeOptFine:
        return "kvarn4_decode_fine";
    case kModeOptCoarse:
        return "kvarn4_gqa_coarse";
    default:
        return "invalid";
    }
}

// Decode split policy for single-query native KVarN4 (nq == 1):
// Round split rows up to whole 128-token record groups (kKvarnGroup = 128)
// so splits never straddle record boundaries, and bound max_splits so the
// reducer and WHT workgroup overhead are proportional to console CU count.
// Mode 1 (balanced): target_rows = 512 (4 groups), max_splits = 32.
// Mode 2 (fine):     target_rows = 256 (2 groups), max_splits = 64.
// Mode 3 (coarse):   keeps the legacy kvarn4_split baseline.
inline KvarnSplit kvarn4_decode_split(uint32_t active_count, uint32_t mode) {
    if (active_count == 0) {
        return {1, 0};
    }
    if (mode == kModeOptCoarse) {
        return kvarn4_split(active_count, mode);
    }
    const uint32_t target_rows = mode == kModeOptFine ? 256u : 512u;
    const uint32_t max_splits = mode == kModeOptFine ? 64u : 32u;
    const uint32_t wanted = (active_count + target_rows - 1) / target_rows;
    const uint32_t k0 = std::max(1u, std::min(max_splits, wanted));
    uint32_t rows = split_rows(active_count, k0);
    rows = (rows + kKvarnGroup - 1) / kKvarnGroup * kKvarnGroup;
    const uint32_t k = rows ? (active_count + rows - 1) / rows : 1;
    return {std::max(1u, k), rows};
}

// Multi-query tile split policy for native KVarN4 (nq 2..9, q_tile 2 or 4):
// For q_tile == 2 or 4 in mode 1 (balanced) and mode 2 (fine):
//   target_rows = 256 (2 record groups of 128 tokens),
//   max_splits = 64 (mode fine) or 32 (mode balanced).
// Mode 3 (coarse) or invalid q_tile keeps kvarn4_split.
// rows is rounded up to whole kKvarnTile (8), and active_count == 0 returns {1, 0}.
inline KvarnSplit kvarn4_mq_split(uint32_t active_count, uint32_t mode, uint32_t q_tile) {
    if (active_count == 0) {
        return {1, 0};
    }
    if ((q_tile != 2 && q_tile != 4) || (mode != kModeOptBalanced && mode != kModeOptFine)) {
        return kvarn4_split(active_count, mode);
    }
    const uint32_t target_rows = 256u;
    const uint32_t max_splits = (mode == kModeOptFine) ? 64u : 32u;
    const uint32_t wanted = (active_count + target_rows - 1) / target_rows;
    const uint32_t k0 = std::max(1u, std::min(max_splits, wanted));
    uint32_t rows = split_rows(active_count, k0);
    rows = (rows + kKvarnTile - 1) / kKvarnTile * kKvarnTile;
    const uint32_t k = rows ? (active_count + rows - 1) / rows : 1;
    return {std::max(1u, k), rows};
}

enum Kind : uint8_t { kUpload = 0, kAttention = 1, kKindCount = 2 };

// Disjoint CPU spans; their sum is the engine wall time of the call (profiling
// bookkeeping after the work is reported separately as overhead).
enum CpuSpan : uint8_t {
    kCpuValidate = 0, // argument/session checks
    kCpuAlloc,        // grow-only scratch/upload buffer creation (cold calls)
    kCpuMapCopy,      // Map + memcpy inputs + Unmap
    kCpuRecord,       // allocator/list reset + command recording
    kCpuSubmit,       // Close + ExecuteCommandLists + Signal
    kCpuFenceWait,    // fence completion wait (GPU execution + queue latency)
    kCpuReadback,     // Map + memcpy outputs + Unmap (attention only)
    kCpuSpanCount,
};

// Upload uses stage 0 (copy). Attention: pass1, reduce, readback copy.
enum GpuStage : uint8_t { kGpuStage0 = 0, kGpuStage1, kGpuStage2, kGpuStageCount };

struct Sample {
    uint8_t kind = kUpload;
    uint8_t storage = 0;
    uint8_t optimized = 0;
    uint8_t mode = 0;
    uint8_t gpu_valid = 0;
    uint8_t direct_output = 0;
    uint8_t fence_blocked = 0;
    uint8_t alloc_grew = 0;
    uint8_t serial_prefix = 0; // native KVarN4 descriptor revision 3
    uint8_t q_tile = 0;
    uint8_t decode_opt = 0;   // multi-query tile (Q4_0/F16/Q8_0/KVarN4 2 or 4); 0 = per-query
    uint8_t bulk_segment = 0; // 1 if recorded as an individual segment of native bulk op 22
    uint32_t n_tokens = 0;
    uint32_t rows = 0; // upload rows, or attention active_count
    uint32_t split_k = 0;
    uint32_t groups = 0; // pass-1 threadgroups
    uint64_t bytes = 0;  // upload payload, or attention Q+positions in / OUT+LSE out
    uint64_t gpu_ticks[kGpuStageCount] = {};
    uint64_t cpu_ns[kCpuSpanCount] = {};
    uint64_t overhead_ns = 0;
};

struct Counters {
    uint64_t calls[kKindCount] = {};
    uint64_t optimized_calls = 0; // optimized Q4_0 attention calls (261 meaning)
    uint64_t reference_q4_calls = 0;
    uint64_t q4_mq_calls[2] = {}; // subset of optimized_calls: [0] 2-query, [1] 4-query tile
    // F16/Q8_0 multi-query tile calls (subset of optimized_by_storage), [storage][tile/4].
    uint64_t fmt_mq_calls[kStorageCount][2] = {};
    // Native KVarN4 revision-3 2-query and 4-query tile calls (subset of
    // optimized_by_storage[kvarn4]).
    uint64_t kvarn4_mq_calls = 0;
    uint64_t kvarn4_decode_calls = 0;
    uint64_t kvarn4_mq4_calls = 0;
    uint64_t optimized_by_storage[kStorageCount] = {};
    uint64_t reference_by_storage[kStorageCount] = {};
    uint64_t failed_calls = 0;
    uint64_t submits = 0;
    uint64_t fence_blocking_waits = 0;
    uint64_t fence_already_complete = 0;
    uint64_t dispatches = 0;
    uint64_t copy_regions = 0;
    uint64_t buffer_creates = 0;
    uint64_t buffer_create_bytes = 0;
    uint64_t timestamp_resolves = 0;
    uint64_t timestamp_map_failures = 0;
    uint64_t bulk_calls = 0;
    uint64_t bulk_segments = 0;
    uint64_t bulk_queries = 0;
    uint64_t fp8_gpu_calls = 0, fp8_cpu_fallback_calls = 0;
};

struct BulkEnvelopeSample {
    uint64_t cycle_id = 0;
    uint32_t session_id = 0, layer = 0, wire = 0;
    uint32_t segment_count = 0;
    uint32_t total_queries = 0;
    uint8_t gpu_valid = 0;
    uint8_t fp8_gpu = 0;
    uint8_t fence_blocked = 0;
    uint8_t alloc_grew = 0;
    uint64_t upload_bytes = 0;
    uint64_t query_bytes = 0;
    uint64_t output_bytes = 0;
    uint64_t cpu_ns[kCpuSpanCount] = {};
    uint64_t gpu_upload_ticks = 0;
    uint64_t gpu_unpack_ticks = 0;
    uint64_t gpu_pack_ticks = 0;
    uint64_t gpu_pass1_ticks = 0;
    uint64_t gpu_reduce_ticks = 0;
    uint64_t gpu_copy_ticks = 0;
    uint64_t gpu_total_ticks = 0;
    uint64_t overhead_ns = 0;
};

struct MemoryInfo {
    uint64_t timestamp_unix_ms = 0;
    bool app_available = false;
    uint64_t app_usage_bytes = 0;
    uint64_t app_limit_bytes = 0;
    bool gpu_local_available = false;
    uint64_t gpu_local_usage_bytes = 0;
    uint64_t gpu_local_budget_bytes = 0;
};

inline std::string memory_json(const MemoryInfo& m) {
    char text[512];
    std::snprintf(text, sizeof(text),
                  "{\"timestamp_unix_ms\":%llu,\"app_available\":%s,"
                  "\"app_usage_bytes\":%llu,\"app_limit_bytes\":%llu,"
                  "\"gpu_local_available\":%s,\"gpu_local_usage_bytes\":%llu,"
                  "\"gpu_local_budget_bytes\":%llu}",
                  static_cast<unsigned long long>(m.timestamp_unix_ms),
                  m.app_available ? "true" : "false",
                  static_cast<unsigned long long>(m.app_usage_bytes),
                  static_cast<unsigned long long>(m.app_limit_bytes),
                  m.gpu_local_available ? "true" : "false",
                  static_cast<unsigned long long>(m.gpu_local_usage_bytes),
                  static_cast<unsigned long long>(m.gpu_local_budget_bytes));
    return text;
}

struct Header {
    MemoryInfo memory;
    bool profiling = false;
    uint32_t q4_kernel_mode = 0;
    bool gpu_timestamps = false;
    uint64_t gpu_timestamp_frequency = 0;
    std::string gpu_timestamp_status;
    bool q4_optimized_available = false;
    std::string q4_optimized_status;
    // Per storage (index = Storage); the Q4 entry mirrors the fields above.
    bool optimized_available[kStorageCount] = {};
    std::string optimized_status[kStorageCount];
    uint32_t wave_lane_min = 0;
    uint32_t wave_lane_max = 0;
    // Native KVarN4 revision-3 (serial prefix) pipelines: [0] reference, [1] optimized.
    bool kvarn4_prefix_available[2] = {};
    std::string kvarn4_prefix_status[2];
    // Q4_0 multi-query tile pipelines: [0] 2-query, [1] 4-query tile.
    bool q4_mq_available[2] = {};
    std::string q4_mq_status[2];
    // F16 / Q8_0 multi-query tile pipelines, [storage][0] 2-query, [storage][1] 4-query.
    bool fmt_mq_available[kStorageCount][2] = {};
    std::string fmt_mq_status[kStorageCount][2];
    // Native KVarN4 revision-3 2-query and 4-query tile pipelines (require the optimized prefix
    // PSO).
    bool kvarn4_mq_available = false;
    std::string kvarn4_mq_status;
    bool kvarn4_decode_available = false;
    std::string kvarn4_decode_status;
    bool kvarn4_mq4_available = false;
    std::string kvarn4_mq4_status;
};

inline const char* storage_name(uint32_t storage) {
    switch (storage) {
    case 0:
        return "f16";
    case 1:
        return "q4_0";
    case 2:
        return "q8_0";
    case 3:
        return "kvarn4";
    default:
        return "unknown";
    }
}

// Coarse bands keep decode (nq1, warm) apart from prefill and cold calls.
inline const char* nq_band(uint32_t n_tokens) {
    return n_tokens <= 1     ? "nq1"
           : n_tokens <= 16  ? "nq2-16"
           : n_tokens <= 128 ? "nq17-128"
                             : "nq129+";
}

inline const char* rows_band(uint32_t rows) {
    return rows == 0       ? "r0"
           : rows <= 256   ? "r1-256"
           : rows <= 1024  ? "r257-1k"
           : rows <= 4096  ? "r1k-4k"
           : rows <= 16384 ? "r4k-16k"
           : rows <= 32768 ? "r16k-32k"
                           : "r32k+";
}

// kind/storage/kernel[-mode]/nq band/rows band/warm|cold. rows is the
// attention active_count or the upload row count; cold means the call grew a
// scratch or upload buffer.
inline std::string bucket_key(const Sample& s) {
    std::string key = s.kind == kUpload ? "upload/" : "attention/";
    key += storage_name(s.storage);
    if (s.serial_prefix) {
        key += "-prefix";
    }
    if (s.kind == kAttention) {
        key += s.optimized ? "/optimized-" : "/reference";
        if (s.optimized) {
            key += s.q_tile ? mq_kernel_name(s.storage, s.q_tile)
                            : (s.storage == kStorageKvarn4 && s.decode_opt
                                   ? kvarn4_decode_kernel_name(s.mode)
                                   : kernel_name(s.storage, s.mode));
        }
    }
    key += '/';
    key += nq_band(s.n_tokens);
    key += '/';
    key += rows_band(s.rows);
    key += s.alloc_grew ? "/cold" : "/warm";
    return key;
}

inline std::string json_escape(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (char c : in) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
            out += buf;
        } else {
            out += c;
        }
    }
    return out;
}

// Fixed-capacity sample ring. Storage is reserved by configure(), so record()
// never allocates during a timed RPC.
class Recorder {
  public:
    static constexpr size_t kDefaultCapacity = 16384;
    static constexpr size_t kDefaultBulkCapacity = 2048;

    void configure(size_t capacity = kDefaultCapacity) {
        if (ring_.size() != capacity) {
            ring_.assign(capacity, Sample{});
            next_ = 0;
            stored_ = 0;
        }
        const size_t bulk_cap = std::min(capacity, kDefaultBulkCapacity);
        if (bulk_ring_.size() != bulk_cap) {
            bulk_ring_.assign(bulk_cap, BulkEnvelopeSample{});
            bulk_next_ = 0;
            bulk_stored_ = 0;
        }
        // Keep phase distributions even after a long decode suffix. Allocate
        // only on the control path; record_bulk never grows these buffers.
        const size_t phase_cap = std::min<size_t>(capacity, 1024);
        for (size_t phase = 0; phase < bulk_phase_ring_.size(); ++phase) {
            if (bulk_phase_ring_[phase].size() != phase_cap) {
                bulk_phase_ring_[phase].assign(phase_cap, BulkEnvelopeSample{});
                bulk_phase_next_[phase] = bulk_phase_stored_[phase] = 0;
            }
        }
    }
    void reset() {
        next_ = 0;
        stored_ = 0;
        recorded_ = 0;
        bulk_next_ = 0;
        bulk_stored_ = 0;
        bulk_recorded_ = 0;
        bulk_phase_next_ = {};
        bulk_phase_stored_ = {};
        counters = Counters{};
    }
    void record(const Sample& sample) {
        ++recorded_;
        if (ring_.empty()) {
            return;
        }
        ring_[next_] = sample;
        next_ = (next_ + 1) % ring_.size();
        stored_ = std::min(stored_ + 1, ring_.size());
    }
    void record_bulk(const BulkEnvelopeSample& sample) {
        ++bulk_recorded_;
        if (bulk_ring_.empty()) {
            return;
        }
        bulk_ring_[bulk_next_] = sample;
        bulk_next_ = (bulk_next_ + 1) % bulk_ring_.size();
        bulk_stored_ = std::min(bulk_stored_ + 1, bulk_ring_.size());
        const size_t phase = sample.total_queries == 1 ? 0 : sample.total_queries <= 9 ? 1 : 2;
        auto& ring = bulk_phase_ring_[phase];
        if (!ring.empty()) {
            ring[bulk_phase_next_[phase]] = sample;
            bulk_phase_next_[phase] = (bulk_phase_next_[phase] + 1) % ring.size();
            bulk_phase_stored_[phase] = std::min(bulk_phase_stored_[phase] + 1, ring.size());
        }
    }
    size_t capacity() const {
        return ring_.size();
    }
    size_t stored() const {
        return stored_;
    }
    uint64_t recorded() const {
        return recorded_;
    }
    size_t bulk_capacity() const {
        return bulk_ring_.size();
    }
    size_t bulk_stored() const {
        return bulk_stored_;
    }
    uint64_t bulk_recorded() const {
        return bulk_recorded_;
    }

    std::string json(const Header& h) const {
        std::string o;
        o.reserve(8192);
        char buf[512];
        o += "{\"schema\":\"rkva-range-profile-v1\"";
        std::snprintf(buf, sizeof(buf),
                      ",\"profiling\":%s,\"q4_kernel_mode\":%u,\"q4_kernel_mode_name\":\"%s\"",
                      h.profiling ? "true" : "false", h.q4_kernel_mode,
                      mode_name(h.q4_kernel_mode));
        o += buf;
        std::snprintf(buf, sizeof(buf),
                      ",\"gpu_timestamps\":%s,\"gpu_timestamp_frequency_hz\":%llu"
                      ",\"gpu_timestamp_status\":\"%s\"",
                      h.gpu_timestamps ? "true" : "false",
                      static_cast<unsigned long long>(h.gpu_timestamp_frequency),
                      json_escape(h.gpu_timestamp_status).c_str());
        o += buf;
        std::snprintf(buf, sizeof(buf),
                      ",\"q4_optimized_available\":%s,\"q4_optimized_status\":\"%s\""
                      ",\"wave_lane_min\":%u,\"wave_lane_max\":%u",
                      h.q4_optimized_available ? "true" : "false",
                      json_escape(h.q4_optimized_status).c_str(), h.wave_lane_min, h.wave_lane_max);
        o += buf;
        o += ",\"cpu_clock\":\"std::chrono::steady_clock\"";
        std::snprintf(buf, sizeof(buf),
                      ",\"sample_capacity\":%llu,\"samples_recorded\":%llu,"
                      "\"samples_in_ring\":%llu,\"samples_overwritten\":%llu",
                      static_cast<unsigned long long>(ring_.size()),
                      static_cast<unsigned long long>(recorded_),
                      static_cast<unsigned long long>(stored_),
                      static_cast<unsigned long long>(recorded_ - stored_));
        o += buf;
        const Counters& c = counters;
        std::snprintf(
            buf, sizeof(buf),
            ",\"counters\":{\"upload_calls\":%llu,\"attention_calls\":%llu,"
            "\"optimized_q4_calls\":%llu,\"reference_q4_calls\":%llu,\"failed_calls\":%llu,"
            "\"submits\":%llu,\"fence_blocking_waits\":%llu,\"fence_already_complete\":%llu,"
            "\"dispatches\":%llu,\"copy_regions\":%llu,",
            ull(c.calls[kUpload]), ull(c.calls[kAttention]), ull(c.optimized_calls),
            ull(c.reference_q4_calls), ull(c.failed_calls), ull(c.submits),
            ull(c.fence_blocking_waits), ull(c.fence_already_complete), ull(c.dispatches),
            ull(c.copy_regions));
        o += buf;
        std::snprintf(buf, sizeof(buf),
                      "\"buffer_creates\":%llu,\"buffer_create_bytes\":%llu,"
                      "\"timestamp_resolves\":%llu,\"timestamp_map_failures\":%llu,"
                      "\"bulk_calls\":%llu,\"bulk_segments\":%llu,\"bulk_queries\":%llu}",
                      ull(c.buffer_creates), ull(c.buffer_create_bytes), ull(c.timestamp_resolves),
                      ull(c.timestamp_map_failures), ull(c.bulk_calls), ull(c.bulk_segments),
                      ull(c.bulk_queries));
        o += buf;
        o += ",\"memory\":" + memory_json(h.memory);
        o += ",\"kvarn4_mq_qualified_precision_tail_tokens\":[128]";
        o += ",\"kernel_mode_scope\":\"legacy name q4_kernel_mode; modes 1..3 select the "
             "optimized kernel of each storage listed available in optimized_kernels, "
             "otherwise reference; Q4_0 nq 2..9 in modes 1/2 use the q4_multi_query tile "
             "kernel when available; F16/Q8_0 nq 2..9 in modes 1/2 use the "
             "format_multi_query tile kernel when available; native KVarN4 descriptor "
             "revision 3 nq 2..9 in modes 1/2 uses the kvarn4_multi_query tile kernel when "
             "available\"";
        o += ",\"optimized_kernels\":{";
        for (uint32_t st = 0; st < kStorageCount; ++st) {
            std::snprintf(
                buf, sizeof(buf),
                "%s\"%s\":{\"available\":%s,\"status\":\"%s\","
                "\"optimized_calls\":%llu,\"reference_calls\":%llu,\"fast_nq_max\":%u}",
                st ? "," : "", storage_name(st), h.optimized_available[st] ? "true" : "false",
                json_escape(h.optimized_status[st]).c_str(), ull(c.optimized_by_storage[st]),
                ull(c.reference_by_storage[st]), h.optimized_available[st] ? kOptMaxQueries : 0u);
            o += buf;
        }
        o += '}';
        std::snprintf(
            buf, sizeof(buf),
            ",\"kvarn4_serial_prefix\":{\"reference\":{\"available\":%s,\"status\":\"%s\"},"
            "\"optimized\":{\"available\":%s,\"status\":\"%s\"}}",
            h.kvarn4_prefix_available[0] ? "true" : "false",
            json_escape(h.kvarn4_prefix_status[0]).c_str(),
            h.kvarn4_prefix_available[1] ? "true" : "false",
            json_escape(h.kvarn4_prefix_status[1]).c_str());
        o += buf;
        std::snprintf(buf, sizeof(buf),
                      ",\"q4_multi_query\":{\"nq_min\":2,\"nq_max\":%u,\"tile_rows\":%u,"
                      "\"mode_map\":\"1:q4_mq4_balanced 2:q4_mq2_fine 3:per-query q4_gqa_coarse\","
                      "\"mq2\":{\"available\":%s,\"status\":\"%s\",\"calls\":%llu},",
                      kOptMaxQueries, kQ4MqRows, h.q4_mq_available[0] ? "true" : "false",
                      json_escape(h.q4_mq_status[0]).c_str(), ull(c.q4_mq_calls[0]));
        o += buf;
        std::snprintf(buf, sizeof(buf),
                      "\"mq4\":{\"available\":%s,\"status\":\"%s\",\"calls\":%llu}}",
                      h.q4_mq_available[1] ? "true" : "false",
                      json_escape(h.q4_mq_status[1]).c_str(), ull(c.q4_mq_calls[1]));
        o += buf;
        std::snprintf(buf, sizeof(buf),
                      ",\"format_multi_query\":{\"nq_min\":2,\"nq_max\":%u,\"tile_rows\":%u,"
                      "\"mode_map\":\"1:<fmt>_mq4_balanced 2:<fmt>_mq2_fine "
                      "3:per-query <fmt>_gqa_coarse\"",
                      kOptMaxQueries, kQ4MqRows);
        o += buf;
        for (uint32_t st :
             {static_cast<uint32_t>(kStorageF16), static_cast<uint32_t>(kStorageQ8)}) {
            std::snprintf(buf, sizeof(buf),
                          ",\"%s\":{\"mq2\":{\"available\":%s,\"status\":\"%s\",\"calls\":%llu},",
                          storage_name(st), h.fmt_mq_available[st][0] ? "true" : "false",
                          json_escape(h.fmt_mq_status[st][0]).c_str(), ull(c.fmt_mq_calls[st][0]));
            o += buf;
            std::snprintf(buf, sizeof(buf),
                          "\"mq4\":{\"available\":%s,\"status\":\"%s\",\"calls\":%llu}}",
                          h.fmt_mq_available[st][1] ? "true" : "false",
                          json_escape(h.fmt_mq_status[st][1]).c_str(), ull(c.fmt_mq_calls[st][1]));
            o += buf;
        }
        o += '}';
        std::snprintf(buf, sizeof(buf),
                      ",\"kvarn4_multi_query\":{\"nq_min\":2,\"nq_max\":%u,"
                      "\"descriptor_revision\":3,\"required_wave_lanes\":%u,"
                      "\"mode_map\":\"1:kvarn4_mq4_balanced 2:kvarn4_mq2_fine "
                      "3:per-query kvarn4_gqa_coarse\","
                      "\"mq2\":{\"available\":%s,\"status\":\"%s\",\"calls\":%llu},",
                      kOptMaxQueries, kKvarn4MqWaveLanes, h.kvarn4_mq_available ? "true" : "false",
                      json_escape(h.kvarn4_mq_status).c_str(), ull(c.kvarn4_mq_calls));
        o += buf;
        std::snprintf(buf, sizeof(buf),
                      "\"mq4\":{\"available\":%s,\"status\":\"%s\",\"calls\":%llu}}",
                      h.kvarn4_mq4_available ? "true" : "false",
                      json_escape(h.kvarn4_mq4_status).c_str(), ull(c.kvarn4_mq4_calls));
        o += buf;
        std::snprintf(buf, sizeof(buf),
                      ",\"kvarn4_decode\":{\"nq\":1,"
                      "\"descriptor_revision\":3,\"required_wave_lanes\":%u,"
                      "\"mode_map\":\"1:kvarn4_decode_balanced 2:kvarn4_decode_fine "
                      "3:per-query kvarn4_gqa_coarse\","
                      "\"available\":%s,\"status\":\"%s\",\"calls\":%llu}",
                      kKvarn4DecodeWaveLanes, h.kvarn4_decode_available ? "true" : "false",
                      json_escape(h.kvarn4_decode_status).c_str(), ull(c.kvarn4_decode_calls));
        o += buf;
        o += ",\"bulk\":";
        append_bulk_json(o, h);

        std::map<std::string, std::vector<const Sample*>> buckets;
        for (size_t i = 0; i < stored_; ++i) {
            const size_t index = (next_ + ring_.size() - stored_ + i) % ring_.size();
            const Sample& s = ring_[index];
            buckets[bucket_key(s)].push_back(&s);
        }
        o += ",\"buckets\":{";
        bool first = true;
        for (const auto& bucket : buckets) {
            if (!first) {
                o += ',';
            }
            first = false;
            append_bucket(o, bucket.first, bucket.second, h);
        }
        o += "}}";
        return o;
    }

    Counters counters;

  private:
    static unsigned long long ull(uint64_t v) {
        return static_cast<unsigned long long>(v);
    }

  public:
    // Control/report path only. Values are sorted in place; nearest-rank percentiles.
    static void append_stats(std::string& o, const char* name, std::vector<double>& values) {
        char buf[320];
        if (values.empty()) {
            std::snprintf(buf, sizeof(buf), "\"%s\":{\"n\":0}", name);
            o += buf;
            return;
        }
        std::sort(values.begin(), values.end());
        double sum = 0;
        for (double v : values) {
            sum += v;
        }
        auto rank = [&](double p) {
            size_t r = static_cast<size_t>(p * static_cast<double>(values.size()) + 0.999999);
            r = std::max<size_t>(1, std::min(values.size(), r));
            return values[r - 1];
        };
        std::snprintf(buf, sizeof(buf),
                      "\"%s\":{\"n\":%llu,\"mean\":%.3f,\"min\":%.3f,\"p50\":%.3f,\"p90\":%.3f,"
                      "\"p95\":%.3f,\"p99\":%.3f,\"max\":%.3f,\"sum\":%.3f}",
                      name, static_cast<unsigned long long>(values.size()),
                      sum / static_cast<double>(values.size()), values.front(), rank(0.5),
                      rank(0.9), rank(0.95), rank(0.99), values.back(), sum);
        o += buf;
    }

  private:
    static void append_bucket(std::string& o, const std::string& key,
                              const std::vector<const Sample*>& samples, const Header& h) {
        const bool attention = !samples.empty() && samples.front()->kind == kAttention;
        o += '"';
        o += json_escape(key);
        o += "\":{";
        char buf[160];
        uint64_t gpu_valid = 0, fence_blocked = 0, alloc_grew = 0, direct = 0;
        for (const Sample* s : samples) {
            gpu_valid += s->gpu_valid;
            fence_blocked += s->fence_blocked;
            alloc_grew += s->alloc_grew;
            direct += s->direct_output;
        }
        std::snprintf(buf, sizeof(buf),
                      "\"samples\":%llu,\"gpu_samples\":%llu,\"fence_blocked\":%llu,"
                      "\"alloc_grew\":%llu,\"direct_output\":%llu,",
                      ull(samples.size()), ull(gpu_valid), ull(fence_blocked), ull(alloc_grew),
                      ull(direct));
        o += buf;
        std::vector<double> v;
        auto collect = [&](auto fn) {
            v.clear();
            for (const Sample* s : samples) {
                v.push_back(fn(*s));
            }
        };
        collect([](const Sample& s) { return double(s.n_tokens); });
        append_stats(o, "n_tokens", v);
        o += ',';
        collect([](const Sample& s) { return double(s.rows); });
        append_stats(o, attention ? "active_rows" : "rows", v);
        o += ',';
        collect([](const Sample& s) { return double(s.bytes); });
        append_stats(o, "bytes", v);
        if (attention) {
            o += ',';
            collect([](const Sample& s) { return double(s.split_k); });
            append_stats(o, "split_k", v);
            o += ',';
            collect([](const Sample& s) { return double(s.groups); });
            append_stats(o, "pass1_groups", v);
        }
        o += ",\"gpu_us\":{";
        if (h.gpu_timestamp_frequency) {
            const double us_per_tick = 1e6 / static_cast<double>(h.gpu_timestamp_frequency);
            static const char* upload_names[] = {"copy"};
            static const char* attn_names[] = {"pass1", "reduce", "readback_copy"};
            const uint32_t stages = attention ? 3 : 1;
            for (uint32_t st = 0; st < stages; ++st) {
                v.clear();
                for (const Sample* s : samples) {
                    if (s->gpu_valid) {
                        v.push_back(double(s->gpu_ticks[st]) * us_per_tick);
                    }
                }
                append_stats(o, attention ? attn_names[st] : upload_names[st], v);
                o += ',';
            }
            v.clear();
            for (const Sample* s : samples) {
                if (s->gpu_valid) {
                    uint64_t total = 0;
                    for (uint32_t st = 0; st < stages; ++st) {
                        total += s->gpu_ticks[st];
                    }
                    v.push_back(double(total) * us_per_tick);
                }
            }
            append_stats(o, "total", v);
        }
        o += "},\"cpu_us\":{";
        static const char* cpu_names[] = {"validate", "alloc",      "map_copy", "record",
                                          "submit",   "fence_wait", "readback"};
        for (uint32_t span = 0; span < kCpuSpanCount; ++span) {
            if (!attention && span == kCpuReadback) {
                continue;
            }
            collect([span](const Sample& s) { return double(s.cpu_ns[span]) / 1000.0; });
            append_stats(o, cpu_names[span], v);
            o += ',';
        }
        collect([](const Sample& s) {
            uint64_t total = 0;
            for (uint32_t span = 0; span < kCpuSpanCount; ++span) {
                total += s.cpu_ns[span];
            }
            return double(total) / 1000.0;
        });
        append_stats(o, "engine_total", v);
        o += "},";
        collect([](const Sample& s) { return double(s.overhead_ns) / 1000.0; });
        append_stats(o, "profile_overhead_us", v);
        o += '}';
    }

    void append_bulk_json(std::string& o, const Header& h) const {
        char buf[512];
        o += '{';
        std::snprintf(buf, sizeof(buf),
                      "\"bulk_calls\":%llu,\"bulk_segments\":%llu,\"bulk_queries\":%llu,"
                      "\"samples_recorded\":%llu,\"samples_in_ring\":%llu,"
                      "\"gpu_timestamps\":%s",
                      ull(counters.bulk_calls), ull(counters.bulk_segments),
                      ull(counters.bulk_queries), ull(bulk_recorded_), ull(bulk_stored_),
                      (h.gpu_timestamps && h.gpu_timestamp_frequency) ? "true" : "false");
        o += buf;
        o += ",\"fp8_gpu_calls\":" + std::to_string(counters.fp8_gpu_calls) +
             ",\"fp8_cpu_fallback_calls\":" + std::to_string(counters.fp8_cpu_fallback_calls);

        std::vector<double> v_val, v_alloc, v_mc, v_rec, v_sub, v_fw, v_rb, v_tot;
        std::vector<double> v_gpu_u, v_gpu_unp, v_gpu_pack, v_gpu_p1, v_gpu_red, v_gpu_cp,
            v_gpu_tot, v_qw;
        std::vector<double> v_b_up, v_b_q, v_b_out, v_b_tot;

        const double us_per_tick = h.gpu_timestamp_frequency
                                       ? (1e6 / static_cast<double>(h.gpu_timestamp_frequency))
                                       : 0.0;

        for (size_t i = 0; i < bulk_stored_; ++i) {
            const size_t idx =
                (bulk_next_ + bulk_ring_.size() - bulk_stored_ + i) % bulk_ring_.size();
            const auto& b = bulk_ring_[idx];
            v_val.push_back(double(b.cpu_ns[kCpuValidate]) / 1000.0);
            v_alloc.push_back(double(b.cpu_ns[kCpuAlloc]) / 1000.0);
            v_mc.push_back(double(b.cpu_ns[kCpuMapCopy]) / 1000.0);
            v_rec.push_back(double(b.cpu_ns[kCpuRecord]) / 1000.0);
            v_sub.push_back(double(b.cpu_ns[kCpuSubmit]) / 1000.0);
            const double fw_us = double(b.cpu_ns[kCpuFenceWait]) / 1000.0;
            v_fw.push_back(fw_us);
            v_rb.push_back(double(b.cpu_ns[kCpuReadback]) / 1000.0);
            uint64_t cpu_tot = 0;
            for (uint32_t sp = 0; sp < kCpuSpanCount; ++sp) {
                cpu_tot += b.cpu_ns[sp];
            }
            v_tot.push_back(double(cpu_tot) / 1000.0);

            v_b_up.push_back(double(b.upload_bytes));
            v_b_q.push_back(double(b.query_bytes));
            v_b_out.push_back(double(b.output_bytes));
            v_b_tot.push_back(double(b.upload_bytes + b.query_bytes + b.output_bytes));

            if (b.gpu_valid && us_per_tick > 0.0) {
                v_gpu_u.push_back(double(b.gpu_upload_ticks) * us_per_tick);
                v_gpu_unp.push_back(double(b.gpu_unpack_ticks) * us_per_tick);
                v_gpu_pack.push_back(double(b.gpu_pack_ticks) * us_per_tick);
                v_gpu_p1.push_back(double(b.gpu_pass1_ticks) * us_per_tick);
                v_gpu_red.push_back(double(b.gpu_reduce_ticks) * us_per_tick);
                v_gpu_cp.push_back(double(b.gpu_copy_ticks) * us_per_tick);
                const double gpu_tot_us = double(b.gpu_total_ticks) * us_per_tick;
                v_gpu_tot.push_back(gpu_tot_us);
                v_qw.push_back(std::max(0.0, fw_us - gpu_tot_us));
            }
        }

        o += ",\"timings\":{";
        append_stats(o, "admission_cpu_us", v_val);
        o += ',';
        append_stats(o, "alloc_cpu_us", v_alloc);
        o += ',';
        append_stats(o, "map_copy_cpu_us", v_mc);
        o += ',';
        append_stats(o, "record_cpu_us", v_rec);
        o += ',';
        append_stats(o, "submit_cpu_us", v_sub);
        o += ',';
        append_stats(o, "fence_wait_cpu_us", v_fw);
        o += ',';
        append_stats(o, "readback_cpu_us", v_rb);
        o += ',';
        append_stats(o, "engine_total_cpu_us", v_tot);
        if (h.gpu_timestamp_frequency) {
            o += ',';
            append_stats(o, "gpu_upload_us", v_gpu_u);
            o += ',';
            append_stats(o, "gpu_unpack_wht_us", v_gpu_unp);
            o += ',';
            append_stats(o, "gpu_pack_us", v_gpu_pack);
            o += ',';
            append_stats(o, "gpu_pass1_us", v_gpu_p1);
            o += ',';
            append_stats(o, "gpu_reduce_us", v_gpu_red);
            o += ',';
            append_stats(o, "gpu_readback_copy_us", v_gpu_cp);
            o += ',';
            append_stats(o, "gpu_total_us", v_gpu_tot);
            o += ',';
            append_stats(o, "queue_wait_us", v_qw);
        }
        o += "},\"recent_envelopes\":[";
        const size_t first_recent = bulk_stored_ > 16 ? bulk_stored_ - 16 : 0;
        for (size_t i = first_recent; i < bulk_stored_; ++i) {
            const auto& b =
                bulk_ring_[(bulk_next_ + bulk_ring_.size() - bulk_stored_ + i) % bulk_ring_.size()];
            if (i != first_recent)
                o += ',';
            std::snprintf(buf, sizeof(buf),
                          "{\"cycle_id\":%llu,\"session\":%u,\"layer\":%u,\"wire\":%u,\"nq\":%u,"
                          "\"segments\":%u,\"gpu_valid\":%u,\"alloc_grew\":%u,\"gpu_total_ticks\":%"
                          "llu,\"cpu_ns\":[",
                          ull(b.cycle_id), b.session_id, b.layer, b.wire, b.total_queries,
                          b.segment_count, unsigned(b.gpu_valid), unsigned(b.alloc_grew),
                          ull(b.gpu_total_ticks));
            o += buf;
            for (uint32_t sp = 0; sp < kCpuSpanCount; ++sp) {
                if (sp)
                    o += ',';
                o += std::to_string(b.cpu_ns[sp]);
            }
            o += "]}";
        }
        o += "],\"phases\":{";
        for (uint32_t phase = 0; phase < 3; ++phase) {
            if (phase)
                o += ',';
            o += phase == 0 ? "\"decode\":{" : phase == 1 ? "\"verification\":{" : "\"prefill\":{";
            const char* cpu_names[] = {"admission_cpu_us", "alloc_cpu_us",  "map_copy_cpu_us",
                                       "record_cpu_us",    "submit_cpu_us", "fence_wait_cpu_us",
                                       "readback_cpu_us"};
            const char* gpu_names[] = {"gpu_upload_us", "gpu_unpack_wht_us", "gpu_pass1_us",
                                       "gpu_reduce_us", "gpu_pack_us",       "gpu_readback_copy_us",
                                       "gpu_total_us"};
            for (uint32_t metric = 0; metric < kCpuSpanCount + 7; ++metric) {
                if (metric)
                    o += ',';
                std::vector<double> values;
                for (size_t i = 0; i < bulk_phase_stored_[phase]; ++i) {
                    const auto& b = bulk_phase_ring_[phase][i];
                    if (metric < kCpuSpanCount)
                        values.push_back(double(b.cpu_ns[metric]) / 1000.0);
                    else if (b.gpu_valid && us_per_tick > 0.0) {
                        const uint64_t ticks[] = {b.gpu_upload_ticks, b.gpu_unpack_ticks,
                                                  b.gpu_pass1_ticks,  b.gpu_reduce_ticks,
                                                  b.gpu_pack_ticks,   b.gpu_copy_ticks,
                                                  b.gpu_total_ticks};
                        values.push_back(double(ticks[metric - kCpuSpanCount]) * us_per_tick);
                    }
                }
                append_stats(o,
                             metric < kCpuSpanCount ? cpu_names[metric]
                                                    : gpu_names[metric - kCpuSpanCount],
                             values);
            }
            o += ",\"recent\":[";
            const auto& ring = bulk_phase_ring_[phase];
            const size_t count = std::min<size_t>(4, bulk_phase_stored_[phase]);
            for (size_t i = 0; i < count; ++i) {
                const auto& b =
                    ring[(bulk_phase_next_[phase] + ring.size() - count + i) % ring.size()];
                if (i)
                    o += ',';
                o += "{\"cycle_id\":" + std::to_string(b.cycle_id) +
                     ",\"session\":" + std::to_string(b.session_id) +
                     ",\"layer\":" + std::to_string(b.layer) +
                     ",\"fp8_gpu\":" + std::to_string(b.fp8_gpu) +
                     ",\"wire\":" + std::to_string(b.wire) +
                     ",\"nq\":" + std::to_string(b.total_queries) + ",\"gpu_ticks\":[";
                const uint64_t ticks[] = {b.gpu_upload_ticks, b.gpu_unpack_ticks, b.gpu_pass1_ticks,
                                          b.gpu_reduce_ticks, b.gpu_pack_ticks,   b.gpu_copy_ticks,
                                          b.gpu_total_ticks};
                for (size_t j = 0; j < 7; ++j) {
                    if (j)
                        o += ',';
                    o += std::to_string(ticks[j]);
                }
                o += "],\"cpu_ns\":[";
                for (size_t j = 0; j < kCpuSpanCount; ++j) {
                    if (j)
                        o += ',';
                    o += std::to_string(b.cpu_ns[j]);
                }
                o += "]}";
            }
            o += "]}";
        }
        o += "},\"bytes\":{";
        append_stats(o, "upload_bytes", v_b_up);
        o += ',';
        append_stats(o, "query_bytes", v_b_q);
        o += ',';
        append_stats(o, "output_bytes", v_b_out);
        o += ',';
        append_stats(o, "total_bytes", v_b_tot);
        o += "},\"timing_breakdown_notes\":{";
        o += "\"inclusive_fence_wait\":\"fence_wait_cpu_us is inclusive host CPU wall time waiting "
             "for fence completion.\",";
        o += "\"exclusive_gpu_total\":\"gpu_total_us measures exclusive GPU execution duration "
             "between first and last queries.\",";
        o += "\"queue_wait\":\"queue_wait_us = max(0, fence_wait_cpu_us - gpu_total_us), capturing "
             "queue/driver/wakeup overhead; it is not pure GPU queue latency.\",";
        o += "\"cpu_copies\":\"CPU copy cost is map_copy_cpu_us + readback_cpu_us.\",";
        o += "\"gpu_kernel\":\"GPU kernel cost is gpu_upload_us + gpu_unpack_wht_us + gpu_pass1_us "
             "+ gpu_reduce_us + gpu_pack_us + "
             "gpu_readback_copy_us.\",";
        o += "\"unmeasurable_network_oneway\":\"One-way network transit over LAN is strictly "
             "unmeasurable without synchronized clocks. Use transport payload_receive_us, "
             "send_enqueue_us, and host RTT minus handle_us as inclusive transport diagnostics "
             "(not pure network time).\"";
        o += "}}";
    }

    std::vector<Sample> ring_;
    size_t next_ = 0;
    size_t stored_ = 0;
    uint64_t recorded_ = 0;
    std::vector<BulkEnvelopeSample> bulk_ring_;
    size_t bulk_next_ = 0;
    size_t bulk_stored_ = 0;
    uint64_t bulk_recorded_ = 0;
    std::array<std::vector<BulkEnvelopeSample>, 3> bulk_phase_ring_;
    std::array<size_t, 3> bulk_phase_next_{}, bulk_phase_stored_{};
};

} // namespace rkva_rprof
