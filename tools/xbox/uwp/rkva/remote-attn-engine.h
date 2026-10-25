#pragma once

// RemoteAttnEngine — D3D12 executor for the RKVA remote KVarN attention
// service on Xbox Series X. Owns the compressed KV (KVarN records + F16 stage
// + exact precision tail) per (session, layer) and runs the KVarN WHT / store /
// flash-attention-over-records kernels on the console APU (RDNA2, wave32).
//
// One engine serves the single persistent RKVA host connection; calls are
// strictly serialized by the listener, so the engine uses one command queue /
// allocator / list / fence and never needs internal locking.
//
// The host sends Q/K/V F32 (post-norm, post-MRoPE, PRE-WHT) plus absolute
// positions. The engine reproduces the host KVarN semantics exactly: WHT
// rotation per the negotiated domain, Sinkhorn record compression, the F16
// stage for the incomplete group, the exact precision tail, causal masking,
// and the inverse WHT on the output. Bit-parity with the CUDA fattn-kvarn path
// is the acceptance gate (see the RKVA PROBE harness).

#ifndef NOMINMAX
    #define NOMINMAX
#endif

#include "remote-attn-protocol.h"
#include "rkva-bulk-fp8.h"
#include "rkva-bulk-scratch.h"
#include "rkva-execution-lifetime.h"
#include <d3d12.h>
#include <wrl/client.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace rkva_rprof {
struct Sample;
struct BulkEnvelopeSample;
} // namespace rkva_rprof

class RemoteAttnEngine {
  public:
    // Geometry negotiated from RKVA HELLO. Mirrors rkva_hello on the wire.
    struct Geometry {
        uint32_t n_head = 0;         // Q heads (qwen35: 24)
        uint32_t n_head_kv = 0;      // KV heads (qwen35: 4)
        uint32_t head_dim = 0;       // 256
        uint32_t head_slices = 0;    // head_dim / 128 (2)
        uint32_t cache_bits_k = 0;   // 4 for kvarn4; 16 = F16 bring-up path
        uint32_t cache_bits_v = 0;   // 4 for kvarn4; 16 = F16 bring-up path
        uint32_t group_tokens = 128; // KVarN record group (KVAR_N_GROUP)
        uint32_t sinkhorn_iters = 0;
        uint32_t tail_tokens = 0;  // exact precision tail (0 = policy default)
        uint32_t tail_groups = 2;  // rollback insurance groups
        uint32_t stage_groups = 3; // F16 stage depth (tail_groups + 1 sink)
        uint32_t domain = 0;       // rkva_domain (AUTO/ROTATED/ORIGINAL_V)
        float kq_scale = 0.0f;     // 1/sqrt(head_dim)
        uint32_t max_ctx = 0;      // per-session position capacity
        uint32_t n_layers = 0;     // remote full-attention layer count (16)
    };

    RemoteAttnEngine();
    ~RemoteAttnEngine();
    RemoteAttnEngine(const RemoteAttnEngine&) = delete;
    RemoteAttnEngine& operator=(const RemoteAttnEngine&) = delete;

    // Create the D3D12 device, root signatures, PSOs and geometry-derived
    // scratch buffers. Session KV resources are allocated per create_session.
    // A failed initialize never leaves the engine half-enabled.
    bool initialize(const Geometry& geo, std::string* error);
    bool submission_unconfirmed() const {
        return unconfirmed_submission_;
    }
    // Owner calls before destruction. Queries the existing fence/device only;
    // never waits, submits a marker, CPU-signals a fence, or retries work.
    bool confirm_execution_retired();
    bool ready() const {
        return ready_;
    }
    bool native_serial_prefix_available() const {
        return ready_ && range_attn_kvarn4_prefix_pso_.Get() != nullptr;
    }
    const Geometry& geometry() const {
        return geo_;
    }

    // Session = one host sequence's KV context. capacity is in positions
    // (0 = geo.max_ctx). Preallocates per-layer record/stage/tail resources;
    // nothing is allocated per token afterwards.
    bool create_session(uint32_t session_id, uint32_t capacity, std::string* error,
                        bool defer_allocation = false);
    bool destroy_session(uint32_t session_id);
    bool has_session(uint32_t session_id) const;
    bool reset_session(uint32_t session_id, std::string* error);
    // Remove every stored position >= pos0 (trim and speculative rewind).
    bool trim_session(uint32_t session_id, int32_t pos0, std::string* error);

    // One layer's full attention step. Appends K/V (WHT + KVarN store into the
    // session's records/stage/tail), then computes causal attention of Q over
    // the stored KV for that layer.
    //   q:  [n_tokens][n_head][head_dim]     F32, post-RoPE, pre-WHT
    //   k:  [n_tokens][n_head_kv][head_dim]  F32, post-norm/RoPE, pre-WHT
    //   v:  [n_tokens][n_head_kv][head_dim]  F32
    //   positions: [n_tokens] absolute int32 within the session
    //   out: [n_tokens][n_head][head_dim]    F32, post inverse-WHT per domain
    // layer_abs is the absolute model layer index (qwen35 full-attn: 3,7,...,63).
    // On any GPU/recording failure after mutation the session is invalidated.
    bool attend(uint32_t session_id, uint32_t layer_abs, uint32_t n_tokens,
                const int32_t* positions, const float* q, const float* k, const float* v,
                float* out, std::string* error);

    // V2 positional overflow. Physical row i represents origin+i. F16/Q4_0/Q8_0
    // uploads preserve host cache values; no transform or requantization applies.
    bool configure_range(uint32_t session_id, uint32_t origin, uint32_t capacity,
                         std::string* error, uint32_t storage = 0);
    bool upload_range(uint32_t session_id, uint32_t layer, uint32_t start_row, uint32_t n_rows,
                      const uint16_t* k, const uint16_t* v, std::string* error);
    uint64_t native_range_generation(uint32_t id) const;
    bool rewind_kvarn_range(uint32_t id, const rkva_kvarn_range_rewind& request,
                            const uint8_t* data, size_t bytes, std::string* error);
    bool configure_kvarn_range(uint32_t id, const rkva_kvarn_range_config& config,
                               std::string* error);
    bool upload_kvarn_range(uint32_t id, uint32_t layer, const rkva_kvarn_range_upload& upload,
                            const uint8_t* data, size_t bytes, std::string* error);
    // Layout-4 Q4_0 envelope: every segment upload and attention is recorded in
    // ONE command list and published by a single submit/fence, then each
    // segment's OUT/LSE is read back. Q wire may be F32, F16 or FP8.
    bool attend_q4_bulk(uint32_t id, uint32_t layer, uint32_t n_tokens, const uint8_t* data,
                        size_t bytes, float* out, float* lse, std::string* error);
    bool attend_q8_bulk(uint32_t id, uint32_t layer, uint32_t n_tokens, const uint8_t* data,
                        size_t bytes, float* out, float* lse, std::string* error);
    // Shared layout-4/5 implementation: storage selects the row format and
    // row_bytes is the per-token plane stride (heads * 144 or heads * 272).
    bool attend_native_rows(uint32_t id, uint32_t layer, uint32_t n_tokens, uint32_t storage,
                            uint64_t row_bytes, const uint8_t* data, size_t bytes, float* out,
                            float* lse, std::string* error);
    bool attend_kvarn_bulk(uint32_t id, uint32_t layer, uint32_t n_tokens, const uint8_t* data,
                           size_t bytes, float* out, float* lse, std::string* error,
                           uint64_t cycle_id = 0);
    bool attend_kvarn_bulk_f16(uint32_t id, uint32_t layer, uint32_t n_tokens, const uint8_t* data,
                               size_t bytes, uint16_t* out_f16, float* lse, std::string* error,
                               uint64_t cycle_id = 0);
    bool ensure_bulk_fp8_gpu_pipelines();
    bool attend_kvarn_bulk_fp8(uint32_t id, uint32_t layer, uint32_t n_tokens, const uint8_t* data,
                               size_t bytes, uint8_t* out_fp8, float* lse, std::string* error,
                               uint64_t cycle_id = 0);
    bool probe_bulk_fp8_pack(const uint8_t* data, size_t bytes, std::vector<uint8_t>& output,
                             std::string* error);
    bool bulk_f16_gpu_available() const {
        return bool(bulk_unpack_q_f16_pso_) && bool(bulk_pack_out_f16_pso_);
    }
    bool attend_range(uint32_t session_id, uint32_t layer, uint32_t n_tokens,
                      const int32_t* positions, const float* q, float* out, float* lse,
                      std::string* error);

    // LocalState path comes only from serve_remote_attn, never from the wire.
    bool disk_snapshot(uint32_t session_id, const std::string& local_folder,
                       const rkva_snapshot_req& request, bool restore, rkva_snapshot_ack* ack,
                       std::string* error);

    // Range-attention profiling and Q4 kernel selection (control path only; call
    // while the engine is idle). Invalid modes or an unavailable optimized kernel
    // return false without mutating any state. Profiling is OFF by default; it
    // records D3D12 timestamps on the existing queue (resolved in the same command
    // list, read after the existing fence wait) plus disjoint CPU spans into a
    // preallocated RAM ring. Mode 0 = reference kernel of every storage (default);
    // modes 1..3 (legacy name q4_kernel_mode) = optimized GQA decode kernel
    // variants for qualified nq=1 calls of Q4_0 and, when their own pipeline was
    // created, F16, Q8_0 and native KVarN4; every other call keeps the reference
    // path of its storage. See docs/rkva-range-profiling.md.
    bool set_range_profiling(bool enabled, bool reset, uint32_t q4_kernel_mode, std::string& error);
    std::string get_range_profile_json();
    std::string get_memory_profile_json();
    uint32_t q4_kernel_mode() const {
        return q4_kernel_mode_;
    }
    bool range_profiling() const {
        return profiling_;
    }

    // Diagnostics / parity harness support.
    uint64_t allocated_bytes() const {
        return allocated_bytes_;
    }
    bool session_invalid(uint32_t session_id) const;

  private:
    struct Session;      // per-session, per-layer KV resources (defined in .cpp)
    struct RangeProfile; // in-RAM profiling state (defined in .cpp)
    struct Buffer {
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
        uint64_t bytes = 0;
    };

    bool fail(std::string* error, const char* stage, HRESULT hr = S_OK);
    bool create_buffer(uint64_t bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_FLAGS flags,
                       D3D12_RESOURCE_STATES state, Buffer* out, std::string* error);
    // Same as create_buffer but never marks the engine failed (control-path use).
    HRESULT try_create_buffer(uint64_t bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_FLAGS flags,
                              D3D12_RESOURCE_STATES state, Buffer* out);
    void create_q4_opt_pipelines();
    // F16/Q8_0/KVarN4 optimized pipelines; needs the Q4 opt prerequisites.
    void create_format_opt_pipelines();
    void create_bulk_f16_pipelines();
    bool attend_kvarn_bulk_compact(uint32_t id, uint32_t layer, uint32_t n_tokens,
                                   const uint8_t* data, size_t bytes, uint16_t* out_f16,
                                   uint8_t* out_fp8, float* lse, std::string* error,
                                   uint64_t cycle_id);
    bool ensure_profile_resources(std::string* error);
    void timestamp(uint32_t slot);
    void resolve_timestamps(uint32_t first, uint32_t count);
    bool read_timestamps(uint32_t first, uint32_t count, uint64_t* ticks);
    // Splits submit() into exec/fence spans, reads timestamps and records the sample.
    void finish_profile_sample(rkva_rprof::Sample& sample, bool timestamps, uint32_t first_slot,
                               uint32_t slot_count, uint32_t copy_regions, uint32_t dispatches);
    void finish_bulk_profile(rkva_rprof::BulkEnvelopeSample& envelope, bool timestamps);
    void transition(Buffer& buffer, D3D12_RESOURCE_STATES state);
    void uav_barrier(Buffer& buffer);
    bool begin(std::string* error);
    bool submit(std::string* error);
    bool create_device(std::string* error);
    bool create_pipelines(std::string* error);
    bool ensure_scratch(uint32_t n_tokens, std::string* error);
    bool ensure_range_scratch(uint32_t n_tokens, std::string* error);
    Session* find_session(uint32_t session_id);
    const Session* find_session(uint32_t session_id) const;

    // Geometry-derived sizes computed in initialize().
    uint64_t record_bytes_k_ = 0; // bytes per K record group (128 tokens, all kv heads/slices)
    uint64_t record_bytes_v_ = 0;
    uint64_t stage_row_bytes_ = 0; // F16 stage bytes per token per layer (K or V)
    uint32_t groups_per_stream_ = 0;

    bool ready_ = false;
    Geometry geo_;

    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue_;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator_;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> cmd_;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
    HANDLE fence_event_ = nullptr;
    UINT64 fence_value_ = 0;

    // Root signatures + PSOs for the KVarN kernels.
    Microsoft::WRL::ComPtr<ID3D12RootSignature> wht_root_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> store_root_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> attn_root_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> store_kvarn_root_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> attn_kvarn_root_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> split_k_reduce_root_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> wht_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> store_k_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> store_v_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> store_kvarn_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> attn_decode_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> attn_prefill_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> attn_kvarn_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> split_k_reduce_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> materialize_pso_;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> range_attn_root_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> range_reduce_root_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> range_attn_pso_[4];
    Microsoft::WRL::ComPtr<ID3D12PipelineState> range_reduce_pso_;
    // Optimized Q4_0 decode kernel (optional; reference stays available).
    Microsoft::WRL::ComPtr<ID3D12RootSignature> range_attn_q4_opt_root_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> range_attn_q4_opt_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> range_reduce_opt_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> range_reduce_kvarn4_pso_;
    std::string q4_opt_status_ = "not initialized";
    // Q4_0 multi-query tile kernels ([0] 2-query, [1] 4-query) for nq 2..9 in
    // modes 1/2; optional, a missing PSO keeps the per-query optimized kernel.
    Microsoft::WRL::ComPtr<ID3D12PipelineState> range_attn_q4_mq_pso_[2];
    std::string q4_mq_status_[2] = {"not initialized", "not initialized"};
    uint32_t wave_lane_min_ = 0;
    uint32_t wave_lane_max_ = 0;
    uint32_t q4_kernel_mode_ = 0;
    Buffer range_opt_partials_; // allocated by set_range_profiling, never in a timed RPC
    // Optimized F16 / Q8_0 / native KVarN4 decode kernels (index = range storage;
    // [1] stays empty, Q4_0 uses the pipeline above). Each one is optional.
    Microsoft::WRL::ComPtr<ID3D12RootSignature> range_attn_fmt_opt_root_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> range_attn_fmt_opt_pso_[4];
    std::string fmt_opt_status_[4] = {"not initialized", "q4 uses the q4 pipeline",
                                      "not initialized", "not initialized"};
    // F16 / Q8_0 multi-query tile kernels [storage][0] 2-query, [storage][1]
    // 4-query (267) for nq 2..9 in modes 1/2; same root signature, optional: a
    // missing PSO keeps that storage's per-query optimized kernel. Q4_0 uses
    // range_attn_q4_mq_pso_; KVarN4 uses range_attn_kvarn4_mq_pso_.
    Microsoft::WRL::ComPtr<ID3D12PipelineState> range_attn_fmt_mq_pso_[4][2];
    std::string fmt_mq_status_[4][2] = {{"not initialized", "not initialized"},
                                        {"q4 uses q4_multi_query", "q4 uses q4_multi_query"},
                                        {"not initialized", "not initialized"},
                                        {"not supported", "not supported"}};
    // Native KVarN4 descriptor revision 3 (serial query prefix, 256-row tail ring).
    // Separate PSOs from the same sources; a revision-3 session never uses the
    // revision 1/2 PSOs. Missing reference PSO: revision 3 configure fails closed.
    // Missing optimized PSO: modes 1..3 keep the revision-3 reference kernel.
    Microsoft::WRL::ComPtr<ID3D12PipelineState> range_attn_kvarn4_prefix_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> range_attn_kvarn4_opt_prefix_pso_;
    std::string kvarn4_prefix_status_ = "not initialized";
    // Native KVarN4 revision-3 2-query (mode 2) and 4-query (mode 1) tiles for nq
    // 2..9 with the optimized prefix PSO; a missing tile PSO falls back to the
    // per-query prefix kernel.
    Microsoft::WRL::ComPtr<ID3D12PipelineState> range_attn_kvarn4_mq_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> range_attn_kvarn4_mq4_pso_;
    std::string kvarn4_mq_status_ = "not initialized";
    std::string kvarn4_mq4_status_ = "not initialized";
    std::string kvarn4_opt_prefix_status_ = "not initialized";
    // Native KVarN4 revision-3 wave64 decode specialization (nq == 1).
    Microsoft::WRL::ComPtr<ID3D12PipelineState> range_attn_kvarn4_decode_prefix_pso_;
    std::string kvarn4_decode_prefix_status_ = "not initialized";

    // Profiling (default off). Timestamp slots are fixed per stage so a future
    // fused upload+attention command list cannot alias them.
    bool profiling_ = false;
    Microsoft::WRL::ComPtr<ID3D12QueryHeap> timestamp_heap_;
    Buffer timestamp_readback_;
    uint64_t timestamp_frequency_ = 0;
    std::string timestamp_status_ = "not initialized";
    uint32_t timestamp_set_ = 0;
    uint64_t last_submit_exec_ns_ = 0;
    bool last_submit_blocked_ = false;
    std::unique_ptr<RangeProfile> profile_;
    // Keep transfer resources alive after an unconfirmed GPU submission.
    Buffer snapshot_upload_;
    Buffer snapshot_readback_;
    // Native bulk: distinct transfer regions survive the single final fence.
    // Grow-only bounded pool (32 segments), also retained on unconfirmed submit.
    struct BulkTransfer {
        Buffer upload, query, readback;
        Buffer query_f16_upload, query_f32_default, packed_out_f16;
    };
    std::vector<BulkTransfer> bulk_transfers_;
    rkva_kvarn_bulk::Scratch bulk_scratch_;
    bool bulk_recording_ = false;
    bool bulk_gpu_query_ = false;
    Buffer bulk_f16_stage_out_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> bulk_unpack_q_f16_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> bulk_pack_out_f16_pso_;
    std::string bulk_f16_status_ = "not initialized";
    Microsoft::WRL::ComPtr<ID3D12PipelineState> bulk_unpack_q_fp8_pso_, bulk_pack_out_fp8_pso_;
    std::string bulk_fp8_status_ = "not initialized";
    Buffer bulk_probe_upload_, bulk_probe_packed_, bulk_probe_readback_;
    Buffer range_upload_;
    Buffer range_query_;
    Buffer range_partials_;
    Buffer range_output_;
    Buffer range_readback_;
    uint32_t range_scratch_tokens_ = 0;

    // Shared scratch (Q/K/V upload ring, output readback, WHT workspace).
    Buffer qkv_upload_;
    Buffer pos_upload_;
    Buffer out_readback_;
    Buffer wht_workspace_;
    Buffer split_accum_; // DEFAULT UAV: split-K partial accumulators
    Buffer q_rotated_;   // DEFAULT UAV: WHT(Q) input to the kvarn attention

    // Absolute model layer index -> dense KV slot (assigned on first use).
    std::map<uint32_t, uint32_t> layer_slot_;

    // Scratch is sized for this many tokens per call (grow-only); decode uses 1.
    uint32_t scratch_tokens_ = 0;

    std::vector<std::unique_ptr<Session>> sessions_;
    uint64_t allocated_bytes_ = 0;
    bool terminal_failure_ = false;
    bool unconfirmed_submission_ = false;
};
