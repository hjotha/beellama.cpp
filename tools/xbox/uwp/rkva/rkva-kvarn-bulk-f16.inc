// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Included in remote-attn-engine.cpp after Session is complete.
// Native KVarN4 bulk FP16 Q/OUT staging, GPU conversion, and readback helpers.

#if __has_include("rkva_range_bulk_unpack_q_f16_dxil.h") && \
                  __has_include("rkva_range_bulk_pack_out_f16_dxil.h")
    #include "rkva_range_bulk_pack_out_f16_dxil.h"
    #include "rkva_range_bulk_unpack_q_f16_dxil.h"
    #define RKVA_HAS_BULK_F16_DXIL 1
#else
    #define RKVA_HAS_BULK_F16_DXIL 0
#endif

#if __has_include("rkva_range_bulk_unpack_q_fp8_dxil.h") && \
                  __has_include("rkva_range_bulk_pack_out_fp8_dxil.h")
    #include "rkva_range_bulk_pack_out_fp8_dxil.h"
    #include "rkva_range_bulk_unpack_q_fp8_dxil.h"
    #define RKVA_HAS_BULK_FP8_DXIL 1
#else
    #define RKVA_HAS_BULK_FP8_DXIL 0
#endif

bool RemoteAttnEngine::ensure_bulk_fp8_gpu_pipelines() {
    if (!ready_ || unconfirmed_submission_ || terminal_failure_ || bulk_recording_)
        return false;
    if (bulk_fp8_status_ != "not initialized")
        return bool(bulk_unpack_q_fp8_pso_) && bool(bulk_pack_out_fp8_pso_);
#if RKVA_HAS_BULK_FP8_DXIL
    if (!ready_ || !device_ || !range_reduce_root_) {
        bulk_fp8_status_ = "range root/device unavailable; CPU fallback";
        return false;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = range_reduce_root_.Get();
    pso.CS = {g_rkva_range_bulk_unpack_q_fp8_dxil, sizeof(g_rkva_range_bulk_unpack_q_fp8_dxil)};
    HRESULT hr = device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&bulk_unpack_q_fp8_pso_));
    if (SUCCEEDED(hr)) {
        pso.CS = {g_rkva_range_bulk_pack_out_fp8_dxil, sizeof(g_rkva_range_bulk_pack_out_fp8_dxil)};
        hr = device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&bulk_pack_out_fp8_pso_));
    }
    if (FAILED(hr)) {
        char text[96];
        std::snprintf(text, sizeof(text), "FP8 PSO 0x%08X; CPU fallback", unsigned(hr));
        bulk_fp8_status_ = text;
        bulk_unpack_q_fp8_pso_.Reset();
        bulk_pack_out_fp8_pso_.Reset();
        return false;
    }
    bulk_fp8_status_ = "ready: CSBulkUnpackQFP8 + CSBulkPackOutFP8 (F32->F16->FP8 RNE)";
    return true;
#else
    bulk_fp8_status_ = "DXIL unavailable; CPU fallback";
    return false;
#endif
}

void RemoteAttnEngine::create_bulk_f16_pipelines() {
#if RKVA_HAS_BULK_F16_DXIL
    if (!device_ || !range_reduce_root_) {
        bulk_f16_status_ = "range reduce root signature unavailable";
        return;
    }
    char text[96];
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso = {};
    pso.pRootSignature = range_reduce_root_.Get();
    pso.CS = {g_rkva_range_bulk_unpack_q_f16_dxil, sizeof(g_rkva_range_bulk_unpack_q_f16_dxil)};
    HRESULT hr = device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&bulk_unpack_q_f16_pso_));
    if (SUCCEEDED(hr)) {
        pso.CS = {g_rkva_range_bulk_pack_out_f16_dxil, sizeof(g_rkva_range_bulk_pack_out_f16_dxil)};
        hr = device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&bulk_pack_out_f16_pso_));
    }
    if (FAILED(hr)) {
        std::snprintf(text, sizeof(text), "bulk f16 pipeline state: 0x%08X", unsigned(hr));
        bulk_unpack_q_f16_pso_.Reset();
        bulk_pack_out_f16_pso_.Reset();
        bulk_f16_status_ = text;
        return;
    }
    bulk_f16_status_ = "ready";
#else
    bulk_f16_status_ = "deferred_dxil_host_fallback";
#endif
}

bool RemoteAttnEngine::attend_kvarn_bulk_f16(uint32_t id, uint32_t layer, uint32_t n_tokens,
                                             const uint8_t* data, size_t bytes, uint16_t* out_f16,
                                             float* lse, std::string* error, uint64_t cycle_id) {
    return attend_kvarn_bulk_compact(id, layer, n_tokens, data, bytes, out_f16, nullptr, lse, error,
                                     cycle_id);
}

bool RemoteAttnEngine::attend_kvarn_bulk_fp8(uint32_t id, uint32_t layer, uint32_t n_tokens,
                                             const uint8_t* data, size_t bytes, uint8_t* out_fp8,
                                             float* lse, std::string* error, uint64_t cycle_id) {
    return attend_kvarn_bulk_compact(id, layer, n_tokens, data, bytes, nullptr, out_fp8, lse, error,
                                     cycle_id);
}

bool RemoteAttnEngine::attend_kvarn_bulk_compact(uint32_t id, uint32_t layer, uint32_t n_tokens,
                                                 const uint8_t* data, size_t bytes,
                                                 uint16_t* out_f16, uint8_t* out_fp8, float* lse,
                                                 std::string* error, uint64_t cycle_id) {
    const bool fp8_gpu = out_fp8 != nullptr;
    const bool prof = profiling_ && profile_ && !bulk_recording_;
    const auto t_val_start = prof ? prof_clock::now() : prof_clock::time_point{};
    rp::BulkEnvelopeSample envelope;
    envelope.total_queries = n_tokens;
    envelope.cycle_id = cycle_id;
    envelope.session_id = id;
    envelope.layer = layer;
    FailedCallGuard guard{prof ? &profile_->recorder.counters.failed_calls : nullptr};
    Session* s = find_session(id);
    if (!ready_ || unconfirmed_submission_ || terminal_failure_ || !s || !s->valid ||
        !s->range_configured || s->range_storage != 3 || layer >= s->layer_kv.size() ||
        (!out_f16 && !out_fp8) || !lse || !range_attn_kvarn4_prefix_pso_ || bulk_recording_)
        return ::fail(error, "native bulk f16: invalid session/arguments");
    uint32_t wire = RKVA_WIRE_F16;
    if (!rkva_kvarn_bulk::peek_wire(data, bytes, &wire))
        return ::fail(error, "native bulk f16: invalid wire");
    if (wire != RKVA_WIRE_F16 && wire != RKVA_WIRE_FP8)
        return ::fail(error, "native bulk f16: unsupported wire format");
    if (fp8_gpu && (wire != RKVA_WIRE_FP8 || !ensure_bulk_fp8_gpu_pipelines()))
        return ::fail(error, "native bulk fp8: GPU codec unavailable/invalid input wire");
    envelope.wire = wire;
    envelope.fp8_gpu = fp8_gpu;
    rkva_kvarn_bulk::ScratchCall scratch_call(bulk_scratch_, n_tokens, wire);
    auto& segments = bulk_scratch_.segments;
    if (segments.capacity() < kTsBulkMaxSegments)
        segments.reserve(kTsBulkMaxSegments);
    if (!rkva_kvarn_bulk::validate_wire(
            data, bytes, n_tokens, s->range_descriptor, geo_.n_head, geo_.n_head_kv,
            s->range_counts[layer], s->range_record_groups[layer], s->range_ready[layer],
            s->prefix_coverage[layer], s->native_generation, wire, segments))
        return ::fail(error, "native bulk f16: invalid envelope/coverage/query");
    envelope.segment_count = uint32_t(segments.size());
    if (prof)
        envelope.cpu_ns[rp::kCpuValidate] = elapsed_ns(t_val_start, prof_clock::now());
    const auto t_alloc_start = prof ? prof_clock::now() : prof_clock::time_point{};
    const uint64_t creates_before = prof ? profile_->recorder.counters.buffer_creates : 0;
    if (bulk_f16_status_ == "not initialized") {
        create_bulk_f16_pipelines();
    }
    const bool gpu_convert = fp8_gpu || bulk_f16_gpu_available();
    // Admission and all grow-only allocation precede the first native mutation.
    uint32_t max_queries = 0;
    for (const auto& segment : segments)
        max_queries = std::max(max_queries, segment.n_queries);
    if (!ensure_range_scratch(max_queries, error))
        return false;
    if (gpu_convert) {
        const uint64_t stage_out_bytes = uint64_t(max_queries) * geo_.n_head * (256 * 4 + 4);
        if (bulk_f16_stage_out_.bytes < stage_out_bytes &&
            !create_buffer(stage_out_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE,
                           D3D12_RESOURCE_STATE_COPY_DEST, &bulk_f16_stage_out_, error))
            return false;
    }
    if (bulk_transfers_.size() < segments.size())
        bulk_transfers_.resize(segments.size());
    for (size_t i = 0; i < segments.size(); ++i) {
        const auto& segment = segments[i];
        auto& transfer = bulk_transfers_[i];
        const uint64_t q_heads = uint64_t(segment.n_queries) * geo_.n_head;
        const uint64_t q_f32_bytes = q_heads * 256 * sizeof(float);
        const uint64_t q_f16_bytes = q_heads * 256 * sizeof(uint16_t);
        const uint64_t pos_bytes = uint64_t(segment.n_queries) * sizeof(int32_t);
        const uint64_t lse_bytes = q_heads * sizeof(float);
        if (transfer.upload.bytes < segment.data_bytes &&
            !create_buffer(segment.data_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                           D3D12_RESOURCE_STATE_GENERIC_READ, &transfer.upload, error))
            return false;
        if (gpu_convert) {
            const uint64_t query_upload_bytes =
                (fp8_gpu ? q_f16_bytes / 2 : q_f16_bytes) + pos_bytes;
            const uint64_t query_default_bytes = 2 * q_f32_bytes + pos_bytes;
            const uint64_t packed_output_bytes =
                fp8_gpu ? rkva_bulk_fp8::readback_bytes(uint32_t(q_heads * 256), uint32_t(q_heads))
                        : q_f16_bytes + lse_bytes;
            if ((transfer.query_f16_upload.bytes < query_upload_bytes &&
                 !create_buffer(query_upload_bytes, D3D12_HEAP_TYPE_UPLOAD,
                                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
                                &transfer.query_f16_upload, error)) ||
                (transfer.query_f32_default.bytes < query_default_bytes &&
                 !create_buffer(query_default_bytes, D3D12_HEAP_TYPE_DEFAULT,
                                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &transfer.query_f32_default,
                                error)) ||
                (transfer.packed_out_f16.bytes < packed_output_bytes &&
                 !create_buffer(packed_output_bytes, D3D12_HEAP_TYPE_DEFAULT,
                                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &transfer.packed_out_f16,
                                error)) ||
                (transfer.readback.bytes < packed_output_bytes &&
                 !create_buffer(packed_output_bytes, D3D12_HEAP_TYPE_READBACK,
                                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                                &transfer.readback, error)))
                return false;
        } else {
            const uint64_t query_bytes = 2 * q_f32_bytes + pos_bytes;
            const uint64_t output_bytes = q_f32_bytes + lse_bytes;
            if ((transfer.query.bytes < query_bytes &&
                 !create_buffer(query_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                                D3D12_RESOURCE_STATE_GENERIC_READ, &transfer.query, error)) ||
                (transfer.readback.bytes < output_bytes &&
                 !create_buffer(output_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                                D3D12_RESOURCE_STATE_COPY_DEST, &transfer.readback, error)))
                return false;
        }
    }
    auto& fallback_q_f32 = bulk_scratch_.queries;
    if (!gpu_convert) {
        fallback_q_f32.resize(size_t(max_queries) * geo_.n_head * 256);
    }
    if (prof) {
        envelope.cpu_ns[rp::kCpuAlloc] = elapsed_ns(t_alloc_start, prof_clock::now());
        envelope.alloc_grew = profile_->recorder.counters.buffer_creates != creates_before;
    }
    const auto t_record_start = prof ? prof_clock::now() : prof_clock::time_point{};
    if (!begin(error))
        return false;
    const bool ts_bulk = prof && timestamp_heap_ && segments.size() <= kTsBulkMaxSegments;
    if (prof) {
        profile_->bulk.active = true;
        profile_->bulk.segment_index = 0;
        profile_->bulk.segments.clear();
        profile_->bulk.segments.resize(segments.size());
    }
    if (ts_bulk)
        timestamp(kTsBulkEnvelopeBegin);
    bulk_recording_ = true;
    float dummy_f32 = 0.0f;
    for (size_t i = 0; i < segments.size(); ++i) {
        if (prof)
            profile_->bulk.segment_index = uint32_t(i);
        const auto& segment = segments[i];
        auto& transfer = bulk_transfers_[i];
        const uint64_t q_heads = uint64_t(segment.n_queries) * geo_.n_head;
        const uint64_t q_elems = q_heads * 256;
        const uint64_t q_f16_bytes = q_elems * sizeof(uint16_t);
        const uint64_t pos_bytes = uint64_t(segment.n_queries) * sizeof(int32_t);
        const uint64_t lse_bytes = q_heads * sizeof(float);
        bool ok = false;
        const auto t_q_map = prof ? prof_clock::now() : prof_clock::time_point{};
        if (gpu_convert) {
            void* mapped_q = nullptr;
            const HRESULT map_hr = transfer.query_f16_upload.resource->Map(0, nullptr, &mapped_q);
            if (SUCCEEDED(map_hr) && mapped_q) {
                if (fp8_gpu) {
                    std::memcpy(mapped_q, data + segment.query_offset, q_elems);
                } else if (wire == RKVA_WIRE_FP8) {
                    uint16_t* dst_q16 = static_cast<uint16_t*>(mapped_q);
                    const uint8_t* src_fp8 = data + segment.query_offset;
                    for (uint64_t k = 0; k < q_elems; ++k) {
                        dst_q16[k] = rkva_kvarn_bulk::fp8_to_f16_bits(src_fp8[k]);
                    }
                } else {
                    std::memcpy(mapped_q, data + segment.query_offset, q_f16_bytes);
                }
                std::memcpy(static_cast<uint8_t*>(mapped_q) + (fp8_gpu ? q_elems : q_f16_bytes),
                            data + segment.positions_offset, pos_bytes);
                transfer.query_f16_upload.resource->Unmap(0, nullptr);
                if (prof)
                    profile_->bulk.segments[i].map_copy_query_ns +=
                        elapsed_ns(t_q_map, prof_clock::now());
                std::swap(range_upload_, transfer.upload);
                ok = upload_kvarn_range(id, layer, segment.upload, data + segment.data_offset,
                                        segment.data_bytes, error);
                std::swap(range_upload_, transfer.upload);
            } else {
                ::fail(error, "Map bulk f16 query upload", map_hr);
            }
            if (ok) {
                const uint32_t q_pairs = uint32_t(q_elems / 2);
                const uint32_t unpack_params[4] = {q_pairs, segment.n_queries, 0u, 0u};
                transition(transfer.query_f32_default, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                cmd_->SetComputeRootSignature(range_reduce_root_.Get());
                cmd_->SetPipelineState(fp8_gpu ? bulk_unpack_q_fp8_pso_.Get()
                                               : bulk_unpack_q_f16_pso_.Get());
                cmd_->SetComputeRoot32BitConstants(0, 4, unpack_params, 0);
                cmd_->SetComputeRootShaderResourceView(
                    1, transfer.query_f16_upload.resource->GetGPUVirtualAddress());
                cmd_->SetComputeRootUnorderedAccessView(
                    2, transfer.query_f32_default.resource->GetGPUVirtualAddress());
                const uint32_t unpack_groups = (std::max(q_pairs, segment.n_queries) + 255u) / 256u;
                cmd_->Dispatch(unpack_groups, 1, 1);
                uav_barrier(transfer.query_f32_default);
                transition(transfer.query_f32_default,
                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                transition(bulk_f16_stage_out_, D3D12_RESOURCE_STATE_COPY_DEST);
                std::swap(range_query_, transfer.query_f32_default);
                std::swap(range_readback_, bulk_f16_stage_out_);
                bulk_gpu_query_ = true;
                ok = attend_range(id, layer, segment.n_queries,
                                  reinterpret_cast<const int32_t*>(data + segment.positions_offset),
                                  &dummy_f32, &dummy_f32, lse, error);
                bulk_gpu_query_ = false;
                std::swap(range_query_, transfer.query_f32_default);
                std::swap(range_readback_, bulk_f16_stage_out_);
            }
            if (ok) {
                const uint32_t q_pairs = uint32_t(q_elems / 2);
                const uint32_t q_heads_u32 = uint32_t(q_heads);
                const uint32_t pack_params[4] = {q_pairs, q_heads_u32, 0u, 0u};
                transition(range_output_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                transition(transfer.packed_out_f16, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                cmd_->SetComputeRootSignature(range_reduce_root_.Get());
                cmd_->SetPipelineState(fp8_gpu ? bulk_pack_out_fp8_pso_.Get()
                                               : bulk_pack_out_f16_pso_.Get());
                cmd_->SetComputeRoot32BitConstants(0, 4, pack_params, 0);
                cmd_->SetComputeRootShaderResourceView(
                    1, range_output_.resource->GetGPUVirtualAddress());
                cmd_->SetComputeRootUnorderedAccessView(
                    2, transfer.packed_out_f16.resource->GetGPUVirtualAddress());
                const uint32_t pack_groups =
                    (std::max(fp8_gpu ? q_pairs / 2 : q_pairs, q_heads_u32) + 255u) / 256u;
                const uint64_t packed_bytes =
                    fp8_gpu ? rkva_bulk_fp8::readback_bytes(uint32_t(q_elems), q_heads_u32)
                            : q_f16_bytes + lse_bytes;
                cmd_->Dispatch(pack_groups, 1, 1);
                uav_barrier(transfer.packed_out_f16);
                transition(transfer.packed_out_f16, D3D12_RESOURCE_STATE_COPY_SOURCE);
                if (ts_bulk)
                    timestamp(1 + uint32_t(i) * kTsBulkSlotsPerSegment + kTsBulkPackEnd);
                cmd_->CopyBufferRegion(transfer.readback.resource.Get(), 0,
                                       transfer.packed_out_f16.resource.Get(), 0, packed_bytes);
                if (ts_bulk)
                    timestamp(1 + uint32_t(i) * kTsBulkSlotsPerSegment + kTsBulkCopyEnd);
            }
        } else {
            if (wire == RKVA_WIRE_FP8) {
                const uint8_t* src_fp8 = data + segment.query_offset;
                for (size_t e = 0; e < q_elems; ++e) {
                    fallback_q_f32[e] = rkva_kvarn_bulk::f16_bits_to_f32(
                        rkva_kvarn_bulk::fp8_to_f16_bits(src_fp8[e]));
                }
            } else {
                const auto* q_f16 = reinterpret_cast<const uint16_t*>(data + segment.query_offset);
                for (size_t e = 0; e < q_elems; ++e) {
                    fallback_q_f32[e] = rkva_kvarn_bulk::f16_bits_to_f32(q_f16[e]);
                }
            }
            if (prof)
                profile_->bulk.segments[i].map_copy_query_ns +=
                    elapsed_ns(t_q_map, prof_clock::now());
            std::swap(range_upload_, transfer.upload);
            std::swap(range_query_, transfer.query);
            std::swap(range_readback_, transfer.readback);
            ok = upload_kvarn_range(id, layer, segment.upload, data + segment.data_offset,
                                    segment.data_bytes, error) &&
                 attend_range(id, layer, segment.n_queries,
                              reinterpret_cast<const int32_t*>(data + segment.positions_offset),
                              fallback_q_f32.data(), &dummy_f32, lse, error);
            std::swap(range_upload_, transfer.upload);
            std::swap(range_query_, transfer.query);
            std::swap(range_readback_, transfer.readback);
        }
        if (prof && ok) {
            auto& sample = profile_->bulk.segments[i];
            sample.query_bytes = q_elems * (wire == RKVA_WIRE_FP8 ? 1 : 2) + pos_bytes;
            sample.out_bytes = (fp8_gpu ? q_elems : q_f16_bytes) + lse_bytes;
            sample.dispatches += gpu_convert ? 2 : 0;
        }
        if (ts_bulk && ok && !gpu_convert) {
            const uint32_t base = 1 + uint32_t(i) * kTsBulkSlotsPerSegment;
            timestamp(base + kTsBulkPackEnd);
            timestamp(base + kTsBulkCopyEnd);
        }
        if (!ok) {
            if (prof)
                profile_->bulk.active = false;
            bulk_recording_ = false;
            bulk_gpu_query_ = false;
            s->valid = false;
            terminal_failure_ = true;
            return false;
        }
    }
    if (prof) {
        // Whole recording interval includes query CPU expansion and upload map/copies.
        // Remove those subspans once so the envelope CPU spans are disjoint.
        uint64_t map_ns = 0;
        for (const auto& seg : profile_->bulk.segments)
            map_ns += seg.map_copy_upload_ns + seg.map_copy_query_ns;
        const uint64_t record_ns = elapsed_ns(t_record_start, prof_clock::now());
        envelope.cpu_ns[rp::kCpuRecord] = record_ns > map_ns ? record_ns - map_ns : 0;
    }
    const uint32_t total_slots = 2 + uint32_t(segments.size()) * kTsBulkSlotsPerSegment;
    if (ts_bulk) {
        timestamp(total_slots - 1);
        resolve_timestamps(0, total_slots);
    }
    bulk_recording_ = false;
    const auto t_submit = prof ? prof_clock::now() : prof_clock::time_point{};
    if (!submit(error)) {
        if (prof)
            profile_->bulk.active = false;
        s->valid = false;
        return false;
    }
    if (prof) {
        const uint64_t total_ns = elapsed_ns(t_submit, prof_clock::now());
        const uint64_t exec_ns = std::min(last_submit_exec_ns_, total_ns);
        envelope.cpu_ns[rp::kCpuSubmit] = exec_ns;
        envelope.cpu_ns[rp::kCpuFenceWait] = total_ns - exec_ns;
        envelope.fence_blocked = last_submit_blocked_;
    }
    const auto t_readback = prof ? prof_clock::now() : prof_clock::time_point{};
    size_t row = 0;
    for (size_t i = 0; i < segments.size(); ++i) {
        const auto t_seg_rb = prof ? prof_clock::now() : prof_clock::time_point{};
        const uint64_t heads = uint64_t(segments[i].n_queries) * geo_.n_head;
        const uint64_t q_elems = heads * 256;
        const uint64_t q_f16_bytes = q_elems * sizeof(uint16_t);
        const uint64_t q_f32_bytes = q_elems * sizeof(float);
        const uint64_t lse_bytes = heads * sizeof(float);
        const uint64_t readback_bytes =
            fp8_gpu ? rkva_bulk_fp8::readback_bytes(uint32_t(q_elems), uint32_t(heads))
                    : (gpu_convert ? q_f16_bytes : q_f32_bytes) + lse_bytes;
        void* mapped = nullptr;
        D3D12_RANGE range{0, static_cast<SIZE_T>(readback_bytes)};
        const HRESULT hr = bulk_transfers_[i].readback.resource->Map(0, &range, &mapped);
        if (FAILED(hr) || !mapped) {
            if (prof)
                profile_->bulk.active = false;
            s->valid = false;
            return fail(error, "Map native bulk f16 readback", hr);
        }
        uint16_t* dst_out = out_f16 ? out_f16 + row * geo_.n_head * 256 : nullptr;
        float* dst_lse = lse + row * geo_.n_head;
        bool finite_ok = true;
        if (fp8_gpu) {
            const uint8_t* packed = static_cast<const uint8_t*>(mapped);
            const uint32_t groups = rkva_bulk_fp8::pack_groups(uint32_t(q_elems), uint32_t(heads));
            for (uint32_t g = 0; g < groups; ++g) {
                uint32_t status;
                std::memcpy(&status, packed + q_elems + lse_bytes + uint64_t(g) * 4, 4);
                if (status) {
                    finite_ok = false;
                    break;
                }
            }
            if (finite_ok) {
                std::memcpy(out_fp8 + row * geo_.n_head * 256, packed, q_elems);
                std::memcpy(dst_lse, packed + q_elems, lse_bytes);
            }
        } else if (gpu_convert) {
            std::memcpy(dst_out, mapped, q_f16_bytes);
            std::memcpy(dst_lse, static_cast<const uint8_t*>(mapped) + q_f16_bytes, lse_bytes);
            for (uint64_t e = 0; e < q_elems; ++e) {
                if (!rkva_kvarn_bulk::f16_bits_finite(dst_out[e])) {
                    finite_ok = false;
                    break;
                }
            }
        } else {
            const auto* src_out = static_cast<const float*>(mapped);
            for (uint64_t e = 0; e < q_elems; ++e) {
                if (!std::isfinite(src_out[e])) {
                    finite_ok = false;
                    break;
                }
                const uint16_t packed = rkva_kvarn_bulk::f32_to_f16_bits(src_out[e]);
                if (!rkva_kvarn_bulk::f16_bits_finite(packed)) {
                    finite_ok = false;
                    break;
                }
                dst_out[e] = packed;
            }
            std::memcpy(dst_lse, static_cast<const uint8_t*>(mapped) + q_f32_bytes, lse_bytes);
        }
        for (uint64_t h = 0; finite_ok && h < heads; ++h) {
            if (!std::isfinite(dst_lse[h])) {
                finite_ok = false;
            }
        }
        D3D12_RANGE empty{0, 0};
        bulk_transfers_[i].readback.resource->Unmap(0, &empty);
        if (!finite_ok) {
            if (prof)
                profile_->bulk.active = false;
            s->valid = false;
            return ::fail(error, "native bulk f16: non-finite or overflowed attention output");
        }
        row += segments[i].n_queries;
        if (prof)
            profile_->bulk.segments[i].readback_ns = elapsed_ns(t_seg_rb, prof_clock::now());
    }
    if (prof) {
        envelope.cpu_ns[rp::kCpuReadback] = elapsed_ns(t_readback, prof_clock::now());
        finish_bulk_profile(envelope, ts_bulk);
        guard.done = true;
    }
    return true;
}

// Explicit diagnostic op11/id5 only. It never looks up a Session or changes KV,
// coverage/generation. Status remains in this diagnostic response for the oracle.
bool RemoteAttnEngine::probe_bulk_fp8_pack(const uint8_t* data, size_t bytes,
                                           std::vector<uint8_t>& output, std::string* error) {
    // Never grow/replace an upload or output resource still referenced by an
    // unconfirmed submission, even if a diagnostic client sends another call.
    if (!ready_ || unconfirmed_submission_ || terminal_failure_ || bulk_recording_)
        return ::fail(error, "bulk fp8 pack probe: executor unavailable/unconfirmed");
    rkva_bulk_fp8::ProbeLayout layout;
    if (!rkva_bulk_fp8::validate_probe(data, bytes, layout))
        return ::fail(error, "bulk fp8 pack probe: invalid bounded payload");
    if (!ensure_bulk_fp8_gpu_pipelines() || bulk_recording_)
        return ::fail(error, "bulk fp8 pack probe: unavailable or busy");
    if ((bulk_probe_upload_.bytes < layout.source_bytes &&
         !create_buffer(layout.source_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                        D3D12_RESOURCE_STATE_GENERIC_READ, &bulk_probe_upload_, error)) ||
        (bulk_probe_packed_.bytes < layout.readback_bytes &&
         !create_buffer(layout.readback_bytes, D3D12_HEAP_TYPE_DEFAULT,
                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &bulk_probe_packed_, error)) ||
        (bulk_probe_readback_.bytes < layout.readback_bytes &&
         !create_buffer(layout.readback_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                        D3D12_RESOURCE_STATE_COPY_DEST, &bulk_probe_readback_, error)))
        return false;
    output.resize(layout.readback_bytes);
    void* mapped = nullptr;
    HRESULT hr = bulk_probe_upload_.resource->Map(0, nullptr, &mapped);
    if (FAILED(hr) || !mapped)
        return fail(error, "Map fp8 probe upload", hr);
    std::memcpy(mapped, data + layout.source_offset, layout.source_bytes);
    bulk_probe_upload_.resource->Unmap(0, nullptr);
    if (!begin(error))
        return false;
    transition(bulk_probe_packed_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const uint32_t params[4] = {layout.elements / 2, layout.heads, 0, 0};
    cmd_->SetComputeRootSignature(range_reduce_root_.Get());
    cmd_->SetPipelineState(bulk_pack_out_fp8_pso_.Get());
    cmd_->SetComputeRoot32BitConstants(0, 4, params, 0);
    cmd_->SetComputeRootShaderResourceView(1, bulk_probe_upload_.resource->GetGPUVirtualAddress());
    cmd_->SetComputeRootUnorderedAccessView(2, bulk_probe_packed_.resource->GetGPUVirtualAddress());
    cmd_->Dispatch(layout.groups, 1, 1);
    uav_barrier(bulk_probe_packed_);
    transition(bulk_probe_packed_, D3D12_RESOURCE_STATE_COPY_SOURCE);
    cmd_->CopyBufferRegion(bulk_probe_readback_.resource.Get(), 0,
                           bulk_probe_packed_.resource.Get(), 0, layout.readback_bytes);
    if (!submit(error)) {
        output.clear();
        return false;
    }
    D3D12_RANGE range{0, layout.readback_bytes};
    hr = bulk_probe_readback_.resource->Map(0, &range, &mapped);
    if (FAILED(hr) || !mapped) {
        output.clear();
        return fail(error, "Map fp8 probe readback", hr);
    }
    std::memcpy(output.data(), mapped, layout.readback_bytes);
    D3D12_RANGE empty{0, 0};
    bulk_probe_readback_.resource->Unmap(0, &empty);
    return true;
}
