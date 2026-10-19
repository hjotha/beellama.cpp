// Host-only v2 RKVA loopback contract test. No model, GPU or Xbox required.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-remote-attn.h"
#include "rkva-snapshot-digest.h"
#include "../ggml/src/ggml-remote-attn/remote-attn-protocol.h"
#include "../ggml/src/ggml-remote-attn/rkva-kvarn-rewind.h"
#undef NDEBUG
#include <cassert>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <map>
#include <string>
#include <thread>
#include <vector>
#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

struct loopback {
    enum class profile_failure { none, cycle, version, oversized, truncated, server_error };
    int listener = -1;
    std::atomic<int> client {-1};
    uint16_t port = 0;
    std::thread worker;
    std::mutex mutex;
    std::vector<rkva_range_upload> uploads;
    std::vector<std::vector<uint8_t>> upload_payloads;
    int attention_calls = 0;
    int fused_calls = 0;
    std::vector<uint16_t> packet_ops;
    std::vector<uint32_t> session_flags;
    rkva_hello geometry {};
    rkva_range_config range {};
    std::vector<float> keys, values;
    uint32_t rows = 0;
    struct history { std::vector<float> keys, values; uint32_t rows = 0; };
    std::map<uint32_t, history> other_layers;
    bool fail_trim = false;
    profile_failure fail_native_rewind = profile_failure::none;
    std::vector<uint32_t> native_counts;
    std::vector<bool> native_ready;
    rkva_kvarn_range_config native_descriptor {};
    uint64_t native_generation = 17;
    std::vector<rkva_range_profile> profile_controls;
    bool profile_enabled = false;
    uint32_t kernel_mode = 0;
    profile_failure fail_report = profile_failure::none;

    static bool read_all(int fd, void * p, size_t n) {
        auto * dst = (char *) p;
        while (n) { const auto got = recv(fd, dst, n, 0); if (got <= 0) return false; dst += got; n -= got; }
        return true;
    }
    static bool write_all(int fd, const void * p, size_t n) {
        const auto * src = (const char *) p;
        while (n) { const auto got = send(fd, src, n, MSG_NOSIGNAL); if (got <= 0) return false; src += got; n -= got; }
        return true;
    }
    // profile_support bits: 1 = capability, 2 = operation. Both are required.
    explicit loopback(bool old_version = false, bool range_ops = true, bool native_q4 = false, bool force_f16 = false, bool native_q8 = false, bool fused_range = false, uint32_t profile_support = 0, uint32_t format_kernels = 0, bool native_kvarn = false, uint32_t native_rewind_support = 0) {
        listener = socket(AF_INET, SOCK_STREAM, 0);
        assert(listener >= 0);
        sockaddr_in address {}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        assert(bind(listener, (sockaddr *) &address, sizeof(address)) == 0);
        socklen_t length = sizeof(address);
        assert(getsockname(listener, (sockaddr *) &address, &length) == 0);
        port = ntohs(address.sin_port);
        assert(listen(listener, 1) == 0);
        worker = std::thread([this, old_version, range_ops, native_q4, force_f16, native_q8, fused_range, profile_support, format_kernels, native_kvarn, native_rewind_support] {
            const int fd = accept(listener, nullptr, nullptr);
            client = fd;
            if (fd < 0) return;
            const int no_delay = 1;
            assert(setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &no_delay, sizeof(no_delay)) == 0);
            timeval timeout {5, 0};
            setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
            rkva_request req {};
            while (read_all(fd, &req, sizeof(req))) {
                assert(req.magic == RKVA_MAGIC && req.version == RKVA_VERSION);
                assert(req.payload_bytes < (32u << 20));
                std::vector<uint8_t> payload(req.payload_bytes), response;
                auto response_failure = profile_failure::none;
                if (req.payload_bytes && !read_all(fd, payload.data(), payload.size())) break;
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    auto & other = other_layers[req.layer_id];
                    auto & keys = req.layer_id == 0 ? this->keys : other.keys;
                    auto & values = req.layer_id == 0 ? this->values : other.values;
                    auto & rows = req.layer_id == 0 ? this->rows : other.rows;
                    auto apply_upload = [&](const std::vector<uint8_t> & data) {
                            rkva_range_upload upload {}; std::memcpy(&upload, data.data(), sizeof(upload));
                            const size_t width = geometry.n_head_kv * geometry.head_dim;
                            const bool native_storage = range.reserved[0] != 0;
                            const auto storage_type = range.reserved[0] == RKVA_RANGE_STORAGE_Q8_0 ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_0;
                            const auto storage_wire = native_storage ? (storage_type == GGML_TYPE_Q8_0 ? RKVA_WIRE_Q8_0 : RKVA_WIRE_Q4_0) : RKVA_WIRE_F16;
                            assert(upload.wire == storage_wire);
                            assert(upload.start_row == 0 || upload.start_row == rows);
                            const size_t row_bytes = native_storage ? ggml_row_size(storage_type, width) : width * 2;
                            assert(data.size() == sizeof(upload) + upload.n_rows * row_bytes * 2);
                            assert(upload.start_row + upload.n_rows <= range.capacity);
                            rows = upload.start_row + upload.n_rows;
                            keys.resize(rows * width); values.resize(rows * width);
                            const uint8_t * src = data.data() + sizeof(upload);
                            for (auto * output : {&keys, &values}) {
                                if (native_storage) {
                                    const auto decode = ggml_get_type_traits(storage_type)->to_float;
                                    for (size_t row = upload.start_row; row < rows; ++row) {
                                        decode(src, output->data() + row * width, width);
                                        src += row_bytes;
                                    }
                                } else {
                                    for (size_t j = upload.start_row * width; j < rows * width; ++j) {
                                        uint16_t half; std::memcpy(&half, src, 2); src += 2;
                                        (*output)[j] = ggml_fp16_to_fp32(half);
                                    }
                                }
                            }
                            uploads.push_back(upload);
                            upload_payloads.emplace_back(data.begin() + sizeof(upload), data.end());
                    };
                    auto apply_attention = [&](const std::vector<uint8_t> & data) {
                            rkva_range_attn attn {}; std::memcpy(&attn, data.data(), sizeof(attn));
                            assert(attn.wire == RKVA_WIRE_F32 && attn.flags == 0);
                            const size_t dh = geometry.n_head * geometry.head_dim;
                            assert(data.size() == sizeof(attn) + req.n_tokens * (4 + dh * 4));
                            const auto * pos = (const int32_t *) (data.data() + sizeof(attn));
                            const auto * q = (const float *) (data.data() + sizeof(attn) + req.n_tokens * 4);
                            std::vector<float> result(req.n_tokens * (dh + geometry.n_head));
                            for (uint32_t t = 0; t < req.n_tokens; ++t) {
                                assert(pos[t] >= (int32_t) range.origin);
                                const uint32_t live = std::min(rows, (uint32_t) pos[t] - range.origin + 1);
                                for (uint32_t h = 0; h < geometry.n_head; ++h) {
                                    const uint32_t kh = h / (geometry.n_head / geometry.n_head_kv);
                                    std::vector<double> scores(live);
                                    for (uint32_t j = 0; j < live; ++j) {
                                        double dot = 0;
                                        for (uint32_t d = 0; d < geometry.head_dim; ++d)
                                            dot += double(q[(t * geometry.n_head + h) * geometry.head_dim + d]) * double(keys[(j * geometry.n_head_kv + kh) * geometry.head_dim + d]);
                                        scores[j] = dot * geometry.kq_scale;
                                    }
                                    const double m = *std::max_element(scores.begin(), scores.end());
                                    double denom = 0; for (const double score : scores) denom += std::exp(score - m);
                                    result[req.n_tokens * dh + t * geometry.n_head + h] = (float) (m + std::log(denom));
                                    for (uint32_t d = 0; d < geometry.head_dim; ++d) {
                                        double sum = 0;
                                        for (uint32_t j = 0; j < live; ++j) sum += std::exp(scores[j] - m) * values[(j * geometry.n_head_kv + kh) * geometry.head_dim + d];
                                        result[t * dh + h * geometry.head_dim + d] = (float) (sum / denom);
                                    }
                                }
                            }
                            response.resize(result.size() * 4); std::memcpy(response.data(), result.data(), response.size());
                            ++attention_calls;
                    };
                    packet_ops.push_back(req.op);
                    switch (req.op) {
                        case RKVA_HELLO: {
                            assert(payload.size() == sizeof(geometry));
                            std::memcpy(&geometry, payload.data(), sizeof(geometry));
                            rkva_hello_ack ack {};
                            ack.supported_ops = (1u << RKVA_ATTN_DECODE) | (1u << RKVA_ATTN_PREFILL);
                            if (range_ops) ack.supported_ops |= (1u << RKVA_REWIND) | (1u << RKVA_RANGE_CONFIG) | (1u << RKVA_RANGE_UPLOAD) | (1u << RKVA_RANGE_ATTN);
                            if (range_ops && fused_range) ack.supported_ops |= (1u << RKVA_RANGE_UPLOAD_ATTN);
                            if (profile_support & 2) ack.supported_ops |= (1u << RKVA_RANGE_PROFILE);
                            if (native_kvarn) ack.supported_ops |= (1u << RKVA_KVARN_RANGE_CONFIG) | (1u << RKVA_KVARN_RANGE_UPLOAD);
                            if ((geometry.proto_flags & RKVA_PROTO_NATIVE_REWIND_V2) && (native_rewind_support & 2)) ack.supported_ops |= (1u << RKVA_KVARN_RANGE_REWIND);
                            ack.chosen_wire = RKVA_WIRE_F16;
                            ack.reserved[0] = (native_q4 ? RKVA_CAP_RANGE_Q4_0 : 0) | (native_q8 ? RKVA_CAP_RANGE_Q8_0 : 0) |
                                              (fused_range ? RKVA_CAP_RANGE_UPLOAD_ATTN : 0);
                            if (native_kvarn) ack.reserved[0] |= RKVA_CAP_RANGE_KVARN4;
                            if ((geometry.proto_flags & RKVA_PROTO_NATIVE_REWIND_V2) && (native_rewind_support & 1)) ack.reserved[0] |= RKVA_CAP_KVARN_REWIND;
                            if ((geometry.proto_flags & RKVA_PROTO_NATIVE_SERIAL_PREFIX_V3) && (native_rewind_support & 4)) ack.reserved[0] |= RKVA_CAP_KVARN_SERIAL_PREFIX;
                            if ((geometry.proto_flags & RKVA_PROTO_NATIVE_SERIAL_PREFIX_V3) && (native_rewind_support & 8)) {
                                ack.reserved[0] |= RKVA_CAP_KVARN_BULK_ATTN;
                                ack.supported_ops |= (1u << RKVA_KVARN_RANGE_BULK_ATTN);
                            }
                            if ((geometry.proto_flags & RKVA_PROTO_NATIVE_SERIAL_PREFIX_V3) && (native_rewind_support & 16)) {
                                ack.reserved[0] |= RKVA_CAP_KVARN_BULK_F16;
                            }
                            if (profile_support & 1) ack.reserved[0] |= RKVA_CAP_RANGE_PROFILE;
                            response.resize(sizeof(ack)); std::memcpy(response.data(), &ack, sizeof(ack));
                        } break;
                        case RKVA_CREATE_SESSION: {
                            rkva_session_req session {};
                            assert(payload.size() == sizeof(session));
                            std::memcpy(&session, payload.data(), sizeof(session));
                            session_flags.push_back(session.flags);
                        } break;
                        case RKVA_KVARN_RANGE_CONFIG:
                            assert(payload.size() == sizeof(native_descriptor));
                            std::memcpy(&native_descriptor, payload.data(), sizeof(native_descriptor));
                            native_counts.assign(geometry.n_layer_remote, 0);
                            native_ready.assign(geometry.n_layer_remote, false);
                            ++native_generation;
                            assert(native_descriptor.layout_version == 1 || (geometry.proto_flags & RKVA_PROTO_NATIVE_REWIND_V2));
                            assert(native_descriptor.layout_version != 3 || (geometry.proto_flags & RKVA_PROTO_NATIVE_SERIAL_PREFIX_V3));
                            if (native_descriptor.layout_version >= 2) {
                                response.resize(sizeof(native_generation));
                                std::memcpy(response.data(), &native_generation, sizeof(native_generation));
                            }
                            break;
                        case RKVA_KVARN_RANGE_UPLOAD: {
                            rkva_kvarn_range_upload u {};
                            assert(payload.size() >= sizeof(u)); std::memcpy(&u, payload.data(), sizeof(u));
                            assert(req.layer_id < native_counts.size());
                            const uint64_t native_width = uint64_t(geometry.n_head_kv) * 256;
                            const uint64_t record_bytes = uint64_t(u.record_groups) * geometry.n_head_kv * 2 * 8960;
                            const uint64_t stage_rows = u.flags ? 384 : u.append_rows;
                            const uint64_t tail_rows = native_descriptor.precision_tail_tokens ?
                                (u.flags ? std::min(u.live_count, native_descriptor.precision_tail_tokens) : u.append_rows) : 0;
                            assert(payload.size() == sizeof(u) + 2 * (record_bytes + (stage_rows + tail_rows) * native_width * 2));
                            assert(u.flags ? (u.record_begin == 0) :
                                (u.append_begin == native_counts[req.layer_id] && u.append_rows == u.live_count - u.append_begin));
                            ++native_generation;
                            native_counts[req.layer_id] = u.live_count;
                            native_ready[req.layer_id] = true;
                        } break;
                        case RKVA_KVARN_RANGE_REWIND: {
                            assert(payload.size() >= sizeof(rkva_kvarn_range_rewind));
                            rkva_kvarn_range_rewind r {}; std::memcpy(&r, payload.data(), sizeof(r));
                            rkva_kvarn_rewind_plan plan;
                            assert(rkva_kvarn_plan_rewind(r, native_descriptor.origin, native_descriptor.capacity,
                                geometry.n_head_kv, native_descriptor.precision_tail_tokens, native_generation, native_counts,
                                native_ready, payload.data() + sizeof(r), payload.size() - sizeof(r), plan));
                            response_failure = fail_native_rewind;
                            if (response_failure == profile_failure::none) {
                                ++native_generation;
                                for (size_t il = 0; il < native_counts.size(); ++il) {
                                    native_counts[il] = plan.layers[il].new_live;
                                    native_ready[il] = native_counts[il] != 0;
                                }
                            }
                        } break;
                        case RKVA_RANGE_CONFIG:
                            assert(payload.size() == sizeof(range));
                            std::memcpy(&range, payload.data(), sizeof(range));
                            assert(range.reserved[0] == 0 || (!force_f16 &&
                                ((range.reserved[0] == RKVA_RANGE_STORAGE_Q4_0 && native_q4) ||
                                 (range.reserved[0] == RKVA_RANGE_STORAGE_Q8_0 && native_q8))));
                            assert(range.reserved[1] == 0);
                            // Avoid doubling a large decoded history allocation
                            // when the final chunk appends just a few rows.
                            keys.reserve(size_t(range.capacity) * geometry.n_head_kv * geometry.head_dim);
                            values.reserve(size_t(range.capacity) * geometry.n_head_kv * geometry.head_dim);
                            rows = 0; break;
                        case RKVA_RESET:
                            rows = 0; other_layers.clear();
                            if (native_kvarn) {
                                ++native_generation;
                                std::fill(native_counts.begin(), native_counts.end(), 0);
                                std::fill(native_ready.begin(), native_ready.end(), false);
                            }
                            break;
                        case RKVA_REWIND: {
                            assert(payload.size() == sizeof(rkva_trim_req));
                            rkva_trim_req trim {};
                            std::memcpy(&trim, payload.data(), sizeof(trim));
                            assert(trim.pos0 >= 0);
                            if (fail_trim) {
                                response_failure = profile_failure::server_error;
                                break;
                            }
                            const uint32_t keep = uint32_t(trim.pos0) <= range.origin ? 0 :
                                    std::min(uint32_t(trim.pos0) - range.origin, range.capacity);
                            this->rows = std::min(this->rows, keep);
                            for (auto & entry : other_layers) entry.second.rows = std::min(entry.second.rows, keep);
                        } break;
                        case RKVA_RANGE_UPLOAD:
                            apply_upload(payload); break;
                        case RKVA_RANGE_ATTN:
                            if (native_kvarn) {
                                // Legacy native shaders use their resident
                                // image. Every query must see its own prefix,
                                // including grouped host graphs crossing seals.
                                assert(native_descriptor.layout_version == 3 ?
                                        req.n_tokens <= 9 : req.n_tokens == 1);
                                const size_t dh = size_t(geometry.n_head) * geometry.head_dim;
                                assert(payload.size() == sizeof(rkva_range_attn) +
                                        size_t(req.n_tokens) * (sizeof(int32_t) + dh * sizeof(float)));
                                std::vector<float> result(size_t(req.n_tokens) * (dh + geometry.n_head), 0.0f);
                                const auto * positions = reinterpret_cast<const int32_t *>(payload.data() + sizeof(rkva_range_attn));
                                assert(uint32_t(positions[req.n_tokens - 1]) - native_descriptor.origin + 1 == native_counts[req.layer_id]);
                                for (uint32_t t = 0; t < req.n_tokens; ++t) {
                                    const uint32_t live = std::min(native_counts[req.layer_id], uint32_t(positions[t]) - native_descriptor.origin + 1);
                                    for (uint32_t h = 0; h < geometry.n_head; ++h) result[size_t(req.n_tokens) * dh + size_t(t) * geometry.n_head + h] = float(std::log(double(live)));
                                }
                                response.resize(result.size() * sizeof(float));
                                std::memcpy(response.data(), result.data(), response.size());
                                ++attention_calls;
                            } else apply_attention(payload);
                            break;
                        case RKVA_RANGE_PROFILE: {
                            assert(profile_support == 3 && payload.size() == sizeof(rkva_range_profile));
                            assert(req.n_tokens == 0 && req.layer_id == 0);
                            rkva_range_profile control{};
                            std::memcpy(&control, payload.data(), sizeof(control));
                            assert(control.reserved == 0 && control.q4_kernel_mode <= 3);
                            profile_controls.push_back(control);
                            if (control.flags & RKVA_PROFILE_REPORT_ONLY) {
                                assert(control.flags == RKVA_PROFILE_REPORT_ONLY && control.q4_kernel_mode == 0);
                                response_failure = fail_report;
                                fail_report = profile_failure::none;
                            } else {
                                assert((control.flags & ~(RKVA_PROFILE_ENABLE | RKVA_PROFILE_RESET)) == 0);
                                profile_enabled = (control.flags & RKVA_PROFILE_ENABLE) != 0;
                                kernel_mode = control.q4_kernel_mode;
                            }
                            std::string formats;
                            if (format_kernels) {
                                formats = ",\"optimized_kernels\":{";
                                for (const char * name : {"f16", "q4_0", "q8_0", "kvarn4"}) {
                                    if (formats.back() != '{') formats += ',';
                                    formats += std::string("\"") + name + "\":{\"available\":" +
                                        (format_kernels == 1 ? "true" : "false") + ",\"status\":\"fixture\"}";
                                }
                                formats += "},\"buckets\":{}";
                            }
                            const std::string json = "{\"engine\":{\"enabled\":" +
                                std::string(profile_enabled ? "true" : "false") +
                                ",\"q4_kernel_mode\":" + std::to_string(kernel_mode) + formats + "},\"transport\":{}}";
                            response.assign(json.begin(), json.end());
                        } break;
                        case RKVA_RANGE_UPLOAD_ATTN: {
                            assert(fused_range && payload.size() >= sizeof(rkva_range_upload_attn));
                            rkva_range_upload_attn combined{};
                            std::memcpy(&combined, payload.data(), sizeof(combined));
                            const size_t width = geometry.n_head_kv * geometry.head_dim;
                            const bool native_storage = range.reserved[0] != 0;
                            const auto storage_type = range.reserved[0] == RKVA_RANGE_STORAGE_Q8_0 ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_0;
                            const size_t row_bytes = native_storage ? ggml_row_size(storage_type, width) : width * 2;
                            const size_t kv_bytes = combined.upload.n_rows * row_bytes * 2;
                            assert(payload.size() >= sizeof(combined) + kv_bytes);
                            std::vector<uint8_t> upload_data(sizeof(combined.upload) + kv_bytes);
                            std::memcpy(upload_data.data(), &combined.upload, sizeof(combined.upload));
                            std::memcpy(upload_data.data() + sizeof(combined.upload), payload.data() + sizeof(combined), kv_bytes);
                            std::vector<uint8_t> query_data(sizeof(combined.attention) + payload.size() - sizeof(combined) - kv_bytes);
                            std::memcpy(query_data.data(), &combined.attention, sizeof(combined.attention));
                            std::memcpy(query_data.data() + sizeof(combined.attention), payload.data() + sizeof(combined) + kv_bytes,
                                        payload.size() - sizeof(combined) - kv_bytes);
                            apply_upload(upload_data);
                            apply_attention(query_data);
                            ++fused_calls;
                        } break;
                        case RKVA_KVARN_RANGE_BULK_ATTN: {
                            assert(native_kvarn && (native_rewind_support & 8));
                            rkva_kvarn_range_bulk header{};
                            assert(payload.size() >= sizeof(header));
                            std::memcpy(&header, payload.data(), sizeof(header));
                            assert(header.version == 1 && header.total_queries == req.n_tokens);
                            assert(header.segment_count > 0);
                            rkva_kvarn_range_bulk_segment seg{};
                            assert(payload.size() >= sizeof(header) + sizeof(seg));
                            std::memcpy(&seg, payload.data() + sizeof(header), sizeof(seg));
                            rkva_kvarn_range_upload u{};
                            assert(seg.upload_bytes >= sizeof(u));
                            std::memcpy(&u, payload.data() + sizeof(header) + sizeof(seg), sizeof(u));
                            native_counts[req.layer_id] = u.live_count;
                            native_ready[req.layer_id] = true;
                            const size_t attn_offset = sizeof(header) + sizeof(seg) + seg.upload_bytes;
                            assert(payload.size() >= attn_offset + sizeof(rkva_range_attn));
                            rkva_range_attn attn{};
                            std::memcpy(&attn, payload.data() + attn_offset, sizeof(attn));
                            const bool wire_f16 = (attn.wire == RKVA_WIRE_F16);
                            assert(attn.wire == RKVA_WIRE_F32 || (wire_f16 && (native_rewind_support & 16)));
                            const size_t dh = size_t(geometry.n_head) * geometry.head_dim;
                            const size_t out_elem_bytes = wire_f16 ? sizeof(uint16_t) : sizeof(float);
                            const size_t out_bytes = size_t(req.n_tokens) * dh * out_elem_bytes;
                            const size_t lse_bytes = size_t(req.n_tokens) * geometry.n_head * sizeof(float);
                            response.resize(out_bytes + lse_bytes, 0);
                            if (wire_f16) {
                                auto * dst_f16 = reinterpret_cast<uint16_t *>(response.data());
                                auto * dst_lse = reinterpret_cast<float *>(response.data() + out_bytes);
                                for (uint32_t t = 0; t < req.n_tokens; ++t) {
                                    for (size_t d = 0; d < dh; ++d) dst_f16[t * dh + d] = 0x0000u;
                                    for (uint32_t h = 0; h < geometry.n_head; ++h)
                                        dst_lse[t * geometry.n_head + h] = float(std::log(double(native_counts[req.layer_id] - req.n_tokens + t + 1)));
                                }
                            } else {
                                auto * dst = reinterpret_cast<float *>(response.data());
                                for (uint32_t t = 0; t < req.n_tokens; ++t) {
                                    for (size_t d = 0; d < dh; ++d) dst[t * dh + d] = 0.0f;
                                    for (uint32_t h = 0; h < geometry.n_head; ++h)
                                        dst[req.n_tokens * dh + t * geometry.n_head + h] = float(std::log(double(native_counts[req.layer_id] - req.n_tokens + t + 1)));
                                }
                            }
                            ++attention_calls;
                        } break;
                        default: assert(false);
                    }
                    rkva_response rsp {RKVA_MAGIC, (uint16_t) (old_version ? RKVA_VERSION_LEGACY : RKVA_VERSION), req.op, req.cycle_id,
                                       0, (uint32_t) response.size(), 0, 0, 0, 0};
                    if (response_failure == profile_failure::cycle) ++rsp.cycle_id;
                    if (response_failure == profile_failure::version) rsp.version = RKVA_VERSION_LEGACY;
                    if (response_failure == profile_failure::oversized) rsp.payload_bytes = (256u << 20) + 1;
                    if (response_failure == profile_failure::server_error) {
                        rsp.status = RKVA_ERR_STATE;
                        rsp.payload_bytes = 0;
                        response.clear();
                    }
                    if (!write_all(fd, &rsp, sizeof(rsp))) break;
                    if (response_failure == profile_failure::truncated) {
                        write_all(fd, response.data(), response.size() / 2);
                        break;
                    }
                    if (response_failure == profile_failure::oversized) break;
                    if (!response.empty() && !write_all(fd, response.data(), response.size())) break;
                }
            }
            close(fd); client = -1;
        });
    }
    ~loopback() {
        const int fd = client.load(); if (fd >= 0) shutdown(fd, SHUT_RDWR);
        shutdown(listener, SHUT_RDWR); close(listener); worker.join();
    }
};

static ggml_backend_t connect(loopback & server, uint32_t n_layers = 1) {
    auto * backend = ggml_backend_remote_attn_init("127.0.0.1", server.port);
    ggml_remote_attn_geometry geometry {};
    geometry.n_layer_remote = n_layers; geometry.n_head = 2; geometry.n_head_kv = 1;
    geometry.head_dim = 32; geometry.max_ctx = 8; geometry.kq_scale = 0.125f;
    assert(ggml_backend_remote_attn_set_geometry(backend, &geometry));
    return backend;
}

struct storage_env {
    bool had_previous;
    std::string previous;
    const char * key;
    explicit storage_env(const char * value, const char * name = "GGML_RKVA_RANGE_STORAGE") : key(name) {
        const char * old = std::getenv(key);
        had_previous = old != nullptr;
        if (old) previous = old;
        if (value) setenv(key, value, 1);
        else unsetenv(key);
    }
    ~storage_env() {
        if (had_previous) setenv(key, previous.c_str(), 1);
        else unsetenv(key);
    }
};

static void test_snapshot_digest() {
    // Independent hashlib/NIST vectors cover padding and streaming boundaries.
    const struct { size_t size; const char * expected; } cases[] = {
        {0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {3, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {55, "2900465fcb533e05a158fd2b3be0e5e3b03740d83060aa3580e0d98a96bf2384"},
        {56, "31454ff48ef36af2f08fd511bdc37d9d5855ac23e992e5ff5445cb6b7674a674"},
        {63, "5f6401b96532c36de4e65beec0409b69b1d181864c8009b7a04f43e5d56350d1"},
        {64, "94eb5de4943613fd048dc93393ab06877405faa39c11f53e9386083339833e7e"},
        {65, "fc518669b6eb4b4dd91827ecacef86689c725bd5bab888fd3b26dbb196eec954"},
        {1024, "ffbad8f947474cfdd5b2bb22d7e0bf5ee8ba2b7af859d0c2bb28622db6a4be47"},
        {1000000, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"},
    };
    for (const auto & sample : cases) {
        std::vector<uint8_t> data(sample.size);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = sample.size == 3 ? uint8_t("abc"[i]) :
                      sample.size == 1000000 ? uint8_t('a') : uint8_t(i * 37 + 11);
        }
        const auto direct = rkva_snapshot::digest(data.data(), data.size());
        assert(rkva_snapshot::hex(direct.data()) == sample.expected);
        for (size_t chunk : {size_t(1), size_t(7), size_t(55), size_t(64), size_t(8192)}) {
            rkva_snapshot::Sha256 streamed;
            for (size_t i = 0; i < data.size(); i += chunk) {
                streamed.update(data.data() + i, std::min(chunk, data.size() - i));
            }
            const auto digest = streamed.finish();
            assert(rkva_snapshot::hex(digest.data()) == sample.expected);
        }
    }
}

static void test_prefill_query_chunking() {
    for (const auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_F16}) {
        for (const uint32_t queries : {1u, 9u, 20u}) {
            std::vector<float> reference;
            for (bool chunked : {false, true}) {
                storage_env storage(ggml_type_name(type));
                storage_env chunk(chunked ? "8" : "0", "GGML_RKVA_PREFILL_QUERY_CHUNK");
                storage_env fuse("1", "GGML_RKVA_RANGE_FUSE");
                storage_env kernel("reference", "GGML_RKVA_Q4_KERNEL");
                storage_env profiling("0", "GGML_RKVA_PROFILE");
                constexpr uint32_t origin = 256, capacity = 32, dim = 32, heads = 2;
                loopback server(false, true, type == GGML_TYPE_Q4_0, false,
                                type == GGML_TYPE_Q8_0, true);
                auto * backend = ggml_backend_remote_attn_init("127.0.0.1", server.port);
                ggml_remote_attn_geometry geometry {};
                geometry.n_layer_remote = geometry.n_head_kv = 1;
                geometry.n_head = heads; geometry.head_dim = dim;
                geometry.max_ctx = capacity; geometry.kq_scale = 0.125f;
                assert(ggml_backend_remote_attn_set_geometry(backend, &geometry));
                assert(ggml_backend_remote_attn_connect(backend));
                assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, capacity));
                ggml_backend_remote_attn_set_active_session(backend, 0);
                assert(ggml_backend_remote_attn_configure_range(backend, origin, capacity, type));
                ggml_remote_attn_set_active(backend);
                auto * ctx = ggml_init({1u << 20, nullptr, true});
                auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, heads, queries);
                auto * k = ggml_new_tensor_2d(ctx, type, dim, capacity);
                auto * v = ggml_new_tensor_2d(ctx, type, dim, capacity);
                auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, capacity);
                auto * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, queries);
                auto * node = ggml_remote_attn_range(ctx, q, k, v, ids, pos, 0, origin, capacity, 0.125f);
                auto * graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, node);
                auto * cpu = ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(cpu, 1);
                auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu); assert(buffer);
                std::vector<float> query(queries * heads * dim), keys(capacity * dim), values(capacity * dim);
                for (size_t i = 0; i < query.size(); ++i) { query[i] = float(int(i % 17) - 8) * 0.07f; }
                for (size_t i = 0; i < keys.size(); ++i) {
                    keys[i] = float(int(i % 23) - 11) * 0.09f;
                    values[i] = float(int(i % 29) - 14) * 0.11f;
                }
                std::vector<uint8_t> packed(capacity * ggml_row_size(type, dim));
                ggml_quantize_chunk(type, keys.data(), packed.data(), 0, capacity, dim, nullptr);
                ggml_backend_tensor_set(k, packed.data(), 0, packed.size());
                ggml_quantize_chunk(type, values.data(), packed.data(), 0, capacity, dim, nullptr);
                ggml_backend_tensor_set(v, packed.data(), 0, packed.size());
                ggml_backend_tensor_set(q, query.data(), 0, query.size() * sizeof(float));
                std::vector<int32_t> cells(capacity, -1), positions(queries);
                for (uint32_t i = 0; i < queries; ++i) { cells[i] = positions[i] = origin + i; }
                ggml_backend_tensor_set(ids, cells.data(), 0, cells.size() * sizeof(int32_t));
                ggml_backend_tensor_set(pos, positions.data(), 0, positions.size() * sizeof(int32_t));
                assert(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
                std::vector<float> output(queries * (heads * dim + heads));
                ggml_backend_tensor_get(node, output.data(), 0, output.size() * sizeof(float));
                if (!chunked) { reference = output; } else { assert(output == reference); }
                {
                    std::lock_guard<std::mutex> lock(server.mutex);
                    assert(server.uploads.size() == 1 && server.uploads[0].start_row == 0 && server.uploads[0].n_rows == queries);
                    const bool tiled = chunked && queries > 9;
                    assert(server.attention_calls == (tiled ? 3 : 1));
                    assert(server.fused_calls == (tiled ? 0 : 1));
                    assert(server.rows == queries);
                }
                ggml_remote_attn_set_active(nullptr);
                ggml_backend_buffer_free(buffer); ggml_free(ctx); ggml_backend_free(cpu); ggml_backend_free(backend);
            }
        }
    }
    storage_env invalid("16", "GGML_RKVA_PREFILL_QUERY_CHUNK");
    assert(!ggml_backend_remote_attn_init("127.0.0.1", 1));
}

static void test_range_chunking() {
    // Cross the real 16MiB upload limit with the smallest head width. F16
    // transport limits the decoded fake-server history to about 32MiB, keeping
    // total peak memory near 100MiB instead of a large model-like fixture.
    storage_env storage("f16");
    storage_env fuse("1", "GGML_RKVA_RANGE_FUSE");
    storage_env kernel("reference", "GGML_RKVA_Q4_KERNEL");
    storage_env profiling("0", "GGML_RKVA_PROFILE");
    constexpr uint32_t first_rows = (16u << 20) / (32 * 4);
    constexpr uint32_t rows = first_rows + 3;
    constexpr uint32_t origin = 256;
    loopback server(false, true, true, true, false, true);
    auto * backend = ggml_backend_remote_attn_init("127.0.0.1", server.port);
    ggml_remote_attn_geometry geometry {};
    geometry.n_layer_remote = geometry.n_head = geometry.n_head_kv = 1;
    geometry.head_dim = 32; geometry.max_ctx = rows; geometry.kq_scale = 0.125f;
    assert(ggml_backend_remote_attn_set_geometry(backend, &geometry));
    assert(ggml_backend_remote_attn_connect(backend));
    assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, rows));
    ggml_backend_remote_attn_set_active_session(backend, 0);
    assert(ggml_backend_remote_attn_configure_range(backend, origin, rows, GGML_TYPE_Q4_0));
    ggml_remote_attn_set_active(backend);
    auto * ctx = ggml_init({1u << 20, nullptr, true});
    auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 1, 1);
    auto * k = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 32, rows);
    auto * v = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 32, rows);
    auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, rows);
    auto * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    auto * node = ggml_remote_attn_range(ctx, q, k, v, ids, pos, 0, origin, rows, 0.125f);
    auto * graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, node);
    auto * cpu = ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(cpu, 1);
    auto * buffers = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    assert(buffers);
    const float query[32] = {};
    ggml_backend_tensor_set(q, query, 0, sizeof(query));
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_0, 32);
    std::vector<uint8_t> packed(rows * row_bytes, 0);
    ggml_backend_tensor_set(k, packed.data(), 0, packed.size());
    float value[32]; std::fill(value, value + 32, 1.0f);
    std::vector<uint8_t> packed_row(row_bytes);
    ggml_quantize_chunk(GGML_TYPE_Q4_0, value, packed_row.data(), 0, 1, 32, nullptr);
    for (uint32_t row = 0; row < rows; ++row) std::memcpy(packed.data() + row * row_bytes, packed_row.data(), row_bytes);
    ggml_backend_tensor_set(v, packed.data(), 0, packed.size());
    std::vector<int32_t> cells(rows);
    for (uint32_t row = 0; row < rows; ++row) cells[row] = origin + row;
    const int32_t position = origin + rows - 1;
    ggml_backend_tensor_set(ids, cells.data(), 0, cells.size() * sizeof(int32_t));
    ggml_backend_tensor_set(pos, &position, 0, sizeof(position));
    assert(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
    float result[33]; ggml_backend_tensor_get(node, result, 0, sizeof(result));
    for (int d = 0; d < 32; ++d) assert(std::abs(result[d] - 1.0f) < 1e-6f);
    assert(std::abs(result[32] - std::log(double(rows))) < 1e-6);
    {
        std::lock_guard<std::mutex> lock(server.mutex);
        const std::vector<uint16_t> expected_ops = {
            RKVA_HELLO, RKVA_CREATE_SESSION, RKVA_RANGE_CONFIG, RKVA_RANGE_UPLOAD, RKVA_RANGE_UPLOAD_ATTN,
        };
        assert(server.packet_ops == expected_ops);
        assert(server.uploads.size() == 2 && server.attention_calls == 1 && server.fused_calls == 1);
        assert(server.uploads[0].start_row == 0 && server.uploads[0].n_rows == first_rows);
        assert(server.uploads[1].start_row == first_rows && server.uploads[1].n_rows == 3);
        assert(server.rows == rows);
        assert(server.upload_payloads[0].size() == (16u << 20));
        assert(server.upload_payloads[1].size() == 3 * 32 * 4);
    }
    ggml_remote_attn_set_active(nullptr);
    ggml_backend_buffer_free(buffers); ggml_free(ctx); ggml_backend_free(cpu); ggml_backend_free(backend);
}

static void test_range_profile() {
    // Exercise the actual client negotiation/control/report path without a GPU.
    const struct { const char * name; uint32_t mode; } modes[] = {
        {"reference", 0}, {"0", 0}, {"optimized", 1}, {"balanced", 1}, {"1", 1},
        {"fine", 2}, {"2", 2}, {"coarse", 3}, {"3", 3},
    };
    storage_env automatic("auto");
    storage_env fuse("auto", "GGML_RKVA_RANGE_FUSE");
    for (const auto & mode : modes) {
        for (const bool profiling : {false, true}) {
            storage_env kernel(mode.name, "GGML_RKVA_Q4_KERNEL");
            storage_env enabled(profiling ? "1" : "0", "GGML_RKVA_PROFILE");
            loopback server(false, true, true, false, false, false, 3);
            auto * backend = connect(server);
            assert(ggml_backend_remote_attn_connect(backend));
            assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 8));
            ggml_backend_remote_attn_set_active_session(backend, 0);
            assert(ggml_backend_remote_attn_configure_range(backend, 256, 8, GGML_TYPE_Q4_0));
            const bool controlled = profiling || mode.mode != 0;
            {
                std::lock_guard<std::mutex> lock(server.mutex);
                assert(server.profile_controls.size() == (controlled ? 1u : 0u));
                assert(server.kernel_mode == mode.mode && server.profile_enabled == profiling);
                if (controlled) {
                    const auto & control = server.profile_controls.front();
                    assert(control.flags == uint32_t(RKVA_PROFILE_RESET | (profiling ? RKVA_PROFILE_ENABLE : 0)));
                    assert(control.q4_kernel_mode == mode.mode);
                }
            }
            const std::string stats = ggml_backend_remote_attn_stats_json(backend);
            assert(stats.find("\"range_kernel_mode\":" + std::to_string(mode.mode)) != std::string::npos);
            assert(!ggml_backend_remote_attn_failed(backend));
            {
                std::lock_guard<std::mutex> lock(server.mutex);
                assert(server.profile_controls.size() == (controlled ? 2u : 0u));
                if (controlled) {
                    const auto & report = server.profile_controls.back();
                    assert(report.flags == RKVA_PROFILE_REPORT_ONLY && report.q4_kernel_mode == 0);
                }
                // A report must never switch off profiling or change the kernel.
                assert(server.kernel_mode == mode.mode && server.profile_enabled == profiling);
            }
            ggml_backend_free(backend);
        }
    }
    // Reconfiguring the same connection must explicitly turn off the previous
    // engine state, including when the overflow storage format changes.
    for (const char * next_storage : {"q4_0", "q8_0", "f16"}) {
        loopback server(false, true, true, false, true, false, 3);
        auto * backend = connect(server);
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 8));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        {
            storage_env storage("q4_0");
            storage_env kernel("optimized", "GGML_RKVA_Q4_KERNEL");
            storage_env enabled("1", "GGML_RKVA_PROFILE");
            assert(ggml_backend_remote_attn_configure_range(backend, 256, 8, GGML_TYPE_Q4_0));
        }
        {
            storage_env storage(next_storage);
            storage_env kernel("reference", "GGML_RKVA_Q4_KERNEL");
            storage_env enabled("0", "GGML_RKVA_PROFILE");
            const auto next_type = std::strcmp(next_storage, "q8_0") == 0 ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_0;
            assert(ggml_backend_remote_attn_configure_range(backend, 256, 8, next_type));
            size_t packets_after_disable;
            {
                std::lock_guard<std::mutex> lock(server.mutex);
                assert(!server.profile_enabled && server.kernel_mode == 0);
                assert(server.profile_controls.size() == 2);
                assert(server.profile_controls.front().flags == (RKVA_PROFILE_ENABLE | RKVA_PROFILE_RESET));
                assert(server.profile_controls.front().q4_kernel_mode == 1);
                assert(server.profile_controls.back().flags == RKVA_PROFILE_RESET);
                assert(server.profile_controls.back().q4_kernel_mode == 0);
                packets_after_disable = server.packet_ops.size();
            }
            const std::string stats = ggml_backend_remote_attn_stats_json(backend);
            assert(stats.find("\"range_kernel_mode\":0") != std::string::npos);
            assert(stats.find("\"range_cache_type\":\"" + std::string(next_storage) + "\"") != std::string::npos);
            ggml_backend_remote_attn_stats_json(backend);
            {
                std::lock_guard<std::mutex> lock(server.mutex);
                assert(server.packet_ops.size() == packets_after_disable);
                assert(!server.profile_enabled && server.kernel_mode == 0);
            }
        }
        ggml_backend_free(backend);
    }
    // Profiling itself also works with Q8 and F16; only optimized Q4 modes
    // require native Q4 records.
    for (const bool f16 : {false, true}) {
        storage_env storage(f16 ? "f16" : "auto");
        storage_env kernel("reference", "GGML_RKVA_Q4_KERNEL");
        storage_env enabled("1", "GGML_RKVA_PROFILE");
        loopback server(false, true, false, f16, true, false, 3);
        auto * backend = connect(server);
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 8));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(ggml_backend_remote_attn_configure_range(backend, 256, 8, GGML_TYPE_Q8_0));
        ggml_backend_remote_attn_stats_json(backend);
        {
            std::lock_guard<std::mutex> lock(server.mutex);
            assert(server.profile_enabled && server.kernel_mode == 0 && server.profile_controls.size() == 2);
        }
        ggml_backend_free(backend);
    }
    const auto rejects_before_config = [](const char * kernel_name, const char * profiling,
            uint32_t support, ggml_type source_type = GGML_TYPE_Q4_0, bool f16 = false) {
        storage_env storage(f16 ? "f16" : "auto");
        storage_env kernel(kernel_name, "GGML_RKVA_Q4_KERNEL");
        storage_env enabled(profiling, "GGML_RKVA_PROFILE");
        loopback server(false, true, true, f16, true, false, support);
        auto * backend = connect(server);
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 8));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(!ggml_backend_remote_attn_configure_range(backend, 256, 8, source_type));
        {
            std::lock_guard<std::mutex> lock(server.mutex);
            assert(server.range.capacity == 0 && server.profile_controls.empty());
        }
        ggml_backend_free(backend);
    };
    for (const uint32_t support : {0u, 1u, 2u}) {
        rejects_before_config("reference", "1", support);
        rejects_before_config("optimized", "0", support);
    }
    for (const char * invalid : {"4", "-1", "invalid", ""}) rejects_before_config(invalid, "0", 3);
    for (const char * invalid : {"2", "true", "invalid", ""}) rejects_before_config("reference", invalid, 3);
    rejects_before_config("optimized", "0", 3, GGML_TYPE_Q8_0);
    rejects_before_config("optimized", "0", 3, GGML_TYPE_Q4_0, true);

    // Optional diagnostic failures must not permit reuse of a broken stream.
    // A valid server error with an empty payload leaves the framing intact.
    for (const auto failure : {loopback::profile_failure::cycle, loopback::profile_failure::version,
            loopback::profile_failure::oversized, loopback::profile_failure::truncated,
            loopback::profile_failure::server_error}) {
        storage_env kernel("reference", "GGML_RKVA_Q4_KERNEL");
        storage_env enabled("1", "GGML_RKVA_PROFILE");
        loopback server(false, true, true, false, false, false, 3);
        auto * backend = connect(server);
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 8));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(ggml_backend_remote_attn_configure_range(backend, 256, 8, GGML_TYPE_Q4_0));
        {
            std::lock_guard<std::mutex> lock(server.mutex);
            server.fail_report = failure;
        }
        ggml_backend_remote_attn_stats_json(backend);
        const bool fatal = failure != loopback::profile_failure::server_error;
        assert(ggml_backend_remote_attn_failed(backend) == fatal);
        assert(ggml_backend_remote_attn_connected(backend) != fatal);
        size_t packets_after_report;
        {
            std::lock_guard<std::mutex> lock(server.mutex);
            packets_after_report = server.packet_ops.size();
            assert(server.profile_controls.size() == 2);
        }
        assert(ggml_backend_remote_attn_reset(backend, 0) != fatal);
        ggml_backend_remote_attn_stats_json(backend);
        {
            std::lock_guard<std::mutex> lock(server.mutex);
            assert(server.packet_ops.size() == packets_after_report + (fatal ? 0u : 2u));
            assert(server.profile_controls.size() == (fatal ? 2u : 3u));
        }
        ggml_backend_free(backend);
    }
}

static void test_range_format_kernel() {
    storage_env old(nullptr, "GGML_RKVA_Q4_KERNEL");
    storage_env generic("balanced", "GGML_RKVA_RANGE_KERNEL");
    storage_env profiling("0", "GGML_RKVA_PROFILE");
    storage_env fuse("auto", "GGML_RKVA_RANGE_FUSE");
    for (const char * format : {"q4_0", "q8_0", "f16"}) {
        storage_env wire(format);
        for (uint32_t available : {0u, 1u, 2u}) {
            loopback server(false, true, true, false, true, true, 3, available);
            auto * backend = connect(server);
            assert(ggml_backend_remote_attn_connect(backend));
            assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 8));
            ggml_backend_remote_attn_set_active_session(backend, 0);
            const auto source = !std::strcmp(format, "q8_0") ? GGML_TYPE_Q8_0 :
                !std::strcmp(format, "f16") ? GGML_TYPE_F16 : GGML_TYPE_Q4_0;
            assert(ggml_backend_remote_attn_configure_range(backend, 256, 8, source) == (available == 1));
            assert(ggml_backend_remote_attn_failed(backend) == (available != 1));
            {
                std::lock_guard<std::mutex> lock(server.mutex);
                assert(server.kernel_mode == 1 && server.profile_controls.size() == 1);
            }
            ggml_backend_free(backend);
        }
    }
    // Conflicting knobs must not silently select one policy.
    storage_env wire("q4_0");
    storage_env conflict("fine", "GGML_RKVA_Q4_KERNEL");
    loopback server(false, true, true, false, false, false, 3, 1);
    auto * backend = connect(server);
    assert(ggml_backend_remote_attn_connect(backend));
    assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 8));
    ggml_backend_remote_attn_set_active_session(backend, 0);
    assert(!ggml_backend_remote_attn_configure_range(backend, 256, 8, GGML_TYPE_Q4_0));
    { std::lock_guard<std::mutex> lock(server.mutex); assert(server.range.capacity == 0); }
    ggml_backend_free(backend);
}

static void test_old_native_client_ack_compatibility() {
    loopback server(false, true, false, false, false, false, 0, 0, true, 3);
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    assert(fd >= 0);
    sockaddr_in address {}; address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK); address.sin_port = htons(server.port);
    assert(::connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0);
    uint64_t cycle = 0;
    auto transact = [&](rkva_op op, const void * payload, uint32_t bytes, uint32_t expected) {
        const rkva_request req {RKVA_MAGIC, RKVA_VERSION, uint16_t(op), ++cycle, 0, 0, 0, bytes};
        assert(loopback::write_all(fd, &req, sizeof(req)));
        if (bytes) assert(loopback::write_all(fd, payload, bytes));
        rkva_response response {};
        assert(loopback::read_all(fd, &response, sizeof(response)));
        assert(response.magic == RKVA_MAGIC && response.version == RKVA_VERSION &&
            response.op == uint16_t(op) && response.cycle_id == cycle && response.status == RKVA_OK);
        assert(response.payload_bytes == expected);
        std::vector<uint8_t> reply(expected);
        if (expected) assert(loopback::read_all(fd, reply.data(), reply.size()));
        return reply;
    };
    rkva_hello hello {};
    hello.proto_flags = 3; // original client: no descriptor-v2/rewind advertisement
    hello.n_layer_remote = 1; hello.n_head = 2; hello.n_head_kv = 1;
    hello.head_dim = 256; hello.max_ctx = 256; hello.cache_bits_k = hello.cache_bits_v = 16;
    hello.group_tokens = 128; hello.sinkhorn_iters = 3; hello.tail_tokens = 128; hello.tail_groups = 2;
    hello.kq_scale = 0.0625f;
    const auto ack_bytes = transact(RKVA_HELLO, &hello, sizeof(hello), sizeof(rkva_hello_ack));
    rkva_hello_ack ack {}; std::memcpy(&ack, ack_bytes.data(), sizeof(ack));
    assert(!(ack.reserved[0] & RKVA_CAP_KVARN_REWIND));
    assert(!(ack.supported_ops & (1u << RKVA_KVARN_RANGE_REWIND)));
    const rkva_session_req session {0, 0, 256, RKVA_SESSION_RANGE};
    transact(RKVA_CREATE_SESSION, &session, sizeof(session), 0);
    const rkva_kvarn_range_config descriptor {1024, 256, RKVA_RANGE_STORAGE_KVARN4, 1,
        128, 128, 256, 2, 4, 4, 8960, 3, 2, 128, 128, 1, 1, 1};
    transact(RKVA_KVARN_RANGE_CONFIG, &descriptor, sizeof(descriptor), 0);
    for (uint32_t live : {1u, 2u}) {
        const rkva_kvarn_range_upload upload {live, 0, 0, live == 1 ? 1u : 0u,
            live == 1 ? 0u : 1u, live == 1 ? 0u : 1u, {0, 0}};
        const size_t half_rows = (live == 1 ? 384 : 1) + 1;
        std::vector<uint8_t> payload(sizeof(upload) + 2 * half_rows * 256 * 2, 0);
        std::memcpy(payload.data(), &upload, sizeof(upload));
        transact(RKVA_KVARN_RANGE_UPLOAD, payload.data(), uint32_t(payload.size()), 0);
    }
    shutdown(fd, SHUT_RDWR); close(fd);
}

static void test_native_rewind() {
    storage_env native("kvarn4");
    storage_env experimental("1", "GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL");
    storage_env fusion("0", "GGML_RKVA_RANGE_FUSE");
    ggml_backend_load_all();
    for (uint32_t tail : {0u, 128u}) {
    storage_env precision(tail ? "128" : "0", "GGML_RKVA_NATIVE_TAIL_TOKENS");
    for (uint32_t support : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u}) {
      for (auto failure : {loopback::profile_failure::none, loopback::profile_failure::server_error,
                           loopback::profile_failure::cycle}) {
        if ((support & 3) != 3 && failure != loopback::profile_failure::none) continue;
        loopback server(false, true, false, false, false, false, 0, 0, true, support);
        auto * backend = ggml_backend_remote_attn_init("127.0.0.1", server.port);
        ggml_remote_attn_geometry geometry {};
        geometry.n_layer_remote = 2; geometry.n_head = 2; geometry.n_head_kv = 1;
        geometry.head_dim = 256; geometry.max_ctx = 512; geometry.kq_scale = 0.0625f;
        geometry.group_tokens = 128; geometry.sinkhorn_iters = 3;
        assert(ggml_backend_remote_attn_set_geometry(backend, &geometry));
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 512));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(ggml_backend_remote_attn_configure_range(backend, 1024, 512, GGML_TYPE_Q4_0));
        assert(ggml_backend_remote_attn_native_kvarn_can_rewind(backend) == ((support & 3) == 3));
        {
            std::lock_guard<std::mutex> lock(server.mutex);
            assert(server.native_descriptor.layout_version == (support == 7 ? 3u : support == 3 ? 2u : 1u));
        }
        const std::string stats = ggml_backend_remote_attn_stats_json(backend);
        assert(stats.find("\"native_precision_tail_tokens\":" + std::to_string(tail)) != std::string::npos);
        ggml_remote_attn_set_active(backend);
        auto * ctx = ggml_init({1u << 20, nullptr, true});
        auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 2, 2);
        auto * k = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 512);
        auto * v = ggml_dup_tensor(ctx, k);
        auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
        auto * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
        auto * original_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, 2);
        auto * original_v = ggml_dup_tensor(ctx, original_k);
        auto * node = ggml_remote_attn_range(ctx, q, k, v, ids, pos, 0, 1024, 2, geometry.kq_scale);
        ggml_remote_attn_range_set_native_sources(node, original_k, original_v);
        auto * cpu = ggml_backend_cpu_init();
        auto * buffers = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        assert(buffers); ggml_backend_buffer_clear(buffers, 0);
        auto run = [&](uint32_t live, uint32_t layer) {
            std::vector<int32_t> cells(512, -1);
            for (uint32_t i = 0; i < live; ++i) cells[i] = 1024 + i;
            const int32_t positions[2] = {int32_t(1024 + live - 2), int32_t(1024 + live - 1)};
            ggml_backend_tensor_set(ids, cells.data(), 0, cells.size() * sizeof(int32_t));
            ggml_backend_tensor_set(pos, positions, 0, sizeof(positions));
            node->op_params[GGML_REMOTE_ATTN_PARAM_N_LIVE] = int32_t(live);
            node->op_params[GGML_REMOTE_ATTN_PARAM_LAYER_ID] = int32_t(layer);
            assert(ggml_remote_attn_exec(node));
            const float * output = static_cast<const float *>(node->data);
            for (uint32_t t = 0; t < 2; ++t) {
                for (size_t d = 0; d < 512; ++d) assert(output[t * 514 + d] == 0.0f);
                for (uint32_t h = 0; h < 2; ++h)
                    assert(std::abs(double(output[t * 514 + 512 + h]) - std::log(double(live - 1 + t))) < 1e-6);
            }
        };
        for (uint32_t live = 2; live <= 130; live += 2) { run(live, 0); run(live, 1); }
        if ((support & 3) != 3) {
            assert(!ggml_backend_remote_attn_trim(backend, 0, 1024 + 128));
            assert(!ggml_backend_remote_attn_failed(backend));
        } else {
            assert(ggml_backend_remote_attn_native_kvarn_can_restore(backend, 1024 + 128));
            assert(!ggml_backend_remote_attn_native_kvarn_can_restore(backend, 1024 + 131));
            assert(ggml_backend_remote_attn_trim(backend, 0, 1024 + 128));
            assert(ggml_backend_remote_attn_trim(backend, 0, 1024 + 127));
            run(129, 0); run(129, 1); // rejected suffix overwritten across seal boundary
            if (failure == loopback::profile_failure::none) {
                assert(ggml_backend_remote_attn_trim(backend, 0, 1024 + 127));
                run(129, 0); run(129, 1); // repeated earlier anchor
                for (uint32_t live = 131; live <= 385; live += 2) { run(live, 0); run(live, 1); }
                assert(ggml_backend_remote_attn_trim(backend, 0, 1024 + 257));
                // A second 128-token rewind needs originals that have expired,
                // despite the metadata distance itself being legal.
                assert(!ggml_backend_remote_attn_native_kvarn_can_restore(backend, 1024 + 129));
                assert(!ggml_backend_remote_attn_trim(backend, 0, 1024 + 129));
                assert(!ggml_backend_remote_attn_failed(backend));
                run(259, 0); run(259, 1); // refusal did not publish changed counts
                assert(ggml_backend_remote_attn_reset(backend, 0));
                assert(!ggml_backend_remote_attn_native_kvarn_has_state(backend));
                assert(!ggml_backend_remote_attn_native_kvarn_can_restore(backend, 1024 + 129));
                run(2, 0); run(2, 1);
                assert(ggml_backend_remote_attn_native_kvarn_can_restore(backend, 1024 + 1));
                assert(ggml_backend_remote_attn_trim(backend, 0, 1024));
                assert(ggml_backend_remote_attn_native_kvarn_can_restore(backend, 1024));
                ggml_backend_remote_attn_invalidate_range(backend);
                assert(!ggml_backend_remote_attn_native_kvarn_can_restore(backend, 1024));
                assert(!ggml_backend_remote_attn_trim(backend, 0, 1024));
                assert(!ggml_backend_remote_attn_native_kvarn_can_restore(backend, 1024 + 1));
                assert(!ggml_backend_remote_attn_native_kvarn_can_restore(backend, 1024 + 127));
                assert(!ggml_backend_remote_attn_trim(backend, 0, 1024 + 127)); // stale epoch
                assert(!ggml_remote_attn_exec(node));
            } else {
                { std::lock_guard<std::mutex> lock(server.mutex); server.fail_native_rewind = failure; }
                assert(!ggml_backend_remote_attn_trim(backend, 0, 1024 + 127));
                assert(ggml_backend_remote_attn_failed(backend));
                assert(!ggml_remote_attn_exec(node));
            }
        }
        ggml_remote_attn_set_active(nullptr);
        ggml_backend_buffer_free(buffers); ggml_free(ctx); ggml_backend_free(cpu); ggml_backend_free(backend);
      }
    }
    }
}

// Verify negotiated batching, chunk boundaries and OUT/LSE row assembly on the
// actual TCP transport. Shader numerical semantics have independent GPU oracles.
static void test_native_query_prefix_chunks() {
    storage_env native("kvarn4");
    storage_env experimental("1", "GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL");
    storage_env fusion("0", "GGML_RKVA_RANGE_FUSE");
    for (uint32_t tail : {0u, 128u}) {
        storage_env precision(tail ? "128" : "0", "GGML_RKVA_NATIVE_TAIL_TOKENS");
        for (uint32_t support : {3u, 7u}) {
            for (uint32_t nq : {1u, 2u, 3u, 5u, 9u, 20u}) {
                loopback server(false, true, false, false, false, false, 0, 0, true, support);
                auto * backend = ggml_backend_remote_attn_init("127.0.0.1", server.port);
                ggml_remote_attn_geometry geometry{1, 2, 1, 256, 512, 0, 0, 128, 3, 128, 2, 0, 0, false, false, 1.0f/16};
                assert(ggml_backend_remote_attn_set_geometry(backend, &geometry));
                assert(ggml_backend_remote_attn_connect(backend));
                assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 512));
                ggml_backend_remote_attn_set_active_session(backend, 0);
                assert(ggml_backend_remote_attn_configure_range(backend, 1024, 512, GGML_TYPE_Q4_0));
                ggml_remote_attn_set_active(backend);
                auto * ctx = ggml_init({1u << 20, nullptr, true});
                auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 2, nq);
                auto * k = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 512);
                auto * v = ggml_dup_tensor(ctx, k);
                auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
                auto * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nq);
                auto * original_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, nq);
                auto * original_v = ggml_dup_tensor(ctx, original_k);
                auto * node = ggml_remote_attn_range(ctx, q, k, v, ids, pos, 0, 1024, nq, geometry.kq_scale);
                ggml_remote_attn_range_set_native_sources(node, original_k, original_v);
                auto * cpu = ggml_backend_cpu_init();
                auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
                assert(buffer); ggml_backend_buffer_clear(buffer, 0);
                for (uint32_t live = nq; live < 260 + nq; live += nq) {
                    std::vector<int32_t> cells(512, -1), positions(nq);
                    for (uint32_t i = 0; i < live; ++i) cells[i] = 1024 + i;
                    for (uint32_t i = 0; i < nq; ++i) positions[i] = 1024 + live - nq + i;
                    ggml_backend_tensor_set(ids, cells.data(), 0, cells.size() * sizeof(int32_t));
                    ggml_backend_tensor_set(pos, positions.data(), 0, positions.size() * sizeof(int32_t));
                    node->op_params[GGML_REMOTE_ATTN_PARAM_N_LIVE] = int32_t(live);
                    int calls;
                    { std::lock_guard<std::mutex> lock(server.mutex); calls = server.attention_calls; }
                    assert(ggml_remote_attn_exec(node));
                    {
                        std::lock_guard<std::mutex> lock(server.mutex);
                        assert(server.native_descriptor.layout_version == (support == 7 ? 3u : 2u));
                        const uint32_t expected_calls = support == 3 ? nq : nq <= 9 ? 1u : (nq + 7) / 8;
                        assert(server.attention_calls - calls == int(expected_calls));
                    }
                    const auto * out = static_cast<const float *>(node->data);
                    for (uint32_t t = 0; t < nq; ++t) {
                        for (size_t d = 0; d < 512; ++d) assert(out[t * 514 + d] == 0.0f);
                        for (uint32_t h = 0; h < 2; ++h)
                            assert(std::abs(double(out[t * 514 + 512 + h]) - std::log(double(live - nq + t + 1))) < 1e-6);
                    }
                }
                ggml_remote_attn_set_active(nullptr);
                ggml_backend_buffer_free(buffer); ggml_free(ctx);
            }
        }
    }
}

static void test_native_bulk_f16() {
    storage_env native("kvarn4");
    storage_env experimental("1", "GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL");
    storage_env fusion("0", "GGML_RKVA_RANGE_FUSE");
    storage_env precision("128", "GGML_RKVA_NATIVE_TAIL_TOKENS");
    // Test 1: Requested FP16 bulk without server capability fails closed.
    {
        storage_env bulk_env("f16", "GGML_RKVA_NATIVE_BULK");
        loopback server(false, true, false, false, false, false, 0, 0, true, 7 | 8);
        auto * backend = ggml_backend_remote_attn_init("127.0.0.1", server.port);
        ggml_remote_attn_geometry geometry{1, 2, 1, 256, 512, 0, 0, 128, 3, 128, 2, 0, 0, false, false, 1.0f / 16};
        assert(ggml_backend_remote_attn_set_geometry(backend, &geometry));
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 512));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(!ggml_backend_remote_attn_configure_range(backend, 1024, 512, GGML_TYPE_Q4_0));
        ggml_backend_free(backend);
    }
    // Test 2: Negotiated FP16 bulk executes successfully.
    for (const char * mode : {"f16", "1"}) {
        storage_env bulk_env(mode, "GGML_RKVA_NATIVE_BULK");
        storage_env wire_env(!std::strcmp(mode, "1") ? "f16" : nullptr, "GGML_RKVA_NATIVE_BULK_WIRE");
        const uint32_t nq = 5;
        loopback server(false, true, false, false, false, false, 0, 0, true, 7 | 8 | 16);
        auto * backend = ggml_backend_remote_attn_init("127.0.0.1", server.port);
        ggml_remote_attn_geometry geometry{1, 2, 1, 256, 512, 0, 0, 128, 3, 128, 2, 0, 0, false, false, 1.0f / 16};
        assert(ggml_backend_remote_attn_set_geometry(backend, &geometry));
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 512));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(ggml_backend_remote_attn_configure_range(backend, 1024, 512, GGML_TYPE_Q4_0));
        ggml_remote_attn_set_active(backend);
        auto * ctx = ggml_init({1u << 20, nullptr, true});
        auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 2, nq);
        auto * k = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 512);
        auto * v = ggml_dup_tensor(ctx, k);
        auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
        auto * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nq);
        auto * original_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, nq);
        auto * original_v = ggml_dup_tensor(ctx, original_k);
        auto * node = ggml_remote_attn_range(ctx, q, k, v, ids, pos, 0, 1024, nq, geometry.kq_scale);
        ggml_remote_attn_range_set_native_sources(node, original_k, original_v);
        auto * cpu = ggml_backend_cpu_init();
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        assert(buffer);
        ggml_backend_buffer_clear(buffer, 0);
        const uint32_t live = nq;
        std::vector<int32_t> cells(512, -1), positions(nq);
        for (uint32_t i = 0; i < live; ++i) cells[i] = 1024 + i;
        for (uint32_t i = 0; i < nq; ++i) positions[i] = 1024 + live - nq + i;
        ggml_backend_tensor_set(ids, cells.data(), 0, cells.size() * sizeof(int32_t));
        ggml_backend_tensor_set(pos, positions.data(), 0, positions.size() * sizeof(int32_t));
        node->op_params[GGML_REMOTE_ATTN_PARAM_N_LIVE] = int32_t(live);
        assert(ggml_remote_attn_exec(node));
        const auto * out = static_cast<const float *>(node->data);
        for (uint32_t t = 0; t < nq; ++t) {
            for (size_t d = 0; d < 512; ++d) assert(out[t * 514 + d] == 0.0f);
            for (uint32_t h = 0; h < 2; ++h)
                assert(std::abs(double(out[t * 514 + 512 + h]) - std::log(double(live - nq + t + 1))) < 1e-6);
        }
        ggml_remote_attn_set_active(nullptr);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu);
        ggml_backend_free(backend);
    }
}

int main() {
    test_old_native_client_ack_compatibility();
    test_native_rewind();
    test_native_query_prefix_chunks();
    test_native_bulk_f16();
    storage_env clean_generic(nullptr, "GGML_RKVA_RANGE_KERNEL");
    test_range_format_kernel();
    test_snapshot_digest();
    test_range_profile();
    test_range_chunking();
    test_prefill_query_chunking();
    storage_env automatic("auto");
    storage_env automatic_fusion("auto", "GGML_RKVA_RANGE_FUSE");
    storage_env reference_kernel("reference", "GGML_RKVA_Q4_KERNEL");
    storage_env profiling_off("0", "GGML_RKVA_PROFILE");
    for (const char * invalid : {"1", "invalid"}) {
        storage_env force(invalid, "GGML_RKVA_RANGE_FUSE");
        loopback server(false, true, true);
        auto * backend = connect(server);
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 8));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(!ggml_backend_remote_attn_configure_range(backend, 256, 8, GGML_TYPE_Q4_0));
        { std::lock_guard<std::mutex> lock(server.mutex); assert(server.range.capacity == 0); }
        ggml_backend_free(backend);
    }
    for (const auto source_type : {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}) {
        // F16 expansion retains the requested source precision, even with
        // both native formats available on the server.
        storage_env forced("f16");
        loopback server(false, true, true, true, true);
        auto * backend = connect(server);
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 8));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(ggml_backend_remote_attn_configure_range(backend, 256, 8, source_type));
        assert(std::string(ggml_backend_remote_attn_stats_json(backend)).find("\"range_cache_type\":\"f16\"") != std::string::npos);
        ggml_backend_free(backend);
    }
    for (const auto source_type : {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}) {
        storage_env wrong(source_type == GGML_TYPE_Q4_0 ? "q8_0" : "q4_0");
        loopback server(false, true, true, false, true);
        auto * backend = connect(server);
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 8));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(!ggml_backend_remote_attn_configure_range(backend, 256, 8, source_type));
        { std::lock_guard<std::mutex> lock(server.mutex); assert(server.range.capacity == 0); }
        ggml_backend_free(backend);
    }
    { // A Q4-only server cannot satisfy an explicitly native Q8 request.
        storage_env required("q8_0");
        loopback server(false, true, true);
        auto * backend = connect(server);
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 8));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(!ggml_backend_remote_attn_configure_range(backend, 256, 8, GGML_TYPE_Q8_0));
        { std::lock_guard<std::mutex> lock(server.mutex); assert(server.range.capacity == 0); }
        ggml_backend_free(backend);
    }
    for (const char * setting : {"invalid", "q4_0", "q8_0"}) {
        storage_env forced(setting);
        loopback server; // range supported, native Q4 unavailable
        auto * backend = connect(server);
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 8));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(!ggml_backend_remote_attn_configure_range(backend, 256, 8, GGML_TYPE_Q4_0));
        { std::lock_guard<std::mutex> lock(server.mutex); assert(server.range.capacity == 0); }
        ggml_backend_free(backend);
    }
    { // An old server cannot accidentally supply OUT-only as a range result.
        loopback server(true);
        auto * backend = connect(server);
        assert(!ggml_backend_remote_attn_connect(backend));
        ggml_backend_free(backend);
    }
    {
        loopback server(false, false);
        auto * backend = connect(server);
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_session(backend, 0, 0, 8));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(!ggml_backend_remote_attn_configure_range(backend, 256, 8, GGML_TYPE_Q4_0));
        ggml_backend_free(backend);
    }
    { // Advertising Q4 range support must not change legacy whole-session creation.
        loopback server(false, true, true);
        auto * backend = connect(server);
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_session(backend, 0, 0, 8));
        { std::lock_guard<std::mutex> lock(server.mutex); assert(server.session_flags == std::vector<uint32_t>({0})); }
        ggml_backend_free(backend);
    }
    for (const auto cache_type : {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_F16}) {
    for (bool native_quant : {false, true}) {
    if (cache_type == GGML_TYPE_F16 && native_quant) continue;
    for (int fusion_case : {0, 1, 2}) {
    const bool supports_fusion = fusion_case != 0;
    const bool use_fusion = fusion_case == 1;
    storage_env fuse_mode(fusion_case == 2 ? "0" : "auto", "GGML_RKVA_RANGE_FUSE");
    loopback server(false, true, native_quant && cache_type == GGML_TYPE_Q4_0, false,
                    native_quant && cache_type == GGML_TYPE_Q8_0, supports_fusion);
    auto * backend = connect(server, 2);
    assert(ggml_backend_remote_attn_connect(backend));
    assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 8));
    ggml_backend_remote_attn_set_active_session(backend, 0);
    assert(ggml_backend_remote_attn_configure_range(backend, 256, 8, cache_type));
    const std::string initial_stats = ggml_backend_remote_attn_stats_json(backend);
    assert(initial_stats.find("\"range_attn_calls\":0") != std::string::npos);
    assert(initial_stats.find("\"range_upload_calls\":0") != std::string::npos);
    ggml_remote_attn_set_active(backend);
    ggml_init_params params {1u << 20, nullptr, true};
    auto * ctx = ggml_init(params);
    auto * q_input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 2, 2);
    auto * q_zero = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 2, 2);
    // Force a multi-worker CPU graph. A remote-only graph plans one worker
    // and would hide duplicate RPC execution by the remaining CPU workers.
    auto * q = ggml_add(ctx, q_input, q_zero);
    auto * k = ggml_new_tensor_2d(ctx, cache_type, 32, 8);
    auto * v = ggml_new_tensor_2d(ctx, cache_type, 32, 8);
    auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    auto * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
    auto * node = ggml_remote_attn_range(ctx, q, k, v, ids, pos, 0, 256, 4, 0.125f);
    auto * graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, node);
    auto * cpu = ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(cpu, 4);
    auto * buffers = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    assert(buffers);
    const bool varied = fusion_case != 0;
    std::vector<float> queries(32 * 2 * 2, 0.0f);
    if (varied) {
        for (size_t i = 0; i < queries.size(); ++i) queries[i] = (int(i % 17) - 8) * 0.07f;
    }
    ggml_backend_tensor_set(q_input, queries.data(), 0, queries.size() * 4);
    const std::vector<float> zeros(queries.size(), 0.0f);
    ggml_backend_tensor_set(q_zero, zeros.data(), 0, zeros.size() * 4);
    std::vector<float> raw_k(32 * 8, 0), raw_v(32 * 8);
    for (int row = 0; row < 8; ++row) std::fill(raw_v.begin() + row * 32, raw_v.begin() + (row + 1) * 32, (float) (row + 1));
    if (varied) {
        for (int row = 0; row < 8; ++row) {
            for (int d = 0; d < 32; ++d) {
                raw_k[row * 32 + d] = ((row * 11 + d * 5) % 23 - 11) * 0.125f;
                raw_v[row * 32 + d] = ((row * 7 + d * 3) % 29 - 14) * 0.1875f;
            }
        }
    }
    if (cache_type == GGML_TYPE_F16) {
        // Values with meaningful half mantissa bits would not survive Q4.
        raw_k[1] = 1.0009765625f;
        raw_v[3] = 0.333251953125f;
        raw_v[33] = -0.99951171875f;
    }
    std::vector<uint8_t> quant(ggml_row_size(cache_type, 32) * 8);
    ggml_quantize_chunk(cache_type, raw_k.data(), quant.data(), 0, 8, 32, nullptr);
    ggml_backend_tensor_set(k, quant.data(), 0, quant.size());
    const auto quant_k = quant;
    ggml_quantize_chunk(cache_type, raw_v.data(), quant.data(), 0, 8, 32, nullptr);
    ggml_backend_tensor_set(v, quant.data(), 0, quant.size());
    const auto quant_v = quant;
    std::vector<float> expected_k(raw_k.size()), expected_v(raw_v.size());
    ggml_get_type_traits(cache_type)->to_float(quant_k.data(), expected_k.data(), expected_k.size());
    ggml_get_type_traits(cache_type)->to_float(quant_v.data(), expected_v.data(), expected_v.size());
    if (!native_quant) {
        for (auto & value : expected_k) value = ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
        for (auto & value : expected_v) value = ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
    }
    auto run = [&](int count) {
        int32_t cells[4] = {256,257,258,259};
        for (int i = count; i < 4; ++i) cells[i] = -1;
        int32_t positions[2] = {256 + count - 2, 256 + count - 1};
        ggml_backend_tensor_set(ids, cells, 0, sizeof(cells));
        ggml_backend_tensor_set(pos, positions, 0, sizeof(positions));
        assert(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
        std::vector<float> result(66 * 2); ggml_backend_tensor_get(node, result.data(), 0, result.size() * 4);
        for (int t = 0; t < 2; ++t) {
            const int live = count - 1 + t;
            for (int head = 0; head < 2; ++head) {
                // Independent FP64 reference over the original host tensors;
                // unequal signed Q/K/V expose query/head/plane misalignment.
                std::vector<double> weights(live);
                for (int row = 0; row < live; ++row) {
                    double dot = 0;
                    for (int d = 0; d < 32; ++d) dot += double(queries[t * 64 + head * 32 + d]) * double(expected_k[row * 32 + d]);
                    weights[row] = dot * 0.125;
                }
                const double max_score = *std::max_element(weights.begin(), weights.end());
                double denominator = 0;
                for (auto & weight : weights) { weight = std::exp(weight - max_score); denominator += weight; }
                for (int d = 0; d < 32; ++d) {
                    double numerator = 0;
                    for (int row = 0; row < live; ++row) numerator += weights[row] * expected_v[row * 32 + d];
                    assert(std::abs(result[t * 66 + head * 32 + d] - numerator / denominator) < 1e-6);
                }
                assert(std::abs(result[t * 66 + 64 + head] - (max_score + std::log(denominator))) < 1e-6);
            }
        }
    };
    run(2); run(4);
    ggml_backend_remote_attn_invalidate_range(backend); run(4);
    run(2); // rewind/shrink replaces the complete layer prefix
    const std::string stats = ggml_backend_remote_attn_stats_json(backend);
    assert(stats.find("\"range_attn_calls\":4") != std::string::npos);
    assert(stats.find("\"range_fused_calls\":" + std::to_string(use_fusion ? 4 : 0)) != std::string::npos);
    assert(stats.find("\"calls\":" + std::to_string(use_fusion ? 7 : 11) + ",") != std::string::npos);
    assert(stats.find("\"range_upload_rows\":10") != std::string::npos);
    assert(stats.find("\"range_query_min\":256") != std::string::npos);
    assert(stats.find("\"range_query_max\":259") != std::string::npos);
    const std::string expected_storage = native_quant ? ggml_type_name(cache_type) : "f16";
    assert(stats.find("\"range_cache_type\":\"" + expected_storage + "\"") != std::string::npos);
    assert(stats.find("\"range_source_cache_type\":\"" + std::string(ggml_type_name(cache_type)) + "\"") != std::string::npos);
    const size_t uploaded_bytes = 10 * 2 * (native_quant ? ggml_row_size(cache_type, 32) : 64);
    assert(stats.find("\"range_upload_bytes\":" + std::to_string(uploaded_bytes)) != std::string::npos);
    {
        std::lock_guard<std::mutex> lock(server.mutex);
        assert(server.attention_calls == 4 && server.uploads.size() == 4);
        assert(server.fused_calls == (use_fusion ? 4 : 0));
        assert(std::count(server.packet_ops.begin(), server.packet_ops.end(), RKVA_RANGE_UPLOAD_ATTN) == (use_fusion ? 4 : 0));
        assert(server.session_flags == std::vector<uint32_t>({native_quant ? RKVA_SESSION_RANGE : 0}));
        assert(server.uploads[0].start_row == 0 && server.uploads[0].n_rows == 2);
        assert(server.uploads[1].start_row == 2 && server.uploads[1].n_rows == 2);
        assert(server.uploads[2].start_row == 0 && server.uploads[2].n_rows == 4);
        assert(server.uploads[3].start_row == 0 && server.uploads[3].n_rows == 2);
        if (native_quant || cache_type == GGML_TYPE_F16) {
            const size_t row_bytes = ggml_row_size(cache_type, 32);
            for (size_t upload = 0; upload < server.uploads.size(); ++upload) {
                const auto first = server.uploads[upload].start_row * row_bytes;
                const auto bytes = server.uploads[upload].n_rows * row_bytes;
                const auto & raw = server.upload_payloads[upload];
                assert(raw.size() == 2 * bytes);
                assert(std::memcmp(raw.data(), quant_k.data() + first, bytes) == 0);
                assert(std::memcmp(raw.data() + bytes, quant_v.data() + first, bytes) == 0);
            }
        }
    }
    // A validated suffix removal preserves both layers with one session-wide
    // ACK. Rewriting the rejected rows uploads only the delta, never the full
    // retained prefix. An unannounced same-count rewrite above still resyncs.
    node->op_params[GGML_REMOTE_ATTN_PARAM_LAYER_ID] = 1;
    run(4);
    node->op_params[GGML_REMOTE_ATTN_PARAM_LAYER_ID] = 0;
    run(4);
    assert(ggml_backend_remote_attn_trim(backend, 0, 258));
    const size_t suffix_offset = 2 * ggml_row_size(cache_type, 32);
    const size_t suffix_bytes = 2 * ggml_row_size(cache_type, 32);
    ggml_backend_tensor_set(k, quant_v.data() + suffix_offset, suffix_offset, suffix_bytes);
    ggml_backend_tensor_set(v, quant_k.data() + suffix_offset, suffix_offset, suffix_bytes);
    for (size_t i = 2 * 32; i < 4 * 32; ++i) std::swap(expected_k[i], expected_v[i]);
    run(4);
    node->op_params[GGML_REMOTE_ATTN_PARAM_LAYER_ID] = 1;
    run(4);
    assert(ggml_backend_remote_attn_trim(backend, 0, 260)); // no-op, no RPC
    {
        std::lock_guard<std::mutex> lock(server.mutex);
        assert(std::count(server.packet_ops.begin(), server.packet_ops.end(), RKVA_REWIND) == 1);
        assert(server.uploads.size() == 8 && server.attention_calls == 8);
        for (size_t i = 6; i < 8; ++i) {
            assert(server.uploads[i].start_row == 2 && server.uploads[i].n_rows == 2);
        }
    }
    // Invalid global IDs fail before transmitting any upload/attention.
    int32_t hole[4] = {256,258,-1,-1}; ggml_backend_tensor_set(ids, hole, 0, sizeof(hole));
    assert(!ggml_remote_attn_exec(node));
    { std::lock_guard<std::mutex> lock(server.mutex); assert(server.attention_calls == 8); }
    { std::lock_guard<std::mutex> lock(server.mutex); server.fail_trim = true; }
    assert(!ggml_backend_remote_attn_trim(backend, 0, 258));
    assert(ggml_backend_remote_attn_failed(backend));
    assert(!ggml_remote_attn_exec(node)); // failed ACK never resumes remote compute
    ggml_remote_attn_set_active(nullptr);
    ggml_backend_buffer_free(buffers); ggml_free(ctx); ggml_backend_free(cpu); ggml_backend_free(backend);
    }
    }
    }
    return 0;
}
#else
int main() { return 0; } // POSIX transport is unavailable on Windows hosts.
#endif
