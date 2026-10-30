// Host-only v2 RKVA loopback contract test. No model, GPU or Xbox required.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-remote-attn.h"
#include "ggml-remote-attn-cuda.h"
#include "rkva-snapshot-digest.h"
#include "../ggml/src/ggml-remote-attn/remote-attn-protocol.h"
#include "../ggml/src/ggml-remote-attn/rkva-kvarn-rewind.h"
#include "../ggml/src/ggml-remote-attn/rkva-kvarn-producer.h"
#undef NDEBUG
#include <cassert>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <future>
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
    enum class profile_failure { none, cycle, version, oversized, truncated, server_error, short_header, snapshot_miss };
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
    std::atomic<bool> fail_attention{false}; // fail the next native/range attention reply (valid server error, framed)
    std::atomic<bool> drop_attention{false}; // close the stream instead of replying to the next attention call
    std::atomic<profile_failure> fail_bulk_reply{profile_failure::none};
    profile_failure fail_native_rewind = profile_failure::none;
    std::vector<uint32_t> native_counts;
    std::vector<bool> native_ready;
    rkva_kvarn_range_config native_descriptor {};
    uint64_t native_generation = 17;
    bool native_snapshot_saved = false;
    rkva_snapshot::Digest native_snapshot_hash{};
    uint64_t native_snapshot_generation = 0;
    uint32_t native_snapshot_rows = 0;
    std::vector<rkva_range_profile> profile_controls;
    bool profile_enabled = false;
    uint32_t kernel_mode = 0;
    std::vector<uint8_t> fp8_query_prefix;
    std::vector<uint8_t> fp8_query_data;
    std::vector<int32_t> fp8_query_positions;
    std::vector<uint32_t> bulk_query_segments;
    std::vector<uint32_t> bulk_live_counts;
    uint8_t fp8_output_byte = 0;
    std::vector<uint8_t> fp8_output_data; // optional independent per-row/head/dimension fixture
    float lse_head_delta = 0.0f;
    profile_failure fail_report = profile_failure::none;
    enum class cohort_failure { none, last_ticket, last_epoch, last_out_nan, last_lse_nan,
                                last_length, server_suffix_error };
    cohort_failure fail_cohort = cohort_failure::none;
    unsigned cohort_calls = 0;
    std::vector<uint32_t> cohort_counts, cohort_queries, cohort_layers;
    std::vector<uint64_t> cohort_tickets, cohort_epochs;
    std::atomic<bool> pause_mode{false};
    int reply_tokens = 0;
    unsigned paused_replies = 0; // protected by pause_mutex
    std::mutex pause_mutex;
    std::condition_variable pause_cv;

    void set_pause(bool p, int tokens = 0) {
        {
            std::lock_guard<std::mutex> lk(pause_mutex);
            pause_mode.store(p);
            reply_tokens = tokens;
        }
        pause_cv.notify_all();
    }

    void grant_reply(int n = 1) {
        {
            std::lock_guard<std::mutex> lk(pause_mutex);
            reply_tokens += n;
        }
        pause_cv.notify_all();
    }

    bool wait_until_reply_paused() {
        std::unique_lock<std::mutex> lk(pause_mutex);
        return pause_cv.wait_for(lk, std::chrono::seconds(2), [this] { return paused_replies != 0; });
    }

    void unpause() {
        {
            std::lock_guard<std::mutex> lk(pause_mutex);
            pause_mode.store(false);
        }
        pause_cv.notify_all();
    }

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
                            if ((geometry.proto_flags & RKVA_PROTO_NATIVE_SERIAL_PREFIX_V3) && (native_rewind_support & 512)) {
                                ack.reserved[0] |= RKVA_CAP_KVARN_BULK_512;
                            }
                            if ((geometry.proto_flags & RKVA_PROTO_NATIVE_SERIAL_PREFIX_V3) && (native_rewind_support & 256)) {
                                ack.reserved[0] |= RKVA_CAP_KVARN_COHORT_ATTN;
                                ack.supported_ops |= (1u << RKVA_KVARN_RANGE_COHORT_ATTN);
                            }
                            if ((geometry.proto_flags & RKVA_PROTO_NATIVE_SERIAL_PREFIX_V3) && (native_rewind_support & 16)) {
                                ack.reserved[0] |= RKVA_CAP_KVARN_BULK_F16;
                            }
                            if ((geometry.proto_flags & RKVA_PROTO_NATIVE_SERIAL_PREFIX_V3) && (native_rewind_support & 32)) {
                                ack.reserved[0] |= RKVA_CAP_KVARN_BULK_FP8;
                                if (!(native_rewind_support & 128))
                                    ack.reserved[0] |= RKVA_CAP_KVARN_BULK_FP8_RNE;
                            }
                            if ((geometry.proto_flags & RKVA_PROTO_NATIVE_SERIAL_PREFIX_V3) && (native_rewind_support & 64)) {
                                ack.reserved[0] |= RKVA_CAP_KVARN_BULK_SPLITZIP;
                            }
                            if ((geometry.proto_flags & RKVA_PROTO_NATIVE_SNAPSHOT_V4) &&
                                (native_rewind_support & 512)) {
                                ack.reserved[0] |= RKVA_CAP_NATIVE_KVARN_SNAPSHOT;
                                ack.supported_ops |= (1u << RKVA_SNAPSHOT_SAVE) |
                                                     (1u << RKVA_SNAPSHOT_RESTORE);
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
                        case RKVA_DESTROY_SESSION:
                            assert(payload.empty()); native_counts.clear(); native_ready.clear(); ++native_generation; break;
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
                        case RKVA_SNAPSHOT_SAVE:
                        case RKVA_SNAPSHOT_RESTORE: {
                            assert(payload.size() == sizeof(rkva_snapshot_req) +
                                                           sizeof(rkva_native_snapshot_req));
                            rkva_snapshot_req snapshot {};
                            rkva_native_snapshot_req native {};
                            std::memcpy(&snapshot, payload.data(), sizeof(snapshot));
                            std::memcpy(&native, payload.data() + sizeof(snapshot), sizeof(native));
                            assert(snapshot.revision == RKVA_SNAPSHOT_REVISION_NATIVE);
                            assert(snapshot.flags == 0 && snapshot.reserved == 0);
                            assert(native.format_version == RKVA_NATIVE_SNAPSHOT_FORMAT);
                            assert(native.flags == 0);
                            assert(native.expected_generation == native_generation);
                            rkva_snapshot_ack ack {};
                            rkva_native_snapshot_ack native_ack {};
                            if (req.op == RKVA_SNAPSHOT_SAVE) {
                                assert(native.saved_generation == native_generation);
                                assert(snapshot.live_rows);
                                for (size_t il = 0; il < native_counts.size(); ++il)
                                    assert(native_counts[il] == snapshot.live_rows && native_ready[il]);
                                native_snapshot_saved = true;
                                std::memcpy(native_snapshot_hash.data(),
                                            snapshot.host_state_sha256, 32);
                                native_snapshot_generation = native_generation;
                                native_snapshot_rows = snapshot.live_rows;
                                native_ack.generation = native_generation;
                            } else {
                                if (!native_snapshot_saved ||
                                    native.saved_generation != native_snapshot_generation ||
                                    snapshot.live_rows != native_snapshot_rows ||
                                    std::memcmp(native_snapshot_hash.data(),
                                                snapshot.host_state_sha256, 32) != 0) {
                                    response_failure = profile_failure::snapshot_miss;
                                    break;
                                }
                                ++native_generation;
                                std::fill(native_counts.begin(), native_counts.end(),
                                          snapshot.live_rows);
                                std::fill(native_ready.begin(), native_ready.end(), true);
                                native_ack.generation = native_generation;
                            }
                            std::memcpy(ack.host_state_sha256, snapshot.host_state_sha256, 32);
                            ack.live_rows = snapshot.live_rows;
                            ack.n_layers = geometry.n_layer_remote;
                            ack.disk_bytes = 1;
                            native_ack.layers = geometry.n_layer_remote;
                            response.resize(sizeof(ack) + sizeof(native_ack));
                            std::memcpy(response.data(), &ack, sizeof(ack));
                            std::memcpy(response.data() + sizeof(ack), &native_ack,
                                        sizeof(native_ack));
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
                        case RKVA_KVARN_RANGE_COHORT_ATTN: {
                            assert(native_kvarn && (native_rewind_support & 256));
                            assert(req.layer_id == 0 && payload.size() >= sizeof(rkva_kvarn_cohort_header));
                            rkva_kvarn_cohort_header header{};
                            std::memcpy(&header, payload.data(), sizeof(header));
                            assert(header.version == 1 && header.entry_count >= 1 && header.entry_count <= 4);
                            assert(!header.reserved && header.total_queries == req.n_tokens && req.n_tokens <= 256);
                            struct item {
                                rkva_kvarn_cohort_entry entry{};
                                uint32_t live = 0, segments = 0;
                            };
                            std::vector<item> items;
                            size_t offset = sizeof(header);
                            uint32_t queries = 0, segments = 0;
                            uint64_t ticket = 0, epoch = 0;
                            // Parse the complete request before simulating any KV mutation.
                            for (uint32_t i = 0; i < header.entry_count; ++i) {
                                assert(payload.size() - offset >= sizeof(rkva_kvarn_cohort_entry));
                                item next;
                                std::memcpy(&next.entry, payload.data() + offset, sizeof(next.entry));
                                offset += sizeof(next.entry);
                                const auto& e = next.entry;
                                assert(e.layer < geometry.n_layer_remote && e.n_queries && !e.reserved);
                                assert(e.ticket > ticket && e.epoch && (!i || e.epoch == epoch));
                                assert(e.body_bytes <= payload.size() - offset);
                                for (const auto& previous : items) assert(previous.entry.layer != e.layer);
                                ticket = e.ticket; epoch = e.epoch;
                                const size_t end = offset + e.body_bytes;
                                assert(end - offset >= sizeof(rkva_kvarn_range_bulk));
                                rkva_kvarn_range_bulk inner{};
                                std::memcpy(&inner, payload.data() + offset, sizeof(inner));
                                offset += sizeof(inner);
                                assert(inner.version == 1 && !inner.reserved && inner.total_queries == e.n_queries);
                                assert(inner.segment_count && inner.segment_count <= 32);
                                uint32_t entry_queries = 0;
                                for (uint32_t j = 0; j < inner.segment_count; ++j) {
                                    assert(end - offset >= sizeof(rkva_kvarn_range_bulk_segment));
                                    rkva_kvarn_range_bulk_segment segment{};
                                    std::memcpy(&segment, payload.data() + offset, sizeof(segment));
                                    offset += sizeof(segment);
                                    assert(segment.n_queries && segment.n_queries <= 9);
                                    assert(segment.upload_bytes >= sizeof(rkva_kvarn_range_upload) && segment.upload_bytes <= end - offset);
                                    rkva_kvarn_range_upload upload{};
                                    std::memcpy(&upload, payload.data() + offset, sizeof(upload));
                                    next.live = upload.live_count;
                                    offset += segment.upload_bytes;
                                    assert(end - offset >= sizeof(rkva_range_attn));
                                    rkva_range_attn attention{};
                                    std::memcpy(&attention, payload.data() + offset, sizeof(attention));
                                    assert(attention.wire == RKVA_WIRE_FP8 && !attention.flags && !attention.reserved[0] && !attention.reserved[1]);
                                    offset += sizeof(attention);
                                    const size_t query_bytes = size_t(segment.n_queries) *
                                        (sizeof(int32_t) + size_t(geometry.n_head) * geometry.head_dim);
                                    assert(query_bytes <= end - offset);
                                    offset += query_bytes;
                                    entry_queries += segment.n_queries;
                                }
                                assert(offset == end && entry_queries == e.n_queries);
                                next.segments = inner.segment_count;
                                queries += e.n_queries; segments += inner.segment_count;
                                items.push_back(next);
                            }
                            assert(offset == payload.size() && queries == req.n_tokens && segments <= 32);
                            ++cohort_calls;
                            cohort_counts.push_back(header.entry_count);
                            cohort_queries.push_back(queries);
                            if (fail_cohort == cohort_failure::server_suffix_error) {
                                response_failure = profile_failure::server_error;
                                break; // simulated pre-admission failure, no layer is mutated
                            }
                            response.resize(sizeof(header));
                            std::memcpy(response.data(), &header, sizeof(header));
                            for (uint32_t i = 0; i < items.size(); ++i) {
                                const auto& next = items[i];
                                auto echo = next.entry;
                                const size_t out_bytes = size_t(echo.n_queries) * geometry.n_head * geometry.head_dim;
                                const size_t lse_bytes = size_t(echo.n_queries) * geometry.n_head * sizeof(float);
                                echo.body_bytes = uint32_t(out_bytes + lse_bytes);
                                native_counts[echo.layer] = next.live;
                                native_ready[echo.layer] = true;
                                native_generation += next.segments;
                                cohort_layers.push_back(echo.layer);
                                cohort_tickets.push_back(echo.ticket);
                                cohort_epochs.push_back(echo.epoch);
                                const bool last = i + 1 == items.size();
                                if (last && fail_cohort == cohort_failure::last_ticket) ++echo.ticket;
                                if (last && fail_cohort == cohort_failure::last_epoch) ++echo.epoch;
                                if (last && fail_cohort == cohort_failure::last_length) ++echo.body_bytes;
                                const size_t before = response.size();
                                response.resize(before + sizeof(echo) + out_bytes + lse_bytes);
                                std::memcpy(response.data() + before, &echo, sizeof(echo));
                                uint8_t* body = response.data() + before + sizeof(echo);
                                std::memset(body, uint8_t(0x3c + echo.layer), out_bytes);
                                for (uint32_t q = 0; q < echo.n_queries; ++q) {
                                    const float lse = float(std::log(double(next.live - echo.n_queries + q + 1))) + float(echo.layer) * 0.125f;
                                    for (uint32_t h = 0; h < geometry.n_head; ++h)
                                        std::memcpy(body + out_bytes + (size_t(q) * geometry.n_head + h) * sizeof(float), &lse, sizeof(lse));
                                }
                                if (last && fail_cohort == cohort_failure::last_out_nan) body[out_bytes - 1] = 0x7f;
                                if (last && fail_cohort == cohort_failure::last_lse_nan) {
                                    const float nan = std::numeric_limits<float>::quiet_NaN();
                                    std::memcpy(body + out_bytes + lse_bytes - sizeof(float), &nan, sizeof(nan));
                                }
                            }
                        } break;
                        case RKVA_KVARN_RANGE_BULK_ATTN: {
                            assert(native_kvarn && (native_rewind_support & 8));
                            rkva_kvarn_range_bulk header{};
                            assert(payload.size() >= sizeof(header));
                            std::memcpy(&header, payload.data(), sizeof(header));
                            assert(header.version == 1 && header.total_queries == req.n_tokens);
                            assert(header.segment_count > 0);
                            native_generation += header.segment_count;

                            size_t seg_offset = sizeof(header);
                            rkva_range_attn attn{};
                            fp8_query_data.clear();
                            fp8_query_positions.clear();
                            bulk_query_segments.clear();
                            bulk_live_counts.clear();
                            uint32_t total_segment_queries = 0;
                            for (uint32_t s = 0; s < header.segment_count; ++s) {
                                assert(payload.size() >= seg_offset + sizeof(rkva_kvarn_range_bulk_segment));
                                rkva_kvarn_range_bulk_segment seg{};
                                std::memcpy(&seg, payload.data() + seg_offset, sizeof(seg));
                                assert(seg.n_queries > 0 && seg.n_queries <= header.total_queries - total_segment_queries);
                                total_segment_queries += seg.n_queries;
                                bulk_query_segments.push_back(seg.n_queries);
                                seg_offset += sizeof(seg);
                                assert(payload.size() >= seg_offset + seg.upload_bytes);
                                rkva_kvarn_range_upload u{};
                                assert(seg.upload_bytes >= sizeof(u));
                                std::memcpy(&u, payload.data() + seg_offset, sizeof(u));
                                bulk_live_counts.push_back(u.live_count);
                                native_counts[req.layer_id] = u.live_count;
                                native_ready[req.layer_id] = true;
                                seg_offset += seg.upload_bytes;
                                assert(payload.size() >= seg_offset + sizeof(rkva_range_attn));
                                std::memcpy(&attn, payload.data() + seg_offset, sizeof(attn));
                                seg_offset += sizeof(rkva_range_attn);
                                const bool wire_f16_seg = (attn.wire == RKVA_WIRE_F16);
                                const bool wire_fp8_seg = (attn.wire == RKVA_WIRE_FP8);
                                const bool wire_sz_seg  = (attn.wire == RKVA_WIRE_SPLITZIP);
                                const size_t elem_bytes = wire_fp8_seg ? sizeof(uint8_t) :
                                                          (wire_f16_seg || wire_sz_seg) ? sizeof(uint16_t) :
                                                          sizeof(float);
                                const size_t q_bytes = seg.n_queries * (sizeof(int32_t) + geometry.n_head * geometry.head_dim * elem_bytes);
                                assert(payload.size() >= seg_offset + q_bytes);
                                if (wire_fp8_seg) {
                                    const size_t before = fp8_query_positions.size();
                                    fp8_query_positions.resize(before + seg.n_queries);
                                    std::memcpy(fp8_query_positions.data() + before,
                                                payload.data() + seg_offset, seg.n_queries * sizeof(int32_t));
                                    const size_t begin = seg_offset + seg.n_queries * sizeof(int32_t);
                                    fp8_query_data.insert(fp8_query_data.end(), payload.begin() + begin,
                                            payload.begin() + begin + seg.n_queries * geometry.n_head * geometry.head_dim);
                                }
                                if (s == 0 && wire_fp8_seg && q_bytes >= sizeof(int32_t) + 8) {
                                    fp8_query_prefix.assign(payload.begin() + seg_offset + seg.n_queries * sizeof(int32_t),
                                                           payload.begin() + seg_offset + seg.n_queries * sizeof(int32_t) + 8);
                                }
                                seg_offset += q_bytes;
                            }
                            assert(total_segment_queries == header.total_queries && seg_offset == payload.size());
                            const bool wire_f16 = (attn.wire == RKVA_WIRE_F16);
                            const bool wire_fp8 = (attn.wire == RKVA_WIRE_FP8);
                            const bool wire_sz = (attn.wire == RKVA_WIRE_SPLITZIP);
                            assert(attn.wire == RKVA_WIRE_F32 || (wire_f16 && (native_rewind_support & 16)) ||
                                   (wire_fp8 && (native_rewind_support & 32)) || (wire_sz && (native_rewind_support & 64)));
                            const size_t dh = size_t(geometry.n_head) * geometry.head_dim;
                            const size_t out_elem_bytes = wire_fp8 ? sizeof(uint8_t) :
                                                          (wire_f16 || wire_sz) ? sizeof(uint16_t) :
                                                          sizeof(float);
                            const size_t out_bytes = size_t(req.n_tokens) * dh * out_elem_bytes;
                            const size_t lse_bytes = size_t(req.n_tokens) * geometry.n_head * sizeof(float);
                            response.resize(out_bytes + lse_bytes, 0);
                            if (wire_fp8) {
                                if (!fp8_output_data.empty()) {
                                    assert(fp8_output_data.size() == size_t(req.n_tokens) * dh);
                                }
                                auto * dst_fp8 = reinterpret_cast<uint8_t *>(response.data());
                                auto * dst_lse = reinterpret_cast<float *>(response.data() + out_bytes);
                                for (uint32_t t = 0; t < req.n_tokens; ++t) {
                                    for (size_t d = 0; d < dh; ++d)
                                        dst_fp8[t * dh + d] = fp8_output_data.empty() ? fp8_output_byte : fp8_output_data[t * dh + d];
                                    for (uint32_t h = 0; h < geometry.n_head; ++h)
                                        dst_lse[t * geometry.n_head + h] =
                                            float(std::log(double(native_counts[req.layer_id] - req.n_tokens + t + 1))) + float(h) * lse_head_delta;
                                }
                            } else if (wire_f16 || wire_sz) {
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
                    if (pause_mode.load() && (req.op == RKVA_KVARN_RANGE_BULK_ATTN || req.op == RKVA_KVARN_RANGE_COHORT_ATTN || req.op == RKVA_RANGE_ATTN || req.op == RKVA_RANGE_UPLOAD_ATTN)) {
                        std::unique_lock<std::mutex> lk(pause_mutex);
                        ++paused_replies;
                        pause_cv.notify_all();
                        pause_cv.wait(lk, [this] { return !pause_mode.load() || reply_tokens > 0 || client.load() < 0; });
                        --paused_replies;
                        if (reply_tokens > 0) {
                            --reply_tokens;
                        }
                    }
                    if (fail_attention &&
                            (req.op == RKVA_KVARN_RANGE_BULK_ATTN || req.op == RKVA_RANGE_ATTN ||
                             req.op == RKVA_RANGE_UPLOAD_ATTN)) {
                        response_failure = profile_failure::server_error;
                    }
                    if (req.op == RKVA_KVARN_RANGE_BULK_ATTN && fail_bulk_reply != profile_failure::none) {
                        response_failure = fail_bulk_reply;
                    }
                    if (drop_attention &&
                            (req.op == RKVA_KVARN_RANGE_BULK_ATTN || req.op == RKVA_RANGE_ATTN ||
                             req.op == RKVA_RANGE_UPLOAD_ATTN)) {
                        break; // broken transport: no reply, close the stream
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
                    if (response_failure == profile_failure::snapshot_miss) {
                        rsp.status = RKVA_ERR_SNAPSHOT_MISS;
                        rsp.payload_bytes = 0;
                        response.clear();
                    }
                    if (response_failure == profile_failure::short_header) {
                        write_all(fd, &rsp, sizeof(rsp) / 2);
                        break;
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
        unpause();
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
    // This fixture exercises legacy rewind layouts (1/2/3) with servers that do
    // not advertise the now-default bulk RPC. Keep the opt-out explicit.
    storage_env legacy_bulk("0", "GGML_RKVA_NATIVE_BULK");
    storage_env legacy_async("0", "GGML_RKVA_ASYNC_LOCAL_REMOTE");
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

// Host-side transport/state regression for the revision-2 native snapshot
// envelope: save, reset to an empty resident session, parse the canonical owner
// trailer, restore, continue, and rewind. This exercises the real request/ACK
// parser and the staged owner commit over TCP; the Xbox disk codec remains in
// the MSVC/D3D12 build.
static void test_native_snapshot_transport() {
    storage_env native("kvarn4");
    storage_env experimental("1", "GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL");
    storage_env fusion("0", "GGML_RKVA_RANGE_FUSE");
    storage_env legacy_bulk("0", "GGML_RKVA_NATIVE_BULK");
    storage_env legacy_async("0", "GGML_RKVA_ASYNC_LOCAL_REMOTE");
    storage_env precision("128", "GGML_RKVA_NATIVE_TAIL_TOKENS");
    storage_env snapshots("1", "GGML_RKVA_DISK_SNAPSHOTS");
    ggml_backend_load_all();
    loopback server(false, true, false, false, false, false, 0, 0, true, 3u | 4u | 512u);
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
    assert(ggml_backend_remote_attn_native_kvarn_snapshot_supported(backend));
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
    };
    for (uint32_t live = 2; live <= 130; live += 2) { run(live, 0); run(live, 1); }
    assert(ggml_backend_remote_attn_native_kvarn_has_state(backend));

    std::vector<uint8_t> owner_state;
    uint8_t owner_key[32] = {};
    assert(ggml_backend_remote_attn_native_kvarn_state_write(
        backend, [](void * user, const void * data, size_t bytes) {
            auto * out = static_cast<std::vector<uint8_t> *>(user);
            const auto * begin = static_cast<const uint8_t *>(data);
            out->insert(out->end(), begin, begin + bytes);
            return true;
        }, &owner_state, owner_key));
    assert(!owner_state.empty());
    // The server appends its SCKP checkpoint block after the llama state. The
    // native key binds the complete host prefix through the RKVK frame, so
    // these trailing bytes must neither be consumed by the owner parser nor
    // alter the key.
    std::vector<uint8_t> framed_state = owner_state;
    const uint32_t sckp[3] = {0x504b4353u, 1u, 0u};
    framed_state.insert(framed_state.end(), reinterpret_cast<const uint8_t *>(sckp),
                        reinterpret_cast<const uint8_t *>(sckp) + sizeof(sckp));
    // Simulated host prefix (server header, tokens and local KV/recurrent
    // state) precedes RKVK in the real file. A same-size flip here must change
    // the key while leaving RKVK byte-identical.
    std::vector<uint8_t> local_prefix(137, 0x33);
    const auto prefix_key = [](const std::vector<uint8_t> & prefix,
                               const std::vector<uint8_t> & frame) {
        std::vector<uint8_t> bytes = prefix;
        bytes.insert(bytes.end(), frame.begin(), frame.end());
        return rkva_snapshot::digest(bytes.data(), bytes.size());
    };
    const auto good_key = prefix_key(local_prefix, owner_state);
    auto mutated_local = local_prefix;
    mutated_local[7] ^= 0x01;
    const auto bad_local_key = prefix_key(mutated_local, owner_state);
    assert(std::memcmp(good_key.data(), bad_local_key.data(), good_key.size()) != 0);
    assert(ggml_backend_remote_attn_native_kvarn_remote_save(backend, good_key.data(), 130));

    // Fresh resident session (empty counts/coverage), same connection.
    assert(ggml_backend_remote_attn_reset(backend, 0));
    assert(!ggml_backend_remote_attn_native_kvarn_has_state(backend));
    uint32_t rows = 0;
    uint64_t saved_generation = 0;
    std::pair<const std::vector<uint8_t> *, size_t> source_state{&framed_state, 0};
    uint8_t restored_key[32] = {};
    assert(ggml_backend_remote_attn_native_kvarn_state_read(
        backend, [](void * user, void * data, size_t bytes) {
            auto * source = static_cast<std::pair<const std::vector<uint8_t> *, size_t> *>(user);
            if (bytes > source->first->size() - source->second) return false;
            std::memcpy(data, source->first->data() + source->second, bytes);
            source->second += bytes;
            return true;
        }, &source_state, &rows, &saved_generation, restored_key));
    assert(source_state.second == owner_state.size());
    assert(std::memcmp(restored_key, owner_key, sizeof(owner_key)) == 0);
    uint64_t new_generation = 0;
    assert(rows == 130);
    // A corrupted local prefix preserves the RKVK frame but changes the key;
    // the Xbox must reject the restore and leave the fresh destination empty.
    assert(!ggml_backend_remote_attn_native_kvarn_remote_restore(
        backend, bad_local_key.data(), rows, saved_generation, &new_generation));
    assert(!ggml_backend_remote_attn_native_kvarn_has_state(backend));
    ggml_backend_remote_attn_native_kvarn_state_abort(backend);
    // Retry the valid prefix into the same new session.
    source_state.second = 0;
    assert(ggml_backend_remote_attn_native_kvarn_state_read(
        backend, [](void * user, void * data, size_t bytes) {
            auto * source = static_cast<std::pair<const std::vector<uint8_t> *, size_t> *>(user);
            if (bytes > source->first->size() - source->second) return false;
            std::memcpy(data, source->first->data() + source->second, bytes);
            source->second += bytes;
            return true;
        }, &source_state, &rows, &saved_generation, restored_key));
    assert(ggml_backend_remote_attn_native_kvarn_remote_restore(
        backend, good_key.data(), rows, saved_generation, &new_generation));
    assert(ggml_backend_remote_attn_native_kvarn_state_commit(backend, new_generation));
    assert(ggml_backend_remote_attn_native_kvarn_has_state(backend));

    // Owner rows and epoch must remain coherent for the first continuation.
    run(132, 0); run(132, 1);
    assert(ggml_backend_remote_attn_native_kvarn_can_restore(backend, 1024 + 132));
    assert(ggml_backend_remote_attn_trim(backend, 0, 1024 + 128));
    run(130, 0); run(130, 1);

    // Same-size trailer mutation must fail the owner checksum before any
    // transport restore; the fresh destination remains empty.
    assert(ggml_backend_remote_attn_reset(backend, 0));
    auto mutated_trailer = owner_state;
    mutated_trailer[mutated_trailer.size() / 2] ^= 0x01;
    std::pair<const std::vector<uint8_t> *, size_t> bad_source{&mutated_trailer, 0};
    uint8_t ignored_key[32] = {};
    assert(!ggml_backend_remote_attn_native_kvarn_state_read(
        backend, [](void * user, void * data, size_t bytes) {
            auto * source = static_cast<std::pair<const std::vector<uint8_t> *, size_t> *>(user);
            if (bytes > source->first->size() - source->second) return false;
            std::memcpy(data, source->first->data() + source->second, bytes);
            source->second += bytes;
            return true;
        }, &bad_source, &rows, &saved_generation, ignored_key));
    assert(!ggml_backend_remote_attn_native_kvarn_has_state(backend));

    // Unknown/invalid bytes after RKVK are not part of the owner frame. A
    // valid owner frame followed by junk still parses and restores under the
    // same prefix key; the server rejects the appendix separately.
    std::vector<uint8_t> bad_scpk = owner_state;
    const uint8_t junk[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    bad_scpk.insert(bad_scpk.end(), junk, junk + sizeof(junk));
    std::pair<const std::vector<uint8_t> *, size_t> junk_source{&bad_scpk, 0};
    assert(ggml_backend_remote_attn_native_kvarn_state_read(
        backend, [](void * user, void * data, size_t bytes) {
            auto * source = static_cast<std::pair<const std::vector<uint8_t> *, size_t> *>(user);
            if (bytes > source->first->size() - source->second) return false;
            std::memcpy(data, source->first->data() + source->second, bytes);
            source->second += bytes;
            return true;
        }, &junk_source, &rows, &saved_generation, ignored_key));
    assert(junk_source.second == owner_state.size());
    assert(ggml_backend_remote_attn_native_kvarn_remote_restore(
        backend, good_key.data(), rows, saved_generation, &new_generation));
    assert(ggml_backend_remote_attn_native_kvarn_state_commit(backend, new_generation));
    assert(ggml_backend_remote_attn_native_kvarn_has_state(backend));

    ggml_remote_attn_set_active(nullptr);
    ggml_backend_buffer_free(buffers); ggml_free(ctx); ggml_backend_free(cpu); ggml_backend_free(backend);
}

// Verify negotiated batching, chunk boundaries and OUT/LSE row assembly on the
// actual TCP transport. Shader numerical semantics have independent GPU oracles.
static void test_native_query_prefix_chunks() {
    storage_env native("kvarn4");
    storage_env experimental("1", "GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL");
    storage_env fusion("0", "GGML_RKVA_RANGE_FUSE");
    // Legacy layout 2/3 coverage: explicit non-bulk opt-out (bulk is default).
    storage_env legacy_bulk("0", "GGML_RKVA_NATIVE_BULK");
    storage_env legacy_async("0", "GGML_RKVA_ASYNC_LOCAL_REMOTE");
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
                ggml_backend_free(cpu); ggml_backend_free(backend);
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

static void test_native_bulk_fp8_splitzip() {
    storage_env native("kvarn4");
    storage_env experimental("1", "GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL");
    storage_env fusion("0", "GGML_RKVA_RANGE_FUSE");
    storage_env precision("128", "GGML_RKVA_NATIVE_TAIL_TOKENS");

    // The explicit synchronous option is valid; all other spellings fail closed.
    for (const char * value : {"0", "1", "2", "invalid"}) {
        storage_env overlap(value, "GGML_RKVA_ASYNC_LOCAL_REMOTE");
        storage_env bulk_env("1", "GGML_RKVA_NATIVE_BULK");
        storage_env wire_env("fp8", "GGML_RKVA_WIRE_FORMAT");
        loopback server(false, true, false, false, false, false, 0, 0, true, 7 | 8 | 16 | 32);
        auto * backend = ggml_backend_remote_attn_init("127.0.0.1", server.port);
        ggml_remote_attn_geometry geometry{1, 2, 1, 256, 512, 0, 0, 128, 3, 128, 2, 0, 0, false, false, 1.0f / 16};
        assert(ggml_backend_remote_attn_set_geometry(backend, &geometry));
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 512));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        const bool valid = !std::strcmp(value, "0") || !std::strcmp(value, "1");
        assert(ggml_backend_remote_attn_configure_range(backend, 1024, 512, GGML_TYPE_Q4_0) == valid);
        if (valid) assert(ggml_backend_remote_attn_async_enabled(backend) == !std::strcmp(value, "1"));
        ggml_backend_free(backend);
    }

    // Explicit FP8 cannot be weakened by the old strictness environment override.
    {
        storage_env strict_env("0", "GGML_RKVA_WIRE_STRICT");
        storage_env bulk_env("1", "GGML_RKVA_NATIVE_BULK");
        storage_env wire_env("fp8", "GGML_RKVA_WIRE_FORMAT");
        loopback server(false, true, false, false, false, false, 0, 0, true, 7 | 8 | 16);
        auto * backend = ggml_backend_remote_attn_init("127.0.0.1", server.port);
        ggml_remote_attn_geometry geometry{1, 2, 1, 256, 512, 0, 0, 128, 3, 128, 2, 0, 0, false, false, 1.0f / 16};
        assert(ggml_backend_remote_attn_set_geometry(backend, &geometry));
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 512));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(!ggml_backend_remote_attn_configure_range(backend, 1024, 512, GGML_TYPE_Q4_0));
        ggml_backend_free(backend);
    }
    // An old FP8 server without the RNE codec capability is also rejected.
    {
        storage_env strict_env("0", "GGML_RKVA_WIRE_STRICT");
        storage_env bulk_env("1", "GGML_RKVA_NATIVE_BULK");
        storage_env wire_env("fp8", "GGML_RKVA_WIRE_FORMAT");
        loopback server(false, true, false, false, false, false, 0, 0, true, 7 | 8 | 16 | 32 | 128);
        auto * backend = ggml_backend_remote_attn_init("127.0.0.1", server.port);
        ggml_remote_attn_geometry geometry{1, 2, 1, 256, 512, 0, 0, 128, 3, 128, 2, 0, 0, false, false, 1.0f / 16};
        assert(ggml_backend_remote_attn_set_geometry(backend, &geometry));
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 512));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(!ggml_backend_remote_attn_configure_range(backend, 1024, 512, GGML_TYPE_Q4_0));
        ggml_backend_free(backend);
    }
    // Test 2: Explicit 'auto' without FP8 server capability gracefully falls back to F16
    {
        storage_env bulk_env("1", "GGML_RKVA_NATIVE_BULK");
        storage_env wire_env("auto", "GGML_RKVA_WIRE_FORMAT");
        loopback server(false, true, false, false, false, false, 0, 0, true, 7 | 8 | 16);
        auto * backend = ggml_backend_remote_attn_init("127.0.0.1", server.port);
        ggml_remote_attn_geometry geometry{1, 2, 1, 256, 512, 0, 0, 128, 3, 128, 2, 0, 0, false, false, 1.0f / 16};
        assert(ggml_backend_remote_attn_set_geometry(backend, &geometry));
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 512));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(ggml_backend_remote_attn_configure_range(backend, 1024, 512, GGML_TYPE_Q4_0));
        ggml_backend_free(backend);
    }
    // Test 3: SplitZip wire format is deactivated and fails closed
    {
        storage_env bulk_env("1", "GGML_RKVA_NATIVE_BULK");
        storage_env wire_env("splitzip", "GGML_RKVA_WIRE_FORMAT");
        loopback server(false, true, false, false, false, false, 0, 0, true, 7 | 8 | 16 | 64);
        auto * backend = ggml_backend_remote_attn_init("127.0.0.1", server.port);
        ggml_remote_attn_geometry geometry{1, 2, 1, 256, 512, 0, 0, 128, 3, 128, 2, 0, 0, false, false, 1.0f / 16};
        assert(ggml_backend_remote_attn_set_geometry(backend, &geometry));
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 512));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(!ggml_backend_remote_attn_configure_range(backend, 1024, 512, GGML_TYPE_Q4_0));
        ggml_backend_free(backend);
    }
    // Test 4: Negotiated FP8 bulk executes successfully and clamps extreme finite values
    {
        storage_env bulk_env("1", "GGML_RKVA_NATIVE_BULK");
        storage_env wire_env("fp8", "GGML_RKVA_WIRE_FORMAT");
        const uint32_t nq = 5;
        loopback server(false, true, false, false, false, false, 0, 0, true, 7 | 8 | 16 | 32);
        {
            std::lock_guard<std::mutex> lock(server.mutex);
            server.fp8_output_byte = 0x3c;
        }
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
        // Put extreme finite float values in q to test clamping (> 57344.0f and < -57344.0f)
        std::vector<float> extreme_q(256 * 2 * nq, 0.0f);
        extreme_q[0] = 60000.0f;
        extreme_q[1] = -60000.0f;
        extreme_q[2] = 57344.0f;
        extreme_q[3] = -57344.0f;
        extreme_q[4] = 1.125f;
        extreme_q[5] = std::nextafter(1.125f, 0.0f);
        extreme_q[6] = std::ldexp(1.0f, -17);
        extreme_q[7] = std::nextafter(extreme_q[6], 1.0f);
        ggml_backend_tensor_set(q, extreme_q.data(), 0, extreme_q.size() * sizeof(float));
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
            for (size_t d = 0; d < 512; ++d) assert(out[t * 514 + d] == 1.0f);
            for (uint32_t h = 0; h < 2; ++h)
                assert(std::abs(double(out[t * 514 + 512 + h]) - std::log(double(live - nq + t + 1))) < 1e-6);
        }
        {
            std::lock_guard<std::mutex> lock(server.mutex);
            assert(server.fp8_query_prefix.size() == 8);
            assert(server.fp8_query_prefix[0] == 0x7b && server.fp8_query_prefix[1] == 0xfb);
            assert(server.fp8_query_prefix[2] == 0x7b && server.fp8_query_prefix[3] == 0xfb);
            assert(server.fp8_query_prefix[4] == 0x3c && server.fp8_query_prefix[5] == 0x3c);
            assert(server.fp8_query_prefix[6] == 0x00 && server.fp8_query_prefix[7] == 0x01);
        }
        ggml_remote_attn_set_active(nullptr);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu);
        ggml_backend_free(backend);
    }
    // Test 4b: Non-finite inputs (Inf/NaN) in FP8 fail closed
    {
        storage_env bulk_env("1", "GGML_RKVA_NATIVE_BULK");
        storage_env wire_env("fp8", "GGML_RKVA_WIRE_FORMAT");
        const uint32_t nq = 5;
        loopback server(false, true, false, false, false, false, 0, 0, true, 7 | 8 | 16 | 32);
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
        std::vector<float> bad_q(256 * 2 * nq, 0.0f);
        bad_q[0] = std::numeric_limits<float>::infinity();
        ggml_backend_tensor_set(q, bad_q.data(), 0, bad_q.size() * sizeof(float));
        const uint32_t live = nq;
        std::vector<int32_t> cells(512, -1), positions(nq);
        for (uint32_t i = 0; i < live; ++i) cells[i] = 1024 + i;
        for (uint32_t i = 0; i < nq; ++i) positions[i] = 1024 + live - nq + i;
        ggml_backend_tensor_set(ids, cells.data(), 0, cells.size() * sizeof(int32_t));
        ggml_backend_tensor_set(pos, positions.data(), 0, positions.size() * sizeof(int32_t));
        node->op_params[GGML_REMOTE_ATTN_PARAM_N_LIVE] = int32_t(live);
        assert(!ggml_remote_attn_exec(node)); // Fails closed on non-finite
        ggml_remote_attn_set_active(nullptr);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu);
        ggml_backend_free(backend);
    }
    // Test 5: Async native bulk buffer recycling across sequential ubatches
    {
        storage_env bulk_env("1", "GGML_RKVA_NATIVE_BULK");
        storage_env wire_env("fp8", "GGML_RKVA_WIRE_FORMAT");
        storage_env async_env("1", "GGML_RKVA_ASYNC_LOCAL_REMOTE");
        const uint32_t nq = 4;
        loopback server(false, true, false, false, false, false, 0, 0, true, 7 | 8 | 16 | 32);
        auto * backend = ggml_backend_remote_attn_init("127.0.0.1", server.port);
        ggml_remote_attn_geometry geometry{1, 2, 1, 256, 512, 0, 0, 128, 3, 128, 2, 0, 0, false, false, 1.0f / 16};
        assert(ggml_backend_remote_attn_set_geometry(backend, &geometry));
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 512));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(ggml_backend_remote_attn_configure_range(backend, 1024, 512, GGML_TYPE_Q4_0));
        assert(ggml_backend_remote_attn_async_enabled(backend));
        ggml_remote_attn_set_active(backend);

        for (uint32_t iter = 0; iter < 3; ++iter) {
            auto * ctx = ggml_init({2u << 20, nullptr, true});
            auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 2, nq);
            auto * k = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 512);
            auto * v = ggml_dup_tensor(ctx, k);
            auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
            auto * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nq);
            auto * original_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, nq);
            auto * original_v = ggml_dup_tensor(ctx, original_k);
            auto * node = ggml_remote_attn_range(ctx, q, k, v, ids, pos, 0, 1024, nq, geometry.kq_scale);
            ggml_remote_attn_range_set_native_sources(node, original_k, original_v);
            ggml_remote_attn_range_set_async(node);
            auto * wait_node = ggml_remote_attn_range_wait(ctx, node);

            auto * cpu = ggml_backend_cpu_init();
            auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
            assert(buffer);
            ggml_backend_buffer_clear(buffer, 0);

            const uint32_t live = (iter + 1) * nq;
            std::vector<int32_t> cells(512, -1), positions(nq);
            for (uint32_t i = 0; i < live; ++i) cells[i] = 1024 + i;
            for (uint32_t i = 0; i < nq; ++i) positions[i] = 1024 + live - nq + i;
            ggml_backend_tensor_set(ids, cells.data(), 0, cells.size() * sizeof(int32_t));
            ggml_backend_tensor_set(pos, positions.data(), 0, positions.size() * sizeof(int32_t));
            node->op_params[GGML_REMOTE_ATTN_PARAM_N_LIVE] = int32_t(live);

            assert(ggml_remote_attn_exec(node));
            if (iter == 2) {
                // A terminal graph boundary may discard an unconsumed completion.
                ggml_backend_remote_attn_drain(backend);
            } else {
                // Synchronization must not consume the completion needed by wait.
                ggml_backend_synchronize(backend);
                assert(ggml_remote_attn_exec(wait_node));
            }

            const auto * out = static_cast<const float *>((iter == 2 ? node : wait_node)->data);
            for (uint32_t t = 0; t < nq; ++t) {
                for (size_t d = 0; d < 512; ++d) assert(out[t * 514 + d] == 0.0f);
                for (uint32_t h = 0; h < 2; ++h)
                    assert(std::abs(double(out[t * 514 + 512 + h]) - std::log(double(live - nq + t + 1))) < 1e-6);
            }
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            ggml_backend_free(cpu);
        }
        ggml_remote_attn_set_active(nullptr);
        ggml_backend_free(backend);
    }
}

static void test_phase6_double_buffer_async(uint32_t nq = 4) {
    storage_env native("kvarn4");
    storage_env experimental("1", "GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL");
    storage_env fusion("0", "GGML_RKVA_RANGE_FUSE");
    storage_env precision("128", "GGML_RKVA_NATIVE_TAIL_TOKENS");
    storage_env bulk_env("1", "GGML_RKVA_NATIVE_BULK");
    storage_env wire_env("fp8", "GGML_RKVA_WIRE_FORMAT");
    storage_env async_env("1", "GGML_RKVA_ASYNC_LOCAL_REMOTE");

    // An older peer must never advertise the enlarged envelope; a matching
    // peer may advertise it only for the configured native async KVarN path.
    assert(!ggml_backend_remote_attn_kvarn_bulk_512_enabled(nullptr));
    {
        loopback extended(false, true, false, false, false, false, 0, 0, true, 7 | 8 | 16 | 32 | 512);
        auto * peer = ggml_backend_remote_attn_init("127.0.0.1", extended.port);
        ggml_remote_attn_geometry geo{1, 2, 1, 256, 512, 0, 0, 128, 3, 128, 2, 0, 0, false, false, 1.0f / 16};
        assert(ggml_backend_remote_attn_set_geometry(peer, &geo));
        assert(ggml_backend_remote_attn_connect(peer));
        assert(!ggml_backend_remote_attn_kvarn_bulk_512_enabled(peer)); // not configured
        assert(ggml_backend_remote_attn_create_range_session(peer, 0, 0, 512));
        ggml_backend_remote_attn_set_active_session(peer, 0);
        assert(ggml_backend_remote_attn_configure_range(peer, 1024, 512, GGML_TYPE_Q4_0));
        assert(ggml_backend_remote_attn_kvarn_bulk_512_enabled(peer));
        ggml_backend_free(peer);
    }

    loopback server(false, true, false, false, false, false, 0, 0, true, 7 | 8 | 16 | 32);
    auto * backend = ggml_backend_remote_attn_init("127.0.0.1", server.port);
    ggml_remote_attn_geometry geometry{1, 2, 1, 256, 512, 0, 0, 128, 3, 128, 2, 0, 0, false, false, 1.0f / 16};
    assert(ggml_backend_remote_attn_set_geometry(backend, &geometry));
    assert(ggml_backend_remote_attn_connect(backend));
    assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 512));
    ggml_backend_remote_attn_set_active_session(backend, 0);
    assert(ggml_backend_remote_attn_configure_range(backend, 1024, 512, GGML_TYPE_Q4_0));
    assert(ggml_backend_remote_attn_async_enabled(backend));
    assert(!ggml_backend_remote_attn_kvarn_bulk_512_enabled(backend));
    assert(ggml_backend_remote_attn_async_capacity(backend) == 2);
    assert(ggml_backend_remote_attn_async_in_flight(backend) == 0);
    ggml_remote_attn_set_active(backend);

    struct async_node_pair {
        ggml_tensor * node;
        ggml_tensor * wait_node;
        ggml_tensor * ids;
        ggml_tensor * pos;
        uint32_t n_queries;

        void set_inputs(uint32_t live) {
            std::vector<int32_t> cells(512, -1), positions(n_queries);
            for (uint32_t i = 0; i < live; ++i) cells[i] = 1024 + i;
            for (uint32_t i = 0; i < n_queries; ++i) positions[i] = 1024 + live - n_queries + i;
            ggml_backend_tensor_set(ids, cells.data(), 0, cells.size() * sizeof(int32_t));
            ggml_backend_tensor_set(pos, positions.data(), 0, positions.size() * sizeof(int32_t));
            node->op_params[GGML_REMOTE_ATTN_PARAM_N_LIVE] = int32_t(live);
        }
    };

    auto make_node = [&](ggml_context * ctx, uint32_t n_queries) {
        auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 2, n_queries);
        auto * k = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 512);
        auto * v = ggml_dup_tensor(ctx, k);
        auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
        auto * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_queries);
        auto * original_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, n_queries);
        auto * original_v = ggml_dup_tensor(ctx, original_k);
        auto * node = ggml_remote_attn_range(ctx, q, k, v, ids, pos, 0, 1024, n_queries, geometry.kq_scale);
        ggml_remote_attn_range_set_native_sources(node, original_k, original_v);
        ggml_remote_attn_range_set_async(node);
        auto * wait_node = ggml_remote_attn_range_wait(ctx, node);

        return async_node_pair{node, wait_node, ids, pos, n_queries};
    };

    // -------------------------------------------------------------------------
    // Subtest 1: 2 overlapping async slots in flight & capacity check
    // -------------------------------------------------------------------------
    {
        auto * ctx = ggml_init({4u << 20, nullptr, true});
        auto p0 = make_node(ctx, nq);
        auto p1 = make_node(ctx, nq);
        auto * cpu = ggml_backend_cpu_init();
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        assert(buffer);
        ggml_backend_buffer_clear(buffer, 0);
        p0.set_inputs(nq);
        p1.set_inputs(2*nq);

        server.pause_mode = true;
        server.reply_tokens = 0;

        assert(ggml_remote_attn_exec(p0.node));
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 1);

        assert(ggml_remote_attn_exec(p1.node));
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 2);

        // Server replies to slot 0
        server.grant_reply(1);
        assert(ggml_remote_attn_exec(p0.wait_node));
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 1);

        const auto * out0 = static_cast<const float *>(p0.wait_node->data);
        for (uint32_t t = 0; t < nq; ++t) {
            for (size_t d = 0; d < 512; ++d) assert(out0[t * 514 + d] == 0.0f);
            for (uint32_t h = 0; h < 2; ++h)
                assert(std::abs(double(out0[t * 514 + 512 + h]) - std::log(double(t + 1))) < 1e-6);
        }

        // Server replies to slot 1
        server.grant_reply(1);
        assert(ggml_remote_attn_exec(p1.wait_node));
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 0);

        const auto * out1 = static_cast<const float *>(p1.wait_node->data);
        for (uint32_t t = 0; t < nq; ++t) {
            for (size_t d = 0; d < 512; ++d) assert(out1[t * 514 + d] == 0.0f);
            for (uint32_t h = 0; h < 2; ++h)
                assert(std::abs(double(out1[t * 514 + 512 + h]) - std::log(double(nq + t + 1))) < 1e-6);
        }

        server.unpause();
        ggml_backend_remote_attn_drain(backend);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu);
    }

    // -------------------------------------------------------------------------
    // Subtest 2: Backpressure queue saturation blocks 3rd submit until slot freed
    // -------------------------------------------------------------------------
    {
        assert(ggml_backend_remote_attn_configure_range(backend, 1024, 512, GGML_TYPE_Q4_0));
        auto * ctx = ggml_init({6u << 20, nullptr, true});
        auto p0 = make_node(ctx, nq);
        auto p1 = make_node(ctx, nq);
        auto p2 = make_node(ctx, nq);
        auto * cpu = ggml_backend_cpu_init();
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        assert(buffer);
        ggml_backend_buffer_clear(buffer, 0);
        p0.set_inputs(nq);
        p1.set_inputs(2*nq);
        p2.set_inputs(3*nq);

        server.pause_mode = true;
        server.reply_tokens = 0;

        assert(ggml_remote_attn_exec(p0.node));
        assert(ggml_remote_attn_exec(p1.node));
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 2);

        std::atomic<bool> thread_started{false};
        std::atomic<bool> thread_done{false};
        std::thread submit_thread([&] {
            thread_started = true;
            assert(ggml_remote_attn_exec(p2.node));
            thread_done = true;
        });

        while (!thread_started.load()) { std::this_thread::yield(); }
        // Allow time for the submit thread to enter wait on async_cv
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        assert(!thread_done.load()); // Must still be blocked on backpressure
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 2);

        // Unblock slot 0 by releasing 1 reply and consuming wait_node0
        server.grant_reply(1);
        assert(ggml_remote_attn_exec(p0.wait_node)); // Frees slot 0

        // Submitting thread now acquires slot 0 and finishes
        submit_thread.join();
        assert(thread_done.load());
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 2); // p1 + p2

        // Complete remaining two
        server.grant_reply(2);
        assert(ggml_remote_attn_exec(p1.wait_node));
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 1);
        assert(ggml_remote_attn_exec(p2.wait_node));
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 0);

        server.unpause();
        ggml_backend_remote_attn_drain(backend);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu);
    }

    // -------------------------------------------------------------------------
    // Subtest 3: Backpressure timeout fail-closed
    // -------------------------------------------------------------------------
    {
        storage_env timeout_env("50", "GGML_RKVA_ASYNC_TIMEOUT_MS"); // 50ms timeout
        assert(ggml_backend_remote_attn_create_range_session(backend, 1, 1, 512));
        ggml_backend_remote_attn_set_active_session(backend, 1);
        assert(ggml_backend_remote_attn_configure_range(backend, 1024, 512, GGML_TYPE_Q4_0));

        auto * ctx = ggml_init({6u << 20, nullptr, true});
        auto p0 = make_node(ctx, nq);
        auto p1 = make_node(ctx, nq);
        auto p2 = make_node(ctx, nq);
        auto * cpu = ggml_backend_cpu_init();
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        assert(buffer);
        ggml_backend_buffer_clear(buffer, 0);
        p0.set_inputs(nq);
        p1.set_inputs(2*nq);
        p2.set_inputs(3*nq);

        server.pause_mode = true;
        server.reply_tokens = 0;

        assert(ggml_remote_attn_exec(p0.node));
        assert(ggml_remote_attn_exec(p1.node));
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 2);

        // Submitting 3rd node with 0 free slots and timeout=50ms must fail-closed
        const auto t0 = std::chrono::steady_clock::now();
        assert(!ggml_remote_attn_exec(p2.node)); // Must fail!
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        assert(elapsed >= 35); // Confirms wait occurred
        assert(ggml_backend_remote_attn_failed(backend));

        server.unpause();
        ggml_backend_remote_attn_drain(backend);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu);
    }
    ggml_remote_attn_set_active(nullptr);
    ggml_backend_free(backend);

    // -------------------------------------------------------------------------
    // Subtest 4: Control Barrier - Drain
    // -------------------------------------------------------------------------
    {
        loopback server_drain(false, true, false, false, false, false, 0, 0, true, 7 | 8 | 16 | 32);
        auto * backend = ggml_backend_remote_attn_init("127.0.0.1", server_drain.port);
        assert(ggml_backend_remote_attn_set_geometry(backend, &geometry));
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 512));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(ggml_backend_remote_attn_configure_range(backend, 1024, 512, GGML_TYPE_Q4_0));
        ggml_remote_attn_set_active(backend);

        auto * ctx = ggml_init({4u << 20, nullptr, true});
        auto p0 = make_node(ctx, nq);
        auto p1 = make_node(ctx, nq);
        auto * cpu = ggml_backend_cpu_init();
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        assert(buffer);
        ggml_backend_buffer_clear(buffer, 0);
        p0.set_inputs(nq);
        p1.set_inputs(2*nq);

        server_drain.pause_mode = true;
        server_drain.reply_tokens = 0;

        assert(ggml_remote_attn_exec(p0.node));
        assert(ggml_remote_attn_exec(p1.node));
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 2);

        std::thread unpause_thr([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
            server_drain.unpause();
        });
        ggml_backend_remote_attn_drain(backend);
        unpause_thr.join();
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 0);

        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu);
        ggml_remote_attn_set_active(nullptr);
        ggml_backend_free(backend);
    }

    // -------------------------------------------------------------------------
    // Subtest 5: Control Barrier - Epoch Invalidation & Safe Failure
    // -------------------------------------------------------------------------
    {
        loopback server_inval(false, true, false, false, false, false, 0, 0, true, 7 | 8 | 16 | 32);
        auto * backend = ggml_backend_remote_attn_init("127.0.0.1", server_inval.port);
        assert(ggml_backend_remote_attn_set_geometry(backend, &geometry));
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 512));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(ggml_backend_remote_attn_configure_range(backend, 1024, 512, GGML_TYPE_Q4_0));
        ggml_remote_attn_set_active(backend);

        auto * ctx = ggml_init({2u << 20, nullptr, true});
        auto p0 = make_node(ctx, nq);
        auto p1 = make_node(ctx, nq);
        auto * cpu = ggml_backend_cpu_init();
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        assert(buffer);
        ggml_backend_buffer_clear(buffer, 0);
        p0.set_inputs(nq);
        p1.set_inputs(2*nq);

        server_inval.set_pause(true, 0);
        assert(ggml_remote_attn_exec(p0.node));
        assert(ggml_remote_attn_exec(p1.node));
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 2);
        server_inval.unpause();

        // Invalidate epoch before consuming completion
        ggml_backend_remote_attn_invalidate_range(backend);
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 0);

        // Waiting on invalidated slot must fail safely
        assert(!ggml_remote_attn_exec(p0.wait_node));
        assert(!ggml_remote_attn_exec(p1.wait_node));

        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu);
        ggml_remote_attn_set_active(nullptr);
        ggml_backend_free(backend);
    }

    // -------------------------------------------------------------------------
    // Subtest 6: Control Barrier - Trim/Rewind Drains Active Slots
    // -------------------------------------------------------------------------
    {
        loopback server_trim(false, true, false, false, false, false, 0, 0, true, 7 | 8 | 16 | 32);
        auto * backend = ggml_backend_remote_attn_init("127.0.0.1", server_trim.port);
        assert(ggml_backend_remote_attn_set_geometry(backend, &geometry));
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 512));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(ggml_backend_remote_attn_configure_range(backend, 1024, 512, GGML_TYPE_Q4_0));
        ggml_remote_attn_set_active(backend);

        auto * ctx = ggml_init({2u << 20, nullptr, true});
        auto p0 = make_node(ctx, nq);
        auto p1 = make_node(ctx, nq);
        auto * cpu = ggml_backend_cpu_init();
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        assert(buffer);
        ggml_backend_buffer_clear(buffer, 0);
        p0.set_inputs(nq);
        p1.set_inputs(2*nq);

        server_trim.pause_mode = true;
        server_trim.reply_tokens = 0;

        assert(ggml_remote_attn_exec(p0.node));
        assert(ggml_remote_attn_exec(p1.node));
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 2);

        std::thread unpause_thr([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
            server_trim.unpause();
        });
        assert(ggml_backend_remote_attn_trim(backend, 0, 1024+nq));
        unpause_thr.join();
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 0);

        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu);
        ggml_remote_attn_set_active(nullptr);
        ggml_backend_free(backend);
    }

    // -------------------------------------------------------------------------
    // Subtest 7: A128 and B8 Pipeline - Sequential Append and Confirmation
    // -------------------------------------------------------------------------
    {
        loopback server_pipe(false, true, false, false, false, false, 0, 0, true, 7 | 8 | 16 | 32);
        auto * backend = ggml_backend_remote_attn_init("127.0.0.1", server_pipe.port);
        assert(ggml_backend_remote_attn_set_geometry(backend, &geometry));
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 512));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(ggml_backend_remote_attn_configure_range(backend, 1024, 512, GGML_TYPE_Q4_0));
        ggml_remote_attn_set_active(backend);

        auto * ctx = ggml_init({64u << 20, nullptr, true});
        const uint32_t nq_a = 128;
        const uint32_t nq_b = nq > 9 ? 128 : 8;
        auto p_a = make_node(ctx, nq_a);
        auto p_b = make_node(ctx, nq_b);
        auto * cpu = ggml_backend_cpu_init();
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        assert(buffer);
        ggml_backend_buffer_clear(buffer, 0);

        p_a.set_inputs(128);
        p_b.set_inputs(128 + nq_b);

        server_pipe.set_pause(true, 0);

        // Submit A (128 tokens)
        assert(ggml_remote_attn_exec(p_a.node));
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 1);

        // Submit B (8 tokens) while A is still in-flight
        assert(ggml_remote_attn_exec(p_b.node));
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 2);

        // Allow server to process both
        server_pipe.unpause();

        // Consume A
        assert(ggml_remote_attn_exec(p_a.wait_node));
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 1);

        // Consume B
        assert(ggml_remote_attn_exec(p_b.wait_node));
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 0);

        ggml_backend_remote_attn_drain(backend);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu);
        ggml_remote_attn_set_active(nullptr);
        ggml_backend_free(backend);
    }
}

static void test_native_tail_serialization() {
    storage_env producer("cpu", "GGML_RKVA_NATIVE_PRODUCER");
    // Isolate original-domain tail serialization from record quantization.
    const auto store = +[](ggml_tensor *) { return true; };
    constexpr size_t width = 256;
    const float values[] = {
        0.0f, -0.0f, 0x1p-149f, -0x1p-149f, 0x1p-25f, 0x1p-24f,
        0x1.002p0f, 0x1.006p0f, -0x1.002p0f, 65504.0f, 65520.0f,
        -65520.0f, 0x1.fffffep127f, -0x1.fffffep127f,
    };
    for (const uint32_t origin : {0u, 128u}) {
    for (const uint32_t tail : {0u, 128u}) {
    for (const bool compact : {false, true}) {
        rkva_kvarn_producer owner(origin, 1024, 1, 3, store, tail, compact);
        std::vector<float> all_k, all_v;
        std::vector<uint8_t> packet;
        uint32_t before = 0;
        for (const uint32_t count : {1u, 7u, 9u, 127u, 128u, 129u, 255u, 256u}) {
            std::vector<float> k(count * width), v(k.size());
            std::vector<int32_t> pos(count);
            for (size_t i = 0; i < k.size(); ++i) {
                k[i] = values[(before * width + i) % (sizeof(values) / sizeof(*values))];
                v[i] = values[(before * width + i + 5) % (sizeof(values) / sizeof(*values))];
            }
            for (uint32_t row = 0; row < count; ++row) pos[row] = origin + before + row;
            all_k.insert(all_k.end(), k.begin(), k.end());
            all_v.insert(all_v.end(), v.begin(), v.end());
            const bool reset = before == 0 || count == 129;
            assert(owner.append(k.data(), v.data(), pos.data(), count, reset, packet));
            const uint32_t after = before + count;
            const bool omitted = compact && origin >= 128 && tail == 128;
            const uint32_t record_begin = reset ? 0 : before / 128;
            const size_t records_bytes = (after / 128 - record_begin) * 2 * 8960 * 2;
            const size_t stage_bytes = omitted ? 0 : (reset ? 384 : count) * width * 2 * 2;
            const uint32_t tail_rows = tail ? (reset ? std::min(after, tail) : count) : 0;
            const size_t tail_offset = sizeof(rkva_kvarn_range_upload) + records_bytes + stage_bytes;
            assert(packet.size() == tail_offset + tail_rows * width * 2 * 2);
            std::vector<uint8_t> expected;
            const size_t first = (reset ? after - tail_rows : before) * width;
            for (const auto * source : {&all_k, &all_v}) {
                for (size_t i = 0; i < tail_rows * width; ++i) {
                    const ggml_fp16_t half = ggml_fp32_to_fp16((*source)[first + i]);
                    const auto * bytes = reinterpret_cast<const uint8_t *>(&half);
                    expected.insert(expected.end(), bytes, bytes + sizeof(half));
                }
            }
            assert(std::equal(expected.begin(), expected.end(), packet.begin() + tail_offset));
            assert(owner.rows() == after);
            before = after;
        }
        const auto saved = packet;
        std::vector<float> bad(width, 0.0f), good(width, 0.0f);
        const int32_t pos = origin + before;
        for (const uint32_t bits : {0x7f800000u, 0xff800000u, 0x7fc00001u}) {
            std::memcpy(bad.data(), &bits, sizeof(bits));
            assert(!owner.append(bad.data(), good.data(), &pos, 1, false, packet));
            assert(!owner.append(good.data(), bad.data(), &pos, 1, false, packet));
            assert(owner.rows() == before && packet == saved);
        }
    }
    }
    }
}

static void test_native_physical_peer() {
    const char * peer = std::getenv("GGML_RKVA_TEST_PEER");
    if (!peer || !*peer) return;
    storage_env native("kvarn4");
    storage_env experimental("1", "GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL");
    storage_env fusion("0", "GGML_RKVA_RANGE_FUSE");
    storage_env bulk("1", "GGML_RKVA_NATIVE_BULK");
    storage_env overlap("1", "GGML_RKVA_ASYNC_LOCAL_REMOTE");
    storage_env mode("fine", "GGML_RKVA_RANGE_KERNEL");
    storage_env tail("128", "GGML_RKVA_NATIVE_TAIL_TOKENS");
    constexpr uint32_t nq = 256, origin = 96256, capacity = 20000, heads = 24, kv_heads = 4;
    for (const char * wire_name : {"fp8", "f32"}) {
    const char * selected_wire = std::getenv("GGML_RKVA_TEST_PEER_WIRE");
    if (selected_wire && std::strcmp(selected_wire, wire_name)) continue;
    storage_env wire(wire_name, "GGML_RKVA_WIRE_FORMAT");
    auto * backend = ggml_backend_remote_attn_init(peer, 50054);
    ggml_remote_attn_geometry geo{1, heads, kv_heads, 256, capacity, 0, 0, 128, 3, 128, 2, 0,
        GGML_REMOTE_ATTN_DOMAIN_AUTO, false, false, 1.0f/16};
    assert(ggml_backend_remote_attn_set_geometry(backend, &geo));
    assert(ggml_backend_remote_attn_connect(backend));
    assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, capacity));
    ggml_backend_remote_attn_set_active_session(backend, 0);
    assert(ggml_backend_remote_attn_configure_range(backend, origin, capacity, GGML_TYPE_Q4_0));
    ggml_remote_attn_set_active(backend);
    auto * ctx = ggml_init({4u << 20, nullptr, true});
    auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, heads, nq);
    auto * k = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256 * kv_heads, capacity);
    auto * v = ggml_dup_tensor(ctx, k);
    auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, capacity);
    auto * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nq);
    auto * original_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, kv_heads, nq);
    auto * original_v = ggml_dup_tensor(ctx, original_k);
    auto * node = ggml_remote_attn_range(ctx, q, k, v, ids, positions, 0, origin, nq, geo.kq_scale);
    ggml_remote_attn_range_set_native_sources(node, original_k, original_v);
    ggml_remote_attn_range_set_async(node);
    auto * wait = ggml_remote_attn_range_wait(ctx, node);
    auto * cpu = ggml_backend_cpu_init();
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    assert(buffer);
    ggml_backend_buffer_clear(buffer, 0);
    std::vector<float> queries(size_t(nq)*heads*256, 0), keys(size_t(nq)*kv_heads*256, 0), values(keys.size(), 0);
    for (size_t row = 0; row < size_t(nq)*heads; ++row) queries[row*256] = 1.0f;
    for (size_t row = 0; row < size_t(nq)*kv_heads; ++row) { keys[row*256] = 0.25f; values[row*256] = 1.0f; }
    ggml_backend_tensor_set(q, queries.data(), 0, queries.size()*sizeof(float));
    ggml_backend_tensor_set(original_k, keys.data(), 0, keys.size()*sizeof(float));
    ggml_backend_tensor_set(original_v, values.data(), 0, values.size()*sizeof(float));
    std::vector<int32_t> cells(capacity, -1), pos(nq);
    auto store = reinterpret_cast<ggml_backend_cpu_kvarn_store_direct_v1_t>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu)), "ggml_backend_cpu_kvarn_store_direct_v1"));
    assert(store);
    rkva_kvarn_producer golden(origin, 128, kv_heads, 3, store, 128, true);
    std::vector<int32_t> golden_pos(128);
    for (uint32_t i = 0; i < 128; ++i) golden_pos[i] = origin + i;
    std::vector<uint8_t> golden_packet;
    assert(golden.append(keys.data(), values.data(), golden_pos.data(), 128, true, golden_packet));
    std::vector<double> body_k(256), body_v(256);
    const auto half = [](const uint8_t * p) {
        ggml_fp16_t bits; std::memcpy(&bits, p, sizeof(bits)); return ggml_fp16_to_fp32(bits);
    };
    for (size_t dim = 0; dim < 256; ++dim) {
        const size_t local_dim = dim % 128, slice = dim / 128;
        const auto * kr = golden_packet.data() + sizeof(rkva_kvarn_range_upload) + slice * 8960;
        const auto * vr = kr + kv_heads * 2 * 8960;
        const uint8_t k_quant = kr[local_dim * 64] & 15;
        const uint8_t v_quant = (vr[local_dim / 2] >> ((local_dim & 1) * 4)) & 15;
        body_k[dim] = (float(k_quant)*half(kr+8192+local_dim*2)+half(kr+8448+local_dim*2))*half(kr+8704);
        body_v[dim] = (float(v_quant)*half(vr+8192)+half(vr+8448))*half(vr+8704+local_dim*2);
    }
    // Materialize the actual stored records in F64. KVarN is lossy even for
    // this simple input; an oracle over the original V would falsely fail F32.
    for (auto * vector : {&body_k, &body_v}) {
        for (size_t stride = 1; stride < 256; stride *= 2)
            for (size_t base = 0; base < 256; base += 2*stride)
                for (size_t d = 0; d < stride; ++d) {
                    double a = (*vector)[base+d], b = (*vector)[base+d+stride];
                    (*vector)[base+d] = a+b; (*vector)[base+d+stride] = a-b;
                }
        for (double & value : *vector) value /= 16;
    }
    double max_o = 0, max_lse = 0;
    const uint32_t total_rows = !std::strcmp(wire_name, "fp8") ? 16384 : nq;
    for (uint32_t live = nq; live <= total_rows; live += nq) {
        for (uint32_t row = live-nq; row < live; ++row) cells[row] = origin + row;
        for (uint32_t row = 0; row < nq; ++row) pos[row] = origin + live - nq + row;
        ggml_backend_tensor_set(ids, cells.data(), 0, cells.size()*sizeof(int32_t));
        ggml_backend_tensor_set(positions, pos.data(), 0, pos.size()*sizeof(int32_t));
        node->op_params[GGML_REMOTE_ATTN_PARAM_N_LIVE] = int32_t(live);
        assert(ggml_remote_attn_exec(node));
        assert(ggml_remote_attn_exec(wait));
        const auto * out = static_cast<const float *>(wait->data);
        for (size_t row = 0; row < nq; ++row) {
            const size_t count = live-nq+row+1, tail_count = std::min<size_t>(count,128), body_count = count-tail_count;
            const double body_score = body_k[0]/16, tail_score = 1.0/64;
            const double maximum = std::max(body_score,tail_score);
            const double wb = body_count*std::exp(body_score-maximum), wt = tail_count*std::exp(tail_score-maximum);
            for (size_t h = 0; h < heads; ++h) {
                for (size_t d = 0; d < 256; ++d) {
                    const double expected = !std::strcmp(wire_name,"fp8") ? (d==0 ? 1.0 : 0.0) :
                        (wb*body_v[d]+wt*(d==0 ? 1.0 : 0.0))/(wb+wt);
                    const double error = std::abs(double(out[row*(heads*256+heads)+h*256+d]) - expected);
                    max_o = std::max(max_o, error);
                    if (error >= 1e-4) std::fprintf(stderr,
                        "peer OUT mismatch wire=%s live=%u row=%zu head=%zu dim=%zu value=%.9g error=%.9g\n",
                        wire_name, live, row, h, d, out[row*(heads*256+heads)+h*256+d], error);
                    assert(error < 1e-4);
                }
                const double expected = maximum + std::log(wb+wt);
                const double error = std::abs(double(out[row*(heads*256+heads)+heads*256+h])-expected);
                max_lse = std::max(max_lse, error); assert(error < 1e-4);
            }
        }
    }
    std::printf("Physical peer %s: wire=%s rows=%u, ub256, max_abs_O=%.9g max_abs_LSE=%.9g stats=%s\n",
        peer, wire_name, total_rows, max_o, max_lse, ggml_backend_remote_attn_stats_json(backend));
    std::fflush(stdout);
    if (!std::strcmp(wire_name, "f32")) {
        // Finite wire inputs can overflow the GPU score. A valid old body must
        // not hide the invalid new wave/split as though it were empty.
        for (size_t row = 0; row < size_t(nq)*heads; ++row)
            queries[row*256] = std::numeric_limits<float>::max();
        for (size_t row = 0; row < size_t(nq)*kv_heads; ++row) keys[row*256] = 65504.0f;
        for (uint32_t row = nq; row < 2*nq; ++row) cells[row] = origin + row;
        for (uint32_t row = 0; row < nq; ++row) pos[row] = origin + nq + row;
        ggml_backend_tensor_set(q, queries.data(), 0, queries.size()*sizeof(float));
        ggml_backend_tensor_set(original_k, keys.data(), 0, keys.size()*sizeof(float));
        ggml_backend_tensor_set(ids, cells.data(), 0, cells.size()*sizeof(int32_t));
        ggml_backend_tensor_set(positions, pos.data(), 0, pos.size()*sizeof(int32_t));
        node->op_params[GGML_REMOTE_ATTN_PARAM_N_LIVE] = int32_t(2*nq);
        assert(ggml_remote_attn_exec(node));
        assert(!ggml_remote_attn_exec(wait));
        assert(ggml_backend_remote_attn_failed(backend));
        std::puts("Physical peer finite F32 score overflow: rejected before publication");
    }
    ggml_backend_remote_attn_drain(backend);
    ggml_backend_buffer_free(buffer); ggml_free(ctx); ggml_backend_free(cpu);
    ggml_remote_attn_set_active(nullptr); ggml_backend_free(backend);
    }
}

static void test_cuda_fp8_serving_graph() {
    ggml_backend_load_all();
    auto * reg = ggml_backend_reg_by_name("CUDA");
    if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
        std::puts("CUDA serving graph: skipped (CUDA backend unavailable)");
        return;
    }
    storage_env native("kvarn4");
    storage_env experimental("1", "GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL");
    storage_env fusion("0", "GGML_RKVA_RANGE_FUSE");
    storage_env bulk("1", "GGML_RKVA_NATIVE_BULK");
    storage_env wire("fp8", "GGML_RKVA_WIRE_FORMAT");
    storage_env acceleration("1", "GGML_RKVA_CUDA_FP8");
    auto * gpu = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
    assert(gpu);
    auto * cpu = ggml_backend_cpu_init();
    constexpr uint32_t n_head = 24, n_head_kv = 4, head_dim = 256, nq = 9;
    constexpr uint32_t capacity = 512, origin = 1024;
    constexpr size_t out_width = size_t(n_head) * head_dim;
    constexpr size_t kv_width = size_t(n_head_kv) * head_dim;
    constexpr size_t row_width = out_width + n_head;
    constexpr float lse_head_delta = 0.125f;
    // The reference values are literal E5M2 values, independent of the codec
    // under test. Include both signs, signed zeros, a subnormal and extremes.
    const uint8_t fp8_codes[] = {0x00, 0x80, 0x3c, 0xbc, 0x01, 0x04, 0x7b, 0xfb};
    const float fp8_values[] = {0.0f, -0.0f, 1.0f, -1.0f, 0.0000152587890625f,
                              0.00006103515625f, 57344.0f, -57344.0f};
    const auto fixture_index = [](size_t row, size_t head, size_t dimension) {
        return (row * 3 + head * 5 + dimension * 7) % 8;
    };
    for (bool invalidation_case : {false, true}) {
    for (bool q_on_gpu : {false, true}) {
    for (bool async : {false, true}) {
        if (invalidation_case && !async) continue;
        storage_env overlap(async ? "1" : "0", "GGML_RKVA_ASYNC_LOCAL_REMOTE");
        loopback server(false, true, false, false, false, false, 0, 0, true, 7 | 8 | 16 | 32);
        {
            std::lock_guard<std::mutex> lock(server.mutex);
            server.fp8_output_data.resize(size_t(nq) * out_width);
            for (size_t row = 0; row < nq; ++row)
                for (size_t head = 0; head < n_head; ++head)
                    for (size_t d = 0; d < head_dim; ++d)
                        server.fp8_output_data[row * out_width + head * head_dim + d] = fp8_codes[fixture_index(row, head, d)];
            server.lse_head_delta = lse_head_delta;
        }
        auto * backend = ggml_backend_remote_attn_init("127.0.0.1", server.port);
        ggml_remote_attn_geometry geometry{1, n_head, n_head_kv, head_dim, capacity, 0, 0, 128, 3, 128, 2, 0, 0, false, false, 1.0f / 16};
        assert(ggml_backend_remote_attn_set_geometry(backend, &geometry));
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, capacity));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(ggml_backend_remote_attn_configure_range(backend, origin, capacity, GGML_TYPE_Q4_0));
        ggml_remote_attn_set_active(backend);
        assert(ggml_backend_remote_attn_cuda_enabled());

        auto * ctx = ggml_init({4u << 20, nullptr, true});
        auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_head, nq);
        auto * k = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, kv_width, capacity);
        auto * v = ggml_dup_tensor(ctx, k);
        auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, capacity);
        auto * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nq);
        auto * original_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_head_kv, nq);
        auto * original_v = ggml_dup_tensor(ctx, original_k);
        auto * node = ggml_remote_attn_range(ctx, q, k, v, ids, positions, 0, origin, nq, geometry.kq_scale);
        ggml_remote_attn_range_set_native_sources(node, original_k, original_v);
        ggml_tensor * complete = node;
        if (async) {
            ggml_remote_attn_range_set_async(node);
            complete = ggml_remote_attn_range_wait(ctx, node);
            assert(complete->view_src == node);
        }
        // The oracle inspects raw OUT after the terminal consumer. Keep its
        // storage alive: otherwise SCALE/ADD may legally reuse this allocation
        // in place and the later read observes the consumer's result.
        ggml_set_output(node);
        ggml_set_output(complete);
        auto * local_input = ggml_dup_tensor(ctx, complete);
        auto * local_branch = ggml_scale(ctx, local_input, 3.0f);
        auto * consumer = ggml_add(ctx, ggml_scale(ctx, complete, 2.0f), local_branch);
        auto * graph = ggml_new_graph(ctx);
        // The local CUDA branch runs between submit and WAIT. Its consumer
        // must observe the validated remote result after both branches retire.
        ggml_build_forward_expand(graph, node);
        ggml_build_forward_expand(graph, local_branch);
        ggml_build_forward_expand(graph, consumer);
        ggml_backend_t backends[] = {gpu, cpu};
        auto * sched = ggml_backend_sched_new(backends, nullptr, 2, 1024, false, true);
        for (auto * tensor : {original_k, original_v, node, complete, local_input, local_branch, consumer})
            ggml_backend_sched_set_tensor_backend(sched, tensor, gpu);
        ggml_backend_sched_set_tensor_backend(sched, q, q_on_gpu ? gpu : cpu);
        for (auto * tensor : {k, v, ids, positions})
            ggml_backend_sched_set_tensor_backend(sched, tensor, cpu);
        assert(ggml_backend_sched_alloc_graph(sched, graph));
        assert(ggml_backend_sched_get_n_splits(sched) >= (async ? 3 : 2));
        std::vector<float> queries(out_width * nq, 1.125f), originals(kv_width * nq, 0.0f);
        ggml_backend_tensor_set(q, queries.data(), 0, queries.size() * sizeof(float));
        ggml_backend_tensor_set(original_k, originals.data(), 0, originals.size() * sizeof(float));
        ggml_backend_tensor_set(original_v, originals.data(), 0, originals.size() * sizeof(float));
        const std::vector<float> local_values(row_width * nq, 1.0f);
        ggml_backend_tensor_set(local_input, local_values.data(), 0, local_values.size() * sizeof(float));
        std::vector<int32_t> cells(capacity, -1), pos(nq);
        for (uint32_t i = 0; i < nq; ++i) cells[i] = pos[i] = int32_t(origin + i);
        ggml_backend_tensor_set(ids, cells.data(), 0, cells.size() * sizeof(int32_t));
        ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(int32_t));
        assert(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_sched_synchronize(sched);
        assert(node->src[0] == q); // scheduler must not introduce an F32 D2H copy
        assert(node->src[1] == k && node->src[3] == ids); // metadata stays on CPU
        std::vector<float> output(row_width * nq), raw_output(output.size());
        ggml_backend_tensor_get(consumer, output.data(), 0, output.size() * sizeof(float));
        ggml_backend_tensor_get(complete, raw_output.data(), 0, raw_output.size() * sizeof(float));
        for (size_t row = 0; row < nq; ++row) {
            for (size_t head = 0; head < n_head; ++head) {
                for (size_t d = 0; d < head_dim; ++d) {
                    const float expected = fp8_values[fixture_index(row, head, d)];
                    const size_t index = row * row_width + head * head_dim + d;
                    if (raw_output[index] != expected || std::signbit(raw_output[index]) != std::signbit(expected)) {
                        std::fprintf(stderr, "CUDA serving raw mismatch: gpu_q=%d async=%d row=%zu head=%zu d=%zu index=%zu code=0x%02x actual=%a expected=%a consumer=%a node=%p complete=%p consumer_data=%p\n",
                            int(q_on_gpu), int(async), row, head, d, index,
                            unsigned(fp8_codes[fixture_index(row, head, d)]), double(raw_output[index]),
                            double(expected), double(output[index]), node->data, complete->data, consumer->data);
                    }
                    assert(raw_output[index] == expected);
                    assert(std::signbit(raw_output[index]) == std::signbit(expected));
                    assert(output[index] == 2.0f * expected + 3.0f);
                }
                const float expected_lse = float(std::log(double(row + 1))) + float(head) * lse_head_delta;
                const size_t index = row * row_width + out_width + head;
                assert(raw_output[index] == expected_lse);
                assert(std::abs(output[index] - (2.0f * expected_lse + 3.0f)) < 1e-6f);
            }
        }
        const std::string stats = ggml_backend_remote_attn_stats_json(backend);
        assert(stats.find(q_on_gpu ? "\"cuda_pack_calls\":1" : "\"cuda_pack_calls\":0") != std::string::npos);
        assert(stats.find("\"cuda_unpack_calls\":1") != std::string::npos);
        const auto q_bytes_avoided = q_on_gpu ? nq * out_width * sizeof(float) : 0;
        assert(stats.find("\"q_f32_readback_bytes_avoided\":" + std::to_string(q_bytes_avoided)) != std::string::npos);
        assert(stats.find("\"out_f32_upload_bytes_avoided\":" + std::to_string(nq * out_width * sizeof(float))) != std::string::npos);
        if (invalidation_case) {
            // A control boundary must retire the worker's CUDA unpack before
            // graph storage can be overwritten, even while its reply is held.
            assert(ggml_backend_remote_attn_trim(backend, 0, origin));
            server.set_pause(true, 0);
            auto * submit_graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(submit_graph, node);
            assert(ggml_backend_graph_compute(gpu, submit_graph) == GGML_STATUS_SUCCESS);
            assert(ggml_backend_remote_attn_async_in_flight(backend) == 1);
            // A merely PENDING slot can be cancelled without an RPC. Observe
            // the server holding the actual response before testing the drain.
            assert(server.wait_until_reply_paused());

            std::promise<void> control_started;
            auto control_ready = control_started.get_future();
            auto invalidate = std::async(std::launch::async, [&] {
                control_started.set_value();
                ggml_backend_remote_attn_invalidate_range(backend);
            });
            assert(control_ready.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
            assert(invalidate.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);
            server.unpause();
            invalidate.get();
            assert(ggml_backend_remote_attn_async_in_flight(backend) == 0);
            assert(!ggml_remote_attn_exec(complete)); // retired ticket cannot publish
            { std::lock_guard<std::mutex> lock(server.mutex); assert(server.attention_calls == 2); }

            const std::vector<float> replacement(row_width * nq, -17.0f);
            ggml_backend_tensor_set(complete, replacement.data(), 0, replacement.size() * sizeof(float));
            // Keep independent GPU work after the overwrite. No old worker
            // operation may write this storage after invalidate has returned.
            auto * local_graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(local_graph, local_branch);
            assert(ggml_backend_graph_compute(gpu, local_graph) == GGML_STATUS_SUCCESS);
            ggml_backend_synchronize(gpu);
            // Retire the codec's private stream as well, keeping all graph
            // tensors alive until every possible old writer has disappeared.
            ggml_remote_attn_set_active(nullptr);
            ggml_backend_free(backend);
            backend = nullptr;
            ggml_backend_tensor_get(complete, raw_output.data(), 0, raw_output.size() * sizeof(float));
            assert(raw_output == replacement);
        } else {
            queries[0] = std::numeric_limits<float>::quiet_NaN();
            ggml_backend_tensor_set(q, queries.data(), 0, queries.size() * sizeof(float));
            for (uint32_t i = nq; i < 2 * nq; ++i) cells[i] = int32_t(origin + i);
            for (uint32_t i = 0; i < nq; ++i) pos[i] = int32_t(origin + nq + i);
            ggml_backend_tensor_set(ids, cells.data(), 0, cells.size() * sizeof(int32_t));
            ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(int32_t));
            node->op_params[GGML_REMOTE_ATTN_PARAM_N_LIVE] = 2 * nq;
            assert(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_FAILED);
            assert(ggml_backend_remote_attn_failed(backend));
            { std::lock_guard<std::mutex> lock(server.mutex); assert(server.attention_calls == 1); }
            // A false CUDA helper result is conservatively non-reusable, even
            // though this numerical failure actually retired successfully.
            if (q_on_gpu) assert(!ggml_backend_remote_attn_connect(backend));
        }
        ggml_backend_remote_attn_drain(backend);
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        ggml_remote_attn_set_active(nullptr);
        if (backend) ggml_backend_free(backend);
    }
    }
    }
    ggml_backend_free(cpu);
    ggml_backend_free(gpu);
    std::puts("CUDA FP8 serving graph: Hq24/Hkv4/D256/nq9, OUT/LSE, sync/async, mixed operands and invalidation/overwrite retirement passed");
}

static void test_cuda_fp8_prefill_mode_graph() {
    ggml_backend_load_all();
    auto * reg = ggml_backend_reg_by_name("CUDA");
    if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
        std::puts("CUDA prefill codec graph: skipped (CUDA backend unavailable)");
        return;
    }
    storage_env native("kvarn4"), experimental("1", "GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL");
    storage_env fusion("0", "GGML_RKVA_RANGE_FUSE"), bulk("1", "GGML_RKVA_NATIVE_BULK");
    storage_env wire("fp8", "GGML_RKVA_WIRE_FORMAT");
    auto * gpu = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
    auto * cpu = ggml_backend_cpu_init();
    assert(gpu && cpu);
    constexpr uint32_t heads = 24, kv_heads = 4, dim = 256, capacity = 512, origin = 1024;
    constexpr size_t width = size_t(heads) * dim, kv_width = size_t(kv_heads) * dim, row_width = width + heads;
    const uint8_t codes[] = {0x00, 0x80, 0x3c, 0xbc, 0x01, 0x04, 0x7b, 0xfb};
    const float values[] = {0.0f, -0.0f, 1.0f, -1.0f, 0.0000152587890625f,
                            0.00006103515625f, 57344.0f, -57344.0f};
    // Break the eight-row period too, so using segment0's bytes for segment1
    // or2 is detected by the complete query capture below.
    const auto pattern = [](size_t row, size_t head, size_t d) { return (row * 3 + row / 8 + head * 5 + d * 7) % 8; };
    const auto counter = [](const std::string & json, const char * key) {
        const auto pos = json.find(std::string("\"") + key + "\":");
        assert(pos != std::string::npos);
        return std::stoull(json.substr(pos + std::strlen(key) + 3));
    };
    for (const char * policy : {"prefill", "0", "1", "mapped", static_cast<const char *>(nullptr)}) {
    storage_env acceleration(policy, "GGML_RKVA_CUDA_FP8");
    const bool prefill_policy = policy && !std::strcmp(policy, "prefill");
    for (bool async : {false, true}) {
    for (bool q_on_gpu : {false, true}) {
        storage_env overlap(async ? "1" : "0", "GGML_RKVA_ASYNC_LOCAL_REMOTE");
        loopback server(false, true, false, false, false, false, 0, 0, true, 7 | 8 | 16 | 32);
        auto * backend = ggml_backend_remote_attn_init("127.0.0.1", server.port);
        ggml_remote_attn_geometry geometry{1, heads, kv_heads, dim, capacity, 0, 0, 128, 3, 128, 2, 0, 0, false, false, 1.0f / 16};
        assert(ggml_backend_remote_attn_set_geometry(backend, &geometry));
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, capacity));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(ggml_backend_remote_attn_configure_range(backend, origin, capacity, GGML_TYPE_Q4_0));
        ggml_remote_attn_set_active(backend);
        uint32_t live = 0;
        uint64_t packs = 0, unpacks = 0, creates = 0, releases = 0, small_packs = 0, small_unpacks = 0;
        bool context_live = false;
        // One backend/slot pool crosses both directions of the policy boundary.
        // The17-query RPC has8+8+1 segments but exactly one GPU pack/unpack.
        for (uint32_t nq : {17u, 1u, 9u, 10u, 1u}) {
            const bool cuda_range = prefill_policy ? nq > 9 : !(policy && !std::strcmp(policy, "0"));
            auto * ctx = ggml_init({4u << 20, nullptr, true});
            auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, heads, nq);
            auto * k = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, kv_width, capacity);
            auto * v = ggml_dup_tensor(ctx, k);
            auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, capacity);
            auto * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nq);
            auto * original_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, kv_heads, nq);
            auto * original_v = ggml_dup_tensor(ctx, original_k);
            auto * node = ggml_remote_attn_range(ctx, q, k, v, ids, pos, 0, origin, live + nq, geometry.kq_scale);
            ggml_remote_attn_range_set_native_sources(node, original_k, original_v);
            // Compatibility probes are metadata-only, before a graph/request
            // exists. Restore prefill before constructing the real WAIT.
            for (const char * legacy : {"0", "1", static_cast<const char *>(nullptr)}) {
                storage_env mode(legacy, "GGML_RKVA_CUDA_FP8");
                assert(ggml_backend_remote_attn_cuda_enabled_for_node(node) == !(legacy && !std::strcmp(legacy, "0")));
            }
            assert(ggml_backend_remote_attn_cuda_enabled_for_node(node) == cuda_range);
            assert(ggml_backend_supports_op(gpu, node) == cuda_range);
            auto * complete = node;
            if (async) {
                ggml_remote_attn_range_set_async(node);
                complete = ggml_remote_attn_range_wait(ctx, node);
                assert(ggml_backend_remote_attn_cuda_enabled_for_node(complete) == cuda_range);
                assert((complete->view_src == node) == cuda_range);
            }
            ggml_set_output(node);
            ggml_set_output(complete);
            auto * local_input = ggml_dup_tensor(ctx, complete);
            auto * local = ggml_scale(ctx, local_input, 3.0f);
            auto * consumer = ggml_add(ctx, ggml_scale(ctx, complete, 2.0f), local);
            auto * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, node);
            ggml_build_forward_expand(graph, local);
            ggml_build_forward_expand(graph, consumer);
            ggml_backend_t backends[] = {gpu, cpu};
            auto * sched = ggml_backend_sched_new(backends, nullptr, 2, 1024, false, true);
            for (auto * t : {original_k, original_v, local_input, local, consumer})
                ggml_backend_sched_set_tensor_backend(sched, t, gpu);
            for (auto * t : {node, complete})
                ggml_backend_sched_set_tensor_backend(sched, t, cuda_range ? gpu : cpu);
            ggml_backend_sched_set_tensor_backend(sched, q, q_on_gpu ? gpu : cpu);
            for (auto * t : {k, v, ids, pos}) ggml_backend_sched_set_tensor_backend(sched, t, cpu);
            assert(ggml_backend_sched_alloc_graph(sched, graph));
            assert(ggml_backend_sched_get_tensor_backend(sched, node) == (cuda_range ? gpu : cpu));
            assert(ggml_backend_sched_get_tensor_backend(sched, complete) == (cuda_range ? gpu : cpu));
            std::vector<float> queries(width * nq), originals(kv_width * nq, 0.0f);
            std::vector<uint8_t> expected_q(width * nq), expected_out(width * nq);
            for (size_t row = 0; row < nq; ++row)
                for (size_t head = 0; head < heads; ++head)
                    for (size_t d = 0; d < dim; ++d) {
                        const size_t i = row * width + head * dim + d;
                        const size_t code = pattern(row, head, d);
                        queries[i] = values[code]; expected_q[i] = codes[code];
                        expected_out[i] = codes[(code + 3) % 8];
                    }
            { std::lock_guard<std::mutex> lock(server.mutex);
              server.fp8_output_data = expected_out; server.lse_head_delta = 0.125f; }
            ggml_backend_tensor_set(q, queries.data(), 0, queries.size() * sizeof(float));
            ggml_backend_tensor_set(original_k, originals.data(), 0, originals.size() * sizeof(float));
            ggml_backend_tensor_set(original_v, originals.data(), 0, originals.size() * sizeof(float));
            std::vector<float> local_values(row_width * nq, 1.0f);
            ggml_backend_tensor_set(local_input, local_values.data(), 0, local_values.size() * sizeof(float));
            std::vector<int32_t> cells(capacity, -1), positions(nq);
            for (uint32_t i = 0; i < live + nq; ++i) cells[i] = int32_t(origin + i);
            for (uint32_t i = 0; i < nq; ++i) positions[i] = int32_t(origin + live + i);
            ggml_backend_tensor_set(ids, cells.data(), 0, cells.size() * sizeof(int32_t));
            ggml_backend_tensor_set(pos, positions.data(), 0, positions.size() * sizeof(int32_t));
            assert(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
            ggml_backend_sched_synchronize(sched);
            assert((node->src[0] == q) == (cuda_range || !q_on_gpu));
            std::vector<float> raw(row_width * nq), result(raw.size());
            ggml_backend_tensor_get(complete, raw.data(), 0, raw.size() * sizeof(float));
            ggml_backend_tensor_get(consumer, result.data(), 0, result.size() * sizeof(float));
            for (size_t row = 0; row < nq; ++row) {
                for (size_t head = 0; head < heads; ++head) {
                    for (size_t d = 0; d < dim; ++d) {
                        const float expected = values[(pattern(row, head, d) + 3) % 8];
                        const size_t i = row * row_width + head * dim + d;
                        assert(raw[i] == expected && std::signbit(raw[i]) == std::signbit(expected));
                        assert(result[i] == 2.0f * expected + 3.0f);
                    }
                    const size_t i = row * row_width + width + head;
                    const float expected = float(std::log(double(live + row + 1))) + float(head) * 0.125f;
                    assert(raw[i] == expected);
                    assert(std::abs(result[i] - (2.0f * expected + 3.0f)) < 1e-6f);
                }
            }
            { std::lock_guard<std::mutex> lock(server.mutex);
              assert(server.fp8_query_data == expected_q && server.fp8_query_positions == positions);
              const std::vector<uint32_t> segments = nq > 9 ?
                  (nq == 17 ? std::vector<uint32_t>{8, 8, 1} : std::vector<uint32_t>{8, 2}) : std::vector<uint32_t>{nq};
              assert(server.bulk_query_segments == segments);
              uint32_t expected_live = live;
              for (size_t s = 0; s < segments.size(); ++s) {
                  expected_live += segments[s];
                  assert(server.bulk_live_counts[s] == expected_live);
              }
              assert(server.native_counts[0] == live + nq); }
            packs += cuda_range && q_on_gpu;
            unpacks += cuda_range;
            if (policy && !std::strcmp(policy, "mapped") && nq <= 9) {
                small_packs += q_on_gpu;
                ++small_unpacks;
            }
            const auto stats = std::string(ggml_backend_remote_attn_stats_json(backend));
            assert(counter(stats, "cuda_pack_calls") == packs);
            assert(counter(stats, "cuda_unpack_calls") == unpacks);
            assert(counter(stats, "cuda_small_pack_calls") == small_packs);
            assert(counter(stats, "cuda_small_unpack_calls") == small_unpacks);
            if (cuda_range && !context_live) { ++creates; context_live = true; }
            if (prefill_policy && !cuda_range && context_live) { ++releases; context_live = false; }
            assert(counter(stats, "cuda_context_creates") == creates);
            assert(counter(stats, "cuda_context_phase_releases") == releases);
            live += nq;
            ggml_backend_remote_attn_drain(backend);
            ggml_backend_sched_free(sched);
            ggml_free(ctx);
        }
        assert(releases == (prefill_policy ? 2u : 0u));
        assert(creates == (prefill_policy ? 2u : (policy && !std::strcmp(policy, "0") ? 0u : 1u)));
        if (!prefill_policy) {
            ggml_remote_attn_set_active(nullptr);
            ggml_backend_free(backend);
            continue;
        }
        // A direct caller can bypass supports_op/scheduler placement. A small
        // range in prefill mode must reject a VRAM output before sending RPC
        // or letting the CPU decoder dereference that address.
        auto * bad_ctx = ggml_init({4u << 20, nullptr, true});
        auto * bad_q = ggml_new_tensor_3d(bad_ctx, GGML_TYPE_F32, dim, heads, 1);
        auto * bad_k = ggml_new_tensor_2d(bad_ctx, GGML_TYPE_Q4_0, kv_width, capacity);
        auto * bad_v = ggml_dup_tensor(bad_ctx, bad_k);
        auto * bad_ids = ggml_new_tensor_1d(bad_ctx, GGML_TYPE_I32, capacity);
        auto * bad_pos = ggml_new_tensor_1d(bad_ctx, GGML_TYPE_I32, 1);
        auto * bad_original_k = ggml_new_tensor_3d(bad_ctx, GGML_TYPE_F32, dim, kv_heads, 1);
        auto * bad_original_v = ggml_dup_tensor(bad_ctx, bad_original_k);
        auto * bad = ggml_remote_attn_range(bad_ctx, bad_q, bad_k, bad_v, bad_ids, bad_pos,
                                           0, origin, live + 1, geometry.kq_scale);
        ggml_remote_attn_range_set_native_sources(bad, bad_original_k, bad_original_v);
        if (async) ggml_remote_attn_range_set_async(bad);
        auto * bad_buffer = ggml_backend_alloc_ctx_tensors(bad_ctx, gpu);
        assert(bad_buffer);
        ggml_backend_buffer_clear(bad_buffer, 0);
        std::vector<int32_t> bad_cells(capacity, -1);
        for (uint32_t i = 0; i < live + 1; ++i) bad_cells[i] = int32_t(origin + i);
        const int32_t bad_position = int32_t(origin + live);
        ggml_backend_tensor_set(bad_ids, bad_cells.data(), 0, bad_cells.size() * sizeof(int32_t));
        ggml_backend_tensor_set(bad_pos, &bad_position, 0, sizeof(bad_position));
        const std::vector<float> sentinel(row_width, -31.0f);
        ggml_backend_tensor_set(bad, sentinel.data(), 0, sentinel.size() * sizeof(float));
        assert(!ggml_remote_attn_exec(bad));
        assert(ggml_backend_remote_attn_failed(backend));
        ggml_backend_remote_attn_drain(backend);
        { std::lock_guard<std::mutex> lock(server.mutex); assert(server.attention_calls == 5); }
        ggml_remote_attn_set_active(nullptr);
        ggml_backend_free(backend);
        std::vector<float> observed(sentinel.size());
        ggml_backend_tensor_get(bad, observed.data(), 0, observed.size() * sizeof(float));
        assert(observed == sentinel);
        ggml_backend_buffer_free(bad_buffer);
        ggml_free(bad_ctx);
    }
    }
    }
    {
        storage_env policy("prefill", "GGML_RKVA_CUDA_FP8");
        storage_env overlap("1", "GGML_RKVA_ASYNC_LOCAL_REMOTE");
        loopback server(false, true, false, false, false, false, 0, 0, true, 7 | 8 | 16 | 32);
        auto * backend = ggml_backend_remote_attn_init("127.0.0.1", server.port);
        ggml_remote_attn_geometry geometry{1, heads, kv_heads, dim, capacity, 0, 0, 128, 3, 128, 2, 0, 0, false, false, 1.0f / 16};
        assert(ggml_backend_remote_attn_set_geometry(backend, &geometry));
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, capacity));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(ggml_backend_remote_attn_configure_range(backend, origin, capacity, GGML_TYPE_Q4_0));
        ggml_remote_attn_set_active(backend);
        struct pair { ggml_context * ctx; ggml_tensor * node, * wait, * ids, * pos; ggml_backend_buffer_t buffer; uint32_t nq; };
        const auto make_pair = [&](uint32_t nq) {
            auto * ctx = ggml_init({4u << 20, nullptr, true});
            auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, heads, nq);
            auto * k = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, kv_width, capacity);
            auto * v = ggml_dup_tensor(ctx, k);
            auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, capacity);
            auto * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nq);
            auto * original_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, kv_heads, nq);
            auto * original_v = ggml_dup_tensor(ctx, original_k);
            auto * node = ggml_remote_attn_range(ctx, q, k, v, ids, pos, 0, origin, nq, geometry.kq_scale);
            ggml_remote_attn_range_set_native_sources(node, original_k, original_v);
            ggml_remote_attn_range_set_async(node);
            auto * wait = ggml_remote_attn_range_wait(ctx, node);
            auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, nq > 9 ? gpu : cpu);
            assert(buffer);
            ggml_backend_buffer_clear(buffer, 0);
            return pair{ctx, node, wait, ids, pos, buffer, nq};
        };
        auto g0 = make_pair(17), g1 = make_pair(17), c1 = make_pair(1);
        const auto submit = [&](pair & p, uint32_t live) {
            std::vector<int32_t> cells(capacity, -1), positions(p.nq);
            for (uint32_t i = 0; i < live; ++i) cells[i] = int32_t(origin + i);
            for (uint32_t i = 0; i < p.nq; ++i) positions[i] = int32_t(origin + live - p.nq + i);
            ggml_backend_tensor_set(p.ids, cells.data(), 0, cells.size() * sizeof(int32_t));
            ggml_backend_tensor_set(p.pos, positions.data(), 0, positions.size() * sizeof(int32_t));
            p.node->op_params[GGML_REMOTE_ATTN_PARAM_N_LIVE] = int32_t(live);
            assert(ggml_remote_attn_exec(p.node));
        };
        const auto wait = [&](pair & p, uint32_t first_row) {
            assert(ggml_remote_attn_exec(p.wait));
            std::vector<float> result(row_width * p.nq);
            ggml_backend_tensor_get(p.wait, result.data(), 0, result.size() * sizeof(float));
            for (size_t row = 0; row < p.nq; ++row) {
                for (size_t d = 0; d < width; ++d) assert(result[row * row_width + d] == 0.0f);
                for (size_t h = 0; h < heads; ++h)
                    assert(result[row * row_width + width + h] == float(std::log(double(first_row + row + 1))));
            }
        };
        // Both slots acquire private contexts, then reach actual DONE/FREE.
        server.set_pause(true);
        submit(g0, 17); assert(server.wait_until_reply_paused());
        submit(g1, 34); assert(ggml_backend_remote_attn_async_in_flight(backend) == 2);
        server.unpause(); wait(g0, 0); wait(g1, 17);
        assert(counter(ggml_backend_remote_attn_stats_json(backend), "cuda_context_creates") == 2);
        // GPU slot0 is RUNNING; only the retired context of slot1 may go away.
        server.set_pause(true);
        submit(g0, 51); assert(server.wait_until_reply_paused());
        submit(c1, 52); assert(ggml_backend_remote_attn_async_in_flight(backend) == 2);
        assert(counter(ggml_backend_remote_attn_stats_json(backend), "cuda_context_phase_releases") == 1);
        assert(counter(ggml_backend_remote_attn_stats_json(backend), "cuda_context_creates") == 2);
        server.unpause(); wait(g0, 34); wait(c1, 51);
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 0);
        { std::lock_guard<std::mutex> lock(server.mutex); assert(server.attention_calls == 4); }
        ggml_remote_attn_set_active(nullptr);
        ggml_backend_free(backend); // private contexts retire before tensors go away
        for (auto p : {g0, g1, c1}) { ggml_backend_buffer_free(p.buffer); ggml_free(p.ctx); }
    }
    ggml_backend_free(cpu);
    ggml_backend_free(gpu);
    std::puts("CUDA prefill codec graph:20 cases,1/9 CPU,10/17 GPU, full-segment Q/positions,17->1->9->10->1 reuse and4 invalid GPU outputs passed");
    std::puts("CUDA codec contexts: two retired releases per prefill sequence;60 legacy0/1/unset cases without phase release passed");
    std::puts("CUDA codec overlap: GPU17 RUNNING slot0 preserves context while CPU1 retires only old slot1 context; FIFO OUT/LSE passed");
}

static void test_done_epoch_invalidation_race() {
    storage_env native("kvarn4");
    storage_env experimental("1", "GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL");
    storage_env fusion("0", "GGML_RKVA_RANGE_FUSE");
    storage_env precision("128", "GGML_RKVA_NATIVE_TAIL_TOKENS");
    storage_env bulk("1", "GGML_RKVA_NATIVE_BULK");
    storage_env wire("fp8", "GGML_RKVA_WIRE_FORMAT");
    storage_env overlap("1", "GGML_RKVA_ASYNC_LOCAL_REMOTE");
    storage_env acceleration("0", "GGML_RKVA_CUDA_FP8");
    constexpr uint32_t nq = 4, capacity = 512, origin = 1024;
    unsigned consumed = 0;
    for (unsigned iteration = 0; iteration < 64; ++iteration) {
        // A fresh connection also covers the legal losing-WAIT path, which
        // may poison the old context. No failed context is revived for reuse.
        loopback server(false, true, false, false, false, false, 0, 0, true, 7 | 8 | 16 | 32);
        auto * backend = ggml_backend_remote_attn_init("127.0.0.1", server.port);
        ggml_remote_attn_geometry geo{1, 2, 1, 256, capacity, 0, 0, 128, 3, 128, 2, 0, 0, false, false, 1.0f / 16};
        assert(ggml_backend_remote_attn_set_geometry(backend, &geo));
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, capacity));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(ggml_backend_remote_attn_configure_range(backend, origin, capacity, GGML_TYPE_Q4_0));
        ggml_remote_attn_set_active(backend);
        auto * ctx = ggml_init({2u << 20, nullptr, true});
        auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 2, nq);
        auto * k = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, capacity);
        auto * v = ggml_dup_tensor(ctx, k);
        auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, capacity);
        auto * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nq);
        auto * original_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, nq);
        auto * original_v = ggml_dup_tensor(ctx, original_k);
        auto * node = ggml_remote_attn_range(ctx, q, k, v, ids, pos, 0, origin, nq, geo.kq_scale);
        ggml_remote_attn_range_set_native_sources(node, original_k, original_v);
        ggml_remote_attn_range_set_async(node);
        auto * wait = ggml_remote_attn_range_wait(ctx, node);
        auto * cpu = ggml_backend_cpu_init();
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        assert(buffer);
        ggml_backend_buffer_clear(buffer, 0);
        std::vector<int32_t> cells(capacity, -1), positions(nq);
        for (uint32_t i = 0; i < nq; ++i) cells[i] = positions[i] = int32_t(origin + i);
        ggml_backend_tensor_set(ids, cells.data(), 0, cells.size() * sizeof(int32_t));
        ggml_backend_tensor_set(pos, positions.data(), 0, positions.size() * sizeof(int32_t));
        assert(ggml_remote_attn_exec(node));
        ggml_backend_synchronize(backend); // preserve an actual DONE_OK completion
        assert(!ggml_backend_remote_attn_failed(backend));
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 1);

        std::promise<void> start, waiter_ready, invalidator_ready;
        auto gate = start.get_future().share();
        auto ready_wait = waiter_ready.get_future();
        auto ready_invalidate = invalidator_ready.get_future();
        bool accepted = false;
        std::thread waiter([&] {
            waiter_ready.set_value();
            gate.wait();
            accepted = ggml_remote_attn_exec(wait);
        });
        std::thread invalidator([&] {
            invalidator_ready.set_value();
            gate.wait();
            ggml_backend_remote_attn_invalidate_range(backend);
        });
        ready_wait.wait();
        ready_invalidate.wait();
        start.set_value();
        waiter.join();
        invalidator.join();
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 0);
        // Either operation may linearize first. An accepted completion must
        // still contain its validated result; rejected output is never used.
        if (accepted) {
            ++consumed;
            const auto * result = static_cast<const float *>(wait->data);
            for (size_t row = 0; row < nq; ++row) {
                for (size_t d = 0; d < 512; ++d) assert(result[row * 514 + d] == 0.0f);
                for (size_t h = 0; h < 2; ++h)
                    assert(std::abs(result[row * 514 + 512 + h] - float(std::log(double(row + 1)))) < 1e-6f);
            }
        }
        ggml_backend_remote_attn_drain(backend);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu);
        ggml_remote_attn_set_active(nullptr);
        ggml_backend_free(backend);
    }
    std::printf("DONE_OK WAIT/invalidate race: 64 cases, consumed=%u invalidated=%u\n", consumed, 64 - consumed);
}

// P0-D: a failed remote transaction must retire every derived state, reject
// the next request without retrying on the broken ordered stream, and only a
// fresh backend/session may serve again. Runs entirely on loopback (no Xbox).
static void test_failure_retirement() {
    storage_env native("kvarn4");
    storage_env experimental("1", "GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL");
    storage_env fusion("0", "GGML_RKVA_RANGE_FUSE");
    ggml_backend_load_all();

    struct fixture {
        loopback server;
        ggml_backend_t backend = nullptr;
        ggml_context * ctx = nullptr;
        ggml_backend_t cpu = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_tensor * node = nullptr;
        ggml_tensor * ids = nullptr;
        ggml_tensor * pos = nullptr;

        fixture() : server(false, true, false, false, false, false, 0, 0, true, 4 | 8 | 16 | 1 | 2) {
            backend = ggml_backend_remote_attn_init("127.0.0.1", server.port);
            ggml_remote_attn_geometry geometry {};
            geometry.n_layer_remote = 2; geometry.n_head = 2; geometry.n_head_kv = 1;
            geometry.head_dim = 256; geometry.max_ctx = 512; geometry.kq_scale = 0.0625f;
            geometry.group_tokens = 128; geometry.sinkhorn_iters = 3;
            assert(ggml_backend_remote_attn_set_geometry(backend, &geometry));
            assert(ggml_backend_remote_attn_connect(backend));
            assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 512));
            ggml_backend_remote_attn_set_active_session(backend, 0);
            assert(ggml_backend_remote_attn_configure_range(backend, 1024, 512, GGML_TYPE_Q4_0));
            ggml_remote_attn_set_active(backend);
            ctx = ggml_init({1u << 20, nullptr, true});
            auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 2, 2);
            auto * k = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 512);
            auto * v = ggml_dup_tensor(ctx, k);
            ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
            pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
            auto * original_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, 2);
            auto * original_v = ggml_dup_tensor(ctx, original_k);
            node = ggml_remote_attn_range(ctx, q, k, v, ids, pos, 0, 1024, 2, geometry.kq_scale);
            ggml_remote_attn_range_set_native_sources(node, original_k, original_v);
            cpu = ggml_backend_cpu_init();
            buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
            assert(buffer);
            ggml_backend_buffer_clear(buffer, 0);
        }
        ~fixture() {
            ggml_remote_attn_set_active(nullptr);
            if (buffer) ggml_backend_buffer_free(buffer);
            if (ctx) ggml_free(ctx);
            if (cpu) ggml_backend_free(cpu);
            if (backend) ggml_backend_free(backend);
        }
        void run(uint32_t live, uint32_t layer) {
            std::vector<int32_t> cells(512, -1);
            for (uint32_t i = 0; i < live; ++i) cells[i] = 1024 + i;
            const int32_t positions[2] = {int32_t(1024 + live - 2), int32_t(1024 + live - 1)};
            ggml_backend_tensor_set(ids, cells.data(), 0, cells.size() * sizeof(int32_t));
            ggml_backend_tensor_set(pos, positions, 0, sizeof(positions));
            node->op_params[GGML_REMOTE_ATTN_PARAM_N_LIVE] = int32_t(live);
            node->op_params[GGML_REMOTE_ATTN_PARAM_LAYER_ID] = int32_t(layer);
            assert(ggml_remote_attn_exec(node));
        }
    };

    // Failure path: every derived state is retired, the next request is
    // rejected, and reset/trim cannot revive the broken stream or publish a
    // partial result.
    {
        fixture f;
        f.run(2, 0);
        f.run(2, 1);
        assert(ggml_backend_remote_attn_native_kvarn_enabled(f.backend));
        assert(ggml_backend_remote_attn_native_kvarn_has_state(f.backend));
        f.server.fail_attention = true;
        assert(!ggml_remote_attn_exec(f.node));
        assert(ggml_backend_remote_attn_failed(f.backend));
        assert(!ggml_backend_remote_attn_connected(f.backend));
        assert(!ggml_backend_remote_attn_native_kvarn_enabled(f.backend));
        assert(!ggml_backend_remote_attn_native_kvarn_has_state(f.backend));
        assert(!ggml_backend_remote_attn_native_kvarn_can_rewind(f.backend));
        assert(!ggml_backend_remote_attn_async_enabled(f.backend));
        assert(!ggml_backend_remote_attn_trim(f.backend, 0, 1024));
        assert(!ggml_backend_remote_attn_reset(f.backend, 0));
        assert(!ggml_remote_attn_exec(f.node));
    }
    // Broken transport (stream closed without a reply): same terminal poison,
    // physical teardown at drain, and no retry on the dead stream.
    {
        fixture f;
        f.run(2, 0);
        f.run(2, 1);
        f.server.drop_attention = true;
        assert(!ggml_remote_attn_exec(f.node));
        assert(ggml_backend_remote_attn_failed(f.backend));
        assert(!ggml_backend_remote_attn_connected(f.backend));
        assert(!ggml_backend_remote_attn_native_kvarn_enabled(f.backend));
        ggml_backend_remote_attn_drain(f.backend);
        assert(!ggml_backend_remote_attn_native_kvarn_has_state(f.backend));
        assert(!ggml_remote_attn_exec(f.node));
    }
    // Recovery: only a fresh backend + session is valid again.
    {
        fixture f;
        f.run(2, 0);
        f.run(2, 1);
        assert(ggml_backend_remote_attn_native_kvarn_enabled(f.backend));
        assert(ggml_backend_remote_attn_native_kvarn_has_state(f.backend));
    }
    std::puts("P0-D failure retirement: state invalidated, next request rejected, recreation valid passed");
}

// P0-D: a failure while two async slots are in flight must retire both slots
// and every derived state; WAIT nodes must fail (never publish), and no later
// submit may reuse the broken session/stream.
static void test_async_failure_two_slots(uint32_t nq = 4) {
    storage_env native("kvarn4");
    storage_env experimental("1", "GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL");
    storage_env fusion("0", "GGML_RKVA_RANGE_FUSE");
    storage_env precision("128", "GGML_RKVA_NATIVE_TAIL_TOKENS");
    storage_env bulk("1", "GGML_RKVA_NATIVE_BULK");
    storage_env wire("fp8", "GGML_RKVA_WIRE_FORMAT");
    storage_env async_env("1", "GGML_RKVA_ASYNC_LOCAL_REMOTE");

    // mode 0: slot 0 fails while slot 1 is PENDING.
    // mode 1: slot 0 completes; slot 1 fails while RUNNING.
    // mode 2: slot 0 completes; slot 1 is RUNNING and its valid reply is held
    //         after a successful transact; an invalid async submit poisons the
    //         context, then the post-transact epoch check must reject the reply
    //         before any OUT write. Drain must wait for that job.
    auto scenario = [&](int mode) {
    loopback server(false, true, false, false, false, false, 0, 0, true, 7 | 8 | 16 | 32);
    auto * backend = ggml_backend_remote_attn_init("127.0.0.1", server.port);
    ggml_remote_attn_geometry geometry{1, 2, 1, 256, 512, 0, 0, 128, 3, 128, 2, 0, 0, false, false, 1.0f / 16};
    assert(ggml_backend_remote_attn_set_geometry(backend, &geometry));
    assert(ggml_backend_remote_attn_connect(backend));
    assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 512));
    ggml_backend_remote_attn_set_active_session(backend, 0);
    assert(ggml_backend_remote_attn_configure_range(backend, 1024, 512, GGML_TYPE_Q4_0));
    assert(ggml_backend_remote_attn_async_enabled(backend));
    ggml_remote_attn_set_active(backend);

    struct job { ggml_tensor * node; ggml_tensor * wait; ggml_tensor * ids; ggml_tensor * pos; ggml_tensor * q; };
    ggml_context * ctx = ggml_init({8u << 20, nullptr, true});
    auto make = [&]() {
        job j;
        j.q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 2, nq);
        auto * k = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 512);
        auto * v = ggml_dup_tensor(ctx, k);
        j.ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
        j.pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nq);
        auto * ok = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, nq);
        auto * ov = ggml_dup_tensor(ctx, ok);
        j.node = ggml_remote_attn_range(ctx, j.q, k, v, j.ids, j.pos, 0, 1024, nq, geometry.kq_scale);
        ggml_remote_attn_range_set_native_sources(j.node, ok, ov);
        ggml_remote_attn_range_set_async(j.node);
        j.wait = ggml_remote_attn_range_wait(ctx, j.node);
        return j;
    };
    auto set_inputs = [&](job & j, uint32_t live) {
        std::vector<int32_t> cells(512, -1), positions(nq);
        for (uint32_t i = 0; i < live; ++i) cells[i] = 1024 + i;
        for (uint32_t i = 0; i < nq; ++i) positions[i] = 1024 + live - nq + i;
        ggml_backend_tensor_set(j.ids, cells.data(), 0, cells.size() * sizeof(int32_t));
        ggml_backend_tensor_set(j.pos, positions.data(), 0, positions.size() * sizeof(int32_t));
        j.node->op_params[GGML_REMOTE_ATTN_PARAM_N_LIVE] = int32_t(live);
    };

    job a = make(), b = make(), c = make(), d = make();
    auto * cpu = ggml_backend_cpu_init();
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    assert(buffer);
    ggml_backend_buffer_clear(buffer, 0);
    set_inputs(a, nq);
    set_inputs(b, 2*nq);
    set_inputs(c, 3*nq);
    set_inputs(d, 3*nq);
    {
        std::vector<float> bad(size_t(256) * 2 * nq, 0.0f);
        bad[0] = std::numeric_limits<float>::infinity();
        ggml_backend_tensor_set(d.q, bad.data(), 0, bad.size() * sizeof(float));
    }

    // Pre-fill WAIT destinations: a failed WAIT must never overwrite them with
    // an old prefix/OUT-LSE, even if the source was written by a running job.
    std::vector<float> sentinel(size_t(nq) * (2 * 256 + 2), 42.0f);
    ggml_backend_tensor_set(a.wait, sentinel.data(), 0, sentinel.size() * sizeof(float));
    ggml_backend_tensor_set(b.wait, sentinel.data(), 0, sentinel.size() * sizeof(float));

    server.set_pause(true, 0);
    assert(ggml_remote_attn_exec(a.node));
    assert(ggml_remote_attn_exec(b.node));
    assert(ggml_backend_remote_attn_async_in_flight(backend) == 2);

    if (mode != 0) {
        server.grant_reply(1);
        assert(ggml_remote_attn_exec(a.wait)); // slot 0 published before the fault
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 1);
    }

    std::vector<float> published_a;
    if (mode == 1) {
        published_a.assign(static_cast<float *>(a.wait->data), static_cast<float *>(a.wait->data)+sentinel.size());
        ggml_backend_tensor_set(c.wait, sentinel.data(), 0, sentinel.size()*sizeof(float));
        assert(ggml_remote_attn_exec(c.node)); // reuse consumed A's physical slot while B is held
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 2);
    }
    if (mode == 2) {
        // Deterministic handshake: enable before the reply, wait until the
        // worker has a successful transact and is held, poison, prove the drain
        // cannot finish while held, then release and let the epoch check reject.
        ggml_backend_remote_attn_test_handshake_enable(true);
        server.grant_reply(1); // valid reply to slot 1
        assert(ggml_backend_remote_attn_test_handshake_wait_reached(3000));
        assert(!ggml_remote_attn_exec(d.node)); // poison while the worker is held
        assert(ggml_backend_remote_attn_failed(backend));
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 1);

        std::atomic<bool> drain_started{false};
        std::atomic<bool> drained{false};
        std::thread drain_thread([&] {
            drain_started = true;
            ggml_backend_remote_attn_drain(backend);
            drained = true;
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        assert(drain_started.load());
        assert(!drained.load()); // worker held: drain must not finish
        ggml_backend_remote_attn_test_handshake_release();
        assert(!ggml_remote_attn_exec(b.wait)); // epoch check rejects before unpack
        drain_thread.join();
        assert(drained.load());
        ggml_backend_remote_attn_test_handshake_enable(false);
    } else {
        server.fail_attention = true;
        server.grant_reply(1);
        if (mode == 1) {
            assert(!ggml_remote_attn_exec(b.wait)); // slot 1 was RUNNING
        } else {
            assert(!ggml_remote_attn_exec(a.wait));
            assert(!ggml_remote_attn_exec(b.wait));
        }
    }

    if (mode == 1) {
        assert(!ggml_remote_attn_exec(c.wait));
        assert(std::equal(published_a.begin(), published_a.end(), static_cast<float *>(a.wait->data)));
        const auto * out_c = static_cast<float *>(c.wait->data);
        for(size_t i=0;i<sentinel.size();++i)assert(out_c[i]==42.0f);
    }
    const auto * out_a = static_cast<const float *>(a.wait->data);
    const auto * out_b = static_cast<const float *>(b.wait->data);
    if (mode == 0) {
        for (size_t i = 0; i < sentinel.size(); ++i) assert(out_a[i] == 42.0f);
        for (size_t i = 0; i < sentinel.size(); ++i) assert(out_b[i] == 42.0f);
    } else {
        for (size_t i = 0; i < sentinel.size(); ++i) assert(out_b[i] == 42.0f);
    }

    assert(ggml_backend_remote_attn_failed(backend));
    assert(!ggml_backend_remote_attn_connected(backend));
    assert(ggml_backend_remote_attn_async_in_flight(backend) == 0);
    assert(!ggml_backend_remote_attn_async_enabled(backend));
    assert(!ggml_backend_remote_attn_native_kvarn_has_state(backend));
    assert(!ggml_remote_attn_exec(c.node));

    // Drain is the physical teardown point; every logical getter stays invalid.
    ggml_backend_remote_attn_drain(backend);
    assert(!ggml_backend_remote_attn_native_kvarn_has_state(backend));
    assert(!ggml_remote_attn_exec(c.node));

    server.unpause();
    ggml_remote_attn_set_active(nullptr);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu);
    ggml_backend_free(backend);
    };

    scenario(0);
    scenario(1);
    scenario(2);
    std::puts("P0-D async failure: pending/RUNNING/poison-during-run slots retired, WAIT/no-publish/reject passed");
}

// Cohort4 remains unavailable even when a peer advertises the experimental
// extension or native Q4/Q8 capability bits. Ordinary WF uses op22/RPC_BATCH=1.
static void test_native_cohort_async() {
    storage_env native("kvarn4");
    storage_env experimental("1", "GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL");
    storage_env fusion("0", "GGML_RKVA_RANGE_FUSE");
    storage_env precision("128", "GGML_RKVA_NATIVE_TAIL_TOKENS");
    storage_env bulk("1", "GGML_RKVA_NATIVE_BULK");
    storage_env wire("fp8", "GGML_RKVA_WIRE_FORMAT");
    storage_env overlap("1", "GGML_RKVA_ASYNC_LOCAL_REMOTE");
    storage_env acceleration("0", "GGML_RKVA_CUDA_FP8");
    for (uint32_t support : {uint32_t(7 | 8 | 16 | 32), uint32_t(7 | 8 | 16 | 32 | 256)}) {
        loopback server(false, true, true, false, true, false, 0, 0, true, support);
        auto * backend = ggml_backend_remote_attn_init("127.0.0.1", server.port);
        assert(backend);
        ggml_remote_attn_geometry geo{4, 2, 1, 256, 512, 0, 0, 128, 3, 128, 2,
                                     0, 0, false, false, 1.0f / 16};
        assert(ggml_backend_remote_attn_set_geometry(backend, &geo));
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 512));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        {
            storage_env batching("4", "GGML_RKVA_WAVEFRONT_RPC_BATCH");
            assert(!ggml_backend_remote_attn_configure_range(backend, 1024, 512, GGML_TYPE_Q4_0));
            assert(!ggml_backend_remote_attn_cohort_enabled(backend));
            std::lock_guard<std::mutex> lock(server.mutex);
            assert(server.range.capacity == 0 && server.cohort_calls == 0);
        }
        {
            storage_env batching("1", "GGML_RKVA_WAVEFRONT_RPC_BATCH");
            assert(ggml_backend_remote_attn_configure_range(backend, 1024, 512, GGML_TYPE_Q4_0));
            assert(!ggml_backend_remote_attn_cohort_enabled(backend));
            assert(ggml_backend_remote_attn_async_capacity(backend) == 2);
            std::lock_guard<std::mutex> lock(server.mutex);
            assert(server.cohort_calls == 0);
        }
        ggml_backend_free(backend);
    }
    std::puts("cohort4 unavailable: native Q4/Q8 bits and experimental peer cannot enable op23; RPC batch1 admitted");
}

// Marker-only ownership must work with the ordinary two-slot op22 path,
// including an already-warm private slot. This checks FIFO/admission/lifetime;
// the paused wire transaction is not a measurement of CPU/GPU concurrency.
static void test_cuda_wavefront_marker_only_two_slots() {
    ggml_backend_load_all();
    auto * reg = ggml_backend_reg_by_name("CUDA");
    if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
        std::puts("CUDA WF marker-only two slots: skipped (CUDA backend unavailable)");
        return;
    }
    storage_env native("kvarn4"), experimental("1", "GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL");
    storage_env fusion("0", "GGML_RKVA_RANGE_FUSE"), bulk("1", "GGML_RKVA_NATIVE_BULK");
    storage_env wire("fp8", "GGML_RKVA_WIRE_FORMAT"), overlap("1", "GGML_RKVA_ASYNC_LOCAL_REMOTE");
    storage_env batching("1", "GGML_RKVA_WAVEFRONT_RPC_BATCH");
    constexpr uint32_t heads = 24, kv_heads = 4, dim = 256, capacity = 1024, origin = 1024;
    constexpr size_t width = size_t(heads)*dim, kv_width = size_t(kv_heads)*dim, row_width = width + heads;
    constexpr float lse_head_delta = 0.125f;
    // Literal, exactly-representable E5M2 values are an independent byte/value
    // oracle. The pattern varies by stage, row, head and dimension.
    const uint8_t codes[] = {0x00, 0x80, 0x3c, 0xbc, 0x40, 0xc0, 0x34, 0xb4};
    const float values[] = {0.0f, -0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.25f, -0.25f};
    const auto fixture_index = [](uint32_t seed, uint32_t row, uint32_t head, uint32_t d) {
        return (seed*3 + row*3 + head*5 + d*7) % 8;
    };
    const auto counter = [](const std::string & json, const char * key) {
        const auto pos = json.find(std::string("\"") + key + "\":");
        assert(pos != std::string::npos);
        return std::stoull(json.substr(pos + std::strlen(key) + 3));
    };
    struct phase_case { uint32_t tails; const char * policy; uint32_t nq; };
    const phase_case cases[] = {{0, "prefill", 17}, {0, "mapped", 17},
                                {2, "prefill", 128}, {2, "mapped", 128},
                                {0, "prefill", 256}, {0, "mapped", 256}};
    for (const auto & test_case : cases) {
        const uint32_t tails = test_case.tails;
        const char * policy = test_case.policy;
        const uint32_t large_nq = test_case.nq;
        const uint32_t last = 3 + tails;
        const uint32_t small_live = (2 + tails)*large_nq + 1;
        storage_env acceleration(policy, "GGML_RKVA_CUDA_FP8");
        const bool mapped = std::strcmp(policy, "mapped") == 0;
        auto * gpu = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
        auto * cpu = ggml_backend_cpu_init();
        assert(gpu && cpu);
        // Deliberately omit the cohort capability. The bit alone must not
        // require op23 negotiation or enlarge the legacy two-slot queue.
        loopback server(false, true, false, false, false, false, 0, 0, true, 7 | 8 | 16 | 32);
        auto * backend = ggml_backend_remote_attn_init("127.0.0.1", server.port);
        ggml_remote_attn_geometry geo{2, heads, kv_heads, dim, capacity, 0, 0, 128, 3, 128, 2,
                                      0, 0, false, false, 1.0f / 16};
        assert(ggml_backend_remote_attn_set_geometry(backend, &geo));
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, capacity));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(ggml_backend_remote_attn_configure_range(backend, origin, capacity, GGML_TYPE_Q4_0));
        assert(!ggml_backend_remote_attn_cohort_enabled(backend));
        assert(ggml_backend_remote_attn_async_capacity(backend) == 2);
        ggml_remote_attn_set_active(backend);
        ggml_backend_remote_attn_reset_stats(backend);
        auto * ctx = ggml_init({4u << 20, nullptr, true});
        assert(ctx);
        struct item {
            ggml_tensor * node, * wait, * q, * ids, * pos, * original_k, * original_v;
            ggml_cgraph * submit, * consume;
            uint32_t layer, nq, live, fixture;
        };
        std::vector<item> items;
        for (uint32_t i = 0; i <= last; ++i) {
            // Private warmup, two marked slots, optional two private large
            // tails, then unmarked small decode. The tails reproduce the real
            // serving ownership transition missed by the direct-to-small case.
            const uint32_t layer = i == 2 ? 1 : 0;
            const uint32_t nq = i == last ? 1 : large_nq;
            const uint32_t live = i == 0 || i == 2 ? large_nq : i == 1 ? 2*large_nq : i == last ? small_live : i*large_nq;
            auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, heads, nq);
            auto * k = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, kv_width, capacity);
            auto * v = ggml_dup_tensor(ctx, k);
            auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, capacity);
            auto * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nq);
            auto * original_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, kv_heads, nq);
            auto * original_v = ggml_dup_tensor(ctx, original_k);
            for (auto * tensor : {q, k, v, ids, pos, original_k, original_v}) ggml_set_input(tensor);
            auto * node = ggml_remote_attn_range(ctx, q, k, v, ids, pos, layer, origin, live, geo.kq_scale);
            ggml_remote_attn_range_set_native_sources(node, original_k, original_v);
            ggml_remote_attn_range_set_async(node);
            if (i == 1 || i == 2) ggml_remote_attn_range_set_wavefront(node);
            assert(uint32_t(ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_COHORT)) ==
                    (i == 1 || i == 2 ? uint32_t(GGML_REMOTE_ATTN_WAVEFRONT_SHARED_CUDA) : 0u));
            auto * wait = ggml_remote_attn_range_wait(ctx, node);
            assert(ggml_backend_remote_attn_cuda_enabled_for_node(node) == (i != last || mapped));
            assert((wait->view_src == node) == (i != last || mapped));
            ggml_set_output(node);
            ggml_set_output(wait);
            // Operands are allocated once by the complete scheduler graph.
            // One-op dispatches below use those ready operands and preserve the
            // CUDA submit/WAIT bridge. Expanding WAIT recursively here would
            // accidentally submit its source Range a second time.
            auto * submit = ggml_new_graph_custom(ctx, 1, false);
            auto * consume = ggml_new_graph_custom(ctx, 1, false);
            ggml_graph_add_node(submit, node);
            ggml_graph_add_node(consume, wait);
            items.push_back({node, wait, q, ids, pos, original_k, original_v,
                             submit, consume, layer, nq, live, i});
        }
        auto * allocation_graph = ggml_new_graph_custom(ctx, 1024, false);
        ggml_build_forward_expand(allocation_graph, items[0].wait);
        ggml_build_forward_expand(allocation_graph, items[1].node);
        ggml_build_forward_expand(allocation_graph, items[2].node);
        ggml_build_forward_expand(allocation_graph, items[1].wait);
        ggml_build_forward_expand(allocation_graph, items[2].wait);
        for (uint32_t i = 3; i <= last; ++i) ggml_build_forward_expand(allocation_graph, items[i].wait);
        ggml_backend_t backends[] = {gpu, cpu};
        auto * sched = ggml_backend_sched_new(backends, nullptr, 2, 1024, false, true);
        assert(sched);
        for (uint32_t i = 0; i < items.size(); ++i) {
            const auto & item = items[i];
            const auto selected = i != last || mapped ? gpu : cpu;
            for (auto * tensor : {item.q, item.original_k, item.original_v, item.node, item.wait})
                ggml_backend_sched_set_tensor_backend(sched, tensor, selected);
            for (auto * tensor : {item.node->src[1], item.node->src[2], item.ids, item.pos})
                ggml_backend_sched_set_tensor_backend(sched, tensor, cpu);
        }
        assert(ggml_backend_sched_alloc_graph(sched, allocation_graph));
        for (const auto & item : items) {
            std::vector<float> q(size_t(item.nq)*width), originals(size_t(item.nq)*kv_width, 0.0f);
            for (uint32_t row = 0; row < item.nq; ++row)
                for (uint32_t h = 0; h < heads; ++h)
                    for (uint32_t d = 0; d < dim; ++d)
                        q[size_t(row)*width + size_t(h)*dim + d] = values[fixture_index(item.fixture + 4, row, h, d)];
            std::vector<int32_t> ids(capacity, -1), pos(item.nq);
            for (uint32_t j = 0; j < item.live; ++j) ids[j] = int32_t(origin + j);
            for (uint32_t j = 0; j < item.nq; ++j) pos[j] = int32_t(origin + item.live - item.nq + j);
            ggml_backend_tensor_set(item.q, q.data(), 0, q.size()*sizeof(float));
            ggml_backend_tensor_set(item.original_k, originals.data(), 0, originals.size()*sizeof(float));
            ggml_backend_tensor_set(item.original_v, originals.data(), 0, originals.size()*sizeof(float));
            ggml_backend_tensor_set(item.ids, ids.data(), 0, ids.size()*sizeof(int32_t));
            ggml_backend_tensor_set(item.pos, pos.data(), 0, pos.size()*sizeof(int32_t));
        }
        ggml_backend_synchronize(gpu);
        const auto set_reply_pattern = [&](uint32_t seed, uint32_t nq) {
            // Caller either owns server.mutex while idle, or pause_mutex after
            // the reply latch and before grant. The parser then cannot advance.
            server.fp8_output_data.resize(size_t(nq)*width);
            for (uint32_t row = 0; row < nq; ++row)
                for (uint32_t h = 0; h < heads; ++h)
                    for (uint32_t d = 0; d < dim; ++d)
                        server.fp8_output_data[size_t(row)*width + size_t(h)*dim + d] =
                                codes[fixture_index(seed, row, h, d)];
            server.lse_head_delta = lse_head_delta;
        };
        const auto check_request = [&](const item & item, unsigned packet_count) {
            // Same explicit idle-or-paused ownership contract as above.
            assert(server.cohort_calls == 0);
            assert(std::count(server.packet_ops.begin(), server.packet_ops.end(), RKVA_KVARN_RANGE_BULK_ATTN) == packet_count);
            assert(server.native_counts[item.layer] == item.live);
            assert(server.fp8_query_data.size() == size_t(item.nq)*width);
            assert(server.fp8_query_positions.size() == item.nq);
            for (uint32_t row = 0; row < item.nq; ++row) {
                assert(server.fp8_query_positions[row] == int32_t(origin + item.live - item.nq + row));
                for (uint32_t h = 0; h < heads; ++h)
                    for (uint32_t d = 0; d < dim; ++d)
                        assert(server.fp8_query_data[size_t(row)*width + size_t(h)*dim + d] ==
                                codes[fixture_index(item.fixture + 4, row, h, d)]);
            }
            const uint32_t chunk = item.nq > 9 ? 8 : item.nq;
            const uint32_t nseg = (item.nq + chunk - 1) / chunk;
            assert(server.bulk_query_segments.size() == nseg && server.bulk_live_counts.size() == nseg);
            for (uint32_t s = 0; s < nseg; ++s) {
                const uint32_t end = std::min(item.nq, (s + 1)*chunk);
                assert(server.bulk_query_segments[s] == end - s*chunk);
                assert(server.bulk_live_counts[s] == item.live - item.nq + end);
            }
        };
        const auto check_output = [&](const item & item) {
            std::vector<float> out(size_t(item.nq)*row_width);
            ggml_backend_tensor_get(item.wait, out.data(), 0, out.size()*sizeof(float));
            for (uint32_t row = 0; row < item.nq; ++row) {
                for (uint32_t h = 0; h < heads; ++h) {
                    for (uint32_t d = 0; d < dim; ++d) {
                        const float expected = values[fixture_index(item.fixture, row, h, d)];
                        const float value = out[size_t(row)*row_width + size_t(h)*dim + d];
                        assert(value == expected);
                        if (expected == 0.0f) assert(std::signbit(value) == std::signbit(expected));
                    }
                    const float expected_lse = float(std::log(double(item.live - item.nq + row + 1))) + float(h)*lse_head_delta;
                    assert(out[size_t(row)*row_width + width + h] == expected_lse);
                }
            }
        };
        {
            std::lock_guard<std::mutex> lock(server.mutex);
            set_reply_pattern(0, large_nq);
        }
        assert(ggml_backend_graph_compute(gpu, items[0].submit) == GGML_STATUS_SUCCESS);
        assert(ggml_backend_graph_compute(gpu, items[0].consume) == GGML_STATUS_SUCCESS);
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 0);
        check_output(items[0]);
        {
            std::lock_guard<std::mutex> lock(server.mutex);
            check_request(items[0], 1);
            set_reply_pattern(1, large_nq);
        }
        const std::string warm = ggml_backend_remote_attn_stats_json(backend);
        assert(counter(warm, "cuda_context_creates") == 1);
        assert(counter(warm, "cuda_context_phase_releases") == 0);
        assert(counter(warm, "cuda_context_shared_acquires") == 0);
        server.set_pause(true);
        assert(ggml_backend_graph_compute(gpu, items[1].submit) == GGML_STATUS_SUCCESS);
        assert(server.wait_until_reply_paused());
        {
            // The mock holds server.mutex across its reply wait. Acquiring it
            // here would deadlock. Its pause_cv wait released pause_mutex after
            // parsing; holding pause_mutex with zero tokens freezes all parser
            // fields through these reads and the next-reply configuration.
            std::lock_guard<std::mutex> pause(server.pause_mutex);
            assert(server.paused_replies == 1 && server.pause_mode.load() && server.reply_tokens == 0);
            check_request(items[1], 2);
            assert(!server.native_ready[1]); // the second marked request is not on wire yet
            set_reply_pattern(2, large_nq);
        }
        assert(ggml_backend_graph_compute(gpu, items[2].submit) == GGML_STATUS_SUCCESS);
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 2);
        const auto ticket = [](ggml_tensor * node) {
            return uint64_t(uint32_t(ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_TICKET_LO))) |
                   (uint64_t(uint32_t(ggml_get_op_params_i32(node, GGML_REMOTE_ATTN_PARAM_TICKET_HI))) << 32);
        };
        assert(ticket(items[1].node) != 0 && ticket(items[2].node) == ticket(items[1].node) + 1);
        const std::string prepared = ggml_backend_remote_attn_stats_json(backend);
        assert(counter(prepared, "cuda_context_creates") == 2);
        assert(counter(prepared, "cuda_context_phase_releases") == 1); // warm private owner dropped
        assert(counter(prepared, "cuda_context_shared_acquires") == 2);
        assert(counter(prepared, "cuda_context_shared_releases") == 0);
        assert(counter(prepared, "cuda_pack_calls") == 3);
        assert(counter(prepared, "cuda_unpack_calls") == 1); // only warm reply completed
        server.grant_reply(1);
        assert(ggml_backend_graph_compute(gpu, items[1].consume) == GGML_STATUS_SUCCESS);
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 1);
        check_output(items[1]);
        assert(server.wait_until_reply_paused());
        {
            std::lock_guard<std::mutex> pause(server.pause_mutex);
            assert(server.paused_replies == 1 && server.pause_mode.load() && server.reply_tokens == 0);
            check_request(items[2], 3);
            assert(server.native_counts[0] == 2*large_nq);
        }
        server.grant_reply(1);
        assert(ggml_backend_graph_compute(gpu, items[2].consume) == GGML_STATUS_SUCCESS);
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 0);
        server.unpause();
        check_output(items[2]);
        for (uint32_t i = 3; i < last; ++i) {
            {
                std::lock_guard<std::mutex> lock(server.mutex);
                set_reply_pattern(i, large_nq);
            }
            assert(ggml_backend_graph_compute(gpu, items[i].submit) == GGML_STATUS_SUCCESS);
            assert(ggml_backend_graph_compute(gpu, items[i].consume) == GGML_STATUS_SUCCESS);
            assert(ggml_backend_remote_attn_async_in_flight(backend) == 0);
            check_output(items[i]);
            {
                std::lock_guard<std::mutex> lock(server.mutex);
                check_request(items[i], i + 1);
            }
        }
        {
            std::lock_guard<std::mutex> lock(server.mutex);
            set_reply_pattern(last, 1);
        }
        auto * small_backend = mapped ? gpu : cpu;
        assert(ggml_backend_graph_compute(small_backend, items[last].submit) == GGML_STATUS_SUCCESS);
        assert(ggml_backend_graph_compute(small_backend, items[last].consume) == GGML_STATUS_SUCCESS);
        ggml_backend_sched_synchronize(sched);
        assert(!ggml_backend_remote_attn_failed(backend));
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 0);
        const std::string final = ggml_backend_remote_attn_stats_json(backend);
        assert(counter(final, "cuda_context_creates") == (mapped || tails ? 3u : 2u));
        assert(counter(final, "cuda_context_phase_releases") == (tails && !mapped ? 4u : 3u));
        assert(counter(final, "cuda_context_shared_acquires") == 2);
        assert(counter(final, "cuda_context_shared_releases") == 2);
        assert(counter(final, "cuda_context_shared_last_releases") == 1);
        assert(counter(final, "cuda_pack_calls") == (3u + tails + unsigned(mapped)));
        assert(counter(final, "cuda_unpack_calls") == (3u + tails + unsigned(mapped)));
        assert(counter(final, "cuda_small_pack_calls") == (mapped ? 1u : 0u));
        assert(counter(final, "cuda_small_unpack_calls") == (mapped ? 1u : 0u));
        assert(counter(final, "cohort_calls") == 0 && counter(final, "cohort_entries") == 0);
        // Check all earlier GGML OUT/LSE again after private/shared release.
        for (const auto & item : items) check_output(item);
        {
            std::lock_guard<std::mutex> lock(server.mutex);
            check_request(items[last], last + 1);
            assert(server.native_counts[0] == small_live && server.native_counts[1] == large_nq);
        }
        ggml_backend_remote_attn_drain(backend);
        // A second oracle survives destruction of every codec owner. Outputs,
        // their scheduler allocation and the GPU backend stay alive until after
        // this overwrite is read, so a teardown-retired old write cannot hide.
        for (const auto & item : items) {
            std::vector<float> sentinel(size_t(item.nq)*row_width, -17.0f);
            ggml_backend_tensor_set(item.node, sentinel.data(), 0, sentinel.size()*sizeof(float));
            if (item.wait->data != item.node->data)
                ggml_backend_tensor_set(item.wait, sentinel.data(), 0, sentinel.size()*sizeof(float));
        }
        ggml_backend_synchronize(gpu);
        ggml_remote_attn_set_active(nullptr);
        ggml_backend_free(backend);
        for (const auto & item : items) {
            for (auto * output : {item.node, item.wait}) {
                std::vector<float> observed(size_t(item.nq)*row_width);
                ggml_backend_tensor_get(output, observed.data(), 0, observed.size()*sizeof(float));
                for (float value : observed) assert(value == -17.0f);
            }
        }
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        ggml_backend_free(cpu);
        ggml_backend_free(gpu);
        std::printf("CUDA WF marker-only two slots: policy=%s large_nq=%u private_tails=%u then unmarked%s1; creates=%llu phase_drops=%llu/acquires2/releases2/last1, Q/pos/OUT/LSE and post-backend overwrite passed; no physical concurrency claim\n",
                policy, large_nq, tails, mapped ? "mappedGPU" : "CPU", counter(final, "cuda_context_creates"), counter(final, "cuda_context_phase_releases"));
    }
}


static void test_prefetch_public_controls() {
    storage_env native("kvarn4"), experimental("1", "GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL"),
        fusion("0", "GGML_RKVA_RANGE_FUSE"), precision("128", "GGML_RKVA_NATIVE_TAIL_TOKENS"),
        bulk("1", "GGML_RKVA_NATIVE_BULK"), wire("fp8", "GGML_RKVA_WIRE_FORMAT"),
        overlap("1", "GGML_RKVA_ASYNC_LOCAL_REMOTE"), profiling("1", "GGML_RKVA_PROFILE");
    const char *names[] = {"destroy session",      "reset",         "free/close",
                           "profile idle barrier", "drain/profile", "epoch invalidation",
                           "wrong cycle",          "wrong version", "oversized reply",
                           "truncated body",       "server error",  "truncated header",
                           "EOF before header"};
    for (int mode = 0; mode < 13; ++mode) {
        loopback server(false, true, false, false, false, false, 3, 0, true, 7 | 8 | 16 | 32);
        auto *backend = ggml_backend_remote_attn_init("127.0.0.1", server.port);
        ggml_remote_attn_geometry geo{1, 2, 1, 256, 512, 0, 0, 128, 3, 128, 2, 0, 0, false, false, 1.0f / 16};
        assert(ggml_backend_remote_attn_set_geometry(backend, &geo));
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 512));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(ggml_backend_remote_attn_configure_range(backend, 1024, 512, GGML_TYPE_Q4_0));
        ggml_remote_attn_set_active(backend);
        auto *tensors = ggml_init({4u << 20, nullptr, true});
        struct job {
            ggml_tensor *node, *wait, *ids, *pos;
        };
        auto make = [&] {
            auto *q = ggml_new_tensor_3d(tensors, GGML_TYPE_F32, 256, 2, 128);
            auto *k = ggml_new_tensor_2d(tensors, GGML_TYPE_Q4_0, 256, 512);
            auto *v = ggml_dup_tensor(tensors, k);
            auto *ids = ggml_new_tensor_1d(tensors, GGML_TYPE_I32, 512);
            auto *pos = ggml_new_tensor_1d(tensors, GGML_TYPE_I32, 128);
            auto *ok = ggml_new_tensor_3d(tensors, GGML_TYPE_F32, 256, 1, 128);
            auto *ov = ggml_dup_tensor(tensors, ok);
            auto *node = ggml_remote_attn_range(tensors, q, k, v, ids, pos, 0, 1024, 128, geo.kq_scale);
            ggml_remote_attn_range_set_native_sources(node, ok, ov);
            ggml_remote_attn_range_set_async(node);
            return job{node, ggml_remote_attn_range_wait(tensors, node), ids, pos};
        };
        job a = make(), b = make();
        auto *cpu = ggml_backend_cpu_init();
        auto *buf = ggml_backend_alloc_ctx_tensors(tensors, cpu);
        assert(buf);
        ggml_backend_buffer_clear(buf, 0);
        auto submit = [&](job &j, uint32_t live) {
            std::vector<int32_t> ids(512, -1), pos(128);
            for (uint32_t i = 0; i < live; ++i)
                ids[i] = 1024 + i;
            for (uint32_t i = 0; i < 128; ++i)
                pos[i] = 1024 + live - 128 + i;
            ggml_backend_tensor_set(j.ids, ids.data(), 0, ids.size() * 4);
            ggml_backend_tensor_set(j.pos, pos.data(), 0, pos.size() * 4);
            j.node->op_params[GGML_REMOTE_ATTN_PARAM_N_LIVE] = live;
            assert(ggml_remote_attn_exec(j.node));
        };
        if (mode >= 6) {
            const loopback::profile_failure faults[] = {
                loopback::profile_failure::cycle,        loopback::profile_failure::version,
                loopback::profile_failure::oversized,    loopback::profile_failure::truncated,
                loopback::profile_failure::server_error, loopback::profile_failure::short_header,
                loopback::profile_failure::none};
            server.fail_bulk_reply = faults[mode - 6];
            server.drop_attention = mode == 12;
            std::vector<float> sentinel(128 * 514, 42.0f);
            ggml_backend_tensor_set(a.wait, sentinel.data(), 0, sentinel.size() * sizeof(float));
            ggml_backend_tensor_set(b.wait, sentinel.data(), 0, sentinel.size() * sizeof(float));
        }
        server.set_pause(true);
        submit(a, 128);
        submit(b, 256);
        assert(server.wait_until_reply_paused());
        assert(ggml_backend_remote_attn_async_in_flight(backend) == 2);
        if (mode >= 6) {
            server.unpause();
            assert(!ggml_remote_attn_exec(a.wait));
            assert(!ggml_remote_attn_exec(b.wait));
            assert(ggml_backend_remote_attn_failed(backend));
            assert(!ggml_backend_remote_attn_native_kvarn_has_state(backend));
            for (auto *output : {a.wait, b.wait}) {
                const auto *values = static_cast<const float *>(output->data);
                for (size_t i = 0; i < 128 * 514; ++i)
                    assert(values[i] == 42.0f);
            }
        } else if (mode == 3) {
            auto report = std::async(std::launch::async,
                                     [&] { return std::string(ggml_backend_remote_attn_stats_json(backend)); });
            assert(report.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready);
            assert(report.get().find("tcp_prefetch_admitted") != std::string::npos);
            server.unpause();
            assert(ggml_remote_attn_exec(a.wait));
            assert(ggml_remote_attn_exec(b.wait));
            {
                std::lock_guard<std::mutex> lock(server.mutex);
                assert(server.profile_controls.size() == 1);
            }
            const std::string stats = ggml_backend_remote_attn_stats_json(backend);
            assert(stats.find("\"tcp_prefetch_completed\":2") != std::string::npos);
            {
                std::lock_guard<std::mutex> lock(server.mutex);
                assert(server.profile_controls.size() == 2);
            }
        } else {
            std::promise<void> entered;
            auto started = entered.get_future();
            auto control = std::async(std::launch::async, [&] {
                entered.set_value();
                if (mode == 0)
                    return ggml_backend_remote_attn_destroy_session(backend, 0);
                if (mode == 1)
                    return ggml_backend_remote_attn_reset(backend, 0);
                if (mode == 2) {
                    ggml_backend_free(backend);
                    return true;
                }
                if (mode == 5) {
                    ggml_backend_remote_attn_invalidate_range(backend);
                    return true;
                }
                ggml_backend_remote_attn_drain(backend);
                (void)ggml_backend_remote_attn_stats_json(backend);
                return true;
            });
            started.wait();
            assert(control.wait_for(std::chrono::milliseconds(30)) == std::future_status::timeout);
            server.unpause();
            assert(control.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
            assert(control.get());
            if (mode == 2)
                backend = nullptr;
            if (backend)
                assert(ggml_backend_remote_attn_async_in_flight(backend) == 0);
            if (mode == 5) {
                assert(ggml_backend_remote_attn_failed(backend));
                assert(!ggml_backend_remote_attn_native_kvarn_has_state(backend));
                assert(!ggml_remote_attn_exec(a.wait));
                assert(!ggml_remote_attn_exec(b.wait));
            }
            std::lock_guard<std::mutex> lock(server.mutex);
            if (mode != 5)
                assert(std::count(server.packet_ops.begin(), server.packet_ops.end(), RKVA_KVARN_RANGE_BULK_ATTN) == 2);
            const auto last_bulk =
                std::find(server.packet_ops.rbegin(), server.packet_ops.rend(), RKVA_KVARN_RANGE_BULK_ATTN).base() - 1;
            if (mode == 0)
                assert(std::find(last_bulk + 1, server.packet_ops.end(), RKVA_DESTROY_SESSION) !=
                       server.packet_ops.end());
            if (mode == 1)
                assert(std::find(last_bulk + 1, server.packet_ops.end(), RKVA_RESET) != server.packet_ops.end());
            if (mode == 4)
                assert(server.profile_controls.size() == 2);
        }
        ggml_remote_attn_set_active(nullptr);
        if (backend)
            ggml_backend_free(backend);
        ggml_backend_buffer_free(buf);
        ggml_free(tensors);
        ggml_backend_free(cpu);
        printf("PASS public control case %d: %s (two N128 frames)\n", mode, names[mode]);
    }
}

int main() {
    if (std::getenv("GGML_RKVA_TCP_PREFETCH_TEST_ONLY")) {
        storage_env prefetch("1", "GGML_RKVA_TCP_PREFETCH");
        storage_env producer("cpu", "GGML_RKVA_NATIVE_PRODUCER");
        storage_env codec("0", "GGML_RKVA_CUDA_FP8");
        test_phase6_double_buffer_async(128);
        test_async_failure_two_slots(128);
        test_prefetch_public_controls();
        std::puts("TCP prefetch: N128 public controls, FIFO/reuse, poison/publication passed");
        return 0;
    }

    if (std::getenv("GGML_RKVA_FAILURE_ONLY")) {
        test_failure_retirement();
        test_async_failure_two_slots();
        return 0;
    }
    if (std::getenv("GGML_RKVA_EPOCH_RACE_ONLY")) {
        test_done_epoch_invalidation_race();
        return 0;
    }
    if (std::getenv("GGML_RKVA_COHORT_TEST_ONLY")) {
        test_native_cohort_async();
        return 0;
    }
    test_native_cohort_async();
    test_cuda_fp8_serving_graph();
    test_cuda_fp8_prefill_mode_graph();
    test_cuda_wavefront_marker_only_two_slots();
    test_native_tail_serialization();
    test_native_physical_peer();
    test_old_native_client_ack_compatibility();
    test_native_rewind();
    test_native_snapshot_transport();
    test_failure_retirement();
    test_async_failure_two_slots();
    test_native_query_prefix_chunks();
    test_native_bulk_f16();
    test_native_bulk_fp8_splitzip();
    test_phase6_double_buffer_async();
    test_done_epoch_invalidation_race();
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
