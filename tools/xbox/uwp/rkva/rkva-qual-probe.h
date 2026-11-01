#pragma once

// Test-only console qualification probe core.
//
// This header is compiled only when XLLAMA_QUAL_PROBE is defined by the build.
// It is WinRT-free and performs no file, network, GPU or registry I/O so the
// same code can be exercised by a Linux CPU unit test. The UWP action that
// reads/writes LocalState lives in MainPage.cpp behind the same guard.
//
// Contract (docs/rkva-snapshot-console-qualification-matrix.md):
//   * one autopilot action, "rkva_qual_probe", never advertised in HELLO and
//     never added to the RKVA supported_ops mask;
//   * the response is built from a resident snapshot supplied by the live
//     RKVA engine. A field the engine cannot prove is emitted as JSON null
//     with an exact reason; zero/placeholder values are never substituted;
//   * a durability fault selector is parsed and refused with the exact
//     fail-closed seam. No staging object is written, renamed or deleted.

#if defined(XLLAMA_QUAL_PROBE)

#include "rkva-native-snapshot.h"

#include "../../../../vendor/nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace rkva_qual_probe {

using nlohmann::json;
namespace snap = rkva_native_snapshot;
using digest = rkva_snapshot::Digest;

inline constexpr bool compiled = true;
inline constexpr const char * kAction = "rkva_qual_probe";
inline constexpr const char * kAttestSchema = "rkva-console-attest-v1";
inline constexpr const char * kDurabilitySchema = "rkva-console-durability-v1";
inline constexpr size_t kMaxRequestBytes = 64u << 10;

// ---------------------------------------------------------------------------
// Real engine snapshot. Every bool says whether the value came from the live
// engine; a false bool is never rendered as a number.
// ---------------------------------------------------------------------------

struct layer_coverage {
    uint32_t tail_begin = 0;
    uint32_t stage_begin = 0;
    uint32_t live_rows = 0;
};

struct resident_state {
    bool available = false; // engine/session reachable at all
    std::string reason;     // exact reason when unavailable

    bool has_session_id = false;
    uint32_t session_id = 0;
    bool has_live_rows = false;
    uint32_t live_rows = 0;
    bool has_generation = false;
    uint64_t generation = 0;
    bool has_fence = false;
    uint64_t submitted_fence = 0;
    uint64_t completed_fence = 0;
    bool unconfirmed_submission = false;
    bool has_prefix_key = false; // RKVK host prefix key (host_state_sha256)
    digest prefix_key{};
    bool has_coverage = false;
    std::vector<layer_coverage> coverage;
    bool has_geometry = false;
    snap::Geometry geometry{};
    bool has_descriptor = false;
    snap::Descriptor descriptor{};

    // field name -> exact reason, filled by the engine adapter and extended
    // here for fields this core computes itself.
    std::map<std::string, std::string> unavailable_fields;
};

struct expectation {
    bool has_session_id = false;
    uint32_t session_id = 0;
    bool has_rows = false;
    uint32_t rows = 0;
    bool has_generation = false;
    uint64_t generation = 0;
    bool has_host_key = false;
    digest host_key{};
    bool has_prefix = false;
    digest prefix{};
};

struct fault_selector {
    std::string kind;   // sync_file | manifest_sync | directory_sync |
                        // rename_noreplace | rename_eio | stale_lock | crash_point
    std::string target; // staging | manifest | directory | lease | object
    bool has_chunk = false;
    uint32_t chunk = 0;
};

struct request {
    std::string probe_id;
    std::string case_name;
    bool negative = false;
    bool has_phase_flags = false;
    bool phase3 = false;
    bool phase4 = false;
    std::vector<std::string> caps;
    bool has_expect = false;
    expectation expect;
    bool has_fault = false;
    fault_selector fault;
};

struct build_info {
    std::string commit; // empty means "not embedded"; never invented
    bool phase3_compiled = false;
    bool phase3_enabled = false;
    bool phase4_compiled = false;
    bool phase4_enabled = false;
    // The Phase 3/4 libraries are not linked into the UWP app: the macros only
    // label what the probe translation unit was compiled with.
    bool phase3_linked = false;
    bool phase4_linked = false;
    bool managed_caps_negotiated = false; // never true in this tree
    std::vector<std::string> caps;        // actually advertised caps; empty
    std::string caps_status;
    std::string commit_status;
};

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

inline bool is_hex128(const std::string & s) {
    if (s.size() != 32)
        return false;
    for (char c : s)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
            return false;
    return true;
}

inline bool is_hex64(const std::string & s) {
    if (s.size() != 64)
        return false;
    for (char c : s)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
            return false;
    return true;
}

inline bool parse_hex64(const std::string & s, digest & out) {
    if (!is_hex64(s))
        return false;
    for (size_t i = 0; i < 32; ++i) {
        const auto nibble = [](char c) -> uint8_t {
            return uint8_t(c <= '9' ? c - '0' : c - 'a' + 10);
        };
        out[i] = uint8_t(uint8_t(nibble(s[2 * i]) << 4) | nibble(s[2 * i + 1]));
    }
    return true;
}

inline std::string hex(const digest & value) {
    return rkva_snapshot::hex(value.data());
}

inline void put_u32(std::vector<uint8_t> & out, uint32_t value) {
    for (int i = 0; i < 4; ++i)
        out.push_back(uint8_t(value >> (8 * i)));
}
inline void put_u64(std::vector<uint8_t> & out, uint64_t value) {
    for (int i = 0; i < 8; ++i)
        out.push_back(uint8_t(value >> (8 * i)));
}
inline void put_text(std::vector<uint8_t> & out, const char * text, size_t size) {
    out.insert(out.end(), text, text + size);
}
inline void put_domain(std::vector<uint8_t> & out, const char * domain) {
    put_text(out, domain, std::strlen(domain));
}

// Canonical geometry/descriptor/coverage digests. The host-side producer
// (tools/xbox/scripts/rkva-host-expect.py) mirrors these byte layouts; a
// mismatch is a real geometry/coverage difference, not a formatting choice.
inline digest geometry_digest(const snap::Geometry & g) {
    std::vector<uint8_t> body;
    put_domain(body, "RKVA-QUAL-GEOMETRY-V1");
    put_u32(body, g.n_head);
    put_u32(body, g.n_head_kv);
    put_u32(body, g.head_dim);
    put_u32(body, g.head_slices);
    put_u32(body, g.max_ctx);
    put_u32(body, g.n_layers);
    return rkva_snapshot::digest(body.data(), body.size());
}

inline digest descriptor_digest(const snap::Descriptor & d) {
    std::vector<uint8_t> body;
    put_domain(body, "RKVA-QUAL-DESCRIPTOR-V1");
    const uint32_t fields[18] = {
        d.origin,      d.capacity,    d.storage,       d.layout_version, d.token_group,
        d.record_dim,  d.head_dim,    d.head_slices,   d.key_bits,        d.value_bits,
        d.record_bytes, d.stage_groups, d.tail_groups, d.sink_tokens,     d.precision_tail_tokens,
        d.wht_version, d.domain,      d.record_policy};
    for (uint32_t value : fields)
        put_u32(body, value);
    return rkva_snapshot::digest(body.data(), body.size());
}

inline digest coverage_digest(const resident_state & s) {
    std::vector<uint8_t> body;
    put_domain(body, "RKVA-QUAL-COVERAGE-V1");
    put_u32(body, s.live_rows);
    put_u64(body, s.generation);
    put_u32(body, uint32_t(s.coverage.size()));
    for (size_t i = 0; i < s.coverage.size(); ++i) {
        put_u32(body, uint32_t(i));
        put_u32(body, s.coverage[i].tail_begin);
        put_u32(body, s.coverage[i].stage_begin);
        // One checkpoint: every layer must hold the same live row count.
        put_u32(body, s.live_rows);
    }
    return rkva_snapshot::digest(body.data(), body.size());
}

inline digest state_digest(const resident_state & s) {
    std::vector<uint8_t> body;
    put_domain(body, "RKVA-QUAL-STATE-V1");
    const auto g = geometry_digest(s.geometry);
    const auto d = descriptor_digest(s.descriptor);
    const auto c = coverage_digest(s);
    body.insert(body.end(), g.begin(), g.end());
    body.insert(body.end(), d.begin(), d.end());
    body.insert(body.end(), c.begin(), c.end());
    // Session id and fences are deliberately excluded so the host producer can
    // compute the same structural digest from the native manifest. They are
    // compared as separate fields, never folded into this value.
    return rkva_snapshot::digest(body.data(), body.size());
}

// ---------------------------------------------------------------------------
// Request parsing. Rejects anything ambiguous; never defaults a field that
// the caller must prove (phase flags are the only optional opt-in).
// ---------------------------------------------------------------------------

inline bool get_u32(const json & object, const char * key, bool & present, uint32_t & out) {
    present = object.contains(key);
    if (!present)
        return true;
    if (!object[key].is_number_unsigned())
        return false;
    const auto value = object[key].get<uint64_t>();
    if (value > UINT32_MAX)
        return false;
    out = uint32_t(value);
    return true;
}

inline bool get_u64(const json & object, const char * key, bool & present, uint64_t & out) {
    present = object.contains(key);
    if (!present)
        return true;
    if (!object[key].is_number_unsigned())
        return false;
    out = object[key].get<uint64_t>();
    return true;
}

inline bool get_digest(const json & object, const char * key, bool & present, digest & out) {
    present = object.contains(key);
    if (!present)
        return true;
    if (!object[key].is_string())
        return false;
    return parse_hex64(object[key].get<std::string>(), out);
}

inline bool parse_expect(const json & value, expectation & out) {
    if (!value.is_object())
        return false;
    if (!get_u32(value, "session_id", out.has_session_id, out.session_id) ||
        !get_u32(value, "rows", out.has_rows, out.rows) ||
        !get_u64(value, "generation", out.has_generation, out.generation) ||
        !get_digest(value, "host_key_sha256", out.has_host_key, out.host_key) ||
        !get_digest(value, "prefix_sha256", out.has_prefix, out.prefix))
        return false;
    if (out.has_session_id && out.session_id == 0)
        return false;
    if (out.has_rows && out.rows == 0)
        return false;
    if (out.has_generation && out.generation == 0)
        return false;
    return out.has_session_id || out.has_rows || out.has_generation || out.has_host_key ||
           out.has_prefix;
}

inline bool fault_kind_known(const std::string & kind) {
    static const char * const kKinds[] = {"sync_file",     "manifest_sync", "directory_sync",
                                          "rename_noreplace", "rename_eio", "stale_lock",
                                          "crash_point"};
    return std::find_if(std::begin(kKinds), std::end(kKinds),
                        [&](const char * k) { return kind == k; }) != std::end(kKinds);
}

inline bool fault_target_known(const std::string & target) {
    static const char * const kTargets[] = {"staging", "manifest", "directory", "lease", "object"};
    return target.empty() ||
           std::find_if(std::begin(kTargets), std::end(kTargets),
                        [&](const char * t) { return target == t; }) != std::end(kTargets);
}

inline bool parse_fault(const json & value, fault_selector & out) {
    if (!value.is_object())
        return false;
    if (!value.contains("kind") || !value["kind"].is_string())
        return false;
    out.kind = value["kind"].get<std::string>();
    if (!fault_kind_known(out.kind))
        return false;
    if (value.contains("target")) {
        if (!value["target"].is_string())
            return false;
        out.target = value["target"].get<std::string>();
        if (!fault_target_known(out.target))
            return false;
    }
    if (!get_u32(value, "chunk", out.has_chunk, out.chunk) || out.chunk > 4095)
        return false;
    if (out.kind == "sync_file" && !out.has_chunk)
        return false;
    return true;
}

inline bool parse_request(const std::string & text, request & out, std::string & error) {
    if (text.empty() || text.size() > kMaxRequestBytes) {
        error = "request missing or over 64 KiB";
        return false;
    }
    const auto document = json::parse(text, nullptr, false);
    if (document.is_discarded() || !document.is_object()) {
        error = "request is not valid JSON object";
        return false;
    }
    if (!document.contains("schema") || !document["schema"].is_string()) {
        error = "request.schema missing";
        return false;
    }
    const auto schema = document["schema"].get<std::string>();
    if (schema != kAttestSchema && schema != kDurabilitySchema) {
        error = "request.schema is not " + std::string(kAttestSchema) + " or " +
                kDurabilitySchema;
        return false;
    }
    if (!document.contains("probe_id") || !document["probe_id"].is_string() ||
        !is_hex128(document["probe_id"].get<std::string>())) {
        error = "request.probe_id is not 128-bit lowercase hex";
        return false;
    }
    out.probe_id = document["probe_id"].get<std::string>();
    if (!document.contains("case") || !document["case"].is_string() ||
        document["case"].get<std::string>().empty() ||
        document["case"].get<std::string>().size() > 64) {
        error = "request.case missing or invalid";
        return false;
    }
    out.case_name = document["case"].get<std::string>();
    if (document.contains("negative")) {
        if (!document["negative"].is_boolean()) {
            error = "request.negative is not a boolean";
            return false;
        }
        out.negative = document["negative"].get<bool>();
    }
    if (document.contains("phase_flags")) {
        const auto & flags = document["phase_flags"];
        if (!flags.is_object() || !flags.contains("phase3") || !flags.contains("phase4") ||
            !flags["phase3"].is_boolean() || !flags["phase4"].is_boolean()) {
            error = "request.phase_flags must carry boolean phase3 and phase4";
            return false;
        }
        out.has_phase_flags = true;
        out.phase3 = flags["phase3"].get<bool>();
        out.phase4 = flags["phase4"].get<bool>();
        if (flags.contains("caps")) {
            if (!flags["caps"].is_array() || flags["caps"].empty()) {
                error = "request.phase_flags.caps must be a non-empty array";
                return false;
            }
            for (const auto & cap : flags["caps"]) {
                if (!cap.is_string()) {
                    error = "request.phase_flags.caps entry is not a string";
                    return false;
                }
                const auto name = cap.get<std::string>();
                if (name != "REGISTRY_V1" && name != "AUTH_V2" && name != "FRAMING_V3") {
                    error = "request.phase_flags.caps contains unknown capability " + name;
                    return false;
                }
                out.caps.push_back(name);
            }
        }
    }
    if (document.contains("expect")) {
        out.has_expect = true;
        if (!parse_expect(document["expect"], out.expect)) {
            error = "request.expect is invalid";
            return false;
        }
    }
    if (document.contains("fault")) {
        out.has_fault = true;
        if (!parse_fault(document["fault"], out.fault)) {
            error = "request.fault selector is invalid";
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Comparison policy used for both the positive and the negative case.
// ---------------------------------------------------------------------------

inline bool valid_fences(uint64_t submitted, uint64_t completed) {
    return submitted != 0 && submitted != UINT64_MAX && completed != UINT64_MAX &&
           completed >= submitted;
}

inline std::vector<std::string> compare_expectation(const resident_state & s,
                                                    const expectation & e) {
    std::vector<std::string> failures;
    if (!e.has_session_id || !s.has_session_id || e.session_id != s.session_id)
        failures.push_back("session_mismatch");
    if (!e.has_rows || !s.has_live_rows || e.rows != s.live_rows)
        failures.push_back("rows_mismatch");
    if (!e.has_generation || !s.has_generation || e.generation != s.generation)
        failures.push_back("generation_mismatch");
    if (!e.has_host_key || !s.has_prefix_key || e.host_key != s.prefix_key)
        failures.push_back("key_mismatch");
    if (!e.has_prefix || !s.has_prefix_key || e.prefix != s.prefix_key)
        failures.push_back("prefix_mismatch");
    if (!s.has_fence || !valid_fences(s.submitted_fence, s.completed_fence))
        failures.push_back("fence_regression");
    return failures;
}

inline expectation baseline_expectation(const resident_state & s) {
    expectation e;
    e.has_session_id = s.has_session_id;
    e.session_id = s.session_id;
    e.has_rows = s.has_live_rows;
    e.rows = s.live_rows;
    e.has_generation = s.has_generation;
    e.generation = s.generation;
    e.has_host_key = s.has_prefix_key;
    e.host_key = s.prefix_key;
    e.has_prefix = s.has_prefix_key;
    e.prefix = s.prefix_key;
    return e;
}

inline digest flip_bit(digest value, size_t bit) {
    value[bit / 8] ^= uint8_t(1u << (bit % 8));
    return value;
}

inline json negative_evidence(const resident_state & s) {
    json out;
    const char * const kKinds[] = {"generation_mismatch", "key_mismatch", "session_mismatch",
                                   "prefix_mismatch", "fence_regression"};
    out["kinds"] = json::array();
    for (const char * kind : kKinds)
        out["kinds"].push_back(kind);
    out["policy"] = "rkva-qual-attest-compare-v1";
    out["source"] = "resident engine state; tampered expectations are in-memory copies";
    out["results"] = json::object();
    out["all_refused"] = false;
    out["accepted_any"] = false;
    out["coverage_mac_emitted"] = false;
    out["mac_status"] =
        "no authenticated managed session: the ATTEST HMAC key is not provisioned";
    json results = json::object();
    bool all_refused = true;
    bool accepted_any = false;
    const auto base = baseline_expectation(s);
    if (!s.available || !s.has_session_id || !s.has_generation || !s.has_fence ||
        !s.has_prefix_key || s.submitted_fence == 0 || s.submitted_fence == UINT64_MAX) {
        for (const char * kind : kKinds)
            results[kind] = "unavailable";
        out["results"] = results;
        out["all_refused"] = false;
        return out;
    }
    if (compare_expectation(s, base).empty()) {
        // The untampered baseline must be accepted; otherwise the policy
        // itself is broken and no negative claim may be made.
        out["baseline"] = "accepted";
    } else {
        out["baseline"] = "rejected";
        for (const char * kind : kKinds)
            results[kind] = "policy_unavailable";
        out["results"] = results;
        out["all_refused"] = false;
        return out;
    }
    struct mutation {
        const char * kind;
        expectation value;
    };
    std::vector<mutation> mutations;
    {
        auto v = base;
        v.generation = v.generation + 1;
        mutations.push_back({"generation_mismatch", v});
    }
    {
        auto v = base;
        v.host_key = flip_bit(v.host_key, 7);
        mutations.push_back({"key_mismatch", v});
    }
    {
        auto v = base;
        v.session_id = v.session_id + 1;
        mutations.push_back({"session_mismatch", v});
    }
    {
        auto v = base;
        v.prefix = flip_bit(v.prefix, 11);
        mutations.push_back({"prefix_mismatch", v});
    }
    // A regressed fence is expressed by the same comparator over the real
    // fence pair: completed one below submitted must be refused.
    const bool fence_refused = !valid_fences(s.submitted_fence, s.completed_fence - 1);
    for (const auto & m : mutations) {
        const auto failures = compare_expectation(s, m.value);
        const bool refused =
            std::find(failures.begin(), failures.end(), m.kind) != failures.end();
        results[m.kind] = refused ? "refused" : "accepted";
        all_refused = all_refused && refused;
        accepted_any = accepted_any || !refused;
    }
    results["fence_regression"] = fence_refused ? "refused" : "accepted";
    all_refused = all_refused && fence_refused;
    accepted_any = accepted_any || !fence_refused;
    out["results"] = results;
    out["all_refused"] = all_refused;
    out["accepted_any"] = accepted_any;
    return out;
}

// ---------------------------------------------------------------------------
// Response construction
// ---------------------------------------------------------------------------

inline json build_json(const build_info & b) {
    json out;
    out["commit"] = b.commit.empty() ? json(nullptr) : json(b.commit);
    out["commit_status"] =
        b.commit.empty() ? "the UWP build did not embed XLLAMA_QUAL_COMMIT" : "";
    out["phase3_compiled"] = b.phase3_compiled;
    out["phase3_enabled"] = b.phase3_enabled;
    out["phase4_compiled"] = b.phase4_compiled;
    out["phase4_enabled"] = b.phase4_enabled;
    out["phase3_linked"] = b.phase3_linked;
    out["phase4_linked"] = b.phase4_linked;
    out["managed_caps_negotiated"] = b.managed_caps_negotiated;
    out["caps"] = b.caps;
    out["caps_status"] = b.caps_status;
    return out;
}

inline json coverage_json(const resident_state & s) {
    json layers = json::array();
    for (size_t i = 0; i < s.coverage.size(); ++i) {
        json row;
        row["layer"] = i;
        row["tail_begin"] = s.coverage[i].tail_begin;
        row["stage_begin"] = s.coverage[i].stage_begin;
        row["live_rows"] = s.coverage[i].live_rows;
        layers.push_back(row);
    }
    return layers;
}

inline json geometry_json(const snap::Geometry & g) {
    return json{{"n_head", g.n_head},
                {"n_head_kv", g.n_head_kv},
                {"head_dim", g.head_dim},
                {"head_slices", g.head_slices},
                {"max_ctx", g.max_ctx},
                {"n_layers", g.n_layers}};
}

inline json descriptor_json(const snap::Descriptor & d) {
    return json{{"origin", d.origin},
                {"capacity", d.capacity},
                {"storage", d.storage},
                {"layout_version", d.layout_version},
                {"token_group", d.token_group},
                {"record_dim", d.record_dim},
                {"head_dim", d.head_dim},
                {"head_slices", d.head_slices},
                {"key_bits", d.key_bits},
                {"value_bits", d.value_bits},
                {"record_bytes", d.record_bytes},
                {"stage_groups", d.stage_groups},
                {"tail_groups", d.tail_groups},
                {"sink_tokens", d.sink_tokens},
                {"precision_tail_tokens", d.precision_tail_tokens},
                {"wht_version", d.wht_version},
                {"domain", d.domain},
                {"record_policy", d.record_policy}};
}

inline std::string join(const std::vector<std::string> & values) {
    std::string out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i)
            out += "; ";
        out += values[i];
    }
    return out;
}

// Canonical manifest digest of the live resident state. Only available when
// every input is real: geometry, descriptor, rows, generation, coverage and
// the host prefix key. Nothing is synthesized.
inline bool metadata_digest(const resident_state & s, digest & out, std::string & error) {
    if (!s.has_geometry || !s.has_descriptor || !s.has_live_rows || !s.has_generation ||
        !s.has_coverage || !s.has_prefix_key) {
        error = "canonical metadata needs geometry, descriptor, rows, generation, coverage "
                "and the host prefix key";
        return false;
    }
    std::vector<snap::LayerCoverage> layers;
    layers.reserve(s.coverage.size());
    for (size_t i = 0; i < s.coverage.size(); ++i) {
        snap::LayerCoverage layer;
        layer.tail_begin = s.coverage[i].tail_begin;
        layer.stage_begin = s.coverage[i].stage_begin;
        layer.record_groups = s.coverage[i].live_rows / 128;
        layer.ready = 1;
        layers.push_back(layer);
    }
    snap::Manifest manifest;
    std::vector<uint8_t> bytes;
    if (!snap::build(s.geometry, s.descriptor, s.live_rows, s.generation,
                     s.prefix_key.data(), layers, manifest, &error) ||
        !snap::encode(manifest, bytes, &error)) {
        if (error.empty())
            error = "native manifest rejected";
        return false;
    }
    out = rkva_snapshot::digest(bytes.data(), bytes.size());
    return true;
}

inline json attestation_json(const resident_state & s,
                             std::map<std::string, std::string> unavailable) {
    json out;
    const auto set_null = [&](const char * name) {
        out[name] = nullptr;
        if (!unavailable.count(name))
            unavailable[name] = "engine did not provide " + std::string(name);
    };
    if (s.has_session_id)
        out["session_id"] = s.session_id;
    else
        set_null("session_id");
    if (s.has_live_rows)
        out["live_rows"] = s.live_rows;
    else
        set_null("live_rows");
    if (s.has_generation)
        out["generation"] = s.generation;
    else
        set_null("generation");
    if (s.has_fence) {
        out["submitted_fence"] = s.submitted_fence;
        out["completed_fence"] = s.completed_fence;
    } else {
        set_null("submitted_fence");
        set_null("completed_fence");
    }
    if (s.has_prefix_key) {
        out["host_key_sha256"] = hex(s.prefix_key);
        out["prefix_sha256"] = hex(s.prefix_key);
        // Same real value: host_state_sha256 is the SHA-256 of the complete
        // host prefix (token IDs + KV state) through the end of the RKVK frame.
        out["token_prefix_sha256"] = hex(s.prefix_key);
    } else {
        set_null("host_key_sha256");
        set_null("prefix_sha256");
        set_null("token_prefix_sha256");
        unavailable["prefix_key"] =
            "the engine retains no RKVK host prefix key: this build accepts it only from a "
            "native snapshot SAVE/RESTORE request";
    }
    out["coverage"] = s.has_coverage ? coverage_json(s) : json(nullptr);
    if (!s.has_coverage)
        unavailable["coverage"] = "the live session has no published revision-3 coverage";
    out["geometry"] = s.has_geometry ? geometry_json(s.geometry) : json(nullptr);
    if (!s.has_geometry)
        unavailable["geometry"] = "the engine has no negotiated geometry";
    out["descriptor"] = s.has_descriptor ? descriptor_json(s.descriptor) : json(nullptr);
    if (!s.has_descriptor)
        unavailable["descriptor"] = "the live session has no revision-3 range descriptor";
    if (s.has_geometry)
        out["geometry_sha256"] = hex(geometry_digest(s.geometry));
    else
        out["geometry_sha256"] = nullptr;
    if (s.has_descriptor)
        out["descriptor_sha256"] = hex(descriptor_digest(s.descriptor));
    else
        out["descriptor_sha256"] = nullptr;
    if (s.has_coverage && s.has_live_rows && s.has_generation)
        out["coverage_sha256"] = hex(coverage_digest(s));
    else
        out["coverage_sha256"] = nullptr;
    if (s.has_geometry && s.has_descriptor && s.has_coverage && s.has_live_rows &&
        s.has_generation)
        out["state_digest"] = hex(state_digest(s));
    else
        out["state_digest"] = nullptr;
    digest metadata{};
    std::string metadata_error;
    if (metadata_digest(s, metadata, metadata_error))
        out["metadata_sha256"] = hex(metadata);
    else {
        out["metadata_sha256"] = nullptr;
        unavailable["metadata_sha256"] = metadata_error;
    }
    out["coverage_mac"] = nullptr;
    unavailable["coverage_mac"] =
        "no authenticated managed session: the ATTEST HMAC key is not provisioned";
    out["unconfirmed_submission"] = s.unconfirmed_submission;
    json fields = json::object();
    for (const auto & kv : unavailable)
        fields[kv.first] = kv.second;
    out["unavailable_fields"] = fields;
    return out;
}

inline std::string required_status_reason(const json & attestation) {
    static const char * const kRequired[] = {"session_id",        "live_rows",
                                             "generation",        "submitted_fence",
                                             "completed_fence",   "host_key_sha256",
                                             "prefix_sha256"};
    std::vector<std::string> missing;
    for (const char * name : kRequired)
        if (!attestation.contains(name) || attestation[name].is_null()) {
            const auto & fields = attestation["unavailable_fields"];
            missing.push_back(std::string(name) + ": " +
                              (fields.contains(name) ? fields[name].get<std::string>()
                                                     : "unavailable"));
        }
    return join(missing);
}

inline std::string make_response(const request & req, const build_info & build,
                                 const resident_state & state) {
    json document;
    document["schema"] = kAttestSchema;
    document["probe_id"] = req.probe_id;
    document["case"] = req.case_name;
    document["build"] = build_json(build);
    auto unavailable = state.unavailable_fields;
    if (!state.available && !state.reason.empty())
        unavailable["engine"] = state.reason;
    const auto attestation = attestation_json(state, unavailable);
    document["attestation"] = attestation;
    const auto reason = required_status_reason(attestation);
    document["status"] = reason.empty() ? "available" : "unavailable";
    document["reason"] =
        reason.empty()
            ? ""
            : (state.available
                   ? "live engine state is missing required fields: " + reason
                   : (state.reason.empty() ? "live engine state is unavailable"
                                           : state.reason) +
                         ": " + reason);
    if (req.negative)
        document["negative"] = negative_evidence(state);
    return document.dump(2) + "\n";
}

// A structurally invalid request still gets a response document, so the host
// can distinguish "probe ran and refused" from "probe never ran". It never
// falls back to a fabricated positive attestation.
inline std::string make_invalid_response(const std::string & probe_id,
                                         const std::string & why) {
    json document;
    document["schema"] = kAttestSchema;
    document["probe_id"] = is_hex128(probe_id) ? json(probe_id) : json(nullptr);
    document["status"] = "refused";
    document["reason"] = "invalid probe request: " + why;
    document["build"] = json(nullptr);
    document["attestation"] = json(nullptr);
    return document.dump(2) + "\n";
}

// ---------------------------------------------------------------------------
// Durability staging/fault selector: precise refusal.
//
// The v3 object writer is unconditionally fail-closed before any sync, rename
// or staging I/O (rkva-managed-object-v3.cpp:115 — "concrete authenticated pin
// and metadata proof unavailable"), and no qualified durable backend exists.
// Injecting a fault at a real seam would require opening that writer and the
// managed success path, which this task forbids. The only safe action is to
// record the exact refusal with no filesystem mutation.
// ---------------------------------------------------------------------------

inline std::string fault_refusal_reason(const fault_selector & f) {
    if (f.kind == "sync_file" || f.kind == "manifest_sync" || f.kind == "directory_sync")
        return "staging fault '" + f.kind +
               "' unavailable: rkva_managed_object_v3::write() returns at "
               "tools/xbox/uwp/rkva/rkva-managed-object-v3.cpp:115 ('concrete authenticated "
               "pin and metadata proof unavailable') before durable_filesystem::sync_file/"
               "sync_directory is ever called, so no sync seam exists to inject into";
    if (f.kind == "rename_noreplace" || f.kind == "rename_eio")
        return "staging fault '" + f.kind +
               "' unavailable: the same fail-closed guard at "
               "tools/xbox/uwp/rkva/rkva-managed-object-v3.cpp:115 precedes "
               "durable_filesystem::rename_noreplace_and_sync, and no qualified platform "
               "implementation of that method exists in this tree";
    if (f.kind == "stale_lock")
        return "staging fault 'stale_lock' unavailable: rkva_managed_store::lease is an "
               "in-process exclusive lock only; no cross-process owner-evidence/reclaim "
               "implementation exists, so PID reuse cannot be tested from the app";
    if (f.kind == "crash_point")
        return "staging fault 'crash_point' unavailable: there is no staging writer to "
               "crash between rename and parent-directory sync; the v3 writer is closed at "
               "tools/xbox/uwp/rkva/rkva-managed-object-v3.cpp:115";
    return "staging fault unavailable: unknown selector";
}

inline std::string make_fault_response(const request & req, const build_info & build) {
    json document;
    document["schema"] = kDurabilitySchema;
    document["probe_id"] = req.probe_id;
    document["case"] = req.case_name;
    document["status"] = "refused";
    document["selector_valid"] = true;
    document["fault"] = json{{"kind", req.fault.kind},
                             {"target", req.fault.target},
                             {"chunk", req.fault.has_chunk ? json(req.fault.chunk)
                                                           : json(nullptr)}};
    document["reason"] = fault_refusal_reason(req.fault);
    document["wrote_anything"] = false;
    document["writer_reachable"] = false;
    document["writer_gate"] = json::array(
        {"tools/xbox/uwp/rkva/rkva-managed-object-v3.cpp:115: need(false, \"concrete "
         "authenticated pin and metadata proof unavailable\")",
         "tools/xbox/uwp/rkva/rkva-managed-object-v3.cpp:92: verify() has the same "
         "fail-closed guard",
         "ggml/src/ggml-remote-attn/rkva-anchored-sqlite.cpp: managed apply refuses every "
         "transition without the concrete pin/metadata proof"});
    document["next_action"] =
        "record D3-D7 as BLOCKED; do not open the v3 writer or the managed success path";
    document["build"] = build_json(build);
    return document.dump(2) + "\n";
}

// ---------------------------------------------------------------------------
// Resident-state provider. The live RKVA listener registers a callback under
// its state mutex; there is deliberately no global engine handle.
// ---------------------------------------------------------------------------

using provider_fn = bool (*)(void * context, uint32_t requested_session,
                             resident_state & out, std::string & error);

struct provider_slot {
    provider_fn fn = nullptr;
    void * context = nullptr;
};

inline provider_slot & provider() {
    static provider_slot slot;
    return slot;
}

inline void set_provider(provider_fn fn, void * context) {
    provider().fn = fn;
    provider().context = context;
}

inline bool query_resident(uint32_t requested_session, resident_state & out,
                           std::string & error) {
    const auto & slot = provider();
    if (!slot.fn) {
        out = resident_state{};
        out.reason =
            "no live RKVA engine exists in this app process: the listener has no accepted "
            "HELLO/connection, so no session, coverage, generation or fence can be attested";
        error = out.reason;
        return false;
    }
    return slot.fn(slot.context, requested_session, out, error);
}

} // namespace rkva_qual_probe

#else // !XLLAMA_QUAL_PROBE

namespace rkva_qual_probe {
inline constexpr bool compiled = false;
inline constexpr const char * kAction = "rkva_qual_probe";
} // namespace rkva_qual_probe

#endif // XLLAMA_QUAL_PROBE
