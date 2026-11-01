// CPU contract test for the test-only console qualification probe.
//
// It exercises the WinRT-free core only: request parsing, the response built
// from an injected resident state, the canonical digests the host producer
// mirrors, per-field unavailability, the negative-case comparison policy and
// the precise durability-fault refusal. No UWP, D3D12, engine, network, file
// or console access.

#include "rkva/rkva-qual-probe.h"

#undef NDEBUG
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace qp = rkva_qual_probe;
namespace snap = rkva_native_snapshot;
using json = nlohmann::json;

static constexpr const char* kProbeId = "0123456789abcdef0123456789abcdef";

static qp::digest key_bytes() {
    qp::digest key{};
    for (size_t i = 0; i < key.size(); ++i)
        key[i] = uint8_t(i + 1);
    return key;
}

static qp::resident_state complete_state() {
    qp::resident_state state;
    state.available = true;
    state.has_session_id = true;
    state.session_id = 7;
    state.has_live_rows = true;
    state.live_rows = 513;
    state.has_generation = true;
    state.generation = 0x700000001ull;
    state.has_fence = true;
    state.submitted_fence = 41;
    state.completed_fence = 41;
    state.has_prefix_key = true;
    state.prefix_key = key_bytes();
    state.has_geometry = true;
    state.geometry = snap::Geometry{8, 2, 256, 2, 4096, 2};
    state.has_descriptor = true;
    state.descriptor = snap::Descriptor{0, 4096, RKVA_RANGE_STORAGE_KVARN4, 3, 128, 128,
                                        256, 2, 4, 4, 8960, 3, 2, 128, 128, 1, 1, 1};
    state.has_coverage = true;
    state.coverage = {{385, 384, 513}, {385, 384, 513}};
    return state;
}

static qp::build_info complete_build() {
    qp::build_info build;
    build.commit = "3a02e90a2";
    build.phase3_compiled = true;
    build.phase3_enabled = true;
    build.phase4_compiled = true;
    build.phase4_enabled = true;
    build.phase3_linked = false;
    build.phase4_linked = false;
    return build;
}

static qp::request attest_request(const qp::resident_state& state, bool negative = false) {
    qp::request req;
    req.probe_id = kProbeId;
    req.case_name = negative ? "negative" : "positive";
    req.negative = negative;
    req.has_phase_flags = true;
    req.phase3 = true;
    req.phase4 = true;
    req.caps = {"REGISTRY_V1", "AUTH_V2", "FRAMING_V3"};
    req.has_expect = true;
    req.expect = qp::baseline_expectation(state);
    return req;
}

// Independent byte layout so a silent change to the probe's canonical digests
// fails here rather than only on the console.
static std::string sha_hex(const std::vector<uint8_t>& bytes) {
    const auto digest = rkva_snapshot::digest(bytes.data(), bytes.size());
    return rkva_snapshot::hex(digest.data());
}
static void put32(std::vector<uint8_t>& out, uint32_t value) {
    for (int i = 0; i < 4; ++i)
        out.push_back(uint8_t(value >> (8 * i)));
}
static void put64(std::vector<uint8_t>& out, uint64_t value) {
    for (int i = 0; i < 8; ++i)
        out.push_back(uint8_t(value >> (8 * i)));
}
static std::string expected_geometry_digest(const snap::Geometry& g) {
    std::vector<uint8_t> body{'R', 'K', 'V', 'A', '-', 'Q', 'U', 'A', 'L', '-',
                              'G', 'E', 'O', 'M', 'E', 'T', 'R', 'Y', '-', 'V', '1'};
    for (uint32_t value : {g.n_head, g.n_head_kv, g.head_dim, g.head_slices, g.max_ctx,
                           g.n_layers})
        put32(body, value);
    return sha_hex(body);
}
static std::string expected_descriptor_digest(const snap::Descriptor& d) {
    std::vector<uint8_t> body{'R', 'K', 'V', 'A', '-', 'Q', 'U', 'A', 'L', '-',
                              'D', 'E', 'S', 'C', 'R', 'I', 'P', 'T', 'O', 'R', '-',
                              'V', '1'};
    for (uint32_t value : {d.origin, d.capacity, d.storage, d.layout_version, d.token_group,
                           d.record_dim, d.head_dim, d.head_slices, d.key_bits, d.value_bits,
                           d.record_bytes, d.stage_groups, d.tail_groups, d.sink_tokens,
                           d.precision_tail_tokens, d.wht_version, d.domain, d.record_policy})
        put32(body, value);
    return sha_hex(body);
}
static std::string expected_coverage_digest(const qp::resident_state& s) {
    std::vector<uint8_t> body{'R', 'K', 'V', 'A', '-', 'Q', 'U', 'A', 'L', '-',
                              'C', 'O', 'V', 'E', 'R', 'A', 'G', 'E', '-', 'V', '1'};
    put32(body, s.live_rows);
    put64(body, s.generation);
    put32(body, uint32_t(s.coverage.size()));
    for (size_t i = 0; i < s.coverage.size(); ++i) {
        put32(body, uint32_t(i));
        put32(body, s.coverage[i].tail_begin);
        put32(body, s.coverage[i].stage_begin);
        put32(body, s.live_rows);
    }
    return sha_hex(body);
}
static std::string expected_state_digest(const qp::resident_state& s) {
    std::vector<uint8_t> body{'R', 'K', 'V', 'A', '-', 'Q', 'U', 'A', 'L', '-',
                              'S', 'T', 'A', 'T', 'E', '-', 'V', '1'};
    const auto add = [&](const std::string& hex) {
        for (size_t i = 0; i < hex.size(); i += 2)
            body.push_back(uint8_t(std::stoi(hex.substr(i, 2), nullptr, 16)));
    };
    add(expected_geometry_digest(s.geometry));
    add(expected_descriptor_digest(s.descriptor));
    add(expected_coverage_digest(s));
    return sha_hex(body);
}

static void test_parse_valid() {
    const std::string text = R"({
      "schema": "rkva-console-attest-v1",
      "probe_id": "0123456789abcdef0123456789abcdef",
      "case": "positive",
      "negative": false,
      "phase_flags": {"phase3": true, "phase4": true,
                      "caps": ["REGISTRY_V1", "AUTH_V2", "FRAMING_V3"]},
      "expect": {"session_id": 7, "rows": 513, "generation": 30064771073,
                 "host_key_sha256": "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20",
                 "prefix_sha256": "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20"}
    })";
    qp::request req;
    std::string error;
    assert(qp::parse_request(text, req, error));
    assert(req.probe_id == kProbeId);
    assert(req.case_name == "positive");
    assert(!req.negative);
    assert(req.has_phase_flags && req.phase3 && req.phase4);
    assert(req.caps.size() == 3);
    assert(req.has_expect && req.expect.has_session_id && req.expect.session_id == 7);
    assert(req.expect.has_rows && req.expect.rows == 513);
    assert(!req.has_fault);
}

static void test_parse_fault() {
    const std::string text = R"({
      "schema": "rkva-console-durability-v1",
      "probe_id": "0123456789abcdef0123456789abcdef",
      "case": "chunk_sync",
      "test_only": true,
      "store": {"root": "rkva-qual\\durable\\rkva-snapshots-v3",
                "namespace": "rkva-snapshots-v3",
                "production_namespace": false},
      "fault": {"kind": "chunk_sync", "target": "staging", "at": 2,
                "injected": true}
    })";
    qp::request req;
    std::string error;
    assert(qp::parse_request(text, req, error));
    assert(req.has_fault);
    assert(req.fault.kind == "chunk_sync");
    assert(req.fault.has_store);
    assert(req.fault.store_root.find("durable") != std::string::npos);
    assert(req.fault.has_at && req.fault.at == 2);
    assert(req.fault.injected);

    qp::request hold;
    const std::string hold_text = R"({
      "schema": "rkva-console-durability-v1",
      "probe_id": "0123456789abcdef0123456789abcdef",
      "case": "stale_lock",
      "phase": "hold",
      "test_only": true,
      "fault": {"kind": "stale_lock", "injected": true, "at": 0}
    })";
    assert(qp::parse_request(hold_text, hold, error));
    assert(hold.fault.phase == "hold");
}

static void test_parse_rejects() {
    struct bad_case {
        const char* text;
        const char* needle;
    };
    const bad_case cases[] = {
        {R"({"schema":"rkva-console-attest-v1","probe_id":"0123456789ABCDEF0123456789abcdef","case":"positive"})",
         "probe_id"},
        {R"({"schema":"rkva-console-attest-v1","probe_id":"0123456789abcdef0123456789abcdef"})",
         "case"},
        {R"({"schema":"other","probe_id":"0123456789abcdef0123456789abcdef","case":"positive"})",
         "schema"},
        {R"({"schema":"rkva-console-attest-v1","probe_id":"0123456789abcdef0123456789abcdef",
             "case":"positive","phase_flags":{"phase3":true,"phase4":true,"caps":["MANAGED_GC"]}})",
         "unknown capability"},
        {R"({"schema":"rkva-console-attest-v1","probe_id":"0123456789abcdef0123456789abcdef",
             "case":"positive","expect":{"session_id":0}})",
         "expect"},
        {R"({"schema":"rkva-console-durability-v1","probe_id":"0123456789abcdef0123456789abcdef",
             "case":"chunk_sync","store":{"root":"rkva-snapshots-v3"},
             "fault":{"kind":"chunk_sync","at":2}})",
         "store.root"},
        {R"({"schema":"rkva-console-durability-v1","probe_id":"0123456789abcdef0123456789abcdef",
             "case":"chunk_sync","store":{"root":"rkva-qual\\durable\\rkva-snapshots-v3",
             "namespace":"rkva-snapshots-v3","production_namespace":true},
             "fault":{"kind":"chunk_sync"}})",
         "production_namespace"},
        {R"({"schema":"rkva-console-durability-v1","probe_id":"0123456789abcdef0123456789abcdef",
             "case":"stale_lock","phase":"adopt","fault":{"kind":"stale_lock"}})",
         "phase"},
        {R"({"schema":"rkva-console-durability-v1","probe_id":"0123456789abcdef0123456789abcdef",
             "case":"chunk_sync","test_only":false,"fault":{"kind":"chunk_sync"}})",
         "test_only"},
        {R"({"schema":"rkva-console-durability-v1","probe_id":"0123456789abcdef0123456789abcdef",
             "case":"durability-fault","fault":{"kind":"format_c","chunk":1}})",
         "fault"},
    };
    for (const auto& item : cases) {
        qp::request req;
        std::string error;
        assert(!qp::parse_request(item.text, req, error));
        assert(error.find(item.needle) != std::string::npos);
    }
    qp::request req;
    std::string error;
    assert(!qp::parse_request(std::string(qp::kMaxRequestBytes + 1, 'x'), req, error));
    assert(error.find("64 KiB") != std::string::npos);
}

static void test_complete_response() {
    const auto state = complete_state();
    const auto req = attest_request(state);
    const auto document = json::parse(qp::make_response(req, complete_build(), state));
    assert(document["schema"] == "rkva-console-attest-v1");
    assert(document["probe_id"] == kProbeId);
    assert(document["status"] == "available");
    assert(document["reason"] == "");
    assert(document["build"]["commit"] == "3a02e90a2");
    assert(document["build"]["phase3_compiled"] == true);
    assert(document["build"]["phase3_linked"] == false);
    assert(document["build"]["managed_caps_negotiated"] == false);
    const auto& att = document["attestation"];
    assert(att["session_id"] == 7);
    assert(att["live_rows"] == 513);
    assert(att["generation"] == 0x700000001ull);
    assert(att["submitted_fence"] == 41);
    assert(att["completed_fence"] == 41);
    assert(att["host_key_sha256"] == qp::hex(key_bytes()));
    assert(att["prefix_sha256"] == qp::hex(key_bytes()));
    assert(att["geometry_sha256"] == expected_geometry_digest(state.geometry));
    assert(att["descriptor_sha256"] == expected_descriptor_digest(state.descriptor));
    assert(att["coverage_sha256"] == expected_coverage_digest(state));
    assert(att["state_digest"] == expected_state_digest(state));
    // Cross-language vectors: the Python host producer asserts these exact
    // values for the same geometry/descriptor/coverage input.
    assert(att["geometry_sha256"] ==
           "f872fd04fe0c09da8ee163b191aa45164a0715f1c6d95338be4933f072ea8e0d");
    assert(att["descriptor_sha256"] ==
           "403df8d102e91bc73a2abe2fd3bc24351e24b5da20a2444b307b20a48a0c98c6");
    assert(att["coverage_sha256"] ==
           "e51bfd43789fdbf1e34129895feda0b69077ebb2f2304fd9b9365ab2b3704a1e");
    assert(att["state_digest"] ==
           "2fbd10c8fd7b21728b3935ca4abf73cc74003c09da3660f549fe213b5dad0af8");
    assert(att["coverage"].size() == 2);
    assert(att["coverage"][1]["tail_begin"] == 385);
    assert(att["coverage"][1]["stage_begin"] == 384);
    assert(att["coverage_mac"].is_null());
    assert(att["unavailable_fields"].contains("coverage_mac"));
    // The canonical metadata digest must be the real native manifest digest.
    std::vector<snap::LayerCoverage> layers = {{385, 384, 4, 1}, {385, 384, 4, 1}};
    snap::Manifest manifest;
    std::vector<uint8_t> encoded;
    std::string error;
    assert(snap::build(state.geometry, state.descriptor, state.live_rows, state.generation,
                       state.prefix_key.data(), layers, manifest, &error));
    assert(snap::encode(manifest, encoded, &error));
    const auto metadata = rkva_snapshot::digest(encoded.data(), encoded.size());
    assert(att["metadata_sha256"] == rkva_snapshot::hex(metadata.data()));
    // metadata_sha256 must not be the same value as the coverage identity.
    assert(att["metadata_sha256"] != att["state_digest"]);
}

static void test_unavailable_is_explicit() {
    qp::resident_state state;
    state.available = false;
    state.reason = "no live RKVA engine: no accepted HELLO in this app process";
    const auto req = attest_request(complete_state());
    const auto document = json::parse(qp::make_response(req, complete_build(), state));
    assert(document["status"] == "unavailable");
    assert(document["reason"].get<std::string>().find("no live RKVA engine") !=
           std::string::npos);
    const auto& att = document["attestation"];
    for (const char* field : {"session_id", "live_rows", "generation", "submitted_fence",
                              "completed_fence", "host_key_sha256", "prefix_sha256",
                              "metadata_sha256", "coverage", "geometry", "descriptor"})
        assert(att[field].is_null());
    assert(att["unavailable_fields"].contains("session_id"));
    assert(att["unavailable_fields"].contains("coverage_mac"));
    // No zero placeholder may leak into a missing field.
    assert(!att.contains("session_id") || att["session_id"].is_null());
}

static void test_field_unavailability() {
    auto state = complete_state();
    state.has_prefix_key = false;
    state.prefix_key = {};
    const auto req = attest_request(state);
    const auto document = json::parse(qp::make_response(req, complete_build(), state));
    assert(document["status"] == "unavailable");
    const auto& att = document["attestation"];
    assert(att["host_key_sha256"].is_null());
    assert(att["prefix_sha256"].is_null());
    assert(att["session_id"] == 7);
    assert(att["coverage"].size() == 2);
    assert(att["unavailable_fields"].contains("host_key_sha256"));
    assert(att["unavailable_fields"].contains("metadata_sha256"));
    assert(att["state_digest"].is_string()); // geometry+descriptor+coverage still real
}

static void test_negative_policy() {
    const auto state = complete_state();
    const auto document = json::parse(qp::make_response(attest_request(state, true),
                                                        complete_build(), state));
    assert(document["status"] == "available");
    const auto& negative = document["negative"];
    assert(negative["baseline"] == "accepted");
    assert(negative["all_refused"] == true);
    assert(negative["accepted_any"] == false);
    assert(negative["coverage_mac_emitted"] == false);
    for (const char* kind : {"generation_mismatch", "key_mismatch", "session_mismatch",
                             "prefix_mismatch", "fence_regression"})
        assert(negative["results"][kind] == "refused");
    // Without a real prefix key the negative claim cannot be completed.
    auto missing = complete_state();
    missing.has_prefix_key = false;
    const auto partial = json::parse(qp::make_response(attest_request(missing, true),
                                                       complete_build(), missing));
    assert(partial["negative"]["all_refused"] == false);
    assert(partial["negative"]["results"]["key_mismatch"] == "unavailable");
}

static void test_main_controller_fault_cases_parse() {
    // The console controller requests these exact selectors (11f37de73).
    const char* const cases[] = {"chunk_sync",       "manifest_sync", "dir_sync",
                                 "rename_existing",  "rename_eio",    "ack_lost",
                                 "crash_after_apply", "stale_lock"};
    for (const char* fault_case : cases) {
        const std::string kind(fault_case);
        const std::string text =
            "{\"schema\":\"rkva-console-durability-v1\","
            "\"probe_id\":\"0123456789abcdef0123456789abcdef\","
            "\"case\":\"" + kind + "\","
            "\"test_only\":true,"
            "\"store\":{\"root\":\"rkva-qual\\\\durable\\\\rkva-snapshots-v3\","
            "\"namespace\":\"rkva-snapshots-v3\",\"production_namespace\":false},"
            "\"fault\":{\"kind\":\"" + kind + "\",\"injected\":true,\"at\":2}}";
        qp::request req;
        std::string error;
        assert(qp::parse_request(text, req, error));
        assert(req.has_fault);
        assert(req.fault.kind == fault_case);
        assert(req.fault.has_store);
        assert(req.fault.injected);
        const auto document = json::parse(qp::make_fault_response(req, complete_build()));
        assert(document["status"] == "refused");
        assert(document["writer_reachable"] == false);
        assert(document["wrote_anything"] == false);
        assert(document["build"]["durable_fault_seam"] == false);
    }
}

static void test_fault_refusal_is_precise() {
    qp::request req;
    req.probe_id = kProbeId;
    req.case_name = "chunk_sync";
    req.has_fault = true;
    req.fault.kind = "chunk_sync";
    req.fault.target = "staging";
    req.fault.has_at = true;
    req.fault.at = 2;
    req.fault.has_store = true;
    req.fault.store_root = "rkva-qual\\durable\\rkva-snapshots-v3";
    const auto document = json::parse(qp::make_fault_response(req, complete_build()));
    assert(document["schema"] == "rkva-console-durability-v1");
    assert(document["status"] == "refused");
    assert(document["wrote_anything"] == false);
    assert(document["writer_reachable"] == false);
    const auto reason = document["reason"].get<std::string>();
    assert(reason.find("rkva-managed-object-v3.cpp:37") != std::string::npos);
    assert(reason.find("rkva-managed-object-v3-write.inc") != std::string::npos);
    assert(document["writer_gate"].size() >= 3);
    assert(document["store"]["root"] == "rkva-qual\\durable\\rkva-snapshots-v3");
    assert(document["build"]["durable_fault_seam"] == false);
    assert(document["fault"]["injected"] == false);
    assert(document["claims"]["published"] == false);
    // A different selector must name its own missing seam, not reuse a generic
    // "not implemented" string.
    req.fault.kind = "stale_lock";
    req.fault.phase = "hold";
    const auto lock = json::parse(qp::make_fault_response(req, complete_build()));
    assert(lock["reason"].get<std::string>().find("cross-process") != std::string::npos);
    assert(lock["lock"]["held"] == false);
}

static void test_build_never_invents_commit() {
    auto build = complete_build();
    build.commit.clear();
    const auto state = complete_state();
    const auto document = json::parse(qp::make_response(attest_request(state), build, state));
    assert(document["build"]["commit"].is_null());
    assert(document["build"]["commit_status"].get<std::string>().find("XLLAMA_QUAL_COMMIT") !=
           std::string::npos);
}

int main() {
    test_parse_valid();
    test_parse_fault();
    test_parse_rejects();
    test_complete_response();
    test_unavailable_is_explicit();
    test_field_unavailability();
    test_negative_policy();
    test_main_controller_fault_cases_parse();
    test_fault_refusal_is_precise();
    test_build_never_invents_commit();
    std::cout << "rkva-qual-probe-cpu: all checks passed\n";
    return 0;
}
