#include "rkva-peer-identity.h"

#include "../../ggml/src/ggml-remote-attn/rkva-managed-auth.h"
#include "../../vendor/nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <string>
#include <vector>

namespace rkva_peer_identity {
namespace {

using json = nlohmann::json;

constexpr const char * mac_domain = "RKVA-PEER-IDENTITY-STORE-HMAC-SHA256-V1";
constexpr const char * tofu_boundary = "explicit-operator-first-use";
constexpr const char * provision_boundary = "authorized-provisioning";
constexpr uint64_t max_unix_time = 4102444800ULL;  // 2100-01-01T00:00:00Z
constexpr uint32_t max_rotation_generation = 1000;

struct record {
    peer observed;
    std::string publisher_id;
    uint64_t first_seen = 0;
    bool pinned = true;
    bool revoked = false;

    bool tofu_enrolled = false;
    std::string tofu_boundary_value;
    uint64_t tofu_enrolled_at = 0;
    std::string tofu_operator_label;

    uint32_t rotation_generation = 1;
    uint64_t rotated_at = 0;
    std::vector<sha256> previous_cert_sha256;
    std::string rotation_operator_label;

    std::string revocation_reason;
    uint64_t revoked_at = 0;
    std::string revocation_operator_label;

    std::string integrity_key_id;
    std::string integrity_key_protected;
    sha256 mac{};
    std::string integrity_mac_hex;
};

bool equal_bytes(const uint8_t * a, const uint8_t * b, size_t size) {
    volatile uint8_t difference = 0;
    for (size_t i = 0; i < size; ++i) difference = uint8_t(difference | (a[i] ^ b[i]));
    return difference == 0;
}

template <typename T> bool nonzero(const T & value) {
    return std::any_of(value.begin(), value.end(), [](uint8_t byte) { return byte != 0; });
}

bool valid_unix_time(uint64_t value) {
    return value > 0 && value <= max_unix_time;
}

bool valid_label(const std::string & value, size_t maximum = 128) {
    if (value.empty() || value.size() > maximum) return false;
    for (unsigned char c : value) {
        if (c < 32 || c > 126) return false;
    }
    return true;
}

bool valid_publisher(const std::string & value) {
    if (value.empty() || value.size() > 512) return false;
    if (value.front() == ' ' || value.back() == ' ') return false;
    for (unsigned char c : value) {
        if (c < 32 || c > 126) return false;
    }
    return true;
}

bool valid_host(const std::string & value) {
    if (value.empty() || value.size() > 255) return false;
    for (unsigned char c : value) {
        const bool alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        const bool digit = c >= '0' && c <= '9';
        if (!alpha && !digit && c != '.' && c != '-' && c != ':' && c != '[' && c != ']' &&
            c != '_') {
            return false;
        }
    }
    return true;
}

bool valid_package_name(const std::string & value) {
    if (value.empty() || value.size() > 128) return false;
    if (!std::isalnum(static_cast<unsigned char>(value.front()))) return false;
    for (unsigned char c : value) {
        if (!std::isalnum(c) && c != '.' && c != '-') return false;
    }
    return true;
}

bool valid_version_part(const std::string & value) {
    if (value.empty() || value.size() > 5) return false;
    uint32_t number = 0;
    for (unsigned char c : value) {
        if (c < '0' || c > '9') return false;
        number = number * 10 + uint32_t(c - '0');
    }
    return number <= 65535;
}

bool u64_value(const json & value, uint64_t & out) {
    if (value.is_number_unsigned()) {
        out = value.get<uint64_t>();
        return true;
    }
    if (value.is_number_integer()) {
        const int64_t number = value.get<int64_t>();
        if (number < 0) return false;
        out = uint64_t(number);
        return true;
    }
    return false;
}

bool u32_value(const json & value, uint32_t & out) {
    uint64_t number = 0;
    if (!u64_value(value, number) || number > std::numeric_limits<uint32_t>::max()) return false;
    out = uint32_t(number);
    return true;
}

bool exact_keys(const json & value, std::initializer_list<const char *> keys) {
    if (!value.is_object() || value.size() != keys.size()) return false;
    for (auto it = value.begin(); it != value.end(); ++it) {
        bool found = false;
        for (const char * key : keys) {
            if (it.key() == key) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

std::vector<std::string> split(const std::string & value, char separator) {
    std::vector<std::string> parts;
    size_t begin = 0;
    for (;;) {
        const size_t end = value.find(separator, begin);
        if (end == std::string::npos) {
            parts.push_back(value.substr(begin));
            return parts;
        }
        parts.push_back(value.substr(begin, end - begin));
        begin = end + 1;
    }
}

json tofu_json(const record & value) {
    json out = json::object();
    out["boundary"] = value.tofu_boundary_value;
    out["enrolled"] = value.tofu_enrolled;
    out["enrolled_at"] = value.tofu_enrolled_at;
    out["operator_label"] = value.tofu_operator_label;
    return out;
}

json rotation_json(const record & value) {
    json history = json::array();
    for (const auto & digest : value.previous_cert_sha256) {
        history.push_back(to_hex(digest.data(), digest.size()));
    }
    json out = json::object();
    out["generation"] = value.rotation_generation;
    out["rotated_at"] = value.rotated_at;
    out["previous_cert_sha256"] = std::move(history);
    out["operator_label"] = value.rotation_operator_label;
    return out;
}

json revocation_json(const record & value) {
    json out = json::object();
    out["revoked"] = value.revoked;
    out["reason"] = value.revocation_reason;
    out["revoked_at"] = value.revoked_at;
    out["operator_label"] = value.revocation_operator_label;
    return out;
}

json body_json(const record & value) {
    json out = json::object();
    out["schema"] = schema_name;
    out["version"] = schema_version;
    out["pfn"] = value.observed.pfn;
    out["package_family"] = value.observed.package_family;
    out["publisher"] = value.observed.publisher;
    out["publisher_id"] = value.publisher_id;
    out["cert_sha256"] = to_hex(value.observed.cert_sha256.data(),
                                value.observed.cert_sha256.size());
    out["cert_thumbprint"] = to_hex(value.observed.cert_thumbprint.data(),
                                    value.observed.cert_thumbprint.size());
    out["peer_host"] = value.observed.host;
    out["peer_port"] = value.observed.port;
    out["first_seen"] = value.first_seen;
    out["pin_state"] = value.revoked ? "revoked" : "pinned";
    out["tofu"] = tofu_json(value);
    out["rotation"] = rotation_json(value);
    out["revocation"] = revocation_json(value);
    return out;
}

std::string serialize_record(const record & value) {
    json out = body_json(value);
    json integrity = json::object();
    integrity["algorithm"] = "hmac-sha256";
    integrity["key_id"] = value.integrity_key_id;
    integrity["key_protected"] = value.integrity_key_protected;
    integrity["mac"] = value.integrity_mac_hex;
    out["integrity"] = std::move(integrity);
    return out.dump() + "\n";
}

bool compute_mac(const record & value, const key & integrity_key, sha256 & out) {
    const std::string body = body_json(value).dump();
    const std::string input = std::string(mac_domain) + "\n" + body;
    out = sha256{};
    return rkva_managed_auth::hmac_sha256(
        integrity_key.data(), integrity_key.size(),
        reinterpret_cast<const uint8_t *>(input.data()), input.size(), out);
}

bool validate_record(record & value) {
    std::string family;
    std::string publisher_id;
    if (!parse_pfn(value.observed.pfn, family, publisher_id)) return false;
    if (value.observed.package_family != family) return false;
    if (value.publisher_id != publisher_id) return false;
    if (!valid_publisher(value.observed.publisher)) return false;
    if (!valid_host(value.observed.host)) return false;
    if (canonical_host(value.observed.host) != value.observed.host) return false;
    if (value.observed.port == 0) return false;
    if (!nonzero(value.observed.cert_sha256) || !nonzero(value.observed.cert_thumbprint)) return false;
    if (!valid_unix_time(value.first_seen)) return false;

    if (value.revoked) {
        if (value.pinned) return false;
        if (!valid_label(value.revocation_reason)) return false;
        if (!valid_label(value.revocation_operator_label)) return false;
        if (!valid_unix_time(value.revoked_at) || value.revoked_at < value.first_seen) return false;
    } else {
        if (!value.pinned) return false;
        if (!value.revocation_reason.empty() || !value.revocation_operator_label.empty()) return false;
        if (value.revoked_at != 0) return false;
    }

    if (value.tofu_enrolled) {
        if (value.tofu_boundary_value != tofu_boundary) return false;
        if (!valid_unix_time(value.tofu_enrolled_at)) return false;
        if (!valid_label(value.tofu_operator_label)) return false;
    } else {
        if (value.tofu_boundary_value != provision_boundary) return false;
        if (value.tofu_enrolled_at != 0 || !value.tofu_operator_label.empty()) return false;
    }

    if (value.rotation_generation == 0 || value.rotation_generation > max_rotation_generation) {
        return false;
    }
    if (value.rotation_generation == 1) {
        if (value.rotated_at != 0 || !value.previous_cert_sha256.empty() ||
            !value.rotation_operator_label.empty()) {
            return false;
        }
    } else {
        if (!valid_unix_time(value.rotated_at) || value.rotated_at < value.first_seen) return false;
        if (value.previous_cert_sha256.empty()) return false;
        if (!valid_label(value.rotation_operator_label)) return false;
    }
    if (value.previous_cert_sha256.size() > max_rotation_history) return false;
    for (size_t i = 0; i < value.previous_cert_sha256.size(); ++i) {
        const auto & digest = value.previous_cert_sha256[i];
        if (!nonzero(digest)) return false;
        if (equal_bytes(digest.data(), value.observed.cert_sha256.data(), digest.size())) return false;
        for (size_t j = 0; j < i; ++j) {
            if (equal_bytes(digest.data(), value.previous_cert_sha256[j].data(), digest.size())) {
                return false;
            }
        }
    }

    if (value.integrity_key_id.size() != 16) return false;
    uint8_t key_id[8]{};
    if (!from_hex(value.integrity_key_id, key_id, sizeof(key_id))) return false;
    if (value.integrity_key_protected.size() < 2 ||
        value.integrity_key_protected.size() > 8192 ||
        value.integrity_key_protected.size() % 2 != 0) {
        return false;
    }
    {
        std::vector<uint8_t> scratch(value.integrity_key_protected.size() / 2);
        if (!from_hex(value.integrity_key_protected, scratch.data(), scratch.size())) return false;
    }
    if (value.integrity_mac_hex.size() != 64) return false;
    if (!from_hex(value.integrity_mac_hex, value.mac.data(), value.mac.size())) return false;
    return true;
}

status parse_record(const std::string & text, record & out, std::string & error) {
    error = "peer identity record is corrupt or ambiguous";
    if (text.empty() || text.size() > max_record_bytes) return status::corrupt;
    try {
        const json root = json::parse(text);
        if (!exact_keys(root, {"schema", "version", "pfn", "package_family", "publisher",
                               "publisher_id", "cert_sha256", "cert_thumbprint", "peer_host",
                               "peer_port", "first_seen", "pin_state", "tofu", "rotation",
                               "revocation", "integrity"})) {
            return status::corrupt;
        }
        if (!root["schema"].is_string() || root["schema"].get<std::string>() != schema_name) {
            return status::corrupt;
        }
        uint32_t version = 0;
        if (!u32_value(root["version"], version) || version != schema_version) return status::corrupt;

        if (!root["pfn"].is_string() || !root["package_family"].is_string() ||
            !root["publisher"].is_string() || !root["publisher_id"].is_string() ||
            !root["cert_sha256"].is_string() || !root["cert_thumbprint"].is_string() ||
            !root["peer_host"].is_string() || !root["pin_state"].is_string()) {
            return status::corrupt;
        }
        record parsed;
        parsed.observed.pfn = root["pfn"].get<std::string>();
        parsed.observed.package_family = root["package_family"].get<std::string>();
        parsed.observed.publisher = root["publisher"].get<std::string>();
        parsed.publisher_id = root["publisher_id"].get<std::string>();
        parsed.observed.host = root["peer_host"].get<std::string>();
        uint32_t port = 0;
        if (!u32_value(root["peer_port"], port) || port == 0 || port > 65535) return status::corrupt;
        parsed.observed.port = uint16_t(port);
        if (!from_hex(root["cert_sha256"].get<std::string>(),
                      parsed.observed.cert_sha256.data(),
                      parsed.observed.cert_sha256.size()) ||
            !from_hex(root["cert_thumbprint"].get<std::string>(),
                      parsed.observed.cert_thumbprint.data(),
                      parsed.observed.cert_thumbprint.size())) {
            return status::corrupt;
        }
        if (!u64_value(root["first_seen"], parsed.first_seen)) return status::corrupt;
        const std::string pin_state = root["pin_state"].get<std::string>();
        if (pin_state == "pinned") {
            parsed.pinned = true;
            parsed.revoked = false;
        } else if (pin_state == "revoked") {
            parsed.pinned = false;
            parsed.revoked = true;
        } else {
            return status::corrupt;
        }

        const json & tofu = root["tofu"];
        if (!exact_keys(tofu, {"boundary", "enrolled", "enrolled_at", "operator_label"}) ||
            !tofu["boundary"].is_string() || !tofu["enrolled"].is_boolean() ||
            !tofu["operator_label"].is_string()) {
            return status::corrupt;
        }
        parsed.tofu_boundary_value = tofu["boundary"].get<std::string>();
        parsed.tofu_enrolled = tofu["enrolled"].get<bool>();
        parsed.tofu_operator_label = tofu["operator_label"].get<std::string>();
        if (!u64_value(tofu["enrolled_at"], parsed.tofu_enrolled_at)) return status::corrupt;

        const json & rotation = root["rotation"];
        if (!exact_keys(rotation, {"generation", "rotated_at", "previous_cert_sha256",
                                   "operator_label"}) ||
            !rotation["previous_cert_sha256"].is_array() ||
            !rotation["operator_label"].is_string()) {
            return status::corrupt;
        }
        if (!u32_value(rotation["generation"], parsed.rotation_generation) ||
            !u64_value(rotation["rotated_at"], parsed.rotated_at)) {
            return status::corrupt;
        }
        parsed.rotation_operator_label = rotation["operator_label"].get<std::string>();
        if (rotation["previous_cert_sha256"].size() > max_rotation_history) return status::corrupt;
        for (const auto & item : rotation["previous_cert_sha256"]) {
            if (!item.is_string()) return status::corrupt;
            sha256 digest{};
            if (!from_hex(item.get<std::string>(), digest.data(), digest.size())) {
                return status::corrupt;
            }
            parsed.previous_cert_sha256.push_back(digest);
        }

        const json & revocation = root["revocation"];
        if (!exact_keys(revocation, {"revoked", "reason", "revoked_at", "operator_label"}) ||
            !revocation["revoked"].is_boolean() || !revocation["reason"].is_string() ||
            !revocation["operator_label"].is_string()) {
            return status::corrupt;
        }
        const bool revoked = revocation["revoked"].get<bool>();
        if (revoked != parsed.revoked) return status::corrupt;
        parsed.revocation_reason = revocation["reason"].get<std::string>();
        parsed.revocation_operator_label = revocation["operator_label"].get<std::string>();
        if (!u64_value(revocation["revoked_at"], parsed.revoked_at)) return status::corrupt;

        const json & integrity = root["integrity"];
        if (!exact_keys(integrity, {"algorithm", "key_id", "key_protected", "mac"}) ||
            !integrity["algorithm"].is_string() || !integrity["key_id"].is_string() ||
            !integrity["key_protected"].is_string() || !integrity["mac"].is_string()) {
            return status::corrupt;
        }
        if (integrity["algorithm"].get<std::string>() != "hmac-sha256") return status::corrupt;
        parsed.integrity_key_id = integrity["key_id"].get<std::string>();
        parsed.integrity_key_protected = integrity["key_protected"].get<std::string>();
        parsed.integrity_mac_hex = integrity["mac"].get<std::string>();

        if (!validate_record(parsed)) return status::corrupt;
        if (serialize_record(parsed) != text) return status::corrupt;
        out = std::move(parsed);
        error.clear();
        return status::ok;
    } catch (...) {
        return status::corrupt;
    }
}

struct key_guard {
    key & value;
    ~key_guard() { rkva_managed_auth::wipe(value.data(), value.size()); }
};

status load_record(store & backend, record & out, std::string & error) {
    std::string bytes;
    const status read_status = backend.read(bytes, error);
    if (read_status != status::ok) {
        rkva_managed_auth::wipe(bytes.data(), bytes.size());
        return read_status;
    }
    const status parsed_status = parse_record(bytes, out, error);
    rkva_managed_auth::wipe(bytes.data(), bytes.size());
    if (parsed_status != status::ok) return parsed_status;

    key integrity_key{};
    key_guard guard{integrity_key};
    if (!backend.unseal_key(out.integrity_key_protected, integrity_key, error)) {
        error = "peer identity protected integrity key is unavailable";
        return status::corrupt;
    }

    const auto key_digest = rkva_snapshot::digest(integrity_key.data(), integrity_key.size());
    uint8_t expected_key_id[8]{};
    if (!from_hex(out.integrity_key_id, expected_key_id, sizeof(expected_key_id)) ||
        !equal_bytes(expected_key_id, key_digest.data(), sizeof(expected_key_id))) {
        error = "peer identity integrity key identifier does not match";
        return status::corrupt;
    }

    sha256 expected{};
    if (!compute_mac(out, integrity_key, expected) ||
        !equal_bytes(expected.data(), out.mac.data(), expected.size())) {
        error = "peer identity integrity check failed";
        return status::corrupt;
    }
    error.clear();
    return status::ok;
}

status encode_record(record & value, const key & integrity_key, std::string & bytes,
                     std::string & error) {
    const auto key_digest = rkva_snapshot::digest(integrity_key.data(), integrity_key.size());
    value.integrity_key_id = to_hex(key_digest.data(), 8);
    if (!compute_mac(value, integrity_key, value.mac)) {
        error = "peer identity integrity computation failed";
        return status::corrupt;
    }
    value.integrity_mac_hex = to_hex(value.mac.data(), value.mac.size());
    bytes = serialize_record(value);
    if (bytes.size() > max_record_bytes) {
        error = "peer identity record exceeds the size limit";
        return status::corrupt;
    }
    error.clear();
    return status::ok;
}

status save_record(store & backend, record & value, const key & integrity_key,
                   std::string & error) {
    std::string bytes;
    const status encoded = encode_record(value, integrity_key, bytes, error);
    if (encoded != status::ok) return encoded;
    const status written = backend.write(bytes, error);
    rkva_managed_auth::wipe(bytes.data(), bytes.size());
    return written;
}

status validate_peer(const peer & value, std::string & error) {
    std::string family;
    std::string publisher_id;
    if (!parse_pfn(value.pfn, family, publisher_id)) {
        error = "peer package full name is invalid";
        return status::invalid_argument;
    }
    if (!value.package_family.empty() && value.package_family != family) {
        error = "peer package family is inconsistent with the package full name";
        return status::invalid_argument;
    }
    if (!valid_publisher(value.publisher)) {
        error = "peer publisher is invalid";
        return status::invalid_argument;
    }
    if (!valid_host(value.host)) {
        error = "peer host is invalid";
        return status::invalid_argument;
    }
    if (value.port == 0) {
        error = "peer port is invalid";
        return status::invalid_argument;
    }
    if (!nonzero(value.cert_sha256)) {
        error = "peer certificate sha256 is missing";
        return status::invalid_argument;
    }
    if (!nonzero(value.cert_thumbprint)) {
        error = "peer certificate thumbprint is missing";
        return status::invalid_argument;
    }
    error.clear();
    return status::ok;
}

peer normalized_peer(const peer & value) {
    peer out = value;
    out.host = canonical_host(out.host);
    std::string family;
    std::string publisher_id;
    if (parse_pfn(out.pfn, family, publisher_id)) {
        out.package_family = family;
    }
    return out;
}

record initial_record(const peer & value) {
    record out;
    out.observed = normalized_peer(value);
    std::string family;
    std::string publisher_id;
    if (parse_pfn(out.observed.pfn, family, publisher_id)) out.publisher_id = publisher_id;
    out.first_seen = 0;
    return out;
}

bool same_digest(const sha256 & a, const sha256 & b) {
    return equal_bytes(a.data(), b.data(), a.size());
}

bool same_thumbprint(const thumbprint & a, const thumbprint & b) {
    return equal_bytes(a.data(), b.data(), a.size());
}

uint64_t current_time() {
#ifdef RKVA_PEER_IDENTITY_TESTING
    if (test_clock) return test_clock();
#endif
    using namespace std::chrono;
    return uint64_t(duration_cast<seconds>(system_clock::now().time_since_epoch()).count());
}

bool present_key(const sha256 & digest) {
    return nonzero(digest);
}

}  // namespace

const char * status_name(status value) {
    switch (value) {
        case status::ok: return "ok";
        case status::disabled: return "disabled";
        case status::unknown: return "unknown";
        case status::corrupt: return "corrupt";
        case status::unavailable: return "unavailable";
        case status::mismatch: return "mismatch";
        case status::revoked: return "revoked";
        case status::already_pinned: return "already_pinned";
        case status::unauthorized: return "unauthorized";
        case status::invalid_argument: return "invalid_argument";
    }
    return "unavailable";
}

std::string to_hex(const uint8_t * data, size_t size) {
    static const char digits[] = "0123456789abcdef";
    std::string out;
    out.resize(size * 2);
    for (size_t i = 0; i < size; ++i) {
        out[i * 2] = digits[data[i] >> 4];
        out[i * 2 + 1] = digits[data[i] & 0x0f];
    }
    return out;
}

bool from_hex(const std::string & text, uint8_t * out, size_t size) {
    if (text.size() != size * 2) return false;
    auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < size; ++i) {
        const int high = nibble(text[i * 2]);
        const int low = nibble(text[i * 2 + 1]);
        if (high < 0 || low < 0) return false;
        if (out) out[i] = uint8_t(high * 16 + low);
    }
    return true;
}

bool parse_pfn(const std::string & pfn, std::string & package_family,
               std::string & publisher_id) {
    const std::vector<std::string> parts = split(pfn, '_');
    if (parts.size() != 5) return false;
    if (!valid_package_name(parts[0])) return false;
    const std::vector<std::string> version = split(parts[1], '.');
    if (version.empty() || version.size() > 4) return false;
    for (const auto & item : version) {
        if (!valid_version_part(item)) return false;
    }
    if (parts[2].empty() || parts[2].size() > 16) return false;
    for (unsigned char c : parts[2]) {
        if (!std::isalnum(c)) return false;
    }
    if (!parts[3].empty()) return false;
    if (parts[4].size() != 13) return false;
    for (unsigned char c : parts[4]) {
        const bool lower = c >= 'a' && c <= 'z';
        const bool digit = c >= '0' && c <= '9';
        if (!lower && !digit) return false;
    }
    package_family = parts[0] + "_" + parts[4];
    publisher_id = parts[4];
    return true;
}

std::string canonical_host(const std::string & host) {
    std::string out = host;
    for (char & c : out) {
        if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    }
    return out;
}

bool enabled() {
#ifdef RKVA_PEER_IDENTITY_STORE
    const char * value = std::getenv("RKVA_PEER_IDENTITY_STORE");
    return value && std::strcmp(value, "1") == 0;
#else
    return false;
#endif
}

status verify(store & backend, const peer & observed, std::string & error) {
    if (!enabled()) {
        error = "peer identity store is disabled";
        return status::disabled;
    }
    const status peer_status = validate_peer(observed, error);
    if (peer_status != status::ok) return peer_status;

    record stored;
    const status loaded = load_record(backend, stored, error);
    if (loaded != status::ok) return loaded;
    if (stored.revoked || !stored.pinned) {
        error = "peer identity is revoked";
        return status::revoked;
    }

    const peer presented = normalized_peer(observed);
    if (stored.observed.pfn != presented.pfn) {
        error = "peer identity mismatch: package full name";
        return status::mismatch;
    }
    if (stored.observed.package_family != presented.package_family) {
        error = "peer identity mismatch: package family";
        return status::mismatch;
    }
    if (stored.observed.publisher != presented.publisher) {
        error = "peer identity mismatch: publisher";
        return status::mismatch;
    }
    if (stored.observed.host != presented.host) {
        error = "peer identity mismatch: peer host";
        return status::mismatch;
    }
    if (stored.observed.port != presented.port) {
        error = "peer identity mismatch: peer port";
        return status::mismatch;
    }
    if (!same_digest(stored.observed.cert_sha256, presented.cert_sha256)) {
        error = "peer identity mismatch: certificate sha256";
        return status::mismatch;
    }
    if (!same_thumbprint(stored.observed.cert_thumbprint, presented.cert_thumbprint)) {
        error = "peer identity mismatch: certificate thumbprint";
        return status::mismatch;
    }
    error.clear();
    return status::ok;
}

status enroll_tofu(store & backend, const peer & observed,
                   const enrollment & request, std::string & error) {
    if (!enabled()) {
        error = "peer identity store is disabled";
        return status::disabled;
    }
    const status peer_status = validate_peer(observed, error);
    if (peer_status != status::ok) return peer_status;
    if (!request.explicit_operator_confirmation) {
        error = "explicit operator confirmation is required for first-use enrollment";
        return status::unauthorized;
    }
    if (!valid_label(request.operator_label)) {
        error = "operator label is invalid";
        return status::invalid_argument;
    }
    if (!present_key(request.out_of_band_cert_sha256) ||
        !same_digest(request.out_of_band_cert_sha256, observed.cert_sha256)) {
        error = "out-of-band certificate digest does not match the presented certificate";
        return status::unauthorized;
    }

    std::string bytes;
    const status existing = backend.read(bytes, error);
    rkva_managed_auth::wipe(bytes.data(), bytes.size());
    if (existing == status::ok) {
        error = "peer identity is already enrolled; use rotate, revoke or wipe";
        return status::already_pinned;
    }
    if (existing != status::unknown) return existing;

    key integrity_key{};
    key_guard guard{integrity_key};
    if (!backend.new_key(integrity_key, error)) return status::unavailable;
    record value = initial_record(observed);
    value.first_seen = current_time();
    value.tofu_enrolled = true;
    value.tofu_boundary_value = tofu_boundary;
    value.tofu_enrolled_at = value.first_seen;
    value.tofu_operator_label = request.operator_label;
    value.rotation_generation = 1;

    std::string sealed;
    if (!backend.seal_key(integrity_key, sealed, error)) return status::unavailable;
    value.integrity_key_protected = std::move(sealed);
    return save_record(backend, value, integrity_key, error);
}

status provision(store & backend, const provisioning & request, std::string & error) {
    if (!enabled()) {
        error = "peer identity store is disabled";
        return status::disabled;
    }
    const status peer_status = validate_peer(request.observed, error);
    if (peer_status != status::ok) return peer_status;
    if (!request.authorized_wdp_session) {
        error = "an authorized provisioning session is required";
        return status::unauthorized;
    }
    if (!valid_label(request.operator_label)) {
        error = "operator label is invalid";
        return status::invalid_argument;
    }
    if (!present_key(request.out_of_band_cert_sha256) ||
        !same_digest(request.out_of_band_cert_sha256, request.observed.cert_sha256)) {
        error = "out-of-band certificate digest does not match the presented certificate";
        return status::unauthorized;
    }
    if (request.first_seen != 0 && !valid_unix_time(request.first_seen)) {
        error = "first-seen timestamp is invalid";
        return status::invalid_argument;
    }

    std::string bytes;
    const status existing = backend.read(bytes, error);
    rkva_managed_auth::wipe(bytes.data(), bytes.size());
    if (existing == status::ok) {
        error = "peer identity is already enrolled; use rotate, revoke or wipe";
        return status::already_pinned;
    }
    if (existing != status::unknown) return existing;

    key integrity_key{};
    key_guard guard{integrity_key};
    if (!backend.new_key(integrity_key, error)) return status::unavailable;
    record value = initial_record(request.observed);
    value.first_seen = request.first_seen ? request.first_seen : current_time();
    value.tofu_enrolled = false;
    value.tofu_boundary_value = provision_boundary;
    value.rotation_generation = 1;

    std::string sealed;
    if (!backend.seal_key(integrity_key, sealed, error)) return status::unavailable;
    value.integrity_key_protected = std::move(sealed);
    return save_record(backend, value, integrity_key, error);
}

status rotate(store & backend, const peer & observed, const rotation & request,
              std::string & error) {
    if (!enabled()) {
        error = "peer identity store is disabled";
        return status::disabled;
    }
    const status peer_status = validate_peer(observed, error);
    if (peer_status != status::ok) return peer_status;
    if (!request.explicit_operator_confirmation) {
        error = "explicit operator confirmation is required for rotation";
        return status::unauthorized;
    }
    if (!valid_label(request.operator_label)) {
        error = "operator label is invalid";
        return status::invalid_argument;
    }
    if (!present_key(request.out_of_band_cert_sha256) ||
        !same_digest(request.out_of_band_cert_sha256, observed.cert_sha256)) {
        error = "out-of-band certificate digest does not match the presented certificate";
        return status::unauthorized;
    }

    record stored;
    const status loaded = load_record(backend, stored, error);
    if (loaded != status::ok) return loaded;
    if (stored.revoked || !stored.pinned) {
        error = "peer identity is revoked and cannot rotate";
        return status::revoked;
    }

    const peer presented = normalized_peer(observed);
    if (same_digest(stored.observed.cert_sha256, presented.cert_sha256)) {
        error = "rotation requires a different certificate";
        return status::invalid_argument;
    }
    for (const auto & retired : stored.previous_cert_sha256) {
        if (same_digest(retired, presented.cert_sha256)) {
            error = "rotation cannot re-pin a previously retired certificate";
            return status::mismatch;
        }
    }
    if (stored.rotation_generation >= max_rotation_generation) {
        error = "rotation generation is exhausted";
        return status::mismatch;
    }

    stored.previous_cert_sha256.push_back(stored.observed.cert_sha256);
    if (stored.previous_cert_sha256.size() > max_rotation_history) {
        stored.previous_cert_sha256.erase(stored.previous_cert_sha256.begin());
    }
    stored.observed = presented;
    std::string family;
    std::string publisher_id;
    if (!parse_pfn(stored.observed.pfn, family, publisher_id)) {
        error = "peer package full name is invalid";
        return status::invalid_argument;
    }
    stored.publisher_id = publisher_id;
    stored.rotation_generation += 1;
    stored.rotated_at = current_time();
    stored.rotation_operator_label = request.operator_label;

    key integrity_key{};
    key_guard guard{integrity_key};
    if (!backend.unseal_key(stored.integrity_key_protected, integrity_key, error)) {
        error = "peer identity protected integrity key is unavailable";
        return status::corrupt;
    }
    return save_record(backend, stored, integrity_key, error);
}

status revoke(store & backend, const revocation & request, std::string & error) {
    if (!enabled()) {
        error = "peer identity store is disabled";
        return status::disabled;
    }
    if (!request.explicit_operator_confirmation) {
        error = "explicit operator confirmation is required for revocation";
        return status::unauthorized;
    }
    if (!valid_label(request.operator_label)) {
        error = "operator label is invalid";
        return status::invalid_argument;
    }
    if (!valid_label(request.reason)) {
        error = "revocation reason is invalid";
        return status::invalid_argument;
    }

    record stored;
    const status loaded = load_record(backend, stored, error);
    if (loaded != status::ok) return loaded;
    if (stored.revoked) {
        error = "peer identity is already revoked";
        return status::revoked;
    }
    stored.revoked = true;
    stored.pinned = false;
    stored.revocation_reason = request.reason;
    stored.revocation_operator_label = request.operator_label;
    stored.revoked_at = current_time();
    if (stored.revoked_at < stored.first_seen) {
        error = "revocation timestamp precedes first-seen";
        return status::corrupt;
    }

    key integrity_key{};
    key_guard guard{integrity_key};
    if (!backend.unseal_key(stored.integrity_key_protected, integrity_key, error)) {
        error = "peer identity protected integrity key is unavailable";
        return status::corrupt;
    }
    return save_record(backend, stored, integrity_key, error);
}

status wipe(store & backend, const wipe_request & request, std::string & error) {
    if (!enabled()) {
        error = "peer identity store is disabled";
        return status::disabled;
    }
    if (!request.explicit_operator_confirmation) {
        error = "explicit operator confirmation is required for wipe";
        return status::unauthorized;
    }
    if (!valid_label(request.operator_label)) {
        error = "operator label is invalid";
        return status::invalid_argument;
    }
    const status erased = backend.erase(error);
    if (erased != status::ok) return erased;
    error.clear();
    return status::ok;
}

status inspect(store & backend, view & out, std::string & error) {
    if (!enabled()) {
        error = "peer identity store is disabled";
        return status::disabled;
    }
    record stored;
    const status loaded = load_record(backend, stored, error);
    if (loaded != status::ok) return loaded;
    out.pfn = stored.observed.pfn;
    out.package_family = stored.observed.package_family;
    out.publisher = stored.observed.publisher;
    out.publisher_id = stored.publisher_id;
    out.peer_host = stored.observed.host;
    out.peer_port = stored.observed.port;
    out.cert_sha256 = stored.observed.cert_sha256;
    out.cert_thumbprint = stored.observed.cert_thumbprint;
    out.first_seen = stored.first_seen;
    out.pinned = stored.pinned && !stored.revoked;
    out.revoked = stored.revoked;
    out.tofu_enrolled = stored.tofu_enrolled;
    out.tofu_boundary = stored.tofu_boundary_value;
    out.tofu_enrolled_at = stored.tofu_enrolled_at;
    out.rotation_generation = stored.rotation_generation;
    out.rotated_at = stored.rotated_at;
    out.rotation_history = stored.previous_cert_sha256.size();
    out.revoked_at = stored.revoked_at;
    out.revocation_reason = stored.revocation_reason;
    error.clear();
    return status::ok;
}

}  // namespace rkva_peer_identity
