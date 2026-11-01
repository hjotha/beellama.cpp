// CPU-only tests for the default-off peer identity store and certificate pin.
// This never touches Windows, a console, WDP, a GPU or a production service.
// The POSIX file backend mirrors the Windows write sequence for corruption and
// crash-window tests; it is not AppContainer/ACL qualification.
#undef NDEBUG
#include "../tools/rkva-peer-identity/rkva-peer-identity.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef _WIN32
#define setenv(name, value, overwrite) _putenv_s((name), (value))
#define unsetenv(name) _putenv_s((name), "")
#endif

namespace fs = std::filesystem;
using namespace rkva_peer_identity;

namespace {

int failures = 0;

void check(bool condition, const char * what, int line) {
    if (!condition) {
        std::cerr << "FAIL line " << line << ": " << what << "\n";
        ++failures;
    }
}

#define CHECK(condition) check((condition), #condition, __LINE__)

void expect_status(status wanted, status actual, const std::string & error, const char * what,
                   int line) {
    if (actual != wanted) {
        std::cerr << "FAIL line " << line << ": " << what << " returned "
                  << status_name(actual) << " (wanted " << status_name(wanted)
                  << ") error=" << error << "\n";
        ++failures;
    }
}

#define EXPECT(wanted, expression) expect_status((wanted), (expression), error, #expression, __LINE__)

peer make_peer(uint8_t cert_seed = 0x11, uint8_t thumb_seed = 0x22,
               const std::string & host = "192.168.1.200") {
    peer value;
    value.pfn = "GianlucaMazza.xllama_1.6.0.436_x64__pj67f1fcj4n14";
    value.package_family = "GianlucaMazza.xllama_pj67f1fcj4n14";
    value.publisher = "CN=Gianluca Mazza, O=Gianluca Mazza, C=IT";
    value.host = host;
    value.port = 11443;
    for (size_t i = 0; i < value.cert_sha256.size(); ++i) {
        value.cert_sha256[i] = uint8_t(cert_seed + i);
    }
    for (size_t i = 0; i < value.cert_thumbprint.size(); ++i) {
        value.cert_thumbprint[i] = uint8_t(thumb_seed + i);
    }
    return value;
}

void set_gate(const char * value) {
    if (value) {
        setenv("RKVA_PEER_IDENTITY_STORE", value, 1);
    } else {
        unsetenv("RKVA_PEER_IDENTITY_STORE");
    }
}

status enroll_ok(store & backend, const peer & value, std::string & error,
                 const char * label = "lab-console-a") {
    enrollment request;
    request.explicit_operator_confirmation = true;
    request.operator_label = label;
    request.out_of_band_cert_sha256 = value.cert_sha256;
    return enroll_tofu(backend, value, request, error);
}

status provision_ok(store & backend, const peer & value, std::string & error,
                    uint64_t first_seen = 0) {
    provisioning request;
    request.authorized_wdp_session = true;
    request.operator_label = "lab-provision";
    request.out_of_band_cert_sha256 = value.cert_sha256;
    request.first_seen = first_seen;
    request.observed = value;
    return provision(backend, request, error);
}

status rotate_ok(store & backend, const peer & value, std::string & error) {
    rotation request;
    request.explicit_operator_confirmation = true;
    request.operator_label = "lab-rotate";
    request.out_of_band_cert_sha256 = value.cert_sha256;
    return rotate(backend, value, request, error);
}

std::string flip_mac(std::string bytes) {
    const std::string marker = "\"mac\":\"";
    const size_t position = bytes.rfind(marker);
    if (position == std::string::npos || position + marker.size() >= bytes.size()) return bytes;
    const size_t index = position + marker.size();
    bytes[index] = bytes[index] == '0' ? '1' : '0';
    return bytes;
}

void test_gate() {
    std::string error;
    const peer value = make_peer();
    test_memory_store memory;

    set_gate(nullptr);
    CHECK(!enabled());
    EXPECT(status::disabled, verify(memory, value, error));
    EXPECT(status::disabled, enroll_ok(memory, value, error));

    for (const char * gate : {"0", "true", "01", " 1", "1 ", "", "yes"}) {
        set_gate(gate);
        CHECK(!enabled());
        EXPECT(status::disabled, verify(memory, value, error));
    }
    set_gate("1");
    CHECK(enabled());
    set_gate(nullptr);
}

void test_schema_and_mac() {
    std::string error;
    const peer value = make_peer();
    set_gate("1");
    test_clock = [] { return uint64_t(1700000000); };

    test_memory_store memory;
    EXPECT(status::unknown, verify(memory, value, error));
    EXPECT(status::ok, enroll_ok(memory, value, error));

    const std::string enrolled_bytes = memory.bytes;
    EXPECT(status::ok, verify(memory, value, error));
    CHECK(memory.bytes == enrolled_bytes);

    view inspected;
    EXPECT(status::ok, inspect(memory, inspected, error));
    CHECK(inspected.pfn == value.pfn);
    CHECK(inspected.package_family == value.package_family);
    CHECK(inspected.publisher == value.publisher);
    CHECK(inspected.peer_host == value.host);
    CHECK(inspected.peer_port == value.port);
    CHECK(inspected.first_seen == 1700000000);
    CHECK(inspected.tofu_enrolled);
    CHECK(inspected.tofu_boundary == "explicit-operator-first-use");
    CHECK(inspected.tofu_enrolled_at == 1700000000);
    CHECK(inspected.rotation_generation == 1);
    CHECK(inspected.rotation_history == 0);
    CHECK(!inspected.revoked);
    CHECK(inspected.pinned);
    CHECK(inspected.cert_sha256 == value.cert_sha256);
    CHECK(inspected.cert_thumbprint == value.cert_thumbprint);

    // A single flipped MAC character is corruption, not a mismatch.
    const std::string tampered = flip_mac(memory.bytes);
    memory.bytes = tampered;
    EXPECT(status::corrupt, verify(memory, value, error));

    // Non-canonical formatting is refused even when the JSON still parses.
    memory.bytes = tampered;
    test_memory_store pretty;
    EXPECT(status::ok, enroll_ok(pretty, value, error));
    std::string non_canonical = pretty.bytes;
    const size_t colon = non_canonical.find(':');
    CHECK(colon != std::string::npos);
    if (colon != std::string::npos) non_canonical.insert(colon + 1, " ");
    pretty.bytes = non_canonical;
    EXPECT(status::corrupt, verify(pretty, value, error));

    // Extra keys and a newer schema version are refused.
    test_memory_store extra;
    EXPECT(status::ok, enroll_ok(extra, value, error));
    const size_t insert = extra.bytes.find('{');
    CHECK(insert != std::string::npos);
    if (insert != std::string::npos) extra.bytes.insert(insert + 1, "\"extra\":1,");
    EXPECT(status::corrupt, verify(extra, value, error));

    test_memory_store version;
    EXPECT(status::ok, enroll_ok(version, value, error));
    const size_t version_key = version.bytes.find("\"version\":1");
    CHECK(version_key != std::string::npos);
    if (version_key != std::string::npos) version.bytes.replace(version_key, 11, "\"version\":2");
    EXPECT(status::corrupt, verify(version, value, error));

    test_clock = nullptr;
}

void test_pin_comparisons() {
    std::string error;
    const peer value = make_peer();
    set_gate("1");
    test_clock = [] { return uint64_t(1700000001); };

    struct case_value {
        const char * name;
        peer altered;
    };
    std::vector<case_value> cases;
    {
        peer altered = value;
        altered.pfn = "GianlucaMazza.xllama_1.6.0.437_x64__pj67f1fcj4n14";
        altered.package_family = "GianlucaMazza.xllama_pj67f1fcj4n14";
        cases.push_back({"pfn", altered});
    }
    // package_family is derived from the PFN and revalidated, so it cannot
    // differ independently of the PFN in a valid peer claim.
    {
        peer altered = value;
        altered.publisher = "CN=Someone Else, C=IT";
        cases.push_back({"publisher", altered});
    }
    {
        peer altered = value;
        altered.host = "192.168.1.201";
        cases.push_back({"host", altered});
    }
    {
        peer altered = value;
        altered.port = 11444;
        cases.push_back({"port", altered});
    }
    {
        peer altered = value;
        altered.cert_sha256[0] ^= 0x80;
        cases.push_back({"cert", altered});
    }
    {
        peer altered = value;
        altered.cert_thumbprint[0] ^= 0x80;
        cases.push_back({"thumbprint", altered});
    }

    for (const auto & item : cases) {
        test_memory_store memory;
        EXPECT(status::ok, enroll_ok(memory, value, error));
        const status actual = verify(memory, item.altered, error);
        expect_status(status::mismatch, actual, error, item.name, __LINE__);
        CHECK(error.find("mismatch") != std::string::npos);
        CHECK(error.find(to_hex(memory.stored_key.data(), memory.stored_key.size())) ==
              std::string::npos);
        CHECK(error.find("auth_secret") == std::string::npos);
    }

    // A host name is canonicalized before comparison, but not before storage.
    test_memory_store memory;
    peer upper = value;
    upper.host = "XBOX-LAB.LOCAL";
    EXPECT(status::ok, enroll_ok(memory, upper, error));
    CHECK(verify(memory, value, error) == status::mismatch);
    peer lower = value;
    lower.host = "xbox-lab.local";
    EXPECT(status::ok, verify(memory, lower, error));

    test_clock = nullptr;
}

void test_tofu_boundary() {
    std::string error;
    const peer value = make_peer();
    set_gate("1");
    test_clock = [] { return uint64_t(1700000002); };

    test_memory_store memory;
    enrollment request;
    request.operator_label = "lab-console-a";
    request.out_of_band_cert_sha256 = value.cert_sha256;
    EXPECT(status::unauthorized, enroll_tofu(memory, value, request, error));
    CHECK(error.find("explicit operator confirmation") != std::string::npos);

    request.explicit_operator_confirmation = true;
    request.out_of_band_cert_sha256 = make_peer(0x99).cert_sha256;
    EXPECT(status::unauthorized, enroll_tofu(memory, value, request, error));
    CHECK(error.find("out-of-band") != std::string::npos);

    request.out_of_band_cert_sha256 = value.cert_sha256;
    EXPECT(status::ok, enroll_tofu(memory, value, request, error));
    EXPECT(status::ok, verify(memory, value, error));
    EXPECT(status::already_pinned, enroll_tofu(memory, value, request, error));
    EXPECT(status::already_pinned, provision_ok(memory, value, error));

    // Provisioning also requires an explicitly authorized session and an
    // out-of-band digest; it is never a TOFU shortcut.
    test_memory_store provisioning_store;
    provisioning provision_request;
    provision_request.operator_label = "lab-provision";
    provision_request.out_of_band_cert_sha256 = value.cert_sha256;
    provision_request.observed = value;
    EXPECT(status::unauthorized, provision(provisioning_store, provision_request, error));
    provision_request.authorized_wdp_session = true;
    provision_request.out_of_band_cert_sha256 = make_peer(0x99).cert_sha256;
    EXPECT(status::unauthorized, provision(provisioning_store, provision_request, error));
    provision_request.out_of_band_cert_sha256 = value.cert_sha256;
    EXPECT(status::ok, provision(provisioning_store, provision_request, error));
    view inspected;
    EXPECT(status::ok, inspect(provisioning_store, inspected, error));
    CHECK(!inspected.tofu_enrolled);
    CHECK(inspected.tofu_boundary == "authorized-provisioning");
    EXPECT(status::ok, verify(provisioning_store, value, error));
    EXPECT(status::already_pinned, enroll_ok(provisioning_store, value, error));

    test_clock = nullptr;
}

void test_rotation() {
    std::string error;
    const peer first = make_peer(0x11);
    const peer second = make_peer(0x55);
    const peer third = make_peer(0x77);
    set_gate("1");
    test_clock = [] { return uint64_t(1700000003); };

    test_memory_store memory;
    EXPECT(status::ok, enroll_ok(memory, first, error));

    rotation request;
    request.operator_label = "lab-rotate";
    request.out_of_band_cert_sha256 = second.cert_sha256;
    EXPECT(status::unauthorized, rotate(memory, second, request, error));

    request.explicit_operator_confirmation = true;
    request.out_of_band_cert_sha256 = third.cert_sha256;
    EXPECT(status::unauthorized, rotate(memory, second, request, error));

    request.out_of_band_cert_sha256 = second.cert_sha256;
    EXPECT(status::ok, rotate(memory, second, request, error));
    view inspected;
    EXPECT(status::ok, inspect(memory, inspected, error));
    CHECK(inspected.rotation_generation == 2);
    CHECK(inspected.rotation_history == 1);
    CHECK(inspected.rotated_at == 1700000003);
    CHECK(inspected.cert_sha256 == second.cert_sha256);
    EXPECT(status::ok, verify(memory, second, error));
    EXPECT(status::mismatch, verify(memory, first, error));

    request.out_of_band_cert_sha256 = first.cert_sha256;
    EXPECT(status::mismatch, rotate(memory, first, request, error));
    request.out_of_band_cert_sha256 = second.cert_sha256;
    EXPECT(status::invalid_argument, rotate(memory, second, request, error));

    request.out_of_band_cert_sha256 = third.cert_sha256;
    EXPECT(status::ok, rotate(memory, third, request, error));
    EXPECT(status::ok, inspect(memory, inspected, error));
    CHECK(inspected.rotation_generation == 3);
    CHECK(inspected.rotation_history == 2);
    EXPECT(status::mismatch, verify(memory, first, error));
    EXPECT(status::mismatch, verify(memory, second, error));
    EXPECT(status::ok, verify(memory, third, error));

    test_clock = nullptr;
}

void test_revocation_and_wipe() {
    std::string error;
    const peer value = make_peer();
    const peer replacement = make_peer(0x55);
    set_gate("1");
    test_clock = [] { return uint64_t(1700000004); };

    test_memory_store memory;
    EXPECT(status::ok, enroll_ok(memory, value, error));

    revocation request;
    request.operator_label = "lab-revoke";
    request.reason = "certificate-compromised";
    EXPECT(status::unauthorized, revoke(memory, request, error));
    request.explicit_operator_confirmation = true;
    EXPECT(status::ok, revoke(memory, request, error));
    EXPECT(status::revoked, verify(memory, value, error));
    EXPECT(status::revoked, rotate_ok(memory, replacement, error));
    EXPECT(status::revoked, revoke(memory, request, error));

    view inspected;
    EXPECT(status::ok, inspect(memory, inspected, error));
    CHECK(inspected.revoked);
    CHECK(!inspected.pinned);
    CHECK(inspected.revocation_reason == "certificate-compromised");
    CHECK(inspected.revoked_at == 1700000004);

    wipe_request wipe_request_value;
    wipe_request_value.operator_label = "lab-wipe";
    EXPECT(status::unauthorized, wipe(memory, wipe_request_value, error));
    wipe_request_value.explicit_operator_confirmation = true;
    EXPECT(status::ok, wipe(memory, wipe_request_value, error));
    EXPECT(status::unknown, verify(memory, value, error));
    EXPECT(status::ok, enroll_ok(memory, replacement, error));

    test_clock = nullptr;
}

void test_fail_closed_platform_failures() {
    std::string error;
    const peer value = make_peer();
    set_gate("1");
    test_clock = [] { return uint64_t(1700000005); };

    test_memory_store unavailable;
    unavailable.fail_read = true;
    EXPECT(status::unavailable, verify(unavailable, value, error));

    test_memory_store write_failure;
    write_failure.fail_write = true;
    EXPECT(status::unavailable, enroll_ok(write_failure, value, error));

    // An unavailable protected key is corruption, not a fallback to an
    // unauthenticated record.
    test_memory_store key_failure;
    EXPECT(status::ok, enroll_ok(key_failure, value, error));
    key_failure.bytes.replace(key_failure.bytes.find("\"key_protected\":\""),
                              std::string("\"key_protected\":\"").size() + 4, "\"key_protected\":\"zzzz");
    EXPECT(status::corrupt, verify(key_failure, value, error));

    test_clock = nullptr;
}

#ifndef _WIN32
std::string read_text(const fs::path & path) {
    std::ifstream input(path, std::ios::binary);
    std::ostringstream out;
    out << input.rdbuf();
    return out.str();
}

void write_text(const fs::path & path, const std::string & text) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(text.data(), std::streamsize(text.size()));
}

std::vector<std::string> names(const fs::path & root) {
    std::vector<std::string> out;
    std::error_code code;
    for (const auto & entry : fs::directory_iterator(root, code)) {
        out.push_back(entry.path().filename().string());
    }
    std::sort(out.begin(), out.end());
    return out;
}

const char * test_stop_step = nullptr;

bool stop_hook(const char * name) {
    return test_stop_step == nullptr || std::strcmp(name, test_stop_step) != 0;
}

fs::path temp_root() {
    std::string pattern = "/tmp/rkva-peer-identity-XXXXXX";
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    char * made = ::mkdtemp(buffer.data());
    return made ? fs::path(made) : fs::path();
}

void test_file_store() {
    std::string error;
    const peer value = make_peer();
    const peer replacement = make_peer(0x55);
    const fs::path root = temp_root();
    CHECK(!root.empty());
    if (root.empty()) return;
    set_gate("1");
    test_clock = [] { return uint64_t(1700000006); };

    {
        auto backend = open_test_directory(root.string(), error);
        CHECK(backend != nullptr);
        if (!backend) return;
        EXPECT(status::unknown, verify(*backend, value, error));
        EXPECT(status::ok, enroll_ok(*backend, value, error));

        const fs::path namespace_root = root / namespace_name;
        const std::vector<std::string> entries = names(namespace_root);
        CHECK(entries.size() == 2);
        CHECK(entries.size() == 2 && entries[0] == "identity.json" && entries[1] == "identity.lock");
        struct stat info {};
        CHECK(::stat((namespace_root / record_name).c_str(), &info) == 0);
        CHECK((info.st_mode & 0777) == 0600);
        CHECK(::stat((namespace_root / lock_name).c_str(), &info) == 0);
        CHECK((info.st_mode & 0777) == 0600);
        CHECK(::stat(namespace_root.c_str(), &info) == 0);
        CHECK((info.st_mode & 0777) == 0700);

        auto reloaded = open_test_directory(root.string(), error);
        CHECK(reloaded != nullptr);
        if (reloaded) EXPECT(status::ok, verify(*reloaded, value, error));

        // Pending marker at rest is ambiguous and refused.
        write_text(namespace_root / pending_name, "pending\n");
        EXPECT(status::corrupt, verify(*backend, value, error));
        fs::remove(namespace_root / pending_name);

        // Unknown entries are refused, including a reparse/link substitute.
        write_text(namespace_root / "unexpected", "x");
        EXPECT(status::corrupt, verify(*backend, value, error));
        fs::remove(namespace_root / "unexpected");

        const fs::path record = namespace_root / record_name;
        const fs::path backup = root / "identity.backup";
        fs::rename(record, backup);
        CHECK(::symlink(backup.c_str(), record.c_str()) == 0);
        EXPECT(status::corrupt, verify(*backend, value, error));
        fs::remove(record);
        fs::rename(backup, record);

        // Truncation and non-canonical text are corruption.
        const std::string valid = read_text(record);
        write_text(record, valid.substr(0, valid.size() / 2));
        EXPECT(status::corrupt, verify(*backend, value, error));
        write_text(record, valid);
        EXPECT(status::ok, verify(*backend, value, error));

        write_text(record, flip_mac(valid));
        EXPECT(status::corrupt, verify(*backend, value, error));

        wipe_request wipe_request_value;
        wipe_request_value.explicit_operator_confirmation = true;
        wipe_request_value.operator_label = "lab-wipe";
        EXPECT(status::ok, wipe(*backend, wipe_request_value, error));
        EXPECT(status::unknown, verify(*backend, value, error));
        EXPECT(status::ok, enroll_ok(*backend, value, error));
    }

    // A second writer that holds identity.lock excludes every read/write.
    {
        auto backend = open_test_directory(root.string(), error);
        CHECK(backend != nullptr);
        const fs::path lock_path = root / namespace_name / lock_name;
        const int fd = ::open(lock_path.c_str(), O_RDWR | O_CLOEXEC);
        CHECK(fd >= 0);
        if (fd >= 0) {
            CHECK(::flock(fd, LOCK_EX | LOCK_NB) == 0);
            EXPECT(status::unavailable, verify(*backend, value, error));
            EXPECT(status::unavailable, enroll_ok(*backend, replacement, error));
            ::flock(fd, LOCK_UN);
            ::close(fd);
            EXPECT(status::ok, verify(*backend, value, error));
        }
    }

    // Crash-window injection: every partial state must be refused, and only an
    // explicit wipe may recover it.
    for (const char * stop : {"after_pending", "after_staging", "after_rename"}) {
        const fs::path crash_root = temp_root();
        CHECK(!crash_root.empty());
        if (crash_root.empty()) continue;
        auto backend = open_test_directory(crash_root.string(), error);
        CHECK(backend != nullptr);
        if (!backend) continue;
        test_stop_step = stop;
        test_step = stop_hook;
        const status enroll_status = enroll_ok(*backend, value, error);
        test_step = nullptr;
        test_stop_step = nullptr;
        CHECK(enroll_status != status::ok);
        EXPECT(status::corrupt, verify(*backend, value, error));
        // Recovery is explicit, not silent repair or adoption.
        wipe_request wipe_request_value;
        wipe_request_value.explicit_operator_confirmation = true;
        wipe_request_value.operator_label = "lab-recover";
        EXPECT(status::ok, wipe(*backend, wipe_request_value, error));
        EXPECT(status::unknown, verify(*backend, value, error));
        EXPECT(status::ok, enroll_ok(*backend, value, error));
        EXPECT(status::ok, verify(*backend, value, error));
        fs::remove_all(crash_root);
    }

    // A missing LocalState directory is unavailable, never silently created.
    {
        std::string missing_error;
        auto backend = open_test_directory((root / "does-not-exist").string(), missing_error);
        CHECK(backend != nullptr);
        if (backend) {
            EXPECT(status::unavailable, enroll_ok(*backend, value, missing_error));
        }
    }

    test_clock = nullptr;
    fs::remove_all(root);
}
#endif

}  // namespace

int main() {
    test_gate();
    test_schema_and_mac();
    test_pin_comparisons();
    test_tofu_boundary();
    test_rotation();
    test_revocation_and_wipe();
    test_fail_closed_platform_failures();
#ifndef _WIN32
    test_file_store();
#endif
    if (failures != 0) {
        std::cerr << failures << " peer identity checks failed\n";
        return 1;
    }
    std::cout << "peer identity store: schema, pin, TOFU, rotation, revocation, wipe and "
                 "fail-closed behavior passed\n";
    return 0;
}
