// CPU-only contract for the durable native snapshot reference journal:
// append/load/corruption, pending->committed folding, migrated pins, host-file
// removal keeping the record, retire decisions and the read-only audit. The
// append contract is exercised with the real exclusive route-state store lock,
// which is the same critical section used by the explicit and automatic native
// save paths.
#include "../tools/server/server-native-refs.h"
#include "../tools/server/server-route-state.h"

#undef NDEBUG
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace refs = server_native_refs;

namespace {

const char * const kKeyA = "3c1ac4014f51a0ff550c3e1346463623140e22f8c9a21bba80ff66d6b09fc582";
const char * const kKeyB = "fff30e532caeac47750f17351664f3b392ffdafc0801514d09395c5326bee09a";
const char * const kKeyC = "ef1c05ae42b5d79406ba3cc48b15f8107049a5c07e76339b23ad9fd96b972dd7";
const char * const kTime = "2026-11-01T00:00:00Z";

refs::record make_record(const std::string & key, const std::string & host, refs::state st,
        refs::source src = refs::source::explicit_save) {
    refs::record entry;
    entry.st        = st;
    entry.key       = key;
    entry.host_file = host;
    entry.n_tokens  = 128;
    entry.split_p   = 96;
    entry.native    = true;
    entry.timestamp = kTime;
    entry.src       = src;
    return entry;
}

void write_text(const fs::path & path, const std::string & text) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    assert(out);
    out << text;
    out.flush();
    assert(out.good());
}

std::string read_text(const fs::path & path) {
    std::ifstream in(path, std::ios::binary);
    assert(in);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void test_serialize_roundtrip() {
    std::string error;
    auto pending = make_record(kKeyA, "auto-quote\"name.bin", refs::state::pending, refs::source::auto_save);
    assert(refs::valid_record(pending, &error));
    const std::string line = refs::to_json(pending);
    assert(line.find("\\\"") != std::string::npos);
    refs::record parsed;
    assert(refs::parse_line(line, parsed, &error));
    assert(parsed.key == kKeyA && parsed.host_file == pending.host_file);
    assert(parsed.st == refs::state::pending && parsed.src == refs::source::auto_save);
    assert(parsed.n_tokens == 128 && parsed.split_p == 96 && parsed.native);

    auto upper = make_record("3C1AC4014F51A0FF550C3E1346463623140E22F8C9A21BBA80FF66D6B09FC582",
            "explicit.bin", refs::state::committed);
    const std::string upper_line = refs::to_json(upper);
    refs::record normalized;
    assert(refs::parse_line(upper_line, normalized, &error));
    assert(normalized.key == kKeyA);

    auto pinned = make_record(kKeyB, "", refs::state::pinned, refs::source::migrated);
    assert(refs::valid_record(pinned, &error));
    refs::record pinned_parsed;
    assert(refs::parse_line(refs::to_json(pinned), pinned_parsed, &error));
    assert(pinned_parsed.st == refs::state::pinned && pinned_parsed.src == refs::source::migrated);

    // Invalid encodings are refused.
    auto bad = make_record("not-a-key", "x.bin", refs::state::pending);
    assert(!refs::valid_record(bad, &error));
    bad = make_record(kKeyA, "../escape.bin", refs::state::pending);
    assert(!refs::valid_record(bad, &error));
    bad = make_record(kKeyA, "x.bin", refs::state::pinned, refs::source::migrated);
    assert(!refs::valid_record(bad, &error));
    bad = make_record(kKeyA, "x.bin", refs::state::pending, refs::source::migrated);
    assert(!refs::valid_record(bad, &error));
}

void test_append_load_and_lock(const fs::path & root) {
    const fs::path store = root / "store-lock";
    fs::create_directories(store);
    const std::string journal = refs::journal_path(store.string());
    std::string error;

    // Without the exclusive store lock the append must be refused.
    auto pending = make_record(kKeyA, "explicit.bin", refs::state::pending);
    assert(!refs::append(store.string(), pending, [](const std::string &) { return false; }, &error));
    assert(!fs::exists(journal));

    server_route_state_store_lock lock(store.string(), server_route_state_store_lock_mode::exclusive);
    assert(lock.acquired());
    const auto held = [&lock](const std::string & path) { return lock.owns_exclusive_for(path); };
    assert(refs::append(store.string(), pending, held, &error));
    auto committed = pending;
    committed.st = refs::state::committed;
    assert(refs::append(store.string(), committed, held, &error));

    // A shared lock is not sufficient for the journal critical section, and an
    // append after releasing the exclusive lock is refused.
    lock.release();
    {
        server_route_state_store_lock shared(store.string(), server_route_state_store_lock_mode::shared);
        assert(shared.acquired());
        assert(!refs::append(store.string(), pending,
                [&shared](const std::string & path) { return shared.owns_exclusive_for(path); }, &error));
    }
    assert(!refs::append(store.string(), pending, held, &error));

    const refs::load_result loaded = refs::load(journal);
    assert(loaded.ok && !loaded.truncated_tail);
    assert(loaded.raw.size() == 2);
    assert(loaded.records.size() == 1);
    assert(loaded.records.front().st == refs::state::committed);
    assert(loaded.records.front().key == kKeyA);

    // Both sources are accepted and survive a reload.
    auto auto_entry = make_record(kKeyB, "auto-1.bin", refs::state::pending, refs::source::auto_save);
    server_route_state_store_lock relock(store.string(), server_route_state_store_lock_mode::exclusive);
    assert(relock.acquired());
    const auto reheld = [&relock](const std::string & path) { return relock.owns_exclusive_for(path); };
    assert(refs::append(store.string(), auto_entry, reheld, &error));
    const refs::load_result reloaded = refs::load(journal);
    assert(reloaded.ok && reloaded.raw.size() == 3);
    assert(reloaded.records.size() == 2);
    assert(std::any_of(reloaded.records.begin(), reloaded.records.end(), [](const refs::record & r) {
        return r.key == kKeyB && r.src == refs::source::auto_save;
    }));
}

void test_corruption(const fs::path & root) {
    const fs::path journal = root / "corrupt.jsonl";
    const std::string good = refs::to_json(make_record(kKeyA, "a.bin", refs::state::pending)) + "\n";

    // A complete malformed line is corruption, and the load fails closed.
    write_text(journal, good + "{\"schema\":\"rkva-native-refs\"}\n");
    refs::load_result result = refs::load(journal.string());
    assert(!result.ok);
    assert(result.error.find("line 2") != std::string::npos);

    // A torn final append is reported and ignored, keeping earlier records.
    write_text(journal, good + "{\"schema\":\"rkva-native-refs\",\"version\":1");
    result = refs::load(journal.string());
    assert(result.ok && result.truncated_tail);
    assert(result.raw.size() == 1 && result.records.size() == 1);

    // A complete object without its newline terminator is not lost: it is kept
    // and reported, and the next append repairs the missing terminator.
    write_text(journal, good + refs::to_json(make_record(kKeyB, "b.bin", refs::state::committed)));
    result = refs::load(journal.string());
    assert(result.ok && result.truncated_tail);
    assert(result.raw.size() == 2);
    {
        const fs::path store = root / "repair";
        fs::create_directories(store);
        std::error_code ec;
        fs::copy_file(journal, refs::journal_path(store.string()), ec);
        assert(!ec);
        server_route_state_store_lock lock(store.string());
        assert(lock.acquired());
        const auto held = [&lock](const std::string & path) { return lock.owns_exclusive_for(path); };
        std::string append_error;
        assert(refs::append(store.string(), make_record(kKeyC, "c.bin", refs::state::pending),
                held, &append_error));
        const refs::load_result repaired = refs::load(refs::journal_path(store.string()));
        assert(repaired.ok && !repaired.truncated_tail);
        assert(repaired.raw.size() == 3);
    }

    // A torn partial record is never merged into the next append.
    write_text(journal, good + "{\"schema\":\"rkva-native-refs\",\"version\":1");
    {
        const fs::path store = root / "torn";
        fs::create_directories(store);
        std::error_code ec;
        fs::copy_file(journal, refs::journal_path(store.string()), ec);
        assert(!ec);
        server_route_state_store_lock lock(store.string());
        assert(lock.acquired());
        const auto held = [&lock](const std::string & path) { return lock.owns_exclusive_for(path); };
        std::string append_error;
        assert(!refs::append(store.string(), make_record(kKeyC, "c.bin", refs::state::pending),
                held, &append_error));
        assert(append_error.find("incomplete record") != std::string::npos);
        // The torn fragment was not rewritten or deleted.
        const std::string content = read_text(refs::journal_path(store.string()));
        assert(content.find("{\"schema\":\"rkva-native-refs\",\"version\":1") != std::string::npos);
    }

    // Wrong schema, duplicate fields, unknown state and oversized numbers.
    const std::vector<std::string> bad_lines = {
        "{\"schema\":\"other\",\"version\":1,\"state\":\"pending\",\"xbox_key\":\"" + std::string(kKeyA) +
            "\",\"host_file\":\"a.bin\",\"n_tokens\":1,\"split_p\":0,\"native\":true,\"timestamp\":\"" +
            kTime + "\",\"source\":\"explicit\"}",
        "{\"schema\":\"rkva-native-refs\",\"version\":1,\"state\":\"pending\",\"state\":\"committed\",\"xbox_key\":\"" +
            std::string(kKeyA) + "\",\"host_file\":\"a.bin\",\"n_tokens\":1,\"split_p\":0,\"native\":true,\"timestamp\":\"" +
            kTime + "\",\"source\":\"explicit\"}",
        "{\"schema\":\"rkva-native-refs\",\"version\":1,\"state\":\"gone\",\"xbox_key\":\"" + std::string(kKeyA) +
            "\",\"host_file\":\"a.bin\",\"n_tokens\":1,\"split_p\":0,\"native\":true,\"timestamp\":\"" +
            kTime + "\",\"source\":\"explicit\"}",
        "{\"schema\":\"rkva-native-refs\",\"version\":1,\"state\":\"pending\",\"xbox_key\":\"" + std::string(kKeyA) +
            "\",\"host_file\":\"a.bin\",\"n_tokens\":99999999999999999999999999,\"split_p\":0,\"native\":true,\"timestamp\":\"" +
            kTime + "\",\"source\":\"explicit\"}",
    };
    for (const auto & line : bad_lines) {
        refs::record parsed;
        std::string error;
        assert(!refs::parse_line(line, parsed, &error));
        write_text(journal, line + "\n");
        const refs::load_result rejected = refs::load(journal.string());
        assert(!rejected.ok);
    }

    // A missing journal is an empty journal, not an error.
    const refs::load_result missing = refs::load((root / "absent.jsonl").string());
    assert(missing.ok && missing.raw.empty() && !missing.truncated_tail);
}

void test_host_removal_and_retire(const fs::path & root) {
    const fs::path store = root / "host-removal";
    fs::create_directories(store);
    const fs::path host = store / "explicit.bin";
    write_text(host, "host-state");
    server_route_state_store_lock lock(store.string());
    assert(lock.acquired());
    const auto held = [&lock](const std::string & path) { return lock.owns_exclusive_for(path); };
    std::string error;
    auto pending = make_record(kKeyA, host.filename().string(), refs::state::pending);
    assert(refs::append(store.string(), pending, held, &error));
    auto committed = pending;
    committed.st = refs::state::committed;
    assert(refs::append(store.string(), committed, held, &error));
    lock.release();

    fs::remove(host);
    assert(!fs::exists(host));

    // The record survives host eviction: the journal is the durable reference.
    const refs::load_result loaded = refs::load(refs::journal_path(store.string()));
    assert(loaded.ok && loaded.records.size() == 1);
    assert(loaded.records.front().st == refs::state::committed);
    assert(refs::retire_decision(kKeyA, loaded.records, {}) == refs::retire_outcome::referenced);

    // Pending only: referenced while the host file is present, otherwise a
    // candidate (never a remote delete).
    std::vector<refs::record> only_pending = {pending};
    assert(refs::retire_decision(kKeyA, only_pending, {host.filename().string()}) ==
            refs::retire_outcome::referenced);
    assert(refs::retire_decision(kKeyA, only_pending, {}) ==
            refs::retire_outcome::orphan_candidate);

    // A pin always wins and unrelated keys are candidates.
    std::vector<refs::record> pinned = {make_record(kKeyB, "", refs::state::pinned, refs::source::migrated)};
    assert(refs::retire_decision(kKeyB, pinned, {}) == refs::retire_outcome::pinned);
    assert(refs::retire_decision(kKeyC, pinned, {}) == refs::retire_outcome::orphan_candidate);
    assert(refs::retire_decision("invalid", pinned, {}) == refs::retire_outcome::orphan_candidate);
    // A pin wins over a committed record for the same key regardless of order.
    std::vector<refs::record> pinned_and_committed = {committed,
        make_record(kKeyA, "", refs::state::pinned, refs::source::migrated)};
    assert(refs::retire_decision(kKeyA, pinned_and_committed, {}) == refs::retire_outcome::pinned);
}

void test_migration() {
    std::vector<std::string> keys(refs::kBaselineKeys.begin(), refs::kBaselineKeys.end());
    assert(keys.size() == 7);
    for (const auto & key : keys) {
        assert(refs::is_hex_key(key));
    }
    const refs::migration first = refs::plan_migration(keys, {}, kTime);
    assert(first.pins.size() == 7 && first.already_pinned.empty() && first.invalid.empty());
    assert(first.pins.front().st == refs::state::pinned);
    assert(first.pins.front().src == refs::source::migrated);
    assert(first.pins.front().host_file.empty());

    std::vector<refs::record> journal = first.pins;
    const refs::migration second = refs::plan_migration(keys, journal, kTime);
    assert(second.pins.empty() && second.already_pinned.size() == 7);

    // Duplicates inside one request and invalid entries.
    const refs::migration duplicate = refs::plan_migration({kKeyA, kKeyA, "bad"}, {}, kTime);
    assert(duplicate.pins.size() == 1);
    assert(duplicate.invalid.size() == 1 && duplicate.invalid.front() == "bad");

    // A committed key can also be pinned by migration.
    const refs::migration pinned_committed = refs::plan_migration({kKeyB},
            {make_record(kKeyB, "b.bin", refs::state::committed)}, kTime);
    assert(pinned_committed.pins.size() == 1);
}

void test_migration_writes(const fs::path & root) {
    const fs::path store = root / "migrate";
    fs::create_directories(store);
    server_route_state_store_lock lock(store.string());
    assert(lock.acquired());
    const auto held = [&lock](const std::string & path) { return lock.owns_exclusive_for(path); };
    const refs::migration plan = refs::plan_migration(
            std::vector<std::string>(refs::kBaselineKeys.begin(), refs::kBaselineKeys.end()), {}, kTime);
    std::string error;
    for (const auto & pin : plan.pins) {
        assert(refs::append(store.string(), pin, held, &error));
    }
    lock.release();

    const refs::load_result loaded = refs::load(refs::journal_path(store.string()));
    assert(loaded.ok && loaded.records.size() == 7);
    for (const auto & entry : loaded.records) {
        assert(entry.st == refs::state::pinned);
        assert(refs::retire_decision(entry.key, loaded.records, {}) == refs::retire_outcome::pinned);
    }
}

void test_audit() {
    std::vector<refs::record> journal = {
        make_record(kKeyA, "", refs::state::pinned, refs::source::migrated),
        make_record(kKeyB, "b.bin", refs::state::committed),
        make_record(kKeyC, "gone.bin", refs::state::pending, refs::source::auto_save),
    };
    const std::vector<refs::host_key> hosts = {
        {"b.bin", kKeyB},
        {"new.bin", "1111111111111111111111111111111111111111111111111111111111111111"},
        {"unreadable.bin", ""},
    };
    const refs::audit_report report = refs::audit(journal, hosts);
    assert(report.pinned.size() == 1 && report.pinned.front() == kKeyA);
    assert(report.referenced.size() == 1 && report.referenced.front() == kKeyB);
    assert(report.pending_only.size() == 1 && report.pending_only.front() == kKeyC);
    assert(report.orphan_candidates.empty());
    assert(report.unregistered_host_files.size() == 1 &&
           report.unregistered_host_files.front() == "new.bin");
    assert(report.unreadable_host_files.size() == 1 &&
           report.unreadable_host_files.front() == "unreadable.bin");

    // A pending record with its host file present is no longer pending-only.
    std::vector<refs::host_key> with_gone = hosts;
    with_gone.push_back({"gone.bin", kKeyC});
    const refs::audit_report present = refs::audit(journal, with_gone);
    assert(present.pending_only.empty());
    assert(present.referenced.size() == 2);

    // Without the pending host file the key is explicitly only pending.
    const refs::audit_report absent = refs::audit(journal, hosts);
    assert(absent.pending_only.size() == 1);
}

void test_naming() {
    assert(refs::is_journal_file(".rkva-native-refs.jsonl"));
    assert(!refs::is_journal_file("auto-1.bin"));
    assert(refs::journal_path("/tmp/store") == "/tmp/store/.rkva-native-refs.jsonl");
}

} // namespace

int main() {
    const fs::path root = fs::temp_directory_path() /
        (std::string("rkva-native-refs-") + std::to_string((unsigned long long) std::time(nullptr)));
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);
    assert(!ec);

    test_serialize_roundtrip();
    test_append_load_and_lock(root);
    test_corruption(root);
    test_host_removal_and_retire(root);
    test_migration();
    test_migration_writes(root);
    test_audit();
    test_naming();

    fs::remove_all(root, ec);
    assert(!ec);
    std::printf("native snapshot reference journal CPU regression: ok\n");
    return 0;
}
