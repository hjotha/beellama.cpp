// CPU-only contract for the durable native snapshot reference journal:
// append/load/corruption, pending->committed folding, migrated pins, host-file
// removal keeping the record, retire decisions and the read-only audit. The
// append contract is exercised with the real exclusive route-state store lock,
// which is the same critical section used by the explicit and automatic native
// save paths.
#include "../tools/server/server-native-refs.h"
#include "../tools/server/server-native-rotation.h"
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

bool accept_remote(const uint8_t *, bool) { return true; }

const char * const kKeyA = "3c1ac4014f51a0ff550c3e1346463623140e22f8c9a21bba80ff66d6b09fc582";
const char * const kKeyB = "fff30e532caeac47750f17351664f3b392ffdafc0801514d09395c5326bee09a";
const char * const kKeyC = "ef1c05ae42b5d79406ba3cc48b15f8107049a5c07e76339b23ad9fd96b972dd7";
const char * const kKeyD = "1111111111111111111111111111111111111111111111111111111111111111";
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

refs::record make_retired(const std::string & key, const std::string & host) {
    refs::record entry;
    entry.st        = refs::state::retired;
    entry.key       = key;
    entry.host_file = host;
    entry.n_tokens  = 0;
    entry.split_p   = 0;
    entry.native    = true;
    entry.timestamp = kTime;
    entry.src       = refs::source::retired;
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

    auto retired = make_retired(kKeyC, "slot-c.bin");
    assert(refs::valid_record(retired, &error));
    refs::record retired_parsed;
    assert(refs::parse_line(refs::to_json(retired), retired_parsed, &error));
    assert(retired_parsed.st == refs::state::retired);
    assert(retired_parsed.src == refs::source::retired);
    assert(retired_parsed.host_file == "slot-c.bin");
    assert(retired_parsed.n_tokens == 0 && retired_parsed.split_p == 0);

    // Invalid encodings are refused.
    auto bad = make_record("not-a-key", "x.bin", refs::state::pending);
    assert(!refs::valid_record(bad, &error));
    bad = make_record(kKeyA, "../escape.bin", refs::state::pending);
    assert(!refs::valid_record(bad, &error));
    bad = make_record(kKeyA, "x.bin", refs::state::pinned, refs::source::migrated);
    assert(!refs::valid_record(bad, &error));
    bad = make_record(kKeyA, "x.bin", refs::state::pending, refs::source::migrated);
    assert(!refs::valid_record(bad, &error));
    // The retired state has its own source and no token geometry.
    bad = make_record(kKeyA, "x.bin", refs::state::retired, refs::source::explicit_save);
    assert(!refs::valid_record(bad, &error));
    bad = make_record(kKeyA, "x.bin", refs::state::retired, refs::source::retired);
    assert(!refs::valid_record(bad, &error));
    bad = make_record(kKeyA, "x.bin", refs::state::committed, refs::source::retired);
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

void test_retired_fold_and_decision() {
    const auto committed = make_record(kKeyA, "a.bin", refs::state::committed);
    const auto pin = make_record(kKeyA, "", refs::state::pinned, refs::source::migrated);
    const auto retired = make_retired(kKeyA, "a.bin");

    // A retired record supersedes committed and pinned records for the key.
    std::vector<refs::record> folded = refs::fold_records({committed, pin, retired});
    assert(folded.size() == 1);
    assert(folded.front().st == refs::state::retired);
    assert(refs::retire_decision(kKeyA, folded, {"a.bin"}) == refs::retire_outcome::retired);
    assert(refs::retire_decision(kKeyA, folded, {}) == refs::retire_outcome::retired);

    // Append order is operation order: a later committed record is a re-save
    // and clears the retirement again.
    folded = refs::fold_records({committed, retired, committed});
    assert(folded.size() == 1 && folded.front().st == refs::state::committed);
    assert(refs::retire_decision(kKeyA, folded, {}) == refs::retire_outcome::referenced);

    // A later pin is an explicit re-reference.
    folded = refs::fold_records({retired, pin});
    assert(folded.size() == 1 && folded.front().st == refs::state::pinned);
    assert(refs::retire_decision(kKeyA, folded, {}) == refs::retire_outcome::pinned);

    // A pending record after retirement is a save in flight: conservative
    // referenced with its host file present, candidate otherwise.
    const auto pending = make_record(kKeyA, "b.bin", refs::state::pending);
    folded = refs::fold_records({retired, pending});
    assert(folded.size() == 1 && folded.front().st == refs::state::pending);
    assert(refs::retire_decision(kKeyA, folded, {"b.bin"}) == refs::retire_outcome::referenced);
    assert(refs::retire_decision(kKeyA, folded, {}) == refs::retire_outcome::orphan_candidate);

    // Retiring one key never affects another key's records.
    folded = refs::fold_records({committed, make_retired(kKeyB, "b.bin")});
    assert(folded.size() == 2);
    assert(refs::retire_decision(kKeyA, folded, {}) == refs::retire_outcome::referenced);
    assert(refs::retire_decision(kKeyB, folded, {}) == refs::retire_outcome::retired);
    assert(refs::retire_decision(kKeyC, folded, {}) == refs::retire_outcome::orphan_candidate);
}

void test_plan_retire() {
    const auto committed = make_record(kKeyA, "a.bin", refs::state::committed);
    const auto pin = make_record(kKeyA, "", refs::state::pinned, refs::source::migrated);
    const auto pending = make_record(kKeyA, "a.bin", refs::state::pending);
    const auto retired = make_retired(kKeyA, "a.bin");

    refs::retire_plan plan = refs::plan_retire("not-a-key", {}, false, kTime);
    assert(plan.refusal == refs::retire_refusal::invalid_key && !plan.execute);

    plan = refs::plan_retire(kKeyA, {}, false, kTime);
    assert(plan.refusal == refs::retire_refusal::unknown_key && !plan.execute);

    plan = refs::plan_retire(kKeyA, {pending}, false, kTime);
    assert(plan.refusal == refs::retire_refusal::pending && !plan.execute);

    plan = refs::plan_retire(kKeyA, {pin}, false, kTime);
    assert(plan.refusal == refs::retire_refusal::pinned && !plan.execute);
    plan = refs::plan_retire(kKeyA, {pin}, true, kTime);
    assert(plan.refusal == refs::retire_refusal::none && plan.execute);
    assert(plan.host_files.empty() && plan.entry.host_file.empty());
    assert(plan.entry.st == refs::state::retired && plan.entry.src == refs::source::retired);
    assert(plan.entry.key == kKeyA && plan.entry.n_tokens == 0 && plan.entry.split_p == 0);
    assert(plan.entry.timestamp == kTime);

    plan = refs::plan_retire(kKeyA, {committed, pin}, true, kTime);
    assert(plan.execute && plan.host_files.size() == 1 && plan.host_files.front() == "a.bin");
    assert(plan.entry.host_file == "a.bin");

    plan = refs::plan_retire(kKeyA, {retired}, false, kTime);
    assert(plan.refusal == refs::retire_refusal::already_retired && !plan.execute);
    assert(plan.host_files.size() == 1 && plan.host_files.front() == "a.bin");
    assert(plan.entry.key == kKeyA);

    // A committed record appended after retirement reopens the key.
    plan = refs::plan_retire(kKeyA, refs::fold_records({retired, committed}), false, kTime);
    assert(plan.refusal == refs::retire_refusal::none && plan.execute);

    // Uppercase input is normalized.
    std::string upper(kKeyA);
    for (char & c : upper) {
        if (c >= 'a' && c <= 'f') c = char(c - 'a' + 'A');
    }
    plan = refs::plan_retire(upper, {committed}, false, kTime);
    assert(plan.execute && plan.entry.key == kKeyA);
}

void test_retired_journal(const fs::path & root) {
    const fs::path store = root / "retired-journal";
    fs::create_directories(store);
    const fs::path host = store / "a.bin";
    write_text(host, "host-state");
    server_route_state_store_lock lock(store.string());
    assert(lock.acquired());
    const auto held = [&lock](const std::string & path) { return lock.owns_exclusive_for(path); };
    std::string error;
    assert(refs::append(store.string(),
            make_record(kKeyA, host.filename().string(), refs::state::committed), held, &error));
    const auto retired = make_retired(kKeyA, host.filename().string());
    assert(refs::valid_record(retired, &error));
    assert(refs::append(store.string(), retired, held, &error));
    lock.release();

    fs::remove(host);
    const refs::load_result loaded = refs::load(refs::journal_path(store.string()));
    assert(loaded.ok && loaded.raw.size() == 2 && loaded.records.size() == 1);
    assert(loaded.records.front().st == refs::state::retired);
    assert(refs::retire_decision(kKeyA, loaded.records, {}) == refs::retire_outcome::retired);

    const refs::retire_plan again = refs::plan_retire(kKeyA, loaded.records, false, kTime);
    assert(again.refusal == refs::retire_refusal::already_retired && !again.execute);
}

void test_migration() {    std::vector<std::string> keys(refs::kBaselineKeys.begin(), refs::kBaselineKeys.end());
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

    // A retired key is never re-pinned: the remote object is gone until a real
    // save re-references it.
    const refs::migration retired = refs::plan_migration({kKeyA},
            {make_retired(kKeyA, "a.bin")}, kTime);
    assert(retired.pins.empty() && retired.already_pinned.empty());
    assert(retired.retired.size() == 1 && retired.retired.front() == kKeyA);
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
        make_retired(kKeyD, "d.bin"),
    };
    const std::vector<refs::host_key> hosts = {
        {"b.bin", kKeyB},
        {"d.bin", kKeyD},
        {"new.bin", "2222222222222222222222222222222222222222222222222222222222222222"},
        {"unreadable.bin", ""},
    };
    const refs::audit_report report = refs::audit(journal, hosts);
    assert(report.pinned.size() == 1 && report.pinned.front() == kKeyA);
    assert(report.referenced.size() == 1 && report.referenced.front() == kKeyB);
    assert(report.retired.size() == 1 && report.retired.front() == kKeyD);
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
    assert(present.retired.size() == 1);

    // Without the pending host file the key is explicitly only pending.
    const refs::audit_report absent = refs::audit(journal, hosts);
    assert(absent.pending_only.size() == 1);
}

void test_naming() {
    assert(refs::is_journal_file(".rkva-native-refs.jsonl"));
    assert(!refs::is_journal_file("auto-1.bin"));
    assert(refs::journal_path("/tmp/store") == "/tmp/store/.rkva-native-refs.jsonl");
}

void test_rotation_plan() {
    assert(!server_native_rotation::enabled(nullptr, true));
    assert(!server_native_rotation::enabled("0", true));
    assert(!server_native_rotation::enabled("1", false));
    assert(server_native_rotation::enabled("1", true));
    auto pinned = make_record(kKeyC, "", refs::state::pinned, refs::source::migrated);
    auto pending = make_record(kKeyB, "pending.bin", refs::state::pending);
    std::vector<refs::record> journal{
        make_record(kKeyA, "a.bin", refs::state::committed), pending, pinned,
        make_record(kKeyD, "d.bin", refs::state::committed)};
    const auto candidates = server_native_rotation::candidates(journal, "2222222222222222222222222222222222222222222222222222222222222222");
    assert(candidates.size() == 3);
    assert(candidates[0].key == kKeyD && candidates[1].key == kKeyA);
    assert(candidates[2].key == kKeyB && candidates[2].pending_files.size() == 1);
    ggml_remote_native_store_info info{};
    info.used_bytes = 600; info.limit_bytes = 1000; info.incoming_bytes = 200;
    info.objects = 8; info.max_objects = 8; info.matches = 2;
    info.match[0] = {2, 100}; info.match[1] = {1, 100};
    assert(server_native_rotation::select(info, candidates.size(), true) == 1); // remote oldest
    info.objects = 7;
    assert(server_native_rotation::select(info, candidates.size(), true) == -1);
    info.matches = 1; info.match[0] = {2, 100};
    assert(server_native_rotation::select(info, candidates.size(), false) == 0); // missing on Xbox
    info.incoming_bytes = 1001;
    assert(server_native_rotation::select(info, candidates.size(), true) == -2);
    info.matches = 1; info.match[0] = {0, 100}; // already published, no remote extra space
    assert(server_native_rotation::select(info, candidates.size(), true) == -1);
    assert(server_native_rotation::candidates(journal, kKeyA).size() == 2);
}

void test_rotation_pending_orphan(const fs::path & root) {
    const fs::path dir = root / "rotation-pending";
    fs::create_directory(dir);
    server_route_state_store_lock store(dir.string());
    const auto held = [&store](const std::string & p) { return store.owns_exclusive_for(p); };
    std::string error;
    assert(refs::append(dir.string(), make_record(kKeyD, "pending.bin", refs::state::pending), held, &error));
    auto journal = refs::load(refs::journal_path(dir.string()));
    auto options = server_native_rotation::candidates(journal.records, kKeyA);
    assert(options.size() == 1 && options[0].pending_files.size() == 1);
    const auto path = dir / "pending.bin";
    { std::ofstream file(path); file << "possibly published"; }
    unsigned remote_calls = 0; size_t removed = 0;
    auto verify = [](const fs::path &, const std::string &) { return false; };
    auto remote = [&](const uint8_t *) { ++remote_calls; return true; };
    auto sync = [](const std::string & p) { return server_native_rotation::sync_directory(p); };
    assert(server_native_rotation::retire(options[0], dir.string(), store, verify, accept_remote, remote, sync, removed, error) ==
           server_native_rotation::retirement_result::skipped);
    assert(fs::exists(path) && remote_calls == 0);
    fs::remove(path);
    assert(server_native_rotation::retire(options[0], dir.string(), store, verify, accept_remote, remote, sync, removed, error) ==
           server_native_rotation::retirement_result::retired);
    assert(remote_calls == 1 && removed == 0);
    journal = refs::load(refs::journal_path(dir.string()));
    assert(journal.ok && journal.records.size() == 1 && journal.records[0].st == refs::state::retired);
}

void test_rotation_mixed_pending(const fs::path & root) {
    const fs::path dir = root / "rotation-mixed-pending";
    fs::create_directory(dir);
    const auto pending_path = dir / "pending.bin", committed_path = dir / "committed.bin";
    { std::ofstream file(pending_path); file << "pending"; }
    { std::ofstream file(committed_path); file << "valid"; }
    server_route_state_store_lock store(dir.string());
    const auto held = [&store](const std::string & p) { return store.owns_exclusive_for(p); };
    std::string error;
    assert(refs::append(dir.string(), make_record(kKeyD, "pending.bin", refs::state::pending), held, &error));
    assert(refs::append(dir.string(), make_record(kKeyD, "committed.bin", refs::state::committed), held, &error));
    auto journal = refs::load(refs::journal_path(dir.string()));
    auto options = server_native_rotation::candidates(journal.records, kKeyA);
    assert(options.size() == 1 && options[0].pending_files.size() == 1 &&
           options[0].retirement.host_files.size() == 1);
    unsigned remote_calls = 0; size_t removed = 0;
    auto verify = [](const fs::path & p, const std::string & key) {
        std::ifstream file(p); std::string value; file >> value; return key == kKeyD && value == "valid";
    };
    auto remote = [&](const uint8_t *) { ++remote_calls; assert(!fs::exists(committed_path)); return true; };
    auto sync = [](const std::string & p) { return server_native_rotation::sync_directory(p); };
    assert(server_native_rotation::retire(options[0], dir.string(), store, verify, accept_remote, remote, sync, removed, error) ==
           server_native_rotation::retirement_result::skipped);
    assert(fs::exists(pending_path) && fs::exists(committed_path) && remote_calls == 0);
    fs::remove(pending_path);
    assert(server_native_rotation::retire(options[0], dir.string(), store, verify, accept_remote, remote, sync, removed, error) ==
           server_native_rotation::retirement_result::retired);
    assert(removed == 1 && remote_calls == 1 && !fs::exists(committed_path));
    journal = refs::load(refs::journal_path(dir.string()));
    assert(journal.ok && journal.records.size() == 1 && journal.records[0].st == refs::state::retired);
}

void test_rotation_multiple_references(const fs::path & root) {
    const fs::path dir = root / "rotation-multiple";
    fs::create_directory(dir);
    server_route_state_store_lock store(dir.string());
    const auto first = dir / "first.bin";
    { std::ofstream file(first); file << "valid"; }
    fs::path second;
    {
        server_route_state_lease first_lock(first.string(), server_route_state_lock_mode::eviction);
        assert(first_lock.acquired());
        for (int i = 0; i < 128 && second.empty(); ++i) {
            const auto candidate = dir / ("second-" + std::to_string(i) + ".bin");
            server_route_state_lease candidate_lock(candidate.string(), server_route_state_lock_mode::eviction);
            if (candidate_lock.acquired()) second = candidate;
        }
    }
    assert(!second.empty());
    { std::ofstream file(second); file << "invalid"; }
    const auto held = [&store](const std::string & p) { return store.owns_exclusive_for(p); };
    std::string error;
    assert(refs::append(dir.string(), make_record(kKeyD, first.filename().string(), refs::state::committed), held, &error));
    assert(refs::append(dir.string(), make_record(kKeyD, second.filename().string(), refs::state::committed), held, &error));
    auto journal = refs::load(refs::journal_path(dir.string()));
    auto options = server_native_rotation::candidates(journal.records, kKeyA);
    assert(options.size() == 1 && options[0].retirement.host_files.size() == 2);
    unsigned remote_calls = 0; size_t removed = 0;
    auto verify = [](const fs::path & p, const std::string & key) {
        std::ifstream file(p); std::string value; file >> value; return key == kKeyD && value == "valid";
    };
    auto remote = [&](const uint8_t *) { ++remote_calls; assert(!fs::exists(first) && !fs::exists(second)); return true; };
    auto sync = [](const std::string & p) { return server_native_rotation::sync_directory(p); };
    assert(server_native_rotation::retire(options[0], dir.string(), store, verify, accept_remote, remote, sync, removed, error) ==
           server_native_rotation::retirement_result::skipped);
    assert(fs::exists(first) && fs::exists(second) && remote_calls == 0);
    { std::ofstream file(second, std::ios::trunc); file << "valid"; }
    assert(server_native_rotation::retire(options[0], dir.string(), store, verify, accept_remote, remote, sync, removed, error) ==
           server_native_rotation::retirement_result::retired);
    assert(removed == 2 && remote_calls == 1);
    journal = refs::load(refs::journal_path(dir.string()));
    assert(journal.ok && journal.records.size() == 1 && journal.records[0].st == refs::state::retired);
}

void test_rotation_retirement(const fs::path & root) {
    for (int failure : {0, 1, 2}) {
    const fs::path dir = root / (failure == 2 ? "rotation-sync-failed" :
                                 failure == 1 ? "rotation-lost-ack" : "rotation-failed-delete");
    fs::create_directory(dir);
    const auto path = dir / "old.bin";
    { std::ofstream file(path); file << "validated"; }
    server_route_state_store_lock store(dir.string());
    assert(store.acquired());
    const auto held = [&store](const std::string & p) { return store.owns_exclusive_for(p); };
    std::string error;
    assert(refs::append(dir.string(), make_record(kKeyD, "old.bin", refs::state::committed), held, &error));
    auto loaded = refs::load(refs::journal_path(dir.string()));
    auto options = server_native_rotation::candidates(loaded.records, kKeyA);
    assert(options.size() == 1);
    auto verify = [](const fs::path & p, const std::string & key) {
        std::ifstream file(p); std::string value; file >> value;
        return key == kKeyD && value == "validated";
    };
    size_t removed = 0;
    unsigned remote_calls = 0;
    bool remote_exists = true;
    bool synced = false;
    auto fail_remote = [&](const uint8_t *) { assert(synced); ++remote_calls; if (failure == 1) remote_exists = false; return false; };
    auto ok_remote = [&](const uint8_t *) { assert(synced); ++remote_calls; assert(!fs::exists(path)); remote_exists = false; return true; };
    auto bad_verify = [](const fs::path &, const std::string &) { return false; };
    auto preflight = [&](const uint8_t *, bool partial) { assert(partial == !fs::exists(path)); return true; };
    auto sync = [&](const std::string & directory) { synced = server_native_rotation::sync_directory(directory); return synced; };
    auto fail_sync = [](const std::string &) { return false; };
    assert(server_native_rotation::retire(options[0], dir.string(), store, bad_verify, accept_remote, ok_remote, sync, removed, error) ==
           server_native_rotation::retirement_result::skipped);
    assert(fs::exists(path) && remote_calls == 0);
    auto reject_remote = [](const uint8_t *, bool) { return false; };
    assert(server_native_rotation::retire(options[0], dir.string(), store, verify, reject_remote, ok_remote, sync, removed, error) ==
           server_native_rotation::retirement_result::skipped);
    assert(fs::exists(path) && remote_calls == 0);
    { std::ofstream sidecar(path.string() + ".meta"); sidecar << "protected"; }
    assert(server_native_rotation::retire(options[0], dir.string(), store, verify, accept_remote, ok_remote, sync, removed, error) ==
           server_native_rotation::retirement_result::skipped);
    fs::remove(path.string() + ".meta");
    {
        server_route_state_lease live(path.string(), server_route_state_lock_mode::reservation);
        assert(live.acquired());
        assert(server_native_rotation::retire(options[0], dir.string(), store, verify, accept_remote, ok_remote, sync, removed, error) ==
               server_native_rotation::retirement_result::skipped);
    }
    const auto alias = dir / "alias.bin";
    std::error_code hardlink_error;
    fs::create_hard_link(path, alias, hardlink_error);
    if (!hardlink_error) {
        assert(server_native_rotation::retire(options[0], dir.string(), store, verify, accept_remote, ok_remote, sync, removed, error) ==
               server_native_rotation::retirement_result::skipped);
        fs::remove(alias);
    }
    assert(fs::exists(path) && remote_calls == 0);
    const auto failed = failure == 2
        ? server_native_rotation::retire(options[0], dir.string(), store, verify, preflight, ok_remote, fail_sync, removed, error)
        : server_native_rotation::retire(options[0], dir.string(), store, verify, preflight, fail_remote, sync, removed, error);
    assert(failed == server_native_rotation::retirement_result::failed);
    assert(removed == 1 && !fs::exists(path) && remote_calls == (failure == 2 ? 0u : 1u));
    assert(remote_exists == (failure != 1));
    loaded = refs::load(refs::journal_path(dir.string()));
    assert(loaded.ok && loaded.records.size() == 1 && loaded.records[0].st == refs::state::committed);
    options = server_native_rotation::candidates(loaded.records, kKeyA);
    assert(options.size() == 1);
    synced = false;
    assert(server_native_rotation::retire(options[0], dir.string(), store, verify, preflight, ok_remote, sync, removed, error) ==
           server_native_rotation::retirement_result::retired);
    assert(removed == 0 && remote_calls == (failure == 2 ? 1u : 2u) && !remote_exists);
    loaded = refs::load(refs::journal_path(dir.string()));
    assert(loaded.ok && loaded.records.size() == 1 && loaded.records[0].st == refs::state::retired);
    options = server_native_rotation::candidates(loaded.records, kKeyA);
    assert(options.size() == 1 && options[0].retirement.refusal == refs::retire_refusal::already_retired);
    remote_exists = true; // a retired Xbox object reappeared after an interrupted cleanup
    synced = false;
    assert(server_native_rotation::retire(options[0], dir.string(), store, verify, preflight, ok_remote, sync, removed, error) ==
           server_native_rotation::retirement_result::retired);
    assert(!remote_exists && removed == 0);
    assert(refs::load(refs::journal_path(dir.string())).records.size() == 1);
    }
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
    test_retired_fold_and_decision();
    test_plan_retire();
    test_retired_journal(root);
    test_migration();
    test_migration_writes(root);
    test_audit();
    test_naming();
    test_rotation_plan();
    test_rotation_pending_orphan(root);
    test_rotation_mixed_pending(root);
    test_rotation_multiple_references(root);
    test_rotation_retirement(root);

    fs::remove_all(root, ec);
    assert(!ec);
    std::printf("native snapshot reference journal CPU regression: ok\n");
    return 0;
}
