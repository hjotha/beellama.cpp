// Explicit operator entry points for the durable native snapshot reference
// journal. Nothing here runs automatically at server startup:
//
//   llama-native-refs migrate --store <slot-save-path> [--keys <file>] [--dry-run]
//   llama-native-refs audit   --store <slot-save-path> [--json]
//   llama-native-refs retire  --store <slot-save-path> --key <64hex>
//                             [--host-file <path>] [--force-pinned] [--execute]
//
// `migrate` records the known pinned Xbox keys (baseline list, or one key per
// line from --keys) as `pinned`/`migrated`; re-running it is idempotent and a
// retired key is never re-pinned. `audit` is read-only: it lists
// committed/pinned references, retired keys, pending-only keys, host files
// whose key is absent from the journal, and unreadable host files.
//
// `retire` is dry-run unless --execute is passed. It never contacts the Xbox:
// the WDP object delete is a separate operator step (tools/xbox/scripts/
// native-retire.py). With --execute it removes the journal-named host files
// after verifying each file's key, then appends one `retired` record. The
// order is deliberate: unlink first, journal append second. A crash between
// the two leaves the previous committed/pinned record and a missing host file;
// rerunning `retire` converges because the missing file is skipped and the
// retired record is appended. The append runs under the exclusive route-state
// store lock, so the server cannot publish a competing record between the
// reload and the append.

#include "server-native-refs.h"
#include "server-route-state.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

namespace {

namespace fs    = std::filesystem;
namespace refs  = server_native_refs;

void usage(FILE * out) {
    std::fprintf(out,
        "usage:\n"
        "  llama-native-refs migrate --store <slot-save-path> [--keys <file>] [--dry-run]\n"
        "  llama-native-refs audit   --store <slot-save-path> [--json]\n"
        "  llama-native-refs retire  --store <slot-save-path> --key <64hex>\n"
        "                            [--host-file <path>] [--force-pinned] [--execute]\n"
        "\n"
        "migrate records pinned/migrated Xbox keys (baseline list when --keys is\n"
        "omitted) and never runs automatically. audit is read-only and never\n"
        "deletes host or remote state. retire is dry-run unless --execute is\n"
        "passed; it never contacts the Xbox and removes only the host file(s)\n"
        "named for that key before appending the retired record.\n");
}

std::string path_basename(const std::string & path) {
    return fs::path(path).filename().string();
}

bool collect_host_keys(const std::string & store,
        std::vector<refs::host_key> & keyed, std::vector<std::string> & unreadable) {
    std::error_code ec;
    std::vector<std::string> hosts;
    for (fs::directory_iterator it(store, ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code fec;
        if (!it->is_regular_file(fec) || fec) continue;
        const std::string name = it->path().filename().string();
        if (name.size() < 4 || name.compare(name.size() - 4, 4, ".bin") != 0) continue;
        hosts.push_back(name);
    }
    if (ec) {
        std::fprintf(stderr, "error: cannot enumerate %s: %s\n", store.c_str(), ec.message().c_str());
        return false;
    }
    std::sort(hosts.begin(), hosts.end());
    for (const auto & name : hosts) {
        const std::string path = (fs::path(store) / name).string();
        refs::host_key entry{name, {}};
        try {
            const auto file = server_route_state_read(path, UINT64_MAX, 1u << 20, false, nullptr);
            if (!file.has_native_remote_marker || !file.native_remote || file.state_bytes == 0) {
                unreadable.push_back(name);
            } else if (!refs::hash_host_prefix(path, file.state_bytes, entry.key, nullptr)) {
                entry.key.clear();
                unreadable.push_back(name);
            }
        } catch (const std::exception &) {
            unreadable.push_back(name);
        }
        keyed.push_back(std::move(entry));
    }
    return true;
}

int command_audit(const std::string & store, bool json) {
    const refs::load_result journal = refs::load(refs::journal_path(store));
    if (!journal.ok) {
        std::fprintf(stderr, "error: %s\n", journal.error.c_str());
        return 3;
    }
    std::vector<refs::host_key> keyed;
    std::vector<std::string> unreadable;
    if (!collect_host_keys(store, keyed, unreadable)) {
        return 3;
    }
    const refs::audit_report report = refs::audit(journal.records, keyed);

    if (json) {
        const auto print_list = [](const std::vector<std::string> & values) {
            std::printf("[");
            for (size_t i = 0; i < values.size(); ++i) {
                std::printf("%s\"%s\"", i ? "," : "", values[i].c_str());
            }
            std::printf("]");
        };
        std::printf("{\"store\":\"%s\",\"journal_records\":%zu,\"journal_folded\":%zu,"
                    "\"journal_truncated_tail\":%s,\"pinned\":",
                store.c_str(), journal.raw.size(), journal.records.size(),
                journal.truncated_tail ? "true" : "false");
        print_list(report.pinned);
        std::printf(",\"referenced\":");
        print_list(report.referenced);
        std::printf(",\"retired\":");
        print_list(report.retired);
        std::printf(",\"pending_only\":");
        print_list(report.pending_only);
        std::printf(",\"orphan_candidates\":");
        print_list(report.orphan_candidates);
        std::printf(",\"unregistered_host_files\":");
        print_list(report.unregistered_host_files);
        std::printf(",\"unreadable_host_files\":");
        print_list(unreadable);
        std::printf("}\n");
        return 0;
    }

    const auto print_list = [](const char * label, const std::vector<std::string> & values) {
        std::printf("%s (%zu):\n", label, values.size());
        for (const auto & value : values) {
            std::printf("  %s\n", value.c_str());
        }
    };
    std::printf("store: %s\n", store.c_str());
    std::printf("journal: %s\n", refs::journal_path(store).c_str());
    std::printf("records: %zu (%zu folded)%s\n", journal.raw.size(), journal.records.size(),
            journal.truncated_tail ? " [torn final append ignored]" : "");
    print_list("pinned references", report.pinned);
    print_list("committed references", report.referenced);
    print_list("retired keys (no remote reference)", report.retired);
    print_list("pending-only keys", report.pending_only);
    print_list("orphan candidates (no remote delete performed)", report.orphan_candidates);
    print_list("host files without a journal key", report.unregistered_host_files);
    print_list("unreadable host files", unreadable);
    return 0;
}

int command_migrate(const std::string & store, const std::string & keys_path, bool dry_run) {
    refs::load_result journal = refs::load(refs::journal_path(store));
    if (!journal.ok) {
        std::fprintf(stderr, "error: %s\n", journal.error.c_str());
        return 3;
    }
    std::vector<std::string> keys(refs::kBaselineKeys.begin(), refs::kBaselineKeys.end());
    if (!keys_path.empty()) {
        std::string error;
        std::vector<std::string> extra = refs::read_key_file(keys_path, &error);
        if (!error.empty()) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            return 2;
        }
        keys.insert(keys.end(), extra.begin(), extra.end());
    }
    const refs::migration plan = refs::plan_migration(keys, journal.records, refs::timestamp_now());
    if (!plan.invalid.empty()) {
        std::fprintf(stderr, "error: %zu invalid key(s) refused; no records written\n", plan.invalid.size());
        for (const auto & value : plan.invalid) {
            std::fprintf(stderr, "  %s\n", value.c_str());
        }
        return 2;
    }
    std::printf("migrate: %zu new pin(s), %zu already pinned, %zu retired\n", plan.pins.size(),
            plan.already_pinned.size(), plan.retired.size());
    for (const auto & pin : plan.pins) {
        std::printf("  pin %s\n", pin.key.c_str());
    }
    for (const auto & key : plan.retired) {
        std::printf("  retired, not re-pinned: %s\n", key.c_str());
    }
    if (dry_run || plan.pins.empty()) {
        return 0;
    }
    server_route_state_store_lock store_lock(store, server_route_state_store_lock_mode::exclusive);
    if (!store_lock.acquired()) {
        std::fprintf(stderr, "error: exclusive store lock is unavailable\n");
        return 3;
    }
    const auto lock_held = [&store_lock](const std::string & path) {
        return store_lock.owns_exclusive_for(path);
    };
    for (const auto & pin : plan.pins) {
        std::string error;
        if (!refs::append(store, pin, lock_held, &error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            return 3;
        }
    }
    return 0;
}

// True when `path` is a direct child of the store directory. The CLI must not
// delete an arbitrary file even when its content hash happens to match.
bool inside_store(const std::string & store, const std::string & path) {
    std::error_code ec;
    const fs::path store_path = fs::weakly_canonical(fs::u8path(store), ec);
    if (ec) return false;
    const fs::path host_path = fs::weakly_canonical(fs::u8path(path), ec);
    if (ec) return false;
    return host_path.parent_path() == store_path;
}

// Compute the snapshot key of a host file. A published host file is the
// canonical route-state envelope: the key covers [0, state_bytes) through the
// RKVK frame. A raw staged native frame (no manifest) is keyed over the whole
// file, which is exactly the staged prefix. The canonical marker is
// authoritative; a present-but-non-native envelope is refused rather than
// silently hashing the wrong range. The caller must pass the exclusive store
// lock it already holds, so the verification cannot block on its own lease.
bool host_file_key(const std::string & path, std::string & key, std::string & error,
        server_route_state_store_lock * store_lock) {
    std::error_code ec;
    const uint64_t size = fs::file_size(fs::u8path(path), ec);
    if (ec) {
        error = "host file size is unavailable";
        return false;
    }
    try {
        const auto file = server_route_state_read(path, UINT64_MAX, 1u << 20, false, store_lock);
        if (file.has_native_remote_marker) {
            if (!file.native_remote) {
                error = "host file is not a native remote snapshot";
                return false;
            }
            if (file.state_bytes == 0) {
                error = "host file has no native prefix";
                return false;
            }
            return refs::hash_host_prefix(path, file.state_bytes, key, &error);
        }
        error = "host file has no native remote marker";
        return false;
    } catch (const std::exception &) {
        // Not a canonical envelope: fall through to the raw frame.
    }
    return refs::hash_host_prefix(path, size, key, &error);
}

bool remove_host_file(const std::string & path, std::string & error) {
    std::error_code ec;
    const fs::file_status status = fs::symlink_status(fs::u8path(path), ec);
    if (ec) {
        error = "host file status is unavailable: " + ec.message();
        return false;
    }
    if (fs::is_symlink(status)) {
        error = "host file is a symbolic link; refusing to delete";
        return false;
    }
    if (!fs::is_regular_file(status)) {
        error = "host file is not a regular file; refusing to delete";
        return false;
    }
    if (!fs::remove(fs::u8path(path), ec) || ec) {
        error = "host file removal failed: " + (ec ? ec.message() : std::string("unknown error"));
        return false;
    }
    return true;
}

struct retirement {
    std::vector<std::string> paths;    // existing host files to remove
    std::vector<std::string> missing;  // journal-named but already absent
    std::string host_file;             // basename recorded in the retired entry
};

// Resolve, verify and (when executing) remove the host files named for the
// key. The dry-run path never reads the payload, so it cannot create a
// reference-lock sidecar in the store. Execution holds the exclusive store
// lock: every present file is verified before the first unlink, so a key
// mismatch leaves the whole store untouched.
bool prepare_host_files(const std::string & store, const std::string & explicit_file,
        const refs::retire_plan & plan, bool execute,
        server_route_state_store_lock * store_lock, retirement & out, std::string & error) {
    out = retirement{};
    std::vector<std::string> candidates;
    if (!explicit_file.empty()) {
        if (!inside_store(store, explicit_file)) {
            error = "--host-file must be a direct child of --store";
            return false;
        }
        const std::string name = path_basename(explicit_file);
        if (!refs::valid_host_file(name)) {
            error = "--host-file must name a plain basename";
            return false;
        }
        if (!plan.host_files.empty() &&
                std::find(plan.host_files.begin(), plan.host_files.end(), name) == plan.host_files.end()) {
            error = "--host-file does not match the host file named by the journal for this key";
            return false;
        }
        candidates.push_back(explicit_file);
        out.host_file = name;
    } else {
        for (const auto & name : plan.host_files) {
            candidates.push_back((fs::path(store) / name).string());
            out.host_file = name;
        }
    }
    for (const auto & path : candidates) {
        std::error_code ec;
        const fs::file_status status = fs::symlink_status(fs::u8path(path), ec);
        if (ec) {
            if (ec == std::errc::no_such_file_or_directory) {
                out.missing.push_back(path);
                continue;
            }
            error = "host file status is unavailable: " + path + ": " + ec.message();
            return false;
        }
        if (!fs::exists(status)) {
            out.missing.push_back(path);
            continue;
        }
        if (!fs::is_regular_file(status)) {
            error = "host file is not a regular file: " + path;
            return false;
        }
        if (!execute) {
            // Planning only: no payload read and no reference-lock sidecar.
            out.paths.push_back(path);
            continue;
        }
        std::string actual;
        std::string key_error;
        if (!host_file_key(path, actual, key_error, store_lock)) {
            error = path + ": " + key_error;
            return false;
        }
        if (actual != plan.entry.key) {
            error = path + ": host file key " + actual + " does not match " + plan.entry.key;
            return false;
        }
        out.paths.push_back(path);
    }
    if (execute) {
        for (const auto & path : out.paths) {
            if (!remove_host_file(path, error)) return false;
        }
    }
    return true;
}

int command_retire(const std::string & store, const std::string & key_arg,
        const std::string & host_file, bool force_pinned, bool execute) {
    refs::load_result journal = refs::load(refs::journal_path(store));
    if (!journal.ok) {
        std::fprintf(stderr, "error: %s\n", journal.error.c_str());
        return 3;
    }
    refs::retire_plan plan = refs::plan_retire(key_arg, journal.records, force_pinned,
            refs::timestamp_now());
    const auto refusal_message = [](const refs::retire_plan & value) {
        switch (value.refusal) {
            case refs::retire_refusal::invalid_key:
                return "--key must be 64 lowercase hex characters";
            case refs::retire_refusal::unknown_key:
                return "the key has no record in this journal; refusing to invent a retirement";
            case refs::retire_refusal::pending:
                return "a pending record exists for this key; finish or inspect the save first";
            case refs::retire_refusal::pinned:
                return "the key is pinned; pass --force-pinned to retire it explicitly";
            case refs::retire_refusal::already_retired:
                return "";
            case refs::retire_refusal::none:
                return "";
        }
        return "unknown refusal";
    };
    if (plan.refusal != refs::retire_refusal::none &&
            plan.refusal != refs::retire_refusal::already_retired) {
        std::fprintf(stderr, "error: %s\n", refusal_message(plan));
        return 2;
    }

    if (!execute) {
        std::string error;
        retirement preview;
        if (!prepare_host_files(store, host_file, plan, false, nullptr, preview, error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            return 2;
        }
        std::printf("retire (dry-run): %s\n", plan.entry.key.c_str());
        if (plan.refusal == refs::retire_refusal::already_retired) {
            std::printf("  journal state: already retired (no new record)\n");
        } else {
            std::printf("  journal state: eligible\n");
        }
        for (const auto & path : preview.paths) {
            std::printf("  would delete host file: %s\n", path.c_str());
        }
        for (const auto & path : preview.missing) {
            std::printf("  host file already absent: %s\n", path.c_str());
        }
        if (preview.host_file.empty()) {
            std::printf("  host file: none named by the journal\n");
        }
        if (plan.execute) {
            std::printf("  would append retired record%s%s\n",
                    preview.host_file.empty() ? "" : " for ",
                    preview.host_file.empty() ? "" : preview.host_file.c_str());
        }
        std::printf("  Xbox object is not touched; delete rkva-snapshots-v2/%s separately\n",
                plan.entry.key.c_str());
        if (plan.execute) {
            std::printf("re-run with --execute to apply\n");
        } else {
            std::printf("nothing to append; --execute would only remove a leftover host file\n");
        }
        return 0;
    }

    server_route_state_store_lock store_lock(store, server_route_state_store_lock_mode::exclusive);
    if (!store_lock.acquired()) {
        std::fprintf(stderr, "error: exclusive store lock is unavailable\n");
        return 3;
    }
    // Re-read under the lock: a concurrent save may have added a pending or
    // committed record between the first plan and now.
    journal = refs::load(refs::journal_path(store));
    if (!journal.ok) {
        std::fprintf(stderr, "error: %s\n", journal.error.c_str());
        return 3;
    }
    plan = refs::plan_retire(key_arg, journal.records, force_pinned, refs::timestamp_now());
    if (plan.refusal != refs::retire_refusal::none &&
            plan.refusal != refs::retire_refusal::already_retired) {
        std::fprintf(stderr, "error: %s\n", refusal_message(plan));
        return 2;
    }
    retirement removal;
    std::string error;
    if (!prepare_host_files(store, host_file, plan, true, &store_lock, removal, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 3;
    }
    for (const auto & path : removal.paths) {
        std::printf("deleted host file: %s\n", path.c_str());
    }
    for (const auto & path : removal.missing) {
        std::printf("host file already absent: %s\n", path.c_str());
    }
    if (!plan.execute) {
        std::printf("retire: %s is already retired; no record appended\n",
                refs::normalize_key(key_arg).c_str());
        return 0;
    }
    if (!removal.host_file.empty()) {
        // An explicit --host-file is the only way to name a host file for a
        // key whose journal has no committed host record (a pin).
        plan.entry.host_file = removal.host_file;
    }
    const auto lock_held = [&store_lock](const std::string & path) {
        return store_lock.owns_exclusive_for(path);
    };
    if (!refs::append(store, plan.entry, lock_held, &error)) {
        std::fprintf(stderr, "error: host file removal succeeded but the journal append failed: %s\n",
                error.c_str());
        std::fprintf(stderr, "error: rerun the same retire command to record the retirement\n");
        return 3;
    }
    std::printf("retire: appended retired record for %s\n", plan.entry.key.c_str());
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage(stderr);
        return 2;
    }
    const std::string command = argv[1];
    if (command == "-h" || command == "--help" || command == "help") {
        usage(stdout);
        return 0;
    }
    std::string store;
    std::string keys;
    std::string key;
    std::string host_file;
    bool dry_run = false;
    bool execute = false;
    bool force_pinned = false;
    bool json = false;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--store" && i + 1 < argc) {
            store = argv[++i];
        } else if (arg == "--keys" && i + 1 < argc) {
            keys = argv[++i];
        } else if (arg == "--key" && i + 1 < argc) {
            key = argv[++i];
        } else if (arg == "--host-file" && i + 1 < argc) {
            host_file = argv[++i];
        } else if (arg == "--dry-run") {
            dry_run = true;
        } else if (arg == "--execute") {
            execute = true;
        } else if (arg == "--force-pinned") {
            force_pinned = true;
        } else if (arg == "--json") {
            json = true;
        } else {
            std::fprintf(stderr, "error: unknown argument %s\n", arg.c_str());
            usage(stderr);
            return 2;
        }
    }
    if (store.empty()) {
        std::fprintf(stderr, "error: --store is required\n");
        usage(stderr);
        return 2;
    }
    if (command == "audit") {
        return command_audit(store, json);
    }
    if (command == "migrate") {
        return command_migrate(store, keys, dry_run);
    }
    if (command == "retire") {
        if (key.empty()) {
            std::fprintf(stderr, "error: retire requires --key\n");
            usage(stderr);
            return 2;
        }
        if (dry_run && execute) {
            std::fprintf(stderr, "error: --dry-run and --execute are mutually exclusive\n");
            return 2;
        }
        return command_retire(store, key, host_file, force_pinned, execute);
    }
    std::fprintf(stderr, "error: unknown command %s\n", command.c_str());
    usage(stderr);
    return 2;
}
