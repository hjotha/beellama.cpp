// Explicit operator entry points for the durable native snapshot reference
// journal. Nothing here runs automatically at server startup:
//
//   llama-native-refs migrate --store <slot-save-path> [--keys <file>] [--dry-run]
//   llama-native-refs audit   --store <slot-save-path> [--json]
//
// `migrate` records the known pinned Xbox keys (baseline list, or one key per
// line from --keys) as `pinned`/`migrated`; re-running it is idempotent.
// `audit` is read-only: it lists committed/pinned references, pending-only
// keys, host files whose key is absent from the journal, and unreadable host
// files. Neither command deletes any host file or remote object.

#include "server-native-refs.h"
#include "server-route-state.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

namespace {

namespace refs = server_native_refs;

void usage(FILE * out) {
    std::fprintf(out,
        "usage:\n"
        "  llama-native-refs migrate --store <slot-save-path> [--keys <file>] [--dry-run]\n"
        "  llama-native-refs audit   --store <slot-save-path> [--json]\n"
        "\n"
        "migrate records pinned/migrated Xbox keys (baseline list when --keys is\n"
        "omitted) and never runs automatically. audit is read-only and never\n"
        "deletes host or remote state.\n");
}

std::string path_basename(const std::string & path) {
    return std::filesystem::path(path).filename().string();
}

bool collect_host_keys(const std::string & store,
        std::vector<refs::host_key> & keyed, std::vector<std::string> & unreadable) {
    std::error_code ec;
    std::vector<std::string> hosts;
    for (std::filesystem::directory_iterator it(store, ec), end; !ec && it != end; it.increment(ec)) {
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
        const std::string path = (std::filesystem::path(store) / name).string();
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
    std::printf("migrate: %zu new pin(s), %zu already pinned\n", plan.pins.size(),
            plan.already_pinned.size());
    for (const auto & pin : plan.pins) {
        std::printf("  pin %s\n", pin.key.c_str());
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
    bool dry_run = false;
    bool json = false;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--store" && i + 1 < argc) {
            store = argv[++i];
        } else if (arg == "--keys" && i + 1 < argc) {
            keys = argv[++i];
        } else if (arg == "--dry-run") {
            dry_run = true;
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
    std::fprintf(stderr, "error: unknown command %s\n", command.c_str());
    usage(stderr);
    return 2;
}
