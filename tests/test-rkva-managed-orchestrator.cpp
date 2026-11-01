// Phase 2 synthetic lifecycle test for the managed registry + host journal +
// orchestrator state machine. No server, RPC, GPU, Xbox or power-loss test.
//
// Contract under test: whole-file SHA-256 keys, flat <tx>.managed host files,
// peer <key>.blob objects and the journalled pending -> published -> commit
// ack -> release intent/ack transitions. This is the ONE lifecycle contract
// chosen for Phase 2; object v3 manifest directories are deliberately not used.

#include "../tools/server/server-rkva-managed-orchestrator.h"
#include "../ggml/include/rkva-snapshot-digest.h"
#undef NDEBUG
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
namespace hj = rkva_host_journal;
namespace mr = rkva_managed_registry;
namespace mo = rkva_managed_orchestrator;

static std::string contents(const fs::path & path) {
    std::ifstream input(path, std::ios::binary);
    assert(input.good());
    return std::string(std::istreambuf_iterator<char>(input), {});
}

static void write_file(const fs::path & path, const std::string & data) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(data.data(), std::streamsize(data.size()));
    assert(output.good());
}

static std::string hex_digest(const std::string & data) {
    const auto digest = rkva_snapshot::digest(data.data(), data.size());
    return rkva_snapshot::hex(digest.data());
}

static std::string payload(char seed, size_t bytes) {
    std::string data(bytes, '\0');
    for (size_t i = 0; i < bytes; ++i) {
        data[i] = char((uint32_t(uint8_t(seed)) * 131u + uint32_t(i) * 17u) & 0xffu);
    }
    return data;
}

static const hj::transaction * find_host(const hj::snapshot & snap, const std::string & tx) {
    for (const auto & record : snap.transactions) {
        if (record.tx == tx) return &record;
    }
    return nullptr;
}

static const mr::transaction * find_peer(const mr::snapshot & snap, const std::string & tx) {
    for (const auto & record : snap.transactions) {
        if (record.id == tx) return &record;
    }
    return nullptr;
}

int main() {
    char tmp[] = "/tmp/rkva-phase2-orchestrator-XXXXXX";
    char * root = mkdtemp(tmp);
    assert(root != nullptr);
    const fs::path base(root);
    const fs::path host = base / "host";
    const fs::path journal_dir = base / "journal";
    const fs::path peer_dir = base / "peer";
    const fs::path objects = base / "objects";
    const fs::path legacy = base / "rkva-snapshots-v2";
    for (const auto & dir : {host, journal_dir, peer_dir, objects, legacy}) {
        assert(fs::create_directory(dir));
    }
    write_file(legacy / "sentinel", "never touch");

    const std::string owner(64, 'a');
    hj::journal journal(journal_dir.string(), owner);
    mr::registry peer(peer_dir.string(), owner);
    std::string error;
    assert(journal.initialize(error));
    assert(peer.initialize(error));

    mo::orchestrator flow(journal, peer, host.string(), objects.string(), 8ull << 30);

    const std::string tx(64, '1');
    const std::string ref(64, '3');
    const std::string data = payload('A', 4096);
    const std::string key = hex_digest(data);
    const fs::path source = base / "source-1.bin";
    write_file(source, data);
    write_file(objects / (key + ".blob"), data);

    assert(flow.publish(tx, ref, key, source.string(), error));
    assert(contents(host / (tx + ".managed")) == data);

    hj::snapshot host_snapshot;
    mr::snapshot peer_snapshot;
    assert(journal.inspect(host_snapshot, error));
    assert(peer.inspect(peer_snapshot, error));
    const auto * host_record = find_host(host_snapshot, tx);
    const auto * peer_record = find_peer(peer_snapshot, tx);
    assert(host_record && peer_record);
    assert(host_record->state == hj::phase::commit_ack);
    assert(host_record->peer_commit_version == 2);
    assert(host_record->host_path == host.string() + "/" + tx + ".managed");
    assert(host_record->host_sha256 == key);
    assert(peer_record->state == mr::phase::committed);
    assert(peer_record->committed_version == 2);

    // An idempotent replay must not need the source again.
    assert(flow.publish(tx, ref, key, std::string(), error));

    // Missing peer object fails closed after the durable host prepare but
    // before any host staging, peer prepare or commit.
    const std::string tx_missing(64, '2');
    const std::string ref_missing(64, '4');
    const std::string data_missing = payload('B', 3072);
    const std::string key_missing = hex_digest(data_missing);
    const fs::path source_missing = base / "source-2.bin";
    write_file(source_missing, data_missing);
    assert(!flow.publish(tx_missing, ref_missing, key_missing, source_missing.string(), error));
    assert(!error.empty());
    assert(!fs::exists(host / (tx_missing + ".managed")));
    assert(journal.inspect(host_snapshot, error));
    const auto * missing_record = find_host(host_snapshot, tx_missing);
    assert(missing_record && missing_record->state == hj::phase::prepared);
    assert(peer.inspect(peer_snapshot, error));
    assert(find_peer(peer_snapshot, tx_missing) == nullptr);

    // A source whose whole-file digest differs from the requested key is
    // rejected before the host stage is created; the peer pending record may
    // exist and remains pinned.
    const std::string tx_bad(64, '5');
    const std::string ref_bad(64, '6');
    const std::string expected_data = payload('C', 2048);
    const std::string expected_key = hex_digest(expected_data);
    const std::string other_data = payload('D', 2048);
    write_file(objects / (expected_key + ".blob"), expected_data);
    const fs::path bad_source = base / "source-3.bin";
    write_file(bad_source, other_data);
    assert(!flow.publish(tx_bad, ref_bad, expected_key, bad_source.string(), error));
    assert(error.find("source digest differs from key") != std::string::npos);
    assert(!fs::exists(host / (tx_bad + ".managed")));
    assert(journal.inspect(host_snapshot, error));
    assert(find_host(host_snapshot, tx_bad)->state == hj::phase::prepared);
    assert(peer.inspect(peer_snapshot, error));
    assert(find_peer(peer_snapshot, tx_bad)->state == mr::phase::pending);

    // Unknown transactions cannot be released.
    assert(!flow.release(std::string(64, '9'), error));
    assert(error.find("unknown journal transaction") != std::string::npos);

    // Release removes the managed host file only after durable tombstone
    // evidence, then releases the peer reference.
    assert(flow.release(tx, error));
    assert(!fs::exists(host / (tx + ".managed")));
    assert(fs::exists(host / ("." + tx + ".release")));
    assert(journal.inspect(host_snapshot, error));
    assert(find_host(host_snapshot, tx)->state == hj::phase::release_ack);
    assert(peer.inspect(peer_snapshot, error));
    assert(find_peer(peer_snapshot, tx)->state == mr::phase::released);
    assert(flow.release(tx, error));

    // A released transaction cannot publish again; the tombstone forbids it.
    assert(!flow.publish(tx, ref, key, std::string(), error));
    assert(!fs::exists(host / (tx + ".managed")));

    // Pending retries remain intact and are never auto-aborted or deleted.
    assert(journal.inspect(host_snapshot, error));
    assert(find_host(host_snapshot, tx_missing)->state == hj::phase::prepared);
    assert(find_host(host_snapshot, tx_bad)->state == hj::phase::prepared);

    assert(contents(legacy / "sentinel") == "never touch");
    fs::remove_all(base);
    std::cout << "managed registry/journal/orchestrator lifecycle: PASS\n";
    return 0;
}
