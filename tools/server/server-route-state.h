#pragma once

#include "common.h"
#include "server-model-identity.h"

#include <functional>
#include <memory>
#include <vector>

enum class server_route_state_store_lock_mode {
    shared,
    exclusive,
    // Nonblocking exclusive acquisition for an eviction attempt; a live
    // reader causes a safe skip rather than waiting in the server loop.
    exclusive_try,
};

enum class server_route_state_lock_mode {
    // Shared reference reservation used by the router before a child publishes
    // the pathname. It blocks eviction through one of the bounded reference
    // stripes but deliberately does not block the store publisher.
    reservation,
    // Shared reference lock. Eviction takes the matching reference lock
    // exclusively, while publication may atomically replace the pathname. It
    // also shares the publication lock so inspect+stream restore is coherent.
    reference,
    // Exclusive stable publication lock (the store lock is the publication
    // authority; the mode remains part of the compatibility API).
    publication,
    // Exclusive reference lock, used immediately before an eviction/unlink
    // while the caller already holds the exclusive store lock.
    eviction,
};

class server_route_state_store_lock {
public:
    explicit server_route_state_store_lock(const std::string & directory,
            server_route_state_store_lock_mode mode = server_route_state_store_lock_mode::exclusive);
    ~server_route_state_store_lock();

    server_route_state_store_lock(const server_route_state_store_lock &) = delete;
    server_route_state_store_lock & operator=(const server_route_state_store_lock &) = delete;

    bool acquired() const {
#ifndef _WIN32
        return fd >= 0;
#else
        return handle != nullptr;
#endif
    }
    void release();

private:
    std::string path;
#ifndef _WIN32
    int fd = -1;
#else
    void * handle = nullptr;
#endif
};

class server_route_state_lease;

// The unified persistent KV snapshot is text-only, target-only and uses the
// native sequence format. One file contains that native prefix followed by a
// small checksummed manifest. Router handoff and automatic disk-cache entries
// intentionally use this same envelope and reader/writer.
struct server_route_state_file {
    std::string state_path;
    size_t file_bytes = 0;
    size_t state_bytes = 0;
    uint64_t state_checksum = 0;
    uint64_t logits_offset = 0;
    uint64_t logits_bytes = 0;
    uint64_t logits_checksum = 0;
    uint32_t logits_vocab = 0;
    uint32_t logits_tokens = 0;
    uint32_t source_ctx = 0;
    // 0 means the canonical index block (used by route snapshots that predate
    // the field). Auto-cache snapshots record their configured block here so
    // profiles with different block settings can share one index safely.
    uint32_t index_block = 0;
    llama_pos position = -1;
    llama_tokens tokens;
    std::string model;
    std::string layout;
};

size_t server_route_state_save(llama_context * ctx, llama_seq_id seq,
        const llama_tokens & tokens, const server_model_identity & identity,
        const std::string & path, uint64_t max_bytes, uint32_t index_block = 0,
        const std::vector<float> * logits = nullptr,
        const std::function<bool(uint64_t)> & before_publish = {},
        const std::function<bool()> & after_publish = {},
        server_route_state_store_lock * store_lock = nullptr,
        server_route_state_lease * publication_lock = nullptr);

// Bounded metadata/token read, no context mutation. With `verify_payload=true`
// (the default) it also streams the native checksum through an 8 MiB buffer.
// Index callers pass false to avoid hashing every stored payload; the single
// selected candidate is then verified by llama_state_seq_load_file_streaming
// before state commit while its reference lease is held.
server_route_state_file server_route_state_read(const std::string & path,
        uint64_t max_bytes, uint32_t max_tokens,
        bool verify_payload = true);

// Read the optional logits payload bound to the exact canonical snapshot.
// This holds the same shared reference lock and never accepts a mismatched
// vocab/token count or checksum.
bool server_route_state_read_logits(const server_route_state_file & file,
        uint32_t expected_vocab, std::vector<float> & logits);

// Convert an old native-only auto-cache file into the canonical envelope by
// copying it in bounded chunks and atomically replacing the filename. The
// caller must have validated the legacy token/fingerprint sidecar first.
bool server_route_state_adopt_legacy(const std::string & path,
        const std::string & model, const std::string & layout,
        const llama_tokens & tokens, uint32_t source_ctx,
        uint32_t index_block, uint64_t max_bytes);

// Wrap a raw native sequence-state file (for example a freshly converted
// snapshot) into the canonical envelope at `destination`, copying it in
// bounded chunks and publishing atomically. `provenance`, when given, is
// recorded verbatim under the manifest's "converted" member so a converted
// snapshot is never confused with a native one. The native source is left
// untouched on any failure.
bool server_route_state_adopt_native(const std::string & native_path,
        const std::string & destination,
        const std::string & model, const std::string & layout,
        const llama_tokens & tokens, uint32_t source_ctx,
        uint32_t index_block, uint64_t max_bytes,
        const std::string * provenance_json);

// Remove a snapshot only after acquiring the exclusive store and reference
// locks. Lock-identity stripes are retained and are never removed.
bool server_route_state_remove_if_unreferenced(const std::string & path,
        server_route_state_store_lock * store_lock = nullptr);

// Pins one published snapshot with a kernel lock. The lock identity files are
// never unlinked: after SIGKILL the kernel releases the descriptor lock and a
// later operation can acquire it, so there is no stale pin marker. A failed
// acquisition is reported to the caller so it can take the safe miss/skip path.
class server_route_state_lease {
public:
    explicit server_route_state_lease(const std::string & state_path,
            server_route_state_lock_mode mode = server_route_state_lock_mode::reference);
    ~server_route_state_lease();

    server_route_state_lease(const server_route_state_lease &) = delete;
    server_route_state_lease & operator=(const server_route_state_lease &) = delete;

    bool acquired() const {
#ifndef _WIN32
        return mode == server_route_state_lock_mode::publication
            ? shared_store && shared_store->acquired()
            : mode == server_route_state_lock_mode::reference
                ? shared_store && shared_store->acquired() && ref_fd >= 0
                : ref_fd >= 0;
#else
        return mode == server_route_state_lock_mode::publication
            ? shared_store && shared_store->acquired()
            : mode == server_route_state_lock_mode::reference
                ? shared_store && shared_store->acquired() && ref_handle != nullptr
                : ref_handle != nullptr;
#endif
    }
    void release();

private:
    std::string path;
    server_route_state_lock_mode mode = server_route_state_lock_mode::reference;
    std::unique_ptr<server_route_state_store_lock> shared_store;
    std::string lock_path;
    std::string ref_path;
#ifndef _WIN32
    int lock_fd = -1;
    int ref_fd = -1;
#else
    void * lock_handle = nullptr;
    void * ref_handle = nullptr;
#endif
};
