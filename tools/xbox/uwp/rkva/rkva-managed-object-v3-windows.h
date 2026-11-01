#pragma once

#include "rkva-managed-object-v3.h"

#include <memory>

// Test-only Windows/UWP durable-filesystem skeleton for the v3 object writer.
//
// Guard: RKVA_MANAGED_V3_WINDOWS_EXPERIMENTAL_TEST. That macro is defined only
// for tests/test-rkva-managed-object-v3-platform.cpp. Without it every method
// fails closed and no Win32 API is called. With it, the Win32 body is still
// compiled only under _WIN32; a Linux build compiles the fail-closed stub.
//
// qualification() is false in EVERY build, including the experimental one.
// The skeleton can flush a file, flush a directory handle and perform a
// no-replace MoveFileExW, but no test in this repository has established that
// those calls preserve the directory entry across power loss on UWP/Xbox.
// rename_noreplace_and_sync() therefore returns `uncertain` after an observed
// move and parent flush, never `durable`. This is deliberate: it cannot enable
// publication, because the production writer already refuses before calling a
// backend and durable_filesystem::qualified() is the next gate.

namespace rkva_managed_object_v3 {

class windows_filesystem final : public durable_filesystem {
public:
    // Keeps the concrete shared lease alive for the backend lifetime and binds
    // the base class weak authority to the same control block.
    windows_filesystem(std::shared_ptr<rkva_managed_store::lease> store, fs::path root);
    ~windows_filesystem() override;
    windows_filesystem(const windows_filesystem &) = delete;
    windows_filesystem & operator=(const windows_filesystem &) = delete;

    // Opens the already-provisioned absolute v3 root without following a
    // reparse point. It never creates the root or a lock file.
    bool acquire(const fs::path & root, std::string & error);
    bool lock_held() const;

    // Unconditionally false: a successful FlushFileBuffers/MoveFileExW is not a
    // power-loss qualification.
    bool qualified() const override;

    bool sync_file(const fs::path &, std::string &) override;
    bool sync_directory(const fs::path &, std::string &) override;
    // A refused (existing destination) move is no_change. A move that was
    // observed to apply and whose parent handle flushed is still uncertain.
    move_result rename_noreplace_and_sync(const fs::path &, const fs::path &, std::string &) override;

private:
    struct state;
    std::unique_ptr<state> state_;
};

#ifdef RKVA_MANAGED_V3_WINDOWS_EXPERIMENTAL_TEST
// Desktop-only process-death injection hook. It is defined solely for the
// experimental test target; normal builds never see the symbol.
extern void (*windows_test_hook)(const char * phase);
#endif

} // namespace rkva_managed_object_v3
