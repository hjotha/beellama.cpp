#pragma once
// Shared, testable core of the auto-restore DESTINATION TRANSACTION (plan 3.6:
// capture -> clear -> attempt -> rollback -> inspect -> fallback).
//
// Both production guards delegate here so a test exercises the SAME code the
// server runs:
//   tools/server/server-context.cpp  (slot/prompt metadata stays server-side)
//   tools/server/server-task.cpp
//
// Invariants:
//  * residency is decided by POSITIONS, never by "byte size == 0";
//  * capture is FAIL-SAFE BEFORE ANY CLEAR: a missing/unusable disk spool or an
//    uncapturable resident state must refuse mutation, never proceed and lose
//    state that cannot be put back;
//  * the empty-context precondition of llama_state_seq_load_file_streaming is
//    NOT weakened here -- the caller clears first, then attempts the load, then
//    rolls back;
//  * rollback only restores bytes and verifies the position watermark; prompt,
//    checkpoint and speculative-carry metadata are owned by the caller.

#include "llama.h"
#include "llama-ext.h"

#include <sys/stat.h>
#include <sys/statfs.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace llama_state_rollback {

enum class rollback_result { not_attempted, restored, lost };

// The mixed-state reader refuses a tmpfs/ramfs spool, so the spool must be a
// real, writable, disk-backed directory BEFORE any clear. This is the same
// preflight the server resolves (server_mixed_state_spool_dir delegates here),
// so the transaction cannot mutate on a spool that could never restore.
inline bool spool_usable(const std::string & dir, std::string * error) {
    if (dir.empty()) {
        if (error) { *error = "no disk-backed mixed-state spool directory"; }
        return false;
    }
    std::error_code ec;
    const std::filesystem::path p(dir);
    if (!std::filesystem::is_directory(p, ec) || ec) {
        if (error) { *error = "mixed-state spool path is not a directory"; }
        return false;
    }
#if defined(__linux__)
    struct statfs fs_info = {};
    if (::statfs(p.c_str(), &fs_info) != 0) {
        if (error) { *error = "mixed-state spool statfs failed"; }
        return false;
    }
    constexpr long TMPFS_MAGIC_VALUE = 0x01021994;
    constexpr long RAMFS_MAGIC_VALUE = 0x858458f6;
    if (long(fs_info.f_type) == TMPFS_MAGIC_VALUE || long(fs_info.f_type) == RAMFS_MAGIC_VALUE) {
        if (error) { *error = "mixed-state spool directory is tmpfs/ramfs"; }
        return false;
    }
#endif
    const auto spool = p / "beellama-mixed-state";
    std::filesystem::create_directories(spool, ec);
    if (ec || !std::filesystem::is_directory(spool, ec)) {
        if (error) { *error = "cannot create <spool>/beellama-mixed-state"; }
        return false;
    }
    const auto probe = spool / ".f4-spool-write-probe";
    {
        std::ofstream f(probe, std::ios::trunc);
        if (!f.is_open()) {
            if (error) { *error = "mixed-state spool directory is not writable"; }
            return false;
        }
        f << "f4";
        if (!f.good()) {
            if (error) { *error = "mixed-state spool directory is not writable"; }
            return false;
        }
    }
    std::filesystem::remove(probe, ec);
    return true;
}

// Resident iff the context reports a position range for this sequence.
inline bool resident_positions(llama_context * ctx, llama_seq_id seq, llama_pos * out_pmax) {
    if (out_pmax) { *out_pmax = -1; }
    if (!ctx) { return false; }
    const llama_pos pmax = llama_memory_seq_pos_max(llama_get_memory(ctx), seq);
    if (out_pmax) { *out_pmax = pmax; }
    return pmax >= 0;
}

// Capture the resident sequence state. Returns false with a reason when there is
// nothing to capture (resident == false) or when the state cannot be serialized
// (caller must then refuse to mutate). Nothing is cleared here.
inline bool capture_resident(llama_context * ctx, llama_seq_id seq,
                             std::vector<uint8_t> & out, llama_pos * out_pmax,
                             bool * resident, std::string * error) {
    out.clear();
    if (resident) { *resident = false; }
    llama_pos pmax = -1;
    const bool has_pos = resident_positions(ctx, seq, &pmax);
    if (resident) { *resident = has_pos; }
    if (!has_pos) {
        return true; // legitimately empty: nothing to capture
    }
    const size_t n = llama_state_seq_get_size_ext(ctx, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (!n) {
        if (error) {
            *error = "resident positions but no serializable state; refusing before mutation";
        }
        return false;
    }
    out.resize(n);
    if (llama_state_seq_get_data_ext(ctx, out.data(), n, seq, LLAMA_STATE_SEQ_FLAGS_NONE) != n) {
        out.clear();
        if (error) {
            *error = "resident state could not be captured; refusing before mutation";
        }
        return false;
    }
    if (out_pmax) { *out_pmax = pmax; }
    return true;
}

// Undo clear + a failed load for ONE sequence: write the captured bytes back
// and verify the position watermark, so a partial write from a refused restore
// cannot survive. Pure state only -- the caller owns prompt/checkpoint metadata.
inline rollback_result rollback_resident(llama_context * ctx, llama_seq_id seq,
                                         const std::vector<uint8_t> & blob,
                                         llama_pos expected_pos) {
    if (!ctx || blob.empty()) {
        return rollback_result::lost;
    }
    if (llama_state_seq_set_data_ext(ctx, blob.data(), blob.size(), seq,
                                     LLAMA_STATE_SEQ_FLAGS_NONE) != blob.size()) {
        return rollback_result::lost;
    }
    if (llama_memory_seq_pos_max(llama_get_memory(ctx), seq) != expected_pos) {
        return rollback_result::lost;
    }
    return rollback_result::restored;
}

} // namespace llama_state_rollback
