#pragma once

// Bounded streaming encoder and file-backing helpers for the codec wire
// format (see llama-kv-mixed-state.h, versions 1/2/3). Produces byte-identical
// output to snapshot::serialize() without holding whole payloads in memory:
// payload bytes are read on demand from a per-layer producer through a small
// fixed transfer buffer (default 1 MiB, at most 8 MiB), and written straight
// to a sink callback. No full payload vector and no full F32 prefix are ever
// assembled.
//
// Why two passes: the wire format stores each payload's XXH64 checksum in
// its descriptor, which precedes the payload bytes. The encoder therefore
// runs a bounded checksum pass over every payload first, then a write pass
// that re-reads the payloads. Producer mutation between the passes is
// detected: pass 2 recomputes every payload checksum and compares it with
// pass 1 before the tail_crc is emitted; any mismatch fails the encode.
//
// Failure semantics: on any producer failure, sink failure, mutation or
// validation error the encode fails and the output is NOT a valid snapshot.
// The sink is never rolled back (this module does not claim output-sink
// rollback); the caller publishes (e.g. temp-file + rename) only on success.
//
// Adapter notes for the llama I/O interfaces (src/llama-io.h,
// src/llama-io-file.h):
//   - sink adapter: write_bytes -> llama_io_write_i::write(dst, count).
//     The write order is strictly sequential; no seek is needed. The file
//     stream's staged read_tensor budget is 8 MiB
//     (LLAMA_STATE_FILE_BUFFER_SIZE), matching this module's transfer-buffer
//     ceiling.
//   - producer adapter: read_payload must serve the SAME bytes on repeated
//     reads (pass 1 + pass 2) at absolute offsets. llama_io_read_i itself is
//     a sequential one-shot stream WITHOUT rewind, so a direct adapter must
//     not assume seekability: use a memory mapping of the source file, a
//     re-readable file handle, or a deterministic re-generator. A later
//     source may be a ggml tensor; this module does not depend on ggml and
//     does not modify llama-io.
//
// This module is portable C++17; the mmap-backed file owner below is an
// explicitly Linux-guarded convenience helper. The codec itself (parse /
// parse_view) remains portable.

#include "llama-kv-mixed-state.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace llama_kv_mixed {

// per-layer payload producer: serves payload bytes at absolute offsets.
// Must return exactly `count` bytes for every valid (offset, count) within
// the layer's declared payload_bytes; a short return is a producer failure.
// Must be re-readable: the same bytes must be served again on the write pass
// (pass 2). A producer that serves different bytes on the second pass is
// detected by the checksum comparison and fails the encode.
struct stream_read_callbacks {
    // F4 v2: layer_id alone is no longer unique. The range_id is passed
    // alongside it so a producer can serve the two ranges of one full layer
    // from their own storages. v1 frames always pass range_id 0.
    std::function<size_t(uint32_t layer_id, uint32_t range_id, uint64_t offset, uint8_t * dst, size_t count)> read_payload;
};

// output sink. All snapshot bytes are delivered strictly sequentially;
// returning false fails the encode. The sink is NOT rolled back on failure.
struct stream_write_callbacks {
    std::function<bool(const uint8_t * bytes, size_t count)> write_bytes;
};

// ---------------------------------------------------------------------------
// Section map (F4 import forensics).
//
// One contiguous region of a serialized frame, reported with the schema the
// plan demands: {start, end, kind, layer, field, range/stream, count,
// stride, checksum}. Offsets are absolute inside the FRAME (first byte of the
// KMS header is 0); a caller holding a whole sequence blob must skip its own
// envelope first (llama_mixed_state_section_map).
//
// Produced by walking the REAL bytes (stream_describe) and, independently,
// by the writer as it emits them (stream_options::on_section). Comparing the
// two is what proves the remembered write order matches the schema.
struct stream_section {
    uint64_t start     = 0;         // inclusive
    uint64_t end       = 0;         // exclusive
    const char * kind  = "invalid"; // header|metadata|descriptor|payload|cells|tail
    const char * field = "";        // header|manifest|descriptor|payload_kv|cell_entry|tail_crc
    uint32_t layer_id  = UINT32_MAX;// valid for descriptor/payload
    uint32_t range_id  = UINT32_MAX;// 0 = local kvarn, 1 = overflow standard qx
    uint64_t count     = 0;         // payload_cells (payload) / cells (cells)
    uint64_t k_stride  = 0;         // descriptor row/record bytes
    uint64_t v_stride  = 0;
    uint64_t checksum  = 0;         // payload XXH64 for payloads, else 0
};

struct stream_options {
    std::function<void(const std::string & message)> on_fault; // invoked before throwing
    // writer-side section map: invoked once per emitted region, in order,
    // with the absolute frame offset already advanced. Diagnostic only;
    // never affects bytes or validation.
    std::function<void(const stream_section & section)> on_section;
    size_t transfer_buffer_bytes = 1 << 20; // 1 MiB; clamped range 1..8 MiB
    // caller caps, enforced before reading/writing large regions:
    uint64_t max_total_bytes = 0;    // total output; 0 = 4 TiB internal sanity bound
    uint64_t max_payload_bytes = 0;  // per-layer payload; 0 = no cap beyond the sanity bound
    uint64_t max_metadata_bytes = 0; // metadata blob; 0 = no cap beyond the sanity bound
};

// immutable layer descriptors + small opaque metadata + cells (the same
// inputs snapshot::serialize takes, except payload bytes are served by the
// callbacks instead of blobs).
// Compact Qx descriptors use layout_standard_qx_compact_rows only in a v2
// overflow range [P,C), with nontransposed V. Their payload_cells/payload_rows
// both count occupied global IDs >=P (bounded by C-P), including zero for an
// empty range; the shared cell summary keeps its original global IDs. The
// encoder validates this geometry before producer or sink callbacks run.
struct stream_input {
    std::vector<layer_desc> layers;
    std::vector<cell_entry> cells;
    uint32_t metadata_version = 0;
    const uint8_t * metadata = nullptr;
    size_t metadata_size = 0;
    snapshot_flags flags = snapshot_flags::full;
    uint64_t owner_epoch = 0;
    uint64_t owner_id = 0;
    // F4 v2: unset emits a v1 frame (unchanged wire); set emits a v2 frame with
    // the position_split envelope. Mirrors snapshot::serialize exactly.
    std::optional<position_split> split;
    // populated => emit a KMS3 frame carrying the ORIGINAL WRITER identity of a
    // pure (non-split) snapshot, so a split destination can verify the source.
    std::optional<source_identity> source;
};

// Stream-writes a snapshot. Validates the input exactly like
// snapshot::serialize (fail-closed, before any byte is written), computes
// per-layer payload checksums in a bounded first pass, then emits
// header/blob/descriptors/payloads/cells/tail_crc, recomputing payload
// checksums during the write pass and comparing them before completion.
// Returns true only when the complete, checksum-consistent snapshot was
// written. On ANY failure (validation, overflow, producer, sink, mutation,
// callback exception) the function throws kv_mixed_error — callback
// exceptions are wrapped — and opts.on_fault is invoked exactly once.
bool stream_serialize(const stream_input & input,
                      const stream_read_callbacks & reads,
                      const stream_write_callbacks & writes,
                      const stream_options & opts = {});

// ---------------------------------------------------------------------------
// Section map (F4 import forensics), reader side.
//
// Walks a REAL serialized frame and reports every contiguous region using the
// stream_section schema above. This is a read of the bytes that were actually
// written - it never reconstructs layout from a remembered order - so the
// write-side log and this walk are independent measurements of one schema.
//
// Returns the ordered section list. Throws kv_mixed_error on a frame the real
// parser would reject (bad magic/version/header crc/truncation), so a describe
// failure is itself evidence and never a silently empty map.
std::vector<stream_section> stream_describe(const uint8_t * data, size_t size);

// One-line rendering of a section, for logs.
std::string stream_section_text(const stream_section & s);

// Pure sizing: applies the SAME validation and accounting as
// stream_serialize (shared routine) and returns the exact serialized byte
// count WITHOUT reading or hashing any payload byte and WITHOUT calling any
// callback (zero tensor/GPU reads; useful for state_size queries). Throws
// kv_mixed_error on any invalid input (opts.on_fault exactly once).
uint64_t stream_serialized_size(const stream_input & input, const stream_options & opts = {});

// ---------------------------------------------------------------------------
// Linux immutable file backing for parse_view (explicitly guarded).
// Maps a snapshot file read-only with stable lifetime: payload() views from
// snapshot::parse_view point into the mapped file, so no anonymous
// whole-payload copy exists. MAP_PRIVATE yields a private copy-on-write view
// of the pages at map time; it does NOT freeze the external file — the file
// must not be rewritten while any mapping exists (caller contract; publish
// snapshots by temp-file + rename). Integrity (header/blob/payload/tail
// checksums) is verified by snapshot::parse_view.
//
// Lifetime: parse_view takes a std::shared_ptr that OWNS the backing, and the
// snapshot retains it for its whole lifetime. Pass the shared_ptr directly:
//     auto backing = std::make_shared<kv_mixed_file_backing>(path);
//     snapshot s = snapshot::parse_view(backing->data(), backing->size(),
//                                       backing);
// The shared_ptr keeps the mapping alive until the snapshot (and every move
// of it) is destroyed; commit() re-hashes borrowed payloads against their
// stored checksums before any stage runs, detecting in-place mutation of the
// backing — but immutability remains the caller's contract.
// Only regular files are mapped; sizes above the 4 TiB sanity bound (or
// SIZE_MAX on 32-bit hosts) are rejected before the size_t cast.
// On non-Linux the constructor fails with kv_mixed_error("unsupported").
// ---------------------------------------------------------------------------

#if defined(__linux__)

class kv_mixed_file_backing {
public:
    // Maps `path` read-only. Fails (kv_mixed_error) if the file cannot be
    // opened/stat/mapped, if it is smaller than the minimum header, or if
    // opts.max_total_bytes is set and the file exceeds it.
    explicit kv_mixed_file_backing(const std::string & path, const parse_options & opts = {});
    ~kv_mixed_file_backing();

    kv_mixed_file_backing(const kv_mixed_file_backing &) = delete;
    kv_mixed_file_backing & operator=(const kv_mixed_file_backing &) = delete;
    kv_mixed_file_backing(kv_mixed_file_backing && other) noexcept;
    kv_mixed_file_backing & operator=(kv_mixed_file_backing && other) noexcept;

    const uint8_t * data() const { return data_; }
    size_t size() const { return size_; }

private:
    uint8_t * data_ = nullptr;
    size_t size_ = 0;
};

#else // !defined(__linux__)

class kv_mixed_file_backing {
public:
    explicit kv_mixed_file_backing(const std::string &, const parse_options & opts = {});
    kv_mixed_file_backing(const kv_mixed_file_backing &) = delete;
    kv_mixed_file_backing & operator=(const kv_mixed_file_backing &) = delete;
    kv_mixed_file_backing(kv_mixed_file_backing &&) noexcept {}
    kv_mixed_file_backing & operator=(kv_mixed_file_backing &&) noexcept { return *this; }
    const uint8_t * data() const { return nullptr; }
    size_t size() const { return 0; }
};

#endif // defined(__linux__)

} // namespace llama_kv_mixed
