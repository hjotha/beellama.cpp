#pragma once

// IO bridge between the codec snapshot format (llama-kv-mixed-state /
// -stream) and the legacy llama_io_write_i / llama_io_read_i interfaces.
// Exclusive new module: no core/context/cache/llama-io.h changes.
//
// split_writer  - implements llama_io_write_i over a legacy state writer
//                 (e.g. KVarN state_write) WITHOUT reading tensor bytes:
//                 write() bytes and write_tensor() ranges are recorded as a
//                 versioned manifest + ordered events + per-layer payload
//                 sizes. The virtual legacy stream (metadata + tensor bytes)
//                 is only COUNTED (n_bytes), so state_size queries never
//                 touch GPU tensors. read_payload() later serves the exact
//                 layer payload bytes (bounded chunks, ggml_backend_tensor_get)
//                 for stream_serialize. Tensor pointers are borrowed and must
//                 stay immutable until the writer is discarded.
// split_reader  - implements llama_io_read_i over a parsed manifest + borrowed
//                 per-layer payload spans (mmap/codec views). It reverses the
//                 virtual legacy stream: read() serves any byte range
//                 (metadata AND payload events, so legacy remappers that read
//                 payload bytes with read() work), read_tensor() queues
//                 BORROWED segment writes (zero copy, split across events),
//                 stage_tensor_set() is limited to small fixups with a TOTAL
//                 staging budget (default 64 MiB, not per chunk),
//                 stage_tensor_clear() is queued. Nothing touches the
//                 destination until defer_to() transfers everything into ONE
//                 outer.on_commit callback that applies ALL tensor ops (incl.
//                 Qx queue_borrowed() appends) and then the collected legacy
//                 publication callbacks, after the outer frame fully parsed.
//                 Direct commit() is rejected; outer cancel/destruction
//                 before defer cancels with no writes; after defer the
//                 transferred state is owned by the outer io.
//
// The module is a pure IO bridge: no file mapping pool, no model parsing.
// Luna supplies the input backing, the codec snapshot/payloads and the layer
// geometry; ON_DEVICE transport uses wholly separate device buffers in
// llama-context.cpp (llama_io_write_device/read_device) and is NOT supported
// through these helpers - the caller must reject the ON_DEVICE flag.

#include "llama-io.h"
#include "llama-kv-mixed-state.h" // kv_mixed_error

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llama_kv_mixed {

constexpr uint32_t MIXED_IO_MANIFEST_MAGIC = 0x4D534D4B; // "KMSM"
constexpr uint32_t MIXED_IO_MANIFEST_VERSION = 1;
constexpr uint32_t MIXED_IO_WIRE_EVENT_SIZE = 37; // kind1+layer4+payload_off8+size8+stream_off8+tensor_off8
constexpr uint32_t MIXED_IO_WIRE_LAYER_SIZE = 12; // layer_id4+total8
constexpr uint64_t MIXED_IO_MANIFEST_MAX_METADATA = 64ull << 20; // parse/serialize cap

// ordered event in the virtual legacy stream (wire-stable)
struct mixed_manifest_event {
    uint8_t kind = 0;            // 0 = metadata run, 1 = tensor payload
    uint32_t layer_id = 0;       // tensor events only
    uint64_t payload_offset = 0; // layer-local byte offset of this event's bytes
    uint64_t size = 0;
    uint64_t stream_offset = 0;  // absolute offset in the virtual legacy stream
    uint64_t tensor_offset = 0;  // tensor events only: byte offset in the source tensor
};

// versioned manifest: ordinary metadata + ordered events (no process
// pointers) + per-layer payload sizes
struct mixed_manifest {
    uint32_t version = MIXED_IO_MANIFEST_VERSION;
    std::vector<uint8_t> metadata;                    // concatenated write() bytes
    std::vector<mixed_manifest_event> events;         // ordered by stream_offset
    std::vector<std::pair<uint32_t, uint64_t>> layer_payload_sizes; // sorted by layer_id
    uint64_t total_stream_bytes = 0;

    // structural validation shared by writer finalize and reader ctor;
    // returns an empty string when valid
    std::string validate(uint32_t max_layers = 4096, uint64_t max_metadata = 64ull << 20,
                         uint32_t max_events = 1u << 20) const;
};

// strict versioned wire serialization of a manifest (little-endian; version,
// magic and all sizes checked). Returns false + error on malformed input.
std::vector<uint8_t> mixed_manifest_serialize(const mixed_manifest & m, std::string * error = nullptr);
bool mixed_manifest_parse(const uint8_t * data, size_t size, mixed_manifest & out,
                          std::string * error = nullptr);

struct split_writer_limits {
    uint64_t max_metadata_bytes = 64u << 20; // total write() bytes
    uint32_t max_events = 1u << 20;          // total events (metadata runs + tensor)
    uint32_t max_layers = 4096;
    uint64_t max_payload_bytes = 0;          // per layer; 0 = 4 TiB sanity bound
    uint64_t max_virtual_stream_bytes = 0;   // total; 0 = 4 TiB sanity bound
};

// Captures a legacy llama_io_write_i stream as manifest + tensor ranges.
// No tensor bytes are read during recording (only ggml_nbytes for bounds).
class split_writer final : public ::llama_io_write_i {
public:
    // maps a source tensor to its logical codec layer id
    using layer_fn = std::function<uint32_t(const ::ggml_tensor *)>;

    explicit split_writer(layer_fn layer_of, split_writer_limits limits = {});
    ~split_writer() override = default;
    split_writer(const split_writer &) = delete;
    split_writer & operator=(const split_writer &) = delete;

    // llama_io_write_i
    void write(const void * src, size_t size) override;
    void write_tensor(::ggml_tensor * tensor, size_t offset, size_t size) override;
    size_t n_bytes() override; // == virtual_stream_bytes()

    // virtual legacy stream size (write + tensor bytes), overflow-guarded
    uint64_t virtual_stream_bytes() const;

    const mixed_manifest & manifest() const { return manifest_; }
    const split_writer_limits & limits() const { return limits_; }

    // payload producer for stream_serialize: serves layer payload bytes from
    // the captured ggml tensor ranges via ggml_backend_tensor_get in bounded
    // chunks (no full copy). Returns bytes served (== count on success; short
    // when the range is out of bounds).
    size_t read_payload(uint32_t layer_id, uint64_t offset, uint8_t * dst, size_t count) const;

private:
    struct tensor_range {
        const ::ggml_tensor * tensor;
        uint64_t tensor_offset;
    };

    const split_writer_limits limits_;
    layer_fn layer_of_;
    mixed_manifest manifest_;
    std::vector<tensor_range> event_tensors_; // parallel to manifest_.events (kind==1)
    uint64_t stream_size_ = 0;
    std::vector<uint64_t> layer_bytes_;       // layer_id -> total bytes
    mutable std::unordered_map<uint32_t, std::vector<const mixed_manifest_event *>> payload_index_; // lazy per-layer events
    mutable bool sealed_ = false; // first read_payload seals the writer: later writes rejected
                          // (payload_index_ holds pointers into events)

    void fail_terminal(const std::string & message);
};

struct split_reader_limits {
    uint64_t max_fixup_staging = 64u << 20; // TOTAL stage_tensor_set budget (not per chunk)
    uint32_t max_payload_spans = 1u << 20;
    uint64_t max_payload_bytes = 0;         // per span; 0 = 4 TiB sanity bound
    uint32_t max_operations = 4u << 20;     // TOTAL queued ops (consume splits, queue_borrowed,
                                            // stage_clear/set), checked BEFORE append
    uint32_t max_callbacks = 1u << 18;      // TOTAL on_commit callbacks
    bool hash_borrowed = false;             // hash borrowed ops at queue and recheck ALL
                                            // before any tensor write in the commit callback
};

// Reverses a virtual legacy stream over a parsed manifest + borrowed payload
// spans (codec views / mmap). Nothing writes to the destination until
// defer_to() registers the single outer publication callback.
class split_reader final : public ::llama_io_read_i {
public:
    // borrowed immutable payload span for one layer. owner MUST be non-null
    // for any span with size > 0: the zero-copy lifetime claims depend on the
    // retained owner keeping the backing alive until the outer io releases it
    struct payload_span {
        uint32_t layer_id = 0;
        const uint8_t * data = nullptr;
        uint64_t size = 0;
        std::shared_ptr<const void> owner;
    };

    // manifest is copied into owned storage; spans are borrowed and fully
    // bounds-validated before any op is queued
    split_reader(mixed_manifest manifest, std::vector<payload_span> spans,
                 split_reader_limits limits = {});
    ~split_reader() override; // before defer: cancels (no writes); after defer: no-op
    split_reader(const split_reader &) = delete;
    split_reader & operator=(const split_reader &) = delete;

    // llama_io_read_i
    void read(void * dst, size_t size) override;       // serves across metadata AND payload events
    void read_tensor(::ggml_tensor * tensor, size_t offset, size_t size) override; // borrowed queues, split across events
    void stage_tensor_set(::ggml_tensor * tensor, const void * src, size_t offset, size_t size) override; // small fixups, TOTAL cap
    void stage_tensor_clear(::ggml_tensor * tensor, size_t offset, size_t size) override;
    void on_commit(std::function<void()> callback) override; // collected separately
    void commit() override;   // rejected: direct nested commit unsupported
    void cancel() override;   // releases retained owners, drops pending ops, no writes
    size_t n_bytes() override; // virtual bytes consumed

    // Qx companion: enqueue a borrowed tensor write into the SAME pending
    // transaction without consuming the virtual stream
    void queue_borrowed(::ggml_tensor * tensor, size_t offset, const uint8_t * data, size_t count,
                        std::shared_ptr<const void> owner);

    // transfer pending ops + retained owners + callbacks into ONE
    // outer.on_commit callback. Requires the virtual stream fully consumed;
    // repeated defer is rejected. After defer the destructor must NOT cancel
    // the transferred state. Never calls outer.commit()/cancel() itself.
    void defer_to(::llama_io_read_i & outer);

    uint64_t remaining() const;
    bool deferred() const { return deferred_; }

private:
    struct op {
        uint8_t kind = 0; // 0 clear, 1 staged fixup, 2 borrowed
        ::ggml_tensor * tensor = nullptr;
        size_t offset = 0;
        const uint8_t * ptr = nullptr;
        size_t size = 0;
        std::shared_ptr<const void> owner;
        uint64_t hash = 0;
    };
    struct pending_state {
        std::vector<op> ops;
        std::vector<std::vector<uint8_t>> staged_chunks;
        std::vector<std::function<void()>> callbacks;
        bool hash_checked = false;
        bool applied = false; // guards duplicate outer commit callback invocation
    };

    void consume(void * dst, size_t size, bool as_tensor_op, ::ggml_tensor * tensor, size_t tensor_offset);
    uint64_t event_meta_offset(size_t event_index) const;

    mixed_manifest manifest_;
    std::vector<uint64_t> meta_offsets_; // per-event metadata byte offset (kind==0)
    std::vector<payload_span> spans_;
    std::vector<size_t> span_by_layer_; // layer_id -> span index (invalid == SIZE_MAX)
    split_reader_limits limits_;
    uint64_t cursor_ = 0;
    size_t event_idx_ = 0;              // monotonic event cursor (consume is sequential)
    std::vector<op> ops_;
    std::vector<std::vector<uint8_t>> staged_chunks_;
    uint64_t staged_bytes_ = 0;
    std::vector<std::function<void()>> callbacks_;
    bool deferred_ = false;
    bool failed_ = false;               // terminal: any op after a guard failure is rejected

    void check_alive() const;           // throws when deferred_/failed_
    void fail_terminal(const std::string & message); // marks failed_ and throws
};

} // namespace llama_kv_mixed