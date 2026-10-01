// IO bridge between the codec snapshot format and the legacy
// llama_io_write_i / llama_io_read_i interfaces. See llama-kv-mixed-io.h.

#define XXH_INLINE_ALL
#include "../vendor/hash/xxhash/xxhash.h"

#include "llama-kv-mixed-io.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_set>

namespace llama_kv_mixed {

namespace {

constexpr uint64_t MIXED_IO_MAX_TOTAL = uint64_t(1) << 42; // 4 TiB sanity bound

void io_fail(const std::string & message) {
    throw kv_mixed_error(message);
}

void put_u32(std::vector<uint8_t> & out, uint32_t v) {
    out.push_back(uint8_t(v & 0xFF));
    out.push_back(uint8_t((v >> 8) & 0xFF));
    out.push_back(uint8_t((v >> 16) & 0xFF));
    out.push_back(uint8_t((v >> 24) & 0xFF));
}
void put_u64(std::vector<uint8_t> & out, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        out.push_back(uint8_t((v >> (8 * i)) & 0xFF));
    }
}

struct wire_reader {
    const uint8_t * data;
    size_t size;
    size_t pos = 0;
    std::string * error;

    wire_reader(const uint8_t * d, size_t n, std::string * e) : data(d), size(n), error(e) {}

    bool need(size_t n) {
        if (pos > size || n > size - pos) {
            if (error) {
                *error = "truncated mixed-io manifest";
            }
            return false;
        }
        return true;
    }
    bool take(void * dst, size_t n) {
        if (!need(n)) {
            return false;
        }
        std::memcpy(dst, data + pos, n);
        pos += n;
        return true;
    }
    bool u8(uint8_t & v) { return take(&v, 1); }
    bool u32(uint32_t & v) {
        uint8_t b[4];
        if (!take(b, 4)) {
            return false;
        }
        v = uint32_t(b[0]) | (uint32_t(b[1]) << 8) | (uint32_t(b[2]) << 16) | (uint32_t(b[3]) << 24);
        return true;
    }
    bool u64(uint64_t & v) {
        uint8_t b[8];
        if (!take(b, 8)) {
            return false;
        }
        v = 0;
        for (int i = 7; i >= 0; --i) {
            v = (v << 8) | b[i];
        }
        return true;
    }
};

} // namespace

// ---- mixed_manifest --------------------------------------------------------

std::string mixed_manifest::validate(uint32_t max_layers, uint64_t max_metadata,
                                     uint32_t max_events) const {
    if (version != MIXED_IO_MANIFEST_VERSION) {
        return "unsupported mixed-io manifest version";
    }
    if (metadata.size() > max_metadata) {
        return "mixed-io manifest metadata exceeds the limit";
    }
    if (events.size() > max_events) {
        return "mixed-io manifest event count exceeds the limit";
    }
    uint64_t prev_stream = 0;
    uint64_t metadata_bytes = 0;
    for (const mixed_manifest_event & e : events) {
        if (e.kind > 1) {
            return "mixed-io manifest contains an unknown event kind";
        }
        if (e.size == 0) {
            return "mixed-io manifest contains a zero-length event";
        }
        if (e.stream_offset != prev_stream) {
            return "mixed-io manifest events are not contiguous";
        }
        // pre-bounded additions: no unchecked prev_stream + size
        if (e.size > MIXED_IO_MAX_TOTAL - prev_stream) {
            return "mixed-io manifest stream size overflows the sanity bound";
        }
        prev_stream += e.size;
        if (e.kind == 0) {
            metadata_bytes += e.size;
        } else {
            if (e.size > MIXED_IO_MAX_TOTAL - e.tensor_offset) {
                return "mixed-io manifest tensor range overflows the sanity bound";
            }
        }
    }
    if (metadata_bytes != metadata.size()) {
        // exact match: a short/long metadata region would make consume() read
        // out of bounds
        return "mixed-io manifest metadata length does not match its events";
    }
    if (metadata_bytes > max_metadata) {
        return "mixed-io manifest metadata exceeds the limit";
    }
    if (!events.empty() && prev_stream != total_stream_bytes) {
        return "mixed-io manifest total_stream_bytes does not match the events";
    }
    if (events.empty() && total_stream_bytes != 0) {
        return "mixed-io manifest total_stream_bytes is nonzero without events";
    }
    if (layer_payload_sizes.size() > max_layers) {
        return "mixed-io manifest layer count exceeds the limit";
    }
    uint32_t prev_layer = 0;
    bool first = true;
    for (const auto & lp : layer_payload_sizes) {
        if (!first && lp.first <= prev_layer) {
            return "mixed-io manifest layer ids are not strictly ordered";
        }
        first = false;
        prev_layer = lp.first;
        if (lp.second > MIXED_IO_MAX_TOTAL) {
            return "mixed-io manifest layer payload size overflows the sanity bound";
        }
    }
    // every tensor event must reference a declared layer; layer-local
    // payload offsets must be monotonic and within the layer total
    std::unordered_map<uint32_t, uint64_t> layer_used;
    std::unordered_map<uint32_t, uint64_t> layer_prev_off;
    for (const mixed_manifest_event & e : events) {
        if (e.kind != 1) {
            continue;
        }
        const auto it = std::find_if(layer_payload_sizes.begin(), layer_payload_sizes.end(),
                                     [&](const std::pair<uint32_t, uint64_t> & lp) {
                                         return lp.first == e.layer_id;
                                     });
        if (it == layer_payload_sizes.end()) {
            return "mixed-io manifest tensor event references an undeclared layer";
        }
        const uint64_t prev = layer_prev_off.count(e.layer_id) ? layer_prev_off[e.layer_id] : 0;
        if (e.payload_offset != prev) {
            return "mixed-io manifest tensor event payload offsets are not contiguous";
        }
        // pre-bounded additions
        if (e.size > MIXED_IO_MAX_TOTAL - prev) {
            return "mixed-io manifest layer payload offsets overflow the sanity bound";
        }
        layer_prev_off[e.layer_id] = prev + e.size;
        const uint64_t used = layer_used[e.layer_id];
        if (e.size > it->second - used) {
            return "mixed-io manifest tensor event exceeds its layer payload size";
        }
        layer_used[e.layer_id] = used + e.size;
    }
    for (const auto & lp : layer_payload_sizes) {
        if (layer_used[lp.first] != lp.second) {
            return "mixed-io manifest layer payload size does not match its events";
        }
    }
    return "";
}

std::vector<uint8_t> mixed_manifest_serialize(const mixed_manifest & m, std::string * error) {
    const std::string err = m.validate();
    if (!err.empty()) {
        if (error) {
            *error = err;
        }
        return {};
    }
    const uint64_t n_metadata = m.metadata.size();
    const uint64_t n_events = m.events.size();
    const uint64_t n_layers = m.layer_payload_sizes.size();
    // overflow-checked wire body size
    uint64_t body = 16 + n_metadata + 8;
    if (n_events > (MIXED_IO_MAX_TOTAL - body) / MIXED_IO_WIRE_EVENT_SIZE) {
        if (error) {
            *error = "mixed-io manifest event table size overflows";
        }
        return {};
    }
    body += n_events * MIXED_IO_WIRE_EVENT_SIZE + 8;
    if (n_layers > (MIXED_IO_MAX_TOTAL - body) / MIXED_IO_WIRE_LAYER_SIZE) {
        if (error) {
            *error = "mixed-io manifest layer table size overflows";
        }
        return {};
    }
    body += n_layers * MIXED_IO_WIRE_LAYER_SIZE + 8;
    if (body > std::numeric_limits<size_t>::max()) {
        if (error) {
            *error = "mixed-io manifest exceeds the host size_t range";
        }
        return {};
    }
    std::vector<uint8_t> out;
    out.reserve(size_t(body));
    put_u32(out, MIXED_IO_MANIFEST_MAGIC);
    put_u32(out, m.version);
    put_u64(out, n_metadata);
    out.insert(out.end(), m.metadata.begin(), m.metadata.end());
    put_u64(out, n_events);
    for (const mixed_manifest_event & e : m.events) {
        out.push_back(e.kind);
        put_u32(out, e.layer_id);
        put_u64(out, e.payload_offset);
        put_u64(out, e.size);
        put_u64(out, e.stream_offset);
        put_u64(out, e.tensor_offset);
    }
    put_u64(out, n_layers);
    for (const auto & lp : m.layer_payload_sizes) {
        put_u32(out, lp.first);
        put_u64(out, lp.second);
    }
    put_u64(out, m.total_stream_bytes);
    return out;
}

bool mixed_manifest_parse(const uint8_t * data, size_t size, mixed_manifest & out,
                          std::string * error) {
    out = mixed_manifest{};
    if (data == nullptr && size > 0) {
        if (error) {
            *error = "null mixed-io manifest data";
        }
        return false;
    }
    wire_reader r(data, size, error);
    uint32_t magic = 0;
    if (!r.u32(magic) || magic != MIXED_IO_MANIFEST_MAGIC) {
        if (error) {
            *error = "bad mixed-io manifest magic";
        }
        return false;
    }
    if (!r.u32(out.version) || out.version != MIXED_IO_MANIFEST_VERSION) {
        if (error) {
            *error = "unsupported mixed-io manifest version";
        }
        return false;
    }
    uint64_t n_metadata = 0;
    // cap BEFORE copying; SIZE_MAX guard before the size_t cast
    if (!r.u64(n_metadata) || n_metadata > MIXED_IO_MANIFEST_MAX_METADATA ||
            n_metadata > std::numeric_limits<size_t>::max() || !r.need(size_t(n_metadata))) {
        if (error) {
            *error = "truncated or oversized mixed-io manifest metadata";
        }
        return false;
    }
    out.metadata.resize(size_t(n_metadata));
    if (n_metadata && !r.take(out.metadata.data(), size_t(n_metadata))) {
        return false;
    }
    uint64_t n_events = 0;
    if (!r.u64(n_events) || n_events > (1u << 20)) {
        if (error) {
            *error = "mixed-io manifest event count exceeds the limit";
        }
        return false;
    }
    // check the remaining wire bytes cover the event table BEFORE reserve
    if (n_events > (MIXED_IO_MAX_TOTAL - r.pos) / MIXED_IO_WIRE_EVENT_SIZE ||
            n_events * MIXED_IO_WIRE_EVENT_SIZE > r.size - r.pos) {
        if (error) {
            *error = "truncated mixed-io manifest event table";
        }
        return false;
    }
    out.events.reserve(size_t(n_events));
    for (uint64_t i = 0; i < n_events; ++i) {
        mixed_manifest_event e;
        if (!r.u8(e.kind) || !r.u32(e.layer_id) || !r.u64(e.payload_offset) ||
                !r.u64(e.size) || !r.u64(e.stream_offset) || !r.u64(e.tensor_offset)) {
            if (error) {
                *error = "truncated mixed-io manifest event";
            }
            return false;
        }
        out.events.push_back(e);
    }
    uint64_t n_layers = 0;
    if (!r.u64(n_layers) || n_layers > 4096) {
        if (error) {
            *error = "mixed-io manifest layer count exceeds the limit";
        }
        return false;
    }
    if (n_layers * MIXED_IO_WIRE_LAYER_SIZE > r.size - r.pos) {
        if (error) {
            *error = "truncated mixed-io manifest layer table";
        }
        return false;
    }
    out.layer_payload_sizes.reserve(size_t(n_layers));
    for (uint64_t i = 0; i < n_layers; ++i) {
        uint32_t layer_id = 0;
        uint64_t total = 0;
        if (!r.u32(layer_id) || !r.u64(total)) {
            if (error) {
                *error = "truncated mixed-io manifest layer entry";
            }
            return false;
        }
        out.layer_payload_sizes.emplace_back(layer_id, total);
    }
    if (!r.u64(out.total_stream_bytes)) {
        if (error) {
            *error = "truncated mixed-io manifest trailer";
        }
        return false;
    }
    if (r.pos != r.size) {
        if (error) {
            *error = "trailing garbage after the mixed-io manifest";
        }
        return false;
    }
    const std::string err = out.validate();
    if (!err.empty()) {
        if (error) {
            *error = err;
        }
        return false;
    }
    return true;
}

// ---- split_writer ----------------------------------------------------------

void split_writer::fail_terminal(const std::string & message) {
    sealed_ = true; // terminal: no further recording accepted after a cap/seal failure
    io_fail(message);
}

namespace {
void fail(const std::string & message) {
    io_fail(message);
}
} // namespace

split_writer::split_writer(layer_fn layer_of, split_writer_limits limits)
    : limits_(limits), layer_of_(std::move(layer_of)) {
    if (!layer_of_) {
        io_fail("split_writer: layer mapping callback is required");
    }
    if (limits_.max_layers == 0 || limits_.max_events == 0) {
        io_fail("split_writer: invalid limits");
    }
    layer_bytes_.assign(limits_.max_layers, 0);
}

void split_writer::write(const void * src, size_t size) {
    if (sealed_) {
        fail_terminal("split_writer: writes are rejected after the payload producer started");
    }
    if (size == 0) {
        return;
    }
    if (src == nullptr) {
        fail("split_writer: null metadata bytes");
    }
    if (size > limits_.max_metadata_bytes - manifest_.metadata.size()) {
        fail_terminal("split_writer: metadata budget exceeded");
    }
    if (limits_.max_virtual_stream_bytes != 0 &&
            size > limits_.max_virtual_stream_bytes - stream_size_) {
        fail_terminal("split_writer: virtual stream cap exceeded");
    }
    if (size > MIXED_IO_MAX_TOTAL - stream_size_) {
        fail_terminal("split_writer: virtual stream size overflows the sanity bound");
    }
    if (manifest_.events.size() >= limits_.max_events) {
        fail_terminal("split_writer: event budget exceeded");
    }
    // combine adjacent ordinary writes into one metadata run
    if (!manifest_.events.empty() && manifest_.events.back().kind == 0) {
        mixed_manifest_event & last = manifest_.events.back();
        last.size += size;
        manifest_.metadata.insert(manifest_.metadata.end(),
                                  static_cast<const uint8_t *>(src),
                                  static_cast<const uint8_t *>(src) + size);
        stream_size_ += size;
        manifest_.total_stream_bytes = stream_size_;
        return;
    }
    mixed_manifest_event e;
    e.kind = 0;
    e.size = size;
    e.stream_offset = stream_size_;
    manifest_.events.push_back(e);
    event_tensors_.push_back({ nullptr, 0 });
    manifest_.metadata.insert(manifest_.metadata.end(),
                              static_cast<const uint8_t *>(src),
                              static_cast<const uint8_t *>(src) + size);
    stream_size_ += size;
    manifest_.total_stream_bytes = stream_size_;
}

void split_writer::write_tensor(ggml_tensor * tensor, size_t offset, size_t size) {
    if (sealed_) {
        fail_terminal("split_writer: writes are rejected after the payload producer started");
    }
    if (size == 0) {
        return;
    }
    if (tensor == nullptr) {
        fail("split_writer: null tensor");
    }
    const size_t nbytes = ggml_nbytes(tensor);
    if (offset > nbytes || size > nbytes - offset) {
        fail("split_writer: tensor range exceeds the tensor size");
    }
    const uint32_t layer_id = layer_of_(tensor);
    if (layer_id >= limits_.max_layers) {
        fail("split_writer: layer id exceeds the layer limit");
    }
    if (limits_.max_virtual_stream_bytes != 0 &&
            size > limits_.max_virtual_stream_bytes - stream_size_) {
        fail_terminal("split_writer: virtual stream cap exceeded");
    }
    if (size > MIXED_IO_MAX_TOTAL - stream_size_) {
        fail_terminal("split_writer: virtual stream size overflows the sanity bound");
    }
    if (limits_.max_payload_bytes != 0 &&
            size > limits_.max_payload_bytes - layer_bytes_[layer_id]) {
        fail_terminal("split_writer: layer payload exceeds the per-layer cap");
    }
    if (manifest_.events.size() >= limits_.max_events) {
        fail_terminal("split_writer: event budget exceeded");
    }

    mixed_manifest_event e;
    e.kind = 1;
    e.layer_id = layer_id;
    e.payload_offset = layer_bytes_[layer_id];
    e.size = size;
    e.stream_offset = stream_size_;
    e.tensor_offset = offset;
    manifest_.events.push_back(e);
    event_tensors_.push_back({ tensor, offset });

    layer_bytes_[layer_id] += size;
    stream_size_ += size;
    manifest_.total_stream_bytes = stream_size_;

    auto it = std::lower_bound(manifest_.layer_payload_sizes.begin(),
                               manifest_.layer_payload_sizes.end(),
                               std::make_pair(layer_id, uint64_t(0)));
    if (it != manifest_.layer_payload_sizes.end() && it->first == layer_id) {
        it->second += size;
    } else {
        manifest_.layer_payload_sizes.insert(it, std::make_pair(layer_id, uint64_t(size)));
    }
}

size_t split_writer::n_bytes() {
    if (stream_size_ > SIZE_MAX) {
        io_fail("split_writer: virtual stream size exceeds host size_t");
    }
    return size_t(stream_size_);
}

uint64_t split_writer::virtual_stream_bytes() const {
    return stream_size_;
}

size_t split_writer::read_payload(uint32_t layer_id, uint64_t offset, uint8_t * dst, size_t count) const {
    if (count == 0) {
        return 0;
    }
    if (dst == nullptr) {
        io_fail("split_writer: null destination for read_payload");
    }
    if (offset > MIXED_IO_MAX_TOTAL || count > MIXED_IO_MAX_TOTAL - offset) {
        return 0;
    }
    // first producer read seals the writer: payload_index_ holds pointers
    // into events, which later writes would reallocate
    sealed_ = true;
    auto & evs = payload_index_[layer_id];
    if (evs.empty()) {
        for (size_t i = 0; i < manifest_.events.size(); ++i) {
            if (manifest_.events[i].kind == 1 && manifest_.events[i].layer_id == layer_id) {
                evs.push_back(&manifest_.events[i]);
            }
        }
    }
    size_t done = 0;
    uint64_t pos = offset;
    for (const mixed_manifest_event * e : evs) {
        if (done == count) {
            break;
        }
        if (e->payload_offset + e->size <= pos) {
            continue;
        }
        if (e->payload_offset >= offset + count) {
            break;
        }
        const uint64_t local = pos > e->payload_offset ? pos - e->payload_offset : 0;
        const uint64_t avail = e->size - local;
        const uint64_t take = std::min<uint64_t>(avail, count - done);
        const tensor_range & tr = event_tensors_[size_t(e - manifest_.events.data())];
        // recheck the absolute tensor range before the backend read
        const size_t nbytes = ggml_nbytes(tr.tensor);
        const uint64_t abs_off = tr.tensor_offset + local;
        if (abs_off > nbytes || take > nbytes - abs_off) {
            io_fail("split_writer: captured tensor range is out of bounds");
        }
        ggml_backend_tensor_get(tr.tensor, dst + done, size_t(abs_off), size_t(take));
        done += size_t(take);
        pos += take;
    }
    return done;
}

// ---- split_reader ----------------------------------------------------------

void split_reader::check_alive() const {
    if (failed_) {
        io_fail("split_reader: reader is in a failed/terminal state");
    }
    if (deferred_) {
        io_fail("split_reader: reader is already deferred/terminal");
    }
}

void split_reader::fail_terminal(const std::string & message) {
    failed_ = true; // terminal: no partial retry accepted after a guard failure
    io_fail(message);
}

split_reader::split_reader(mixed_manifest manifest, std::vector<payload_span> spans,
                           split_reader_limits limits)
    : manifest_(std::move(manifest)), spans_(std::move(spans)), limits_(limits) {
    const std::string err = manifest_.validate();
    if (!err.empty()) {
        io_fail("split_reader: invalid manifest: " + err);
    }
    if (limits_.max_payload_spans == 0 || limits_.max_operations == 0 || limits_.max_callbacks == 0) {
        io_fail("split_reader: invalid limits");
    }
    if (spans_.size() > limits_.max_payload_spans) {
        io_fail("split_reader: payload span count exceeds the limit");
    }
    // per-event metadata offsets (u64: ordinary metadata can exceed 255 bytes)
    meta_offsets_.resize(manifest_.events.size());
    uint64_t meta_off = 0;
    for (size_t i = 0; i < manifest_.events.size(); ++i) {
        meta_offsets_[i] = manifest_.events[i].kind == 0 ? meta_off : 0;
        if (manifest_.events[i].kind == 0) {
            meta_off += manifest_.events[i].size;
        }
    }
    // spans: unique layer ids, exact size match with the manifest layer table
    span_by_layer_.assign(manifest_.layer_payload_sizes.size(), SIZE_MAX);
    for (size_t s = 0; s < spans_.size(); ++s) {
        const payload_span & sp = spans_[s];
        if (sp.data == nullptr && sp.size > 0) {
            io_fail("split_reader: span with null data and nonzero size");
        }
        if (sp.size > 0 && !sp.owner) {
            // the zero-copy lifetime claims require a retained owner
            io_fail("split_reader: nonzero payload span requires a retained owner");
        }
        if (sp.size > MIXED_IO_MAX_TOTAL) {
            io_fail("split_reader: span exceeds the sanity bound");
        }
        if (limits_.max_payload_bytes != 0 && sp.size > limits_.max_payload_bytes) {
            io_fail("split_reader: span exceeds the per-span cap");
        }
        auto it = std::lower_bound(manifest_.layer_payload_sizes.begin(),
                                   manifest_.layer_payload_sizes.end(),
                                   std::make_pair(sp.layer_id, uint64_t(0)));
        if (it == manifest_.layer_payload_sizes.end() || it->first != sp.layer_id) {
            io_fail("split_reader: span references an undeclared layer");
        }
        if (it->second != sp.size) {
            io_fail("split_reader: span size does not match the manifest layer payload size");
        }
        const size_t idx = size_t(it - manifest_.layer_payload_sizes.begin());
        if (span_by_layer_[idx] != SIZE_MAX) {
            io_fail("split_reader: duplicate payload span for a layer");
        }
        span_by_layer_[idx] = s;
    }
    // every declared layer must have a span
    for (size_t i = 0; i < manifest_.layer_payload_sizes.size(); ++i) {
        if (span_by_layer_[i] == SIZE_MAX) {
            io_fail("split_reader: missing payload span for a declared layer");
        }
    }
}

split_reader::~split_reader() {
    if (!deferred_) {
        cancel();
    }
}

uint64_t split_reader::event_meta_offset(size_t event_index) const {
    return meta_offsets_[event_index];
}

void split_reader::consume(void * dst, size_t size, bool as_tensor_op,
                           ggml_tensor * tensor, size_t tensor_offset) {
    if (size == 0) {
        return;
    }
    check_alive();
    if (size > MIXED_IO_MAX_TOTAL - cursor_) {
        fail_terminal("split_reader: virtual stream read overflows the sanity bound");
    }
    uint8_t * out = static_cast<uint8_t *>(dst); // null for tensor ops
    uint64_t dst_off = 0;                        // running destination offset
    uint64_t need = size;
    size_t idx = event_idx_; // monotonic event cursor: consume is sequential;
                             // invariant: events[idx] contains cursor_ (or idx == size)
    while (need > 0) {
        if (idx >= manifest_.events.size()) {
            fail_terminal("split_reader: virtual stream truncated");
        }
        const mixed_manifest_event & e = manifest_.events[idx];
        const uint64_t local = cursor_ - e.stream_offset;
        const uint64_t avail = e.size - local;
        const uint64_t take = std::min<uint64_t>(avail, need);
        const size_t part_dst_off = size_t(dst_off);
        if (e.kind == 0) {
            const uint8_t * src = manifest_.metadata.data() + event_meta_offset(idx) + local;
            if (as_tensor_op) {
                // a legacy read_tensor crossing a metadata run: copy the
                // ordinary bytes into a staged fixup (bounded by the fixup cap)
                if (take > limits_.max_fixup_staging - staged_bytes_) {
                    fail_terminal("split_reader: fixup staging budget exceeded");
                }
                if (ops_.size() >= limits_.max_operations) {
                    fail_terminal("split_reader: operation budget exceeded");
                }
                staged_chunks_.emplace_back(size_t(take));
                std::vector<uint8_t> & chunk = staged_chunks_.back();
                std::memcpy(chunk.data(), src, size_t(take));
                staged_bytes_ += take;
                ops_.push_back({ 1, tensor, tensor_offset + part_dst_off,
                                 chunk.data(), size_t(take), {}, 0 });
            } else {
                std::memcpy(out + part_dst_off, src, size_t(take));
            }
        } else {
            // tensor event: the span must cover the layer-local range
            const auto it = std::lower_bound(manifest_.layer_payload_sizes.begin(),
                                             manifest_.layer_payload_sizes.end(),
                                             std::make_pair(e.layer_id, uint64_t(0)));
            const size_t span_idx = span_by_layer_[size_t(it - manifest_.layer_payload_sizes.begin())];
            const payload_span & sp = spans_[span_idx];
            const uint64_t local_payload = e.payload_offset + local;
            if (local_payload + take > sp.size) {
                fail_terminal("split_reader: span does not cover the requested tensor range");
            }
            const uint8_t * src = sp.data + local_payload;
            if (as_tensor_op) {
                if (ops_.size() >= limits_.max_operations) {
                    fail_terminal("split_reader: operation budget exceeded");
                }
                uint64_t hash = 0;
                if (limits_.hash_borrowed) {
                    hash = XXH64(src, size_t(take), 0);
                }
                ops_.push_back({ 2, tensor, tensor_offset + part_dst_off,
                                 src, size_t(take), sp.owner, hash });
            } else {
                std::memcpy(out + part_dst_off, src, size_t(take));
            }
        }
        dst_off += take;
        cursor_ += take;
        need -= take;
        if (cursor_ == e.stream_offset + e.size) {
            ++idx; // advance only when the current event is fully consumed
        }
    }
    event_idx_ = idx;
}

void split_reader::read(void * dst, size_t size) {
    if (size > 0 && dst == nullptr) {
        fail("split_reader: null destination for read");
    }
    consume(dst, size, false, nullptr, 0);
}

void split_reader::read_tensor(ggml_tensor * tensor, size_t offset, size_t size) {
    check_alive();
    if (tensor == nullptr) {
        fail("split_reader: null tensor");
    }
    const size_t nbytes = ggml_nbytes(tensor);
    if (offset > nbytes || size > nbytes - offset) {
        fail("split_reader: tensor range exceeds the tensor size");
    }
    consume(nullptr, size, true, tensor, offset);
}

void split_reader::stage_tensor_set(ggml_tensor * tensor, const void * src, size_t offset, size_t size) {
    check_alive();
    if (tensor == nullptr) {
        fail("split_reader: null tensor");
    }
    if (size > 0 && src == nullptr) {
        fail("split_reader: null source for stage_tensor_set");
    }
    const size_t nbytes = ggml_nbytes(tensor);
    if (offset > nbytes || size > nbytes - offset) {
        fail("split_reader: tensor range exceeds the tensor size");
    }
    if (size > limits_.max_fixup_staging - staged_bytes_) {
        fail_terminal("split_reader: fixup staging budget exceeded");
    }
    if (ops_.size() >= limits_.max_operations) {
        fail_terminal("split_reader: operation budget exceeded");
    }
    staged_chunks_.emplace_back(size);
    std::vector<uint8_t> & chunk = staged_chunks_.back();
    std::memcpy(chunk.data(), src, size);
    staged_bytes_ += size;
    ops_.push_back({ 1, tensor, offset, chunk.data(), size, {}, 0 });
}

void split_reader::stage_tensor_clear(ggml_tensor * tensor, size_t offset, size_t size) {
    check_alive();
    if (tensor == nullptr) {
        fail("split_reader: null tensor");
    }
    const size_t nbytes = ggml_nbytes(tensor);
    if (offset > nbytes || size > nbytes - offset) {
        fail("split_reader: tensor range exceeds the tensor size");
    }
    if (ops_.size() >= limits_.max_operations) {
        fail_terminal("split_reader: operation budget exceeded");
    }
    ops_.push_back({ 0, tensor, offset, nullptr, size, {}, 0 });
}

void split_reader::on_commit(std::function<void()> callback) {
    check_alive();
    if (!callback) {
        fail_terminal("split_reader: empty on_commit callback is rejected");
    }
    if (callbacks_.size() >= limits_.max_callbacks) {
        fail_terminal("split_reader: callback budget exceeded");
    }
    callbacks_.push_back(std::move(callback));
}

void split_reader::commit() {
    io_fail("split_reader: direct commit is unsupported; use defer_to(outer)");
}

void split_reader::cancel() {
    // repeated cancel is safe; cancel after defer is a no-op
    if (deferred_) {
        return;
    }
    ops_.clear();
    staged_chunks_.clear();
    staged_bytes_ = 0;
    callbacks_.clear();
    spans_.clear(); // releases retained span owners; no tensor writes
    failed_ = true; // terminal: nothing can be read/queued after cancel
}

size_t split_reader::n_bytes() {
    if (cursor_ > SIZE_MAX) {
        io_fail("split_reader: consumed bytes exceed host size_t");
    }
    return size_t(cursor_);
}

void split_reader::queue_borrowed(ggml_tensor * tensor, size_t offset, const uint8_t * data,
                                  size_t count, std::shared_ptr<const void> owner) {
    check_alive();
    if (tensor == nullptr || (data == nullptr && count > 0)) {
        fail("split_reader: invalid borrowed range");
    }
    if (count > 0 && !owner) {
        // the zero-copy lifetime claims require a retained owner
        fail("split_reader: nonzero borrowed range requires a retained owner");
    }
    const size_t nbytes = ggml_nbytes(tensor);
    if (offset > nbytes || count > nbytes - offset) {
        fail("split_reader: tensor range exceeds the tensor size");
    }
    if (ops_.size() >= limits_.max_operations) {
        fail_terminal("split_reader: operation budget exceeded");
    }
    uint64_t hash = 0;
    if (limits_.hash_borrowed && count > 0) {
        hash = XXH64(data, count, 0);
    }
    ops_.push_back({ 2, tensor, offset, data, count, std::move(owner), hash });
}

uint64_t split_reader::remaining() const {
    return manifest_.total_stream_bytes - cursor_;
}

void split_reader::defer_to(llama_io_read_i & outer) {
    check_alive();
    if (cursor_ != manifest_.total_stream_bytes) {
        fail_terminal("split_reader: virtual stream not fully consumed before defer");
    }
    auto pending = std::make_shared<pending_state>();
    pending->ops = std::move(ops_);
    pending->staged_chunks = std::move(staged_chunks_);
    pending->callbacks = std::move(callbacks_);
    pending->hash_checked = limits_.hash_borrowed;
    ops_.clear();
    staged_chunks_.clear();
    staged_bytes_ = 0;
    callbacks_.clear();
    spans_.clear(); // owners are retained inside the pending ops
    // mark the reader terminal BEFORE registering: if outer.on_commit throws,
    // the reader stays rejected (no retry that would silently succeed empty)
    deferred_ = true;
    try {
        outer.on_commit([pending]() {
            if (pending->applied) {
                io_fail("split_reader: duplicate commit callback invocation");
            }
            pending->applied = true;
            if (pending->hash_checked) {
                for (const op & o : pending->ops) {
                    if (o.kind == 2 && o.size > 0 && XXH64(o.ptr, o.size, 0) != o.hash) {
                        io_fail("split_reader: borrowed payload changed since queue; backing must be immutable");
                    }
                }
            }
            for (const op & o : pending->ops) {
                if (o.kind == 0) {
                    ggml_backend_tensor_memset(o.tensor, 0, o.offset, o.size);
                } else {
                    ggml_backend_tensor_set(o.tensor, o.ptr, o.offset, o.size);
                }
            }
            for (const auto & cb : pending->callbacks) {
                cb();
            }
        });
    } catch (...) {
        // the outer rejected the callback: transferred state is released with
        // the pending shared_ptr; the reader stays terminal
        throw;
    }
}

} // namespace llama_kv_mixed