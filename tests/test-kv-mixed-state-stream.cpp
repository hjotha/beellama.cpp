// Standalone deterministic CPU tests for the bounded streaming encoder and
// file backing. Build without CMake:
//   g++ -std=c++17 -O2 -Wall -Wextra -I src
//     tests/test-kv-mixed-state-stream.cpp src/llama-kv-mixed-state-stream.cpp
//     src/llama-kv-mixed-state.cpp
//     -o /tmp/opencode/test-kv-mixed-state-stream

#include "llama-kv-mixed-state-stream.h"

#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

using namespace llama_kv_mixed;

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond) do { \
    ++g_checks; \
    if (!(cond)) { \
        ++g_failures; \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    } \
} while (0)

#define CHECK_THROWS(expr) do { \
    ++g_checks; \
    bool thrown = false; \
    try { (void)(expr); } catch (const kv_mixed_error &) { thrown = true; } \
    if (!thrown) { \
        ++g_failures; \
        std::fprintf(stderr, "FAIL %s:%d: expected kv_mixed_error from: %s\n", __FILE__, __LINE__, #expr); \
    } \
} while (0)

static void fill_pattern(std::vector<uint8_t> & v, uint32_t seed) {
    uint32_t x = seed * 2654435761u + 12345u;
    for (size_t i = 0; i < v.size(); ++i) {
        x = x * 1664525u + 1013904223u;
        v[i] = uint8_t(x >> 24);
    }
}

static layer_desc make_qx_layer(uint32_t id, uint16_t k_type, uint16_t v_type,
                                uint32_t rows, uint64_t bytes) {
    layer_desc d;
    d.layer_id = id;
    d.kind = cache_kind::standard_qx;
    d.k_type = k_type;
    d.v_type = v_type;
    d.k_rot = uint8_t(rotation::hadamard);
    d.v_rot = uint8_t(rotation::hadamard);
    d.k_rot_width = 256;
    d.v_rot_width = 64;
    d.layout = layout_standard_qx_rows;
    d.owner = owner_draft_mtp;
    d.v_trans = 1;
    d.head_dim_k = 256;
    d.head_dim_v = 64;
    d.n_head_kv = 4;
    d.n_stream = 1;
    d.k_stride = 40;
    d.v_stride = 40;
    d.payload_cells = rows;
    d.payload_rows = rows;
    d.payload_bytes = bytes;
    return d;
}

static layer_desc make_kvarn_layer(uint32_t id, uint16_t kvarn_type_id, uint8_t k_bits, uint8_t v_bits,
                                   uint8_t domain, uint32_t cells, uint64_t rows, uint64_t bytes) {
    layer_desc d;
    d.layer_id = id;
    d.kind = cache_kind::kvarn;
    d.k_type = kvarn_type_id;
    d.v_type = kvarn_type_id;
    d.k_bits = k_bits;
    d.v_bits = v_bits;
    d.kvarn_domain = domain;
    d.layout = layout_kvarn_records_stage_tail;
    d.owner = owner_target;
    d.tail_type = sx_f16;
    d.token_group = 128;
    d.record_dim = 4;
    d.head_dim_k = 256;
    d.head_dim_v = 256;
    d.head_slices = 4;
    d.n_head_kv = 4;
    d.n_stream = 1;
    d.k_stride = 64;
    d.v_stride = 64;
    d.payload_cells = cells;
    d.payload_rows = rows;
    d.payload_bytes = bytes;
    return d;
}

// ---- fixtures ---------------------------------------------------------------

struct fixture {
    std::vector<uint8_t> p0, p1, p2;
    std::vector<uint8_t> md;
    std::vector<layer_blob> blobs;
    std::vector<cell_entry> cells;
    stream_input input;
    std::vector<uint8_t> reference; // output of owning snapshot::serialize
};

static fixture make_full_fixture() {
    fixture f;
    f.p0.resize(4096); fill_pattern(f.p0, 1);
    f.p1.resize(2048); fill_pattern(f.p1, 2);
    f.p2.resize(512);  fill_pattern(f.p2, 3);
    f.md.resize(384);  fill_pattern(f.md, 4);
    f.blobs.push_back({ make_kvarn_layer(0, kv4v5, 4, 5, uint8_t(kvarn_domain::rotated), 8192, 64, f.p0.size()), f.p0.data(), f.p0.size() });
    f.blobs.push_back({ make_qx_layer(3, sx_q4_0, sx_q8_0, 128, f.p1.size()), f.p1.data(), f.p1.size() });
    f.blobs.push_back({ make_qx_layer(7, sx_q6_0, sx_q6_0, 32, f.p2.size()), f.p2.data(), f.p2.size() });
    f.cells.resize(5);
    f.cells[0].pos = 0;  f.cells[0].tok = 100; f.cells[0].seq_ids = { 0 };
    f.cells[1].pos = 1;  f.cells[1].tok = 101; f.cells[1].seq_ids = { 0, 1 };
    f.cells[2].pos = -1;
    f.cells[3].pos = 3;  f.cells[3].x = 7; f.cells[3].y = 9; f.cells[3].tok = 103; f.cells[3].seq_ids = { 1 };
    f.cells[4].pos = 4;  f.cells[4].tok = 104; f.cells[4].seq_ids = { 0 };
    for (const layer_blob & b : f.blobs) {
        f.input.layers.push_back(b.desc);
    }
    f.input.cells = f.cells;
    f.input.metadata_version = 3;
    f.input.metadata = f.md.data();
    f.input.metadata_size = f.md.size();
    f.input.flags = snapshot_flags::full;
    CHECK(snapshot::serialize(f.blobs, f.cells, 3, f.md.data(), f.md.size(),
                              snapshot_flags::full, 0, 0, f.reference));
    return f;
}

static fixture make_partial_fixture() {
    fixture f;
    f.p0.resize(2048); fill_pattern(f.p0, 11); // KVarN overlay
    f.md.resize(256);  fill_pattern(f.md, 12);
    layer_desc k = make_kvarn_layer(1, kv4v5, 4, 5, uint8_t(kvarn_domain::rotated), 32, 2, f.p0.size());
    k.payload_mode = mode_partial_overlay;
    layer_desc q = make_qx_layer(2, sx_q4_0, sx_q4_0, 32, 0);
    q.payload_mode = mode_resident_reference;
    q.payload_bytes = 0;
    f.input.layers = { k, q };
    f.input.metadata_version = 2;
    f.input.metadata = f.md.data();
    f.input.metadata_size = f.md.size();
    f.input.flags = snapshot_flags::partial;
    f.input.owner_epoch = 42;
    f.input.owner_id = 7;
    f.blobs = { { k, f.p0.data(), f.p0.size() }, { q, nullptr, 0 } };
    CHECK(snapshot::serialize(f.blobs, {}, 2, f.md.data(), f.md.size(),
                              snapshot_flags::partial, 42, 7, f.reference));
    return f;
}

static fixture make_body_only_fixture() {
    fixture f;
    f.p0.resize(1024); fill_pattern(f.p0, 21);
    f.blobs.push_back({ make_qx_layer(4, sx_q8_0, sx_q8_0, 16, f.p0.size()), f.p0.data(), f.p0.size() });
    f.input.layers = { f.blobs[0].desc };
    f.input.flags = snapshot_flags::body_only;
    CHECK(snapshot::serialize(f.blobs, {}, 0, nullptr, 0, snapshot_flags::body_only, 0, 0, f.reference));
    return f;
}

// producer/sink helpers --------------------------------------------------------

struct blob_producer {
    std::map<uint32_t, const uint8_t *> data;
    std::map<uint32_t, size_t> size;
    std::map<uint32_t, size_t> calls; // per-layer read call count
    size_t max_chunk = 0;

    size_t read(uint32_t layer_id, uint64_t offset, uint8_t * dst, size_t count) {
        ++calls[layer_id];
        max_chunk = std::max(max_chunk, count);
        const auto it = data.find(layer_id);
        if (it == data.end()) {
            return 0;
        }
        if (offset + count > size[layer_id]) {
            return 0;
        }
        std::memcpy(dst, it->second + offset, count);
        return count;
    }
};

struct vector_sink {
    std::vector<uint8_t> out;
    size_t calls = 0;
    bool ok = true;

    bool write(const uint8_t * bytes, size_t count) {
        ++calls;
        if (!ok) {
            return false;
        }
        out.insert(out.end(), bytes, bytes + count);
        return true;
    }
};

static bool stream_fixture(const fixture & f, size_t chunk, vector_sink & sink,
                           blob_producer & producer) {
    for (const layer_blob & b : f.blobs) {
        producer.data[b.desc.layer_id] = b.data;
        producer.size[b.desc.layer_id] = b.size;
    }
    stream_options opts;
    opts.transfer_buffer_bytes = chunk;
    stream_read_callbacks reads;
    reads.read_payload = [&](uint32_t layer_id, uint64_t offset, uint8_t * dst, size_t count) {
        return producer.read(layer_id, offset, dst, count);
    };
    stream_write_callbacks writes;
    writes.write_bytes = [&](const uint8_t * bytes, size_t count) {
        return sink.write(bytes, count);
    };
    return stream_serialize(f.input, reads, writes, opts);
}

// ---- tests ------------------------------------------------------------------

static void test_stream_equals_serialize_all_fixtures() {
    std::vector<fixture> fixtures;
    fixtures.push_back(make_full_fixture());
    fixtures.push_back(make_partial_fixture());
    fixtures.push_back(make_body_only_fixture());
    for (const fixture & f : fixtures) {
        vector_sink sink;
        blob_producer producer;
        CHECK(stream_fixture(f, 64, sink, producer));
        CHECK(sink.out.size() == f.reference.size());
        CHECK(std::memcmp(sink.out.data(), f.reference.data(), f.reference.size()) == 0);
    }
}

static void test_chunk_boundaries() {
    const fixture f = make_full_fixture();
    std::vector<size_t> chunks = { 1, 2, 3, 7, 17, 64, 4096, 1u << 20 };
    for (size_t chunk : chunks) {
        vector_sink sink;
        blob_producer producer;
        CHECK(stream_fixture(f, chunk, sink, producer));
        CHECK(sink.out.size() == f.reference.size());
        CHECK(std::memcmp(sink.out.data(), f.reference.data(), f.reference.size()) == 0);
        CHECK(producer.max_chunk <= chunk); // bounded transfer buffer
    }
}

static void test_stream_roundtrip_parse() {
    const fixture f = make_full_fixture();
    vector_sink sink;
    blob_producer producer;
    CHECK(stream_fixture(f, 128, sink, producer));
    snapshot s = snapshot::parse(sink.out.data(), sink.out.size());
    CHECK(s.layers().size() == 3);
    CHECK(s.payload_size(0) == f.p0.size() && std::memcmp(s.payload(0), f.p0.data(), f.p0.size()) == 0);
    CHECK(s.payload_size(1) == f.p1.size() && std::memcmp(s.payload(1), f.p1.data(), f.p1.size()) == 0);
    CHECK(s.payload_size(2) == f.p2.size() && std::memcmp(s.payload(2), f.p2.data(), f.p2.size()) == 0);
    CHECK(s.cells().size() == 5);
    CHECK(s.metadata_size() == f.md.size());
}

static void test_reference_layer_never_read() {
    const fixture f = make_partial_fixture();
    vector_sink sink;
    blob_producer producer;
    producer.data[1] = f.p0.data();
    producer.size[1] = f.p0.size();
    stream_options opts;
    opts.transfer_buffer_bytes = 64;
    stream_read_callbacks reads;
    reads.read_payload = [&](uint32_t layer_id, uint64_t offset, uint8_t * dst, size_t count) {
        return producer.read(layer_id, offset, dst, count);
    };
    stream_write_callbacks writes;
    writes.write_bytes = [&](const uint8_t * bytes, size_t count) { return sink.write(bytes, count); };
    CHECK(stream_serialize(f.input, reads, writes, opts));
    CHECK(sink.out == f.reference);
    CHECK(producer.calls[2] == 0); // resident reference layer is never read
    CHECK(producer.calls[1] == f.p0.size() / 64 * 2 + (f.p0.size() % 64 ? 2 : 0)); // overlay read twice
}

static void test_large_virtual_payload_bounded_scratch() {
    const uint64_t layer_bytes = 100u << 20; // 100 MiB per layer
    const uint32_t md_size = 64;
    std::vector<uint8_t> md(md_size);
    fill_pattern(md, 31);

    std::vector<layer_desc> layers;
    for (uint32_t id = 1; id <= 3; ++id) {
        layer_desc d = make_qx_layer(id, sx_q8_0, sx_q8_0, uint32_t(layer_bytes / 40), layer_bytes);
        layers.push_back(d);
    }
    stream_input input;
    input.layers = layers;
    input.metadata_version = 1;
    input.metadata = md.data();
    input.metadata_size = md.size();
    input.flags = snapshot_flags::full;

    const uint64_t total = 96 + md_size + 3 * 101 + 3 * layer_bytes + 8;
    stream_options opts;
    opts.transfer_buffer_bytes = 1u << 20; // 1 MiB
    opts.max_total_bytes = total;          // cap the whole encode
    size_t written = 0;
    size_t write_calls = 0;
    size_t max_read_chunk = 0;
    uint64_t read_calls = 0;
    stream_read_callbacks reads;
    reads.read_payload = [&](uint32_t layer_id, uint64_t offset, uint8_t * dst, size_t count) {
        ++read_calls;
        max_read_chunk = std::max(max_read_chunk, count);
        CHECK(count <= (1u << 20));
        for (size_t i = 0; i < count; ++i) {
            dst[i] = uint8_t((offset + i) * 131u + layer_id);
        }
        return count;
    };
    stream_write_callbacks writes;
    writes.write_bytes = [&](const uint8_t *, size_t count) {
        ++write_calls;
        written += count;
        return true; // byte counter only: no huge file/vector is materialized
    };
    CHECK(stream_serialize(input, reads, writes, opts));
    CHECK(written == total);
    CHECK(max_read_chunk <= (1u << 20));          // bounded transfer
    CHECK(read_calls == 3 * 100 * 2);             // 2 passes x 100 chunks per layer
    CHECK(write_calls == 1 + 1 + 3 + 3 * 100 + 1); // header+blob+descs+payloads+tail
}

static void test_producer_failure() {
    const fixture f = make_full_fixture();
    vector_sink sink;
    stream_options opts;
    opts.transfer_buffer_bytes = 64;
    stream_read_callbacks reads;
    reads.read_payload = [&](uint32_t, uint64_t, uint8_t *, size_t count) {
        return count - 1; // short read: producer failure
    };
    stream_write_callbacks writes;
    writes.write_bytes = [&](const uint8_t * bytes, size_t count) { return sink.write(bytes, count); };
    CHECK_THROWS(stream_serialize(f.input, reads, writes, opts));
    CHECK(sink.out.empty()); // validation of the callback happened... (header written only after pass 1)
    CHECK(sink.calls == 0);

    // producer throwing propagates
    reads.read_payload = [&](uint32_t, uint64_t, uint8_t *, size_t) -> size_t {
        throw std::runtime_error("producer boom");
    };
    bool thrown = false;
    try {
        stream_serialize(f.input, reads, writes, opts);
    } catch (const std::runtime_error &) {
        thrown = true;
    }
    CHECK(thrown);
    CHECK(sink.calls == 0);
}

static void test_producer_mutation() {
    const fixture f = make_full_fixture();
    const uint32_t lid = f.input.layers[0].layer_id; // 4096-byte payload layer
    std::map<uint32_t, const uint8_t *> data = { { 0, f.p0.data() }, { 3, f.p1.data() }, { 7, f.p2.data() } };
    std::map<uint32_t, size_t> sizes = { { 0, f.p0.size() }, { 3, f.p1.size() }, { 7, f.p2.size() } };
    vector_sink sink;
    int lid_calls = 0;
    stream_options opts;
    opts.transfer_buffer_bytes = f.p0.size(); // one chunk per pass
    stream_read_callbacks reads;
    reads.read_payload = [&](uint32_t layer_id, uint64_t offset, uint8_t * dst, size_t count) {
        if (offset + count > sizes[layer_id]) {
            return size_t(0);
        }
        std::memcpy(dst, data[layer_id] + offset, count);
        if (layer_id == lid && ++lid_calls == 2) {
            dst[0] ^= 0xFF; // pass 2 serves different bytes: mutation
        }
        return count;
    };
    stream_write_callbacks writes;
    writes.write_bytes = [&](const uint8_t * bytes, size_t count) { return sink.write(bytes, count); };
    CHECK_THROWS(stream_serialize(f.input, reads, writes, opts));
    // partial output reached the sink (header+blob+desc+payload), but the
    // encode FAILED: the caller must not publish; no tail_crc was written
    CHECK(sink.out.size() < f.reference.size());
    // everything except the tail_crc reached the sink (checksum comparison
    // runs after the last payload), so the output is incomplete and the
    // caller must not publish
    CHECK(sink.out.size() == f.reference.size() - 8);
}

static void test_sink_failure() {
    const fixture f = make_full_fixture();
    blob_producer producer;
    for (const layer_blob & b : f.blobs) {
        producer.data[b.desc.layer_id] = b.data;
        producer.size[b.desc.layer_id] = b.size;
    }
    stream_options opts;
    opts.transfer_buffer_bytes = 64;
    stream_read_callbacks reads;
    reads.read_payload = [&](uint32_t layer_id, uint64_t offset, uint8_t * dst, size_t count) {
        return producer.read(layer_id, offset, dst, count);
    };
    stream_write_callbacks writes;
    writes.write_bytes = [](const uint8_t *, size_t) { return false; }; // sink refuses everything
    CHECK_THROWS(stream_serialize(f.input, reads, writes, opts));

    // sink fails mid-stream
    int calls = 0;
    writes.write_bytes = [&](const uint8_t *, size_t) {
        ++calls;
        return calls != 100;
    };
    CHECK_THROWS(stream_serialize(f.input, reads, writes, opts));
    CHECK(calls == 100);
}

static void test_overflow_and_validation_before_write() {
    const fixture f = make_full_fixture();
    vector_sink sink;
    blob_producer producer;
    stream_write_callbacks writes;
    writes.write_bytes = [&](const uint8_t * bytes, size_t count) { return sink.write(bytes, count); };
    auto reads_ok = [&]() {
        stream_read_callbacks r;
        r.read_payload = [&](uint32_t layer_id, uint64_t offset, uint8_t * dst, size_t count) {
            return producer.read(layer_id, offset, dst, count);
        };
        return r;
    };

    // declared payload far beyond the sanity bound: fails before any read/write
    {
        stream_input bad = f.input;
        bad.layers[0].payload_bytes = uint64_t(1) << 63;
        CHECK_THROWS(stream_serialize(bad, reads_ok(), writes));
        CHECK(sink.calls == 0);
    }
    // metadata beyond the caller cap
    {
        stream_input bad = f.input;
        stream_options opts;
        opts.max_metadata_bytes = f.md.size() - 1;
        CHECK_THROWS(stream_serialize(bad, reads_ok(), writes, opts));
        CHECK(sink.calls == 0);
    }
    // total beyond the caller cap
    {
        stream_options opts;
        opts.max_total_bytes = 96 + f.md.size() + 101; // too small
        CHECK_THROWS(stream_serialize(f.input, reads_ok(), writes, opts));
        CHECK(sink.calls == 0);
    }
    // duplicate layer ids
    {
        stream_input bad = f.input;
        bad.layers[1].layer_id = bad.layers[0].layer_id;
        CHECK_THROWS(stream_serialize(bad, reads_ok(), writes));
        CHECK(sink.calls == 0);
    }
    // overlay empty in partial
    {
        stream_input bad = f.input;
        bad.flags = snapshot_flags::partial;
        bad.owner_epoch = 1;
        bad.layers[0].payload_mode = mode_partial_overlay;
        bad.layers[0].payload_bytes = 0;
        CHECK_THROWS(stream_serialize(bad, reads_ok(), writes));
        CHECK(sink.calls == 0);
    }
    // resident reference with declared bytes
    {
        stream_input bad = f.input;
        bad.flags = snapshot_flags::partial;
        bad.owner_epoch = 1;
        bad.layers[0].payload_mode = mode_resident_reference;
        bad.layers[0].payload_bytes = 64;
        CHECK_THROWS(stream_serialize(bad, reads_ok(), writes));
        CHECK(sink.calls == 0);
    }
    // missing write_bytes callback
    {
        stream_write_callbacks none;
        CHECK_THROWS(stream_serialize(f.input, reads_ok(), none));
    }
    // missing read_payload callback with payloads present
    {
        stream_read_callbacks none;
        CHECK_THROWS(stream_serialize(f.input, none, writes));
    }
    // transfer buffer out of range
    {
        stream_options opts;
        opts.transfer_buffer_bytes = 0;
        CHECK_THROWS(stream_serialize(f.input, reads_ok(), writes, opts));
        opts.transfer_buffer_bytes = (8u << 20) + 1;
        CHECK_THROWS(stream_serialize(f.input, reads_ok(), writes, opts));
    }
}

static void test_overflow_accounting() {
    const fixture f = make_full_fixture();
    vector_sink sink;
    blob_producer producer;
    stream_write_callbacks writes;
    writes.write_bytes = [&](const uint8_t * bytes, size_t count) { return sink.write(bytes, count); };
    auto reads_ok = [&]() {
        stream_read_callbacks r;
        r.read_payload = [&](uint32_t layer_id, uint64_t offset, uint8_t * dst, size_t count) {
            return producer.read(layer_id, offset, dst, count);
        };
        return r;
    };
    const uint64_t bound = uint64_t(1) << 42; // 4 TiB sanity bound

    // metadata_size = SIZE_MAX with a non-null dummy pointer: the accounting
    // must fail BEFORE any dereference of the metadata pointer
    {
        stream_input bad = f.input;
        bad.metadata_size = std::numeric_limits<size_t>::max();
        bad.metadata = reinterpret_cast<const uint8_t *>(uintptr_t(1)); // must not be dereferenced
        CHECK_THROWS(stream_serialize(bad, reads_ok(), writes));
        CHECK(sink.calls == 0);
    }
    // metadata exactly at the 4 TiB bound: no room left for anything
    {
        stream_input bad = f.input;
        bad.metadata_size = size_t(bound);
        CHECK_THROWS(stream_serialize(bad, reads_ok(), writes));
        CHECK(sink.calls == 0);
    }
    // payload exactly fills the bound minus header/blob/tail WITHOUT room for
    // its own descriptor: must fail instead of pushing total past the bound
    {
        stream_input bad = f.input;
        bad.metadata_size = f.md.size();
        bad.layers[0].payload_bytes = bound - 96 - f.md.size() - 8;
        CHECK_THROWS(stream_serialize(bad, reads_ok(), writes));
        CHECK(sink.calls == 0);
    }
    // payload fits exactly INCLUDING its descriptor (single-layer input):
    // accounting passes and pass 1 starts (one read, then the producer
    // truncates)
    {
        stream_input ok_in;
        ok_in.metadata_size = f.md.size();
        ok_in.metadata = f.md.data();
        ok_in.metadata_version = 1;
        ok_in.flags = snapshot_flags::full;
        layer_desc d0 = f.input.layers[0];
        d0.payload_bytes = bound - 96 - f.md.size() - 8 - 101;
        ok_in.layers.push_back(d0);
        stream_read_callbacks reads;
        int read_calls = 0;
        reads.read_payload = [&](uint32_t, uint64_t, uint8_t *, size_t) {
            ++read_calls;
            return size_t(0); // truncate immediately
        };
        CHECK_THROWS(stream_serialize(ok_in, reads, writes));
        CHECK(read_calls == 1);
        CHECK(sink.calls == 0);
    }
    // successive descriptor growth: 99 one-byte layers accumulate, then the
    // 100th layer's payload is one byte too large and must fail with zero
    // callbacks
    {
        const uint32_t n = 100;
        stream_input big;
        big.metadata_size = f.md.size();
        big.metadata = f.md.data();
        big.metadata_version = 1;
        big.flags = snapshot_flags::full;
        const uint64_t budget = bound - 96 - f.md.size() - 8;
        for (uint32_t id = 0; id + 1 < n; ++id) {
            big.layers.push_back(make_qx_layer(id, sx_q4_0, sx_q4_0, 1, 1));
        }
        // after 99 layers total = 96 + md + 8 + 99 * (101 + 1) = 96+md+8+10098
        const uint64_t remaining = budget - 10098;
        big.layers.push_back(make_qx_layer(n - 1, sx_q4_0, sx_q4_0, 1, remaining - 101 + 1));
        CHECK_THROWS(stream_serialize(big, reads_ok(), writes));
        CHECK(sink.calls == 0);
        CHECK(producer.calls.empty());
    }
}

static void test_on_fault_exactly_once() {
    const fixture f = make_full_fixture();
    vector_sink sink;
    blob_producer producer;
    for (const layer_blob & b : f.blobs) {
        producer.data[b.desc.layer_id] = b.data;
        producer.size[b.desc.layer_id] = b.size;
    }
    stream_write_callbacks writes;
    writes.write_bytes = [&](const uint8_t * bytes, size_t count) { return sink.write(bytes, count); };
    auto reads_ok = [&]() {
        stream_read_callbacks r;
        r.read_payload = [&](uint32_t layer_id, uint64_t offset, uint8_t * dst, size_t count) {
            return producer.read(layer_id, offset, dst, count);
        };
        return r;
    };
    auto run = [&](stream_input in, stream_read_callbacks rd, stream_write_callbacks wr,
                   stream_options opts) -> bool {
        int faults = 0;
        opts.on_fault = [&](const std::string & msg) {
            ++faults;
            CHECK(!msg.empty());
        };
        bool threw = false;
        try {
            stream_serialize(in, rd, wr, opts);
        } catch (const kv_mixed_error &) {
            threw = true;
        }
        CHECK(threw);
        CHECK(faults == 1); // exactly once, even for callback exceptions
        return threw && faults == 1;
    };

    // validation failures
    {
        stream_input bad = f.input;
        bad.flags = snapshot_flags::partial | snapshot_flags::body_only;
        CHECK(run(bad, reads_ok(), writes, {}));
    }
    {
        stream_input bad = f.input;
        bad.layers[0].kind = cache_kind::unknown;
        CHECK(run(bad, reads_ok(), writes, {}));
    }
    {
        stream_input bad = f.input;
        bad.layers[1].layer_id = bad.layers[0].layer_id;
        CHECK(run(bad, reads_ok(), writes, {}));
    }
    // producer throwing: wrapped into kv_mixed_error, on_fault once
    {
        stream_read_callbacks reads;
        reads.read_payload = [&](uint32_t, uint64_t, uint8_t *, size_t) -> size_t {
            throw std::runtime_error("producer boom");
        };
        CHECK(run(f.input, reads, writes, {}));
    }
    // sink returning false
    {
        stream_write_callbacks wr;
        wr.write_bytes = [](const uint8_t *, size_t) { return false; };
        CHECK(run(f.input, reads_ok(), wr, {}));
    }
    // sink throwing
    {
        stream_write_callbacks wr;
        wr.write_bytes = [](const uint8_t *, size_t) -> bool {
            throw std::runtime_error("sink boom");
        };
        CHECK(run(f.input, reads_ok(), wr, {}));
    }
}

static void test_body_only_cells_guard() {
    const fixture f = make_full_fixture();
    vector_sink sink;
    stream_write_callbacks writes;
    writes.write_bytes = [&](const uint8_t * bytes, size_t count) { return sink.write(bytes, count); };
    stream_read_callbacks reads;
    reads.read_payload = [](uint32_t, uint64_t, uint8_t *, size_t) { return size_t(0); };

    // body_only with cells: both APIs must reject; stream must not emit
    // n_used != 0 bytes that the parser would refuse
    stream_input bad;
    bad.flags = snapshot_flags::body_only;
    bad.layers = f.input.layers;
    bad.cells = f.cells;
    CHECK_THROWS(stream_serialize(bad, reads, writes));
    CHECK(sink.calls == 0);
    std::vector<layer_blob> blobs;
    for (size_t i = 0; i < f.blobs.size(); ++i) {
        blobs.push_back(f.blobs[i]);
    }
    std::vector<uint8_t> out;
    CHECK_THROWS(snapshot::serialize(blobs, bad.cells, 0, nullptr, 0,
                                     snapshot_flags::body_only, 0, 0, out));
}

// invalid-input matrix: both the streaming encoder and the owning serializer
// must reject the same invalid inputs (guards against validation drift)
static void test_validation_matrix_equivalence() {
    const fixture f = make_full_fixture();
    blob_producer producer;
    for (const layer_blob & b : f.blobs) {
        producer.data[b.desc.layer_id] = b.data;
        producer.size[b.desc.layer_id] = b.size;
    }
    vector_sink sink;
    stream_write_callbacks writes;
    stream_read_callbacks reads;
    reads.read_payload = [&](uint32_t layer_id, uint64_t offset, uint8_t * dst, size_t count) {
        return producer.read(layer_id, offset, dst, count);
    };
    std::vector<uint8_t> out;

    // blobs derived from the (possibly mutated) input descriptors, with the
    // fixture payload buffers attached by layer id
    auto blobs_for = [&](const stream_input & in) {
        std::vector<layer_blob> blobs;
        for (const layer_desc & d : in.layers) {
            layer_blob b;
            b.desc = d;
            const auto it = producer.data.find(d.layer_id);
            if (it != producer.data.end()) {
                b.data = it->second;
                b.size = producer.size[d.layer_id];
            }
            blobs.push_back(b);
        }
        return blobs;
    };

    auto both_reject = [&](const stream_input & in, snapshot_flags flags,
                           uint64_t epoch, uint64_t oid, const char * what) {
        sink.calls = 0; // fresh sink per case
        int faults = 0;
        stream_options opts;
        opts.on_fault = [&](const std::string &) { ++faults; };
        bool stream_threw = false;
        try {
            stream_serialize(in, reads, writes, opts);
        } catch (const kv_mixed_error &) {
            stream_threw = true;
        }
        CHECK(stream_threw);
        CHECK(faults == 1);
        CHECK(sink.calls == 0); // no bytes emitted for invalid input
        bool ser_threw = false;
        try {
            snapshot::serialize(blobs_for(in), in.cells, in.metadata_version,
                                in.metadata, in.metadata_size, flags, epoch, oid, out);
        } catch (const kv_mixed_error &) {
            ser_threw = true;
        }
        CHECK(ser_threw);
        if (!stream_threw || !ser_threw) {
            std::fprintf(stderr, "matrix case failed: %s\n", what);
        }
    };
    writes.write_bytes = [&](const uint8_t * bytes, size_t count) {
        sink.out.insert(sink.out.end(), bytes, bytes + count);
        ++sink.calls;
        return true;
    };

    // flags
    {
        stream_input in = f.input;
        in.flags = snapshot_flags::partial | snapshot_flags::body_only;
        both_reject(in, snapshot_flags::partial | snapshot_flags::body_only, 0, 0, "flags partial|body_only");
    }
    {
        stream_input in = f.input;
        in.flags = snapshot_flags(4);
        both_reject(in, snapshot_flags(4), 0, 0, "unknown flag bit");
    }
    {
        stream_input in = f.input;
        in.flags = snapshot_flags::partial;
        in.owner_epoch = 0; // partial without owner epoch
        both_reject(in, snapshot_flags::partial, 0, 0, "partial without epoch");
    }
    // descriptors
    {
        stream_input in = f.input;
        in.layers[0].kind = cache_kind::unknown;
        both_reject(in, snapshot_flags::full, 0, 0, "unknown kind");
    }
    {
        stream_input in = f.input;
        in.layers[1].kvarn_domain = uint8_t(kvarn_domain::rotated); // standard layer
        both_reject(in, snapshot_flags::full, 0, 0, "standard with kvarn domain");
    }
    {
        stream_input in = f.input;
        in.layers[1].n_head_kv = 0;
        both_reject(in, snapshot_flags::full, 0, 0, "missing geometry");
    }
    {
        stream_input in = f.input;
        in.layers[1].payload_mode = 9;
        both_reject(in, snapshot_flags::full, 0, 0, "unknown payload mode");
    }
    {
        stream_input in = f.input;
        in.layers[1].layer_id = in.layers[0].layer_id;
        both_reject(in, snapshot_flags::full, 0, 0, "duplicate layer ids");
    }
    // cells
    {
        stream_input in = f.input;
        in.cells[0].seq_ids.assign(257, 0);
        both_reject(in, snapshot_flags::full, 0, 0, "cell seq > 256");
    }
    {
        stream_input in = f.input;
        in.cells[2].seq_ids = { 0 }; // pos == -1 with seq ids
        both_reject(in, snapshot_flags::full, 0, 0, "empty cell with seqs");
    }
    {
        stream_input in = f.input;
        in.flags = snapshot_flags::body_only;
        in.metadata = nullptr;
        in.metadata_size = 0;
        both_reject(in, snapshot_flags::body_only, 0, 0, "body_only with cells");
    }
    // metadata
    {
        stream_input in = f.input;
        in.metadata = nullptr;
        in.metadata_size = 0;
        both_reject(in, snapshot_flags::full, 0, 0, "full without blob");
    }
    {
        stream_input in = f.input;
        in.flags = snapshot_flags::body_only;
        in.metadata = f.md.data();
        in.metadata_size = f.md.size();
        both_reject(in, snapshot_flags::body_only, 0, 0, "body_only with blob");
    }
}

static void test_empty_payload_layer() {
    fixture f;
    f.p0.resize(0);
    f.md.resize(32);
    fill_pattern(f.md, 41);
    layer_desc d = make_qx_layer(1, sx_q4_0, sx_q4_0, 0, 0); // empty payload
    std::vector<layer_blob> blobs = { { d, nullptr, 0 } };
    f.blobs = blobs;
    f.input.layers = { d };
    f.input.metadata_version = 1;
    f.input.metadata = f.md.data();
    f.input.metadata_size = f.md.size();
    f.input.flags = snapshot_flags::full;
    CHECK(snapshot::serialize(blobs, {}, 1, f.md.data(), f.md.size(),
                              snapshot_flags::full, 0, 0, f.reference));

    vector_sink sink;
    blob_producer producer;
    CHECK(stream_fixture(f, 64, sink, producer));
    CHECK(sink.out == f.reference);
    CHECK(producer.calls[1] == 0); // no bytes to read
}

#if defined(__linux__)
static void test_file_backing() {
    const fixture f = make_full_fixture();
    // write the snapshot to a real temp file
    char path[] = "/tmp/opencode/kvms-backing-XXXXXX";
    const int fd = ::mkstemp(path);
    CHECK(fd >= 0);
    CHECK(::write(fd, f.reference.data(), f.reference.size()) == (ssize_t) f.reference.size());
    ::close(fd);

    {
        auto backing = std::make_shared<kv_mixed_file_backing>(path);
        CHECK(backing->size() == f.reference.size());
        CHECK(backing->data() != nullptr);

        // parse_view over the mapped file: payload points into the mapping;
        // the snapshot retains the shared_ptr (no raw token)
        snapshot s = snapshot::parse_view(backing->data(), backing->size(), backing);
        CHECK(s.is_borrowed());
        CHECK(s.backing() == backing);
        const uint8_t * p0 = s.payload(0);
        CHECK(p0 >= backing->data());
        CHECK(p0 + s.payload_size(0) <= backing->data() + backing->size());
        CHECK(std::memcmp(p0, f.p0.data(), f.p0.size()) == 0);
        CHECK(s.payload_size(1) == f.p1.size());
        CHECK(s.payload_size(2) == f.p2.size());

        // weak lifetime: dropping the caller handle keeps the mapping alive
        // through the snapshot (and its moves)
        std::weak_ptr<kv_mixed_file_backing> weak = backing;
        backing.reset();
        CHECK(!weak.expired());
        snapshot moved = std::move(s);
        CHECK(std::memcmp(moved.payload(0), f.p0.data(), f.p0.size()) == 0);
        moved = snapshot{}; // destroy the snapshot: mapping released
        CHECK(weak.expired());
    }

    // truncated file rejected by the backing
    {
        char small[] = "/tmp/opencode/kvms-small-XXXXXX";
        const int fd2 = ::mkstemp(small);
        CHECK(fd2 >= 0);
        const uint8_t bytes[64] = { 0 };
        CHECK(::write(fd2, bytes, sizeof(bytes)) == (ssize_t) sizeof(bytes));
        ::close(fd2);
        CHECK_THROWS(kv_mixed_file_backing(small));
        ::unlink(small);
    }
    // size cap rejected by the backing
    {
        parse_options opts;
        opts.max_total_bytes = f.reference.size() - 1;
        CHECK_THROWS(kv_mixed_file_backing(path, opts));
    }
    // non-regular files are rejected (device node, not a snapshot file)
    CHECK_THROWS(kv_mixed_file_backing("/dev/null"));
    ::unlink(path);
}
#endif


static void test_stream_parse_view_integration() {
    // streamed output is directly consumable by parse_view (borrowed mode)
    const fixture f = make_full_fixture();
    std::vector<uint8_t> streamed;
    blob_producer producer;
    for (const layer_blob & b : f.blobs) {
        producer.data[b.desc.layer_id] = b.data;
        producer.size[b.desc.layer_id] = b.size;
    }
    stream_options opts;
    opts.transfer_buffer_bytes = 128;
    stream_read_callbacks reads;
    reads.read_payload = [&](uint32_t layer_id, uint64_t offset, uint8_t * dst, size_t count) {
        return producer.read(layer_id, offset, dst, count);
    };
    stream_write_callbacks writes;
    writes.write_bytes = [&](const uint8_t * bytes, size_t count) {
        streamed.insert(streamed.end(), bytes, bytes + count);
        return true;
    };
    CHECK(stream_serialize(f.input, reads, writes, opts));
    auto owner = std::make_shared<std::vector<uint8_t>>(streamed);
    snapshot v = snapshot::parse_view(owner->data(), owner->size(), owner);
    CHECK(v.is_borrowed());
    CHECK(v.backing() == owner);
    CHECK(std::memcmp(v.payload(0), f.p0.data(), f.p0.size()) == 0);
    CHECK(std::memcmp(v.payload(1), f.p1.data(), f.p1.size()) == 0);
}

int main() {
    test_stream_equals_serialize_all_fixtures();
    test_chunk_boundaries();
    test_stream_roundtrip_parse();
    test_reference_layer_never_read();
    test_large_virtual_payload_bounded_scratch();
    test_producer_failure();
    test_producer_mutation();
    test_sink_failure();
    test_overflow_and_validation_before_write();
    test_overflow_accounting();
    test_on_fault_exactly_once();
    test_body_only_cells_guard();
    test_validation_matrix_equivalence();
    test_empty_payload_layer();
#if defined(__linux__)
    test_file_backing();
#endif
    test_stream_parse_view_integration();

    std::printf("%s: %d checks, %d failures\n", g_failures == 0 ? "PASS" : "FAIL", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}