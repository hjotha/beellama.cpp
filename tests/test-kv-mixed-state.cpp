// Standalone deterministic CPU tests for the mixed KV snapshot codec.
// Build without CMake:
//   g++ -std=c++17 -O2 -Wall -Wextra -I src
//     tests/test-kv-mixed-state.cpp src/llama-kv-mixed-state.cpp
//     -o /tmp/opencode/test-kv-mixed-state

#include "llama-kv-mixed-state.h"

#include <algorithm>
#include "../src/llama-state-rollback.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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

// Qwen-style standard layer: K Q4_0 rotated Hadamard 256, V Q8_0 rotated
// Hadamard 64, n_head_kv 4.
static layer_desc make_qx_layer(uint32_t id, uint16_t k_type, uint16_t v_type,
                                uint32_t rows, uint64_t bytes) {
    layer_desc d;
    d.layer_id = id;
    d.kind = cache_kind::standard_qx;
    d.k_type = k_type;
    d.v_type = v_type;
    d.kvarn_domain = 0;
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

static bool descs_equal(const layer_desc & a, const layer_desc & b) {
    return a.layer_id == b.layer_id && a.kind == b.kind &&
           a.k_type == b.k_type && a.v_type == b.v_type &&
           a.k_bits == b.k_bits && a.v_bits == b.v_bits &&
           a.kvarn_domain == b.kvarn_domain &&
           a.k_rot == b.k_rot && a.v_rot == b.v_rot &&
           a.k_rot_width == b.k_rot_width && a.v_rot_width == b.v_rot_width &&
           a.layout == b.layout && a.owner == b.owner && a.v_trans == b.v_trans &&
           a.tail_type == b.tail_type && a.token_group == b.token_group &&
           a.record_dim == b.record_dim && a.head_dim_k == b.head_dim_k &&
           a.head_dim_v == b.head_dim_v && a.head_slices == b.head_slices &&
           a.n_head_kv == b.n_head_kv && a.n_stream == b.n_stream &&
           a.k_stride == b.k_stride && a.v_stride == b.v_stride &&
           a.payload_mode == b.payload_mode &&
           a.payload_cells == b.payload_cells && a.payload_rows == b.payload_rows &&
           a.payload_bytes == b.payload_bytes;
}

// XXH64 is used by the codec; recompute here for header patching in negative tests.
#define XXH_INLINE_ALL
#include "../vendor/hash/xxhash/xxhash.h"

static void patch_header_crc(std::vector<uint8_t> & buf) {
    CHECK(buf.size() >= 96);
    std::vector<uint8_t> h(buf.begin(), buf.begin() + 96);
    std::memset(h.data() + 88, 0, 8); // crc field is zeroed when hashed
    const uint64_t crc = XXH64(h.data(), 96, 0);
    for (int i = 0; i < 8; ++i) {
        buf[88 + i] = uint8_t((crc >> (8 * i)) & 0xFF);
    }
}

// F4 v2 header is 104 bytes with the crc at offset 96.
static void patch_header_crc_v2(std::vector<uint8_t> & buf) {
    CHECK(buf.size() >= 104);
    std::vector<uint8_t> h(buf.begin(), buf.begin() + 104);
    std::memset(h.data() + 96, 0, 8);
    const uint64_t crc = XXH64(h.data(), 104, 0);
    for (int i = 0; i < 8; ++i) {
        buf[96 + i] = uint8_t((crc >> (8 * i)) & 0xFF);
    }
}

static void write_le_u32_at(std::vector<uint8_t> & b, size_t off, uint64_t v) {
    for (int i = 0; i < 4; ++i) {
        b[off + i] = uint8_t((v >> (8 * i)) & 0xFF);
    }
}

static void write_le_u64_at(std::vector<uint8_t> & b, size_t off, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        b[off + i] = uint8_t((v >> (8 * i)) & 0xFF);
    }
}

static uint64_t read_le_u64_at(const std::vector<uint8_t> & b, size_t off) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) {
        v = (v << 8) | b[off + i];
    }
    return v;
}

// wrapper-owned opaque shared-cache metadata (allocation groups, stage
// slots, checkpoints, ...): opaque to the codec, versioned by the wrapper
static void make_metadata(std::vector<uint8_t> & m, uint32_t seed) {
    m.resize(384);
    fill_pattern(m, seed);
}

// ---------------------------------------------------------------------------

static void test_roundtrip_mixed() {
    std::vector<uint8_t> p0(4096), p1(2048), p2(512);
    fill_pattern(p0, 1);
    fill_pattern(p1, 2);
    fill_pattern(p2, 3);
    std::vector<uint8_t> md;
    make_metadata(md, 4);

    std::vector<layer_blob> layers;
    layers.push_back({ make_kvarn_layer(0, kv4v5, 4, 5, uint8_t(kvarn_domain::rotated), 8192, 64, p0.size()), p0.data(), p0.size() });
    layers.push_back({ make_qx_layer(3, sx_q4_0, sx_q8_0, 128, p1.size()), p1.data(), p1.size() });
    layers.push_back({ make_qx_layer(7, sx_q6_0, sx_q6_0, 32, p2.size()), p2.data(), p2.size() });

    std::vector<cell_entry> cells(5);
    cells[0].pos = 0;  cells[0].tok = 100; cells[0].seq_ids = { 0 };
    cells[1].pos = 1;  cells[1].tok = 101; cells[1].seq_ids = { 0, 1 };
    cells[2].pos = -1; // hole
    cells[3].pos = 3;  cells[3].x = 7; cells[3].y = 9; cells[3].tok = 103; cells[3].seq_ids = { 1 };
    cells[4].pos = 4;  cells[4].tok = 104; cells[4].seq_ids = { 0 };

    std::vector<uint8_t> buf;
    CHECK(snapshot::serialize(layers, cells, 3, md.data(), md.size(),
                              snapshot_flags::full, 0, 0, buf));

    snapshot s = snapshot::parse(buf.data(), buf.size());
    CHECK(s.version() == 1);
    CHECK(uint32_t(s.flags()) == uint32_t(snapshot_flags::full));
    CHECK(s.layers().size() == 3);
    CHECK(s.n_used() == 5);
    CHECK(descs_equal(s.layers()[0], layers[0].desc));
    CHECK(descs_equal(s.layers()[1], layers[1].desc));
    CHECK(descs_equal(s.layers()[2], layers[2].desc));
    // rotations and per-tensor geometry survive the roundtrip
    CHECK(s.layers()[1].k_rot == uint8_t(rotation::hadamard) && s.layers()[1].k_rot_width == 256);
    CHECK(s.layers()[1].v_rot == uint8_t(rotation::hadamard) && s.layers()[1].v_rot_width == 64);
    CHECK(s.layers()[1].head_dim_k == 256 && s.layers()[1].head_dim_v == 64 && s.layers()[1].n_head_kv == 4);
    CHECK(s.layers()[0].kvarn_domain == uint8_t(kvarn_domain::rotated));
    // opaque metadata blob preserved once
    CHECK(s.metadata_version() == 3);
    CHECK(s.metadata_size() == md.size());
    CHECK(std::memcmp(s.metadata_blob(), md.data(), md.size()) == 0);
    // payloads exact
    CHECK(s.payload_size(0) == p0.size() && std::memcmp(s.payload(0), p0.data(), p0.size()) == 0);
    CHECK(s.payload_size(1) == p1.size() && std::memcmp(s.payload(1), p1.data(), p1.size()) == 0);
    CHECK(s.payload_size(2) == p2.size() && std::memcmp(s.payload(2), p2.data(), p2.size()) == 0);
    // cell summary appears exactly once
    CHECK(s.cells().size() == 5);
    CHECK(s.cells()[1].seq_ids.size() == 2);
    CHECK(s.cells()[2].pos == -1 && s.cells()[2].seq_ids.empty());
    CHECK(s.cells()[3].x == 7 && s.cells()[3].y == 9 && s.cells()[3].tok == 103);

    // strict expected set built from the destination's known layers
    expected_snapshot exp;
    for (const layer_blob & b : layers) {
        exp.layers.push_back(expected_layer::from_desc(b.desc));
    }
    exp.capacity_cells = 8192;
    exp.metadata_version = 3;
    s.validate_expected_snapshot(exp);

    // transactional commit: stage then finish
    int staged_meta = 0, staged_layers = 0, finished = 0, aborted = 0;
    s.commit({
        [&](uint32_t id, uint32_t, const layer_desc &, const uint8_t * p, size_t n) {
            ++staged_layers;
            if (id == 0) CHECK(n == p0.size() && std::memcmp(p, p0.data(), n) == 0);
            if (id == 3) CHECK(n == p1.size() && std::memcmp(p, p1.data(), n) == 0);
            if (id == 7) CHECK(n == p2.size() && std::memcmp(p, p2.data(), n) == 0);
            return true;
        },
        [&](uint32_t ver, const uint8_t * m, size_t n) {
            ++staged_meta;
            CHECK(ver == 3 && n == md.size() && std::memcmp(m, md.data(), n) == 0);
            return true;
        },
        [&]() { ++finished; },
        [&]() { ++aborted; },
    });
    CHECK(staged_meta == 1);
    CHECK(staged_layers == 3);
    CHECK(finished == 1);
    CHECK(aborted == 0);
    // ticket consumed: a second commit attempt fails
    CHECK_THROWS(s.commit({}));

    CHECK(s.layers()[0].payload_checksum == XXH64(p0.data(), p0.size(), 0));
}

static void test_empty_prefix() {
    std::vector<uint8_t> p(64);
    fill_pattern(p, 9);
    std::vector<uint8_t> md;
    make_metadata(md, 10);
    std::vector<uint8_t> buf;
    CHECK(snapshot::serialize({ { make_qx_layer(1, sx_q4_0, sx_q4_0, 2, p.size()), p.data(), p.size() } },
                              {}, 1, md.data(), md.size(), snapshot_flags::full, 0, 0, buf));
    snapshot s = snapshot::parse(buf.data(), buf.size());
    CHECK(s.n_used() == 0);
    CHECK(s.cells().empty());
    CHECK(s.metadata_size() == md.size());
    expected_snapshot exp;
    exp.layers.push_back(expected_layer::from_desc(s.layers()[0]));
    exp.capacity_cells = 0;
    exp.metadata_version = 1;
    s.validate_expected_snapshot(exp);
    int staged = 0, finished = 0;
    s.commit({ [&](uint32_t, uint32_t, const layer_desc &, const uint8_t *, size_t) { ++staged; return true; },
               [&](uint32_t, const uint8_t *, size_t) { return true; },
               [&]() { ++finished; }, [&]() {} });
    CHECK(staged == 1 && finished == 1);
}

static void test_validate_strict() {
    std::vector<uint8_t> p(128);
    fill_pattern(p, 11);
    std::vector<uint8_t> md;
    make_metadata(md, 12);
    layer_desc d = make_qx_layer(2, sx_q4_0, sx_q4_0, 8, p.size());
    std::vector<uint8_t> buf;
    CHECK(snapshot::serialize({ { d, p.data(), p.size() } }, {}, 1, md.data(), md.size(),
                              snapshot_flags::full, 0, 0, buf));

    auto fresh = [&]() { return snapshot::parse(buf.data(), buf.size()); };
    auto ok = [&](expected_layer e) {
        snapshot s = fresh();
        expected_snapshot exp;
        exp.layers.push_back(e);
        exp.capacity_cells = 8;
        exp.metadata_version = 1;
        s.validate_expected_snapshot(exp);
    };

    ok(expected_layer::from_desc(d)); // exact match passes

    auto expect_throw = [&](expected_layer e) { CHECK_THROWS(ok(e)); };
    expected_layer bad;
    bad = expected_layer::from_desc(d); bad.kind = cache_kind::kvarn;                expect_throw(bad);
    bad = expected_layer::from_desc(d); bad.k_type = sx_q8_0;                        expect_throw(bad);
    bad = expected_layer::from_desc(d); bad.v_type = sx_q8_0;                        expect_throw(bad);
    bad = expected_layer::from_desc(d); bad.k_rot = uint8_t(rotation::none);         expect_throw(bad);
    bad = expected_layer::from_desc(d); bad.k_rot_width = 512;                       expect_throw(bad);
    bad = expected_layer::from_desc(d); bad.v_rot_width = 32;                        expect_throw(bad);
    bad = expected_layer::from_desc(d); bad.head_dim_k = 128;                        expect_throw(bad);
    bad = expected_layer::from_desc(d); bad.head_dim_v = 128;                        expect_throw(bad);
    bad = expected_layer::from_desc(d); bad.n_head_kv = 8;                           expect_throw(bad);
    bad = expected_layer::from_desc(d); bad.owner = owner_target;                    expect_throw(bad);
    bad = expected_layer::from_desc(d); bad.kvarn_domain = uint8_t(kvarn_domain::rotated); expect_throw(bad);
    bad = expected_layer::from_desc(d); bad.payload_rows = 999;                      expect_throw(bad);
    bad = expected_layer::from_desc(d); bad.layer_id = 5;                            expect_throw(bad);

    // capacity: 255 is a plain number, NOT a wildcard; uses a snapshot with 8 cells
    std::vector<cell_entry> cells8(8);
    for (int i = 0; i < 8; ++i) {
        cells8[i].pos = i;
        cells8[i].tok = i;
        cells8[i].seq_ids = { 0 };
    }
    std::vector<uint8_t> cbuf;
    CHECK(snapshot::serialize({ { d, p.data(), p.size() } }, cells8, 1, md.data(), md.size(),
                              snapshot_flags::full, 0, 0, cbuf));
    {
        snapshot s = snapshot::parse(cbuf.data(), cbuf.size());
        expected_snapshot exp;
        exp.layers.push_back(expected_layer::from_desc(d));
        exp.capacity_cells = 255; // n_used = 8 <= 255: must pass
        exp.metadata_version = 1;
        s.validate_expected_snapshot(exp);
    }
    {
        snapshot s = snapshot::parse(cbuf.data(), cbuf.size());
        expected_snapshot exp;
        exp.layers.push_back(expected_layer::from_desc(d));
        exp.capacity_cells = 7; // n_used = 8 > 7: must fail
        exp.metadata_version = 1;
        CHECK_THROWS(s.validate_expected_snapshot(exp));
        CHECK_THROWS(s.commit({})); // failed validation leaves commit blocked
    }
    // capacity must be specified
    {
        snapshot s = fresh();
        expected_snapshot exp;
        exp.layers.push_back(expected_layer::from_desc(d));
        exp.metadata_version = 1;
        CHECK_THROWS(s.validate_expected_snapshot(exp));
    }
    // metadata schema version is REQUIRED for full snapshots and must match
    {
        snapshot s = fresh();
        expected_snapshot exp;
        exp.layers.push_back(expected_layer::from_desc(d));
        exp.capacity_cells = 8;
        CHECK_THROWS(s.validate_expected_snapshot(exp)); // version missing
        exp.metadata_version = 2;
        CHECK_THROWS(s.validate_expected_snapshot(exp)); // version mismatch
        CHECK_THROWS(s.commit({})); // incompatible blob never reaches commit
    }
    // duplicate expected layer ids cannot match the same actual layer
    {
        snapshot s = fresh();
        expected_snapshot exp;
        exp.layers.push_back(expected_layer::from_desc(d));
        exp.layers.push_back(expected_layer::from_desc(d));
        exp.capacity_cells = 8;
        exp.metadata_version = 1;
        CHECK_THROWS(s.validate_expected_snapshot(exp));
    }
    // layer count mismatch
    {
        snapshot s = fresh();
        expected_snapshot exp;
        exp.capacity_cells = 8;
        exp.metadata_version = 1;
        CHECK_THROWS(s.validate_expected_snapshot(exp));
    }
    // validate is single-shot; a successful validation permits exactly one commit
    {
        snapshot s = fresh();
        expected_snapshot exp;
        exp.layers.push_back(expected_layer::from_desc(d));
        exp.capacity_cells = 8;
        exp.metadata_version = 1;
        s.validate_expected_snapshot(exp);
        CHECK_THROWS(s.validate_expected_snapshot(exp)); // second attempt fails and invalidates
        int staged = 0;
        CHECK_THROWS(s.commit({ [&](uint32_t, uint32_t, const layer_desc &, const uint8_t *, size_t) { ++staged; return true; },
                                [&](uint32_t, const uint8_t *, size_t) { return true; },
                                [&]() {}, [&]() {} }));
        CHECK(staged == 0); // the invalidated ticket never commits
    }
    // exactly one commit after a single successful validation
    {
        snapshot s = fresh();
        expected_snapshot exp;
        exp.layers.push_back(expected_layer::from_desc(d));
        exp.capacity_cells = 8;
        exp.metadata_version = 1;
        s.validate_expected_snapshot(exp);
        int staged = 0;
        s.commit({ [&](uint32_t, uint32_t, const layer_desc &, const uint8_t *, size_t) { ++staged; return true; },
                   [&](uint32_t, const uint8_t *, size_t) { return true; },
                   [&]() {}, [&]() {} });
        CHECK(staged == 1);
        CHECK_THROWS(s.commit({})); // ticket consumed by the successful attempt
    }
    // a FAILED second validation invalidates the ticket: commit must fail
    {
        snapshot s = fresh();
        expected_snapshot exp_ok;
        exp_ok.layers.push_back(expected_layer::from_desc(d));
        exp_ok.capacity_cells = 8;
        exp_ok.metadata_version = 1;
        s.validate_expected_snapshot(exp_ok); // success
        expected_snapshot exp_bad;
        exp_bad.layers.push_back(expected_layer::from_desc(d));
        exp_bad.layers[0].k_type = sx_q8_0; // mismatch
        exp_bad.capacity_cells = 8;
        exp_bad.metadata_version = 1;
        CHECK_THROWS(s.validate_expected_snapshot(exp_bad)); // second attempt fails...
        int staged = 0;
        CHECK_THROWS(s.commit({ [&](uint32_t, uint32_t, const layer_desc &, const uint8_t *, size_t) { ++staged; return true; },
                                [&](uint32_t, const uint8_t *, size_t) { return true; },
                                [&]() {}, [&]() {} }));
        CHECK(staged == 0); // ...and invalidates the ticket: nothing commits
    }
}

static void test_owner_values_are_exact() {
    std::vector<uint8_t> p(64);
    fill_pattern(p, 21);
    std::vector<uint8_t> md;
    make_metadata(md, 22);
    layer_desc d = make_qx_layer(1, sx_q4_0, sx_q4_0, 4, p.size());
    d.payload_mode = mode_resident_reference;
    d.payload_bytes = 0;
    std::vector<uint8_t> buf;

    // owner_id 255 and 65535 are plain values, NOT wildcards
    for (uint64_t oid : { uint64_t(255), uint64_t(65535) }) {
        buf.clear();
        CHECK(snapshot::serialize({ { d, nullptr, 0 } }, {}, 1, md.data(), md.size(),
                                  snapshot_flags::partial, 42, oid, buf));
        snapshot s = snapshot::parse(buf.data(), buf.size());
        expected_snapshot exp;
        exp.layers.push_back(expected_layer::from_desc(s.layers()[0]));
        exp.capacity_cells = 4;
        exp.metadata_version = 1;
        exp.owner_epoch = 42;
        exp.owner_id = oid;
        s.validate_expected_snapshot(exp); // exact owner passes
    }
    // and they must NOT match a different value
    buf.clear();
    CHECK(snapshot::serialize({ { d, nullptr, 0 } }, {}, 1, md.data(), md.size(),
                              snapshot_flags::partial, 42, 255, buf));
    {
        snapshot s = snapshot::parse(buf.data(), buf.size());
        expected_snapshot exp;
        exp.layers.push_back(expected_layer::from_desc(s.layers()[0]));
        exp.capacity_cells = 4;
        exp.metadata_version = 1;
        exp.owner_epoch = 42;
        exp.owner_id = 254;
        CHECK_THROWS(s.validate_expected_snapshot(exp));
    }
    // owner_epoch 65535 exact
    buf.clear();
    CHECK(snapshot::serialize({ { d, nullptr, 0 } }, {}, 1, md.data(), md.size(),
                              snapshot_flags::partial, 65535, 7, buf));
    {
        snapshot s = snapshot::parse(buf.data(), buf.size());
        expected_snapshot exp;
        exp.layers.push_back(expected_layer::from_desc(s.layers()[0]));
        exp.capacity_cells = 4;
        exp.metadata_version = 1;
        exp.owner_epoch = 65535;
        exp.owner_id = 7;
        s.validate_expected_snapshot(exp);
    }
    // full snapshots: optional owner check is exact when provided
    buf.clear();
    layer_desc d_full = make_qx_layer(1, sx_q4_0, sx_q4_0, 4, p.size());
    CHECK(snapshot::serialize({ { d_full, p.data(), p.size() } }, {}, 1, md.data(), md.size(),
                              snapshot_flags::full, 99, 0, buf));
    {
        snapshot s = snapshot::parse(buf.data(), buf.size());
        expected_snapshot exp;
        exp.layers.push_back(expected_layer::from_desc(d_full));
        exp.capacity_cells = 4;
        exp.metadata_version = 1;
        exp.owner_epoch = 98;
        CHECK_THROWS(s.validate_expected_snapshot(exp));
        exp.owner_epoch = 99;
        s.validate_expected_snapshot(exp);
    }
}

static void test_partial_owner_contract() {
    std::vector<uint8_t> p(256);
    fill_pattern(p, 31);
    std::vector<uint8_t> md;
    make_metadata(md, 32);
    // mixed partial: KVarN overlay bytes MUST travel (stages/tail/fixups are
    // mutable state), standard Qx body is a resident reference
    layer_desc k = make_kvarn_layer(1, kv4v5, 4, 5, uint8_t(kvarn_domain::rotated), 32, 2, p.size());
    k.payload_mode = mode_partial_overlay;
    layer_desc q = make_qx_layer(2, sx_q4_0, sx_q4_0, 32, 0);
    q.payload_mode = mode_resident_reference;
    q.payload_bytes = 0;
    std::vector<layer_blob> layers = { { k, p.data(), p.size() }, { q, nullptr, 0 } };
    std::vector<uint8_t> buf;
    CHECK(snapshot::serialize(layers, {}, 2, md.data(), md.size(),
                              snapshot_flags::partial, 42, 7, buf));
    CHECK(buf.size() == 96 + md.size() + 202 + p.size() + 8); // header+blob+2 desc+overlay+tail

    // overlay/reference misuse must be rejected at serialize
    layer_desc q_bad = q;
    q_bad.payload_bytes = 64; // reference cannot declare bytes
    CHECK_THROWS(snapshot::serialize({ { q_bad, nullptr, 0 } }, {}, 2, md.data(), md.size(),
                                     snapshot_flags::partial, 42, 7, buf));
    layer_desc q2 = q;
    q2.payload_mode = mode_partial_overlay;
    CHECK_THROWS(snapshot::serialize({ { q2, nullptr, 0 } }, {}, 2, md.data(), md.size(),
                                     snapshot_flags::partial, 42, 7, buf)); // overlay must be non-empty
    layer_desc q3 = q;
    q3.payload_mode = mode_full; // full mode forbidden in partial snapshots
    CHECK_THROWS(snapshot::serialize({ { q3, nullptr, 0 } }, {}, 2, md.data(), md.size(),
                                     snapshot_flags::partial, 42, 7, buf));
    // full snapshot layers must be mode_full
    CHECK_THROWS(snapshot::serialize({ { q, nullptr, 0 } }, {}, 2, md.data(), md.size(),
                                     snapshot_flags::full, 0, 0, buf));
    // full/partial without the metadata blob must be rejected
    CHECK_THROWS(snapshot::serialize(layers, {}, 2, nullptr, 0, snapshot_flags::partial, 42, 7, buf));
    CHECK_THROWS(snapshot::serialize({ { k, p.data(), p.size() } }, {}, 2, nullptr, 0,
                                     snapshot_flags::full, 0, 0, buf));

    snapshot s = snapshot::parse(buf.data(), buf.size());
    CHECK(uint32_t(s.flags() & snapshot_flags::partial) != 0);
    CHECK(s.owner_epoch() == 42);
    CHECK(s.owner_id() == 7);
    CHECK(s.layers()[0].payload_mode == mode_partial_overlay);
    CHECK(s.payload_size(0) == p.size());
    CHECK(std::memcmp(s.payload(0), p.data(), p.size()) == 0); // overlay preserved
    CHECK(s.layers()[1].payload_mode == mode_resident_reference);
    CHECK(s.payload_size(1) == 0); // Qx body is a resident reference
    CHECK(s.metadata_size() == md.size());

    expected_snapshot exp;
    exp.layers.push_back(expected_layer::from_desc(s.layers()[0]));
    exp.layers.push_back(expected_layer::from_desc(s.layers()[1]));
    exp.capacity_cells = 32;
    exp.metadata_version = 2;
    // owner fields are REQUIRED for partial
    CHECK_THROWS(s.validate_expected_snapshot(exp));
    exp.owner_epoch = 43;
    exp.owner_id = 7;
    CHECK_THROWS(s.validate_expected_snapshot(exp)); // wrong epoch
    exp.owner_epoch = 42;
    exp.owner_id = 8;
    CHECK_THROWS(s.validate_expected_snapshot(exp)); // wrong owner id
    exp.owner_id = 7;
    s.validate_expected_snapshot(exp); // same-owner restore accepted

    int staged_meta = 0, staged_layers = 0, finished = 0, aborted = 0;
    s.commit({
        [&](uint32_t id, uint32_t, const layer_desc &, const uint8_t * ptr, size_t n) {
            ++staged_layers;
            if (id == 1) {
                CHECK(ptr != nullptr && n == p.size()); // overlay bytes staged
            } else {
                CHECK(ptr == nullptr && n == 0); // reference has no bytes
            }
            return true;
        },
        [&](uint32_t, const uint8_t * m [[maybe_unused]], size_t n) {
            ++staged_meta;
            CHECK(n == md.size());
            return true;
        },
        [&]() { ++finished; },
        [&]() { ++aborted; },
    });
    CHECK(staged_meta == 1);
    CHECK(staged_layers == 2);
    CHECK(finished == 1);
    CHECK(aborted == 0);
}

static void test_body_only() {
    std::vector<uint8_t> p(128);
    fill_pattern(p, 41);
    std::vector<uint8_t> md;
    make_metadata(md, 42);
    std::vector<layer_blob> layers = { { make_qx_layer(4, sx_q8_0, sx_q8_0, 16, p.size()), p.data(), p.size() } };

    std::vector<uint8_t> buf;
    CHECK(snapshot::serialize(layers, {}, 0, nullptr, 0, snapshot_flags::body_only, 0, 0, buf));
    // body_only with metadata blob or cell metadata must be rejected
    CHECK_THROWS(snapshot::serialize(layers, {}, 1, md.data(), md.size(), snapshot_flags::body_only, 0, 0, buf));
    std::vector<cell_entry> cells(3);
    cells[0].pos = 0; cells[0].seq_ids = { 0 };
    CHECK_THROWS(snapshot::serialize(layers, cells, 0, nullptr, 0, snapshot_flags::body_only, 0, 0, buf));

    snapshot s = snapshot::parse(buf.data(), buf.size());
    CHECK(uint32_t(s.flags() & snapshot_flags::body_only) != 0);
    CHECK(s.n_used() == 0);
    CHECK(s.cells().empty());
    CHECK(s.metadata_size() == 0);
    CHECK(s.payload_size(0) == p.size());
    CHECK(std::memcmp(s.payload(0), p.data(), p.size()) == 0);

    expected_snapshot exp;
    exp.layers.push_back(expected_layer::from_desc(s.layers()[0]));
    exp.capacity_cells = 16;
    s.validate_expected_snapshot(exp);
    int staged = 0, staged_meta = 0, cell_pubs = 0;
    s.commit({ [&](uint32_t, uint32_t, const layer_desc &, const uint8_t *, size_t) { ++staged; return true; },
               [&](uint32_t, const uint8_t *, size_t) { ++staged_meta; return true; },
               [&]() {}, [&]() {} });
    CHECK(staged == 1);
    CHECK(staged_meta == 0); // body_only never stages metadata
    CHECK(cell_pubs == 0);
}

static void test_transactional_commit() {
    std::vector<uint8_t> p0(256), p1(256);
    fill_pattern(p0, 51);
    fill_pattern(p1, 52);
    std::vector<uint8_t> md;
    make_metadata(md, 53);
    std::vector<layer_blob> layers = {
        { make_qx_layer(1, sx_q4_0, sx_q4_0, 16, p0.size()), p0.data(), p0.size() },
        { make_qx_layer(2, sx_q4_0, sx_q4_0, 16, p1.size()), p1.data(), p1.size() },
    };
    std::vector<uint8_t> buf;
    CHECK(snapshot::serialize(layers, {}, 1, md.data(), md.size(), snapshot_flags::full, 0, 0, buf));

    // stage failure at layer 2: abort invoked, destination untouched, no finish
    {
        snapshot s = snapshot::parse(buf.data(), buf.size());
        expected_snapshot exp;
        for (const layer_blob & b : layers) {
            exp.layers.push_back(expected_layer::from_desc(b.desc));
        }
        exp.capacity_cells = 32;
        exp.metadata_version = 1;
        s.validate_expected_snapshot(exp);

        // simulated live destination: only finish() may mutate it
        int live = 0;
        int staged = 0, aborted = 0, finished = 0;
        CHECK_THROWS(s.commit({
            [&](uint32_t id, uint32_t, const layer_desc &, const uint8_t *, size_t) {
                ++staged;
                return id != 2; // fail staging layer 2
            },
            [&](uint32_t, const uint8_t *, size_t) { return true; },
            [&]() { ++finished; live = 99; }, // swap-in
            [&]() { ++aborted; },
        }));
        CHECK(staged == 2);       // layer 1 staged, layer 2 rejected
        CHECK(aborted == 1);      // rollback invoked
        CHECK(finished == 0);
        CHECK(live == 0);         // destination never touched
        CHECK_THROWS(s.commit({})); // ticket consumed by the failed attempt
    }
    // metadata stage failure: abort, no finish, destination untouched
    {
        snapshot s = snapshot::parse(buf.data(), buf.size());
        expected_snapshot exp;
        for (const layer_blob & b : layers) {
            exp.layers.push_back(expected_layer::from_desc(b.desc));
        }
        exp.capacity_cells = 32;
        exp.metadata_version = 1;
        s.validate_expected_snapshot(exp);
        int aborted = 0, finished = 0, live = 0;
        CHECK_THROWS(s.commit({
            [&](uint32_t, uint32_t, const layer_desc &, const uint8_t *, size_t) { return true; },
            [&](uint32_t, const uint8_t *, size_t) { return false; },
            [&]() { ++finished; live = 99; },
            [&]() { ++aborted; },
        }));
        CHECK(aborted == 1);
        CHECK(finished == 0);
        CHECK(live == 0);
    }
    // stage handler throwing: abort, no finish
    {
        snapshot s = snapshot::parse(buf.data(), buf.size());
        expected_snapshot exp;
        for (const layer_blob & b : layers) {
            exp.layers.push_back(expected_layer::from_desc(b.desc));
        }
        exp.capacity_cells = 32;
        exp.metadata_version = 1;
        s.validate_expected_snapshot(exp);
        int aborted = 0, finished = 0;
        bool thrown = false;
        try {
            s.commit({
                [&](uint32_t, uint32_t, const layer_desc &, const uint8_t *, size_t) -> bool {
                    throw std::runtime_error("handler boom");
                },
                [&](uint32_t, const uint8_t *, size_t) { return true; },
                [&]() { ++finished; },
                [&]() { ++aborted; },
            });
        } catch (const std::runtime_error &) {
            thrown = true;
        }
        CHECK(thrown);
        CHECK(aborted == 1);
        CHECK(finished == 0);
    }
    // success path: stage all, finish once, abort never, destination swapped
    {
        snapshot s = snapshot::parse(buf.data(), buf.size());
        expected_snapshot exp;
        for (const layer_blob & b : layers) {
            exp.layers.push_back(expected_layer::from_desc(b.desc));
        }
        exp.capacity_cells = 32;
        exp.metadata_version = 1;
        s.validate_expected_snapshot(exp);
        int staged = 0, aborted = 0, finished = 0, live = 0;
        s.commit({
            [&](uint32_t, uint32_t, const layer_desc &, const uint8_t *, size_t) { ++staged; return true; },
            [&](uint32_t, const uint8_t *, size_t) { return true; },
            [&]() { ++finished; live = 1; },
            [&]() { ++aborted; },
        });
        CHECK(staged == 2);
        CHECK(aborted == 0);
        CHECK(finished == 1);
        CHECK(live == 1); // only finish() may touch the destination
        CHECK_THROWS(s.commit({})); // exactly one commit attempt
    }
    // commit without validation is unreachable
    {
        snapshot s = snapshot::parse(buf.data(), buf.size());
        CHECK_THROWS(s.commit({}));
    }
    // required handlers must not be silently optional: a missing handler
    // fails the attempt BEFORE any staging and consumes the ticket
    {
        snapshot s = snapshot::parse(buf.data(), buf.size());
        expected_snapshot exp;
        for (const layer_blob & b : layers) {
            exp.layers.push_back(expected_layer::from_desc(b.desc));
        }
        exp.capacity_cells = 32;
        exp.metadata_version = 1;
        s.validate_expected_snapshot(exp);

        int ran = 0;
        auto stage = [&](uint32_t, uint32_t, const layer_desc &, const uint8_t *, size_t) { ++ran; return true; };
        auto meta  = [&](uint32_t, const uint8_t *, size_t) { ++ran; return true; };
        // missing stage_metadata
        CHECK_THROWS(s.commit({ stage, nullptr, [&]() { ++ran; }, [&]() { ++ran; } }));
        // missing abort (staging may allocate; rollback is required)
        CHECK_THROWS(s.commit({ stage, meta, [&]() { ++ran; }, nullptr }));
        // missing finish
        CHECK_THROWS(s.commit({ stage, meta, nullptr, [&]() { ++ran; } }));
        // missing stage_layer
        CHECK_THROWS(s.commit({ nullptr, meta, [&]() { ++ran; }, [&]() { ++ran; } }));
        CHECK(ran == 0); // no handler ran, no state dropped silently
        // ticket consumed by the failed attempts: no retry
        CHECK_THROWS(s.commit({ stage, meta, [&]() { ++ran; }, [&]() { ++ran; } }));
        CHECK(ran == 0);
    }
}

static void test_duplicate_and_bad_inputs() {
    std::vector<uint8_t> p(16);
    fill_pattern(p, 61);
    std::vector<uint8_t> md;
    make_metadata(md, 62);
    layer_desc a = make_qx_layer(1, sx_q4_0, sx_q4_0, 1, p.size());
    layer_desc b = a;
    b.layer_id = 1; // duplicate
    std::vector<uint8_t> buf;
    CHECK_THROWS(snapshot::serialize({ { a, p.data(), p.size() }, { b, p.data(), p.size() } },
                                     {}, 1, md.data(), md.size(), snapshot_flags::full, 0, 0, buf));
    CHECK_THROWS(snapshot::serialize({ { a, p.data(), 8 } }, {}, 1, md.data(), md.size(),
                                     snapshot_flags::full, 0, 0, buf));

    layer_desc bad = a;
    bad.kind = cache_kind::unknown;
    CHECK_THROWS(snapshot::serialize({ { bad, p.data(), p.size() } }, {}, 1, md.data(), md.size(),
                                     snapshot_flags::full, 0, 0, buf));
    bad = a; bad.k_type = 999;
    CHECK_THROWS(snapshot::serialize({ { bad, p.data(), p.size() } }, {}, 1, md.data(), md.size(),
                                     snapshot_flags::full, 0, 0, buf));
    bad = a; bad.layout = 99;
    CHECK_THROWS(snapshot::serialize({ { bad, p.data(), p.size() } }, {}, 1, md.data(), md.size(),
                                     snapshot_flags::full, 0, 0, buf));
    bad = a; bad.owner = 9;
    CHECK_THROWS(snapshot::serialize({ { bad, p.data(), p.size() } }, {}, 1, md.data(), md.size(),
                                     snapshot_flags::full, 0, 0, buf));
    bad = a; bad.n_stream = 2;
    CHECK_THROWS(snapshot::serialize({ { bad, p.data(), p.size() } }, {}, 1, md.data(), md.size(),
                                     snapshot_flags::full, 0, 0, buf));
    bad = a; bad.tail_type = sx_f16; // standard cache cannot carry a tail
    CHECK_THROWS(snapshot::serialize({ { bad, p.data(), p.size() } }, {}, 1, md.data(), md.size(),
                                     snapshot_flags::full, 0, 0, buf));
    bad = a; bad.n_head_kv = 0; // missing geometry
    CHECK_THROWS(snapshot::serialize({ { bad, p.data(), p.size() } }, {}, 1, md.data(), md.size(),
                                     snapshot_flags::full, 0, 0, buf));
    bad = a; bad.head_dim_k = 0;
    CHECK_THROWS(snapshot::serialize({ { bad, p.data(), p.size() } }, {}, 1, md.data(), md.size(),
                                     snapshot_flags::full, 0, 0, buf));

    // rotation consistency: hadamard needs a width; none must not carry one
    bad = a; bad.k_rot = uint8_t(rotation::hadamard); bad.k_rot_width = 0;
    CHECK_THROWS(snapshot::serialize({ { bad, p.data(), p.size() } }, {}, 1, md.data(), md.size(),
                                     snapshot_flags::full, 0, 0, buf));
    bad = a; bad.v_rot = uint8_t(rotation::none); bad.v_rot_width = 64;
    CHECK_THROWS(snapshot::serialize({ { bad, p.data(), p.size() } }, {}, 1, md.data(), md.size(),
                                     snapshot_flags::full, 0, 0, buf));
    bad = a; bad.k_rot = 7;
    CHECK_THROWS(snapshot::serialize({ { bad, p.data(), p.size() } }, {}, 1, md.data(), md.size(),
                                     snapshot_flags::full, 0, 0, buf));
    bad = a; bad.kvarn_domain = uint8_t(kvarn_domain::rotated); // standard cannot carry kvarn domain
    CHECK_THROWS(snapshot::serialize({ { bad, p.data(), p.size() } }, {}, 1, md.data(), md.size(),
                                     snapshot_flags::full, 0, 0, buf));

    // kvarn descriptor consistency: joint type id + bit fields
    layer_desc k = make_kvarn_layer(2, kv4v5, 4, 5, uint8_t(kvarn_domain::rotated), 128, 1, 64);
    std::vector<uint8_t> kp(64);
    fill_pattern(kp, 63);
    CHECK(snapshot::serialize({ { k, kp.data(), kp.size() } }, {}, 1, md.data(), md.size(),
                             snapshot_flags::full, 0, 0, buf));
    k.k_bits = 3; // contradicts type id kv4v5 (K4V5)
    CHECK_THROWS(snapshot::serialize({ { k, kp.data(), kp.size() } }, {}, 1, md.data(), md.size(),
                                     snapshot_flags::full, 0, 0, buf));
    k.k_bits = 4; k.v_type = kv4v6; // k_type != v_type
    CHECK_THROWS(snapshot::serialize({ { k, kp.data(), kp.size() } }, {}, 1, md.data(), md.size(),
                                     snapshot_flags::full, 0, 0, buf));
    k.v_type = kv4v5; k.v_bits = 5; k.tail_type = 999;
    CHECK_THROWS(snapshot::serialize({ { k, kp.data(), kp.size() } }, {}, 1, md.data(), md.size(),
                                     snapshot_flags::full, 0, 0, buf));
    k.tail_type = sx_f16; k.k_rot = uint8_t(rotation::hadamard); k.k_rot_width = 256; // kvarn rotation
    CHECK_THROWS(snapshot::serialize({ { k, kp.data(), kp.size() } }, {}, 1, md.data(), md.size(),
                                     snapshot_flags::full, 0, 0, buf));
    k.k_rot = 0; k.k_rot_width = 0; k.kvarn_domain = 7; // unknown kvarn domain
    CHECK_THROWS(snapshot::serialize({ { k, kp.data(), kp.size() } }, {}, 1, md.data(), md.size(),
                                     snapshot_flags::full, 0, 0, buf));

    // flag combination partial|body_only and unknown bits are invalid
    CHECK_THROWS(snapshot::serialize({ { a, p.data(), p.size() } }, {}, 1, md.data(), md.size(),
                                     snapshot_flags::partial | snapshot_flags::body_only, 1, 1, buf));
    CHECK_THROWS(snapshot::serialize({ { a, p.data(), p.size() } }, {}, 1, md.data(), md.size(),
                                     snapshot_flags(4), 0, 0, buf));
    CHECK_THROWS(snapshot::serialize({ { a, nullptr, 0 } }, {}, 1, md.data(), md.size(),
                                     snapshot_flags::partial, 0, 1, buf));

    // parse rejects an empty buffer and a truncated header
    CHECK_THROWS(snapshot::parse(nullptr, 0));
    std::vector<uint8_t> short_buf(40, 0xAB);
    CHECK_THROWS(snapshot::parse(short_buf.data(), short_buf.size()));
}

static void test_truncation_and_overflow() {
    std::vector<uint8_t> p0(1024), p1(512);
    fill_pattern(p0, 71);
    fill_pattern(p1, 72);
    std::vector<uint8_t> md(256);
    fill_pattern(md, 73);
    std::vector<layer_blob> layers = {
        { make_kvarn_layer(0, kv4v5, 4, 5, uint8_t(kvarn_domain::rotated), 2048, 16, p0.size()), p0.data(), p0.size() },
        { make_qx_layer(1, sx_q4_0, sx_q4_0, 64, p1.size()), p1.data(), p1.size() },
    };
    std::vector<cell_entry> cells(4);
    for (int i = 0; i < 4; ++i) {
        cells[i].pos = i;
        cells[i].tok = 200 + i;
        cells[i].seq_ids = { 0 };
    }
    std::vector<uint8_t> buf;
    CHECK(snapshot::serialize(layers, cells, 1, md.data(), md.size(), snapshot_flags::full, 0, 0, buf));

    const size_t blob_off = 96;
    const size_t desc0_off = blob_off + md.size();
    const size_t payload0_off = desc0_off + 100;
    CHECK(read_le_u64_at(buf, desc0_off + 85) == p0.size()); // layer 0 payload_bytes field

    // truncated at every section
    CHECK_THROWS(snapshot::parse(buf.data(), 40));                                  // mid header
    CHECK_THROWS(snapshot::parse(buf.data(), blob_off + 100));                      // mid metadata blob
    CHECK_THROWS(snapshot::parse(buf.data(), desc0_off + 96));                      // descriptor 0 ends, payload missing
    CHECK_THROWS(snapshot::parse(buf.data(), payload0_off + 100));                  // mid payload 0
    CHECK_THROWS(snapshot::parse(buf.data(), buf.size() - 1));                      // missing tail crc byte
    CHECK_THROWS(snapshot::parse(buf.data(), buf.size() - 8));                      // tail crc entirely missing
    CHECK_THROWS(snapshot::parse(buf.data(), buf.size() - 8 - 4 * 22 - 1));         // mid cell summary

    // malicious layer count: no large allocation, immediate bound check
    std::vector<uint8_t> evil = buf;
    write_le_u32_at(evil, 28, 0xFFFFFFFFu); // n_layers
    patch_header_crc(evil);
    CHECK_THROWS(snapshot::parse(evil.data(), evil.size()));

    // malicious n_used
    evil = buf;
    write_le_u32_at(evil, 32, 0xFFFFFFFEu); // n_used
    patch_header_crc(evil);
    CHECK_THROWS(snapshot::parse(evil.data(), evil.size()));

    // malicious metadata_bytes larger than the remaining input
    evil = buf;
    write_le_u64_at(evil, 40, 0x7FFFFFFFFFFFFFFFull);
    patch_header_crc(evil);
    CHECK_THROWS(snapshot::parse(evil.data(), evil.size()));

    // malicious payload_bytes larger than the remaining input
    evil = buf;
    write_le_u64_at(evil, payload0_off - 16, 0x7FFFFFFFFFFFFFFFull); // payload_bytes field
    CHECK_THROWS(snapshot::parse(evil.data(), evil.size()));

    // malicious cell n_seq count
    evil = buf;
    const size_t cells_off = payload0_off + p0.size() + 100 + p1.size();
    write_le_u32_at(evil, cells_off + 16, 0xFFFFu); // n_seq of cell 0
    CHECK_THROWS(snapshot::parse(evil.data(), evil.size()));

    // unknown version must be rejected (patched header crc)
    evil = buf;
    write_le_u32_at(evil, 4, 2);
    patch_header_crc(evil);
    CHECK_THROWS(snapshot::parse(evil.data(), evil.size()));

    // nonzero reserved header field
    evil = buf;
    write_le_u32_at(evil, 56, 1);
    patch_header_crc(evil);
    CHECK_THROWS(snapshot::parse(evil.data(), evil.size()));
}

static void test_caller_caps() {
    std::vector<uint8_t> p(512);
    fill_pattern(p, 81);
    std::vector<uint8_t> md(256);
    fill_pattern(md, 82);
    std::vector<uint8_t> buf;
    CHECK(snapshot::serialize({ { make_kvarn_layer(0, kv4v5, 4, 5, uint8_t(kvarn_domain::rotated), 512, 4, p.size()), p.data(), p.size() } },
                              {}, 1, md.data(), md.size(), snapshot_flags::full, 0, 0, buf));

    parse_options caps;
    caps.max_total_bytes = buf.size() - 1;
    CHECK_THROWS(snapshot::parse(buf.data(), buf.size(), caps));
    caps.max_total_bytes = 0;

    caps.max_metadata_bytes = md.size() - 1;
    CHECK_THROWS(snapshot::parse(buf.data(), buf.size(), caps));
    caps.max_metadata_bytes = 0;

    caps.max_payload_bytes = p.size() - 1;
    CHECK_THROWS(snapshot::parse(buf.data(), buf.size(), caps));
    caps.max_payload_bytes = 0;

    // caps at exactly the real sizes must pass
    caps.max_total_bytes = buf.size();
    caps.max_metadata_bytes = md.size();
    caps.max_payload_bytes = p.size();
    snapshot s = snapshot::parse(buf.data(), buf.size(), caps);
    CHECK(s.payload_size(0) == p.size());
    CHECK(s.metadata_size() == md.size());

    // caps are enforced identically in borrowed view mode
    auto owner = std::make_shared<std::vector<uint8_t>>(buf);
    caps.max_payload_bytes = p.size() - 1;
    CHECK_THROWS(snapshot::parse_view(buf.data(), buf.size(), owner, caps));
    caps.max_payload_bytes = 0;
    caps.max_total_bytes = buf.size() - 1;
    CHECK_THROWS(snapshot::parse_view(buf.data(), buf.size(), owner, caps));
    caps.max_total_bytes = 0;
    snapshot v = snapshot::parse_view(buf.data(), buf.size(), owner, caps);
    CHECK(v.is_borrowed() && v.payload_size(0) == p.size());
}

static void test_checksum_corruption() {
    std::vector<uint8_t> p(512);
    fill_pattern(p, 91);
    std::vector<uint8_t> md(128);
    fill_pattern(md, 92);
    std::vector<uint8_t> buf;
    CHECK(snapshot::serialize({ { make_kvarn_layer(0, kv4v5, 4, 5, uint8_t(kvarn_domain::rotated), 512, 4, p.size()), p.data(), p.size() } },
                              {}, 1, md.data(), md.size(), snapshot_flags::full, 0, 0, buf));

    // header byte flip -> header checksum mismatch
    std::vector<uint8_t> evil = buf;
    evil[12] ^= 0x40; // inside owner_epoch
    CHECK_THROWS(snapshot::parse(evil.data(), evil.size()));

    // metadata blob byte flip -> metadata checksum mismatch
    evil = buf;
    evil[96 + 100] ^= 0xFF;
    CHECK_THROWS(snapshot::parse(evil.data(), evil.size()));

    // payload byte flip -> payload checksum mismatch
    evil = buf;
    const size_t payload_off = 96 + md.size() + 100;
    evil[payload_off + 100] ^= 0xFF;
    CHECK_THROWS(snapshot::parse(evil.data(), evil.size()));

    // trailing checksum byte flip -> whole-file mismatch
    evil = buf;
    evil[evil.size() - 1] ^= 0x01;
    CHECK_THROWS(snapshot::parse(evil.data(), evil.size()));
}

static void test_fault_callback_and_no_commit() {
    std::vector<uint8_t> p(64);
    fill_pattern(p, 101);
    std::vector<uint8_t> md(64);
    fill_pattern(md, 102);
    std::vector<uint8_t> buf;
    CHECK(snapshot::serialize({ { make_qx_layer(1, sx_q4_0, sx_q4_0, 4, p.size()), p.data(), p.size() } },
                              {}, 1, md.data(), md.size(), snapshot_flags::full, 0, 0, buf));

    // fault callback fires on parse failure; no snapshot object is produced
    int faults = 0;
    parse_options opts;
    opts.on_fault = [&](const std::string & msg) {
        ++faults;
        CHECK(!msg.empty());
    };
    buf[0] ^= 0xFF; // corrupt magic; also breaks header crc
    bool threw = false;
    try {
        snapshot::parse(buf.data(), buf.size(), opts);
    } catch (const kv_mixed_error &) {
        threw = true;
    }
    CHECK(threw);
    CHECK(faults == 1);

    // fault callback fires on validate failure and commit stays blocked
    buf[0] ^= 0xFF; // restore
    snapshot s = snapshot::parse(buf.data(), buf.size());
    faults = 0;
    expected_snapshot exp;
    exp.layers.push_back(expected_layer::from_desc(s.layers()[0]));
    exp.layers[0].layer_id = 2; // wrong layer id
    exp.capacity_cells = 4;
    exp.metadata_version = 1;
    threw = false;
    try {
        s.validate_expected_snapshot(exp, opts);
    } catch (const kv_mixed_error &) {
        threw = true;
    }
    CHECK(threw);
    CHECK(faults == 1);
    int publishes = 0;
    CHECK_THROWS(s.commit({ [&](uint32_t, uint32_t, const layer_desc &, const uint8_t *, size_t) { ++publishes; return true; },
                            nullptr, nullptr, nullptr }));
    CHECK(publishes == 0);

    // serialize faults fire through the callback too
    faults = 0;
    std::vector<layer_blob> dup = { { make_qx_layer(1, sx_q4_0, sx_q4_0, 4, p.size()), p.data(), p.size() },
                                    { make_qx_layer(1, sx_q4_0, sx_q4_0, 4, p.size()), p.data(), p.size() } };
    std::vector<uint8_t> out;
    CHECK_THROWS(snapshot::serialize(dup, {}, 1, md.data(), md.size(), snapshot_flags::full, 0, 0, out, opts));
    CHECK(faults == 1);
}

static void test_borrowed_view() {
    std::vector<uint8_t> p(512);
    fill_pattern(p, 121);
    std::vector<uint8_t> md(128);
    fill_pattern(md, 122);
    layer_desc d = make_qx_layer(1, sx_q4_0, sx_q4_0, 32, p.size());
    std::vector<uint8_t> buf;
    CHECK(snapshot::serialize({ { d, p.data(), p.size() } }, {}, 1, md.data(), md.size(),
                              snapshot_flags::full, 0, 0, buf));
    const size_t payload0_off = 96 + md.size() + 101;

    // the shared_ptr OWNS the backing; the snapshot retains it
    auto owner = std::make_shared<std::vector<uint8_t>>(buf);
    const uint8_t * base = owner->data();
    snapshot v = snapshot::parse_view(base, owner->size(), owner);
    CHECK(v.is_borrowed());
    CHECK(v.backing() == owner); // the retained shared_ptr is the same one
    // zero-copy: payload points exactly into the caller backing, no copy made
    CHECK(v.payload(0) == base + payload0_off);
    CHECK(v.payload_size(0) == p.size());
    CHECK(std::memcmp(v.payload(0), p.data(), p.size()) == 0);
    // pointer range: every payload lies within the backing
    CHECK(v.payload(0) >= base);
    CHECK(v.payload(0) + v.payload_size(0) <= base + owner->size());
    // metadata blob and cells are still copied (small)
    CHECK(v.metadata_size() == md.size());
    CHECK(std::memcmp(v.metadata_blob(), md.data(), md.size()) == 0);

    // strict validation and transaction semantics are preserved in view mode
    expected_snapshot exp;
    exp.layers.push_back(expected_layer::from_desc(v.layers()[0]));
    exp.capacity_cells = 32;
    exp.metadata_version = 1;
    v.validate_expected_snapshot(exp);
    int staged = 0, finished = 0, aborted = 0;
    v.commit({
        [&](uint32_t, uint32_t, const layer_desc &, const uint8_t * ptr, size_t n) {
            ++staged;
            CHECK(ptr == base + payload0_off); // stage sees the borrowed bytes
            CHECK(n == p.size());
            return true;
        },
        [&](uint32_t, const uint8_t *, size_t) { return true; },
        [&]() { ++finished; },
        [&]() { ++aborted; },
    });
    CHECK(staged == 1 && finished == 1 && aborted == 0);

    // snapshot stays valid through moves (it is move-only)
    snapshot m = std::move(v);
    CHECK(m.is_borrowed());
    CHECK(m.payload(0) == base + payload0_off);
    CHECK_THROWS(v.commit({})); // moved-from snapshot cannot commit
    snapshot c = std::move(m);
    CHECK(c.is_borrowed());
    CHECK(c.payload(0) == base + payload0_off);
    CHECK(std::memcmp(c.payload(0), p.data(), p.size()) == 0);

    // weak_ptr lifetime: the snapshot retains the backing until it dies, even
    // after the original shared_ptr handle is released
    std::weak_ptr<const void> weak = owner;
    owner.reset();
    CHECK(!weak.expired());
    CHECK(c.payload(0) == base + payload0_off); // still mapped through the retained owner
    CHECK(std::memcmp(c.payload(0), p.data(), p.size()) == 0);
    c = snapshot{}; // drop the snapshot: backing must be released now
    CHECK(weak.expired());

    // a missing shared lifetime owner is rejected
    CHECK_THROWS(snapshot::parse_view(buf.data(), buf.size(), {}));

    // owning mode stays independent of input mutations...
    std::vector<uint8_t> copy = buf;
    snapshot o = snapshot::parse(copy.data(), copy.size());
    copy[payload0_off + 10] ^= 0xFF;
    CHECK(std::memcmp(o.payload(0), p.data(), p.size()) == 0);
    // ...while a borrowed view reflects the mutation: the caller MUST keep
    // the backing immutable by contract (atomic snapshot files, temp+rename)
    auto mutable_backing = std::make_shared<std::vector<uint8_t>>(buf);
    snapshot bv = snapshot::parse_view(mutable_backing->data(), mutable_backing->size(), mutable_backing);
    CHECK(std::memcmp(bv.payload(0), p.data(), p.size()) == 0);
    mutable_backing->data()[payload0_off + 10] ^= 0xFF;
    CHECK(std::memcmp(bv.payload(0), p.data(), p.size()) != 0); // view reflects mutation
}

static void test_move_semantics() {
    std::vector<uint8_t> p(256);
    fill_pattern(p, 131);
    std::vector<uint8_t> md(64);
    fill_pattern(md, 132);
    layer_desc d = make_qx_layer(1, sx_q4_0, sx_q4_0, 16, p.size());
    std::vector<uint8_t> buf;
    CHECK(snapshot::serialize({ { d, p.data(), p.size() } }, {}, 1, md.data(), md.size(),
                              snapshot_flags::full, 0, 0, buf));

    auto make_expected = [&](snapshot & s) {
        expected_snapshot exp;
        exp.layers.push_back(expected_layer::from_desc(s.layers()[0]));
        exp.capacity_cells = 16;
        exp.metadata_version = 1;
        return exp;
    };
    auto handlers = [&](int & staged) {
        return commit_handlers{
            [&](uint32_t, uint32_t, const layer_desc &, const uint8_t *, size_t) { ++staged; return true; },
            [&](uint32_t, const uint8_t *, size_t) { return true; },
            [&]() {}, [&]() {} };
    };

    // owning move: content survives the original object's destruction
    {
        snapshot s2;
        {
            snapshot s = snapshot::parse(buf.data(), buf.size());
            s2 = std::move(s); // move assignment
            CHECK_THROWS(s.commit({})); // moved-from cannot commit
        }
        CHECK(!s2.is_borrowed());
        CHECK(std::memcmp(s2.payload(0), p.data(), p.size()) == 0);
    }
    // validated ticket moves to the destination only
    {
        snapshot s = snapshot::parse(buf.data(), buf.size());
        s.validate_expected_snapshot(make_expected(s));
        snapshot s2 = std::move(s);
        CHECK_THROWS(s.commit({}));               // moved-from ticket invalidated
        CHECK_THROWS(s.validate_expected_snapshot(make_expected(s2))); // moved-from is inert
        int staged = 0;
        s2.commit(handlers(staged));              // destination commits once
        CHECK(staged == 1);
        CHECK_THROWS(s2.commit({}));              // ticket consumed
    }
    // borrowed move keeps the backing and transfers the ticket
    {
        auto owner = std::make_shared<std::vector<uint8_t>>(buf);
        snapshot s = snapshot::parse_view(owner->data(), owner->size(), owner);
        CHECK(s.is_borrowed());
        s.validate_expected_snapshot(make_expected(s));
        owner.reset(); // snapshot retains the backing
        snapshot s2 = std::move(s);
        CHECK(s2.is_borrowed());
        CHECK(s2.backing() != nullptr);
        CHECK(std::memcmp(s2.payload(0), p.data(), p.size()) == 0);
        int staged = 0;
        s2.commit(handlers(staged));
        CHECK(staged == 1);
    }
    // move assignment frees the old owned payloads
    {
        snapshot a = snapshot::parse(buf.data(), buf.size());
        snapshot b = snapshot::parse(buf.data(), buf.size());
        a = std::move(b); // old owned buffers of a are released
        CHECK(std::memcmp(a.payload(0), p.data(), p.size()) == 0);
        CHECK_THROWS(b.commit({}));
    }
    // moving an ALREADY moved-from snapshot keeps the destination inert: the
    // moved_from_ flag transfers, so an empty moved-from cannot be resurrected
    // by re-validating an empty expected set
    {
        snapshot a = snapshot::parse(buf.data(), buf.size());
        snapshot b = std::move(a);
        CHECK_THROWS(b.commit({}));
        snapshot c = std::move(b); // moving a moved-from source
        expected_snapshot exp_empty; // zero layers
        exp_empty.capacity_cells = 0;
        CHECK_THROWS(c.validate_expected_snapshot(exp_empty)); // stays inert
        CHECK_THROWS(c.commit({}));
        CHECK_THROWS(b.commit({})); // b also inert
    }
    // assigning a VALID parsed snapshot into a previously moved-from
    // destination restores a fully valid object
    {
        snapshot x = snapshot::parse(buf.data(), buf.size());
        snapshot dead = snapshot::parse(buf.data(), buf.size());
        snapshot z = std::move(dead);
        CHECK_THROWS(z.commit({})); // inert before
        z = std::move(x);           // assignment into moved-from destination
        CHECK(!z.is_borrowed());
        CHECK(std::memcmp(z.payload(0), p.data(), p.size()) == 0);
        z.validate_expected_snapshot(make_expected(z));
        int staged = 0;
        z.commit(handlers(staged)); // valid again
        CHECK(staged == 1);
        CHECK_THROWS(x.commit({})); // source is moved-from now
    }
    // self-move is a defined no-op: valid stays valid, inert stays inert
    // (routed through a helper so the compiler's -Wself-move does not fire)
    auto selfmove = [](snapshot & x) -> snapshot && { return std::move(x); };
    {
        snapshot s = snapshot::parse(buf.data(), buf.size());
        s.validate_expected_snapshot(make_expected(s));
        s = selfmove(s);
        CHECK(std::memcmp(s.payload(0), p.data(), p.size()) == 0);
        int staged = 0;
        s.commit(handlers(staged)); // ticket preserved by the no-op self-move
        CHECK(staged == 1);
        snapshot t = snapshot::parse(buf.data(), buf.size());
        snapshot u = std::move(t);
        t = selfmove(t); // self-move of a moved-from object
        CHECK_THROWS(t.commit({})); // stays inert
        CHECK_THROWS(t.validate_expected_snapshot(make_expected(u)));
    }
}

static void test_commit_rechecks_borrowed_payload() {
    std::vector<uint8_t> p(256);
    fill_pattern(p, 141);
    std::vector<uint8_t> md(64);
    fill_pattern(md, 142);
    layer_desc d = make_qx_layer(1, sx_q4_0, sx_q4_0, 16, p.size());
    std::vector<uint8_t> buf;
    CHECK(snapshot::serialize({ { d, p.data(), p.size() } }, {}, 1, md.data(), md.size(),
                              snapshot_flags::full, 0, 0, buf));
    const size_t payload0_off = 96 + md.size() + 101;

    // mutating the backing after parse is DETECTED at commit, before any
    // stage handler runs (detection, not protection: immutability is the
    // caller's contract)
    auto owner = std::make_shared<std::vector<uint8_t>>(buf);
    snapshot s = snapshot::parse_view(owner->data(), owner->size(), owner);
    expected_snapshot exp;
    exp.layers.push_back(expected_layer::from_desc(s.layers()[0]));
    exp.capacity_cells = 16;
    exp.metadata_version = 1;
    s.validate_expected_snapshot(exp);
    owner->data()[payload0_off + 10] ^= 0xFF; // mutate the backing
    int staged = 0, aborted = 0, finished = 0;
    CHECK_THROWS(s.commit({
        [&](uint32_t, uint32_t, const layer_desc &, const uint8_t *, size_t) { ++staged; return true; },
        [&](uint32_t, const uint8_t *, size_t) { return true; },
        [&]() { ++finished; }, [&]() { ++aborted; },
    }));
    CHECK(staged == 0); // nothing staged: recheck runs before stage handlers
    CHECK(aborted == 0); // nothing staged, so nothing to abort
    CHECK(finished == 0);
    // unmutated backing commits normally (covered by test_borrowed_view)
}

static void test_trailing_garbage() {
    std::vector<uint8_t> p(32);
    fill_pattern(p, 111);
    std::vector<uint8_t> md(32);
    fill_pattern(md, 112);
    std::vector<uint8_t> buf;
    CHECK(snapshot::serialize({ { make_qx_layer(1, sx_q4_0, sx_q4_0, 2, p.size()), p.data(), p.size() } },
                              {}, 1, md.data(), md.size(), snapshot_flags::full, 0, 0, buf));
    buf.push_back(0x00); // garbage after tail crc
    CHECK_THROWS(snapshot::parse(buf.data(), buf.size()));
}

// ---------------------------------------------------------------------------
// F4 v2: position_split envelope
// ---------------------------------------------------------------------------

static constexpr uint32_t SPLIT_P = 512;
static constexpr uint32_t SPLIT_C = 1024;

// One full layer as it exists in P-mode: local KVarN [0,P) on CUDA and
// overflow standard-Qx [P,C) on Vulkan. This is exactly the pair that v1
// rejected with "duplicate layer ownership".
static layer_blob make_split_pair(uint32_t layer_id,
                                  std::vector<uint8_t> & local,
                                  std::vector<uint8_t> & overflow,
                                  bool empty_overflow = false) {
    layer_desc l = make_kvarn_layer(layer_id, kv4v4, 4, 4,
            uint8_t(kvarn_domain::rotated), SPLIT_P, SPLIT_P / 128, 0);
    l.owner = owner_target;
    l.range_id = 0;
    l.range_start = 0;
    l.range_end = SPLIT_P;
    fill_pattern(local, layer_id * 7 + 1);
    l.payload_bytes = local.size();

    layer_desc o = make_qx_layer(layer_id, sx_q4_0, sx_q4_0, SPLIT_C - SPLIT_P, 0);
    o.owner = owner_target;
    o.range_id = 1;
    o.range_start = SPLIT_P;
    o.range_end = SPLIT_C;
    fill_pattern(overflow, layer_id * 7 + 2);
    // an empty overflow range keeps its descriptor and simply omits payload bytes
    o.payload_bytes = empty_overflow ? 0 : overflow.size();

    return { l, local.data(), local.size() };
}

static std::vector<layer_blob> build_split_layers(
        std::vector<std::vector<uint8_t>> & locals,
        std::vector<std::vector<uint8_t>> & overflows,
        uint32_t n_layers, bool empty_overflow = false) {
    std::vector<layer_blob> out;
    locals.assign(n_layers, std::vector<uint8_t>(256, 0));
    overflows.assign(n_layers, std::vector<uint8_t>(128, 0));
    for (uint32_t i = 0; i < n_layers; ++i) {
        const layer_blob local = make_split_pair(i, locals[i], overflows[i], empty_overflow);
        out.push_back(local);
        layer_desc o = make_qx_layer(i, sx_q4_0, sx_q4_0, SPLIT_C - SPLIT_P, 0);
        o.owner = owner_target;
        o.range_id = 1;
        o.range_start = SPLIT_P;
        o.range_end = SPLIT_C;
        o.payload_bytes = empty_overflow ? 0 : overflows[i].size();
        out.push_back({ o, empty_overflow ? nullptr : overflows[i].data(),
                        empty_overflow ? 0 : overflows[i].size() });
    }
    return out;
}

static position_split make_split() {
    position_split s;
    s.p = SPLIT_P;
    s.c = SPLIT_C;
    s.model_id = 0xA1B2C3D4E5F60718ull;
    s.rope_id  = 0x1122334455667788ull;
    return s;
}

static std::vector<cell_entry> make_split_cells() {
    std::vector<cell_entry> cells;
    for (uint32_t i = 0; i < 4; ++i) {
        cell_entry c;
        c.pos = int32_t(i);
        c.x = 0;
        c.y = int32_t(i + 1);
        c.tok = int32_t(1000 + i);
        c.seq_ids.push_back(0);
        cells.push_back(c);
    }
    return cells;
}

static void test_v2_split_roundtrip() {
    std::vector<std::vector<uint8_t>> locals, overflows;
    auto layers = build_split_layers(locals, overflows, 3);
    const auto md = std::vector<uint8_t>{ 'm', 'd', 1, 2, 3 };
    const auto cells = make_split_cells();

    std::vector<uint8_t> buf;
    serialize_options ser;
    ser.split = make_split();
    CHECK(snapshot::serialize(layers, cells, 7, md.data(), md.size(),
            snapshot_flags::full, 0, 0, buf, {}, ser));

    // the frame is v2 on the wire: new magic and version, not a v1 relabel
    CHECK(buf.size() >= 104);
    const uint32_t wire_magic = uint32_t(buf[0]) | (uint32_t(buf[1]) << 8) |
        (uint32_t(buf[2]) << 16) | (uint32_t(buf[3]) << 24);
    const uint32_t wire_version = uint32_t(buf[4]) | (uint32_t(buf[5]) << 8) |
        (uint32_t(buf[6]) << 16) | (uint32_t(buf[7]) << 24);
    CHECK(wire_magic == 0x32534D4Bull); // "KMS2"
    CHECK(wire_version == 2);

    auto s = snapshot::parse(buf.data(), buf.size());
    CHECK(s.version() == 2);
    CHECK(s.has_split());
    CHECK(s.split().has_value());
    if (s.split().has_value()) {
        CHECK(s.split()->p == SPLIT_P);
        CHECK(s.split()->c == SPLIT_C);
        CHECK(s.split()->model_id == 0xA1B2C3D4E5F60718ull);
        CHECK(s.split()->rope_id  == 0x1122334455667788ull);
    }
    CHECK(s.layers().size() == 6); // 3 layers x 2 ranges
    CHECK(s.cells().size() == cells.size());
    CHECK(s.metadata_version() == 7);

    // both ranges of every layer survive with their payloads intact
    for (uint32_t i = 0; i < 3; ++i) {
        bool saw_local = false, saw_overflow = false;
        for (size_t j = 0; j < s.layers().size(); ++j) {
            const layer_desc & d = s.layers()[j];
            if (d.layer_id != i) {
                continue;
            }
            if (d.range_id == 0) {
                saw_local = true;
                CHECK(d.range_start == 0 && d.range_end == SPLIT_P);
                CHECK(d.kind == cache_kind::kvarn);
                CHECK(s.payload_size(j) == locals[i].size());
                CHECK(std::memcmp(s.payload(j), locals[i].data(), locals[i].size()) == 0);
            } else {
                saw_overflow = true;
                CHECK(d.range_start == SPLIT_P && d.range_end == SPLIT_C);
                CHECK(d.kind == cache_kind::standard_qx);
                CHECK(s.payload_size(j) == overflows[i].size());
                CHECK(std::memcmp(s.payload(j), overflows[i].data(), overflows[i].size()) == 0);
            }
        }
        CHECK(saw_local);
        CHECK(saw_overflow);
    }
}

static void test_v2_empty_range_keeps_descriptor() {
    std::vector<std::vector<uint8_t>> locals, overflows;
    auto layers = build_split_layers(locals, overflows, 2, /*empty_overflow=*/true);
    const auto md = std::vector<uint8_t>{ 'x' };
    std::vector<uint8_t> buf;
    serialize_options ser;
    ser.split = make_split();
    CHECK(snapshot::serialize(layers, {}, 1, md.data(), md.size(),
            snapshot_flags::full, 0, 0, buf, {}, ser));
    auto s = snapshot::parse(buf.data(), buf.size());
    // descriptors stay complete; only the payload bytes are omitted
    CHECK(s.layers().size() == 4);
    for (const layer_desc & d : s.layers()) {
        if (d.range_id == 1) {
            CHECK(d.payload_bytes == 0);
            CHECK(d.range_start == SPLIT_P && d.range_end == SPLIT_C);
        } else {
            CHECK(d.payload_bytes > 0);
        }
    }
}

static void test_v2_rejects_bad_ranges() {
    const auto md = std::vector<uint8_t>{ 'm' };
    std::vector<std::vector<uint8_t>> locals, overflows;
    auto layers = build_split_layers(locals, overflows, 1);

    // invalid P/C: P == 0, P >= C, and a misaligned P are all rejected
    for (auto bad : { uint32_t(0), uint32_t(SPLIT_C), uint32_t(SPLIT_C + 256) }) {
        serialize_options ser;
        position_split s = make_split();
        s.p = bad;
        s.c = SPLIT_C;
        ser.split = s;
        std::vector<uint8_t> buf;
        CHECK_THROWS(snapshot::serialize(layers, {}, 1, md.data(), md.size(),
                snapshot_flags::full, 0, 0, buf, {}, ser));
    }
    {   // P not a multiple of the alignment
        serialize_options ser;
        position_split s = make_split();
        s.p = SPLIT_P - 1;
        ser.split = s;
        std::vector<uint8_t> buf;
        CHECK_THROWS(snapshot::serialize(layers, {}, 1, md.data(), md.size(),
                snapshot_flags::full, 0, 0, buf, {}, ser));
    }
    // local range that does not start at 0
    {
        auto bad = layers;
        bad[0].desc.range_start = 8;
        serialize_options ser;
        ser.split = make_split();
        std::vector<uint8_t> buf;
        CHECK_THROWS(snapshot::serialize(bad, {}, 1, md.data(), md.size(),
                snapshot_flags::full, 0, 0, buf, {}, ser));
    }
    // gap/overlap at the P boundary
    {
        auto bad = layers;
        bad[1].desc.range_start = SPLIT_P - 8;
        serialize_options ser;
        ser.split = make_split();
        std::vector<uint8_t> buf;
        CHECK_THROWS(snapshot::serialize(bad, {}, 1, md.data(), md.size(),
                snapshot_flags::full, 0, 0, buf, {}, ser));
    }
    // range running past C
    {
        auto bad = layers;
        bad[1].desc.range_end = SPLIT_C + 128;
        serialize_options ser;
        ser.split = make_split();
        std::vector<uint8_t> buf;
        CHECK_THROWS(snapshot::serialize(bad, {}, 1, md.data(), md.size(),
                snapshot_flags::full, 0, 0, buf, {}, ser));
    }
    // unknown range_id
    {
        auto bad = layers;
        bad[0].desc.range_id = 2;
        serialize_options ser;
        ser.split = make_split();
        std::vector<uint8_t> buf;
        CHECK_THROWS(snapshot::serialize(bad, {}, 1, md.data(), md.size(),
                snapshot_flags::full, 0, 0, buf, {}, ser));
    }
    // local range carrying a standard cache kind (backend/format pairing)
    {
        auto bad = layers;
        bad[0].desc.kind = cache_kind::standard_qx;
        bad[0].desc.k_type = sx_q4_0;
        bad[0].desc.v_type = sx_q4_0;
        bad[0].desc.layout = layout_standard_qx_rows;
        serialize_options ser;
        ser.split = make_split();
        std::vector<uint8_t> buf;
        CHECK_THROWS(snapshot::serialize(bad, {}, 1, md.data(), md.size(),
                snapshot_flags::full, 0, 0, buf, {}, ser));
    }
    // duplicate (layer_id, range_id) pair
    {
        auto bad = layers;
        bad.push_back(bad[0]); // same (layer_id, range_id) pair twice
        serialize_options ser;
        ser.split = make_split();
        std::vector<uint8_t> buf;
        CHECK_THROWS(snapshot::serialize(bad, {}, 1, md.data(), md.size(),
                snapshot_flags::full, 0, 0, buf, {}, ser));
    }
    // a layer with only a local range and no overflow is incomplete
    {
        std::vector<layer_blob> bad = { layers[0] };
        serialize_options ser;
        ser.split = make_split();
        std::vector<uint8_t> buf;
        CHECK_THROWS(snapshot::serialize(bad, {}, 1, md.data(), md.size(),
                snapshot_flags::full, 0, 0, buf, {}, ser));
    }
    // v1 path must refuse range identity it cannot validate
    {
        auto bad = layers;
        bad[0].desc.range_id = 1;
        std::vector<uint8_t> buf;
        CHECK_THROWS(snapshot::serialize(bad, {}, 1, md.data(), md.size(),
                snapshot_flags::full, 0, 0, buf));
    }
}

static void test_v2_validate_requires_matching_envelope() {
    std::vector<std::vector<uint8_t>> locals, overflows;
    auto layers = build_split_layers(locals, overflows, 2);
    const auto md = std::vector<uint8_t>{ 'm', 'd' };
    std::vector<uint8_t> buf;
    serialize_options ser;
    ser.split = make_split();
    CHECK(snapshot::serialize(layers, {}, 3, md.data(), md.size(),
            snapshot_flags::full, 0, 0, buf, {}, ser));

    auto build_expected = [&](const position_split & split) {
        expected_snapshot e;
        e.capacity_cells = 64;
        e.metadata_version = 3;
        e.split = split;
        for (const layer_blob & b : layers) {
            e.layers.push_back(expected_layer::from_desc(b.desc));
        }
        return e;
    };

    {   // exact match commits
        auto s = snapshot::parse(buf.data(), buf.size());
        s.validate_expected_snapshot(build_expected(make_split()));
        int staged = 0, finished = 0, aborted = 0;
        int split_staged = 0;
        s.commit({
            [&](uint32_t, uint32_t, const layer_desc &, const uint8_t *, size_t) { ++staged; return true; },
            [&](uint32_t, const uint8_t *, size_t) { return true; },
            [&]() { ++finished; },
            [&]() { ++aborted; },
            [&](const position_split &) { ++split_staged; return true; },
        });
        CHECK(staged == 4);
        CHECK(finished == 1);
        CHECK(aborted == 0);
        // stage_split runs exactly once, before any KV byte
        CHECK(split_staged == 1);
    }
    {   // P mismatch: rejected before any staging, and the ticket is consumed so
    // no partial restore can follow
        auto s = snapshot::parse(buf.data(), buf.size());
        position_split wrong = make_split();
        wrong.p = SPLIT_P - 256;
        CHECK_THROWS(s.validate_expected_snapshot(build_expected(wrong)));
        int staged = 0, finished = 0, aborted = 0, split_staged = 0;
        CHECK_THROWS(s.commit({
            [&](uint32_t, uint32_t, const layer_desc &, const uint8_t *, size_t) { ++staged; return true; },
            [&](uint32_t, const uint8_t *, size_t) { return true; },
            [&]() { ++finished; },
            [&]() { ++aborted; },
            [&](const position_split &) { ++split_staged; return true; },
        }));
        // nothing may be staged or published when the envelope does not match
        CHECK(staged == 0);
        CHECK(split_staged == 0);
        CHECK(finished == 0);
    }
    {   // model identity mismatch must not reach commit
        auto s = snapshot::parse(buf.data(), buf.size());
        position_split wrong = make_split();
        wrong.model_id ^= 1;
        CHECK_THROWS(s.validate_expected_snapshot(build_expected(wrong)));
        int staged = 0, finished = 0, split_staged = 0;
        CHECK_THROWS(s.commit({
            [&](uint32_t, uint32_t, const layer_desc &, const uint8_t *, size_t) { ++staged; return true; },
            [&](uint32_t, const uint8_t *, size_t) { return true; },
            [&]() { ++finished; },
            [&]() {},
            [&](const position_split &) { ++split_staged; return true; },
        }));
        CHECK(staged == 0);
        CHECK(split_staged == 0);
        CHECK(finished == 0);
    }
    {   // RoPE identity mismatch must not reach commit
        auto s = snapshot::parse(buf.data(), buf.size());
        position_split wrong = make_split();
        wrong.rope_id ^= 1;
        CHECK_THROWS(s.validate_expected_snapshot(build_expected(wrong)));
        int staged = 0, finished = 0, split_staged = 0;
        CHECK_THROWS(s.commit({
            [&](uint32_t, uint32_t, const layer_desc &, const uint8_t *, size_t) { ++staged; return true; },
            [&](uint32_t, const uint8_t *, size_t) { return true; },
            [&]() { ++finished; },
            [&]() {},
            [&](const position_split &) { ++split_staged; return true; },
        }));
        CHECK(staged == 0);
        CHECK(split_staged == 0);
        CHECK(finished == 0);
    }
    {   // a v2 snapshot cannot be validated against a destination that has no split
        auto s = snapshot::parse(buf.data(), buf.size());
        expected_snapshot e;
        e.capacity_cells = 64;
        e.metadata_version = 3;
        for (const layer_blob & b : layers) {
            e.layers.push_back(expected_layer::from_desc(b.desc));
        }
        CHECK_THROWS(s.validate_expected_snapshot(e));
    }
    {   // a v2 snapshot requires stage_split: absence aborts with nothing published
        auto s = snapshot::parse(buf.data(), buf.size());
        s.validate_expected_snapshot(build_expected(make_split()));
        int staged = 0, aborted = 0, finished = 0;
        CHECK_THROWS(s.commit({
            [&](uint32_t, uint32_t, const layer_desc &, const uint8_t *, size_t) { ++staged; return true; },
            [&](uint32_t, const uint8_t *, size_t) { return true; },
            [&]() { ++finished; },
            [&]() { ++aborted; },
        }));
        CHECK(staged == 0); // fails BEFORE any layer staging
        CHECK(finished == 0);
        CHECK(aborted == 1);
    }
    {   // stage_split failing aborts with nothing published
        auto s = snapshot::parse(buf.data(), buf.size());
        s.validate_expected_snapshot(build_expected(make_split()));
        int staged = 0, aborted = 0, finished = 0;
        CHECK_THROWS(s.commit({
            [&](uint32_t, uint32_t, const layer_desc &, const uint8_t *, size_t) { ++staged; return true; },
            [&](uint32_t, const uint8_t *, size_t) { return true; },
            [&]() { ++finished; },
            [&]() { ++aborted; },
            [&](const position_split &) { return false; },
        }));
        CHECK(staged == 0);
        CHECK(finished == 0);
        CHECK(aborted == 1);
    }
    {   // a range whose identity differs from the destination fails validation
        auto s = snapshot::parse(buf.data(), buf.size());
        position_split wrong = make_split();
        wrong.c = SPLIT_C + 256;
        CHECK_THROWS(s.validate_expected_snapshot(build_expected(wrong)));
    }
}

static void test_v2_corruption_and_version_fails_closed() {
    std::vector<std::vector<uint8_t>> locals, overflows;
    auto layers = build_split_layers(locals, overflows, 2);
    const auto md = std::vector<uint8_t>{ 'm', 'd', 3 };
    std::vector<uint8_t> buf;
    serialize_options ser;
    ser.split = make_split();
    CHECK(snapshot::serialize(layers, {}, 2, md.data(), md.size(),
            snapshot_flags::full, 0, 0, buf, {}, ser));

    {   // corrupted split_p in the header, crc recomputed: still fails the range set
        auto bad = buf;
        bad[56] ^= 0xFF; // split_p low byte
        patch_header_crc_v2(bad);
        bool threw = false;
        try { (void) snapshot::parse(bad.data(), bad.size()); }
        catch (const kv_mixed_error &) { threw = true; }
        CHECK(threw);
    }
    {   // corrupted model_id: detected by header crc
        auto bad = buf;
        bad[64] ^= 0x01;
        CHECK_THROWS(snapshot::parse(bad.data(), bad.size()));
    }
    {   // nonzero reserved field is rejected
        auto bad = buf;
        bad[80] = 1;
        patch_header_crc_v2(bad);
        CHECK_THROWS(snapshot::parse(bad.data(), bad.size()));
    }
    {   // v2 magic with a v1 version is refused
        auto bad = buf;
        bad[4] = 1;
        patch_header_crc_v2(bad);
        CHECK_THROWS(snapshot::parse(bad.data(), bad.size()));
    }
    {   // unknown version is refused
        auto bad = buf;
        bad[4] = 3;
        patch_header_crc_v2(bad);
        CHECK_THROWS(snapshot::parse(bad.data(), bad.size()));
    }
    {   // truncated v2 frame is refused, not partially read
        auto bad = buf;
        bad.resize(bad.size() - 1);
        CHECK_THROWS(snapshot::parse(bad.data(), bad.size()));
    }
    {   // a v1 frame must NOT parse as v2 and vice versa
        std::vector<uint8_t> v1;
        layer_desc d = make_qx_layer(0, sx_q4_0, sx_q4_0, 4, 0);
        std::vector<uint8_t> p1(16, 0xAB);
        d.payload_bytes = p1.size();
        CHECK(snapshot::serialize({ { d, p1.data(), p1.size() } }, {}, 1,
                md.data(), md.size(), snapshot_flags::full, 0, 0, v1));
        auto s1 = snapshot::parse(v1.data(), v1.size());
        CHECK(s1.version() == 1);
        CHECK(!s1.has_split());
    }
    {   // payload corruption inside a range is caught by its own checksum
        // flip a byte inside the first payload, well past the header+descriptor
        auto bad = buf;
        const size_t target = 104 + 113 + 8;
        CHECK(bad.size() > target);
        bad[target] ^= 0xFF;
        bool threw = false;
        try { (void) snapshot::parse(bad.data(), bad.size()); }
        catch (const kv_mixed_error &) { threw = true; }
        CHECK(threw);
    }
}

static void test_v2_partial_and_body_only() {
    std::vector<std::vector<uint8_t>> locals, overflows;
    auto layers = build_split_layers(locals, overflows, 2);
    const auto md = std::vector<uint8_t>{ 'm', 'd' };
    serialize_options ser;
    ser.split = make_split();

    {   // partial snapshot: kvarn ranges carry overlays, qx ranges are references
        auto pl = layers;
        // layout: [0]=layer0 local (kvarn), [1]=layer0 overflow (qx),
        //         [2]=layer1 local (kvarn), [3]=layer1 overflow (qx)
        pl[0].desc.payload_mode = mode_partial_overlay;
        pl[1].desc.payload_mode = mode_resident_reference;
        pl[1].desc.payload_bytes = 0;
        pl[1].size = 0;
        pl[2].desc.payload_mode = mode_partial_overlay;
        pl[3].desc.payload_mode = mode_resident_reference;
        pl[3].desc.payload_bytes = 0;
        pl[3].size = 0;
        std::vector<uint8_t> buf;
        CHECK(snapshot::serialize(pl, {}, 5, md.data(), md.size(),
                snapshot_flags::partial, 77, 88, buf, {}, ser));
        auto s = snapshot::parse(buf.data(), buf.size());
        CHECK(s.has_split());
        CHECK(uint32_t(s.flags() & snapshot_flags::partial) != 0);
        for (size_t i = 0; i < s.layers().size(); ++i) {
            const layer_desc & d = s.layers()[i];
            if (d.range_id == 0) {
                // local kvarn range: overlay bytes travel with the snapshot
                CHECK(d.payload_mode == mode_partial_overlay);
                CHECK(s.payload_size(i) > 0);
            } else {
                // overflow qx range: resident reference, no bytes
                CHECK(d.payload_mode == mode_resident_reference);
                CHECK(s.payload_size(i) == 0);
                // the descriptor still declares its full [P,C) extent
                CHECK(d.range_start == SPLIT_P && d.range_end == SPLIT_C);
            }
        }
    }
    {   // body_only with a split envelope: no cells, no metadata blob
        std::vector<uint8_t> buf;
        CHECK(snapshot::serialize(layers, {}, 0, nullptr, 0,
                snapshot_flags::body_only, 0, 0, buf, {}, ser));
        auto s = snapshot::parse(buf.data(), buf.size());
        CHECK(s.has_split());
        CHECK(s.cells().empty());
        CHECK(s.metadata_size() == 0);
    }
}

// XXL import shape: a pure local cache whose logical occupancy never reaches P.
// The snapshot must carry the overflow range as a descriptor with no payload,
// so a fresh process can import it without prefilling the prefix.
static void test_v2_xxl_import_shape() {
    std::vector<uint8_t> local(4096, 0);
    fill_pattern(local, 99);
    layer_desc l = make_kvarn_layer(0, kv4v4, 4, 4,
            uint8_t(kvarn_domain::rotated), SPLIT_P, SPLIT_P / 128, 0);
    l.range_id = 0;
    l.range_start = 0;
    l.range_end = SPLIT_P;
    l.payload_bytes = local.size();

    layer_desc o = make_qx_layer(0, sx_q4_0, sx_q4_0, SPLIT_C - SPLIT_P, 0);
    o.range_id = 1;
    o.range_start = SPLIT_P;
    o.range_end = SPLIT_C;
    o.payload_bytes = 0; // overflow empty for an XXL import

    std::vector<cell_entry> cells;
    for (uint32_t i = 0; i < 3; ++i) {
        cell_entry c;
        c.pos = int32_t(i);
        c.tok = int32_t(500 + i);
        c.seq_ids.push_back(0);
        cells.push_back(c);
    }

    const auto md = std::vector<uint8_t>{ 'x', 'x', 'l' };
    std::vector<uint8_t> buf;
    serialize_options ser;
    ser.split = make_split();
    CHECK(snapshot::serialize({ { l, local.data(), local.size() },
                               { o, nullptr, 0 } },
            cells, 2, md.data(), md.size(), snapshot_flags::full, 0, 0, buf, {}, ser));

    auto s = snapshot::parse(buf.data(), buf.size());
    CHECK(s.layers().size() == 2);
    CHECK(s.has_split());
    // the local range is fully populated, the overflow range is present but empty
    CHECK(s.layers()[0].payload_bytes == local.size());
    CHECK(s.layers()[1].payload_bytes == 0);
    CHECK(s.layers()[1].range_start == SPLIT_P);
    CHECK(s.layers()[1].range_end == SPLIT_C);
    // and the descriptor set still validates as a complete split
    expected_snapshot e;
    e.capacity_cells = SPLIT_C;
    e.metadata_version = 2;
    e.split = make_split();
    for (const layer_desc & d : s.layers()) {
        e.layers.push_back(expected_layer::from_desc(d));
    }
    s.validate_expected_snapshot(e);
}


// §3.6 negative fallback matrix: a staging failure at ANY point must abort and
// leave the destination untouched, for EVERY owner (target / draft) and for
// both payload transport styles (full body and carry = partial overlay +
// resident reference). abort() must never run after finish(); finish() is the
// only hook allowed to touch the destination, and only after every stage
// succeeded. Ticket consumption means no partial restore can follow a failure.
static void test_owner_fallback_matrix() {
    struct Owner { uint8_t id; const char * name; };
    const Owner owners[] = {
        { (uint8_t) owner_target,    "target" },
        { (uint8_t) owner_draft_mtp, "draft"  },
    };

    for (const Owner & o : owners) {
        // ---- full body: fail at layer 1, fail at layer 2, or no failure ----
        std::vector<uint8_t> p0(128), p1(128);
        fill_pattern(p0, 201); fill_pattern(p1, 202);
        std::vector<uint8_t> md; make_metadata(md, 203);
        layer_desc l1 = make_qx_layer(1, sx_q4_0, sx_q4_0, 16, p0.size()); l1.owner = o.id;
        layer_desc l2 = make_qx_layer(2, sx_q4_0, sx_q4_0, 16, p1.size()); l2.owner = o.id;
        std::vector<layer_blob> layers = { { l1, p0.data(), p0.size() },
                                           { l2, p1.data(), p1.size() } };
        std::vector<uint8_t> buf;
        // 3rd arg is the metadata SCHEMA version (matches exp.metadata_version)
        CHECK(snapshot::serialize(layers, {}, 1, md.data(), md.size(),
                                  snapshot_flags::full, 0, 0, buf));

        for (uint32_t fail_id = 0; fail_id <= 2; ++fail_id) {
            snapshot s = snapshot::parse(buf.data(), buf.size());
            expected_snapshot exp;
            for (const layer_blob & b : layers) {
                exp.layers.push_back(expected_layer::from_desc(b.desc));
            }
            exp.capacity_cells = 32;
            exp.metadata_version = 1;
            s.validate_expected_snapshot(exp);

            int staged = 0, aborted = 0, finished = 0, dest = 0;
            auto attempt = [&]() {
                s.commit({
                    [&](uint32_t id, uint32_t, const layer_desc &, const uint8_t *, size_t) {
                        ++staged;
                        CHECK(dest == 0);              // destination untouched before finish
                        return id != fail_id;
                    },
                    [&](uint32_t, const uint8_t *, size_t) { CHECK(dest == 0); return true; },
                    [&]() { ++finished; dest = 1; },   // only finish may touch
                    [&]() { ++aborted; CHECK(finished == 0); }, // never after finish
                });
            };
            if (fail_id != 0) {
                CHECK_THROWS(attempt());
                CHECK(staged == (int) fail_id);  // stopped at the failing layer
                CHECK(aborted == 1);
                CHECK(finished == 0);
                CHECK(dest == 0);                // destination intact -> explicit miss/reprocess
                CHECK_THROWS(attempt());         // ticket consumed: no partial restore follows
            } else {
                attempt();
                CHECK(staged == 2);
                CHECK(aborted == 0);
                CHECK(finished == 1);
                CHECK(dest == 1);
            }
        }

        // ---- metadata stage failure: abort, destination intact ----
        {
            snapshot s = snapshot::parse(buf.data(), buf.size());
            expected_snapshot exp;
            for (const layer_blob & b : layers) {
                exp.layers.push_back(expected_layer::from_desc(b.desc));
            }
            exp.capacity_cells = 32;
            exp.metadata_version = 1;
            s.validate_expected_snapshot(exp);
            int aborted = 0, finished = 0, dest = 0;
            CHECK_THROWS(s.commit({
                [&](uint32_t, uint32_t, const layer_desc &, const uint8_t *, size_t) {
                    CHECK(dest == 0); return true;
                },
                [&](uint32_t, const uint8_t *, size_t) { CHECK(dest == 0); return false; },
                [&]() { ++finished; dest = 1; },
                [&]() { ++aborted; CHECK(finished == 0); },
            }));
            CHECK(aborted == 1);
            CHECK(finished == 0);
            CHECK(dest == 0);
        }

        // ---- carry: partial overlay travels, Qx body is a resident reference.
        //      Failure on the overlay layer must abort with the destination intact;
        //      the reference layer must never be asked for bytes.
        {
            std::vector<uint8_t> po(256); fill_pattern(po, 211);
            std::vector<uint8_t> mdc; make_metadata(mdc, 212);
            layer_desc k = make_kvarn_layer(1, kv4v5, 4, 5, uint8_t(kvarn_domain::rotated), 32, 2, po.size());
            k.owner = o.id;
            k.payload_mode = mode_partial_overlay;
            layer_desc q = make_qx_layer(2, sx_q4_0, sx_q4_0, 32, 0);
            q.owner = o.id;
            q.payload_mode = mode_resident_reference;
            q.payload_bytes = 0;
            std::vector<layer_blob> cl = { { k, po.data(), po.size() }, { q, nullptr, 0 } };
            std::vector<uint8_t> cbuf;
            CHECK(snapshot::serialize(cl, {}, 2, mdc.data(), mdc.size(),
                                      snapshot_flags::partial, 42, 7, cbuf));

            for (uint32_t fail_id = 1; fail_id <= 2; ++fail_id) {
                snapshot s = snapshot::parse(cbuf.data(), cbuf.size());
                expected_snapshot exp;
                for (const layer_blob & b : cl) exp.layers.push_back(expected_layer::from_desc(b.desc));
                exp.capacity_cells = 32;
                exp.metadata_version = 2;
                exp.owner_epoch = 42;
                exp.owner_id = 7;
                s.validate_expected_snapshot(exp);
                int aborted = 0, finished = 0, dest = 0, refs_asked = 0;
                CHECK_THROWS(s.commit({
                    [&](uint32_t id, uint32_t, const layer_desc & d, const uint8_t * ptr, size_t n) {
                        CHECK(dest == 0);
                        if (d.payload_mode == mode_resident_reference) {
                            ++refs_asked;
                            CHECK(ptr == nullptr && n == 0);
                        }
                        return id != fail_id;
                    },
                    [&](uint32_t, const uint8_t *, size_t) { CHECK(dest == 0); return true; },
                    [&]() { ++finished; dest = 1; },
                    [&]() { ++aborted; CHECK(finished == 0); },
                }));
                CHECK(aborted == 1);
                CHECK(finished == 0);
                CHECK(dest == 0);
                CHECK(refs_asked == (fail_id == 1 ? 0 : 1)); // stage stops at the failing layer
            }
            // success path: overlay staged with bytes, reference with none
            {
                snapshot s = snapshot::parse(cbuf.data(), cbuf.size());
                expected_snapshot exp;
                for (const layer_blob & b : cl) exp.layers.push_back(expected_layer::from_desc(b.desc));
                exp.capacity_cells = 32;
                exp.metadata_version = 2;
                exp.owner_epoch = 42;
                exp.owner_id = 7;
                s.validate_expected_snapshot(exp);
                int overlay_n = -1, refs = 0, finished = 0;
                s.commit({
                    [&](uint32_t, uint32_t, const layer_desc & d, const uint8_t * ptr, size_t n) {
                        if (d.payload_mode == mode_partial_overlay) {
                            overlay_n = (int) n;
                            CHECK(ptr != nullptr && n == po.size());
                        } else {
                            ++refs;
                            CHECK(ptr == nullptr && n == 0);
                        }
                        return true;
                    },
                    [&](uint32_t, const uint8_t *, size_t) { return true; },
                    [&]() { ++finished; },
                    [&]() { CHECK(false); },
                });
                CHECK(overlay_n == (int) po.size());
                CHECK(refs == 1);
                CHECK(finished == 1);
            }
        }
    }
}


// F4 plan 3.6: authorized PURE KVarN -> position_split import adapter.
// Positive case plus the negatives that "equal byte length" must not satisfy:
// format, dimension, stride, domain, model, RoPE, duplicate source, missing
// range, corrupt payload, byte-carrying synthesis, unavailable identity.
// F4 plan 3.6: authorized PURE KVarN -> position_split import adapter.
// The SOURCE identity now travels in the frame (KMS3, header-crc protected and
// bound to these exact bytes). These cases cover the identity/format negatives
// the reviewer required: legacy v1 (no identity), mismatched model/RoPE,
// config mismatch, unknown version, truncation, corruption, equal-byte-length
// incompatible format/dimension/stride/domain, duplicate/missing range,
// byte-carrying synthesis, and "stale" state across loads.

// ---------------------------------------------------------------------------
// F4 normalized STATE fingerprint (acceptance addendum).
//
// Hashes ONLY documented SEMANTIC state:
//   * KV payload            -> XXH64 over every mapped payload byte
//   * table/cell metadata   -> cell_entry {pos, x, y, tok, seq_ids}
//   * descriptor contract   -> the config-relevant layer_desc fields
//   * recurrent state       -> caller-supplied recurrent/DeltaNet region
//
// EXCLUDED as documented irrelevant envelope metadata (never part of the
// fingerprint): header magic/version/checksum bytes, owner_epoch, owner_id,
// snapshot container/footer, file path, source/backend identity, generation
// counters and any transient load statistics. Excluding only these is what
// makes "fingerprints agree" meaningful; whole-file SHA would not be.
uint64_t normalized_state_fingerprint(const snapshot & s,
                                      const void * recurrent, size_t recurrent_bytes) {
    XXH64_state_t h;
    XXH64_reset(&h, 0);
    const auto put = [&](const void * d, size_t n) { XXH64_update(&h, d, n); };

    // 1) descriptor contract + payload identity (sorted for stability)
    std::vector<size_t> order(s.layers().size());
    for (size_t i = 0; i < order.size(); ++i) { order[i] = i; }
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        const layer_desc & x = s.layers()[a];
        const layer_desc & y = s.layers()[b];
        if (x.layer_id != y.layer_id) { return x.layer_id < y.layer_id; }
        return x.range_id < y.range_id;
    });
    for (size_t i : order) {
        const layer_desc & d = s.layers()[i];
        put(&d.kind, sizeof(d.kind));
        put(&d.k_type, sizeof(d.k_type));
        put(&d.v_type, sizeof(d.v_type));
        put(&d.k_bits, sizeof(d.k_bits));
        put(&d.v_bits, sizeof(d.v_bits));
        put(&d.kvarn_domain, sizeof(d.kvarn_domain));
        put(&d.layout, sizeof(d.layout));
        put(&d.tail_type, sizeof(d.tail_type));
        put(&d.token_group, sizeof(d.token_group));
        put(&d.record_dim, sizeof(d.record_dim));
        put(&d.head_dim_k, sizeof(d.head_dim_k));
        put(&d.head_dim_v, sizeof(d.head_dim_v));
        put(&d.head_slices, sizeof(d.head_slices));
        put(&d.n_head_kv, sizeof(d.n_head_kv));
        put(&d.k_stride, sizeof(d.k_stride));
        put(&d.v_stride, sizeof(d.v_stride));
        put(&d.payload_cells, sizeof(d.payload_cells));
        put(&d.payload_rows, sizeof(d.payload_rows));
        put(&d.payload_bytes, sizeof(d.payload_bytes));
        put(&d.payload_checksum, sizeof(d.payload_checksum)); // content identity
        put(&d.range_id, sizeof(d.range_id));
        // 2) the KV payload itself
        const uint8_t * payload = s.payload(i);
        const size_t n = s.payload_size(i);
        put(&n, sizeof(n));
        if (payload != nullptr && n != 0) {
            put(payload, n);
        }
    }
    // 3) the cell/table metadata
    const auto & cells = s.cells();
    const uint64_t n_cells = cells.size();
    put(&n_cells, sizeof(n_cells));
    for (const cell_entry & c : cells) {
        put(&c.pos, sizeof(c.pos));
        put(&c.x, sizeof(c.x));
        put(&c.y, sizeof(c.y));
        put(&c.tok, sizeof(c.tok));
        const uint64_t n_seq = c.seq_ids.size();
        put(&n_seq, sizeof(n_seq));
        if (n_seq != 0) { put(c.seq_ids.data(), n_seq * sizeof(int32_t)); }
    }
    // 4) the recurrent/DeltaNet region, when the caller supplies it
    put(&recurrent_bytes, sizeof(recurrent_bytes));
    if (recurrent != nullptr && recurrent_bytes != 0) {
        put(recurrent, recurrent_bytes);
    }
    return XXH64_digest(&h);
}

static void test_pure_import_adapter() {
    std::vector<uint8_t> p(512);
    fill_pattern(p, 71);
    std::vector<uint8_t> md;
    make_metadata(md, 72);
    layer_desc d = make_kvarn_layer(1, kv4v5, 4, 5, uint8_t(kvarn_domain::rotated), 32, 2, p.size());
    d.range_id = 0;
    d.range_start = 0;
    d.range_end = 0;

    const uint64_t kModel = 0x1111222233334444ull;
    const uint64_t kRoPE  = 0xaaaabbbbccccddddull;
    const uint64_t kCfg   = descriptor_config_id({ d });

    // KMS3 frame carrying the ORIGINAL WRITER identity
    const auto frame_kms3 = [&](uint64_t m, uint64_t r, uint64_t c) {
        std::vector<uint8_t> out;
        serialize_options ser;
        source_identity sid;
        sid.model_id = m; sid.rope_id = r; sid.config_id = c;
        ser.source = sid;
        CHECK(snapshot::serialize({ { d, p.data(), p.size() } }, {}, 3, md.data(), md.size(),
                                  snapshot_flags::full, 0, 0, out, {}, ser));
        return out;
    };
    const auto frame_ok = [&]() { return frame_kms3(kModel, kRoPE, kCfg); };
    // legacy v1 frame (no identity) -- old read path preserved
    const auto frame_v1 = [&]() {
        std::vector<uint8_t> out;
        CHECK(snapshot::serialize({ { d, p.data(), p.size() } }, {}, 3, md.data(), md.size(),
                                  snapshot_flags::full, 0, 0, out));
        return out;
    };

    const uint32_t kP = 512, kC = 1024;
    const auto want = [&]() {
        position_split w;
        w.p = kP; w.c = kC; w.model_id = kModel; w.rope_id = kRoPE;
        return w;
    };
    const auto build_dest = [&](const layer_desc & src, std::vector<layer_desc> & out) {
        out.clear();
        layer_desc local = src;
        local.range_start = 0;
        local.range_end = kP;
        out.push_back(local);
        layer_desc ovf = src;
        ovf.range_id = 1;
        ovf.range_start = kP;
        ovf.range_end = kC;
        ovf.payload_bytes = 0;
        ovf.payload_checksum = 0;
        out.push_back(ovf);
    };
    const auto imp = [&](bool ok, const auto & tag) {
        ++g_checks;
        if (!ok) {
            ++g_failures;
            std::fprintf(stderr, "FAIL: pure import: %s\n", std::string(tag).c_str());
        }
    };
    const auto rejects = [&](const std::vector<uint8_t> & frame, const char * tag,
                             const position_split * force_want = nullptr) {
        bool threw = false;
        std::string msg;
        try {
            snapshot s = snapshot::parse(frame.data(), frame.size());
            std::vector<layer_desc> dest;
            build_dest(d, dest);
            s.adopt_split_for_import(force_want ? *force_want : want(), dest,
                                     s.frame_source_identity());
        } catch (const std::exception & e) {
            threw = true;
            msg = e.what();
        }
        imp(threw, std::string("rejects ") + tag + (threw ? "" : std::string(" [no throw]")));
        return msg;
    };

    {   // POSITIVE: KMS3 identity accepted, geometry adopted, overflow empty,
        // provenance and payload preserved.
        const std::vector<uint8_t> f = frame_ok();
        snapshot s = snapshot::parse(f.data(), f.size());
        imp(s.has_source_identity(), "positive: KMS3 frame carries source identity");
        imp(s.frame_source_identity().model_id == kModel &&
            s.frame_source_identity().rope_id == kRoPE &&
            s.frame_source_identity().config_id == kCfg,
            "positive: source identity round-trips through the header");
        std::vector<layer_desc> dest;
        build_dest(d, dest);
        bool threw = false;
        try { s.adopt_split_for_import(want(), dest, s.frame_source_identity()); }
        catch (const std::exception &) { threw = true; }
        imp(!threw, "positive: authorized pure->split adopt succeeds");
        imp(s.has_split() && s.split()->p == kP && s.split()->c == kC,
            "positive: destination envelope adopted");
        imp(s.layers().size() == 2 && s.layers()[1].range_id == 1 &&
            s.layers()[1].payload_bytes == 0,
            "positive: empty overflow descriptor synthesized");
        imp(s.layers()[0].range_start == 0 && s.layers()[0].range_end == kP,
            "positive: local range geometry adopted");
        imp(s.payload_size(0) == p.size() && s.payload(0) != nullptr,
            "positive: mapped payload bytes untouched");
        imp(s.layers()[0].k_stride == d.k_stride && s.layers()[0].head_dim_k == d.head_dim_k,
            "positive: source dimensions/strides preserved");
    }

    {   // legacy v1 frame: no source identity -> fail closed (old read path kept)
        const std::vector<uint8_t> f = frame_v1();
        snapshot s = snapshot::parse(f.data(), f.size());
        imp(!s.has_source_identity(), "legacy v1 frame parses with NO source identity");
        const std::string msg = rejects(f, "a legacy v1 frame with no source identity");
        imp(msg.find("no source identity") != std::string::npos,
            "legacy refusal names the missing provenance");
    }

    {   // mismatched source model / RoPE (destination never substituted)
        rejects(frame_kms3(kModel + 1, kRoPE, kCfg), "a mismatched SOURCE model identity");
        rejects(frame_kms3(kModel, kRoPE + 1, kCfg), "a mismatched SOURCE RoPE identity");
        // config id that does not describe the frame's own descriptors
        rejects(frame_kms3(kModel, kRoPE, kCfg ^ 0x5a5a5a5a5a5a5a5aull),
                "a SOURCE config id that does not match its descriptors");
        // equal-size incompatible identity against a DIFFERENT destination
        position_split other = want();
        other.model_id = kModel ^ 0xff;
        rejects(frame_ok(), "an equal-byte-size frame against another destination identity", &other);
    }

    {   // format negatives: unknown version, truncation, header corruption
        std::vector<uint8_t> f = frame_ok();
        // unknown version: bump version to 9 and repair the prefix checksum
        f[4] = 9;
        std::vector<uint8_t> prefix(f.begin(), f.begin() + 96);
        std::memset(prefix.data() + 88, 0, 8);
        const uint64_t crc = XXH64(prefix.data(), 96, 0);
        for (int i = 0; i < 8; ++i) { f[88 + i] = uint8_t((crc >> (8 * i)) & 0xFF); }
        bool threw = false; std::string msg;
        try { snapshot::parse(f.data(), f.size()); } catch (const std::exception & e) { threw = true; msg = e.what(); }
        imp(threw && msg.find("unsupported snapshot version") != std::string::npos,
            "unknown version is refused before any read");

        // truncation
        f = frame_ok();
        f.resize(64);
        threw = false; msg.clear();
        try { snapshot::parse(f.data(), f.size()); } catch (const std::exception & e) { threw = true; msg = e.what(); }
        imp(threw && msg.find("truncated header") != std::string::npos,
            "truncated header is refused");

        // corruption: flip a byte inside the protected header
        f = frame_ok();
        f[20] ^= 0xFF;
        threw = false; msg.clear();
        try { snapshot::parse(f.data(), f.size()); } catch (const std::exception & e) { threw = true; msg = e.what(); }
        imp(threw && msg.find("header checksum mismatch") != std::string::npos,
            "corrupted header is refused by the checksum");

        // corruption of the identity block itself (crc over identity only)
        f = frame_ok();
        f[96] ^= 0xFF;
        threw = false; msg.clear();
        try { snapshot::parse(f.data(), f.size()); } catch (const std::exception & e) { threw = true; msg = e.what(); }
        imp(threw && msg.find("source identity checksum mismatch") != std::string::npos,
            "corrupted source identity is refused by its checksum");
    }

    {   // stale carrier: a prior successful adopt must not authorize a later
        // legacy frame -- provenance lives in the frame, never in process state.
        snapshot good = snapshot::parse(frame_ok().data(), frame_ok().size());
        std::vector<layer_desc> dest;
        build_dest(d, dest);
        bool threw = false;
        try { good.adopt_split_for_import(want(), dest, good.frame_source_identity()); }
        catch (const std::exception &) { threw = true; }
        imp(!threw, "stale: first load adopts successfully");
        snapshot later = snapshot::parse(frame_v1().data(), frame_v1().size());
        threw = false;
        try { later.adopt_split_for_import(want(), dest, later.frame_source_identity()); }
        catch (const std::exception &) { threw = true; }
        imp(threw, "stale: a later legacy frame is still refused (no retained authorization)");
    }

    {   // equal byte length, incompatible format / dimension / stride / domain
        const auto bad = [&](const layer_desc & x, const char * tag) {
            snapshot s = snapshot::parse(frame_ok().data(), frame_ok().size());
            std::vector<layer_desc> dest;
            build_dest(x, dest);
            bool threw = false;
            try { s.adopt_split_for_import(want(), dest, s.frame_source_identity()); }
            catch (const std::exception &) { threw = true; }
            imp(threw, std::string("rejects ") + tag);
        };
        layer_desc x = d; x.k_type = uint16_t(sx_q8_0); bad(x, "incompatible K format (equal length)");
        x = d; x.head_dim_k = 128;                bad(x, "incompatible head dimension");
        x = d; x.k_stride = d.k_stride + 8;       bad(x, "incompatible stride");
        x = d; x.kvarn_domain = uint8_t(0x7f);    bad(x, "incompatible KVarN domain");
        x = d; x.layout = uint16_t(layout_standard_qx_rows); bad(x, "incompatible layout");
        x = d; x.tail_type = uint16_t(sx_q4_0);   bad(x, "incompatible tail type");
        x = d; x.record_dim = 8;                  bad(x, "incompatible record dimension");
    }

    {   // source range the destination does not own
        snapshot s = snapshot::parse(frame_ok().data(), frame_ok().size());
        std::vector<layer_desc> dest;
        build_dest(d, dest);
        for (layer_desc & dd : dest) { dd.layer_id = 2; }
        bool threw = false;
        try { s.adopt_split_for_import(want(), dest, s.frame_source_identity()); }
        catch (const std::exception &) { threw = true; }
        imp(threw, "rejects a source range the destination does not own");
    }

    {   // synthesized non-empty overflow must be refused
        snapshot s = snapshot::parse(frame_ok().data(), frame_ok().size());
        std::vector<layer_desc> dest;
        build_dest(d, dest);
        dest[1].payload_bytes = 64;
        bool threw = false;
        try { s.adopt_split_for_import(want(), dest, s.frame_source_identity()); }
        catch (const std::exception &) { threw = true; }
        imp(threw, "refuses a synthesized non-empty overflow range");
    }
}


// Positive mutation controls: a fingerprint that never moves would make
// "fingerprints agree" vacuous, so each semantic class is mutated and the
// fingerprint must change. Envelope-only mutations (owner_epoch/owner_id) are
// mutated too and MUST NOT move it, proving the exclusions are real.
static void test_fingerprint_controls() {
    std::vector<uint8_t> p(512);
    fill_pattern(p, 91);
    std::vector<uint8_t> md;
    make_metadata(md, 92);
    layer_desc d = make_kvarn_layer(1, kv4v5, 4, 5, uint8_t(kvarn_domain::rotated), 32, 2, p.size());
    std::vector<uint8_t> buf;
    CHECK(snapshot::serialize({ { d, p.data(), p.size() } }, {}, 3, md.data(), md.size(),
                              snapshot_flags::full, 0, 0, buf));
    snapshot base = snapshot::parse(buf.data(), buf.size());
    std::vector<uint8_t> rec(256);
    fill_pattern(rec, 93);
    const uint64_t f0 = normalized_state_fingerprint(base, rec.data(), rec.size());
    const auto imp = [&](bool ok, const auto & tag) {
        ++g_checks;
        if (!ok) { ++g_failures; std::fprintf(stderr, "FAIL: fingerprint: %s\n", std::string(tag).c_str()); }
    };
    imp(f0 != 0, "baseline fingerprint is non-zero");

    // (a) KV payload mutation must move it
    {
        std::vector<uint8_t> p2 = p;
        p2[17] ^= 0xFF;
        std::vector<uint8_t> b2;
        CHECK(snapshot::serialize({ { d, p2.data(), p2.size() } }, {}, 3, md.data(), md.size(),
                                  snapshot_flags::full, 0, 0, b2));
        snapshot s = snapshot::parse(b2.data(), b2.size());
        imp(normalized_state_fingerprint(s, rec.data(), rec.size()) != f0,
            "KV payload mutation changes the fingerprint");
    }
    // (b) table/cell mutation must move it
    {
        std::vector<cell_entry> cells;
        cell_entry c;
        c.pos = 7; c.x = 1; c.y = 2; c.tok = 99; c.seq_ids = {0};
        cells.push_back(c);
        std::vector<uint8_t> b3;
        CHECK(snapshot::serialize({ { d, p.data(), p.size() } }, cells, 3, md.data(), md.size(),
                                  snapshot_flags::full, 0, 0, b3));
        snapshot s = snapshot::parse(b3.data(), b3.size());
        imp(normalized_state_fingerprint(s, rec.data(), rec.size()) != f0,
            "table/cell mutation changes the fingerprint");
        cells[0].tok = 100;
        std::vector<uint8_t> b4;
        CHECK(snapshot::serialize({ { d, p.data(), p.size() } }, cells, 3, md.data(), md.size(),
                                  snapshot_flags::full, 0, 0, b4));
        snapshot s2 = snapshot::parse(b4.data(), b4.size());
        const uint64_t f4 = normalized_state_fingerprint(s2, rec.data(), rec.size());
        imp(f4 != f0 && f4 != normalized_state_fingerprint(s, rec.data(), rec.size()),
            "a single cell field change moves the fingerprint");
    }
    // (c) recurrent/DeltaNet region mutation must move it
    {
        std::vector<uint8_t> rec2 = rec;
        rec2[33] ^= 0xFF;
        imp(normalized_state_fingerprint(base, rec2.data(), rec2.size()) != f0,
            "recurrent/DeltaNet region mutation changes the fingerprint");
        imp(normalized_state_fingerprint(base, rec.data(), rec.size()) == f0,
            "same inputs reproduce the same fingerprint (determinism)");
    }
    // (d) documented envelope-only metadata must NOT move it
    {
        std::vector<uint8_t> b5;
        CHECK(snapshot::serialize({ { d, p.data(), p.size() } }, {}, 3, md.data(), md.size(),
                                  snapshot_flags::full, /*owner_epoch*/ 777, /*owner_id*/ 888, b5));
        snapshot s = snapshot::parse(b5.data(), b5.size());
        imp(normalized_state_fingerprint(s, rec.data(), rec.size()) == f0,
            "documented envelope metadata (owner_epoch/owner_id) is excluded");
    }
    // (e) descriptor contract mutation must move it
    {
        layer_desc d2 = d;
        d2.k_stride = d.k_stride + 8;
        std::vector<uint8_t> b6;
        CHECK(snapshot::serialize({ { d2, p.data(), p.size() } }, {}, 3, md.data(), md.size(),
                                  snapshot_flags::full, 0, 0, b6));
        snapshot s = snapshot::parse(b6.data(), b6.size());
        imp(normalized_state_fingerprint(s, rec.data(), rec.size()) != f0,
            "descriptor contract mutation changes the fingerprint");
    }
}


// Shared transaction preflight: the comment on disk-backed spool protection has
// to be TRUE, so the three refusal classes are exercised directly.
static void test_spool_preflight() {
    const auto imp = [&](bool ok, const auto & tag) {
        ++g_checks;
        if (!ok) { ++g_failures; std::fprintf(stderr, "FAIL: spool: %s\n", std::string(tag).c_str()); }
    };
    std::string err;

    // (1) empty
    err.clear();
    imp(!llama_state_rollback::spool_usable("", &err) && !err.empty(),
        "empty spool is refused with a reason");

    // (2) not a directory (a regular file)
    const std::string file_path = "/home/hjotha/f4-spool/f4-not-a-dir.bin";
    { FILE * f = fopen(file_path.c_str(), "wb"); if (f) { fputc('x', f); fclose(f); } }
    err.clear();
    imp(!llama_state_rollback::spool_usable(file_path, &err),
        "a non-directory path is refused");
    std::remove(file_path.c_str());

    // (3) unwritable directory
    const std::string locked = "/home/hjotha/f4-spool/f4-locked";
    std::filesystem::create_directories(locked);
    std::filesystem::permissions(locked, std::filesystem::perms::owner_read |
                                          std::filesystem::perms::owner_exec);
    err.clear();
    const bool locked_ok = llama_state_rollback::spool_usable(locked, &err);
    std::filesystem::permissions(locked, std::filesystem::perms::owner_all);
    // refusal happens either while preparing <spool>/beellama-mixed-state or
    // on the write probe; both are legitimate, but it MUST refuse with a reason
    imp(!locked_ok && !err.empty() &&
            (err.find("writable") != std::string::npos ||
             err.find("cannot create") != std::string::npos),
        std::string("an unwritable directory is refused with a reason (") + err + ")");

    // (4) tmpfs / ramfs (the exact class the mixed-state reader refuses)
#if defined(__linux__)
    if (std::filesystem::is_directory("/dev/shm")) {
        err.clear();
        const bool shm_ok = llama_state_rollback::spool_usable("/dev/shm", &err);
        imp(!shm_ok && err.find("tmpfs") != std::string::npos,
            "a tmpfs spool directory is refused with a reason");
    }
#endif

    // (5) a real disk-backed directory is accepted and prepares its subdir
    const std::string good = "/home/hjotha/f4-spool/tmp-preflight";
    std::filesystem::create_directories(good);
    err.clear();
    imp(llama_state_rollback::spool_usable(good, &err),
        std::string("a disk-backed directory is accepted (") + err + ")");
    imp(std::filesystem::is_directory(std::filesystem::path(good) / "beellama-mixed-state"),
        "the shared preflight prepares <spool>/beellama-mixed-state");
    // calling it twice must stay idempotent
    err.clear();
    imp(llama_state_rollback::spool_usable(good, &err), "preflight is idempotent");
}


// KMS3 move semantics: both move operations must carry the ORIGINAL WRITER
// identity, or a parse() that returns by value silently loses it (observed:
// version read fine, identity all-zero). Covers owning AND borrowed modes,
// moved-from safety, and transaction behaviour after the move.
static void test_kms3_move_identity() {
    std::vector<uint8_t> p(256);
    fill_pattern(p, 101);
    std::vector<uint8_t> md;
    make_metadata(md, 102);
    layer_desc d = make_kvarn_layer(1, kv4v5, 4, 5, uint8_t(kvarn_domain::rotated), 32, 2, p.size());
    const uint64_t kModel = 0x1111222233334444ull;
    const uint64_t kRoPE  = 0xaaaabbbbccccddddull;
    const uint64_t kCfg   = descriptor_config_id({ d });
    std::vector<uint8_t> buf;
    serialize_options ser;
    source_identity sid;
    sid.model_id = kModel; sid.rope_id = kRoPE; sid.config_id = kCfg;
    ser.source = sid;
    CHECK(snapshot::serialize({ { d, p.data(), p.size() } }, {}, 3, md.data(), md.size(),
                              snapshot_flags::full, 0, 0, buf, {}, ser));
    const auto imp = [&](bool ok, const auto & tag) {
        ++g_checks;
        if (!ok) { ++g_failures; std::fprintf(stderr, "FAIL: move-identity: %s\n", std::string(tag).c_str()); }
    };
    const auto same = [](const source_identity & a, const source_identity & b) {
        return a.model_id == b.model_id && a.rope_id == b.rope_id && a.config_id == b.config_id;
    };

    // (1) move CONSTRUCTION, owning mode
    {
        snapshot src = snapshot::parse(buf.data(), buf.size());
        imp(src.has_source_identity() && same(src.frame_source_identity(), sid),
            "parse keeps the identity (owning)");
        snapshot moved(std::move(src));
        imp(moved.version() == 3 && moved.has_source_identity() &&
                same(moved.frame_source_identity(), sid),
            "move CONSTRUCTION preserves the KMS3 source identity");
        imp(!src.has_source_identity(), "moved-from source no longer claims an identity");
        imp(!src.has_split() && src.layers().empty(),
            "moved-from source is inert (no layers, no envelope)");
    }

    // (2) move ASSIGNMENT, owning mode
    {
        snapshot a = snapshot::parse(buf.data(), buf.size());
        snapshot b;
        b = std::move(a);
        imp(b.version() == 3 && b.has_source_identity() &&
                same(b.frame_source_identity(), sid),
            "move ASSIGNMENT preserves the KMS3 source identity");
        imp(!a.has_source_identity(), "moved-from assign source has no identity");
    }

    // (3) BORROWED mode: payload pointers must stay valid after the move
    {
        auto owner = std::make_shared<std::vector<uint8_t>>(buf);
        snapshot src = snapshot::parse_view(owner->data(), owner->size(),
                                            std::static_pointer_cast<const void>(owner));
        const uint8_t * before = src.payload(0);
        snapshot moved(std::move(src));
        imp(moved.has_source_identity() && same(moved.frame_source_identity(), sid),
            "move CONSTRUCTION preserves identity in BORROWED mode");
        imp(moved.payload(0) == before && moved.payload_size(0) == p.size(),
            "borrowed payload pointer and size survive the move");
    }

    // (4b) SELF-MOVE must be a no-op (guarded) and must not lose the identity
    {
        snapshot s = snapshot::parse(buf.data(), buf.size());
        snapshot & alias = s;
        s = std::move(alias);
        imp(s.version() == 3 && s.has_source_identity() &&
                same(s.frame_source_identity(), sid),
            "self-move assignment keeps the KMS3 source identity");
        std::vector<layer_desc> dest0;
        layer_desc l0 = d; l0.range_start = 0; l0.range_end = 512; dest0.push_back(l0);
        imp(s.layers().size() == 1 && s.payload_size(0) == p.size(),
            "self-move keeps layers and payload intact");
        (void) dest0;
    }

    // (4c) parse -> move -> SERIALIZE: the identity must survive a round trip
    {
        snapshot parsed = snapshot::parse(buf.data(), buf.size());
        snapshot moved(std::move(parsed));
        std::vector<uint8_t> again;
        serialize_options ser2;
        ser2.source = moved.frame_source_identity();
        CHECK(snapshot::serialize({ { d, p.data(), p.size() } }, {}, 3, md.data(), md.size(),
                                  snapshot_flags::full, 0, 0, again, {}, ser2));
        snapshot back = snapshot::parse(again.data(), again.size());
        imp(back.version() == 3 && back.has_source_identity() &&
                same(back.frame_source_identity(), sid),
            "parse -> move -> serialize preserves the KMS3 source identity");
    }

    // (4) transaction behaviour after the move: the identity must still gate
    //     the authorized pure->split import (a moved-from frame must refuse)
    {
        snapshot moved = snapshot::parse(buf.data(), buf.size());
        snapshot carrier(std::move(moved));
        position_split want;
        want.p = 512; want.c = 1024; want.model_id = kModel; want.rope_id = kRoPE;
        std::vector<layer_desc> dest;
        layer_desc local = d; local.range_start = 0; local.range_end = 512; dest.push_back(local);
        layer_desc ovf = d; ovf.range_id = 1; ovf.range_start = 512; ovf.range_end = 1024;
        ovf.payload_bytes = 0; ovf.payload_checksum = 0; dest.push_back(ovf);
        bool threw = false;
        try { carrier.adopt_split_for_import(want, dest, carrier.frame_source_identity()); }
        catch (const std::exception &) { threw = true; }
        imp(!threw && carrier.has_split(),
            "moved KMS3 frame still authorizes the pure->split transaction");
        bool refused = false;
        try { moved.adopt_split_for_import(want, dest, moved.frame_source_identity()); }
        catch (const std::exception &) { refused = true; }
        imp(refused, "moved-FROM frame refuses the transaction (no identity reuse)");
    }
}

static void test_split_to_pure_import_adapter() {
    const uint64_t model = 0x1111222233334444ull;
    const uint64_t rope = 0xaaaabbbbccccddddull;
    position_split split;
    split.p = 512; split.c = 1024; split.model_id = model; split.rope_id = rope;
    std::vector<uint8_t> payload(512), metadata;
    fill_pattern(payload, 81);
    make_metadata(metadata, 82);
    layer_desc pure = make_kvarn_layer(1, kv4v5, 4, 5,
            uint8_t(kvarn_domain::rotated), 2, 2, payload.size());
    pure.record_dim = 128;
    pure.head_slices = 2;
    layer_desc local = pure;
    local.range_start = 0; local.range_end = split.p;
    layer_desc overflow = make_qx_layer(1, sx_q4_0, sx_q4_0, 2, 0);
    overflow.owner = pure.owner;
    overflow.head_dim_v = pure.head_dim_v;
    overflow.v_rot_width = 256;
    overflow.range_id = 1; overflow.range_start = split.p; overflow.range_end = split.c;
    std::vector<cell_entry> cells(2);
    cells[0].pos = 0; cells[0].tok = 11; cells[0].seq_ids = {0};
    cells[1].pos = 1; cells[1].tok = 12; cells[1].seq_ids = {0};
    const auto frame = [&](const layer_desc & ovf, const std::vector<cell_entry> & table,
                           snapshot_flags flags = snapshot_flags::full) {
        auto k = local;
        k.payload_cells = uint32_t(table.size());
        k.payload_rows = uint64_t(std::count_if(table.begin(), table.end(), [](const cell_entry & c) { return c.pos >= 0; }));
        auto q = ovf;
        q.payload_cells = k.payload_cells; q.payload_rows = k.payload_rows;
        std::vector<uint8_t> qbytes(size_t(q.payload_bytes), 17);
        std::vector<uint8_t> encoded;
        serialize_options options; options.split = split;
        if (flags == snapshot_flags::partial) {
            k.payload_mode = mode_partial_overlay;
            q.payload_mode = mode_resident_reference;
            q.payload_bytes = 0;
            qbytes.clear();
        }
        CHECK(snapshot::serialize({{k, payload.data(), payload.size()}, {q, qbytes.data(), qbytes.size()}},
                table, 3, metadata.data(), metadata.size(), flags,
                flags == snapshot_flags::partial ? 10 : 0,
                flags == snapshot_flags::partial ? 20 : 0, encoded, {}, options));
        return encoded;
    };
    const auto valid_frame = frame(overflow, cells);
    const uint64_t original_hash = XXH64(valid_frame.data(), valid_frame.size(), 0);
    const auto expected = [&](const layer_desc & d, uint32_t capacity) {
        expected_snapshot want;
        want.capacity_cells = capacity; want.metadata_version = 3;
        want.layers = {expected_layer::from_desc(d)};
        return want;
    };
    for (bool borrowed : {false, true}) {
        auto backing = std::make_shared<std::vector<uint8_t>>(valid_frame);
        auto state = borrowed ? snapshot::parse_view(backing->data(), backing->size(), backing)
                              : snapshot::parse(backing->data(), backing->size());
        const auto original_checksum = state.layers()[0].payload_checksum;
        state.project_split_to_pure_for_import(256, model, rope, {pure});
        CHECK(!state.has_split());
        CHECK(state.has_source_identity());
        CHECK(state.frame_source_identity().model_id == model && state.frame_source_identity().rope_id == rope);
        CHECK(state.layers().size() == 1 && state.layers()[0].range_id == 0);
        CHECK(state.layers()[0].range_start == 0 && state.layers()[0].range_end == 0);
        CHECK(state.layers()[0].payload_checksum == original_checksum);
        CHECK(state.payload_size(0) == payload.size() && std::memcmp(state.payload(0), payload.data(), payload.size()) == 0);
        CHECK(state.metadata_size() == metadata.size() && std::memcmp(state.metadata_blob(), metadata.data(), metadata.size()) == 0);
        CHECK(XXH64(backing->data(), backing->size(), 0) == original_hash);
        state.validate_expected_snapshot(expected(pure, 256));
        std::vector<uint8_t> live{9, 8, 7}, staged;
        int finished = 0, split_staged = 0;
        commit_handlers handlers;
        handlers.stage_metadata = [&](uint32_t version, const uint8_t * data, size_t size) {
            CHECK(version == 3 && size == metadata.size() && std::memcmp(data, metadata.data(), size) == 0);
            CHECK(live == std::vector<uint8_t>({9, 8, 7}));
            return true;
        };
        handlers.stage_layer = [&](uint32_t layer, uint32_t range, const layer_desc &, const uint8_t * data, size_t size) {
            CHECK(layer == 1 && range == 0 && size == payload.size());
            CHECK(live == std::vector<uint8_t>({9, 8, 7}));
            staged.assign(data, data+size); return true;
        };
        handlers.stage_split = [&](const position_split &) { ++split_staged; return true; };
        handlers.finish = [&] { live = staged; ++finished; };
        handlers.abort = [&] { staged.clear(); };
        state.commit(handlers);
        CHECK(live == payload && finished == 1 && split_staged == 0);
    }
    const auto refuses = [&](const std::vector<uint8_t> & source, uint32_t capacity,
                             uint64_t destination_model, uint64_t destination_rope,
                             const std::vector<layer_desc> & destination) {
        auto state = snapshot::parse(source.data(), source.size());
        const auto state_before = normalized_state_fingerprint(state, nullptr, 0);
        const auto split_before = state.split();
        std::vector<uint8_t> live{9, 8, 7};
        int finished = 0;
        bool rejected = false;
        try {
            state.project_split_to_pure_for_import(capacity, destination_model, destination_rope, destination);
            state.validate_expected_snapshot(expected(pure, capacity));
            commit_handlers handlers;
            handlers.stage_metadata = [](uint32_t, const uint8_t *, size_t) { return true; };
            handlers.stage_layer = [](uint32_t, uint32_t, const layer_desc &, const uint8_t *, size_t) { return true; };
            handlers.finish = [&] { live.clear(); ++finished; };
            handlers.abort = [] {};
            state.commit(handlers);
        } catch (const kv_mixed_error &) { rejected = true; }
        CHECK(rejected && live == std::vector<uint8_t>({9, 8, 7}) && finished == 0);
        CHECK(state.has_split() && state.split()->p == split_before->p && state.split()->c == split_before->c);
        CHECK(normalized_state_fingerprint(state, nullptr, 0) == state_before);
    };
    refuses(valid_frame, 256, model^1, rope, {pure});
    refuses(valid_frame, 256, model, rope^1, {pure});
    refuses(valid_frame, 1, model, rope, {pure});
    refuses(valid_frame, 256, 0, rope, {pure});
    refuses(valid_frame, 256, model, rope, {});
    refuses(valid_frame, 256, model, rope, {pure, pure});
    auto unowned_layer = pure;
    unowned_layer.layer_id = 2;
    refuses(valid_frame, 256, model, rope, {unowned_layer});
    refuses(valid_frame, 256, model, rope, {pure, unowned_layer});
    auto occupied = cells;
    occupied[1].pos = int32_t(split.p);
    refuses(frame(overflow, occupied), 1024, model, rope, {pure});
    auto nonempty = overflow;
    nonempty.payload_bytes = 32;
    refuses(frame(nonempty, cells), 256, model, rope, {pure});
    auto wrong_overflow_geometry = overflow;
    wrong_overflow_geometry.head_dim_v = 512;
    refuses(frame(wrong_overflow_geometry, cells), 256, model, rope, {pure});
    refuses(frame(overflow, cells, snapshot_flags::partial), 256, model, rope, {pure});
    for (int mismatch = 0; mismatch < 6; ++mismatch) {
        auto incompatible = pure;
        if (mismatch == 0) { incompatible.kvarn_domain = uint8_t(kvarn_domain::original); }
        if (mismatch == 1) { incompatible.k_bits = 3; }
        if (mismatch == 2) { incompatible.tail_type = sx_bf16; }
        if (mismatch == 3) { ++incompatible.k_stride; }
        if (mismatch == 4) { incompatible.head_slices = 1; }
        if (mismatch == 5) { ++incompatible.payload_bytes; }
        refuses(valid_frame, 256, model, rope, {incompatible});
    }
    // A validated source cannot be projected after acquiring a commit ticket.
    auto validated = snapshot::parse(valid_frame.data(), valid_frame.size());
    expected_snapshot split_expected;
    split_expected.capacity_cells = split.c; split_expected.metadata_version = 3; split_expected.split = split;
    for (const auto & d : validated.layers()) { split_expected.layers.push_back(expected_layer::from_desc(d)); }
    validated.validate_expected_snapshot(split_expected);
    CHECK_THROWS(validated.project_split_to_pure_for_import(256, model, rope, {pure}));
    // A staged failure after successful projection still cannot publish state.
    auto failed_commit = snapshot::parse(valid_frame.data(), valid_frame.size());
    failed_commit.project_split_to_pure_for_import(256, model, rope, {pure});
    failed_commit.validate_expected_snapshot(expected(pure, 256));
    int published = 0, aborted = 0;
    commit_handlers failure;
    failure.stage_metadata = [](uint32_t, const uint8_t *, size_t) { return true; };
    failure.stage_layer = [](uint32_t, uint32_t, const layer_desc &, const uint8_t *, size_t) { return false; };
    failure.finish = [&] { ++published; };
    failure.abort = [&] { ++aborted; };
    CHECK_THROWS(failed_commit.commit(failure));
    CHECK(published == 0 && aborted == 1);
    // Projection keeps borrowed-payload integrity checking live: changing the
    // original local bytes after validation must fail before staging begins.
    auto mutable_backing = std::make_shared<std::vector<uint8_t>>(valid_frame);
    auto changed = snapshot::parse_view(mutable_backing->data(), mutable_backing->size(), mutable_backing);
    changed.project_split_to_pure_for_import(256, model, rope, {pure});
    changed.validate_expected_snapshot(expected(pure, 256));
    auto * byte = const_cast<uint8_t *>(changed.payload(0));
    byte[0] ^= 1;
    int staged_after_mutation = 0;
    commit_handlers mutated;
    mutated.stage_metadata = [&](uint32_t, const uint8_t *, size_t) { ++staged_after_mutation; return true; };
    mutated.stage_layer = [&](uint32_t, uint32_t, const layer_desc &, const uint8_t *, size_t) { ++staged_after_mutation; return true; };
    mutated.finish = [&] { ++published; };
    mutated.abort = [&] { ++aborted; };
    CHECK_THROWS(changed.commit(mutated));
    CHECK(staged_after_mutation == 0 && published == 0);
    auto corrupt = valid_frame;
    corrupt[56] ^= 1; // authenticated split-P header field
    CHECK_THROWS(snapshot::parse(corrupt.data(), corrupt.size()));
    CHECK(XXH64(valid_frame.data(), valid_frame.size(), 0) == original_hash);
}

static void test_compact_qx_rows(uint16_t type = sx_q4_0) {
    // The same strict mapper is used by the native cache index setters and
    // scatter writer/reader: metadata IDs remain global, payload IDs do not.
    CHECK(qx_payload_row_for_cell(102400, 131072, 102400) == 0);
    CHECK(qx_payload_row_for_cell(102400, 131072, 131071) == 28671);
    CHECK(qx_payload_row_for_cell(0, 1024, 512) == 512);
    CHECK_THROWS(qx_payload_row_for_cell(102400, 131072, 102399));
    CHECK_THROWS(qx_payload_row_for_cell(102400, 131072, 131072));
    CHECK_THROWS(qx_payload_row_for_cell(131072, 131072, 131072));
    CHECK_THROWS(qx_payload_row_for_cell(131073, 131072, 131073));

    serialize_options options;
    options.split = make_split();
    std::vector<cell_entry> cells(SPLIT_P + 3);
    cells[SPLIT_P-1].pos = SPLIT_P-1; cells[SPLIT_P-1].seq_ids = {0};
    cells[SPLIT_P].pos = SPLIT_P; cells[SPLIT_P].seq_ids = {0};
    cells[SPLIT_P+2].pos = SPLIT_P+2; cells[SPLIT_P+2].seq_ids = {0};
    const uint64_t row_stride = type == sx_f16 ? 2048 : 576;
    std::vector<uint8_t> local_bytes(128), overflow_bytes(4*row_stride), metadata;
    fill_pattern(local_bytes, 91); fill_pattern(overflow_bytes, 92);
    make_metadata(metadata, 93);
    auto local = make_kvarn_layer(1, kv4v4, 4, 4, uint8_t(kvarn_domain::rotated),
            uint32_t(cells.size()), 3, local_bytes.size());
    local.range_start = 0; local.range_end = SPLIT_P;
    auto overflow = make_qx_layer(1, type, type, 2, overflow_bytes.size());
    overflow.layout = layout_standard_qx_compact_rows;
    overflow.owner = owner_target; overflow.v_trans = 0;
    overflow.head_dim_v = 256; overflow.v_rot_width = 256;
    overflow.k_stride = row_stride; overflow.v_stride = row_stride;
    if (type == sx_f16) {
        overflow.k_rot = overflow.v_rot = uint8_t(rotation::none);
        overflow.k_rot_width = overflow.v_rot_width = 0;
    }
    overflow.range_id = 1; overflow.range_start = SPLIT_P; overflow.range_end = SPLIT_C;
    const auto encode = [&](const layer_desc & q, const std::vector<cell_entry> & table,
                            std::vector<uint8_t> & output) {
        auto k = local;
        k.payload_cells = uint32_t(table.size());
        k.payload_rows = uint64_t(std::count_if(table.begin(), table.end(), [](const cell_entry & cell) {
            return cell.pos >= 0;
        }));
        return snapshot::serialize({{k, local_bytes.data(), local_bytes.size()},
                {q, overflow_bytes.data(), size_t(q.payload_bytes)}}, table,
                1, metadata.data(), metadata.size(), snapshot_flags::full, 0, 0, output, {}, options);
    };
    std::vector<uint8_t> compact;
    CHECK(encode(overflow, cells, compact));
    const auto original = compact;
    expected_snapshot expected;
    expected.split = *options.split; expected.capacity_cells = SPLIT_C; expected.metadata_version = 1;
    const auto parsed = snapshot::parse(compact.data(), compact.size());
    for (const auto & d : parsed.layers()) { expected.layers.push_back(expected_layer::from_desc(d)); }
    CHECK(parsed.layers()[1].layout == layout_standard_qx_compact_rows);
    CHECK(parsed.layers()[1].payload_cells == 2 && parsed.layers()[1].payload_rows == 2);
    CHECK(parsed.cells().size() == SPLIT_P+3 && parsed.cells()[SPLIT_P+1].pos == -1);
    for (bool borrowed : {false, true}) {
        auto backing = std::make_shared<std::vector<uint8_t>>(compact);
        auto state = borrowed ? snapshot::parse_view(backing->data(), backing->size(), backing)
                              : snapshot::parse(backing->data(), backing->size());
        state.validate_expected_snapshot(expected);
        std::vector<uint8_t> live{7, 8}, staged;
        int stages = 0, published = 0;
        commit_handlers handlers;
        handlers.stage_metadata = [](uint32_t, const uint8_t *, size_t) { return true; };
        handlers.stage_split = [](const position_split &) { return true; };
        handlers.stage_layer = [&](uint32_t, uint32_t range, const layer_desc & d, const uint8_t * bytes, size_t size) {
            CHECK(live == std::vector<uint8_t>({7, 8})); ++stages;
            if (range == 1) {
                CHECK(d.layout == layout_standard_qx_compact_rows && size == overflow_bytes.size());
                staged.assign(bytes, bytes + size);
            }
            return true;
        };
        handlers.finish = [&] { live = staged; ++published; };
        handlers.abort = [&] { staged.clear(); };
        state.commit(handlers);
        CHECK(stages == 2 && published == 1 && live == overflow_bytes);
        CHECK(*backing == original);
    }

    if (type == sx_f16) {
        // Format identity is authoritative even with equal padded row sizes:
        // a Q4 destination must never consume a native F16 snapshot.
        auto mismatch = expected;
        mismatch.layers[1].k_type = mismatch.layers[1].v_type = sx_q4_0;
        auto state = snapshot::parse(compact.data(), compact.size());
        CHECK_THROWS(state.validate_expected_snapshot(mismatch));
        int mutations = 0;
        commit_handlers handlers;
        handlers.stage_metadata = [&](uint32_t, const uint8_t *, size_t) { ++mutations; return true; };
        handlers.stage_layer = [&](uint32_t, uint32_t, const layer_desc &, const uint8_t *, size_t) { ++mutations; return true; };
        handlers.stage_split = [&](const position_split &) { ++mutations; return true; };
        handlers.finish = [&] { ++mutations; }; handlers.abort = [] {};
        CHECK_THROWS(state.commit(handlers));
        CHECK(mutations == 0);
    }

    // Legacy absolute-row and compact-row frames are distinct contracts even
    // when their row bytes are identical. Refusal must not stage or publish.
    for (bool legacy_source : {false, true}) {
        auto source = compact;
        auto destination = expected;
        if (legacy_source) {
            auto q = overflow; q.layout = layout_standard_qx_rows;
            CHECK(encode(q, cells, source));
        } else {
            destination.layers[1].layout = layout_standard_qx_rows;
        }
        auto state = snapshot::parse(source.data(), source.size());
        int mutations = 0;
        commit_handlers handlers;
        handlers.stage_metadata = [&](uint32_t, const uint8_t *, size_t) { ++mutations; return true; };
        handlers.stage_layer = [&](uint32_t, uint32_t, const layer_desc &, const uint8_t *, size_t) { ++mutations; return true; };
        handlers.stage_split = [&](const position_split &) { ++mutations; return true; };
        handlers.finish = [&] { ++mutations; };
        handlers.abort = [] {};
        CHECK_THROWS(state.validate_expected_snapshot(destination));
        CHECK_THROWS(state.commit(handlers));
        CHECK(mutations == 0);
    }
    const auto refuses = [&](layer_desc q, const std::vector<cell_entry> & table) {
        std::vector<uint8_t> destination{4, 5, 6};
        CHECK_THROWS(encode(q, table, destination));
        CHECK(destination == std::vector<uint8_t>({4, 5, 6}));
    };
    auto bad = overflow;
    bad.payload_cells = SPLIT_C-SPLIT_P+1; bad.payload_rows = bad.payload_cells;
    refuses(bad, cells);
    bad = overflow; bad.payload_rows = 1; refuses(bad, cells);
    bad = overflow; bad.payload_cells = 1; bad.payload_rows = 1; refuses(bad, cells);
    bad = overflow; bad.v_trans = 1; refuses(bad, cells);
    bad = overflow; bad.n_stream = 2; refuses(bad, cells);
    auto bad_cells = cells; bad_cells[SPLIT_P].pos = SPLIT_P-1; refuses(overflow, bad_cells);
    bad_cells = cells; bad_cells[SPLIT_P-1].pos = SPLIT_P; refuses(overflow, bad_cells);
    bad_cells = cells; bad_cells[SPLIT_P].pos = SPLIT_C; refuses(overflow, bad_cells);
    bad_cells = cells; bad_cells[SPLIT_P].seq_ids.clear(); refuses(overflow, bad_cells);
    bad_cells = cells; bad_cells.resize(SPLIT_C+1); refuses(overflow, bad_cells);

    // Recompute the whole-file checksum to exercise parser geometry checks,
    // rather than rejecting these mutations only as generic corrupt bytes.
    const size_t overflow_desc = 104 + metadata.size() + 113 + local_bytes.size();
    const auto rehash = [](std::vector<uint8_t> & frame) {
        write_le_u64_at(frame, frame.size()-8, XXH64(frame.data(), frame.size()-8, 0));
    };
    auto malformed = compact;
    write_le_u32_at(malformed, overflow_desc+73, SPLIT_C-SPLIT_P+1);
    write_le_u64_at(malformed, overflow_desc+77, SPLIT_C-SPLIT_P+1);
    rehash(malformed);
    CHECK_THROWS(snapshot::parse(malformed.data(), malformed.size()));
    malformed = compact;
    write_le_u32_at(malformed, overflow_desc+73, 1);
    write_le_u64_at(malformed, overflow_desc+77, 1);
    rehash(malformed);
    CHECK_THROWS(snapshot::parse(malformed.data(), malformed.size()));

    // Compact wire IDs never belong to v1/v3 layer placement frames.
    bad = overflow; bad.range_id = 0; bad.range_start = 0; bad.range_end = 0;
    std::vector<uint8_t> v1;
    CHECK_THROWS(snapshot::serialize({{bad, overflow_bytes.data(), overflow_bytes.size()}}, {},
            1, metadata.data(), metadata.size(), snapshot_flags::full, 0, 0, v1));

    // An empty compact overflow still travels as an explicit descriptor.
    auto empty = overflow; empty.payload_cells = 0; empty.payload_rows = 0; empty.payload_bytes = 0;
    std::vector<cell_entry> below(2);
    below[0].pos = 0; below[0].seq_ids = {0}; below[1].pos = 1; below[1].seq_ids = {0};
    std::vector<uint8_t> empty_frame;
    CHECK(encode(empty, below, empty_frame));
    auto empty_state = snapshot::parse(empty_frame.data(), empty_frame.size());
    CHECK(empty_state.layers().size() == 2 && empty_state.payload_size(1) == 0);
    auto pure = local;
    pure.range_start = 0; pure.range_end = 0;
    pure.payload_cells = 2; pure.payload_rows = 2;
    empty_state.project_split_to_pure_for_import(SPLIT_P, options.split->model_id, options.split->rope_id, {pure});
    CHECK(!empty_state.has_split() && empty_state.layers().size() == 1);

    source_identity identity;
    identity.model_id = options.split->model_id;
    identity.rope_id = options.split->rope_id;
    identity.config_id = descriptor_config_id({pure});
    serialize_options pure_options; pure_options.source = identity;
    std::vector<uint8_t> pure_frame;
    CHECK(snapshot::serialize({{pure, local_bytes.data(), local_bytes.size()}}, below,
            1, metadata.data(), metadata.size(), snapshot_flags::full, 0, 0, pure_frame, {}, pure_options));
    auto destination_local = pure; destination_local.range_end = SPLIT_P;
    auto imported = snapshot::parse(pure_frame.data(), pure_frame.size());
    imported.adopt_split_for_import(*options.split, {destination_local, empty}, identity);
    CHECK(imported.has_split() && imported.layers().size() == 2);
    CHECK(imported.layers()[1].layout == layout_standard_qx_compact_rows);
    CHECK(imported.layers()[1].payload_cells == 0 && imported.layers()[1].payload_rows == 0);
    auto refused_import = snapshot::parse(pure_frame.data(), pure_frame.size());
    const auto before_import = normalized_state_fingerprint(refused_import, nullptr, 0);
    auto nonempty_target = empty; nonempty_target.payload_cells = 1; nonempty_target.payload_rows = 1;
    CHECK_THROWS(refused_import.adopt_split_for_import(*options.split,
            {destination_local, nonempty_target}, identity));
    CHECK(normalized_state_fingerprint(refused_import, nullptr, 0) == before_import);
    CHECK(compact == original);
}

int main() {
    test_roundtrip_mixed();
    test_empty_prefix();
    test_validate_strict();
    test_owner_values_are_exact();
    test_partial_owner_contract();
    test_body_only();
    test_transactional_commit();
    test_owner_fallback_matrix();
    test_pure_import_adapter();
    test_split_to_pure_import_adapter();
    test_compact_qx_rows();
    test_compact_qx_rows(sx_f16);
    test_fingerprint_controls();
    test_spool_preflight();
    test_kms3_move_identity();
    test_duplicate_and_bad_inputs();
    test_truncation_and_overflow();
    test_caller_caps();
    test_checksum_corruption();
    test_fault_callback_and_no_commit();
    test_borrowed_view();
    test_move_semantics();
    test_commit_rechecks_borrowed_payload();
    test_trailing_garbage();
    test_v2_split_roundtrip();
    test_v2_empty_range_keeps_descriptor();
    test_v2_rejects_bad_ranges();
    test_v2_validate_requires_matching_envelope();
    test_v2_corruption_and_version_fails_closed();
    test_v2_partial_and_body_only();
    test_v2_xxl_import_shape();

    std::printf("%s: %d checks, %d failures\n", g_failures == 0 ? "PASS" : "FAIL", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
