// Standalone deterministic CPU tests for the mixed KV snapshot codec.
// Build without CMake:
//   g++ -std=c++17 -O2 -Wall -Wextra -I src
//     tests/test-kv-mixed-state.cpp src/llama-kv-mixed-state.cpp
//     -o /tmp/opencode/test-kv-mixed-state

#include "llama-kv-mixed-state.h"

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
        [&](uint32_t id, const layer_desc &, const uint8_t * p, size_t n) {
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
    s.commit({ [&](uint32_t, const layer_desc &, const uint8_t *, size_t) { ++staged; return true; },
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
        CHECK_THROWS(s.commit({ [&](uint32_t, const layer_desc &, const uint8_t *, size_t) { ++staged; return true; },
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
        s.commit({ [&](uint32_t, const layer_desc &, const uint8_t *, size_t) { ++staged; return true; },
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
        CHECK_THROWS(s.commit({ [&](uint32_t, const layer_desc &, const uint8_t *, size_t) { ++staged; return true; },
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
        [&](uint32_t id, const layer_desc &, const uint8_t * ptr, size_t n) {
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
    s.commit({ [&](uint32_t, const layer_desc &, const uint8_t *, size_t) { ++staged; return true; },
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
            [&](uint32_t id, const layer_desc &, const uint8_t *, size_t) {
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
            [&](uint32_t, const layer_desc &, const uint8_t *, size_t) { return true; },
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
                [&](uint32_t, const layer_desc &, const uint8_t *, size_t) -> bool {
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
            [&](uint32_t, const layer_desc &, const uint8_t *, size_t) { ++staged; return true; },
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
        auto stage = [&](uint32_t, const layer_desc &, const uint8_t *, size_t) { ++ran; return true; };
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
    CHECK_THROWS(s.commit({ [&](uint32_t, const layer_desc &, const uint8_t *, size_t) { ++publishes; return true; },
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
        [&](uint32_t, const layer_desc &, const uint8_t * ptr, size_t n) {
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
            [&](uint32_t, const layer_desc &, const uint8_t *, size_t) { ++staged; return true; },
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
        [&](uint32_t, const layer_desc &, const uint8_t *, size_t) { ++staged; return true; },
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

int main() {
    test_roundtrip_mixed();
    test_empty_prefix();
    test_validate_strict();
    test_owner_values_are_exact();
    test_partial_owner_contract();
    test_body_only();
    test_transactional_commit();
    test_duplicate_and_bad_inputs();
    test_truncation_and_overflow();
    test_caller_caps();
    test_checksum_corruption();
    test_fault_callback_and_no_commit();
    test_borrowed_view();
    test_move_semantics();
    test_commit_rechecks_borrowed_payload();
    test_trailing_garbage();

    std::printf("%s: %d checks, %d failures\n", g_failures == 0 ? "PASS" : "FAIL", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}