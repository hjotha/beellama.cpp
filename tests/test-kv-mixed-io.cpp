// Standalone CPU tests for the mixed-io bridge (split_writer / split_reader)
// against real GGML CPU tensors and the immutable baseline libraries.
// Build (recorded in the artifact report):
//   g++ -std=c++17 -O2 -Wall -Wextra -I src -I ggml/include
//     tests/test-kv-mixed-io.cpp src/llama-kv-mixed-io.cpp
//     src/llama-kv-mixed-state.cpp src/llama-kv-mixed-state-stream.cpp
//     -L /home/hjotha/beellama-mixed-kv-20261001-130910/baseline-bin \
//     -lggml -lggml-cpu \
//     -Wl,-rpath,/home/hjotha/beellama-mixed-kv-20261001-130910/baseline-bin \
//     -Wl,--wrap=ggml_backend_tensor_get -Wl,--wrap=ggml_backend_tensor_set \
//     -Wl,--wrap=ggml_backend_tensor_memset
//     -o /home/hjotha/beellama-mixed-kv-20261001-130910/opencode/snapshots/test-kv-mixed-io

#include "llama-kv-mixed-io.h"
#include "llama-kv-mixed-state-stream.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

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

// ---- ggml call interposition (counts only calls from THIS binary) --------

extern "C" {
int g_wrap_get = 0;
int g_wrap_set = 0;
int g_wrap_memset = 0;

void __real_ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
void __real_ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void __real_ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size);

void __wrap_ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ++g_wrap_get;
    __real_ggml_backend_tensor_get(tensor, data, offset, size);
}
void __wrap_ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ++g_wrap_set;
    __real_ggml_backend_tensor_set(tensor, data, offset, size);
}
void __wrap_ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ++g_wrap_memset;
    __real_ggml_backend_tensor_memset(tensor, value, offset, size);
}
}

static void fill_pattern(std::vector<uint8_t> & v, uint32_t seed) {
    uint32_t x = seed * 2654435761u + 12345u;
    for (size_t i = 0; i < v.size(); ++i) {
        x = x * 1664525u + 1013904223u;
        v[i] = uint8_t(x >> 24);
    }
}

// ---- real CPU ggml fixture --------------------------------------------------

struct tensors {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * t0 = nullptr; // F32 ne{100,8} = 3200 B
    ggml_tensor * t1 = nullptr; // F16 ne{64,16} = 2048 B
    std::vector<uint8_t> d0;
    std::vector<uint8_t> d1;
    ~tensors() {
        if (buf) { ggml_backend_buffer_free(buf); }
        if (backend) { ggml_backend_free(backend); }
        if (ctx) { ggml_free(ctx); }
    }
};

static tensors make_tensors() {
    tensors ts;
    ggml_init_params params{};
    params.mem_size = 1u << 20;
    params.no_alloc = true;
    ts.ctx = ggml_init(params);
    ts.t0 = ggml_new_tensor_2d(ts.ctx, GGML_TYPE_F32, 100, 8);
    ts.t1 = ggml_new_tensor_2d(ts.ctx, GGML_TYPE_F16, 64, 16);
    ts.backend = ggml_backend_cpu_init();
    ts.buf = ggml_backend_alloc_ctx_tensors(ts.ctx, ts.backend);
    ts.d0.resize(ggml_nbytes(ts.t0));
    ts.d1.resize(ggml_nbytes(ts.t1));
    fill_pattern(ts.d0, 11);
    fill_pattern(ts.d1, 12);
    ggml_backend_tensor_set(ts.t0, ts.d0.data(), 0, ts.d0.size());
    ggml_backend_tensor_set(ts.t1, ts.d1.data(), 0, ts.d1.size());
    g_wrap_get = 0;
    g_wrap_set = 0;
    g_wrap_memset = 0;
    return ts;
}

// one context with multiple destination tensors (produced from prototypes)
struct dests {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> t;
    ~dests() {
        if (buf) { ggml_backend_buffer_free(buf); }
        if (backend) { ggml_backend_free(backend); }
        if (ctx) { ggml_free(ctx); }
    }
};

static dests make_dests(const tensors &, std::initializer_list<ggml_tensor *> protos) {
    dests out;
    ggml_init_params params{};
    params.mem_size = 1u << 21;
    params.no_alloc = true;
    out.ctx = ggml_init(params);
    for (ggml_tensor * proto : protos) {
        out.t.push_back(ggml_new_tensor_2d(out.ctx, proto->type, proto->ne[0], proto->ne[1]));
    }
    out.backend = ggml_backend_cpu_init();
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, out.backend);
    g_wrap_set = 0;
    return out;
}

// independent layer payload reference built from the source byte buffers
static std::vector<uint8_t> ref_payload(const mixed_manifest & m, const tensors & ts, uint32_t layer) {
    std::vector<uint8_t> out;
    for (const mixed_manifest_event & e : m.events) {
        if (e.kind == 1 && e.layer_id == layer) {
            const std::vector<uint8_t> & src = layer == 0 ? ts.d0 : ts.d1;
            out.insert(out.end(), src.begin() + long(e.tensor_offset),
                       src.begin() + long(e.tensor_offset) + long(e.size));
        }
    }
    return out;
}

// mock outer llama_io_read_i (file-local classes cannot be instantiated here)
class mock_outer final : public llama_io_read_i {
public:
    std::vector<std::function<void()>> cbs;
    ~mock_outer() { cancel(); }
    void read(void *, size_t) override { throw std::runtime_error("mock read"); }
    void read_tensor(ggml_tensor *, size_t, size_t) override { throw std::runtime_error("mock read_tensor"); }
    void stage_tensor_set(ggml_tensor *, const void *, size_t, size_t) override { throw std::runtime_error("mock stage"); }
    void stage_tensor_clear(ggml_tensor *, size_t, size_t) override { throw std::runtime_error("mock clear"); }
    void on_commit(std::function<void()> cb) override { cbs.push_back(std::move(cb)); }
    void commit() override {
        for (auto & cb : cbs) { cb(); }
        cbs.clear();
    }
    void cancel() override { cbs.clear(); }
    size_t n_bytes() override { return 0; }
};

// ---- tests ------------------------------------------------------------------

static void test_split_writer_records_without_reads() {
    tensors ts = make_tensors();
    split_writer w([](const ggml_tensor * t) { return t->ne[1] == 8 ? 0u : 1u; });
    const char a[] = "A";
    const char bc[] = "BC";
    const char d[] = "D";
    w.write(a, 1);
    w.write_tensor(ts.t0, 64, 320);
    w.write(bc, 2);
    w.write_tensor(ts.t1, 0, 2048);
    w.write(d, 1);

    CHECK(g_wrap_get == 0); // no tensor reads during recording
    CHECK(g_wrap_set == 0);
    CHECK(w.n_bytes() == 2372);
    CHECK(w.virtual_stream_bytes() == 2372);
    const mixed_manifest & m = w.manifest();
    // A | t0(320) | BC | t1(2048) | D  -- "D" follows a tensor event so it
    // cannot combine with "BC"
    CHECK(m.events.size() == 5);
    CHECK(m.events[0].kind == 0 && m.events[0].size == 1 && m.events[0].stream_offset == 0);
    CHECK(m.events[1].kind == 1 && m.events[1].layer_id == 0 && m.events[1].size == 320);
    CHECK(m.events[1].stream_offset == 1 && m.events[1].tensor_offset == 64);
    CHECK(m.events[2].kind == 0 && m.events[2].size == 2 && m.events[2].stream_offset == 321);
    CHECK(m.events[3].kind == 1 && m.events[3].layer_id == 1 && m.events[3].size == 2048);
    CHECK(m.events[4].kind == 0 && m.events[4].size == 1 && m.events[4].stream_offset == 2371);
    CHECK(m.metadata.size() == 4 && std::memcmp(m.metadata.data(), "ABCD", 4) == 0);
    CHECK(m.layer_payload_sizes.size() == 2);
    CHECK(m.layer_payload_sizes[0].first == 0 && m.layer_payload_sizes[0].second == 320);
    CHECK(m.layer_payload_sizes[1].first == 1 && m.layer_payload_sizes[1].second == 2048);
    CHECK(m.total_stream_bytes == 2372);
    CHECK(m.validate().empty());

    // adjacent ordinary writes combine into one metadata run
    split_writer w2([](const ggml_tensor *) { return 0u; });
    w2.write("XY", 2);
    w2.write("Z", 1);
    CHECK(w2.manifest().events.size() == 1);
    CHECK(w2.manifest().events[0].size == 3);
    CHECK(w2.manifest().metadata.size() == 3);
}

static void test_split_writer_bounds() {
    tensors ts = make_tensors();
    split_writer w([](const ggml_tensor *) { return 0u; });
    const uint8_t b[4] = { 1, 2, 3, 4 };
    CHECK_THROWS(w.write_tensor(nullptr, 0, 1));
    CHECK_THROWS(w.write_tensor(ts.t0, 3200, 1)); // beyond nbytes
    CHECK_THROWS(w.write_tensor(ts.t0, 0, 3201)); // beyond nbytes
    w.write_tensor(ts.t0, 0, 3200);               // exact fit ok

    split_writer_limits lim;
    lim.max_metadata_bytes = 4;
    split_writer wm([](const ggml_tensor *) { return 0u; }, lim);
    wm.write(b, 4);
    CHECK_THROWS(wm.write(b, 1)); // metadata budget

    split_writer_limits le;
    le.max_events = 2;
    split_writer we([](const ggml_tensor *) { return 0u; }, le);
    we.write(b, 1);
    we.write_tensor(ts.t0, 0, 16);
    CHECK_THROWS(we.write(b, 1)); // event budget

    split_writer_limits lp;
    lp.max_payload_bytes = 100;
    split_writer wp([](const ggml_tensor *) { return 0u; }, lp);
    wp.write_tensor(ts.t0, 0, 100);
    CHECK_THROWS(wp.write_tensor(ts.t0, 0, 1));    // per-layer cap
    CHECK_THROWS(wp.write_tensor(ts.t0, 101, 1));  // layer total overflow check
    CHECK(g_wrap_get == 0);
}

static void test_manifest_wire_roundtrip() {
    tensors ts = make_tensors();
    split_writer w([](const ggml_tensor * t) { return t->ne[1] == 8 ? 0u : 1u; });
    w.write("A", 1);
    w.write_tensor(ts.t0, 64, 320);
    w.write("BC", 2);
    w.write_tensor(ts.t1, 0, 2048);
    std::vector<uint8_t> wire = mixed_manifest_serialize(w.manifest());
    CHECK(!wire.empty());
    mixed_manifest parsed;
    std::string err;
    CHECK(mixed_manifest_parse(wire.data(), wire.size(), parsed, &err));
    CHECK(parsed.events.size() == w.manifest().events.size());
    CHECK(parsed.metadata == w.manifest().metadata);
    CHECK(parsed.total_stream_bytes == w.manifest().total_stream_bytes);
    for (size_t i = 0; i < parsed.events.size(); ++i) {
        const mixed_manifest_event & x = parsed.events[i];
        const mixed_manifest_event & y = w.manifest().events[i];
        CHECK(x.kind == y.kind && x.layer_id == y.layer_id &&
              x.payload_offset == y.payload_offset && x.size == y.size &&
              x.stream_offset == y.stream_offset && x.tensor_offset == y.tensor_offset);
    }
    CHECK(parsed.layer_payload_sizes == w.manifest().layer_payload_sizes);

    // strict parse failures
    std::vector<uint8_t> bad = wire;
    bad[4] = 2; // version
    CHECK(!mixed_manifest_parse(bad.data(), bad.size(), parsed, &err));
    bad = wire;
    bad[0] ^= 0xFF; // magic
    CHECK(!mixed_manifest_parse(bad.data(), bad.size(), parsed, &err));
    CHECK(!mixed_manifest_parse(wire.data(), wire.size() - 1, parsed, &err)); // truncated
    CHECK(!mixed_manifest_parse(wire.data(), 0, parsed, &err));
    bad = wire;
    bad.push_back(0); // trailing garbage
    CHECK(!mixed_manifest_parse(bad.data(), bad.size(), parsed, &err));
}

static void test_read_payload_slices() {
    tensors ts = make_tensors();
    split_writer w([](const ggml_tensor * t) { return t->ne[1] == 8 ? 0u : 1u; });
    w.write("TAG", 3);
    w.write_tensor(ts.t0, 64, 320);   // layer 0 event 1
    w.write_tensor(ts.t1, 0, 2048);   // layer 1
    w.write_tensor(ts.t0, 384, 128);  // layer 0 event 2 (contiguous layer payload)
    const std::vector<uint8_t> ref0 = ref_payload(w.manifest(), ts, 0);
    const std::vector<uint8_t> ref1 = ref_payload(w.manifest(), ts, 1);
    CHECK(ref0.size() == 448 && ref1.size() == 2048);

    // arbitrary slices with diverse boundaries
    const std::vector<std::pair<uint64_t, size_t>> slices = {
        { 0, 1 }, { 1, 7 }, { 100, 64 }, { 319, 1 }, { 320, 1 }, { 320, 128 },
        { 319, 2 }, { 447, 1 }, { 0, 448 }, { 63, 200 },
    };
    for (const auto & sl : slices) {
        std::vector<uint8_t> dst(sl.second);
        const size_t got = w.read_payload(0, sl.first, dst.data(), sl.second);
        CHECK(got == sl.second);
        CHECK(std::memcmp(dst.data(), ref0.data() + sl.first, sl.second) == 0);
    }
    // out-of-range: short/zero returns
    std::vector<uint8_t> dst(16);
    CHECK(w.read_payload(0, 448, dst.data(), 16) == 0);
    CHECK(w.read_payload(0, 1000, dst.data(), 16) == 0);
    CHECK(w.read_payload(9, 0, dst.data(), 16) == 0);
    CHECK(w.read_payload(1, 2040, dst.data(), 16) == 8); // clipped at layer end
    CHECK(std::memcmp(dst.data(), ref1.data() + 2040, 8) == 0);
    // cross-event read
    std::vector<uint8_t> xdst(50);
    CHECK(w.read_payload(0, 300, xdst.data(), 50) == 50);
    CHECK(std::memcmp(xdst.data(), ref0.data() + 300, 50) == 0);
}

static void test_split_reader_reverse_and_outer_commit() {
    tensors ts = make_tensors();
    split_writer w([](const ggml_tensor * t) { return t->ne[1] == 8 ? 0u : 1u; });
    w.write("A", 1);
    w.write_tensor(ts.t0, 64, 320);
    w.write("BC", 2);
    w.write_tensor(ts.t1, 0, 2048);
    w.write("D", 1);

    // spans: RAM copies with retained owners
    const std::vector<uint8_t> ref0 = ref_payload(w.manifest(), ts, 0);
    const std::vector<uint8_t> ref1 = ref_payload(w.manifest(), ts, 1);
    auto owner0 = std::make_shared<std::vector<uint8_t>>(ref0);
    auto owner1 = std::make_shared<std::vector<uint8_t>>(ref1);
    std::weak_ptr<const void> weak0 = owner0;
    std::weak_ptr<const void> weak1 = owner1;
    split_reader r(w.manifest(), { { 0, owner0->data(), owner0->size(), owner0 },
                                   { 1, owner1->data(), owner1->size(), owner1 } });
    CHECK(r.remaining() == 2372);

    dests dd = make_dests(ts, { ts.t0, ts.t1 });
    ggml_tensor * d0 = dd.t[0];
    ggml_tensor * d1 = dd.t[1];

    // replay the legacy stream with DIFFERENT read boundaries than the events
    uint8_t tag[8];
    r.read(tag, 1); CHECK(tag[0] == 'A');
    r.read_tensor(d0, 0, 7);                 // small slice of the 320-byte event
    r.read_tensor(d0, 7, 64);
    r.read_tensor(d0, 71, 249);              // rest of the event
    r.read(tag, 2); CHECK(tag[0] == 'B' && tag[1] == 'C');
    r.read_tensor(d1, 0, 1000);
    r.read_tensor(d1, 1000, 1048);
    r.read(tag, 1); CHECK(tag[0] == 'D');
    CHECK(r.remaining() == 0);
    CHECK(r.n_bytes() == 2372);
    CHECK(g_wrap_set == 0); // no actual tensor write before the outer commit
    CHECK(!weak0.expired() && !weak1.expired()); // owners retained by the reader

    {
        mock_outer outer;
        r.defer_to(outer);
        // destructor after defer must NOT cancel the transferred state; the
        // reader goes out of scope here (deferred -> no-op)
        CHECK(!weak0.expired()); // owners moved into the outer callback
        outer.commit();          // publication happens only here, after the whole frame
        CHECK(g_wrap_set > 0);
        CHECK(g_wrap_memset == 0);
        std::vector<uint8_t> got0(320), got1(2048);
        ggml_backend_tensor_get(d0, got0.data(), 0, 320);
        CHECK(std::memcmp(got0.data(), ts.d0.data() + 64, 320) == 0);
        ggml_backend_tensor_get(d1, got1.data(), 0, 2048);
        CHECK(std::memcmp(got1.data(), ts.d1.data(), 2048) == 0);
        CHECK(std::memcmp(got1.data(), ref1.data(), 2048) == 0);
        CHECK(!weak0.expired()); // owners released only with the outer io
    }
    owner0.reset();
    owner1.reset(); // the test scope no longer holds the owners
    CHECK(weak0.expired());
    CHECK(weak1.expired());
}

static void test_read_across_events_and_tensor_splits() {
    tensors ts = make_tensors();
    split_writer w([](const ggml_tensor *) { return 0u; });
    w.write("XY", 2);
    w.write_tensor(ts.t0, 0, 100);
    w.write("Z", 1);
    w.write_tensor(ts.t0, 100, 200);
    const std::vector<uint8_t> ref = ref_payload(w.manifest(), ts, 0);
    CHECK(ref.size() == 300);

    auto owner = std::make_shared<std::vector<uint8_t>>(ref);
    std::weak_ptr<const void> weak = owner;
    split_reader r(w.manifest(), { { 0, owner->data(), owner->size(), owner } });

    // read() spanning metadata -> tensor event
    uint8_t buf[4];
    r.read(buf, 4);
    CHECK(std::memcmp(buf, "XY", 2) == 0);
    CHECK(std::memcmp(buf + 2, ref.data(), 2) == 0);

    // read_tensor() spanning tensor -> metadata -> tensor events:
    // stream: XY | t0[0,100) | Z | t0[100,300); cursor 4 -> t0[2,102) gives
    // 98 borrowed bytes, "Z" is staged, then t0[103,254) 151 borrowed
    dests dd = make_dests(ts, { ts.t0 });
    ggml_tensor * d0 = dd.t[0];
    r.read_tensor(d0, 0, 250);
    CHECK(r.remaining() == 49);
    r.read_tensor(d0, 250, 49); // consume the rest
    CHECK(r.remaining() == 0);

    mock_outer outer;
    r.defer_to(outer);
    outer.commit();
    CHECK(g_wrap_set == 4); // borrowed98 + staged1 + borrowed151 + borrowed49
    std::vector<uint8_t> got(300);
    ggml_backend_tensor_get(d0, got.data(), 0, 300);
    CHECK(std::memcmp(got.data(), ref.data() + 2, 98) == 0);
    CHECK(got[98] == 'Z');
    CHECK(std::memcmp(got.data() + 99, ref.data() + 100, 151) == 0);
    CHECK(std::memcmp(got.data() + 250, ref.data() + 251, 49) == 0);
}

static void test_fixup_caps_and_big_borrowed() {
    // >8 MiB borrowed payload with a TINY fixup cap succeeds; the TOTAL fixup
    // budget (default 64 MiB, configurable) is enforced across calls
    const uint64_t big = 10u << 20; // 10 MiB
    mixed_manifest m;
    mixed_manifest_event e;
    e.kind = 1;
    e.layer_id = 0;
    e.size = big;
    e.stream_offset = 0;
    m.events.push_back(e);
    m.layer_payload_sizes.emplace_back(0, big);
    m.total_stream_bytes = big;

    auto owner = std::make_shared<std::vector<uint8_t>>(size_t(big));
    fill_pattern(*owner, 21);
    split_reader_limits lim;
    lim.max_fixup_staging = 1u << 20; // 1 MiB total

    // destination tensor of 10 MiB
    struct local_dests {
        ggml_context * ctx = nullptr;
        ggml_backend_t be = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        ggml_tensor * d0 = nullptr;
        ~local_dests() {
            if (buf) { ggml_backend_buffer_free(buf); }
            if (be) { ggml_backend_free(be); }
            if (ctx) { ggml_free(ctx); }
        }
    } lds;
    ggml_init_params p2{};
    p2.mem_size = 1u << 20;
    p2.no_alloc = true;
    lds.ctx = ggml_init(p2);
    lds.d0 = ggml_new_tensor_2d(lds.ctx, GGML_TYPE_F32, 256, 10240);
    lds.be = ggml_backend_cpu_init();
    lds.buf = ggml_backend_alloc_ctx_tensors(lds.ctx, lds.be);
    g_wrap_set = 0;

    split_reader r(m, { { 0, owner->data(), owner->size(), owner } }, lim);
    r.read_tensor(lds.d0, 0, size_t(big)); // borrowed: no staging needed
    CHECK(r.remaining() == 0);
    std::vector<uint8_t> fix(64);
    fill_pattern(fix, 22);
    r.stage_tensor_set(lds.d0, fix.data(), 0, 64);
    // cap failure is TERMINAL: checked on a separate reader instance, and the
    // failed reader must not be reused (no partial retry accepted)
    {
        split_reader r2(m, { { 0, owner->data(), owner->size(), owner } }, lim);
        r2.stage_tensor_set(lds.d0, fix.data(), 0, 64);
        CHECK_THROWS(r2.stage_tensor_set(lds.d0, fix.data(), 64, (1u << 20) - 32)); // TOTAL 1 MiB + 64 B
        CHECK_THROWS(r2.read_tensor(lds.d0, 0, 1)); // terminal: no retry
    }
    mock_outer outer;
    r.defer_to(outer);
    outer.commit();
    CHECK(g_wrap_set == 2);
    std::vector<uint8_t> got(64);
    ggml_backend_tensor_get(lds.d0, got.data(), 0, 64);
    CHECK(std::memcmp(got.data(), fix.data(), 64) == 0);
}

static void test_defer_ordering_and_qx_append() {
    tensors ts = make_tensors();
    split_writer w([](const ggml_tensor * t) { return t->ne[1] == 8 ? 0u : 1u; });
    w.write_tensor(ts.t0, 0, 3200);
    w.write_tensor(ts.t1, 0, 2048);
    const std::vector<uint8_t> ref0 = ref_payload(w.manifest(), ts, 0);
    const std::vector<uint8_t> ref1 = ref_payload(w.manifest(), ts, 1);
    auto owner0 = std::make_shared<std::vector<uint8_t>>(ref0);
    auto owner1 = std::make_shared<std::vector<uint8_t>>(ref1);
    split_reader r(w.manifest(), { { 0, owner0->data(), owner0->size(), owner0 },
                                   { 1, owner1->data(), owner1->size(), owner1 } });

    dests dd = make_dests(ts, { ts.t0, ts.t1, ts.t1 });
    ggml_tensor * d0 = dd.t[0];
    ggml_tensor * d1 = dd.t[1];
    ggml_tensor * dqx = dd.t[2];

    bool callback_saw_updated = false;
    r.on_commit([&]() {
        // runs AFTER all tensor ops: the destination must already hold the data
        uint8_t b = 0;
        ggml_backend_tensor_get(d1, &b, 100, 1);
        callback_saw_updated = (b == ts.d1[100]);
    });
    r.read_tensor(d0, 0, 3200);
    r.read_tensor(d1, 0, 2048);
    // Qx companion append: borrowed write outside the virtual stream
    std::vector<uint8_t> qx(512);
    fill_pattern(qx, 31);
    auto qx_owner = std::make_shared<std::vector<uint8_t>>(qx);
    std::weak_ptr<const void> weak_qx = qx_owner;
    r.queue_borrowed(dqx, 0, qx_owner->data(), qx.size(), qx_owner);
    CHECK(r.remaining() == 0);

    {
        mock_outer outer;
        r.defer_to(outer);
        outer.commit();
        CHECK(callback_saw_updated); // publication callbacks ran AFTER all ops
        uint8_t got_qx[512];
        ggml_backend_tensor_get(dqx, got_qx, 0, 512);
        CHECK(std::memcmp(got_qx, qx.data(), 512) == 0); // Qx borrowed op applied
        CHECK(!weak_qx.expired());
    }
    qx_owner.reset(); // the test scope no longer holds the owner
    CHECK(weak_qx.expired()); // owners released with the outer io
}

static void test_outer_cancel_releases_everything() {
    tensors ts = make_tensors();
    split_writer w([](const ggml_tensor * t) { return t->ne[1] == 8 ? 0u : 1u; });
    w.write_tensor(ts.t0, 0, 3200);
    const std::vector<uint8_t> ref0 = ref_payload(w.manifest(), ts, 0);
    auto owner0 = std::make_shared<std::vector<uint8_t>>(ref0);
    std::weak_ptr<const void> weak = owner0;
    split_reader r(w.manifest(), { { 0, owner0->data(), owner0->size(), owner0 } });

    dests dd = make_dests(ts, { ts.t0 });
    r.read_tensor(dd.t[0], 0, 3200);
    CHECK(r.remaining() == 0);
    {
        mock_outer outer;
        r.defer_to(outer);
        outer.cancel(); // outer frame error -> cancel: no writes, owners released
        CHECK(g_wrap_set == 0);
        CHECK(g_wrap_memset == 0);
    } // reader destructor after defer must NOT cancel (already transferred)
    owner0.reset(); // the test scope no longer holds the owner
    CHECK(weak.expired());
}

static void test_borrowed_mutation_detected() {
    const uint64_t size = 100 * 1024;
    mixed_manifest m;
    mixed_manifest_event e;
    e.kind = 1;
    e.layer_id = 0;
    e.size = size;
    e.stream_offset = 0;
    m.events.push_back(e);
    m.layer_payload_sizes.emplace_back(0, size);
    m.total_stream_bytes = size;

    auto owner = std::make_shared<std::vector<uint8_t>>(size_t(size));
    fill_pattern(*owner, 41);
    split_reader_limits lim;
    lim.hash_borrowed = true; // opt-in: recheck ALL borrowed bytes before any write

    ggml_init_params params{};
    params.mem_size = 1u << 20;
    params.no_alloc = true;
    struct local_dests {
        ggml_context * ctx = nullptr;
        ggml_backend_t be = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        ggml_tensor * d0 = nullptr;
        ~local_dests() {
            if (buf) { ggml_backend_buffer_free(buf); }
            if (be) { ggml_backend_free(be); }
            if (ctx) { ggml_free(ctx); }
        }
    } lds;
    lds.ctx = ggml_init(params);
    lds.d0 = ggml_new_tensor_2d(lds.ctx, GGML_TYPE_F32, 256, 100);
    lds.be = ggml_backend_cpu_init();
    lds.buf = ggml_backend_alloc_ctx_tensors(lds.ctx, lds.be);
    g_wrap_set = 0;

    split_reader r(m, { { 0, owner->data(), owner->size(), owner } }, lim);
    r.read_tensor(lds.d0, 0, size_t(size));
    CHECK(r.remaining() == 0);
    owner->data()[12345] ^= 0xFF; // mutate the backing after queue

    mock_outer outer;
    r.defer_to(outer);
    bool threw = false;
    try {
        outer.commit();
    } catch (const kv_mixed_error &) {
        threw = true;
    }
    CHECK(threw);
    CHECK(g_wrap_set == 0); // detected BEFORE any tensor write
}

static void test_commit_defer_contract() {
    tensors ts = make_tensors();
    split_writer w([](const ggml_tensor * t) { return t->ne[1] == 8 ? 0u : 1u; });
    w.write_tensor(ts.t0, 0, 3200);
    const std::vector<uint8_t> ref0 = ref_payload(w.manifest(), ts, 0);
    auto owner0 = std::make_shared<std::vector<uint8_t>>(ref0);

    // direct commit rejected
    {
        split_reader r(w.manifest(), { { 0, owner0->data(), owner0->size(), owner0 } });
        CHECK_THROWS(r.commit());
    }
    // repeated defer rejected
    {
        split_reader r(w.manifest(), { { 0, owner0->data(), owner0->size(), owner0 } });
        dests dd = make_dests(ts, { ts.t0 });
        r.read_tensor(dd.t[0], 0, 3200);
        CHECK(r.remaining() == 0);
        mock_outer outer;
        r.defer_to(outer);
        CHECK_THROWS(r.defer_to(outer));
        CHECK(outer.cbs.size() == 1);
        outer.cancel();
        CHECK(g_wrap_set == 0);
    }
    // defer with unconsumed stream rejected
    {
        split_reader r(w.manifest(), { { 0, owner0->data(), owner0->size(), owner0 } });
        mock_outer outer;
        CHECK_THROWS(r.defer_to(outer));
    }
    // destructor before defer cancels: owners released, no writes
    {
        std::weak_ptr<const void> weak = owner0;
        dests dd = make_dests(ts, { ts.t0 });
        {
            split_reader r(w.manifest(), { { 0, owner0->data(), owner0->size(), owner0 } });
            r.read_tensor(dd.t[0], 0, 3200);
            CHECK(r.remaining() == 0);
            CHECK(!weak.expired());
        } // destructor without defer -> cancel
        CHECK(g_wrap_set == 0);
        owner0.reset(); // the test scope no longer holds the owner
        CHECK(weak.expired());
    }
}

static void test_reader_ctor_validation() {
    tensors ts = make_tensors();
    split_writer w([](const ggml_tensor * t) { return t->ne[1] == 8 ? 0u : 1u; });
    w.write_tensor(ts.t0, 0, 3200);
    w.write_tensor(ts.t1, 0, 2048);
    const std::vector<uint8_t> ref0 = ref_payload(w.manifest(), ts, 0);
    auto owner0 = std::make_shared<std::vector<uint8_t>>(ref0);

    // missing span for a declared layer
    CHECK_THROWS(split_reader(w.manifest(), { { 0, owner0->data(), owner0->size(), owner0 } }));
    // duplicate span for the same layer
    CHECK_THROWS(split_reader(w.manifest(), { { 0, owner0->data(), 3200, owner0 },
                                              { 0, owner0->data(), 3200, owner0 },
                                              { 1, owner0->data(), 2048, owner0 } }));
    // span size mismatch
    CHECK_THROWS(split_reader(w.manifest(), { { 0, owner0->data(), 3199, owner0 },
                                              { 1, owner0->data(), 2048, owner0 } }));
    // span for an undeclared layer
    CHECK_THROWS(split_reader(w.manifest(), { { 9, owner0->data(), 3200, owner0 },
                                              { 1, owner0->data(), 2048, owner0 } }));
    // null data with nonzero size
    CHECK_THROWS(split_reader(w.manifest(), { { 0, nullptr, 3200, owner0 },
                                              { 1, owner0->data(), 2048, owner0 } }));
    // span count cap
    {
        split_reader_limits lim;
        lim.max_payload_spans = 1;
        CHECK_THROWS(split_reader(w.manifest(), { { 0, owner0->data(), 3200, owner0 },
                                                  { 1, owner0->data(), 2048, owner0 } }, lim));
    }
    // per-span byte cap
    {
        split_reader_limits lim;
        lim.max_payload_bytes = 3000;
        CHECK_THROWS(split_reader(w.manifest(), { { 0, owner0->data(), 3200, owner0 },
                                                  { 1, owner0->data(), 2048, owner0 } }, lim));
    }
    // malformed manifests (structural validation)
    {
        mixed_manifest m = w.manifest();
        m.events[0].stream_offset = 1; // non-contiguous
        CHECK(!m.validate().empty());
        m = w.manifest();
        m.events[0].layer_id = 7; // undeclared layer in an event
        CHECK(!m.validate().empty());
        m = w.manifest();
        m.events[1].payload_offset = 100; // non-contiguous layer offsets
        CHECK(!m.validate().empty());
        m = w.manifest();
        m.layer_payload_sizes[0].second = 2000; // layer total mismatch
        CHECK(!m.validate().empty());
        m = w.manifest();
        m.total_stream_bytes += 1;
        CHECK(!m.validate().empty());
        m = w.manifest();
        m.events[0].kind = 9;
        CHECK(!m.validate().empty());
        m = w.manifest();
        m.version = 2;
        CHECK(!m.validate().empty());
    }
}

static void test_callback_failure_during_outer_commit() {
    tensors ts = make_tensors();
    split_writer w([](const ggml_tensor * t) { return t->ne[1] == 8 ? 0u : 1u; });
    w.write_tensor(ts.t0, 0, 3200);
    const std::vector<uint8_t> ref0 = ref_payload(w.manifest(), ts, 0);
    auto owner0 = std::make_shared<std::vector<uint8_t>>(ref0);
    split_reader r(w.manifest(), { { 0, owner0->data(), owner0->size(), owner0 } });
    dests dd = make_dests(ts, { ts.t0 });
    r.read_tensor(dd.t[0], 0, 3200);
    r.on_commit([]() -> void { throw std::runtime_error("publication callback failed"); });
    mock_outer outer;
    r.defer_to(outer);
    bool threw = false;
    try {
        outer.commit();
    } catch (const std::runtime_error &) {
        threw = true;
    }
    CHECK(threw);
    CHECK(g_wrap_set == 1); // tensor ops applied BEFORE publication callbacks
}

// ---- stream_serialized_size + split writer sizing --------------------------

static layer_desc make_qx_desc(uint32_t id, uint64_t bytes) {
    layer_desc d;
    d.layer_id = id;
    d.kind = cache_kind::standard_qx;
    d.k_type = sx_q4_0;
    d.v_type = sx_q4_0;
    d.layout = layout_standard_qx_rows;
    d.owner = owner_target;
    d.head_dim_k = 256;
    d.head_dim_v = 64;
    d.n_head_kv = 4;
    d.n_stream = 1;
    d.payload_mode = mode_full;
    d.payload_bytes = bytes;
    return d;
}

static void test_stream_serialized_size() {
    // small full fixture
    {
        std::vector<uint8_t> md(64);
        fill_pattern(md, 51);
        stream_input in;
        in.layers.push_back(make_qx_desc(1, 4096));
        in.layers.push_back(make_qx_desc(2, 2048));
        in.metadata_version = 1;
        in.metadata = md.data();
        in.metadata_size = md.size();
        in.flags = snapshot_flags::full;
        const uint64_t expected = 96 + 64 + 2 * 101 + 4096 + 2048 + 8;
        CHECK(stream_serialized_size(in) == expected);
        // independent of transfer buffer size
        for (size_t chunk : { size_t(1), size_t(64), size_t(1u << 20) }) {
            stream_options opts;
            opts.transfer_buffer_bytes = chunk;
            CHECK(stream_serialized_size(in, opts) == expected);
        }
        // equals the actual streamed output size
        std::vector<uint8_t> p0(4096), p1(2048);
        fill_pattern(p0, 52);
        fill_pattern(p1, 53);
        stream_read_callbacks reads;
        reads.read_payload = [&](uint32_t layer_id, uint64_t offset, uint8_t * dst, size_t count) {
            const std::vector<uint8_t> & p = layer_id == 1 ? p0 : p1;
            if (offset + count > p.size()) {
                return size_t(0);
            }
            std::memcpy(dst, p.data() + offset, count);
            return count;
        };
        uint64_t written = 0;
        stream_write_callbacks writes;
        writes.write_bytes = [&](const uint8_t *, size_t count) { written += count; return true; };
        CHECK(stream_serialize(in, reads, writes));
        CHECK(written == expected);
    }
    // partial and body_only fixtures
    {
        std::vector<uint8_t> md(32);
        fill_pattern(md, 54);
        stream_input p;
        layer_desc k = make_qx_desc(1, 1024);
        k.payload_mode = mode_partial_overlay;
        p.layers.push_back(k);
        layer_desc q = make_qx_desc(2, 0);
        q.payload_mode = mode_resident_reference;
        p.layers.push_back(q);
        p.metadata_version = 2;
        p.metadata = md.data();
        p.metadata_size = md.size();
        p.flags = snapshot_flags::partial;
        p.owner_epoch = 42;
        p.owner_id = 7;
        CHECK(stream_serialized_size(p) == 96 + 32 + 2 * 101 + 1024 + 8);

        stream_input b;
        b.layers.push_back(make_qx_desc(3, 512));
        b.flags = snapshot_flags::body_only;
        CHECK(stream_serialized_size(b) == 96 + 101 + 512 + 8);
    }
    // virtual 300 MiB: pure sizing, no callbacks, no payload reads
    {
        stream_input big;
        for (uint32_t id = 1; id <= 3; ++id) {
            big.layers.push_back(make_qx_desc(id, 100u << 20));
        }
        std::vector<uint8_t> md(64);
        fill_pattern(md, 55);
        big.metadata_version = 1;
        big.metadata = md.data();
        big.metadata_size = md.size();
        big.flags = snapshot_flags::full;
        CHECK(stream_serialized_size(big) == 96 + 64 + 3 * 101 + 3 * (100u << 20) + 8);
    }
    // same validation as stream_serialize: invalid inputs rejected
    {
        stream_input bad;
        bad.flags = snapshot_flags::partial | snapshot_flags::body_only;
        CHECK_THROWS(stream_serialized_size(bad));
        stream_input bad2;
        bad2.layers.push_back(make_qx_desc(1, 0));
        CHECK_THROWS(stream_serialized_size(bad2)); // full without metadata blob
    }
}

static void test_split_writer_sizing_no_gpu_read() {
    tensors ts = make_tensors();
    split_writer w([](const ggml_tensor * t) { return t->ne[1] == 8 ? 0u : 1u; });
    w.write_tensor(ts.t0, 64, 320);
    w.write_tensor(ts.t0, 384, 128);
    w.write_tensor(ts.t1, 0, 2048);
    CHECK(g_wrap_get == 0); // recording does not read tensors

    const mixed_manifest & m = w.manifest();
    std::vector<uint8_t> wire = mixed_manifest_serialize(m);
    CHECK(!wire.empty());

    stream_input in;
    for (const auto & lp : m.layer_payload_sizes) {
        in.layers.push_back(make_qx_desc(lp.first, lp.second));
    }
    in.metadata_version = 1;
    in.metadata = wire.data();
    in.metadata_size = wire.size();
    in.flags = snapshot_flags::full;

    const uint64_t expected = 96 + wire.size() + 2 * 101 + 320 + 128 + 2048 + 8;
    CHECK(stream_serialized_size(in) == expected);
    CHECK(g_wrap_get == 0); // sizing performs zero tensor reads

    // and the real streamed size matches, with read_payload served by the writer
    stream_read_callbacks reads;
    reads.read_payload = [&](uint32_t layer_id, uint64_t offset, uint8_t * dst, size_t count) {
        return w.read_payload(layer_id, offset, dst, count);
    };
    uint64_t written = 0;
    std::vector<uint8_t> streamed;
    stream_write_callbacks writes;
    writes.write_bytes = [&](const uint8_t * bytes, size_t count) {
        streamed.insert(streamed.end(), bytes, bytes + count);
        written += count;
        return true;
    };
    CHECK(stream_serialize(in, reads, writes));
    CHECK(written == expected);
    CHECK(g_wrap_get > 0); // the write pass did read the tensors (bounded)
    // the streamed bytes parse back to the exact layer payloads
    auto backing = std::make_shared<std::vector<uint8_t>>(streamed);
    snapshot s = snapshot::parse_view(backing->data(), backing->size(), backing);
    CHECK(s.layers().size() == 2);
    const std::vector<uint8_t> ref0 = ref_payload(m, ts, 0);
    CHECK(s.payload_size(0) == ref0.size());
    CHECK(std::memcmp(s.payload(0), ref0.data(), ref0.size()) == 0);
    const std::vector<uint8_t> ref1 = ref_payload(m, ts, 1);
    CHECK(std::memcmp(s.payload(1), ref1.data(), ref1.size()) == 0);
}

// ---- review fixes -----------------------------------------------------------

static std::vector<uint8_t> ref1_payload(const mixed_manifest & m, const tensors & ts, uint32_t layer) {
    std::vector<uint8_t> out;
    for (const mixed_manifest_event & e : m.events) {
        if (e.kind == 1 && e.layer_id == layer) {
            const std::vector<uint8_t> & src = layer == 0 ? ts.d0 : ts.d1;
            out.insert(out.end(), src.begin() + long(e.tensor_offset),
                       src.begin() + long(e.tensor_offset) + long(e.size));
        }
    }
    return out;
}

static void test_large_metadata_offsets() {
    // CRITICAL regression: ordinary metadata larger than 255 bytes BEFORE a
    // tensor event plus trailing metadata; meta_offsets must be u64
    tensors ts = make_tensors();
    split_writer w([](const ggml_tensor * t) { return t->ne[1] == 8 ? 0u : 1u; });
    std::vector<uint8_t> head(1024);
    fill_pattern(head, 61);
    std::vector<uint8_t> tail_meta(300);
    fill_pattern(tail_meta, 62);
    w.write(head.data(), head.size());
    w.write_tensor(ts.t0, 64, 320);
    w.write_tensor(ts.t1, 0, 2048);
    w.write(tail_meta.data(), tail_meta.size());

    const mixed_manifest & m = w.manifest();
    CHECK(m.validate().empty());
    CHECK(m.events.size() == 4);
    CHECK(m.events[2].kind == 1 && m.events[2].layer_id == 1); // t1 tensor event
    CHECK(m.events[3].kind == 0 && m.events[3].stream_offset == 1024 + 320 + 2048);
    CHECK(m.events[3].size == 300);

    const std::vector<uint8_t> ref0 = ref_payload(m, ts, 0);
    auto owner0 = std::make_shared<std::vector<uint8_t>>(ref0);
    auto owner1 = std::make_shared<std::vector<uint8_t>>(ref1_payload(m, ts, 1));
    split_reader r(m, { { 0, owner0->data(), owner0->size(), owner0 },
                        { 1, owner1->data(), owner1->size(), owner1 } });

    // read() must serve the head metadata correctly across the >255 boundary
    std::vector<uint8_t> got_head(1024);
    r.read(got_head.data(), got_head.size());
    CHECK(std::memcmp(got_head.data(), head.data(), head.size()) == 0);
    // tensor reads with varied boundaries
    dests dd = make_dests(ts, { ts.t0, ts.t1 });
    r.read_tensor(dd.t[0], 0, 320);
    r.read_tensor(dd.t[1], 0, 2048);
    // trailing metadata read correctly (offset 1024+320+2048 = 3392 > 255)
    std::vector<uint8_t> got_tail(300);
    r.read(got_tail.data(), got_tail.size());
    CHECK(std::memcmp(got_tail.data(), tail_meta.data(), tail_meta.size()) == 0);
    CHECK(r.remaining() == 0);
    mock_outer outer;
    r.defer_to(outer);
    outer.commit();
    std::vector<uint8_t> got0(320);
    ggml_backend_tensor_get(dd.t[0], got0.data(), 0, 320);
    CHECK(std::memcmp(got0.data(), ts.d0.data() + 64, 320) == 0);
}
static void test_manifest_malformed() {
    constexpr uint64_t MAX_T = uint64_t(1) << 42; // 4 TiB sanity bound (module-internal)
    tensors ts = make_tensors();
    split_writer w([](const ggml_tensor * t) { return t->ne[1] == 8 ? 0u : 1u; });
    w.write("META", 4); // metadata event so the length-mismatch checks apply
    w.write_tensor(ts.t0, 0, 3200);
    const std::vector<uint8_t> wire = mixed_manifest_serialize(w.manifest());
    CHECK(!wire.empty());
    mixed_manifest parsed;
    std::string err;

    // metadata shorter than the sum of kind-0 events
    {
        mixed_manifest m = w.manifest();
        m.metadata.resize(m.metadata.size() / 2); // events still claim the full length
        CHECK(!m.validate().empty());
        std::vector<uint8_t> w2 = mixed_manifest_serialize(m);
        CHECK(w2.empty()); // serialize refuses too
    }
    // zero-length event
    {
        mixed_manifest m = w.manifest();
        mixed_manifest_event e = m.events[0];
        e.size = 0;
        m.events.push_back(e);
        CHECK(!m.validate().empty());
    }
    // invalid tensor offset (would overflow on addition); the tensor event is
    // events[1] (events[0] is the "META" metadata run)
    {
        mixed_manifest m = w.manifest();
        m.events[1].tensor_offset = MAX_T;
        m.events[1].size = 16;
        CHECK(!m.validate().empty());
    }
    // invalid tensor offset: size + offset overflows
    {
        mixed_manifest m = w.manifest();
        m.events[1].tensor_offset = MAX_T - 8;
        CHECK(!m.validate().empty());
    }
    // bad layer id (undeclared) and unordered layer table
    {
        mixed_manifest m = w.manifest();
        m.events[1].layer_id = 42;
        CHECK(!m.validate().empty());
        m = w.manifest();
        m.layer_payload_sizes.clear();
        m.layer_payload_sizes.emplace_back(1, 3200); // id not matching the event
        CHECK(!m.validate().empty());
    }
    // parse: null data with size
    CHECK(!mixed_manifest_parse(nullptr, 8, parsed, &err));
    // parse: huge encoded metadata length with a dummy pointer (no deref)
    {
        std::vector<uint8_t> evil(wire);
        // overwrite the metadata length field (offset 8..16) with 4 TiB
        for (int i = 0; i < 8; ++i) { evil[8 + i] = i == 7 ? 0x04 : 0; }
        CHECK(!mixed_manifest_parse(evil.data(), evil.size(), parsed, &err));
    }
    // parse: event count too large for the remaining bytes (no big reserve)
    {
        std::vector<uint8_t> evil(wire);
        // n_events field at offset 16 + metadata_size; metadata is empty in
        // this manifest, so n_events sits at offset 16
        const size_t off = 16;
        for (int i = 0; i < 8; ++i) { evil[off + i] = 0; }
        evil[off] = 0xFF;
        evil[off + 1] = 0xFF;
        CHECK(!mixed_manifest_parse(evil.data(), evil.size(), parsed, &err));
    }
    // exact wire length: serialize(parse(wire)) == wire
    {
        CHECK(mixed_manifest_parse(wire.data(), wire.size(), parsed, &err));
        std::vector<uint8_t> wire2 = mixed_manifest_serialize(parsed);
        CHECK(wire2 == wire);
    }
}

static void test_reader_terminal_state() {
    tensors ts = make_tensors();
    split_writer w([](const ggml_tensor * t) { return t->ne[1] == 8 ? 0u : 1u; });
    w.write_tensor(ts.t0, 0, 3200);
    const std::vector<uint8_t> ref0 = ref_payload(w.manifest(), ts, 0);
    auto owner0 = std::make_shared<std::vector<uint8_t>>(ref0);

    // cancel is terminal: every later operation is rejected; repeated cancel ok
    {
        split_reader r(w.manifest(), { { 0, owner0->data(), owner0->size(), owner0 } });
        dests dd = make_dests(ts, { ts.t0 });
        r.read_tensor(dd.t[0], 0, 3200);
        r.cancel();
        r.cancel(); // repeated cancel is safe
        CHECK_THROWS(r.read_tensor(dd.t[0], 0, 1));
        CHECK_THROWS(r.stage_tensor_set(dd.t[0], owner0->data(), 0, 8));
        CHECK_THROWS(r.stage_tensor_clear(dd.t[0], 0, 8));
        CHECK_THROWS(r.queue_borrowed(dd.t[0], 0, owner0->data(), 8, owner0));
        CHECK_THROWS(r.on_commit([]() {}));
        mock_outer outer;
        CHECK_THROWS(r.defer_to(outer));
        CHECK(outer.cbs.empty()); // nothing transferred
        CHECK(g_wrap_set == 0);   // no writes ever
    }
    // defer is terminal too
    {
        split_reader r(w.manifest(), { { 0, owner0->data(), owner0->size(), owner0 } });
        dests dd = make_dests(ts, { ts.t0 });
        r.read_tensor(dd.t[0], 0, 3200);
        mock_outer outer;
        r.defer_to(outer);
        CHECK_THROWS(r.read_tensor(dd.t[0], 0, 1));
        CHECK_THROWS(r.on_commit([]() {}));
        CHECK_THROWS(r.defer_to(outer));
        outer.cancel();
    }
    // empty on_commit callback rejected
    {
        split_reader r(w.manifest(), { { 0, owner0->data(), owner0->size(), owner0 } });
        CHECK_THROWS(r.on_commit({}));
    }
    // guard failure leaves the reader terminal (no partial retry)
    {
        split_reader_limits lim;
        lim.max_fixup_staging = 64;
        split_reader r(w.manifest(), { { 0, owner0->data(), owner0->size(), owner0 } }, lim);
        dests dd = make_dests(ts, { ts.t0 });
        std::vector<uint8_t> fix(64);
        r.stage_tensor_set(dd.t[0], fix.data(), 0, 64);
        CHECK_THROWS(r.stage_tensor_set(dd.t[0], fix.data(), 64, 1)); // cap failure
        CHECK_THROWS(r.stage_tensor_set(dd.t[0], fix.data(), 128, 1)); // terminal now
        CHECK_THROWS(r.read_tensor(dd.t[0], 0, 1)); // even reads are rejected
    }
    // operation budget: total ops enforced (consume splits count too)
    {
        split_reader_limits lim;
        lim.max_operations = 3;
        split_reader r(w.manifest(), { { 0, owner0->data(), owner0->size(), owner0 } }, lim);
        dests dd = make_dests(ts, { ts.t0 });
        r.read_tensor(dd.t[0], 0, 3200); // 1 op (single event)
        r.stage_tensor_clear(dd.t[0], 0, 8); // 2 ops
        r.queue_borrowed(dd.t[0], 8, owner0->data(), 8, owner0); // 3 ops
        CHECK_THROWS(r.stage_tensor_clear(dd.t[0], 0, 8)); // budget exhausted
        CHECK_THROWS(r.queue_borrowed(dd.t[0], 0, owner0->data(), 8, owner0));
    }
    // callback budget
    {
        split_reader_limits lim;
        lim.max_callbacks = 1;
        split_reader r(w.manifest(), { { 0, owner0->data(), owner0->size(), owner0 } }, lim);
        r.on_commit([]() {});
        CHECK_THROWS(r.on_commit([]() {}));
    }
}

static void test_writer_seal_and_limits() {
    tensors ts = make_tensors();
    // seal: read_payload then write/write_tensor are rejected
    {
        split_writer w([](const ggml_tensor * t) { return t->ne[1] == 8 ? 0u : 1u; });
        w.write_tensor(ts.t0, 0, 64);
        uint8_t dst[16];
        CHECK(w.read_payload(0, 0, dst, 16) == 16);
        CHECK_THROWS(w.write_tensor(ts.t0, 64, 64));   // after producer read
        CHECK_THROWS(w.write("X", 1));                 // after producer read
        const std::vector<uint8_t> ref0 = ref_payload(w.manifest(), ts, 0);
        CHECK(std::memcmp(dst, ref0.data(), 16) == 0); // first read still valid
    }
    // read_payload null dst / overflow guards
    {
        split_writer w([](const ggml_tensor *) { return 0u; });
        w.write_tensor(ts.t0, 0, 3200);
        CHECK_THROWS(w.read_payload(0, 0, nullptr, 16));
        std::vector<uint8_t> dst(16);
        CHECK(w.read_payload(0, (uint64_t(1) << 42), dst.data(), 16) == 0); // overflow-safe
    }
    // max_virtual_stream_bytes enforced in write and write_tensor
    {
        split_writer_limits lim;
        lim.max_virtual_stream_bytes = 100;
        split_writer w([](const ggml_tensor *) { return 0u; }, lim);
        w.write_tensor(ts.t0, 0, 100);
        CHECK_THROWS(w.write_tensor(ts.t0, 0, 1));
        CHECK_THROWS(w.write("X", 1));
    }
    // a guard failure leaves the writer terminal
    {
        split_writer_limits lim;
        lim.max_virtual_stream_bytes = 100;
        split_writer w([](const ggml_tensor *) { return 0u; }, lim);
        w.write_tensor(ts.t0, 0, 100);
        CHECK_THROWS(w.write_tensor(ts.t0, 0, 1));
        CHECK_THROWS(w.write("X", 1)); // terminal
    }
}

static void test_many_events_sequential() {
    // 10k+ events: consume stays sequential (monotonic event cursor) and
    // correct; no quadratic event-0 rescans
    const uint32_t n = 10001;
    tensors ts = make_tensors();
    split_writer w([](const ggml_tensor *) { return 0u; });
    std::vector<uint8_t> meta(n);
    for (uint32_t i = 0; i < n; ++i) {
        meta[i] = uint8_t(i);
    }
    for (uint32_t i = 0; i < n; ++i) {
        w.write(meta.data() + i, 1); // each becomes its own 1-byte event? no:
        // adjacent writes combine; force a tensor event between runs
        w.write_tensor(ts.t0, 0, 1);
    }
    const mixed_manifest & m = w.manifest();
    CHECK(m.events.size() == 2 * n); // n metadata + n tensor events
    CHECK(m.validate().empty());

    const std::vector<uint8_t> ref0 = ref_payload(m, ts, 0);
    auto owner0 = std::make_shared<std::vector<uint8_t>>(ref0);
    split_reader r(m, { { 0, owner0->data(), owner0->size(), owner0 } });
    // destination tensor of n bytes (10001 F32 elements -> 40004 B)
    ggml_init_params params{};
    params.mem_size = 1u << 20;
    params.no_alloc = true;
    struct dd_local {
        ggml_context * ctx = nullptr;
        ggml_backend_t be = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        ggml_tensor * t = nullptr;
        ~dd_local() {
            if (buf) { ggml_backend_buffer_free(buf); }
            if (be) { ggml_backend_free(be); }
            if (ctx) { ggml_free(ctx); }
        }
    } dd;
    dd.ctx = ggml_init(params);
    dd.t = ggml_new_tensor_1d(dd.ctx, GGML_TYPE_F32, n);
    dd.be = ggml_backend_cpu_init();
    dd.buf = ggml_backend_alloc_ctx_tensors(dd.ctx, dd.be);
    g_wrap_set = 0;
    uint8_t b = 0;
    for (uint32_t i = 0; i < n; ++i) {
        r.read(&b, 1);
        CHECK(b == uint8_t(i));
        r.read_tensor(dd.t, i, 1); // each event serves t0[0..1)
    }
    CHECK(r.remaining() == 0);
    mock_outer outer;
    r.defer_to(outer);
    outer.commit();
    CHECK(g_wrap_set == n); // one borrowed op per 1-byte tensor event
    std::vector<uint8_t> got_v(size_t(n), 0);
    ggml_backend_tensor_get(dd.t, got_v.data(), 0, size_t(n));
    for (uint32_t i = 0; i < n; ++i) {
        CHECK(got_v[i] == ts.d0[0]); // every event carried the same source byte
    }
}

static void test_duplicate_outer_callback_invocation() {
    tensors ts = make_tensors();
    split_writer w([](const ggml_tensor * t) { return t->ne[1] == 8 ? 0u : 1u; });
    w.write_tensor(ts.t0, 0, 64);
    const std::vector<uint8_t> ref0 = ref_payload(w.manifest(), ts, 0);
    auto owner0 = std::make_shared<std::vector<uint8_t>>(ref0);
    split_reader r(w.manifest(), { { 0, owner0->data(), owner0->size(), owner0 } });
    dests dd = make_dests(ts, { ts.t0 });
    g_wrap_set = 0;
    r.read_tensor(dd.t[0], 0, 64);
    mock_outer outer;
    r.defer_to(outer);
    outer.commit();
    CHECK(g_wrap_set == 1); // single application
    // a buggy outer invoking the callback twice must be detected
    {
        // fresh transaction
        split_writer w2([](const ggml_tensor * t) { return t->ne[1] == 8 ? 0u : 1u; });
        w2.write_tensor(ts.t0, 0, 64);
        const std::vector<uint8_t> r2 = ref_payload(w2.manifest(), ts, 0);
        auto o2 = std::make_shared<std::vector<uint8_t>>(r2);
        split_reader r3(w2.manifest(), { { 0, o2->data(), o2->size(), o2 } });
        dests dd2 = make_dests(ts, { ts.t0 });
        r3.read_tensor(dd2.t[0], 0, 64);
        mock_outer bad;
        r3.defer_to(bad);
        CHECK(bad.cbs.size() == 1);
        std::function<void()> cb = bad.cbs[0];
        g_wrap_set = 0;
        cb(); // first invocation applies
        CHECK(g_wrap_set == 1);
        bool threw = false;
        try {
            cb(); // duplicate invocation
        } catch (const kv_mixed_error &) {
            threw = true;
        }
        CHECK(threw);
    }
}

static void test_owner_required() {
    tensors ts = make_tensors();
    split_writer w([](const ggml_tensor * t) { return t->ne[1] == 8 ? 0u : 1u; });
    w.write_tensor(ts.t0, 0, 3200);
    const std::vector<uint8_t> ref0 = ref_payload(w.manifest(), ts, 0);

    // nonzero span without an owner is rejected (zero-copy lifetime claims)
    CHECK_THROWS(split_reader(w.manifest(), { { 0, ref0.data(), ref0.size(), nullptr } }));
    // queue_borrowed without an owner is rejected
    {
        split_reader r(w.manifest(), { { 0, ref0.data(), ref0.size(), std::make_shared<std::vector<uint8_t>>(ref0) } });
        dests dd = make_dests(ts, { ts.t0 });
        CHECK_THROWS(r.queue_borrowed(dd.t[0], 0, ref0.data(), 8, nullptr));
    }
}

int main() {
    test_split_writer_records_without_reads();
    test_split_writer_bounds();
    test_manifest_wire_roundtrip();
    test_read_payload_slices();
    test_split_reader_reverse_and_outer_commit();
    test_read_across_events_and_tensor_splits();
    test_fixup_caps_and_big_borrowed();
    test_defer_ordering_and_qx_append();
    test_outer_cancel_releases_everything();
    test_borrowed_mutation_detected();
    test_commit_defer_contract();
    test_reader_ctor_validation();
    test_callback_failure_during_outer_commit();
    test_stream_serialized_size();
    test_split_writer_sizing_no_gpu_read();
    test_large_metadata_offsets();
    test_manifest_malformed();
    test_reader_terminal_state();
    test_writer_seal_and_limits();
    test_many_events_sequential();
    test_duplicate_outer_callback_invocation();
    test_owner_required();

    std::printf("%s: %d checks, %d failures\n", g_failures == 0 ? "PASS" : "FAIL", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}