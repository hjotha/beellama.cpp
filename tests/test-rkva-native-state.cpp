// CPU regression for the canonical native KVarN4 owner-state codec and the real
// producer export/import path. Synthetic values only; no model, GPU or D3D12.
#include "../ggml/include/rkva-native-state.h"
#include "../ggml/include/rkva-native-snapshot.h"
#include "../ggml/src/ggml-remote-attn/rkva-kvarn-producer.h"
#include "../tools/xbox/uwp/rkva/rkva-kvarn-range.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kOrigin = 1024;
constexpr uint32_t kCapacity = 1024;
constexpr uint32_t kHeads = 2;
constexpr uint32_t kIterations = 3;

struct Fixture {
    uint32_t tail;
    bool compact_stage;
    ggml_backend_cpu_kvarn_store_direct_v1_t store = nullptr;
    std::vector<float> keys;
    std::vector<float> values;
    std::vector<int32_t> positions;
    std::unique_ptr<rkva_kvarn_producer> producer;

    explicit Fixture(uint32_t precision_tail, bool compact = false, uint32_t total = 900)
        : tail(precision_tail), compact_stage(compact) {
        auto * reg = ggml_backend_cpu_reg();
        store = reinterpret_cast<ggml_backend_cpu_kvarn_store_direct_v1_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_kvarn_store_direct_v1"));
        assert(store);
        const size_t width = size_t(kHeads) * 256;
        keys.resize(size_t(total) * width);
        values.resize(keys.size());
        positions.resize(total);
        for (uint32_t t = 0; t < total; ++t) {
            positions[t] = int32_t(kOrigin + t);
            for (size_t d = 0; d < width; ++d) {
                keys[size_t(t) * width + d] = std::sin(float(t * 7 + d * 3) * 0.003f) * 0.7f;
                values[size_t(t) * width + d] = std::cos(float(t * 11 + d * 5) * 0.007f) * 0.9f;
            }
        }
        producer = std::make_unique<rkva_kvarn_producer>(
            kOrigin, kCapacity, kHeads, kIterations, store, tail, compact);
    }

    void append(uint32_t first, uint32_t count, bool reset = false,
                const std::vector<float> * k = nullptr,
                const std::vector<float> * v = nullptr) {
        std::vector<uint8_t> packet;
        assert(producer->append(k ? k->data() + size_t(first) * kHeads * 256 : keys.data() + size_t(first) * kHeads * 256,
                                v ? v->data() + size_t(first) * kHeads * 256 : values.data() + size_t(first) * kHeads * 256,
                                positions.data() + first, count, reset, packet));
        assert(!packet.empty());
        rkva_kvarn_range_upload header{};
        assert(packet.size() >= sizeof(header));
        std::memcpy(&header, packet.data(), sizeof(header));
        assert(header.live_count == producer->rows());
    }
};

std::vector<uint8_t> export_one(const rkva_kvarn_producer & producer, uint64_t generation) {
    rkva_native_state::View view;
    assert(producer.state_export(generation, view));
    std::vector<rkva_native_state::View> views{view};
    std::vector<uint8_t> bytes;
    std::string error;
    assert(rkva_native_state::write_set(views, bytes, &error));
    return bytes;
}

void import_one(rkva_kvarn_producer & producer, const std::vector<uint8_t> & bytes) {
    std::vector<rkva_native_state::Storage> layers;
    std::string error;
    if (!rkva_native_state::read_set(bytes, layers, &error)) {
        std::fprintf(stderr, "read_set failed: %s\n", error.c_str());
        assert(false);
    }
    assert(layers.size() == 1);
    assert(producer.state_import(layers.front(), &error));
}

void test_roundtrip(uint32_t tail) {
    Fixture source(tail);
    source.append(0, 127);
    source.append(127, 256);
    source.append(383, 130);
    assert(source.producer->rows() == 513);
    const auto original = export_one(*source.producer, 0x1234000000000001ull);

    std::vector<rkva_native_state::Storage> layers;
    std::string error;
    assert(rkva_native_state::read_set(original, layers, &error));
    assert(layers.size() == 1);
    const auto & layout = layers[0].layout;
    assert(layout.origin == kOrigin && layout.capacity == kCapacity);
    assert(layout.heads == kHeads && layout.iterations == kIterations);
    assert(layout.precision_tail_tokens == tail && layout.rows == 513);
    assert(layout.record_base_group == 0);
    assert(layout.saved_generation == 0x1234000000000001ull);
    assert(layout.compact_stage == 0);
    assert(layers[0].stage_k.size() == size_t(384) * kHeads * 256);
    assert(layers[0].tail_k.size() == size_t(tail) * kHeads * 256);
    assert(layers[0].original_k.size() == size_t(256) * kHeads * 256);
    assert(layers[0].original_pos.size() == 256);
    // CPU/global records are normalized to the origin group.
    assert(layers[0].records_k.size() == size_t(513 / 128) * kHeads * 2 * 8960);

    Fixture restored(tail);
    import_one(*restored.producer, original);
    assert(restored.producer->rows() == source.producer->rows());
    assert(export_one(*restored.producer, 0x1234000000000001ull) == original);

    // Continue after import: seals, tail and original ring must advance the same.
    source.append(513, 200);
    restored.append(513, 200);
    assert(export_one(*source.producer, 0x1234000000000002ull) ==
           export_one(*restored.producer, 0x1234000000000002ull));

    // Tail 0 has no tail payload; tail 128 covers the physical ring inputs.
    if (tail) {
        assert(!layers[0].tail_k.empty());
    } else {
        assert(layers[0].tail_k.empty() && layers[0].tail_v.empty());
    }
}

void test_compact_layout_normalization() {
    Fixture source(128);
    source.append(0, 513);
    rkva_native_state::View global;
    assert(source.producer->state_export(0x100000001ull, global));
    const size_t groups = size_t(source.producer->rows() / 128) * size_t(kHeads) * 2 * 8960;
    std::vector<uint8_t> compact_records(global.records_k, global.records_k + groups);
    auto compact = global;
    compact.layout.record_base_group = kOrigin / 128;
    compact.records_k = compact_records.data();
    std::vector<rkva_native_state::View> a{global}, b{compact};
    std::string error;
    std::vector<uint8_t> global_bytes, compact_bytes;
    assert(rkva_native_state::write_set(a, global_bytes, &error));
    assert(rkva_native_state::write_set(b, compact_bytes, &error));
    // The payload is normalized; only the diagnostic source-layout word in the
    // header differs. A compact-exported set imports into the global CPU layout
    // and re-exports byte-identically.
    Fixture restored(128);
    import_one(*restored.producer, compact_bytes);
    assert(export_one(*restored.producer, 0x100000001ull) == global_bytes);
}

void test_multilayer() {
    Fixture a(128), b(128), c(128);
    a.append(0, 300);
    b.append(0, 300);
    c.append(0, 300);
    rkva_native_state::View va, vb, vc;
    assert(a.producer->state_export(0x200000001ull, va));
    assert(b.producer->state_export(0x200000001ull, vb));
    assert(c.producer->state_export(0x200000001ull, vc));
    std::vector<rkva_native_state::View> views{va, vb, vc};
    std::string error;
    std::vector<uint8_t> bytes;
    assert(rkva_native_state::write_set(views, bytes, &error));
    std::vector<rkva_native_state::Storage> layers;
    assert(rkva_native_state::read_set(bytes, layers, &error));
    assert(layers.size() == 3);
    Fixture da(128), db(128), dc(128);
    assert(da.producer->state_import(layers[0], &error));
    assert(db.producer->state_import(layers[1], &error));
    assert(dc.producer->state_import(layers[2], &error));
    assert(export_one(*da.producer, 0x200000001ull) == export_one(*a.producer, 0x200000001ull));
    assert(export_one(*db.producer, 0x200000001ull) == export_one(*b.producer, 0x200000001ull));
    assert(export_one(*dc.producer, 0x200000001ull) == export_one(*c.producer, 0x200000001ull));
}

void test_rewind_and_reappend() {
    Fixture source(128);
    source.append(0, 513);
    assert(source.producer->can_rewind(385));
    assert(!source.producer->can_rewind(384)); // 129 removed
    const auto before = export_one(*source.producer, 0x300000001ull);
    std::vector<uint8_t> rejected;
    assert(!source.producer->rewind_tail(384, rejected));
    assert(rejected.empty());
    assert(!source.producer->rewind(384));
    assert(export_one(*source.producer, 0x300000001ull) == before);

    std::vector<uint8_t> tail_rows;
    assert(source.producer->rewind_tail(385, tail_rows));
    const size_t row_bytes = size_t(kHeads) * 256 * sizeof(uint16_t);
    assert(tail_rows.size() == 2 * 128 * row_bytes);
    assert(source.producer->rewind(385));
    assert(source.producer->rows() == 385);
    const auto rewound = export_one(*source.producer, 0x300000002ull);

    Fixture restored(128);
    import_one(*restored.producer, rewound);
    assert(restored.producer->rows() == 385);
    // The ring retains the 257..384 window after this rewind; a second removal
    // from 385 is rejected until new rows advance the required window.
    assert(restored.producer->can_rewind(385));
    assert(!restored.producer->can_rewind(384));

    source.append(385, 128);
    restored.append(385, 128);
    assert(restored.producer->can_rewind(385));
    assert(export_one(*source.producer, 0x300000003ull) ==
           export_one(*restored.producer, 0x300000003ull));
}

void test_corruption_and_short_io() {
    Fixture source(128);
    source.append(0, 300);
    auto bytes = export_one(*source.producer, 0x400000001ull);
    std::vector<rkva_native_state::Storage> layers;
    std::string error;

    auto bad_magic = bytes;
    bad_magic[0] ^= 0xff;
    assert(!rkva_native_state::read_set(bad_magic, layers, &error));
    assert(layers.empty());

    auto bad_version = bytes;
    bad_version[4] = 2;
    assert(!rkva_native_state::read_set(bad_version, layers, &error));

    auto bad_checksum = bytes;
    bad_checksum.back() ^= 0x01;
    assert(!rkva_native_state::read_set(bad_checksum, layers, &error));

    auto truncated = bytes;
    truncated.resize(truncated.size() - 1);
    assert(!rkva_native_state::read_set(truncated, layers, &error));

    auto zero_generation = bytes;
    std::memset(zero_generation.data() + 48, 0, 8);
    // Recompute the footer through a direct header rewrite is not required:
    // generation zero is rejected before the checksum, proving no state is read.
    assert(!rkva_native_state::read_set(zero_generation, layers, &error));

    class short_source final : public rkva_native_state::Source {
      public:
        short_source(const std::vector<uint8_t> & data, size_t limit) : data(data), limit(limit) {}
        bool read(void * dst, size_t size) override {
            if (size > limit - offset) return false;
            std::memcpy(dst, data.data() + offset, size);
            offset += size;
            return true;
        }
      private:
        const std::vector<uint8_t> & data;
        size_t limit = 0;
        size_t offset = 0;
    } source_short(bytes, bytes.size() - 17);
    assert(!rkva_native_state::read_set(source_short, layers, &error));

    class short_sink final : public rkva_native_state::Sink {
      public:
        explicit short_sink(size_t limit) : limit(limit) {}
        bool write(const void * data, size_t size) override {
            (void) data;
            if (size > limit - written) return false;
            written += size;
            return true;
        }
      private:
        size_t limit = 0;
        size_t written = 0;
    } sink(bytes.size() - 13);
    rkva_native_state::View view;
    assert(source.producer->state_export(0x400000002ull, view));
    std::vector<rkva_native_state::View> views{view};
    assert(!rkva_native_state::write_set(views, sink, &error));
}

void test_import_is_atomic() {
    Fixture source(128);
    source.append(0, 300);
    auto before = export_one(*source.producer, 0x500000001ull);
    std::vector<rkva_native_state::Storage> layers;
    std::string error;
    assert(rkva_native_state::read_set(before, layers, &error));

    auto bad_layout = layers[0];
    bad_layout.layout.capacity += 128;
    bad_layout.layout.rows = bad_layout.layout.capacity;
    assert(!source.producer->state_import(bad_layout, &error));
    assert(export_one(*source.producer, 0x500000001ull) == before);

    auto bad_ring = layers[0];
    bad_ring.original_pos[0] = UINT32_MAX;
    assert(!source.producer->state_import(bad_ring, &error));
    assert(export_one(*source.producer, 0x500000001ull) == before);

    auto bad_size = layers[0];
    bad_size.stage_k.pop_back();
    assert(!source.producer->state_import(bad_size, &error));
    assert(export_one(*source.producer, 0x500000001ull) == before);
}

void test_generation_validation() {
    Fixture source(128);
    source.append(0, 130);
    rkva_native_state::View view;
    assert(source.producer->state_export(0x6FFFFFFFFull, view));
    // Low 32 bits all ones is the reserved exhausted namespace.
    assert(!rkva_native_state::valid_layout(view.layout, nullptr));
    rkva_native_state::View zero = view;
    zero.layout.saved_generation = 0;
    assert(!rkva_native_state::valid_layout(zero.layout, nullptr));
    rkva_native_state::View good = view;
    good.layout.saved_generation = 0x600000001ull;
    assert(rkva_native_state::valid_layout(good.layout, nullptr));
}

void test_aggregate_layer_budget() {
    // Exact review fixture: 544,227,328 bytes for 16 production layers. A
    // fixed 256-layer product would wrongly reject it (8,707,637,248 bytes).
    rkva_native_state::Layout production;
    production.origin = 102400;
    production.capacity = 26624;
    production.heads = 4;
    production.iterations = 3;
    production.precision_tail_tokens = 128;
    production.rows = production.capacity;
    production.saved_generation = 0x800000002ull;
    std::string production_error;
    assert(rkva_native_state::valid_layout(production, &production_error));
    assert(rkva_native_state::layer_bytes(production) == 34014208ull);
    assert(rkva_native_state::valid_payload_size(production, 16));
    assert(!rkva_native_state::valid_payload_size(production, 256));

    rkva_native_state::Layout layout;
    layout.origin = 1024;
    layout.capacity = 700000;
    layout.heads = 1;
    layout.iterations = 3;
    layout.precision_tail_tokens = 128;
    layout.rows = layout.capacity;
    layout.record_base_group = 0;
    layout.saved_generation = 0x800000001ull;
    std::string error;
    assert(rkva_native_state::valid_layout(layout, &error));
    assert(rkva_native_state::valid_payload_size(layout, 16));
    assert(!rkva_native_state::valid_payload_size(layout, 64));
    rkva_native_state::SetHeader header;
    header.layers = 16;
    header.layout = layout;
    header.layer_bytes = rkva_native_state::layer_bytes(layout);
    header.payload_bytes = rkva_native_state::payload_bytes(layout, header.layers);
    uint8_t encoded[rkva_native_state::kHeaderBytes];
    assert(rkva_native_state::encode_header(header, encoded, &error));
    header.layers = 64;
    header.payload_bytes = rkva_native_state::payload_bytes(layout, header.layers);
    assert(!rkva_native_state::encode_header(header, encoded, &error));
}

void test_reset_coverage_and_seals() {
    // Real revision-3 coverage transitions used by the engine manifest.
    auto c = rkva_kvarn_range::coverage_after_reset(513, 128);
    assert(c.tail_begin == 385);
    assert(c.stage_begin == 384);
    c = rkva_kvarn_range::coverage_after_append(c, 640, 128);
    assert(c.tail_begin == 385);
    assert(c.stage_begin == 384);
    assert(rkva_kvarn_range::coverage_after_rewind(c, 600, 472, 128).tail_begin == 385);
    assert(rkva_kvarn_range::coverage_after_reset(300, 0).tail_begin == 300);
}

} // namespace

int main() {
    test_roundtrip(128);
    test_roundtrip(0);
    test_compact_layout_normalization();
    test_multilayer();
    test_rewind_and_reappend();
    test_corruption_and_short_io();
    test_import_is_atomic();
    test_generation_validation();
    test_aggregate_layer_budget();
    test_reset_coverage_and_seals();
    std::printf("native owner-state CPU regression: ok\n");
    return 0;
}
