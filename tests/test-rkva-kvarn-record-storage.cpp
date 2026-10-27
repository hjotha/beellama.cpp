// Host-record storage qualification. Build this same source once against a
// frozen OLD producer header and once against the NEW header:
//   -DRKVA_KVARN_PRODUCER_HEADER='"/abs/old/ggml/src/ggml-remote-attn/rkva-kvarn-producer.h"'
// The frozen header needs its sibling headers, so freeze the whole old
// ggml/src tree (see docs/agent-reports/rkva-compact-host-records.md).
// Default CPU mode checks the global host span and canonical record bytes and
// exercises the packet harness; it never initializes a GPU backend. With
// GGML_RKVA_GPU_TEST=1 the same grid runs on the CUDA producer so packets from
// the OLD and NEW headers can be compared byte-for-byte:
//   <bin> --record DIR [--expect-memory global|compact]
//   <bin> --compare DIR [--expect-memory global|compact]
// --expect-memory asserts the ctor host_bytes before any transient buffer:
// `global` is the pre-compaction span, `compact` the CUDA-only span.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#ifndef RKVA_KVARN_PRODUCER_HEADER
#define RKVA_KVARN_PRODUCER_HEADER "../ggml/src/ggml-remote-attn/rkva-kvarn-producer.h"
#endif
#include RKVA_KVARN_PRODUCER_HEADER
#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <utility>
#include <vector>

static void mode(const char * value) {
#ifdef _WIN32
    _putenv_s("GGML_RKVA_NATIVE_PRODUCER", value);
#else
    setenv("GGML_RKVA_NATIVE_PRODUCER", value, 1);
#endif
}

static bool gpu_mode() {
    const char * value = std::getenv("GGML_RKVA_GPU_TEST");
    return value && *value && std::strcmp(value, "0") != 0;
}

static ggml_backend_cpu_kvarn_store_direct_v1_t cpu_store() {
    const auto store = reinterpret_cast<ggml_backend_cpu_kvarn_store_direct_v1_t>(
        ggml_backend_reg_get_proc_address(ggml_backend_cpu_reg(), "ggml_backend_cpu_kvarn_store_direct_v1"));
    assert(store);
    return store;
}

enum class expect_memory { none, global, compact };

struct scenario {
    uint32_t origin, capacity, heads, tail;
    bool compact_stage;
};

static std::string tag(const scenario & s) {
    return "o" + std::to_string(s.origin) + "-c" + std::to_string(s.capacity) +
        "-h" + std::to_string(s.heads) + "-t" + std::to_string(s.tail) +
        "-s" + std::to_string(unsigned(s.compact_stage));
}

static std::string geometry(const scenario & s) {
    return tag(s) + " origin=" + std::to_string(s.origin) + " capacity=" + std::to_string(s.capacity) +
        " heads=" + std::to_string(s.heads) + " tail=" + std::to_string(s.tail) +
        " compact_stage=" + std::to_string(unsigned(s.compact_stage));
}

// Host bytes immediately after construction: record planes plus the fixed
// stage/original rings and the optional precision tail. No prepared or append
// scratch exists yet, and the CPU store owns no host mirror here.
static size_t expected_host_bytes(const scenario & s, bool compact_records) {
    const size_t heads = size_t(s.heads) * 2, width = size_t(s.heads) * 256;
    const size_t groups = (uint64_t(s.origin) + s.capacity + 127) / 128;
    const size_t records = compact_records ? groups - s.origin / 128 : groups;
    return records * heads * 8960 * 2 + 2048 * width + (1536 + 4 * size_t(s.tail)) * width;
}

struct sink {
    std::string dir;
    bool compare = false;
    std::vector<std::pair<std::string, size_t>> manifest;
    std::vector<std::string> memory_lines;
    std::set<std::string> packet_names;
    size_t next = 0, memory_next = 0, packets = 0, scenarios = 0;

    sink(std::string path, bool compare_) : dir(std::move(path)), compare(compare_) {
        if (dir.empty()) return;
        if (!compare) {
            assert(!std::filesystem::exists(dir) || std::filesystem::is_empty(dir));
            std::filesystem::create_directories(dir);
            return;
        }
        assert(std::filesystem::is_directory(dir));
        std::ifstream in(dir + "/manifest.txt"); assert(in);
        std::string name; size_t bytes = 0;
        std::set<std::string> reference_names;
        while (in >> name >> bytes) {
            assert(reference_names.insert(name).second);
            manifest.emplace_back(name, bytes);
        }
        std::ifstream memory(dir + "/memory.txt"); assert(memory);
        for (std::string line; std::getline(memory, line);) memory_lines.push_back(line);
    }

    void packet(const std::string & name, const std::vector<uint8_t> & bytes) {
        assert(packet_names.insert(name).second);
        ++packets;
        if (dir.empty()) return;
        if (!compare) {
            std::ofstream out(dir + "/" + name, std::ios::binary); assert(out);
            out.write(reinterpret_cast<const char *>(bytes.data()), std::streamsize(bytes.size())); assert(out);
            manifest.emplace_back(name, bytes.size());
            return;
        }
        assert(next < manifest.size());
        assert(manifest[next].first == name && manifest[next].second == bytes.size());
        std::ifstream in(dir + "/" + name, std::ios::binary); assert(in);
        const std::vector<char> reference((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        assert(reference.size() == bytes.size());
        assert(!std::memcmp(reference.data(), bytes.data(), bytes.size()));
        ++next;
    }

    void memory(const std::string & line, size_t host_bytes) {
        ++scenarios;
        if (dir.empty()) return;
        const std::string actual = line + " host_bytes=" + std::to_string(host_bytes);
        if (!compare) { memory_lines.push_back(actual); return; }
        assert(memory_next < memory_lines.size());
        const std::string & reference = memory_lines[memory_next++];
        const size_t at = reference.find(" host_bytes=");
        assert(at != std::string::npos && reference.substr(0, at) == line);
        const long long previous = std::stoll(reference.substr(at + 12));
        std::printf("BASELINE %s reference_host_bytes=%lld new_host_bytes=%zu saved_bytes=%lld\n",
            line.c_str(), previous, host_bytes, previous - static_cast<long long>(host_bytes));
    }

    void finish() const {
        if (!dir.empty() && !compare) {
            std::ofstream out(dir + "/manifest.txt"); assert(out);
            for (const auto & entry : manifest) out << entry.first << ' ' << entry.second << '\n';
            std::ofstream memory(dir + "/memory.txt"); assert(memory);
            for (const auto & line : memory_lines) memory << line << '\n';
        }
        if (compare) assert(next == manifest.size() && memory_next == memory_lines.size());
        std::printf("%s packets=%zu scenarios=%zu dir=%s\n",
            compare ? "COMPARE byte_exact=1" : "RECORD", packets, scenarios, dir.c_str());
    }
};

// Unaligned append tails, seals, a late RESET that replays the committed
// prefix from the first host record, rewind by 128 with reappend, and a
// rejected 129-row rewind. The rewind/reappend path after the rejection must
// reproduce the earlier one packet-for-packet.
static void drive(const scenario & s, sink & out, expect_memory expect) {
    const size_t width = size_t(s.heads) * 256, heads = size_t(s.heads) * 2, record = 8960;
    const bool compact_stage = s.compact_stage && s.origin >= 128 && s.tail == 128;
    const std::string id = tag(s);
    std::vector<float> keys(size_t(s.capacity) * width), values(keys.size());
    std::vector<int32_t> positions(s.capacity);
    for (uint32_t t = 0; t < s.capacity; ++t) {
        positions[t] = int32_t(s.origin + t);
        for (size_t d = 0; d < width; ++d) {
            keys[size_t(t) * width + d] = 0.7f * std::sin(float(t * 7 + d * 3) * 0.003f);
            values[size_t(t) * width + d] = 0.9f * std::cos(float(t * 11 + d * 5) * 0.007f);
        }
    }
    rkva_kvarn_producer producer(s.origin, s.capacity, s.heads, 8, cpu_store(), s.tail, s.compact_stage);
    const size_t host_bytes = producer.host_bytes();
    if (expect != expect_memory::none) {
        assert(host_bytes == expected_host_bytes(s, expect == expect_memory::compact));
    }
    out.memory(geometry(s), host_bytes);
    uint32_t at = 0, index = 0;
    std::vector<uint8_t> packet, first_record;
    const auto append = [&](uint32_t count, bool reset) {
        assert(producer.append(keys.data() + size_t(at) * width, values.data() + size_t(at) * width,
            positions.data() + at, count, reset, packet));
        rkva_kvarn_range_upload header {};
        std::memcpy(&header, packet.data(), sizeof(header));
        assert(header.live_count == at + count);
        assert(header.append_rows == (reset ? 0u : count));
        assert(header.record_begin == (reset ? 0u : at / 128));
        assert(header.record_groups == (at + count) / 128 - header.record_begin);
        assert((header.flags & RKVA_KVARN_UPLOAD_RESET) == (reset ? RKVA_KVARN_UPLOAD_RESET : 0u));
        assert((header.flags & RKVA_KVARN_UPLOAD_OMIT_STAGE) ==
            (compact_stage ? RKVA_KVARN_UPLOAD_OMIT_STAGE : 0u));
        size_t bytes = sizeof(header) + 2 * size_t(header.record_groups) * heads * record;
        if (reset) {
            if (!compact_stage) bytes += 2 * size_t(384) * width * 2;
            bytes += 2 * size_t(std::min<uint32_t>(at + count, s.tail)) * width * 2;
        } else {
            if (!compact_stage) bytes += 2 * size_t(count) * width * 2;
            if (s.tail) bytes += 2 * size_t(count) * width * 2;
        }
        assert(packet.size() == bytes);
        // Keep the first group ever published: a RESET replay starts at the
        // first remote record even when the host sink stores a compact span.
        if (first_record.empty() && header.record_groups) {
            first_record.assign(packet.begin() + sizeof(header),
                packet.begin() + sizeof(header) + heads * record);
        }
        if (reset && !first_record.empty()) {
            assert(packet.size() >= sizeof(header) + heads * record);
            assert(!std::memcmp(packet.data() + sizeof(header), first_record.data(), heads * record));
        }
        out.packet(id + "-p" + std::to_string(index++) + ".bin", packet);
        at += count;
    };
    assert(producer.prepare(keys.data(), values.data(), positions.data(), 120));
    for (int i = 0; i < 15; ++i) append(8, at == 0);          // prepare 120, seal at 128
    assert(producer.prepare(keys.data() + size_t(at) * width, values.data() + size_t(at) * width,
        positions.data() + at, 256));
    for (int i = 0; i < 32; ++i) append(8, false);            // prepare 256
    append(1, false);                                         // unaligned prepare
    append(127, false);                                       // several closing groups
    append(9, true);                                          // late RESET: replay prefix
    assert(producer.rows() == 513);
    std::vector<std::vector<uint8_t>> disjoint_prefix;
    assert(producer.rewind(385));
    at = 385;
    for (int i = 0; i < 16; ++i) {                            // rewind 128 + reappend
        append(8, false);
        disjoint_prefix.push_back(packet);
    }
    assert(producer.rows() == 513);
    assert(!producer.rewind(384) && producer.rows() == 513);  // rejected 129: no change
    assert(producer.rewind(385));
    at = 385;
    for (size_t i = 0; i < disjoint_prefix.size(); ++i) {     // same path after the rejection
        append(8, false);
        assert(packet == disjoint_prefix[i]);
    }
    assert(producer.rows() == 513);
}

static std::vector<scenario> grid(bool gpu) {
    std::vector<scenario> result;
    for (const uint32_t origin : {0u, 128u}) {
        for (const uint32_t tail : {0u, 128u}) {
            for (const bool compact_stage : {false, true}) {
                result.push_back({origin, 640, 1, tail, compact_stage});
            }
        }
    }
    if (gpu) {
        for (const uint32_t tail : {0u, 128u}) {
            for (const bool compact_stage : {false, true}) {
                result.push_back({102400, 640, 1, tail, compact_stage});
            }
        }
        // Production width (Hkv=4) at the task origin: the old header holds
        // 800 unnecessary prefix groups, the compact span drops them.
        result.push_back({102400, 640, 4, 128, true});
    }
    result.push_back({1024, 513, 1, 128, false});          // non-multiple 128 span
    result.push_back({1024, 513, 1, 128, true});
    return result;
}

static void run_grid(sink & out, bool gpu, expect_memory expect) {
    for (const auto & s : grid(gpu)) drive(s, out, expect);
}

static void test_grid_names() {
    std::set<std::string> names;
    const auto scenarios = grid(true);
    assert(scenarios.size() == 15);
    for (const auto & s : scenarios) assert(names.insert(tag(s)).second);
    assert(names.size() == 15);
    std::puts("PASS all 15 GPU scenario names are unique, including Hkv1/Hkv4");
}

static void cpu_allocation() {
    for (const uint32_t origin : {0u, 128u, 102400u}) {
        for (const uint32_t tail : {0u, 128u}) {
            const scenario s {origin, 513, 1, tail, false};
            rkva_kvarn_producer producer(origin, s.capacity, 1, 3, cpu_store(), tail, false);
            assert(producer.host_bytes() == expected_host_bytes(s, false));
        }
    }
    std::puts("PASS CPU host records retain the global group span for the canonical store");
}

// Replay the emitted record planes against one batch store over the same
// original F32 rows; the CPU path must keep byte-identical global record IDs.
static void cpu_reference() {
    constexpr uint32_t origin = 1024, capacity = 384, heads = 1, count = 256;
    constexpr size_t width = 256, physical = 2, record = 8960;
    const size_t global_groups = (origin + capacity + 127) / 128;
    std::vector<float> keys(size_t(count) * width), values(keys.size());
    std::vector<int32_t> positions(count);
    std::vector<int64_t> indices(count);
    for (uint32_t t = 0; t < count; ++t) {
        positions[t] = origin + t; indices[t] = positions[t];
        for (size_t d = 0; d < width; ++d) {
            keys[size_t(t) * width + d] = std::sin(float(t * 7 + d * 3) * 0.003f) * 0.7f;
            values[size_t(t) * width + d] = std::cos(float(t * 11 + d * 5) * 0.007f) * 0.9f;
        }
    }
    std::vector<uint8_t> expected_k(global_groups * physical * record), expected_v(expected_k.size());
    std::vector<ggml_fp16_t> stage_k(384 * width), stage_v(stage_k.size());
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({16 * ggml_tensor_overhead(), nullptr, true}), ggml_free);
    assert(ctx);
    auto * current = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 128, physical, count);
    auto * index = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, count);
    index->data = indices.data();
    auto * stage = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, 128, physical, 384);
    auto * records = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I8, record, physical, global_groups);
    auto * kstore = ggml_kvarn_store(ctx.get(), current, index, stage, records, 4, 3, false, 3);
    auto * vstore = ggml_kvarn_store(ctx.get(), current, index, stage, records, 4, 3, true, 3);
    for (auto * op : {kstore, vstore}) {
        ggml_set_op_params_i32(op, 5, 2);
        ggml_set_op_params_i32(op, 9, 1);
    }
    current->data = keys.data(); stage->data = stage_k.data(); records->data = expected_k.data();
    assert(cpu_store()(kstore));
    current->data = values.data(); stage->data = stage_v.data(); records->data = expected_v.data();
    assert(cpu_store()(vstore));
    rkva_kvarn_producer producer(origin, capacity, heads, 3, cpu_store(), 128, true);
    std::vector<uint8_t> packet;
    for (const uint32_t begin : {0u, 128u}) {
        assert(producer.append(keys.data() + size_t(begin) * width, values.data() + size_t(begin) * width,
            positions.data() + begin, 128, begin == 0, packet));
        rkva_kvarn_range_upload header {};
        std::memcpy(&header, packet.data(), sizeof(header));
        assert(header.record_begin == begin / 128 && header.record_groups == 1);
        const size_t global = size_t(origin / 128 + header.record_begin) * physical * record;
        assert(!std::memcmp(packet.data() + sizeof(header), expected_k.data() + global, physical * record));
        assert(!std::memcmp(packet.data() + sizeof(header) + physical * record,
            expected_v.data() + global, physical * record));
    }
    std::puts("PASS CPU packets replay the canonical global record bytes");
}

int main(int argc, char ** argv) {
    test_grid_names();
    std::string record_dir, compare_dir;
    expect_memory expect = expect_memory::none;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        assert(i + 1 < argc);
        if (arg == "--record") record_dir = argv[++i];
        else if (arg == "--compare") compare_dir = argv[++i];
        else if (arg == "--expect-memory") {
            const std::string value = argv[++i];
            assert(value == "global" || value == "compact");
            expect = value == "global" ? expect_memory::global : expect_memory::compact;
        } else assert(false && "unknown argument");
    }
    assert(record_dir.empty() || compare_dir.empty());
    sink out(record_dir.empty() ? compare_dir : record_dir, !compare_dir.empty());
    if (!gpu_mode()) {
        mode("cpu");
        cpu_allocation();
        cpu_reference();
        run_grid(out, false, expect);
        out.finish();
        std::puts("PASS CPU host-record allocation, canonical bytes and packet harness");
        return 0;
    }
    ggml_backend_load_all();
    assert(ggml_backend_reg_by_name("CUDA"));
    mode("cuda");
    run_grid(out, true, expect);
    out.finish();
    mode("cpu");
    std::puts("PASS CUDA host-record compaction packets recorded/compared byte-exactly");
    return 0;
}
