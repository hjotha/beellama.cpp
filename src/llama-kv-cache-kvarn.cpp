#include "llama-kv-cache-kvarn.h"

#include "ggml-backend.h"
#include "llama-context.h"
#include "llama-hparams.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-io-file.h"
#include "llama-model.h"
#include "llama-state-q4.h"
#include "llama-kv-mixed-io.h"
#include "llama-kv-mixed-state.h"
#include "llama-kv-mixed-state-stream.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <functional>
#include <mutex>
#include <limits>
#include <map>
#include <deque>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/vfs.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

// ggml/src is not on this target's include path; only the q4_0 dequantizer is
// needed and it is part of the public ggml API surface at link time.
struct block_q4_0;
extern "C" {
GGML_API void dequantize_row_q4_0(const block_q4_0 * x, float * y, int64_t k);
}

namespace {

constexpr uint32_t MIXED_KV_STATE_ENVELOPE_MAGIC = 0x4D4B5351u; // "QSKM"
constexpr uint32_t MIXED_KV_STATE_ENVELOPE_VERSION = 1;
constexpr uint32_t MIXED_KV_STATE_METADATA_VERSION = 1;
constexpr uint64_t MIXED_KV_STATE_MAX_FRAME_BYTES = 4ull << 40;
// split_writer metadata is capped at 64 MiB; the serialized event table can
// add up to 37 MiB (2^20 events x 37 bytes), plus layer/header bookkeeping.
constexpr uint64_t MIXED_KV_STATE_MAX_MANIFEST_BYTES = 102ull << 20;
constexpr uint64_t MIXED_KV_STATE_FRAME_MARGIN_BYTES = 128ull << 20;

std::string create_conversion_temp_path(const std::string & destination);

struct mixed_state_frame_backing {
    std::string path;
    std::shared_ptr<llama_kv_mixed::kv_mixed_file_backing> mapping;

    ~mixed_state_frame_backing() {
        mapping.reset();
        if (!path.empty()) {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    }
};

std::shared_ptr<mixed_state_frame_backing> spool_mixed_state_frame(
        llama_io_read_i & io, uint64_t frame_bytes, uint64_t frame_limit) {
    if (frame_bytes < 104 || frame_bytes > frame_limit ||
            frame_bytes > MIXED_KV_STATE_MAX_FRAME_BYTES ||
            frame_bytes > uint64_t(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("mixed state frame length is outside the supported range");
    }
#ifdef _WIN32
    const char * configured_dir = std::getenv("TMPDIR");
#else
    const char * configured_dir = std::getenv("TMPDIR");
#endif
    const auto temp_dir = configured_dir && configured_dir[0]
            ? std::filesystem::absolute(std::filesystem::path(configured_dir))
            : std::filesystem::temp_directory_path();
    std::error_code fs_error;
    if (!std::filesystem::is_directory(temp_dir, fs_error) || fs_error) {
        throw std::runtime_error("mixed state spool directory is unavailable; set TMPDIR to a disk-backed directory");
    }
#if defined(__linux__)
    struct statfs fs_info = {};
    if (::statfs(temp_dir.c_str(), &fs_info) != 0) {
        throw std::runtime_error("cannot inspect mixed state spool filesystem");
    }
    // /tmp is commonly tmpfs. Spooling a multi-gigabyte prompt snapshot there
    // would create another anonymous-RAM-sized copy while looking like disk IO.
    constexpr long TMPFS_MAGIC_VALUE = 0x01021994;
    constexpr long RAMFS_MAGIC_VALUE = 0x858458f6;
    if (long(fs_info.f_type) == TMPFS_MAGIC_VALUE || long(fs_info.f_type) == RAMFS_MAGIC_VALUE) {
        throw std::runtime_error("mixed state spool directory is tmpfs/ramfs; set TMPDIR to a disk-backed directory");
    }
#endif
    const std::string destination = (temp_dir / "beellama-mixed-state").string();
    auto result = std::make_shared<mixed_state_frame_backing>();
    result->path = create_conversion_temp_path(destination);
    {
        llama_file file(result->path.c_str(), "wb");
        std::vector<uint8_t> buffer(1u << 20);
        uint64_t remaining = frame_bytes;
        while (remaining != 0) {
            const size_t chunk = size_t(std::min<uint64_t>(remaining, buffer.size()));
            io.read(buffer.data(), chunk);
            file.write_raw(buffer.data(), chunk);
            remaining -= chunk;
        }
        file.close();
    }
    result->mapping = std::make_shared<llama_kv_mixed::kv_mixed_file_backing>(result->path);
    return result;
}

uint64_t next_mixed_state_owner_id() {
    static const uint64_t process_nonce = [] {
        std::random_device random;
        uint64_t value = (uint64_t(random()) << 32) ^ uint64_t(random());
        if (value == 0) {
            value = uint64_t(random()) | 1u;
        }
        return value;
    }();
    static std::atomic<uint64_t> next{1};
    uint64_t id = next.load(std::memory_order_relaxed);
    for (;;) {
        if (id == 0 || id == UINT64_MAX) {
            throw std::overflow_error("mixed KV state owner id exhausted");
        }
        if (next.compare_exchange_weak(id, id + 1,
                std::memory_order_relaxed, std::memory_order_relaxed)) {
            // The odd multiplier is invertible modulo 2^64, so the sequence
            // is unique within this process; the random nonce separates cache
            // owners across processes that might otherwise both start at 1.
            uint64_t owner = process_nonce + id * 0x9e3779b97f4a7c15ull;
            if (owner == 0) {
                owner = process_nonce;
            }
            return owner;
        }
    }
}

std::vector<llama_kv_mixed::layer_desc> mixed_state_layer_descs(
        const llama_kv_cache_kvarn & cache,
        const std::unordered_map<uint32_t, uint64_t> & payload_sizes,
        bool partial,
        uint32_t payload_cells,
        uint64_t payload_rows) {
    using namespace llama_kv_mixed;
    std::vector<layer_desc> result;
    const auto payload_size = [&](uint32_t layer_id) {
        const auto it = payload_sizes.find(layer_id);
        return it == payload_sizes.end() ? uint64_t(0) : it->second;
    };

    for (const auto & layer : cache.kvarn_layer_layout()) {
        if (layer.record_dim_k != layer.record_dim_v ||
                layer.head_slices_k != layer.head_slices_v) {
            throw kv_mixed_error(format(
                    "mixed state layer %u has incompatible K/V KVarN record geometry", layer.layer_id));
        }
        layer_desc desc = {};
        desc.layer_id = layer.layer_id;
        desc.kind = cache_kind::kvarn;
        desc.k_type = layer.kvarn_type;
        desc.v_type = layer.kvarn_type;
        desc.k_bits = layer.key_bits;
        desc.v_bits = layer.value_bits;
        desc.kvarn_domain = uint8_t(kvarn_domain::rotated);
        desc.layout = layout_kvarn_records_stage_tail;
        desc.owner = owner_target;
        desc.tail_type = layer.tail_type == GGML_TYPE_COUNT ? 0 : uint16_t(layer.tail_type);
        desc.token_group = KVAR_N_GROUP;
        desc.record_dim = layer.record_dim_k;
        desc.head_dim_k = layer.head_dim_k;
        desc.head_dim_v = layer.head_dim_v;
        desc.head_slices = layer.head_slices_k;
        desc.n_head_kv = layer.n_head_kv;
        desc.n_stream = 1;
        desc.k_stride = layer.record_stride_k;
        desc.v_stride = layer.record_stride_v;
        desc.payload_mode = partial ? mode_partial_overlay : mode_full;
        desc.payload_cells = payload_cells;
        desc.payload_rows = payload_rows;
        desc.payload_bytes = payload_size(layer.layer_id);
        if (partial && desc.payload_bytes == 0) {
            desc.payload_mode = mode_resident_reference;
        }
        result.push_back(desc);
    }

    for (const auto & layer : cache.standard_layer_layout()) {
        layer_desc desc = {};
        desc.layer_id = layer.layer_id;
        desc.kind = cache_kind::standard_qx;
        desc.k_type = uint16_t(layer.type_k);
        desc.v_type = uint16_t(layer.type_v);
        desc.k_rot = layer.rotation_k == 0 ? uint8_t(rotation::none) : uint8_t(rotation::hadamard);
        desc.v_rot = layer.rotation_v == 0 ? uint8_t(rotation::none) : uint8_t(rotation::hadamard);
        desc.k_rot_width = layer.rotation_k;
        desc.v_rot_width = layer.rotation_v;
        desc.layout = layout_standard_qx_rows;
        desc.owner = owner_target;
        desc.v_trans = layer.v_transposed ? 1 : 0;
        desc.token_group = uint32_t(std::max<int64_t>(1, ggml_blck_size(layer.type_k)));
        desc.head_dim_k = layer.head_dim_k;
        desc.head_dim_v = layer.head_dim_v;
        desc.n_head_kv = layer.n_head_kv;
        desc.n_stream = 1;
        desc.k_stride = layer.row_stride_k;
        desc.v_stride = layer.row_stride_v;
        desc.payload_mode = partial ? mode_resident_reference : mode_full;
        desc.payload_cells = payload_cells;
        desc.payload_rows = payload_rows;
        desc.payload_bytes = partial ? 0 : payload_size(layer.layer_id);
        if (partial && payload_size(layer.layer_id) != 0) {
            throw kv_mixed_error(format(
                    "mixed partial Qx layer %u unexpectedly has serialized payload bytes", layer.layer_id));
        }
        result.push_back(desc);
    }

    std::sort(result.begin(), result.end(), [](const layer_desc & a, const layer_desc & b) {
        return a.layer_id < b.layer_id;
    });
    for (size_t i = 1; i < result.size(); ++i) {
        if (result[i - 1].layer_id == result[i].layer_id) {
            throw kv_mixed_error("mixed state has duplicate layer ownership");
        }
    }
    return result;
}

std::vector<llama_kv_mixed::cell_entry> mixed_state_cells(
        const llama_kv_cache * metadata, llama_seq_id seq_id, uint32_t & row_count) {
    using namespace llama_kv_mixed;
    if (!metadata || metadata->get_n_stream() != 1) {
        throw kv_mixed_error("mixed state requires one shared metadata stream");
    }
    if (seq_id < -1 || (seq_id >= 0 && uint32_t(seq_id) >= LLAMA_MAX_SEQ)) {
        throw kv_mixed_error("mixed state sequence id is out of range");
    }
    const auto & cells = metadata->get_cells(0);
    uint32_t used_end = 0;
    row_count = 0;
    for (uint32_t cell = 0; cell < cells.size(); ++cell) {
        const bool include = !cells.is_empty(cell) &&
                (seq_id < 0 || cells.seq_has(cell, seq_id));
        if (include) {
            used_end = cell + 1;
            ++row_count;
        }
    }
    std::vector<cell_entry> result(used_end);
    for (uint32_t cell = 0; cell < used_end; ++cell) {
        auto & entry = result[cell];
        if (cells.is_empty(cell) || (seq_id >= 0 && !cells.seq_has(cell, seq_id))) {
            continue;
        }
        entry.pos = int32_t(cells.pos_get(cell));
        const auto & ext = cells.ext_get(cell);
        entry.x = int32_t(ext.x);
        entry.y = int32_t(ext.y);
        entry.tok = int32_t(ext.tok);
        for (llama_seq_id seq = 0; uint32_t(seq) < LLAMA_MAX_SEQ; ++seq) {
            if (cells.seq_has(cell, seq) && (seq_id < 0 || seq == seq_id)) {
                entry.seq_ids.push_back(int32_t(seq));
            }
        }
    }
    return result;
}

using backend_kvarn_capabilities_t = bool (*)(
        ggml_backend_dev_t,
        ggml_backend_kvarn_capabilities *);

static backend_kvarn_capabilities_t kvarn_capabilities_proc(ggml_backend_dev_t dev) {
    auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    return reg ? reinterpret_cast<backend_kvarn_capabilities_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_kvarn_capabilities")) : nullptr;
}

static bool query_kvarn_capabilities(
        ggml_backend_dev_t dev,
        backend_kvarn_capabilities_t fn,
        ggml_backend_kvarn_capabilities & capabilities) {
    if (fn == nullptr) {
        return false;
    }
    capabilities = {};
    capabilities.struct_size = sizeof(capabilities);
    capabilities.abi_version = GGML_BACKEND_KVARN_CAPABILITIES_ABI_VERSION;
    return fn(dev, &capabilities) &&
        capabilities.struct_size == sizeof(capabilities) &&
        capabilities.abi_version == GGML_BACKEND_KVARN_CAPABILITIES_ABI_VERSION;
}

using backend_kv_tail_attention_supported_t = bool (*)(
        ggml_type, ggml_type, ggml_type, ggml_type, int64_t, int64_t);
using backend_kvarn_tail_attention_supported_t = bool (*)(
        ggml_backend_dev_t,
        ggml_type, ggml_type, ggml_type, ggml_type, int64_t, int64_t);

bool kvarn_backend_supports_native_tail(
        ggml_backend_dev_t dev, ggml_type exact_type, int64_t d_k, int64_t d_v) {
    if (dev == nullptr) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    if (ggml_backend_dev_is_meta(dev)) {
        const size_t count = ggml_backend_meta_device_count(dev);
        for (size_t i = 0; i < count; ++i) {
            if (!kvarn_backend_supports_native_tail(
                        ggml_backend_meta_device_get(dev, i), exact_type, d_k, d_v)) {
                return false;
            }
        }
        return count > 0;
    }
    auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    auto * kvarn_fn = reg ? reinterpret_cast<backend_kvarn_tail_attention_supported_t>(
            ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_kvarn_tail_attention_supported")) : nullptr;
    if (kvarn_fn) {
        return kvarn_fn(dev, GGML_TYPE_F16, GGML_TYPE_F16,
            exact_type, exact_type, d_k, d_v);
    }
    auto * segmented_fn = reg ? reinterpret_cast<backend_kv_tail_attention_supported_t>(
            ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_kv_tail_segmented_attention_supported")) : nullptr;
    if (!segmented_fn || !segmented_fn(
            GGML_TYPE_F16, GGML_TYPE_F16, exact_type, exact_type, d_k, d_v)) {
        return false;
    }
    auto * fn = reg ? reinterpret_cast<backend_kv_tail_attention_supported_t>(
            ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_kv_tail_attention_supported")) : nullptr;
    return fn && fn(GGML_TYPE_F16, GGML_TYPE_F16, exact_type, exact_type, d_k, d_v);
}

using backend_kvarn_convert_q4_t = bool (*)(
        const void *, size_t, int, int, int, int, int, int, bool, int, void *);

// Opt-out switch for A/B validation of the GPU converter (default: enabled).
bool kvarn_convert_gpu_enabled() {
    static const bool enabled = [] {
        const char * raw = std::getenv("LLAMA_KVARN_CONVERT_GPU");
        return raw == nullptr || std::atoi(raw) != 0;
    }();
    return enabled;
}

// First backend device exposing the CUDA/HIP bulk q4_0 -> KVarN converter.
backend_kvarn_convert_q4_t kvarn_convert_gpu_proc() {
    static backend_kvarn_convert_q4_t proc = [] {
        const size_t count = ggml_backend_dev_count();
        for (size_t i = 0; i < count; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            auto * fn = reg ? reinterpret_cast<backend_kvarn_convert_q4_t>(
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_kvarn_convert_q4_v2")) : nullptr;
            if (fn != nullptr) {
                return fn;
            }
        }
        return (backend_kvarn_convert_q4_t) nullptr;
    }();
    return proc;
}

bool kvarn_backend_supports_tail_write(
        ggml_backend_dev_t dev, ggml_type exact_type, int64_t n_embd) {
    if (!dev) {
        return false;
    }
    if (ggml_backend_dev_is_meta(dev)) {
        const size_t count = ggml_backend_meta_device_count(dev);
        for (size_t i = 0; i < count; ++i) {
            if (!kvarn_backend_supports_tail_write(
                        ggml_backend_meta_device_get(dev, i), exact_type, n_embd)) {
                return false;
            }
        }
        return count > 0;
    }
    ggml_init_params params = {
        /*.mem_size   =*/ 16*ggml_tensor_overhead() + 4096,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(params) };
    if (!ctx) {
        throw std::runtime_error("failed to create KVarN exact-tail capability context");
    }
    auto * dst = ggml_new_tensor_2d(ctx.get(), exact_type, n_embd, 16);
    auto * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, 1);
    auto * idx = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, 1);
    return ggml_backend_dev_supports_op(dev, ggml_set_rows(ctx.get(), dst, src, idx));
}

// SWA keeps only local tail groups in F16; older window groups are served from
// records. Keep this low enough that KVarN remains a KV-memory win over q5_0.
constexpr uint32_t KVAR_N_SWA_TAIL_GROUPS = 2;
constexpr uint32_t KVAR_N_STATE_MAGIC = 0x4e52564b; // "KVRN"
constexpr uint32_t KVAR_N_MIXED_STATE_MAGIC = 0x584d564b; // "KVMX"
constexpr uint32_t KVAR_N_MIXED_STATE_VERSION = 1;
// Version 16 stores full unified non-SWA stages as source-cell rows so state
// can remap across contexts with different sequence-dependent stage depths.
// Version 15 adds self-contained selective record groups with cell remapping.
// Version 14 stores selective per-sequence stage rows by logical source cell.
// Version 13 stores exact-tail payloads component-major in contiguous physical
// slot runs. Version 12 stores canonical exact-tail payloads interleaved by row
// and remaps SWA record rings across physical ubatch layouts. Version 11: tail_groups is explicit and SWA
// stages no longer allocate a non-existent sink slot. Version 9: D256/D512 records use the full logical-head Hadamard instead of
// independent 128-wide slice rotations. Version 8: KVarN K/V stage rows and compressed records are all
// rotated-domain. Older states are rejected because their staged V rows may
// otherwise be restored in the wrong domain. Version 5 added stage_groups
// validation. Version 10 rejects states with the pre-dedup SWA record-ring layout.
constexpr uint32_t KVAR_N_STATE_VERSION_MIN = 12;
constexpr uint32_t KVAR_N_STATE_VERSION = 16;
constexpr uint32_t KVAR_N_STATE_RECORDS_FULL = 0;
constexpr uint32_t KVAR_N_STATE_STAGE_ONLY_PARTIAL = 1;
constexpr uint32_t KVAR_N_STATE_RECORDS_SELECTIVE = 2;
constexpr uint32_t KVAR_N_STATE_RECORDS_FULL_REMAP_STAGE = 3;

} // namespace

bool llama_kvarn_backend_supports_native_ops(ggml_backend_dev_t dev) {
    if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return true; // the built-in CPU backend consumes KVarN views directly
    }

    if (ggml_backend_dev_is_meta(dev)) {
        const size_t count = ggml_backend_meta_device_count(dev);
        for (size_t i = 0; i < count; ++i) {
            if (!llama_kvarn_backend_supports_native_ops(
                        ggml_backend_meta_device_get(dev, i))) {
                return false;
            }
        }
        return count > 0;
    }
    if (auto * capabilities_fn = kvarn_capabilities_proc(dev)) {
        ggml_backend_kvarn_capabilities capabilities = {};
        return query_kvarn_capabilities(dev, capabilities_fn, capabilities) &&
            (capabilities.portable_direct_body ||
             capabilities.specialized_generic_mma ||
             capabilities.specialized_decode_split ||
             capabilities.specialized_decode_vector);
    }
    using ggml_backend_kvarn_native_ops_t = bool (*)(ggml_backend_dev_t dev);
    auto * reg = ggml_backend_dev_backend_reg(dev);
    auto * fn = reg ? (ggml_backend_kvarn_native_ops_t) ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_kvarn_native_ops") : nullptr;
    return fn != nullptr && fn(dev);
}

bool llama_kvarn_backend_native_attention_uses_original_v(ggml_backend_dev_t dev) {
    if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return false;
    }

    if (ggml_backend_dev_is_meta(dev)) {
        const size_t count = ggml_backend_meta_device_count(dev);
        for (size_t i = 0; i < count; ++i) {
            if (!llama_kvarn_backend_native_attention_uses_original_v(
                        ggml_backend_meta_device_get(dev, i))) {
                return false;
            }
        }
        return count > 0;
    }
    if (auto * capabilities_fn = kvarn_capabilities_proc(dev)) {
        ggml_backend_kvarn_capabilities capabilities = {};
        return query_kvarn_capabilities(dev, capabilities_fn, capabilities) &&
            capabilities.original_v_domain;
    }
    using ggml_backend_kvarn_native_original_v_t = bool (*)(ggml_backend_dev_t dev);
    auto * reg = ggml_backend_dev_backend_reg(dev);
    auto * fn = reg ? (ggml_backend_kvarn_native_original_v_t) ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_kvarn_native_original_v") : nullptr;
    return fn != nullptr && fn(dev);
}

uint32_t llama_kvarn_backend_native_rotated_max_query_tokens(ggml_backend_dev_t dev) {
    if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return 0;
    }

    if (ggml_backend_dev_is_meta(dev)) {
        const size_t count = ggml_backend_meta_device_count(dev);
        uint32_t result = UINT32_MAX;
        for (size_t i = 0; i < count; ++i) {
            result = std::min(result, llama_kvarn_backend_native_rotated_max_query_tokens(
                    ggml_backend_meta_device_get(dev, i)));
        }
        return count > 0 ? result : 0;
    }
    if (auto * capabilities_fn = kvarn_capabilities_proc(dev)) {
        ggml_backend_kvarn_capabilities capabilities = {};
        if (!query_kvarn_capabilities(dev, capabilities_fn, capabilities)) {
            return 0;
        }
        const bool specialized = capabilities.original_v_domain &&
            (capabilities.specialized_generic_mma ||
             capabilities.specialized_decode_split ||
             capabilities.specialized_decode_vector);
        return specialized ?
            capabilities.rotated_query_max_specialized :
            capabilities.rotated_query_max_portable;
    }
    using ggml_backend_kvarn_native_rotated_max_query_tokens_t = uint32_t (*)(ggml_backend_dev_t dev);
    auto * reg = ggml_backend_dev_backend_reg(dev);
    auto * fn = reg ? (ggml_backend_kvarn_native_rotated_max_query_tokens_t)
        ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_kvarn_native_rotated_max_query_tokens") : nullptr;
    return fn != nullptr ? fn(dev) : 0;
}

bool llama_kvarn_backend_mixed_tail_native_preferred(ggml_backend_dev_t dev) {
    if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return true;
    }

    if (ggml_backend_dev_is_meta(dev)) {
        const size_t count = ggml_backend_meta_device_count(dev);
        for (size_t i = 0; i < count; ++i) {
            if (!llama_kvarn_backend_mixed_tail_native_preferred(
                        ggml_backend_meta_device_get(dev, i))) {
                return false;
            }
        }
        return count > 0;
    }
    using ggml_backend_kvarn_mixed_tail_native_preferred_t = bool (*)(
        ggml_backend_dev_t dev);
    auto * reg = ggml_backend_dev_backend_reg(dev);
    auto * fn = reg ? (ggml_backend_kvarn_mixed_tail_native_preferred_t)
        ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_kvarn_mixed_tail_native_preferred") : nullptr;
    return fn == nullptr || fn(dev);
}

bool llama_kvarn_backend_supports_ops(ggml_backend_dev_t dev, int head_dim) {
    uint32_t head_dim_bit = 0;
    switch (head_dim) {
        case  64: head_dim_bit = GGML_BACKEND_KVARN_HEAD_DIM_64;  break;
        case 128: head_dim_bit = GGML_BACKEND_KVARN_HEAD_DIM_128; break;
        case 256: head_dim_bit = GGML_BACKEND_KVARN_HEAD_DIM_256; break;
        case 512: head_dim_bit = GGML_BACKEND_KVARN_HEAD_DIM_512; break;
        default: return false;
    }
    if (dev == nullptr) {
        return true; // the built-in CPU backend implements every admitted geometry
    }

    if (ggml_backend_dev_is_meta(dev)) {
        const size_t count = ggml_backend_meta_device_count(dev);
        for (size_t i = 0; i < count; ++i) {
            if (!llama_kvarn_backend_supports_ops(
                        ggml_backend_meta_device_get(dev, i), head_dim)) {
                return false;
            }
        }
        return count > 0;
    }
    if (auto * capabilities_fn = kvarn_capabilities_proc(dev)) {
        ggml_backend_kvarn_capabilities capabilities = {};
        return query_kvarn_capabilities(dev, capabilities_fn, capabilities) &&
            capabilities.store_materialize &&
            (capabilities.supported_head_dims & head_dim_bit) != 0;
    }
    // Legacy boolean-only backends predate rectangular records. Preserve their
    // legacy geometry support, but never infer D64 support from that boolean.
    if (head_dim == 64) {
        return false;
    }
    using ggml_backend_kvarn_ops_t = bool (*)(ggml_backend_dev_t dev);
    auto * reg = ggml_backend_dev_backend_reg(dev);
    auto * fn = reg ? (ggml_backend_kvarn_ops_t) ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_kvarn_ops") : nullptr;
    return fn != nullptr && fn(dev);
}

namespace {

size_t kvarn_record_bytes(int record_dim, int bits, bool value) {
    return llama_kvarn_make_record_layout(record_dim, bits, value).record_bytes;
}

void write_kvarn_tensor(llama_io_write_i & io, ggml_tensor * tensor) {
    const uint64_t size = ggml_nbytes(tensor);
    io.write(&size, sizeof(size));
    io.write_tensor(tensor, 0, size);
}

void write_kvarn_tensor_slice(llama_io_write_i & io, ggml_tensor * tensor, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= (size_t) ggml_nbytes(tensor));
    const uint64_t size64 = size;
    io.write(&size64, sizeof(size64));
    io.write_tensor(tensor, offset, size);
}

void read_kvarn_tensor(llama_io_read_i & io, ggml_tensor * tensor) {
    uint64_t size;
    io.read(&size, sizeof(size));
    if (size != (uint64_t) ggml_nbytes(tensor)) {
        throw std::runtime_error("mismatched KVarN cache tensor size");
    }
    io.read_tensor(tensor, 0, size);
}

void read_kvarn_tensor_slice(llama_io_read_i & io, ggml_tensor * tensor, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= (size_t) ggml_nbytes(tensor));
    uint64_t saved_size;
    io.read(&saved_size, sizeof(saved_size));
    if (saved_size != size) {
        throw std::runtime_error("mismatched KVarN cache tensor slice size");
    }
    io.read_tensor(tensor, offset, size);
}

void read_kvarn_swa_records(
        llama_io_read_i & io,
        ggml_tensor * tensor,
        uint32_t saved_groups,
        uint32_t current_groups,
        llama_pos saved_pos_max,
        bool on_device) {
    const size_t group_bytes = tensor->nb[2];
    uint64_t saved_size;
    io.read(&saved_size, sizeof(saved_size));
    if (saved_size != uint64_t(saved_groups)*group_bytes) {
        throw std::runtime_error("mismatched KVarN SWA record tensor size");
    }
    if (on_device) {
        if (saved_groups != current_groups) {
            throw std::runtime_error("on-device KVarN SWA state cannot remap record-ring depth");
        }
        io.read_tensor(tensor, 0, saved_size);
        return;
    }
    std::vector<uint8_t> saved(saved_size);
    if (!saved.empty()) {
        io.read(saved.data(), saved.size());
    }
    io.stage_tensor_clear(tensor, 0, ggml_nbytes(tensor));
    if (saved_pos_max < llama_pos(KVAR_N_GROUP - 1)) {
        return;
    }

    const int64_t last_complete = (int64_t(saved_pos_max) + 1) / KVAR_N_GROUP - 1;
    const int64_t first_saved = std::max<int64_t>(0, last_complete - int64_t(saved_groups) + 1);
    const int64_t first_current = std::max<int64_t>(first_saved, last_complete - int64_t(current_groups) + 1);
    for (int64_t group = first_current; group <= last_complete; ++group) {
        const uint32_t src = uint32_t(group % saved_groups);
        const uint32_t dst = uint32_t(group % current_groups);
        io.stage_tensor_set(
                tensor,
                saved.data() + size_t(src)*group_bytes,
                size_t(dst)*group_bytes,
                group_bytes);
    }
}

void zero_kvarn_tensor_range(llama_io_read_i & io, ggml_tensor * tensor, size_t offset, size_t size) {
    if (size == 0) {
        return;
    }
    io.stage_tensor_clear(tensor, offset, size);
}

struct kvarn_tail_tensor_span {
    size_t offset;
    size_t size;
};

kvarn_tail_tensor_span kvarn_tail_checked_span(
        ggml_tensor * tensor, int32_t slot_begin, uint32_t length, uint64_t row_size) {
    if (!tensor || slot_begin < 0 || length == 0 || row_size == 0) {
        throw std::runtime_error(format(
            "invalid KVarN exact-tail tensor span (tensor=%s type=%s ne0=%lld slot=%d length=%u row=%llu)",
            tensor ? tensor->name : "null", tensor ? ggml_type_name(tensor->type) : "none",
            tensor ? (long long) tensor->ne[0] : -1LL, slot_begin, length,
            (unsigned long long) row_size));
    }
    const uint64_t slot = uint64_t(slot_begin);
    if (slot > uint64_t(std::numeric_limits<size_t>::max())/row_size ||
            uint64_t(length) > uint64_t(std::numeric_limits<size_t>::max())/row_size) {
        throw std::overflow_error("KVarN exact-tail tensor span overflows size_t");
    }
    const size_t offset = size_t(slot*row_size);
    const size_t size = size_t(uint64_t(length)*row_size);
    if (offset > size_t(ggml_nbytes(tensor)) || size > size_t(ggml_nbytes(tensor)) - offset) {
        throw std::runtime_error("KVarN exact-tail tensor span exceeds its allocation");
    }
    return { offset, size };
}

uint64_t kvarn_tail_checked_bytes(uint32_t payloads, uint64_t row_size) {
    if (row_size != 0 && uint64_t(payloads) > std::numeric_limits<uint64_t>::max()/row_size) {
        throw std::overflow_error("KVarN exact-tail state byte count overflows uint64_t");
    }
    return uint64_t(payloads)*row_size;
}

void kvarn_tail_add_bytes(uint64_t & total, uint64_t bytes) {
    if (bytes > std::numeric_limits<uint64_t>::max() - total) {
        throw std::overflow_error("KVarN exact-tail state byte total overflows uint64_t");
    }
    total += bytes;
}

size_t read_kvarn_exact_tail_v12_interleaved(
        llama_io_read_i & io,
        ggml_tensor * k_tail,
        ggml_tensor * v_tail,
        uint64_t k_tail_row,
        uint64_t v_tail_row,
        const std::vector<std::vector<int32_t>> & destinations,
        bool on_device) {
    size_t tensor_ops = 0;
    for (const auto & payload_destinations : destinations) {
        if (k_tail) {
            if (on_device) {
                const auto span = kvarn_tail_checked_span(k_tail, payload_destinations[0], 1, k_tail_row);
                io.read_tensor(k_tail, span.offset, span.size);
                ++tensor_ops;
            } else {
                std::vector<uint8_t> row(static_cast<size_t>(k_tail_row));
                io.read(row.data(), row.size());
                for (const int32_t slot : payload_destinations) {
                    const auto span = kvarn_tail_checked_span(k_tail, slot, 1, k_tail_row);
                    io.stage_tensor_set(k_tail, row.data(), span.offset, span.size);
                    ++tensor_ops;
                }
            }
        }
        if (v_tail) {
            if (on_device) {
                const auto span = kvarn_tail_checked_span(v_tail, payload_destinations[0], 1, v_tail_row);
                io.read_tensor(v_tail, span.offset, span.size);
                ++tensor_ops;
            } else {
                std::vector<uint8_t> row(static_cast<size_t>(v_tail_row));
                io.read(row.data(), row.size());
                for (const int32_t slot : payload_destinations) {
                    const auto span = kvarn_tail_checked_span(v_tail, slot, 1, v_tail_row);
                    io.stage_tensor_set(v_tail, row.data(), span.offset, span.size);
                    ++tensor_ops;
                }
            }
        }
    }
    return tensor_ops;
}

size_t read_kvarn_exact_tail_v13_component(
        llama_io_read_i & io,
        ggml_tensor * tensor,
        uint64_t row_size,
        const std::vector<std::vector<int32_t>> & destinations,
        bool on_device) {
    if (!tensor) {
        if (row_size != 0) {
            throw std::runtime_error("KVarN exact-tail state has a row for an absent tensor");
        }
        return 0;
    }

    const uint64_t expected_bytes = kvarn_tail_checked_bytes(uint32_t(destinations.size()), row_size);
    if (expected_bytes > uint64_t(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("KVarN exact-tail component exceeds size_t");
    }
    const size_t begin = io.n_bytes();
    size_t tensor_ops = 0;
    if (on_device) {
        // Device state buffers retain one tensor per write operation. Keep a
        // canonical row topology so a wrapped source can restore into a
        // destination with different physical run boundaries.
        for (const auto & payload_destinations : destinations) {
            if (payload_destinations.size() != 1) {
                throw std::runtime_error("on-device KVarN exact-tail state requires one destination slot per payload");
            }
            const auto span = kvarn_tail_checked_span(tensor, payload_destinations[0], 1, row_size);
            io.read_tensor(tensor, span.offset, span.size);
            ++tensor_ops;
        }
        return tensor_ops;
    }

    for (size_t payload = 0; payload < destinations.size();) {
        if (destinations[payload].size() == 1) {
            uint32_t length = 1;
            while (payload + length < destinations.size() &&
                    destinations[payload + length].size() == 1 &&
                    int64_t(destinations[payload + length][0]) ==
                            int64_t(destinations[payload + length - 1][0]) + 1) {
                ++length;
            }
            const auto span = kvarn_tail_checked_span(tensor, destinations[payload][0], length, row_size);
            io.read_tensor(tensor, span.offset, span.size);
            ++tensor_ops;
            payload += length;
            continue;
        }

        std::vector<uint8_t> row(static_cast<size_t>(row_size));
        io.read(row.data(), row.size());
        for (const int32_t slot : destinations[payload]) {
            const auto span = kvarn_tail_checked_span(tensor, slot, 1, row_size);
            io.stage_tensor_set(tensor, row.data(), span.offset, span.size);
            ++tensor_ops;
        }
        ++payload;
    }

    if (io.n_bytes() < begin || io.n_bytes() - begin != size_t(expected_bytes)) {
        throw std::runtime_error("KVarN exact-tail component byte count mismatch");
    }
    return tensor_ops;
}

size_t kvarn_exact_tail_destination_runs(
        const std::vector<std::vector<int32_t>> & destinations) {
    size_t runs = 0;
    for (size_t payload = 0; payload < destinations.size();) {
        if (destinations[payload].size() != 1) {
            runs += destinations[payload].size();
            ++payload;
            continue;
        }
        ++runs;
        do {
            ++payload;
        } while (payload < destinations.size() && destinations[payload].size() == 1 &&
                int64_t(destinations[payload][0]) == int64_t(destinations[payload - 1][0]) + 1);
    }
    return runs;
}

int32_t kvarn_workspace_tokens_per_stream_hint(const llama_kv_cache::slot_info & sinfo) {
    if (sinfo.empty() || sinfo.idxs.empty() || sinfo.idxs[0].empty()) {
        return 0;
    }

    const size_t n_tokens = sinfo.idxs[0].size();
    if (n_tokens > (size_t) std::numeric_limits<int32_t>::max()) {
        return 0;
    }

    for (const auto & idxs : sinfo.idxs) {
        if (idxs.size() != n_tokens || idxs.empty()) {
            return 0;
        }
        for (size_t i = 1; i < idxs.size(); ++i) {
            const uint32_t prev = idxs[i - 1];
            const uint32_t cur = idxs[i];
            if (cur != prev + 1u &&
                    (cur <= prev || prev % KVAR_N_GROUP != KVAR_N_GROUP - 1u || cur % KVAR_N_GROUP != 0u)) {
                return 0;
            }
        }
    }

    return (int32_t) n_tokens;
}

void kvarn_gen_hadamard(std::vector<float> & data, int n) {
    GGML_ASSERT(n == 64 || n == 128 || n == 256 || n == 512);
    data.assign(n * n, 0.0f);
    data[0] = 1.0f / std::sqrt(float(n));

    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; ++i) {
            for (int j = 0; j < s; ++j) {
                const float val = data[i * n + j];

                data[(i + s) * n + j]       =  val;
                data[i * n + (j + s)]       =  val;
                data[(i + s) * n + (j + s)] = -val;
            }
        }
    }
}

const std::vector<float> & kvarn_hadamard(int n) {
    static const std::vector<float> h64 = [] {
        std::vector<float> result;
        kvarn_gen_hadamard(result, 64);
        return result;
    }();
    static const std::vector<float> h128 = [] {
        std::vector<float> result;
        kvarn_gen_hadamard(result, 128);
        return result;
    }();
    static const std::vector<float> h256 = [] {
        std::vector<float> result;
        kvarn_gen_hadamard(result, 256);
        return result;
    }();
    static const std::vector<float> h512 = [] {
        std::vector<float> result;
        kvarn_gen_hadamard(result, 512);
        return result;
    }();

    switch (n) {
        case  64: return h64;
        case 128: return h128;
        case 256: return h256;
        case 512: return h512;
        default:  GGML_ABORT("unsupported KVarN Hadamard width");
    }
}

uint32_t kvarn_stage_tail_groups(
        uint32_t n_batch, uint32_t n_ubatch, bool is_swa, uint32_t n_seq_max) {
    if (is_swa) {
        return KVAR_N_SWA_TAIL_GROUPS;
    }

    return llama_kvarn_non_swa_tail_groups(n_batch, n_ubatch)*std::max(1u, n_seq_max);
}

uint32_t kvarn_swa_visible_groups(uint32_t kv_size, uint32_t n_swa) {
    const uint32_t window_cells = n_swa > 0 ? std::min(kv_size, n_swa) : kv_size;
    return ((window_cells + KVAR_N_GROUP - 1u) / KVAR_N_GROUP) + 1u;
}

uint32_t kvarn_record_groups_per_stream(uint32_t kv_size, uint32_t n_ubatch, uint32_t n_swa, bool is_swa, uint32_t tail_groups) {
    if (!is_swa) {
        return (kv_size + KVAR_N_GROUP - 1u) / KVAR_N_GROUP;
    }

    GGML_UNUSED(tail_groups);
    const uint32_t visible_groups = kvarn_swa_visible_groups(kv_size, n_swa);
    const uint32_t in_flight_groups = std::max<uint32_t>(1u, (n_ubatch + KVAR_N_GROUP - 1u) / KVAR_N_GROUP);
    return std::max<uint32_t>(1u, visible_groups + in_flight_groups - 1u);
}

} // namespace

llama_kv_cache_kvarn_context::llama_kv_cache_kvarn_context(
        llama_kv_cache_kvarn * cache,
        llama_memory_context_ptr base,
        llama_context * update_lctx,
        std::vector<int32_t> shared_graph_layers) :
    llama_kv_cache_context(base ? base->get_status() : LLAMA_MEMORY_STATUS_FAILED_PREPARE),
    cache(cache),
    base_ctx(std::move(base)),
    shared_graph_layers(std::move(shared_graph_layers)),
    update_lctx(update_lctx) {
}

llama_kv_cache_context * llama_kv_cache_kvarn_context::base() const {
    return static_cast<llama_kv_cache_context *>(base_ctx.get());
}

int32_t llama_kv_cache_kvarn_context::graph_layer_for(int32_t il) const {
    if (shared_graph_layers.empty()) {
        return il;
    }
    const int32_t shared = shared_graph_layers.at(il);
    GGML_ASSERT(shared >= 0);
    return shared;
}

bool llama_kv_cache_kvarn_context::next() {
    compact_read_plan_cache.clear();
    return base()->next();
}

bool llama_kv_cache_kvarn_context::apply() {
    compact_read_plan_cache.clear();
    if (!base()->apply()) {
        return false;
    }

    return !update_lctx || cache->apply_pending_stream_copies(update_lctx);
}

void llama_kv_cache_kvarn_context::graph_compute_start() {
    base()->graph_compute_start();
}

void llama_kv_cache_kvarn_context::graph_compute_finish(ggml_status compute_status) {
    base()->graph_compute_finish(compute_status);
}

void llama_kv_cache_kvarn_context::graph_compute_complete(
        ggml_backend_sched_t sched, ggml_status compute_status) {
    if (compute_status == GGML_STATUS_SUCCESS && !shared_graph_layers.empty()) {
        return;
    }
    if (compute_status == GGML_STATUS_SUCCESS) {
        cache->enqueue_prefill_migration(current_sinfo(), sched);
    }
}

llama_memory_status llama_kv_cache_kvarn_context::get_status() const {
    const auto status = base_ctx ? base_ctx->get_status() : LLAMA_MEMORY_STATUS_FAILED_PREPARE;
    if (status == LLAMA_MEMORY_STATUS_NO_UPDATE && cache->has_pending_stream_copies()) {
        return LLAMA_MEMORY_STATUS_SUCCESS;
    }
    return status;
}

const llama_ubatch & llama_kv_cache_kvarn_context::get_ubatch() const {
    return base()->get_ubatch();
}

uint32_t llama_kv_cache_kvarn_context::get_n_kv() const {
    return uses_compact_read_indices() ?
            uint32_t(compact_read_plan().size()) : base()->get_n_kv();
}

bool llama_kv_cache_kvarn_context::uses_compact_read_indices() const {
    return (!shared_graph_layers.empty() && !cache->is_swa() && cache->get_kv_n_stream() == 1) ||
            cache->uses_compact_read_indices();
}

bool llama_kv_cache_kvarn_context::uses_shared_live_indices() const {
    return !shared_graph_layers.empty() && !cache->is_swa() && cache->get_kv_n_stream() > 1;
}

bool llama_kv_cache_kvarn_context::uses_materialization_indices() const {
    return uses_compact_read_indices() || uses_shared_live_indices();
}

const std::vector<int64_t> & llama_kv_cache_kvarn_context::compact_read_plan() const {
    GGML_ASSERT(uses_compact_read_indices());
    if (!compact_read_plan_cache.empty()) {
        return compact_read_plan_cache;
    }
    const auto * kv = cache->get_metadata_cache();
    const auto & cells = kv->get_cells(0);
    uint32_t scan_end = std::min<uint32_t>(cells.size(), base()->get_n_kv());
    if (shared_graph_layers.empty() && !current_sinfo().empty()) {
        GGML_ASSERT(current_sinfo().n_stream() == 1);
        for (const uint32_t cell : current_sinfo().idxs[0]) {
            scan_end = std::max(scan_end, cell + 1u);
        }
    }
    std::vector<std::pair<llama_pos, uint32_t>> ordered;
    ordered.reserve(cells.get_used());
    for (uint32_t cell = 0; cell < scan_end; ++cell) {
        if (!cells.is_empty(cell)) {
            ordered.emplace_back(cells.pos_get(cell), cell);
        }
    }
    std::stable_sort(ordered.begin(), ordered.end(), [](const auto & a, const auto & b) {
        return a.first < b.first || (a.first == b.first && a.second < b.second);
    });
    std::vector<uint32_t> occupied;
    occupied.reserve(ordered.size());
    for (const auto & entry : ordered) {
        occupied.push_back(entry.second);
    }
    std::vector<uint32_t> pending;
    if (!current_sinfo().empty()) {
        GGML_ASSERT(current_sinfo().n_stream() == 1);
        pending.assign(current_sinfo().idxs[0].begin(), current_sinfo().idxs[0].end());
    }
    compact_read_plan_cache = llama_kvarn_compact_read_plan(
            occupied, pending, cells.size(), 256, KVAR_N_GROUP);
    return compact_read_plan_cache;
}

std::vector<int64_t> llama_kv_cache_kvarn_context::shared_live_indices() const {
    GGML_ASSERT(uses_shared_live_indices());
    // Shared MTP layers have no K/V projections, and the metadata view does not
    // apply its pending ubatch. Only committed target cells have backing KVarN
    // stage or record data, so auxiliary slot reservations must not advance
    // these per-stream live boundaries.
    const auto * metadata = cache->get_metadata_cache();
    const uint32_t n_stream = cache->get_kv_n_stream();
    const uint32_t kv_size = cache->get_kv_size();
    std::vector<int64_t> result(n_stream, -1);
    for (uint32_t stream = 0; stream < n_stream; ++stream) {
        const auto & cells = metadata->get_cells(stream);
        for (uint32_t cell = uint32_t(cells.size()); cell-- > 0;) {
            if (!cells.is_empty(cell)) {
                result[stream] = int64_t(stream)*kv_size + cell;
                break;
            }
        }
    }
    return result;
}

llama_kv_cache * llama_kv_cache_kvarn_context::get_kv() const {
    return cache->get_metadata_cache();
}

const llama_kv_cache::slot_info & llama_kv_cache_kvarn_context::current_sinfo() const {
    return base()->current_sinfo();
}

const llama_kv_cache::slot_info_vec_t & llama_kv_cache_kvarn_context::get_sinfos() const {
    return base()->get_sinfos();
}

void llama_kv_cache_kvarn_context::get_prev_tokens(
        const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {
    base()->get_prev_tokens(ubatch, n, res);
}

ggml_type llama_kv_cache_kvarn_context::type_k() const {
    return GGML_TYPE_F16;
}

ggml_type llama_kv_cache_kvarn_context::type_v() const {
    return GGML_TYPE_F16;
}

ggml_tensor * llama_kv_cache_kvarn_context::get_k(ggml_context * ctx, int32_t il) const {
    return get_k_for_attention(ctx, il, uses_native_attention(il));
}

ggml_tensor * llama_kv_cache_kvarn_context::get_k_for_attention(
        ggml_context * ctx, int32_t il, bool native_attention) const {
    if (uses_standard_layer(il)) {
        return cache->standard_get_k(ctx, il, get_n_kv(), current_sinfo());
    }
    const int32_t shared_il = graph_layer_for(il);
    const auto it = stored_k.find(cache->mapped_layer_id(shared_il));
    ggml_tensor * stored = it != stored_k.end() ? it->second :
            cache->get_materialization_source(shared_il, false);
    GGML_ASSERT(stored != nullptr);
    return native_attention ? get_k_native(ctx, il) :
        cache->materialize(ctx, stored, shared_il, get_n_kv(), current_sinfo(), false,
                mat_idxs, uses_compact_read_indices());
}

ggml_tensor * llama_kv_cache_kvarn_context::get_v(ggml_context * ctx, int32_t il) const {
    return get_v_for_attention(ctx, il, uses_native_attention(il));
}

ggml_tensor * llama_kv_cache_kvarn_context::get_v_for_attention(
        ggml_context * ctx, int32_t il, bool native_attention) const {
    if (uses_standard_layer(il)) {
        return cache->standard_get_v(ctx, il, get_n_kv(), current_sinfo());
    }
    const int32_t shared_il = graph_layer_for(il);
    const auto it = stored_v.find(cache->mapped_layer_id(shared_il));
    ggml_tensor * stored = it != stored_v.end() ? it->second :
            cache->get_materialization_source(shared_il, true);
    GGML_ASSERT(stored != nullptr);
    return native_attention ? get_v_native(ctx, il) :
        cache->materialize(ctx, stored, shared_il, get_n_kv(), current_sinfo(), true,
                mat_idxs, uses_compact_read_indices());
}

ggml_tensor * llama_kv_cache_kvarn_context::get_k_tail(ggml_context * ctx, int32_t il) const {
    if (uses_standard_layer(il)) {
        GGML_UNUSED(ctx);
        return nullptr;
    }
    return shared_graph_layers.empty() ? cache->get_tail(ctx, il, false) : nullptr;
}

ggml_tensor * llama_kv_cache_kvarn_context::get_v_tail(ggml_context * ctx, int32_t il) const {
    if (uses_standard_layer(il)) {
        GGML_UNUSED(ctx);
        return nullptr;
    }
    return shared_graph_layers.empty() ? cache->get_tail(ctx, il, true) : nullptr;
}

uint32_t llama_kv_cache_kvarn_context::get_tail_slots() const {
    return base()->get_tail_slots();
}

ggml_type llama_kv_cache_kvarn_context::get_tail_type() const {
    return base()->get_tail_type();
}

uint32_t llama_kv_cache_kvarn_context::get_tail_tokens() const {
    return base()->get_tail_tokens();
}

uint32_t llama_kv_cache_kvarn_context::get_tail_tokens(int32_t il) const {
    return cache->uses_kvarn_layer(graph_layer_for(il)) ? base()->get_tail_tokens() : 0;
}

uint32_t llama_kv_cache_kvarn_context::get_tail_arena_stride() const {
    return base()->get_tail_arena_stride();
}

uint32_t llama_kv_cache_kvarn_context::get_tail_attention_stride(uint32_t n_query_tokens) const {
    return base()->get_tail_attention_stride(n_query_tokens);
}

uint32_t llama_kv_cache_kvarn_context::get_tail_body_execution_stride() const {
    return cache->get_metadata_cache()->get_tail_body_execution_stride();
}

uint32_t llama_kv_cache_kvarn_context::get_tail_body_execution_rows(int32_t il) const {
    if (cache->uses_standard_layer(graph_layer_for(il))) return 0;
    return shared_graph_layers.empty() ?
            cache->get_metadata_cache()->get_tail_body_execution_rows(il) : 0;
}

bool llama_kv_cache_kvarn_context::has_compact_tail() const {
    return shared_graph_layers.empty() && base()->has_compact_tail();
}

bool llama_kv_cache_kvarn_context::has_compact_tail(int32_t il) const {
    if (uses_standard_layer(il)) return false;
    return shared_graph_layers.empty() && base()->has_compact_tail();
}

bool llama_kv_cache_kvarn_context::has_kv_body() const {
    return !shared_graph_layers.empty() || base()->has_kv_body();
}

bool llama_kv_cache_kvarn_context::has_kv_body(int32_t il) const {
    if (uses_standard_layer(il)) return true;
    return !shared_graph_layers.empty() || cache->get_metadata_cache()->has_kv_body(il);
}

bool llama_kv_cache_kvarn_context::uses_kvarn_layer(int32_t il) const {
    return cache->uses_kvarn_layer(graph_layer_for(il));
}

bool llama_kv_cache_kvarn_context::uses_standard_layer(int32_t il) const {
    return shared_graph_layers.empty() && cache->uses_standard_layer(graph_layer_for(il));
}

bool llama_kv_cache_kvarn_context::has_standard_cache() const {
    return shared_graph_layers.empty() && cache->has_standard_cache();
}

bool llama_kv_cache_kvarn_context::has_tail_current(int32_t il) const {
    if (uses_standard_layer(il)) return false;
    return shared_graph_layers.empty() && cache->get_metadata_cache()->has_tail_current(il);
}

ggml_backend_dev_t llama_kv_cache_kvarn_context::get_tail_backend(int32_t il) const {
    if (uses_standard_layer(il)) return nullptr;
    return shared_graph_layers.empty() ? cache->get_metadata_cache()->get_tail_backend(il) : nullptr;
}

llama_kv_tail_storage_kind llama_kv_cache_kvarn_context::get_tail_storage_kind() const {
    return base()->get_tail_storage_kind();
}

uint32_t llama_kv_cache_kvarn_context::get_tail_rollback_tokens() const {
    return base()->get_tail_rollback_tokens();
}

llama_kv_tail_route llama_kv_cache_kvarn_context::get_tail_route(int32_t il) const {
    if (uses_standard_layer(il)) return LLAMA_KV_TAIL_ROUTE_NONE;
    return shared_graph_layers.empty() ? cache->get_tail_route(il) : LLAMA_KV_TAIL_ROUTE_NONE;
}

const llama_kv_tail_layer_route * llama_kv_cache_kvarn_context::get_tail_layer_route(int32_t il) const {
    if (uses_standard_layer(il)) return nullptr;
    return shared_graph_layers.empty() ? cache->get_metadata_cache()->get_tail_layer_route(il) : nullptr;
}

bool llama_kv_cache_kvarn_context::get_tail_explicit_bias(int32_t il) const {
    if (uses_standard_layer(il)) return false;
    return shared_graph_layers.empty() && cache->get_tail_explicit_bias(il);
}

bool llama_kv_cache_kvarn_context::can_pack_tail_body(const llama_ubatch & ubatch) const {
    GGML_UNUSED(ubatch);
    // Structured persistent records are not row-addressable cache payloads,
    // even when a backend materializes them for attention.
    return false;
}

ggml_tensor * llama_kv_cache_kvarn_context::get_k_native(ggml_context * ctx, int32_t il) const {
    const int32_t shared_il = graph_layer_for(il);
    const auto it = stored_k.find(cache->mapped_layer_id(shared_il));
    ggml_tensor * stored = it != stored_k.end() ? it->second :
            cache->get_materialization_source(shared_il, false);
    GGML_ASSERT(stored != nullptr);
    return cache->view(ctx, stored, shared_il, get_n_kv(), current_sinfo(), false, mat_idxs);
}

ggml_tensor * llama_kv_cache_kvarn_context::get_v_native(ggml_context * ctx, int32_t il) const {
    const int32_t shared_il = graph_layer_for(il);
    const auto it = stored_v.find(cache->mapped_layer_id(shared_il));
    ggml_tensor * stored = it != stored_v.end() ? it->second :
            cache->get_materialization_source(shared_il, true);
    GGML_ASSERT(stored != nullptr);
    return cache->view(ctx, stored, shared_il, get_n_kv(), current_sinfo(), true, mat_idxs);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_kvarn_rot(ggml_context * ctx, int n_rot) const {
    GGML_ASSERT(n_rot == 64 || n_rot == 128 || n_rot == 256 || n_rot == 512);
    ggml_tensor * res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rot, n_rot);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kvarn_rot");
    return res;
}

ggml_tensor * llama_kv_cache_kvarn_context::cpy_k(
        ggml_context * ctx,
        ggml_tensor * k_cur,
        ggml_tensor * k_idxs,
        int32_t il) const {
    if (!shared_graph_layers.empty() || !k_cur) {
        return nullptr;
    }
    const int32_t shared_il = graph_layer_for(il);
    if (cache->uses_standard_layer(shared_il)) {
        return cache->standard_cpy_k(ctx, k_cur, k_idxs, shared_il, current_sinfo());
    }
    auto * result = cache->store(ctx, k_cur, k_idxs, shared_il, current_sinfo(), false);
    stored_k[cache->mapped_layer_id(shared_il)] = result;
    return result;
}

ggml_tensor * llama_kv_cache_kvarn_context::cpy_v(
        ggml_context * ctx,
        ggml_tensor * v_cur,
        ggml_tensor * v_idxs,
        int32_t il) const {
    if (!shared_graph_layers.empty() || !v_cur) {
        return nullptr;
    }
    const int32_t shared_il = graph_layer_for(il);
    if (cache->uses_standard_layer(shared_il)) {
        return cache->standard_cpy_v(ctx, v_cur, v_idxs, shared_il, current_sinfo());
    }
    auto * result = cache->store(ctx, v_cur, v_idxs, shared_il, current_sinfo(), true);
    stored_v[cache->mapped_layer_id(shared_il)] = result;
    return result;
}

ggml_tensor * llama_kv_cache_kvarn_context::cpy_k_with_tail(
        ggml_context *, ggml_tensor *, ggml_tensor *, ggml_tensor *, int32_t) const {
    return nullptr;
}

ggml_tensor * llama_kv_cache_kvarn_context::cpy_v_with_tail(
        ggml_context *, ggml_tensor *, ggml_tensor *, ggml_tensor *, int32_t) const {
    return nullptr;
}

ggml_tensor * llama_kv_cache_kvarn_context::cpy_k_tail(
        ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * tail_idxs,
        int32_t il, ggml_tensor * dependency) const {
    if (uses_standard_layer(il)) return nullptr;
    if (!shared_graph_layers.empty() || !k_cur || !tail_idxs) {
        return nullptr;
    }
    return cache->store_tail(ctx, k_cur, tail_idxs, il, false, dependency);
}

ggml_tensor * llama_kv_cache_kvarn_context::cpy_v_tail(
        ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * tail_idxs,
        int32_t il, ggml_tensor * dependency) const {
    if (uses_standard_layer(il)) return nullptr;
    if (!shared_graph_layers.empty() || !v_cur || !tail_idxs) {
        return nullptr;
    }
    return cache->store_tail(ctx, v_cur, tail_idxs, il, true, dependency);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return base()->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return base()->build_input_v_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_tail_idxs(
        ggml_context * ctx, const llama_ubatch & ubatch) const {
    return base()->build_input_tail_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_tail_body_idxs(ggml_context * ctx) const {
    return base()->build_input_tail_body_idxs(ctx);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_k_rot(ggml_context * ctx) const {
    if (has_standard_cache()) return cache->standard_build_input_k_rot(ctx);
    return base()->build_input_k_rot(ctx);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_v_rot(ggml_context * ctx) const {
    if (has_standard_cache()) return cache->standard_build_input_v_rot(ctx);
    return base()->build_input_v_rot(ctx);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_kvarn_mat_idxs(ggml_context * ctx) const {
    // SWA and compact reads use one index per output cache cell. Shared
    // multi-stream reads need one high-watermark index per target stream so
    // materialization can locate each stream's live stage group.
    const uint32_t n_indices = cache->is_swa() ? get_n_kv()*cache->get_kv_n_stream() :
            (uses_shared_live_indices() ? cache->get_kv_n_stream() : get_n_kv());
    ggml_tensor * res = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_indices);
    ggml_set_input(res);
    ggml_set_name(res, cache->is_swa() ?
            "attn_inp_kvarn_mat_idxs_swa" : "attn_inp_kvarn_read_idxs");
    return res;
}

void llama_kv_cache_kvarn_context::set_input_kvarn_mat_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(dst->type == GGML_TYPE_I64);

    if (uses_shared_live_indices()) {
        const auto indices = shared_live_indices();
        GGML_ASSERT(indices.size() == size_t(dst->ne[0]));
        std::memcpy(dst->data, indices.data(), indices.size()*sizeof(indices[0]));
        return;
    }

    if (uses_compact_read_indices()) {
        const auto & plan = compact_read_plan();
        GGML_ASSERT(plan.size() == size_t(dst->ne[0]));
        const auto * metadata = cache->get_metadata_cache();
        int64_t * data = static_cast<int64_t *>(dst->data);
        for (size_t read = 0; read < plan.size(); ++read) {
            const int64_t cell = plan[read];
            if (cell < 0) {
                data[read] = cell;
                continue;
            }
            int32_t stage_slot = -1;
            if (metadata->allocation_cell_uses_stage(uint32_t(cell))) {
                const auto & sinfo = current_sinfo();
                if (!sinfo.empty() && !sinfo.stage_slots.empty()) {
                    for (uint32_t stream = 0; stream < sinfo.n_stream() && stage_slot < 0; ++stream) {
                        for (size_t i = 0; i < sinfo.idxs[stream].size(); ++i) {
                            if (sinfo.idxs[stream][i] == uint32_t(cell)) {
                                stage_slot = int32_t(sinfo.stage_slots[stream][i]);
                                break;
                            }
                        }
                    }
                }
                if (stage_slot < 0) {
                    stage_slot = metadata->allocation_cell_stage_slot(uint32_t(cell));
                }
                GGML_ASSERT(stage_slot >= 0);
            }
            data[read] = stage_slot >= 0 ?
                    llama_kvarn_encode_stage_cell(uint32_t(cell), uint32_t(stage_slot)) : cell;
        }
        return;
    }

    const auto * kv = cache->get_metadata_cache();
    const uint32_t n_stream = cache->get_kv_n_stream();
    GGML_ASSERT(dst->ne[0] % n_stream == 0);
    const uint32_t n_kv = uint32_t(dst->ne[0]) / n_stream;
    int64_t * data = (int64_t *) dst->data;

    for (uint32_t stream = 0; stream < n_stream; ++stream) {
        const auto & cells = kv->get_cells(stream);
        for (uint32_t cell = 0; cell < n_kv; ++cell) {
            data[stream*n_kv + cell] = cells.is_empty(cell) ? -1 :
                    llama_kvarn_encode_swa_position(stream, uint32_t(cells.pos_get(cell)));
        }
    }

    // Metadata commits after compute. Mirror pending target writes; shared MTP
    // contexts have no K/V projections and read only committed target records.
    if (ubatch != nullptr && shared_graph_layers.empty()) {
        const auto & sinfo = current_sinfo();
        if (!sinfo.empty()) {
            GGML_ASSERT(ubatch->n_tokens == sinfo.size()*sinfo.n_stream());
            for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
                const uint32_t stream = sinfo.strm[s];
                GGML_ASSERT(stream < n_stream);
                for (uint32_t i = 0; i < sinfo.size(); ++i) {
                    GGML_ASSERT(sinfo.idxs[s][i] < n_kv);
                    data[stream*n_kv + sinfo.idxs[s][i]] = llama_kvarn_encode_swa_position(
                            stream, uint32_t(ubatch->pos[s*sinfo.size() + i]));
                }
            }
        }
    }
}

void llama_kv_cache_kvarn_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    if (cache->is_swa()) {
        GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
        int64_t * data = (int64_t *) dst->data;
        const auto & sinfo = current_sinfo();
        GGML_ASSERT(ubatch->n_tokens == sinfo.size()*sinfo.n_stream());
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                data[s*sinfo.size() + i] = llama_kvarn_encode_swa_position(
                        sinfo.strm[s], uint32_t(ubatch->pos[s*sinfo.size() + i]));
            }
        }
        return;
    }
    base()->set_input_k_idxs(dst, ubatch);
    if (cache->uses_compact_read_indices()) {
        GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
        const auto & sinfo = current_sinfo();
        GGML_ASSERT(sinfo.n_stream() == 1 && !sinfo.stage_slots.empty() &&
                sinfo.stage_slots[0].size() == ubatch->n_tokens);
        auto * data = static_cast<int64_t *>(dst->data);
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            data[i] = llama_kvarn_encode_store_cell(
                    llama_kvarn_decode_cell(data[i]), sinfo.stage_slots[0][i]);
        }
    }
}

void llama_kv_cache_kvarn_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    if (cache->is_swa()) {
        set_input_k_idxs(dst, ubatch);
        return;
    }
    base()->set_input_v_idxs(dst, ubatch);
    if (cache->uses_compact_read_indices()) {
        GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
        const auto & sinfo = current_sinfo();
        GGML_ASSERT(sinfo.n_stream() == 1 && !sinfo.stage_slots.empty() &&
                sinfo.stage_slots[0].size() == ubatch->n_tokens);
        auto * data = static_cast<int64_t *>(dst->data);
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            data[i] = llama_kvarn_encode_store_cell(
                    llama_kvarn_decode_cell(data[i]), sinfo.stage_slots[0][i]);
        }
    }
}

void llama_kv_cache_kvarn_context::set_input_tail_idxs(
        ggml_tensor * dst, const llama_ubatch * ubatch) const {
    base()->set_input_tail_idxs(dst, ubatch);
}

void llama_kv_cache_kvarn_context::set_input_tail_body_idxs(ggml_tensor * dst) const {
    base()->set_input_tail_body_idxs(dst);
}

void llama_kv_cache_kvarn_context::set_input_k_idxs_backend(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    if (cache->is_swa()) {
        const auto & sinfo = current_sinfo();
        GGML_ASSERT(ubatch->n_tokens == sinfo.size()*sinfo.n_stream());
        std::vector<int64_t> data(ubatch->n_tokens);
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                data[s*sinfo.size() + i] = llama_kvarn_encode_swa_position(
                        sinfo.strm[s], uint32_t(ubatch->pos[s*sinfo.size() + i]));
            }
        }
        ggml_backend_tensor_set(dst, data.data(), 0, data.size() * sizeof(int64_t));
        return;
    }
    if (cache->uses_compact_read_indices()) {
        const auto & sinfo = current_sinfo();
        GGML_ASSERT(sinfo.n_stream() == 1 && !sinfo.stage_slots.empty() &&
                sinfo.stage_slots[0].size() == ubatch->n_tokens);
        std::vector<int64_t> data(ubatch->n_tokens);
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            data[i] = llama_kvarn_encode_store_cell(sinfo.idxs[0][i], sinfo.stage_slots[0][i]);
        }
        ggml_backend_tensor_set(dst, data.data(), 0, data.size()*sizeof(int64_t));
        return;
    }
    base()->set_input_k_idxs_backend(dst, ubatch);
}

void llama_kv_cache_kvarn_context::set_input_v_idxs_backend(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    if (cache->is_swa()) {
        const auto & sinfo = current_sinfo();
        GGML_ASSERT(ubatch->n_tokens == sinfo.size()*sinfo.n_stream());
        std::vector<int64_t> data(ubatch->n_tokens);
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                data[s*sinfo.size() + i] = llama_kvarn_encode_swa_position(
                        sinfo.strm[s], uint32_t(ubatch->pos[s*sinfo.size() + i]));
            }
        }
        ggml_backend_tensor_set(dst, data.data(), 0, data.size() * sizeof(int64_t));
        return;
    }
    if (cache->uses_compact_read_indices()) {
        const auto & sinfo = current_sinfo();
        GGML_ASSERT(sinfo.n_stream() == 1 && !sinfo.stage_slots.empty() &&
                sinfo.stage_slots[0].size() == ubatch->n_tokens);
        std::vector<int64_t> data(ubatch->n_tokens);
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            data[i] = llama_kvarn_encode_store_cell(sinfo.idxs[0][i], sinfo.stage_slots[0][i]);
        }
        ggml_backend_tensor_set(dst, data.data(), 0, data.size()*sizeof(int64_t));
        return;
    }
    base()->set_input_v_idxs_backend(dst, ubatch);
}

void llama_kv_cache_kvarn_context::set_input_k_shift(ggml_tensor * dst) const {
    base()->set_input_k_shift(dst);
}

void llama_kv_cache_kvarn_context::set_input_kq_mask(
        ggml_tensor * dst,
        const llama_ubatch * ubatch,
        bool causal_attn) const {
    if (uses_compact_read_indices()) {
        cache->get_metadata_cache()->set_input_kq_mask_mapped(
                dst, ubatch, causal_attn, compact_read_plan());
    } else {
        base()->set_input_kq_mask(dst, ubatch, causal_attn);
    }
}

void llama_kv_cache_kvarn_context::set_standard_input_kq_mask(
        ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    if (shared_graph_layers.empty() && cache->has_standard_cache()) {
        cache->standard_cache_set_input_kq_mask(dst, ubatch, causal_attn);
        return;
    }
    throw std::logic_error("standard KV mask requested without a target mixed cache");
}

void llama_kv_cache_kvarn_context::set_input_kq_mask_tail(
        ggml_tensor * body, ggml_tensor * exact,
        ggml_tensor * read_idxs, ggml_tensor * body_read_idxs, ggml_tensor * bias_read_idxs,
        const llama_ubatch * ubatch, bool causal_attn) const {
    if (uses_compact_read_indices()) {
        cache->get_metadata_cache()->set_input_kq_mask_tail_mapped(
                body, exact, read_idxs, body_read_idxs, bias_read_idxs,
                ubatch, causal_attn, compact_read_plan());
    } else {
        base()->set_input_kq_mask_tail(
                body, exact, read_idxs, body_read_idxs, bias_read_idxs, ubatch, causal_attn);
    }
}

void llama_kv_cache_kvarn_context::set_input_tail_body_plan(
        ggml_tensor * query_order, ggml_tensor * run_desc,
        ggml_tensor * body_mask, const llama_ubatch * ubatch, bool causal_attn) const {
    base()->set_input_tail_body_plan(query_order, run_desc, body_mask, ubatch, causal_attn);
}

void llama_kv_cache_kvarn_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    base()->set_input_pos_bucket(dst, ubatch);
}

void llama_kv_cache_kvarn_context::set_input_k_rot(ggml_tensor * dst) const {
    if (has_standard_cache()) {
        cache->standard_set_input_k_rot(dst);
        return;
    }
    base()->set_input_k_rot(dst);
}

void llama_kv_cache_kvarn_context::set_input_v_rot(ggml_tensor * dst) const {
    if (has_standard_cache()) {
        cache->standard_set_input_v_rot(dst);
        return;
    }
    base()->set_input_v_rot(dst);
}

void llama_kv_cache_kvarn_context::set_input_k_rot_backend(ggml_tensor * dst) const {
    if (has_standard_cache()) {
        cache->standard_set_input_k_rot_backend(dst);
        return;
    }
    base()->set_input_k_rot_backend(dst);
}

void llama_kv_cache_kvarn_context::set_input_v_rot_backend(ggml_tensor * dst) const {
    if (has_standard_cache()) {
        cache->standard_set_input_v_rot_backend(dst);
        return;
    }
    base()->set_input_v_rot_backend(dst);
}

void llama_kv_cache_kvarn_context::set_input_kvarn_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT((dst->ne[0] == 64 || dst->ne[0] == 128 || dst->ne[0] == 256 || dst->ne[0] == 512) && dst->ne[1] == dst->ne[0]);

    const auto & data = kvarn_hadamard((int) dst->ne[0]);
    memcpy(dst->data, data.data(), ggml_nbytes(dst));
}

struct llama_kv_cache_kvarn::migration_queue {
    struct span {
        ggml_tensor * src;
        ggml_tensor * dst;
        size_t src_offset;
        size_t dst_offset;
        size_t size;
    };

    struct job {
        bool source_cuda = true;
        ggml_backend_event_ptr producer_event;
        std::vector<span> spans;
    };

    migration_queue(ggml_backend_dev_t cuda_dev, ggml_backend_dev_t vulkan_dev) :
        cuda_dev(cuda_dev),
        vulkan_dev(vulkan_dev),
        cuda_backend(ggml_backend_dev_init(cuda_dev, nullptr)),
        vulkan_backend(ggml_backend_dev_init(vulkan_dev, nullptr)) {
        if (!cuda_backend || !vulkan_backend) {
            throw std::runtime_error("failed to initialize KVarN migration transfer backends");
        }
        constexpr size_t staging_bytes = 4u * 1024u * 1024u;
        cuda_host.reset(ggml_backend_buft_alloc_buffer(
                ggml_backend_dev_host_buffer_type(cuda_dev), staging_bytes));
        vulkan_host.reset(ggml_backend_buft_alloc_buffer(
                ggml_backend_dev_host_buffer_type(vulkan_dev), staging_bytes));
        if (!cuda_host || !vulkan_host) {
            throw std::runtime_error("failed to allocate pinned KVarN migration staging buffers");
        }
        cuda_host_ptr = ggml_backend_buffer_get_base(cuda_host.get());
        vulkan_host_ptr = ggml_backend_buffer_get_base(vulkan_host.get());
        staging_size = staging_bytes;
        worker = std::thread([this] { worker_loop(); });
    }

    migration_queue(const migration_queue &) = delete;
    migration_queue & operator=(const migration_queue &) = delete;

    ~migration_queue() {
        if (!drain()) {
            LLAMA_LOG_ERROR("KVarN migration drain failed during destruction; active cache remains authoritative\n");
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        work.notify_all();
        if (worker.joinable()) worker.join();
    }

    bool enqueue(ggml_backend_t producer, bool source_cuda, std::vector<span> spans) {
        if (spans.empty()) return true;
        job next;
        next.source_cuda = source_cuda;
        next.spans = std::move(spans);
        if (producer != nullptr) {
            auto * dev = ggml_backend_get_device(producer);
            // Incremental mirroring is intentionally CUDA -> Vulkan only.
            // Vulkan event synchronization mutates its producer command pool,
            // so a scheduler-produced Vulkan event cannot be waited by worker.
            if (!source_cuda || dev != cuda_dev) {
                return false;
            }
        }

        std::unique_lock<std::mutex> lock(mutex);
        slots_available.wait(lock, [this] { return pending < max_pending || error || stopping; });
        if (error || stopping) return false;
        if (producer != nullptr) {
            next.producer_event.reset(ggml_backend_event_new(cuda_dev));
            if (!next.producer_event) return false;
            ggml_backend_event_record(next.producer_event.get(), producer);
        }
        try {
            queue.push_back(std::move(next));
        } catch (...) {
            lock.unlock();
            // The event may already be recorded when deque growth fails.
            // Complete the producer before its RAII wrapper releases it.
            if (producer != nullptr) ggml_backend_synchronize(producer);
            throw;
        }
        ++pending;
        lock.unlock();
        work.notify_one();
        return true;
    }

    bool drain() {
        std::unique_lock<std::mutex> lock(mutex);
        finished.wait(lock, [this] { return pending == 0; });
        return !error;
    }

    uint64_t copied_bytes() const {
        return bytes_copied.load(std::memory_order_relaxed);
    }

private:
    void copy_span(
            ggml_backend_t source_backend,
            ggml_backend_t destination_backend,
            bool source_cuda,
            const span & item) {
        void * source_host = source_cuda ? cuda_host_ptr : vulkan_host_ptr;
        void * destination_host = source_cuda ? vulkan_host_ptr : cuda_host_ptr;
        for (size_t done = 0; done < item.size;) {
            const size_t chunk = std::min(staging_size, item.size - done);
            ggml_backend_tensor_get_async(
                    source_backend, item.src, source_host,
                    item.src_offset + done, chunk);
            ggml_backend_synchronize(source_backend);
            std::memcpy(destination_host, source_host, chunk);
            ggml_backend_tensor_set_async(
                    destination_backend, item.dst, destination_host,
                    item.dst_offset + done, chunk);
            ggml_backend_synchronize(destination_backend);
            done += chunk;
        }
        bytes_copied.fetch_add(item.size, std::memory_order_relaxed);
    }

    void worker_loop() {
        while (true) {
            job current;
            {
                std::unique_lock<std::mutex> lock(mutex);
                work.wait(lock, [this] { return stopping || !queue.empty(); });
                if (stopping && queue.empty()) return;
                current = std::move(queue.front());
                queue.pop_front();
            }

            try {
                if (!current.spans.empty()) {
                    ggml_backend_t source_backend = current.source_cuda ? cuda_backend.get() : vulkan_backend.get();
                    ggml_backend_t destination_backend = current.source_cuda ? vulkan_backend.get() : cuda_backend.get();
                    if (current.producer_event) {
                        // Wait on the worker, not the inference thread. This
                        // ensures the producer has committed the record before
                        // the transfer reads it, while later compute can run
                        // concurrently on its independent backend stream. The
                        // only producer events here are from CUDA.
                        ggml_backend_event_synchronize(current.producer_event.get());
                        current.producer_event.reset();
                    }
                    for (const auto & item : current.spans) {
                        copy_span(source_backend, destination_backend, current.source_cuda, item);
                    }
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex);
                if (!error) error = std::current_exception();
            }

            {
                std::lock_guard<std::mutex> lock(mutex);
                GGML_ASSERT(pending > 0);
                --pending;
                if (pending == 0) finished.notify_all();
                slots_available.notify_one();
            }
        }
    }

    static constexpr size_t max_pending = 4;
    ggml_backend_dev_t cuda_dev;
    ggml_backend_dev_t vulkan_dev;
    ggml_backend_ptr cuda_backend;
    ggml_backend_ptr vulkan_backend;
    ggml_backend_buffer_ptr cuda_host;
    ggml_backend_buffer_ptr vulkan_host;
    void * cuda_host_ptr = nullptr;
    void * vulkan_host_ptr = nullptr;
    size_t staging_size = 0;
    std::mutex mutex;
    std::condition_variable work;
    std::condition_variable finished;
    std::condition_variable slots_available;
    std::deque<job> queue;
    std::thread worker;
    size_t pending = 0;
    bool stopping = false;
    std::exception_ptr error;
    std::atomic<uint64_t> bytes_copied { 0 };
};

llama_kv_cache_kvarn::~llama_kv_cache_kvarn() = default;

llama_kv_cache_kvarn::llama_kv_cache_kvarn(
        const llama_model & model,
        const llama_hparams & hparams,
        llama_kvarn_params params,
        bool offload,
        bool unified,
        uint32_t kv_size,
        uint32_t n_seq_max,
        uint32_t n_batch,
        uint32_t n_ubatch,
        uint32_t n_pad,
        uint32_t n_swa,
        llama_swa_type swa_type,
        const layer_filter_cb & filter,
        const layer_reuse_cb & reuse,
        uint32_t tail_tokens,
        ggml_type tail_type_requested,
        uint32_t tail_tokens_requested,
        uint32_t tail_rollback_tokens,
        const layer_device_cb & device_for_layer,
        const layer_device_cb & migration_device_for_layer,
        const layer_filter_cb & standard_layer_filter,
        ggml_type standard_type_k,
        ggml_type standard_type_v,
        bool standard_v_trans,
        const layer_device_cb & standard_device_for_layer) :
    model(model),
    hparams(hparams),
    params(params),
    n_stream(unified ? 1u : n_seq_max),
    n_seq_max(n_seq_max),
    kv_size(kv_size),
    // Dynamic staging: size the lossless F16 ring from position semantics.
    // Non-SWA keeps a permanent sink slot plus the scheduler-span tail. SWA has
    // no sink, so every stage slot is part of the local tail.
    tail_groups(kvarn_stage_tail_groups(
        n_batch, n_ubatch, n_swa > 0 && swa_type != LLAMA_SWA_TYPE_NONE,
        unified ? n_seq_max : 1u)),
    stage_groups((n_swa > 0 && swa_type != LLAMA_SWA_TYPE_NONE) ? tail_groups : tail_groups + 1u),
    swa(n_swa > 0 && swa_type != LLAMA_SWA_TYPE_NONE),
    // SWA: the metadata window may span one more 128-token tile than its nominal
    // size, and a batched prefill needs the union of every row's sliding window.
    // The record ring stores only tiles older than the F16 tail; loaders account
    // for the tail offset when deciding whether a ring slot is live.
    n_groups_per_stream(kvarn_record_groups_per_stream(kv_size, n_ubatch, n_swa, swa, tail_groups)),
    exact_tail_tokens(tail_tokens),
    metadata_n_pad(n_pad),
    metadata_n_swa(n_swa),
    metadata_swa_type(swa_type),
    metadata_n_ubatch(n_ubatch),
    exact_tail_tokens_requested(tail_tokens_requested),
    exact_tail_type_requested(tail_type_requested),
    exact_tail_type(tail_type_requested),
    state_owner_id(next_mixed_state_owner_id()),
    metadata(std::make_unique<llama_kv_cache>(
        model,
        hparams,
        GGML_TYPE_F16,
        GGML_TYPE_F16,
        false,
        false,
        unified,
        kv_size,
        n_seq_max,
        n_pad,
        n_swa,
        swa_type,
        nullptr,
        [](int32_t) { return false; },
        nullptr,
        nullptr,
        "",
        n_ubatch,
        tail_tokens,
        tail_type_requested,
            tail_tokens_requested,
            true,
        tail_rollback_tokens)),
    migration_enabled(bool(migration_device_for_layer)) {
    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(swa || kv_size % KVAR_N_GROUP == 0);
    GGML_ASSERT(stage_groups >= 2 && "KVarN stage depth must be at least 2");
    GGML_ASSERT(tail_groups >= 1 && tail_groups <= stage_groups &&
        "KVarN tail depth must fit within the F16 stage");
    exact_tail_type = metadata->get_tail_type();
    if (!swa) {
        metadata->set_allocation_group_size(KVAR_N_GROUP, n_stream == 1 ? tail_groups : 1u);
    }
    if (standard_type_k != GGML_TYPE_COUNT || standard_type_v != GGML_TYPE_COUNT) {
        if (standard_type_k == GGML_TYPE_COUNT || standard_type_v == GGML_TYPE_COUNT ||
                !standard_layer_filter || !standard_device_for_layer) {
            throw std::invalid_argument("mixed KVarN/standard KV requires paired types and remote layer/device filters");
        }
        if (swa || n_seq_max != 1 || n_stream != 1) {
            throw std::invalid_argument("mixed KVarN/standard KV currently requires non-SWA and one sequence");
        }
        standard_cache = std::make_unique<llama_kv_cache>(
                model, hparams, standard_type_k, standard_type_v, standard_v_trans,
                offload, unified, kv_size, n_seq_max, n_pad, n_swa, swa_type,
                metadata.get(), standard_layer_filter, nullptr, nullptr, "remote_",
                n_ubatch, 0, exact_tail_type_requested, 0, false, 0, 0, false,
                standard_device_for_layer, false, true);
        for (const uint32_t il : standard_cache->get_layer_ids()) {
            const ggml_tensor * key = standard_cache->get_k_storage(int32_t(il));
            const ggml_tensor * value = standard_cache->get_v_storage(int32_t(il));
            if (!key || !value || !key->buffer || !value->buffer) {
                throw std::runtime_error(format("mixed standard KV layer %u has no allocated K/V payload", il));
            }
            const auto key_buft = ggml_backend_buffer_get_type(key->buffer);
            const auto value_buft = ggml_backend_buffer_get_type(value->buffer);
            const auto key_dev = key_buft ? ggml_backend_buft_get_device(key_buft) : nullptr;
            const auto value_dev = value_buft ? ggml_backend_buft_get_device(value_buft) : nullptr;
            if (key_dev != value_dev) {
                throw std::runtime_error(format("mixed standard KV layer %u places K/V on different devices", il));
            }
            const bool rotated = standard_cache->uses_attn_rot_k() || standard_cache->uses_attn_rot_v();
            LLAMA_LOG_INFO("mixed KV layer=%u device=%s type_k=%s type_v=%s domain=%s\n",
                    il, key_dev ? ggml_backend_dev_name(key_dev) : "CPU",
                    ggml_type_name(key->type), ggml_type_name(value->type),
                    rotated ? "standard-hadamard" : "standard-original");
        }
        if (standard_cache->get_layer_ids().empty()) {
            throw std::runtime_error("mixed KV override selected no remote standard-cache layers");
        }
    }
    if (swa) {
        const uint32_t in_flight_groups = std::max<uint32_t>(1u, (n_ubatch + KVAR_N_GROUP - 1u) / KVAR_N_GROUP);
        // Backstop for the ring-size invariant above: the record ring must have
        // enough slots for the compressed portion of the worst-case visible tile
        // span after subtracting the F16 tail and adding the active ubatch span.
        GGML_ASSERT(n_groups_per_stream + tail_groups >=
                kvarn_swa_visible_groups(kv_size, n_swa) + in_flight_groups - 1u &&
            "SWA KVarN record ring is too small for the deduplicated sliding window");
    }
    // Dynamic staging keeps the F16/compressed mix stable across physical ubatch
    // splits. Log the configured stage depth and its memory cost at cache
    // creation so regressions in the propagation are visible at startup.
    LLAMA_LOG_INFO("KVarN cache: stage_groups=%u tail_groups=%u n_batch=%u n_ubatch=%u%s\n",
            stage_groups, tail_groups, n_batch, n_ubatch, swa ? " (SWA ring)" : "");

    using ctx_key = std::pair<ggml_backend_buffer_type_t, int32_t>;
    struct ctx_key_comparator {
        bool operator()(const ctx_key & lhs, const ctx_key & rhs) const {
            const int by_buft = std::strcmp(
                    ggml_backend_buft_name(lhs.first), ggml_backend_buft_name(rhs.first));
            return by_buft != 0 ? by_buft < 0 : lhs.second < rhs.second;
        }
    };

    std::map<ctx_key, ggml_context_ptr, ctx_key_comparator> ctx_map;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft, int32_t migration_layer = -1) -> ggml_context * {
        const ctx_key key { buft, migration_layer };
        const auto it = ctx_map.find(key);
        if (it != ctx_map.end()) {
            return it->second.get();
        }

        ggml_init_params ctx_params = {
            /*.mem_size   =*/ size_t((12u + 8u * n_stream) * hparams.n_layer_kv() * ggml_tensor_overhead()),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx { ggml_init(ctx_params) };
        if (!ctx) {
            return nullptr;
        }

        auto * result = ctx.get();
        ctx_map.emplace(key, std::move(ctx));
        return result;
    };

    const int64_t n_record_groups = int64_t(n_groups_per_stream) * n_stream;
    // Stage depth is a cache property derived from position semantics. Non-SWA
    // caches cover the logical scheduler batch plus one physical ubatch; SWA
    // keeps only the live tail while older visible tiles use the record ring.
    // Backends read stage_groups from op_params[7] instead of assuming 3.
    const int64_t n_stage_tokens = int64_t(KVAR_N_GROUP) * int64_t(stage_groups) * n_stream;
    size_t raw_bytes = 0;

    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        if (!hparams.has_kv(il) || hparams.is_recr(il)) {
            continue;
        }
        if (filter && !filter(il)) {
            continue;
        }

        auto * dev = offload ? (device_for_layer ? device_for_layer(il) : model.dev_layer(il)) : nullptr;
        auto * mirror_dev = migration_device_for_layer ? migration_device_for_layer(il) : nullptr;
        if (mirror_dev != nullptr && (!offload || dev == nullptr || mirror_dev == dev || swa || n_stream != 1)) {
            throw std::runtime_error(format(
                "KVarN prefill migration layer %u requires distinct device buffers, non-SWA, and one stream",
                il));
        }
        auto * buft = offload ? ggml_backend_dev_buffer_type(dev) : ggml_backend_cpu_buffer_type();
        auto * ctx = ctx_for_buft(buft, mirror_dev != nullptr ? int32_t(il) : -1);
        if (!ctx) {
            throw std::runtime_error("failed to create KVarN cache tensor context");
        }

        const uint32_t n_head_kv = hparams.n_head_kv(il);
        const uint32_t head_dim_k = hparams.n_embd_head_k(il);
        const uint32_t head_dim_v = hparams.n_embd_head_v(il);
        llama_kvarn_geometry k_geometry = {};
        llama_kvarn_geometry v_geometry = {};
        if (!llama_kvarn_geometry_for(head_dim_k, k_geometry) ||
                !llama_kvarn_geometry_for(head_dim_v, v_geometry)) {
            throw std::runtime_error(format(
                "KVarN cache layer %u has unsupported K/V head dimensions %u/%u",
                il, head_dim_k, head_dim_v));
        }
        if (!llama_kvarn_backend_supports_ops(dev, head_dim_k) ||
                !llama_kvarn_backend_supports_ops(dev, head_dim_v)) {
            throw std::runtime_error(format(
                "KVarN cache layer %u is assigned to backend %s, which cannot store and attend KVarN dimensions %u/%u",
                il, dev ? ggml_backend_dev_name(dev) : "CPU", head_dim_k, head_dim_v));
        }
        const int k_slices = int(k_geometry.head_slices);
        const int v_slices = int(v_geometry.head_slices);
        const int k_record_dim = int(k_geometry.record_dim);
        const int v_record_dim = int(v_geometry.record_dim);
        const size_t k_record_size = kvarn_record_bytes(k_record_dim, this->params.key_bits, false);
        const size_t v_record_size = kvarn_record_bytes(v_record_dim, this->params.value_bits, true);
        const bool explicit_bias = model.self_attention_uses_explicit_bias(il);
        const bool native_tail = exact_tail_tokens == 0 ||
            (!explicit_bias && kvarn_backend_supports_native_tail(
                dev, exact_tail_type, head_dim_k, head_dim_v));
        const bool native_attention =
            llama_kvarn_backend_supports_native_ops(dev) && native_tail && !(swa && n_stream > 1);
        const bool mixed_tail_native = native_attention &&
            llama_kvarn_backend_mixed_tail_native_preferred(dev);
        const bool native_original_v = native_attention &&
            llama_kvarn_backend_native_attention_uses_original_v(dev);
        const uint32_t native_rotated_max_query_tokens = native_attention ?
            llama_kvarn_backend_native_rotated_max_query_tokens(dev) : 0;

        const uint32_t n_head_k_sliced = n_head_kv * (uint32_t) k_slices;
        const uint32_t n_head_v_sliced = n_head_kv * (uint32_t) v_slices;
        auto * k_records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, k_record_size, n_head_k_sliced, n_record_groups);
        auto * v_records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, v_record_size, n_head_v_sliced, n_record_groups);
        auto * k_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, k_record_dim, n_head_k_sliced, n_stage_tokens);
        auto * v_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, v_record_dim, n_head_v_sliced, n_stage_tokens);
        ggml_tensor * k_tail = nullptr;
        ggml_tensor * v_tail = nullptr;

        ggml_format_name(k_records, "cache_kvarn_k_records_l%d", il);
        ggml_format_name(v_records, "cache_kvarn_v_records_l%d", il);
        ggml_format_name(k_stage, "cache_kvarn_k_stage_l%d", il);
        ggml_format_name(v_stage, "cache_kvarn_v_stage_l%d", il);

        std::vector<ggml_tensor *> k_records_stream;
        std::vector<ggml_tensor *> v_records_stream;
        std::vector<ggml_tensor *> k_stage_stream;
        std::vector<ggml_tensor *> v_stage_stream;
        k_records_stream.reserve(n_stream);
        v_records_stream.reserve(n_stream);
        k_stage_stream.reserve(n_stream);
        v_stage_stream.reserve(n_stream);

        for (uint32_t s = 0; s < n_stream; ++s) {
            auto * k_records_view = ggml_view_3d(
                    ctx, k_records,
                    k_record_size, n_head_k_sliced, n_groups_per_stream,
                    k_records->nb[1], k_records->nb[2],
                    size_t(s) * n_groups_per_stream * k_records->nb[2]);
            auto * v_records_view = ggml_view_3d(
                    ctx, v_records,
                    v_record_size, n_head_v_sliced, n_groups_per_stream,
                    v_records->nb[1], v_records->nb[2],
                    size_t(s) * n_groups_per_stream * v_records->nb[2]);
            auto * k_stage_view = ggml_view_3d(
                    ctx, k_stage,
                    k_record_dim, n_head_k_sliced, KVAR_N_GROUP * stage_groups,
                    k_stage->nb[1], k_stage->nb[2],
                    size_t(s) * KVAR_N_GROUP * stage_groups * k_stage->nb[2]);
            auto * v_stage_view = ggml_view_3d(
                    ctx, v_stage,
                    v_record_dim, n_head_v_sliced, KVAR_N_GROUP * stage_groups,
                    v_stage->nb[1], v_stage->nb[2],
                    size_t(s) * KVAR_N_GROUP * stage_groups * v_stage->nb[2]);

            ggml_format_name(k_records_view, "cache_kvarn_k_records_l%d_s%d", il, s);
            ggml_format_name(v_records_view, "cache_kvarn_v_records_l%d_s%d", il, s);
            ggml_format_name(k_stage_view, "cache_kvarn_k_stage_l%d_s%d", il, s);
            ggml_format_name(v_stage_view, "cache_kvarn_v_stage_l%d_s%d", il, s);

            k_records_stream.push_back(k_records_view);
            v_records_stream.push_back(v_records_view);
            k_stage_stream.push_back(k_stage_view);
            v_stage_stream.push_back(v_stage_view);
        }

        map_layer_ids[il] = layers.size();
        layers.push_back({
            il,
            n_head_kv,
            head_dim_k,
            head_dim_v,
            (uint32_t) k_slices,
            (uint32_t) v_slices,
            native_attention,
            mixed_tail_native,
            native_original_v,
            native_rotated_max_query_tokens,
            k_records,
            v_records,
            k_stage,
            v_stage,
            k_tail,
            v_tail,
            std::move(k_records_stream),
            std::move(v_records_stream),
            std::move(k_stage_stream),
            std::move(v_stage_stream),
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            {}, {}, {}, {},
            false, false, false, 0, nullptr,
        });

        auto & layer = layers.back();
        layer.mirror_dev = mirror_dev;
        if (mirror_dev != nullptr) {
            if (!llama_kvarn_backend_supports_ops(mirror_dev, head_dim_k) ||
                    !llama_kvarn_backend_supports_ops(mirror_dev, head_dim_v)) {
                throw std::runtime_error(format(
                    "KVarN migration mirror layer %u backend %s cannot store/attend its head dimensions",
                    il, ggml_backend_dev_name(mirror_dev)));
            }
            auto * mirror_buft = ggml_backend_dev_buffer_type(mirror_dev);
            auto * mirror_ctx = ctx_for_buft(mirror_buft, int32_t(il));
            if (!mirror_ctx) {
                throw std::runtime_error("failed to create KVarN migration mirror tensor context");
            }
            layer.mirror_k_records = ggml_new_tensor_3d(
                mirror_ctx, GGML_TYPE_I8, k_record_size, n_head_k_sliced, n_record_groups);
            layer.mirror_v_records = ggml_new_tensor_3d(
                mirror_ctx, GGML_TYPE_I8, v_record_size, n_head_v_sliced, n_record_groups);
            layer.mirror_k_stage = ggml_new_tensor_3d(
                mirror_ctx, GGML_TYPE_F16, k_record_dim, n_head_k_sliced, n_stage_tokens);
            layer.mirror_v_stage = ggml_new_tensor_3d(
                mirror_ctx, GGML_TYPE_F16, v_record_dim, n_head_v_sliced, n_stage_tokens);
            ggml_format_name(layer.mirror_k_records, "cache_kvarn_k_records_mirror_l%d", il);
            ggml_format_name(layer.mirror_v_records, "cache_kvarn_v_records_mirror_l%d", il);
            ggml_format_name(layer.mirror_k_stage, "cache_kvarn_k_stage_mirror_l%d", il);
            ggml_format_name(layer.mirror_v_stage, "cache_kvarn_v_stage_mirror_l%d", il);
            layer.mirror_k_records_stream.reserve(n_stream);
            layer.mirror_v_records_stream.reserve(n_stream);
            layer.mirror_k_stage_stream.reserve(n_stream);
            layer.mirror_v_stage_stream.reserve(n_stream);
            for (uint32_t stream = 0; stream < n_stream; ++stream) {
                auto * kr = ggml_view_3d(
                    mirror_ctx, layer.mirror_k_records, k_record_size, n_head_k_sliced,
                    n_groups_per_stream, layer.mirror_k_records->nb[1],
                    layer.mirror_k_records->nb[2],
                    size_t(stream) * n_groups_per_stream * layer.mirror_k_records->nb[2]);
                auto * vr = ggml_view_3d(
                    mirror_ctx, layer.mirror_v_records, v_record_size, n_head_v_sliced,
                    n_groups_per_stream, layer.mirror_v_records->nb[1],
                    layer.mirror_v_records->nb[2],
                    size_t(stream) * n_groups_per_stream * layer.mirror_v_records->nb[2]);
                auto * ks = ggml_view_3d(
                    mirror_ctx, layer.mirror_k_stage, k_record_dim, n_head_k_sliced,
                    KVAR_N_GROUP * stage_groups, layer.mirror_k_stage->nb[1],
                    layer.mirror_k_stage->nb[2],
                    size_t(stream) * KVAR_N_GROUP * stage_groups * layer.mirror_k_stage->nb[2]);
                auto * vs = ggml_view_3d(
                    mirror_ctx, layer.mirror_v_stage, v_record_dim, n_head_v_sliced,
                    KVAR_N_GROUP * stage_groups, layer.mirror_v_stage->nb[1],
                    layer.mirror_v_stage->nb[2],
                    size_t(stream) * KVAR_N_GROUP * stage_groups * layer.mirror_v_stage->nb[2]);
                ggml_format_name(kr, "cache_kvarn_k_records_mirror_l%d_s%d", il, stream);
                ggml_format_name(vr, "cache_kvarn_v_records_mirror_l%d_s%d", il, stream);
                ggml_format_name(ks, "cache_kvarn_k_stage_mirror_l%d_s%d", il, stream);
                ggml_format_name(vs, "cache_kvarn_v_stage_mirror_l%d_s%d", il, stream);
                layer.mirror_k_records_stream.push_back(kr);
                layer.mirror_v_records_stream.push_back(vr);
                layer.mirror_k_stage_stream.push_back(ks);
                layer.mirror_v_stage_stream.push_back(vs);
            }
            const bool mirror_native_tail = exact_tail_tokens == 0 ||
                (!explicit_bias && kvarn_backend_supports_native_tail(
                    mirror_dev, exact_tail_type, head_dim_k, head_dim_v));
            layer.mirror_native_attention =
                llama_kvarn_backend_supports_native_ops(mirror_dev) && mirror_native_tail;
            layer.mirror_mixed_tail_native = layer.mirror_native_attention &&
                llama_kvarn_backend_mixed_tail_native_preferred(mirror_dev);
            layer.mirror_native_original_v = layer.mirror_native_attention &&
                llama_kvarn_backend_native_attention_uses_original_v(mirror_dev);
            layer.mirror_native_rotated_max_query_tokens = layer.mirror_native_attention ?
                llama_kvarn_backend_native_rotated_max_query_tokens(mirror_dev) : 0;
        }

        raw_bytes += size_t(kv_size) * n_stream * n_head_kv * (head_dim_k + head_dim_v) * sizeof(ggml_fp16_t);
    }

    if (reuse) {
        for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
            const int32_t il_reuse = reuse(il);
            if (il_reuse < 0 || !hparams.has_kv(il) || hparams.is_recr(il)) {
                continue;
            }
            if (filter && !filter(il)) {
                continue;
            }
            const auto src = map_layer_ids.find(il_reuse);
            if (src == map_layer_ids.end()) {
                throw std::runtime_error(format("KVarN cache layer %u cannot reuse missing layer %d", il, il_reuse));
            }

            const auto & reused = layers.at(src->second);
            if (hparams.n_head_kv(il) != reused.n_head_kv ||
                hparams.n_embd_head_k(il) != reused.head_dim_k ||
                hparams.n_embd_head_v(il) != reused.head_dim_v) {
                throw std::runtime_error(format(
                    "KVarN cache layer %u cannot reuse layer %d with different KV shape",
                    il, il_reuse));
            }

            map_layer_ids[il] = src->second;
        }
    }

    size_t total_bytes = 0;
    for (auto & [key, ctx] : ctx_map) {
        const auto buft = key.first;
        ggml_backend_buffer_t buf;
        if (hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, 0);
            for (auto * tensor = ggml_get_first_tensor(ctx.get()); tensor != nullptr; tensor = ggml_get_next_tensor(ctx.get(), tensor)) {
                tensor->buffer = buf;
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate KVarN cache buffer");
        }

        ggml_backend_buffer_clear(buf, 0);
        total_bytes += ggml_backend_buffer_get_size(buf);
        LLAMA_LOG_INFO("%s: %10s KVarN buffer size = %8.2f MiB\n",
                __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf) / 1024.0 / 1024.0);
        ctxs_bufs.push_back({ key.second, buft, std::move(ctx), ggml_backend_buffer_ptr(buf) });
    }

    const auto tensor_buft = [](const ggml_tensor * tensor) -> ggml_backend_buffer_type_t {
        return tensor && tensor->buffer ? ggml_backend_buffer_get_type(tensor->buffer) : nullptr;
    };
    // Records and stages are always the persistent KVarN body, including when
    // no enlarged exact tail was requested.  Validate their complete-head
    // contract once for every construction path so tail=0 cannot bypass the
    // same fail-closed placement guarantees.
    for (const auto & layer : layers) {
        const auto k_buft = tensor_buft(layer.k_records);
        const auto v_buft = tensor_buft(layer.v_records);
        if (!k_buft || !v_buft) {
            throw std::runtime_error(format("KVarN body layer %u has no realized backend buffer", layer.il));
        }
        if (k_buft != v_buft) {
            throw std::runtime_error(format(
                    "KVarN body layer %u places K and V records on different owners", layer.il));
        }
        auto * dev = ggml_backend_buft_get_device(k_buft);
        if (ggml_backend_dev_is_meta(dev)) {
            for (const auto & component : {
                    std::pair<const ggml_tensor *, const char *> { layer.k_records, "K records" },
                    std::pair<const ggml_tensor *, const char *> { layer.v_records, "V records" },
                    std::pair<const ggml_tensor *, const char *> { layer.k_stage,   "K stage" },
                    std::pair<const ggml_tensor *, const char *> { layer.v_stage,   "V stage" } }) {
                const auto split = llama_meta_device_get_split_state(
                        component.first,
                        const_cast<llama_meta_device_get_split_state_userdata *>(&model.get_split_state_ud));
                if (split.axis != GGML_BACKEND_SPLIT_AXIS_1 || split.n_segments == 0) {
                    throw std::runtime_error(format(
                            "KVarN tensor/meta split is invalid for layer %u %s: expected complete-head axis 1, got %s",
                            layer.il, component.second,
                            ggml_backend_meta_split_axis_name(split.axis)));
                }
            }
        }
    }

    uint32_t exact_slots = 0;
    std::vector<llama_kv_tail_layer_route> tail_routes;
    if (exact_tail_tokens > 0) {

        for (const auto & layer_entry : map_layer_ids) {
            // plain locals instead of structured bindings: fail_route below
            // captures logical_il, which C++17 does not allow for structured bindings
            const auto logical_il  = layer_entry.first;
            const auto layer_index = layer_entry.second;
            const auto & layer = layers.at(layer_index);
            auto * buft = tensor_buft(layer.k_records);
            auto * dev = ggml_backend_buft_get_device(buft);
            const char * backend = dev ? ggml_backend_dev_name(dev) : ggml_backend_buft_name(buft);
            const bool device_route = dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU;
            const bool explicit_bias = model.self_attention_uses_explicit_bias(uint32_t(logical_il));
            const auto fail_route = [&](llama_kv_tail_operation operation) {
                throw std::runtime_error(format(
                        "KVarN exact-tail route is unsupported for architecture %s group %s layer %u backend %s "
                        "body f16/f16 exact %s/%s: missing %s",
                        llm_arch_name(model.arch), swa ? "swa" : "full", uint32_t(logical_il),
                        backend ? backend : "unknown",
                        ggml_type_name(exact_tail_type), ggml_type_name(exact_tail_type),
                        llama_kv_tail_operation_name(operation)));
            };
            llama_kv_tail_route tail_route = LLAMA_KV_TAIL_ROUTE_GENERIC;
            if (device_route) {
                if (!kvarn_backend_supports_tail_write(
                            dev, exact_tail_type, uint64_t(layer.head_dim_k)*layer.n_head_kv)) {
                    fail_route(LLAMA_KV_TAIL_OP_WRITE_K);
                }
                if (!kvarn_backend_supports_tail_write(
                            dev, exact_tail_type, uint64_t(layer.head_dim_v)*layer.n_head_kv)) {
                    fail_route(LLAMA_KV_TAIL_OP_WRITE_V);
                }
            }
            if (layer.native_attention && !explicit_bias) {
                if (!kvarn_backend_supports_native_tail(
                            dev, exact_tail_type, layer.head_dim_k, layer.head_dim_v)) {
                    fail_route(LLAMA_KV_TAIL_OP_NATIVE_ATTENTION);
                }
                tail_route = LLAMA_KV_TAIL_ROUTE_NATIVE;
            }
            tail_routes.push_back({
                    uint32_t(logical_il), backend ? backend : "unknown",
                    GGML_TYPE_F16, GGML_TYPE_F16, exact_tail_type, exact_tail_type,
                    false, hparams.causal_attn, swa, explicit_bias, true,
                    tail_route == LLAMA_KV_TAIL_ROUTE_NATIVE,
                    llama_kv_tail_packed_body_stride(
                            uint64_t(exact_tail_tokens) + metadata->get_tail_rollback_tokens(), 256),
                    dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU),
                    { true, tail_route, LLAMA_KV_TAIL_OP_NONE },
            });
        }

        if (!layers.empty()) {
            for (const auto & route : tail_routes) {
                LLAMA_LOG_INFO("KV tail: group=%s layer=%u dev=%s route=%s body=kvarn/%s exact=%s/%s "
                        "presence=%s current=%s execution_rows=%u requested=%u effective=%u\n",
                        swa ? "swa" : "full", route.layer_id, route.backend.c_str(),
                        route.capability.route == LLAMA_KV_TAIL_ROUTE_NATIVE ? "native" : "generic",
                        "kvarn", ggml_type_name(route.exact_type_k), ggml_type_name(route.exact_type_v),
                        route.has_body ? "body" : "bodyless", route.has_current ? "current" : "no-current",
                        route.body_execution_rows, exact_tail_tokens_requested, exact_tail_tokens);
            }
            metadata->set_tail_routes(std::move(tail_routes));
            metadata->finalize_tail_overlay_metadata();
            if (migration_enabled) {
                migration_local_tail_routes = metadata->get_tail_layer_routes();
                migration_remote_tail_routes = migration_local_tail_routes;
                for (auto & route : migration_remote_tail_routes) {
                    const auto & layer = layer_for(route.layer_id);
                    if (layer.mirror_dev == nullptr) {
                        continue;
                    }
                    route.backend = ggml_backend_dev_name(layer.mirror_dev);
                    route.owner = layer.mirror_dev;
                    route.capability = {
                        true,
                        layer.mirror_native_attention && !route.explicit_bias
                            ? LLAMA_KV_TAIL_ROUTE_NATIVE
                            : LLAMA_KV_TAIL_ROUTE_GENERIC,
                        LLAMA_KV_TAIL_OP_NONE,
                    };
                }
            }
            exact_slots = metadata->get_tail_slots();
            if (exact_slots == 0) {
                throw std::logic_error("KVarN exact-tail metadata finalized without storage slots");
            }

            std::map<ctx_key, ggml_context_ptr, ctx_key_comparator> tail_ctx_map;
            const auto tail_ctx_for_buft = [&](ggml_backend_buffer_type_t buft,
                                               int32_t migration_layer = -1) -> ggml_context * {
                const ctx_key key { buft, migration_layer };
                const auto it = tail_ctx_map.find(key);
                if (it != tail_ctx_map.end()) {
                    return it->second.get();
                }
                ggml_init_params ctx_params = {
                    /*.mem_size   =*/ size_t(2u*hparams.n_layer_kv()*ggml_tensor_overhead()),
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ggml_context_ptr ctx { ggml_init(ctx_params) };
                if (!ctx) {
                    return nullptr;
                }
                auto * result = ctx.get();
                tail_ctx_map.emplace(key, std::move(ctx));
                return result;
            };

            for (auto & layer : layers) {
                auto * buft = tensor_buft(layer.k_records);
                const int32_t migration_layer = layer.mirror_dev != nullptr ? int32_t(layer.il) : -1;
                auto * ctx = tail_ctx_for_buft(buft, migration_layer);
                if (!ctx) {
                    throw std::runtime_error("failed to create KVarN exact-tail tensor context");
                }
                layer.k_tail = ggml_new_tensor_2d(
                        ctx, exact_tail_type, uint64_t(layer.head_dim_k)*layer.n_head_kv, exact_slots);
                layer.v_tail = ggml_new_tensor_2d(
                        ctx, exact_tail_type, uint64_t(layer.head_dim_v)*layer.n_head_kv, exact_slots);
                ggml_format_name(layer.k_tail, "cache_kvarn_k_tail_l%d", layer.il);
                ggml_format_name(layer.v_tail, "cache_kvarn_v_tail_l%d", layer.il);
                if (layer.mirror_dev != nullptr) {
                    auto * mirror_ctx = tail_ctx_for_buft(
                            ggml_backend_dev_buffer_type(layer.mirror_dev), int32_t(layer.il));
                    if (!mirror_ctx) {
                        throw std::runtime_error("failed to create KVarN migration mirror tail context");
                    }
                    layer.mirror_k_tail = ggml_new_tensor_2d(
                            mirror_ctx, exact_tail_type,
                            uint64_t(layer.head_dim_k)*layer.n_head_kv, exact_slots);
                    layer.mirror_v_tail = ggml_new_tensor_2d(
                            mirror_ctx, exact_tail_type,
                            uint64_t(layer.head_dim_v)*layer.n_head_kv, exact_slots);
                    ggml_format_name(layer.mirror_k_tail, "cache_kvarn_k_tail_mirror_l%d", layer.il);
                    ggml_format_name(layer.mirror_v_tail, "cache_kvarn_v_tail_mirror_l%d", layer.il);
                }
            }

            for (auto & [key, ctx] : tail_ctx_map) {
                const auto buft = key.first;
                ggml_backend_buffer_t buf;
                if (hparams.no_alloc) {
                    buf = ggml_backend_buft_alloc_buffer(buft, 0);
                    for (auto * tensor = ggml_get_first_tensor(ctx.get()); tensor != nullptr; tensor = ggml_get_next_tensor(ctx.get(), tensor)) {
                        tensor->buffer = buf;
                    }
                } else {
                    buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
                }
                if (!buf) {
                    throw std::runtime_error("failed to allocate KVarN exact-tail buffer");
                }
                ggml_backend_buffer_clear(buf, 0);
                total_bytes += ggml_backend_buffer_get_size(buf);
                LLAMA_LOG_INFO("%s: %10s KVarN tail buffer size = %8.2f MiB\n",
                        __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
                ctxs_bufs.push_back({ key.second, buft, std::move(ctx), ggml_backend_buffer_ptr(buf) });
            }

            for (const auto & layer : layers) {
                const uintptr_t owner = reinterpret_cast<uintptr_t>(tensor_buft(layer.k_records));
                auto ownership = llama_kv_tail_plan_layer_ownership(
                        layer.il, owner, owner, true, true);
                ownership.shadow_k_owner = reinterpret_cast<uintptr_t>(tensor_buft(layer.k_tail));
                ownership.shadow_v_owner = reinterpret_cast<uintptr_t>(tensor_buft(layer.v_tail));
                const auto error = llama_kv_tail_validate_layer_ownership(ownership);
                if (error != LLAMA_KV_TAIL_OWNERSHIP_OK) {
                    throw std::runtime_error(format("KVarN exact-tail ownership validation failed for layer %u (error %d)",
                            layer.il, int(error)));
                }
            }
        }
    } else {
        metadata->set_tail_routes({});
    }

    if (migration_enabled) {
        ggml_backend_dev_t prefill_dev = nullptr;
        ggml_backend_dev_t mirror_dev = nullptr;
        for (const auto & layer : layers) {
            if (layer.mirror_dev == nullptr) continue;
            auto * layer_prefill_dev = ggml_backend_buft_get_device(
                    ggml_backend_buffer_get_type(layer.k_records->buffer));
            if (prefill_dev == nullptr) {
                prefill_dev = layer_prefill_dev;
                mirror_dev = layer.mirror_dev;
            } else if (prefill_dev != layer_prefill_dev || mirror_dev != layer.mirror_dev) {
                throw std::runtime_error("KVarN prefill migration currently requires one CUDA/Vulkan device pair");
            }
        }
        if (hparams.no_alloc || prefill_dev == nullptr || mirror_dev == nullptr ||
                strncmp(ggml_backend_dev_name(prefill_dev), "CUDA", 4) != 0 ||
                strncmp(ggml_backend_dev_name(mirror_dev), "Vulkan", 6) != 0) {
            throw std::runtime_error(
                "KVarN prefill migration requires allocated CUDA payloads and a Vulkan mirror");
        }
        migration_prefill_dev = prefill_dev;
        migration_remote_dev = mirror_dev;
        migration_copies = std::make_unique<migration_queue>(prefill_dev, mirror_dev);
        migration_mirror_stale = false;
        LLAMA_LOG_INFO("KVarN prefill migration enabled: CUDA owner with Vulkan mirror; "
                "ownership changes only at explicit request boundaries\n");
    }

    LLAMA_LOG_INFO("%s: type = %s, layers = %zu, groups/stream = %u, streams = %u, KVarN = %.2f MiB, equivalent F16 = %.2f MiB\n",
            __func__, llama_kvarn_type_name(this->params.type), layers.size(), n_groups_per_stream, n_stream,
            total_bytes / 1024.0 / 1024.0, raw_bytes / 1024.0 / 1024.0);
}

void llama_kv_cache_kvarn::ensure_migration_mirror_storage(layer & cache_layer) {
    GGML_ASSERT(migration_enabled && migration_prefill_dev != nullptr && migration_remote_dev != nullptr);
    const auto target_dev = migration_remote_active ? migration_prefill_dev : migration_remote_dev;
    const auto target_buft = ggml_backend_dev_buffer_type(target_dev);
    bool found = false;
    bool allocated = false;

    for (auto & storage : ctxs_bufs) {
        if (storage.migration_layer != int32_t(cache_layer.il) ||
                ggml_backend_buft_get_device(storage.buft) != target_dev) {
            continue;
        }
        found = true;
        if (storage.buffer) {
            continue;
        }
        if (storage.buft != target_buft) {
            throw std::runtime_error(format(
                    "KVarN migration layer %u has a detached buffer on an unexpected owner",
                    cache_layer.il));
        }
        const auto detach_tensors = [&]() {
            for (auto * tensor = ggml_get_first_tensor(storage.ctx.get());
                    tensor != nullptr; tensor = ggml_get_next_tensor(storage.ctx.get(), tensor)) {
                tensor->data = nullptr;
                tensor->buffer = nullptr;
            }
        };
        // A failed GGML allocation can free partially-created backend buffers
        // without detaching the tensors it already initialized. Reset the
        // descriptor bundle before every retry and again on failure.
        detach_tensors();
        auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(storage.ctx.get(), storage.buft);
        if (buffer == nullptr) {
            detach_tensors();
            throw std::runtime_error(format(
                    "failed to reallocate KVarN migration storage for layer %u on %s",
                    cache_layer.il, ggml_backend_dev_name(target_dev)));
        }
        storage.buffer.reset(buffer);
        ggml_backend_buffer_clear(storage.buffer.get(), 0);
        allocated = true;
    }

    if (!found) {
        throw std::runtime_error(format(
                "KVarN migration layer %u has no allocation bundle for %s",
                cache_layer.il, ggml_backend_dev_name(target_dev)));
    }
    if (allocated) {
        // A newly allocated target has no complete cache image. The handoff
        // must copy all records, not only groups copied during this prefill.
        migration_mirror_stale = true;
    }
}

void llama_kv_cache_kvarn::release_migration_layer_storage(
        layer & cache_layer, ggml_backend_dev_t device) noexcept {
    for (auto & storage : ctxs_bufs) {
        if (storage.migration_layer != int32_t(cache_layer.il) ||
                ggml_backend_buft_get_device(storage.buft) != device || !storage.buffer) {
            continue;
        }
        // Keep tensor descriptors and their view relationships so the same
        // context can be reallocated on the next reverse handoff. The GGML
        // allocator skips tensors with non-null data, so detach every base and
        // view before freeing the backing buffer.
        for (auto * tensor = ggml_get_first_tensor(storage.ctx.get());
                tensor != nullptr; tensor = ggml_get_next_tensor(storage.ctx.get(), tensor)) {
            tensor->data = nullptr;
            tensor->buffer = nullptr;
        }
        storage.buffer.reset();
    }
}

void llama_kv_cache_kvarn::release_prefill_migration_inactive_buffers() noexcept {
    if (!migration_enabled) {
        return;
    }
    if (migration_copies && !migration_copies->drain()) {
        migration_mirror_stale = true;
    }
    const auto inactive_dev = migration_remote_active ? migration_prefill_dev : migration_remote_dev;
    size_t released_bytes = 0;
    for (auto & storage : ctxs_bufs) {
        if (storage.migration_layer >= 0 && storage.buffer &&
                ggml_backend_buft_get_device(storage.buft) == inactive_dev) {
            released_bytes += ggml_backend_buffer_get_size(storage.buffer.get());
        }
    }
    for (auto & cache_layer : layers) {
        if (cache_layer.mirror_dev != nullptr) {
            release_migration_layer_storage(cache_layer, inactive_dev);
        }
    }
    migration_mirror_stale = true;
    if (released_bytes > 0) {
        LLAMA_LOG_INFO("KVarN migration released inactive %s cache buffers: %.2f MiB\n",
                ggml_backend_dev_name(inactive_dev), released_bytes / 1024.0 / 1024.0);
    }
}

void llama_kv_cache_kvarn::enqueue_prefill_migration(const llama_kv_cache::slot_info & sinfo,
                                                     ggml_backend_sched_t              sched) {
    if (!migration_enabled || !migration_copies || sinfo.empty()) {
        return;
    }
    // Once Vulkan owns the live payload, the CUDA mirror is deliberately left
    // untouched. In particular, never wait on a Vulkan scheduler event from
    // this transfer worker: Vulkan event synchronization recycles producer
    // command buffers and races the inference thread.
    if (migration_remote_active) {
        migration_mirror_stale = true;
        return;
    }

    try {
        for (auto & layer : layers) {
            if (layer.mirror_dev != nullptr) {
                ensure_migration_mirror_storage(layer);
            }
        }
        if (swa || n_stream != 1 || sinfo.n_stream() != 1) {
            throw std::runtime_error("KVarN prefill migration does not support SWA or multiple streams");
        }

        std::set<uint32_t> sealed_groups;
        for (const uint32_t cell : sinfo.idxs[0]) {
            if (cell % KVAR_N_GROUP == KVAR_N_GROUP - 1u) {
                const uint32_t group = cell / KVAR_N_GROUP;
                // Group zero is the permanent sink and remains in the F16 stage.
                if (group > 0 && group < n_groups_per_stream) {
                    sealed_groups.insert(group);
                }
            }
        }
        if (sealed_groups.empty()) {
            return;
        }

        std::vector<migration_queue::span> spans;
        ggml_backend_t                     producer = nullptr;
        for (const auto & layer : layers) {
            if (layer.mirror_dev == nullptr) {
                continue;
            }
            ggml_tensor *  source_probe   = layer.k_records;
            ggml_backend_t layer_producer = ggml_backend_sched_get_tensor_backend(sched, source_probe);
            if (layer_producer == nullptr) {
                layer_producer = ggml_backend_sched_get_tensor_backend(sched, layer.k_stage);
            }
            if (layer_producer == nullptr ||
                ggml_backend_get_device(layer_producer) !=
                    ggml_backend_buft_get_device(ggml_backend_buffer_get_type(source_probe->buffer))) {
                throw std::runtime_error(
                    format("KVarN migration cannot resolve the producer backend for layer %u", layer.il));
            }
            if (producer == nullptr) {
                producer = layer_producer;
            } else if (ggml_backend_get_device(producer) != ggml_backend_get_device(layer_producer)) {
                throw std::runtime_error("KVarN migration spans more than one producer device");
            }
            const auto append_records = [&](ggml_tensor * src, ggml_tensor * dst) {
                const size_t group_bytes = src->nb[2];
                auto         it          = sealed_groups.begin();
                while (it != sealed_groups.end()) {
                    const uint32_t begin = *it;
                    uint32_t       end   = begin + 1;
                    ++it;
                    while (it != sealed_groups.end() && *it == end) {
                        ++end;
                        ++it;
                    }
                    spans.push_back({
                        src,
                        dst,
                        size_t(begin) * group_bytes,
                        size_t(begin) * dst->nb[2],
                        size_t(end - begin) * group_bytes,
                    });
                }
            };
            append_records(layer.k_records_stream[0], layer.mirror_k_records_stream[0]);
            append_records(layer.v_records_stream[0], layer.mirror_v_records_stream[0]);
        }
        // One CUDA producer event orders the newly sealed records behind this
        // ubatch's cache stores while keeping the transfer queue bounded.
        if (!spans.empty() && !migration_copies->enqueue(producer, true, std::move(spans))) {
            migration_mirror_stale = true;
            LLAMA_LOG_WARN("%s: CUDA-to-Vulkan cache mirror queue is unavailable; retaining CUDA ownership\n",
                           __func__);
        }
    } catch (const std::exception & e) {
        migration_mirror_stale = true;
        release_prefill_migration_inactive_buffers();
        LLAMA_LOG_ERROR("%s: cache mirror copy was skipped: %s; retaining CUDA ownership\n", __func__, e.what());
    } catch (...) {
        migration_mirror_stale = true;
        release_prefill_migration_inactive_buffers();
        LLAMA_LOG_ERROR("%s: cache mirror copy was skipped; retaining CUDA ownership\n", __func__);
    }
}

bool llama_kv_cache_kvarn::handoff_prefill_migration(bool to_remote) {
    if (!migration_enabled || !migration_copies) return false;
    if (migration_remote_active == to_remote) return true;
    if (swa || n_stream != 1) return false;

    const int64_t migration_start_us = ggml_time_us();
    const uint64_t copied_before = migration_copies->copied_bytes();
    try {
        if (!migration_copies->drain()) {
            migration_mirror_stale = true;
            LLAMA_LOG_WARN("%s: migration queue failed; keeping current %s cache owner\n",
                    __func__, migration_remote_active ? "Vulkan" : "CUDA");
            return false;
        }
        for (auto & layer : layers) {
            if (layer.mirror_dev != nullptr) {
                ensure_migration_mirror_storage(layer);
            }
        }
        std::vector<migration_queue::span> spans;
        for (const auto & layer : layers) {
            if (layer.mirror_dev == nullptr) continue;
            if (migration_mirror_stale) {
                spans.push_back({
                    layer.k_records, layer.mirror_k_records, 0, 0,
                    size_t(ggml_nbytes(layer.k_records)),
                });
                spans.push_back({
                    layer.v_records, layer.mirror_v_records, 0, 0,
                    size_t(ggml_nbytes(layer.v_records)),
                });
            }
            // The sink, partial records, rollback window, and exact FP16 tail
            // are mutable during prefill/decode. Snapshot them at the boundary.
            spans.push_back({
                layer.k_stage, layer.mirror_k_stage, 0, 0,
                size_t(ggml_nbytes(layer.k_stage)),
            });
            spans.push_back({
                layer.v_stage, layer.mirror_v_stage, 0, 0,
                size_t(ggml_nbytes(layer.v_stage)),
            });
            if (layer.k_tail != nullptr) {
                spans.push_back({
                    layer.k_tail, layer.mirror_k_tail, 0, 0,
                    size_t(ggml_nbytes(layer.k_tail)),
                });
                spans.push_back({
                    layer.v_tail, layer.mirror_v_tail, 0, 0,
                    size_t(ggml_nbytes(layer.v_tail)),
                });
            }
        }
        if (!migration_copies->enqueue(nullptr, !migration_remote_active, std::move(spans)) ||
                !migration_copies->drain()) {
            migration_mirror_stale = true;
            LLAMA_LOG_WARN("%s: payload copy failed; keeping current %s cache owner\n",
                    __func__, migration_remote_active ? "Vulkan" : "CUDA");
            return false;
        }

        metadata->rebind_tail_routes(to_remote
                ? migration_remote_tail_routes
                : migration_local_tail_routes);
        for (auto & layer : layers) {
            if (layer.mirror_dev == nullptr) continue;
            std::swap(layer.k_records, layer.mirror_k_records);
            std::swap(layer.v_records, layer.mirror_v_records);
            std::swap(layer.k_stage, layer.mirror_k_stage);
            std::swap(layer.v_stage, layer.mirror_v_stage);
            std::swap(layer.k_tail, layer.mirror_k_tail);
            std::swap(layer.v_tail, layer.mirror_v_tail);
            std::swap(layer.k_records_stream, layer.mirror_k_records_stream);
            std::swap(layer.v_records_stream, layer.mirror_v_records_stream);
            std::swap(layer.k_stage_stream, layer.mirror_k_stage_stream);
            std::swap(layer.v_stage_stream, layer.mirror_v_stage_stream);
            std::swap(layer.native_attention, layer.mirror_native_attention);
            std::swap(layer.mixed_tail_native, layer.mirror_mixed_tail_native);
            std::swap(layer.native_original_v, layer.mirror_native_original_v);
            std::swap(layer.native_rotated_max_query_tokens,
                    layer.mirror_native_rotated_max_query_tokens);
        }
        migration_remote_active = to_remote;
        migration_mirror_stale = false;
        const uint64_t copied_after = migration_copies->copied_bytes();
        LLAMA_LOG_INFO("%s: KVarN migration handoff complete: owner=%s, bytes=%llu, elapsed=%.2f ms\n",
                __func__, to_remote ? "Vulkan" : "CUDA",
                (unsigned long long) (copied_after - copied_before),
                (ggml_time_us() - migration_start_us) / 1000.0);
        return true;
    } catch (const std::exception & e) {
        migration_mirror_stale = true;
        release_prefill_migration_inactive_buffers();
        LLAMA_LOG_ERROR("KVarN prefill migration handoff failed: %s\n", e.what());
        return false;
    } catch (...) {
        migration_mirror_stale = true;
        release_prefill_migration_inactive_buffers();
        LLAMA_LOG_ERROR("KVarN prefill migration handoff failed with an unknown error\n");
        return false;
    }
}

bool llama_kv_cache_kvarn::drain_prefill_migration() {
    if (!migration_copies) return true;
    const bool ok = migration_copies->drain();
    if (!ok) migration_mirror_stale = true;
    return ok;
}

std::unique_ptr<llama_kv_cache> llama_kv_cache_kvarn::make_metadata_cache() const {
    auto result = std::make_unique<llama_kv_cache>(
            model,
            hparams,
            GGML_TYPE_F16,
            GGML_TYPE_F16,
            false,
            false,
            n_stream == 1,
            kv_size,
            n_seq_max,
            metadata_n_pad,
            metadata_n_swa,
            metadata_swa_type,
            nullptr,
            [](int32_t) { return false; },
            nullptr,
            nullptr,
            "",
            metadata_n_ubatch,
            exact_tail_tokens,
            exact_tail_type_requested,
            exact_tail_tokens_requested,
            true,
            metadata->get_tail_rollback_tokens());

    const auto & routes = metadata->get_tail_layer_routes();
    result->set_tail_routes(std::vector<llama_kv_tail_layer_route>(routes.begin(), routes.end()));
    if (!routes.empty()) {
        result->finalize_tail_overlay_metadata();
    }
    if (!swa) {
        result->set_allocation_group_size(KVAR_N_GROUP, n_stream == 1 ? tail_groups : 1u);
    }
    return result;
}

llama_memory_context_ptr llama_kv_cache_kvarn::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    return std::make_unique<llama_kv_cache_kvarn_context>(
        this, metadata->init_batch(balloc, n_ubatch, embd_all));
}

llama_memory_context_ptr llama_kv_cache_kvarn::init_full() {
    return std::make_unique<llama_kv_cache_kvarn_context>(this, metadata->init_full());
}

llama_memory_context_ptr llama_kv_cache_kvarn::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_kv_cache_kvarn_context>(this, metadata->init_update(lctx, optimize), lctx);
}

uint32_t llama_kv_cache_kvarn::get_kv_n_stream() const {
    return metadata->get_n_stream();
}

uint32_t llama_kv_cache_kvarn::get_kv_size() const {
    return metadata->get_size();
}

llama_memory_context_ptr llama_kv_cache_kvarn::init_kv_batch(const std::vector<llama_ubatch> & ubatches) {
    auto sinfos = metadata->prepare(ubatches);
    if (sinfos.empty()) {
        return std::make_unique<llama_kv_cache_kvarn_context>(
                this, std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE));
    }

    return std::make_unique<llama_kv_cache_kvarn_context>(
            this, std::make_unique<llama_kv_cache_context>(metadata.get(), std::move(sinfos), ubatches));
}

bool llama_kv_cache_kvarn::get_can_shift() const {
    return false;
}

llama_memory_i::seq_rm_capability llama_kv_cache_kvarn::get_seq_rm_capability() const {
    // Metadata can discard an unbounded suffix because KVarN owns the body, but
    // this cache only guarantees exact in-place rollback through the current
    // and previous staged record.
    return llama_memory_clamp_suffix_rollback_capability(
            metadata->get_seq_rm_capability(), KVAR_N_GROUP);
}

void llama_kv_cache_kvarn::clear(bool data) {
    drain_prefill_migration();
    pending_stream_copies = {};
    if (standard_cache) {
        if (state_owner_epoch == UINT64_MAX) {
            throw std::overflow_error("mixed KV state owner epoch exhausted");
        }
        ++state_owner_epoch;
    }
    if (standard_cache) standard_cache->clear(data);
    metadata->clear(false);
    if (data) {
        for (auto & storage : ctxs_bufs) {
            if (storage.buffer) {
                ggml_backend_buffer_clear(storage.buffer.get(), 0);
            }
        }
    }
    migration_mirror_stale = !data;
}

bool llama_kv_cache_kvarn::can_remove(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    if (seq_id < 0) {
        return p0 <= 0 && p1 < 0;
    }

    // seq_rm() delegates the mutation to the metadata cache, which owns the
    // compact precision tail and its much tighter suffix-rollback reserve. The
    // capability query has to honour that too: callers treat can_seq_rm() as a
    // promise, and the server turns a seq_rm() that fails after can_seq_rm()
    // succeeded into a hard error instead of falling back to reprocessing.
    if (!metadata->can_seq_rm(seq_id, p0, p1)) {
        return false;
    }

    const llama_pos pos_max = metadata->seq_pos_max(seq_id);
    if (llama_kvarn_can_remove_range(pos_max, p0, p1, KVAR_N_GROUP)) {
        return true;
    }
    if (p0 <= 0 || p1 >= 0) {
        return false;
    }

    llama_pos planned_p0 = -1;
    llama_pos planned_p1 = -1;
    return llama_kvarn_plan_remove_range(
            pos_max, p0, p1, KVAR_N_GROUP,
            stream_is_exclusive_for(seq_id), planned_p0, planned_p1) &&
           planned_p0 == p0 && planned_p1 == p1;
}

bool llama_kv_cache_kvarn::can_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    return can_remove(seq_id, p0, p1);
}

bool llama_kv_cache_kvarn::seq_rm_plan(
        llama_seq_id seq_id, llama_pos p0, llama_pos p1,
        llama_pos & planned_p0, llama_pos & planned_p1) const {
    if (seq_id < 0) {
        if (p0 > 0 || p1 >= 0) {
            return false;
        }
        planned_p0 = p0;
        planned_p1 = p1;
        return true;
    }
    if (uint32_t(seq_id) >= n_seq_max) {
        return false;
    }
    if (llama_kvarn_can_remove_range(
            metadata->seq_pos_max(seq_id), p0, p1, KVAR_N_GROUP)) {
        if (!metadata->can_seq_rm(seq_id, p0, p1)) {
            return false;
        }
        planned_p0 = p0;
        planned_p1 = p1;
        return true;
    }
    if (p0 <= 0 || p1 >= 0) {
        return false;
    }
    // A widened plan still has to survive the metadata cache's own tail rules.
    return llama_kvarn_plan_remove_range(
            metadata->seq_pos_max(seq_id), p0, p1, KVAR_N_GROUP,
            stream_is_exclusive_for(seq_id), planned_p0, planned_p1) &&
           metadata->can_seq_rm(seq_id, planned_p0, planned_p1);
}

bool llama_kv_cache_kvarn::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    drain_prefill_migration();
    apply_pending_stream_copies(nullptr);
    if (!can_seq_rm(seq_id, p0, p1)) {
        const bool valid_seq = seq_id >= 0 && uint32_t(seq_id) < n_seq_max;
        const llama_pos pos_min = valid_seq ? metadata->seq_pos_min(seq_id) : -1;
        const llama_pos pos_max = valid_seq ? metadata->seq_pos_max(seq_id) : -1;
        const llama_pos earliest_exact =
                std::max<llama_pos>(0, pos_max / llama_pos(KVAR_N_GROUP) - 1) *
                llama_pos(KVAR_N_GROUP);
        LLAMA_LOG_WARN("%s: KVarN can only remove a complete sequence or the current/previous fp16 tail groups "
                       "(seq_id = %d, p0 = %d, p1 = %d, seq_pos_min = %d, seq_pos_max = %d, "
                       "group = %u, earliest_exact = %d, meta_can_seq_rm = %d)\n",
                       __func__, seq_id, p0, p1, pos_min, pos_max,
                       unsigned(KVAR_N_GROUP), earliest_exact,
                       int(metadata->can_seq_rm(seq_id, p0, p1)));
        return false;
    }
    const bool removed = metadata->seq_rm(seq_id, p0, p1);
    if (removed && migration_enabled) migration_mirror_stale = true;
    return removed;
}

bool llama_kv_cache_kvarn::seq_rm_cell(llama_seq_id seq_id, uint32_t cell_idx) {
    drain_prefill_migration();
    apply_pending_stream_copies(nullptr);
    if (swa) {
        // SWA ring: the metadata cache manages window eviction; records follow the ring.
        const bool removed = metadata->seq_rm_cell(seq_id, cell_idx);
        if (removed && migration_enabled) migration_mirror_stale = true;
        return removed;
    }
    const llama_pos pos_max = metadata->seq_pos_max(seq_id);
    if (pos_max >= 0) {
        const uint32_t earliest_exact = uint32_t(std::max<llama_pos>(0, pos_max / KVAR_N_GROUP - 1) * KVAR_N_GROUP);
        if (cell_idx < earliest_exact) {
            return false;
        }
    }
    const bool removed = metadata->seq_rm_cell(seq_id, cell_idx);
    if (removed && migration_enabled) migration_mirror_stale = true;
    return removed;
}

int llama_kv_cache_kvarn::cells_at_pos(llama_seq_id seq_id, llama_pos pos, uint32_t * cell_indices, int n_max) {
    return metadata->cells_at_pos(seq_id, pos, cell_indices, n_max);
}

void llama_kv_cache_kvarn::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    drain_prefill_migration();
    apply_pending_stream_copies(nullptr);
    const uint32_t stream_src = metadata->get_stream_for_seq(seq_id_src);
    const uint32_t stream_dst = metadata->get_stream_for_seq(seq_id_dst);

    if (stream_src != stream_dst) {
        bool is_full = true;

        if (p0 > 0 && p0 + 1 < (int) get_kv_size()) {
            is_full = false;
        }

        if (p1 > 0 && p1 + 1 < (int) get_kv_size()) {
            is_full = false;
        }

        GGML_ASSERT(is_full && "KVarN cross-stream seq_cp() is only supported for full KV buffers");

        pending_stream_copies.ssrc.push_back(stream_src);
        pending_stream_copies.sdst.push_back(stream_dst);
    }

    metadata->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    auto tail_copies = metadata->take_pending_tail_copies();
    pending_stream_copies.tail_src_slots.insert(
            pending_stream_copies.tail_src_slots.end(),
            tail_copies.tail_src_slots.begin(), tail_copies.tail_src_slots.end());
    pending_stream_copies.tail_dst_slots.insert(
            pending_stream_copies.tail_dst_slots.end(),
            tail_copies.tail_dst_slots.begin(), tail_copies.tail_dst_slots.end());
    pending_stream_copies.tail_transaction =
            pending_stream_copies.tail_transaction || tail_copies.tail_transaction;
    if (migration_enabled) migration_mirror_stale = true;
}

void llama_kv_cache_kvarn::seq_keep(llama_seq_id seq_id) {
    drain_prefill_migration();
    apply_pending_stream_copies(nullptr);
    metadata->seq_keep(seq_id);
    if (migration_enabled) migration_mirror_stale = true;
}

GGML_NORETURN void llama_kv_cache_kvarn::seq_add(llama_seq_id, llama_pos, llama_pos, llama_pos) {
    GGML_ABORT("KVarN does not support position shifts");
}

GGML_NORETURN void llama_kv_cache_kvarn::seq_div(llama_seq_id, llama_pos, llama_pos, int) {
    GGML_ABORT("KVarN does not support position division");
}

llama_pos llama_kv_cache_kvarn::seq_pos_min(llama_seq_id seq_id) const {
    return metadata->seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_kvarn::seq_pos_max(llama_seq_id seq_id) const {
    return metadata->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_kvarn::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> result;
    for (const auto & storage : ctxs_bufs) {
        if (!storage.buffer) {
            continue;
        }
        auto * buft = storage.buft;
        result[buft] += hparams.no_alloc
            ? ggml_backend_alloc_ctx_tensors_from_buft_size(storage.ctx.get(), buft)
            : ggml_backend_buffer_get_size(storage.buffer.get());
    }
    if (standard_cache) {
        for (const auto & [buft, bytes] : standard_cache->memory_breakdown()) {
            result[buft] += bytes;
        }
    }
    return result;
}

bool llama_kv_cache_kvarn::requires_state_for_partial_restore() const {
    return true;
}

bool llama_kv_cache_kvarn::stream_is_exclusive_for(llama_seq_id seq_id) const {
    return llama_kvarn_stream_is_exclusive_for(
            n_stream, n_seq_max, seq_id,
            [&](llama_seq_id other) { return metadata->seq_pos_max(other); });
}

bool llama_kv_cache_kvarn::state_seq_can_save(llama_seq_id seq_id) const {
    return stream_is_exclusive_for(seq_id);
}

bool llama_kv_cache_kvarn::state_seq_can_restore(llama_seq_id seq_id) const {
    return stream_is_exclusive_for(seq_id);
}

bool llama_kv_cache_kvarn::state_seq_can_save(
        llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if (seq_id < 0) {
        return false;
    }
    constexpr uint32_t supported = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY |
            LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED;
    if ((uint32_t(flags) & ~supported) != 0 ||
            ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != 0 &&
             (flags & LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED) != 0)) {
        return false;
    }
    const bool selective = (flags & (LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY |
                                     LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED)) != 0;
    if (swa && !stream_is_exclusive_for(seq_id)) {
        return false;
    }
    return !has_pending_stream_copies() && (selective || stream_is_exclusive_for(seq_id));
}

bool llama_kv_cache_kvarn::state_seq_can_restore(
        llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if (seq_id < 0) {
        return false;
    }
    constexpr uint32_t supported = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY |
            LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED;
    if ((uint32_t(flags) & ~supported) != 0 ||
            ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != 0 &&
             (flags & LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED) != 0)) {
        return false;
    }
    const bool selective = (flags & (LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY |
                                     LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED)) != 0;
    if (swa && !stream_is_exclusive_for(seq_id)) {
        return false;
    }
    return !has_pending_stream_copies() && (selective || stream_is_exclusive_for(seq_id));
}

bool llama_kv_cache_kvarn::has_pending_stream_copies() const {
    return !pending_stream_copies.empty();
}

void llama_kv_cache_kvarn::copy_kvarn_stream(uint32_t stream_src, uint32_t stream_dst) {
    GGML_ASSERT(stream_src < n_stream);
    GGML_ASSERT(stream_dst < n_stream);
    GGML_ASSERT(stream_src != stream_dst);

    LLAMA_LOG_DEBUG("%s: copying KVarN stream %u to stream %u\n", __func__, stream_src, stream_dst);

    for (auto & layer : layers) {
        ggml_backend_tensor_copy(layer.k_records_stream[stream_src], layer.k_records_stream[stream_dst]);
        ggml_backend_tensor_copy(layer.v_records_stream[stream_src], layer.v_records_stream[stream_dst]);
        ggml_backend_tensor_copy(layer.k_stage_stream[stream_src], layer.k_stage_stream[stream_dst]);
        ggml_backend_tensor_copy(layer.v_stage_stream[stream_src], layer.v_stage_stream[stream_dst]);
    }
}

bool llama_kv_cache_kvarn::apply_pending_stream_copies(llama_context * lctx) {
    if (pending_stream_copies.empty()) {
        return true;
    }

    GGML_ASSERT(pending_stream_copies.ssrc.size() == pending_stream_copies.sdst.size());
    if (lctx) {
        llama_synchronize(lctx);
    }

    const size_t n_copy = pending_stream_copies.ssrc.size();
    for (size_t i = 0; i < n_copy; ++i) {
        copy_kvarn_stream(pending_stream_copies.ssrc[i], pending_stream_copies.sdst[i]);
    }

    const size_t n_tail_copy = pending_stream_copies.tail_src_slots.size();
    GGML_ASSERT(n_tail_copy == pending_stream_copies.tail_dst_slots.size());
    for (auto & layer : layers) {
        for (ggml_tensor * tensor : { layer.k_tail, layer.v_tail }) {
            if (!tensor || n_tail_copy == 0) {
                continue;
            }
            const size_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);
            std::vector<uint8_t> rows(n_tail_copy*row_size);
            for (size_t i = 0; i < n_tail_copy; ++i) {
                const int32_t slot = pending_stream_copies.tail_src_slots[i];
                GGML_ASSERT(slot >= 0 && int64_t(slot) < tensor->ne[1]);
                ggml_backend_tensor_get(tensor, rows.data() + i*row_size, size_t(slot)*row_size, row_size);
            }
            for (size_t i = 0; i < n_tail_copy; ++i) {
                const int32_t slot = pending_stream_copies.tail_dst_slots[i];
                GGML_ASSERT(slot >= 0 && int64_t(slot) < tensor->ne[1]);
                ggml_backend_tensor_set(tensor, rows.data() + i*row_size, size_t(slot)*row_size, row_size);
            }
        }
    }

    if (pending_stream_copies.tail_transaction) {
        metadata->commit_pending_tail_copy();
    }

    pending_stream_copies.ssrc.clear();
    pending_stream_copies.sdst.clear();
    pending_stream_copies.tail_src_slots.clear();
    pending_stream_copies.tail_dst_slots.clear();
    pending_stream_copies.tail_transaction = false;
    return true;
}

llama_kv_memory_stats llama_kv_cache_kvarn::kv_memory_stats() const {
    llama_kv_memory_stats result;
    llama_kv_memory_component_stats & component = swa ? result.swa : result.global;
    for (const auto & route : metadata->get_tail_layer_routes()) {
        const bool cpu = !route.owner || ggml_backend_dev_type(route.owner) == GGML_BACKEND_DEVICE_TYPE_CPU;
        if (cpu) {
            component.tail_cpu_layers++;
        } else if (route.capability.route != LLAMA_KV_TAIL_ROUTE_NATIVE) {
            component.tail_device_fallback_layers++;
        } else if (route.has_body) {
            component.tail_native_mixed_layers++;
        } else {
            component.tail_native_bodyless_layers++;
        }
    }

    const uint64_t active_slots = uint64_t(exact_tail_tokens)*n_seq_max;
    const uint64_t rollback_slots = uint64_t(metadata->get_tail_rollback_tokens())*n_seq_max;
    const uint64_t total_tail_slots = active_slots + rollback_slots;
    const auto account_tail = [&](const ggml_tensor * tensor) {
        if (!tensor) {
            return;
        }
        const uint64_t bytes = ggml_nbytes(tensor);
        if (total_tail_slots == 0) {
            component.exact_tail_bytes += bytes;
            return;
        }
        GGML_ASSERT(uint64_t(tensor->ne[1]) >= total_tail_slots);
        const uint64_t row_bytes = tensor->nb[1];
        GGML_ASSERT(row_bytes*uint64_t(tensor->ne[1]) <= bytes);
        component.exact_tail_bytes += row_bytes*active_slots;
        component.rollback_reserve_bytes += row_bytes*rollback_slots;
        component.transient_estimate_bytes += row_bytes*metadata_n_ubatch;
    };
    for (const auto & layer : layers) {
        component.k_payload_bytes += ggml_nbytes(layer.k_records);
        component.v_payload_bytes += ggml_nbytes(layer.v_records);
        component.staging_bytes += ggml_nbytes(layer.k_stage) + ggml_nbytes(layer.v_stage);
        component.stage_rotated_bytes += ggml_nbytes(layer.k_stage) + ggml_nbytes(layer.v_stage);
        account_tail(layer.k_tail);
        account_tail(layer.v_tail);
    }

    uint64_t allocated_kvarn = 0;
    for (const auto & storage : ctxs_bufs) {
        if (!storage.buffer) {
            continue;
        }
        allocated_kvarn += hparams.no_alloc
            ? ggml_backend_alloc_ctx_tensors_from_buft_size(storage.ctx.get(), storage.buft)
            : ggml_backend_buffer_get_size(storage.buffer.get());
    }
    const uint64_t accounted = component.k_payload_bytes + component.v_payload_bytes +
            component.exact_tail_bytes + component.rollback_reserve_bytes + component.staging_bytes;
    component.padding_bytes = allocated_kvarn > accounted ? allocated_kvarn - accounted : 0;
    component.allocated_capacity_tokens = kv_size;
    if (standard_cache) result.add(standard_cache->kv_memory_stats());
    return result;
}

bool llama_kv_cache_kvarn::get_kv_tail_coverage(
        uint32_t group_index, llama_seq_id seq_id, llama_kv_tail_coverage_info & out) const {
    return metadata->get_kv_tail_coverage(group_index, seq_id, out);
}

void llama_kv_cache_kvarn::reset_kv_tail_planner_timing() {
    metadata->reset_kv_tail_planner_timing();
}

uint64_t llama_kv_cache_kvarn::get_kv_tail_planner_timing_ns() const {
    return metadata->get_kv_tail_planner_timing_ns();
}

void llama_kv_cache_kvarn::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if (standard_cache) {
        state_write_mixed(io, seq_id, flags);
        return;
    }
    state_write_kvarn_body(io, seq_id, flags);
}

void llama_kv_cache_kvarn::state_write_kvarn_body(
        llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if (migration_copies && !migration_copies->drain()) migration_mirror_stale = true;
    // Unlike the dense cache, a sliding ring overwrites its historical body.
    // A partial checkpoint must own that ring, not reference its live records.
    if (swa && (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != 0) {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED) != 0 || !stream_is_exclusive_for(seq_id)) {
            throw std::invalid_argument("KVarN SWA partial state requires an exclusive stream and non-conflicting flags");
        }
        flags &= ~LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    }
    metadata->state_write(io, seq_id, flags);
    const bool partial_state = (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != 0 && seq_id >= 0;
    const bool self_contained = (flags & LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED) != 0 && seq_id >= 0;
    if (partial_state && self_contained) {
        throw std::invalid_argument("KVarN state cannot be partial-only and self-contained");
    }
    if (self_contained && swa && !stream_is_exclusive_for(seq_id)) {
        throw std::runtime_error("self-contained KVarN SWA state requires an exclusive stream");
    }
    const bool body_only = (flags & LLAMA_STATE_SEQ_FLAGS_BODY_ONLY) != 0;

    std::vector<uint32_t> saved_streams;
    if (seq_id == -1) {
        saved_streams.reserve(n_stream);
        for (uint32_t stream = 0; stream < n_stream; ++stream) {
            saved_streams.push_back(stream);
        }
    } else {
        const uint32_t stream = metadata->get_stream_for_seq(seq_id);
        GGML_ASSERT(stream < n_stream);
        saved_streams.push_back(stream);
    }

    io.write(&KVAR_N_STATE_MAGIC, sizeof(KVAR_N_STATE_MAGIC));
    io.write(&KVAR_N_STATE_VERSION, sizeof(KVAR_N_STATE_VERSION));
    const int32_t type = params.type;
    const uint32_t n_layers = layers.size();
    const uint32_t n_saved_streams = saved_streams.size();
    io.write(&type, sizeof(type));
    io.write(&n_layers, sizeof(n_layers));
    io.write(&n_saved_streams, sizeof(n_saved_streams));
    for (const uint32_t stream : saved_streams) {
        io.write(&stream, sizeof(stream));
    }
    // Each SWA stream is a complete position-addressed ring. Full state saves
    // preserve every selected stream independently.
    const uint32_t state_kind = self_contained && !swa ? KVAR_N_STATE_RECORDS_SELECTIVE :
            (partial_state ? KVAR_N_STATE_STAGE_ONLY_PARTIAL :
                (!swa && n_stream == 1 ? KVAR_N_STATE_RECORDS_FULL_REMAP_STAGE :
                    KVAR_N_STATE_RECORDS_FULL));
    io.write(&state_kind, sizeof(state_kind));
    // Record both stage and workspace depth so SWA no-sink stages and
    // non-SWA sink stages cannot be restored into each other's layout.
    io.write(&stage_groups, sizeof(stage_groups));
    io.write(&tail_groups, sizeof(tail_groups));
    const uint32_t has_exact_tail = exact_tail_tokens > 0 ? 1u : 0u;
    const int32_t state_exact_type = int32_t(exact_tail_type);
    const std::vector<int32_t> exact_payload_slots = has_exact_tail && !body_only ?
        metadata->state_tail_payload_slots(seq_id) : std::vector<int32_t>{};
    if (exact_payload_slots.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("KVarN exact-tail payload count overflows uint32_t");
    }
    const uint32_t n_exact_payloads = uint32_t(exact_payload_slots.size());
    for (const int32_t slot : exact_payload_slots) {
        if (slot < 0) {
            throw std::runtime_error("KVarN exact-tail state contains a negative source slot");
        }
    }
    const auto exact_payload_runs = llama_kv_tail_contiguous_slot_runs(exact_payload_slots);
    const bool on_device = (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) != 0;
    io.write(&has_exact_tail, sizeof(has_exact_tail));
    io.write(&exact_tail_tokens, sizeof(exact_tail_tokens));
    io.write(&state_exact_type, sizeof(state_exact_type));
    io.write(&n_exact_payloads, sizeof(n_exact_payloads));

    // n_groups_used is single-valued across all saved streams. This is correct
    // because when seq_id >= 0, saved_streams has exactly 1 entry (the stream
    // for that sequence), so n_groups_used applies to that one stream only.
    // When seq_id == -1, n_groups_used equals n_groups_per_stream (no compression).
    // If multi-stream partial writes are ever added, n_groups_used must become
    // per-stream.
    uint32_t n_groups_used = n_groups_per_stream;
    if (seq_id >= 0 && !swa) {
        const auto source_cells = metadata->state_source_cells(seq_id);
        if (!source_cells.empty()) {
            n_groups_used = std::min(
                    n_groups_per_stream,
                    *std::max_element(source_cells.begin(), source_cells.end())/KVAR_N_GROUP + 1u);
        }
    }
    // SWA ring: always serialize all ring slots — the live window may wrap around
    // and occupy any slot, so partial serialization by group index is not meaningful.
    io.write(&n_groups_used, sizeof(n_groups_used));
    llama_pos saved_pos_max = -1;
    if (seq_id >= 0) {
        saved_pos_max = metadata->seq_pos_max(seq_id);
    } else {
        for (uint32_t id = 0; id < n_seq_max; ++id) {
            saved_pos_max = std::max(saved_pos_max, metadata->seq_pos_max(id));
        }
    }
    io.write(&saved_pos_max, sizeof(saved_pos_max));

    std::vector<llama_kvarn_state_stage_cell> selective_stage_cells;
    if (partial_state || state_kind == KVAR_N_STATE_RECORDS_SELECTIVE ||
            state_kind == KVAR_N_STATE_RECORDS_FULL_REMAP_STAGE) {
        std::vector<uint32_t> source_cells;
        if (seq_id >= 0) {
            source_cells = metadata->state_source_cells(seq_id);
        } else {
            GGML_ASSERT(n_stream == 1);
            const auto & cells = metadata->get_cells(0);
            source_cells.reserve(cells.get_used());
            for (uint32_t cell = 0; cell < cells.size(); ++cell) {
                if (!cells.is_empty(cell)) {
                    source_cells.push_back(cell);
                }
            }
        }
        std::vector<uint32_t> staged_groups;
        // The store only encodes allocator-assigned stage slots when the compact
        // read plan is active; otherwise it uses the parity/window layout, so the
        // state rows must be selected under the same rule.
        const bool explicit_stage = !swa && uses_compact_read_indices();
        if (explicit_stage) {
            for (const uint32_t cell : source_cells) {
                if (metadata->allocation_cell_uses_stage(cell)) {
                    staged_groups.push_back(cell/KVAR_N_GROUP);
                }
            }
            std::sort(staged_groups.begin(), staged_groups.end());
            staged_groups.erase(std::unique(staged_groups.begin(), staged_groups.end()), staged_groups.end());
        }
        const uint32_t source_max_p1 = source_cells.empty() ? 0 :
                *std::max_element(source_cells.begin(), source_cells.end()) + 1u;
        selective_stage_cells = llama_kvarn_select_state_stage_cells(
                source_cells,
                source_max_p1,
                stage_groups,
                tail_groups,
                swa,
                explicit_stage ? &staged_groups : nullptr,
                explicit_stage ? &metadata->get_allocation_stage_slots() : nullptr);
    }
    if (selective_stage_cells.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("KVarN selective stage row count overflows uint32_t");
    }
    const uint32_t n_selective_stage_cells = uint32_t(selective_stage_cells.size());
    io.write(&n_selective_stage_cells, sizeof(n_selective_stage_cells));
    for (const auto & cell : selective_stage_cells) {
        io.write(&cell.source_cell, sizeof(cell.source_cell));
        io.write(&cell.stage_row, sizeof(cell.stage_row));
    }

    std::vector<uint32_t> selective_record_groups;
    if (state_kind == KVAR_N_STATE_RECORDS_SELECTIVE) {
        selective_record_groups = llama_kvarn_select_state_record_groups(
                metadata->state_source_cells(seq_id), selective_stage_cells,
                n_groups_per_stream);
    }
    if (selective_record_groups.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("KVarN selective record group count overflows uint32_t");
    }
    const uint32_t n_selective_record_groups = uint32_t(selective_record_groups.size());
    io.write(&n_selective_record_groups, sizeof(n_selective_record_groups));
    for (const uint32_t group : selective_record_groups) {
        if (group >= n_groups_per_stream) {
            throw std::runtime_error("KVarN selective record group is out of range");
        }
        io.write(&group, sizeof(group));
    }

    uint64_t exact_payload_bytes = 0;
    uint64_t selective_stage_bytes = 0;
    size_t exact_tensor_ops = 0;
    for (const auto & layer : layers) {
        io.write(&layer.il, sizeof(layer.il));
        for (const uint32_t stream : saved_streams) {
            io.write(&stream, sizeof(stream));

            if (state_kind == KVAR_N_STATE_RECORDS_FULL ||
                    state_kind == KVAR_N_STATE_RECORDS_FULL_REMAP_STAGE) {
                const size_t k_records_used = n_groups_used * layer.k_records_stream[stream]->nb[2];
                const size_t v_records_used = n_groups_used * layer.v_records_stream[stream]->nb[2];
                write_kvarn_tensor_slice(io, layer.k_records_stream[stream], 0, k_records_used);
                write_kvarn_tensor_slice(io, layer.v_records_stream[stream], 0, v_records_used);
            }
            if (state_kind == KVAR_N_STATE_RECORDS_SELECTIVE) {
                for (const uint32_t group : selective_record_groups) {
                    write_kvarn_tensor_slice(
                            io, layer.k_records_stream[stream],
                            size_t(group)*layer.k_records_stream[stream]->nb[2],
                            layer.k_records_stream[stream]->nb[2]);
                    write_kvarn_tensor_slice(
                            io, layer.v_records_stream[stream],
                            size_t(group)*layer.v_records_stream[stream]->nb[2],
                            layer.v_records_stream[stream]->nb[2]);
                }
            }
            if (state_kind == KVAR_N_STATE_STAGE_ONLY_PARTIAL ||
                    state_kind == KVAR_N_STATE_RECORDS_SELECTIVE ||
                    state_kind == KVAR_N_STATE_RECORDS_FULL_REMAP_STAGE) {
                for (const auto & cell : selective_stage_cells) {
                    const size_t k_offset = size_t(cell.stage_row)*layer.k_stage_stream[stream]->nb[2];
                    const size_t v_offset = size_t(cell.stage_row)*layer.v_stage_stream[stream]->nb[2];
                    write_kvarn_tensor_slice(
                            io, layer.k_stage_stream[stream], k_offset, layer.k_stage_stream[stream]->nb[2]);
                    write_kvarn_tensor_slice(
                            io, layer.v_stage_stream[stream], v_offset, layer.v_stage_stream[stream]->nb[2]);
                    selective_stage_bytes += layer.k_stage_stream[stream]->nb[2] +
                            layer.v_stage_stream[stream]->nb[2];
                }
            } else {
                write_kvarn_tensor(io, layer.k_stage_stream[stream]);
                write_kvarn_tensor(io, layer.v_stage_stream[stream]);
            }
        }
        const uint64_t k_tail_row = layer.k_tail ? ggml_row_size(layer.k_tail->type, layer.k_tail->ne[0]) : 0;
        const uint64_t v_tail_row = layer.v_tail ? ggml_row_size(layer.v_tail->type, layer.v_tail->ne[0]) : 0;
        io.write(&k_tail_row, sizeof(k_tail_row));
        io.write(&v_tail_row, sizeof(v_tail_row));
        if (layer.k_tail) {
            kvarn_tail_add_bytes(exact_payload_bytes, kvarn_tail_checked_bytes(n_exact_payloads, k_tail_row));
            if (on_device) {
                // Match the layout-independent row topology used by the
                // on-device reader; host checkpoints use the batched runs.
                for (const int32_t slot : exact_payload_slots) {
                    const auto span = kvarn_tail_checked_span(layer.k_tail, slot, 1, k_tail_row);
                    io.write_tensor(layer.k_tail, span.offset, span.size);
                    ++exact_tensor_ops;
                }
            } else {
                for (const auto & run : exact_payload_runs) {
                    const auto span = kvarn_tail_checked_span(layer.k_tail, run.slot_begin, run.length, k_tail_row);
                    io.write_tensor(layer.k_tail, span.offset, span.size);
                    ++exact_tensor_ops;
                }
            }
        }
        if (layer.v_tail) {
            kvarn_tail_add_bytes(exact_payload_bytes, kvarn_tail_checked_bytes(n_exact_payloads, v_tail_row));
            if (on_device) {
                for (const int32_t slot : exact_payload_slots) {
                    const auto span = kvarn_tail_checked_span(layer.v_tail, slot, 1, v_tail_row);
                    io.write_tensor(layer.v_tail, span.offset, span.size);
                    ++exact_tensor_ops;
                }
            } else {
                for (const auto & run : exact_payload_runs) {
                    const auto span = kvarn_tail_checked_span(layer.v_tail, run.slot_begin, run.length, v_tail_row);
                    io.write_tensor(layer.v_tail, span.offset, span.size);
                    ++exact_tensor_ops;
                }
            }
        }
    }

    LLAMA_LOG_DEBUG(
            "%s: KVarN state save: kind=%s version=%u stage_rows=%u stage_bytes=%llu "
            "payloads=%u tail_bytes=%llu runs=%zu tensor_ops=%zu device=%s\n",
            __func__, partial_state ? "partial" : (self_contained ? "selective" : "full"), KVAR_N_STATE_VERSION,
            n_selective_stage_cells, (unsigned long long) selective_stage_bytes, n_exact_payloads,
            (unsigned long long) exact_payload_bytes, exact_payload_runs.size(), exact_tensor_ops,
            on_device ? "true" : "false");
}

void llama_kv_cache_kvarn::state_write_mixed(
        llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if (!standard_cache || migration_enabled || swa || n_stream != 1 || n_seq_max != 1) {
        throw std::runtime_error("mixed state currently requires static non-SWA one-stream KV ownership");
    }
    constexpr uint32_t supported_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY |
            LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED;
    if ((uint32_t(flags) & ~supported_flags) != 0 ||
            ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != 0 &&
             (flags & LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED) != 0)) {
        throw std::invalid_argument("mixed state supports host full, partial-only, or self-contained snapshots only");
    }
    const bool partial = (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != 0;
    const bool self_contained = (flags & LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED) != 0;
    if (seq_id < -1 || (partial && seq_id < 0) ||
            (self_contained && seq_id < 0) ||
            (seq_id >= 0 && uint32_t(seq_id) >= n_seq_max)) {
        throw std::invalid_argument("invalid sequence ID for mixed state save");
    }
    if (has_pending_stream_copies() || (migration_copies && !migration_copies->drain())) {
        throw std::runtime_error("mixed KV state cannot be saved while a stream copy is pending");
    }

    std::unordered_map<const ggml_tensor *, uint32_t> tensor_layers;
    const auto add_tensor = [&](const ggml_tensor * tensor, uint32_t layer_id) {
        while (tensor && tensor->view_src) {
            tensor = tensor->view_src;
        }
        if (!tensor) {
            return;
        }
        const auto [it, inserted] = tensor_layers.emplace(tensor, layer_id);
        if (!inserted && it->second != layer_id) {
            throw std::runtime_error("mixed state tensor is shared across different layer IDs");
        }
    };
    for (const auto & layer : layers) {
        for (ggml_tensor * tensor : { layer.k_records, layer.v_records,
                layer.k_stage, layer.v_stage, layer.k_tail, layer.v_tail }) {
            add_tensor(tensor, layer.il);
        }
        for (ggml_tensor * tensor : layer.k_records_stream) add_tensor(tensor, layer.il);
        for (ggml_tensor * tensor : layer.v_records_stream) add_tensor(tensor, layer.il);
        for (ggml_tensor * tensor : layer.k_stage_stream) add_tensor(tensor, layer.il);
        for (ggml_tensor * tensor : layer.v_stage_stream) add_tensor(tensor, layer.il);
    }
    for (const uint32_t il : standard_cache->get_layer_ids()) {
        add_tensor(standard_cache->get_k_storage(int32_t(il)), il);
        add_tensor(standard_cache->get_v_storage(int32_t(il)), il);
    }

    llama_kv_mixed::split_writer split([&](const ggml_tensor * tensor) -> uint32_t {
        while (tensor && tensor->view_src) {
            tensor = tensor->view_src;
        }
        const auto it = tensor_layers.find(tensor);
        if (it == tensor_layers.end()) {
            throw llama_kv_mixed::kv_mixed_error(
                    "mixed state writer received a tensor without a layer descriptor");
        }
        return it->second;
    });
    state_write_kvarn_body(split, seq_id, flags);
    standard_cache->state_write_shared_payload(split, seq_id, partial);

    uint32_t payload_rows = 0;
    auto cells = mixed_state_cells(metadata.get(), seq_id, payload_rows);
    const uint32_t payload_cells = uint32_t(cells.size());
    std::unordered_map<uint32_t, uint64_t> payload_sizes;
    for (const auto & entry : split.manifest().layer_payload_sizes) {
        payload_sizes.emplace(entry.first, entry.second);
    }
    auto layer_descs = mixed_state_layer_descs(*this, payload_sizes, partial,
            payload_cells, payload_rows);

    std::string manifest_error;
    const auto manifest = llama_kv_mixed::mixed_manifest_serialize(
            split.manifest(), &manifest_error);
    if (manifest.empty()) {
        throw llama_kv_mixed::kv_mixed_error(
                manifest_error.empty() ? "failed to encode mixed state manifest" : manifest_error);
    }

    llama_kv_mixed::stream_input input = {};
    input.layers = std::move(layer_descs);
    input.cells = std::move(cells);
    input.metadata_version = MIXED_KV_STATE_METADATA_VERSION;
    input.metadata = manifest.data();
    input.metadata_size = manifest.size();
    input.flags = partial ? llama_kv_mixed::snapshot_flags::partial : llama_kv_mixed::snapshot_flags::full;
    input.owner_epoch = partial ? state_owner_epoch : 0;
    input.owner_id = partial ? state_owner_id : 0;

    const uint64_t frame_size = llama_kv_mixed::stream_serialized_size(input);
    if (frame_size > MIXED_KV_STATE_MAX_FRAME_BYTES || frame_size > size_t(-1)) {
        throw std::overflow_error("mixed KV state frame exceeds the supported size");
    }
    const uint32_t magic = MIXED_KV_STATE_ENVELOPE_MAGIC;
    const uint32_t version = MIXED_KV_STATE_ENVELOPE_VERSION;
    io.write(&magic, sizeof(magic));
    io.write(&version, sizeof(version));
    io.write(&frame_size, sizeof(frame_size));
    if (io.counts_only()) {
        io.write(nullptr, size_t(frame_size));
        return;
    }

    llama_kv_mixed::stream_read_callbacks reads;
    reads.read_payload = [&](uint32_t layer_id, uint64_t offset, uint8_t * dst, size_t count) {
        return split.read_payload(layer_id, offset, dst, count);
    };
    llama_kv_mixed::stream_write_callbacks writes;
    writes.write_bytes = [&](const uint8_t * bytes, size_t count) {
        io.write(bytes, count);
        return true;
    };
    if (!llama_kv_mixed::stream_serialize(input, reads, writes)) {
        throw llama_kv_mixed::kv_mixed_error("failed to serialize mixed KV state");
    }
}

void llama_kv_cache_kvarn::state_read_mixed(
        llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags,
        llama_kv_cache::slot_info_vec_t * sinfos_out,
        const llama_kv_cache::slot_info_vec_t * sinfos_in) {
    using namespace llama_kv_mixed;
    if (!standard_cache || migration_enabled || swa || n_stream != 1 || n_seq_max != 1) {
        throw std::runtime_error("mixed state currently requires static non-SWA one-stream KV ownership");
    }
    constexpr uint32_t supported_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY |
            LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED;
    if ((uint32_t(flags) & ~supported_flags) != 0 ||
            ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != 0 &&
             (flags & LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED) != 0)) {
        throw std::invalid_argument("mixed state restore supports host full, partial-only, or self-contained snapshots only");
    }
    const bool requested_partial = (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != 0;
    if (seq_id < -1 || (requested_partial && seq_id < 0) ||
            ((flags & LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED) != 0 && seq_id < 0) ||
            (seq_id >= 0 && uint32_t(seq_id) >= n_seq_max)) {
        throw std::invalid_argument("invalid sequence ID for mixed state restore");
    }
    if (has_pending_stream_copies() || (migration_copies && !migration_copies->drain())) {
        throw std::runtime_error("mixed KV state cannot be restored while a stream copy is pending");
    }
    if (!requested_partial && state_owner_epoch == UINT64_MAX) {
        throw std::overflow_error("mixed KV state owner epoch exhausted");
    }

    // Bound the incoming frame by the actual persistent K/V allocations plus a
    // bounded allowance for the legacy metadata stream, codec descriptors,
    // cell summaries, and split-I/O event table. This rejects attacker-sized
    // lengths before creating a spool file or mapping any bytes.
    uint64_t kv_payload_capacity = 0;
    std::unordered_set<const ggml_tensor *> counted_tensors;
    const auto add_tensor_capacity = [&](const ggml_tensor * tensor) {
        while (tensor && tensor->view_src) {
            tensor = tensor->view_src;
        }
        if (!tensor || !counted_tensors.insert(tensor).second) {
            return;
        }
        const uint64_t bytes = ggml_nbytes(tensor);
        if (bytes > UINT64_MAX - kv_payload_capacity) {
            throw std::overflow_error("mixed KV allocation size overflows uint64_t");
        }
        kv_payload_capacity += bytes;
    };
    for (const auto & layer : layers) {
        for (const ggml_tensor * tensor : { layer.k_records, layer.v_records,
                layer.k_stage, layer.v_stage, layer.k_tail, layer.v_tail }) {
            add_tensor_capacity(tensor);
        }
        for (const ggml_tensor * tensor : layer.k_records_stream) add_tensor_capacity(tensor);
        for (const ggml_tensor * tensor : layer.v_records_stream) add_tensor_capacity(tensor);
        for (const ggml_tensor * tensor : layer.k_stage_stream) add_tensor_capacity(tensor);
        for (const ggml_tensor * tensor : layer.v_stage_stream) add_tensor_capacity(tensor);
    }
    for (const uint32_t il : standard_cache->get_layer_ids()) {
        add_tensor_capacity(standard_cache->get_k_storage(int32_t(il)));
        add_tensor_capacity(standard_cache->get_v_storage(int32_t(il)));
    }
    const uint64_t cell_margin = uint64_t(metadata->get_size()) * 32u;
    const uint64_t fixed_margin = MIXED_KV_STATE_FRAME_MARGIN_BYTES + cell_margin;
    const uint64_t frame_limit = kv_payload_capacity >= MIXED_KV_STATE_MAX_FRAME_BYTES - fixed_margin
            ? MIXED_KV_STATE_MAX_FRAME_BYTES
            : kv_payload_capacity + fixed_margin;

    uint32_t envelope_magic = 0;
    uint32_t envelope_version = 0;
    uint64_t frame_bytes = 0;
    io.read(&envelope_magic, sizeof(envelope_magic));
    io.read(&envelope_version, sizeof(envelope_version));
    io.read(&frame_bytes, sizeof(frame_bytes));
    if (envelope_magic != MIXED_KV_STATE_ENVELOPE_MAGIC ||
            envelope_version != MIXED_KV_STATE_ENVELOPE_VERSION) {
        throw std::runtime_error("incompatible mixed KV state envelope");
    }
    if (frame_bytes > frame_limit) {
        throw std::runtime_error(format(
                "mixed KV state frame (%llu bytes) exceeds the cache-derived limit (%llu bytes)",
                (unsigned long long) frame_bytes, (unsigned long long) frame_limit));
    }

    auto frame = spool_mixed_state_frame(io, frame_bytes, frame_limit);
    parse_options parse_opts;
    parse_opts.max_total_bytes = frame_bytes;
    parse_opts.max_payload_bytes = kv_payload_capacity;
    parse_opts.max_metadata_bytes = MIXED_KV_STATE_MAX_MANIFEST_BYTES;
    auto parsed = snapshot::parse_view(
            frame->mapping->data(), frame->mapping->size(),
            std::static_pointer_cast<const void>(frame), parse_opts);
    if (parsed.metadata_version() != MIXED_KV_STATE_METADATA_VERSION) {
        throw kv_mixed_error("unsupported mixed KV state metadata version");
    }
    const uint32_t snapshot_flags_value = uint32_t(parsed.flags());
    if ((snapshot_flags_value & ~uint32_t(snapshot_flags::partial)) != 0) {
        throw kv_mixed_error("mixed KV state BODY_ONLY and unknown snapshot flags are unsupported");
    }
    const bool partial = (snapshot_flags_value & uint32_t(snapshot_flags::partial)) != 0;
    if (partial != requested_partial) {
        throw kv_mixed_error("mixed KV snapshot partial/full kind does not match restore flags");
    }
    if (partial && (seq_id < 0 || (flags & LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED) != 0)) {
        throw kv_mixed_error("mixed partial snapshot requires an owner-bound ordinary sequence restore");
    }
    if (!partial && seq_id < 0 && (flags & LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED) != 0) {
        throw kv_mixed_error("self-contained mixed state restore requires a destination sequence");
    }

    mixed_manifest manifest;
    std::string manifest_error;
    if (!mixed_manifest_parse(parsed.metadata_blob(), parsed.metadata_size(), manifest, &manifest_error)) {
        throw kv_mixed_error(manifest_error.empty() ? "invalid mixed state I/O manifest" : manifest_error);
    }

    std::unordered_map<uint32_t, uint64_t> payload_sizes;
    payload_sizes.reserve(manifest.layer_payload_sizes.size());
    for (const auto & entry : manifest.layer_payload_sizes) {
        payload_sizes.emplace(entry.first, entry.second);
    }
    uint64_t payload_rows = 0;
    for (const cell_entry & cell : parsed.cells()) {
        if (cell.pos >= 0) {
            ++payload_rows;
        }
    }
    if (parsed.n_used() != parsed.cells().size()) {
        throw kv_mixed_error("mixed state cell count is inconsistent");
    }
    auto expected_descs = mixed_state_layer_descs(
            *this, payload_sizes, partial, uint32_t(parsed.cells().size()), payload_rows);
    std::unordered_set<uint32_t> expected_layer_ids;
    expected_layer_ids.reserve(expected_descs.size());
    expected_snapshot expected;
    expected.layers.reserve(expected_descs.size());
    for (const layer_desc & desc : expected_descs) {
        expected_layer_ids.insert(desc.layer_id);
        expected.layers.push_back(expected_layer::from_desc(desc));
    }
    for (const auto & entry : manifest.layer_payload_sizes) {
        if (expected_layer_ids.count(entry.first) == 0) {
            throw kv_mixed_error("mixed state I/O manifest names a layer not owned by this cache");
        }
    }
    for (const layer_desc & desc : parsed.layers()) {
        const auto it = payload_sizes.find(desc.layer_id);
        const uint64_t manifest_bytes = it == payload_sizes.end() ? 0 : it->second;
        if (desc.payload_bytes != manifest_bytes) {
            throw kv_mixed_error("mixed state descriptor and I/O manifest payload sizes differ");
        }
    }
    expected.capacity_cells = metadata->get_size();
    expected.metadata_version = MIXED_KV_STATE_METADATA_VERSION;
    if (partial) {
        expected.owner_epoch = state_owner_epoch;
        expected.owner_id = state_owner_id;
    }
    parsed.validate_expected_snapshot(expected, parse_opts);

    if (partial) {
        // Qx payloads in a partial snapshot are resident references. The
        // physical cell identities must still name the same live rows; an
        // append beyond this prefix remains valid, while slot reuse or a
        // changed prefix fails closed.
        const auto & current_cells = metadata->get_cells(0);
        for (uint32_t cell_index = 0; cell_index < parsed.cells().size(); ++cell_index) {
            const cell_entry & saved = parsed.cells()[cell_index];
            if (saved.pos < 0) {
                continue;
            }
            if (cell_index >= current_cells.size() || current_cells.is_empty(cell_index) ||
                    current_cells.pos_get(cell_index) != saved.pos) {
                throw kv_mixed_error("mixed partial snapshot references a missing or changed KV cell");
            }
            const auto & ext = current_cells.ext_get(cell_index);
            if (int32_t(ext.x) != saved.x || int32_t(ext.y) != saved.y || int32_t(ext.tok) != saved.tok) {
                throw kv_mixed_error("mixed partial snapshot references a changed KV cell identity");
            }
            std::vector<int32_t> current_seq_ids;
            for (llama_seq_id current_seq = 0; uint32_t(current_seq) < n_seq_max; ++current_seq) {
                if (current_cells.seq_has(cell_index, current_seq)) {
                    current_seq_ids.push_back(int32_t(current_seq));
                }
            }
            if (current_seq_ids != saved.seq_ids ||
                    std::find(saved.seq_ids.begin(), saved.seq_ids.end(), int32_t(seq_id)) == saved.seq_ids.end()) {
                throw kv_mixed_error("mixed partial snapshot sequence membership no longer matches its owner");
            }
        }
    }

    std::vector<split_reader::payload_span> spans;
    spans.reserve(manifest.layer_payload_sizes.size());
    for (const auto & entry : manifest.layer_payload_sizes) {
        const uint32_t layer_id = entry.first;
        const auto desc_it = std::find_if(parsed.layers().begin(), parsed.layers().end(),
                [layer_id](const layer_desc & desc) { return desc.layer_id == layer_id; });
        if (desc_it == parsed.layers().end()) {
            throw kv_mixed_error("mixed state payload has no layer descriptor");
        }
        const size_t layer_index = size_t(desc_it - parsed.layers().begin());
        const size_t size = parsed.payload_size(layer_index);
        if (entry.second != size) {
            throw kv_mixed_error("mixed state payload span size differs from its I/O manifest");
        }
        if (size != 0) {
            const uint8_t * data = parsed.payload(layer_index);
            if (data == nullptr) {
                throw kv_mixed_error("mixed state payload has no mapped backing data");
            }
            spans.push_back({ layer_id, data, size, parsed.backing() });
        }
    }

    split_reader_limits reader_limits;
    reader_limits.max_fixup_staging = 64u << 20;
    reader_limits.hash_borrowed = true;
    split_reader reader(std::move(manifest), std::move(spans), reader_limits);
    llama_kv_cache::slot_info_vec_t restored_sinfos;
    const auto validate_prepared_cells = [&](const llama_kv_cache & prepared,
            const std::unordered_map<uint32_t, uint32_t> & cell_remap) {
        const auto & prepared_cells = prepared.get_cells(0);
        for (uint32_t source_cell = 0; source_cell < parsed.cells().size(); ++source_cell) {
            const cell_entry & saved = parsed.cells()[source_cell];
            if (saved.pos < 0) {
                continue;
            }
            uint32_t destination_cell = source_cell;
            const auto remapped = cell_remap.find(source_cell);
            if (remapped != cell_remap.end()) {
                destination_cell = remapped->second;
            }
            if (destination_cell >= prepared_cells.size() || prepared_cells.is_empty(destination_cell) ||
                    prepared_cells.pos_get(destination_cell) != saved.pos) {
                throw kv_mixed_error("mixed snapshot cell summary does not match parsed KVarN metadata");
            }
            const auto & ext = prepared_cells.ext_get(destination_cell);
            if (int32_t(ext.x) != saved.x || int32_t(ext.y) != saved.y || int32_t(ext.tok) != saved.tok) {
                throw kv_mixed_error("mixed snapshot cell identity differs from parsed KVarN metadata");
            }
            std::vector<int32_t> actual_seq_ids;
            for (llama_seq_id current_seq = 0; uint32_t(current_seq) < n_seq_max; ++current_seq) {
                if (prepared_cells.seq_has(destination_cell, current_seq)) {
                    actual_seq_ids.push_back(int32_t(current_seq));
                }
            }
            std::vector<int32_t> expected_seq_ids = saved.seq_ids;
            if (seq_id >= 0) {
                for (int32_t & saved_seq : expected_seq_ids) {
                    saved_seq = int32_t(seq_id);
                }
                std::sort(expected_seq_ids.begin(), expected_seq_ids.end());
                expected_seq_ids.erase(std::unique(expected_seq_ids.begin(), expected_seq_ids.end()),
                        expected_seq_ids.end());
            }
            if (actual_seq_ids != expected_seq_ids) {
                throw kv_mixed_error("mixed snapshot sequence membership differs from parsed KVarN metadata");
            }
        }
    };
    state_read_kvarn_body(reader, seq_id, flags, &restored_sinfos, sinfos_in, validate_prepared_cells);
    uint64_t restored_rows = 0;
    if (restored_sinfos.size() != 1) {
        throw kv_mixed_error("mixed KVarN metadata restored an unexpected stream count");
    }
    for (const auto & sinfo : restored_sinfos) {
        if (!sinfo.empty()) {
            if (sinfo.n_stream() != 1 || sinfo.idxs.size() != 1 || sinfo.strm.size() != 1 ||
                    sinfo.strm[0] < 0 || uint32_t(sinfo.strm[0]) >= n_stream) {
                throw kv_mixed_error("mixed KVarN metadata returned an invalid slot map");
            }
            restored_rows += sinfo.idxs[0].size();
        }
    }
    if (restored_rows != payload_rows) {
        throw kv_mixed_error("mixed snapshot cell summary does not match the restored KVarN rows");
    }
    standard_cache->state_read_shared_payload(reader, seq_id, partial, restored_sinfos);
    if (reader.remaining() != 0) {
        throw kv_mixed_error("mixed state I/O manifest contains trailing legacy bytes");
    }
    if (!partial) {
        reader.on_commit([this]() {
            ++state_owner_epoch;
        });
    }
    reader.defer_to(io);
    if (sinfos_out) {
        *sinfos_out = std::move(restored_sinfos);
    }
}

bool llama_kv_cache_kvarn_context::uses_native_attention(int32_t il) const {
    return shared_graph_layers.empty() && cache->uses_kvarn_layer(graph_layer_for(il)) &&
            cache->uses_native_attention(graph_layer_for(il));
}

bool llama_kv_cache_kvarn_context::mixed_tail_native_preferred(int32_t il) const {
    return shared_graph_layers.empty() && cache->mixed_tail_native_preferred(il);
}

bool llama_kv_cache_kvarn_context::native_attention_uses_original_v(int32_t il) const {
    return shared_graph_layers.empty() && cache->native_attention_uses_original_v(il);
}

uint32_t llama_kv_cache_kvarn_context::native_rotated_max_query_tokens(int32_t il) const {
    return shared_graph_layers.empty() ? cache->native_rotated_max_query_tokens(il) : 0;
}

void llama_kv_cache_kvarn::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    state_read_sinfo(io, seq_id, flags, nullptr, nullptr);
}

void llama_kv_cache_kvarn::state_read_sinfo(
        llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags,
        llama_kv_cache::slot_info_vec_t * sinfos_out,
        const llama_kv_cache::slot_info_vec_t * sinfos_in) {
    if (standard_cache) {
        state_read_mixed(io, seq_id, flags, sinfos_out, sinfos_in);
        return;
    }
    state_read_kvarn_body(io, seq_id, flags, sinfos_out, sinfos_in);
}

void llama_kv_cache_kvarn::state_read_kvarn_body(
        llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags,
        llama_kv_cache::slot_info_vec_t * sinfos_out,
        const llama_kv_cache::slot_info_vec_t * sinfos_in,
        const std::function<void(const llama_kv_cache &,
                const std::unordered_map<uint32_t, uint32_t> &)> & validate_prepared) {
    if (has_pending_stream_copies()) {
        throw std::runtime_error("cannot restore KVarN state while a stream copy is pending");
    }
    drain_prefill_migration();
    if (migration_enabled) migration_mirror_stale = true;

    if (swa && (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != 0) {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED) != 0 || !stream_is_exclusive_for(seq_id)) {
            throw std::invalid_argument("KVarN SWA partial state requires an exclusive stream and non-conflicting flags");
        }
        flags &= ~LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    }

    // Parse into a restore-only metadata cache. The live metadata remains
    // untouched until every KVarN descriptor and payload has validated and the
    // outer state reader commits its queued tensor writes.
    auto metadata_prepared = make_metadata_cache();
    const bool self_contained = (flags & LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED) != 0;
    if (seq_id >= 0 && ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != 0 || self_contained)) {
        metadata_prepared->clone_logical_state_from(*metadata);
    }
    if (self_contained) {
        metadata_prepared->set_state_remap_group_size(KVAR_N_GROUP);
    }
    metadata_prepared->state_read_sinfo(io, seq_id, flags, sinfos_out, sinfos_in);
    const auto & state_cell_remap_pairs = metadata_prepared->get_state_cell_remap();
    std::unordered_map<uint32_t, uint32_t> state_cell_remap(
            state_cell_remap_pairs.begin(), state_cell_remap_pairs.end());
    if (validate_prepared) {
        validate_prepared(*metadata_prepared, state_cell_remap);
    }
    std::vector<std::vector<int32_t>> exact_destinations = metadata_prepared->take_restored_tail_payload_slots();
    const bool on_device = (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) != 0;

    uint32_t magic;
    uint32_t version;
    int32_t type;
    uint32_t n_layers;
    io.read(&magic, sizeof(magic));
    io.read(&version, sizeof(version));
    io.read(&type, sizeof(type));
    io.read(&n_layers, sizeof(n_layers));
    if (magic != KVAR_N_STATE_MAGIC || version > KVAR_N_STATE_VERSION ||
        type != params.type || n_layers != layers.size()) {
        throw std::runtime_error("incompatible KVarN cache state");
    }
    if (version < KVAR_N_STATE_VERSION_MIN) {
        throw std::runtime_error(
            "incompatible KVarN cache state version 11: canonical exact-tail payloads are absent; "
            "re-save the prompt cache with this build");
    }

    uint32_t n_saved_streams;
    io.read(&n_saved_streams, sizeof(n_saved_streams));
    if (n_saved_streams == 0 || n_saved_streams > n_stream) {
        throw std::runtime_error("invalid KVarN cache stream count");
    }

    std::vector<uint32_t> saved_streams(n_saved_streams);
    for (uint32_t & stream : saved_streams) {
        io.read(&stream, sizeof(stream));
        if (stream >= n_stream) {
            throw std::runtime_error("invalid KVarN cache stream");
        }
    }

    uint32_t state_kind;
    io.read(&state_kind, sizeof(state_kind));
    if (state_kind != KVAR_N_STATE_RECORDS_FULL &&
            state_kind != KVAR_N_STATE_STAGE_ONLY_PARTIAL &&
            state_kind != KVAR_N_STATE_RECORDS_SELECTIVE &&
            state_kind != KVAR_N_STATE_RECORDS_FULL_REMAP_STAGE) {
        throw std::runtime_error("invalid KVarN cache state kind");
    }
    if (state_kind == KVAR_N_STATE_RECORDS_SELECTIVE && version < 15) {
        throw std::runtime_error("KVarN selective record state predates its format version");
    }
    if (state_kind == KVAR_N_STATE_RECORDS_FULL_REMAP_STAGE && version < 16) {
        throw std::runtime_error("KVarN remappable full state predates its format version");
    }

    // Full unified non-SWA state stores only live stage rows and remaps them by
    // source cell. Older and SWA layouts still require identical stage depth.
    uint32_t saved_stage_groups;
    uint32_t saved_tail_groups;
    io.read(&saved_stage_groups, sizeof(saved_stage_groups));
    io.read(&saved_tail_groups, sizeof(saved_tail_groups));
    if (saved_stage_groups < 2 || saved_tail_groups == 0) {
        throw std::runtime_error("invalid KVarN cache stage depth");
    }
    const bool remappable_full_stage = state_kind == KVAR_N_STATE_RECORDS_FULL_REMAP_STAGE;
    if (remappable_full_stage) {
        if (swa || saved_tail_groups != saved_stage_groups - 1u) {
            throw std::runtime_error("invalid remappable KVarN full-state stage layout");
        }
    } else if (saved_stage_groups != stage_groups) {
        throw std::runtime_error(format(
            "KVarN cache stage depth mismatch: state has %u stage groups, cache has %u; "
            "re-save the prompt cache with the current --ubatch setting",
            saved_stage_groups, stage_groups));
    }
    if (!remappable_full_stage && saved_tail_groups != tail_groups) {
        throw std::runtime_error(format(
            "KVarN cache tail depth mismatch: state has %u tail groups, cache has %u; "
            "re-save the prompt cache with this build",
            saved_tail_groups, tail_groups));
    }

    uint32_t has_exact_tail;
    uint32_t saved_exact_tail_tokens;
    int32_t saved_exact_type;
    uint32_t n_exact_payloads;
    io.read(&has_exact_tail, sizeof(has_exact_tail));
    io.read(&saved_exact_tail_tokens, sizeof(saved_exact_tail_tokens));
    io.read(&saved_exact_type, sizeof(saved_exact_type));
    io.read(&n_exact_payloads, sizeof(n_exact_payloads));
    if (has_exact_tail != uint32_t(exact_tail_tokens > 0) ||
            saved_exact_tail_tokens != exact_tail_tokens ||
            saved_exact_type != int32_t(exact_tail_type) ||
            n_exact_payloads != exact_destinations.size()) {
        throw std::runtime_error(format(
                "KVarN exact-tail state configuration does not match the context "
                "(present %u/%u, tokens %u/%u, type %d/%d, payloads %u/%zu)",
                has_exact_tail, uint32_t(exact_tail_tokens > 0),
                saved_exact_tail_tokens, exact_tail_tokens,
                saved_exact_type, int32_t(exact_tail_type),
                n_exact_payloads, exact_destinations.size()));
    }
    for (const auto & destinations : exact_destinations) {
        if (destinations.empty()) {
            throw std::runtime_error("KVarN exact-tail state contains an unreferenced payload");
        }
        if (on_device && destinations.size() != 1) {
            throw std::runtime_error("on-device KVarN exact-tail state requires one destination slot per payload");
        }
        for (const int32_t slot : destinations) {
            if (slot < 0) {
                throw std::runtime_error("KVarN exact-tail state contains a negative destination slot");
            }
        }
    }

    uint32_t n_groups_used;
    io.read(&n_groups_used, sizeof(n_groups_used));
    if (n_groups_used == 0 || (!swa && n_groups_used > n_groups_per_stream)) {
        throw std::runtime_error("invalid KVarN cache group count");
    }
    llama_pos saved_pos_max;
    io.read(&saved_pos_max, sizeof(saved_pos_max));

    std::vector<llama_kvarn_state_stage_cell> selective_stage_cells;
    if (version >= 14) {
        uint32_t n_selective_stage_cells;
        io.read(&n_selective_stage_cells, sizeof(n_selective_stage_cells));
        if (n_selective_stage_cells > uint64_t(saved_stage_groups)*KVAR_N_GROUP) {
            throw std::runtime_error("invalid KVarN selective stage row count");
        }
        selective_stage_cells.resize(n_selective_stage_cells);
        std::set<uint32_t> source_cells;
        std::set<uint32_t> stage_rows;
        for (auto & cell : selective_stage_cells) {
            io.read(&cell.source_cell, sizeof(cell.source_cell));
            io.read(&cell.stage_row, sizeof(cell.stage_row));
            if (uint64_t(cell.stage_row) >= uint64_t(saved_stage_groups)*KVAR_N_GROUP ||
                    !source_cells.insert(cell.source_cell).second ||
                    !stage_rows.insert(cell.stage_row).second) {
                throw std::runtime_error("invalid KVarN selective stage cell mapping");
            }
        }
    }

    std::vector<uint32_t> selective_record_groups;
    if (version >= 15) {
        uint32_t n_selective_record_groups;
        io.read(&n_selective_record_groups, sizeof(n_selective_record_groups));
        if (n_selective_record_groups > n_groups_per_stream) {
            throw std::runtime_error("invalid KVarN selective record group count");
        }
        selective_record_groups.resize(n_selective_record_groups);
        std::set<uint32_t> unique_groups;
        for (uint32_t & group : selective_record_groups) {
            io.read(&group, sizeof(group));
            if (group >= n_groups_per_stream || !unique_groups.insert(group).second) {
                throw std::runtime_error("invalid KVarN selective record group mapping");
            }
        }
    }

    const uint32_t seq_stream = seq_id == -1 ? 0 : metadata_prepared->get_stream_for_seq(seq_id);
    if (seq_id != -1 && seq_stream >= n_stream) {
        throw std::runtime_error("invalid KVarN sequence stream");
    }
    if (state_kind == KVAR_N_STATE_STAGE_ONLY_PARTIAL) {
        if (swa) {
            throw std::runtime_error("legacy KVarN SWA partial state lacks checkpoint-owned ring records; re-save the checkpoint");
        }
        if (seq_id < 0) {
            throw std::runtime_error("KVarN stage-only state requires a destination sequence");
        }
        if (version < 14 && !stream_is_exclusive_for(seq_id)) {
            throw std::runtime_error(
                    "legacy KVarN partial state cannot restore into a shared physical stream");
        }
    } else if (state_kind == KVAR_N_STATE_RECORDS_SELECTIVE) {
        if (seq_id < 0 || !self_contained) {
            throw std::runtime_error("KVarN selective record state requires a self-contained sequence restore");
        }
        if (swa) {
            throw std::runtime_error("KVarN selective record state does not support SWA ring remapping");
        }
    } else if (state_kind == KVAR_N_STATE_RECORDS_FULL_REMAP_STAGE) {
        if (seq_id >= 0 && self_contained) {
            throw std::runtime_error("remappable full KVarN state cannot be self-contained");
        }
        if (!selective_record_groups.empty()) {
            throw std::runtime_error("remappable full KVarN state contains selective records");
        }
    } else if (!selective_stage_cells.empty() || !selective_record_groups.empty()) {
        throw std::runtime_error("full KVarN state contains selective rows");
    }

    std::unordered_map<uint32_t, uint32_t> desired_stage_rows;
    std::unordered_map<uint32_t, uint32_t> install_stage_rows;
    if ((state_kind == KVAR_N_STATE_STAGE_ONLY_PARTIAL ||
            state_kind == KVAR_N_STATE_RECORDS_SELECTIVE ||
            state_kind == KVAR_N_STATE_RECORDS_FULL_REMAP_STAGE) && version >= 14) {
        std::vector<uint32_t> destination_cells;
        if (seq_id >= 0) {
            destination_cells = metadata_prepared->state_source_cells(seq_id);
        } else {
            GGML_ASSERT(n_stream == 1);
            const auto & cells = metadata_prepared->get_cells(0);
            destination_cells.reserve(cells.get_used());
            for (uint32_t cell = 0; cell < cells.size(); ++cell) {
                if (!cells.is_empty(cell)) {
                    destination_cells.push_back(cell);
                }
            }
        }
        std::vector<uint32_t> staged_groups;
        // mirror the store's stage layout: allocator slots only with compact reads
        const bool explicit_stage = !swa && uses_compact_read_indices();
        if (explicit_stage) {
            for (const uint32_t cell : destination_cells) {
                if (metadata_prepared->allocation_cell_uses_stage(cell)) {
                    staged_groups.push_back(cell/KVAR_N_GROUP);
                }
            }
            std::sort(staged_groups.begin(), staged_groups.end());
            staged_groups.erase(std::unique(staged_groups.begin(), staged_groups.end()), staged_groups.end());
        }
        const uint32_t destination_max_p1 = destination_cells.empty() ? 0 :
                *std::max_element(destination_cells.begin(), destination_cells.end()) + 1u;
        const auto desired = llama_kvarn_select_state_stage_cells(
                destination_cells,
                destination_max_p1,
                stage_groups,
                tail_groups,
                swa,
                explicit_stage ? &staged_groups : nullptr,
                explicit_stage ? &metadata_prepared->get_allocation_stage_slots() : nullptr);
        for (const auto & cell : desired) {
            desired_stage_rows.emplace(cell.source_cell, cell.stage_row);
        }
        for (const auto & saved : selective_stage_cells) {
            const auto remapped = state_cell_remap.find(saved.source_cell);
            const uint32_t destination_cell = remapped != state_cell_remap.end() ?
                    remapped->second : saved.source_cell;
            const auto desired_row = desired_stage_rows.find(destination_cell);
            if (desired_row != desired_stage_rows.end()) {
                install_stage_rows.emplace(saved.source_cell, desired_row->second);
            }
        }
        if (install_stage_rows.size() != desired_stage_rows.size()) {
            throw std::runtime_error("KVarN selective state is missing a required live stage row");
        }
        if (on_device && state_kind == KVAR_N_STATE_STAGE_ONLY_PARTIAL &&
                desired_stage_rows.size() != selective_stage_cells.size()) {
            throw std::runtime_error(
                    "on-device KVarN selective restore cannot discard superseded stage rows");
        }
    }

    std::unordered_map<uint32_t, uint32_t> selective_record_destinations;
    if (state_kind == KVAR_N_STATE_RECORDS_SELECTIVE) {
        for (const auto & [source_cell, destination_cell] : state_cell_remap) {
            const uint32_t source_group = source_cell/KVAR_N_GROUP;
            const uint32_t source_offset = source_cell%KVAR_N_GROUP;
            const uint32_t destination_group = destination_cell/KVAR_N_GROUP;
            if (destination_cell%KVAR_N_GROUP != source_offset) {
                throw std::runtime_error("KVarN selective state did not preserve record row offsets");
            }
            const auto [it, inserted] = selective_record_destinations.emplace(source_group, destination_group);
            if (!inserted && it->second != destination_group) {
                throw std::runtime_error("KVarN selective record group remapped inconsistently");
            }
        }
        for (const uint32_t source_group : selective_record_groups) {
            if (selective_record_destinations.count(source_group) == 0) {
                throw std::runtime_error("KVarN selective record group has no destination mapping");
            }
        }
    }

    uint64_t exact_payload_bytes = 0;
    uint64_t selective_stage_bytes = 0;
    size_t exact_tensor_ops = 0;
    for (const auto & layer : layers) {
        uint32_t il;
        io.read(&il, sizeof(il));
        if (il != layer.il) {
            throw std::runtime_error("mismatched KVarN cache layer");
        }

        for (uint32_t i = 0; i < n_saved_streams; ++i) {
            uint32_t stream;
            io.read(&stream, sizeof(stream));
            if (stream != saved_streams[i]) {
                throw std::runtime_error("mismatched KVarN cache stream");
            }

            const uint32_t stream_dst = seq_id == -1 ? stream : seq_stream;

            const size_t k_records_used = n_groups_used * layer.k_records_stream[stream_dst]->nb[2];
            const size_t v_records_used = n_groups_used * layer.v_records_stream[stream_dst]->nb[2];
            const size_t k_records_total = n_groups_per_stream * layer.k_records_stream[stream_dst]->nb[2];
            const size_t v_records_total = n_groups_per_stream * layer.v_records_stream[stream_dst]->nb[2];

            if (state_kind == KVAR_N_STATE_RECORDS_FULL ||
                    state_kind == KVAR_N_STATE_RECORDS_FULL_REMAP_STAGE) {
                if (swa) {
                    read_kvarn_swa_records(io, layer.k_records_stream[stream_dst],
                            n_groups_used, n_groups_per_stream, saved_pos_max, on_device);
                    read_kvarn_swa_records(io, layer.v_records_stream[stream_dst],
                            n_groups_used, n_groups_per_stream, saved_pos_max, on_device);
                } else {
                    read_kvarn_tensor_slice(io, layer.k_records_stream[stream_dst], 0, k_records_used);
                    zero_kvarn_tensor_range(io, layer.k_records_stream[stream_dst], k_records_used, k_records_total - k_records_used);

                    read_kvarn_tensor_slice(io, layer.v_records_stream[stream_dst], 0, v_records_used);
                    zero_kvarn_tensor_range(io, layer.v_records_stream[stream_dst], v_records_used, v_records_total - v_records_used);
                }
            }

            if (state_kind == KVAR_N_STATE_RECORDS_SELECTIVE) {
                for (const uint32_t source_group : selective_record_groups) {
                    const uint32_t destination_group = selective_record_destinations.at(source_group);
                    read_kvarn_tensor_slice(
                            io, layer.k_records_stream[stream_dst],
                            size_t(destination_group)*layer.k_records_stream[stream_dst]->nb[2],
                            layer.k_records_stream[stream_dst]->nb[2]);
                    read_kvarn_tensor_slice(
                            io, layer.v_records_stream[stream_dst],
                            size_t(destination_group)*layer.v_records_stream[stream_dst]->nb[2],
                            layer.v_records_stream[stream_dst]->nb[2]);
                }
            }

            if ((state_kind == KVAR_N_STATE_STAGE_ONLY_PARTIAL ||
                    state_kind == KVAR_N_STATE_RECORDS_SELECTIVE ||
                    state_kind == KVAR_N_STATE_RECORDS_FULL_REMAP_STAGE) && version >= 14) {
                const auto read_selective_row = [&](ggml_tensor * tensor,
                                                     const llama_kvarn_state_stage_cell & cell) {
                    const size_t row_size = tensor->nb[2];
                    uint64_t saved_size;
                    io.read(&saved_size, sizeof(saved_size));
                    if (saved_size != row_size) {
                        throw std::runtime_error("mismatched KVarN selective stage row size");
                    }
                    const auto desired = install_stage_rows.find(cell.source_cell);
                    const bool install = desired != install_stage_rows.end();
                    const size_t offset = install ? size_t(desired->second)*row_size : 0;
                    if (install) {
                        // Indexed row installs stay within the bounded
                        // transfer contract of the streaming reader; a
                        // multi-layer state has far more stage rows than the
                        // staging budget allows, and the empty-destination
                        // contract makes late failures unreachable.
                        io.read_tensor(tensor, offset, row_size);
                    } else {
                        if (on_device) {
                            throw std::runtime_error("on-device KVarN selective stage row became stale");
                        }
                        std::vector<uint8_t> row(row_size);
                        io.read(row.data(), row.size());
                    }
                    selective_stage_bytes += install ? row_size : 0;
                };
                for (const auto & cell : selective_stage_cells) {
                    read_selective_row(layer.k_stage_stream[stream_dst], cell);
                    read_selective_row(layer.v_stage_stream[stream_dst], cell);
                }
            } else {
                read_kvarn_tensor(io, layer.k_stage_stream[stream_dst]);
                read_kvarn_tensor(io, layer.v_stage_stream[stream_dst]);
            }
        }
        uint64_t k_tail_row;
        uint64_t v_tail_row;
        io.read(&k_tail_row, sizeof(k_tail_row));
        io.read(&v_tail_row, sizeof(v_tail_row));
        const uint64_t expected_k_tail_row = layer.k_tail ? ggml_row_size(layer.k_tail->type, layer.k_tail->ne[0]) : 0;
        const uint64_t expected_v_tail_row = layer.v_tail ? ggml_row_size(layer.v_tail->type, layer.v_tail->ne[0]) : 0;
        if (k_tail_row != expected_k_tail_row || v_tail_row != expected_v_tail_row) {
            throw std::runtime_error("KVarN exact-tail state layer layout mismatch");
        }
        kvarn_tail_add_bytes(exact_payload_bytes, kvarn_tail_checked_bytes(n_exact_payloads, k_tail_row));
        kvarn_tail_add_bytes(exact_payload_bytes, kvarn_tail_checked_bytes(n_exact_payloads, v_tail_row));
        if (version == 12) {
            exact_tensor_ops += read_kvarn_exact_tail_v12_interleaved(
                    io, layer.k_tail, layer.v_tail, k_tail_row, v_tail_row,
                    exact_destinations, on_device);
        } else if (version >= 13) {
            exact_tensor_ops += read_kvarn_exact_tail_v13_component(
                    io, layer.k_tail, k_tail_row, exact_destinations, on_device);
            exact_tensor_ops += read_kvarn_exact_tail_v13_component(
                    io, layer.v_tail, v_tail_row, exact_destinations, on_device);
        } else {
            throw std::runtime_error("unsupported KVarN exact-tail state layout");
        }
    }

    auto prepared_owner = std::make_shared<std::unique_ptr<llama_kv_cache>>(std::move(metadata_prepared));
    const size_t exact_destination_runs = kvarn_exact_tail_destination_runs(exact_destinations);
    const char * log_function = __func__;
    const uint32_t n_selective_stage_cells = uint32_t(selective_stage_cells.size());
    io.on_commit([this, prepared_owner, state_kind, version, n_exact_payloads,
                  n_selective_stage_cells, selective_stage_bytes,
                  log_function,
                  exact_payload_bytes, exact_destination_runs, exact_tensor_ops, on_device]() mutable {
        metadata->swap_logical_state_from(**prepared_owner);
        LLAMA_LOG_DEBUG(
                "%s: KVarN state restore: kind=%s version=%u stage_rows=%u stage_bytes=%llu "
                "payloads=%u tail_bytes=%llu runs=%zu tensor_ops=%zu device=%s\n",
                log_function, state_kind == KVAR_N_STATE_STAGE_ONLY_PARTIAL ? "partial" :
                        (state_kind == KVAR_N_STATE_RECORDS_SELECTIVE ? "selective" :
                            (state_kind == KVAR_N_STATE_RECORDS_FULL_REMAP_STAGE ? "full-remap" : "full")),
                version, n_selective_stage_cells, (unsigned long long) selective_stage_bytes,
                n_exact_payloads, (unsigned long long) exact_payload_bytes,
                exact_destination_runs, exact_tensor_ops, on_device ? "true" : "false");
    });
}

llama_kv_cache * llama_kv_cache_kvarn::get_metadata_cache() const {
    return metadata.get();
}

bool llama_kv_cache_kvarn::uses_kvarn_layer(int32_t il) const {
    return map_layer_ids.find(il) != map_layer_ids.end();
}

bool llama_kv_cache_kvarn::uses_standard_layer(int32_t il) const {
    return standard_cache && standard_cache->has_layer(il);
}

ggml_tensor * llama_kv_cache_kvarn::standard_get_k(
        ggml_context * ctx, int32_t il, uint32_t n_kv,
        const llama_kv_cache::slot_info & sinfo) const {
    return uses_standard_layer(il) ? standard_cache->get_k(ctx, il, n_kv, sinfo) : nullptr;
}

ggml_tensor * llama_kv_cache_kvarn::standard_get_v(
        ggml_context * ctx, int32_t il, uint32_t n_kv,
        const llama_kv_cache::slot_info & sinfo) const {
    return uses_standard_layer(il) ? standard_cache->get_v(ctx, il, n_kv, sinfo) : nullptr;
}

ggml_tensor * llama_kv_cache_kvarn::standard_cpy_k(
        ggml_context * ctx, ggml_tensor * current, ggml_tensor * indices,
        int32_t il, const llama_kv_cache::slot_info & sinfo) const {
    return uses_standard_layer(il) ? standard_cache->cpy_k(ctx, current, indices, il, sinfo) : nullptr;
}

ggml_tensor * llama_kv_cache_kvarn::standard_cpy_v(
        ggml_context * ctx, ggml_tensor * current, ggml_tensor * indices,
        int32_t il, const llama_kv_cache::slot_info & sinfo) const {
    return uses_standard_layer(il) ? standard_cache->cpy_v(ctx, current, indices, il, sinfo) : nullptr;
}

ggml_tensor * llama_kv_cache_kvarn::standard_build_input_k_rot(ggml_context * ctx) const {
    return standard_cache ? standard_cache->build_input_k_rot(ctx) : nullptr;
}

ggml_tensor * llama_kv_cache_kvarn::standard_build_input_v_rot(ggml_context * ctx) const {
    return standard_cache ? standard_cache->build_input_v_rot(ctx) : nullptr;
}

void llama_kv_cache_kvarn::standard_set_input_k_rot(ggml_tensor * tensor) const {
    if (standard_cache && tensor) standard_cache->set_input_k_rot(tensor);
}

void llama_kv_cache_kvarn::standard_set_input_v_rot(ggml_tensor * tensor) const {
    if (standard_cache && tensor) standard_cache->set_input_v_rot(tensor);
}

void llama_kv_cache_kvarn::standard_set_input_k_rot_backend(ggml_tensor * tensor) const {
    if (standard_cache && tensor) standard_cache->set_input_k_rot_backend(tensor);
}

void llama_kv_cache_kvarn::standard_set_input_v_rot_backend(ggml_tensor * tensor) const {
    if (standard_cache && tensor) standard_cache->set_input_v_rot_backend(tensor);
}

void llama_kv_cache_kvarn::standard_cache_set_input_kq_mask(
        ggml_tensor * tensor, const llama_ubatch * ubatch, bool causal_attn) const {
    if (standard_cache && tensor) standard_cache->set_input_kq_mask(tensor, ubatch, causal_attn);
}

std::vector<uint32_t> llama_kv_cache_kvarn::standard_layer_ids() const {
    return standard_cache ? standard_cache->get_layer_ids() : std::vector<uint32_t>{};
}

std::vector<llama_kv_cache_standard_layer_layout> llama_kv_cache_kvarn::standard_layer_layout() const {
    std::vector<llama_kv_cache_standard_layer_layout> result;
    if (!standard_cache) {
        return result;
    }
    const uint32_t rotation_k = uint32_t(std::max(0, standard_cache->rotation_k()));
    const uint32_t rotation_v = uint32_t(std::max(0, standard_cache->rotation_v()));
    for (const uint32_t il : standard_cache->get_layer_ids()) {
        const ggml_tensor * key = standard_cache->get_k_storage(int32_t(il));
        const ggml_tensor * value = standard_cache->get_v_storage(int32_t(il));
        if (!key || !value) {
            throw std::runtime_error(format("mixed standard KV layer %u is missing a K/V tensor", il));
        }
        std::string device = "CPU";
        if (key->buffer) {
            const auto buft = ggml_backend_buffer_get_type(key->buffer);
            const auto dev = buft ? ggml_backend_buft_get_device(buft) : nullptr;
            if (dev) {
                device = ggml_backend_dev_name(dev);
            }
        }
        result.push_back({
            il,
            key->type,
            value->type,
            rotation_k,
            rotation_v,
            standard_cache->v_transposed(),
            hparams.n_embd_head_k(il),
            hparams.n_embd_head_v(il),
            hparams.n_head_kv(il),
            ggml_row_size(key->type, hparams.n_embd_k_gqa(il)),
            ggml_row_size(value->type, hparams.n_embd_v_gqa(il)),
            std::move(device),
        });
    }
    return result;
}

std::vector<llama_kv_cache_kvarn_layer_layout> llama_kv_cache_kvarn::kvarn_layer_layout() const {
    std::vector<llama_kv_cache_kvarn_layer_layout> result;
    result.reserve(layers.size());
    for (const auto & layer : layers) {
        if (!layer.k_records || !layer.v_records) {
            throw std::runtime_error(format("KVarN layer %u is missing record tensors", layer.il));
        }
        llama_kvarn_geometry k_geometry = {};
        llama_kvarn_geometry v_geometry = {};
        if (!llama_kvarn_geometry_for(layer.head_dim_k, k_geometry) ||
                !llama_kvarn_geometry_for(layer.head_dim_v, v_geometry)) {
            throw std::runtime_error(format("KVarN layer %u has unsupported geometry", layer.il));
        }
        result.push_back({
            layer.il,
            uint16_t(params.type),
            uint8_t(params.key_bits),
            uint8_t(params.value_bits),
            k_geometry.record_dim,
            v_geometry.record_dim,
            layer.head_dim_k,
            layer.head_dim_v,
            layer.k_slices,
            layer.v_slices,
            layer.n_head_kv,
            layer.k_records->nb[1],
            layer.v_records->nb[1],
            exact_tail_tokens > 0 ? exact_tail_type : GGML_TYPE_COUNT,
        });
    }
    return result;
}

ggml_tensor * llama_kv_cache_kvarn::get_materialization_source(int32_t il, bool value) const {
    const auto & layer = layer_for(il);
    return value ? layer.v_stage : layer.k_stage;
}

std::unique_ptr<llama_kv_cache> llama_kv_cache_kvarn::make_shared_metadata_cache(
        const llama_model & model_view) const {
    return std::make_unique<llama_kv_cache>(
            model_view,
            model_view.hparams,
            GGML_TYPE_F16,
            GGML_TYPE_F16,
            false,
            false,
            n_stream == 1,
            kv_size,
            n_seq_max,
            metadata_n_pad,
            metadata_n_swa,
            metadata_swa_type,
            metadata.get(),
            [](int32_t) { return false; },
            nullptr,
            nullptr,
            "",                 // name_tag
            metadata_n_ubatch,  // n_ubatch
            0,                  // tail_tokens
            GGML_TYPE_F16,      // tail_type
            UINT32_MAX,         // tail_tokens_requested
            false,              // tail_metadata_only
            0,                  // tail_rollback_tokens
            0,                  // tail_visibility_window
            false);             // disable_attn_rot
}

int32_t llama_kv_cache_kvarn::mapped_layer_id(int32_t il) const {
    return map_layer_ids.at(il);
}

llama_kv_tail_route llama_kv_cache_kvarn::get_tail_route(int32_t il) const {
    return metadata->get_tail_route(il);
}

bool llama_kv_cache_kvarn::get_tail_explicit_bias(int32_t il) const {
    return metadata->get_tail_explicit_bias(il);
}

const llama_kv_cache_kvarn::layer & llama_kv_cache_kvarn::layer_for(int32_t il) const {
    return layers.at(map_layer_ids.at(il));
}

bool llama_kv_cache_kvarn::uses_native_attention(int32_t il) const {
    return layer_for(il).native_attention;
}

bool llama_kv_cache_kvarn::mixed_tail_native_preferred(int32_t il) const {
    return layer_for(il).mixed_tail_native;
}

bool llama_kv_cache_kvarn::native_attention_uses_original_v(int32_t il) const {
    return layer_for(il).native_original_v;
}

uint32_t llama_kv_cache_kvarn::native_rotated_max_query_tokens(int32_t il) const {
    return layer_for(il).native_rotated_max_query_tokens;
}

ggml_tensor * llama_kv_cache_kvarn::get_tail(
        ggml_context * ctx, int32_t il, bool value) const {
    const auto & layer = layer_for(il);
    ggml_tensor * tensor = value ? layer.v_tail : layer.k_tail;
    if (!tensor) {
        return nullptr;
    }
    const uint32_t head_dim = value ? layer.head_dim_v : layer.head_dim_k;
    return ggml_view_4d(ctx, tensor,
            head_dim, layer.n_head_kv, metadata->get_tail_slots(), 1,
            ggml_row_size(tensor->type, head_dim), tensor->nb[1], tensor->nb[2], 0);
}

ggml_tensor * llama_kv_cache_kvarn::store_tail(
        ggml_context * ctx, ggml_tensor * current, ggml_tensor * indices,
        int32_t il, bool value, ggml_tensor * dependency) const {
    const auto & layer = layer_for(il);
    ggml_tensor * dst = value ? layer.v_tail : layer.k_tail;
    if (!dst || !indices) {
        return nullptr;
    }
    const uint32_t head_dim = value ? layer.head_dim_v : layer.head_dim_k;
    const int64_t n_embd = int64_t(head_dim)*layer.n_head_kv;
    const int64_t n_tokens = current->ne[2];
    GGML_ASSERT(current->type == GGML_TYPE_F32 && current->ne[0] == head_dim &&
            current->ne[1] == layer.n_head_kv && dst->ne[0] == n_embd);
    GGML_ASSERT(indices->type == GGML_TYPE_I64 && indices->ne[0] == n_tokens);
    current = ggml_is_contiguous(current)
        ? ggml_reshape_2d(ctx, current, n_embd, n_tokens)
        : ggml_cont_2d(ctx, current, n_embd, n_tokens);
    ggml_tensor * written = dst;
    for (int64_t level = 0; level < indices->ne[1]; ++level) {
        ggml_tensor * level_idxs = indices->ne[1] == 1
            ? indices
            : ggml_view_1d(ctx, indices, n_tokens, level*indices->nb[1]);
        written = ggml_set_rows_ordered(ctx, written, current, level_idxs, dependency);
        dependency = written;
    }
    return ggml_view_4d(ctx, written,
            head_dim, layer.n_head_kv, metadata->get_tail_slots(), 1,
            ggml_row_size(written->type, head_dim), written->nb[1], written->nb[2], 0);
}

ggml_tensor * llama_kv_cache_kvarn::store(
        ggml_context * ctx,
        ggml_tensor * current,
        ggml_tensor * indices,
        int32_t il,
        const llama_kv_cache::slot_info & sinfo,
        bool value) const {
    const auto & layer = layer_for(il);
    if (!ggml_is_contiguous(current)) {
        current = ggml_cont(ctx, current);
    }

    const uint32_t head_dim = value ? layer.head_dim_v : layer.head_dim_k;
    const uint32_t slices = value ? layer.v_slices : layer.k_slices;
    GGML_ASSERT((uint32_t) current->ne[0] == head_dim);
    GGML_ASSERT((uint32_t) current->ne[1] == layer.n_head_kv);
    if (slices > 1) {
        current = ggml_reshape_3d(ctx, current, KVAR_N_GROUP, layer.n_head_kv * slices, current->ne[2]);
    }

    ggml_tensor * result = ggml_kvarn_store(
        ctx,
        current,
        indices,
        value ? layer.v_stage : layer.k_stage,
        value ? layer.v_records : layer.k_records,
        value ? params.value_bits : params.key_bits,
        params.sinkhorn_iters,
        value,
        int32_t(stage_groups));
    result->op_params[3] = kvarn_workspace_tokens_per_stream_hint(sinfo);
    result->op_params[4] = swa ? 1 : 0; // SWA sliding-window ring store
    result->op_params[5] = (int32_t) slices; // KVarN head-wide Hadamard slice count
    result->op_params[8] = int32_t(tail_groups);
    result->op_params[9] = 1; // commit every completed record before it leaves the live workspace
    return result;
}

ggml_tensor * llama_kv_cache_kvarn::view(
        ggml_context * ctx,
        ggml_tensor * stored,
        int32_t il,
        uint32_t n_kv,
        const llama_kv_cache::slot_info & sinfo,
        bool value,
        ggml_tensor * mat_idxs) const {
    const auto & layer = layer_for(il);
    const uint32_t stream_start = sinfo.s0;
    const uint32_t stream_count = sinfo.s1 - sinfo.s0 + 1;
    ggml_tensor * indices = mat_idxs ? mat_idxs : stored->src[1];

    ggml_tensor * result = ggml_kvarn_view(
        ctx,
        value ? layer.v_records : layer.k_records,
        stored,
        indices,
        n_kv,
        stream_start,
        stream_count,
        value ? params.value_bits : params.key_bits,
        value,
        int32_t(stage_groups));
    result->op_params[6] = swa ? 1 : 0;
    result->op_params[8] = int32_t(tail_groups);
    result->op_params[9] = 1;
    result->op_params[10] = !swa && mat_idxs ? 1 : 0;
    const uint32_t slices = value ? layer.v_slices : layer.k_slices;
    if (slices > 1) {
        result = ggml_reshape_4d(
                ctx,
                result,
                value ? layer.head_dim_v : layer.head_dim_k,
                layer.n_head_kv,
                n_kv,
                stream_count);
    }

    return result;
}

ggml_tensor * llama_kv_cache_kvarn::materialize(
        ggml_context * ctx,
        ggml_tensor * stored,
        int32_t il,
        uint32_t n_kv,
        const llama_kv_cache::slot_info & sinfo,
        bool value,
        ggml_tensor * mat_idxs,
        bool read_indirect) const {
    const auto & layer = layer_for(il);
    const uint32_t stream_start = sinfo.s0;
    const uint32_t stream_count = sinfo.s1 - sinfo.s0 + 1;
    ggml_tensor * indices = mat_idxs ? mat_idxs : stored->src[1];
    GGML_ASSERT(indices != nullptr);

    ggml_tensor * result = ggml_kvarn_materialize(
        ctx,
        value ? layer.v_records : layer.k_records,
        stored,
        indices,
        n_kv,
        stream_start,
        stream_count,
        value ? params.value_bits : params.key_bits,
        value,
        int32_t(stage_groups));
    // Materialized fallback attention stays in the same rotated domain as the
    // persistent KVarN body. This avoids a full inverse transform per layer.
    result->op_params[4] = 1;
    result->op_params[5] = int32_t(value ? layer.v_slices : layer.k_slices);
    result->op_params[6] = swa ? 1 : 0;
    result->op_params[8] = int32_t(tail_groups);
    result->op_params[9] = 1;
    result->op_params[10] = !swa && mat_idxs && read_indirect ? 1 : 0;
    const uint32_t slices = value ? layer.v_slices : layer.k_slices;
    if (slices > 1) {
        result = ggml_reshape_4d(
                ctx,
                result,
                value ? layer.head_dim_v : layer.head_dim_k,
                layer.n_head_kv,
                n_kv,
                stream_count);
    }
    return result;
}
// ---------------------------------------------------------------------------
// q4_0/q4_0 -> KVarN conversion
//
// The source q4 cache stores post-RoPE K and V rows in the original domain.
// This cache stores both in the rotated domain: every 128-dim head slice is
// Hadamard-transformed, then corresponding lanes across slices are mixed by a
// second Hadamard step (matching ggml_cuda_kvarn_store). Completed 128-token
// groups are quantized into records; the sink group and the live tail groups
// stay as F16 stage rows. The conversion reproduces that pipeline on the host
// from dequantized q4 rows and emits a canonical KVarN "full" state image.
// ---------------------------------------------------------------------------

namespace {

// A conversion group must join every worker even when one of the dequantizers
// or quantizers throws. Letting a joinable std::thread escape would call
// std::terminate and could leave the caller without the normal atomic cleanup.
// Keep the workers alive for the entire conversion: creating and joining a
// fresh set for every 128-token group otherwise adds thousands of thread
// lifetimes to the large q4 -> KVarN handoff.
class conversion_thread_pool {
public:
    explicit conversion_thread_pool(size_t count) {
        count = std::max<size_t>(1, count);
        workers.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            workers.emplace_back([this] { worker_loop(); });
        }
    }

    ~conversion_thread_pool() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        work.notify_all();
        for (auto & worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    template<typename F>
    void submit(F && function) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (stopping) {
                throw std::runtime_error("q4 conversion worker pool is stopped");
            }
            tasks.emplace_back(std::forward<F>(function));
            ++pending;
        }
        work.notify_one();
    }

    void join() {
        std::unique_lock<std::mutex> lock(mutex);
        finished.wait(lock, [this] { return pending == 0; });
        if (error) {
            std::rethrow_exception(error);
        }
    }

private:
    void worker_loop() {
        while (true) {
            std::function<void()> function;
            {
                std::unique_lock<std::mutex> lock(mutex);
                work.wait(lock, [this] { return stopping || !tasks.empty(); });
                if (stopping && tasks.empty()) {
                    return;
                }
                function = std::move(tasks.front());
                tasks.pop_front();
            }

            try {
                function();
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex);
                if (!error) {
                    error = std::current_exception();
                }
            }

            {
                std::lock_guard<std::mutex> lock(mutex);
                GGML_ASSERT(pending > 0);
                --pending;
                if (pending == 0) {
                    finished.notify_all();
                }
            }
        }
    }

    std::vector<std::thread> workers;
    std::deque<std::function<void()>> tasks;
    std::mutex mutex;
    std::condition_variable work;
    std::condition_variable finished;
    size_t pending = 0;
    bool stopping = false;
    std::exception_ptr error;
};

std::string create_conversion_temp_path(const std::string & destination) {
#ifndef _WIN32
    std::string pattern = destination + ".tmp-convert-XXXXXX";
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    const int fd = ::mkstemp(name.data());
    if (fd < 0) {
        throw std::runtime_error("cannot create converted sequence state temporary file");
    }
    ::close(fd);
    return name.data();
#else
    static std::atomic<uint64_t> nonce{0};
    for (int attempt = 0; attempt < 32; ++attempt) {
        const std::string path = destination + ".tmp-convert-" + std::to_string((unsigned long) GetCurrentProcessId()) + "-" +
            std::to_string(nonce.fetch_add(1, std::memory_order_relaxed));
        const int fd = _open(path.c_str(), _O_CREAT | _O_EXCL | _O_BINARY | _O_RDWR,
                _S_IREAD | _S_IWRITE);
        if (fd >= 0) {
            _close(fd);
            return path;
        }
    }
    throw std::runtime_error("cannot create converted sequence state temporary file");
#endif
}

// Collects a host-only state stream (the metadata mirror owns no tensors).
class llama_io_write_vector final : public llama_io_write_i {
public:
    void write(const void * src, size_t size) override {
        const auto * bytes = static_cast<const uint8_t *>(src);
        data_.insert(data_.end(), bytes, bytes + size);
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        GGML_UNUSED(tensor);
        GGML_UNUSED(offset);
        GGML_UNUSED(size);
        throw std::runtime_error("metadata conversion stream contains a tensor payload");
    }

    size_t n_bytes() override { return data_.size(); }

    const std::vector<uint8_t> & data() const { return data_; }

private:
    std::vector<uint8_t> data_;
};

struct kvarn_convert_shape {
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t slices;
    uint32_t n_head_sliced;
    int32_t source_rotation = 0;
};

kvarn_convert_shape kvarn_convert_shape_for(uint32_t n_head_kv, uint32_t head_dim) {
    const int slices = llama_kvarn_head_slices(int(head_dim));
    if (slices <= 0 || head_dim % KVAR_N_GROUP != 0) {
        throw std::runtime_error(format("unsupported KVarN head dimension %u in conversion", head_dim));
    }
    return { n_head_kv, head_dim, uint32_t(slices), n_head_kv * uint32_t(slices) };
}

// Standalone record layouts: payload, then the three coefficient axes in the
// exact native order (scale axis, zero point, other axis). K records have the
// head dimension on the scale axis; V records have the token on it.
llama_kvarn_tile_layout kvarn_convert_k_layout(int bits) {
    llama_kvarn_tile_layout layout = {};
    layout.k_payload_off = 0;
    layout.k_payload_bytes = llama_kvarn_packed_bytes(KVAR_N_GROUP * KVAR_N_GROUP, bits);
    layout.k_s_col_off = layout.k_payload_bytes;
    layout.k_zp_off = layout.k_s_col_off + KVAR_N_GROUP * sizeof(uint16_t);
    layout.k_s_row_off = layout.k_zp_off + KVAR_N_GROUP * sizeof(uint16_t);
    layout.tile_bytes = layout.k_s_row_off + KVAR_N_GROUP * sizeof(uint16_t);
    return layout;
}

llama_kvarn_tile_layout kvarn_convert_v_layout(int bits) {
    llama_kvarn_tile_layout layout = {};
    layout.v_payload_off = 0;
    layout.v_payload_bytes = llama_kvarn_packed_bytes(KVAR_N_GROUP * KVAR_N_GROUP, bits);
    layout.v_s_row_off = layout.v_payload_bytes;
    layout.v_zp_off = layout.v_s_row_off + KVAR_N_GROUP * sizeof(uint16_t);
    layout.v_s_col_off = layout.v_zp_off + KVAR_N_GROUP * sizeof(uint16_t);
    layout.tile_bytes = layout.v_s_col_off + KVAR_N_GROUP * sizeof(uint16_t);
    return layout;
}

// Rotated-domain row for every head slice, rounded to F16 exactly like the
// runtime stage that the record quantizer reads.
void kvarn_convert_rotate_row(const float * row, const kvarn_convert_shape & shape, float * rotated) {
    const size_t row_len = size_t(shape.n_head_sliced) * KVAR_N_GROUP;
    std::copy(row, row + row_len, rotated);
    const uint32_t first = std::max(1, shape.source_rotation);
    const float scale = 1.0f / std::sqrt(float(shape.head_dim / first));
    for (uint32_t head = 0; head < shape.n_head_kv; ++head) {
        float * dst = rotated + size_t(head) * shape.head_dim;
        for (uint32_t stride = first; stride < shape.head_dim; stride *= 2) {
            for (uint32_t base = 0; base < shape.head_dim; base += 2 * stride) {
                for (uint32_t i = 0; i < stride; ++i) {
                    const float a = dst[base + i], b = dst[base + stride + i];
                    dst[base + i] = a + b;
                    dst[base + stride + i] = a - b;
                }
            }
        }
        for (uint32_t i = 0; i < shape.head_dim; ++i) {
            dst[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(dst[i] * scale));
        }
    }
}

void kvarn_convert_dequantize_row(const uint8_t * bytes, float * row, uint32_t n_embd) {
    dequantize_row_q4_0(reinterpret_cast<const block_q4_0 *>(bytes), row, n_embd);
}

// All head-slice records of one 128-token group for a single component.
// `rows` holds n_tokens contiguous source rows; `out` receives
// n_head_sliced * layout.tile_bytes bytes.
void kvarn_convert_component_records(
        const uint8_t * rows,
        uint32_t n_tokens,
        uint32_t row_bytes,
        uint32_t n_embd,
        const kvarn_convert_shape & shape,
        int bits,
        int sinkhorn_iters,
        bool value,
        uint8_t * out) {
    const llama_kvarn_tile_layout layout = value ? kvarn_convert_v_layout(bits) : kvarn_convert_k_layout(bits);

    std::vector<float> rotated(size_t(n_tokens) * shape.n_head_sliced * KVAR_N_GROUP);
    std::vector<float> row(n_embd);
    for (uint32_t token = 0; token < n_tokens; ++token) {
        kvarn_convert_dequantize_row(rows + size_t(token) * row_bytes, row.data(), n_embd);
        kvarn_convert_rotate_row(row.data(), shape,
                rotated.data() + size_t(token) * shape.n_head_sliced * KVAR_N_GROUP);
    }

    std::vector<float> tile(KVAR_N_GROUP * KVAR_N_GROUP);
    for (uint32_t hs = 0; hs < shape.n_head_sliced; ++hs) {
        for (uint32_t token = 0; token < n_tokens; ++token) {
            const float * src = rotated.data() +
                (size_t(token) * shape.n_head_sliced + hs) * KVAR_N_GROUP;
            if (value) {
                for (uint32_t d = 0; d < KVAR_N_GROUP; ++d) {
                    tile[size_t(token) * KVAR_N_GROUP + d] = src[d];
                }
            } else {
                for (uint32_t d = 0; d < KVAR_N_GROUP; ++d) {
                    tile[size_t(d) * KVAR_N_GROUP + token] = src[d];
                }
            }
        }
        uint8_t * record = out + size_t(hs) * layout.tile_bytes;
        if (value) {
            llama_kvarn_quantize_v_tile(tile.data(), sinkhorn_iters, bits, layout, record);
        } else {
            llama_kvarn_quantize_k_tile(tile.data(), sinkhorn_iters, bits, layout, record);
        }
    }
}

// Native exact tails store the original domain (store_tail copies current
// K/V directly), unlike the rotated records and stage. Undo only the source
// q4 Hadamard here; applying the destination rotation corrupts continuation.
void kvarn_convert_tail_row(
        const uint8_t * q4_row, uint32_t n_embd,
        const kvarn_convert_shape & shape, std::vector<uint8_t> & out) {
    std::vector<float> row(n_embd);
    kvarn_convert_dequantize_row(q4_row, row.data(), n_embd);
    const uint32_t width = std::max(1, shape.source_rotation);
    for (uint32_t base = 0; base < n_embd; base += width) {
        for (uint32_t stride = 1; stride < width; stride *= 2) {
            for (uint32_t b = 0; b < width; b += 2 * stride) {
                for (uint32_t i = 0; i < stride; ++i) {
                    const float a = row[base+b+i], c = row[base+b+stride+i];
                    row[base+b+i] = a+c;
                    row[base+b+stride+i] = a-c;
                }
            }
        }
    }
    out.resize(size_t(n_embd) * sizeof(ggml_fp16_t));
    for (size_t i = 0; i < row.size(); ++i) {
        const ggml_fp16_t half = ggml_fp32_to_fp16(row[i] / std::sqrt(float(width)));
        std::memcpy(out.data() + i * sizeof(half), &half, sizeof(half));
    }
}

// Fills the stage rows of one head slice from a group of source rows.
void kvarn_convert_stage_slice(
        const uint8_t * rows,
        uint32_t n_valid,
        uint32_t row_bytes,
        uint32_t n_embd,
        const kvarn_convert_shape & shape,
        uint32_t hs,
        uint32_t stage_slot,
        uint8_t * image) {
    std::vector<float> row(n_embd);
    std::vector<float> rotated(shape.n_head_sliced * KVAR_N_GROUP);
    for (uint32_t token = 0; token < n_valid; ++token) {
        kvarn_convert_dequantize_row(rows + size_t(token) * row_bytes, row.data(), n_embd);
        kvarn_convert_rotate_row(row.data(), shape, rotated.data());

        const size_t stage_pos = size_t(stage_slot) * KVAR_N_GROUP + token;
        ggml_fp16_t * dst = reinterpret_cast<ggml_fp16_t *>(image) +
            (stage_pos * shape.n_head_sliced + hs) * KVAR_N_GROUP;
        const float * src = rotated.data() + size_t(hs) * KVAR_N_GROUP;
        for (uint32_t d = 0; d < KVAR_N_GROUP; ++d) {
            dst[d] = ggml_fp32_to_fp16(src[d]);
        }
    }
}

} // namespace

bool llama_kv_cache_kvarn::state_streaming_restore_supported() const {
    // The bounded streaming reader can restore KVarN states whose payload
    // travels through indexed tensor reads: records, full stage images and
    // row-sized exact-tail payloads. SWA ring remapping still materializes
    // whole record vectors, and a pending stream copy owns the destination.
    return !migration_enabled && !swa && !has_pending_stream_copies();
}

bool llama_kv_cache_kvarn::state_parse_q4(
        llama_state_q4_source & src,
        const llama_hparams & hparams_ref,
        llama_state_q4_info & info,
        std::string & error) {
    if (standard_cache) {
        error = "Q4 state conversion is not supported for mixed per-layer KV layouts";
        return false;
    }
    std::vector<uint32_t> attn_layers;
    attn_layers.reserve(layers.size());
    for (const auto & layer : layers) {
        attn_layers.push_back(layer.il);
    }
    return llama_state_q4_parse(src, hparams_ref, attn_layers, metadata->has_cell_ext(), info, error);
}

size_t llama_kv_cache_kvarn::state_convert_q4(
        llama_state_q4_source & src,
        const llama_state_q4_info & info,
        const char * dst_path,
        std::vector<uint8_t> * out_mem) {
    if (standard_cache) {
        throw std::runtime_error("Q4 conversion is not supported for mixed per-layer KV layouts");
    }
    if ((dst_path == nullptr && out_mem == nullptr) || info.n_tokens == 0) {
        return 0;
    }
    if (swa) {
        throw std::runtime_error("q4 conversion requires a non-SWA destination");
    }
    if (n_stream != 1) {
        throw std::runtime_error("q4 conversion requires a single-stream destination");
    }
    if (info.layers.size() != layers.size()) {
        throw std::runtime_error("q4 conversion source layer count does not match the destination");
    }
    if (info.has_cell_ext != metadata->has_cell_ext()) {
        throw std::runtime_error("q4 conversion source cell-extension layout does not match the destination");
    }

    const uint32_t n_tokens = info.n_tokens;
    if (n_tokens > metadata->get_size()) {
        throw std::runtime_error("q4 conversion source does not fit the destination context");
    }
    const uint32_t n_groups_used = (n_tokens + KVAR_N_GROUP - 1) / KVAR_N_GROUP;
    if (n_groups_used > n_groups_per_stream) {
        throw std::runtime_error("q4 conversion source exceeds the destination record ring");
    }

    const size_t k_record_bytes = kvarn_record_bytes(KVAR_N_GROUP, params.key_bits, false);
    const size_t v_record_bytes = kvarn_record_bytes(KVAR_N_GROUP, params.value_bits, true);
    const llama_kvarn_tile_layout k_layout = kvarn_convert_k_layout(params.key_bits);
    const llama_kvarn_tile_layout v_layout = kvarn_convert_v_layout(params.value_bits);
    if (k_layout.tile_bytes != k_record_bytes || v_layout.tile_bytes != v_record_bytes) {
        throw std::runtime_error("q4 conversion destination record size is inconsistent");
    }

    for (size_t i = 0; i < layers.size(); ++i) {
        const auto & layer = layers[i];
        const auto & source = info.layers[i];
        if (source.il != layer.il) {
            throw std::runtime_error("q4 conversion source layer order does not match the destination");
        }
        if (source.k_row_size != ggml_row_size(GGML_TYPE_Q4_0, hparams.n_embd_k_gqa(layer.il)) ||
                source.v_row_size != ggml_row_size(GGML_TYPE_Q4_0, hparams.n_embd_v_gqa(layer.il))) {
            throw std::runtime_error("q4 conversion source row sizes do not match the destination");
        }
    }

    // Memory output writes straight into the caller's buffer (no file at all);
    // otherwise a temporary file in the destination directory is used and
    // atomically renamed only after the complete stream was written and synced.
    const std::string tmp_path = out_mem ? std::string() : create_conversion_temp_path(dst_path);

    struct tmp_guard {
        std::string path;
        bool armed = true;
        ~tmp_guard() {
            if (armed && !path.empty()) {
                std::error_code ec;
                std::filesystem::remove(path, ec);
            }
        }
    } guard { tmp_path };

    std::error_code ec;

    std::unique_ptr<llama_file> out;
    if (out_mem) {
        out_mem->clear();
    } else {
        out = std::make_unique<llama_file>(tmp_path.c_str(), "wb");
    }
    auto write_bytes = [&](const void * data, size_t size) {
        if (out_mem) {
            const uint8_t * p = static_cast<const uint8_t *>(data);
            out_mem->insert(out_mem->end(), p, p + size);
        } else {
            out->write_raw(data, size);
        }
    };
    auto write_u32 = [&](uint32_t value) {
        if (out_mem) {
            write_bytes(&value, sizeof(value));
        } else {
            out->write_u32(value);
        }
    };

    const uint32_t header[3] = { LLAMA_STATE_SEQ_MAGIC, LLAMA_STATE_SEQ_VERSION, n_tokens };
    write_bytes(header, sizeof(header));
    write_bytes(info.tokens.data(), size_t(n_tokens) * sizeof(llama_token));

    // Metadata prefix: populate a private metadata mirror and serialize it
    // with its own writer, so the destination parser sees exactly the native
    // manifest, body and exact-tail record layout.
    auto metadata_prepared = make_metadata_cache();
    {
        std::vector<llama_kv_cell_ext> exts;
        if (metadata->has_cell_ext()) {
            exts.resize(n_tokens);
            for (uint32_t i = 0; i < n_tokens; ++i) {
                llama_kv_cell_ext ext = info.exts[i];
                if (ext.tok == LLAMA_TOKEN_NULL) {
                    ext.tok = info.tokens[i];
                }
                exts[i] = ext;
            }
        }
        metadata_prepared->import_sequence_prefix(0, n_tokens, exts);
    }

    std::vector<int32_t> payload_slots;
    std::vector<uint32_t> payload_positions;
    if (exact_tail_tokens > 0) {
        const uint32_t tail_begin = n_tokens > exact_tail_tokens ? n_tokens - exact_tail_tokens : 0;
        metadata_prepared->import_sequence_tail(0, tail_begin, n_tokens);
        payload_slots = metadata_prepared->state_tail_payload_slots(0);
        std::unordered_map<int32_t, uint32_t> position_by_slot;
        for (const auto & entry : metadata_prepared->state_tail_snapshot(0)) {
            position_by_slot.emplace(entry.slot, uint32_t(entry.position));
        }
        for (const int32_t slot : payload_slots) {
            const auto it = position_by_slot.find(slot);
            if (it == position_by_slot.end()) {
                throw std::runtime_error("q4 conversion tail payload has no source position");
            }
            payload_positions.push_back(it->second);
        }
        if (payload_slots.size() != std::min<uint32_t>(n_tokens, exact_tail_tokens)) {
            throw std::runtime_error("q4 conversion tail payload count is inconsistent");
        }
    }

    {
        llama_io_write_vector metadata_io;
        metadata_prepared->state_write(metadata_io, 0, 0);
        write_bytes(metadata_io.data().data(), metadata_io.data().size());
    }

    // KVarN header: a full record image plus the complete F16 stage image.
    write_u32(KVAR_N_STATE_MAGIC);
    write_u32(KVAR_N_STATE_VERSION);
    write_u32(uint32_t(params.type));
    write_u32(uint32_t(layers.size()));
    write_u32(1); // saved streams
    write_u32(0); // stream 0
    write_u32(KVAR_N_STATE_RECORDS_FULL);
    write_u32(stage_groups);
    write_u32(tail_groups);
    write_u32(exact_tail_tokens > 0 ? 1u : 0u);
    write_u32(exact_tail_tokens);
    write_u32(uint32_t(int32_t(exact_tail_type)));
    write_u32(uint32_t(payload_slots.size()));
    write_u32(n_groups_used);
    const llama_pos saved_pos_max = llama_pos(n_tokens) - 1;
    write_bytes(&saved_pos_max, sizeof(saved_pos_max));
    write_u32(0); // selective stage rows
    write_u32(0); // selective record groups

    const uint32_t complete_groups = n_tokens / KVAR_N_GROUP;
    const uint32_t last_group = (n_tokens - 1) / KVAR_N_GROUP;
    const uint32_t stage_first_group = last_group >= tail_groups ? last_group - tail_groups + 1 : 1;

    std::vector<uint32_t> staged_groups;
    staged_groups.push_back(0);
    for (uint32_t group = stage_first_group; group <= last_group; ++group) {
        if (group != 0) {
            staged_groups.push_back(group);
        }
    }

    const unsigned hw = std::thread::hardware_concurrency();
    const uint32_t n_workers = std::max(1u, std::min(16u, hw ? hw : 1u));
    conversion_thread_pool workers(n_workers);
    uint64_t gpu_groups = 0, cpu_groups = 0;

    for (size_t li = 0; li < layers.size(); ++li) {
        const auto & layer = layers[li];
        const auto & source = info.layers[li];

        kvarn_convert_shape k_shape = kvarn_convert_shape_for(layer.n_head_kv, layer.head_dim_k);
        kvarn_convert_shape v_shape = kvarn_convert_shape_for(layer.n_head_kv, layer.head_dim_v);

        k_shape.source_rotation = info.rotation_k;
        v_shape.source_rotation = info.rotation_v;
        for (const auto & shape : {k_shape, v_shape}) {
            const int r = shape.source_rotation;
            if (r < 0 || (r && ((r & (r-1)) || r > int(shape.head_dim) || shape.head_dim % r))) {
                throw std::runtime_error("unsupported source attention rotation");
            }
        }

        write_u32(layer.il);
        write_u32(0); // stream

        struct component {
            bool value;
            uint64_t data;
            uint64_t row_size;
            uint32_t n_embd;
            kvarn_convert_shape shape;
            size_t record_bytes;
            int bits;
            uint64_t saved_size;
        };
        const component components[2] = {
            { false, source.k_data, source.k_row_size, hparams.n_embd_k_gqa(layer.il), k_shape,
              k_record_bytes, params.key_bits,
              uint64_t(n_groups_used) * uint64_t(layer.k_records_stream[0]->nb[2]) },
            { true, source.v_data, source.v_row_size, hparams.n_embd_v_gqa(layer.il), v_shape,
              v_record_bytes, params.value_bits,
              uint64_t(n_groups_used) * uint64_t(layer.v_records_stream[0]->nb[2]) },
        };

        for (const auto & component : components) {
            write_bytes(&component.saved_size, sizeof(component.saved_size));

            const size_t block_bytes = size_t(component.shape.n_head_sliced) * component.record_bytes;
            const std::vector<uint8_t> zeros(block_bytes, 0);

            // Stream bounded GPU chunks into the output sink. Never build an
            // entire component's record image on the host. A failed GPU chunk
            // is recomputed on CPU before publishing that chunk.
            backend_kvarn_convert_q4_t gpu_convert =
                kvarn_convert_gpu_enabled() ? kvarn_convert_gpu_proc() : nullptr;
            if (gpu_convert != nullptr && complete_groups > 1) {
                write_bytes(zeros.data(), zeros.size()); // sink group
                const uint32_t chunk_groups = 32;
                for (uint32_t g = 1; g < complete_groups; g += chunk_groups) {
                    const uint32_t count = std::min(chunk_groups, complete_groups - g);
                    const size_t rows_bytes = size_t(count) * KVAR_N_GROUP * component.row_size;
                    std::vector<uint8_t> rows_in(rows_bytes);
                    src.seek(component.data + size_t(g) * KVAR_N_GROUP * component.row_size);
                    src.read_raw(rows_in.data(), rows_in.size());
                    std::vector<uint8_t> chunk_out(size_t(count) * block_bytes, 0);
                    const bool gpu_ok = gpu_convert(
                            rows_in.data(), size_t(component.row_size),
                            int(count * KVAR_N_GROUP), int(component.n_embd),
                            int(layer.n_head_kv),
                            int(component.value ? layer.head_dim_v : layer.head_dim_k),
                            int(component.bits), int(params.sinkhorn_iters), component.value,
                            component.shape.source_rotation, chunk_out.data());
                    if (gpu_ok) { gpu_groups += count; }
                    else { cpu_groups += count; }
                    if (!gpu_ok) {
                        for (uint32_t i = 0; i < count; ++i) {
                            workers.submit([&, i] {
                                kvarn_convert_component_records(
                                    rows_in.data() + size_t(i) * KVAR_N_GROUP * component.row_size,
                                    KVAR_N_GROUP, uint32_t(component.row_size), component.n_embd,
                                    component.shape, component.bits, params.sinkhorn_iters,
                                    component.value, chunk_out.data() + size_t(i) * block_bytes);
                            });
                        }
                        workers.join();
                    }
                    write_bytes(chunk_out.data(), chunk_out.size());
                }
                for (uint32_t g = complete_groups; g < n_groups_used; ++g) {
                    write_bytes(zeros.data(), zeros.size());
                }
                continue;
            }

            std::vector<std::vector<uint8_t>> in(n_workers);
            std::vector<std::vector<uint8_t>> out_buf(n_workers);

            for (uint32_t g0 = 0; g0 < n_groups_used; g0 += n_workers) {
                const uint32_t count = std::min(n_workers, n_groups_used - g0);
                for (uint32_t i = 0; i < count; ++i) {
                    const uint32_t group = g0 + i;
                    in[i].clear();
                    out_buf[i].clear();
                    if (group == 0 || group >= complete_groups) {
                        continue;
                    }
                    in[i].resize(size_t(KVAR_N_GROUP) * component.row_size);
                    src.seek(component.data + size_t(group) * KVAR_N_GROUP * component.row_size);
                    src.read_raw(in[i].data(), in[i].size());
                    out_buf[i].assign(block_bytes, 0);
                }

                for (uint32_t i = 0; i < count; ++i) {
                    const uint32_t group = g0 + i;
                    if (group == 0 || group >= complete_groups) {
                        continue;
                    }
                    workers.submit([&, i] {
                        kvarn_convert_component_records(
                                in[i].data(), KVAR_N_GROUP, uint32_t(component.row_size),
                                component.n_embd, component.shape, component.bits,
                                params.sinkhorn_iters, component.value, out_buf[i].data());
                    });
                }
                workers.join();

                for (uint32_t i = 0; i < count; ++i) {
                    const uint32_t group = g0 + i;
                    if (group == 0 || group >= complete_groups) {
                        write_bytes(zeros.data(), zeros.size());
                    } else {
                        ++cpu_groups;
                        write_bytes(out_buf[i].data(), out_buf[i].size());
                    }
                }
            }
        }

        // Full F16 stage image: sink slot plus the live tail groups.
        for (const auto & stage : components) {
            const uint64_t stage_size = uint64_t(ggml_nbytes(stage.value
                    ? layer.v_stage_stream[0] : layer.k_stage_stream[0]));
            write_bytes(&stage_size, sizeof(stage_size));

            std::vector<uint8_t> image(
                    size_t(KVAR_N_GROUP) * stage_groups * stage.shape.n_head_sliced *
                    KVAR_N_GROUP * sizeof(uint16_t), 0);

            for (const uint32_t group : staged_groups) {
                const uint32_t n_valid = std::min<uint32_t>(KVAR_N_GROUP,
                        n_tokens - group * KVAR_N_GROUP);
                const uint32_t slot = group == 0 ? 0 : 1 + ((group - 1) % tail_groups);

                std::vector<uint8_t> rows(size_t(n_valid) * stage.row_size);
                src.seek(stage.data + size_t(group) * KVAR_N_GROUP * stage.row_size);
                src.read_raw(rows.data(), rows.size());

                for (uint32_t hs = 0; hs < stage.shape.n_head_sliced; ++hs) {
                    workers.submit([&, hs] {
                        kvarn_convert_stage_slice(rows.data(), n_valid, uint32_t(stage.row_size),
                                stage.n_embd, stage.shape, hs, slot, image.data());
                    });
                }
                workers.join();
            }

            write_bytes(image.data(), image.size());
        }

        // Exact tail rows, payload order (per layer: K rows, then V rows).
        const uint64_t k_tail_row = layer.k_tail
            ? uint64_t(ggml_row_size(layer.k_tail->type, layer.k_tail->ne[0])) : 0;
        const uint64_t v_tail_row = layer.v_tail
            ? uint64_t(ggml_row_size(layer.v_tail->type, layer.v_tail->ne[0])) : 0;
        write_bytes(&k_tail_row, sizeof(k_tail_row));
        write_bytes(&v_tail_row, sizeof(v_tail_row));
        if ((k_tail_row != 0) != (exact_tail_tokens > 0) ||
                (v_tail_row != 0) != (exact_tail_tokens > 0)) {
            throw std::runtime_error("q4 conversion destination tail layout is inconsistent");
        }
        if (!payload_positions.empty()) {
            for (const auto & tail : components) {
                std::vector<uint8_t> row_bytes(size_t(tail.row_size));
                std::vector<uint8_t> tail_out;
                for (const uint32_t pos : payload_positions) {
                    src.seek(tail.data + size_t(pos) * tail.row_size);
                    src.read_raw(row_bytes.data(), row_bytes.size());
                    kvarn_convert_tail_row(row_bytes.data(), tail.n_embd, tail.shape, tail_out);
                    write_bytes(tail_out.data(), tail_out.size());
                }
            }
        }
    }

    // Hybrid models carry a recurrent/conv state after the attention state.
    // It is not part of the quantized representation and travels verbatim.
    if (info.recr_bytes != 0) {
        std::vector<uint8_t> buffer(LLAMA_STATE_FILE_BUFFER_SIZE);
        src.seek(info.recr_offset);
        for (uint64_t left = info.recr_bytes; left;) {
            const size_t count = size_t(std::min<uint64_t>(left, buffer.size()));
            src.read_raw(buffer.data(), count);
            write_bytes(buffer.data(), count);
            left -= count;
        }
    }

    const size_t total = out_mem ? out_mem->size() : out->tell();
    if (!out_mem) {
#ifndef _WIN32
        if (::fsync(out->file_id()) != 0) {
            throw std::runtime_error("converted sequence state sync failed");
        }
#endif
        out->close();
        std::filesystem::rename(tmp_path, dst_path, ec);
        if (ec) {
            throw std::runtime_error("converted sequence state publication failed: " + ec.message());
        }
    }
    guard.armed = false;
    LLAMA_LOG_INFO("%s: converted q4_0 state into %s (tokens=%u bytes=%zu type=%s gpu_groups=%llu cpu_groups=%llu)\n",
            __func__, out_mem ? "memory" : dst_path, n_tokens, total, llama_kvarn_type_name(params.type),
            (unsigned long long) gpu_groups, (unsigned long long) cpu_groups);
    return total;
}
