// Conversion tests: q4_0/q4_0 -> KVarN4/KVarN4 sequence state conversion.
//
// Covers the file and in-memory sources sharing one parser, prefix boundaries
// around the 128-token group size, incomplete groups, continuation after
// generation, an independent numeric reference for orientation/rotation/stage
// placement, native save/load of the converted state, and failure paths
// (corruption, truncation, unsupported formats, occupied destination, late
// I/O) that must never publish a partial cache or destroy the source.

#include "common.h"
#include "llama-cpp.h"
#include "../src/llama-kvarn.h"
#include "../src/llama-state-q4.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <typeinfo>
#include <vector>

#include "../src/llama-io-file.h"
#include "../src/llama-model.h"

#define XXH_INLINE_ALL
#include "../vendor/hash/xxhash/xxhash.h"

struct block_q4_0;
extern "C" {
void dequantize_row_q4_0(const block_q4_0 * x, float * y, int64_t k);
}

static void require(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

static std::vector<uint8_t> read_file(const std::string & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(bool(input), "cannot open state file");
    const auto size = size_t(input.tellg());
    input.seekg(0);
    std::vector<uint8_t> bytes(size);
    input.read(reinterpret_cast<char *>(bytes.data()), (std::streamsize) size);
    require(input.good() || input.gcount() == (std::streamsize) size, "short state file read");
    return bytes;
}

static uint64_t checksum(const std::vector<uint8_t> & bytes) {
    return XXH64(bytes.data(), bytes.size(), 0);
}

static void write_file(const std::string & path, const std::vector<uint8_t> & bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char *>(bytes.data()), (std::streamsize) bytes.size());
    require(out.good(), "cannot write state file");
}

static void decode(llama_context * ctx, const llama_tokens & tokens, int start = 0) {
    for (size_t i = 0; i < tokens.size();) {
        const int count = std::min<size_t>(llama_n_batch(ctx), tokens.size() - i);
        llama_batch batch = llama_batch_init(count, 0, 1);
        for (int j = 0; j < count; ++j) {
            common_batch_add(batch, tokens[i + j], start + int(i) + j, {0}, j == count - 1);
        }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        require(rc == 0, "decode failed");
        i += count;
    }
}

// softmax KL(p||q) over the last logits of two contexts.
static double logits_kl(llama_context * a, llama_context * b, int n_vocab, float * max_diff) {
    const float * pa = llama_get_logits_ith(a, -1);
    const float * pb = llama_get_logits_ith(b, -1);
    require(pa && pb, "missing logits");
    std::vector<double> p(n_vocab), q(n_vocab);
    double m_p = -1e30, m_q = -1e30;
    for (int i = 0; i < n_vocab; ++i) { m_p = std::max(m_p, (double) pa[i]); m_q = std::max(m_q, (double) pb[i]); }
    double sp = 0, sq = 0;
    for (int i = 0; i < n_vocab; ++i) { p[i] = std::exp((double) pa[i] - m_p); q[i] = std::exp((double) pb[i] - m_q); sp += p[i]; sq += q[i]; }
    double kl = 0;
    float md = 0;
    for (int i = 0; i < n_vocab; ++i) {
        p[i] /= sp; q[i] /= sq;
        md = std::max(md, std::fabs(pa[i] - pb[i]));
        if (p[i] > 0) { kl += p[i] * std::log(p[i] / std::max(q[i], 1e-30)); }
    }
    if (max_diff) { *max_diff = md; }
    return kl;
}

// --- converted-file reader: independent of the producer ------------------

struct kvarn_reader {
    struct layer {
        uint32_t il = 0;
        std::vector<uint8_t> k_records;
        std::vector<uint8_t> v_records;
        std::vector<uint8_t> k_stage;
        std::vector<uint8_t> v_stage;
        std::vector<uint8_t> k_tail;
        std::vector<uint8_t> v_tail;
    };
    uint32_t n_tokens = 0;
    std::vector<llama_token> tokens;
    uint32_t stage_groups = 0;
    uint32_t tail_groups = 0;
    uint32_t n_groups_used = 0;
    uint32_t type = 0;
    uint32_t tail_tokens = 0;
    uint32_t n_exact_payloads = 0;
    uint64_t recr_bytes = 0;
    std::vector<layer> layers;
};

class cursor {
public:
    explicit cursor(const std::vector<uint8_t> & bytes) : bytes(bytes) {}
    template<typename T> T read() {
        require(offset + sizeof(T) <= bytes.size(), "converted file truncated");
        T value;
        std::memcpy(&value, bytes.data() + offset, sizeof(T));
        offset += sizeof(T);
        return value;
    }
    std::vector<uint8_t> read_bytes(uint64_t size) {
        require(size <= bytes.size() && offset + size <= bytes.size(), "converted file truncated");
        std::vector<uint8_t> value(bytes.begin() + offset, bytes.begin() + offset + size);
        offset += size;
        return value;
    }
    uint64_t offset = 0;
    const std::vector<uint8_t> & bytes;
};

static kvarn_reader parse_kvarn(const std::vector<uint8_t> & bytes, uint32_t n_layer, bool has_cell_ext) {
    kvarn_reader out;
    cursor cur(bytes);
    require(cur.read<uint32_t>() == 0x67677371u, "converted file magic");
    require(cur.read<uint32_t>() == 3u, "converted file version");
    out.n_tokens = cur.read<uint32_t>();
    out.tokens.resize(out.n_tokens);
    for (auto & token : out.tokens) { token = cur.read<llama_token>(); }

    // The metadata prefix is either the framed compact-tail state (magic
    // KVTL with explicit section sizes) or the legacy layer-less body.
    const uint32_t meta_magic = cur.read<uint32_t>();
    if (meta_magic == 0x4c54564bu) {
        cur.read<uint32_t>(); // version
        cur.read<uint32_t>(); // flags
        out.tail_tokens = cur.read<uint32_t>();
        cur.read<int32_t>();  // tail type
        cur.read<uint32_t>(); // storage kind
        cur.read<uint32_t>(); // rollback tokens
        const uint32_t group_len = cur.read<uint32_t>();
        require(group_len > 0 && group_len < 256, "converted metadata group id");
        cur.offset += group_len;
        const uint64_t manifest_size = cur.read<uint64_t>();
        const uint64_t body_size = cur.read<uint64_t>();
        const uint64_t tail_size = cur.read<uint64_t>();
        require(manifest_size > 0 && manifest_size < bytes.size() &&
                body_size < bytes.size() && tail_size < bytes.size(),
                "converted metadata section sizes");
        cur.offset += manifest_size + body_size + tail_size;
    } else {
        const uint32_t meta_stream = meta_magic;
        const uint32_t cell_count = cur.read<uint32_t>();
        require(meta_stream == 1 && cell_count == out.n_tokens, "converted metadata header");
        for (uint32_t i = 0; i < cell_count; ++i) {
            require(cur.read<llama_pos>() == (llama_pos) i, "converted metadata position");
            require(cur.read<uint32_t>() == 1u, "converted metadata owner count");
            if (has_cell_ext) {
                cur.read<llama_pos>(); // x
                cur.read<llama_pos>(); // y
                cur.read<llama_token>(); // tok
            }
            require(cur.read<llama_seq_id>() == 0, "converted metadata owner");
        }
        require(cur.read<uint32_t>() == 0u, "converted metadata v_trans");
        require(cur.read<uint32_t>() == 0u, "converted metadata layer count");
    }

    require(cur.read<uint32_t>() == 0x4e52564bu, "converted KVarN magic");
    require(cur.read<uint32_t>() == 16u, "converted KVarN version");
    out.type = cur.read<uint32_t>();
    require(cur.read<uint32_t>() == n_layer, "converted layer count");
    require(cur.read<uint32_t>() == 1u, "converted stream count");
    require(cur.read<uint32_t>() == 0u, "converted stream id");
    require(cur.read<uint32_t>() == 0u, "converted state kind is not full");
    out.stage_groups = cur.read<uint32_t>();
    out.tail_groups = cur.read<uint32_t>();
    const uint32_t has_exact_tail = cur.read<uint32_t>();
    const uint32_t exact_tail_tokens = cur.read<uint32_t>();
    cur.read<int32_t>(); // exact tail type
    out.n_exact_payloads = cur.read<uint32_t>();
    if (has_exact_tail) {
        require(out.tail_tokens == 0 || exact_tail_tokens == out.tail_tokens,
                "converted exact tail token count");
        require(out.n_exact_payloads == std::min<uint32_t>(out.n_tokens, exact_tail_tokens),
                "converted exact payload count");
    } else {
        require(exact_tail_tokens == 0 && out.n_exact_payloads == 0, "converted tail header");
    }
    out.n_groups_used = cur.read<uint32_t>();
    require(cur.read<llama_pos>() == (llama_pos) out.n_tokens - 1, "converted position");
    require(cur.read<uint32_t>() == 0u, "converted selective stage rows");
    require(cur.read<uint32_t>() == 0u, "converted selective record groups");

    out.layers.resize(n_layer);
    for (auto & layer : out.layers) {
        layer.il = cur.read<uint32_t>();
        require(cur.read<uint32_t>() == 0u, "converted layer stream id");
        layer.k_records = cur.read_bytes(cur.read<uint64_t>());
        layer.v_records = cur.read_bytes(cur.read<uint64_t>());
        layer.k_stage = cur.read_bytes(cur.read<uint64_t>());
        layer.v_stage = cur.read_bytes(cur.read<uint64_t>());
        const uint64_t k_tail_row = cur.read<uint64_t>();
        const uint64_t v_tail_row = cur.read<uint64_t>();
        if (out.n_exact_payloads != 0) {
            require(k_tail_row != 0 && v_tail_row != 0, "converted tail row sizes");
        }
        layer.k_tail = cur.read_bytes(k_tail_row * out.n_exact_payloads);
        layer.v_tail = cur.read_bytes(v_tail_row * out.n_exact_payloads);
    }
    // The hybrid recurrent/conv section follows the attention state verbatim.
    require(cur.offset <= bytes.size(), "converted file trailing bytes");
    out.recr_bytes = bytes.size() - cur.offset;
    return out;
}

// --- independent rotation reference --------------------------------------

static void reference_rotate(const float * row, uint32_t n_head_kv, uint32_t head_dim, std::vector<float> & rotated) {
    const uint32_t slices = head_dim / 128;
    rotated.assign(size_t(n_head_kv) * head_dim, 0.0f);
    for (uint32_t head = 0; head < n_head_kv; ++head) {
        for (uint32_t slice = 0; slice < slices; ++slice) {
            float slice_values[128];
            std::memcpy(slice_values, row + size_t(head) * head_dim + size_t(slice) * 128, sizeof(slice_values));
            for (uint32_t stride = 1; stride < 128; stride *= 2) {
                for (uint32_t base = 0; base < 128; base += 2 * stride) {
                    for (uint32_t i = 0; i < stride; ++i) {
                        const float a = slice_values[base + i];
                        const float b = slice_values[base + stride + i];
                        slice_values[base + i] = a + b;
                        slice_values[base + stride + i] = a - b;
                    }
                }
            }
            for (uint32_t i = 0; i < 128; ++i) {
                rotated[size_t(head) * head_dim + size_t(slice) * 128 + i] =
                    ggml_fp16_to_fp32(ggml_fp32_to_fp16(slice_values[i] * 0.08838834764831845f));
            }
        }
    }
}

static void q4_row(const uint8_t * bytes, float * row, uint32_t n_embd) {
    dequantize_row_q4_0(reinterpret_cast<const block_q4_0 *>(bytes), row, n_embd);
}

struct source_info {
    llama_state_q4_info info;
    std::vector<uint8_t> bytes;
};

static std::vector<uint32_t> attention_layers(const llama_hparams & hparams) {
    std::vector<uint32_t> attn_layers;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.is_recr(il)) {
            attn_layers.push_back(il);
        }
    }
    return attn_layers;
}

static source_info parse_source(const std::string & path, const llama_hparams & hparams) {
    source_info out;
    out.bytes = read_file(path);
    struct memory_source final : llama_state_q4_source {
        const std::vector<uint8_t> & bytes;
        uint64_t pos = 0;
        explicit memory_source(const std::vector<uint8_t> & bytes) : bytes(bytes) {}
        uint64_t size() const override { return bytes.size(); }
        uint64_t tell() const override { return pos; }
        void seek(uint64_t offset) override { require(offset <= bytes.size(), "source seek"); pos = offset; }
        void read_raw(void * dst, size_t size) override {
            require(size <= bytes.size() - pos, "source read");
            std::memcpy(dst, bytes.data() + pos, size);
            pos += size;
        }
    } source(out.bytes);
    // Skip the canonical file header; the body parser starts at the memory.
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t n_tokens = 0;
    source.read_raw(&magic, sizeof(magic));
    require(magic == 0x67677371u && (source.read_raw(&version, sizeof(version)),
            source.read_raw(&n_tokens, sizeof(n_tokens)), version == 3u),
            "source file header");
    source.seek(3 * sizeof(uint32_t) + size_t(n_tokens) * sizeof(llama_token));
    out.info.n_tokens = n_tokens;

    std::string error;
    const bool has_ext = hparams.n_pos_per_embd() > 1 || hparams.ple_n_heads > 0;
    const bool parsed = llama_state_q4_parse(
            source, hparams, attention_layers(hparams), has_ext, out.info, error);
    if (!parsed) {
        throw std::runtime_error(error.empty() ? "source parse failed" : error);
    }
    return out;
}

int main(int argc, char ** argv) {
    try {
        require(argc == 2, "usage: test-state-convert-q4-kvarn MODEL");
        const char * tmpdir = std::getenv("TMPDIR");
        require(tmpdir && *tmpdir, "TMPDIR must be an explicit disk directory");
        std::string pattern = std::string(tmpdir) + "/convert-XXXXXX";
        std::vector<char> name(pattern.begin(), pattern.end()); name.push_back(0);
        require(mkdtemp(name.data()) != nullptr, "mkdtemp failed");
        const std::string dir(name.data());

        ggml_backend_load_all();
        auto mp = llama_model_default_params();
        ggml_backend_dev_t devices[] = {nullptr};
        mp.n_gpu_layers = 0; mp.devices = devices;
        llama_model_ptr model(llama_model_load_from_file(argv[1], mp));
        require(bool(model), "model load failed");
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        const llama_hparams & hparams = model->hparams;
        const std::vector<uint32_t> attn_layers = attention_layers(hparams);
        const uint32_t n_layer = uint32_t(attn_layers.size());
        const bool has_cell_ext = hparams.n_pos_per_embd() > 1 || hparams.ple_n_heads > 0;

        auto base_params = [&]() {
            auto params = llama_context_default_params();
            params.n_ctx = 512;
            params.n_batch = 128;
            params.n_ubatch = 128;
            params.n_threads = params.n_threads_batch = 2;
            params.type_k = params.type_v = GGML_TYPE_Q4_0;
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            params.offload_kqv = false;
            return params;
        };
        auto kvarn_params = [&]() {
            auto params = base_params();
            params.n_batch = params.n_ubatch = 64;
            params.kvarn = llama_kvarn_params_for_type(LLAMA_KVARN_K4V4_G128);
            return params;
        };

        // Prefix boundaries around the 128-token group size, an incomplete
        // group, and a continuation after generation.
        const int boundaries[] = { 1, 127, 128, 129, 255, 256, 257, 300 };
        double worst_kl_native = 0.0;
        double worst_kl_source = 0.0;
        float worst_diff_native = 0.0f;
        double worst_record_rel = 0.0;
        for (const int prompt_len : boundaries) {
            llama_tokens tokens;
            for (int i = 0; i < prompt_len + 4; ++i) { tokens.push_back(3 + (i * 17 + prompt_len) % (n_vocab - 3)); }
            const llama_tokens evaluated(tokens.begin(), tokens.begin() + prompt_len);

            auto src_params = base_params();
            llama_context_ptr source(llama_init_from_model(model.get(), src_params));
            require(bool(source), "source context failed");
            decode(source.get(), evaluated);

            const std::string q4_path = dir + "/q4-" + std::to_string(prompt_len) + ".bin";
            const size_t written = llama_state_seq_save_file(source.get(), q4_path.c_str(), 0,
                    evaluated.data(), evaluated.size());
            require(written > 0, "q4 save failed");

            llama_context_ptr dest(llama_init_from_model(model.get(), kvarn_params()));
            require(bool(dest), "conversion destination context failed");

            const auto q4_bytes = read_file(q4_path);
            const std::string kvarn_path = dir + "/kvarn4-" + std::to_string(prompt_len) + ".bin";
            llama_tokens out_tokens(evaluated.size());
            size_t out_count = 777;
            const size_t converted = llama_state_seq_convert_file(dest.get(), q4_path.c_str(), 0,
                    q4_bytes.size(), 0, kvarn_path.c_str(), out_tokens.data(), out_tokens.size(), &out_count);
            require(converted > 0 && out_count == evaluated.size() && out_tokens == evaluated,
                    "conversion failed or returned wrong tokens");
            require(read_file(q4_path) == q4_bytes, "conversion modified the q4 source");

            // The same conversion through a real in-memory snapshot (the
            // state_seq_get_data form, which carries no token header) must
            // produce the identical converted state file.
            {
                llama_context_ptr dest_ram(llama_init_from_model(model.get(), kvarn_params()));
                require(bool(dest_ram), "RAM conversion destination context failed");
                const size_t ram_size = llama_state_seq_get_size_ext(source.get(), 0,
                        LLAMA_STATE_SEQ_FLAGS_NONE);
                require(ram_size > 0, "RAM snapshot size failed");
                std::vector<uint8_t> ram_data(ram_size);
                require(llama_state_seq_get_data_ext(source.get(), ram_data.data(), ram_data.size(), 0,
                            LLAMA_STATE_SEQ_FLAGS_NONE) == ram_data.size(), "RAM snapshot failed");
                const std::string ram_path = dir + "/kvarn4-ram-" + std::to_string(prompt_len) + ".bin";
                llama_tokens ram_tokens(evaluated.size());
                size_t ram_count = 0;
                const size_t ram_converted = llama_state_seq_convert_data(dest_ram.get(),
                        ram_data.data(), ram_data.size(), 0,
                        evaluated.data(), evaluated.size(), ram_path.c_str(),
                        ram_tokens.data(), ram_tokens.size(), &ram_count);
                require(ram_converted == converted && ram_count == evaluated.size() && ram_tokens == evaluated,
                        "RAM conversion differs from file conversion");
                require(read_file(ram_path) == read_file(kvarn_path), "RAM conversion bytes differ");
            }

            // Restore through the shared streaming loader into the empty child.
            const auto kvarn_bytes = read_file(kvarn_path);
            llama_tokens restored(evaluated.size());
            size_t restored_count = 0;
            const size_t read = llama_state_seq_load_file_streaming(dest.get(), kvarn_path.c_str(), 0,
                    restored.data(), restored.size(), &restored_count, kvarn_bytes.size(), checksum(kvarn_bytes));
            require(read == kvarn_bytes.size() && restored_count == evaluated.size() && restored == evaluated,
                    "streaming restore of converted state failed");
            require(llama_memory_seq_pos_max(llama_get_memory(dest.get()), 0) == (llama_pos) prompt_len - 1,
                    "converted restore position mismatch");

            // Failure: a nonempty destination must be rejected unchanged.
            {
                llama_tokens probe(evaluated.size());
                size_t probe_count = 0;
                const size_t before_pos = llama_memory_seq_pos_max(llama_get_memory(dest.get()), 0);
                require(llama_state_seq_load_file_streaming(dest.get(), kvarn_path.c_str(), 0,
                            probe.data(), probe.size(), &probe_count, kvarn_bytes.size(), checksum(kvarn_bytes)) == 0,
                        "nonempty destination accepted converted state");
                require(llama_memory_seq_pos_max(llama_get_memory(dest.get()), 0) == before_pos &&
                        llama_memory_seq_pos_max(llama_get_memory(dest.get()), 0) == (llama_pos) prompt_len - 1,
                        "rejected conversion restore damaged the destination");
            }

            const auto parsed = parse_kvarn(kvarn_bytes, n_layer, has_cell_ext);
            require(parsed.n_tokens == (uint32_t) prompt_len && parsed.tokens == evaluated,
                    "converted file token roundtrip");
            require(parsed.stage_groups >= 2 && parsed.tail_groups >= 1, "converted stage layout");
            require(parsed.n_groups_used == (uint32_t) ((prompt_len + 127) / 128), "converted group count");
            require(parsed.type == (uint32_t) LLAMA_KVARN_K4V4_G128, "converted KVarN type");

            // --- independent numeric reference for layer 0 -----------------
            // Group 0 K rows live in the stage as F16 rotations of the q4 rows.
            {
                const auto source_parsed = parse_source(q4_path, hparams);
                require(source_parsed.info.layers.size() == n_layer, "source parser layer count");
                const auto & sl = source_parsed.info.layers[0];
                require(sl.il == parsed.layers[0].il, "source/destination layer order");
                const uint32_t n_head_kv = hparams.n_head_kv(0);
                const uint32_t head_dim_k = hparams.n_embd_head_k(0);
                const uint32_t n_embd_k = hparams.n_embd_k_gqa(0);
                const uint32_t n_head_sliced = n_head_kv * (head_dim_k / 128);
                const uint32_t n_valid_group0 = std::min<uint32_t>(128, prompt_len);

                std::vector<float> row(n_embd_k), rotated;
                std::vector<uint8_t> raw(sl.k_row_size);
                for (uint32_t token = 0; token < n_valid_group0; token += 17) {
                    std::memcpy(raw.data(), source_parsed.bytes.data() + sl.k_data + size_t(token) * sl.k_row_size,
                            raw.size());
                    q4_row(raw.data(), row.data(), n_embd_k);
                    reference_rotate(row.data(), n_head_kv, head_dim_k, rotated);
                    // K records are split into 128-dim slices; the fixture has
                    // one slice per head.
                    for (uint32_t hs = 0; hs < n_head_sliced; ++hs) {
                        for (uint32_t d = 0; d < 128; ++d) {
                            const size_t idx = (size_t(token) * n_head_sliced + hs) * 128 + d;
                            require(2 * idx + 2 <= parsed.layers[0].k_stage.size(), "stage extent");
                            const ggml_fp16_t stored = *reinterpret_cast<const ggml_fp16_t *>(
                                    parsed.layers[0].k_stage.data() + 2 * idx);
                            const float expected = rotated[size_t(hs) * 128 + d];
                            require(std::fabs(ggml_fp16_to_fp32(stored) - expected) <=
                                    1e-3f + 1e-3f * std::fabs(expected),
                                    "stage K value does not match the rotated q4 reference");
                        }
                    }
                }

                // Exact tail: the last payload row must be the rotated F16 row
                // of the last token, not the unrotated domain.
                if (prompt_len > 0 && parsed.n_exact_payloads != 0) {
                    const uint32_t last = uint32_t(prompt_len - 1);
                    std::memcpy(raw.data(), source_parsed.bytes.data() + sl.k_data + size_t(last) * sl.k_row_size,
                            raw.size());
                    q4_row(raw.data(), row.data(), n_embd_k);
                    reference_rotate(row.data(), n_head_kv, head_dim_k, rotated);
                    const size_t tail_row_bytes = parsed.layers[0].k_tail.size() / parsed.n_exact_payloads;
                    require(tail_row_bytes == rotated.size() * sizeof(ggml_fp16_t), "tail row extent");
                    const uint8_t * last_tail = parsed.layers[0].k_tail.data() +
                        (parsed.n_exact_payloads - 1) * tail_row_bytes;
                    for (size_t i = 0; i < rotated.size(); ++i) {
                        ggml_fp16_t stored;
                        std::memcpy(&stored, last_tail + i * sizeof(stored), sizeof(stored));
                        require(std::fabs(ggml_fp16_to_fp32(stored) - rotated[i]) <=
                                1e-3f + 1e-3f * std::fabs(rotated[i]),
                                "exact tail row does not match the rotated q4 reference");
                    }
                }

                // Complete group 1 (two full groups): the K record must dequantize
                // to the rotated values in [dim][token] orientation, and must
                // be closer to that orientation than to the transposed one.
                if (prompt_len >= 256) {
                    // Standalone K record layout, independent of the producer:
                    // packed payload then scale axis, zero point, other axis.
                    llama_kvarn_tile_layout layout = {};
                    layout.k_payload_off = 0;
                    layout.k_payload_bytes = llama_kvarn_packed_bytes(128 * 128, 4);
                    layout.k_s_col_off = layout.k_payload_bytes;
                    layout.k_zp_off = layout.k_s_col_off + 128 * sizeof(uint16_t);
                    layout.k_s_row_off = layout.k_zp_off + 128 * sizeof(uint16_t);
                    layout.tile_bytes = layout.k_s_row_off + 128 * sizeof(uint16_t);
                    const size_t record_bytes = layout.tile_bytes;
                    require(parsed.layers[0].k_records.size() >=
                                2u * record_bytes * n_head_sliced, "K record extent");
                    const uint32_t n_valid_group1 = std::min<uint32_t>(128, prompt_len - 128);
                    std::vector<float> tile(128 * 128);
                    double sum_sq = 0, sum_err = 0, sum_err_t = 0;
                    for (uint32_t hs = 0; hs < n_head_sliced; ++hs) {
                        llama_kvarn_dequantize_k_tile(
                                parsed.layers[0].k_records.data() +
                                    (size_t(1) * n_head_sliced + hs) * record_bytes,
                                4, layout, tile.data());
                        for (uint32_t token = 0; token < n_valid_group1; token += 13) {
                            std::memcpy(raw.data(),
                                    source_parsed.bytes.data() + sl.k_data + size_t(128 + token) * sl.k_row_size,
                                    raw.size());
                            q4_row(raw.data(), row.data(), n_embd_k);
                            reference_rotate(row.data(), n_head_kv, head_dim_k, rotated);
                            for (uint32_t d = 0; d < 128; ++d) {
                                const float value = tile[size_t(d) * 128 + token];
                                const float reference = rotated[size_t(hs) * 128 + d];
                                sum_sq += reference * reference;
                                sum_err += (value - reference) * (value - reference);
                                const float transposed = tile[size_t(token) * 128 + d];
                                sum_err_t += (transposed - reference) * (transposed - reference);
                            }
                        }
                    }
                    const double rel = std::sqrt(sum_err / std::max(sum_sq, 1e-12));
                    const double rel_t = std::sqrt(sum_err_t / std::max(sum_sq, 1e-12));
                    worst_record_rel = std::max(worst_record_rel, rel);
                    std::cout << "  case " << prompt_len << " K record rel=" << rel
                              << " rel_transposed=" << rel_t << "\n";
                    require(rel < rel_t * 0.5, "K record orientation is transposed");
                    require(rel < 0.35, "K record dequantization error is too large");
                }
            }

            // Continue both contexts and compare.
            const llama_tokens suffix(tokens.begin() + prompt_len, tokens.begin() + prompt_len + 4);
            decode(source.get(), suffix, prompt_len);
            decode(dest.get(), suffix, prompt_len);
            float max_diff_source = 0;
            const double kl_source = logits_kl(source.get(), dest.get(), n_vocab, &max_diff_source);
            worst_kl_source = std::max(worst_kl_source, kl_source);

            // Native KVarN continuation of the same prefix.
            llama_context_ptr native_ctx(llama_init_from_model(model.get(), kvarn_params()));
            require(bool(native_ctx), "native KVarN context failed");
            decode(native_ctx.get(), evaluated);
            decode(native_ctx.get(), suffix, prompt_len);
            float max_diff_native = 0;
            const double kl_native = logits_kl(native_ctx.get(), dest.get(), n_vocab, &max_diff_native);
            worst_kl_native = std::max(worst_kl_native, kl_native);
            worst_diff_native = std::max(worst_diff_native, max_diff_native);

            // Native roundtrip of the converted destination state must be
            // reusable: save it natively after the suffix, restore elsewhere
            // and continue identically from there.
            const llama_tokens continued = [&] {
                llama_tokens result = evaluated;
                result.insert(result.end(), suffix.begin(), suffix.end());
                return result;
            }();
            const std::string native_path = dir + "/kvarn-native-" + std::to_string(prompt_len) + ".bin";
            const size_t native_written = llama_state_seq_save_file(dest.get(), native_path.c_str(), 0,
                    continued.data(), continued.size());
            require(native_written > 0, "native KVarN save of converted state failed");
            llama_context_ptr reload(llama_init_from_model(model.get(), kvarn_params()));
            require(bool(reload), "native reload context failed");
            const auto native_bytes = read_file(native_path);
            llama_tokens reload_tokens(continued.size());
            size_t reload_count = 0;
            require(llama_state_seq_load_file_streaming(reload.get(), native_path.c_str(), 0,
                        reload_tokens.data(), reload_tokens.size(), &reload_count,
                        native_bytes.size(), checksum(native_bytes)) == native_bytes.size() &&
                    reload_count == continued.size() && reload_tokens == continued,
                    "native KVarN roundtrip of converted state failed");
            const llama_token probe_token = llama_token(3 + (prompt_len * 7) % (n_vocab - 3));
            decode(dest.get(), {probe_token}, prompt_len + int(suffix.size()));
            decode(reload.get(), {probe_token}, prompt_len + int(suffix.size()));
            float reload_diff = 0;
            const double reload_kl = logits_kl(dest.get(), reload.get(), n_vocab, &reload_diff);
            require(reload_kl < 1e-5 && reload_diff < 1e-3, "native roundtrip continuation differs");

            std::cout << "case " << prompt_len << ": kl(q4|cvt)=" << kl_source
                      << " maxdiff=" << max_diff_source
                      << " kl(native|cvt)=" << kl_native << " maxdiff=" << max_diff_native << '\n';
            require(std::isfinite(kl_source) && std::isfinite(kl_native), "non-finite conversion comparison");
            require(kl_source < 0.5 && kl_native < 0.5, "converted continuation diverged");

            // Failure: corrupted payload and checksum mismatch are rejected
            // without touching the (now nonempty) live destination.
            const std::string bad_path = dir + "/bad-" + std::to_string(prompt_len) + ".bin";
            auto bad = kvarn_bytes;
            bad.back() ^= 1;
            write_file(bad_path, bad);
            llama_tokens probe(evaluated.size());
            size_t probe_count = 0;
            const llama_pos before_pos = llama_memory_seq_pos_max(llama_get_memory(dest.get()), 0);
            require(llama_state_seq_load_file_streaming(dest.get(), bad_path.c_str(), 0,
                        probe.data(), probe.size(), &probe_count, bad.size(), checksum(kvarn_bytes)) == 0,
                    "corrupted converted state accepted");
            require(llama_memory_seq_pos_max(llama_get_memory(dest.get()), 0) == before_pos,
                    "corrupted restore damaged the live destination");
        }
        std::cout << "worst kl vs q4 source=" << worst_kl_source
                  << " vs native KVarN=" << worst_kl_native
                  << " worst native maxdiff=" << worst_diff_native
                  << " worst record rel=" << worst_record_rel << '\n';

        // Unsupported source: a KVarN file is not a q4 conversion source.
        {
            auto dest_params = kvarn_params();
            llama_context_ptr dest(llama_init_from_model(model.get(), dest_params));
            require(bool(dest), "format rejection context failed");
            const std::string kvarn_src = dir + "/kvarn4-128.bin";
            const std::string out_path = dir + "/reject.bin";
            llama_tokens tokens_out(8);
            size_t count = 0;
            const auto bytes = read_file(kvarn_src);
            require(llama_state_seq_convert_file(dest.get(), kvarn_src.c_str(), 0, bytes.size(), 0,
                        out_path.c_str(), tokens_out.data(), tokens_out.size(), &count) == 0 && count == 0,
                    "non-q4 source was accepted");
            require(!std::filesystem::exists(out_path), "rejected conversion published an output file");
        }

        // Truncated source and wrong context family are rejected.
        {
            llama_context_ptr dest(llama_init_from_model(model.get(), kvarn_params()));
            require(bool(dest), "truncation context failed");
            const auto bytes = read_file(dir + "/q4-300.bin");
            const std::string trunc_path = dir + "/trunc.bin";
            write_file(trunc_path, std::vector<uint8_t>(bytes.begin(), bytes.begin() + bytes.size() / 2));
            llama_tokens tokens_out(300);
            size_t count = 0;
            const std::string trunc_out = dir + "/trunc-out.bin";
            require(llama_state_seq_convert_file(dest.get(), trunc_path.c_str(), 0,
                        bytes.size() / 2, 0, trunc_out.c_str(),
                        tokens_out.data(), tokens_out.size(), &count) == 0 && count == 0,
                    "truncated source was accepted");

            auto f16_params = base_params();
            f16_params.kvarn = llama_kvarn_default_params();
            llama_context_ptr f16_ctx(llama_init_from_model(model.get(), f16_params));
            require(bool(f16_ctx), "f16 context failed");
            count = 0;
            const std::string q4_300 = dir + "/q4-300.bin";
            const std::string f16_out = dir + "/f16-out.bin";
            require(llama_state_seq_convert_file(f16_ctx.get(), q4_300.c_str(), 0,
                        bytes.size(), 0, f16_out.c_str(),
                        tokens_out.data(), tokens_out.size(), &count) == 0 && count == 0,
                    "conversion into a standard KV context was accepted");
            require(!std::filesystem::exists(dir + "/f16-out.bin"), "standard-KV conversion published output");
        }

        std::cout << "PASS: boundaries, reference orientation, native roundtrip, failure rejection\n";
        std::cout << "Artifacts: " << dir << '\n';
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "FAIL(" << typeid(e).name() << "): " << std::string(e.what()) << '\n';
        return 1;
    }
}
