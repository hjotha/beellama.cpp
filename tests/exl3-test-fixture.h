#pragma once

#include "ggml.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace exl3_test_fixture {
namespace fs = std::filesystem;

// a safetensors file; the header is padded so that the data start at 2 mod 8, so every tensor starts at 2 mod 4 as in real EXL3 files
struct st_file {
    struct entry {
        std::string          name;
        std::string          dtype;
        std::vector<int64_t> shape;
        std::vector<uint8_t> data;
    };
    std::vector<entry> tensors;

    void add(const std::string & name, const std::string & dtype, const std::vector<int64_t> & shape, const std::vector<uint8_t> & data) {
        tensors.push_back({ name, dtype, shape, data });
    }

    void write(const fs::path & path, const std::string & metadata_json = "") const {
        std::string hdr = "{";
        bool first = true;
        if (!metadata_json.empty()) {
            hdr += "\"__metadata__\": " + metadata_json;
            first = false;
        }
        uint64_t offs = 0;
        for (size_t i = 0; i < tensors.size(); i++) {
            const entry & t = tensors[i];
            hdr += std::string(first ? "" : ", ") + "\"" + t.name + "\": {\"dtype\": \"" + t.dtype + "\", \"shape\": [";
            first = false;
            for (size_t j = 0; j < t.shape.size(); j++) {
                hdr += (j ? ", " : "") + std::to_string(t.shape[j]);
            }
            hdr += "], \"data_offsets\": [" + std::to_string(offs) + ", " + std::to_string(offs + t.data.size()) + "]}";
            offs += t.data.size();
        }
        hdr += "}";
        while ((8 + hdr.size()) % 8 != 2) {
            hdr += ' ';
        }
        std::ofstream f(path, std::ios::binary);
        for (int i = 0; i < 8; i++) {
            f.put((char) ((uint64_t) hdr.size() >> 8*i));
        }
        f << hdr;
        for (const entry & t : tensors) {
            f.write((const char *) t.data.data(), t.data.size());
        }
    }
};

static uint32_t rng_state = 1;

static float rnd(float lo, float hi) {
    rng_state = rng_state*1664525u + 1013904223u;
    return lo + (hi - lo)*(float) (rng_state >> 8)/(float) (1u << 24);
}

static std::vector<uint8_t> rnd_data(const std::string & dtype, int64_t n, float lo, float hi) {
    std::vector<uint8_t> out;
    for (int64_t i = 0; i < n; i++) {
        const float x = rnd(lo, hi);
        if (dtype == "F32") {
            out.insert(out.end(), (const uint8_t *) &x, (const uint8_t *) &x + 4);
        } else if (dtype == "F16") {
            const ggml_fp16_t h = ggml_fp32_to_fp16(x);
            out.insert(out.end(), (const uint8_t *) &h, (const uint8_t *) &h + 2);
        } else if (dtype == "BF16") {
            const ggml_bf16_t h = ggml_fp32_to_bf16(x);
            out.insert(out.end(), (const uint8_t *) &h, (const uint8_t *) &h + 2);
        } else {
            const uint16_t h = (uint16_t) (rnd(0, 65536));
            out.insert(out.end(), (const uint8_t *) &h, (const uint8_t *) &h + 2);
        }
    }
    return out;
}

static std::vector<uint8_t> i64_data(const std::vector<int64_t> & vals) {
    std::vector<uint8_t> out(vals.size() * 8);
    if (!vals.empty()) {
        memcpy(out.data(), vals.data(), out.size());
    }
    return out;
}

static void add_exl3_cb(st_file & f, const std::string & base, int64_t k, int64_t n, int64_t K, const std::string & cb) {
    f.add(base + ".trellis", "I16", { k/16, n/16, 16*K }, rnd_data("I16", k/16*n/16*16*K, 0, 0));
    f.add(base + ".suh", "F16", { k }, rnd_data("F16", k, -1.0f, 1.0f));
    f.add(base + ".svh", "F16", { n }, rnd_data("F16", n, -0.05f, 0.05f));
    if (cb == "mcg" || cb == "mul1") {
        const uint32_t marker = cb == "mcg" ? 0xCBAC1FEDu : 0x83DCD12Du;
        f.add(base + "." + cb, "I32", {}, std::vector<uint8_t>((const uint8_t *) &marker, (const uint8_t *) &marker + 4));
    }
}

static void add_exl3(st_file & f, const std::string & base, int64_t k, int64_t n, int64_t K) {
    add_exl3_cb(f, base, k, n, K, "mcg");
}

static void add_exl3_mul1(st_file & f, const std::string & base, int64_t k, int64_t n, int64_t K) {
    add_exl3_cb(f, base, k, n, K, "mul1");
}

// a qwen35 EXL3 checkpoint of 3 linear attention layers and 1 full attention layer, written in three files: two shards
// with an index, and lm_head.safetensors that the index does not list; the config declares an MTP layer the files do not
// have. With mtp the files also hold that MTP layer; with rows the token embedding is a row-ring table (K = 4, one group)
static void write_synthetic_tokenizer(const fs::path & dir) {
    // the byte-level alphabet of GPT-2, two merges, added tokens in tokenizer.json and one only in tokenizer_config.json
    std::string vocab;
    for (int b = 0, n = 0; b < 256; b++) {
        const bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
        const int  cp     = direct ? b : 256 + n++;
        std::string u;
        if (cp < 0x80) {
            u = std::string(1, (char) cp);
        } else {
            u = std::string(1, (char) (0xC0 | (cp >> 6))) + (char) (0x80 | (cp & 0x3F));
        }
        if (u == "\"" || u == "\\") {
            u = "\\" + u;
        }
        vocab += "\"" + u + "\": " + std::to_string(b) + ", ";
    }
    vocab += "\"\xc4\xa0t\": 256, \"he\": 257";
    std::ofstream(dir / "tokenizer.json") << R"JSON({"version": "1.0", "truncation": null, "padding": null,
 "added_tokens": [{"id": 300, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
  {"id": 301, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
  {"id": 302, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
  {"id": 303, "content": "<think>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false}],
 "normalizer": {"type": "NFC"},
 "pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false}, {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}]},
 "post_processor": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
 "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
 "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "", "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
  "vocab": {)JSON" << vocab << R"JSON(}, "merges": [[")JSON" << "\\u0120" << R"JSON(", "t"], ["h", "e"]]}})JSON";
    std::ofstream(dir / "tokenizer_config.json") << R"JSON({"add_bos_token": false, "add_prefix_space": false, "bos_token": null,
 "eos_token": "<|im_end|>", "pad_token": "<|endoftext|>", "chat_template": "{% for m in messages %}{{ m['content'] }}{% endfor %}",
 "added_tokens_decoder": {"300": {"content": "<|endoftext|>", "special": true}, "304": {"content": "<|audio_pad|>", "special": true}}})JSON";
}

static void write_synthetic(const fs::path & dir, std::map<std::string, st_file::entry> & src, int64_t & n_params,
        bool mtp = false, bool rows = false) {
    const int64_t n_embd = 256, n_ff = 256, n_vocab = 384, head_dim = 256, n_head = 2, n_head_kv = 1;
    const int64_t n_kh = 2, n_vh = 4, head_k = 128, head_v = 128;
    fs::create_directories(dir);

    st_file s1, s2, head;
    const std::string m = "model.language_model.";
    if (rows) {
        // rings of 1 + 16*4 words: an fp16 scale, then the stream; signs +/-1
        std::vector<uint8_t> table = rnd_data("I16", n_vocab*65, 0, 0);
        for (int64_t r = 0; r < n_vocab; r++) {
            const ggml_fp16_t sc = ggml_fp32_to_fp16(rnd(0.01f, 0.02f));
            memcpy(table.data() + 2*r*65, &sc, 2);
        }
        std::vector<uint8_t> signs;
        for (int i = 0; i < 256; i++) {
            const ggml_fp16_t h = ggml_fp32_to_fp16(rnd(0, 1) < 0.5f ? -1.0f : 1.0f);
            signs.insert(signs.end(), (const uint8_t *) &h, (const uint8_t *) &h + 2);
        }
        s1.add(m + "embed_tokens.trellis", "I16", { n_vocab, 65 }, table);
        s1.add(m + "embed_tokens.signs",   "F16", { 1, 256 }, signs);
    } else {
        s1.add(m + "embed_tokens.weight", "F16", { n_vocab, n_embd }, rnd_data("F16", n_vocab*n_embd, -0.05f, 0.05f));
    }
    s2.add(m + "norm.weight", "BF16", { n_embd }, rnd_data("BF16", n_embd, -0.1f, 0.1f));
    for (int il = 0; il < 4; il++) {
        st_file & f = il < 2 ? s1 : s2;
        const std::string p = m + "layers." + std::to_string(il) + ".";
        f.add(p + "input_layernorm.weight",          "BF16", { n_embd }, rnd_data("BF16", n_embd, -0.1f, 0.1f));
        f.add(p + "post_attention_layernorm.weight", "BF16", { n_embd }, rnd_data("BF16", n_embd, -0.1f, 0.1f));
        add_exl3(f, p + "mlp.gate_proj", n_embd, n_ff, 4);
        add_exl3(f, p + "mlp.up_proj",   n_embd, n_ff, 4);
        add_exl3(f, p + "mlp.down_proj", n_ff, n_embd, 4);
        if (il < 3) {
            const int64_t qk = 2*n_kh*head_k, v = n_vh*head_v;
            add_exl3(f, p + "linear_attn.in_proj_qkv", n_embd, qk + v, 4);
            add_exl3(f, p + "linear_attn.in_proj_z",   n_embd, v, 4);
            add_exl3(f, p + "linear_attn.out_proj",    v, n_embd, 4);
            f.add(p + "linear_attn.in_proj_a.weight", "F16",  { n_vh, n_embd }, rnd_data("F16", n_vh*n_embd, -0.05f, 0.05f));
            f.add(p + "linear_attn.in_proj_b.weight", "F16",  { n_vh, n_embd }, rnd_data("F16", n_vh*n_embd, -0.05f, 0.05f));
            f.add(p + "linear_attn.A_log",            "F32",  { n_vh }, rnd_data("F32", n_vh, -1.0f, 1.0f));
            f.add(p + "linear_attn.dt_bias",          "BF16", { n_vh }, rnd_data("BF16", n_vh, -0.5f, 0.5f));
            f.add(p + "linear_attn.conv1d.weight",    "BF16", { qk + v, 1, 4 }, rnd_data("BF16", (qk + v)*4, -0.3f, 0.3f));
            f.add(p + "linear_attn.norm.weight",      "F32",  { head_v }, rnd_data("F32", head_v, 0.9f, 1.1f));
        } else {
            add_exl3(f, p + "self_attn.q_proj", n_embd, n_head*head_dim*2, 4);
            add_exl3(f, p + "self_attn.k_proj", n_embd, n_head_kv*head_dim, 4);
            add_exl3(f, p + "self_attn.v_proj", n_embd, n_head_kv*head_dim, 4);
            add_exl3(f, p + "self_attn.o_proj", n_head*head_dim, n_embd, 4);
            f.add(p + "self_attn.q_norm.weight", "BF16", { head_dim }, rnd_data("BF16", head_dim, -0.1f, 0.1f));
            f.add(p + "self_attn.k_norm.weight", "BF16", { head_dim }, rnd_data("BF16", head_dim, -0.1f, 0.1f));
        }
    }
    if (mtp) {
        add_exl3(s2, "mtp.fc", 2*n_embd, n_embd, 4);
        s2.add("mtp.pre_fc_norm_embedding.weight", "BF16", { n_embd }, rnd_data("BF16", n_embd, -0.1f, 0.1f));
        s2.add("mtp.pre_fc_norm_hidden.weight",    "BF16", { n_embd }, rnd_data("BF16", n_embd, -0.1f, 0.1f));
        s2.add("mtp.norm.weight",                  "BF16", { n_embd }, rnd_data("BF16", n_embd, -0.1f, 0.1f));
        const std::string p = "mtp.layers.0.";
        s2.add(p + "input_layernorm.weight",          "BF16", { n_embd }, rnd_data("BF16", n_embd, -0.1f, 0.1f));
        s2.add(p + "post_attention_layernorm.weight", "BF16", { n_embd }, rnd_data("BF16", n_embd, -0.1f, 0.1f));
        add_exl3(s2, p + "self_attn.q_proj", n_embd, n_head*head_dim*2, 4);
        add_exl3(s2, p + "self_attn.k_proj", n_embd, n_head_kv*head_dim, 4);
        add_exl3(s2, p + "self_attn.v_proj", n_embd, n_head_kv*head_dim, 4);
        add_exl3(s2, p + "self_attn.o_proj", n_head*head_dim, n_embd, 4);
        s2.add(p + "self_attn.q_norm.weight", "BF16", { head_dim }, rnd_data("BF16", head_dim, -0.1f, 0.1f));
        s2.add(p + "self_attn.k_norm.weight", "BF16", { head_dim }, rnd_data("BF16", head_dim, -0.1f, 0.1f));
        add_exl3(s2, p + "mlp.gate_proj", n_embd, n_ff, 4);
        add_exl3(s2, p + "mlp.up_proj",   n_embd, n_ff, 4);
        add_exl3(s2, p + "mlp.down_proj", n_ff, n_embd, 4);
    }
    s2.add("model.visual.blocks.0.norm1.weight", "BF16", { 16 }, rnd_data("BF16", 16, -1.0f, 1.0f));
    s2.add("model.visual.patch_embed.proj.weight", "BF16", { 16, 48 }, rnd_data("BF16", 16*48, -1.0f, 1.0f));
    add_exl3(head, "lm_head", n_embd, n_vocab, 6);

    s1.write(dir / "model-00001-of-00002.safetensors");
    s2.write(dir / "model-00002-of-00002.safetensors");
    head.write(dir / "lm_head.safetensors");

    n_params = 0;
    std::string index = "{\"metadata\": {}, \"weight_map\": {";
    bool first = true;
    for (const st_file * f : { &s1, &s2, &head }) {
        for (const auto & t : f->tensors) {
            src[t.name] = t;
            if (f != &head) {
                index += std::string(first ? "" : ", ") + "\"" + t.name + "\": \"model-0000" + (f == &s1 ? "1" : "2") + "-of-00002.safetensors\"";
                first = false;
            }
            auto ends_with = [&](const std::string & suffix) {
                return t.name.size() >= suffix.size() && t.name.compare(t.name.size() - suffix.size(), suffix.size(), suffix) == 0;
            };
            if (t.name.find("visual.") != std::string::npos || ends_with(".suh") || ends_with(".svh") || ends_with(".mcg") ||
                    t.name.rfind("mtp.", 0) == 0) {
                continue;
            }
            int64_t n = 1;
            for (const int64_t x : t.shape) {
                n *= x;
            }
            n_params += ends_with(".trellis") ? t.shape[0]*16*t.shape[1]*16 : n;
        }
    }
    std::ofstream(dir / "model.safetensors.index.json") << index << "}}";

    std::ofstream(dir / "config.json") << R"JSON({
 "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "tie_word_embeddings": true,
 "text_config": {"attn_output_gate": true, "full_attention_interval": 4, "head_dim": 256, "hidden_size": 256, "intermediate_size": 256,
  "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"], "linear_conv_kernel_dim": 4,
  "linear_key_head_dim": 128, "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_value_head_dim": 128,
  "max_position_embeddings": 4096, "mtp_num_hidden_layers": 1, "num_attention_heads": 2, "num_hidden_layers": 4, "num_key_value_heads": 1,
  "rms_norm_eps": 1e-06, "tie_word_embeddings": true, "vocab_size": 384, "eos_token_id": 300,
  "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25}},
 "vision_config": {"depth": 1},
 "quantization_config": {"quant_method": "exl3", "version": "0.0.22", "bits": 4.0, "head_bits": 6, "codebook": "mcg"}})JSON";
    // tensor_storage lists one module only: the importer must not take it as the inventory
    std::ofstream(dir / "quantization_config.json") << R"JSON({"quant_method": "exl3", "tensor_storage": {"model.language_model.layers.0.mlp.gate_proj": {}}})JSON";
    write_synthetic_tokenizer(dir);
}

static void write_synthetic_qwen4exp(const fs::path & dir) {
    fs::create_directories(dir);
    st_file s1, s2, ngram;
    const std::string m = "model.language_model.";
    s1.add(m + "embed_tokens.weight", "BF16", { 384, 640 }, rnd_data("BF16", 384*640, -0.05f, 0.05f));

    auto add_hc = [](st_file & f, const std::string & pfx, bool with_inject) {
        f.add(pfx + ".hc_norm.weight",               "F16", { 2560 },      rnd_data("F16", 2560,      -0.05f, 0.05f));
        f.add(pfx + ".input_mix_weight_down.weight", "F16", { 128, 2560 }, rnd_data("F16", 128*2560,  -0.05f, 0.05f));
        f.add(pfx + ".input_mix_weight_up.weight",   "F16", { 2560, 128 }, rnd_data("F16", 2560*128,  -0.05f, 0.05f));
        if (with_inject) {
            f.add(pfx + ".block_inject_weight.weight", "F16", { 4, 2560 }, rnd_data("F16", 4*2560,    -0.05f, 0.05f));
        }
    };
    auto add_moe = [](st_file & f, const std::string & pfx) {
        f.add(pfx + ".mlp.gate.weight",               "F16", { 4, 640 }, rnd_data("F16", 4*640, -0.05f, 0.05f));
        f.add(pfx + ".mlp.shared_expert_gate.weight", "F16", { 1, 640 }, rnd_data("F16", 640,   -0.05f, 0.05f));
        add_exl3_mul1(f, pfx + ".mlp.shared_expert.gate_proj", 640, 128, 4);
        add_exl3_mul1(f, pfx + ".mlp.shared_expert.up_proj",   640, 128, 4);
        add_exl3_mul1(f, pfx + ".mlp.shared_expert.down_proj", 128, 640, 4);
        for (int e = 0; e < 4; e++) {
            const std::string ep = pfx + ".mlp.experts." + std::to_string(e);
            add_exl3_mul1(f, ep + ".gate_proj", 640, 128, 4);
            add_exl3_mul1(f, ep + ".up_proj",   640, 128, 4);
            add_exl3_mul1(f, ep + ".down_proj", 128, 640, 4);
        }
    };
    auto add_qsa_layer = [&](st_file & f, const std::string & pfx) {
        add_hc(f, pfx + ".attn_hyper_connection", true);
        add_exl3_mul1(f, pfx + ".self_attn.q_proj", 640, 1024, 4);
        add_exl3_mul1(f, pfx + ".self_attn.k_proj", 640, 256,  4);
        add_exl3_mul1(f, pfx + ".self_attn.v_proj", 640, 256,  4);
        add_exl3_mul1(f, pfx + ".self_attn.o_proj", 512, 640,  4);
        f.add(pfx + ".self_attn.q_norm.weight", "BF16", { 256 }, rnd_data("BF16", 256, -0.05f, 0.05f));
        f.add(pfx + ".self_attn.k_norm.weight", "BF16", { 256 }, rnd_data("BF16", 256, -0.05f, 0.05f));
        add_exl3_mul1(f, pfx + ".self_attn.indexer.index_qk_proj", 640, 256, 4);
        f.add(pfx + ".self_attn.indexer.q_layernorm.weight", "BF16", { 128 }, rnd_data("BF16", 128, -0.05f, 0.05f));
        f.add(pfx + ".self_attn.indexer.k_layernorm.weight", "BF16", { 128 }, rnd_data("BF16", 128, -0.05f, 0.05f));
        add_hc(f, pfx + ".mlp_hyper_connection", true);
        add_moe(f, pfx);
    };

    // layer 0: linear_attention + PLE
    {
        const std::string p = m + "layers.0";
        add_hc(s1, p + ".attn_hyper_connection", true);
        add_exl3_mul1(s1, p + ".linear_attn.in_proj_qkv", 640, 1024, 4);
        add_exl3_mul1(s1, p + ".linear_attn.in_proj_z",   640, 512,  4);
        add_exl3_mul1(s1, p + ".linear_attn.out_proj",    512, 640,  4);
        s1.add(p + ".linear_attn.in_proj_a.weight", "F16",  { 4, 640 },     rnd_data("F16",  4*640,     -0.05f, 0.05f));
        s1.add(p + ".linear_attn.in_proj_b.weight", "F16",  { 4, 640 },     rnd_data("F16",  4*640,     -0.05f, 0.05f));
        s1.add(p + ".linear_attn.A_log",            "BF16", { 4 },          rnd_data("BF16", 4,         -0.5f,  0.5f));
        s1.add(p + ".linear_attn.dt_bias",          "BF16", { 4 },          rnd_data("BF16", 4,         -0.5f,  0.5f));
        s1.add(p + ".linear_attn.conv1d.weight",    "BF16", { 1024, 1, 4 }, rnd_data("BF16", 1024*4,    -0.05f, 0.05f));
        s1.add(p + ".linear_attn.norm.weight",      "BF16", { 128 },        rnd_data("BF16", 128,        0.5f,  1.5f));
        s1.add(p + ".ple.norm_conv.weight",         "F16",  { 2560 },       rnd_data("F16",  2560,      -0.05f, 0.05f));
        s1.add(p + ".ple.norm_key.weight",          "F16",  { 2560 },       rnd_data("F16",  2560,      -0.05f, 0.05f));
        s1.add(p + ".ple.norm_query.weight",        "F16",  { 2560 },       rnd_data("F16",  2560,      -0.05f, 0.05f));
        s1.add(p + ".ple.conv1d.weight",            "F16",  { 2560, 1, 4 }, rnd_data("F16",  2560*4,    -0.05f, 0.05f));
        s1.add(p + ".ple.key_proj.weight",          "F16",  { 2560, 640 },  rnd_data("F16",  2560*640,  -0.05f, 0.05f));
        s1.add(p + ".ple.value_proj.weight",        "F16",  { 640, 640 },   rnd_data("F16",  640*640,   -0.05f, 0.05f));
        add_hc(s1, p + ".mlp_hyper_connection", true);
        add_moe(s1, p);
    }
    // layer 1: full_attention (QSA)
    add_qsa_layer(s2, m + "layers.1");

    // output mixer + lm_head
    add_hc(s2, m + "hyper_connection_mixer", false);
    add_exl3_mul1(s2, "lm_head", 640, 384, 4);

    // MTP head
    s2.add("mtp.pre_fc_norm_embedding.weight", "BF16", { 640 },  rnd_data("BF16", 640,  -0.05f, 0.05f));
    s2.add("mtp.pre_fc_norm_hidden.weight",    "F16",  { 2560 }, rnd_data("F16",  2560, -0.05f, 0.05f));
    add_exl3_mul1(s2, "mtp.fc_embedding", 640, 640, 4);
    add_exl3_mul1(s2, "mtp.fc_hidden",    640, 640, 4);
    add_hc(s2, "mtp.hyper_connection_mixer", false);
    add_qsa_layer(s2, "mtp.layers.0");

    // unlisted PLE ngram_embedding sidecar
    const std::string npfx = m + "layers.0.ple.ple_embedding.ngram_embedding";
    ngram.add(npfx + ".head_bias",         "F16", { 4, 160 }, rnd_data("F16", 4*160, -0.05f, 0.05f));
    ngram.add(npfx + ".head_offsets",      "I64", { 4 },      i64_data({ 0, 4, 8, 12 }));
    ngram.add(npfx + ".head_vocab_sizes",  "I64", { 4 },      i64_data({ 4, 4, 4, 4 }));
    ngram.add(npfx + ".layer_multipliers", "I64", { 3 },      i64_data({ 1000000000003LL, 2000000000033LL, 3000000000041LL }));
    {
        std::vector<uint8_t> tr = rnd_data("I16", 16*21, 0, 0);
        for (int r = 0; r < 16; r++) {
            // word 0 of each n-gram ring row is its FP16 scale; random uint16 bits can hit NaN/Inf
            const ggml_fp16_t sc = ggml_fp32_to_fp16(rnd(-0.05f, 0.05f));
            memcpy(tr.data() + (size_t) r * 21 * 2, &sc, sizeof(sc));
        }
        ngram.add(npfx + ".trellis", "I16", { 16, 21 }, std::move(tr));
    }

    s1.write(dir / "model-00001-of-00002.safetensors");
    s2.write(dir / "model-00002-of-00002.safetensors");
    ngram.write(dir / "ngram_embedding.safetensors",
        R"JSON({"format": "exl3_ngram_trellis", "version": "1", "K": "2", "codebook": "mul1", "row_dim": "160", "rows": "16"})JSON");

    // calibration trace and vision sidecars must be skipped
    st_file sidecar;
    sidecar.add("cal_trace.dummy", "F16", { 4 }, rnd_data("F16", 4, 0.0f, 1.0f));
    sidecar.write(dir / "cal_trace.safetensors");
    sidecar.write(dir / "vision_k6.safetensors");

    std::string index = "{\"metadata\": {}, \"weight_map\": {";
    bool first = true;
    for (const st_file * f : { &s1, &s2 }) {
        for (const auto & t : f->tensors) {
            index += std::string(first ? "" : ", ") + "\"" + t.name + "\": \"model-0000" + (f == &s1 ? "1" : "2") + "-of-00002.safetensors\"";
            first = false;
        }
    }
    std::ofstream(dir / "model.safetensors.index.json") << index << "}}";

    std::ofstream(dir / "config.json") << R"JSON({
 "architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp", "tie_word_embeddings": false, "image_token_id": 304,
 "text_config": {"attn_output_gate": true, "full_attention_interval": 2, "hc_count": 4, "hc_lowrank": 128,
  "head_dim": 256, "heads_per_ngram": 2, "hidden_size": 640, "indexer_budget": 32, "indexer_compress_ratio": 4,
  "indexer_head_dim": 128, "indexer_kv_heads": 1, "indexer_n_heads": 1,
  "layer_types": ["linear_attention", "full_attention"], "linear_conv_kernel_dim": 4,
  "linear_key_head_dim": 128, "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_value_head_dim": 128,
  "max_position_embeddings": 4096, "moe_intermediate_size": 128, "mtp_num_hidden_layers": 1, "ngram_size": 3,
  "num_attention_heads": 2, "num_experts": 4, "num_experts_per_tok": 2, "num_hidden_layers": 2, "num_key_value_heads": 1,
  "partial_rotary_factor": 0.25, "ple_conv_kernel_size": 4, "ple_embed_dim": 640, "ple_layer_ids": [1],
  "rms_norm_eps": 1e-06, "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
  "shared_expert_intermediate_size": 128, "tie_word_embeddings": false, "vocab_size": 384, "eos_token_id": 300},
 "vision_config": {"depth": 1},
 "quantization_config": {"quant_method": "exl3", "version": "1.4.4", "bits": 4.05, "head_bits": 6, "codebook": "mul1"}})JSON";

    write_synthetic_tokenizer(dir);
}

} // namespace exl3_test_fixture
