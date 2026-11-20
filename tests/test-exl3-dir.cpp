#include "exl3-test-fixture.h"
// EXL3 directory importer, common/exl3-dir.cpp
//
// test-exl3-dir                            a small synthetic qwen35 EXL3 directory is written, imported and run
// test-exl3-dir --compare DIR CONT.gguf    the import of DIR against the converter's container of it: tensors, bytes, keys
// test-exl3-dir --dump DIR OUT.gguf        write the import as a GGUF file
//
// --compare hashes both sides with the same code: sha256 over the tensors sorted by name (name, type, ne, sha256 of the
// data), and sha256 over the keys sorted by name except the general.* provenance keys, which are listed apart

#include "exl3-dir.h"

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

extern "C" {
#include "hash/sha256/sha256.h"
}

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <vector>

using namespace exl3_test_fixture;
namespace fs = std::filesystem;

static std::string to_hex(const unsigned char * d, size_t n) {
    static const char digits[] = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; i++) {
        s += digits[d[i] >> 4];
        s += digits[d[i] & 15];
    }
    return s;
}

struct sha256 {
    sha256_t s;
    sha256() { sha256_init(&s); }
    void add(const void * p, size_t n) { sha256_update(&s, (const unsigned char *) p, n); }
    template <typename T> void add_val(T v) { add(&v, sizeof(v)); }
    void add_str(const std::string & v) { add_val((uint64_t) v.size()); add(v.data(), v.size()); }
    std::string hex() {
        unsigned char d[SHA256_DIGEST_SIZE];
        sha256_final(&s, d);
        return to_hex(d, sizeof(d));
    }
};

static size_t gguf_elem_size(gguf_type t) {
    switch (t) {
        case GGUF_TYPE_UINT8:  case GGUF_TYPE_INT8:  case GGUF_TYPE_BOOL:    return 1;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16:                         return 2;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return 8;
        default:                                                             return 0;
    }
}

// the value of a key: type, then the value; strings with their length
static std::string kv_value(const gguf_context * g, int64_t i) {
    sha256 h;
    const gguf_type t = gguf_get_kv_type(g, i);
    h.add_val((int32_t) t);
    if (t == GGUF_TYPE_STRING) {
        h.add_str(gguf_get_val_str(g, i));
    } else if (t == GGUF_TYPE_ARRAY) {
        const gguf_type at = gguf_get_arr_type(g, i);
        const size_t    n  = gguf_get_arr_n(g, i);
        h.add_val((int32_t) at);
        h.add_val((uint64_t) n);
        if (at == GGUF_TYPE_STRING) {
            for (size_t j = 0; j < n; j++) {
                h.add_str(gguf_get_arr_str(g, i, j));
            }
        } else {
            h.add(gguf_get_arr_data(g, i), n*gguf_elem_size(at));
        }
    } else {
        h.add(gguf_get_val_data(g, i), gguf_elem_size(t));
    }
    return h.hex();
}

static bool kv_is_provenance(const std::string & key) {
    return (key.rfind("general.", 0) == 0 && key != "general.architecture" && key != "general.type" &&
            key != "general.file_type" && key != "general.quantization_version" && key != "general.alignment") ||
           key == "llama.tensor_set_complete";
}

struct structure {
    std::map<std::string, std::string> kv;         // key -> hash of the value
    std::map<std::string, std::string> provenance; // key -> value as text
    struct tensor {
        ggml_type   type;
        int64_t     ne[4];
        std::string sha;
    };
    std::map<std::string, tensor> tensors;
    std::vector<std::string>      order;
    std::string                   kv_hash;
    std::string                   tensor_hash;
};

static structure read_structure(const gguf_context * g, const std::function<void(int64_t, std::vector<uint8_t> &)> & data) {
    structure s;
    for (int64_t i = 0; i < gguf_get_n_kv(g); i++) {
        const std::string key = gguf_get_key(g, i);
        if (kv_is_provenance(key)) {
            s.provenance[key] = gguf_get_kv_type(g, i) == GGUF_TYPE_STRING ? gguf_get_val_str(g, i) : "(" + std::string(gguf_type_name(gguf_get_kv_type(g, i))) + ")";
        } else {
            s.kv[key] = kv_value(g, i);
        }
    }
    std::vector<uint8_t> buf;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
        structure::tensor t;
        t.type = gguf_get_tensor_type(g, i);
        memcpy(t.ne, gguf_get_tensor_ne(g, i), sizeof(t.ne));
        buf.resize(gguf_get_tensor_size(g, i));
        data(i, buf);
        unsigned char d[SHA256_DIGEST_SIZE];
        sha256_hash(d, buf.data(), buf.size());
        t.sha = to_hex(d, sizeof(d));
        s.tensors[gguf_get_tensor_name(g, i)] = t;
        s.order.push_back(gguf_get_tensor_name(g, i));
    }
    sha256 hk;
    for (const auto & [k, v] : s.kv) {
        hk.add_str(k);
        hk.add_str(v);
    }
    s.kv_hash = hk.hex();
    sha256 ht;
    for (const auto & [name, t] : s.tensors) {
        ht.add_str(name);
        ht.add_val((int32_t) t.type);
        for (int j = 0; j < 4; j++) {
            ht.add_val(t.ne[j]);
        }
        ht.add_str(t.sha);
    }
    s.tensor_hash = ht.hex();
    return s;
}

static int compare(const std::string & dir, const std::string & container) {
    ggml_context * ctx = nullptr;
    gguf_init_params params = { true, &ctx };
    gguf_context * gc = gguf_init_from_file(container.c_str(), params);
    if (!gc) {
        fprintf(stderr, "cannot read %s\n", container.c_str());
        return 1;
    }
    std::ifstream f(container, std::ios::binary);
    const structure a = read_structure(gc, [&](int64_t i, std::vector<uint8_t> & buf) {
        f.seekg((std::streamoff) (gguf_get_data_offset(gc) + gguf_get_tensor_offset(gc, i)));
        if (!f.read((char *) buf.data(), (std::streamsize) buf.size())) {
            throw std::runtime_error("cannot read the container");
        }
    });

    const common_exl3_dir_ptr d = common_exl3_dir_open(dir);
    const gguf_context * gd = common_exl3_dir_gguf(d.get());
    const structure b = read_structure(gd, [&](int64_t i, std::vector<uint8_t> & buf) {
        common_exl3_dir_read(d.get(), gguf_get_tensor_name(gd, i), buf.data(), buf.size());
    });

    int n_diff = 0;
    for (const auto & [k, v] : a.kv) {
        if (!b.kv.count(k)) {
            printf("key only in the container: %s\n", k.c_str());
            n_diff++;
        } else if (b.kv.at(k) != v) {
            printf("key differs: %s\n", k.c_str());
            n_diff++;
        }
    }
    for (const auto & [k, v] : b.kv) {
        if (!a.kv.count(k)) {
            printf("key only in the import: %s\n", k.c_str());
            n_diff++;
        }
    }
    for (const auto & [name, t] : a.tensors) {
        if (!b.tensors.count(name)) {
            printf("tensor only in the container: %s\n", name.c_str());
            n_diff++;
            continue;
        }
        const auto & u = b.tensors.at(name);
        if (t.type != u.type || memcmp(t.ne, u.ne, sizeof(t.ne)) != 0) {
            printf("tensor type or shape differs: %s\n", name.c_str());
            n_diff++;
        } else if (t.sha != u.sha) {
            printf("tensor data differ: %s\n", name.c_str());
            n_diff++;
        }
    }
    for (const auto & [name, t] : b.tensors) {
        if (!a.tensors.count(name)) {
            printf("tensor only in the import: %s\n", name.c_str());
            n_diff++;
        }
    }
    for (const auto & [k, v] : a.provenance) {
        printf("provenance, container: %s = %s\n", k.c_str(), v.c_str());
    }
    for (const auto & [k, v] : b.provenance) {
        printf("provenance, import:    %s = %s\n", k.c_str(), v.c_str());
    }
    printf("container: %zu tensors sha256 %s, %zu keys sha256 %s\n", a.tensors.size(), a.tensor_hash.c_str(), a.kv.size(), a.kv_hash.c_str());
    printf("import:    %zu tensors sha256 %s, %zu keys sha256 %s\n", b.tensors.size(), b.tensor_hash.c_str(), b.kv.size(), b.kv_hash.c_str());
    printf("tensor order equal: %s\n", a.order == b.order ? "yes" : "no");
    printf("%s (%d differences)\n", n_diff == 0 ? "IDENTICAL" : "DIFFERENT", n_diff);

    gguf_free(gc);
    ggml_free(ctx);
    return n_diff == 0 ? 0 : 1;
}

static int dump(const std::string & dir, const std::string & out) {
    const common_exl3_dir_ptr d = common_exl3_dir_open(dir);
    const gguf_context * g = common_exl3_dir_gguf(d.get());
    std::vector<uint8_t> meta(gguf_get_meta_size(g));
    gguf_get_meta_data(g, meta.data());
    FILE * f = fopen(out.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", out.c_str());
        return 1;
    }
    const size_t align      = gguf_get_alignment(g);
    const size_t data_start = (meta.size() + align - 1)/align*align;
    bool ok = fwrite(meta.data(), 1, meta.size(), f) == meta.size();
    size_t pos = meta.size();
    std::vector<uint8_t> buf;
    for (int64_t i = 0; ok && i < gguf_get_n_tensors(g); i++) {
        const size_t offs = data_start + gguf_get_tensor_offset(g, i);
        const std::vector<uint8_t> pad(offs - pos, 0);
        buf.resize(gguf_get_tensor_size(g, i));
        common_exl3_dir_read(d.get(), gguf_get_tensor_name(g, i), buf.data(), buf.size());
        ok = fwrite(pad.data(), 1, pad.size(), f) == pad.size() && fwrite(buf.data(), 1, buf.size(), f) == buf.size();
        pos = offs + buf.size();
    }
    ok = fclose(f) == 0 && ok;
    printf("%s: %" PRId64 " tensors, %zu bytes\n", out.c_str(), gguf_get_n_tensors(g), pos);
    return ok ? 0 : 1;
}

static int n_fail = 0;

static void check(bool ok, const char * what) {
    printf("  %s: %s\n", ok ? "ok  " : "FAIL", what);
    n_fail += !ok;
}

// [K, V, unit] -> [V, K, unit]: the old index of new index p of a range of n_heads*unit elements
static int64_t v_head_src(int64_t p, int64_t K, int64_t V, int64_t unit) {
    const int64_t head = p / unit;
    return ((head % K)*V + head / K)*unit + p % unit;
}

static std::vector<uint8_t> import_data(const common_exl3_dir * d, const std::string & name) {
    const gguf_context * g = common_exl3_dir_gguf(d);
    const int64_t id = gguf_find_tensor(g, name.c_str());
    if (id < 0) {
        return {};
    }
    std::vector<uint8_t> out(gguf_get_tensor_size(g, id));
    common_exl3_dir_read(d, name.c_str(), out.data(), out.size());
    return out;
}

static float f32_of(const std::string & dtype, const uint8_t * p) {
    uint16_t h;
    memcpy(&h, p, 2);
    if (dtype == "BF16") {
        const uint32_t u = (uint32_t) h << 16;
        float x;
        memcpy(&x, &u, 4);
        return x;
    }
    if (dtype == "F16") {
        return ggml_fp16_to_fp32(h);
    }
    float x;
    memcpy(&x, p, 4);
    return x;
}

static int synthetic() {
    const fs::path dir = fs::current_path() / ("test-exl3-dir-" + std::to_string((unsigned) time(nullptr)));
    std::map<std::string, st_file::entry> src;
    int64_t n_params = 0;
    write_synthetic(dir / "model", src, n_params);
    const std::string m = "model.language_model.";

    printf("import of a synthetic qwen35 EXL3 directory (%s)\n", dir.string().c_str());
    const common_exl3_dir_ptr d = common_exl3_dir_open((dir / "model").string());
    const gguf_context * g = common_exl3_dir_gguf(d.get());
    auto u32 = [&](const char * key) { const int64_t id = gguf_find_key(g, key); return id < 0 ? UINT32_MAX : gguf_get_val_u32(g, id); };
    auto str = [&](const char * key) { const int64_t id = gguf_find_key(g, key); return id < 0 ? std::string() : std::string(gguf_get_val_str(g, id)); };

    check(gguf_get_n_tensors(g) == 26*3 + 30, "26 EXL3 linears and 30 other tensors, vision tower not loaded");
    check(gguf_find_tensor(g, "output.exl3_trellis") >= 0, "lm_head found in a file the index does not list, not replaced by the tied embedding");
    check(u32("qwen35.block_count") == 4 && gguf_find_key(g, "qwen35.nextn_predict_layers") < 0, "no MTP layer although config.json declares one");
    check(str("tokenizer.ggml.pre") == "qwen35" && str("exl3.codebook") == "mcg", "pre-tokenizer and codebook");
    check(u32("tokenizer.ggml.eos_token_id") == 302 && u32("tokenizer.ggml.padding_token_id") == 300, "eos and pad from tokenizer_config.json");
    {
        const int64_t kt = gguf_find_key(g, "tokenizer.ggml.token_type");
        const int64_t ks = gguf_find_key(g, "tokenizer.ggml.tokens");
        const int64_t km = gguf_find_key(g, "tokenizer.ggml.merges");
        const int32_t * tt = (const int32_t *) gguf_get_arr_data(g, kt);
        check(gguf_get_arr_n(g, ks) == 384 && tt[65] == 1 && tt[300] == 3 && tt[303] == 4 && tt[304] == 3 && tt[305] == 5 &&
              std::string(gguf_get_arr_str(g, ks, 304)) == "<|audio_pad|>" && std::string(gguf_get_arr_str(g, ks, 260)) == "[PAD260]" &&
              std::string(gguf_get_arr_str(g, km, 0)) == "\xc4\xa0 t", "tokens, token types and merges");
    }

    bool ok = true;
    {
        const auto & s = src.at(m + "layers.0.linear_attn.A_log");
        const auto   y = import_data(d.get(), "blk.0.ssm_a");
        for (int64_t p = 0; p < 4; p++) {
            const float x = f32_of("F32", s.data.data() + 4*v_head_src(p, 2, 2, 1));
            const float e = (float) -std::exp((double) x);
            ok = ok && memcmp(&e, y.data() + 4*p, 4) == 0;
        }
        check(ok, "ssm_a = -exp(A_log) in the tiled V head order, rounded once from float64");
    }
    {
        const auto & s = src.at(m + "layers.1.input_layernorm.weight");
        const auto   y = import_data(d.get(), "blk.1.attn_norm.weight");
        ok = y.size() == 4*256;
        for (int64_t i = 0; ok && i < 256; i++) {
            const float e = f32_of("BF16", s.data.data() + 2*i) + 1.0f;
            ok = memcmp(&e, y.data() + 4*i, 4) == 0;
        }
        check(ok, "zero-centered norm + 1 in F32");
    }
    {
        const auto & s = src.at(m + "layers.2.linear_attn.conv1d.weight");
        const auto   y = import_data(d.get(), "blk.2.ssm_conv1d.weight");
        ok = y.size() == 4*1024*4;
        for (int64_t r = 0; ok && r < 1024; r++) {
            const int64_t q = r < 512 ? r : 512 + v_head_src(r - 512, 2, 2, 128);
            for (int64_t c = 0; ok && c < 4; c++) {
                const float e = f32_of("BF16", s.data.data() + 2*(q*4 + c));
                ok = memcmp(&e, y.data() + 4*(r*4 + c), 4) == 0;
            }
        }
        check(ok, "conv1d squeezed, V channels reordered, F32");
    }
    {
        const auto & tr = src.at(m + "layers.0.linear_attn.in_proj_qkv.trellis");
        const auto & sv = src.at(m + "layers.0.linear_attn.in_proj_qkv.svh");
        const auto   yt = import_data(d.get(), "blk.0.attn_qkv.exl3_trellis");
        const auto   ys = import_data(d.get(), "blk.0.attn_qkv.exl3_svh");
        const int64_t tn = 64, tile = 64*2;
        ok = yt.size() == tr.data.size() && ys.size() == sv.data.size();
        for (int64_t kt = 0; ok && kt < 16; kt++) {
            for (int64_t j = 0; ok && j < tn; j++) {
                const int64_t q = j < 32 ? j : 32 + v_head_src(j - 32, 2, 2, 8);
                ok = memcmp(yt.data() + (kt*tn + j)*tile, tr.data.data() + (kt*tn + q)*tile, tile) == 0;
            }
        }
        for (int64_t i = 0; ok && i < 1024; i++) {
            const int64_t q = i < 512 ? i : 512 + v_head_src(i - 512, 2, 2, 128);
            ok = memcmp(ys.data() + 2*i, sv.data.data() + 2*q, 2) == 0;
        }
        check(ok, "in_proj_qkv: V tiles and svh reordered in whole 128-blocks, q and k untouched");
    }
    {
        const auto & tr = src.at(m + "layers.1.linear_attn.out_proj.trellis");
        const auto & su = src.at(m + "layers.1.linear_attn.out_proj.suh");
        const auto   yt = import_data(d.get(), "blk.1.ssm_out.exl3_trellis");
        const auto   ys = import_data(d.get(), "blk.1.ssm_out.exl3_suh");
        const int64_t row = 16*64*2;
        ok = yt.size() == tr.data.size() && ys.size() == su.data.size();
        for (int64_t kt = 0; ok && kt < 32; kt++) {
            ok = memcmp(yt.data() + kt*row, tr.data.data() + v_head_src(kt, 2, 2, 8)*row, row) == 0;
        }
        for (int64_t i = 0; ok && i < 512; i++) {
            ok = memcmp(ys.data() + 2*i, su.data.data() + 2*v_head_src(i, 2, 2, 128), 2) == 0;
        }
        check(ok, "out_proj: tile rows and suh reordered on the input side");
    }
    {
        const auto & s = src.at(m + "layers.2.linear_attn.in_proj_a.weight");
        const auto   y = import_data(d.get(), "blk.2.ssm_alpha.weight");
        ok = y.size() == s.data.size();
        for (int64_t r = 0; ok && r < 4; r++) {
            ok = memcmp(y.data() + r*512, s.data.data() + v_head_src(r, 2, 2, 1)*512, 512) == 0;
        }
        check(ok, "in_proj_a rows reordered, F16 kept");
    }
    check(import_data(d.get(), "token_embd.weight") == src.at(m + "embed_tokens.weight").data, "token_embd copied from an offset that is 2 mod 4");

    {
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        llama_model * model = common_exl3_load_model((dir / "model").string(), mp);
        check(model != nullptr, "the model loads through llama_model_init_from_user");
        if (model) {
            check(llama_model_n_params(model) == (uint64_t) n_params, "parameters counted as the k x n weights of the linears");
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx     = 64;
            cp.n_batch   = 16;
            cp.n_ubatch  = 16;
            cp.n_threads = 4;
            llama_context * ctx = llama_init_from_model(model, cp);
            std::vector<llama_token> tokens = { 65, 66, 256, 257, 302 };
            ok = ctx && llama_decode(ctx, llama_batch_get_one(tokens.data(), (int32_t) tokens.size())) == 0;
            const float * logits = ok ? llama_get_logits_ith(ctx, -1) : nullptr;
            for (int i = 0; ok && i < 384; i++) {
                ok = std::isfinite(logits[i]);
            }
            check(ok, "a prompt runs and gives finite logits");
            llama_free(ctx);
            llama_model_free(model);
        }
    }

    // a stored MTP layer, named after the trunk layers as the converter names it, and a quantized token embedding
    {
        std::map<std::string, st_file::entry> src2;
        int64_t n_params2 = 0;
        write_synthetic(dir / "mtp", src2, n_params2, true, true);
        const common_exl3_dir_ptr d2 = common_exl3_dir_open((dir / "mtp").string());
        const gguf_context * g2 = common_exl3_dir_gguf(d2.get());
        auto u32_2 = [&](const char * key) { const int64_t id = gguf_find_key(g2, key); return id < 0 ? UINT32_MAX : gguf_get_val_u32(g2, id); };
        const int64_t kr = gguf_find_key(g2, "qwen35.attention.recurrent_layers");
        check(u32_2("qwen35.block_count") == 5 && u32_2("qwen35.nextn_predict_layers") == 1 && gguf_get_arr_n(g2, kr) == 5 &&
              !((const int8_t *) gguf_get_arr_data(g2, kr))[4], "MTP layer counted from the tensors: block 4, full attention");
        check(gguf_get_n_tensors(g2) == (26 + 8)*3 + 30 + 7 + 1, "8 MTP linears and 7 other MTP tensors imported, embedding as rows and signs");
        check(import_data(d2.get(), "blk.4.nextn.eh_proj.exl3_trellis") == src2.at("mtp.fc.trellis").data &&
              import_data(d2.get(), "blk.4.attn_q.exl3_trellis") == src2.at("mtp.layers.0.self_attn.q_proj.trellis").data,
              "mtp.fc -> blk.4.nextn.eh_proj, mtp.layers.0 -> blk.4");
        {
            const auto & s = src2.at("mtp.pre_fc_norm_embedding.weight");
            const auto   y = import_data(d2.get(), "blk.4.nextn.enorm.weight");
            ok = y.size() == 4*256 && gguf_find_tensor(g2, "blk.4.nextn.hnorm.weight") >= 0 &&
                 gguf_find_tensor(g2, "blk.4.nextn.shared_head_norm.weight") >= 0;
            for (int64_t i = 0; ok && i < 256; i++) {
                const float e = f32_of("BF16", s.data.data() + 2*i) + 1.0f;
                ok = memcmp(&e, y.data() + 4*i, 4) == 0;
            }
            check(ok, "MTP norms: enorm, hnorm, shared_head_norm, + 1 in F32");
        }
        check(import_data(d2.get(), "token_embd.exl3r_trellis") == src2.at(m + "embed_tokens.trellis").data &&
              import_data(d2.get(), "token_embd.exl3r_signs") == src2.at(m + "embed_tokens.signs").data &&
              gguf_find_tensor(g2, "token_embd.weight") < 0 && u32_2("exl3.rows.version") == 1,
              "quantized token embedding carried as token_embd.exl3r_trellis/_signs, exl3.rows.version 1");

        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.load_mtp     = true;
        llama_model * model = common_exl3_load_model((dir / "mtp").string(), mp);
        check(model != nullptr && llama_model_n_layer_nextn(model) == 1, "the model loads with its MTP layer");
        if (model) {
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx     = 64;
            cp.n_batch   = 16;
            cp.n_ubatch  = 16;
            cp.n_threads = 4;
            llama_context * ctx = llama_init_from_model(model, cp);
            std::vector<llama_token> tokens = { 65, 66, 256, 257, 302 };
            ok = ctx && llama_decode(ctx, llama_batch_get_one(tokens.data(), (int32_t) tokens.size())) == 0;
            const float * logits = ok ? llama_get_logits_ith(ctx, -1) : nullptr;
            for (int i = 0; ok && i < 384; i++) {
                ok = std::isfinite(logits[i]);
            }
            check(ok, "a prompt runs through the row-ring embedding and gives finite logits");
            llama_free(ctx);
            llama_model_free(model);
        }
    }

    fs::copy(dir / "model", dir / "dup", fs::copy_options::recursive);
    st_file extra;
    extra.add(m + "norm.weight", "BF16", { 256 }, rnd_data("BF16", 256, -0.1f, 0.1f));
    extra.write(dir / "dup" / "extra.safetensors");
    std::string err;
    try {
        common_exl3_dir_open((dir / "dup").string());
    } catch (const std::exception & e) {
        err = e.what();
    }
    check(err.find("is in both") != std::string::npos, "a tensor name in two files is refused");

    {
        write_synthetic_qwen4exp(dir / "qwen4exp");
        if (const char * keep = getenv("TEST_EXL3_KEEP_QWEN4EXP")) {
            fs::remove_all(keep);
            fs::copy(dir / "qwen4exp", keep, fs::copy_options::recursive);
        }
        const common_exl3_dir_ptr dq = common_exl3_dir_open((dir / "qwen4exp").string());
        const gguf_context * gq = common_exl3_dir_gguf(dq.get());
        auto q_str = [&](const char * k) { return std::string(gguf_get_val_str(gq, gguf_find_key(gq, k))); };
        auto q_u32 = [&](const char * k) { return gguf_get_val_u32(gq, gguf_find_key(gq, k)); };

        check(q_str("general.architecture") == "qwen4exp" && q_u32("qwen4exp.block_count") == 3 &&
              q_u32("qwen4exp.nextn_predict_layers") == 1 && q_str("exl3.codebook") == "mul1" &&
              q_u32("exl3.rows.version") == 1 && q_u32("qwen4exp.embedding_length_per_layer_input") == 160,
              "qwen4exp metadata, mul1 codebook, MTP and PLE keys");
        const int64_t kmul = gguf_find_key(gq, "qwen4exp.ple.layer_multipliers");
        const uint64_t * muls = kmul >= 0 ? (const uint64_t *) gguf_get_arr_data(gq, kmul) : nullptr;
        check(muls && gguf_get_arr_n(gq, kmul) == 3 &&
              muls[0] == 1000000000003ULL && muls[1] == 2000000000033ULL && muls[2] == 3000000000041ULL,
              "64-bit PLE layer multipliers preserved exactly");

        const std::vector<uint8_t> ple_tr = import_data(dq.get(), "per_layer_token_embd.exl3r_trellis");
        const std::vector<uint8_t> ple_bs = import_data(dq.get(), "per_layer_token_embd.exl3r_bias");
        const int64_t gate_idx = gguf_find_tensor(gq, "blk.0.ffn_gate_exps.exl3_trellis");
        check(gate_idx >= 0 &&
              gguf_get_tensor_size(gq, gate_idx) == (size_t) (4 * 40 * 8 * 64 * 2) &&
              gguf_find_tensor(gq, "blk.1.indexer.q_proj.exl3_trellis") >= 0 &&
              gguf_find_tensor(gq, "blk.1.indexer.k_proj.exl3_trellis") >= 0 &&
              gguf_find_tensor(gq, "blk.2.nextn.e_proj.exl3_trellis") >= 0 &&
              gguf_find_tensor(gq, "blk.2.nextn.h_proj.exl3_trellis") >= 0 &&
              ple_tr.size() == (size_t) (16 * 21 * 2) && ple_bs.size() == (size_t) (4 * 160 * 2),
              "qwen4exp stacked experts, split indexer, MTP projections, and PLE row table imported");

        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.load_mtp     = true;
        llama_model * model = common_exl3_load_model((dir / "qwen4exp").string(), mp);
        check(model != nullptr && llama_model_n_layer_nextn(model) == 1, "qwen4exp synthetic model loads on CPU with MTP");
        if (model) {
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx     = 64;
            cp.n_batch   = 16;
            cp.n_ubatch  = 16;
            cp.n_threads = 4;
            llama_context * ctx = llama_init_from_model(model, cp);
            std::vector<llama_token> tokens = { 65, 66, 256, 257, 302 };
            const int rc = ctx ? llama_decode(ctx, llama_batch_get_one(tokens.data(), (int32_t) tokens.size())) : -99;
            const float * logits = rc == 0 ? llama_get_logits_ith(ctx, -1) : nullptr;
            bool ok = rc == 0 && logits != nullptr;
            int bad_i = -1;
            float bad_v = 0.0f;
            for (int i = 0; ok && i < 384; i++) {
                if (!std::isfinite(logits[i])) {
                    ok = false;
                    bad_i = i;
                    bad_v = logits[i];
                }
            }
            if (!ok) {
                fprintf(stderr, "qwen4exp decode diag: ctx=%p rc=%d logits=%p bad_i=%d bad_v=%f\n",
                        (void *) ctx, rc, (const void *) logits, bad_i, bad_v);
            }
            check(ok, "qwen4exp forward pass with PLE, QSA, and MoE produces finite logits");
            llama_free(ctx);
            llama_model_free(model);
        }
    }

    fs::remove_all(dir);
    printf("%s\n", n_fail == 0 ? "OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    try {
        if (argc == 4 && strcmp(argv[1], "--compare") == 0) {
            return compare(argv[2], argv[3]);
        }
        if (argc == 4 && strcmp(argv[1], "--dump") == 0) {
            return dump(argv[2], argv[3]);
        }
        if (argc != 1) {
            fprintf(stderr, "usage: %s [--compare DIR CONTAINER.gguf | --dump DIR OUT.gguf]\n", argv[0]);
            return 1;
        }
        llama_backend_init();
        const int ret = synthetic();
        llama_backend_free();
        return ret;
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
