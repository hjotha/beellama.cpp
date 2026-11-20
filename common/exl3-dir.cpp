#include "exl3-dir.h"

#include "common.h"
#include "download.h"
#include "json.h"
#include "log.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include "hash/hash.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

// see ggml/exl3/README.md and conversion/exl3.py
static const uint32_t EXL3_MARKER_MCG  = 0xCBAC1FED;
static const uint32_t EXL3_MARKER_MUL1 = 0x83DCD12D;
static const char *   EXL3_PERMUTATION = "linear_attn_v_heads_grouped_to_tiled";

// sha256 of json.dumps({"normalizer": ..., "pre_tokenizer": ...}, sort_keys=True) of tokenizer.json -> tokenizer.ggml.pre
// the converter hashes tokenizer output instead, which needs the HF tokenizer; a new entry must be checked against a converted file
static const struct {
    const char * fingerprint;
    const char * pre;
} EXL3_PRE_TOKENIZERS[] = {
    { "821105e4f95dacb2d90e189de93c8273343574e7d2aae0ea82fc552ff0bc84b9", "qwen35" },
};

// one tensor of a safetensors file
struct st_tensor {
    std::string          dtype;
    std::vector<int64_t> shape;
    int                  shard = 0;
    uint64_t             offs  = 0; // file offset, often not aligned (2 mod 4 in real EXL3 files): only read into buffers
    uint64_t             size  = 0;
};

// one tensor of the import: the source tensor(s), optional slice, its V-head reorder, its type conversion and arithmetic
struct exl3_out_tensor {
    std::string              name;
    std::string              src;
    std::vector<std::string> srcs; // stacked along axis 0 when non-empty (MoE experts)
    ggml_type                type = GGML_TYPE_F32; // F32 converts the source, any other type is a copy of it
    std::vector<int64_t>     shape;                // row-major shape of the result
    const st_tensor *        stream_src = nullptr;
    int                      arith = 0;            // 1: x + 1 (zero-centered norm), 2: -exp(x) (A_log)

    // slice [slice_start, slice_start + slice_len) on `slice_axis` of the source tensor (e.g. fused indexer_qk -> q/k)
    int     slice_axis  = -1;
    int64_t slice_start = 0;
    int64_t slice_len   = 0;

    // [K, V, unit] -> [V, K, unit] on the elements [start, end) of axis `perm_axis`, see _reorder_v_heads in conversion/qwen.py
    int     perm_axis  = -1;
    int64_t perm_start = 0;
    int64_t perm_unit  = 0;
    int64_t perm_k     = 0;
    int64_t perm_v     = 0;
};

struct common_exl3_dir {
    std::vector<fs::path>                    shards;
    std::map<std::string, st_tensor>         st;
    std::vector<common_download_stream_file_info> stream_files;
    std::vector<size_t>                     shard_sizes;
    std::vector<size_t>                     shard_data_offsets;
    std::vector<std::pair<const st_tensor *, uint32_t>> stream_markers;
    std::vector<exl3_out_tensor>             out;
    std::unordered_map<std::string, size_t>  out_idx;
    gguf_context *                           meta     = nullptr;
    ggml_context *                           ctx_meta = nullptr;
    mutable std::vector<std::ifstream>       files;

    ~common_exl3_dir() {
        if (meta) {
            gguf_free(meta);
        }
        if (ctx_meta) {
            ggml_free(ctx_meta);
        }
    }
};

void common_exl3_dir_deleter::operator()(common_exl3_dir * dir) {
    delete dir;
}

static common_json exl3_read_json(const fs::path & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open " + path.string());
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return common_json::parse(ss.str());
}

static bool str_ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool str_starts_with(const std::string & s, const std::string & prefix) {
    return s.compare(0, prefix.size(), prefix) == 0;
}

static std::string str_replace_all(std::string s, const std::string & from, const std::string & to) {
    for (size_t pos = s.find(from); pos != std::string::npos; pos = s.find(from, pos + to.size())) {
        s.replace(pos, from.size(), to);
    }
    return s;
}

// json.dumps(v, sort_keys=True) of Python: ", " and ": " separators, ASCII only, shortest float repr
static void exl3_py_json(const common_json & v, std::string & out) {
    if (v.is_null()) {
        out += "null";
    } else if (v.is_boolean()) {
        out += v.get<bool>() ? "true" : "false";
    } else if (v.is_number_integer()) {
        out += v.dump();
    } else if (v.is_number_float()) {
        const double x = v.get<double>();
        if (!std::isfinite(x)) {
            out += std::isnan(x) ? "NaN" : x > 0 ? "Infinity" : "-Infinity";
            return;
        }
        // shortest round trip digits, then the layout of repr(): fixed for 1e-4 <= |x| < 1e16, else d.ddde+XX
        char buf[64];
        for (int prec = 1; prec <= 17; prec++) {
            snprintf(buf, sizeof(buf), "%.*e", prec - 1, x);
            if (strtod(buf, nullptr) == x) {
                break;
            }
        }
        std::string s = buf;
        const size_t epos = s.find('e');
        const int    exp  = atoi(s.c_str() + epos + 1);
        std::string  digits;
        for (size_t i = 0; i < epos; i++) {
            if (isdigit((unsigned char) s[i])) {
                digits += s[i];
            }
        }
        const std::string sign = std::signbit(x) ? "-" : "";
        std::string r;
        if (exp >= -4 && exp < 16) {
            if (exp >= 0) {
                std::string ip = digits.substr(0, std::min(digits.size(), (size_t) exp + 1));
                ip += std::string(exp + 1 - ip.size(), '0');
                std::string fp = digits.size() > (size_t) exp + 1 ? digits.substr(exp + 1) : "0";
                r = ip + "." + fp;
            } else {
                r = "0." + std::string(-exp - 1, '0') + digits;
            }
        } else {
            r = digits.substr(0, 1) + (digits.size() > 1 ? "." + digits.substr(1) : "");
            char eb[16];
            snprintf(eb, sizeof(eb), "e%c%02d", exp < 0 ? '-' : '+', std::abs(exp));
            r += eb;
        }
        out += sign + r;
    } else if (v.is_string()) {
        const std::string s = v.get<std::string>();
        out += '"';
        for (size_t i = 0; i < s.size(); ) {
            const unsigned char c = s[i];
            uint32_t cp = c;
            size_t   n  = 1;
            if (c >= 0xF0) {
                cp = c & 0x07; n = 4;
            } else if (c >= 0xE0) {
                cp = c & 0x0F; n = 3;
            } else if (c >= 0xC0) {
                cp = c & 0x1F; n = 2;
            }
            for (size_t k = 1; k < n && i + k < s.size(); k++) {
                cp = (cp << 6) | (s[i + k] & 0x3F);
            }
            i += n;
            char eb[16];
            switch (cp) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                case '\b': out += "\\b";  break;
                case '\f': out += "\\f";  break;
                default:
                    if (cp < 0x20 || cp > 0x7E) {
                        if (cp >= 0x10000) {
                            cp -= 0x10000;
                            snprintf(eb, sizeof(eb), "\\u%04x\\u%04x", 0xD800 + (cp >> 10), 0xDC00 + (cp & 0x3FF));
                        } else {
                            snprintf(eb, sizeof(eb), "\\u%04x", cp);
                        }
                        out += eb;
                    } else {
                        out += (char) cp;
                    }
            }
        }
        out += '"';
    } else if (v.is_array()) {
        out += '[';
        bool first = true;
        for (const auto & e : v) {
            out += first ? "" : ", ";
            first = false;
            exl3_py_json(e, out);
        }
        out += ']';
    } else if (v.is_object()) {
        std::vector<std::string> keys;
        for (const auto & [k, e] : v.items()) {
            keys.push_back(k);
        }
        std::sort(keys.begin(), keys.end());
        out += '{';
        for (size_t i = 0; i < keys.size(); i++) {
            out += i ? ", " : "";
            exl3_py_json(common_json(keys[i]), out);
            out += ": ";
            exl3_py_json(v.at(keys[i]), out);
        }
        out += '}';
    }
}

static std::string exl3_py_json(const common_json & v) {
    std::string out;
    exl3_py_json(v, out);
    return out;
}

static size_t st_dtype_size(const std::string & dtype) {
    if (dtype == "F64" || dtype == "I64" || dtype == "U64") {
        return 8;
    }
    if (dtype == "F32" || dtype == "I32" || dtype == "U32") {
        return 4;
    }
    if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16") {
        return 2;
    }
    if (dtype == "I8" || dtype == "U8" || dtype == "BOOL" || dtype == "F8_E4M3" || dtype == "F8_E5M2") {
        return 1;
    }
    return 0;
}

static ggml_type st_dtype_ggml(const std::string & dtype) {
    if (dtype == "F32")  { return GGML_TYPE_F32;  }
    if (dtype == "F16")  { return GGML_TYPE_F16;  }
    if (dtype == "BF16") { return GGML_TYPE_BF16; }
    if (dtype == "I16")  { return GGML_TYPE_I16;  }
    if (dtype == "I32")  { return GGML_TYPE_I32;  }
    return GGML_TYPE_COUNT;
}

// first key of `keys` that hp has, as find_hparam of the converter
static const common_json * exl3_find(const common_json & hp, std::initializer_list<const char *> keys) {
    for (const char * k : keys) {
        if (hp.contains(k)) {
            return &hp.at(k);
        }
    }
    return nullptr;
}

static int64_t exl3_get_int(const common_json & hp, const char * key) {
    if (!hp.contains(key) || !hp.at(key).is_number_integer()) {
        throw std::runtime_error(string_format("config.json: %s is missing or not an integer", key));
    }
    return hp.at(key).get<long long>();
}

static uint32_t exl3_u32(int64_t v, const char * what) {
    if (v < 0 || v > (int64_t) UINT32_MAX) {
        throw std::runtime_error(string_format("config.json: %s = %" PRId64 " is out of range", what, v));
    }
    return (uint32_t) v;
}

static bool exl3_looks_special(const std::string & t) {
    return t == "<pad>" || t == "<mask>" || t == "<2mass>" || t == "[@BOS@]" ||
           (str_starts_with(t, "<|") && str_ends_with(t, "|>")) ||
           (str_starts_with(t, "<\xef\xbd\x9c") && str_ends_with(t, "\xef\xbd\x9c>")) ||
           (str_starts_with(t, "<unused") && str_ends_with(t, ">"));
}

static std::string exl3_read_text(const fs::path & path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// a small JSON scanner: common_json keeps the order of object keys and parses the 250k keys of model.vocab in quadratic time
struct exl3_json_scan {
    const std::string & s;
    size_t p = 0;

    [[noreturn]] void fail() const {
        throw std::runtime_error(string_format("tokenizer.json: bad JSON at byte %zu", p));
    }
    void ws() {
        while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\r' || s[p] == '\t')) {
            p++;
        }
    }
    void expect(char c) {
        ws();
        if (p >= s.size() || s[p] != c) {
            fail();
        }
        p++;
    }
    bool next_is(char c) {
        ws();
        return p < s.size() && s[p] == c;
    }
    uint32_t hex4() {
        if (p + 4 > s.size()) {
            fail();
        }
        uint32_t v = 0;
        for (int i = 0; i < 4; i++) {
            const char c = s[p++];
            v = v*16 + (c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : (fail(), 0));
        }
        return v;
    }
    std::string str() {
        expect('"');
        std::string out;
        while (true) {
            if (p >= s.size()) {
                fail();
            }
            const char c = s[p++];
            if (c == '"') {
                return out;
            }
            if ((unsigned char) c < 0x20) {
                fail();
            }
            if (c != '\\') {
                out += c;
                continue;
            }
            const char e = p < s.size() ? s[p++] : (fail(), 0);
            switch (e) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'b':  out += '\b'; break;
                case 'f':  out += '\f'; break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00) {
                        if (p + 2 > s.size() || s[p] != '\\' || s[p + 1] != 'u') {
                            fail();
                        }
                        p += 2;
                        const uint32_t lo = hex4();
                        if (lo < 0xDC00 || lo >= 0xE000) {
                            fail();
                        }
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else if (cp >= 0xDC00 && cp < 0xE000) {
                        fail();
                    }
                    if (cp < 0x80) {
                        out += (char) cp;
                    } else if (cp < 0x800) {
                        out += (char) (0xC0 | (cp >> 6));
                        out += (char) (0x80 | (cp & 0x3F));
                    } else if (cp < 0x10000) {
                        out += (char) (0xE0 | (cp >> 12));
                        out += (char) (0x80 | ((cp >> 6) & 0x3F));
                        out += (char) (0x80 | (cp & 0x3F));
                    } else {
                        out += (char) (0xF0 | (cp >> 18));
                        out += (char) (0x80 | ((cp >> 12) & 0x3F));
                        out += (char) (0x80 | ((cp >> 6) & 0x3F));
                        out += (char) (0x80 | (cp & 0x3F));
                    }
                } break;
                default: fail();
            }
        }
    }
    void skip() {
        ws();
        if (p >= s.size()) {
            fail();
        }
        const char c = s[p];
        if (c == '"') {
            str();
        } else if (c == '{' || c == '[') {
            const char close = c == '{' ? '}' : ']';
            p++;
            if (next_is(close)) {
                p++;
                return;
            }
            while (true) {
                if (c == '{') {
                    str();
                    expect(':');
                }
                skip();
                if (next_is(',')) {
                    p++;
                    continue;
                }
                expect(close);
                return;
            }
        } else {
            const size_t b = p;
            while (p < s.size() && (isalnum((unsigned char) s[p]) || s[p] == '-' || s[p] == '+' || s[p] == '.')) {
                p++;
            }
            if (p == b) {
                fail();
            }
        }
    }
    uint64_t u64() {
        ws();
        if (p >= s.size() || !isdigit((unsigned char) s[p])) {
            fail();
        }
        uint64_t v = 0;
        while (p < s.size() && isdigit((unsigned char) s[p])) {
            const uint64_t d = (uint64_t) (s[p++] - '0');
            if (v > (UINT64_MAX - d) / 10) {
                fail();
            }
            v = v * 10 + d;
        }
        return v;
    }
    int64_t i64() {
        ws();
        bool neg = false;
        if (p < s.size() && s[p] == '-') {
            neg = true;
            p++;
        }
        const uint64_t u = u64();
        if (neg) {
            if (u > (uint64_t) INT64_MAX + 1) {
                fail();
            }
            return u == (uint64_t) INT64_MAX + 1 ? INT64_MIN : -(int64_t) u;
        }
        if (u > (uint64_t) INT64_MAX) {
            fail();
        }
        return (int64_t) u;
    }
    // [b, e) of the value of `key` in the object at p, false if it has none
    bool member(const std::string & key, size_t & b, size_t & e) {
        expect('{');
        if (next_is('}')) {
            return false;
        }
        while (true) {
            const std::string k = str();
            expect(':');
            ws();
            const size_t vb = p;
            skip();
            if (k == key) {
                b = vb;
                e = p;
                return true;
            }
            if (next_is(',')) {
                p++;
                continue;
            }
            expect('}');
            return false;
        }
    }
};

// every tensor of one safetensors file in linear time via exl3_json_scan (303k-tensor checkpoints take O(N^2) under common_json)
static size_t st_read_header(const fs::path & path, uint64_t fsize, int shard, std::map<std::string, st_tensor> & st,
        std::map<std::string, std::string> & owner, std::map<std::string, std::string> * st_meta, const common_download_stream_file_info * stream) {
    std::ifstream f(path, std::ios::binary);
    unsigned char len[8];
    if (!f || fsize < 8 || !f.read((char *) len, 8)) {
        throw std::runtime_error("cannot read " + path.string());
    }
    uint64_t n = 0;
    for (int i = 7; i >= 0; i--) {
        n = (n << 8) | len[i];
    }
    if (n < 2 || n > fsize - 8 || n > (uint64_t) 256 << 20) {
        throw std::runtime_error(string_format("%s: bad safetensors header size %" PRIu64, path.string().c_str(), n));
    }
    std::string hdr(n, '\0');
    if (!f.read(&hdr[0], (std::streamsize) n)) {
        throw std::runtime_error("cannot read the header of " + path.string());
    }
    const uint64_t data_start = 8 + n;
    if (stream && data_start != stream->header_end) {
        throw std::runtime_error("safetensors stream header size changed: " + path.string());
    }
    const std::string filename = stream ? stream->rel_path : path.filename().string();
    exl3_json_scan sc = { hdr };
    sc.expect('{');
    if (sc.next_is('}')) {
        return data_start;
    }
    while (true) {
        const std::string name = sc.str();
        sc.expect(':');
        if (name == "__metadata__") {
            if (st_meta && sc.next_is('{')) {
                sc.expect('{');
                if (!sc.next_is('}')) {
                    while (true) {
                        const std::string mk = sc.str();
                        sc.expect(':');
                        if (sc.next_is('"')) {
                            (*st_meta)[mk] = sc.str();
                        } else {
                            sc.skip();
                        }
                        if (sc.next_is(',')) {
                            sc.p++;
                            continue;
                        }
                        sc.expect('}');
                        break;
                    }
                } else {
                    sc.p++;
                }
            } else {
                sc.skip();
            }
        } else {
            st_tensor t;
            t.shard = shard;
            bool has_dtype = false, has_shape = false, has_offs = false;
            uint64_t b = 0, e = 0;
            size_t n_offs = 0;
            sc.expect('{');
            if (!sc.next_is('}')) {
                while (true) {
                    const std::string field = sc.str();
                    sc.expect(':');
                    if (field == "dtype") {
                        t.dtype = sc.str();
                        has_dtype = true;
                    } else if (field == "shape") {
                        sc.expect('[');
                        if (!sc.next_is(']')) {
                            while (true) {
                                const int64_t v = sc.i64();
                                if (v < 0) {
                                    throw std::runtime_error(string_format("%s: tensor %s has a bad shape", path.string().c_str(), name.c_str()));
                                }
                                t.shape.push_back(v);
                                if (sc.next_is(',')) {
                                    sc.p++;
                                    continue;
                                }
                                break;
                            }
                        }
                        sc.expect(']');
                        has_shape = true;
                    } else if (field == "data_offsets") {
                        sc.expect('[');
                        if (!sc.next_is(']')) {
                            while (true) {
                                const uint64_t off = sc.u64();
                                if (n_offs == 0) { b = off; }
                                else if (n_offs == 1) { e = off; }
                                n_offs++;
                                if (sc.next_is(',')) {
                                    sc.p++;
                                    continue;
                                }
                                break;
                            }
                        }
                        sc.expect(']');
                        has_offs = true;
                    } else {
                        sc.skip();
                    }
                    if (sc.next_is(',')) {
                        sc.p++;
                        continue;
                    }
                    sc.expect('}');
                    break;
                }
            } else {
                sc.p++;
            }
            if (!has_dtype || !has_shape || !has_offs) {
                throw std::runtime_error(string_format("%s: tensor %s is missing header fields", path.string().c_str(), name.c_str()));
            }
            const size_t es = st_dtype_size(t.dtype);
            if (es == 0) {
                throw std::runtime_error(string_format("%s: tensor %s has unknown dtype %s", path.string().c_str(), name.c_str(), t.dtype.c_str()));
            }
            uint64_t ne = 1;
            for (const int64_t v : t.shape) {
                if (v > 0 && ne > UINT64_MAX / (uint64_t) v) {
                    throw std::runtime_error(string_format("%s: tensor %s has a bad shape", path.string().c_str(), name.c_str()));
                }
                ne *= (uint64_t) v;
            }
            if (n_offs != 2 || b > e || e > fsize - data_start || ne > UINT64_MAX / es || e - b != ne*es) {
                throw std::runtime_error(string_format("%s: tensor %s has bad data offsets", path.string().c_str(), name.c_str()));
            }
            t.offs = data_start + b;
            t.size = e - b;
            const auto it = owner.find(name);
            if (it != owner.end()) {
                throw std::runtime_error(string_format("tensor %s is in both %s and %s", name.c_str(), it->second.c_str(), filename.c_str()));
            }
            owner.emplace(name, filename);
            st.emplace(name, std::move(t));
        }
        if (sc.next_is(',')) {
            sc.p++;
            continue;
        }
        sc.expect('}');
        break;
    }
    return data_start;
}

static void exl3_verify_index(const fs::path & path, const std::map<std::string, std::string> & owner) {
    const std::string text = exl3_read_text(path);
    exl3_json_scan sc = { text };
    size_t wb = 0, we = 0;
    if (!sc.member("weight_map", wb, we)) {
        throw std::runtime_error("model.safetensors.index.json has no weight_map");
    }
    sc.p = wb;
    sc.expect('{');
    if (sc.next_is('}')) {
        return;
    }
    while (true) {
        const std::string name = sc.str();
        sc.expect(':');
        const std::string file = sc.str();
        const auto it = owner.find(name);
        if (it == owner.end() || it->second != file) {
            throw std::runtime_error("model.safetensors.index.json lists " + name + " in " + file + ", but it is not there");
        }
        if (sc.next_is(',')) {
            sc.p++;
            continue;
        }
        sc.expect('}');
        break;
    }
}

// tokenizer.json, and its model.vocab apart; the rest of the file goes through common_json without the vocab
static common_json exl3_read_tokenizer(const fs::path & path, std::vector<std::pair<std::string, int64_t>> & vocab) {
    const std::string text = exl3_read_text(path);
    exl3_json_scan sc = { text };
    size_t mb = 0, me = 0, vb = 0, ve = 0;
    if (!sc.member("model", mb, me)) {
        throw std::runtime_error("tokenizer.json has no model");
    }
    sc.p = mb;
    if (!sc.member("vocab", vb, ve)) {
        throw std::runtime_error("tokenizer.json has no model.vocab");
    }
    sc.p = vb;
    sc.expect('{');
    if (!sc.next_is('}')) {
        while (true) {
            std::string tok = sc.str();
            sc.expect(':');
            sc.ws();
            const size_t ib = sc.p;
            while (sc.p < text.size() && isdigit((unsigned char) text[sc.p])) {
                sc.p++;
            }
            if (sc.p == ib || sc.p - ib > 18) {
                sc.fail();
            }
            vocab.emplace_back(std::move(tok), std::stoll(text.substr(ib, sc.p - ib)));
            if (sc.next_is(',')) {
                sc.p++;
                continue;
            }
            sc.expect('}');
            break;
        }
    }
    return common_json::parse(text.substr(0, vb) + "{}" + text.substr(ve));
}

// tokenizer.ggml.* and tokenizer.chat_template, as _set_vocab_gpt2 and gguf.SpecialVocab write them
static void exl3_set_vocab(gguf_context * meta, const fs::path & dir, const common_json & config, const common_json & hp) {
    std::vector<std::pair<std::string, int64_t>> vocab;
    const common_json tj = exl3_read_tokenizer(dir / "tokenizer.json", vocab);
    const bool has_tc = fs::is_regular_file(dir / "tokenizer_config.json");
    const common_json tc = has_tc ? exl3_read_json(dir / "tokenizer_config.json") : common_json::object();

    const common_json & model = tj.at("model");
    if (model.value("type", "") != "BPE") {
        throw std::runtime_error("tokenizer.json: only BPE tokenizers are supported");
    }

    std::string pre;
    {
        common_json fp = common_json::object();
        fp["normalizer"]    = tj.value("normalizer", common_json());
        fp["pre_tokenizer"] = tj.value("pre_tokenizer", common_json());
        const std::string s  = exl3_py_json(fp);
        const std::string fh = hash_sha256_hex(s.data(), s.size());
        for (const auto & e : EXL3_PRE_TOKENIZERS) {
            if (fh == e.fingerprint) {
                pre = e.pre;
            }
        }
        if (pre.empty()) {
            throw std::runtime_error("tokenizer.json: unknown pre-tokenizer " + fh + ", convert the model with convert_hf_to_gguf.py");
        }
    }

    // added tokens of tokenizer.json and tokenizer_config.json, as the HF tokenizer merges them
    struct added_token {
        std::string content;
        bool        special;
    };
    std::map<int64_t, added_token> added;
    auto add_token = [&](int64_t id, const common_json & a) {
        const added_token t = { a.at("content").get<std::string>(), a.value("special", false) };
        const auto it = added.find(id);
        if (it != added.end() && (it->second.content != t.content || it->second.special != t.special)) {
            throw std::runtime_error(string_format("added token %" PRId64 " differs between tokenizer.json and tokenizer_config.json", id));
        }
        added[id] = t;
    };
    const common_json added_tj = tj.value("added_tokens", common_json::array());
    for (const auto & a : added_tj) {
        add_token(a.at("id").get<long long>(), a);
    }
    if (tc.contains("added_tokens_decoder")) {
        for (const auto & [id, a] : tc.at("added_tokens_decoder").items()) {
            add_token(std::stoll(id), a);
        }
    }

    std::unordered_map<std::string, int64_t> vocab_ids;
    for (const auto & [t, id] : vocab) {
        if (!vocab_ids.emplace(t, id).second) {
            throw std::runtime_error("tokenizer.json: token " + t + " is twice in model.vocab");
        }
    }
    int64_t n_vocab = (int64_t) vocab.size();
    for (const auto & [id, t] : added) {
        n_vocab += vocab_ids.count(t.content) == 0;
    }
    if (hp.contains("vocab_size")) {
        n_vocab = exl3_get_int(hp, "vocab_size");
    }

    std::vector<std::string> tokens(n_vocab);
    std::vector<int32_t>     types(n_vocab, 0);
    for (const auto & [t, id] : vocab) {
        if (id >= n_vocab || types[id] != 0) {
            throw std::runtime_error(string_format("tokenizer.json: token id %" PRId64 " is not below vocab_size %" PRId64 " or is used twice", id, n_vocab));
        }
        tokens[id] = t;
        types[id]  = 1; // NORMAL
    }
    for (const auto & [id, t] : added) {
        if (id < 0 || id >= n_vocab || (types[id] != 0 && tokens[id] != t.content)) {
            throw std::runtime_error(string_format("added token %" PRId64 " is out of range or clashes with the vocabulary", id));
        }
        // the converter also round-trips a token that is not normalized through the HF tokenizer; for added tokens that gives the token itself
        if (t.special || exl3_looks_special(t.content)) {
            tokens[id] = t.content;
            types[id]  = 3; // CONTROL
        } else {
            tokens[id] = str_replace_all(t.content, "\xe2\x96\x81", " ");
            types[id]  = 4; // USER_DEFINED
        }
    }
    for (int64_t i = 0; i < n_vocab; i++) {
        if (types[i] == 0) {
            tokens[i] = "[PAD" + std::to_string(i) + "]";
            types[i]  = 5; // UNUSED
        }
    }

    std::vector<std::string> merges;
    for (const auto & m : model.value("merges", common_json::array())) {
        if (m.is_string()) {
            merges.push_back(m.get<std::string>());
        } else if (m.is_array() && m.size() == 2) {
            // a space inside a pair is written as U+0120
            merges.push_back(str_replace_all(m.at(0).get<std::string>(), " ", "\xc4\xa0") + " " +
                             str_replace_all(m.at(1).get<std::string>(), " ", "\xc4\xa0"));
        } else {
            throw std::runtime_error("tokenizer.json: unknown merges format");
        }
    }

    // special tokens and add_*_token flags, in the order gguf.SpecialVocab finds them
    static const char * typs[] = { "bos", "eos", "unk", "sep", "pad", "cls", "mask" };
    std::vector<std::pair<std::string, int64_t>> ids;
    std::vector<std::pair<std::string, bool>>    adds;
    auto find_id = [&](const std::string & typ) {
        return std::find_if(ids.begin(), ids.end(), [&](const auto & e) { return e.first == typ; });
    };
    auto find_add = [&](const std::string & typ) {
        return std::find_if(adds.begin(), adds.end(), [&](const auto & e) { return e.first == typ; });
    };
    auto set_id = [&](const std::string & typ, const common_json & id) {
        if (!id.is_number_integer()) {
            return;
        }
        if (id.get<long long>() < 0) {
            throw std::runtime_error("invalid value for special token type " + typ);
        }
        if (find_id(typ) == ids.end()) {
            ids.emplace_back(typ, id.get<long long>());
        }
    };

    std::map<std::string, std::string> content;
    for (const char * typ : typs) {
        const std::string key = std::string(typ) + "_token";
        if (tc.contains(key)) {
            const common_json & e = tc.at(key);
            if (e.is_string()) {
                content[typ] = e.get<std::string>();
            } else if (e.is_object() && e.contains("content") && e.at("content").is_string()) {
                content[typ] = e.at("content").get<std::string>();
            }
        }
    }
    if (content["bos"].empty() && !content["cls"].empty()) {
        content["bos"] = content["cls"];
    }
    if (content["eos"].empty() && !content["sep"].empty()) {
        content["eos"] = content["sep"];
    }
    if (tj.contains("post_processor") && !tj.at("post_processor").is_null()) {
        const common_json & pp = tj.at("post_processor");
        const common_json procs = pp.contains("processors") ? pp.at("processors") : common_json::array({ pp });
        for (const auto & p : procs) {
            const std::string type = p.value("type", "");
            if (type == "ByteLevel") {
                if (find_add("bos") == adds.end()) {
                    adds.emplace_back("bos", false);
                }
                if (find_add("eos") == adds.end()) {
                    adds.emplace_back("eos", false);
                }
            } else if (type == "RobertaProcessing" || type == "TemplateProcessing") {
                throw std::runtime_error("tokenizer.json: post_processor " + type + " is not supported, convert the model with convert_hf_to_gguf.py");
            }
        }
    }

    common_json chat_template;
    if (has_tc) {
        common_json alt;
        bool alt_is_list = false;
        if (fs::is_regular_file(dir / "chat_template.jinja")) {
            alt = exl3_read_text(dir / "chat_template.jinja");
            if (fs::is_directory(dir / "additional_chat_templates")) {
                for (const auto & e : fs::directory_iterator(dir / "additional_chat_templates")) {
                    alt_is_list = alt_is_list || e.path().extension() == ".jinja";
                }
            }
        } else if (fs::is_regular_file(dir / "chat_template.json")) {
            alt = exl3_read_json(dir / "chat_template.json").value("chat_template", common_json());
        }
        if (tc.contains("chat_template")) {
            chat_template = tc.at("chat_template");
        } else if (alt_is_list) {
            throw std::runtime_error("additional_chat_templates is not supported, convert the model with convert_hf_to_gguf.py");
        } else {
            chat_template = alt;
        }
        if (chat_template.is_array()) {
            throw std::runtime_error("a list of chat templates is not supported, convert the model with convert_hf_to_gguf.py");
        }
        if (!chat_template.is_null() && !chat_template.is_string()) {
            LOG_WRN("%s: bad type for chat_template in tokenizer_config.json - ignoring\n", __func__);
            chat_template = common_json();
        }

        for (const char * typ : typs) {
            const std::string add_key = std::string("add_") + typ + "_token";
            if (tc.contains(add_key) && tc.at(add_key).is_boolean()) {
                const bool v  = tc.at(add_key).get<bool>();
                const auto it = find_add(typ);
                if (it == adds.end()) {
                    adds.emplace_back(typ, v);
                } else if (it->second != v) {
                    LOG_WRN("%s: %s = %d in tokenizer_config.json differs from the post_processor, using the post_processor\n", __func__, add_key.c_str(), v);
                }
            }
            if (content[typ].empty()) {
                continue;
            }
            for (const auto & a : added_tj) {
                if (a.value("content", "") == content[typ]) {
                    set_id(typ, a.at("id"));
                    break;
                }
            }
        }
    }
    for (const char * typ : typs) {
        const std::string key = std::string(typ) + "_token_id";
        common_json id = config.value(key, common_json());
        if (id.is_null() && config.contains("text_config")) {
            id = config.at("text_config").value(key, common_json());
        }
        set_id(typ, id);
    }

    gguf_set_val_str(meta, "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(meta, "tokenizer.ggml.pre", pre.c_str());
    {
        std::vector<const char *> ptrs;
        for (const auto & t : tokens) {
            ptrs.push_back(t.c_str());
        }
        gguf_set_arr_str (meta, "tokenizer.ggml.tokens", ptrs.data(), ptrs.size());
        gguf_set_arr_data(meta, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    }
    if (!merges.empty()) {
        std::vector<const char *> ptrs;
        for (const auto & m : merges) {
            ptrs.push_back(m.c_str());
        }
        gguf_set_arr_str(meta, "tokenizer.ggml.merges", ptrs.data(), ptrs.size());
    } else {
        LOG_WRN("%s: tokenizer.json has no merges\n", __func__);
    }

    static const std::map<std::string, const char *> id_keys = {
        { "bos", "tokenizer.ggml.bos_token_id" }, { "eos", "tokenizer.ggml.eos_token_id" }, { "unk", "tokenizer.ggml.unknown_token_id" },
        { "sep", "tokenizer.ggml.seperator_token_id" }, { "pad", "tokenizer.ggml.padding_token_id" }, { "mask", "tokenizer.ggml.mask_token_id" },
    };
    static const std::map<std::string, const char *> add_keys = {
        { "bos", "tokenizer.ggml.add_bos_token" }, { "eos", "tokenizer.ggml.add_eos_token" }, { "sep", "tokenizer.ggml.add_sep_token" },
    };
    for (const auto & [typ, id] : ids) {
        const auto it = id_keys.find(typ);
        if (it == id_keys.end()) {
            LOG_WRN("%s: no key for special token type %s with id %" PRId64 " - skipping\n", __func__, typ.c_str(), id);
            continue;
        }
        gguf_set_val_u32(meta, it->second, exl3_u32(id, it->second));
    }
    for (const auto & [typ, v] : adds) {
        const auto it = add_keys.find(typ);
        if (it != add_keys.end()) {
            gguf_set_val_bool(meta, it->second, v);
        }
    }
    if (chat_template.is_string()) {
        gguf_set_val_str(meta, "tokenizer.chat_template", chat_template.get<std::string>().c_str());
    }
    if (has_tc && tc.contains("clean_up_tokenization_spaces") && tc.at("clean_up_tokenization_spaces").is_boolean()) {
        gguf_set_val_bool(meta, "tokenizer.ggml.clean_spaces", tc.at("clean_up_tokenization_spaces").get<bool>());
    }

    int nfc = -1;
    int lowercase = -1;
    int strip_accents = -1;
    std::function<void(const common_json &)> parse_normalizer = [&](const common_json & n) {
        if (!n.is_object()) {
            return;
        }
        const std::string type = n.value("type", "");
        if (type == "Lowercase") {
            lowercase = 1;
        } else if (type == "NFC") {
            nfc = 1;
        } else if (type == "StripAccents") {
            strip_accents = 1;
        } else if (type == "BertNormalizer") {
            if (n.contains("lowercase")) {
                lowercase = n.at("lowercase").get<bool>();
            }
            if (n.contains("strip_accents")) {
                strip_accents = n.at("strip_accents").get<bool>();
            }
        } else if (type == "Sequence") {
            for (const auto & e : n.value("normalizers", common_json::array())) {
                parse_normalizer(e);
            }
        }
    };
    parse_normalizer(tj.value("normalizer", common_json()));
    if (nfc >= 0) {
        gguf_set_val_bool(meta, "tokenizer.ggml.normalizer.nfc", nfc);
    }
    if (lowercase >= 0) {
        gguf_set_val_bool(meta, "tokenizer.ggml.normalizer.lowercase", lowercase);
    }
    if (strip_accents >= 0) {
        gguf_set_val_bool(meta, "tokenizer.ggml.normalizer.strip_accents", strip_accents);
    }
}

// HF linear (or EXL3 module) of a qwen35 layer -> GGUF base name
static const struct {
    const char * hf;
    const char * gguf;
} QWEN35_LINEARS[] = {
    { "mlp.gate_proj",           "ffn_gate"    },
    { "mlp.up_proj",             "ffn_up"      },
    { "mlp.down_proj",           "ffn_down"    },
    { "self_attn.q_proj",        "attn_q"      },
    { "self_attn.k_proj",        "attn_k"      },
    { "self_attn.v_proj",        "attn_v"      },
    { "self_attn.o_proj",        "attn_output" },
    { "linear_attn.in_proj_qkv", "attn_qkv"    },
    { "linear_attn.in_proj_z",   "attn_gate"   },
    { "linear_attn.out_proj",    "ssm_out"     },
    { "linear_attn.in_proj_a",   "ssm_alpha"   },
    { "linear_attn.in_proj_b",   "ssm_beta"    },
};

// other tensors of a qwen35 layer
static const struct {
    const char * hf;
    const char * gguf;
} QWEN35_LAYER_TENSORS[] = {
    { "input_layernorm.weight",          "attn_norm.weight"           },
    { "post_attention_layernorm.weight", "post_attention_norm.weight" },
    { "self_attn.q_norm.weight",         "attn_q_norm.weight"         },
    { "self_attn.k_norm.weight",         "attn_k_norm.weight"         },
    { "linear_attn.norm.weight",         "ssm_norm.weight"            },
    { "linear_attn.A_log",               "ssm_a"                      },
    { "linear_attn.dt_bias",             "ssm_dt.bias"                },
    { "linear_attn.conv1d.weight",       "ssm_conv1d.weight"          },
};

// the NextN tensors of an MTP layer of qwen35 / qwen4exp
static const struct {
    const char * hf;
    const char * gguf;
} QWEN35_NEXTN[] = {
    { "eh_proj",                          "nextn.eh_proj"                 },
    { "nextn_fc_embedding",               "nextn.e_proj"                  },
    { "nextn_fc_hidden",                  "nextn.h_proj"                  },
    { "enorm.weight",                     "nextn.enorm.weight"            },
    { "hnorm.weight",                     "nextn.hnorm.weight"            },
    { "shared_head.norm.weight",          "nextn.shared_head_norm.weight" },
    { "nextn_hc_head.hc_norm.weight",              "nextn.hc_head_norm.weight"     },
    { "nextn_hc_head.input_mix_weight_down.weight","nextn.hc_head_down.weight"     },
    { "nextn_hc_head.input_mix_weight_up.weight",  "nextn.hc_head_up.weight"       },
};

// extra linear projections & plain tensors of qwen4exp layers
static const struct {
    const char * hf;
    const char * gguf;
} QWEN4EXP_LINEARS[] = {
    { "mlp.shared_expert.gate_proj",              "ffn_gate_shexp" },
    { "mlp.shared_expert.up_proj",                "ffn_up_shexp"   },
    { "mlp.shared_expert.down_proj",              "ffn_down_shexp" },
    { "mlp.gate",                                 "ffn_gate_inp"       },
    { "mlp.shared_expert_gate",                   "ffn_gate_inp_shexp" },
    { "attn_hyper_connection.input_mix_weight_down", "hc_attn_down"    },
    { "attn_hyper_connection.input_mix_weight_up",   "hc_attn_up"      },
    { "attn_hyper_connection.block_inject_weight",   "hc_attn_inject"  },
    { "mlp_hyper_connection.input_mix_weight_down",  "hc_ffn_down"     },
    { "mlp_hyper_connection.input_mix_weight_up",    "hc_ffn_up"       },
    { "mlp_hyper_connection.block_inject_weight",    "hc_ffn_inject"   },
    { "ple.key_proj",                             "ple_key"            },
    { "ple.value_proj",                           "ple_value"          },
};

static const struct {
    const char * hf;
    const char * gguf;
} QWEN4EXP_LAYER_TENSORS[] = {
    { "attn_hyper_connection.hc_norm.weight", "hc_attn_norm.weight"    },
    { "mlp_hyper_connection.hc_norm.weight",  "hc_ffn_norm.weight"     },
    { "self_attn.indexer.q_layernorm.weight", "indexer.q_norm.weight"  },
    { "self_attn.indexer.k_layernorm.weight", "indexer.k_norm.weight"  },
    { "ple.norm_key.weight",                  "ple_norm_key.weight"    },
    { "ple.norm_query.weight",                "ple_norm_query.weight"  },
    { "ple.norm_conv.weight",                 "ple_norm_conv.weight"   },
    { "ple.conv1d.weight",                    "ple_conv1d.weight"      },
};

// "mtp.layers.<i>.<rest>" -> "model.layers.<n_layer + i>.<rest>", "mtp.<hf>.<part>" -> the NextN
// names of layer n_layer, as _QwenMtpMixin / Qwen4ExpTextModel.filter_tensors; n_mtp is the number of MTP layers seen
static std::string exl3_mtp_name(const std::string & n, int64_t n_layer, int64_t & n_mtp, bool is_qwen4exp) {
    static const struct {
        const char * hf;
        const char * name;
        bool         q4_only;
    } remap[] = {
        { "fc",                     "eh_proj",          false },
        { "fc_embedding",           "nextn_fc_embedding", true },
        { "fc_hidden",              "nextn_fc_hidden",    true },
        { "hyper_connection_mixer", "nextn_hc_head",      true },
        { "pre_fc_norm_embedding",  "enorm",            false },
        { "pre_fc_norm_hidden",     "hnorm",            false },
        { "norm",                   "shared_head.norm", false },
    };
    const std::string rest = n.substr(strlen("mtp."));
    if (str_starts_with(rest, "layers.")) {
        const size_t dot = rest.find('.', strlen("layers."));
        const std::string num = dot == std::string::npos ? "" : rest.substr(strlen("layers."), dot - strlen("layers."));
        if (!num.empty() && num.find_first_not_of("0123456789") == std::string::npos) {
            const int64_t i = std::stoll(num);
            n_mtp = std::max(n_mtp, i + 1);
            return "model.layers." + std::to_string(n_layer + i) + rest.substr(dot);
        }
    }
    const size_t dot = rest.find('.');
    if (dot != std::string::npos && (is_qwen4exp || rest.find('.', dot + 1) == std::string::npos)) {
        for (const auto & e : remap) {
            if ((!e.q4_only || is_qwen4exp) && rest.compare(0, dot, e.hf) == 0 && strlen(e.hf) == dot) {
                return "model.layers." + std::to_string(n_layer) + "." + e.name + rest.substr(dot);
            }
        }
    }
    throw std::runtime_error(string_format("MTP tensor %s is not known to the %s importer",
            n.c_str(), is_qwen4exp ? "qwen4exp" : "qwen35"));
}

// "model.layers.<il>.<local>" -> il, local
static bool exl3_split_layer(const std::string & name, int64_t & il, std::string & local) {
    static const std::string prefix = "model.layers.";
    if (!str_starts_with(name, prefix)) {
        return false;
    }
    const size_t dot = name.find('.', prefix.size());
    if (dot == std::string::npos || dot == prefix.size()) {
        return false;
    }
    const std::string num = name.substr(prefix.size(), dot - prefix.size());
    if (num.find_first_not_of("0123456789") != std::string::npos) {
        return false;
    }
    il    = std::stoll(num);
    local = name.substr(dot + 1);
    return true;
}

// "<prefix>.experts.<e>.<proj>" -> prefix + ".experts." + proj, e
static bool exl3_split_expert(const std::string & base, std::string & merged, int64_t & eid) {
    const size_t pos = base.rfind(".experts.");
    if (pos == std::string::npos) {
        return false;
    }
    const size_t num_start = pos + strlen(".experts.");
    const size_t dot = base.find('.', num_start);
    if (dot == std::string::npos || dot == num_start || base.find('.', dot + 1) != std::string::npos) {
        return false;
    }
    const std::string num = base.substr(num_start, dot - num_start);
    if (num.find_first_not_of("0123456789") != std::string::npos) {
        return false;
    }
    eid    = std::stoll(num);
    merged = base.substr(0, pos) + ".experts." + base.substr(dot + 1);
    return true;
}

static void exl3_read_exact(std::ifstream & f, uint64_t offs, void * dst, uint64_t size, const char * what) {
    f.seekg((std::streamoff) offs);
    uint8_t * p = (uint8_t *) dst;
    uint64_t rem = size;
    while (rem > 0) {
        const size_t chunk = (size_t) std::min<uint64_t>(rem, 64ULL << 20);
        if (!f.read((char *) p, (std::streamsize) chunk)) {
            throw std::runtime_error(string_format("cannot read %s", what));
        }
        p   += chunk;
        rem -= chunk;
    }
}

static common_json exl3_prepare_rope(const common_json & hp, const char * arch_label) {
    if (hp.contains("is_causal") && hp.at("is_causal").is_boolean() && !hp.at("is_causal").get<bool>()) {
        throw std::runtime_error(string_format("config.json: is_causal false is not supported by the %s importer", arch_label));
    }
    common_json rope = hp.contains("rope_parameters") ? hp.at("rope_parameters") : hp.value("rope_scaling", common_json());
    if (rope.is_null()) {
        rope = common_json::object();
    }
    if (rope.contains("full_attention") || rope.contains("sliding_attention")) {
        throw std::runtime_error(string_format("config.json: per-attention-type rope parameters are not supported by the %s importer", arch_label));
    }
    if (!rope.contains("rope_theta")) {
        if (const common_json * v = exl3_find(hp, { "global_rope_theta", "rope_global_theta", "rope_theta_global", "rope_theta", "rotary_emb_base" })) {
            rope["rope_theta"] = *v;
        }
    }
    if (!rope.contains("rope_type") && rope.contains("type")) {
        rope["rope_type"] = rope.at("type");
    }
    if (!rope.contains("partial_rotary_factor")) {
        if (const common_json * v = exl3_find(hp, { "partial_rotary_factor", "rope_pct", "rope_percent" })) {
            rope["partial_rotary_factor"] = *v;
        }
    }
    if (rope.contains("rope_type") && !rope.at("rope_type").is_null() && rope.at("rope_type").get<std::string>() != "default") {
        throw std::runtime_error("config.json: rope_type " + rope.at("rope_type").get<std::string>() + " is not supported by the importer");
    }
    return rope;
}

// the <arch>.* and general.* keys of Qwen3_5TextModel / Qwen4ExpTextModel set_gguf_parameters
static void exl3_emit_hparams(gguf_context * meta, bool is_qwen4exp, const common_json & hp, const common_json & config,
        const std::string & name, uint32_t ftype, int64_t n_mtp, int64_t ple_row_dim,
        const std::vector<uint64_t> & ple_multipliers,
        const std::vector<uint64_t> & ple_offsets,
        const std::vector<uint64_t> & ple_vocab_sizes) {
    const std::string arch = is_qwen4exp ? "qwen4exp" : "qwen35";
    auto k = [&](const char * suffix) -> std::string { return arch + "." + suffix; };

    if (!is_qwen4exp) {
        static const char * refused[] = {
            "moe_intermediate_size", "shared_expert_intermediate_size", "num_local_experts", "num_experts", "n_routed_experts",
            "num_experts_per_tok", "num_experts_per_token", "top_k_experts", "n_group", "topk_group",
            "prefix_dense_intermediate_size", "dense_intermediate_size", "layer_norm_eps", "layer_norm_epsilon", "norm_epsilon",
            "local_rope_theta", "rope_local_theta", "rope_theta_local", "swa_rope_theta", "rope_local_base_freq",
            "score_function", "scoring_func", "score_func", "moe_router_activation", "moe_router_activation_func",
            "expert_selection_fn", "router_score_func",
        };
        for (const char * key : refused) {
            if (hp.contains(key) && !hp.at(key).is_null()) {
                throw std::runtime_error(string_format("config.json: %s is not supported by the qwen35 importer, convert the model with convert_hf_to_gguf.py", key));
            }
        }
    }

    const common_json rope = exl3_prepare_rope(hp, arch.c_str());
    const int64_t n_layer  = exl3_get_int(hp, "num_hidden_layers");

    gguf_set_val_str(meta, "general.architecture", arch.c_str());
    gguf_set_val_str(meta, "general.type", "model");
    gguf_set_val_str(meta, "general.name", name.c_str());

    // block_count counts the MTP layers too, as _QwenMtpMixin does
    gguf_set_val_u32(meta, k("block_count").c_str(), exl3_u32(n_layer + n_mtp, "num_hidden_layers"));
    if (const common_json * v = exl3_find(hp, { "max_position_embeddings", "n_ctx", "n_positions", "max_length", "max_sequence_length", "model_max_length" })) {
        gguf_set_val_u32(meta, k("context_length").c_str(), exl3_u32(v->get<long long>(), "context length"));
    }
    if (const common_json * v = exl3_find(hp, { "hidden_size", "n_embd", "dim" })) {
        gguf_set_val_u32(meta, k("embedding_length").c_str(), exl3_u32(v->get<long long>(), "hidden_size"));
    }
    if (const common_json * v = exl3_find(hp, { "prefix_dense_intermediate_size", "dense_intermediate_size", "intermediate_size", "n_inner", "hidden_dim" })) {
        gguf_set_val_u32(meta, k("feed_forward_length").c_str(), exl3_u32(v->get<long long>(), "intermediate_size"));
    }
    if (const common_json * v = exl3_find(hp, { "num_attention_heads", "n_head", "n_heads" })) {
        gguf_set_val_u32(meta, k("attention.head_count").c_str(), exl3_u32(v->get<long long>(), "num_attention_heads"));
    }
    if (const common_json * v = exl3_find(hp, { "num_key_value_heads", "n_kv_heads" })) {
        gguf_set_val_u32(meta, k("attention.head_count_kv").c_str(), exl3_u32(v->get<long long>(), "num_key_value_heads"));
    }
    std::vector<int32_t> sections = { 11, 11, 10, 0 };
    const bool has_sections = rope.contains("mrope_section");
    if (has_sections) {
        sections.clear();
        for (const auto & s : rope.at("mrope_section")) {
            sections.push_back((int32_t) s.get<long long>());
        }
        sections.resize(std::max<size_t>(sections.size(), 4), 0);
        sections.resize(4);
        gguf_set_arr_data(meta, k("rope.dimension_sections").c_str(), GGUF_TYPE_INT32, sections.data(), sections.size());
    }
    if (rope.contains("rope_theta") && !rope.at("rope_theta").is_null()) {
        gguf_set_val_f32(meta, k("rope.freq_base").c_str(), (float) rope.at("rope_theta").get<double>());
    }
    if (const common_json * v = exl3_find(hp, { "rms_norm_eps", "norm_eps" })) {
        gguf_set_val_f32(meta, k("attention.layer_norm_rms_epsilon").c_str(), (float) v->get<double>());
    }
    if (is_qwen4exp) {
        if (const common_json * v = exl3_find(hp, { "num_local_experts", "num_experts", "n_routed_experts" })) {
            gguf_set_val_u32(meta, k("expert_count").c_str(), exl3_u32(v->get<long long>(), "num_experts"));
        }
        if (const common_json * v = exl3_find(hp, { "num_experts_per_tok", "num_experts_per_token", "top_k_experts" })) {
            gguf_set_val_u32(meta, k("expert_used_count").c_str(), exl3_u32(v->get<long long>(), "num_experts_per_tok"));
        }
    }
    const bool has_head_dim = hp.contains("head_dim") && !hp.at("head_dim").is_null();
    if (has_head_dim) {
        const uint32_t head_dim = exl3_u32(exl3_get_int(hp, "head_dim"), "head_dim");
        gguf_set_val_u32(meta, k("attention.key_length").c_str(), head_dim);
        gguf_set_val_u32(meta, k("attention.value_length").c_str(), head_dim);
    }
    gguf_set_val_u32(meta, "general.file_type", ftype);
    if (is_qwen4exp) {
        if (hp.contains("moe_intermediate_size") && !hp.at("moe_intermediate_size").is_null()) {
            gguf_set_val_u32(meta, k("expert_feed_forward_length").c_str(), exl3_u32(exl3_get_int(hp, "moe_intermediate_size"), "moe_intermediate_size"));
        }
        if (hp.contains("shared_expert_intermediate_size") && !hp.at("shared_expert_intermediate_size").is_null()) {
            gguf_set_val_u32(meta, k("expert_shared_feed_forward_length").c_str(), exl3_u32(exl3_get_int(hp, "shared_expert_intermediate_size"), "shared_expert_intermediate_size"));
        }
    }

    gguf_set_val_u32(meta, k("ssm.conv_kernel").c_str(),    exl3_u32(exl3_get_int(hp, "linear_conv_kernel_dim"), "linear_conv_kernel_dim"));
    gguf_set_val_u32(meta, k("ssm.state_size").c_str(),     exl3_u32(exl3_get_int(hp, "linear_key_head_dim"), "linear_key_head_dim"));
    gguf_set_val_u32(meta, k("ssm.group_count").c_str(),    exl3_u32(exl3_get_int(hp, "linear_num_key_heads"), "linear_num_key_heads"));
    gguf_set_val_u32(meta, k("ssm.time_step_rank").c_str(), exl3_u32(exl3_get_int(hp, "linear_num_value_heads"), "linear_num_value_heads"));
    gguf_set_val_u32(meta, k("ssm.inner_size").c_str(),
        exl3_u32(exl3_get_int(hp, "linear_value_head_dim") * exl3_get_int(hp, "linear_num_value_heads"), "linear value size"));
    if (hp.contains("layer_types") && !hp.at("layer_types").is_null()) {
        std::vector<int8_t> recurrent;
        for (const auto & t : hp.at("layer_types")) {
            recurrent.push_back(t.get<std::string>() == "linear_attention");
        }
        if ((int64_t) recurrent.size() != n_layer) {
            throw std::runtime_error(string_format("config.json: layer_types has %zu entries, expected %" PRId64, recurrent.size(), n_layer));
        }
        // MTP layers are full attention
        recurrent.resize(n_layer + n_mtp, 0);
        gguf_set_arr_data(meta, k("attention.recurrent_layers").c_str(), GGUF_TYPE_BOOL, recurrent.data(), recurrent.size());
    }
    gguf_set_val_u32(meta, k("full_attention_interval").c_str(), exl3_u32(hp.value("full_attention_interval", (long long) 4), "full_attention_interval"));
    const int64_t rope_dim = has_head_dim ? exl3_get_int(hp, "head_dim") : exl3_get_int(hp, "hidden_size") / exl3_get_int(hp, "num_attention_heads");
    const double  partial  = rope.contains("partial_rotary_factor") ? rope.at("partial_rotary_factor").get<double>() : 0.25;
    gguf_set_val_u32(meta, k("rope.dimension_count").c_str(), exl3_u32((int64_t) ((double) rope_dim * partial), "rope dimension count"));
    if (!has_sections) {
        gguf_set_arr_data(meta, k("rope.dimension_sections").c_str(), GGUF_TYPE_INT32, sections.data(), sections.size());
    }
    if (is_qwen4exp) {
        gguf_set_val_u32(meta, k("hyper_connection.count").c_str(),          exl3_u32(exl3_get_int(hp, "hc_count"), "hc_count"));
        gguf_set_val_u32(meta, k("hyper_connection.low_rank").c_str(),       exl3_u32(exl3_get_int(hp, "hc_lowrank"), "hc_lowrank"));
        gguf_set_val_u32(meta, k("attention.indexer.head_count").c_str(),    exl3_u32(exl3_get_int(hp, "indexer_n_heads"), "indexer_n_heads"));
        gguf_set_val_u32(meta, k("attention.indexer.key_length").c_str(),    exl3_u32(exl3_get_int(hp, "indexer_head_dim"), "indexer_head_dim"));
        gguf_set_val_u32(meta, k("attention.indexer.top_k").c_str(),         exl3_u32(exl3_get_int(hp, "indexer_budget"), "indexer_budget"));
        const int32_t ratio = (int32_t) exl3_get_int(hp, "indexer_compress_ratio");
        std::vector<int32_t> ratios;
        for (const auto & lt : hp.at("layer_types")) {
            const std::string s = lt.get<std::string>();
            ratios.push_back((s == "full_attention" || s == "indexed_attention") ? ratio : 0);
        }
        ratios.resize(n_layer + n_mtp, ratio);
        gguf_set_arr_data(meta, k("attention.compress_ratios").c_str(), GGUF_TYPE_INT32, ratios.data(), ratios.size());

        std::vector<int32_t> ple_layers;
        if (hp.contains("ple_layer_ids") && !hp.at("ple_layer_ids").is_null()) {
            for (const auto & v : hp.at("ple_layer_ids")) {
                ple_layers.push_back((int32_t) (v.get<long long>() - 1));
            }
        }
        if (!ple_layers.empty()) {
            gguf_set_arr_data(meta, k("ple.layers").c_str(),          GGUF_TYPE_INT32, ple_layers.data(), ple_layers.size());
            gguf_set_val_u32(meta,  k("ple.ngram_size").c_str(),      exl3_u32(exl3_get_int(hp, "ngram_size"), "ngram_size"));
            gguf_set_val_u32(meta,  k("ple.heads_per_ngram").c_str(), exl3_u32(exl3_get_int(hp, "heads_per_ngram"), "heads_per_ngram"));
            gguf_set_val_u32(meta,  k("ple.conv_kernel").c_str(),     exl3_u32(exl3_get_int(hp, "ple_conv_kernel_size"), "ple_conv_kernel_size"));
            if (!hp.contains("eos_token_id") || hp.at("eos_token_id").is_null()) {
                throw std::runtime_error("config.json: eos_token_id is required for PLE");
            }
            const common_json & eos_v = hp.at("eos_token_id");
            const int64_t eos = eos_v.is_array() ? eos_v.back().get<long long>() : eos_v.get<long long>();
            gguf_set_val_u32(meta, k("ple.eos_token_id").c_str(), exl3_u32(eos, "eos_token_id"));
            const common_json * img = hp.contains("image_token_id") && !hp.at("image_token_id").is_null()
                ? &hp.at("image_token_id")
                : (config.contains("image_token_id") && !config.at("image_token_id").is_null() ? &config.at("image_token_id") : nullptr);
            if (img != nullptr) {
                gguf_set_val_u32(meta, k("ple.image_token_id").c_str(), exl3_u32(img->get<long long>(), "image_token_id"));
            }
            if (ple_row_dim > 0) {
                gguf_set_val_u32(meta, k("embedding_length_per_layer_input").c_str(), exl3_u32(ple_row_dim, "ple row dim"));
            }
            if (!ple_multipliers.empty()) {
                gguf_set_arr_data(meta, k("ple.layer_multipliers").c_str(), GGUF_TYPE_UINT64, ple_multipliers.data(), ple_multipliers.size());
            }
            if (!ple_offsets.empty()) {
                gguf_set_arr_data(meta, k("ple.head_offsets").c_str(),      GGUF_TYPE_UINT64, ple_offsets.data(),     ple_offsets.size());
            }
            if (!ple_vocab_sizes.empty()) {
                gguf_set_arr_data(meta, k("ple.head_vocab_sizes").c_str(),  GGUF_TYPE_UINT64, ple_vocab_sizes.data(), ple_vocab_sizes.size());
            }
        }
    }
    if (n_mtp > 0) {
        gguf_set_val_u32(meta, k("nextn_predict_layers").c_str(), exl3_u32(n_mtp, "MTP layers"));
    }
    gguf_set_val_u32(meta, "general.quantization_version", GGML_QNT_VERSION);
}

// copy [slice_start, slice_start + slice_len) along `slice_axis` from row-major `src_shape`
static void exl3_slice_copy(const uint8_t * src, uint8_t * dst, const std::vector<int64_t> & src_shape, size_t es, const exl3_out_tensor & t) {
    int64_t outer = 1;
    int64_t inner = (int64_t) es;
    for (int i = 0; i < t.slice_axis; i++) {
        outer *= src_shape[i];
    }
    for (size_t i = (size_t) t.slice_axis + 1; i < src_shape.size(); i++) {
        inner *= src_shape[i];
    }
    const int64_t dim = src_shape[t.slice_axis];
    GGML_ASSERT(t.slice_start >= 0 && t.slice_len >= 0 && t.slice_start + t.slice_len <= dim);
    for (int64_t o = 0; o < outer; o++) {
        memcpy(dst + o*t.slice_len*inner, src + (o*dim + t.slice_start)*inner, (size_t) (t.slice_len*inner));
    }
}

// [K, V, unit] -> [V, K, unit] on the elements [start, end) of one axis of a row-major tensor, the rest is copied
static void exl3_permute(const uint8_t * src, uint8_t * dst, const std::vector<int64_t> & shape, size_t es, const exl3_out_tensor & t) {
    int64_t outer = 1;
    int64_t inner = (int64_t) es;
    for (int i = 0; i < t.perm_axis; i++) {
        outer *= shape[i];
    }
    for (size_t i = t.perm_axis + 1; i < shape.size(); i++) {
        inner *= shape[i];
    }
    const int64_t dim  = shape[t.perm_axis];
    const int64_t unit = t.perm_unit*inner;
    GGML_ASSERT(dim - t.perm_start == t.perm_k*t.perm_v*t.perm_unit);
    for (int64_t o = 0; o < outer; o++) {
        const uint8_t * s = src + o*dim*inner;
        uint8_t       * d = dst + o*dim*inner;
        memcpy(d, s, t.perm_start*inner);
        for (int64_t g = 0; g < t.perm_k*t.perm_v; g++) {
            const int64_t v = g / t.perm_k;
            const int64_t k = g % t.perm_k;
            memcpy(d + t.perm_start*inner + g*unit, s + t.perm_start*inner + (k*t.perm_v + v)*unit, unit);
        }
    }
}

bool common_exl3_is_dir(const std::string & path) {
    std::error_code ec;
    return fs::is_directory(path, ec);
}

common_exl3_dir_ptr common_exl3_dir_open(const std::string & path) {
    const fs::path dir(path);
    common_exl3_dir_ptr d(new common_exl3_dir);

    const common_json config = exl3_read_json(dir / "config.json");
    common_json hp = config;
    if (config.contains("text_config") && config.at("text_config").is_object()) {
        for (const auto & [k, v] : config.at("text_config").items()) {
            hp[k] = v;
        }
    }
    // only the quant_method and the provenance are taken from it: quantization_config.json (tensor_storage) is not read,
    // it lists the modules of the default model walk only, the safetensors headers are the inventory
    const common_json qc = config.value("quantization_config", common_json());
    if (!qc.is_object() || qc.value("quant_method", "") != "exl3") {
        throw std::runtime_error("config.json has no quantization_config with quant_method exl3");
    }
    std::string arch;
    if (config.contains("architectures") && config.at("architectures").is_array() && !config.at("architectures").empty()) {
        arch = config.at("architectures").at(0).get<std::string>();
    }
    const bool is_qwen35  = (arch == "Qwen3_5ForConditionalGeneration"  || arch == "Qwen3_5ForCausalLM");
    const bool is_qwen4exp = (arch == "Qwen4ExpForConditionalGeneration" || arch == "Qwen4ExpForCausalLM");
    if (!is_qwen35 && !is_qwen4exp) {
        throw std::runtime_error("only Qwen3.5 (qwen35) and Qwen4Exp (qwen4exp) EXL3 models are loaded from a directory, not " + arch + ": convert the model with convert_hf_to_gguf.py");
    }
    if (fs::exists(dir / "modules.json")) {
        throw std::runtime_error("sentence-transformers pooling (modules.json) is not supported, convert the model with convert_hf_to_gguf.py");
    }

    // every *.safetensors file of the directory except calibration traces and the vision tower sidecar
    for (const auto & e : fs::directory_iterator(dir)) {
        if (e.is_regular_file() && e.path().extension() == ".safetensors") {
            const std::string fn = e.path().filename().string();
            if (str_starts_with(fn, "cal_trace") || str_starts_with(fn, "vision_")) {
                continue;
            }
            d->shards.push_back(e.path());
        }
    }
    if (common_download_get_stream_file_infos(path, d->stream_files)) {
        d->shards.clear();
        std::vector<common_download_stream_file_info> kept_stream_files;
        for (const auto & info : d->stream_files) {
            const std::string fn = info.rel_path.empty() ? fs::path(info.temp_path).filename().string() : info.rel_path;
            if (str_starts_with(fn, "cal_trace") || str_starts_with(fn, "vision_")) {
                continue;
            }
            d->shards.emplace_back(info.temp_path);
            kept_stream_files.push_back(info);
        }
        d->stream_files = std::move(kept_stream_files);
    } else {
        std::sort(d->shards.begin(), d->shards.end());
    }
    if (d->shards.empty()) {
        throw std::runtime_error("no .safetensors files");
    }
    std::map<std::string, std::string> owner;
    for (size_t i = 0; i < d->shards.size(); i++) {
        std::map<std::string, std::string> st_meta;
        const auto * stream = d->stream_files.empty() ? nullptr : &d->stream_files[i];
        const auto size = stream ? stream->total_size : fs::file_size(d->shards[i]);
        d->shard_sizes.push_back(size);
        d->shard_data_offsets.push_back(st_read_header(d->shards[i], size, (int) i, d->st, owner, &st_meta, stream));
        if ((stream ? stream->rel_path : d->shards[i].filename().string()) == "ngram_embedding.safetensors") {
            if (st_meta["format"] != "exl3_ngram_trellis" || st_meta["version"] != "1") {
                throw std::runtime_error("ngram_embedding.safetensors: not an exllamav3 n-gram table of version 1");
            }
        }
        if (!stream) {
            d->files.emplace_back(d->shards[i], std::ios::binary);
        }
    }
    // an index is not needed, but it must agree with the files
    if (fs::is_regular_file(dir / "model.safetensors.index.json")) {
        exl3_verify_index(dir / "model.safetensors.index.json", owner);
    }

    const int64_t n_layer = exl3_get_int(hp, "num_hidden_layers");

    // the converter's names drop "language_model."; MTP layers follow the trunk layers
    // the MTP layer count comes from the tensors, not from mtp_num_hidden_layers (the 4B EXL3 file declares 1 and has none)
    int64_t n_mtp = 0;
    std::map<std::string, std::string> names;
    for (const auto & [orig, t] : d->st) {
        std::string n = str_replace_all(orig, "language_model.", "");
        if (str_starts_with(n, "model.mtp.")) {
            n = n.substr(strlen("model."));
        }
        if (str_starts_with(n, "mtp.")) {
            n = exl3_mtp_name(n, n_layer, n_mtp, is_qwen4exp);
        }
        if (!names.emplace(n, orig).second) {
            throw std::runtime_error("tensors " + orig + " and " + names.at(n) + " have the same name " + n);
        }
    }

    int n_vision = 0;
    std::map<std::string, std::map<std::string, std::string>> modules;
    std::map<std::string, std::string> plain;
    for (const auto & [n, orig] : names) {
        // the vision tower is for an mmproj, not for the text model
        if (str_starts_with(n, "model.visual.") || str_starts_with(n, "visual.")) {
            n_vision++;
            continue;
        }
        const size_t dot  = n.rfind('.');
        const std::string part = dot == std::string::npos ? "" : n.substr(dot + 1);
        if (part == "trellis" || part == "suh" || part == "svh" || part == "mcg" || part == "mul1" || part == "su" || part == "sv" || part == "signs") {
            modules[n.substr(0, dot)][part] = n;
        } else {
            plain[n] = orig;
        }
    }
    for (auto it = modules.begin(); it != modules.end(); ) {
        if (it->second.count("trellis") == 0) {
            for (const auto & [part, n] : it->second) {
                plain[n] = names.at(n);
            }
            it = modules.erase(it);
        } else {
            ++it;
        }
    }
    if (modules.empty()) {
        throw std::runtime_error("quant_method is exl3 but the checkpoint has no .trellis tensors");
    }

    const int64_t n_k_heads = exl3_get_int(hp, "linear_num_key_heads");
    const int64_t n_v_heads = exl3_get_int(hp, "linear_num_value_heads");
    const int64_t head_k    = exl3_get_int(hp, "linear_key_head_dim");
    const int64_t head_v    = exl3_get_int(hp, "linear_value_head_dim");
    const bool    reorder   = n_k_heads > 0 && n_v_heads > 0 && n_k_heads != n_v_heads;
    if (reorder && n_v_heads % n_k_heads != 0) {
        throw std::runtime_error("config.json: linear_num_value_heads is not a multiple of linear_num_key_heads");
    }
    const int64_t n_v_per_k = reorder ? n_v_heads / n_k_heads : 1;
    const int64_t qk_dim    = 2*head_k*n_k_heads;
    const int64_t v_dim     = n_v_heads*head_v;

    // tie_word_embeddings may be set while lm_head is stored (the 4B EXL3 file): a stored head is always loaded
    const bool tie      = hp.value("tie_word_embeddings", false);
    const bool has_head = modules.count("lm_head") > 0 || plain.count("lm_head.weight") > 0;
    if (!has_head && !tie) {
        throw std::runtime_error("config.json does not tie the output head, but there is no lm_head");
    }
    if (has_head && tie) {
        LOG_INF("%s: tie_word_embeddings is set, but lm_head is stored: the stored head is used\n", __func__);
    }
    if (n_mtp != hp.value("mtp_num_hidden_layers", (long long) 0)) {
        LOG_INF("%s: config.json declares %lld MTP layers, the checkpoint has %" PRId64 "\n",
                __func__, hp.value("mtp_num_hidden_layers", (long long) 0), n_mtp);
    }
    if (n_vision > 0) {
        LOG_INF("%s: %d vision tensors are not loaded (text model)\n", __func__, n_vision);
    }

    auto read_raw = [&](const std::string & orig, void * dst) {
        const st_tensor & s = d->st.at(orig);
        if (d->stream_files.empty()) {
            exl3_read_exact(d->files[s.shard], s.offs, dst, s.size, orig.c_str());
        } else if (!common_download_stream_read_range(path, s.shard, s.offs, s.size, dst)) {
            throw std::runtime_error("cannot fetch EXL3 constant " + orig);
        }
    };
    auto read_u64_const = [&](const std::string & suffix) -> std::vector<uint64_t> {
        for (auto it = plain.begin(); it != plain.end(); ++it) {
            if (str_ends_with(it->first, suffix)) {
                const st_tensor & s = d->st.at(it->second);
                if (s.dtype != "I64") {
                    throw std::runtime_error(it->second + ": expected I64 PLE constant");
                }
                const size_t n_vals = (size_t) (s.size / 8);
                std::vector<int64_t> tmp(n_vals);
                read_raw(it->second, tmp.data());
                std::vector<uint64_t> out(n_vals);
                for (size_t i = 0; i < n_vals; i++) {
                    if (tmp[i] < 0) {
                        throw std::runtime_error(it->second + ": negative PLE constant");
                    }
                    out[i] = (uint64_t) tmp[i];
                }
                plain.erase(it);
                return out;
            }
        }
        return {};
    };

    std::vector<uint64_t> ple_multipliers;
    std::vector<uint64_t> ple_offsets;
    std::vector<uint64_t> ple_vocab_sizes;
    if (is_qwen4exp) {
        ple_multipliers = read_u64_const("ple_embedding.ngram_embedding.layer_multipliers");
        if (ple_multipliers.empty()) { ple_multipliers = read_u64_const("ple_embedding.layer_multipliers"); }
        ple_offsets     = read_u64_const("ple_embedding.ngram_embedding.head_offsets");
        if (ple_offsets.empty())     { ple_offsets     = read_u64_const("ple_embedding.ngram_heads_offsets"); }
        ple_vocab_sizes = read_u64_const("ple_embedding.ngram_embedding.head_vocab_sizes");
        if (ple_vocab_sizes.empty()) { ple_vocab_sizes = read_u64_const("ple_embedding.ngram_heads_vocab_sizes"); }
    }

    auto gguf_base = [&](const std::string & base) -> std::string {
        if (base == "lm_head") {
            return "output";
        }
        int64_t il;
        std::string local;
        if (exl3_split_layer(base, il, local) && il < n_layer + n_mtp) {
            for (const auto & e : QWEN35_LINEARS) {
                if (local == e.hf) {
                    return "blk." + std::to_string(il) + "." + e.gguf;
                }
            }
            if (is_qwen4exp) {
                for (const auto & e : QWEN4EXP_LINEARS) {
                    if (local == e.hf) {
                        return "blk." + std::to_string(il) + "." + e.gguf;
                    }
                }
                if (local == "mlp.experts.gate_proj") { return "blk." + std::to_string(il) + ".ffn_gate_exps"; }
                if (local == "mlp.experts.up_proj")   { return "blk." + std::to_string(il) + ".ffn_up_exps";   }
                if (local == "mlp.experts.down_proj") { return "blk." + std::to_string(il) + ".ffn_down_exps"; }
            }
            if (il >= n_layer) {
                for (const auto & e : QWEN35_NEXTN) {
                    if (local == e.hf) {
                        return "blk." + std::to_string(il) + "." + e.gguf;
                    }
                }
            }
        }
        return "";
    };

    auto check_codebook = [&](const std::string & base, const std::map<std::string, std::string> & parts, std::string & codebook) -> std::string {
        if (parts.count("su") || parts.count("sv")) {
            throw std::runtime_error(base + ": legacy packed su/sv EXL3 tensors are not supported");
        }
        if (parts.count("mcg") && parts.count("mul1")) {
            throw std::runtime_error(base + ": both .mcg and .mul1 codebook markers are present");
        }
        if (!parts.count("suh") || !parts.count("svh")) {
            throw std::runtime_error(base + ": EXL3 linear without .suh/.svh");
        }
        const std::string cb = parts.count("mcg") ? "mcg" : parts.count("mul1") ? "mul1" : "3inst";
        if (cb != "3inst") {
            const st_tensor & m = d->st.at(names.at(parts.at(cb)));
            if (m.dtype != "I32" || !m.shape.empty()) {
                throw std::runtime_error(base + ": malformed ." + cb + " marker");
            }
            const uint32_t expected = cb == "mcg" ? EXL3_MARKER_MCG : EXL3_MARKER_MUL1;
            if (!d->stream_files.empty()) {
                d->stream_markers.emplace_back(&m, expected);
            } else {
                uint8_t b[4];
                read_raw(names.at(parts.at(cb)), b);
                const uint32_t marker = (uint32_t) b[0] | (uint32_t) b[1] << 8 | (uint32_t) b[2] << 16 | (uint32_t) b[3] << 24;
                if (marker != expected) {
                    throw std::runtime_error(string_format("%s: .%s marker holds 0x%08x", base.c_str(), cb.c_str(), marker));
                }
            }
        }
        if (!codebook.empty() && codebook != cb) {
            throw std::runtime_error("EXL3 checkpoints that mix codebooks are not supported");
        }
        codebook = cb;
        return cb;
    };

    std::string codebook;
    std::vector<std::string> permuted;
    bool rows = false;
    int64_t ple_row_dim = 0;
    size_t n_exl3_linears = 0;
    size_t n_row_tables = 0;

    // stack per-expert EXL3 linears (<prefix>.experts.<e>.<w>) matching conversion/exl3.py
    if (is_qwen4exp) {
        std::map<std::string, std::map<int64_t, std::map<std::string, std::string>>> expert_stacks;
        for (auto it = modules.begin(); it != modules.end(); ) {
            std::string merged;
            int64_t eid = -1;
            if (exl3_split_expert(it->first, merged, eid)) {
                expert_stacks[merged][eid] = std::move(it->second);
                it = modules.erase(it);
            } else {
                ++it;
            }
        }
        const common_json * v_exp = exl3_find(hp, { "num_local_experts", "num_experts", "n_routed_experts" });
        const int64_t cfg_experts = v_exp ? v_exp->get<long long>() : -1;
        for (const auto & [merged, experts] : expert_stacks) {
            if (experts.empty() || experts.begin()->first != 0 || experts.rbegin()->first != (int64_t) experts.size() - 1) {
                throw std::runtime_error(merged + ": experts are not numbered sequentially from 0");
            }
            if (cfg_experts >= 0 && (int64_t) experts.size() != cfg_experts) {
                throw std::runtime_error(string_format("%s: %zu experts stored, config.json has %" PRId64,
                        merged.c_str(), experts.size(), cfg_experts));
            }
            const std::string gb = gguf_base(merged);
            if (gb.empty()) {
                throw std::runtime_error("EXL3 experts " + merged + " are not known to the qwen4exp importer");
            }
            std::vector<int64_t> tr_shape, su_shape, sv_shape;
            std::vector<std::string> src_tr, src_su, src_sv;
            for (const auto & [eid, parts] : experts) {
                const std::string cb = check_codebook(merged, parts, codebook);
                const std::string & n_tr = names.at(parts.at("trellis"));
                const std::string & n_su = names.at(parts.at("suh"));
                const std::string & n_sv = names.at(parts.at("svh"));
                const st_tensor & tr = d->st.at(n_tr);
                const st_tensor & su = d->st.at(n_su);
                const st_tensor & sv = d->st.at(n_sv);
                if (tr.dtype != "I16" || su.dtype != "F16" || sv.dtype != "F16" || tr.shape.size() != 3 ||
                        su.shape.size() != 1 || sv.shape.size() != 1) {
                    throw std::runtime_error(merged + ": unexpected expert dtypes or shapes");
                }
                if (tr_shape.empty()) {
                    tr_shape = tr.shape; su_shape = su.shape; sv_shape = sv.shape;
                    const int64_t k = tr_shape[0]*16, n = tr_shape[1]*16, tw = tr_shape[2];
                    const bool rate_ok = (tw % 16 == 0 && tw >= 16 && tw <= 128) || (cb == "mul1" && (tw == 24 || tw == 40 || tw == 56));
                    if (!rate_ok || k % 128 || n % 128 || su_shape[0] != k || sv_shape[0] != n) {
                        throw std::runtime_error(merged + ": unsupported EXL3 rate or inconsistent shapes");
                    }
                } else if (tr.shape != tr_shape || su.shape != su_shape || sv.shape != sv_shape) {
                    throw std::runtime_error(merged + ": experts of different shapes or rates cannot be stacked");
                }
                src_tr.push_back(n_tr);
                src_su.push_back(n_su);
                src_sv.push_back(n_sv);
            }
            const int64_t n_ex = (int64_t) experts.size();
            exl3_out_tensor out_tr = { gb + ".exl3_trellis", "", std::move(src_tr), GGML_TYPE_I16, { n_ex, tr_shape[0], tr_shape[1], tr_shape[2] } };
            exl3_out_tensor out_su = { gb + ".exl3_suh",     "", std::move(src_su), GGML_TYPE_F16, { n_ex, su_shape[0] } };
            exl3_out_tensor out_sv = { gb + ".exl3_svh",     "", std::move(src_sv), GGML_TYPE_F16, { n_ex, sv_shape[0] } };
            d->out.push_back(std::move(out_tr));
            d->out.push_back(std::move(out_su));
            d->out.push_back(std::move(out_sv));
            n_exl3_linears++;
        }
    }

    // EXL3 linears and row tables, in the order of conversion/exl3.py
    for (const auto & [base, parts] : modules) {
        // n-gram PLE row table (exllamav3 ngram_codec.py): rows of one ring of 160 + head_bias
        if (is_qwen4exp && parts.size() == 1 && plain.count(base + ".head_bias")) {
            const std::string & orig_tr   = names.at(parts.at("trellis"));
            const std::string & orig_bias = plain.at(base + ".head_bias");
            const st_tensor & tr   = d->st.at(orig_tr);
            const st_tensor & bias = d->st.at(orig_bias);
            if (!str_ends_with(base, ".ple_embedding.ngram_embedding")) {
                throw std::runtime_error(base + ": unknown n-gram row table");
            }
            if (tr.dtype != "I16" || tr.shape.size() != 2 || bias.dtype != "F16" || bias.shape.size() != 2 || bias.shape[1] != 160) {
                throw std::runtime_error(base + ": unexpected EXL3 n-gram table dtypes or shapes");
            }
            const int64_t rw = tr.shape[1];
            const int64_t K  = (rw - 1) * 16 / 160;
            if (K < 1 || K > 8 || 1 + 160 * K / 16 != rw) {
                throw std::runtime_error(base + ": row width does not hold one ring of 160");
            }
            d->out.push_back({ "per_layer_token_embd.exl3r_trellis", orig_tr,   {}, GGML_TYPE_I16, tr.shape });
            d->out.push_back({ "per_layer_token_embd.exl3r_bias",    orig_bias, {}, GGML_TYPE_F16, bias.shape });
            plain.erase(base + ".head_bias");
            ple_row_dim = 160;
            rows = true;
            n_row_tables++;
            continue;
        }
        // a 2-D trellis with signs: a quantized token embedding (ngram_codec.py), rows of rings of 256, always mul1
        if (parts.count("signs")) {
            const st_tensor & tr = d->st.at(names.at(parts.at("trellis")));
            const st_tensor & sg = d->st.at(names.at(parts.at("signs")));
            const int64_t hidden = exl3_get_int(hp, "hidden_size");
            if (parts.size() != 2 || base != "model.embed_tokens") {
                throw std::runtime_error(base + ": only the token embedding can be an EXL3 row table");
            }
            if (tr.dtype != "I16" || tr.shape.size() != 2 || sg.dtype != "F16" || sg.shape.size() != 2 || sg.shape[1] != 256) {
                throw std::runtime_error(base + ": unexpected EXL3 row table dtypes or shapes");
            }
            const int64_t groups = sg.shape[0];
            const int64_t rw     = tr.shape[1] % groups == 0 ? tr.shape[1]/groups - 1 : 0;
            if (rw < 16 || rw > 128 || rw % 16 != 0 || groups*256 < hidden || (groups - 1)*256 >= hidden) {
                throw std::runtime_error(base + ": the row width does not hold the rings of 256 for the hidden size");
            }
            d->out.push_back({ "token_embd.exl3r_trellis", names.at(parts.at("trellis")), {}, GGML_TYPE_I16, tr.shape });
            d->out.push_back({ "token_embd.exl3r_signs",   names.at(parts.at("signs")),   {}, GGML_TYPE_F16, sg.shape });
            rows = true;
            n_row_tables++;
            continue;
        }
        const std::string cb = check_codebook(base, parts, codebook);

        const std::string & n_tr = names.at(parts.at("trellis"));
        const std::string & n_su = names.at(parts.at("suh"));
        const std::string & n_sv = names.at(parts.at("svh"));
        const st_tensor & tr = d->st.at(n_tr);
        const st_tensor & su = d->st.at(n_su);
        const st_tensor & sv = d->st.at(n_sv);
        if (tr.shape.size() != 3) {
            throw std::runtime_error(base + ": a " + std::to_string(tr.shape.size()) + "-D trellis (row-ring embedding or n-gram table) is not supported yet");
        }
        if (tr.dtype != "I16" || su.dtype != "F16" || sv.dtype != "F16") {
            throw std::runtime_error(base + ": unexpected EXL3 dtypes");
        }
        const int64_t k  = tr.shape[0]*16;
        const int64_t n  = tr.shape[1]*16;
        const int64_t tw = tr.shape[2];
        const bool rate_ok = (tw % 16 == 0 && tw >= 16 && tw <= 128) || (cb == "mul1" && (tw == 24 || tw == 40 || tw == 56));
        if (!rate_ok || k % 128 || n % 128 || su.shape != std::vector<int64_t>{ k } || sv.shape != std::vector<int64_t>{ n }) {
            throw std::runtime_error(base + ": unsupported EXL3 rate or inconsistent shapes");
        }

        // Qwen4Exp fused QSA indexer projection: split columns [0, n_q) -> indexer.q_proj, [n_q, n) -> indexer.k_proj
        if (is_qwen4exp && str_ends_with(base, ".self_attn.indexer.index_qk_proj")) {
            int64_t il = -1;
            std::string local;
            if (!exl3_split_layer(base, il, local) || il >= n_layer + n_mtp) {
                throw std::runtime_error("EXL3 linear " + base + " is out of range");
            }
            const int64_t idx_dim = exl3_get_int(hp, "indexer_head_dim");
            const int64_t n_q     = exl3_get_int(hp, "indexer_n_heads") * idx_dim;
            if (n != n_q + idx_dim || n_q % 128 != 0 || idx_dim % 128 != 0) {
                throw std::runtime_error(base + ": indexer_qk split does not cut columns at 128-blocks");
            }
            const std::string pfx = "blk." + std::to_string(il) + ".indexer.";
            const std::pair<const char *, std::pair<int64_t, int64_t>> halves[2] = {
                { "q_proj", { 0, n_q } },
                { "k_proj", { n_q, idx_dim } },
            };
            for (const auto & [proj, range] : halves) {
                const int64_t n0 = range.first;
                const int64_t nl = range.second;
                exl3_out_tensor t_tr = { pfx + proj + ".exl3_trellis", n_tr, {}, GGML_TYPE_I16, { tr.shape[0], nl/16, tr.shape[2] } };
                t_tr.slice_axis = 1; t_tr.slice_start = n0/16; t_tr.slice_len = nl/16;
                exl3_out_tensor t_su = { pfx + proj + ".exl3_suh",     n_su, {}, GGML_TYPE_F16, su.shape };
                exl3_out_tensor t_sv = { pfx + proj + ".exl3_svh",     n_sv, {}, GGML_TYPE_F16, { nl } };
                t_sv.slice_axis = 0; t_sv.slice_start = n0;    t_sv.slice_len = nl;
                d->out.push_back(std::move(t_tr));
                d->out.push_back(std::move(t_su));
                d->out.push_back(std::move(t_sv));
                n_exl3_linears++;
            }
            continue;
        }

        const std::string gb = gguf_base(base);
        if (gb.empty()) {
            throw std::runtime_error(string_format("EXL3 linear %s is not known to the %s importer",
                    base.c_str(), is_qwen4exp ? "qwen4exp" : "qwen35"));
        }

        exl3_out_tensor t_tr = { gb + ".exl3_trellis", n_tr, {}, GGML_TYPE_I16, tr.shape };
        exl3_out_tensor t_su = { gb + ".exl3_suh",     n_su, {}, GGML_TYPE_F16, su.shape };
        exl3_out_tensor t_sv = { gb + ".exl3_svh",     n_sv, {}, GGML_TYPE_F16, sv.shape };
        // the V head reorder of qwen35/qwen4exp on whole 128-blocks, a lossless reorder of tiles (H is block diagonal)
        if (reorder && base.find("linear_attn.") != std::string::npos) {
            if (head_v % 128 || qk_dim % 128) {
                throw std::runtime_error(base + ": V heads do not fill whole EXL3 128-blocks");
            }
            int     axis  = -1;
            int64_t start = 0;
            if (str_ends_with(base, ".in_proj_qkv") && n == qk_dim + v_dim) {
                axis = 1; start = qk_dim;
            } else if (str_ends_with(base, ".in_proj_z") && n == v_dim) {
                axis = 1;
            } else if (str_ends_with(base, ".out_proj") && k == v_dim) {
                axis = 0;
            } else {
                throw std::runtime_error(base + ": no EXL3 V head reorder for this projection");
            }
            t_tr.perm_axis = axis;
            t_tr.perm_start = start/16;
            t_tr.perm_unit  = head_v/16;
            t_tr.perm_k     = n_k_heads;
            t_tr.perm_v     = n_v_per_k;
            exl3_out_tensor & t_s = axis == 0 ? t_su : t_sv;
            t_s.perm_axis  = 0;
            t_s.perm_start = start;
            t_s.perm_unit  = head_v;
            t_s.perm_k     = n_k_heads;
            t_s.perm_v     = n_v_per_k;
            permuted.push_back(gb);
        }
        d->out.push_back(std::move(t_tr));
        d->out.push_back(std::move(t_su));
        d->out.push_back(std::move(t_sv));
        n_exl3_linears++;
    }

    // the other tensors, with the transforms of Qwen3NextModel / Qwen4ExpTextModel and _LinearAttentionVReorderBase.modify_tensors
    for (const auto & [n, orig] : plain) {
        const st_tensor & s = d->st.at(orig);
        exl3_out_tensor t;
        t.src   = orig;
        t.shape = s.shape;
        int64_t il = -1;
        std::string local;
        if (n == "model.embed_tokens.weight") {
            t.name = "token_embd.weight";
        } else if (n == "model.norm.weight") {
            t.name = "output_norm.weight";
        } else if (n == "lm_head.weight") {
            t.name = "output.weight";
        } else if (is_qwen4exp && n == "model.hyper_connection_mixer.hc_norm.weight") {
            t.name = "output_hc_norm.weight";
        } else if (is_qwen4exp && n == "model.hyper_connection_mixer.input_mix_weight_down.weight") {
            t.name = "output_hc_down.weight";
        } else if (is_qwen4exp && n == "model.hyper_connection_mixer.input_mix_weight_up.weight") {
            t.name = "output_hc_up.weight";
        } else if (exl3_split_layer(n, il, local) && il < n_layer + n_mtp) {
            if (is_qwen4exp && local == "self_attn.indexer.index_qk_proj.weight") {
                const int64_t idx_dim = exl3_get_int(hp, "indexer_head_dim");
                const int64_t n_q     = exl3_get_int(hp, "indexer_n_heads") * idx_dim;
                if (s.shape.size() != 2 || s.shape[0] != n_q + idx_dim) {
                    throw std::runtime_error(orig + ": unexpected dense indexer_qk shape");
                }
                const ggml_type dt = (s.dtype == "F16" || s.dtype == "BF16") ? st_dtype_ggml(s.dtype) : GGML_TYPE_F32;
                const std::string pfx = "blk." + std::to_string(il) + ".indexer.";
                exl3_out_tensor tq = { pfx + "q_proj.weight", orig, {}, dt, { n_q, s.shape[1] } };
                tq.slice_axis = 0; tq.slice_start = 0; tq.slice_len = n_q;
                exl3_out_tensor tk = { pfx + "k_proj.weight", orig, {}, dt, { idx_dim, s.shape[1] } };
                tk.slice_axis = 0; tk.slice_start = n_q; tk.slice_len = idx_dim;
                d->out.push_back(std::move(tq));
                d->out.push_back(std::move(tk));
                continue;
            }
            for (const auto & e : QWEN35_LAYER_TENSORS) {
                if (local == e.hf) {
                    t.name = "blk." + std::to_string(il) + "." + e.gguf;
                }
            }
            for (const auto & e : QWEN35_LINEARS) {
                if (local == std::string(e.hf) + ".weight") {
                    t.name = "blk." + std::to_string(il) + "." + e.gguf + ".weight";
                }
            }
            if (is_qwen4exp) {
                for (const auto & e : QWEN4EXP_LAYER_TENSORS) {
                    if (local == e.hf) {
                        t.name = "blk." + std::to_string(il) + "." + e.gguf;
                    }
                }
                for (const auto & e : QWEN4EXP_LINEARS) {
                    if (local == std::string(e.hf) + ".weight") {
                        t.name = "blk." + std::to_string(il) + "." + e.gguf + ".weight";
                    }
                }
            }
            if (il >= n_layer) {
                for (const auto & e : QWEN35_NEXTN) {
                    if (local == e.hf || local == std::string(e.hf) + ".weight") {
                        t.name = "blk." + std::to_string(il) + "." + e.gguf + (str_ends_with(local, ".weight") && !str_ends_with(e.gguf, ".weight") ? ".weight" : "");
                    }
                }
            }
        }
        if (t.name.empty()) {
            throw std::runtime_error(string_format("tensor %s is not known to the %s importer",
                    orig.c_str(), is_qwen4exp ? "qwen4exp" : "qwen35"));
        }

        if (local == "linear_attn.conv1d.weight" || local == "ple.conv1d.weight") {
            if (t.shape.size() != 3 || t.shape[1] != 1 || t.shape[2] <= 1) {
                throw std::runtime_error(orig + ": unexpected conv1d shape");
            }
            t.shape = { t.shape[0], t.shape[2] };
        }
        if (reorder && str_starts_with(local, "linear_attn.")) {
            auto set_perm = [&](int axis, int64_t start, int64_t unit) {
                if (t.shape.size() <= (size_t) axis || t.shape[axis] - start != n_v_heads*unit) {
                    throw std::runtime_error(orig + ": shape does not match the linear attention heads");
                }
                t.perm_axis  = axis;
                t.perm_start = start;
                t.perm_unit  = unit;
                t.perm_k     = n_k_heads;
                t.perm_v     = n_v_per_k;
            };
            if (local == "linear_attn.in_proj_qkv.weight") {
                set_perm(0, qk_dim, head_v);
            } else if (local == "linear_attn.in_proj_z.weight") {
                set_perm(0, 0, head_v);
            } else if (local == "linear_attn.in_proj_a.weight" || local == "linear_attn.in_proj_b.weight") {
                set_perm(0, 0, 1);
            } else if (local == "linear_attn.A_log" || local == "linear_attn.dt_bias") {
                if (t.shape.size() != 1) {
                    throw std::runtime_error(orig + ": expected a 1-D tensor");
                }
                set_perm(0, 0, 1);
            } else if (local == "linear_attn.conv1d.weight") {
                set_perm(0, qk_dim, head_v);
            } else if (local == "linear_attn.out_proj.weight") {
                set_perm(1, 0, head_v);
            }
        }
        if (local == "linear_attn.A_log") {
            t.arith = 2;
        } else if ((str_ends_with(n, "norm.weight") && !str_ends_with(n, "linear_attn.norm.weight")) ||
                (is_qwen4exp && (str_ends_with(n, ".ple.norm_key.weight") || str_ends_with(n, ".ple.norm_query.weight") ||
                str_ends_with(n, ".ple.norm_conv.weight") || str_ends_with(n, ".indexer.q_layernorm.weight") ||
                str_ends_with(n, ".indexer.k_layernorm.weight")))) {
            t.arith = 1;
        }

        // the type rule of ModelBase.prepare_tensors for EXL3 checkpoints
        const bool req_f32 = t.shape.size() <= 1 || str_ends_with(t.name, "_norm.weight") ||
            str_ends_with(t.name, ".ssm_conv1d.weight") || str_ends_with(t.name, ".ffn_gate_inp.weight") ||
            str_ends_with(t.name, ".ffn_gate_inp_shexp.weight") || !str_ends_with(t.name, ".weight");
        if (req_f32) {
            if (s.dtype != "F32" && s.dtype != "F16" && s.dtype != "BF16") {
                throw std::runtime_error(orig + ": unexpected dtype " + s.dtype);
            }
            t.type = GGML_TYPE_F32;
        } else if (s.dtype == "F16" || s.dtype == "BF16") {
            t.type = st_dtype_ggml(s.dtype);
        } else {
            throw std::runtime_error(orig + ": " + s.dtype + " matrices are not supported by the importer, convert the model with convert_hf_to_gguf.py");
        }
        if (t.arith && t.type != GGML_TYPE_F32) {
            throw std::runtime_error(orig + ": unexpected shape");
        }
        d->out.push_back(std::move(t));
    }

    // general.file_type: the type of the first float matrix, as the converter guesses the output type
    uint32_t ftype = 1; // MOSTLY_F16
    for (const auto & [n, orig] : names) {
        if (str_starts_with(n, "model.visual.") || str_starts_with(n, "visual.")) {
            continue;
        }
        const st_tensor & s = d->st.at(orig);
        if (s.shape.size() >= 2 && (s.dtype == "BF16" || s.dtype == "F16")) {
            ftype = s.dtype == "BF16" ? 32 : 1; // MOSTLY_BF16 or MOSTLY_F16
            break;
        }
    }

    d->meta = gguf_init_empty();
    gguf_set_val_bool(d->meta, "llama.tensor_set_complete", true);
    exl3_emit_hparams(d->meta, is_qwen4exp, hp, config,
        dir.filename().string().empty() ? dir.parent_path().filename().string() : dir.filename().string(),
        ftype, n_mtp, ple_row_dim, ple_multipliers, ple_offsets, ple_vocab_sizes);

    gguf_set_val_u32(d->meta, "exl3.version", 1);
    gguf_set_val_str(d->meta, "exl3.codebook", codebook.c_str());
    gguf_set_val_u32(d->meta, "exl3.hadamard_size", 128);
    if (rows) {
        gguf_set_val_u32(d->meta, "exl3.rows.version", 1);
    }
    if (!permuted.empty()) {
        std::vector<const char *> ptrs;
        for (const auto & p : permuted) {
            ptrs.push_back(p.c_str());
        }
        gguf_set_val_str(d->meta, "exl3.permutation", EXL3_PERMUTATION);
        gguf_set_arr_str(d->meta, "exl3.permuted_tensors", ptrs.data(), ptrs.size());
    }
    gguf_set_val_str(d->meta, "exl3.source.quantization_config", exl3_py_json(qc).c_str());

    exl3_set_vocab(d->meta, dir, config, hp);

    ggml_init_params params = { ggml_tensor_overhead()*d->out.size(), NULL, true };
    d->ctx_meta = ggml_init(params);
    for (size_t i = 0; i < d->out.size(); i++) {
        const exl3_out_tensor & t = d->out[i];
        int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
        for (size_t j = 0; j < t.shape.size(); j++) {
            ne[j] = t.shape[t.shape.size() - 1 - j];
        }
        ggml_tensor * cur = ggml_new_tensor(d->ctx_meta, t.type, std::max<int>(1, (int) t.shape.size()), ne);
        ggml_set_name(cur, t.name.c_str());
        if (!d->out_idx.emplace(t.name, i).second) {
            throw std::runtime_error("two tensors map to " + t.name);
        }
        gguf_add_tensor(d->meta, cur);
    }

    LOG_INF("%s: %s: %zu tensors (%zu EXL3 linears, codebook %s, %zu block-permuted, %zu row tables, %" PRId64 " MTP layers) from %zu safetensors file(s)\n",
        __func__, path.c_str(), d->out.size(), n_exl3_linears, codebook.c_str(), permuted.size(), n_row_tables, n_mtp, d->shards.size());
    return d;
}

const gguf_context * common_exl3_dir_gguf(const common_exl3_dir * d) {
    return d->meta;
}

static bool exl3_needs_transform(const exl3_out_tensor & t, const st_tensor & s) {
    return t.slice_axis >= 0 || t.perm_axis >= 0 || t.arith != 0 ||
           (t.type == GGML_TYPE_F32 && s.dtype != "F32");
}

static void exl3_transform(const exl3_out_tensor & t, const st_tensor & s, const void * src, void * dst, size_t ne) {
    const bool copy = t.type != GGML_TYPE_F32 || s.dtype == "F32";
    const size_t st_es = st_dtype_size(s.dtype);
    std::vector<uint8_t> raw;
    const auto * data = static_cast<const uint8_t *>(src);
    if (t.slice_axis >= 0) {
        raw.resize((size_t) ne*st_es);
        exl3_slice_copy(data, raw.data(), s.shape, st_es, t);
        data = raw.data();
    }
    if (t.perm_axis >= 0) {
        std::vector<uint8_t> tmp((size_t) ne*st_es);
        exl3_permute(data, tmp.data(), t.shape, st_es, t);
        raw.swap(tmp);
        data = raw.data();
    }
    if (copy && src != dst) {
        memcpy(dst, data, (size_t) ne*st_es);
    }
    if (t.type != GGML_TYPE_F32) {
        return;
    }

    float * y = (float *) dst;
    if (!copy) {
        for (int64_t i = 0; i < ne; i++) {
            uint16_t h;
            memcpy(&h, data + 2*i, 2);
            if (s.dtype == "BF16") {
                const uint32_t u = (uint32_t) h << 16;
                memcpy(&y[i], &u, 4);
            } else {
                y[i] = ggml_fp16_to_fp32(h);
            }
        }
    }
    if (t.arith == 1) {
        for (int64_t i = 0; i < ne; i++) {
            y[i] = y[i] + 1.0f;
        }
    } else if (t.arith == 2) {
        // one rounding from float64, as the converter does for EXL3 checkpoints
        for (int64_t i = 0; i < ne; i++) {
            y[i] = (float) -std::exp((double) y[i]);
        }
    }
}

void common_exl3_dir_read(const common_exl3_dir * d, const char * name, void * dst, size_t size) {
    const auto it = d->out_idx.find(name);
    if (it == d->out_idx.end()) {
        throw std::runtime_error(string_format("tensor %s is not in the import", name));
    }
    const exl3_out_tensor & t = d->out[it->second];
    int64_t ne = 1;
    for (const int64_t x : t.shape) {
        ne *= x;
    }
    if (size != (size_t) ne*ggml_type_size(t.type)) {
        throw std::runtime_error(string_format("tensor %s: %zu bytes asked, it has %zu", name, size, (size_t) ne*ggml_type_size(t.type)));
    }

    if (!t.srcs.empty()) {
        uint8_t * p = (uint8_t *) dst;
        for (const std::string & sname : t.srcs) {
            const st_tensor & s = d->st.at(sname);
            if ((uint64_t) (p - (uint8_t *) dst) + s.size > size) {
                throw std::runtime_error(string_format("tensor %s: stacked shard size overflow", name));
            }
            exl3_read_exact(d->files[s.shard], s.offs, p, s.size, sname.c_str());
            p += s.size;
        }
        if ((size_t) (p - (uint8_t *) dst) != size) {
            throw std::runtime_error(string_format("tensor %s: stacked size mismatch", name));
        }
        return;
    }

    const st_tensor & s = d->st.at(t.src);
    auto & f = d->files[s.shard];
    if (!exl3_needs_transform(t, s)) {
        exl3_read_exact(f, s.offs, dst, s.size, t.src.c_str());
        return;
    }
    if (t.slice_axis < 0 && t.perm_axis < 0 && s.dtype == "F32" && t.type == GGML_TYPE_F32) {
        exl3_read_exact(f, s.offs, dst, s.size, t.src.c_str());
        exl3_transform(t, s, dst, dst, ne);
        return;
    }
    std::vector<uint8_t> raw(s.size);
    exl3_read_exact(f, s.offs, raw.data(), s.size, t.src.c_str());
    exl3_transform(t, s, raw.data(), dst, ne);
}

static void exl3_set_tensor_data(ggml_tensor * t, void * ud) {
    const auto * d = (const common_exl3_dir *) ud;
    const size_t n = ggml_nbytes(t);
    if (t->buffer && ggml_backend_buffer_is_host(t->buffer)) {
        common_exl3_dir_read(d, ggml_get_name(t), t->data, n);
        return;
    }
    std::vector<uint8_t> buf(n);
    common_exl3_dir_read(d, ggml_get_name(t), buf.data(), n);
    ggml_backend_tensor_set(t, buf.data(), 0, n);
}

static bool exl3_register_tensor(ggml_tensor * dst, const llama_model_stream_target_sink * sink, void * ctx, void * ud) {
    auto & d = *static_cast<common_exl3_dir *>(ud);
    auto & t = d.out.at(d.out_idx.at(ggml_get_name(dst)));
    if (t.srcs.empty()) {
        const auto & s = d.st.at(t.src);
        llama_model_stream_target_desc desc{static_cast<uint16_t>(s.shard), s.offs, s.size, 0, nullptr, nullptr};
        if (exl3_needs_transform(t, s)) {
            t.stream_src = &s;
            desc.transform_user_data = &t;
            desc.transform = [](const void * src, size_t src_bytes, void * out, size_t out_bytes, void * data) {
                const auto & t = *static_cast<const exl3_out_tensor *>(data);
                const auto & s = *t.stream_src;
                if (src_bytes != s.size || !out) {
                    return false;
                }
                exl3_transform(t, s, src, out, out_bytes / ggml_type_size(t.type));
                return true;
            };
        }
        return sink->add_target(ctx, dst, &desc);
    }
    size_t offset = 0;
    for (const auto & name : t.srcs) {
        const auto & s = d.st.at(name);
        const llama_model_stream_target_desc desc{static_cast<uint16_t>(s.shard), s.offs, s.size, offset, nullptr, nullptr};
        if (s.size > ggml_nbytes(dst) - offset || !sink->add_target(ctx, dst, &desc)) {
            return false;
        }
        offset += s.size;
    }
    return offset == ggml_nbytes(dst);
}

static bool exl3_register_markers(const llama_model_stream_target_sink * sink, void * ctx, void * ud) {
    auto & d = *static_cast<common_exl3_dir *>(ud);
    for (auto & [s, expected] : d.stream_markers) {
        const llama_model_stream_target_desc desc{
            static_cast<uint16_t>(s->shard), s->offs, s->size, 0,
            [](const void * src, size_t n, void *, size_t, void * ud) {
                if (n != 4) {
                    return false;
                }
                const auto * b = static_cast<const uint8_t *>(src);
                return ((uint32_t) b[0] | (uint32_t) b[1] << 8 | (uint32_t) b[2] << 16 | (uint32_t) b[3] << 24) ==
                    *static_cast<const uint32_t *>(ud);
            }, &expected,
        };
        if (!sink->add_target(ctx, nullptr, &desc)) {
            return false;
        }
    }
    return true;
}

llama_model * common_exl3_load_model(const std::string & path, llama_model_params params) {
    if (!common_exl3_is_dir(path)) {
        return llama_model_load_from_file(path.c_str(), params);
    }
    try {
        const common_exl3_dir_ptr d = common_exl3_dir_open(path);
        if (!d->stream_files.empty()) {
            params.stream_loader = common_download_stream_loader;
            params.stream_loader_user_data = common_download_get_stream_session_ud(path);
            if (!params.stream_loader_user_data) {
                throw std::runtime_error("EXL3 download session disappeared during metadata load");
            }
        }
        if (params.stream_loader && d->shard_sizes.size() > UINT16_MAX) {
            throw std::runtime_error("too many EXL3 shards for streaming");
        }
        const llama_model_stream_source_desc source{
            static_cast<uint16_t>(d->shard_sizes.size()), d->shard_sizes.data(), d->shard_data_offsets.data(),
            exl3_register_tensor, exl3_register_markers, d.get(),
        };
        if (params.stream_loader) {
            params.stream_source = &source;
            params.load_mode = LLAMA_LOAD_MODE_NONE;
        }
        return llama_model_init_from_user(d->meta, exl3_set_tensor_data, d.get(), params);
    } catch (const std::exception & e) {
        LOG_ERR("%s: cannot load %s: %s\n", __func__, path.c_str(), e.what());
        return nullptr;
    }
}

