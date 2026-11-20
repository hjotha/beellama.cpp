#include "arg.h"
#include "common.h"
#include "download.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"
#include "llama-model-loader.h"
#include "llama-model.h"
#ifdef TEST_HF_STREAM_EXL3
#include "exl3-dir.h"
#include "exl3-test-fixture.h"
#endif

#include "http.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

static void require(bool cond, const char * msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

static uint32_t xorshift32(uint32_t & state) {
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

static void fill_tensor_data(const char * name, uint8_t * dst, size_t n_bytes) {
    uint32_t seed = 0x12345678u;
    for (const char * p = name; *p; ++p) {
        seed = seed * 131u + static_cast<uint8_t>(*p);
    }
    if (n_bytes % sizeof(float) == 0) {
        float * fdst = reinterpret_cast<float *>(dst);
        const size_t n = n_bytes / sizeof(float);
        for (size_t i = 0; i < n; ++i) {
            const float v = (static_cast<float>(xorshift32(seed) & 0xFFFF) / 65535.0f - 0.5f) * 0.05f;
            fdst[i] = v;
        }
    } else {
        for (size_t i = 0; i < n_bytes; ++i) {
            dst[i] = static_cast<uint8_t>(xorshift32(seed) & 0xFF);
        }
    }
}

static std::string read_file_bytes(const fs::path & p) {
    std::ifstream ifs(p, std::ios::binary | std::ios::ate);
    require(ifs.is_open(), ("failed to open file: " + p.string()).c_str());
    const size_t sz = static_cast<size_t>(ifs.tellg());
    std::string data(sz, '\0');
    ifs.seekg(0, std::ios::beg);
    if (sz > 0) {
        ifs.read(&data[0], static_cast<std::streamsize>(sz));
    }
    require(ifs.good() || ifs.eof(), ("failed to read file: " + p.string()).c_str());
    return data;
}

// Create a small valid LLaMA GGUF (~25 MiB) so tensors span many 2 MiB chunks
static void create_test_gguf(const fs::path & out_path) {
    gguf_context * gctx = gguf_init_empty();
    require(gctx != nullptr, "gguf_init_empty failed");
    const LLM_KV kv(LLM_ARCH_LLAMA);

    const uint32_t n_vocab = 256;
    const uint32_t n_embd  = 1024;
    const uint32_t n_head  = 8;
    const uint32_t n_ff    = 2048;
    const uint32_t n_layer = 1;
    const uint32_t n_ctx   = 64;

    gguf_set_val_str(gctx, kv(LLM_KV_GENERAL_ARCHITECTURE).c_str(), "llama");
    gguf_set_val_str(gctx, kv(LLM_KV_GENERAL_NAME).c_str(), "test-hf-stream-vram");
    gguf_set_val_u32(gctx, kv(LLM_KV_CONTEXT_LENGTH).c_str(), n_ctx);
    gguf_set_val_u32(gctx, kv(LLM_KV_EMBEDDING_LENGTH).c_str(), n_embd);
    gguf_set_val_u32(gctx, kv(LLM_KV_FEED_FORWARD_LENGTH).c_str(), n_ff);
    gguf_set_val_u32(gctx, kv(LLM_KV_ATTENTION_HEAD_COUNT).c_str(), n_head);
    gguf_set_val_u32(gctx, kv(LLM_KV_ATTENTION_HEAD_COUNT_KV).c_str(), n_head);
    gguf_set_val_u32(gctx, kv(LLM_KV_BLOCK_COUNT).c_str(), n_layer);
    gguf_set_val_f32(gctx, kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS).c_str(), 1e-5f);
    gguf_set_val_u32(gctx, kv(LLM_KV_ROPE_DIMENSION_COUNT).c_str(), n_embd / n_head);
    gguf_set_val_f32(gctx, kv(LLM_KV_ROPE_FREQ_BASE).c_str(), 10000.0f);

    std::vector<std::string> tokens(n_vocab);
    std::vector<float> scores(n_vocab, 0.0f);
    std::vector<int32_t> ttypes(n_vocab, LLAMA_TOKEN_TYPE_NORMAL);
    for (uint32_t i = 0; i < n_vocab; ++i) {
        tokens[i] = "t" + std::to_string(i);
    }
    ttypes[0] = LLAMA_TOKEN_TYPE_CONTROL;
    ttypes[1] = LLAMA_TOKEN_TYPE_CONTROL;
    ttypes[2] = LLAMA_TOKEN_TYPE_CONTROL;
    gguf_set_val_str(gctx, kv(LLM_KV_TOKENIZER_MODEL).c_str(), "llama");
    std::vector<const char *> token_ptrs;
    for (const auto & token : tokens) {
        token_ptrs.push_back(token.c_str());
    }
    gguf_set_arr_str(gctx, kv(LLM_KV_TOKENIZER_LIST).c_str(), token_ptrs.data(), token_ptrs.size());
    gguf_set_arr_data(gctx, kv(LLM_KV_TOKENIZER_SCORES).c_str(), GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    gguf_set_arr_data(gctx, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, ttypes.data(), ttypes.size());
    gguf_set_val_u32(gctx, kv(LLM_KV_TOKENIZER_BOS_ID).c_str(), uint32_t(1));
    gguf_set_val_u32(gctx, kv(LLM_KV_TOKENIZER_EOS_ID).c_str(), uint32_t(2));

    struct tdesc { std::string name; std::vector<int64_t> ne; };
    const std::vector<tdesc> tdescs = {
        {"token_embd.weight",          {n_embd, n_vocab}},
        {"output_norm.weight",         {n_embd}},
        {"output.weight",              {n_embd, n_vocab}},
        {"blk.0.attn_norm.weight",     {n_embd}},
        {"blk.0.attn_q.weight",        {n_embd, n_embd}},
        {"blk.0.attn_k.weight",        {n_embd, n_embd}},
        {"blk.0.attn_v.weight",        {n_embd, n_embd}},
        {"blk.0.attn_output.weight",   {n_embd, n_embd}},
        {"blk.0.ffn_norm.weight",      {n_embd}},
        {"blk.0.ffn_gate.weight",      {n_embd, n_ff}},
        {"blk.0.ffn_down.weight",      {n_ff, n_embd}},
        {"blk.0.ffn_up.weight",        {n_embd, n_ff}},
    };

    const size_t ctx_size = ggml_tensor_overhead() * tdescs.size() + 64 * 1024 * 1024;
    ggml_context * ctx = ggml_init({ctx_size, nullptr, false});
    require(ctx != nullptr, "ggml_init failed");

    for (const auto & td : tdescs) {
        ggml_tensor * t = td.ne.size() == 1
            ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, td.ne[0])
            : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, td.ne[0], td.ne[1]);
        ggml_set_name(t, td.name.c_str());
        fill_tensor_data(td.name.c_str(), static_cast<uint8_t *>(t->data), ggml_nbytes(t));
        gguf_add_tensor(gctx, t);
    }

    gguf_write_to_file(gctx, out_path.c_str(), false);
    gguf_free(gctx);
    ggml_free(ctx);
}

static std::map<std::string, std::vector<uint8_t>> snapshot_model_tensors(const llama_model * model) {
    std::map<std::string, std::vector<uint8_t>> out;
    for (const auto & kv : model->tensors_by_name) {
        const std::string & name = kv.first;
        const ggml_tensor * t = kv.second;
        std::vector<uint8_t> buf(ggml_nbytes(t));
        if (!t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
            std::memcpy(buf.data(), t->data, buf.size());
        } else {
            ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
        }
        out.emplace(name, std::move(buf));
    }
    return out;
}

static std::vector<float> eval_model_logits(llama_model * model) {
    auto cparams = llama_context_default_params();
    cparams.n_ctx = 32;
    cparams.n_batch = 16;
    cparams.n_ubatch = 16;
    llama_context * ctx = llama_init_from_model(model, cparams);
    require(ctx != nullptr, "llama_init_from_model failed");

    std::vector<llama_token> toks = {1, 5, 17, 42};
    llama_batch batch = llama_batch_get_one(toks.data(), static_cast<int32_t>(toks.size()));
    const int rc = llama_decode(ctx, batch);
    require(rc == 0, "llama_decode failed");

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * logits = llama_get_logits_ith(ctx, static_cast<int32_t>(toks.size()) - 1);
    require(logits != nullptr, "llama_get_logits_ith returned null");
    std::vector<float> res(logits, logits + n_vocab);
    llama_free(ctx);
    return res;
}

struct temp_dir_guard {
    fs::path dir;
    explicit temp_dir_guard(fs::path p) : dir(std::move(p)) {
        fs::create_directories(dir);
    }
    ~temp_dir_guard() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
};

} // namespace

#ifdef TEST_HF_STREAM_EXL3
static std::vector<std::string> write_ple_gguf(const fs::path & dir, const std::string & exl3_path) {
    const auto exl3 = common_exl3_dir_open(exl3_path);
    const auto * meta = common_exl3_dir_gguf(exl3.get());
    const int64_t count = gguf_get_n_tensors(meta);
    ggml_context_ptr ctx(ggml_init({(size_t) count * ggml_tensor_overhead(), nullptr, true}));
    gguf_context_ptr shards[] = {gguf_context_ptr(gguf_init_empty()), gguf_context_ptr(gguf_init_empty())};
    for (uint16_t i = 0; i < 2; ++i) {
        gguf_set_kv(shards[i].get(), meta);
        gguf_set_val_u16(shards[i].get(), "split.count", 2);
        gguf_set_val_u16(shards[i].get(), "split.no", i);
        gguf_set_val_i32(shards[i].get(), "split.tensors.count", count - 1);
    }
    std::vector<std::vector<uint8_t>> data(count);
    for (int64_t i = 0; i < count; ++i) {
        const std::string name = gguf_get_tensor_name(meta, i);
        if (name == "per_layer_token_embd.exl3r_bias") {
            continue;
        }
        ggml_tensor * tensor;
        if (name == "per_layer_token_embd.exl3r_trellis") {
            tensor = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, 160, 16);
            ggml_set_name(tensor, "per_layer_token_embd.weight");
            data[i].resize(ggml_nbytes(tensor), 0x28);
        } else {
            tensor = ggml_new_tensor(ctx.get(), gguf_get_tensor_type(meta, i), GGML_MAX_DIMS, gguf_get_tensor_ne(meta, i));
            ggml_set_name(tensor, name.c_str());
            data[i].resize(ggml_nbytes(tensor));
            common_exl3_dir_read(exl3.get(), name.c_str(), data[i].data(), data[i].size());
        }
        tensor->data = data[i].data();
        gguf_add_tensor(shards[i % 2].get(), tensor);
    }
    std::vector<std::string> paths = {
        (dir / "ple-00001-of-00002.gguf").string(), (dir / "ple-00002-of-00002.gguf").string(),
    };
    for (size_t i = 0; i < paths.size(); ++i) {
        require(gguf_write_to_file(shards[i].get(), paths[i].c_str(), false), "could not write split PLE GGUF");
    }
    return paths;
}
// Replay the range payloads in reverse order for environments without loopback access.
static bool replay_ranges(const llama_model_stream_loader_ops * ops, void * ctx, uint16_t n,
                          const size_t * offsets, void * data) {
    const auto * shards = static_cast<const std::string *>(data);
    for (uint16_t i = n; i-- > 0;) {
        for (size_t end = shards[i].size(); end > offsets[i];) {
            const size_t begin = end - std::min<size_t>(131072, end - offsets[i]);
            int id = 0;
            size_t capacity = 0;
            void * buffer = nullptr;
            if (!ops->acquire_buffer(ctx, 0, &id, &buffer, &capacity) || capacity < end - begin) {
                return false;
            }
            memcpy(buffer, shards[i].data() + begin, end - begin);
            if (!ops->submit_chunk(ctx, 0, id, i, begin, end - begin)) {
                return false;
            }
            end = begin;
        }
    }
    return true;
}

static void test_gguf_replay(const fs::path & dir, const std::vector<std::string> & paths) {
    std::string body[] = {read_file_bytes(paths[0]), read_file_bytes(paths[1])};
    const size_t sizes[] = {body[0].size(), body[1].size()};
    std::string headers[2];
    const char * originals[] = {paths[0].c_str(), paths[1].c_str()};
    const char * header_paths[2];
    for (int i = 0; i < 2; ++i) {
        gguf_context_ptr gguf(gguf_init_from_file(paths[i].c_str(), {true, nullptr}));
        require(gguf != nullptr, "could not read split fixture header");
        headers[i] = (dir / ("header-" + std::to_string(i) + ".gguf")).string();
        header_paths[i] = headers[i].c_str();
        std::ofstream header(headers[i], std::ios::binary);
        header.write(body[i].data(), gguf_get_data_offset(gguf.get()));
        require(header.good(), "could not write split fixture header");
    }
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mtp = true;
    mp.lazy_mode = LLAMA_LAZY_MODE_ON;
    llama_model_ptr reference(llama_model_load_from_splits(originals, 2, mp));
    require(reference != nullptr, "could not load full split fixture");
    const llama_model_stream_source_desc source{2, sizes, nullptr, nullptr, nullptr, nullptr};
    mp.stream_source = &source;
    mp.stream_loader_user_data = body;
    mp.stream_chunk_bytes = 1024 * 1024;
    mp.stream_loader = replay_ranges;
    llama_model_ptr replayed(llama_model_load_from_splits(header_paths, 2, mp));
    require(replayed != nullptr && snapshot_model_tensors(reference.get()) == snapshot_model_tensors(replayed.get()),
            "replayed split GGUF tensors differ from local load");
    const auto * ple = replayed->get_tensor("per_layer_token_embd.weight");
    require(ple && ggml_backend_buffer_is_host(ple->buffer), "replayed lazy PLE weight is not CPU resident");
    printf("PASS: header-only two-shard GGUF replay with lazy CPU PLE weight\n");
}
static void test_exl3_replay(const fs::path & dir) {
    std::vector<fs::path> paths;
    for (const auto & entry : fs::directory_iterator(dir)) {
        const auto name = entry.path().filename().string();
        if (entry.path().extension() == ".safetensors" &&
            name.rfind("cal_trace", 0) != 0 && name.rfind("vision_", 0) != 0) {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    std::vector<std::string> bodies;
    for (const auto & path : paths) {
        bodies.push_back(read_file_bytes(path));
    }
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mtp = true;
    llama_model_ptr local(common_exl3_load_model(dir.string(), mp));
    mp.stream_chunk_bytes = 1024 * 1024;
    mp.stream_loader = replay_ranges;
    mp.stream_loader_user_data = bodies.data();
    llama_model_ptr replayed(common_exl3_load_model(dir.string(), mp));
    require(local && replayed && snapshot_model_tensors(local.get()) == snapshot_model_tensors(replayed.get()),
            "replayed EXL3 tensors differ from local load");
    printf("PASS: bit-exact EXL3 %s replay\n", dir.filename().string().c_str());
}
#endif

int main(int argc, char ** argv) {
    llama_backend_init();

    const fs::path tmp_root = fs::current_path() /
        ("test-hf-stream-vram-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    temp_dir_guard tmp_guard(tmp_root);

#ifdef TEST_HF_STREAM_EXL3
    if (argc == 2 && std::strcmp(argv[1], "--local-replay") == 0) {
        const auto moe = tmp_root / "moe";
        const auto dense = tmp_root / "dense";
        std::map<std::string, exl3_test_fixture::st_file::entry> src;
        int64_t n_params = 0;
        exl3_test_fixture::write_synthetic(dense, src, n_params);
        exl3_test_fixture::write_synthetic_qwen4exp(moe);
        test_gguf_replay(tmp_root, write_ple_gguf(tmp_root, moe.string()));
        test_exl3_replay(dense);
        test_exl3_replay(moe);
        llama_backend_free();
        return 0;
    }
#endif
    require(argc == 1, "unexpected test arguments");
    (void) argv;

    const fs::path ref_path = tmp_root / "ref.gguf";
    create_test_gguf(ref_path);
    const std::string ref_bytes = read_file_bytes(ref_path);
    printf("Created synthetic GGUF: %s (%.2f MiB)\n",
           ref_path.c_str(), static_cast<double>(ref_bytes.size()) / (1024.0 * 1024.0));
    require(ref_bytes.size() > 16 * 1024 * 1024, "expected >16 MiB GGUF to span multiple chunks");

    // Load reference model (GPU if >= 256 MiB free VRAM, else CPU-only so production GPU processes aren't OOM'd)
    llama_model_params ref_mparams = llama_model_default_params();
    ref_mparams.load_mode = LLAMA_LOAD_MODE_NONE;
    ref_mparams.n_gpu_layers = 0;
    if (ggml_backend_dev_t gdev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
        size_t free_b = 0, total_b = 0;
        ggml_backend_dev_memory(gdev, &free_b, &total_b);
        if (free_b >= size_t(256) * 1024 * 1024) {
            ref_mparams.n_gpu_layers = 99;
        } else {
            printf("GPU free VRAM %.1f MiB < 256 MiB; testing streaming path with n_gpu_layers=0\n",
                   static_cast<double>(free_b) / (1024.0 * 1024.0));
        }
    }

    llama_model * ref_model = llama_model_load_from_file(ref_path.c_str(), ref_mparams);
    require(ref_model != nullptr, "failed to load reference model");
    const auto ref_tensors = snapshot_model_tensors(ref_model);
    const auto ref_logits  = eval_model_logits(ref_model);
    const int32_t test_ngl = ref_mparams.n_gpu_layers;
    llama_model_free(ref_model);

    // Local HTTP Range server simulating HF Hub redirect + CDN Range + transient 503 + expired 403 signed URL refresh
    httplib::Server svr;
    std::atomic<int> range_reqs{0};
    std::atomic<int> injected_503{0};
    std::atomic<int> injected_403{0};
    std::atomic<int> redirect_gen{1};
    const std::string etag = "\"test-hf-stream-etag\"";

    svr.Get("/redirect/model.gguf", [&](const httplib::Request &, httplib::Response & res) {
        const int g = redirect_gen.load();
        res.set_redirect("/cdn/model.gguf?gen=" + std::to_string(g), 302);
    });
    svr.Get("/cdn/model.gguf", [&](const httplib::Request & req, httplib::Response & res) {
        res.set_header("ETag", etag);
        res.set_header("Accept-Ranges", "bytes");
        if (req.method == "HEAD") {
            res.set_header("Content-Length", std::to_string(ref_bytes.size()));
            res.status = 200;
            return;
        }
        auto it = req.headers.find("Range");
        if (it == req.headers.end()) {
            res.set_content(ref_bytes, "application/octet-stream");
            return;
        }
        size_t s = 0, e = 0;
        if (sscanf(it->second.c_str(), "bytes=%zu-%zu", &s, &e) != 2 || s >= ref_bytes.size()) {
            res.status = 416;
            return;
        }
        const int rnum = ++range_reqs;
        if (s > 0 && rnum == 3 && injected_503.exchange(1) == 0) {
            res.status = 503;
            res.set_content("transient error", "text/plain");
            return;
        }
        if (s > 0 && rnum == 5 && req.get_param_value("gen") == "1" && injected_403.exchange(1) == 0) {
            redirect_gen.store(2);
            res.status = 403;
            res.set_content("expired signed URL", "text/plain");
            return;
        }
        // cpp-httplib::Server handles Range slicing + 206 Partial Content natively on the full payload
        res.set_content(ref_bytes, "application/octet-stream");
    });

#ifdef TEST_HF_STREAM_EXL3
    const auto dense_dir = tmp_root / "dense";
    const auto moe_dir = tmp_root / "moe";
    std::map<std::string, exl3_test_fixture::st_file::entry> src;
    int64_t n_params = 0;
    exl3_test_fixture::write_synthetic(dense_dir, src, n_params);
    exl3_test_fixture::write_synthetic_qwen4exp(moe_dir);
    const auto gguf_splits = write_ple_gguf(tmp_root, moe_dir.string());

    std::map<std::string, std::string> range_files;
    const auto route = [&](const fs::path & p) {
        return "/" + p.parent_path().filename().string() + "/" + p.filename().string();
    };
    for (const auto & dir : {dense_dir, moe_dir}) {
        for (const auto & entry : fs::directory_iterator(dir)) {
            if (entry.path().extension() == ".safetensors") {
                range_files.emplace(route(entry.path()), read_file_bytes(entry.path()));
            }
        }
    }
    for (const auto & path : gguf_splits) {
        range_files.emplace(route(path), read_file_bytes(path));
    }
    for (const auto & [name, bytes] : range_files) {
        svr.Get(name, [&, body = &bytes](const httplib::Request & req, httplib::Response & res) {
            res.set_header("ETag", etag);
            res.set_header("Accept-Ranges", "bytes");
            if (req.method == "HEAD") {
                res.set_header("Content-Length", std::to_string(body->size()));
            } else {
                res.set_content(*body, "application/octet-stream");
            }
        });
    }
#endif

#ifdef TEST_HF_STREAM_EXL3
    const std::string commit(40, 'a');
    const std::string repo = "fixtures/moe";
    const std::string resolve = "/" + repo + "/resolve/" + commit + "/";
    svr.Get("/api/models/" + repo + "/refs", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"branches\":[{\"name\":\"main\",\"targetCommit\":\"" + commit + "\"}]}", "application/json");
    });
    std::string tree = "[{\"type\":\"file\",\"path\":\"old_cal_trace.safetensors\"},"
                        "{\"type\":\"file\",\"path\":\"extra_vision_tower.safetensors\"}";
    for (const auto & entry : fs::directory_iterator(moe_dir)) {
        const auto name = entry.path().filename().string();
        tree += ",{\"type\":\"file\",\"path\":\"" + name + "\"}";
        svr.Get(resolve + name, [&, bytes = read_file_bytes(entry.path())](const httplib::Request & req, httplib::Response & res) {
            res.set_header("ETag", etag);
            res.set_header("Accept-Ranges", "bytes");
            if (req.method == "HEAD") {
                res.set_header("Content-Length", std::to_string(bytes.size()));
            } else {
                res.set_content(bytes, "application/octet-stream");
            }
        });
    }
    tree += "]";
    svr.Get("/api/models/" + repo + "/tree/" + commit, [&](const httplib::Request &, httplib::Response & res) {
        res.set_content(tree, "application/json");
    });
#endif

    const int port = svr.bind_to_any_port("127.0.0.1");
    require(port > 0, "failed to bind local HTTP test server");
    std::thread svr_thread([&]() { svr.listen_after_bind(); });

    const std::string url = "http://127.0.0.1:" + std::to_string(port) + "/redirect/model.gguf";

    auto verify_loaded_model = [&](llama_model * model, const char * label) {
        require(model != nullptr, (std::string(label) + ": model load failed").c_str());
        const auto got = snapshot_model_tensors(model);
        require(got.size() == ref_tensors.size(), "tensor count mismatch");
        for (const auto & [name, exp_vec] : ref_tensors) {
            auto git = got.find(name);
            require(git != got.end(), ("missing tensor: " + name).c_str());
            require(git->second.size() == exp_vec.size(), ("tensor size mismatch: " + name).c_str());
            require(std::memcmp(git->second.data(), exp_vec.data(), exp_vec.size()) == 0,
                    (std::string(label) + ": tensor byte mismatch for " + name).c_str());
        }
        const auto got_logits = eval_model_logits(model);
        require(got_logits.size() == ref_logits.size(), "logits size mismatch");
        for (size_t i = 0; i < ref_logits.size(); ++i) {
            require(std::fabs(got_logits[i] - ref_logits[i]) < 1e-6f, "logits mismatch");
        }
        printf("PASS: %s tensors + logits bit-identical (ngl=%d)\n", label, test_ngl);
    };

    try {
        // Case 1: STREAM + SYNC disk tee (4 threads, 2 MiB chunks)
        {
            const fs::path sync_path = tmp_root / "sync.gguf";
            common_download_task t;
            t.url        = url;
            t.local_path = sync_path.string();
            t.opts.n_threads = 4;

            common_download_stream_opts sopts;
            sopts.n_threads   = 4;
            sopts.chunk_bytes = 2 * 1024 * 1024;
            sopts.disk_mode   = COMMON_HF_STREAM_DISK_SYNC;

            std::vector<std::string> stream_paths;
            // Verify prepare + abort cleans up temp file & lock and allows immediate re-prepare
            require(common_download_prepare_stream({t}, sopts, stream_paths), "prepare_stream(abort-test) failed");
            require(stream_paths.size() == 1 && fs::exists(stream_paths[0]), "expected temp file after prepare");
            common_download_stream_abort(stream_paths[0]);
            require(!fs::exists(stream_paths[0]), "temp file should be removed by common_download_stream_abort");
            require(common_download_get_stream_session_ud(stream_paths[0]) == nullptr, "session should be cleared after abort");

            require(common_download_prepare_stream({t}, sopts, stream_paths), "prepare_stream(sync) failed");
            require(stream_paths.size() == 1, "expected 1 stream path");
            require(common_download_stream_final_path(stream_paths[0]) == sync_path.string(), "unexpected final path mapping");
            void * ud = common_download_get_stream_session_ud(stream_paths[0]);
            require(ud != nullptr, "missing stream session ud");

            llama_model_params mp = llama_model_default_params();
            mp.load_mode               = LLAMA_LOAD_MODE_NONE;
            mp.n_gpu_layers            = test_ngl;
            mp.check_tensors           = true;
            mp.stream_loader           = common_download_stream_loader;
            mp.stream_loader_user_data = ud;
            mp.stream_chunk_bytes      = 2 * 1024 * 1024;

            llama_model * m = llama_model_load_from_file(stream_paths[0].c_str(), mp);
            verify_loaded_model(m, "stream-sync");
            llama_model_free(m);

            require(injected_503.load() == 1, "expected transient 503 injection to have triggered and recovered");
            require(injected_403.load() == 1 && redirect_gen.load() == 2, "expected expired 403 signed URL refresh to have triggered");
            require(range_reqs.load() > 6, "expected multiple parallel HTTP Range requests");
            require(fs::exists(sync_path), "sync.gguf should exist on disk");
            require(read_file_bytes(sync_path) == ref_bytes, "sync.gguf byte mismatch vs ref.gguf");
            printf("PASS: sync disk tee matches ref.gguf byte-for-byte (%d Range reqs, 503 recovered)\n",
                   range_reqs.load());

            // Case 5: Cache-hit check on existing sync.gguf -> prepare_stream returns false cleanly
            std::vector<std::string> cached_paths;
            require(!common_download_prepare_stream({t}, sopts, cached_paths),
                    "prepare_stream should return false when file is already cached on disk");
            printf("PASS: cache-hit detection skips redundant streaming\n");
        }

        // Case 2: STREAM + BACKGROUND disk tee
        {
            const fs::path bg_path = tmp_root / "bg.gguf";
            common_download_task t;
            t.url        = url;
            t.local_path = bg_path.string();
            t.opts.n_threads = 4;

            common_download_stream_opts sopts;
            sopts.n_threads   = 4;
            sopts.chunk_bytes = 2 * 1024 * 1024;
            sopts.disk_mode   = COMMON_HF_STREAM_DISK_BACKGROUND;

            std::vector<std::string> stream_paths;
            require(common_download_prepare_stream({t}, sopts, stream_paths), "prepare_stream(bg) failed");
            llama_model_params mp = llama_model_default_params();
            mp.load_mode               = LLAMA_LOAD_MODE_NONE;
            mp.n_gpu_layers            = test_ngl;
            mp.stream_loader           = common_download_stream_loader;
            mp.stream_loader_user_data = common_download_get_stream_session_ud(stream_paths[0]);
            mp.stream_chunk_bytes      = 2 * 1024 * 1024;

            llama_model * m = llama_model_load_from_file(stream_paths[0].c_str(), mp);
            verify_loaded_model(m, "stream-background");
            llama_model_free(m);

            common_download_wait_background();
            require(fs::exists(bg_path), "bg.gguf should exist after background wait");
            require(read_file_bytes(bg_path) == ref_bytes, "bg.gguf byte mismatch vs ref.gguf");
            printf("PASS: background disk tee finalized byte-identical bg.gguf\n");
        }

        // Case 3: STREAM + DISK NONE
        {
            const fs::path none_path = tmp_root / "none.gguf";
            common_download_task t;
            t.url        = url;
            t.local_path = none_path.string();
            t.opts.n_threads = 4;

            common_download_stream_opts sopts;
            sopts.n_threads   = 4;
            sopts.chunk_bytes = 2 * 1024 * 1024;
            sopts.disk_mode   = COMMON_HF_STREAM_DISK_NONE;

            std::vector<std::string> stream_paths;
            require(common_download_prepare_stream({t}, sopts, stream_paths), "prepare_stream(none) failed");
            llama_model_params mp = llama_model_default_params();
            mp.load_mode               = LLAMA_LOAD_MODE_NONE;
            mp.n_gpu_layers            = test_ngl;
            mp.stream_loader           = common_download_stream_loader;
            mp.stream_loader_user_data = common_download_get_stream_session_ud(stream_paths[0]);
            mp.stream_chunk_bytes      = 2 * 1024 * 1024;

            llama_model * m = llama_model_load_from_file(stream_paths[0].c_str(), mp);
            verify_loaded_model(m, "stream-disk-none");
            llama_model_free(m);

            require(!fs::exists(none_path), "none.gguf should not persist in disk-mode=none");
            require(!fs::exists(none_path.string() + ".downloadInProgress"), "temp partial should be removed");
            require(!fs::exists(none_path.string() + ".hfstream.tmp"), "stream temp should be removed");
            printf("PASS: disk-mode=none kept VRAM/RAM intact with zero persistent file\n");
        }

#ifdef TEST_HF_STREAM_EXL3
        const std::string origin = "http://127.0.0.1:" + std::to_string(port);
        const auto make_task = [&](const fs::path & original, const fs::path & dest) {
            common_download_task task;
            task.url = origin + route(original);
            task.local_path = dest.string();
            return task;
        };
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.load_mode = LLAMA_LOAD_MODE_NONE;
        mp.load_mtp = true;
        mp.stream_chunk_bytes = 1024 * 1024;
        common_download_stream_opts opts;
        opts.n_threads = 4;
        opts.chunk_bytes = mp.stream_chunk_bytes;

        for (const auto & original : {dense_dir, moe_dir}) {
            llama_model_ptr local(common_exl3_load_model(original.string(), mp));
            require(local != nullptr, "local EXL3 fixture failed to load");
            const auto expected = snapshot_model_tensors(local.get());
            for (auto mode : {COMMON_HF_STREAM_DISK_SYNC, COMMON_HF_STREAM_DISK_BACKGROUND, COMMON_HF_STREAM_DISK_NONE}) {
                opts.disk_mode = mode;
                const auto dir = tmp_root / (original.filename().string() + "-stream-" + std::to_string(mode));
                fs::create_directory(dir);
                std::vector<common_download_task> tasks;
                std::vector<std::string> names;
                for (const auto & entry : fs::directory_iterator(original)) {
                    const std::string fn = entry.path().filename().string();
                    if (fn.rfind("cal_trace", 0) == 0 || fn.rfind("vision_", 0) == 0) {
                        continue;
                    }
                    if (entry.path().extension() == ".safetensors") {
                        names.push_back(fn);
                        tasks.push_back(make_task(entry.path(), dir / names.back()));
                    } else {
                        fs::copy_file(entry.path(), dir / fn);
                    }
                }
                require(common_download_prepare_stream_exl3(dir.string(), tasks, names, opts), "EXL3 prepare failed");
                llama_model_ptr streamed(common_exl3_load_model(dir.string(), mp));
                require(streamed != nullptr, "streamed EXL3 model failed to load");
                require(snapshot_model_tensors(streamed.get()) == expected, "streamed EXL3 tensors differ from local load");
                if (original == moe_dir) {
                    require(streamed->hparams.ple_layer_multipliers == local->hparams.ple_layer_multipliers &&
                            streamed->hparams.ple_head_offsets == local->hparams.ple_head_offsets &&
                            streamed->hparams.ple_head_vocab_sizes == local->hparams.ple_head_vocab_sizes,
                            "streamed PLE constants differ from local load");
                }
                common_download_wait_background();
                for (const auto & name : names) {
                    if (mode == COMMON_HF_STREAM_DISK_NONE) {
                        require(!fs::exists(dir / name), "EXL3 shard persisted in none mode");
                    } else {
                        require(read_file_bytes(dir / name) == read_file_bytes(original / name), "EXL3 disk tee mismatch");
                    }
                    require(!fs::exists((dir / name).string() + ".hfstream.tmp"), "EXL3 temp was left behind");
                }
                printf("PASS: EXL3 %s bit-exact tensors and disk mode %d\n", original.filename().string().c_str(), mode);
            }
        }

        const auto cache = (tmp_root / "hf-cache").string();
#ifdef _WIN32
        require(_putenv_s("MODEL_ENDPOINT", origin.c_str()) == 0 && _putenv_s("HF_HUB_CACHE", cache.c_str()) == 0,
                "could not configure local HF fixture");
#else
        require(setenv("MODEL_ENDPOINT", origin.c_str(), 1) == 0 && setenv("HF_HUB_CACHE", cache.c_str(), 1) == 0,
                "could not configure local HF fixture");
#endif
        common_params hf;
        hf.model.hf_repo = repo;
        hf.hf_stream_vram = true;
        hf.hf_stream_disk_mode = "none";
        hf.hf_stream_chunk_mb = 1;
        auto handler = common_models_handler_init(hf, LLAMA_EXAMPLE_SERVER);
        require(handler.plan.is_exl3 && handler.plan.exl3_tensor_files.size() == 3, "HF EXL3 plan did not filter sidecars");
        common_models_handler_apply(handler, hf, nullptr);
        require(common_download_get_stream_session_ud(hf.model.path) != nullptr, "HF EXL3 stream was not prepared");
        llama_model_ptr hf_model(common_exl3_load_model(hf.model.path, mp));
        llama_model_ptr local_moe(common_exl3_load_model(moe_dir.string(), mp));
        require(hf_model && local_moe && snapshot_model_tensors(hf_model.get()) == snapshot_model_tensors(local_moe.get()),
                "HF-discovered EXL3 tensors differ from local load");
        printf("PASS: HF EXL3 repository discovery, metadata and range streaming\n");

        mp.lazy_mode = LLAMA_LAZY_MODE_ON;
        const char * originals[] = {gguf_splits[0].c_str(), gguf_splits[1].c_str()};
        llama_model_ptr local(llama_model_load_from_splits(originals, 2, mp));
        require(local != nullptr, "local split PLE GGUF failed to load");
        const auto expected = snapshot_model_tensors(local.get());
        for (auto mode : {COMMON_HF_STREAM_DISK_SYNC, COMMON_HF_STREAM_DISK_BACKGROUND, COMMON_HF_STREAM_DISK_NONE}) {
            opts.disk_mode = mode;
            const auto dir = tmp_root / ("gguf-stream-" + std::to_string(mode));
            fs::create_directory(dir);
            std::vector<common_download_task> tasks;
            for (const auto & path : gguf_splits) {
                tasks.push_back(make_task(path, dir / fs::path(path).filename()));
            }
            std::vector<std::string> paths;
            require(common_download_prepare_stream(tasks, opts, paths) && paths.size() == 2, "split GGUF prepare failed");
            if (mode == COMMON_HF_STREAM_DISK_NONE) {
                require(fs::file_size(paths[0]) < fs::file_size(gguf_splits[0]), "none mode GGUF temp extended to full size");
            }
            const char * splits[] = {paths[0].c_str(), paths[1].c_str()};
            auto stream_mp = mp;
            stream_mp.stream_loader = common_download_stream_loader;
            stream_mp.stream_loader_user_data = common_download_get_stream_session_ud(paths[0]);
            llama_model_ptr streamed(llama_model_load_from_splits(splits, 2, stream_mp));
            require(streamed != nullptr, "streamed split PLE GGUF failed to load");
            const auto * ple = streamed->get_tensor("per_layer_token_embd.weight");
            require(ple && ggml_backend_buffer_is_host(ple->buffer), "lazy PLE weight is not CPU resident");
            require(snapshot_model_tensors(streamed.get()) == expected, "streamed split GGUF tensors differ from local load");
            common_download_wait_background();
            for (size_t i = 0; i < paths.size(); ++i) {
                require(!fs::exists(paths[i]), "GGUF temp was left behind");
                if (mode == COMMON_HF_STREAM_DISK_NONE) {
                    require(!fs::exists(tasks[i].local_path), "GGUF shard persisted in none mode");
                } else {
                    require(read_file_bytes(tasks[i].local_path) == read_file_bytes(gguf_splits[i]), "GGUF split disk tee mismatch");
                }
            }
            printf("PASS: split GGUF with lazy CPU PLE weight and disk mode %d\n", mode);
        }
#endif

        // Case 4: Parallel multi-thread disk download (common_download_file_single)
        {
            const fs::path par_path = tmp_root / "parallel.gguf";
            common_download_opts dopts;
            dopts.n_threads = 4;
            const int st = common_download_file_single(url, par_path.string(), dopts);
            require(st >= 200 && st < 300, "parallel common_download_file_single failed");
            require(fs::exists(par_path), "parallel.gguf missing");
            require(read_file_bytes(par_path) == ref_bytes, "parallel.gguf byte mismatch vs ref.gguf");
            printf("PASS: multi-thread parallel disk download byte-identical\n");
        }
    } catch (...) {
        svr.stop();
        svr_thread.join();
        throw;
    }

    svr.stop();
    svr_thread.join();
    llama_backend_free();
    printf("ALL test-hf-stream-vram CHECKS PASSED\n");
    return 0;
}
