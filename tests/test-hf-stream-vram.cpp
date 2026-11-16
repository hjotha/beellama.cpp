#include "arg.h"
#include "common.h"
#include "download.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"
#include "llama-model-loader.h"
#include "llama-model.h"
#include "llama-model-saver.h"

#include "http.h"

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
    llama_model_saver ms(LLM_ARCH_LLAMA, gctx);

    const uint32_t n_vocab = 256;
    const uint32_t n_embd  = 1024;
    const uint32_t n_head  = 8;
    const uint32_t n_ff    = 2048;
    const uint32_t n_layer = 1;
    const uint32_t n_ctx   = 64;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE, "llama");
    ms.add_kv(LLM_KV_GENERAL_NAME, "test-hf-stream-vram");
    ms.add_kv(LLM_KV_CONTEXT_LENGTH, n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head);
    ms.add_kv(LLM_KV_BLOCK_COUNT, n_layer);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT, n_embd / n_head);
    ms.add_kv(LLM_KV_ROPE_FREQ_BASE, 10000.0f);

    std::vector<std::string> tokens(n_vocab);
    std::vector<float> scores(n_vocab, 0.0f);
    std::vector<int32_t> ttypes(n_vocab, LLAMA_TOKEN_TYPE_NORMAL);
    for (uint32_t i = 0; i < n_vocab; ++i) {
        tokens[i] = "t" + std::to_string(i);
    }
    ttypes[0] = LLAMA_TOKEN_TYPE_CONTROL;
    ttypes[1] = LLAMA_TOKEN_TYPE_CONTROL;
    ttypes[2] = LLAMA_TOKEN_TYPE_CONTROL;
    ms.add_kv(LLM_KV_TOKENIZER_MODEL, "llama");
    ms.add_kv(LLM_KV_TOKENIZER_LIST, tokens);
    ms.add_kv(LLM_KV_TOKENIZER_SCORES, scores);
    gguf_set_arr_data(gctx, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, ttypes.data(), ttypes.size());
    ms.add_kv(LLM_KV_TOKENIZER_BOS_ID, uint32_t(1));
    ms.add_kv(LLM_KV_TOKENIZER_EOS_ID, uint32_t(2));

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

int main() {
    llama_backend_init();

    const fs::path tmp_root = fs::temp_directory_path() /
        ("test-hf-stream-vram-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    temp_dir_guard tmp_guard(tmp_root);

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

    // Local HTTP Range server simulating HF Hub redirect + CDN Range + transient 503 failure
    httplib::Server svr;
    std::atomic<int> range_reqs{0};
    std::atomic<int> injected_503{0};
    const std::string etag = "\"test-hf-stream-etag\"";

    svr.Get("/redirect/model.gguf", [&](const httplib::Request &, httplib::Response & res) {
        res.set_redirect("/cdn/model.gguf", 302);
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
        // cpp-httplib::Server handles Range slicing + 206 Partial Content natively on the full payload
        res.set_content(ref_bytes, "application/octet-stream");
    });

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
            require(common_download_prepare_stream({t}, sopts, stream_paths), "prepare_stream(sync) failed");
            require(stream_paths.size() == 1, "expected 1 stream path");
            void * ud = common_download_get_stream_session_ud(stream_paths[0]);
            require(ud != nullptr, "missing stream session ud");

            llama_model_params mp = llama_model_default_params();
            mp.load_mode               = LLAMA_LOAD_MODE_NONE;
            mp.n_gpu_layers            = test_ngl;
            mp.stream_loader           = common_download_stream_loader;
            mp.stream_loader_user_data = ud;
            mp.stream_chunk_bytes      = 2 * 1024 * 1024;

            llama_model * m = llama_model_load_from_file(stream_paths[0].c_str(), mp);
            verify_loaded_model(m, "stream-sync");
            llama_model_free(m);

            require(injected_503.load() == 1, "expected transient 503 injection to have triggered and recovered");
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
            printf("PASS: disk-mode=none kept VRAM/RAM intact with zero persistent file\n");
        }

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
