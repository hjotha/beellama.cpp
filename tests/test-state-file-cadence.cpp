// Standalone batch-cadence vs serialization diagnostic for the
// state-file-stream regression ("pre-last restore logits differ from cold").
// Uses ONLY the public C API + ggml (no libcommon dependency). Compares three
// decode cadences over the SAME token sequence:
//   A cold_full : all tokens in ONE batch (matches the existing test's cold
//                 decode of 184 tokens with n_batch 256)
//   B live_prefix: prefix N-1 in one batch, then the last token in its own
//                 batch (NO serialization)
//   C prefix native save -> streaming restore into a fresh empty context,
//                 then the last token (llama_state_seq_save_file +
//                 llama_state_seq_load_file_streaming with exact byte count
//                 and the XXH64 file hash, matching the original save/read
//                 semantics)
// Reports all-vocab max-abs and RMSE for (A,B), (A,C), (B,C), the greedy
// token ids and the final sequence positions. If the baseline binary already
// shows A!=C while B==C, the difference is BATCH CADENCE, not
// serialization.
//
// Build baseline (ABI: compile against the OLD committed public header
// obtained with `git show HEAD:include/llama.h`, never the working tree
// header which added incompatible context-param fields):
//   g++ -std=c++17 -O2 -I <artifact>/cadence-include -I ggml/include
//     tests/test-state-file-cadence.cpp
//     -L <baseline-bin> -lllama -lggml -lggml-base
//     -Wl,-rpath,<baseline-bin> -o <out>/cadence-baseline
// Build current (repo headers + freshly built build-optimized libs):
//   g++ -std=c++17 -O2 -I include -I ggml/include
//     tests/test-state-file-cadence.cpp
//     -L <build>/bin -lllama -lggml -lggml-base
//     -Wl,-rpath,<build>/bin -o <out>/cadence-current
// Run (TMPDIR = NVMe spool; the diagnostic cleans its own files):
//   TMPDIR=<spool> LD_LIBRARY_PATH=<libdir> ./cadence-<variant>
//     MODEL FIXTURE_SEQUENCE_BIN
// where FIXTURE_SEQUENCE_BIN is a read-only legacy sequence file whose native
// header/tokens are the exact input tokens (spool legacy-sequence.bin), or
// the path "synthetic" for the built-in fixture.

#define XXH_INLINE_ALL
#include "../vendor/hash/xxhash/xxhash.h"

#include "llama.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h> // rmdir (Linux diagnostic)

static void die(const std::string & msg) {
    std::fprintf(stderr, "CADENCE-FATAL: %s\n", msg.c_str());
    std::exit(2);
}

// ---- token fixture ----------------------------------------------------------

static std::vector<llama_token> load_fixture(const char * path, int n_vocab) {
    std::vector<llama_token> tokens;
    if (path != nullptr && std::string(path) != "synthetic") {
        FILE * f = std::fopen(path, "rb");
        if (f == nullptr) {
            die("cannot open fixture sequence file (read-only)");
        }
        uint32_t magic = 0, version = 0, n = 0;
        if (std::fread(&magic, 4, 1, f) != 1 || std::fread(&version, 4, 1, f) != 1 ||
                std::fread(&n, 4, 1, f) != 1) {
            std::fclose(f);
            die("fixture sequence header truncated");
        }
        tokens.resize(n);
        if (n > 0 && std::fread(tokens.data(), sizeof(llama_token), n, f) != n) {
            std::fclose(f);
            die("fixture sequence token payload truncated");
        }
        std::fclose(f);
        std::printf("CADENCE fixture: file=%s magic=%08x version=%u n_tokens=%zu\n",
                    path, (unsigned) magic, (unsigned) version, tokens.size());
        return tokens;
    }
    for (int i = 0; i < 184; ++i) {
        tokens.push_back(llama_token(3 + (i * 17) % (n_vocab - 3)));
    }
    std::printf("CADENCE fixture: synthetic n_tokens=%zu\n", tokens.size());
    return tokens;
}

// ---- manual batch decode (public API only) ----------------------------------

struct ctx_ptr {
    llama_context * ptr = nullptr;
    explicit ctx_ptr(llama_context * p) : ptr(p) {}
    ~ctx_ptr() { if (ptr) { llama_free(ptr); } }
    llama_context * operator->() { return ptr; }
    operator llama_context *() { return ptr; }
};

struct model_ptr {
    llama_model * ptr = nullptr;
    explicit model_ptr(llama_model * p) : ptr(p) {}
    ~model_ptr() { if (ptr) { llama_model_free(ptr); } }
    operator llama_model *() { return ptr; }
};

static bool decode_manual(llama_context * ctx, const std::vector<llama_token> & toks, llama_pos start) {
    const int n = (int) toks.size();
    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        batch.token[i] = toks[i];
        batch.pos[i] = start + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 0;
    }
    batch.logits[n - 1] = 1;
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc == 0;
}

// ---- comparison helpers -----------------------------------------------------

struct cmp {
    double max_abs = 0.0;
    double rmse = 0.0;
};

static cmp compare_logits(const float * a, const float * b, int n) {
    cmp out;
    double sum_sq = 0.0;
    bool a_fin = true, b_fin = true;
    for (int i = 0; i < n; ++i) {
        a_fin = a_fin && std::isfinite(a[i]);
        b_fin = b_fin && std::isfinite(b[i]);
        const double d = double(a[i]) - double(b[i]);
        if (std::fabs(d) > out.max_abs) {
            out.max_abs = std::fabs(d);
        }
        sum_sq += d * d;
    }
    if (!a_fin || !b_fin) {
        die("non-finite logits in comparison");
    }
    out.rmse = std::sqrt(sum_sq / double(n));
    return out;
}

static llama_token greedy(const float * logits, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return llama_token(best);
}

static llama_pos seq_pos(llama_context * ctx) {
    return llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
}

// ---- main -------------------------------------------------------------------

int main(int argc, char ** argv) {
    if (argc != 3) {
        die("usage: test-state-file-cadence MODEL FIXTURE_SEQUENCE_BIN");
    }
    const char * model_path = argv[1];
    const char * fixture_path = argv[2];
    const char * tmpdir = std::getenv("TMPDIR");
    if (tmpdir == nullptr || *tmpdir == '\0') {
        die("TMPDIR must be set to an NVMe spool directory");
    }

    ggml_backend_load_all();

    auto mp = llama_model_default_params();
    ggml_backend_dev_t devices[] = { nullptr };
    mp.n_gpu_layers = 0;
    mp.devices = devices;
    model_ptr model(llama_model_load_from_file(model_path, mp));
    if (!model.ptr) {
        die("model load failed");
    }
    auto params = llama_context_default_params();
    params.n_ctx = 512;
    params.n_batch = 256;
    params.n_ubatch = 256;
    params.n_threads = params.n_threads_batch = 2;
    params.type_k = params.type_v = GGML_TYPE_Q4_0;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.offload_kqv = false;

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.ptr));
    const std::vector<llama_token> tokens = load_fixture(fixture_path, n_vocab);
    const size_t N = tokens.size();
    if (N < 2) {
        die("fixture needs at least two tokens");
    }
    const std::vector<llama_token> prefix(tokens.begin(), tokens.end() - 1);
    const llama_token last = tokens.back();

    // A: cold full decode, all N tokens in one batch
    ctx_ptr ctx_a(llama_init_from_model(model.ptr, params));
    if (!ctx_a.ptr) {
        die("context A failed");
    }
    if (!decode_manual(ctx_a.ptr, tokens, 0)) {
        die("phase A decode failed");
    }
    const float * logits_a = llama_get_logits_ith(ctx_a.ptr, -1);
    const llama_pos pos_a = seq_pos(ctx_a.ptr);

    // B: live prefix N-1 in one batch, then the last token (NO serialization)
    ctx_ptr ctx_b(llama_init_from_model(model.ptr, params));
    if (!ctx_b.ptr) {
        die("context B failed");
    }
    if (!decode_manual(ctx_b.ptr, prefix, 0) ||
            !decode_manual(ctx_b.ptr, { last }, (llama_pos) prefix.size())) {
        die("phase B decode failed");
    }
    const float * logits_b = llama_get_logits_ith(ctx_b.ptr, -1);
    const llama_pos pos_b = seq_pos(ctx_b.ptr);

    // C: prefix save -> fresh empty context -> streaming restore -> last token
    ctx_ptr ctx_c(llama_init_from_model(model.ptr, params));
    if (!ctx_c.ptr) {
        die("context C failed");
    }
    if (!decode_manual(ctx_c.ptr, prefix, 0)) {
        die("phase C prefix decode failed");
    }
    const std::string pattern = std::string(tmpdir) + "/cadence-XXXXXX";
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back(0);
    if (mkdtemp(name.data()) == nullptr) {
        die("mkdtemp failed on TMPDIR spool");
    }
    const std::string workdir = name.data();
    const std::string state_path = workdir + "/prefix183.bin";
    const size_t written = llama_state_seq_save_file(ctx_c.ptr, state_path.c_str(), 0,
            prefix.data(), prefix.size());
    if (written == 0) {
        die("phase C native save failed");
    }
    FILE * f = std::fopen(state_path.c_str(), "rb");
    if (f == nullptr) {
        die("phase C saved file unreadable");
    }
    std::vector<uint8_t> bytes(written);
    const size_t got = std::fread(bytes.data(), 1, written, f);
    std::fclose(f);
    if (got != written) {
        die("phase C saved file size mismatch");
    }
    const uint64_t file_hash = XXH64(bytes.data(), bytes.size(), 0);
    llama_free(ctx_c.ptr);
    ctx_c.ptr = llama_init_from_model(model.ptr, params);
    if (!ctx_c.ptr) {
        die("phase C restore context failed");
    }
    std::vector<llama_token> restored(prefix.size());
    size_t restored_count = 0;
    const size_t read = llama_state_seq_load_file_streaming(ctx_c.ptr, state_path.c_str(), 0,
            restored.data(), restored.size(), &restored_count, written, file_hash);
    if (read != written || restored_count != prefix.size() || restored != prefix) {
        die("phase C streaming restore failed (read/count/tokens mismatch)");
    }
    if (!decode_manual(ctx_c.ptr, { last }, (llama_pos) prefix.size())) {
        die("phase C suffix decode failed");
    }
    const float * logits_c = llama_get_logits_ith(ctx_c.ptr, -1);
    const llama_pos pos_c = seq_pos(ctx_c.ptr);

    // cleanup our own spool files
    std::remove(state_path.c_str());
    ::rmdir(workdir.c_str());

    const cmp ab = compare_logits(logits_a, logits_b, n_vocab);
    const cmp ac = compare_logits(logits_a, logits_c, n_vocab);
    const cmp bc = compare_logits(logits_b, logits_c, n_vocab);

    std::printf("CADENCE-RESULT A-B maxabs=%.9f rmse=%.9f\n", ab.max_abs, ab.rmse);
    std::printf("CADENCE-RESULT A-C maxabs=%.9f rmse=%.9f\n", ac.max_abs, ac.rmse);
    std::printf("CADENCE-RESULT B-C maxabs=%.9f rmse=%.9f\n", bc.max_abs, bc.rmse);
    std::printf("CADENCE-RESULT greedy A=%d B=%d C=%d\n",
                (int) greedy(logits_a, n_vocab), (int) greedy(logits_b, n_vocab),
                (int) greedy(logits_c, n_vocab));
    std::printf("CADENCE-RESULT pos A=%lld B=%lld C=%lld (n_tokens=%zu)\n",
                (long long) pos_a, (long long) pos_b, (long long) pos_c, N);
    std::printf("CADENCE-RESULT saved_bytes=%zu xxh64=%016llx\n",
                written, (unsigned long long) file_hash);
    std::printf("CADENCE-DONE\n");
    return 0;
}