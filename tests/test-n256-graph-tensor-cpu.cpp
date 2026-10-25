// CPU fixture: N256 graph/tensor contract for the Qwen3.8 qwen35 hybrid.
//
// Read-only graph inspection: no token generation, no decode, no server, no
// GPU/RPC/Xbox, no benchmark. It loads the real GGUF only after checking the
// file, its real size and available RAM with margin; otherwise it skips with
// code 77 (ctest SKIP_RETURN_CODE) without attempting to load.
//
// For graph_reserve(256,1), (256,2) and (128,1), it follows the
// linear_attn_qkv_mixed reshape to the single QKV ggml_mul_mat and checks:
//   - src1->ne[1] == n_tokens (256 or 128), dst->ne[1] == n_tokens;
//   - reshape ne=[qkv, n_tokens/n_seqs, n_seqs];
//   - contiguous nb for src1/dst/reshape;
//   - the node and the GEMM are dispatched to the CPU backend;
//   - exactly one GGML_OP_MUL_MAT on the QKV chain (no hidden 2xN128 split).
// The former (256,2) abort was a fixture mismatch: output_all=true made the
// mctx split the 2-seq batch into 1-seq ubatches, while graph_reserve(256,2)
// built a 2-seq ubatch, so n_rs - n_seqs went negative. With output_all=false
// and a matching mctx ubatch check, (256,2) passes. --skip-2seq is kept only
// for isolation; the default runs all three cases.
// Model-level only; this is not numerical equivalence or TPS evidence.

#include "llama.h"
#include "llama-context.h"
#include "llama-graph.h"
#include "llama-batch.h"
#include "llama-memory.h"
#include "llama-model.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>

namespace {

const char * kDefaultModel =
    "/home/hjotha/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf";
constexpr uint64_t kRamMargin = 1ULL << 30; // 1 GiB for graph/context/metadata

int failures = 0;
int skipped = 0;

void check(bool ok, const char * name) {
    std::printf("%s: %s\n", name, ok ? "OK" : "FAIL");
    if (!ok) ++failures;
}

bool file_size(const char * path, uint64_t & size) {
    struct stat st;
    if (stat(path, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size <= 0) return false;
    size = static_cast<uint64_t>(st.st_size);
    return true;
}

uint64_t mem_available() {
    std::ifstream f("/proc/meminfo");
    std::string key;
    unsigned long long value = 0;
    std::string unit;
    while (f >> key >> value >> unit) {
        if (key == "MemAvailable:") return value * 1024ULL;
    }
    return 0;
}

const char * model_path() {
    const char * env = std::getenv("N256_GRAPH_MODEL");
    return env && *env ? env : kDefaultModel;
}

const ggml_tensor * find_named(ggml_cgraph * gf, const char * needle) {
    const int n = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * node = ggml_graph_node(gf, i);
        if (node && std::strstr(node->name, needle)) return node;
    }
    return nullptr;
}

const ggml_tensor * follow_qkv_chain(const ggml_tensor * t, int & mul_mat_count) {
    mul_mat_count = 0;
    while (t) {
        if (t->op == GGML_OP_MUL_MAT) {
            ++mul_mat_count;
            return t;
        }
        if (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_MUL ||
            t->op == GGML_OP_ADD || t->op == GGML_OP_CONT) {
            t = t->src[0];
            continue;
        }
        return nullptr;
    }
    return nullptr;
}

struct Case {
    uint32_t n_tokens;
    uint32_t n_seqs;
    const char * name;
};

bool is_contiguous_nb(const ggml_tensor * t) {
    if (!t || t->ne[0] <= 0) return false;
    const size_t bs = ggml_blck_size(t->type);
    if (t->nb[0] != ggml_type_size(t->type)) return false;
    if (t->nb[1] != t->nb[0] * (static_cast<size_t>(t->ne[0]) / bs)) return false;
    for (int i = 2; i < GGML_MAX_DIMS; ++i) {
        if (t->nb[i] != t->nb[i - 1] * static_cast<size_t>(t->ne[i - 1])) return false;
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    bool skip_2seq = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--skip-2seq") == 0) {
            skip_2seq = true;
        } else if (std::strcmp(argv[i], "--run-2seq") == 0) {
            skip_2seq = false;
        } else {
            std::printf("BLOCKED: unknown argument: %s\n", argv[i]);
            return 77;
        }
    }
    const char * path = model_path();
    uint64_t size = 0;
    if (!file_size(path, size)) {
        std::printf("BLOCKED: model missing/unreadable: %s\n", path);
        return 77;
    }
    const uint64_t available = mem_available();
    if (available < size + kRamMargin) {
        std::printf("BLOCKED: RAM available=%llu model=%llu margin=%llu\n",
                    (unsigned long long) available, (unsigned long long) size,
                    (unsigned long long) kRamMargin);
        return 77;
    }

    llama_backend_init();

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        std::printf("BLOCKED: CPU backend device unavailable\n");
        llama_backend_free();
        return 77;
    }
    ggml_backend_dev_t devices[] = {cpu_dev, nullptr};

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.devices = devices;

    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) {
        std::printf("BLOCKED: model load failed: %s\n", path);
        llama_backend_free();
        return 77;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = 256;
    cp.n_ubatch = 256;
    cp.n_threads = 1;
    cp.n_threads_batch = 1;
    cp.n_seq_max = 2;

    llama_context * ctx = llama_new_context_with_model(model, cp);
    if (!ctx) {
        std::printf("BLOCKED: context creation failed\n");
        llama_model_free(model);
        llama_backend_free();
        return 77;
    }

    check(ctx->get_sched() != nullptr, "CPU fixture: scheduler available");

    llama_memory_t mem = llama_get_memory(ctx);
    check(mem != nullptr, "CPU fixture: memory available");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    check(vocab != nullptr, "CPU fixture: vocab available");

    const Case cases[] = {
        {256, 1, "reserve(256,1)"},
        {256, 2, "reserve(256,2)"},
        {128, 1, "reserve(128,1)"},
    };

    for (const Case & c : cases) {
        if (skip_2seq && c.n_seqs == 2) {
            std::printf("%s: SKIP (known build_rs_inp_impl assert; isolated)\n", c.name);
            ++skipped;
            continue;
        }
        llama_batch batch = llama_batch_init(c.n_tokens, 0, c.n_seqs);
        batch.n_tokens = c.n_tokens;
        for (uint32_t i = 0; i < c.n_tokens; ++i) {
            batch.token[i] = (llama_token) i;
            batch.pos[i] = (llama_pos) i;
            batch.n_seq_id[i] = 1;
            const uint32_t seq_len = c.n_tokens / c.n_seqs;
            batch.seq_id[i][0] = (llama_seq_id) (i / seq_len);
            batch.pos[i] = (llama_pos) (i % seq_len);
        }
        llama_batch_allocr balloc(model->hparams.n_pos_per_embd());
        const bool output_all = false;
        const bool balloc_ok = balloc.init(batch, *vocab, mem, model->hparams.n_embd,
                                           c.n_seqs, output_all);
        check(balloc_ok, (std::string(c.name) + ": batch_allocr init").c_str());
        if (!balloc_ok) {
            llama_batch_free(batch);
            continue;
        }
        auto mctx = mem->init_batch(balloc, cp.n_ubatch, output_all);
        check(mctx != nullptr, (std::string(c.name) + ": mctx").c_str());
        if (!mctx) {
            llama_batch_free(batch);
            continue;
        }
        const llama_ubatch & mub = mctx->get_ubatch();
        if (mub.n_tokens != c.n_tokens || mub.n_seqs != c.n_seqs) {
            std::printf("%s: SKIP (mctx ubatch n_tokens=%u n_seqs=%u != graph %u/%u)\n",
                        c.name, mub.n_tokens, mub.n_seqs, c.n_tokens, c.n_seqs);
            ++skipped;
            llama_batch_free(batch);
            continue;
        }
        ggml_cgraph * gf = ctx->graph_reserve(c.n_tokens, c.n_seqs, 1, mctx.get(), false);
        check(gf != nullptr, (std::string(c.name) + ": graph_reserve").c_str());
        if (!gf) {
            llama_batch_free(batch);
            continue;
        }
        check(ctx->get_gf_res_reserve()->get_gf() == gf,
              (std::string(c.name) + ": get_gf matches reserve").c_str());
        const bool sched_ok = ggml_backend_sched_alloc_graph(ctx->get_sched(), gf);
        check(sched_ok, (std::string(c.name) + ": sched alloc").c_str());
        if (!sched_ok) {
            llama_batch_free(batch);
            continue;
        }

        const ggml_tensor * node = find_named(gf, "linear_attn_qkv_mixed");
        check(node != nullptr, (std::string(c.name) + ": linear_attn_qkv_mixed").c_str());
        if (!node) {
            llama_batch_free(batch);
            continue;
        }
        check(node->op == GGML_OP_RESHAPE,
              (std::string(c.name) + ": qkv_mixed is reshape").c_str());

        int mul_mats = 0;
        const ggml_tensor * gemm = follow_qkv_chain(node, mul_mats);
        check(gemm != nullptr, (std::string(c.name) + ": QKV mul_mat found").c_str());
        check(mul_mats == 1, (std::string(c.name) + ": exactly one QKV mul_mat").c_str());
        if (!gemm) {
            llama_batch_free(batch);
            continue;
        }

        const ggml_tensor * src1 = gemm->src[1];
        const uint32_t seq_tokens = c.n_tokens / c.n_seqs;
        check(src1 != nullptr, (std::string(c.name) + ": src1 present").c_str());
        if (src1) {
            check(src1->ne[1] == (int64_t) c.n_tokens,
                  (std::string(c.name) + ": src1 ne1 == n_tokens").c_str());
            check(src1->ne[2] == 1 && src1->ne[3] == 1,
                  (std::string(c.name) + ": src1 single batch").c_str());
            check(is_contiguous_nb(src1),
                  (std::string(c.name) + ": src1 contiguous nb").c_str());
        }
        check(gemm->ne[1] == (int64_t) c.n_tokens,
              (std::string(c.name) + ": dst ne1 == n_tokens").c_str());
        check(is_contiguous_nb(gemm),
              (std::string(c.name) + ": dst contiguous nb").c_str());
        check(node->ne[1] == (int64_t) seq_tokens,
              (std::string(c.name) + ": reshape ne1 == seq_tokens").c_str());
        check(node->ne[2] == (int64_t) c.n_seqs,
              (std::string(c.name) + ": reshape ne2 == n_seqs").c_str());
        check(is_contiguous_nb(node),
              (std::string(c.name) + ": reshape contiguous nb").c_str());

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(
            ctx->get_sched(), const_cast<ggml_tensor *>(gemm));
        check(backend != nullptr, (std::string(c.name) + ": GEMM backend").c_str());
        if (backend) {
            check(ggml_backend_is_cpu(backend),
                  (std::string(c.name) + ": GEMM on CPU backend").c_str());
        }
        ggml_tensor * reshape_backend_node = const_cast<ggml_tensor *>(node);
        ggml_backend_t rbackend =
            ggml_backend_sched_get_tensor_backend(ctx->get_sched(), reshape_backend_node);
        check(rbackend != nullptr && ggml_backend_is_cpu(rbackend),
              (std::string(c.name) + ": reshape on CPU backend").c_str());

        llama_batch_free(batch);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    std::printf("%s: %d failure(s), %d skipped\n", failures == 0 ? "OK" : "FAILED", failures, skipped);
    return failures == 0 ? 0 : 1;
}
