// Compare the host and device DFlash2 paths through draft KV state and lazy host taps.
#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    const bool cpu_fc = argc == 4 && std::strcmp(argv[3], "--cpu-fc") == 0;
    if (argc != 3 && !cpu_fc) {
        std::fprintf(stderr, "usage: %s target.gguf drafter.gguf [--cpu-fc]\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    ggml_backend_load_all();
    ggml_backend_dev_t devices[] = {ggml_backend_dev_by_name("CUDA0"), nullptr};
    if (!devices[0]) {
        std::fprintf(stderr, "CUDA device unavailable, skipping\n");
        llama_backend_free();
        return 77;
    }
    auto mparams = llama_model_default_params();
    mparams.devices = devices;
    mparams.n_gpu_layers = 999;
    auto * target = llama_model_load_from_file(argv[1], mparams);
    const llama_model_tensor_buft_override overrides[] = {
        {"^fc[.]weight$", ggml_backend_cpu_buffer_type()},
        {nullptr, nullptr},
    };
    if (cpu_fc) {
        mparams.no_host = true;
        mparams.use_extra_bufts = false;
        mparams.tensor_buft_overrides = overrides;
    }
    auto * draft = llama_model_load_from_file(argv[2], mparams);
    GGML_ASSERT(target && draft);
    constexpr int rows = 128;
    auto cparams = llama_context_default_params();
    cparams.n_ctx = 256;
    cparams.n_batch = rows;
    cparams.n_ubatch = rows;
    cparams.n_seq_max = 1;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    auto * ctx_tgt = llama_init_from_model(target, cparams);
    constexpr int draft_rows = 48;
    cparams.n_ubatch = draft_rows;
    cparams.ctx_other = ctx_tgt;
    if (cpu_fc) {
        cparams.op_offload = false;
    }
    auto * ctx_dft = llama_init_from_model(draft, cparams);
    GGML_ASSERT(ctx_tgt && ctx_dft);
    const int n_embd = llama_model_n_embd(target);
    const int n_layers = llama_model_target_layer_ids_n(draft);
    const int32_t * layers = llama_model_target_layer_ids(draft);
    const int width = n_embd*n_layers;
    GGML_ASSERT(n_layers == 2 && layers[0] == 1 && layers[1] == 0);
    for (int k = 0; k < n_layers; ++k) {
        llama_set_embeddings_layer_inp(ctx_tgt, layers[k], true);
    }

    auto tb = llama_batch_init(rows, 0, 1);
    auto db = llama_batch_init(draft_rows, width, 1);
    // M-RoPE embedding batches use four section-major position arrays.
    std::free(db.pos);
    db.pos = static_cast<llama_pos *>(std::calloc(draft_rows*4, sizeof(llama_pos)));
    GGML_ASSERT(db.pos);
    tb.n_tokens = rows;
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(target));
    for (int i = 0; i < rows; ++i) {
        tb.token[i] = (3 + i*7) % vocab;
        tb.pos[i] = i;
        tb.n_seq_id[i] = 1;
        tb.seq_id[i][0] = 0;
        tb.logits[i] = 0;
    }
    GGML_ASSERT(llama_decode(ctx_tgt, tb) == 0);
    std::vector<float> host_taps(rows*width);
    for (int k = 0; k < n_layers; ++k) {
        const float * tap = llama_get_embeddings_layer_inp(ctx_tgt, layers[k]);
        GGML_ASSERT(tap);
        for (int i = 0; i < rows; ++i) {
            std::memcpy(host_taps.data() + i*width + k*n_embd, tap + i*n_embd, n_embd*sizeof(float));
        }
    }
    GGML_ASSERT(std::memcmp(host_taps.data(), host_taps.data() + n_embd, n_embd*sizeof(float)) != 0);

    auto batch_at = [&](int start) {
        db.n_tokens = std::min(draft_rows, rows - start);
        for (int i = 0; i < db.n_tokens; ++i) {
            db.pos[i] = db.pos[db.n_tokens + i] = db.pos[2*db.n_tokens + i] = start + i;
            db.pos[3*db.n_tokens + i] = 0;
            db.n_seq_id[i] = 1;
            db.seq_id[i][0] = 0;
            db.logits[i] = 0;
        }
    };
    for (int start = 0; start < rows; start += draft_rows) {
        batch_at(start);
        std::memcpy(db.embd, host_taps.data() + start*width, db.n_tokens*width*sizeof(float));
        GGML_ASSERT(llama_decode(ctx_dft, db) == 0);
    }
    GGML_ASSERT(llama_memory_seq_pos_max(llama_get_memory(ctx_dft), 0) == rows - 1);
    std::vector<uint8_t> wanted(llama_state_seq_get_size(ctx_dft, 0));
    GGML_ASSERT(!wanted.empty() && llama_state_seq_get_data(ctx_dft, wanted.data(), wanted.size(), 0) == wanted.size());

    llama_memory_clear(llama_get_memory(ctx_tgt), true);
    llama_memory_clear(llama_get_memory(ctx_dft), true);
    llama_prepare_dflash2_prefill_capture(ctx_tgt, ctx_dft, layers, n_layers, rows);
    GGML_ASSERT(llama_decode(ctx_tgt, tb) == 0);
    // A successful device injection must ignore host placeholder bytes.
    std::fill_n(db.embd, draft_rows*width, 0.0f);
    int32_t rc = 0;
    batch_at(draft_rows);
    GGML_ASSERT(!llama_decode_dflash2_prefill_capture(ctx_dft, ctx_tgt, db, draft_rows, &rc));
    for (int start = 0; start < rows; start += draft_rows) {
        batch_at(start);
        GGML_ASSERT(llama_decode_dflash2_prefill_capture(ctx_dft, ctx_tgt, db, start, &rc) && rc == 0);
    }
    GGML_ASSERT(!llama_decode_dflash2_prefill_capture(ctx_dft, ctx_tgt, db, 0, &rc));
    for (int k = 0; k < n_layers; ++k) {
        const float * tap = llama_get_embeddings_layer_inp(ctx_tgt, layers[k]);
        GGML_ASSERT(tap);
        for (int i = 0; i < rows; ++i) {
            GGML_ASSERT(std::memcmp(host_taps.data() + i*width + k*n_embd, tap + i*n_embd, n_embd*sizeof(float)) == 0);
        }
    }
    GGML_ASSERT(llama_memory_seq_pos_max(llama_get_memory(ctx_dft), 0) == rows - 1);
    std::vector<uint8_t> actual(llama_state_seq_get_size(ctx_dft, 0));
    GGML_ASSERT(llama_state_seq_get_data(ctx_dft, actual.data(), actual.size(), 0) == actual.size());
    GGML_ASSERT(actual == wanted);

    tb.n_tokens = db.n_tokens = 1;
    tb.pos[0] = rows;
    GGML_ASSERT(llama_decode(ctx_tgt, tb) == 0);
    GGML_ASSERT(!llama_decode_dflash2_prefill_capture(ctx_dft, ctx_tgt, db, 0, &rc));
    std::printf("device/host DFlash2 KV state equal for %d rows, %zu bytes; lazy taps and short fallback passed\n", rows, actual.size());

    llama_batch_free(db);
    llama_batch_free(tb);
    llama_free(ctx_dft);
    llama_free(ctx_tgt);
    llama_model_free(draft);
    llama_model_free(target);
    llama_backend_free();
}
