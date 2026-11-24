// Qwen4Exp (hyper-connections) DFlash2 target-feature capture, CPU-only.
//
// The DFlash drafter consumes target hidden states from the INPUT of layers
// listed in `dflash.target_layers` (HF layer id + 1). On Qwen4Exp the residual
// lives in `hc` hyper-connection streams and the reference (SGLang patch
// qwen38-dflash2-capture.patch) taps the COMPLETED layer output averaged over
// the streams. `src/models/qwen4exp.cpp` publishes exactly that mean as
// t_layer_inp[il], guarded by cparams.embeddings_layer_inp[il] (default off).
//
// Fixture: tests/gen-tiny-qwen4exp-hc.py. Layer 1 of the target is all-zero, so
// its block is a no-op and the residual after layer 0 reaches t_h_nextn
// unchanged. The test therefore has an independent oracle for the tap:
//     tap(il=1) == mean over hc streams of llama_get_embeddings_nextn()
// checked in double precision, per token row.
//
// Covered: prefill (multi-token), decode (single token), two sequences in one
// batch, rollback + re-decode, and the real common_speculative DFlash2 handoff
// (target taps -> process() -> drafter KV -> selector draft).

#include "llama.h"
#include "llama-ext.h"
#include "speculative.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void fail(const char * what) {
    fprintf(stderr, "FAIL: %s\n", what);
    exit(1);
}

static void check_taps_vs_nextn(llama_context * ctx, const char * phase, int n_tokens) {
    const int n_embd = llama_model_n_embd(llama_get_model(ctx));
    const int hc     = llama_model_n_embd_out(llama_get_model(ctx)) / n_embd;
    if (hc <= 1) {
        fail("fixture did not load hyper-connections");
    }

    const float * tap = llama_get_embeddings_layer_inp(ctx, 1);
    const float * hn  = llama_get_embeddings_nextn(ctx);
    if (!tap || !hn) {
        fail("capture buffers are null (tap not enabled or graph did not emit h_nextn)");
    }

    double max_err = 0.0;
    double max_spread = 0.0;
    for (int i = 0; i < n_tokens; ++i) {
        for (int j = 0; j < n_embd; ++j) {
            double mean = 0.0;
            double lo = 1e30;
            double hi = -1e30;
            for (int s = 0; s < hc; ++s) {
                const double v = hn[(size_t) i*hc*n_embd + (size_t) s*n_embd + j];
                mean += v;
                lo = std::min(lo, v);
                hi = std::max(hi, v);
            }
            mean /= hc;
            const double got = tap[(size_t) i*n_embd + j];
            max_err = std::max(max_err, std::fabs(mean - got));
            max_spread = std::max(max_spread, hi - lo);
        }
    }

    printf("%s: tokens=%d hc=%d max|tap-mean| = %.3e, max stream spread = %.3e\n",
            phase, n_tokens, hc, max_err, max_spread);
    if (max_err > 1e-6) {
        fail("tap is not the HC stream mean");
    }
    if (max_spread <= 1e-4) {
        fail("HC streams are identical; the mean check is vacuous");
    }
}

static int decode_tokens(llama_context * ctx, const std::vector<llama_token> & toks,
        const std::vector<llama_pos> & pos, const std::vector<llama_seq_id> & seq,
        bool logits_all) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 2);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i]    = toks[i];
        b.pos[i]      = pos[i];
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = seq[i];
        b.logits[i]   = logits_all || i + 1 == toks.size();
    }
    b.n_tokens = (int32_t) toks.size();
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <qwen4exp-target.gguf> <dflash2-draft.gguf>\n", argv[0]);
        return 2;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model * model_tgt = llama_model_load_from_file(argv[1], mparams);
    llama_model * model_dft = llama_model_load_from_file(argv[2], mparams);
    if (!model_tgt) fail("cannot load target fixture");
    if (!model_dft) fail("cannot load draft fixture");

    const int n_embd = llama_model_n_embd(model_tgt);
    if (llama_model_n_embd(model_dft) != n_embd) {
        fail("draft/target embedding width mismatch");
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 64;
    cparams.n_batch   = 16;
    cparams.n_ubatch  = 16;
    cparams.n_seq_max = 2;
    // the server sizes the recurrent rollback window from the draft depth
    cparams.n_rs_seq  = 7;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;

    llama_context * ctx_tgt = llama_init_from_model(model_tgt, cparams);
    if (!ctx_tgt) fail("cannot create target context");

    // ---- 0. enabling the tap must not change the target's own arithmetic ----
    {
        llama_context_params oparams = cparams;
        oparams.n_seq_max = 1;
        llama_context * octx = llama_init_from_model(model_tgt, oparams);
        if (!octx) fail("cannot create tap-off context");

        const std::vector<llama_token> toks = {3, 7, 11, 5};
        const std::vector<llama_pos>   pos  = {0, 1, 2, 3};
        const std::vector<llama_seq_id> seq(toks.size(), 0);
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));

        if (decode_tokens(octx, toks, pos, seq, true) != 0) fail("tap-off decode failed");
        std::vector<float> logits_off((size_t) n_vocab * toks.size());
        for (size_t i = 0; i < toks.size(); ++i) {
            std::memcpy(logits_off.data() + i*n_vocab, llama_get_logits_ith(octx, (int32_t) i),
                    (size_t) n_vocab * sizeof(float));
        }

        llama_set_embeddings_layer_inp(octx, 1, true);
        llama_memory_clear(llama_get_memory(octx), false);
        if (decode_tokens(octx, toks, pos, seq, true) != 0) fail("tap-on decode failed");

        for (size_t i = 0; i < toks.size(); ++i) {
            const float * got = llama_get_logits_ith(octx, (int32_t) i);
            if (std::memcmp(logits_off.data() + i*n_vocab, got, (size_t) n_vocab * sizeof(float)) != 0) {
                fail("enabling the layer tap changed the target logits");
            }
        }
        printf("tap-off: target logits bit-identical with the capture enabled\n");

        llama_free(octx);
    }

    // h_nextn is the widened (hc * n_embd) residual; unmasked keeps one row per token.
    llama_set_embeddings_nextn(ctx_tgt, true, false);
    llama_set_embeddings_layer_inp(ctx_tgt, 1, true);

    // ---- 1. prefill ----
    {
        const std::vector<llama_token> toks = {3, 7, 11, 5};
        const std::vector<llama_pos>   pos  = {0, 1, 2, 3};
        const std::vector<llama_seq_id> seq = {0, 0, 0, 0};
        if (decode_tokens(ctx_tgt, toks, pos, seq, true) != 0) fail("prefill decode failed");
        check_taps_vs_nextn(ctx_tgt, "prefill", (int) toks.size());
    }

    // ---- 2. ubatch split: one decode call, n_ubatch < n_tokens ----
    {
        llama_context_params uparams = cparams;
        uparams.n_seq_max = 1;
        // the rollback window (n_rs_seq + 1 = 8) must fit in one ubatch
        uparams.n_ubatch  = 10;
        int64_t projected_rows = 0;
        uparams.cb_eval_user_data = &projected_rows;
        uparams.cb_eval = [](ggml_tensor * t, bool ask, void * data) {
            if (ask && std::strcmp(t->name, "result_output") == 0) {
                *static_cast<int64_t *>(data) += t->ne[1];
            }
            return false;
        };
        llama_context * uctx = llama_init_from_model(model_tgt, uparams);
        if (!uctx) fail("cannot create ubatch target context");
        llama_set_embeddings_nextn(uctx, true, false);
        llama_set_embeddings_layer_inp(uctx, 1, true);

        const std::vector<llama_token> toks = {3, 7, 11, 5, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59};
        const std::vector<llama_pos>   pos  = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        const std::vector<llama_seq_id> seq(toks.size(), 0);
        if (decode_tokens(uctx, toks, pos, seq, true) != 0) fail("ubatch prefill failed");
        if (projected_rows != (int64_t) toks.size()) fail("ubatch full logits were not computed");
        check_taps_vs_nextn(uctx, "ubatch", (int) toks.size());

        llama_memory_clear(llama_get_memory(uctx), false);
        projected_rows = 0;
        if (decode_tokens(uctx, toks, pos, seq, false) != 0) fail("ubatch prefill with last-token logits failed");
        if (projected_rows != 1) fail("ubatch projected unrequested logits");
        check_taps_vs_nextn(uctx, "ubatch last-token logits", (int) toks.size());

        llama_free(uctx);
    }

    // ---- 3. decode one token ----
    {
        if (decode_tokens(ctx_tgt, {13}, {4}, {0}, true) != 0) fail("decode failed");
        check_taps_vs_nextn(ctx_tgt, "decode", 1);
    }

    // ---- 4. rollback the rejected suffix and re-decode the same position ----
    {
        if (!llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, 4, -1)) {
            fail("rollback of seq 0 failed");
        }
        if (decode_tokens(ctx_tgt, {31}, {4}, {0}, true) != 0) fail("re-decode failed");
        check_taps_vs_nextn(ctx_tgt, "rollback", 1);
    }

    // ---- 5. two sequences in one batch ----
    {
        llama_memory_clear(llama_get_memory(ctx_tgt), false);
        const std::vector<llama_token> toks = {17, 19, 23, 29};
        const std::vector<llama_pos>   pos  = {0, 0, 1, 1};
        const std::vector<llama_seq_id> seq = {0, 1, 0, 1};
        if (decode_tokens(ctx_tgt, toks, pos, seq, true) != 0) fail("two-sequence decode failed");
        check_taps_vs_nextn(ctx_tgt, "two-seq", (int) toks.size());
    }

    // ---- 6. real DFlash2 spec handoff ----
    {
        llama_memory_clear(llama_get_memory(ctx_tgt), false);

        llama_context_params dparams = llama_context_default_params();
        dparams.n_ctx      = 64;
        dparams.n_batch    = 8;
        dparams.n_ubatch   = 8;
        dparams.n_seq_max  = 1;
        dparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        dparams.ctx_other  = ctx_tgt;

        llama_context * ctx_dft = llama_init_from_model(model_dft, dparams);
        if (!ctx_dft) fail("cannot create draft context (ctx_other handoff)");

        common_params_speculative sparams;
        sparams.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH };
        sparams.draft.ctx_tgt = ctx_tgt;
        sparams.draft.ctx_dft = ctx_dft;
        sparams.draft.n_max   = 7;
        sparams.draft.n_min   = 0;
        sparams.draft.p_min   = 0.0f;
        sparams.draft.backend_sampling = false;

        common_speculative * spec = common_speculative_init(sparams, 1);
        if (!spec) fail("common_speculative_init failed");

        const llama_tokens prompt = {3, 7, 11, 5, 13};
        common_batch batch(ctx_tgt);
        for (size_t i = 0; i < prompt.size(); ++i) {
            batch.add(prompt[i], (llama_pos) i, 0, i + 1 == prompt.size());
        }

        llama_memory_clear(llama_get_memory(ctx_tgt), false);

        // explicit target decode in the same order common_batch reports
        {
            llama_batch b = llama_batch_init((int32_t) prompt.size(), 0, 1);
            for (size_t i = 0; i < prompt.size(); ++i) {
                b.token[i]     = prompt[i];
                b.pos[i]       = (llama_pos) i;
                b.n_seq_id[i]  = 1;
                b.seq_id[i][0] = 0;
                b.logits[i]    = i + 1 == prompt.size();
            }
            b.n_tokens = (int32_t) prompt.size();
            if (llama_decode(ctx_tgt, b) != 0) fail("spec prompt decode failed");
            llama_batch_free(b);
        }
        check_taps_vs_nextn(ctx_tgt, "spec-prefill", (int) prompt.size());

        common_speculative_begin(spec, 0, prompt);

        auto & dp = common_speculative_get_draft_params(spec, 0);
        llama_tokens result;
        dp.drafting = true;
        dp.pos0     = (llama_pos) prompt.size();
        dp.id_last  = prompt.back();
        dp.prompt   = &prompt;
        dp.result   = &result;
        dp.n_max    = 7;

        if (!common_speculative_process(spec, batch)) {
            fail("common_speculative_process failed");
        }
        if (!common_speculative_draft(spec)) {
            fail("common_speculative_draft failed");
        }
        if (result.empty()) {
            fail("DFlash2 produced an empty draft");
        }
        if ((int) result.size() > 7) {
            fail("DFlash2 exceeded the block/n_max window");
        }
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_dft));
        for (llama_token t : result) {
            if (t < 0 || t >= n_vocab) {
                fail("draft token out of vocab");
            }
        }
        printf("spec: drafted %zu tokens from a %zu-token prompt (first id %d)\n",
                result.size(), prompt.size(), result.front());

        common_speculative_free(spec);
        llama_free(ctx_dft);
    }

    llama_free(ctx_tgt);
    llama_model_free(model_dft);
    llama_model_free(model_tgt);
    llama_backend_free();
    printf("OK: Qwen4Exp HC target capture and DFlash2 handoff\n");
    return 0;
}
