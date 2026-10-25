// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "../src/bridge/state_identity.h"
#include "xllama/api_policy.h"
#include "xllama/path_utils.h"
#include "xllama/session.h"
#include "xllama/session_hub.h"
#include <cmath>

#include <algorithm>
#include <array>
#include <chrono>
#include <doctest/doctest.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <thread>
#include <vector>

#ifdef __linux__
    #include <chrono>
    #include <csignal>
    #include <cstdio>
    #include <fcntl.h>
    #include <sys/inotify.h>
    #include <sys/resource.h>
    #include <sys/stat.h>
    #include <sys/wait.h>
    #include <unistd.h>
#endif

TEST_CASE("SessionHub: failed create leaves the hub empty and bumps generation") {
    auto& hub = xllama::session_hub();
    std::lock_guard<std::mutex> lk(hub.mtx);
    const uint64_t g0 = hub.generation;
    xllama::SessionParams sp;
    sp.model_path = "/nonexistent/hub-model.gguf";
    std::string err;
    CHECK(hub.ensure_locked("hub-model", sp, &err) == nullptr);
    CHECK(hub.session == nullptr);
    CHECK(hub.model.empty());
    // The attempted swap must invalidate any holder's generation even though
    // creation failed — its old session pointer is gone either way.
    CHECK(hub.generation == g0 + 1);
    hub.reset_locked(); // no-op when already empty
    CHECK(hub.generation == g0 + 1);
}

TEST_CASE("resolve_max_length: the #130 ladder has one home") {
    using xllama::resolve_max_length;
    // override < 0: saturate to n_ctx — what Session always requests.
    CHECK(resolve_max_length(2048, 1289, 256, -1) == 2048);
    CHECK(resolve_max_length(3072, 10, 96, -1) == 3072);
    // override == 0: derive min(n_ctx, prompt + n_predict) — bench default,
    // keeps every historical row's meaning.
    CHECK(resolve_max_length(2048, 1289, 256, 0) == 1545); // the shipping default
                                                           // that landed in the valley
    CHECK(resolve_max_length(2048, 1800, 512, 0) == 2048); // clamped to context
    // override > 0: explicit, clamped to (n_prompt, n_ctx].
    CHECK(resolve_max_length(2048, 1289, 256, 1801) == 1801);
    CHECK(resolve_max_length(2048, 1289, 256, 100) == 1290);  // floor: prompt+1
    CHECK(resolve_max_length(2048, 1289, 256, 9999) == 2048); // ceiling: n_ctx
}

TEST_CASE("Session::create rejects non-existent model path") {
    xllama::SessionParams sp;
    sp.model_path = "/nonexistent/path/model.gguf";
    std::string err;
    auto s = xllama::Session::create(sp, &err);
    CHECK(s == nullptr);
    CHECK(!err.empty());
}

TEST_CASE("Session::create rejects empty model path") {
    xllama::SessionParams sp;
    sp.model_path = "";
    std::string err;
    auto s = xllama::Session::create(sp, &err);
    CHECK(s == nullptr);
    CHECK(!err.empty());
}

TEST_CASE("model_uses_llama_backend detects .gguf suffix and dir layout") {
    using xllama::model_uses_llama_backend;

    CHECK(model_uses_llama_backend("foo.gguf"));
    CHECK(model_uses_llama_backend("/abs/path/model.Q4_K.gguf"));
    CHECK_FALSE(model_uses_llama_backend("smollm2-360m"));
    CHECK_FALSE(model_uses_llama_backend("some-ort-model-dir"));

    // Temp dir containing a .gguf file (simulates catalogue layout on disk)
    auto tmp = std::filesystem::temp_directory_path() / "xllama-gguf-layout-test";
    std::filesystem::create_directories(tmp);
    {
        std::ofstream f((tmp / "model.gguf").string());
        f << "dummy";
    }
    CHECK(model_uses_llama_backend(tmp.string()));

    // Dir without .gguf should be false
    auto tmp2 = std::filesystem::temp_directory_path() / "xllama-no-gguf";
    std::filesystem::create_directories(tmp2);
    CHECK_FALSE(model_uses_llama_backend(tmp2.string()));

    // Cleanup
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::remove_all(tmp2, ec);
}

TEST_CASE("first_gguf_in_dir prefers model.gguf over adapter.gguf") {
    using xllama::first_gguf_in_dir;
    auto tmp = std::filesystem::temp_directory_path() / "xllama-gguf-prefer-test";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp);
    {
        std::ofstream a((tmp / "adapter.gguf").string());
        a << "adapter";
        std::ofstream m((tmp / "model.gguf").string());
        m << "model";
    }
    const std::string picked = first_gguf_in_dir(tmp.string(), "adapter.gguf");
    CHECK(picked.find("model.gguf") != std::string::npos);
    // Without exclude, model.gguf still preferred by name
    const std::string picked2 = first_gguf_in_dir(tmp.string());
    CHECK(picked2.find("model.gguf") != std::string::npos);
    std::filesystem::remove_all(tmp, ec);
}

TEST_CASE("Session::create fails when LoRA path is invalid (llama)") {
    // Real base not required: create fails at model load first if path is junk.
    // Here we only assert empty model + lora still rejects cleanly.
    xllama::SessionParams sp;
    sp.model_path = "/nonexistent/base.gguf";
    sp.lora_path = "/nonexistent/adapter.gguf";
    sp.backend = xllama::Backend::LlamaCpp;
    std::string err;
    auto s = xllama::Session::create(sp, &err);
    CHECK(s == nullptr);
    CHECK(!err.empty());
}

TEST_CASE("Session::create Auto with explicit Backend::LlamaCpp on GGUF layout") {
    // Even without real weights the dispatch must select the llama path.
    // We only assert that it does not succeed via the ORT path and that an error is produced.
    auto tmp = std::filesystem::temp_directory_path() / "xllama-session-gguf-auto";
    std::filesystem::create_directories(tmp);
    {
        std::ofstream f((tmp / "mini.gguf").string());
        f << "not-a-real-gguf";
    }

    xllama::SessionParams sp;
    sp.model_path = tmp.string();
    sp.backend = xllama::Backend::Auto; // should resolve to Llama via layout

    std::string err;
    auto s = xllama::Session::create(sp, &err);
    // On Linux this build only has llama; we expect a load failure from the llama code path.
    CHECK(s == nullptr);
    CHECK(!err.empty());

    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
}

// #170a: a full-prompt turn rewinds the resident KV to the common token prefix
// instead of clearing. Opt-in — needs a real GGUF:
//   XLLAMA_TEST_MODEL=/path/to/model.gguf ./xllama-tests
TEST_CASE("Session: KV prefix reuse collapses a repeated prefill (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env)
        return;

    xllama::SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 512;
    std::string err;
    auto sess = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(sess, err);

    auto mkgp = [](const std::string& p) {
        xllama::GenerateParams gp;
        gp.prompt = p;
        gp.n_predict = 12;
        gp.temperature = 0.0f; // greedy: outputs must be reproducible
        return gp;
    };
    const std::string prompt = "The capital of France is";

    const auto r_cold = sess->generate(mkgp(prompt));
    REQUIRE(r_cold.success);
    REQUIRE(r_cold.n_p_eval > 1);

    // Regenerate: identical prompt on the same session. Two legal regimes:
    // pure-attention caches rewind and re-prefill exactly 1 token (the logits
    // source); hybrid caches (LFM2) cannot erase a partial tail — seq_rm
    // returns false — so the session degrades to a full clear + re-prefill.
    // Byte-identical output is the correctness bar in both regimes.
    const auto r_regen = sess->generate(mkgp(prompt));
    REQUIRE(r_regen.success);
    CHECK(r_regen.n_p_eval <= r_cold.n_p_eval);
    CHECK(r_regen.output_text == r_cold.output_text);

    // Extended prompt (the LAN-API shape). This prompt diverges from the
    // resident generated tail right after the original prompt, so it exercises
    // the tail-removal path in both regimes.
    const bool full_rewind = r_regen.n_p_eval == 1;
    const std::string ext = prompt + " Paris, and the capital of Italy is";
    const auto r_ext = sess->generate(mkgp(ext));
    REQUIRE(r_ext.success);

    auto fresh = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(fresh, err);
    const auto r_fresh = fresh->generate(mkgp(ext));
    REQUIRE(r_fresh.success);
    if (full_rewind) {
        // Pure-attention: only the extension past the common prefix re-prefills.
        CHECK(r_ext.n_p_eval < sess->count_tokens(ext));
        // Exact equality is the wrong bar here (it holds for the regenerate
        // case above, where the resident bytes are identical): the resident
        // prefix was accumulated in a different batch shape than the fresh
        // session's single prefill, so the K/V last bits differ and greedy can
        // flip at a NEAR-TIE downstream — inherent to every prefix cache,
        // observed as " Rome." vs " Rome.\n\nThe answer is Paris.". The
        // correctness signal is the leading strong-signal token agreeing.
        REQUIRE(r_fresh.output_text.size() >= 5);
        REQUIRE(r_ext.output_text.size() >= 5);
        CHECK(r_ext.output_text.substr(0, 5) == r_fresh.output_text.substr(0, 5));
    } else {
        // Hybrid (LFM2): the divergent tail cannot be erased — seq_rm refuses —
        // so the session must have degraded to a full clear + single-batch
        // re-prefill, which is exactly a fresh session: byte-identical output.
        CHECK(r_ext.output_text == r_fresh.output_text);
    }
}

// A prompt longer than the logical batch must be prefilled in chunks. This is
// not a "nicer error" test: llama_decode ASSERTS n_tokens <= n_batch (GGML_ABORT,
// Release included), so before the chunking this test process died. n_batch
// defaults to min(n_ctx, 2048), so the regime only exists above the shipping
// context — which is exactly what the coding catalogue opens (n_ctx 4096, ceiling
// 3846 prompt tokens). Opt-in — needs a real GGUF:
//   XLLAMA_TEST_MODEL=/path/to/model.gguf ./xllama-tests
TEST_CASE("Session: a prompt past n_batch prefills in chunks (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env)
        return;

    // Same regime as a 4096-token coding session with the default 2048 batch,
    // at 1/30th of the tokens: what matters is prompt > n_batch, and n_batch is
    // a knob. A 2100-token prefill in a Debug build takes minutes; this runs in
    // seconds and fails identically (an abort, not a failed assertion) without
    // the chunking.
    xllama::SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 512;
    sp.n_batch = 64;
    std::string err;
    auto sess = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(sess, err);

    // Grow past n_batch with the model's own tokenizer, so the test does not
    // depend on one vocabulary's density.
    std::string prompt = "Summarize the following log.\n";
    int n_prompt = 0;
    for (int i = 0; i < 200 && n_prompt <= 96; ++i) {
        prompt += "line " + std::to_string(i) + ": nothing happened\n";
        if ((i % 4) == 3)
            n_prompt = sess->count_tokens(prompt);
    }
    n_prompt = sess->count_tokens(prompt);
    REQUIRE(n_prompt > sp.n_batch); // past the logical batch: the regime under test
    REQUIRE(n_prompt < 400);        // still inside n_ctx with room to generate

    xllama::GenerateParams gp;
    gp.prompt = prompt;
    gp.n_predict = 8;
    gp.temperature = 0.0f; // greedy: the two sessions below must agree
    const auto r = sess->generate(gp);
    REQUIRE_MESSAGE(r.success, r.error_msg); // before the chunking: SIGABRT here
    CHECK(r.n_p_eval == n_prompt);           // every token was prefilled, not just the first chunk

    // And chunking must not change what the model reads: the same prompt in ONE
    // batch (default n_batch, well above this prompt) sees the same context.
    // Only the leading token is compared — a different batch shape moves the
    // last K/V bits, so greedy can flip at a near-tie downstream (same caveat as
    // the #170a prefix-reuse case above).
    sp.n_batch = 0;
    auto whole = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(whole, err);
    const auto r_whole = whole->generate(gp);
    REQUIRE(r_whole.success);
    CHECK(r_whole.n_p_eval == n_prompt);
    CHECK(r.n_eval == r_whole.n_eval);
    if (!r.output_text.empty() && !r_whole.output_text.empty())
        CHECK(r.output_text.substr(0, 1) == r_whole.output_text.substr(0, 1));
}

// A full prompt that cannot fit the context has nothing to evict — the trimmer
// upstream owns that — so it must fail fast rather than reach the decode, where
// an over-context batch is an abort. Opt-in:
//   XLLAMA_TEST_MODEL=/path/to/model.gguf ./xllama-tests
TEST_CASE("Session: a prompt past n_ctx fails fast (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env)
        return;

    xllama::SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 512;
    std::string err;
    auto sess = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(sess, err);

    std::string prompt;
    for (int i = 0; i < 400; ++i)
        prompt += "token " + std::to_string(i) + " ";
    REQUIRE(sess->count_tokens(prompt) > 512);

    xllama::GenerateParams gp;
    gp.prompt = prompt;
    gp.n_predict = 8;
    const auto r = sess->generate(gp);
    CHECK_FALSE(r.success);
    CHECK(r.error_msg.find("prompt too long") != std::string::npos);
    CHECK(r.n_eval == 0);
}

// #170b: the resident KV round-trips through a file, so a conversation the
// session no longer holds resumes without re-reading its history. Opt-in —
// needs a real GGUF:
//   XLLAMA_TEST_MODEL=/path/to/model.gguf ./xllama-tests
TEST_CASE("Session: KV state round-trips through a file (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env)
        return;

    const auto state_path = std::filesystem::temp_directory_path() / "xllama-kv-state.bin";
    std::error_code ec;
    std::filesystem::remove(state_path, ec);

    xllama::SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 2048; // the shipping context: file size scales with it
    std::string err;
    auto a = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(a, err);

    auto mkgp = [](const std::string& p, bool reset) {
        xllama::GenerateParams gp;
        gp.prompt = p;
        gp.n_predict = 16;
        gp.temperature = 0.0f; // greedy: the two sessions must agree exactly
        gp.reuse_kv = true;
        gp.reset_kv = reset;
        return gp;
    };
    // A conversation of realistic length: re-reading exactly this is the cost
    // the file exists to avoid, and its size is what the eviction policy has
    // to budget for.
    std::string filler;
    for (int i = 0; i < 80; ++i)
        filler += "Item " + std::to_string(i) +
                  ": the harbour master filed a report on tide tables and cargo manifests. ";
    const std::string seed_prompt =
        "<|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\n" + filler +
        "The capital of France is Paris. The capital of Italy is Rome. Remember "
        "both.<|im_end|>\n<|im_start|>assistant\n";
    const std::string follow = "<|im_end|>\n<|im_start|>user\nWhich city did I name "
                               "first?<|im_end|>\n<|im_start|>assistant\n";

    const auto r_seed = a->generate(mkgp(seed_prompt, /*reset=*/true));
    REQUIRE_MESSAGE(r_seed.success, r_seed.error_msg);

    const auto t0 = std::chrono::steady_clock::now();
    const bool saved = a->save_state(state_path.string(), &err);
    const double save_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    REQUIRE_MESSAGE(saved, err);
    const auto bytes = std::filesystem::file_size(state_path, ec);
    MESSAGE("state file: " << bytes / 1024 << " KiB for " << r_seed.n_p_eval << " prompt tokens ("
                           << bytes / 1024 / (r_seed.n_p_eval ? r_seed.n_p_eval : 1)
                           << " KiB/token), save " << save_ms << " ms, prefill was "
                           << r_seed.t_p_eval_ms << " ms");

    // The continuation the resident session would produce, for comparison.
    const auto r_cont_resident = a->generate(mkgp(follow, /*reset=*/false));
    REQUIRE_MESSAGE(r_cont_resident.success, r_cont_resident.error_msg);

    // A second session that never saw the conversation: restoring the file
    // must put it in the same state, not merely a similar one.
    auto b = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(b, err);
    const auto t1 = std::chrono::steady_clock::now();
    const bool loaded = b->load_state(state_path.string(), &err);
    const double load_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    REQUIRE_MESSAGE(loaded, err);
    MESSAGE("load " << load_ms << " ms");

    const auto r_cont_restored = b->generate(mkgp(follow, /*reset=*/false));
    REQUIRE_MESSAGE(r_cont_restored.success, r_cont_restored.error_msg);
    // Only the follow-up is prefilled — the history came from the file.
    CHECK(r_cont_restored.n_p_eval == r_cont_resident.n_p_eval);
    CHECK(r_cont_restored.n_p_eval < r_seed.n_p_eval);
    CHECK(r_cont_restored.output_text == r_cont_resident.output_text);

    // A file whose fingerprint does not match must be refused, not fed to the
    // cache: the pin checks shape only, so this is the whole defence against
    // loading another model's KV.
    xllama::SessionParams sp2 = sp;
    sp2.n_ctx = 1024; // same weights, different context configuration
    auto c = xllama::Session::create(sp2, &err);
    REQUIRE_MESSAGE(c, err);
    err.clear();
    CHECK_FALSE(c->load_state(state_path.string(), &err));
    CHECK(!err.empty());
    // Refusal must leave the session usable.
    const auto r_after = c->generate(mkgp("<|im_start|>user\nHi.<|im_end|>\n"
                                          "<|im_start|>assistant\n",
                                          /*reset=*/true));
    CHECK(r_after.success);

    std::filesystem::remove(state_path, ec);
}

// #169: a continuation turn that would overflow n_ctx evicts the oldest
// resident tokens past n_keep (front-drop + RoPE shift) instead of failing,
// so a long chat never leaves the reuse regime. Opt-in — needs a real GGUF:
//   XLLAMA_TEST_MODEL=/path/to/model.gguf ./xllama-tests
TEST_CASE(
    "Session: context shift keeps continuations alive past n_ctx (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env)
        return;

    xllama::SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 256; // small on purpose: a few turns must overflow it
    std::string err;
    auto sess = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(sess, err);

    const std::string sys = "<|im_start|>system\nYou are terse.<|im_end|>\n";
    const int n_keep = sess->count_tokens(sys);
    REQUIRE(n_keep > 0);

    // Seed the persistent context (reuse + reset).
    xllama::GenerateParams seed;
    seed.prompt = sys + "<|im_start|>user\nSay ok.<|im_end|>\n<|im_start|>assistant\n";
    seed.n_predict = 32;
    seed.temperature = 0.0f;
    seed.reuse_kv = true;
    seed.reset_kv = true;
    seed.n_keep = n_keep;
    const auto r_seed = sess->generate(seed);
    REQUIRE_MESSAGE(r_seed.success, r_seed.error_msg);

    // Enough continuation turns to overflow n_ctx=256 several times over.
    // Capability decides the contract (known once the lazy context exists):
    // LFM2 and pure-attention archs shift, so every turn must succeed;
    // an arch that cannot shift (Qwen3.5's imrope cache — seq_add would be
    // a GGML_ASSERT abort, which the gate must prevent) keeps the #173
    // fail-fast, so the overflowing turn must fail CLEANLY with n_eval == 0 —
    // the exact signature the UI's full-prefill retry keys on.
    const bool shifts = sess->can_context_shift();
    int total_tokens = sess->count_tokens(seed.prompt) + r_seed.n_eval;
    bool saw_clean_overflow = false;
    for (int i = 0; i < 8; ++i) {
        xllama::GenerateParams cont;
        cont.prompt = "<|im_end|>\n<|im_start|>user\nAnd again, tell me more "
                      "about that, please.<|im_end|>\n<|im_start|>assistant\n";
        cont.n_predict = 48;
        cont.temperature = 0.0f;
        cont.reuse_kv = true;
        cont.reset_kv = false;
        cont.n_keep = n_keep;
        const auto rc = sess->generate(cont);
        if (shifts) {
            REQUIRE_MESSAGE(rc.success, rc.error_msg);
            CHECK(rc.n_p_eval > 0);
        } else if (!rc.success) {
            CHECK(rc.n_eval == 0);
            CHECK(rc.error_msg.find("context full") != std::string::npos);
            saw_clean_overflow = true;
            break;
        }
        total_tokens += rc.n_p_eval + rc.n_eval;
    }
    if (shifts) {
        // The conversation genuinely outgrew the context — the loop above
        // only proves the point if eviction actually had to happen.
        CHECK(total_tokens > sp.n_ctx);
    } else {
        CHECK(saw_clean_overflow);
    }
}

// Snapshot restoration is not a promise that a hybrid recurrent state can
// rewind to edited/re-rendered history. Compare its fallback with a fresh run.
TEST_CASE("Session: divergent restored snapshot matches cold prefill (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model = std::getenv("XLLAMA_TEST_MODEL");
    if (!model)
        return;
    const auto path = std::filesystem::temp_directory_path() / "xllama-divergent-state.bin";
    xllama::SessionParams sp;
    sp.model_path = model;
    sp.n_ctx = 512;
    sp.n_threads = 2;
    std::string error;
    auto source = xllama::Session::create(sp, &error);
    REQUIRE_MESSAGE(source, error);
    xllama::GenerateParams gp;
    gp.prompt = "<|im_start|>user\nSay hello.<|im_end|>\n<|im_start|>assistant\n";
    gp.n_predict = 12;
    gp.temperature = 0;
    REQUIRE(source->generate(gp).success);
    REQUIRE_MESSAGE(source->save_state(path.string(), &error), error);
    source.reset();
    auto restored = xllama::Session::create(sp, &error);
    REQUIRE_MESSAGE(restored, error);
    REQUIRE_MESSAGE(restored->load_state(path.string(), &error), error);
    gp.prompt += "Edited assistant reply.<|im_end|>\n<|im_start|>user\nSay goodbye."
                 "<|im_end|>\n<|im_start|>assistant\n";
    const auto actual = restored->generate(gp);
    REQUIRE_MESSAGE(actual.success, actual.error_msg);
    restored.reset();
    auto cold = xllama::Session::create(sp, &error);
    REQUIRE_MESSAGE(cold, error);
    const auto expected = cold->generate(gp);
    REQUIRE_MESSAGE(expected.success, expected.error_msg);
    CHECK(actual.n_p_eval <= expected.n_p_eval);
    if (actual.n_p_eval == expected.n_p_eval)
        CHECK(actual.output_text == expected.output_text);
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

namespace {
// Minimal resident session for hub identity tests: ensure_locked's reuse path
// returns the stored pointer without touching the backend.
class StubSessionForHub : public xllama::Session {
  public:
    xllama::InferenceResult generate(const xllama::GenerateParams&) override {
        return {};
    }
    int count_tokens(const std::string&) override {
        return 0;
    }
};
} // namespace

TEST_CASE("SessionHub: batch-thread identity reuses or recreates the session") {
    CHECK(xllama::SessionHub::effective_threads_batch(2, 0) == 2);
    CHECK(xllama::SessionHub::effective_threads_batch(2, 4) == 4);

    xllama::SessionHub hub;
    xllama::SessionParams sp;
    sp.model_path = "/nonexistent/batch-threads.gguf";
    sp.n_threads = 2;
    sp.n_threads_batch = 0; // 0 means "same as n_threads" -> effective 2

    // Seed a resident session with the identity ensure_locked stores for sp.
    hub.session = std::make_unique<StubSessionForHub>();
    hub.model = "batch-threads";
    hub.n_ctx = sp.n_ctx;
    hub.n_batch = sp.n_batch;
    hub.n_ubatch = sp.n_ubatch;
    hub.n_threads = 2;
    hub.n_threads_batch = 2; // stored effective value
    hub.gpu_layers = sp.n_gpu_layers;
    hub.mtp = sp.mtp;
    hub.mtp_depth = sp.mtp_n_max;
    hub.mtp_pmin = sp.mtp_p_min;
    hub.kv_q8 = sp.kv_q8;
    xllama::Session* resident = hub.session.get();

    std::string err;
    // Absent (0) batch threads resolve to the same identity: reuse.
    CHECK(hub.ensure_locked("batch-threads", sp, &err) == resident);
    // An explicit equal value is also the same identity: reuse.
    sp.n_threads_batch = 2;
    CHECK(hub.ensure_locked("batch-threads", sp, &err) == resident);
    // Changing ONLY the batch count is a different load: the hub destroys the
    // resident session first, then creation fails (no model) and leaves the hub
    // empty — the old session must not survive.
    sp.n_threads_batch = 4;
    CHECK(hub.ensure_locked("batch-threads", sp, &err) == nullptr);
    CHECK(hub.session == nullptr);
    CHECK(hub.model.empty());
}

TEST_CASE("session: sequence capacity defaults to the single-session context") {
    // llama.ini [n_seq_max] / [n_parallel] raise it; nothing lowers the default
    // below one sequence, and the hub still serves one request at a time.
    xllama::SessionParams sp;
    CHECK(sp.n_seq_max == 1);
    sp.n_seq_max = 2;
    CHECK(sp.n_seq_max == 2);
}

#ifdef __linux__
// plan 009 / T18: a write that dies in the middle of the target blob must
// leave no file, no .tmp, and the previous snapshot byte-for-byte intact.
// The failure is SIMULATED with RLIMIT_FSIZE inside this single-threaded
// test process: a per-process kernel limit scoped to the armed window, not
// a knob or a shared path, so no other save (or request) can be affected.
// It exercises the real failure path (partial fwrite -> tmp removed ->
// false), NOT a process death — a SIGKILL mid-write would skip that path,
// which stays covered only structurally. Opt-in — needs a real GGUF:
//   XLLAMA_TEST_MODEL=/path/to/model.gguf ./xllama-tests
TEST_CASE("Session: save_state fails cleanly when the write dies mid-blob (T18)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env)
        return;

    std::error_code ec;
    // Exclusive disposable directory: unique per run, never LocalState and
    // never the shared xllama-kv-state.bin of the round-trip test.
    const auto dir = std::filesystem::temp_directory_path() /
                     ("xllama-t18-midwrite-" + std::to_string(getpid()) + "-" +
                      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir, ec);
    REQUIRE_FALSE(ec);
    struct DirCleanup {
        std::filesystem::path dir;
        ~DirCleanup() {
            std::error_code c;
            std::filesystem::remove_all(dir, c);
        }
    } dir_cleanup{dir};

    xllama::SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 512;
    std::string err;
    auto sess = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(sess, err);

    // Resident KV: save_state refuses without it.
    xllama::GenerateParams gp;
    gp.prompt = "The harbour master filed a report on tide tables and cargo "
                "manifests for the autumn season. The clerk copied every row "
                "into the ledger twice, once for the port and once for the "
                "customs house.";
    gp.n_predict = 4;
    gp.temperature = 0.0f;
    gp.reuse_kv = true;
    gp.reset_kv = true;
    const auto seeded = sess->generate(gp);
    REQUIRE_MESSAGE(seeded.success, seeded.error_msg);

    const auto ref = dir / "ref.kv";
    REQUIRE_MESSAGE(sess->save_state(ref.string(), &err), err);
    const auto full_size = std::filesystem::file_size(ref, ec);
    REQUIRE_FALSE(ec);

    // Header: "xllama-kv\t<n_state>\t<n_tokens>\t<fingerprint>\n". Field 2 is
    // the blob size — the region the fault must land strictly inside.
    unsigned long long n_state = 0;
    unsigned long long n_tokens = 0;
    {
        std::ifstream in(ref);
        REQUIRE(in.good());
        std::string header;
        REQUIRE(static_cast<bool>(std::getline(in, header)));
        REQUIRE(std::sscanf(header.c_str(), "xllama-kv\t%llu\t%llu", &n_state, &n_tokens) == 2);
    }
    REQUIRE(n_state > 0);
    REQUIRE(n_tokens > 0);

    // The header fits in its 1024-byte buffer and the token-id array in
    // n_ctx * 8 bytes of slack (sizeof(llama_token) == 4). With
    // n_state > 4 * pre_bound the fault cannot land before the blob starts,
    // and L < full_size keeps it before the blob ends: strictly mid-blob.
    const unsigned long long pre_bound = 1024ull + 8ull * sp.n_ctx;
    REQUIRE(n_state > 4 * pre_bound);
    const rlim_t limit = static_cast<rlim_t>(pre_bound + n_state / 2);
    REQUIRE(static_cast<unsigned long long>(limit) < full_size);
    MESSAGE("T18 fault limit L=" << static_cast<unsigned long long>(limit) << " B inside blob ["
                                 << pre_bound << ", " << pre_bound + n_state
                                 << ") of S=" << full_size << " B, tokens=" << n_tokens);

    auto read_all = [](const std::filesystem::path& p) {
        std::ifstream in(p, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    };
    const std::string ref_bytes = read_all(ref);
    REQUIRE(ref_bytes.size() == full_size);

    // Cooperative observation (plan 009 / T18): a forked, passive watcher —
    // test-only, unprivileged (inotify + pipes), no production code touched
    // and no write faked or replaced. The watcher PRE-CREATES both disposable
    // tmp names and holds the fds before signalling ready; save_state opens
    // them with "wb" (truncate, no O_EXCL, no pre-remove — session.cpp:1589)
    // and writes into those very inodes, its failure path only unlinks the
    // name, and the held fds keep the inodes alive. The ready-pipe
    // handshake fixes the ordering, so fstat reports the bytes the REAL
    // save_state persisted — deterministically, with no scheduling race.
    // Forked BEFORE arming: it never inherits the fault limit and no process
    // is created inside the armed window. (Earlier variants that raced
    // IN_CREATE or spun on open() lost 1 run in 3 — recorded in the receipt.)
    int ready_pipe[2] = {-1, -1};
    int report_pipe[2] = {-1, -1};
    REQUIRE(::pipe(ready_pipe) == 0);
    REQUIRE(::pipe(report_pipe) == 0);
    const pid_t watcher = ::fork();
    REQUIRE(watcher >= 0);
    if (watcher == 0) {
        // Child: observer only; _exit so doctest never unwinds in here.
        ::close(ready_pipe[0]);
        ::close(report_pipe[0]);
        const int ino = inotify_init1(IN_CLOEXEC | IN_NONBLOCK);
        if (ino < 0 || inotify_add_watch(ino, dir.c_str(), IN_CREATE | IN_MODIFY | IN_DELETE) < 0)
            _exit(2);
        const char* names[2] = {"fault.kv.tmp", "ref.kv.tmp"};
        int fds[2] = {-1, -1};
        // Pre-create both tmp names and hold the fds BEFORE signalling
        // ready: save_state's "wb" open (no O_EXCL, no pre-remove —
        // session.cpp:1589) truncates THESE inodes, and its failure path
        // only unlinks the name, so the held fds keep the written bytes
        // observable. Handshake ordering removes the scheduling race that
        // cost the open()-spin variant 1 run in 3.
        for (int i = 0; i < 2; ++i) {
            fds[i] = ::open((dir / names[i]).c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
            if (fds[i] < 0)
                _exit(7);
        }
        const char ready = 'R';
        if (::write(ready_pipe[1], &ready, 1) != 1)
            _exit(3);
        long long max_bytes[2] = {0, 0};
        int deleted[2] = {0, 0};
        int unexpected = 0;
        // 10 s watchdog: the watcher can never hang the suite.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        char buf[4096] __attribute__((aligned(__alignof__(struct inotify_event))));
        while (!(deleted[0] && deleted[1])) {
            for (ssize_t n = ::read(ino, buf, sizeof buf); n > 0;
                 n = ::read(ino, buf, sizeof buf)) {
                for (char* cur = buf; cur < buf + n;) {
                    const auto* ev = reinterpret_cast<const inotify_event*>(cur);
                    cur += sizeof(inotify_event) + ev->len;
                    if (ev->mask & IN_Q_OVERFLOW) {
                        ++unexpected;
                        continue;
                    }
                    const std::string name = ev->len ? ev->name : "";
                    const int idx = name == names[0] ? 0 : (name == names[1] ? 1 : -1);
                    if (idx < 0) {
                        if (!name.empty())
                            ++unexpected;
                        continue;
                    }
                    if (ev->mask & IN_DELETE)
                        deleted[idx] = 1;
                    struct stat st{};
                    if (::fstat(fds[idx], &st) == 0 &&
                        static_cast<long long>(st.st_size) > max_bytes[idx])
                        max_bytes[idx] = static_cast<long long>(st.st_size);
                }
            }
            if (std::chrono::steady_clock::now() >= deadline)
                _exit(4);
        }
        // Final capture through the held fds (valid past the unlink).
        for (int i = 0; i < 2; ++i) {
            struct stat st{};
            if (::fstat(fds[i], &st) == 0 && static_cast<long long>(st.st_size) > max_bytes[i])
                max_bytes[i] = static_cast<long long>(st.st_size);
        }
        char rep[256];
        const int len = std::snprintf(rep, sizeof rep,
                                      "fault.kv.tmp max=%lld open=%d del=%d\n"
                                      "ref.kv.tmp max=%lld open=%d del=%d\n"
                                      "unexpected=%d\n",
                                      max_bytes[0], fds[0] >= 0 ? 1 : 0, deleted[0], max_bytes[1],
                                      fds[1] >= 0 ? 1 : 0, deleted[1], unexpected);
        if (len <= 0 || ::write(report_pipe[1], rep, static_cast<size_t>(len)) != len)
            _exit(6);
        _exit(0);
    }
    ::close(ready_pipe[1]);
    ::close(report_pipe[1]);
    {
        char rdy = 0;
        REQUIRE(::read(ready_pipe[0], &rdy, 1) == 1);
        REQUIRE(rdy == 'R');
        ::close(ready_pipe[0]);
    }

    {
        // Armed window: the kernel rejects any write past L, and SIGXFSZ is
        // ignored for the duration (its default action kills the process —
        // precisely what this test does NOT exercise). Both are restored by
        // the destructor, on REQUIRE unwinding included.
        struct Arm {
            rlimit orig{};
            struct sigaction prev{};
            explicit Arm(rlim_t l) {
                REQUIRE(::getrlimit(RLIMIT_FSIZE, &orig) == 0);
                struct sigaction sa{};
                sa.sa_handler = SIG_IGN;
                REQUIRE(::sigaction(SIGXFSZ, &sa, &prev) == 0);
                rlimit armed = orig;
                armed.rlim_cur = l;
                REQUIRE(::setrlimit(RLIMIT_FSIZE, &armed) == 0);
            }
            ~Arm() {
                ::setrlimit(RLIMIT_FSIZE, &orig);
                ::sigaction(SIGXFSZ, &prev, nullptr);
            }
        } arm(limit);

        // 1) A fresh target: the save fails mid-blob and the routine's own
        //    failure path must leave neither the file nor the tmp behind.
        const auto fault = dir / "fault.kv";
        err.clear();
        CHECK_FALSE(sess->save_state(fault.string(), &err));
        CHECK(err == "write failed (disk full?)");
        CHECK_FALSE(std::filesystem::exists(fault, ec));
        CHECK_FALSE(std::filesystem::exists(fault.string() + ".tmp", ec));

        // 2) Over the previous valid snapshot: it must survive byte-for-byte.
        err.clear();
        CHECK_FALSE(sess->save_state(ref.string(), &err));
        CHECK(err == "write failed (disk full?)");
        REQUIRE(std::filesystem::exists(ref, ec));
        CHECK(read_all(ref) == ref_bytes);
        CHECK_FALSE(std::filesystem::exists(ref.string() + ".tmp", ec));
    }
    // Window closed: limit and signal disposition are back to normal.

    // The watcher's report: bytes effectively persisted in each disposable
    // .tmp before save_state removed it (fd survives the unlink), plus the
    // unlink itself. Direct observation, not an inference from L.
    std::string report;
    {
        char buf[256];
        ssize_t n = 0;
        while ((n = ::read(report_pipe[0], buf, sizeof buf)) > 0)
            report.append(buf, static_cast<size_t>(n));
        ::close(report_pipe[0]);
    }
    int status = 0;
    REQUIRE(::waitpid(watcher, &status, 0) == watcher);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);
    long long obs_fault = 0;
    long long obs_ref = 0;
    int obs_fault_open = 0;
    int obs_fault_del = 0;
    int obs_ref_open = 0;
    int obs_ref_del = 0;
    int obs_unexpected = -1;
    REQUIRE(std::sscanf(report.c_str(),
                        "fault.kv.tmp max=%lld open=%d del=%d\n"
                        "ref.kv.tmp max=%lld open=%d del=%d\n"
                        "unexpected=%d",
                        &obs_fault, &obs_fault_open, &obs_fault_del, &obs_ref, &obs_ref_open,
                        &obs_ref_del, &obs_unexpected) == 7);
    MESSAGE("T18 observed persisted bytes before removal:\n" << report);
    REQUIRE(obs_fault > 0);
    REQUIRE(obs_fault <= static_cast<long long>(limit));
    REQUIRE(obs_ref > 0);
    REQUIRE(obs_ref <= static_cast<long long>(limit));
    REQUIRE(obs_fault_del == 1);
    REQUIRE(obs_ref_del == 1);
    REQUIRE(obs_unexpected == 0);

    // Space accounting: the directory holds exactly the untouched snapshot.
    {
        std::vector<std::string> names;
        for (const auto& e : std::filesystem::directory_iterator(dir, ec))
            names.push_back(e.path().filename().string());
        CHECK(names == std::vector<std::string>{"ref.kv"});
    }

    // A save without the limit succeeds and reproduces the same state file.
    err.clear();
    REQUIRE_MESSAGE(sess->save_state(ref.string(), &err), err);
    CHECK(std::filesystem::file_size(ref, ec) == full_size);
    CHECK(read_all(ref) == ref_bytes);

    // Integrity: the file still restores into a fresh session that continues.
    auto restored = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(restored, err);
    REQUIRE_MESSAGE(restored->load_state(ref.string(), &err), err);
    gp.reset_kv = false;
    gp.prompt = "The harbour master filed a report on tide tables and cargo "
                "manifests for the autumn season. What did the clerk do?";
    const auto cont = restored->generate(gp);
    CHECK(cont.success);

    // Cleanup is verifiable: the exclusive directory is gone.
    std::filesystem::remove_all(dir, ec);
    CHECK_FALSE(std::filesystem::exists(dir, ec));
}

// T18: the HTTP slice for the test-only X-Test-Nonce header lives in
// xllama/api_policy.h so its exact byte handling is provable on host —
// uwp/api-server.cpp only forwards the raw header block. Zero/one/many
// spaces, absent, divergent, oversized, mid-token and shadow-substring
// cases, all before any fault can be armed.
TEST_CASE("api_policy: X-Test-Nonce slice is exact (T18)") {
    using xllama::parse_test_nonce_header;
    // Zero spaces: the value starts exactly after the colon (key = 13 bytes
    // incl. ':') — no byte may be lost.
    CHECK(parse_test_nonce_header("X-Test-Nonce:t18p-3f9a7c21\r\nContent-Length: 2") ==
          "t18p-3f9a7c21");
    // One space.
    CHECK(parse_test_nonce_header("X-Test-Nonce: t18p-3f9a7c21\r\n") == "t18p-3f9a7c21");
    // Multiple spaces and tabs are skipped.
    CHECK(parse_test_nonce_header("X-Test-Nonce:   t18p-3f9a7c21\r\n") == "t18p-3f9a7c21");
    CHECK(parse_test_nonce_header("X-Test-Nonce: \t  t18p-3f9a7c21\r\n") == "t18p-3f9a7c21");
    // Case-insensitive name; the value keeps its case (a divergent casing
    // stays divergent and can never match the armed nonce).
    CHECK(parse_test_nonce_header("x-TEST-Nonce: T18P-AbC12345\r\n") == "T18P-AbC12345");
    CHECK(parse_test_nonce_header("X-Test-NONCE: t18p-deadbeef0\r\n") == "t18p-deadbeef0");
    // Absent / empty -> "" (can never match an armed nonce).
    CHECK(parse_test_nonce_header("Content-Length: 2\r\nX-Other: 1\r\n").empty());
    CHECK(parse_test_nonce_header("X-Test-Nonce:\r\n").empty());
    CHECK(parse_test_nonce_header("X-Test-Nonce: \r\n").empty());
    // Not at a line start (inside another token/value) -> ignored.
    CHECK(parse_test_nonce_header("Foo: x-test-nonce:t18p-3f9a7c21\r\n").empty());
    // A longer name containing the key is not the key.
    CHECK(parse_test_nonce_header("X-Test-Nonce-Extra: t18p-3f9a7c21\r\n").empty());
    // Size cap: 64 kept, 65 rejected.
    CHECK(parse_test_nonce_header("X-Test-Nonce: " + std::string(64, 'a') + "\r\n") ==
          std::string(64, 'a'));
    CHECK(parse_test_nonce_header("X-Test-Nonce: " + std::string(65, 'a') + "\r\n").empty());
    // Last line without a trailing newline still yields the value.
    CHECK(parse_test_nonce_header("X-Test-Nonce: t18p-3f9a7c21") == "t18p-3f9a7c21");
    // Shadow first: the key inside another header's value is skipped and the
    // real header line still matches exactly.
    CHECK(parse_test_nonce_header("X-Custom: prefix x-test-nonce: fake\r\n"
                                  "X-Test-Nonce: t18p-3f9a7c21\r\n") == "t18p-3f9a7c21");
    // Shadow only: no line-start match -> "" (never honored, never fires).
    CHECK(parse_test_nonce_header("X-Custom: prefix x-test-nonce: fake\r\n").empty());
    // Real header first: the first line-start match wins, a later shadow is
    // never even examined.
    CHECK(parse_test_nonce_header("X-Test-Nonce: t18p-3f9a7c21\r\n"
                                  "X-Custom: x-test-nonce: fake\r\n") == "t18p-3f9a7c21");
    // Case-insensitive shadow skipped, real header honored.
    CHECK(parse_test_nonce_header("X-Note: X-TEST-NONCE: fake\r\n"
                                  "X-Test-Nonce: t18p-3f9a7c21\r\n") == "t18p-3f9a7c21");
}

// T18 probe preflight (plan 009, authorized msg 1414): the mid-write fault
// knob binds to BOTH the exact slot basename and the exact request nonce,
// verified BEFORE the atomic claim. Proves by direct execution that:
//   - a save with no nonce never fires and never consumes the arm;
//   - a divergent nonce never fires and never consumes the arm;
//   - another slot never fires and never consumes the arm;
//   - the exact request fires exactly once (one-shot rename claim), keeps
//     the previous snapshot byte-identical, and leaves no .tmp;
//   - an unrelated save racing the matching one neither captures the arm
//     nor loses it (the matching save still fires, exactly once per arm).
//   - a stale ".claimed" blocks the fire fail-closed: never replaced, never
//     deleted, arm untouched (single consumption preserved on every OS).
// OFF by default: without the knob file every save above behaves normally.
// Needs a real GGUF: XLLAMA_TEST_MODEL=/path/to/model.gguf ./xllama-tests
TEST_CASE("Session: fault probe binds exact nonce and slot before claiming (T18)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env)
        return;

    std::error_code ec;

    // The knob resolves to a fixed name (LOCALSTATE on device, CWD on host),
    // i.e. it is GLOBAL to the working directory, not inside this test's
    // disposable dir. Prove ownership before touching anything: an existing
    // file must be a stale arm written by THIS test (exact format
    // "probe.kv <digits> t18p-*", the nonce prefix only these arms use).
    // Anything else is foreign state: fail without removing it.
    const std::string knob = xllama::resolve_local_path("slot-fault.target.txt");
    const std::string claimed = knob + ".claimed";
    auto stale_arm_is_ours = [](const std::filesystem::path& p) {
        std::error_code c;
        if (!std::filesystem::is_regular_file(p, c) || c)
            return false;
        std::ifstream in(p);
        const std::string content((std::istreambuf_iterator<char>(in)),
                                  std::istreambuf_iterator<char>());
        return content.rfind("probe.kv ", 0) == 0 && content.find(" t18p-") != std::string::npos;
    };
    if (std::filesystem::exists(knob, ec) && !stale_arm_is_ours(knob))
        FAIL("pre-existing " << knob << " is not a stale arm of this test — refusing to touch "
                             << "foreign state in the working directory");
    if (std::filesystem::exists(claimed, ec) && !stale_arm_is_ours(claimed))
        FAIL("pre-existing " << claimed << " is not a stale claim of this test — refusing to "
                             << "touch foreign state in the working directory");

    const auto dir = std::filesystem::temp_directory_path() /
                     ("xllama-t18-probe-" + std::to_string(getpid()) + "-" +
                      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir, ec);
    REQUIRE_FALSE(ec);
    // Stale arms proven ours above (or absent): clear them so every arm in
    // this run is freshly written, then remove whatever this run leaves —
    // even on REQUIRE unwinding.
    struct ProbeCleanup {
        std::filesystem::path dir;
        std::string knob;
        std::string claimed;
        ~ProbeCleanup() {
            std::error_code c;
            std::filesystem::remove_all(dir, c);
            std::filesystem::remove(knob, c);
            std::filesystem::remove(claimed, c);
        }
    } cleanup{dir, knob, claimed};
    std::filesystem::remove(knob, ec);
    std::filesystem::remove(claimed, ec);

    xllama::SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 512;
    std::string err;
    auto sess = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(sess, err);

    // Resident KV: save_state refuses without it.
    xllama::GenerateParams gp;
    gp.prompt = "The harbour master filed a report on tide tables and cargo "
                "manifests for the autumn season. The clerk copied every row "
                "into the ledger twice, once for the port and once for the "
                "customs house.";
    gp.n_predict = 4;
    gp.temperature = 0.0f;
    gp.reuse_kv = true;
    gp.reset_kv = true;
    const auto seeded = sess->generate(gp);
    REQUIRE_MESSAGE(seeded.success, seeded.error_msg);

    const auto target = dir / "probe.kv";
    REQUIRE_MESSAGE(sess->save_state(target.string(), &err), err);

    // Header field 2 is the blob size; the armed offset lands mid-blob.
    unsigned long long n_state = 0;
    {
        std::ifstream in(target);
        REQUIRE(in.good());
        std::string header;
        REQUIRE(static_cast<bool>(std::getline(in, header)));
        unsigned long long n_tokens = 0;
        REQUIRE(std::sscanf(header.c_str(), "xllama-kv\t%llu\t%llu", &n_state, &n_tokens) == 2);
    }
    REQUIRE(n_state > 1);

    auto read_all = [](const std::filesystem::path& p) {
        std::ifstream in(p, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    };
    const std::string nonce_a = "t18p-3f9a7c21";        // the exact disposable request
    const std::string nonce_b_plain = "t18p-deadbeef0"; // divergent
    // Every nonce used below comes from the HTTP slice itself — the exact
    // code path uwp/api-server.cpp forwards — and the zero/one/multi-space
    // variants are asserted BEFORE anything can fire.
    using xllama::parse_test_nonce_header;
    const std::string hdr_zero = "X-Test-Nonce:" + nonce_a + "\r\nAccept: */*";
    const std::string hdr_one = "X-Test-Nonce: " + nonce_a + "\r\n";
    const std::string hdr_many = "X-Test-Nonce:   " + nonce_a + "\r\n";
    CHECK(parse_test_nonce_header(hdr_zero) == nonce_a);
    CHECK(parse_test_nonce_header(hdr_one) == nonce_a);
    CHECK(parse_test_nonce_header(hdr_many) == nonce_a);
    const std::string nonce_b = parse_test_nonce_header("X-Test-Nonce: " + nonce_b_plain + "\r\n");
    CHECK(nonce_b == nonce_b_plain); // divergent header -> divergent nonce
    const std::string nonce_absent = parse_test_nonce_header("Content-Length: 18\r\n");
    CHECK(nonce_absent.empty()); // absent header -> "" -> can never match
    const std::string arm_line = "probe.kv " + std::to_string(n_state / 2) + " " + nonce_a;
    auto arm = [&] {
        std::ofstream f(knob, std::ios::trunc | std::ios::binary);
        f << arm_line << "\n";
        REQUIRE_FALSE(f.fail());
    };
    auto knob_intact = [&] {
        CHECK(std::filesystem::exists(knob, ec));
        CHECK_FALSE(std::filesystem::exists(claimed, ec));
        std::ifstream in(knob);
        std::string line;
        std::getline(in, line);
        CHECK(line == arm_line); // not consumed, not rewritten
    };

    // (1) Slot divergence: exact nonce, but a different slot — fires no,
    // consumes no, and the unrelated save still succeeds.
    arm();
    const auto ghost = dir / "ghost.kv";
    err.clear();
    REQUIRE_MESSAGE(sess->save_state(ghost.string(), &err, nonce_a), err);
    CHECK(std::filesystem::exists(ghost, ec));
    knob_intact();

    // (2) Nonce divergence: right slot, wrong nonce.
    err.clear();
    REQUIRE_MESSAGE(sess->save_state(target.string(), &err, nonce_b), err);
    knob_intact();

    // (3) No nonce: the default argument (GUI snapshots and ordinary saves)
    // and the absent-header slice both stay "" — fires no, consumes no.
    err.clear();
    REQUIRE_MESSAGE(sess->save_state(target.string(), &err), err);
    knob_intact();
    err.clear();
    REQUIRE_MESSAGE(sess->save_state(target.string(), &err, nonce_absent), err);
    knob_intact();

    // Valid snapshot in place right before the fire (cases 2-3 rewrote it).
    const std::string snap_bytes = read_all(target);
    REQUIRE_FALSE(snap_bytes.empty());

    // (4) Exact nonce (from the MULTI-SPACE header slice) + exact slot:
    // fires once, through the normal failure path — error text, no .tmp,
    // previous snapshot byte-identical, arm consumed by the rename claim.
    const std::string nonce_fire = parse_test_nonce_header(hdr_many);
    REQUIRE(nonce_fire == nonce_a);
    err.clear();
    CHECK_FALSE(sess->save_state(target.string(), &err, nonce_fire));
    CHECK(err == "write failed (disk full?)");
    CHECK_FALSE(std::filesystem::exists(target.string() + ".tmp", ec));
    CHECK(read_all(target) == snap_bytes);
    CHECK_FALSE(std::filesystem::exists(knob, ec));
    CHECK(std::filesystem::exists(claimed, ec));

    // (5) One-shot: with the arm consumed, the same correct request succeeds.
    err.clear();
    REQUIRE_MESSAGE(sess->save_state(target.string(), &err, nonce_a), err);
    CHECK(read_all(target) == snap_bytes);

    // (5b) Malformed arms fail a PRE-claim check: the save proceeds
    // normally and the arm is neither fired nor consumed. Each entry is
    // written raw and read back byte-identical after the save. The
    // .claimed left by the legitimate fire in (4) is ours — remove it so
    // any re-appearance here would be a claim by a malformed arm.
    std::filesystem::remove(claimed, ec);
    REQUIRE_FALSE(std::filesystem::exists(claimed, ec));
    const std::string bytes_mid = std::to_string(n_state / 2);
    const std::vector<std::array<std::string, 3>> bad_arms = {
        {"probe.kv " + bytes_mid + "\n", nonce_a, "two tokens: no nonce at all"},
        {"probe.kv\n", nonce_a, "one token only"},
        {"probe.kv " + nonce_a + "\n", nonce_a, "no bytes field"},
        {"probe.kv " + bytes_mid + " " + nonce_a + " extra\n", nonce_a, "fourth token"},
        {"probe.kv " + bytes_mid + " " + nonce_a + "\ngarbage\n", nonce_a, "second line"},
        {"probe.kv 0 " + nonce_a + "\n", nonce_a, "zero offset"},
        {"probe.kv abc " + nonce_a + "\n", nonce_a, "non-digit offset"},
        {"probe.kv " + std::string(40, '9') + " " + nonce_a + "\n", nonce_a,
         "40-digit decimal (size_t overflow attempt)"},
        {"probe.kv " + bytes_mid + " t18p$bad!\n", "t18p$bad!",
         "invalid nonce charset even when matched exactly"},
    };
    for (const auto& bad : bad_arms) {
        {
            std::ofstream f(knob, std::ios::trunc | std::ios::binary);
            f << bad[0];
            REQUIRE_FALSE(f.fail());
        }
        err.clear();
        INFO("malformed arm: " << bad[2]);
        REQUIRE_MESSAGE(sess->save_state(target.string(), &err, bad[1]), err);
        CHECK_FALSE(std::filesystem::exists(target.string() + ".tmp", ec));
        CHECK_FALSE(std::filesystem::exists(claimed, ec)); // never claimed
        std::ifstream in(knob, std::ios::binary);
        const std::string back((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
        CHECK(back == bad[0]); // not consumed, not rewritten
    }

    // (5c) Stale claim marker: a pre-existing ".claimed" — another round's
    // marker, possibly foreign — is never replaced nor deleted, and the
    // matching save proceeds normally without firing. Fail-closed on every
    // OS: the claim never depends on destination-replace semantics (POSIX
    // replaces, the UWP STL behavior is unprovable from the Linux host).
    // The knob stays armed and untouched; the device runbook clears
    // ".claimed" before re-arming. Nothing here re-arms by accident.
    {
        std::ofstream c(claimed, std::ios::trunc | std::ios::binary);
        c << "probe.kv 1 t18p-older\n";
        REQUIRE_FALSE(c.fail());
    }
    arm();
    err.clear();
    REQUIRE_MESSAGE(sess->save_state(target.string(), &err, nonce_a), err);
    CHECK_FALSE(std::filesystem::exists(target.string() + ".tmp", ec));
    CHECK(read_all(target) == snap_bytes);    // a real, complete normal save
    CHECK(std::filesystem::exists(knob, ec)); // still armed, untouched
    {
        std::ifstream in(knob);
        std::string line;
        std::getline(in, line);
        CHECK(line == arm_line);
    }
    {
        std::ifstream in(claimed, std::ios::binary);
        const std::string back((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
        CHECK(back == "probe.kv 1 t18p-older\n"); // others' marker intact
    }
    std::filesystem::remove(claimed, ec); // runbook cleanup before continuing
    REQUIRE_FALSE(std::filesystem::exists(claimed, ec));

    // (6) Concurrency: an unrelated save (same nonce, other slot) races the
    // matching one. The probe may be read by either thread, but only the
    // exact nonce+slot can claim: the racer neither fires nor loses the arm,
    // and the matching save still fires — three rounds, deterministic each
    // time. Cross-slot on purpose: same-slot saves share one tmp name and
    // are serialized by the LAN layer, not by this probe.
    auto sess2 = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(sess2, err);
    REQUIRE_MESSAGE(sess2->load_state(target.string(), &err), err);
    const auto other = dir / "other.kv";
    for (int round = 1; round <= 3; ++round) {
        // Runbook cleanup between rounds: a previous round's ".claimed"
        // blocks a new fire fail-closed, so it is cleared before re-arming —
        // exactly what the device runbook does. Never cleared by the probe.
        std::filesystem::remove(claimed, ec);
        arm();
        std::string e1, e2;
        bool ok1 = false;
        bool ok2 = false;
        std::thread t1([&] { ok1 = sess->save_state(target.string(), &e1, nonce_a); });
        std::thread t2([&] { ok2 = sess2->save_state(other.string(), &e2, nonce_a); });
        t1.join();
        t2.join();
        INFO("concurrency round " << round);
        CHECK_FALSE(ok1);
        CHECK(e1 == "write failed (disk full?)"); // the matching save fired
        CHECK_MESSAGE(ok2, e2);                   // the racer was never captured
        CHECK_FALSE(std::filesystem::exists(target.string() + ".tmp", ec));
        CHECK(read_all(target) == snap_bytes);
        CHECK_FALSE(std::filesystem::exists(knob, ec)); // consumed exactly once
        CHECK(std::filesystem::exists(claimed, ec));
    }

    // Space accounting: only the expected slot files, no residual .tmp.
    {
        std::vector<std::string> names;
        for (const auto& e : std::filesystem::directory_iterator(dir, ec))
            names.push_back(e.path().filename().string());
        std::sort(names.begin(), names.end());
        CHECK(names == std::vector<std::string>{"ghost.kv", "other.kv", "probe.kv"});
    }

    // Restore integrity: the preserved snapshot still loads and continues.
    auto restored = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(restored, err);
    REQUIRE_MESSAGE(restored->load_state(target.string(), &err), err);
    gp.reset_kv = false;
    gp.prompt = "The harbour master filed a report on tide tables and cargo "
                "manifests for the autumn season. What did the clerk do?";
    const auto cont = restored->generate(gp);
    CHECK(cont.success);
}
#endif // __linux__

// Every resolved field is part of the persisted identity, including adjacent
// float values that previously collapsed under %.6g / %.4f formatting.
TEST_CASE("Session: RoPE identity preserves every resolved field exactly") {
    llama_context_rope_params p{1, 2048, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f};
    const auto reference = xllama::state_rope_identity(p);
    auto q = p;
    ++q.scaling_type;
    CHECK(xllama::state_rope_identity(q) != reference);
    q = p;
    ++q.orig_ctx;
    CHECK(xllama::state_rope_identity(q) != reference);
    for (auto field :
         {&llama_context_rope_params::freq_base, &llama_context_rope_params::freq_scale,
          &llama_context_rope_params::ext_factor, &llama_context_rope_params::attn_factor,
          &llama_context_rope_params::beta_fast, &llama_context_rope_params::beta_slow}) {
        q = p;
        q.*field = std::nextafter(q.*field, INFINITY);
        CHECK(xllama::state_rope_identity(q) != reference);
    }
    CHECK(xllama::state_rope_identity(p) == reference);
}

TEST_CASE("Session: changed GGUF RoPE identity refuses snapshot before import (opt-in)") {
    const char* fixtures = std::getenv("XLLAMA_TEST_ROPE_FIXTURES");
    if (!fixtures)
        return;
    const std::filesystem::path dir(fixtures);
    const auto state = dir / "identity-test.kv";
    REQUIRE_FALSE(std::filesystem::exists(state));
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    } cleanup{state};
    xllama::SessionParams sp;
    sp.model_path = (dir / "base.gguf").string();
    sp.n_ctx = 512;
    std::string err;
    auto original = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(original, err);
    xllama::GenerateParams gp;
    gp.prompt = "The harbour master filed a report.";
    gp.n_predict = 2;
    gp.temperature = 0;
    gp.reuse_kv = true;
    gp.reset_kv = true;
    REQUIRE(original->generate(gp).success);
    REQUIRE_MESSAGE(original->save_state(state.string(), &err), err);
    // The same model remains compatible, including a fresh session.
    auto same = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(same, err);
    REQUIRE_MESSAGE(same->load_state(state.string(), &err), err);
    for (const char* variant : {"base-adjacent", "scale", "orig", "scaling", "attention"}) {
        INFO("GGUF variant: " << variant);
        sp.model_path = (dir / (std::string(variant) + ".gguf")).string();
        auto changed = xllama::Session::create(sp, &err);
        REQUIRE_MESSAGE(changed, err);
        err.clear();
        CHECK_FALSE(changed->load_state(state.string(), &err));
        CHECK(err == "state file belongs to a different model or context");
        CHECK(changed->generate(gp).success);
    }
    // Version transition refuses v2 without deleting or rewriting the source.
    std::ifstream in(state, std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(in)), {});
    // A different FA reduction policy, or an older identity without the policy,
    // must fail before importing state computed with different arithmetic.
    const auto policy = bytes.find(" fa_query=");
    REQUIRE(policy != std::string::npos);
    for (char other : {'0', '1', '2', '3', '4', '-'}) {
        const auto value_pos = policy + std::string(" fa_query=").size();
        if (other == bytes[value_pos])
            continue;
        auto incompatible = bytes;
        if (other == '-')
            incompatible.erase(policy, std::string(" fa_query=0").size());
        else
            incompatible[value_pos] = other;
        {
            std::ofstream out(state, std::ios::binary);
            out.write(incompatible.data(), incompatible.size());
        }
        err.clear();
        CHECK_FALSE(same->load_state(state.string(), &err));
        CHECK(err == "state file belongs to a different model or context");
        CHECK(same->generate(gp).success);
    }
    auto pos = bytes.find("\tv3 ");
    REQUIRE(pos != std::string::npos);
    bytes[pos + 2] = '2';
    {
        std::ofstream out(state, std::ios::binary);
        out.write(bytes.data(), bytes.size());
    }
    err.clear();
    CHECK_FALSE(same->load_state(state.string(), &err));
    CHECK(err == "state file belongs to a different model or context");
    CHECK(same->generate(gp).success);
}
