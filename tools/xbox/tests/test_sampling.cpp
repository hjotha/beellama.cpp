// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// #125: the CLI/bench surface and the GUI/API surface ran different samplers.
// These tests pin the things that made that possible.

#include <doctest/doctest.h>

#include <cstdlib>

#include "xllama/cli.h"
#include "xllama/inference.h"
#include "xllama/inference_params.h"
#include "xllama/sampling.h"
#include "xllama/session.h"

using namespace xllama;

TEST_CASE("sampling: the two surfaces start from identical defaults") {
    // InferenceParams drives run_inference (CLI, bench); GenerateParams drives
    // Session (GUI, LAN API). They are separate structs for good reasons — one
    // carries a model path and training flags, the other carries KV-reuse state
    // — but a user changing top_p in the GUI and passing --top-p on the CLI must
    // be configuring the same thing. Before #125 the CLI struct had no top_p at
    // all, so the two could not even be compared.
    const InferenceParams ip;
    const GenerateParams gp;

    CHECK(ip.temperature == gp.temperature);
    CHECK(ip.top_p == gp.top_p);
    CHECK(ip.top_k == gp.top_k);
    CHECK(ip.repetition_penalty == gp.repetition_penalty);
    CHECK(ip.seed == gp.seed);

    // Pin the values themselves too. Equality alone would still pass if someone
    // changed both structs together, which is exactly the silent behaviour
    // change this test exists to make visible.
    CHECK(ip.temperature == doctest::Approx(0.8f));
    CHECK(ip.top_p == doctest::Approx(0.9f));
    CHECK(ip.top_k == 40);
    CHECK(ip.repetition_penalty == doctest::Approx(1.1f));
}

TEST_CASE("sampling: both surfaces project to the same SamplingConfig") {
    InferenceParams ip;
    ip.temperature = 0.55f;
    ip.top_p = 0.8f;
    ip.top_k = 20;
    ip.repetition_penalty = 1.25f;
    ip.seed = 1234;

    GenerateParams gp;
    gp.temperature = 0.55f;
    gp.top_p = 0.8f;
    gp.top_k = 20;
    gp.repetition_penalty = 1.25f;
    gp.seed = 1234;

    const SamplingConfig a = ip.sampling();
    const SamplingConfig b = gp.sampling();

    CHECK(a.temperature == b.temperature);
    CHECK(a.top_p == b.top_p);
    CHECK(a.top_k == b.top_k);
    CHECK(a.repetition_penalty == b.repetition_penalty);
    CHECK(a.seed == b.seed);
}

TEST_CASE("sampling: temperature 0 is greedy, independently of the greedy flag") {
    // The full chain must not run at temperature 0. The repetition penalty
    // reweighs prompt tokens BEFORE the temp stage's argmax and can flip the top
    // token — observed on device, where LFM2.5 answered "User\n\n<|end|>" at
    // temperature 0 through the endpoint while pure argmax answered "Hello!".
    SamplingConfig sc;
    CHECK_FALSE(sc.is_greedy());

    sc.temperature = 0.0f;
    CHECK(sc.is_greedy());

    sc.temperature = -1.0f; // nonsense input must not fall through to sampling
    CHECK(sc.is_greedy());

    sc.temperature = 0.8f;
    sc.greedy = true;
    CHECK(sc.is_greedy());
}

TEST_CASE("sampling: same_chain gates the #175 persistent-chain reuse") {
    // Sampler state follows the KV lifecycle; a persistent chain may only be
    // reused while it would be assembled identically. Any stage parameter
    // change must force a rebuild — reusing across a mismatch would sample
    // with parameters the caller no longer holds.
    SamplingConfig a;
    SamplingConfig b;
    CHECK(same_chain(a, b));

    b.top_k = a.top_k + 1;
    CHECK_FALSE(same_chain(a, b));
    b = a;
    b.seed = a.seed + 1;
    CHECK_FALSE(same_chain(a, b));

    // Greedy chains have a single stage: two greedy configs match regardless
    // of the (unused) sampling values, and greedy never matches non-greedy —
    // including via temperature 0, which is greedy without the flag.
    b = a;
    a.greedy = b.greedy = true;
    b.top_p = 0.123f;
    CHECK(same_chain(a, b));
    b.greedy = false;
    CHECK_FALSE(same_chain(a, b));
    a.greedy = false;
    a.temperature = 0.0f;
    b.temperature = 0.8f;
    CHECK_FALSE(same_chain(a, b));
}

TEST_CASE("sampling: the CLI can express every value the GUI can") {
    // Regression for the concrete complaint in #125: a generation observed in
    // the GUI must be reproducible from the command line. Parse the flags and
    // check they land where run_inference reads them.
    InferenceParams p;
    const char* argv[] = {"xllama-cli",
                          "-m",
                          "model.gguf",
                          "-p",
                          "hi",
                          "--temp",
                          "0.55",
                          "--top-p",
                          "0.8",
                          "--top-k",
                          "20",
                          "--repetition-penalty",
                          "1.25",
                          "--seed",
                          "7",
                          "--system",
                          "You are a terse assistant."};
    REQUIRE(parse_cli_args(static_cast<int>(sizeof(argv) / sizeof(argv[0])),
                           const_cast<char**>(argv), p));

    CHECK(p.temperature == doctest::Approx(0.55f));
    CHECK(p.top_p == doctest::Approx(0.8f));
    CHECK(p.top_k == 20);
    CHECK(p.repetition_penalty == doctest::Approx(1.25f));
    CHECK(p.seed == 7u);
    CHECK(p.system_prompt == "You are a terse assistant.");
}

// Opt-in end-to-end parity. Needs a GGUF model:
//   XLLAMA_TEST_MODEL=/path/to/model.gguf ./xllama-tests
// The unit tests above prove the two surfaces hold the same VALUES and call the
// same chain builder. They cannot prove the two code paths feed it the same way
// — for that the only honest check is running both and comparing the tokens.
TEST_CASE("sampling: CLI and Session produce the same text (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env) {
        MESSAGE("XLLAMA_TEST_MODEL not set — skipping CLI/Session parity");
        return;
    }

    const std::string prompt = "The capital of France is";
    const float temperature = 0.9f;
    const float top_p = 0.8f;
    const int top_k = 20;
    const float rep = 1.3f;
    const uint32_t seed = 42;
    const int n_predict = 24;

    InferenceParams ip;
    ip.model_path = model_env;
    ip.prompt = prompt;
    ip.n_predict = n_predict;
    ip.temperature = temperature;
    ip.top_p = top_p;
    ip.top_k = top_k;
    ip.repetition_penalty = rep;
    ip.seed = seed;
    const InferenceResult cli = run_inference(ip);
    REQUIRE(cli.success);

    SessionParams sp;
    sp.model_path = model_env;
    std::string err;
    auto session = Session::create(sp, &err);
    REQUIRE_MESSAGE(session != nullptr, err);

    GenerateParams gp;
    gp.prompt = prompt;
    gp.n_predict = n_predict;
    gp.temperature = temperature;
    gp.top_p = top_p;
    gp.top_k = top_k;
    gp.repetition_penalty = rep;
    gp.seed = seed;
    const InferenceResult gui = session->generate(gp);
    REQUIRE(gui.success);

    // This is the complaint in #125 stated as an assertion: a generation seen on
    // one surface must be reproducible on the other.
    CHECK(cli.output_text == gui.output_text);
}

// MTP + Session: a continuation turn (KV reuse, delta prefill) must draft and
// verify identically to the cold full-prompt run of the same two turns
// (plan 003, F3: the carry is seeded from the real last-batch row count, not
// from the absolute position — a delta after a reused prefix used to read a
// stale row). Opt-in: XLLAMA_TEST_MODEL must carry the MTP head.
TEST_CASE("mtp: session KV-reuse delta parity with MTP (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env) {
        MESSAGE("XLLAMA_TEST_MODEL not set — skipping mtp session delta");
        return;
    }

    SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 2048;
    sp.mtp = true;
    sp.mtp_n_max = 4;
    sp.mtp_p_min = 0.75f;
    std::string err;
    auto session = Session::create(sp, &err);
    REQUIRE_MESSAGE(session != nullptr, err);

    // Turn 1 with KV reuse: full prompt, reset.
    GenerateParams g1;
    g1.prompt = "The capital of France is";
    g1.n_predict = 12;
    g1.temperature = 0.0f;
    g1.reuse_kv = true;
    g1.reset_kv = true;
    const InferenceResult r1 = session->generate(g1);
    REQUIRE(r1.success);

    // Turn 2: a SHORT delta appended after the reused prefix — the case F3
    // fixes (the last batch is 3 rows at positions 100+, not 103 rows).
    GenerateParams g2;
    g2.prompt = " and its largest city is";
    g2.n_predict = 12;
    g2.temperature = 0.0f;
    g2.reuse_kv = true;
    g2.reset_kv = false;
    const InferenceResult r2 = session->generate(g2);
    REQUIRE(r2.success);
    CHECK(r2.n_eval > 0);

    // The continuation must continue the same greedy text the cold run
    // produced: the cold output starts with turn-1's text and the turn-2 delta
    // must be the continuation the cold run generated after that prefix.
    // Drafting must also survive the delta prefill: F3 replays the delta chunk
    // into the private draft context, so a reuse turn keeps proposing instead
    // of degrading to classic decoding.
    CHECK(r2.n_drafted > 0);
    INFO("r1 n_eval=" << r1.n_eval << " mtp_active=" << r1.mtp_active
                      << " n_drafted=" << r1.n_drafted << " text=[" << r1.output_text << "]");
    INFO("r2 n_eval=" << r2.n_eval << " mtp_active=" << r2.mtp_active
                      << " n_drafted=" << r2.n_drafted << " text=[" << r2.output_text << "]");

    // Cold reference over the EXACT cumulative text: the session state after
    // turn 2 is prompt + turn-1 output + delta as one token stream, so a cold
    // run over that text must produce the same continuation.
    auto session2 = Session::create(sp, &err);
    REQUIRE_MESSAGE(session2 != nullptr, err);
    GenerateParams g3;
    g3.prompt = "The capital of France is" + r1.output_text + " and its largest city is";
    g3.n_predict = static_cast<int32_t>(r2.n_eval);
    g3.temperature = 0.0f;
    g3.reuse_kv = true;
    g3.reset_kv = true;
    const InferenceResult r3 = session2->generate(g3);
    REQUIRE(r3.success);
    INFO("r3 n_eval=" << r3.n_eval << " mtp_active=" << r3.mtp_active
                      << " n_drafted=" << r3.n_drafted << " text=[" << r3.output_text << "]");
    CHECK(r3.output_text == r2.output_text);
}

// Plan 003 §3.4: a prefill larger than n_batch arrives in chunks, so the drafter
// is fed several replay batches instead of one. Chunking must not change what the
// model reads, and it must not leave the drafter unable to propose.
// Opt-in: XLLAMA_TEST_MODEL=/path/to/model.gguf
TEST_CASE("mtp: chunked prefill keeps parity and keeps drafting (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env) {
        MESSAGE("XLLAMA_TEST_MODEL not set — skipping mtp chunked prefill");
        return;
    }

    SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 512;
    sp.n_batch = 16;
    sp.mtp = true;
    sp.mtp_n_max = 4;
    sp.mtp_p_min = 0.75f;
    std::string err;
    auto chunked = Session::create(sp, &err);
    REQUIRE_MESSAGE(chunked != nullptr, err);

    // Grow past n_batch with the model's own tokenizer, as the non-MTP chunking
    // case does, so the test does not depend on one vocabulary's density. The
    // batch is deliberately small: what is under test is that a prefill split
    // across chunks reaches the drafter intact, not how long a large prefill
    // takes on a slow host.
    std::string prompt = "Summarize the following log.\n";
    int n_prompt = 0;
    for (int i = 0; i < 200 && n_prompt <= 24; ++i) {
        prompt += "line " + std::to_string(i) + ": nothing happened\n";
        if ((i % 4) == 3)
            n_prompt = chunked->count_tokens(prompt);
    }
    n_prompt = chunked->count_tokens(prompt);
    REQUIRE(n_prompt > sp.n_batch);
    REQUIRE(n_prompt < 64);
    // Eager snapshot: doctest INFO re-evaluates its stream expression when a
    // later assertion fails, and sp.n_batch is reset to 0 below — dividing by
    // it then would divide by zero inside the reporter's own mutex (SIGFPE
    // self-deadlock), hanging the runner instead of reporting the failure.
    const int exercised_n_batch = sp.n_batch;
    const int exercised_chunks = (n_prompt + exercised_n_batch - 1) / exercised_n_batch;
    INFO("exercised config: n_ctx=" << sp.n_ctx << " n_batch=" << exercised_n_batch
                                    << " n_prompt=" << n_prompt << " chunks=" << exercised_chunks);

    GenerateParams g;
    g.prompt = prompt;
    g.n_predict = 8;
    g.temperature = 0.0f;
    const InferenceResult r_chunked = chunked->generate(g);
    REQUIRE_MESSAGE(r_chunked.success, r_chunked.error_msg);
    CHECK(r_chunked.n_p_eval == n_prompt);
    CHECK(r_chunked.mtp_active);
    CHECK(r_chunked.n_drafted > 0);
    INFO("chunked n_eval=" << r_chunked.n_eval << " drafted=" << r_chunked.n_drafted << " text=["
                           << r_chunked.output_text << "]");

    // Same prompt, same MTP config, one batch: n_batch = 0 restores the default,
    // well above this prompt.
    sp.n_batch = 0;
    auto whole = Session::create(sp, &err);
    REQUIRE_MESSAGE(whole != nullptr, err);
    const InferenceResult r_whole = whole->generate(g);
    REQUIRE(r_whole.success);
    CHECK(r_whole.mtp_active);
    CHECK(r_whole.n_drafted > 0);
    INFO("whole n_eval=" << r_whole.n_eval << " drafted=" << r_whole.n_drafted << " text=["
                         << r_whole.output_text << "]");

    // Full-sequence parity, not a leading character: the chunked prefill and the
    // single-batch prefill must produce the same greedy continuation. If they do
    // not, the first divergence is reported so a near-tie can be told apart from
    // a real state difference.
    CHECK(r_chunked.n_eval == r_whole.n_eval);
    const std::string& chunked_text = r_chunked.output_text;
    const std::string& whole_text = r_whole.output_text;
    size_t first_diff = 0;
    while (first_diff < chunked_text.size() && first_diff < whole_text.size() &&
           chunked_text[first_diff] == whole_text[first_diff])
        ++first_diff;
    INFO("first divergence at char " << first_diff << " (sizes " << chunked_text.size() << " vs "
                                     << whole_text.size() << ")");
    CHECK(chunked_text == whole_text);
}

// Plan 003 §3.5: reset and prompt swap are part of the draft lifetime. After a
// reset the private draft context describes a history the target no longer has,
// so the drafter must either rebuild it or decline explicitly — never speculate
// on the stale mirror. What is under test is that drafting survives the swap and
// that the post-reset text is the cold text.
// Opt-in: XLLAMA_TEST_MODEL=/path/to/model.gguf
TEST_CASE("mtp: reset and prompt swap keep the drafter honest (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env) {
        MESSAGE("XLLAMA_TEST_MODEL not set — skipping mtp reset/prompt swap");
        return;
    }

    SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 1024;
    sp.mtp = true;
    sp.mtp_n_max = 4;
    sp.mtp_p_min = 0.75f;
    std::string err;
    auto session = Session::create(sp, &err);
    REQUIRE_MESSAGE(session != nullptr, err);

    GenerateParams g1;
    g1.prompt = "The capital of France is";
    g1.n_predict = 8;
    g1.temperature = 0.0f;
    g1.reuse_kv = true;
    g1.reset_kv = true;
    const InferenceResult r1 = session->generate(g1);
    REQUIRE(r1.success);
    CHECK(r1.mtp_active);
    CHECK(r1.n_drafted > 0);
    INFO("r1 n_eval=" << r1.n_eval << " drafted=" << r1.n_drafted << " text=[" << r1.output_text
                      << "]");

    // Unrelated prompt with reset_kv: the KV is wiped, so every position the
    // drafter mirrored is gone. Drafting must come back from the new prefill.
    GenerateParams g2;
    g2.prompt = "A thermometer measures";
    g2.n_predict = 8;
    g2.temperature = 0.0f;
    g2.reuse_kv = true;
    g2.reset_kv = true;
    const InferenceResult r2 = session->generate(g2);
    REQUIRE(r2.success);
    CHECK(r2.mtp_active);
    CHECK(r2.n_drafted > 0);
    INFO("r2 n_eval=" << r2.n_eval << " drafted=" << r2.n_drafted << " text=[" << r2.output_text
                      << "]");

    // Cold reference for the swapped prompt: after a reset the session state is
    // that prompt alone, so a fresh session must produce the same continuation.
    auto cold = Session::create(sp, &err);
    REQUIRE_MESSAGE(cold != nullptr, err);
    GenerateParams g3 = g2;
    g3.reset_kv = true;
    const InferenceResult r3 = cold->generate(g3);
    REQUIRE(r3.success);
    INFO("r3 n_eval=" << r3.n_eval << " drafted=" << r3.n_drafted << " text=[" << r3.output_text
                      << "]");
    CHECK(r3.output_text == r2.output_text);
}

// Plan 003 F3.5: the other full_prompt shape. kv_keep > 0 — the prompt was edited
// but still shares its prefix with the resident tokens — so the prefill catch-up
// has to replay a tail that starts above position 0. Rebuilding the mirror from a
// full re-prefill is allowed; dropping the drafter is not.
// Opt-in: XLLAMA_TEST_MODEL=/path/to/model.gguf
TEST_CASE(
    "mtp: edited full prompt with a reused prefix keeps drafting (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env) {
        MESSAGE("XLLAMA_TEST_MODEL not set — skipping mtp reused-prefix edit");
        return;
    }

    SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 1024;
    sp.mtp = true;
    sp.mtp_n_max = 4;
    sp.mtp_p_min = 0.75f;
    std::string err;
    auto session = Session::create(sp, &err);
    REQUIRE_MESSAGE(session != nullptr, err);

    GenerateParams g1;
    g1.prompt = "The capital of France is";
    g1.n_predict = 8;
    g1.temperature = 0.0f;
    g1.reuse_kv = true;
    g1.reset_kv = true;
    const InferenceResult r1 = session->generate(g1);
    REQUIRE(r1.success);
    CHECK(r1.mtp_active);
    CHECK(r1.n_drafted > 0);
    INFO("r1 n_eval=" << r1.n_eval << " drafted=" << r1.n_drafted << " text=[" << r1.output_text
                      << "]");

    // reset_kv forces the full_prompt path and the prompt still shares its prefix
    // with the resident tokens, so kv_keep > 0.
    GenerateParams g2;
    g2.prompt = "The capital of France is Paris, and its largest city is";
    g2.n_predict = 8;
    g2.temperature = 0.0f;
    g2.reuse_kv = true;
    g2.reset_kv = true;
    const InferenceResult r2 = session->generate(g2);
    REQUIRE(r2.success);
    CHECK(r2.mtp_active);
    CHECK(r2.n_drafted > 0);
    INFO("r2 n_eval=" << r2.n_eval << " drafted=" << r2.n_drafted << " text=[" << r2.output_text
                      << "]");

    // Cold reference over the same prompt: whether the tail was replayed on the
    // kept prefix or the whole prompt was re-prefilled, the resulting state is
    // that prompt from position 0, so a fresh session must produce the same text.
    auto cold = Session::create(sp, &err);
    REQUIRE_MESSAGE(cold != nullptr, err);
    const InferenceResult r3 = cold->generate(g2);
    REQUIRE(r3.success);
    INFO("r3 n_eval=" << r3.n_eval << " drafted=" << r3.n_drafted << " text=[" << r3.output_text
                      << "]");
    CHECK(r3.output_text == r2.output_text);
}
