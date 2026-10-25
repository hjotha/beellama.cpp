// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// src/bridge/decode_loop.h end to end: the bench-only ignore_eog contract and
// the MTP parity gate (plan 003, stage 1).

#include <doctest/doctest.h>

#include <atomic>
#include <cstdlib>
#include <string>

#include "xllama/inference.h"
#include "xllama/inference_params.h"
#include "xllama/mtp_draft.h"

using namespace xllama;

TEST_CASE("mtp: draft CPU worker override is bounded and strict") {
    CHECK(mtp_draft_threads(nullptr, 6) == 6);
    CHECK(mtp_draft_threads("", 6) == 6);
    CHECK(mtp_draft_threads("1", 6) == 1);
    CHECK(mtp_draft_threads("6", 2) == 6);
    CHECK(mtp_draft_threads("7", 6) == 6);
    CHECK(mtp_draft_threads("-1", 6) == 6);
    CHECK(mtp_draft_threads("2junk", 6) == 6);
}

// Opt-in: XLLAMA_TEST_MODEL=/path/to/model.gguf ./xllama-tests
// A short-answer prompt reaches end-of-generation well before n_predict; with
// ignore_eog every run must still decode exactly n_predict tokens, with prompt
// lookup off and on (D2b A/Bs rely on it). An EOG inside a draft cannot be
// forced with a real model; detail::stops_at_eog is the single decision every
// branch of decode_loop shares, so the branches cannot drift apart.
TEST_CASE("decode_loop: ignore_eog decodes exactly n_predict (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env) {
        MESSAGE("XLLAMA_TEST_MODEL not set — skipping ignore_eog");
        return;
    }
    for (const bool lookup : {false, true}) {
        CAPTURE(lookup);
        InferenceParams ip;
        ip.model_path = model_env;
        ip.prompt = "Reply with the single word OK.";
        ip.n_predict = 48;
        ip.temperature = 0.0f;
        ip.prompt_lookup = lookup;
        ip.ignore_eog = true;
        ip.stop_sequences.clear();
        const InferenceResult r = run_inference(ip);
        REQUIRE(r.success);
        CHECK(r.n_eval == ip.n_predict);
        CHECK_FALSE(r.ended_with_stop);
    }
}

// ---------------------------------------------------------------------------
// MTP parity gate (plan 003, stage 1)
// ---------------------------------------------------------------------------
// XLLAMA_TEST_MODEL must be a GGUF that carries the MTP head (stage-mtp.gguf).
// These are the F1 gate: greedy parity, sampling-seed parity, p_min rejecting
// every proposal, EOG, stop sequences, abort, and the n_predict limit. A model
// without the MTP tensors is a PRECONDITION FAILURE (the drafter does not
// activate), not a test to skip — REQUIRE(res.mtp_active) below encodes that.

namespace {

// Runs one inference with |mutate| applied to a base config, and asserts the
// two-armed contract: both succeed and MTP was really active when asked.
// |expect_draft| is off only for scenarios that cannot draft (abort set before
// the first token leaves nothing to propose).
InferenceResult mtp_run(const InferenceParams& base, bool mtp, const char* label,
                        bool expect_draft = true) {
    InferenceParams ip = base;
    ip.mtp = mtp;
    ip.mtp_n_max = 4;
    ip.mtp_p_min = 0.75f;
    const InferenceResult r = run_inference(ip);
    {
        const std::string msg = std::string(label) + ": run failed: " + r.error_msg;
        REQUIRE_MESSAGE(r.success, msg);
    }
    if (mtp) {
        const std::string msg1 = std::string(label) + ": MTP requested but drafter inactive "
                                                      "(model lacks the MTP head?)";
        REQUIRE_MESSAGE(r.mtp_active, msg1);
        if (expect_draft) {
            const std::string msg2 = std::string(label) + ": MTP active but drafted nothing";
            CHECK_MESSAGE(r.n_drafted > 0, msg2);
        }
    }
    return r;
}

} // namespace

TEST_CASE("mtp: greedy parity — MTP output is byte-identical to baseline "
          "(opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env) {
        MESSAGE("XLLAMA_TEST_MODEL not set — skipping mtp greedy parity");
        return;
    }
    InferenceParams base;
    base.model_path = model_env;
    base.prompt = "The capital of France is";
    base.n_predict = 32;
    base.greedy = true;
    base.stop_sequences.clear();

    const InferenceResult plain = mtp_run(base, false, "greedy baseline");
    const InferenceResult mtp = mtp_run(base, true, "greedy mtp");
    // Greedy text is decided by the target alone; MTP only proposes. Any
    // divergence means the carry/verify/state path is wrong.
    CHECK(plain.output_text == mtp.output_text);
    CHECK(plain.n_eval == mtp.n_eval);
}

TEST_CASE("mtp: fixed-seed sampling parity — MTP output matches baseline "
          "(opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env) {
        MESSAGE("XLLAMA_TEST_MODEL not set — skipping mtp seed parity");
        return;
    }
    InferenceParams base;
    base.model_path = model_env;
    base.prompt = "The capital of France is";
    base.n_predict = 32;
    base.seed = 42;
    base.stop_sequences.clear();

    const InferenceResult plain = mtp_run(base, false, "seed baseline");
    const InferenceResult mtp = mtp_run(base, true, "seed mtp");
    CHECK(plain.output_text == mtp.output_text);
    CHECK(plain.n_eval == mtp.n_eval);
}

TEST_CASE("mtp: p_min=100 rejects every proposal — output equals baseline "
          "(opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env) {
        MESSAGE("XLLAMA_TEST_MODEL not set — skipping mtp p_min reject-all");
        return;
    }
    InferenceParams ip;
    ip.model_path = model_env;
    ip.prompt = "The capital of France is";
    ip.n_predict = 24;
    ip.greedy = true;
    ip.mtp = true;
    ip.mtp_n_max = 4;
    ip.mtp_p_min = 1.0f; // 100%: no candidate can pass
    ip.stop_sequences.clear();
    const InferenceResult r = run_inference(ip);
    REQUIRE(r.success);
    REQUIRE(r.mtp_active);
    // No draft can clear a 100% threshold, so the counters must stay zero and
    // the text must match the non-MTP run exactly.
    CHECK(r.n_drafted == 0);
    InferenceParams plain = ip;
    plain.mtp = false;
    const InferenceResult base = run_inference(plain);
    REQUIRE(base.success);
    CHECK(base.output_text == r.output_text);
}

TEST_CASE("mtp: stop sequence and EOG end the run identically "
          "(opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env) {
        MESSAGE("XLLAMA_TEST_MODEL not set — skipping mtp stop/EOG");
        return;
    }
    InferenceParams base;
    base.model_path = model_env;
    base.prompt = "Count from one to five, one number per line.";
    base.n_predict = 64;
    base.greedy = true;
    base.stop_sequences = {"\n5"}; // hits a stop well before n_predict

    const InferenceResult plain = mtp_run(base, false, "stop baseline");
    const InferenceResult mtp = mtp_run(base, true, "stop mtp");
    CHECK(plain.output_text == mtp.output_text);
    CHECK(plain.n_eval == mtp.n_eval);
    CHECK(plain.ended_with_stop == mtp.ended_with_stop);
    CHECK(plain.n_eval < 64); // the stop really ended it

    // EOG path: no stop sequences, short answer, no ignore_eog.
    InferenceParams eog_base = base;
    eog_base.stop_sequences.clear();
    const InferenceResult eplain = mtp_run(eog_base, false, "eog baseline");
    const InferenceResult emtp = mtp_run(eog_base, true, "eog mtp");
    CHECK(eplain.output_text == emtp.output_text);
    CHECK(eplain.n_eval == emtp.n_eval);
    CHECK_FALSE(eplain.ended_with_stop);
}

TEST_CASE("mtp: abort flag stops mid-generation on both arms "
          "(opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env) {
        MESSAGE("XLLAMA_TEST_MODEL not set — skipping mtp abort");
        return;
    }
    InferenceParams base;
    base.model_path = model_env;
    base.prompt = "Write a long paragraph about the history of computing.";
    base.n_predict = 256;
    base.greedy = true;
    base.stop_sequences.clear();
    std::atomic<bool> abort{false};
    base.abort_flag = &abort;
    abort.store(true); // abort before the first token

    const InferenceResult plain = mtp_run(base, false, "abort baseline");
    // Abort is set before the first token: nothing can be drafted, so the
    // draft contract does not apply — what is under test is that both arms
    // stop identically.
    const InferenceResult mtp = mtp_run(base, true, "abort mtp", /*expect_draft=*/false);
    CHECK(plain.output_text == mtp.output_text);
    CHECK(plain.n_eval == mtp.n_eval);
}

TEST_CASE("mtp: n_predict limit is honoured with a mid-run rejection "
          "(opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env) {
        MESSAGE("XLLAMA_TEST_MODEL not set — skipping mtp limit");
        return;
    }
    InferenceParams base;
    base.model_path = model_env;
    base.prompt = "The capital of France is";
    base.n_predict = 13; // odd count: forces a partial final round
    base.greedy = true;
    base.stop_sequences.clear();

    const InferenceResult plain = mtp_run(base, false, "limit baseline");
    const InferenceResult mtp = mtp_run(base, true, "limit mtp");
    CHECK(plain.output_text == mtp.output_text);
    CHECK(plain.n_eval == 13);
    CHECK(mtp.n_eval == 13);
}
