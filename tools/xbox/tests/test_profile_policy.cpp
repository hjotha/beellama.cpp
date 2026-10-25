// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Rev130 measurement-integrity regression: profile OFF must not emit the
// profile-only step/window logs on ANY path, including the non-session prefill
// catch-up; profile ON must emit them with the prefill catch-up before the
// decode steps. Opt-in with XLLAMA_TEST_MODEL pointing at an MTP GGUF
// (stage-mtp.gguf), like the other MTP gates.
//
// Honest coverage limits of this gate:
//  * Log absence proves no profile line was emitted on the exercised paths. It
//    does not independently observe clock reads; the read gating is
//    expression-level (`profile ? d3d12_wall_ms() : 0.0`) and the mandatory
//    parameter removal makes an omitted flag a compile error.
//  * The Linux D3D12 backend is a no-op stub, so the real shape-log/histogram
//    state and D3D12 clocks can only be validated on the console (rev130
//    runbook: ON/OFF profile toggles in both block orders).
//  * Debug host build with the 4B MTP fixture: this is a correctness/coverage
//    gate, never a timing gate.
//  * The separate `mtp:` tests remain the full parity gates; this test only
//    asserts that drafting happened (REQUIRE(n_drafted > 0)) before requiring
//    the verify marker, so it cannot silently pass without exercising MTP.

#include <doctest/doctest.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#if defined(__unix__) || defined(__APPLE__)
    #include <unistd.h>
    #define XLLAMA_PROFILE_TEST_POSIX 1
#else
    #define XLLAMA_PROFILE_TEST_POSIX 0
#endif

#include "xllama/inference.h"
#include "xllama/inference_params.h"

using namespace xllama;

#if XLLAMA_PROFILE_TEST_POSIX

namespace {

// Captures everything the bridge writes to stderr (log_output on Linux).
class StderrCapture {
  public:
    StderrCapture() {
        std::fflush(stderr);
        m_tmp = std::tmpfile();
        m_saved = ::dup(fileno(stderr));
        if (m_tmp && m_saved >= 0)
            ::dup2(fileno(m_tmp), fileno(stderr));
    }
    std::string stop() {
        std::fflush(stderr);
        if (m_saved >= 0)
            ::dup2(m_saved, fileno(stderr));
        std::string out;
        if (m_tmp) {
            std::fseek(m_tmp, 0, SEEK_SET);
            char buf[8192];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof(buf), m_tmp)) > 0)
                out.append(buf, n);
            std::fclose(m_tmp);
            m_tmp = nullptr;
        }
        if (m_saved >= 0) {
            ::close(m_saved);
            m_saved = -1;
        }
        return out;
    }
    ~StderrCapture() {
        if (m_tmp)
            stop();
    }
    StderrCapture(const StderrCapture&) = delete;
    StderrCapture& operator=(const StderrCapture&) = delete;

  private:
    FILE* m_tmp = nullptr;
    int m_saved = -1;
};

struct ArmResult {
    bool ok = false;
    int drafted = 0;
    std::string log;
};

bool contains(const std::string& hay, const char* needle) {
    return hay.find(needle) != std::string::npos;
}

long first_pos(const std::string& hay, const char* needle) {
    return static_cast<long>(hay.find(needle));
}

ArmResult run_captured(const InferenceParams& ip) {
    StderrCapture cap;
    const InferenceResult r = run_inference(ip);
    ArmResult a;
    a.ok = r.success;
    a.drafted = r.n_drafted;
    a.log = cap.stop();
    return a;
}

} // namespace

TEST_CASE("profile policy: OFF emits no profile clocks; ON emits ordered steps "
          "(opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env) {
        MESSAGE("XLLAMA_TEST_MODEL not set — skipping profile policy gate");
        return;
    }
    InferenceParams ip;
    ip.model_path = model_env;
    ip.prompt = "The capital of France is";
    ip.n_predict = 8;
    ip.greedy = true;
    ip.ignore_eog = true; // fixed token count: deterministic drafting coverage
    ip.stop_sequences.clear();
    ip.mtp = true;
    ip.mtp_n_max = 2;
    ip.mtp_p_min = 0.5f;
    ip.n_threads = 2;

    ip.profile_phases = false;
    const ArmResult off = run_captured(ip);
    ip.profile_phases = true;
    const ArmResult on = run_captured(ip);
    ip.profile_phases = false;
    const ArmResult off2 = run_captured(ip);

    REQUIRE_MESSAGE(off.ok, "profile OFF arm failed");
    REQUIRE_MESSAGE(on.ok, "profile ON arm failed");
    REQUIRE_MESSAGE(off2.ok, "profile OFF after ON arm failed");
    // No silent pass without real MTP engagement (same bar as the mtp: gates).
    REQUIRE_MESSAGE(on.drafted > 0, "profile ON arm drafted nothing — gate would not cover MTP");

    for (const char* marker : {"DSTEP", "CSTEP", "VROUND", "DSPLIT"}) {
        CAPTURE(marker);
        const std::string m_off = std::string("profile OFF arm emitted ") + marker;
        const std::string m_off2 =
            std::string("profile OFF after ON arm emitted ") + marker + " (state leak)";
        const std::string m_on = std::string("profile ON arm did not emit ") + marker;
        CHECK_MESSAGE(!contains(off.log, marker), m_off);
        CHECK_MESSAGE(!contains(off2.log, marker), m_off2);
        CHECK_MESSAGE(contains(on.log, marker), m_on);
    }
    // Non-session prefill: the catch-up replay runs before any decode step, so
    // the first DSTEP must precede the first CSTEP in the ON arm.
    const long dstep = first_pos(on.log, "DSTEP");
    const long cstep = first_pos(on.log, "CSTEP");
    const bool order_ok = (dstep >= 0) && (cstep < 0 || dstep < cstep);
    CHECK_MESSAGE(order_ok, "prefill catch-up DSTEP did not precede decode CSTEP");
}

#else

TEST_CASE("profile policy: OFF emits no profile clocks (POSIX-only capture)") {
    MESSAGE("stderr capture is POSIX-only — skipping");
}

#endif
