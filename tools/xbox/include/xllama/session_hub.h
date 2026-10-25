// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Single process-wide owner of the loaded model Session.
//
// Before this existed the GUI (MainPageController::m_session) and the LAN API
// (api-server.cpp g_session) each owned an independent Session: both could
// hold a model at the same time, silently breaking the "never 2x model in
// RAM" invariant EnsureSession enforced only within the GUI. The hub is the
// one owner; both surfaces lock it for the duration of a turn.
//
// SDK usage:
//   xllama::SessionHub hub;           // explicit instance (SDK)
//   auto* s = hub.ensure_locked(...); // same API as session_hub().ensure_locked()
//
// The global session_hub() function remains for backward compatibility;
// it returns a reference to a static SessionHub instance.
//
// Locking contract:
//   - Take `mtx` before touching `session` / `model`, and HOLD it across
//     ensure_locked() + generate() so a concurrent surface cannot swap the
//     resident model mid-turn. The GUI blocks on the lock (its turns are
//     serialized anyway); the API uses try_lock and reports busy.
//   - `generation` increments every time the resident session changes. A
//     surface that keeps per-conversation state tied to the live session
//     (the GUI's KV-reuse flag) must record it and drop that state when it
//     no longer matches — the other surface may have swapped models between
//     turns.
#pragma once

#include "xllama/session.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace xllama {

// SessionHub owns one resident Session at a time.
// It is a plain, constructible struct — no singleton pattern.
// The global session_hub() function provides a static instance for
// backward compatibility; SDK users may create their own instance.
struct SessionHub {
    std::mutex mtx;
    std::unique_ptr<Session> session; // guarded by mtx
    std::string model;                // model id loaded into `session`; guarded by mtx
    int gpu_layers = 0;               // n_gpu_layers `session` was loaded with; guarded by mtx
    // Full effective identity of the resident session's shaping config
    // (guarded by mtx; review): any change here is a different load — the
    // hub recreates rather than reusing (editing 64/64 must not reuse a
    // stale context). The twocol/repack startup profile is process-immutable
    // (restart required), so it cannot change under a live session and needs
    // no identity term.
    bool mtp = false;
    int mtp_depth = 0;
    float mtp_pmin = 0.0f;
    int n_ctx = 0;
    int n_batch = 0;
    int n_ubatch = 0;
    int n_threads = 0;
    // Effective batch-graph thread count (plan 006 C1): 0/absent in a request
    // means "same as n_threads", so the hub stores and compares the resolved
    // value; a request that changes only this field is a different load.
    int n_threads_batch = 0;
    bool kv_q8 = false;
    uint64_t generation = 0; // bumps on every resident-session change; guarded by mtx

    // True while a background pre-load holds (or is about to take) mtx.
    // Lets the LAN API distinguish "warming up, wait briefly" from "another
    // request is generating, report busy" — without it the first request
    // right after app-Ready bounced with a 503 while the preload held the
    // lock (observed on-console during Phase C validation).
    std::atomic<bool> preloading{false};

    // Effective batch thread count for an identity request: a non-positive
    // n_threads_batch means "same as n_threads" exactly as the loaders resolve
    // it, so 0 and an explicit equal value are the same session identity.
    static int effective_threads_batch(int n_threads, int n_threads_batch) {
        return n_threads_batch > 0 ? n_threads_batch : n_threads;
    }

    // Under mtx: return the resident session for |model_id|, creating it (and
    // destroying any other model's session FIRST — never 2x model in RAM) if
    // needed. On creation failure the hub is left empty and nullptr returns.
    Session* ensure_locked(const std::string& model_id, const SessionParams& sp,
                           std::string* err = nullptr) {
        // A different GPU-layer request is a different load (weights move
        // between CPU and d3d12 buffers), so it reloads like a model switch.
        if (session && model == model_id && gpu_layers == sp.n_gpu_layers && mtp == sp.mtp &&
            mtp_depth == sp.mtp_n_max && mtp_pmin == sp.mtp_p_min && n_ctx == sp.n_ctx &&
            n_batch == sp.n_batch && n_ubatch == sp.n_ubatch && n_threads == sp.n_threads &&
            n_threads_batch == effective_threads_batch(sp.n_threads, sp.n_threads_batch) &&
            kv_q8 == sp.kv_q8)
            return session.get();
        session.reset(); // release the old model before loading the new one
        model.clear();
        ++generation;
        auto s = Session::create(sp, err);
        if (!s)
            return nullptr;
        session = std::move(s);
        model = model_id;
        gpu_layers = sp.n_gpu_layers;
        mtp = sp.mtp;
        mtp_depth = sp.mtp_n_max;
        mtp_pmin = sp.mtp_p_min;
        n_ctx = sp.n_ctx;
        n_batch = sp.n_batch;
        n_ubatch = sp.n_ubatch;
        n_threads = sp.n_threads;
        n_threads_batch = effective_threads_batch(sp.n_threads, sp.n_threads_batch);
        kv_q8 = sp.kv_q8;
        ++generation;
        return session.get();
    }

    // Under mtx: drop the resident session (e.g. to free RAM on demand).
    void reset_locked() {
        if (session) {
            session.reset();
            model.clear();
            ++generation;
        }
    }
};

// The process-wide hub (defined in session.cpp).
// SDK users may create their own SessionHub instance instead.
SessionHub& session_hub();

} // namespace xllama
