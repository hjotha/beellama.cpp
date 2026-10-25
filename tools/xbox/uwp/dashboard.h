// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// TV dashboard for the Xbox-as-server deployment.
//
// The owner runs this app on a console that sits on a shelf: no prompt is
// typed on the console, the host drives inference over the LAN. This screen is
// therefore read-only telemetry, designed for a 1080p TV at viewing distance.
//
// Two rules shape every number here:
//   1. Nothing is estimated. A metric is either read from the component that
//      produced it or rendered as "--" / "unavailable".
//   2. RKVA attention/RPC activity on the console is NOT token throughput.
//      End-to-end prefill/decode tok/s belongs to the host llama-server and is
//      fetched from ITS metrics endpoint; when that endpoint is not configured
//      or unreachable the dashboard says so instead of substituting a local
//      counter. The split case never falls back to local numbers.
//
// Collection (log tail + optional HTTP metrics poll) runs on one background
// thread at 1 Hz; the UI thread only copies a snapshot and repaints a bounded
// 300-sample in-memory history. No prompt or conversation text is read or
// displayed: the dashboard receives aggregate counters only.

#pragma once

#ifdef XLLAMA_UWP

    #include "pch.h"

    #include <chrono>
    #include <cstddef>
    #include <cstdint>
    #include <functional>
    #include <memory>
    #include <string>
    #include <utility>

namespace xllama::ui {

struct DashboardTelemetry;

// LAN/API link state rendered as a coloured row. The value itself is supplied
// by MainPage from xllama::api::server_status() (or "unavailable" in a Store
// build), so this module never reaches into the server.
enum class DashboardLink { Off, Pending, On, Error };

// Last successful/failed read of the host llama-server /metrics endpoint.
// `configured` means LocalState\dashboard.json named a URL; `ok` means the
// most recent poll returned parseable llama-server metrics.
struct DashboardHostSnapshot {
    bool configured = false;
    bool ok = false;
    std::wstring status{L"not configured"};
    std::wstring endpoint;
    std::wstring model_label;
    bool has_model = false;
    bool has_context_capacity = false;
    long long context_capacity = 0;
    bool has_prefill = false;
    bool has_decode = false;
    bool has_kv_ratio = false;
    bool has_kv_tokens = false;
    bool has_context_peak = false;
    bool has_requests = false;
    double prefill_tps = 0.0;
    double decode_tps = 0.0;
    double kv_ratio = 0.0;
    long long kv_tokens = 0;
    long long context_peak_tokens = 0;
    int requests_processing = 0;
};

// RKVA listener state, tailed from LocalState\remote-attn.jsonl. The log has
// no timestamps, so the dashboard shows state only -- never a made-up
// connection duration.
struct DashboardRkvaSnapshot {
    // absent is a confirmed "no log file" (explicit ERROR_FILE_NOT_FOUND), the
    // only state that permits local inference. present-but-unreadable/stale or
    // not-yet-polled is unknown: never local.
    bool absent = false;
    bool present = false;
    bool known = false;
    bool listening = false;
    bool connected = false;
    bool error = false;
    std::wstring status{L"not started"};
};

// Everything the controller knows that the dashboard may display. Prompt and
// conversation text are deliberately absent.
struct DashboardInputs {
    std::wstring model;
    std::wstring status_short{L"Starting"};
    bool model_ready = false;
    bool running = false;
    bool error = false;
    // Local (on-console) inference. Only used when RKVA is not connected.
    double local_prefill_tps = 0.0;
    bool has_local_prefill = false;
    double local_decode_tps = 0.0;
    bool has_local_decode = false;
    // True only after the first token of the current turn, i.e. a live decode
    // rate exists. During prefill `running` is true but there is no rate yet.
    bool local_live = false;
    // True for the first UI tick after a local turn completed, so history gets
    // the measured spike instead of a flat line at the last value.
    bool local_turn_fresh = false;
    // Context: exact n_ctx of the resident session and the exact token count
    // of the last turn's prompt. 0 means "unknown", never an estimate.
    long long ctx_capacity = 0;
    long long ctx_used = 0;
    size_t peak_ws_mb = 0;
    DashboardLink lan = DashboardLink::Off;
    std::wstring lan_detail;
    unsigned long long uptime_seconds = 0;
};

class Dashboard {
  public:
    Dashboard();
    ~Dashboard();
    Dashboard(const Dashboard&) = delete;
    Dashboard& operator=(const Dashboard&) = delete;

    winrt::Windows::UI::Xaml::Controls::Grid Root() const {
        return m_root;
    }

    // Starts the read-only sampler. Safe to call once, after BuildUI.
    void Start(const std::wstring& local_folder);
    // The Chat button hands control back to the controller.
    void OnChat(std::function<void()> handler) {
        m_on_chat = std::move(handler);
    }
    // UI thread, 1 Hz: repaint from a sampler snapshot.
    void Tick(const DashboardInputs& inputs);
    // UI thread: a completed local turn's measured rates (0 = not measured).
    void RecordTurn(double prefill_tps, double decode_tps);
    // Move gamepad focus to the one control on this screen.
    void FocusPrimary();

  private:
    struct Sample {
        double decode = -1.0;  // -1 = no data for this second
        double prefill = -1.0; // -1 = no prefill event in this second
    };

    // The history belongs to exactly one source. Any transition clears it so
    // local samples can never be drawn under split (or vice versa).
    enum class DisplayMode { Unknown, Local, SplitHost, SplitUnavailable };

    void Build();
    void PushSample(double decode, double prefill);
    void ResetHistory();
    // Classifies local/split/unknown from the current snapshots and resets the
    // multi-source presentation before the 500 ms repaint throttle. Returns
    // true when the mode changed.
    bool EnsureDisplayMode(const DashboardHostSnapshot& host, const DashboardRkvaSnapshot& rkva);
    void RedrawDecode();
    void RedrawPrefill();
    void ApplyBars();
    void UpdateClock();
    void UpdateStatus(const DashboardInputs& inputs, const DashboardHostSnapshot& host,
                      const DashboardRkvaSnapshot& rkva);
    void UpdateThroughput(const DashboardInputs& inputs, const DashboardHostSnapshot& host,
                          const DashboardRkvaSnapshot& rkva);
    void UpdateBars(const DashboardInputs& inputs, const DashboardHostSnapshot& host,
                    const DashboardRkvaSnapshot& rkva);
    void UpdateNetwork(const DashboardInputs& inputs, const DashboardHostSnapshot& host,
                       const DashboardRkvaSnapshot& rkva);

    winrt::Windows::UI::Xaml::Controls::Grid m_root{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_tagText{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_modelText{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_clockText{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_dateText{nullptr};
    winrt::Windows::UI::Xaml::Controls::Border m_statusPill{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_statusPillText{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_statusShort{nullptr};
    winrt::Windows::UI::Xaml::Controls::Button m_chatButton{nullptr};

    winrt::Windows::UI::Xaml::Controls::TextBlock m_tpsValue{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_tpsUnit{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_tpsState{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_tpsSource{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_prefillValue{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_turnDetail{nullptr};

    winrt::Windows::UI::Xaml::Controls::Canvas m_decodeCanvas{nullptr};
    winrt::Windows::UI::Xaml::Controls::Canvas m_prefillCanvas{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_decodeMax{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_prefillMax{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_graphSource{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_graphEmpty{nullptr};

    winrt::Windows::UI::Xaml::Controls::Grid m_memoryTrack{nullptr};
    winrt::Windows::UI::Xaml::Controls::Border m_memoryFill{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_memoryValue{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_memoryDetail{nullptr};

    winrt::Windows::UI::Xaml::Controls::Grid m_contextTrack{nullptr};
    winrt::Windows::UI::Xaml::Controls::Border m_contextFill{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_contextValue{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_contextDetail{nullptr};

    winrt::Windows::UI::Xaml::Controls::TextBlock m_lanValue{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_rkvaDot{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_rkvaValue{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_hostValue{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_uptimeValue{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_modeValue{nullptr};
    winrt::Windows::UI::Xaml::Controls::TextBlock m_note{nullptr};

    std::function<void()> m_on_chat;
    std::unique_ptr<DashboardTelemetry> m_telemetry;
    bool m_started = false;

    static constexpr size_t kWindowSeconds = 300; // 5 min at 1 Hz
    Sample m_history[kWindowSeconds]{};
    size_t m_history_next = 0;
    size_t m_history_count = 0;
    DisplayMode m_display_mode = DisplayMode::Unknown;
    double m_pending_prefill = -1.0;
    double m_last_host_prefill = -1.0;
    double m_memory_ratio = 0.0;
    double m_context_ratio = 0.0;
    bool m_have_last_tick = false;
    std::chrono::steady_clock::time_point m_last_tick{};
};

} // namespace xllama::ui

#endif // XLLAMA_UWP
