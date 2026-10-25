// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#ifdef XLLAMA_UWP

// clang-format off
#include "pch.h"
#include "dashboard.h"
// clang-format on

    #include "xllama/utf8_utils.h"

    #include <winrt/Windows.System.h>
    #include <winrt/Windows.UI.Xaml.Shapes.h>
    #include <winrt/Windows.Web.Http.h>

    #include <algorithm>
    #include <atomic>
    #include <cmath>
    #include <condition_variable>
    #include <cstdio>
    #include <cstdlib>
    #include <cstring>
    #include <ctime>
    #include <fstream>
    #include <mutex>
    #include <string_view>
    #include <thread>
    #include <vector>

using namespace winrt;
using namespace winrt::Windows::Foundation;
using namespace winrt::Windows::UI;
using namespace winrt::Windows::UI::Text;
using namespace winrt::Windows::UI::Xaml;
using namespace winrt::Windows::UI::Xaml::Controls;
using namespace winrt::Windows::UI::Xaml::Media;
namespace Shapes = winrt::Windows::UI::Xaml::Shapes;

namespace xllama::ui {
namespace {

// ---------------------------------------------------------------------------
// Palette. Dark, high-contrast, no themed brushes: the console's accent colour
// changes per user and a TV dashboard must render the same everywhere.
// ---------------------------------------------------------------------------
constexpr double kDesignWidth = 1824.0;
constexpr double kDesignHeight = 1026.0;
constexpr double kLeftWidth = 1176.0;
constexpr double kRightWidth = 624.0;
constexpr double kGap = 24.0;
constexpr double kGraphWidth = 1128.0;
constexpr double kDecodeHeight = 190.0;
constexpr double kPrefillHeight = 110.0;

Color argb(uint8_t a, uint8_t r, uint8_t g, uint8_t b) {
    return ColorHelper::FromArgb(a, r, g, b);
}
Color rgb(uint8_t r, uint8_t g, uint8_t b) {
    return argb(255, r, g, b);
}

const Color kBg = rgb(7, 10, 14);
const Color kCard = rgb(17, 23, 32);
const Color kCardAlt = rgb(22, 30, 41);
const Color kBorder = rgb(38, 50, 66);
const Color kGrid = rgb(27, 37, 50);
const Color kText = rgb(232, 238, 247);
const Color kMuted = rgb(143, 163, 184);
const Color kAccent = rgb(79, 195, 247);
const Color kGreen = rgb(76, 217, 130);
const Color kAmber = rgb(255, 184, 76);
const Color kRed = rgb(244, 96, 90);

constexpr wchar_t kUiFont[] = L"Segoe UI";
constexpr wchar_t kNumFont[] = L"Consolas";

SolidColorBrush brush(Color c) {
    return SolidColorBrush(c);
}

TextBlock make_text(const std::wstring& text, double size, Color color, bool semibold = false,
                    bool mono = false) {
    TextBlock block;
    block.Text(hstring(text));
    block.FontSize(size);
    block.Foreground(brush(color));
    block.FontFamily(FontFamily(hstring(mono ? kNumFont : kUiFont)));
    block.TextWrapping(TextWrapping::NoWrap);
    block.TextTrimming(TextTrimming::CharacterEllipsis);
    if (semibold)
        block.FontWeight(FontWeights::SemiBold());
    return block;
}

TextBlock make_label(const std::wstring& text) {
    TextBlock block = make_text(text, 18, kMuted, true);
    block.CharacterSpacing(120);
    return block;
}

Border make_card(double padding) {
    Border card;
    card.Background(brush(kCard));
    card.BorderBrush(brush(kBorder));
    card.BorderThickness(ThicknessHelper::FromUniformLength(1));
    card.CornerRadius(CornerRadius{8, 8, 8, 8});
    card.Padding(ThicknessHelper::FromUniformLength(padding));
    return card;
}

ColumnDefinition star_column() {
    ColumnDefinition column;
    column.Width(GridLengthHelper::FromValueAndType(1, GridUnitType::Star));
    return column;
}

ColumnDefinition auto_column() {
    ColumnDefinition column;
    column.Width(GridLengthHelper::Auto());
    return column;
}

ColumnDefinition fixed_column(double width) {
    ColumnDefinition column;
    column.Width(GridLengthHelper::FromValueAndType(width, GridUnitType::Pixel));
    return column;
}

RowDefinition auto_row() {
    RowDefinition row;
    row.Height(GridLengthHelper::Auto());
    return row;
}

RowDefinition fixed_row(double height) {
    RowDefinition row;
    row.Height(GridLengthHelper::FromValueAndType(height, GridUnitType::Pixel));
    return row;
}

RowDefinition star_row() {
    RowDefinition row;
    row.Height(GridLengthHelper::FromValueAndType(1, GridUnitType::Star));
    return row;
}

std::wstring group_digits(long long value) {
    const bool negative = value < 0;
    std::wstring digits = std::to_wstring(negative ? -value : value);
    for (int i = static_cast<int>(digits.size()) - 3; i > 0; i -= 3)
        digits.insert(static_cast<size_t>(i), L" ");
    return negative ? L"-" + digits : digits;
}

std::wstring rate_text(double value) {
    if (!std::isfinite(value) || value < 0.0)
        return L"--";
    wchar_t buf[64];
    if (value >= 1000.0)
        swprintf_s(buf, L"%.1fk", value / 1000.0);
    else if (value >= 100.0)
        swprintf_s(buf, L"%.0f", value);
    else
        swprintf_s(buf, L"%.1f", value);
    return buf;
}

std::wstring percent_text(double ratio) {
    if (!std::isfinite(ratio) || ratio < 0.0)
        return L"--";
    wchar_t buf[32];
    swprintf_s(buf, L"%.0f%%", ratio * 100.0);
    return buf;
}

std::wstring gb_text(uint64_t bytes) {
    wchar_t buf[64];
    swprintf_s(buf, L"%.1f GB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    return buf;
}

std::wstring mb_text(size_t mb) {
    wchar_t buf[64];
    swprintf_s(buf, L"%zu MB", mb);
    return buf;
}

std::wstring uptime_text(uint64_t seconds) {
    const uint64_t days = seconds / 86400;
    const uint64_t hours = (seconds % 86400) / 3600;
    const uint64_t minutes = (seconds % 3600) / 60;
    const uint64_t secs = seconds % 60;
    wchar_t buf[64];
    if (days > 0)
        swprintf_s(buf, L"%llud %02llu:%02llu:%02llu", days, hours, minutes, secs);
    else
        swprintf_s(buf, L"%02llu:%02llu:%02llu", hours, minutes, secs);
    return buf;
}

double nice_ceiling(double value) {
    if (!std::isfinite(value) || value <= 0.0)
        return 1.0;
    if (value <= 1.0)
        return 1.0;
    if (value <= 5.0)
        return 5.0;
    if (value <= 10.0)
        return 10.0;
    const double step = value <= 100.0 ? 10.0 : 100.0;
    return std::ceil(value / step) * step;
}

double clamp01(double value) {
    if (!std::isfinite(value))
        return 0.0;
    return std::max(0.0, std::min(1.0, value));
}

// ---------------------------------------------------------------------------
// Metrics parsing. Prometheus text is line-oriented; every metric this module
// consumes is a single unlabelled series in llama-server's /metrics output.
// Unknown or labelled series are accepted (the value is the last token).
// ---------------------------------------------------------------------------
bool metric_value(const std::string& body, const char* name, double* out) {
    const size_t name_len = std::strlen(name);
    size_t pos = 0;
    while (pos < body.size()) {
        size_t eol = body.find('\n', pos);
        if (eol == std::string::npos)
            eol = body.size();
        std::string_view line(body.data() + pos, eol - pos);
        if (!line.empty() && line.back() == '\r')
            line.remove_suffix(1);
        if (line.size() > name_len && line.compare(0, name_len, name) == 0) {
            const char next = line[name_len];
            if (next == ' ' || next == '\t' || next == '{') {
                const size_t split = line.find_last_of(" \t");
                if (split != std::string_view::npos && split + 1 < line.size()) {
                    const std::string token(line.substr(split + 1));
                    char* end = nullptr;
                    const double value = std::strtod(token.c_str(), &end);
                    if (end && *end == '\0' && std::isfinite(value)) {
                        *out = value;
                        return true;
                    }
                }
            }
        }
        pos = eol + 1;
    }
    return false;
}

bool metrics_url(const std::wstring& url) {
    if (url.rfind(L"http://", 0) != 0 && url.rfind(L"https://", 0) != 0)
        return false;
    // No userinfo: the dashboard must not put credentials in a URL, display
    // them on a TV, or imply an auth mechanism that does not exist.
    return url.find(L'@') == std::wstring::npos;
}

std::wstring endpoint_label(const std::wstring& url) {
    std::wstring label = url;
    const size_t scheme = label.find(L"://");
    if (scheme != std::wstring::npos)
        label = label.substr(scheme + 3);
    const size_t at = label.find(L'@');
    if (at != std::wstring::npos)
        label = label.substr(at + 1);
    const size_t slash = label.find(L'/');
    if (slash != std::wstring::npos)
        label = label.substr(0, slash);
    if (label.empty())
        label = L"host";
    return label;
}

// Minimal reader for LocalState\dashboard.json. One optional key keeps this
// dependency-free; the rest of the app's settings parser lives in MainPage.cpp.
std::wstring read_host_metrics_url(const std::wstring& local_folder) {
    const std::wstring path = local_folder + L"\\dashboard.json";
    std::ifstream file(path.c_str(), std::ios::binary);
    if (!file)
        return {};
    std::string text;
    char buf[4096];
    while (file.read(buf, sizeof(buf)) || file.gcount() > 0) {
        text.append(buf, static_cast<size_t>(file.gcount()));
        if (text.size() > 16384)
            break;
    }
    const std::string key = "\"host_metrics_url\"";
    size_t pos = text.find(key);
    if (pos == std::string::npos)
        return {};
    pos = text.find(':', pos + key.size());
    if (pos == std::string::npos)
        return {};
    ++pos;
    while (pos < text.size() &&
           (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\r' || text[pos] == '\n'))
        ++pos;
    if (pos >= text.size() || text[pos] != '"')
        return {};
    ++pos;
    std::string value;
    while (pos < text.size() && text[pos] != '"') {
        if (text[pos] == '\\' && pos + 1 < text.size()) {
            ++pos;
            if (text[pos] == 'n' || text[pos] == 'r' || text[pos] == 't') {
                ++pos;
                continue;
            }
        }
        value.push_back(text[pos++]);
    }
    return ::xllama::utf8_to_wstring(value);
}

// Minimal JSON string reader for the two keys this module needs from
// llama-server /props. The app's general JSON helpers live in
// include/xllama/json_utils.h; this stays local because the dashboard must not
// grow a second general-purpose parser.
std::string json_string_value(const std::string& body, const char* key) {
    const std::string quoted = std::string("\"") + key + "\"";
    size_t pos = body.find(quoted);
    if (pos == std::string::npos)
        return {};
    pos = body.find(':', pos + quoted.size());
    if (pos == std::string::npos)
        return {};
    ++pos;
    while (pos < body.size() &&
           (body[pos] == ' ' || body[pos] == '\t' || body[pos] == '\r' || body[pos] == '\n'))
        ++pos;
    if (pos >= body.size() || body[pos] != '"')
        return {};
    ++pos;
    std::string value;
    while (pos < body.size() && body[pos] != '"') {
        if (body[pos] == '\\' && pos + 1 < body.size()) {
            ++pos;
            switch (body[pos]) {
            case 'n':
                value.push_back('\n');
                break;
            case 'r':
                value.push_back('\r');
                break;
            case 't':
                value.push_back('\t');
                break;
            default:
                value.push_back(body[pos]);
                break;
            }
            ++pos;
            continue;
        }
        value.push_back(body[pos++]);
    }
    return value;
}

// n_ctx lives inside default_generation_settings; anchoring the search on the
// object key avoids picking up another n_ctx from nested generation params.
bool json_n_ctx(const std::string& body, long long* out) {
    size_t pos = body.find("\"default_generation_settings\"");
    if (pos == std::string::npos)
        return false;
    pos = body.find('{', pos);
    if (pos == std::string::npos)
        return false;
    pos = body.find("\"n_ctx\"", pos);
    if (pos == std::string::npos)
        return false;
    pos = body.find(':', pos);
    if (pos == std::string::npos)
        return false;
    ++pos;
    while (pos < body.size() && (body[pos] == ' ' || body[pos] == '\t'))
        ++pos;
    char* end = nullptr;
    const long long value = std::strtoll(body.c_str() + pos, &end, 10);
    if (end == body.c_str() + pos || value <= 0)
        return false;
    *out = value;
    return true;
}

std::string basename_utf8(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

// llama-server exposes /props next to /metrics. Deriving it from the configured
// metrics URL keeps one config key and no second host setting.
std::wstring props_url_from_metrics(const std::wstring& metrics_url) {
    const size_t query = metrics_url.find(L'?');
    const std::wstring base =
        query == std::wstring::npos ? metrics_url : metrics_url.substr(0, query);
    const std::wstring tail =
        query == std::wstring::npos ? std::wstring{} : metrics_url.substr(query);
    const std::wstring suffix = L"/metrics";
    if (base.size() <= suffix.size() ||
        base.compare(base.size() - suffix.size(), suffix.size(), suffix) != 0)
        return {};
    return base.substr(0, base.size() - suffix.size()) + L"/props" + tail;
}

uint64_t filetime_value(const FILETIME& time) {
    ULARGE_INTEGER value{};
    value.LowPart = time.dwLowDateTime;
    value.HighPart = time.dwHighDateTime;
    return value.QuadPart;
}

uint64_t steady_us() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
}

} // namespace

// ---------------------------------------------------------------------------
// DashboardTelemetry -- the only code that touches disk/network for this
// screen. One thread, 1 Hz, snapshot under a mutex. It never reads prompt or
// conversation files; the RKVA log it tails carries link/engine events only.
// ---------------------------------------------------------------------------
struct DashboardTelemetry {
    DashboardTelemetry(std::wstring local_folder, uint64_t process_start_100ns)
        : m_folder(std::move(local_folder)), m_process_start(process_start_100ns) {
        // Two independent workers: the log tail stays at 1 Hz even while the
        // optional HTTP metrics endpoint times out. Snapshots are mutex-guarded
        // and expire if a worker dies.
        m_thread = std::thread([this]() {
            try {
                Run();
            } catch (...) {
                // Telemetry must never take the app down; TTL exposes the stop.
            }
        });
        m_host_thread = std::thread([this]() {
            try {
                RunHost();
            } catch (...) {
                // Same: no exception escapes; the host snapshot goes stale.
            }
        });
    }

    ~DashboardTelemetry() {
        m_stop.store(true);
        m_wait.notify_all();
        m_host_wait.notify_all();
        if (m_thread.joinable())
            m_thread.join();
        if (m_host_thread.joinable())
            m_host_thread.join();
    }

    DashboardHostSnapshot Host() {
        std::lock_guard<std::mutex> lock(m_mutex);
        DashboardHostSnapshot snapshot = m_host;
        const uint64_t last = m_host_last_us.load();
        if (snapshot.configured && last != 0 && steady_us() - last > 20'000'000ULL) {
            // Worker died or stalled: never keep old model/context/KV alive.
            snapshot.ok = false;
            snapshot.has_model = false;
            snapshot.model_label.clear();
            snapshot.has_context_capacity = false;
            snapshot.context_capacity = 0;
            snapshot.has_prefill = false;
            snapshot.has_decode = false;
            snapshot.has_kv_ratio = false;
            snapshot.has_kv_tokens = false;
            snapshot.has_context_peak = false;
            snapshot.has_requests = false;
            snapshot.status = L"stale";
        }
        return snapshot;
    }

    DashboardRkvaSnapshot Rkva() {
        std::lock_guard<std::mutex> lock(m_mutex);
        DashboardRkvaSnapshot snapshot = m_rkva;
        const uint64_t last = m_rkva_last_us.load();
        if (last != 0 && steady_us() - last > 5'000'000ULL) {
            // A stopped/stalled log sampler cannot confirm local or split.
            snapshot.absent = false;
            snapshot.known = false;
            snapshot.listening = false;
            snapshot.connected = false;
            snapshot.error = false;
            snapshot.status = L"stale";
        }
        return snapshot;
    }

  private:
    void Run() {
        try {
            winrt::init_apartment(winrt::apartment_type::multi_threaded);
        } catch (...) {
            // A thread that is already in an apartment still works for the
            // WinRT calls below; only the initialisation is best-effort.
        }
        auto next = std::chrono::steady_clock::now();
        while (!m_stop.load()) {
            PollRkva();
            next += std::chrono::seconds(1);
            const auto now = std::chrono::steady_clock::now();
            if (next < now)
                next = now; // drop missed slots; do not burst-catch-up
            std::unique_lock<std::mutex> lock(m_wait_mutex);
            if (m_wait.wait_until(lock, next, [this]() { return m_stop.load(); }))
                break;
        }
    }

    void RunHost() {
        try {
            winrt::init_apartment(winrt::apartment_type::multi_threaded);
        } catch (...) {
        }
        RefreshConfig();
        if (!m_url.empty() && metrics_url(m_url))
            PollHost();
        auto next = std::chrono::steady_clock::now();
        while (!m_stop.load()) {
            next += std::chrono::seconds(10);
            const auto now = std::chrono::steady_clock::now();
            if (next < now)
                next = now; // drop missed slots; do not burst-catch-up
            std::unique_lock<std::mutex> lock(m_host_wait_mutex);
            if (m_host_wait.wait_until(lock, next, [this]() { return m_stop.load(); }))
                break;
            RefreshConfig();
            if (!m_url.empty() && metrics_url(m_url))
                PollHost();
        }
    }

    void RefreshConfig() {
        const std::wstring url = read_host_metrics_url(m_folder);
        if (url == m_url)
            return;
        m_url = url;
        m_props_url = props_url_from_metrics(url);
        m_host_polls = 0;
        DashboardHostSnapshot snapshot;
        if (url.empty()) {
            snapshot.configured = false;
            snapshot.status = L"not configured";
        } else if (!metrics_url(url)) {
            snapshot.configured = true;
            snapshot.status = L"invalid url";
        } else {
            snapshot.configured = true;
            snapshot.status = L"waiting";
            snapshot.endpoint = endpoint_label(url);
        }
        std::lock_guard<std::mutex> lock(m_mutex);
        m_host = snapshot;
    }

    bool HttpGet(const std::wstring& url, std::string* body, std::wstring* error) {
        try {
            const Uri uri{hstring(url)};
            winrt::Windows::Web::Http::HttpClient client;
            auto operation = client.GetAsync(
                uri, winrt::Windows::Web::Http::HttpCompletionOption::ResponseContentRead);

            // Windows.Web.Http has no Timeout property. Cancel the operation
            // from a watchdog instead of letting a silent host freeze the
            // sampler (and with it the RKVA state) for the HTTP default.
            std::mutex gate;
            std::condition_variable gate_cv;
            bool finished = false;
            std::thread watchdog([&]() {
                try {
                    winrt::init_apartment(winrt::apartment_type::multi_threaded);
                } catch (...) {
                }
                std::unique_lock<std::mutex> lock(gate);
                if (!gate_cv.wait_for(lock, std::chrono::seconds(3), [&]() { return finished; })) {
                    try {
                        operation.Cancel();
                    } catch (...) {
                    }
                }
            });
            const auto finish_watchdog = [&]() {
                {
                    std::lock_guard<std::mutex> lock(gate);
                    finished = true;
                }
                gate_cv.notify_all();
                watchdog.join();
            };

            winrt::Windows::Web::Http::HttpResponseMessage response{nullptr};
            try {
                response = operation.get();
            } catch (const winrt::hresult_canceled&) {
                finish_watchdog();
                *error = L"timeout";
                return false;
            } catch (const winrt::hresult_error& e) {
                const bool canceled = e.code().value == HRESULT_FROM_WIN32(ERROR_CANCELLED) ||
                                      e.code().value == E_ABORT;
                finish_watchdog();
                *error = canceled ? L"timeout" : L"request failed";
                return false;
            } catch (...) {
                finish_watchdog();
                *error = L"request failed";
                return false;
            }
            finish_watchdog();

            if (!response.IsSuccessStatusCode()) {
                *error = L"HTTP " + std::to_wstring(static_cast<int>(response.StatusCode()));
                return false;
            }
            *body = winrt::to_string(response.Content().ReadAsStringAsync().get());
            return true;
        } catch (const winrt::hresult_error& e) {
            const bool canceled =
                e.code().value == HRESULT_FROM_WIN32(ERROR_CANCELLED) || e.code().value == E_ABORT;
            *error = canceled ? L"timeout" : L"request failed";
            return false;
        } catch (...) {
            *error = L"request failed";
            return false;
        }
    }

    void PollHost() {
        DashboardHostSnapshot previous;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            previous = m_host;
        }
        DashboardHostSnapshot snapshot;
        snapshot.configured = previous.configured;
        snapshot.endpoint = previous.endpoint;
        snapshot.model_label = previous.model_label;
        snapshot.has_model = previous.has_model;
        snapshot.context_capacity = previous.context_capacity;
        snapshot.has_context_capacity = previous.has_context_capacity;
        ++m_host_polls;

        std::string body;
        std::wstring error;
        if (!HttpGet(m_url, &body, &error)) {
            // A failed metrics read invalidates every derived gauge. Keeping
            // the old model/context/KV would present stale data as current.
            snapshot = DashboardHostSnapshot{};
            snapshot.configured = previous.configured;
            snapshot.endpoint = previous.endpoint;
            snapshot.status = error;
        } else {
            double value = 0.0;
            snapshot.has_prefill = metric_value(body, "llamacpp:prompt_tokens_seconds", &value);
            if (snapshot.has_prefill)
                snapshot.prefill_tps = std::max(0.0, value);
            snapshot.has_decode = metric_value(body, "llamacpp:predicted_tokens_seconds", &value);
            if (snapshot.has_decode)
                snapshot.decode_tps = std::max(0.0, value);
            snapshot.has_kv_ratio = metric_value(body, "llamacpp:kv_cache_usage_ratio", &value);
            if (snapshot.has_kv_ratio)
                snapshot.kv_ratio = clamp01(value);
            snapshot.has_kv_tokens = metric_value(body, "llamacpp:kv_cache_tokens", &value);
            if (snapshot.has_kv_tokens)
                snapshot.kv_tokens = static_cast<long long>(std::max(0.0, value));
            // This fork exposes the largest observed sequence length, not the
            // KV usage ratio. It is a real host number and is labelled as a
            // high-watermark; no capacity is inferred from it alone.
            snapshot.has_context_peak = metric_value(body, "llamacpp:n_tokens_max", &value);
            if (snapshot.has_context_peak)
                snapshot.context_peak_tokens = static_cast<long long>(std::max(0.0, value));
            snapshot.has_requests = metric_value(body, "llamacpp:requests_processing", &value);
            if (snapshot.has_requests)
                snapshot.requests_processing = static_cast<int>(std::max(0.0, value));

            const bool any = snapshot.has_prefill || snapshot.has_decode || snapshot.has_kv_ratio ||
                             snapshot.has_kv_tokens || snapshot.has_context_peak ||
                             snapshot.has_requests;
            if (any) {
                snapshot.ok = true;
                snapshot.status = L"online";
            } else {
                snapshot.status = L"no llama-server metrics";
            }
        }

        // /props is static: fetch it on the first poll and every 30 seconds
        // (three 10 s host polls). It supplies the host model alias and n_ctx,
        // which /metrics does not expose. A failed /props explicitly clears
        // the previous model/capacity instead of presenting it as current.
        if (!m_props_url.empty() && (m_host_polls == 1 || m_host_polls % 3 == 0)) {
            std::string props;
            std::wstring props_error;
            if (HttpGet(m_props_url, &props, &props_error)) {
                // A 200 with an empty/incomplete body must not preserve the
                // previous model/capacity; renew only what this response has.
                snapshot.has_model = false;
                snapshot.model_label.clear();
                snapshot.has_context_capacity = false;
                snapshot.context_capacity = 0;
                std::string label = json_string_value(props, "model_alias");
                if (label.empty())
                    label = basename_utf8(json_string_value(props, "model_path"));
                if (!label.empty()) {
                    snapshot.model_label = ::xllama::utf8_to_wstring(label);
                    snapshot.has_model = true;
                }
                long long n_ctx = 0;
                if (json_n_ctx(props, &n_ctx)) {
                    snapshot.context_capacity = n_ctx;
                    snapshot.has_context_capacity = true;
                }
            } else {
                snapshot.has_model = false;
                snapshot.model_label.clear();
                snapshot.has_context_capacity = false;
                snapshot.context_capacity = 0;
            }
        }

        m_host_last_us.store(steady_us());
        std::lock_guard<std::mutex> lock(m_mutex);
        m_host = snapshot;
    }

    void PollRkva() {
        const std::wstring path = m_folder + L"\\remote-attn.jsonl";
        WIN32_FILE_ATTRIBUTE_DATA attributes{};
        if (!GetFileAttributesExW(path.c_str(), GetFileExInfoStandard, &attributes)) {
            const DWORD error = GetLastError();
            DashboardRkvaSnapshot snapshot;
            if (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND) {
                // Confirmed absence is the ONLY state that permits local mode.
                snapshot.absent = true;
                snapshot.status = L"not started";
            } else {
                // Access denied / transient I/O failure: unknown, not local.
                snapshot.present = true;
                snapshot.status = L"log unavailable";
            }
            SetRkva(snapshot);
            m_rkva_offset = 0;
            m_rkva_last_us.store(steady_us());
            return;
        }
        // A log last written before this process started describes a previous
        // run. It is "unknown", never "local confirmed".
        if (filetime_value(attributes.ftLastWriteTime) < m_process_start) {
            DashboardRkvaSnapshot stale;
            stale.present = true;
            stale.status = L"stale log";
            SetRkva(stale);
            m_rkva_offset = 0;
            m_rkva_last_us.store(steady_us());
            return;
        }

        std::ifstream file(path.c_str(), std::ios::binary);
        file.seekg(0, std::ios::end);
        const std::streamoff size =
            file ? static_cast<std::streamoff>(file.tellg()) : std::streamoff(-1);
        if (!file || size < 0) {
            DashboardRkvaSnapshot unreadable;
            unreadable.present = true;
            unreadable.status = L"log unreadable";
            SetRkva(unreadable);
            m_rkva_last_us.store(steady_us());
            return;
        }
        const uint64_t total = static_cast<uint64_t>(size);
        if (m_rkva_offset > total)
            m_rkva_offset = 0;

        // The producer appends. The last listen event is the boundary of the
        // current listener run; scan backwards once so a limited tail cannot
        // omit the current accept or reuse an old one.
        if (m_rkva_offset == 0) {
            constexpr uint64_t kScanChunk = 256 * 1024;
            constexpr uint64_t kMaxScanChunks = 64; // bounded startup cost: 16 MiB
            uint64_t scanned = 0;
            uint64_t end = total;
            uint64_t found = 0;
            bool have = false;
            while (end > 0 && scanned < kMaxScanChunks) {
                if (m_stop.load())
                    return;
                ++scanned;
                const uint64_t begin = end > kScanChunk ? end - kScanChunk : 0;
                std::string scan(static_cast<size_t>(end - begin), '\0');
                file.seekg(static_cast<std::streamoff>(begin));
                file.read(scan.data(), static_cast<std::streamsize>(scan.size()));
                scan.resize(static_cast<size_t>(file.gcount()));
                const size_t hit = scan.rfind("\"phase\":\"listen\"");
                if (hit != std::string::npos) {
                    const size_t line = scan.rfind('\n', hit);
                    found = begin + (line == std::string::npos ? 0 : line + 1);
                    have = true;
                    break;
                }
                if (begin == 0)
                    break;
                end = begin + 64; // overlap a token split across chunks
            }
            if (!have) {
                DashboardRkvaSnapshot unknown;
                unknown.present = true;
                unknown.status = L"no current listen event";
                SetRkva(unknown);
                m_rkva_offset = total;
                m_rkva_last_us.store(steady_us());
                return;
            }
            m_rkva_offset = found;
        }

        const uint64_t remaining = total - m_rkva_offset;
        constexpr size_t kChunk = 256 * 1024;
        const size_t wanted = static_cast<size_t>(std::min<uint64_t>(remaining, kChunk));
        if (wanted == 0) {
            m_rkva_last_us.store(steady_us());
            return;
        }
        std::string chunk(wanted, '\0');
        file.seekg(static_cast<std::streamoff>(m_rkva_offset));
        file.read(chunk.data(), static_cast<std::streamsize>(wanted));
        const size_t got = static_cast<size_t>(file.gcount());
        chunk.resize(got);
        const size_t newline = chunk.rfind('\n');
        if (newline == std::string::npos) {
            if (got == kChunk)
                m_rkva_offset += got;
            m_rkva_last_us.store(steady_us());
            return;
        }
        chunk.resize(newline + 1);
        m_rkva_offset += newline + 1;

        DashboardRkvaSnapshot snapshot;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            snapshot = m_rkva;
        }
        size_t pos = 0;
        while (pos < chunk.size()) {
            size_t eol = chunk.find('\n', pos);
            if (eol == std::string::npos)
                eol = chunk.size();
            const std::string_view line(chunk.data() + pos, eol - pos);
            pos = eol + 1;
            const auto has = [&](const char* text) {
                return line.find(text) != std::string_view::npos;
            };
            if (has("\"phase\":\"listen\"")) {
                // A listen event starts a new listener run: reset any state
                // from the previous run, including an old "connected".
                snapshot = DashboardRkvaSnapshot{};
                snapshot.present = true;
                snapshot.known = true;
                if (has("\"status\":\"ready\"")) {
                    snapshot.listening = true;
                    snapshot.status = L"listening on :50054";
                } else if (has("owner_already_active")) {
                    snapshot.listening = true;
                    snapshot.status = L"listener already active";
                } else {
                    snapshot.listening = false;
                    snapshot.status = L"listener unavailable";
                }
            } else if (has("\"phase\":\"accept\"")) {
                if (!snapshot.known)
                    continue; // accept from an earlier run
                if (has("\"status\":\"connected\"")) {
                    snapshot.listening = true;
                    snapshot.connected = true;
                    snapshot.error = false;
                    snapshot.status = L"host connected";
                } else if (has("\"status\":\"disconnected\"")) {
                    snapshot.connected = false;
                    snapshot.status =
                        snapshot.listening ? L"waiting for host" : L"listener stopped";
                }
            } else if (has("\"phase\":\"engine_error\"") || has("\"phase\":\"gpu_retirement\"")) {
                if (!snapshot.known)
                    continue;
                snapshot.error = true;
                // An engine error is a terminal link state: never keep the
                // connected flag that would let host TPS/context through.
                snapshot.connected = false;
                snapshot.status = L"engine error";
            }
        }
        SetRkva(snapshot);
        m_rkva_last_us.store(steady_us());
    }

    void SetRkva(const DashboardRkvaSnapshot& snapshot) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_rkva = snapshot;
    }

    std::wstring m_folder;
    uint64_t m_process_start = 0;
    std::wstring m_url;
    std::wstring m_props_url;
    uint64_t m_host_polls = 0;
    uint64_t m_rkva_offset = 0;
    std::mutex m_mutex;
    std::mutex m_wait_mutex;
    std::condition_variable m_wait;
    std::mutex m_host_wait_mutex;
    std::condition_variable m_host_wait;
    std::atomic<uint64_t> m_host_last_us{0};
    std::atomic<uint64_t> m_rkva_last_us{0};
    std::atomic<bool> m_stop{false};
    // Declared before the thread: the worker reads both snapshots as soon as
    // it starts, so their construction must precede it.
    DashboardHostSnapshot m_host;
    DashboardRkvaSnapshot m_rkva;
    std::thread m_thread;
    std::thread m_host_thread;
};

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------
Dashboard::Dashboard() {
    Build();
}

Dashboard::~Dashboard() = default;

void Dashboard::Start(const std::wstring& local_folder) {
    if (m_started)
        return;
    m_started = true;
    FILETIME creation{}, exit{}, kernel{}, user{};
    uint64_t process_start = 0;
    if (GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user)) {
        process_start = filetime_value(creation);
    } else {
        // Rare fallback: a conservative Start-time boundary, never zero.
        FILETIME now{};
        GetSystemTimeAsFileTime(&now);
        process_start = filetime_value(now);
    }
    m_telemetry = std::make_unique<DashboardTelemetry>(local_folder, process_start);
}

void Dashboard::FocusPrimary() {
    if (m_chatButton)
        m_chatButton.Focus(FocusState::Programmatic);
}

void Dashboard::RecordTurn(double prefill_tps, double decode_tps) {
    (void)decode_tps; // the live counter and the controller supply decode samples
    if (prefill_tps > 0.0 && std::isfinite(prefill_tps))
        m_pending_prefill = prefill_tps;
}

void Dashboard::Build() {
    m_root = Grid();
    m_root.Background(brush(kBg));

    // Fixed 1824x1026 design, uniformly scaled. On a 1080p TV with the 5% safe
    // area this is exactly 1:1; on a desktop window it shrinks instead of
    // reflowing into an unreadable pile.
    Viewbox viewbox;
    viewbox.Stretch(Stretch::Uniform);
    viewbox.StretchDirection(StretchDirection::Both);
    viewbox.HorizontalAlignment(HorizontalAlignment::Center);
    viewbox.VerticalAlignment(VerticalAlignment::Center);
    viewbox.Margin(ThicknessHelper::FromLengths(48, 27, 48, 27)); // TV safe area

    Grid design;
    design.Width(kDesignWidth);
    design.Height(kDesignHeight);
    design.RowDefinitions().Append(fixed_row(120));
    design.RowDefinitions().Append(fixed_row(20));
    design.RowDefinitions().Append(star_row());
    design.RowDefinitions().Append(fixed_row(20));
    design.RowDefinitions().Append(fixed_row(40));
    viewbox.Child(design);
    m_root.Children().Append(viewbox);

    // ---- header -----------------------------------------------------------
    Grid header;
    Grid::SetRow(header, 0);
    header.ColumnDefinitions().Append(star_column());
    header.ColumnDefinitions().Append(auto_column());

    {
        StackPanel left;
        m_tagText = make_label(L"XBOX  \u00b7  LOCAL INFERENCE SERVER");
        m_tagText.FontSize(18);
        m_tagText.Margin(ThicknessHelper::FromLengths(0, 0, 0, 4));
        left.Children().Append(m_tagText);

        m_modelText = make_text(L"xllama", 44, kText, true);
        m_modelText.MaxWidth(1220);
        left.Children().Append(m_modelText);

        StackPanel status_row;
        status_row.Orientation(Orientation::Horizontal);
        status_row.Margin(ThicknessHelper::FromLengths(0, 6, 0, 0));

        m_statusPill = Border();
        m_statusPill.CornerRadius(CornerRadius{4, 4, 4, 4});
        m_statusPill.Padding(ThicknessHelper::FromLengths(12, 3, 12, 3));
        m_statusPill.VerticalAlignment(VerticalAlignment::Center);
        m_statusPillText = make_text(L"STARTING", 17, kAmber, true);
        m_statusPill.Child(m_statusPillText);
        status_row.Children().Append(m_statusPill);

        m_statusShort = make_text(L"Loading model", 18, kMuted);
        m_statusShort.Margin(ThicknessHelper::FromLengths(12, 0, 0, 0));
        m_statusShort.VerticalAlignment(VerticalAlignment::Center);
        status_row.Children().Append(m_statusShort);

        left.Children().Append(status_row);
        header.Children().Append(left);
        Grid::SetColumn(left, 0);
    }

    {
        StackPanel right;
        right.HorizontalAlignment(HorizontalAlignment::Right);

        m_clockText = make_text(L"--:--:--", 40, kText, true, true);
        m_clockText.HorizontalAlignment(HorizontalAlignment::Right);
        right.Children().Append(m_clockText);

        m_dateText = make_text(L"---", 18, kMuted, true);
        m_dateText.HorizontalAlignment(HorizontalAlignment::Right);
        m_dateText.CharacterSpacing(80);
        right.Children().Append(m_dateText);

        m_chatButton = Button();
        m_chatButton.Content(box_value(L"Open chat  \u203a"));
        m_chatButton.FontSize(20);
        m_chatButton.FontFamily(FontFamily(hstring(kUiFont)));
        m_chatButton.Foreground(brush(kText));
        m_chatButton.Background(brush(kCardAlt));
        m_chatButton.BorderBrush(brush(kBorder));
        m_chatButton.BorderThickness(ThicknessHelper::FromUniformLength(1));
        m_chatButton.CornerRadius(CornerRadius{6, 6, 6, 6});
        m_chatButton.Padding(ThicknessHelper::FromLengths(22, 7, 22, 7));
        m_chatButton.MinHeight(38);
        m_chatButton.Margin(ThicknessHelper::FromLengths(0, 10, 0, 0));
        m_chatButton.HorizontalAlignment(HorizontalAlignment::Right);
        m_chatButton.Click([this](IInspectable const&, RoutedEventArgs const&) {
            if (m_on_chat)
                m_on_chat();
        });
        right.Children().Append(m_chatButton);

        header.Children().Append(right);
        Grid::SetColumn(right, 1);
    }
    design.Children().Append(header);

    // ---- content ----------------------------------------------------------
    Grid content;
    Grid::SetRow(content, 2);
    content.ColumnDefinitions().Append(fixed_column(kLeftWidth));
    content.ColumnDefinitions().Append(fixed_column(kGap));
    content.ColumnDefinitions().Append(fixed_column(kRightWidth));
    design.Children().Append(content);

    // Left column: throughput hero + history.
    Grid left;
    Grid::SetColumn(left, 0);
    left.RowDefinitions().Append(fixed_row(300));
    left.RowDefinitions().Append(fixed_row(20));
    left.RowDefinitions().Append(star_row());
    content.Children().Append(left);

    // Right column: memory, context, network.
    Grid right;
    Grid::SetColumn(right, 2);
    right.RowDefinitions().Append(fixed_row(220));
    right.RowDefinitions().Append(fixed_row(16));
    right.RowDefinitions().Append(fixed_row(220));
    right.RowDefinitions().Append(fixed_row(16));
    right.RowDefinitions().Append(star_row());
    content.Children().Append(right);

    // ---- throughput hero --------------------------------------------------
    {
        Border card = make_card(28);
        Grid::SetRow(card, 0);
        Grid inner;
        inner.RowDefinitions().Append(auto_row());
        inner.RowDefinitions().Append(star_row());
        inner.RowDefinitions().Append(auto_row());

        Grid top;
        top.ColumnDefinitions().Append(star_column());
        top.ColumnDefinitions().Append(auto_column());
        TextBlock label = make_label(L"DECODE THROUGHPUT");
        label.FontSize(18);
        top.Children().Append(label);

        Border source_badge = Border();
        source_badge.Background(brush(kCardAlt));
        source_badge.CornerRadius(CornerRadius{4, 4, 4, 4});
        source_badge.Padding(ThicknessHelper::FromLengths(10, 3, 10, 3));
        source_badge.HorizontalAlignment(HorizontalAlignment::Right);
        m_tpsSource = make_text(L"WAITING", 15, kMuted, true);
        m_tpsSource.CharacterSpacing(80);
        source_badge.Child(m_tpsSource);
        top.Children().Append(source_badge);
        Grid::SetColumn(source_badge, 1);
        inner.Children().Append(top);

        Grid hero;
        Grid::SetRow(hero, 1);
        hero.ColumnDefinitions().Append(auto_column());
        hero.ColumnDefinitions().Append(star_column());
        m_tpsValue = make_text(L"--", 128, kAccent, true, true);
        m_tpsValue.VerticalAlignment(VerticalAlignment::Center);
        hero.Children().Append(m_tpsValue);

        StackPanel hero_right;
        hero_right.VerticalAlignment(VerticalAlignment::Bottom);
        hero_right.Margin(ThicknessHelper::FromLengths(18, 0, 0, 22));
        m_tpsUnit = make_text(L"tok/s", 28, kMuted, true);
        hero_right.Children().Append(m_tpsUnit);
        m_tpsState = make_text(L"WAITING FOR A TURN", 16, kMuted, true);
        m_tpsState.CharacterSpacing(80);
        m_tpsState.Margin(ThicknessHelper::FromLengths(0, 4, 0, 0));
        hero_right.Children().Append(m_tpsState);
        hero.Children().Append(hero_right);
        Grid::SetColumn(hero_right, 1);
        inner.Children().Append(hero);

        StackPanel bottom;
        bottom.Orientation(Orientation::Horizontal);
        bottom.VerticalAlignment(VerticalAlignment::Bottom);
        bottom.Margin(ThicknessHelper::FromLengths(0, 8, 0, 0));
        m_prefillValue = make_text(L"PREFILL  --", 22, kAmber, true);
        bottom.Children().Append(m_prefillValue);
        TextBlock dot = make_text(L"\u00b7", 22, kMuted);
        dot.Margin(ThicknessHelper::FromLengths(14, 0, 14, 0));
        bottom.Children().Append(dot);
        m_turnDetail = make_text(L"no turn measured yet", 18, kMuted);
        m_turnDetail.VerticalAlignment(VerticalAlignment::Center);
        m_turnDetail.MaxWidth(720);
        bottom.Children().Append(m_turnDetail);
        Grid::SetRow(bottom, 2);
        inner.Children().Append(bottom);

        card.Child(inner);
        left.Children().Append(card);
    }

    // ---- history card -----------------------------------------------------
    {
        Border card = make_card(24);
        Grid::SetRow(card, 2);
        Grid inner;
        inner.RowDefinitions().Append(auto_row()); // title
        inner.RowDefinitions().Append(auto_row()); // decode label
        inner.RowDefinitions().Append(fixed_row(kDecodeHeight));
        inner.RowDefinitions().Append(auto_row()); // prefill label
        inner.RowDefinitions().Append(fixed_row(kPrefillHeight));
        inner.RowDefinitions().Append(auto_row()); // axis

        Grid title;
        title.ColumnDefinitions().Append(star_column());
        title.ColumnDefinitions().Append(auto_column());
        TextBlock history = make_label(L"HISTORY  \u00b7  LAST 5 MIN");
        title.Children().Append(history);
        m_graphSource = make_text(L"source: none", 15, kMuted);
        m_graphSource.HorizontalAlignment(HorizontalAlignment::Right);
        title.Children().Append(m_graphSource);
        Grid::SetColumn(m_graphSource, 1);
        inner.Children().Append(title);

        Grid decode_label;
        Grid::SetRow(decode_label, 1);
        decode_label.Margin(ThicknessHelper::FromLengths(0, 8, 0, 4));
        decode_label.ColumnDefinitions().Append(star_column());
        decode_label.ColumnDefinitions().Append(auto_column());
        TextBlock dlabel = make_label(L"DECODE  tok/s");
        dlabel.Foreground(brush(kAccent));
        decode_label.Children().Append(dlabel);
        m_decodeMax = make_text(L"--", 16, kMuted, false, true);
        m_decodeMax.HorizontalAlignment(HorizontalAlignment::Right);
        decode_label.Children().Append(m_decodeMax);
        Grid::SetColumn(m_decodeMax, 1);
        inner.Children().Append(decode_label);

        m_decodeCanvas = Canvas();
        m_decodeCanvas.Width(kGraphWidth);
        m_decodeCanvas.Height(kDecodeHeight);
        m_decodeCanvas.IsHitTestVisible(false);
        Grid::SetRow(m_decodeCanvas, 2);
        inner.Children().Append(m_decodeCanvas);

        Grid prefill_label;
        Grid::SetRow(prefill_label, 3);
        prefill_label.Margin(ThicknessHelper::FromLengths(0, 8, 0, 4));
        prefill_label.ColumnDefinitions().Append(star_column());
        prefill_label.ColumnDefinitions().Append(auto_column());
        TextBlock plabel = make_label(L"PREFILL  tok/s");
        plabel.Foreground(brush(kAmber));
        prefill_label.Children().Append(plabel);
        m_prefillMax = make_text(L"--", 16, kMuted, false, true);
        m_prefillMax.HorizontalAlignment(HorizontalAlignment::Right);
        prefill_label.Children().Append(m_prefillMax);
        Grid::SetColumn(m_prefillMax, 1);
        inner.Children().Append(prefill_label);

        m_prefillCanvas = Canvas();
        m_prefillCanvas.Width(kGraphWidth);
        m_prefillCanvas.Height(kPrefillHeight);
        m_prefillCanvas.IsHitTestVisible(false);
        Grid::SetRow(m_prefillCanvas, 4);
        inner.Children().Append(m_prefillCanvas);

        Grid axis;
        Grid::SetRow(axis, 5);
        axis.Margin(ThicknessHelper::FromLengths(0, 6, 0, 0));
        axis.ColumnDefinitions().Append(star_column());
        axis.ColumnDefinitions().Append(auto_column());
        TextBlock ago = make_text(L"5 min ago", 14, kMuted);
        axis.Children().Append(ago);
        TextBlock now = make_text(L"now", 14, kMuted);
        now.HorizontalAlignment(HorizontalAlignment::Right);
        axis.Children().Append(now);
        Grid::SetColumn(now, 1);
        inner.Children().Append(axis);

        m_graphEmpty =
            make_text(L"No samples yet -- run a turn or configure host metrics.", 20, kMuted);
        m_graphEmpty.HorizontalAlignment(HorizontalAlignment::Center);
        m_graphEmpty.VerticalAlignment(VerticalAlignment::Center);
        m_graphEmpty.IsHitTestVisible(false);
        m_graphEmpty.TextWrapping(TextWrapping::Wrap);
        m_graphEmpty.TextAlignment(TextAlignment::Center);
        m_graphEmpty.MaxWidth(760);
        Grid::SetRow(m_graphEmpty, 1);
        Grid::SetRowSpan(m_graphEmpty, 5);
        inner.Children().Append(m_graphEmpty);

        card.Child(inner);
        left.Children().Append(card);
    }

    // ---- memory card ------------------------------------------------------
    {
        Border card = make_card(24);
        Grid::SetRow(card, 0);
        StackPanel stack;
        Grid title;
        title.ColumnDefinitions().Append(star_column());
        title.ColumnDefinitions().Append(auto_column());
        TextBlock label = make_label(L"APP MEMORY");
        title.Children().Append(label);
        m_memoryValue = make_text(L"--", 26, kText, true, true);
        m_memoryValue.HorizontalAlignment(HorizontalAlignment::Right);
        title.Children().Append(m_memoryValue);
        Grid::SetColumn(m_memoryValue, 1);
        stack.Children().Append(title);

        m_memoryTrack = Grid();
        m_memoryTrack.Height(14);
        m_memoryTrack.Margin(ThicknessHelper::FromLengths(0, 18, 0, 0));
        Border track = Border();
        track.Background(brush(kCardAlt));
        track.CornerRadius(CornerRadius{7, 7, 7, 7});
        m_memoryTrack.Children().Append(track);
        m_memoryFill = Border();
        m_memoryFill.Background(brush(kAccent));
        m_memoryFill.CornerRadius(CornerRadius{7, 7, 7, 7});
        m_memoryFill.HorizontalAlignment(HorizontalAlignment::Left);
        m_memoryFill.Width(0);
        m_memoryTrack.Children().Append(m_memoryFill);
        m_memoryTrack.SizeChanged(
            [this](IInspectable const&, SizeChangedEventArgs const&) { ApplyBars(); });
        stack.Children().Append(m_memoryTrack);

        m_memoryDetail = make_text(L"UWP app memory", 17, kMuted);
        m_memoryDetail.Margin(ThicknessHelper::FromLengths(0, 14, 0, 0));
        stack.Children().Append(m_memoryDetail);

        card.Child(stack);
        right.Children().Append(card);
    }

    // ---- context card -----------------------------------------------------
    {
        Border card = make_card(24);
        Grid::SetRow(card, 2);
        StackPanel stack;
        Grid title;
        title.ColumnDefinitions().Append(star_column());
        title.ColumnDefinitions().Append(auto_column());
        TextBlock label = make_label(L"CONTEXT");
        title.Children().Append(label);
        m_contextValue = make_text(L"--", 26, kText, true, true);
        m_contextValue.HorizontalAlignment(HorizontalAlignment::Right);
        title.Children().Append(m_contextValue);
        Grid::SetColumn(m_contextValue, 1);
        stack.Children().Append(title);

        m_contextTrack = Grid();
        m_contextTrack.Height(14);
        m_contextTrack.Margin(ThicknessHelper::FromLengths(0, 18, 0, 0));
        Border track = Border();
        track.Background(brush(kCardAlt));
        track.CornerRadius(CornerRadius{7, 7, 7, 7});
        m_contextTrack.Children().Append(track);
        m_contextFill = Border();
        m_contextFill.Background(brush(kAccent));
        m_contextFill.CornerRadius(CornerRadius{7, 7, 7, 7});
        m_contextFill.HorizontalAlignment(HorizontalAlignment::Left);
        m_contextFill.Width(0);
        m_contextTrack.Children().Append(m_contextFill);
        m_contextTrack.SizeChanged(
            [this](IInspectable const&, SizeChangedEventArgs const&) { ApplyBars(); });
        stack.Children().Append(m_contextTrack);

        m_contextDetail = make_text(L"no resident session", 17, kMuted);
        m_contextDetail.Margin(ThicknessHelper::FromLengths(0, 14, 0, 0));
        m_contextDetail.TextWrapping(TextWrapping::Wrap);
        stack.Children().Append(m_contextDetail);

        card.Child(stack);
        right.Children().Append(card);
    }

    // ---- network / activity card -----------------------------------------
    {
        Border card = make_card(24);
        Grid::SetRow(card, 4);
        Grid inner;
        inner.RowDefinitions().Append(auto_row());
        for (int i = 0; i < 5; ++i)
            inner.RowDefinitions().Append(fixed_row(40));
        inner.RowDefinitions().Append(star_row());

        TextBlock label = make_label(L"NETWORK / ACTIVITY");
        inner.Children().Append(label);

        const auto add_row = [&](int row, const wchar_t* name, TextBlock& value, TextBlock* dot) {
            Grid line;
            Grid::SetRow(line, row);
            line.ColumnDefinitions().Append(fixed_column(190));
            line.ColumnDefinitions().Append(star_column());
            TextBlock name_text = make_text(name, 17, kMuted);
            name_text.VerticalAlignment(VerticalAlignment::Center);
            line.Children().Append(name_text);
            StackPanel value_row;
            value_row.Orientation(Orientation::Horizontal);
            value_row.VerticalAlignment(VerticalAlignment::Center);
            if (dot) {
                *dot = make_text(L"\u25cf", 16, kMuted);
                dot->Margin(ThicknessHelper::FromLengths(0, 0, 8, 0));
                dot->VerticalAlignment(VerticalAlignment::Center);
                value_row.Children().Append(*dot);
            }
            value = make_text(L"--", 20, kText, true);
            value.VerticalAlignment(VerticalAlignment::Center);
            value_row.Children().Append(value);
            line.Children().Append(value_row);
            Grid::SetColumn(value_row, 1);
            inner.Children().Append(line);
        };

        add_row(1, L"LAN API", m_lanValue, nullptr);
        add_row(2, L"RKVA ATTENTION", m_rkvaValue, &m_rkvaDot);
        add_row(3, L"HOST METRICS", m_hostValue, nullptr);
        add_row(4, L"APP UPTIME", m_uptimeValue, nullptr);
        add_row(5, L"MODE", m_modeValue, nullptr);

        m_note =
            make_text(L"RPC/attention is link activity, never converted to tokens/s.", 14, kMuted);
        m_note.TextWrapping(TextWrapping::Wrap);
        m_note.VerticalAlignment(VerticalAlignment::Bottom);
        Grid::SetRow(m_note, 6);
        inner.Children().Append(m_note);

        card.Child(inner);
        right.Children().Append(card);
    }

    // ---- footer -----------------------------------------------------------
    Grid footer;
    Grid::SetRow(footer, 4);
    footer.ColumnDefinitions().Append(star_column());
    footer.ColumnDefinitions().Append(auto_column());
    TextBlock hint = make_text(L"A: open chat   \u00b7   B: cancel the active job", 16, kMuted);
    hint.VerticalAlignment(VerticalAlignment::Center);
    footer.Children().Append(hint);
    TextBlock refresh =
        make_text(L"live telemetry \u00b7 1 Hz \u00b7 no prompt content", 14, kMuted);
    refresh.HorizontalAlignment(HorizontalAlignment::Right);
    refresh.VerticalAlignment(VerticalAlignment::Center);
    footer.Children().Append(refresh);
    Grid::SetColumn(refresh, 1);
    design.Children().Append(footer);

    UpdateClock();
}

// ---------------------------------------------------------------------------
// Tick -- all work here is O(window) and bounded (300 samples, two small
// canvases). The sampler thread does the I/O; this runs once per second.
// ---------------------------------------------------------------------------
void Dashboard::Tick(const DashboardInputs& inputs) {
    UpdateClock();

    DashboardHostSnapshot host;
    DashboardRkvaSnapshot rkva;
    if (m_telemetry) {
        host = m_telemetry->Host();
        rkva = m_telemetry->Rkva();
    }
    UpdateStatus(inputs, host, rkva);
    const bool mode_changed = EnsureDisplayMode(host, rkva);

    const auto now = std::chrono::steady_clock::now();
    if (m_have_last_tick && now - m_last_tick < std::chrono::milliseconds(500) && !mode_changed) {
        UpdateBars(inputs, host, rkva);
        UpdateNetwork(inputs, host, rkva);
        return;
    }
    m_last_tick = now;
    m_have_last_tick = true;

    UpdateThroughput(inputs, host, rkva);
    UpdateBars(inputs, host, rkva);
    UpdateNetwork(inputs, host, rkva);
}

void Dashboard::UpdateClock() {
    SYSTEMTIME time{};
    GetLocalTime(&time);
    wchar_t clock[32];
    swprintf_s(clock, L"%02u:%02u:%02u", time.wHour, time.wMinute, time.wSecond);
    m_clockText.Text(clock);

    static const wchar_t* const kDays[] = {L"SUN", L"MON", L"TUE", L"WED", L"THU", L"FRI", L"SAT"};
    static const wchar_t* const kMonths[] = {L"JAN", L"FEB", L"MAR", L"APR", L"MAY", L"JUN",
                                             L"JUL", L"AUG", L"SEP", L"OCT", L"NOV", L"DEC"};
    wchar_t date[64];
    swprintf_s(date, L"%s  %02u  %s", kDays[time.wDayOfWeek % 7], time.wDay,
               kMonths[(time.wMonth + 11) % 12]);
    m_dateText.Text(date);
}

bool Dashboard::EnsureDisplayMode(const DashboardHostSnapshot& host,
                                  const DashboardRkvaSnapshot& rkva) {
    const bool local_mode = rkva.absent;
    const bool unknown = !rkva.absent && !rkva.known;
    const bool host_ok = host.ok && host.has_decode;
    DisplayMode mode = DisplayMode::Unknown;
    if (local_mode)
        mode = DisplayMode::Local;
    else if (unknown)
        mode = DisplayMode::Unknown;
    else if (rkva.connected && !rkva.error && host_ok)
        mode = DisplayMode::SplitHost;
    else
        mode = DisplayMode::SplitUnavailable;
    if (mode == m_display_mode)
        return false;
    // A source/link transition must invalidate the presentation immediately,
    // even when the 500 ms sample/render throttle would otherwise skip the
    // full throughput update. A prefill measured for a local turn is kept only
    // when the new mode is local.
    const double pending = m_pending_prefill;
    m_display_mode = mode;
    ResetHistory();
    if (mode == DisplayMode::Local)
        m_pending_prefill = pending;
    return true;
}

void Dashboard::UpdateStatus(const DashboardInputs& inputs, const DashboardHostSnapshot& host,
                             const DashboardRkvaSnapshot& rkva) {
    const bool local_mode = rkva.absent;
    const bool unknown = !rkva.absent && !rkva.known;
    const bool has_log = !local_mode;
    const bool listener_failure =
        has_log && rkva.known && !rkva.listening && !rkva.connected && !rkva.error;
    const bool host_busy = host.ok && host.has_requests && host.requests_processing > 0;
    const bool error = inputs.error || rkva.error;

    if (!local_mode) {
        m_tagText.Text(L"XBOX  \u00b7  RKVA ATTENTION CO-PROCESSOR");
        if (host.has_model)
            m_modelText.Text(L"HOST  \u00b7  " + host.model_label);
        else if (rkva.connected)
            m_modelText.Text(L"Host inference  \u00b7  model not exposed");
        else
            m_modelText.Text(unknown ? L"Host inference  \u00b7  link state unknown"
                                     : L"Host inference  \u00b7  waiting for host");
    } else {
        m_tagText.Text(L"XBOX  \u00b7  LOCAL INFERENCE SERVER");
        m_modelText.Text(inputs.model.empty() ? hstring(L"No model selected")
                                              : hstring(inputs.model));
    }
    m_statusShort.Text(
        unknown ? L"RKVA link state unknown"
                : (rkva.error ? L"Attention engine error"
                              : (rkva.connected
                                     ? L"Host inference active"
                                     : (has_log ? (rkva.listening ? L"Waiting for host"
                                                                  : L"RKVA listener unavailable")
                                                : inputs.status_short))));

    std::wstring pill = L"OFFLINE";
    Color color = kAmber;
    if (unknown) {
        pill = L"LINK UNKNOWN";
        color = kAmber;
    } else if (error) {
        pill = L"ERROR";
        color = kRed;
    } else if (rkva.connected) {
        pill = host_busy ? L"ONLINE  \u00b7  BUSY" : L"ONLINE  \u00b7  HOST";
        color = kAccent;
    } else if (has_log) {
        pill = listener_failure ? L"LINK UNAVAILABLE" : L"ONLINE  \u00b7  WAIT";
        color = kAmber;
    } else if (!inputs.model_ready) {
        pill = L"OFFLINE";
        color = kAmber;
    } else if (inputs.running) {
        pill = L"ONLINE  \u00b7  BUSY";
        color = kAccent;
    } else {
        pill = L"ONLINE";
        color = kGreen;
    }
    m_statusPillText.Text(pill);
    m_statusPillText.Foreground(brush(color));
    m_statusPill.Background(brush(ColorHelper::FromArgb(34, color.R, color.G, color.B)));
    m_statusPill.BorderBrush(brush(ColorHelper::FromArgb(120, color.R, color.G, color.B)));
    m_statusPill.BorderThickness(ThicknessHelper::FromUniformLength(1));
}

void Dashboard::UpdateThroughput(const DashboardInputs& inputs, const DashboardHostSnapshot& host,
                                 const DashboardRkvaSnapshot& rkva) {
    const bool local_mode = rkva.absent;
    const bool unknown = !rkva.absent && !rkva.known;
    const bool host_available = host.ok && host.has_decode;
    const bool host_split = rkva.known && rkva.connected && !rkva.error && host_available;
    const bool host_live = host_available && (host.has_requests ? host.requests_processing > 0
                                                                : host.decode_tps > 0.001);
    const bool local_live = local_mode && inputs.local_live;

    enum class Source { None, Host, Local };
    Source source = Source::None;
    bool live = false;
    if (host_split) {
        source = Source::Host;
        live = host_live;
    } else if (local_mode && local_live) {
        source = Source::Local;
        live = true;
    } else if (local_mode && inputs.running) {
        source = Source::Local; // prefill in progress: no decode rate yet
    } else if (local_mode && inputs.has_local_decode) {
        source = Source::Local;
    }

    double decode = -1.0;
    double prefill = -1.0;
    double prefill_event = -1.0;
    if (source == Source::Host) {
        decode = host.decode_tps;
        prefill = host.has_prefill ? host.prefill_tps : -1.0;
        m_tpsSource.Text(L"HOST");
        m_tpsState.Text(live ? L"LIVE" : L"LAST SAMPLE");
        m_tpsValue.Text(rate_text(decode));
        m_tpsValue.Foreground(brush(kAccent));
        m_tpsUnit.Opacity(1.0);
        m_prefillValue.Text(host.has_prefill ? L"PREFILL  " + rate_text(prefill) + L" tok/s"
                                             : L"PREFILL  --");
        std::wstring detail = L"host llama-server \u00b7 end-to-end";
        if (host.has_requests && host.requests_processing > 0)
            detail += L" \u00b7 " + std::to_wstring(host.requests_processing) + L" req";
        else if (!live)
            detail += L" \u00b7 idle";
        if (!host.endpoint.empty())
            detail += L" \u00b7 " + host.endpoint;
        m_turnDetail.Text(detail);
        m_graphSource.Text(L"source: host llama-server");
        if (host.has_prefill && (m_last_host_prefill < 0.0 ||
                                 std::fabs(host.prefill_tps - m_last_host_prefill) > 0.05)) {
            prefill_event = host.prefill_tps;
            m_last_host_prefill = host.prefill_tps;
        }
    } else if (source == Source::Local) {
        decode = inputs.local_decode_tps;
        if (inputs.running && !inputs.local_live)
            decode = 0.0; // prefill: no decode rate exists yet
        prefill = inputs.has_local_prefill ? inputs.local_prefill_tps : -1.0;
        m_tpsSource.Text(L"XBOX \u00b7 LOCAL");
        m_tpsState.Text(live ? L"LIVE"
                             : (inputs.running
                                    ? L"PREFILLING"
                                    : (inputs.local_turn_fresh ? L"JUST FINISHED" : L"LAST TURN")));
        m_tpsValue.Text(rate_text(decode));
        m_tpsValue.Foreground(brush(kAccent));
        m_tpsUnit.Opacity(1.0);
        m_prefillValue.Text(!inputs.running && inputs.has_local_prefill
                                ? L"PREFILL  " + rate_text(prefill) + L" tok/s"
                                : L"PREFILL  --");
        std::wstring detail = L"measured on this console";
        if (inputs.ctx_used > 0)
            detail += L" \u00b7 " + group_digits(inputs.ctx_used) + L" prompt tok";
        if (inputs.peak_ws_mb > 0)
            detail += L" \u00b7 peak " + mb_text(inputs.peak_ws_mb);
        m_turnDetail.Text(detail);
        m_graphSource.Text(L"source: Xbox local");
        m_last_host_prefill = -1.0;
        prefill_event = m_pending_prefill;
    } else {
        decode = -1.0;
        prefill = -1.0;
        m_tpsSource.Text(L"NO DATA");
        m_tpsValue.Text(L"--");
        m_tpsValue.Foreground(brush(kMuted));
        m_tpsUnit.Opacity(0.55);
        if (local_mode) {
            m_tpsState.Text(L"WAITING FOR A TURN");
            m_turnDetail.Text(L"run a turn or configure host metrics");
            m_graphSource.Text(L"source: none");
        } else if (unknown) {
            m_tpsState.Text(L"LINK STATE UNKNOWN");
            m_turnDetail.Text(L"RKVA log state unknown \u2014 no local fallback");
            m_graphSource.Text(L"source: none (link unknown)");
        } else if (rkva.error) {
            m_tpsState.Text(L"ATTENTION ENGINE ERROR");
            m_turnDetail.Text(L"RKVA engine error \u2014 no tokens/s");
            m_graphSource.Text(L"source: none (engine error)");
        } else if (!rkva.connected) {
            m_tpsState.Text(rkva.listening ? L"WAITING FOR HOST" : L"LISTENER UNAVAILABLE");
            m_turnDetail.Text(
                rkva.listening
                    ? L"RKVA listener active \u2014 host not connected \u2014 no tokens/s"
                    : L"RKVA listener unavailable \u2014 no tokens/s");
            m_graphSource.Text(rkva.listening ? L"source: none (waiting for host)"
                                              : L"source: none (listener unavailable)");
        } else {
            m_tpsState.Text(L"HOST METRICS UNAVAILABLE");
            m_turnDetail.Text(host.configured
                                  ? (host.ok
                                         ? L"host reachable, tokens/s metric missing"
                                         : L"host endpoint unreachable \u2014 check dashboard.json")
                                  : L"host metrics not configured \u2014 see dashboard.json");
            m_graphSource.Text(L"source: none (host unavailable)");
        }
        m_prefillValue.Text(L"PREFILL  --");
        m_last_host_prefill = -1.0;
    }
    if (local_mode)
        m_pending_prefill = -1.0;

    double decode_sample = -1.0;
    if (source == Source::Host)
        decode_sample = decode;
    else if (source == Source::Local)
        decode_sample = (inputs.local_live || inputs.local_turn_fresh) ? decode : 0.0;

    PushSample(decode_sample, prefill_event);
    RedrawDecode();
    RedrawPrefill();

    bool any = false;
    const size_t start = (m_history_next + kWindowSeconds - m_history_count) % kWindowSeconds;
    for (size_t i = 0; i < m_history_count; ++i) {
        const Sample& sample = m_history[(start + i) % kWindowSeconds];
        if (sample.decode >= 0.0 || sample.prefill >= 0.0) {
            any = true;
            break;
        }
    }
    m_graphEmpty.Visibility(any ? Visibility::Collapsed : Visibility::Visible);
    if (!any) {
        if (unknown)
            m_graphEmpty.Text(L"RKVA link state unknown \u2014 tokens/s unavailable.");
        else if (!local_mode && !host_split) {
            if (rkva.error)
                m_graphEmpty.Text(L"RKVA engine error \u2014 tokens/s unavailable.");
            else if (!rkva.connected)
                m_graphEmpty.Text(rkva.listening
                                      ? L"Waiting for host \u2014 tokens/s unavailable."
                                      : L"RKVA listener unavailable \u2014 tokens/s unavailable.");
            else
                m_graphEmpty.Text(L"Host metrics unavailable \u2014 tokens/s needs the host "
                                  L"llama-server metrics endpoint.");
        } else
            m_graphEmpty.Text(L"No samples yet \u2014 run a turn or configure host metrics.");
    }
}

void Dashboard::UpdateBars(const DashboardInputs& inputs, const DashboardHostSnapshot& host,
                           const DashboardRkvaSnapshot& rkva) {
    // Memory: the real UWP app budget, as the OS reports it.
    uint64_t used = 0;
    uint64_t limit = 0;
    bool memory_ok = false;
    try {
        used = winrt::Windows::System::MemoryManager::AppMemoryUsage();
        limit = winrt::Windows::System::MemoryManager::AppMemoryUsageLimit();
        memory_ok = limit > 0;
    } catch (...) {
        memory_ok = false;
    }
    if (memory_ok) {
        m_memory_ratio = clamp01(static_cast<double>(used) / static_cast<double>(limit));
        m_memoryValue.Text(gb_text(used) + L" / " + gb_text(limit));
        std::wstring detail = percent_text(m_memory_ratio) + L" of the UWP app budget";
        const bool local_mode = rkva.absent;
        if (inputs.peak_ws_mb > 0 && local_mode)
            detail += L" \u00b7 peak " + mb_text(inputs.peak_ws_mb);
        m_memoryDetail.Text(detail);
        Color fill = kAccent;
        if (m_memory_ratio >= 0.90)
            fill = kRed;
        else if (m_memory_ratio >= 0.70)
            fill = kAmber;
        m_memoryFill.Background(brush(fill));
    } else {
        m_memory_ratio = 0.0;
        m_memoryValue.Text(L"--");
        m_memoryDetail.Text(L"memory API unavailable");
    }

    // Context: prefer the host's own gauge while the host drives inference.
    // Otherwise show the exact n_ctx and the exact prompt size of the last
    // turn. Never a chars/token estimate, and never a stale host value.
    const bool local_mode = rkva.absent;
    const bool unknown = !rkva.absent && !rkva.known;
    const bool host_drives = rkva.known && rkva.connected && !rkva.error && host.ok;
    m_contextTrack.Visibility(Visibility::Visible);
    if (host_drives && host.has_kv_ratio) {
        m_context_ratio = clamp01(host.kv_ratio);
        std::wstring value = percent_text(m_context_ratio);
        if (host.has_kv_tokens)
            value = group_digits(host.kv_tokens) + L" tok";
        m_contextValue.Text(value);
        m_contextDetail.Text(host.has_kv_tokens
                                 ? percent_text(m_context_ratio) + L" of host KV cache \u00b7 " +
                                       group_digits(host.kv_tokens) + L" tokens"
                                 : L"host llama-server KV cache");
        m_contextFill.Background(brush(m_context_ratio >= 0.90   ? kRed
                                       : m_context_ratio >= 0.70 ? kAmber
                                                                 : kAccent));
    } else if (host_drives && host.has_context_capacity) {
        // /props gives the host n_ctx; /metrics gives the largest observed
        // sequence. The bar is therefore peak-usage / capacity, labelled as a
        // peak, not current KV occupancy.
        const long long capacity = host.context_capacity;
        if (host.has_context_peak) {
            m_context_ratio = clamp01(static_cast<double>(host.context_peak_tokens) / capacity);
            m_contextValue.Text(L"peak " + group_digits(host.context_peak_tokens) + L" / " +
                                group_digits(capacity) + L" tok");
            m_contextDetail.Text(L"host context high-watermark / n_ctx");
            m_contextFill.Background(brush(m_context_ratio >= 0.90   ? kRed
                                           : m_context_ratio >= 0.70 ? kAmber
                                                                     : kAccent));
        } else {
            m_context_ratio = 0.0;
            m_contextValue.Text(group_digits(capacity) + L" tok");
            m_contextDetail.Text(L"host context capacity (n_ctx) \u00b7 no sequence yet");
            m_contextTrack.Visibility(Visibility::Collapsed);
        }
    } else if (host_drives && host.has_context_peak) {
        // High-watermark without a capacity denominator: show the number,
        // never a made-up bar.
        m_context_ratio = 0.0;
        m_contextValue.Text(L"peak " + group_digits(host.context_peak_tokens) + L" tok");
        m_contextDetail.Text(L"host context high-watermark \u00b7 capacity not exposed");
        m_contextTrack.Visibility(Visibility::Collapsed);
    } else if (!local_mode) {
        m_context_ratio = 0.0;
        m_contextValue.Text(L"--");
        m_contextDetail.Text(
            unknown
                ? L"RKVA link state unknown \u00b7 no host context"
                : (rkva.error ? L"attention engine error \u00b7 no host context"
                              : (rkva.connected
                                     ? L"host context metrics unavailable"
                                     : (rkva.listening
                                            ? L"waiting for host \u00b7 no host context"
                                            : L"listener unavailable \u00b7 no host context"))));
        m_contextTrack.Visibility(Visibility::Collapsed);
    } else if (inputs.ctx_capacity > 0) {
        const long long capacity = inputs.ctx_capacity;
        const long long used_tokens = std::max<long long>(0, inputs.ctx_used);
        m_context_ratio = clamp01(static_cast<double>(used_tokens) / capacity);
        m_contextValue.Text(used_tokens > 0 ? group_digits(used_tokens) + L" prompt tok"
                                            : L"-- / " + group_digits(capacity) + L" tok");
        m_contextDetail.Text(used_tokens > 0
                                 ? L"of " + group_digits(capacity) +
                                       L" context \u00b7 last turn prompt (this console)"
                                 : L"capacity of the resident session \u00b7 no turn yet");
        m_contextFill.Background(brush(m_context_ratio >= 0.90   ? kRed
                                       : m_context_ratio >= 0.70 ? kAmber
                                                                 : kAccent));
    } else {
        m_context_ratio = 0.0;
        m_contextValue.Text(L"--");
        m_contextDetail.Text(L"no resident session");
        m_contextTrack.Visibility(Visibility::Collapsed);
    }
    ApplyBars();
}

void Dashboard::UpdateNetwork(const DashboardInputs& inputs, const DashboardHostSnapshot& host,
                              const DashboardRkvaSnapshot& rkva) {
    const bool local_mode = rkva.absent;
    const bool unknown = !rkva.absent && !rkva.known;
    const bool has_log = !local_mode;
    switch (inputs.lan) {
    case DashboardLink::On:
        m_lanValue.Text(L"ONLINE" +
                        (inputs.lan_detail.empty() ? L"" : L" \u00b7 " + inputs.lan_detail));
        m_lanValue.Foreground(brush(kGreen));
        break;
    case DashboardLink::Pending:
        m_lanValue.Text(inputs.lan_detail.empty() ? L"STARTING" : inputs.lan_detail);
        m_lanValue.Foreground(brush(kAmber));
        break;
    case DashboardLink::Error:
        m_lanValue.Text(inputs.lan_detail.empty() ? L"ERROR"
                                                  : L"ERROR \u00b7 " + inputs.lan_detail);
        m_lanValue.Foreground(brush(kRed));
        break;
    default:
        m_lanValue.Text(inputs.lan_detail.empty() ? L"OFF" : L"OFF \u00b7 " + inputs.lan_detail);
        m_lanValue.Foreground(brush(kMuted));
        break;
    }

    if (rkva.error) {
        m_rkvaDot.Foreground(brush(kRed));
        m_rkvaValue.Text(L"ENGINE ERROR");
        m_rkvaValue.Foreground(brush(kRed));
    } else if (rkva.connected) {
        m_rkvaDot.Foreground(brush(kGreen));
        m_rkvaValue.Text(L"HOST CONNECTED");
        m_rkvaValue.Foreground(brush(kGreen));
    } else if (rkva.listening) {
        m_rkvaDot.Foreground(brush(kAmber));
        m_rkvaValue.Text(L"LISTENING \u00b7 :50054");
        m_rkvaValue.Foreground(brush(kAmber));
    } else if (unknown) {
        m_rkvaDot.Foreground(brush(kAmber));
        m_rkvaValue.Text(L"STATE UNKNOWN");
        m_rkvaValue.Foreground(brush(kAmber));
    } else if (rkva.known) {
        m_rkvaDot.Foreground(brush(kAmber));
        m_rkvaValue.Text(rkva.status.empty() ? hstring(L"LISTENER UNAVAILABLE")
                                             : hstring(rkva.status));
        m_rkvaValue.Foreground(brush(kAmber));
    } else {
        m_rkvaDot.Foreground(brush(kMuted));
        m_rkvaValue.Text(L"NOT STARTED");
        m_rkvaValue.Foreground(brush(kMuted));
    }

    if (host.ok) {
        m_hostValue.Text(host.endpoint.empty() ? L"ONLINE" : L"ONLINE \u00b7 " + host.endpoint);
        m_hostValue.Foreground(brush(kGreen));
    } else if (host.configured) {
        m_hostValue.Text(L"UNAVAILABLE \u00b7 " + host.status);
        m_hostValue.Foreground(brush(kAmber));
    } else {
        m_hostValue.Text(L"NOT CONFIGURED");
        m_hostValue.Foreground(brush(kMuted));
    }

    m_uptimeValue.Text(uptime_text(inputs.uptime_seconds));
    m_uptimeValue.Foreground(brush(kText));

    if (unknown) {
        m_modeValue.Text(L"SPLIT \u00b7 link unknown");
        m_modeValue.Foreground(brush(kAmber));
    } else if (rkva.error) {
        m_modeValue.Text(L"SPLIT \u00b7 engine error");
        m_modeValue.Foreground(brush(kRed));
    } else if (rkva.connected) {
        m_modeValue.Text(L"SPLIT \u00b7 host inference");
        m_modeValue.Foreground(brush(kAccent));
    } else if (has_log) {
        if (rkva.listening) {
            m_modeValue.Text(L"SPLIT \u00b7 waiting for host");
            m_modeValue.Foreground(brush(kAmber));
        } else {
            m_modeValue.Text(L"SPLIT \u00b7 listener unavailable");
            m_modeValue.Foreground(brush(kAmber));
        }
    } else if (local_mode && (inputs.model_ready || inputs.running)) {
        m_modeValue.Text(L"LOCAL \u00b7 Xbox inference");
        m_modeValue.Foreground(brush(kText));
    } else {
        m_modeValue.Text(L"IDLE");
        m_modeValue.Foreground(brush(kMuted));
    }
}

void Dashboard::PushSample(double decode, double prefill) {
    m_history[m_history_next] = Sample{decode, prefill};
    m_history_next = (m_history_next + 1) % kWindowSeconds;
    if (m_history_count < kWindowSeconds)
        ++m_history_count;
}

void Dashboard::ResetHistory() {
    // A source/mode transition invalidates every sample from the previous
    // source; local history can never be drawn under split (or vice versa).
    for (auto& sample : m_history)
        sample = Sample{};
    m_history_next = 0;
    m_history_count = 0;
    m_pending_prefill = -1.0;
    m_last_host_prefill = -1.0;
    if (m_decodeMax)
        m_decodeMax.Text(L"--");
    if (m_prefillMax)
        m_prefillMax.Text(L"--");
    RedrawDecode();
    RedrawPrefill();
}

void Dashboard::RedrawDecode() {
    m_decodeCanvas.Children().Clear();
    const double width = m_decodeCanvas.Width();
    const double height = m_decodeCanvas.Height();
    for (int i = 0; i <= 4; ++i) {
        Shapes::Line line;
        line.X1(0);
        line.X2(width);
        line.Y1(height * i / 4.0);
        line.Y2(height * i / 4.0);
        line.Stroke(brush(i == 4 ? kBorder : kGrid));
        line.StrokeThickness(1);
        m_decodeCanvas.Children().Append(line);
    }
    if (m_history_count == 0) {
        m_decodeMax.Text(L"--");
        return;
    }

    double max_value = 0.0;
    const size_t start = (m_history_next + kWindowSeconds - m_history_count) % kWindowSeconds;
    for (size_t i = 0; i < m_history_count; ++i) {
        const double value = m_history[(start + i) % kWindowSeconds].decode;
        if (value >= 0.0)
            max_value = std::max(max_value, value);
    }
    if (max_value <= 0.0) {
        m_decodeMax.Text(L"--");
        return;
    }
    const double scale = nice_ceiling(max_value);
    m_decodeMax.Text(L"max " + rate_text(max_value) + L" tok/s");

    const double step = width / static_cast<double>(kWindowSeconds - 1);
    PointCollection points;
    auto flush = [&]() {
        if (points.Size() >= 2) {
            Shapes::Polyline line;
            line.Points(points);
            line.Stroke(brush(kAccent));
            line.StrokeThickness(2.5);
            line.StrokeLineJoin(PenLineJoin::Round);
            line.IsHitTestVisible(false);
            m_decodeCanvas.Children().Append(line);
        } else if (points.Size() == 1) {
            const Point point = points.GetAt(0);
            Shapes::Ellipse dot;
            dot.Width(4);
            dot.Height(4);
            dot.Fill(brush(kAccent));
            Canvas::SetLeft(dot, point.X - 2);
            Canvas::SetTop(dot, point.Y - 2);
            m_decodeCanvas.Children().Append(dot);
        }
        points = PointCollection();
    };

    for (size_t i = 0; i < m_history_count; ++i) {
        const Sample& sample = m_history[(start + i) % kWindowSeconds];
        if (sample.decode < 0.0) {
            flush();
            continue;
        }
        const double x = static_cast<double>(kWindowSeconds - m_history_count + i) * step;
        const double y = height - 3.0 - clamp01(sample.decode / scale) * (height - 6.0);
        points.Append(Point{static_cast<float>(x), static_cast<float>(y)});
    }
    flush();
}

void Dashboard::RedrawPrefill() {
    m_prefillCanvas.Children().Clear();
    const double width = m_prefillCanvas.Width();
    const double height = m_prefillCanvas.Height();
    for (int i = 0; i <= 2; ++i) {
        Shapes::Line line;
        line.X1(0);
        line.X2(width);
        line.Y1(height * i / 2.0);
        line.Y2(height * i / 2.0);
        line.Stroke(brush(i == 2 ? kBorder : kGrid));
        line.StrokeThickness(1);
        m_prefillCanvas.Children().Append(line);
    }
    if (m_history_count == 0) {
        m_prefillMax.Text(L"--");
        return;
    }

    double max_value = 0.0;
    const size_t start = (m_history_next + kWindowSeconds - m_history_count) % kWindowSeconds;
    for (size_t i = 0; i < m_history_count; ++i) {
        const double value = m_history[(start + i) % kWindowSeconds].prefill;
        if (value >= 0.0)
            max_value = std::max(max_value, value);
    }
    if (max_value <= 0.0) {
        m_prefillMax.Text(L"--");
        return;
    }
    const double scale = nice_ceiling(max_value);
    m_prefillMax.Text(L"max " + rate_text(max_value) + L" tok/s");

    const double step = width / static_cast<double>(kWindowSeconds - 1);
    const double bar_width = std::max(3.0, step * 0.9);
    for (size_t i = 0; i < m_history_count; ++i) {
        const double value = m_history[(start + i) % kWindowSeconds].prefill;
        if (value < 0.0 || value <= 0.0)
            continue;
        const double bar_height = std::max(2.0, clamp01(value / scale) * (height - 4.0));
        Shapes::Rectangle bar;
        bar.Width(bar_width);
        bar.Height(bar_height);
        bar.Fill(brush(kAmber));
        bar.Opacity(0.9);
        bar.RadiusX(1);
        bar.RadiusY(1);
        bar.IsHitTestVisible(false);
        Canvas::SetLeft(bar, static_cast<double>(kWindowSeconds - m_history_count + i) * step -
                                 bar_width / 2.0);
        Canvas::SetTop(bar, height - bar_height);
        m_prefillCanvas.Children().Append(bar);
    }
}

void Dashboard::ApplyBars() {
    if (m_memoryFill) {
        const double width = m_memoryTrack.ActualWidth();
        if (width > 0.0)
            m_memoryFill.Width(m_memory_ratio * width);
    }
    if (m_contextFill) {
        const double width = m_contextTrack.ActualWidth();
        if (width > 0.0)
            m_contextFill.Width(m_context_ratio * width);
    }
}

} // namespace xllama::ui

#endif // XLLAMA_UWP
