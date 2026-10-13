// F4 TEST-ONLY full-logit capture hook. See f4-test-logit-capture.h.
//
// Environment (test scripts only; the production server never sets them):
//   F4_TEST_LOGIT_CAPTURE   base path for the capture files (unset -> no-op)
//   F4_TEST_EXPECT_TOKENS   comma-separated token counts that ARE the fixed
//                           checkpoint(s); any other count is not captured, so
//                           a phase that never reaches its checkpoint produces
//                           NO file and can therefore not be compared.

#include "f4-test-logit-capture.h"

#if defined(LLAMA_F4_TEST_LOGIT_CAPTURE)

#include "ggml-f4-counters.h"
#include "hash/xxhash/xxhash.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace {

std::string hex64(uint64_t v) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) v);
    return std::string(buf);
}

std::string esc(const std::string & in) {
    std::string out;
    for (const char c : in) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((unsigned char) c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned) (unsigned char) c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

std::string hash_of(const std::string & s) {
    return hex64(XXH64(s.data(), s.size(), 0));
}

std::set<size_t> expected_counts() {
    std::set<size_t> out;
    const char * raw = std::getenv("F4_TEST_EXPECT_TOKENS");
    if (raw == nullptr || *raw == '\0') {
        return out;
    }
    const std::string s(raw);
    size_t i = 0;
    while (i <= s.size()) {
        const size_t j = s.find(',', i);
        const std::string tok = s.substr(i, j == std::string::npos ? std::string::npos : j - i);
        if (!tok.empty()) {
            out.insert((size_t) std::strtoull(tok.c_str(), nullptr, 10));
        }
        if (j == std::string::npos) {
            break;
        }
        i = j + 1;
    }
    return out;
}

// Per-process attempt counter so a retried decode at the same checkpoint is
// visible instead of silently overwriting the first capture.
int & attempt_counter() {
    static int n = 0;
    return n;
}

} // namespace

int f4_test_logit_capture_checkpoint(const f4_test_capture_binding & b,
                                     const std::vector<int32_t> & tokens,
                                     const float * logits, size_t n_logits) {
    const char * base = std::getenv("F4_TEST_LOGIT_CAPTURE");
    if (base == nullptr || *base == '\0' || logits == nullptr || n_logits == 0) {
        return 0;
    }
    const std::set<size_t> want = expected_counts();
    if (want.find(tokens.size()) == want.end()) {
        return 0; // not the explicitly fixed checkpoint
    }

    const std::string phase = b.phase ? *b.phase : "unknown";
    const std::string stem  = std::string(base) + "." + phase + ".n" + std::to_string(tokens.size());

    // Full logits payload (exactly what the server itself would serialize).
    {
        std::ofstream f(stem + ".logits", std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char *>(logits), (std::streamsize) (n_logits * sizeof(float)));
        if (!f) {
            fprintf(stderr, "[F4CAP] failed to write %s.logits\n", stem.c_str());
            return 0;
        }
    }
    // The evaluated token ids, so the comparator can verify the sequence hash
    // (and the fixed next token) independently of this hook.
    {
        std::ofstream f(stem + ".tokens", std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char *>(tokens.data()),
                (std::streamsize) (tokens.size() * sizeof(int32_t)));
    }

    bool finite = true;
    double mx = -INFINITY, mn = INFINITY, sum_exp = 0.0;
    for (size_t i = 0; i < n_logits; ++i) {
        const double x = (double) logits[i];
        if (!std::isfinite(x)) {
            finite = false;
            continue;
        }
        if (x > mx) { mx = x; }
        if (x < mn) { mn = x; }
    }
    if (std::isfinite(mx)) {
        for (size_t i = 0; i < n_logits; ++i) {
            const double x = (double) logits[i];
            if (std::isfinite(x)) {
                sum_exp += std::exp(x - mx);
            }
        }
    }
    const double lse_vocab = std::isfinite(mx) && sum_exp > 0.0 ? mx + std::log(sum_exp) : NAN;

    const std::string tok_hash = tokens.empty() ? std::string("0")
        : hash_of(std::string(reinterpret_cast<const char *>(tokens.data()),
                              tokens.size() * sizeof(int32_t)));

    const auto st = [](const std::string * p) { return p ? *p : std::string(); };
    std::ofstream j(stem + ".json", std::ios::trunc);
    j << "{\n";
    j << "  \"run_id\": \""             << esc(st(b.run_id))             << "\",\n";
    j << "  \"phase\": \""              << esc(phase)                    << "\",\n";
    j << "  \"build_id\": \""           << esc(st(b.build_id))           << "\",\n";
    j << "  \"model_fingerprint\": \""  << esc(st(b.model_fingerprint))  << "\",\n";
    j << "  \"config_core\": \""        << esc(st(b.config_core))        << "\",\n";
    j << "  \"config_hash\": \""        << hash_of(st(b.config_core))    << "\",\n";
    j << "  \"layout\": \""             << esc(st(b.layout))             << "\",\n";
    j << "  \"layout_hash\": \""        << hash_of(st(b.layout))         << "\",\n";
    j << "  \"remote\": \""             << esc(st(b.remote))             << "\",\n";
    j << "  \"cache_source\": \""       << esc(st(b.cache_source))       << "\",\n";
    j << "  \"cache_reason\": \""       << esc(st(b.cache_reason))       << "\",\n";
    j << "  \"n_params\": "             << b.n_params                    << ",\n";
    j << "  \"n_ctx\": "                << b.n_ctx                       << ",\n";
    j << "  \"n_vocab\": "              << b.n_vocab                     << ",\n";
    j << "  \"tokens_n\": "             << tokens.size()                 << ",\n";
    j << "  \"tokens_seq_hash\": \""    << tok_hash                      << "\",\n";
    j << "  \"input_pos\": "            << (tokens.empty() ? -1 : (int64_t) tokens.size() - 1) << ",\n";
    j << "  \"output_pos\": "           << (tokens.empty() ? -1 : (int64_t) tokens.size() - 1) << ",\n";
    j << "  \"logits_index\": "         << b.logits_index                << ",\n";
    j << "  \"logits_count\": "         << n_logits                      << ",\n";
    j << "  \"pos_max\": "              << b.pos_max                     << ",\n";
    j << "  \"state_bytes\": "          << b.state_bytes                 << ",\n";
    j << "  \"attempt\": "              << ++attempt_counter()           << ",\n";
    j << "  \"finite\": "               << (finite ? "true" : "false")  << ",\n";
    j << "  \"logits_max\": "           << mx                            << ",\n";
    j << "  \"logits_min\": "           << mn                            << ",\n";
    j << "  \"lse_vocab\": "            << lse_vocab                     << "\n";
    j << "}\n";
    j.close();

    // TEST-ONLY: close the actual copy/dispatch measurement window exactly at
    // this checkpoint, so the printed delta covers only the work since the
    // previous capture and excludes model-load initialization.
    ggml_f4_counters_dump_and_reset(stem.c_str());

    fprintf(stderr, "[F4CAP] captured phase=%s tokens=%zu input_pos=%zu output_pos=%zu "
                    "vocab=%zu pos_max=%lld seq_hash=%s layout_hash=%s cache=%s/%s\n",
            phase.c_str(), tokens.size(), tokens.empty() ? size_t(0) : tokens.size() - 1,
            tokens.empty() ? size_t(0) : tokens.size() - 1, n_logits, (long long) b.pos_max, tok_hash.c_str(),
            hash_of(b.layout ? *b.layout : std::string()).c_str(),
            b.cache_source ? b.cache_source->c_str() : "?",
            b.cache_reason ? b.cache_reason->c_str() : "?");
    return 1;
}

#endif // LLAMA_F4_TEST_LOGIT_CAPTURE