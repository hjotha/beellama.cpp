#pragma once

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Zero cadence is the documented sync-only control. Reject malformed values
// instead of silently enabling expensive work (or overflowing atoi).
inline int32_t dflash_shadow_env_integer(const char * name, const char * value,
                                         int32_t fallback, int32_t maximum) {
    if (!value) return fallback;
    char * end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || errno == ERANGE || parsed < 0 || parsed > maximum) {
        throw std::invalid_argument(std::string(name) + " must be an integer between 0 and " +
                std::to_string(maximum));
    }
    return (int32_t) parsed;
}

// Count primary draft attempts before running the primary in both modes. A
// proposal-free attempt may cancel an early job, but cannot shift later samples.
inline bool dflash_shadow_schedule_due(int32_t every, uint64_t & attempt) {
    return every > 0 && attempt++ % (uint64_t) every == 0;
}

// Records one shadow observation. The parent owns cancellation and the id/epoch
// lifecycle; completion, commit, and decision may be delivered in any order.
struct dflash_shadow_observation {
    struct evaluation {
        bool ready = false;
        bool prefix_match = false;
        bool position_match = false;
        bool usable = false;
        int32_t matched = 0;
        int32_t remaining = 0;
        int32_t usable_tokens = 0;
    };

    uint64_t id = 0;
    uint64_t epoch = 0;
    int32_t pos0 = -1;
    int32_t anchor = -1;
    std::vector<int32_t> proposed;
    std::vector<int32_t> confirmed;
    int64_t done_us = 0;
    int64_t decision_us = 0;
    int32_t next_pos = -1;
    int32_t next_anchor = -1;
    int32_t capacity = 0;
    bool completed = false;
    bool committed = false;
    bool decided = false;

    void finish(std::vector<int32_t> tokens, int64_t timestamp) {
        proposed = std::move(tokens);
        done_us = timestamp;
        completed = true;
    }

    void commit(std::vector<int32_t> tokens) {
        confirmed = std::move(tokens);
        committed = true;
    }

    void decide(int32_t position, int32_t token, int32_t max_tokens, int64_t timestamp) {
        if (decided) {
            return;
        }
        next_pos = position;
        next_anchor = token;
        capacity = max_tokens;
        decision_us = timestamp;
        decided = true;
    }

    bool resolved() const {
        return completed && committed && decided;
    }

    evaluation evaluate() const {
        evaluation result;
        result.ready = completed && decided && done_us <= decision_us;

        std::size_t matched = 0;
        while (matched < confirmed.size() && matched < proposed.size() &&
               confirmed[matched] == proposed[matched]) {
            ++matched;
        }
        result.matched = static_cast<int32_t>(matched);
        result.remaining = static_cast<int32_t>(proposed.size() - std::min(proposed.size(), confirmed.size()));
        result.prefix_match = committed && !confirmed.empty() && matched == confirmed.size();
        result.position_match = !confirmed.empty() &&
            int64_t(next_pos) == int64_t(pos0) + int64_t(confirmed.size()) && next_anchor == confirmed.back();
        result.usable = resolved() && result.ready && result.prefix_match && result.position_match &&
            capacity > 0 && result.remaining > 0;
        result.usable_tokens = result.usable ? std::min(capacity, result.remaining) : 0;
        return result;
    }

    // The caller may consume this only at the first decision, then retires the
    // observation. Revalidate identity/anchor rather than exporting a bare tail
    // that could survive a request switch or a later polling cycle.
    std::vector<int32_t> suffix(uint64_t current_epoch, int32_t position, int32_t token,
                                int32_t minimum) const {
        const auto result = evaluate();
        if (!id || epoch != current_epoch || next_pos != position || next_anchor != token ||
                !result.usable || minimum < 1 || result.usable_tokens < minimum) {
            return {};
        }
        const auto begin = proposed.begin() + confirmed.size();
        return {begin, begin + result.usable_tokens};
    }
};
