// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#pragma once

#include <cctype>
#include <string>

namespace xllama {

/// Parsed tool-related request fields. The transport parser owns JSON typing;
/// this small policy stays host-testable and has no WinRT dependency.
struct ApiToolFields {
    bool tools = false;
    bool functions = false;
    bool tool_choice = false;
};

/// xllama has no tool executor. Any advertised tool capability is rejected so
/// clients cannot mistake local inference for permission to cause side effects.
inline bool api_tool_execution_requested(const ApiToolFields& fields) {
    return fields.tools || fields.functions || fields.tool_choice;
}

/// Slice the value of the test-only X-Test-Nonce header (T18 fault probe,
/// plan 009). `raw_headers` is the original-case header block of one HTTP
/// request, lines separated by \r\n. The NAME matches case-insensitively at
/// a line start (offsets are stable: lowering is char-wise); the VALUE keeps
/// its case, skips any number of leading spaces/tabs, stops at the line
/// break, and is capped at 64 bytes. The FIRST line-start match wins: an
/// earlier mid-line occurrence (e.g. inside another header's value) is
/// skipped, never honored — but a line-start match with an empty/oversized
/// value still ends the scan with "", so duplicates never fall through to a
/// later header. Returns the value, or "" when no header line matches —
/// "" can never match an armed probe nonce, so such requests simply never
/// fire.
inline std::string parse_test_nonce_header(const std::string& raw_headers) {
    std::string lowered = raw_headers;
    for (char& c : lowered)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    static constexpr char kKey[] = "x-test-nonce:"; // 13 bytes, incl. colon
    static constexpr size_t kKeyLen = sizeof(kKey) - 1;
    size_t from = 0;
    while (from < lowered.size()) {
        const size_t p = lowered.find(kKey, from);
        if (p == std::string::npos)
            return {};
        if (p == 0 || lowered[p - 1] == '\r' || lowered[p - 1] == '\n') {
            size_t i = p + kKeyLen; // exact end of the colon — value starts here
            while (i < raw_headers.size() && (raw_headers[i] == ' ' || raw_headers[i] == '\t'))
                ++i;
            const size_t start = i;
            while (i < raw_headers.size() && raw_headers[i] != '\r' && raw_headers[i] != '\n')
                ++i;
            if (i == start || i - start > 64)
                return {};
            return raw_headers.substr(start, i - start);
        }
        from = p + 1; // shadow inside another token/value: keep scanning
    }
    return {};
}

} // namespace xllama
