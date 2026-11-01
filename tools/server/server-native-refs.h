#pragma once

// Durable host-side reference journal for native KVarN host/Xbox snapshot
// pairs. A host file may be evicted by slot-cache retention while its Xbox
// object still exists; this journal is the authoritative reference that keeps
// the Xbox key known (and pinned) across host-file removal. It never deletes
// remote state.
//
// One JSON object per line. Appends are serialized by the caller's exclusive
// route-state store lock. `pending` is written before the Xbox SAVE and host
// publication, `committed` after host publication (append-only, last state
// wins on load), and `pinned` only by the explicit migration CLI. A crash or
// failure therefore leaves a `pending` record, never a silent drop.
//
// The key is the SHA-256 of the complete host prefix through the end of the
// RKVK frame, exactly as computed by the core native save path. The staged
// host file passed to server_route_state_save's staged callback contains
// nothing after that frame (logits/manifest/SCKP are appended later), so
// hashing [0, staged_bytes) with the same SHA-256 implementation yields the
// key without re-parsing the frame. tests/test-native-save-file.cpp asserts
// equality with llama_native_save::hash_file_prefix.

#include "rkva-snapshot-digest.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace server_native_refs {

inline constexpr const char * kJournalName = ".rkva-native-refs.jsonl";
inline constexpr const char * kSchema      = "rkva-native-refs";
inline constexpr int          kVersion     = 1;

enum class state { pending, committed, pinned };
enum class source { explicit_save, auto_save, migrated };

// The six published keys from the original campaign plus the same-build key
// recorded on 2026-11-01. Migration is explicit (CLI/file), never automatic.
inline constexpr std::array<const char *, 7> kBaselineKeys = {{
    "3c1ac4014f51a0ff550c3e1346463623140e22f8c9a21bba80ff66d6b09fc582",
    "3c7218bdc40af2312e5dba397b7d2a846d3f4beb717f760353cdcc7b6f627ab1",
    "4d5532221ee86514043d1cabb790afb0b09e757ff11b824931b0de3bb0b24fc3",
    "674016a7971750acd47f65809f25c08733712390e717792ec41685224bb2378b",
    "68b109a4d14bbd26d7937cc9f8268a6bdb706584239531c440762857036d82c5",
    "ef1c05ae42b5d79406ba3cc48b15f8107049a5c07e76339b23ad9fd96b972dd7",
    "fff30e532caeac47750f17351664f3b392ffdafc0801514d09395c5326bee09a",
}};

inline const char * state_name(state value) {
    switch (value) {
        case state::pending:   return "pending";
        case state::committed: return "committed";
        case state::pinned:    return "pinned";
    }
    return "unknown";
}

inline bool parse_state(const std::string & text, state & out) {
    if (text == "pending")   { out = state::pending;   return true; }
    if (text == "committed") { out = state::committed; return true; }
    if (text == "pinned")    { out = state::pinned;    return true; }
    return false;
}

inline const char * source_name(source value) {
    switch (value) {
        case source::explicit_save: return "explicit";
        case source::auto_save:     return "auto";
        case source::migrated:      return "migrated";
    }
    return "unknown";
}

inline bool parse_source(const std::string & text, source & out) {
    if (text == "explicit") { out = source::explicit_save; return true; }
    if (text == "auto")     { out = source::auto_save;     return true; }
    if (text == "migrated") { out = source::migrated;      return true; }
    return false;
}

struct record {
    state       st = state::pending;
    std::string key;        // 64 lowercase hex, SHA-256 through the RKVK frame
    std::string host_file;  // basename of the published host file ("" for pins)
    uint64_t    n_tokens = 0;
    uint32_t    split_p  = 0;
    bool        native   = true;
    std::string timestamp;  // RFC3339 UTC
    source      src = source::explicit_save;
};

inline std::string journal_path(const std::string & store_dir) {
    return (std::filesystem::path(store_dir) / kJournalName).string();
}

inline bool is_journal_file(const std::string & basename) {
    return basename == kJournalName;
}

inline bool is_hex_key(const std::string & key) {
    if (key.size() != 64) return false;
    for (char c : key) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

inline std::string normalize_key(std::string key) {
    for (char & c : key) {
        if (c >= 'A' && c <= 'F') c = char(c - 'A' + 'a');
    }
    return key;
}

inline bool valid_host_file(const std::string & name) {
    if (name.empty() || name == "." || name == "..") return false;
    return name.find('/') == std::string::npos && name.find('\\') == std::string::npos &&
           name.find('\0') == std::string::npos;
}

// A short RFC3339 UTC timestamp; seconds resolution is enough for ordering.
inline std::string timestamp_now() {
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &now);
#else
    gmtime_r(&now, &tm);
#endif
    char buffer[32] = {};
    if (std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &tm) == 0) {
        return {};
    }
    return buffer;
}

inline bool valid_timestamp(const std::string & text) {
    return text.size() == 20 && text[4] == '-' && text[7] == '-' && text[10] == 'T' &&
           text[13] == ':' && text[16] == ':' && text[19] == 'Z';
}

inline bool valid_record(const record & value, std::string * error) {
    const auto reject = [&](const char * message) {
        if (error) *error = message;
        return false;
    };
    if (!is_hex_key(value.key)) return reject("xbox_key is not 64 lowercase hex characters");
    if (value.st == state::pinned) {
        if (value.src != source::migrated) return reject("pinned records must be migrated");
        if (!value.host_file.empty()) return reject("pinned records cannot name a host file");
    } else {
        if (value.src == source::migrated) return reject("only pinned records may be migrated");
        if (!valid_host_file(value.host_file)) return reject("host_file must be a plain basename");
    }
    if (!valid_timestamp(value.timestamp)) return reject("timestamp is not RFC3339 UTC seconds");
    return true;
}

namespace detail {

inline std::string escape_json(const std::string & text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (unsigned char c : text) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", unsigned(c));
                    out += buffer;
                } else {
                    out.push_back(char(c));
                }
        }
    }
    return out;
}

inline bool hex_nibble(char c, unsigned & out) {
    if (c >= '0' && c <= '9') { out = unsigned(c - '0'); return true; }
    if (c >= 'a' && c <= 'f') { out = unsigned(c - 'a' + 10); return true; }
    if (c >= 'A' && c <= 'F') { out = unsigned(c - 'A' + 10); return true; }
    return false;
}

// Strict parser for the flat record object. Unknown keys are skipped, duplicate
// known keys and malformed values are corruption.
class line_parser {
public:
    explicit line_parser(const std::string & text) : text(text) {}

    bool parse(record & out, std::string * error) {
        skip_space();
        if (!consume('{')) return fail("record does not start with '{'", error);
        while (true) {
            skip_space();
            if (consume('}')) break;
            std::string name;
            if (!parse_string(name, error)) return false;
            skip_space();
            if (!consume(':')) return fail("missing ':' after field name", error);
            skip_space();
            std::string text_value;
            bool is_string = false;
            bool is_number = false;
            bool is_bool   = false;
            bool bool_value = false;
            skip_space();
            if (peek() == '"') {
                if (!parse_string(text_value, error)) return false;
                is_string = true;
            } else if (starts_with("true") || starts_with("false")) {
                bool_value = starts_with("true");
                text_value = bool_value ? "true" : "false";
                advance(text_value.size());
                is_bool = true;
            } else {
                if (!parse_number(text_value, error)) return false;
                is_number = true;
            }
            try {
                if (!apply(name, text_value, is_string, is_number, is_bool, bool_value, out, error)) {
                    return false;
                }
            } catch (const std::exception &) {
                return fail("numeric field is out of range", error);
            }
            skip_space();
            if (consume(',')) continue;
            if (consume('}')) break;
            return fail("expected ',' or '}'", error);
        }
        skip_space();
        if (!at_end()) return fail("trailing content after record", error);
        if (!seen_schema)  return fail("missing schema field", error);
        if (!seen_version) return fail("missing version field", error);
        if (!seen_state)   return fail("missing state field", error);
        if (!seen_key)     return fail("missing xbox_key field", error);
        if (!seen_host)    return fail("missing host_file field", error);
        if (!seen_tokens)  return fail("missing n_tokens field", error);
        if (!seen_split)   return fail("missing split_p field", error);
        if (!seen_native)  return fail("missing native field", error);
        if (!seen_time)    return fail("missing timestamp field", error);
        if (!seen_source)  return fail("missing source field", error);
        return true;
    }

private:
    const std::string & text;
    size_t pos = 0;
    bool seen_schema = false, seen_version = false, seen_state = false;
    bool seen_key = false, seen_host = false, seen_tokens = false, seen_split = false;
    bool seen_native = false, seen_time = false, seen_source = false;

    bool fail(const char * message, std::string * error) {
        if (error) *error = message;
        return false;
    }
    bool at_end() const { return pos >= text.size(); }
    char peek() const { return at_end() ? '\0' : text[pos]; }
    bool consume(char c) {
        if (peek() != c) return false;
        ++pos;
        return true;
    }
    bool starts_with(const char * word) const {
        const size_t size = std::strlen(word);
        return text.compare(pos, size, word) == 0;
    }
    void advance(size_t count) { pos = std::min(text.size(), pos + count); }
    void skip_space() {
        while (!at_end() && (text[pos] == ' ' || text[pos] == '\t')) ++pos;
    }
    bool parse_string(std::string & out, std::string * error) {
        if (!consume('"')) return fail("expected string", error);
        out.clear();
        while (true) {
            if (at_end()) return fail("unterminated string", error);
            const unsigned char c = (unsigned char) text[pos++];
            if (c == '"') return true;
            if (c < 0x20) return fail("raw control character in string", error);
            if (c != '\\') { out.push_back(char(c)); continue; }
            if (at_end()) return fail("unterminated escape", error);
            const char escape = text[pos++];
            switch (escape) {
                case '"':  out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/');  break;
                case 'b':  out.push_back('\b'); break;
                case 'f':  out.push_back('\f'); break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  out.push_back('\r'); break;
                case 't':  out.push_back('\t'); break;
                case 'u': {
                    // Only control characters are allowed in this schema; any
                    // other code point would need UTF-8 encoding and is refused.
                    if (pos + 4 > text.size()) return fail("short \\u escape", error);
                    unsigned code = 0;
                    for (int i = 0; i < 4; ++i) {
                        unsigned nibble = 0;
                        if (!hex_nibble(text[pos + i], nibble)) return fail("invalid \\u escape", error);
                        code = code * 16 + nibble;
                    }
                    pos += 4;
                    if (code >= 0x20 || code == 0) return fail("unsupported \\u escape", error);
                    out.push_back(char(code));
                    break;
                }
                default: return fail("invalid escape", error);
            }
        }
    }
    bool parse_number(std::string & out, std::string * error) {
        const size_t start = pos;
        if (peek() == '-') return fail("negative number is not allowed", error);
        while (!at_end() && text[pos] >= '0' && text[pos] <= '9') ++pos;
        if (pos == start) return fail("expected number", error);
        out = text.substr(start, pos - start);
        return true;
    }
    bool apply(const std::string & name, const std::string & value, bool is_string,
               bool is_number, bool is_bool, bool bool_value, record & out, std::string * error) {
        if (name == "schema") {
            if (seen_schema) return fail("duplicate schema field", error);
            seen_schema = true;
            if (!is_string || value != kSchema) return fail("unknown schema", error);
        } else if (name == "version") {
            if (seen_version) return fail("duplicate version field", error);
            seen_version = true;
            if (!is_number || value != std::to_string(kVersion)) return fail("unsupported version", error);
        } else if (name == "state") {
            if (seen_state) return fail("duplicate state field", error);
            seen_state = true;
            if (!is_string || !parse_state(value, out.st)) return fail("unknown state", error);
        } else if (name == "xbox_key") {
            if (seen_key) return fail("duplicate xbox_key field", error);
            seen_key = true;
            if (!is_string) return fail("xbox_key must be a string", error);
            out.key = normalize_key(value);
            if (!is_hex_key(out.key)) return fail("xbox_key is not 64 hex characters", error);
        } else if (name == "host_file") {
            if (seen_host) return fail("duplicate host_file field", error);
            seen_host = true;
            if (!is_string) return fail("host_file must be a string", error);
            out.host_file = value;
        } else if (name == "n_tokens") {
            if (seen_tokens) return fail("duplicate n_tokens field", error);
            seen_tokens = true;
            if (!is_number) return fail("n_tokens must be a number", error);
            out.n_tokens = std::stoull(value);
        } else if (name == "split_p") {
            if (seen_split) return fail("duplicate split_p field", error);
            seen_split = true;
            if (!is_number) return fail("split_p must be a number", error);
            const unsigned long long split = std::stoull(value);
            if (split > UINT32_MAX) return fail("split_p is out of range", error);
            out.split_p = uint32_t(split);
        } else if (name == "native") {
            if (seen_native) return fail("duplicate native field", error);
            seen_native = true;
            if (!is_bool) return fail("native must be a boolean", error);
            out.native = bool_value;
        } else if (name == "timestamp") {
            if (seen_time) return fail("duplicate timestamp field", error);
            seen_time = true;
            if (!is_string) return fail("timestamp must be a string", error);
            out.timestamp = value;
        } else if (name == "source") {
            if (seen_source) return fail("duplicate source field", error);
            seen_source = true;
            if (!is_string || !parse_source(value, out.src)) return fail("unknown source", error);
        }
        // Unknown fields are ignored for forward compatibility.
        return true;
    }
};

} // namespace detail

inline std::string to_json(const record & value) {
    std::string out;
    out.reserve(256);
    out += "{\"schema\":\"";
    out += kSchema;
    out += "\",\"version\":";
    out += std::to_string(kVersion);
    out += ",\"state\":\"";
    out += state_name(value.st);
    out += "\",\"xbox_key\":\"";
    out += value.key;
    out += "\",\"host_file\":\"";
    out += detail::escape_json(value.host_file);
    out += "\",\"n_tokens\":";
    out += std::to_string(value.n_tokens);
    out += ",\"split_p\":";
    out += std::to_string(value.split_p);
    out += ",\"native\":";
    out += value.native ? "true" : "false";
    out += ",\"timestamp\":\"";
    out += detail::escape_json(value.timestamp);
    out += "\",\"source\":\"";
    out += source_name(value.src);
    out += "\"}";
    return out;
}

inline bool parse_line(const std::string & line, record & out, std::string * error) {
    out = record{};
    detail::line_parser parser(line);
    std::string local_error;
    if (!parser.parse(out, &local_error)) {
        if (error) *error = local_error;
        return false;
    }
    std::string record_error;
    if (!valid_record(out, &record_error)) {
        if (error) *error = record_error;
        return false;
    }
    return true;
}

struct load_result {
    bool ok = false;
    std::string error;
    // A crash mid-append can leave an unterminated final line. A complete
    // record without its terminator is kept (nothing complete is dropped) and
    // the next append repairs the terminator; an incomplete fragment is
    // ignored. Both are reported here, never silently rewritten.
    bool truncated_tail = false;
    std::vector<record> raw;
    std::vector<record> records; // folded, insertion ordered, last state wins
};

// Fold append-only records: (xbox_key, host_file) identifies one reference and
// a later state replaces the earlier one. Pins have no host file, so a pin and
// a committed pair for the same key remain distinct records.
inline std::vector<record> fold_records(const std::vector<record> & raw) {
    std::vector<record> folded;
    for (const auto & entry : raw) {
        const auto match = std::find_if(folded.begin(), folded.end(), [&](const record & cur) {
            return cur.key == entry.key && cur.host_file == entry.host_file;
        });
        if (match == folded.end()) {
            folded.push_back(entry);
        } else {
            *match = entry;
        }
    }
    return folded;
}

inline load_result load(const std::string & path) {
    load_result out;
    std::ifstream input(std::filesystem::u8path(path), std::ios::binary);
    if (!input) {
        // A missing journal is an empty journal, not corruption: the first save
        // creates it.
        std::error_code ec;
        if (!std::filesystem::exists(std::filesystem::u8path(path), ec) && !ec) {
            out.ok = true;
            return out;
        }
        out.error = "reference journal cannot be opened";
        return out;
    }
    const std::string content((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (input.bad()) {
        out.error = "reference journal read failed";
        return out;
    }
    size_t begin = 0;
    size_t line_number = 0;
    while (begin < content.size()) {
        size_t end = content.find('\n', begin);
        bool complete = true;
        if (end == std::string::npos) {
            end = content.size();
            complete = false;
        }
        std::string line = content.substr(begin, end - begin);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        begin = end + 1;
        ++line_number;
        if (line.empty()) {
            // Blank separators carry no record.
            continue;
        }
        record parsed;
        std::string error;
        if (!parse_line(line, parsed, &error)) {
            if (!complete) {
                // Unterminated final line that is not a complete record: an
                // interrupted append. Ignore the fragment and surface it.
                out.truncated_tail = true;
                break;
            }
            out.error = "reference journal line " + std::to_string(line_number) + ": " + error;
            return out;
        }
        if (!complete) {
            // A complete record without its newline terminator is kept (nothing
            // complete is ever dropped) but the missing terminator is reported;
            // the next append repairs it before writing.
            out.truncated_tail = true;
        }
        out.raw.push_back(parsed);
    }
    out.records = fold_records(out.raw);
    out.ok = true;
    return out;
}

namespace detail {

inline std::FILE * open_append(const std::string & path) {
#ifdef _WIN32
    return _wfopen(std::filesystem::u8path(path).wstring().c_str(), L"ab");
#else
    return std::fopen(path.c_str(), "ab");
#endif
}

inline bool sync_file(std::FILE * file) {
    if (std::fflush(file) != 0) return false;
#ifdef _WIN32
    return _commit(_fileno(file)) == 0;
#else
    return ::fsync(::fileno(file)) == 0;
#endif
}

// Classify a non-empty journal that does not end with '\n': either the final
// record is complete (only its terminator was lost) or the append was torn in
// the middle of a record.
inline bool inspect_tail(const std::string & path, bool & needs_terminator, bool & torn,
        std::string * error) {
    needs_terminator = false;
    torn = false;
    std::error_code exists_ec;
    if (!std::filesystem::exists(std::filesystem::u8path(path), exists_ec)) {
        if (exists_ec) {
            if (error) *error = "reference journal cannot be inspected";
            return false;
        }
        return true; // the first append creates the journal
    }
    std::ifstream input(std::filesystem::u8path(path), std::ios::binary);
    if (!input) {
        if (error) *error = "reference journal cannot be inspected";
        return false;
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size <= 0) {
        return true;
    }
    input.seekg(size - 1, std::ios::beg);
    char last = '\0';
    input.get(last);
    if (!input || last == '\n') {
        return true;
    }
    // Records are small; a tail larger than this window is not a record.
    constexpr std::streamoff window = 64 * 1024;
    const std::streamoff begin = size > window ? size - window : 0;
    std::string tail(size_t(size - begin), '\0');
    input.clear();
    input.seekg(begin, std::ios::beg);
    input.read(tail.data(), std::streamsize(tail.size()));
    if (!input && !input.eof()) {
        if (error) *error = "reference journal tail cannot be read";
        return false;
    }
    const size_t newline = tail.find_last_of('\n');
    if (newline == std::string::npos && begin != 0) {
        torn = true;
        return true;
    }
    const std::string fragment = newline == std::string::npos ? tail : tail.substr(newline + 1);
    record parsed;
    std::string parse_error;
    if (fragment.empty() || !parse_line(fragment, parsed, &parse_error)) {
        torn = true;
        return true;
    }
    needs_terminator = true;
    return true;
}

} // namespace detail

// Append one record. `lock_held(path)` must prove the caller holds the
// exclusive route-state store lock for the journal; both save paths call this
// while the publication store lock is still held, so the append and the host
// publication share one critical section.
template <class LockHeld>
inline bool append(const std::string & store_dir, const record & value, LockHeld && lock_held,
        std::string * error) {
    const auto reject = [&](const std::string & message) {
        if (error) *error = message;
        return false;
    };
    std::string record_error;
    if (!valid_record(value, &record_error)) {
        return reject(record_error);
    }
    const std::string path = journal_path(store_dir);
    if (!lock_held(path)) {
        return reject("exclusive route-state store lock is not held for the reference journal");
    }
    bool needs_terminator = false;
    bool torn = false;
    {
        std::string tail_error;
        if (!detail::inspect_tail(path, needs_terminator, torn, &tail_error)) {
            return reject(tail_error);
        }
    }
    if (torn) {
        return reject("reference journal ends with an incomplete record; repair required");
    }
    std::FILE * file = detail::open_append(path);
    if (file == nullptr) {
        return reject("reference journal cannot be opened for append");
    }
    const std::string line = (needs_terminator ? std::string("\n") : std::string()) +
        to_json(value) + "\n";
    const bool written = std::fwrite(line.data(), 1, line.size(), file) == line.size();
    const bool synced = written && detail::sync_file(file);
    const bool closed = std::fclose(file) == 0;
    if (!written || !synced || !closed) {
        return reject("reference journal append failed");
    }
    return true;
}

enum class retire_outcome { referenced, pinned, orphan_candidate };

// Pure decision input: only the journal and the host files currently present.
// A committed or pinned record keeps the key referenced even when the host
// file was evicted: losing the host file must never imply losing the Xbox
// object. A pending record whose host file is still present also counts as
// referenced (crash between publication and the committed append). Only keys
// with no committed/pinned record and no surviving host file are candidates,
// and even then the caller performs no remote delete.
inline retire_outcome retire_decision(const std::string & key, const std::vector<record> & journal,
        const std::set<std::string> & host_files) {
    const std::string normalized = normalize_key(key);
    bool has_pinned  = false;
    bool has_committed = false;
    bool pending_with_host = false;
    for (const auto & entry : journal) {
        if (entry.key != normalized) continue;
        if (entry.st == state::pinned) {
            has_pinned = true;
        } else if (entry.st == state::committed) {
            has_committed = true;
        } else if (!entry.host_file.empty() && host_files.count(entry.host_file)) {
            pending_with_host = true;
        }
    }
    // A pin is the strongest reference and must win regardless of record order.
    if (has_pinned) return retire_outcome::pinned;
    if (has_committed) return retire_outcome::referenced;
    return pending_with_host ? retire_outcome::referenced : retire_outcome::orphan_candidate;
}

inline const char * retire_name(retire_outcome decision) {
    switch (decision) {
        case retire_outcome::referenced:        return "referenced";
        case retire_outcome::pinned:            return "pinned";
        case retire_outcome::orphan_candidate:  return "orphan_candidate";
    }
    return "unknown";
}

struct host_key {
    std::string host_file;
    std::string key;
};

struct audit_report {
    std::vector<std::string> referenced;      // committed reference (host file may be gone)
    std::vector<std::string> pinned;          // migrated pin
    std::vector<std::string> pending_only;    // only pending records, no live host file
    std::vector<std::string> orphan_candidates;
    std::vector<std::string> unregistered_host_files; // host file key missing from the journal
    std::vector<std::string> unreadable_host_files;   // native marker/key unavailable
};

// Read-only: never mutates the journal or any host/remote state.
inline audit_report audit(const std::vector<record> & journal, const std::vector<host_key> & host_keys) {
    audit_report out;
    std::set<std::string> known_keys;
    std::set<std::string> keys;
    for (const auto & entry : journal) {
        known_keys.insert(entry.key);
        keys.insert(entry.key);
    }
    std::set<std::string> host_present;
    for (const auto & entry : host_keys) {
        host_present.insert(entry.host_file);
    }
    for (const auto & key : keys) {
        switch (retire_decision(key, journal, host_present)) {
            case retire_outcome::pinned:           out.pinned.push_back(key); break;
            case retire_outcome::referenced:       out.referenced.push_back(key); break;
            case retire_outcome::orphan_candidate: {
                bool pending = false;
                for (const auto & entry : journal) {
                    if (entry.key == key && entry.st == state::pending) pending = true;
                }
                (pending ? out.pending_only : out.orphan_candidates).push_back(key);
                break;
            }
        }
    }
    for (const auto & entry : host_keys) {
        if (entry.key.empty()) {
            out.unreadable_host_files.push_back(entry.host_file);
        } else if (!known_keys.count(entry.key)) {
            out.unregistered_host_files.push_back(entry.host_file);
        }
    }
    return out;
}

struct migration {
    std::vector<record> pins;
    std::vector<std::string> already_pinned;
    std::vector<std::string> invalid;
};

// Build pinned/migrated records for explicit keys. Existing pins are not
// duplicated, so re-running the migration is idempotent. Nothing here writes.
inline migration plan_migration(const std::vector<std::string> & keys,
        const std::vector<record> & journal, const std::string & timestamp) {
    migration out;
    std::set<std::string> pinned;
    for (const auto & entry : journal) {
        if (entry.st == state::pinned) pinned.insert(entry.key);
    }
    std::set<std::string> seen;
    for (const auto & raw : keys) {
        const std::string key = normalize_key(raw);
        if (!is_hex_key(key)) {
            out.invalid.push_back(raw);
            continue;
        }
        if (!seen.insert(key).second) continue;
        if (pinned.count(key)) {
            out.already_pinned.push_back(key);
            continue;
        }
        record entry;
        entry.st        = state::pinned;
        entry.key       = key;
        entry.host_file.clear();
        entry.n_tokens  = 0;
        entry.split_p   = 0;
        entry.native    = true;
        entry.timestamp = timestamp;
        entry.src       = source::migrated;
        out.pins.push_back(std::move(entry));
    }
    return out;
}

// Parse one key per line; blank lines and '#' comments are ignored. A trailing
// " key" or tab-separated annotation is not accepted: the first token is read.
inline std::vector<std::string> read_key_file(const std::string & path, std::string * error) {
    std::vector<std::string> keys;
    std::ifstream input(std::filesystem::u8path(path));
    if (!input) {
        if (error) *error = "key file cannot be opened: " + path;
        return keys;
    }
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t begin = line.find_first_not_of(" \t");
        if (begin == std::string::npos || line[begin] == '#') continue;
        const size_t end = line.find_first_of(" \t#", begin);
        keys.push_back(line.substr(begin, end == std::string::npos ? std::string::npos : end - begin));
    }
    return keys;
}

// SHA-256 of [0, bytes) of a staged/published host file. At the staged
// callback `bytes` is the complete RKVK frame end; after publication the audit
// passes the manifest's state_bytes for the same range.
inline bool hash_host_prefix(const std::string & path, uint64_t bytes, std::string & out_hex,
        std::string * error) {
    out_hex.clear();
    if (bytes == 0) {
        if (error) *error = "host prefix is empty";
        return false;
    }
    std::ifstream input(std::filesystem::u8path(path), std::ios::binary);
    if (!input) {
        if (error) *error = "host prefix cannot be opened";
        return false;
    }
    rkva_snapshot::Sha256 hash;
    std::vector<char> buffer(1u << 20);
    uint64_t left = bytes;
    while (left > 0) {
        const size_t chunk = size_t(std::min<uint64_t>(left, buffer.size()));
        input.read(buffer.data(), std::streamsize(chunk));
        if (!input || size_t(input.gcount()) != chunk) {
            if (error) *error = "host prefix is shorter than the declared frame end";
            return false;
        }
        hash.update(buffer.data(), chunk);
        left -= chunk;
    }
    const auto digest = hash.finish();
    static const char * digits = "0123456789abcdef";
    out_hex.reserve(digest.size() * 2);
    for (uint8_t byte : digest) {
        out_hex.push_back(digits[byte >> 4]);
        out_hex.push_back(digits[byte & 0x0f]);
    }
    return true;
}

} // namespace server_native_refs
