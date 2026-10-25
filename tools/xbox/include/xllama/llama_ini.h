// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// LocalState\llama.ini — one home for the llama.cpp session knobs that used
// to need one file each (gguf_gpu_layers.txt, kv_q8.txt) or a catalogue edit
// (n_ctx). Flat `key = value` lines; `#` and `;` start comments, `[section]`
// headers are ignored (one flat namespace), keys are lowercased, surrounding
// whitespace is trimmed, and the last occurrence wins. Unknown keys are kept
// in the map so the caller can log them; a typo never silently changes
// behaviour.
//
// Precedence, documented here and in docs/using-the-app.md: an explicit
// single-purpose LocalState file (gguf_gpu_layers.txt, kv_q8.txt) wins over
// llama.ini, which wins over the catalogue/compiled default.
#pragma once

#include <cctype>
#include <map>
#include <string>

namespace xllama {

// Session knobs llama.ini may carry (SessionParams fields; n_predict and
// sampling stay per-request on the LAN API).
inline constexpr const char* kLlamaIniFile = "llama.ini";

// Output budget default from llama.ini [n_predict]. -1 = key absent or not a
// positive integer, so every surface keeps its own historical default. Written
// once per process by the bridge's apply_llama_ini_session (populate-only, like
// the rest of the parser) and read where a surface needs a fallback: the API
// server (no max_tokens in the request) and the chat UI (catalogue has no
// n_predict). Explicit per-request/per-turn values still win.
inline int llama_ini_n_predict = -1;

using LlamaIni = std::map<std::string, std::string>;

inline std::string llama_ini_trim(const std::string& s) {
    std::size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b])) != 0)
        ++b;
    std::size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])) != 0)
        --e;
    return s.substr(b, e - b);
}

inline LlamaIni parse_llama_ini(const std::string& text) {
    LlamaIni out;
    std::size_t pos = 0;
    while (pos <= text.size()) {
        std::size_t nl = text.find('\n', pos);
        if (nl == std::string::npos)
            nl = text.size();
        std::string line = llama_ini_trim(text.substr(pos, nl - pos));
        pos = nl + 1;
        if (line.empty() || line[0] == '#' || line[0] == ';' || line[0] == '[')
            continue;
        const std::size_t eq = line.find('=');
        if (eq == std::string::npos)
            continue;
        std::string key = llama_ini_trim(line.substr(0, eq));
        for (char& c : key)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (key.empty())
            continue;
        out[key] = llama_ini_trim(line.substr(eq + 1));
    }
    return out;
}

// Strict whole-value int; false when absent or not an integer.
inline bool llama_ini_int(const LlamaIni& ini, const char* key, int& out) {
    const auto it = ini.find(key);
    if (it == ini.end() || it->second.empty())
        return false;
    std::size_t used = 0;
    try {
        const int v = std::stoi(it->second, &used);
        if (used != it->second.size())
            return false;
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

// 1/0 plus true/false/yes/no/on/off (case-insensitive); false otherwise.
inline bool llama_ini_bool(const LlamaIni& ini, const char* key, bool& out) {
    const auto it = ini.find(key);
    if (it == ini.end())
        return false;
    std::string v = it->second;
    for (char& c : v)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (v == "1" || v == "true" || v == "yes" || v == "on") {
        out = true;
        return true;
    }
    if (v == "0" || v == "false" || v == "no" || v == "off") {
        out = false;
        return true;
    }
    return false;
}

} // namespace xllama
