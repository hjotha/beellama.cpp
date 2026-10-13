#pragma once
// TEST-ONLY shared extractor for the KVarN stage capture produced by
// llama-kv-cache-kvarn.cpp f4eval. Included by BOTH
//   tests/test-kv-mixed-destination-intact.cpp
//   tests/test-flash-attn-ext-params.cpp
// so a duplicate implementation cannot diverge from the one under test.
//
// WRITER CONTRACT (src/llama-kv-cache-kvarn.cpp, f4eval):
//     n_slot = st->ne[2];                                  // PHYSICAL stage cells
//     row    = ggml_row_size(st->type, m.head_dim);        // head_dim elements
//     off    = h*row + s*st->nb[2];  ->  cs.raw + s*row
// => one record per PHYSICAL STAGE CELL of `row_width` elements, laid out
//    base = stage_cell * row_width, with elem = n_cells * row_width.
// A logical position maps through the ring descriptor to a physical stage cell
// (group 0 => slot 0, so pos0 -> cell0 and pos1 -> cell1, DIFFERENT rows).
// `pos % group` is NOT a feature-dimension index and is never used here.

#include <cctype>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace kvarn_stage {

enum status { UNAVAILABLE, NONFINITE, ZERO, NONZERO };

inline const char * status_name(int st) {
    switch (st) {
        case UNAVAILABLE: return "unavailable";
        case NONFINITE:   return "nonfinite";
        case ZERO:        return "valid-zero";
        default:          return "valid-nonzero";
    }
}

// Ring-descriptor mapping, matching ggml_cuda_fattn_kvarn_stage_pos and the
// store's stage_slot formula.
inline int stage_cell_for_pos(int pos, int group_dim, int tail_groups, bool ring_known = true) {
    if (group_dim <= 0 || pos < 0) { return -1; }              // refuse before division
    const int g = pos / group_dim;
    const int p = pos % group_dim;
    if (g == 0) { return p; }                                  // group0 => slot0 => cell p
    // Any other group needs the REAL ring assignment. tail_groups<=0 means the
    // ring mapping is unknown -> refuse instead of assuming slot 1.
    if (tail_groups <= 0 || !ring_known) { return -1; }
    return (1 + ((g - 1) % tail_groups)) * group_dim + p;
}

// Derive (n_cells, row_width) from the WRITER's own per-head offsets.
// f4eval does:
//     n = cs.nbytes / sizeof(ggml_fp16_t);  cs.vec.resize(n);
//     ggml_fp16_to_fp32_row(...)            -> ONE FLOAT PER F16 ELEMENT
//     cs.offset = h * row                   -> row = ggml_row_size(F16, head_dim)
// so row_bytes = offset(h1) - offset(h0), row_width = row_bytes / sizeof(fp16),
// n_cells = elem / row_width, and nbytes == elem * sizeof(fp16).
// NOTHING about the model head width is assumed here.
inline bool stage_geometry_from_offsets(size_t elem, size_t nbytes,
                                        long long off_a, long long off_b,
                                        int & n_cells, int & row_width) {
    n_cells = row_width = 0;
    if (elem == 0) { return false; }
    if (nbytes != elem * sizeof(uint16_t)) { return false; }   // fp16-backed capture
    const long long row_bytes = off_b > off_a ? (off_b - off_a) : (off_a - off_b);
    if (row_bytes <= 0) { return false; }                      // positive stride
    if ((row_bytes % 2) != 0) { return false; }                // even (fp16 element)
    const long long rw = row_bytes / (long long) sizeof(uint16_t);
    if (rw <= 0 || elem % size_t(rw) != 0) { return false; }   // exact closure
    row_width = int(rw);
    n_cells = int(elem / size_t(rw));
    return n_cells > 0;
}

// Full-row extract for one physical stage cell. Bounds: base + row_width <= elem.
// Returns false with UNAVAILABLE/NONFINITE instead of fabricating zeros.
inline bool extract_row(const std::vector<float> & snap, size_t elem, int stage_cell,
                        int row_width, std::vector<float> & out, int & st) {
    out.clear();
    st = UNAVAILABLE;
    if (elem == 0 || elem > snap.size()) { return false; }   // bounds BEFORE indexing
    if (row_width <= 0 || stage_cell < 0) { return false; }
    if (size_t(stage_cell) * size_t(row_width) + size_t(row_width) > elem) { return false; }
    const size_t base = size_t(stage_cell) * size_t(row_width);
    out.assign(size_t(row_width), 0.0f);
    for (int i = 0; i < row_width; ++i) { out[size_t(i)] = snap[base + size_t(i)]; }
    st = ZERO;
    for (float v : out) {
        if (!std::isfinite(v)) { st = NONFINITE; return false; }
        if (v != 0.0f) { st = NONZERO; }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Capture-identity family check (llama_kvarn_captured_head info string,
// src/llama-context.cpp:8114):
//   layer= kv= head= pos= route= range= name= type= offset= nbytes= elem=
//   shared= mapped= stored= records= backend= fa= seq= gen=
// Two samples may be combined into ONE row (head0's offset vs head1's offset
// defines the row stride) only if they are the same capture family. Equal
// nbytes is not sufficient; differing identity is not proof of family either.
// Normalized away: ONLY `head=` and that head's expected `offset=`.
// Everything else -- context/clear-generation cohort (seq/gen), layer, V/K,
// pos, route, range, name/shape, type, nbytes, elem, shared/mapped, stored
// leaf, records leaf, backend -- must match byte for byte.
inline bool capture_same_family(const std::string & a, const std::string & b,
                                int * head_a, int * head_b) {
    if (head_a) { *head_a = -1; }
    if (head_b) { *head_b = -1; }
    // "absent layer=..." (miss) is never a family member.
    if (a.rfind("layer=", 0) != 0 || b.rfind("layer=", 0) != 0) { return false; }
    auto get_ll = [](const std::string & s, const char * key) -> long long {
        const size_t p = s.find(key);
        if (p == std::string::npos) { return LLONG_MIN; }
        return strtoll(s.c_str() + p + strlen(key), nullptr, 10);
    };
    const long long ha = get_ll(a, " head=");
    const long long hb = get_ll(b, " head=");
    if (ha == LLONG_MIN || hb == LLONG_MIN) { return false; }
    if (!((ha == 0 && hb == 1) || (ha == 1 && hb == 0))) { return false; }  // heads 0/1
    const auto norm = [](std::string s) -> std::string {
        for (const char * key : {" head=", " offset="}) {
            const size_t p = s.find(key);
            if (p == std::string::npos) { continue; }
            size_t e = p + strlen(key);
            while (e < s.size() && isdigit((unsigned char) s[e])) { ++e; }
            s.replace(p, e - p, std::string(key) + "#");
        }
        return s;
    };
    if (norm(a) != norm(b)) { return false; }
    if (head_a) { *head_a = (int) ha; }
    if (head_b) { *head_b = (int) hb; }
    return true;
}

// `fa=` is the sample's FA-hit count: a fresh FA hit is required before the
// pair may be used (no stale/host-only capture).
inline long long capture_fa_hits(const std::string & info) {
    const size_t p = info.find(" fa=");
    if (p == std::string::npos) { return -1; }
    return strtoll(info.c_str() + p + 4, nullptr, 10);
}

}  // namespace kvarn_stage
