// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/diverge.h"

#include <cstdint>
#include <cstring>
#include <vector>

using namespace xllama;

TEST_CASE("diverge: token-axis rule selects mask, static, or drift") {
    // B3 [d,3] vs B4 [d,4]: token axis is 1.
    {
        const int64_t n3[4] = {2560, 3, 1, 1};
        const int64_t n4[4] = {2560, 4, 1, 1};
        CHECK(diverge_token_axes(n3, n4) == (1 << 1));
    }
    // 4D with token axis 2.
    {
        const int64_t n3[4] = {128, 8, 3, 1};
        const int64_t n4[4] = {128, 8, 4, 1};
        CHECK(diverge_token_axes(n3, n4) == (1 << 2));
    }
    // Multi-axis [heads, T, T]: causal 3x3 block mask.
    {
        const int64_t n3[4] = {32, 3, 3, 1};
        const int64_t n4[4] = {32, 4, 4, 1};
        CHECK(diverge_token_axes(n3, n4) == ((1 << 1) | (1 << 2)));
    }
    // Static tensor (identical both runs): -2, skip without comparing.
    {
        const int64_t n3[4] = {2560, 9216, 1, 1};
        const int64_t n4[4] = {2560, 9216, 1, 1};
        CHECK(diverge_token_axes(n3, n4) == -2);
    }
    // Static dim-3 coincidence (captured via the 3-or-4 union rule): -2.
    {
        const int64_t n3[4] = {3, 64, 1, 1};
        const int64_t n4[4] = {3, 64, 1, 1};
        CHECK(diverge_token_axes(n3, n4) == -2);
    }
    // Static-3 axis beside a token axis is static, not drift.
    {
        const int64_t n3[4] = {3, 2560, 3, 1};
        const int64_t n4[4] = {3, 2560, 4, 1};
        CHECK(diverge_token_axes(n3, n4) == (1 << 2));
    }
    // Non-token axes must match exactly.
    {
        const int64_t n3[4] = {2560, 3, 1, 1};
        const int64_t n4[4] = {2048, 4, 1, 1};
        CHECK(diverge_token_axes(n3, n4) == -1);
    }
    // One side 3/5 where a token pair is required: drift.
    {
        const int64_t n3[4] = {2560, 3, 1, 1};
        const int64_t n4[4] = {2560, 5, 1, 1};
        CHECK(diverge_token_axes(n3, n4) == -1);
    }
}

TEST_CASE("diverge: slice offsets cover first-3 rows along the mask") {
    // Dense 2D [4 x 3] f32, token mask for axis 1: the 3 columns of 4 floats.
    {
        const int64_t ne[4] = {4, 3, 1, 1};
        const size_t nb[4] = {4, 16, 48, 48};
        const auto offs = diverge_slice_offsets(ne, nb, 4, (1 << 1), 0, 3);
        CHECK(offs.size() == 12);
        for (size_t i = 0; i < 4; ++i) {
            CHECK(offs[i] == i * 4);          // first row elements
            CHECK(offs[4 + i] == 16 + i * 4); // second row
            CHECK(offs[8 + i] == 32 + i * 4); // third row
        }
    }
    // Multi-axis mask [.., T, T]: causal 3x3 block of a 4x4 plane.
    {
        const int64_t ne[4] = {2, 4, 4, 1};
        const size_t nb[4] = {4, 8, 32, 128};
        const auto offs = diverge_slice_offsets(ne, nb, 4, (1 << 1) | (1 << 2), 0, 3);
        CHECK(offs.size() == 2 * 3 * 3);
        for (size_t o : offs) {
            const size_t i1 = (o / 8) % 4;
            const size_t i2 = (o / 32) % 4;
            CHECK(i1 < 3);
            CHECK(i2 < 3);
        }
    }
    // nkeep shorter than the axis: only requested rows.
    {
        const int64_t ne[4] = {4, 4, 1, 1};
        const size_t nb[4] = {4, 16, 64, 64};
        const auto offs = diverge_slice_offsets(ne, nb, 4, (1 << 1), 0, 3);
        CHECK(offs.size() == 12);
        for (size_t i = 0; i < offs.size(); ++i)
            CHECK(offs[i] < 48);
    }
    // Invalid mask or empty: empty walk.
    {
        const int64_t ne[4] = {4, 3, 1, 1};
        const size_t nb[4] = {4, 16, 48, 48};
        CHECK(diverge_slice_offsets(ne, nb, 4, 0, 0, 3).empty());
        CHECK(diverge_slice_offsets(ne, nb, 4, 16, 0, 3).empty());
    }
    // Shifted window nstart=1 over 4 rows: rows 1..3 only.
    {
        const int64_t ne[4] = {4, 4, 1, 1};
        const size_t nb[4] = {4, 16, 64, 64};
        const auto offs = diverge_slice_offsets(ne, nb, 4, (1 << 1), 1, 3);
        CHECK(offs.size() == 12);
        for (size_t i = 0; i < offs.size(); ++i) {
            CHECK(offs[i] >= 16);
            CHECK(offs[i] < 64);
        }
    }
    // Shifted multi-axis window: 2x2 block at [1..3)x[1..3) of a 4x4 plane.
    {
        const int64_t ne[4] = {2, 4, 4, 1};
        const size_t nb[4] = {4, 8, 32, 128};
        const auto offs = diverge_slice_offsets(ne, nb, 4, (1 << 1) | (1 << 2), 1, 2);
        CHECK(offs.size() == 2 * 2 * 2);
        for (size_t o : offs) {
            const size_t i1 = (o / 8) % 4;
            const size_t i2 = (o / 32) % 4;
            CHECK(i1 >= 1);
            CHECK(i1 < 3);
            CHECK(i2 >= 1);
            CHECK(i2 < 3);
        }
    }
}

TEST_CASE("diverge: provenance tiers classify ops conservatively") {
    for (const char* op : {"MUL_MAT", "MUL", "ADD", "RMS_NORM", "UNARY", "GLU", "ROPE", "SCALE"})
        CHECK(diverge_tier_by_name(op) == 1);
    // State-family, presentation, scatter, fused recurrent, custom: tier 2.
    for (const char* op : {"VIEW", "CPY", "RESHAPE", "TRANSPOSE", "GET_ROWS", "SSM_CONV", "CONCAT",
                           "SET_ROWS", "GATED_DELTA_NET", "FLASH_ATTN", "SOFT_MAX", ""})
        CHECK(diverge_tier_by_name(op) == 2);
    // Unknown names default to tier 2 (excluded from ranking, never pass).
    CHECK(diverge_tier_by_name("SOME_FUTURE_OP") == 2);
}

TEST_CASE("diverge: fnv distinguishes bytes, maxabs measures floats") {
    const float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float b[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float c[4] = {1.0f, 2.0f, 3.0f, 5.0f};
    CHECK(diverge_fnv(a, sizeof(a)) == diverge_fnv(b, sizeof(b)));
    CHECK(diverge_fnv(a, sizeof(a)) != diverge_fnv(c, sizeof(c)));
    CHECK(diverge_maxabs_f32(a, b, 4) == doctest::Approx(0.0));
    CHECK(diverge_maxabs_f32(a, c, 4) == doctest::Approx(1.0));
}

TEST_CASE("diverge: row comparison reports hash and maxabs") {
    DivergeTensorRow a, b;
    a.slice = {1.0f, 2.0f};
    b.slice = {1.0f, 2.0f};
    DivergeCmp eq = compare_diverge_rows(a, b);
    CHECK(eq.hash_match);
    CHECK(eq.maxabs == doctest::Approx(0.0));
    b.slice = {1.0f, 3.0f};
    DivergeCmp ne = compare_diverge_rows(a, b);
    CHECK_FALSE(ne.hash_match);
    CHECK(ne.maxabs == doctest::Approx(1.0));
    // Size mismatch or empty: no comparison, never a match.
    DivergeTensorRow e;
    DivergeCmp bad = compare_diverge_rows(a, e);
    CHECK_FALSE(bad.hash_match);
    CHECK(bad.maxabs == doctest::Approx(0.0));
}

// Fixture builder: one interleaved (b3,b4) pair. Hashes/val echoes model the
// device verdict fields; the pairing helper under test is production code.
static DivergeTensorRow diverge_fixture_row(const char* branch, int rep, int index, uint64_t hash,
                                            double maxabs, const std::string& skip = "") {
    DivergeTensorRow r;
    r.branch = branch;
    r.rep = rep;
    r.index = index;
    r.name = "attn_out";
    r.op_name = "MUL_MAT";
    r.op = 7;
    r.ne[0] = 2560;
    r.ne[1] = (std::string(branch) == "b3") ? 3 : 4;
    r.ne[2] = 1;
    r.ne[3] = 1;
    r.type = 0;
    r.axes = (1 << 1);
    r.skip_why = skip;
    r.hash = hash;
    r.maxabs = maxabs;
    r.hash_alt = hash; // same-window reference side (B3 repeats unshifted)
    r.maxabs_alt = maxabs;
    return r;
}

static std::vector<DivergeTensorRow> diverge_fixture_rep(int rep) {
    // Sparse graph indices 0, 7, 30: match, symmetric static skip, mismatch.
    std::vector<DivergeTensorRow> rows;
    rows.push_back(diverge_fixture_row("b3", rep, 0, 0xaaa, 0.0));
    rows.push_back(diverge_fixture_row("b4", rep, 0, 0xaaa, 0.0));
    rows.push_back(diverge_fixture_row("b3", rep, 7, 0, -1.0, "static"));
    rows.push_back(diverge_fixture_row("b4", rep, 7, 0, -1.0, "static"));
    rows.push_back(diverge_fixture_row("b3", rep, 30, 0x111, 0.0202318));
    rows.push_back(diverge_fixture_row("b4", rep, 30, 0x222, 0.0202318));
    return rows;
}

TEST_CASE("diverge: pairing validates interleaved sparse layout") {
    std::vector<DivergeTensorRow> rows;
    for (const auto& r : diverge_fixture_rep(0))
        rows.push_back(r);
    for (const auto& r : diverge_fixture_rep(1))
        rows.push_back(r);
    DivergeExport ex = diverge_pair_rows(rows);
    CHECK(ex.fail_why.empty());
    REQUIRE(ex.reps.size() == 2);
    for (const auto& s : ex.reps) {
        CHECK(s.n_tensors == 3);
        CHECK(s.n_matched == 1);
        CHECK(s.n_skipped == 1);
        CHECK(s.first_diff == 2); // pair position, not graph index
    }
    REQUIRE(ex.pairs.size() == 6);
    CHECK(ex.pairs[0].index == 0);
    CHECK(ex.pairs[0].match == 1);
    CHECK(ex.pairs[1].note == "skipped:static");
    CHECK(ex.pairs[1].maxabs == doctest::Approx(-1.0));
    CHECK(ex.pairs[2].index == 30);
    CHECK(ex.pairs[2].match == 0);
    CHECK(ex.pairs[2].maxabs == doctest::Approx(0.0202318));
    // Emission order: rep0 pairs then rep1 pairs.
    CHECK(ex.pairs[3].rep == 1);
    CHECK(ex.pairs[3].index == 0);
}

TEST_CASE("diverge: csv fields are byte-exact") {
    std::vector<DivergeTensorRow> rows = diverge_fixture_rep(0);
    for (const auto& r : diverge_fixture_rep(1))
        rows.push_back(r);
    DivergeExport ex = diverge_pair_rows(rows);
    REQUIRE(ex.pairs.size() == 6);
    CHECK(diverge_csv_line(ex.pairs[2]) ==
          "0,30,attn_out,MUL_MAT,\"[2560,3,1,1]/t0\",\"[2560,4,1,1]/t0\",0,0.0202318,-\n");
    CHECK(diverge_csv_line(ex.pairs[1]) ==
          "0,7,attn_out,MUL_MAT,\"[2560,3,1,1]/t0\",\"[2560,4,1,1]/t0\",1,-1,skipped:static\n");
    // Empty name/op fall back to "-" and "?".
    DivergePairOut p = ex.pairs[0];
    p.name.clear();
    p.op_name.clear();
    CHECK(diverge_csv_line(p).rfind("0,0,-,?,", 0) == 0);
}

TEST_CASE("diverge: hash-only rows compare by stored hash") {
    std::vector<DivergeTensorRow> rows;
    DivergeTensorRow a = diverge_fixture_row("b3", 0, 5, 0xabc, -1.0);
    DivergeTensorRow b = diverge_fixture_row("b4", 0, 5, 0xdef, -1.0);
    a.slice.clear();
    b.slice.clear(); // no floats shipped: hash-only verdict
    rows.push_back(a);
    rows.push_back(b);
    DivergeTensorRow c = diverge_fixture_row("b3", 1, 5, 0xabc, -1.0);
    DivergeTensorRow d = diverge_fixture_row("b4", 1, 5, 0xabc, -1.0);
    c.slice.clear();
    d.slice.clear();
    rows.push_back(c);
    rows.push_back(d);
    DivergeExport ex = diverge_pair_rows(rows);
    CHECK(ex.fail_why.empty());
    REQUIRE(ex.pairs.size() == 2);
    CHECK(ex.pairs[0].match == 0);
    CHECK(ex.pairs[0].maxabs == doctest::Approx(-1.0));
    CHECK(ex.pairs[1].match == 1);
}

TEST_CASE("diverge: layout and discipline breaks fail loudly") {
    // Odd count.
    {
        std::vector<DivergeTensorRow> rows = diverge_fixture_rep(0);
        rows.pop_back();
        DivergeExport ex = diverge_pair_rows(rows);
        CHECK(ex.fail_why == "diverge row count is not 4-balanced");
    }
    // Empty.
    {
        DivergeExport ex = diverge_pair_rows({});
        CHECK(ex.fail_why == "diverge row count is not 4-balanced");
    }
    // Branch mismatch.
    {
        std::vector<DivergeTensorRow> rows = diverge_fixture_rep(0);
        for (const auto& r : diverge_fixture_rep(1))
            rows.push_back(r);
        rows[0].branch = "b4";
        DivergeExport ex = diverge_pair_rows(rows);
        CHECK(ex.fail_why == "diverge pairing broken");
    }
    // Cross-branch index mismatch.
    {
        std::vector<DivergeTensorRow> rows = diverge_fixture_rep(0);
        for (const auto& r : diverge_fixture_rep(1))
            rows.push_back(r);
        rows[1].index = 31;
        DivergeExport ex = diverge_pair_rows(rows);
        CHECK(ex.fail_why == "diverge pairing broken");
    }
    // Name drift within a pair.
    {
        std::vector<DivergeTensorRow> rows = diverge_fixture_rep(0);
        for (const auto& r : diverge_fixture_rep(1))
            rows.push_back(r);
        rows[1].name = "other_out";
        DivergeExport ex = diverge_pair_rows(rows);
        CHECK(ex.fail_why == "diverge structural drift within rep");
    }
    // Asymmetric skip.
    {
        std::vector<DivergeTensorRow> rows = diverge_fixture_rep(0);
        for (const auto& r : diverge_fixture_rep(1))
            rows.push_back(r);
        rows[3].skip_why = "empty";
        DivergeExport ex = diverge_pair_rows(rows);
        CHECK(ex.fail_why == "diverge asymmetric skip");
    }
    // Shape-drift skip is a failure, not evidence.
    {
        std::vector<DivergeTensorRow> rows = diverge_fixture_rep(0);
        for (const auto& r : diverge_fixture_rep(1))
            rows.push_back(r);
        rows[2].skip_why = rows[3].skip_why = "shape-drift";
        DivergeExport ex = diverge_pair_rows(rows);
        CHECK(ex.fail_why == "diverge shape-drift");
    }
    // Stored verdict mismatch.
    {
        std::vector<DivergeTensorRow> rows = diverge_fixture_rep(0);
        for (const auto& r : diverge_fixture_rep(1))
            rows.push_back(r);
        rows[1].maxabs = 0.5;
        DivergeExport ex = diverge_pair_rows(rows);
        CHECK(ex.fail_why == "diverge stored verdict mismatch");
    }
    // Rep field drift.
    {
        std::vector<DivergeTensorRow> rows = diverge_fixture_rep(0);
        for (const auto& r : diverge_fixture_rep(1))
            rows.push_back(r);
        rows[6].rep = 0;
        DivergeExport ex = diverge_pair_rows(rows);
        CHECK(ex.fail_why == "diverge pairing broken");
    }
}

TEST_CASE("diverge: all-match leaves first_diff at -1") {
    std::vector<DivergeTensorRow> rows;
    for (int rep = 0; rep < 2; ++rep) {
        rows.push_back(diverge_fixture_row("b3", rep, 0, 0xaaa, 0.0));
        rows.push_back(diverge_fixture_row("b4", rep, 0, 0xaaa, 0.0));
    }
    DivergeExport ex = diverge_pair_rows(rows);
    CHECK(ex.fail_why.empty());
    REQUIRE(ex.reps.size() == 2);
    CHECK(ex.reps[0].first_diff == -1);
    CHECK(ex.reps[0].n_matched == 1);
}

TEST_CASE("diverge: failures carry the offending pair identity") {
    std::vector<DivergeTensorRow> rows = diverge_fixture_rep(0);
    for (const auto& r : diverge_fixture_rep(1))
        rows.push_back(r);
    rows[1].name = "other_out"; // structural drift at rep0 pair 0
    DivergeExport ex = diverge_pair_rows(rows);
    CHECK(ex.fail_why == "diverge structural drift within rep");
    CHECK(ex.fail_pair == 0);
    CHECK(ex.fail_a.name == "attn_out");
    CHECK(ex.fail_b.name == "other_out");
    CHECK(diverge_row_string(ex.fail_b).find("other_out") != std::string::npos);
    CHECK(diverge_row_string(ex.fail_b).find("idx=0") != std::string::npos);
}

TEST_CASE("diverge: slice-missing fails loudly, not silently skipped") {
    std::vector<DivergeTensorRow> rows = diverge_fixture_rep(0);
    for (const auto& r : diverge_fixture_rep(1))
        rows.push_back(r);
    rows[0].skip_why = rows[1].skip_why = "slice-missing";
    DivergeExport ex = diverge_pair_rows(rows);
    CHECK(ex.fail_why == "diverge slice-missing");
}

TEST_CASE("diverge: forward mismatch stands despite shifted equality") {
    // Anti-min-rule regression: a Tier-1 (MUL_MAT) pair whose unshifted rows
    // differ MUST remain mismatched even when the shifted comparison happens
    // to match. Alignment comes from provenance, never from values.
    std::vector<DivergeTensorRow> rows;
    for (int rep = 0; rep < 2; ++rep) {
        DivergeTensorRow a = diverge_fixture_row("b3", rep, 62, 0x111, 0.05);
        DivergeTensorRow b = diverge_fixture_row("b4", rep, 62, 0x222, 0.05);
        b.hash_alt = 0x111; // shifted equality must NOT clear a Tier-1 row
        b.maxabs_alt = 0.0001;
        rows.push_back(a);
        rows.push_back(b);
    }
    DivergeExport ex = diverge_pair_rows(rows);
    CHECK(ex.fail_why.empty());
    REQUIRE(ex.pairs.size() == 2);
    CHECK(ex.pairs[0].match == 0);
    CHECK(ex.pairs[0].maxabs == doctest::Approx(0.05));
    CHECK(ex.pairs[0].note.empty());
    CHECK(ex.reps[0].first_diff == 0);
    CHECK(ex.reps[0].n_matched == 0);
}

TEST_CASE("diverge: reverse-slot counterpart reports both alignments") {
    // Tier-2 (CPY presenting snapshot slots): unshifted compares different
    // positions, shifted compares the same positions. Both rows are retained
    // as diagnostic evidence; neither ranks (first_diff stays -1).
    std::vector<DivergeTensorRow> rows;
    for (int rep = 0; rep < 2; ++rep) {
        DivergeTensorRow a = diverge_fixture_row("b3", rep, 59, 0x111, 0.05);
        DivergeTensorRow b = diverge_fixture_row("b4", rep, 59, 0x222, 0.05);
        a.op_name = b.op_name = "CPY";
        a.op = b.op = 3;
        b.hash_alt = 0x111; // B4 rows 1..3 equal B3 rows 0..2
        b.maxabs_alt = 0.0001;
        rows.push_back(a);
        rows.push_back(b);
    }
    DivergeExport ex = diverge_pair_rows(rows);
    CHECK(ex.fail_why.empty());
    REQUIRE(ex.pairs.size() == 4); // two diagnostic rows per rep
    CHECK(ex.pairs[0].note == "tier2:unshifted");
    CHECK(ex.pairs[0].match == 0);
    CHECK(ex.pairs[0].maxabs == doctest::Approx(0.05));
    CHECK(ex.pairs[1].note == "tier2:shifted");
    CHECK(ex.pairs[1].match == 1);
    CHECK(ex.pairs[1].maxabs == doctest::Approx(0.0001));
    CHECK(ex.reps[0].first_diff == -1);
}

TEST_CASE("diverge: multi-axis pairs validate by combo mask") {
    // [heads, T, T] attention-style tensor: compared as the causal block.
    std::vector<DivergeTensorRow> rows;
    for (int rep = 0; rep < 2; ++rep) {
        DivergeTensorRow a = diverge_fixture_row("b3", rep, 40, 0xaaa, 0.0);
        DivergeTensorRow b = diverge_fixture_row("b4", rep, 40, 0xaaa, 0.0);
        a.ne[0] = b.ne[0] = 32;
        a.ne[1] = 3;
        b.ne[1] = 4;
        a.ne[2] = 3;
        b.ne[2] = 4;
        rows.push_back(a);
        rows.push_back(b);
    }
    DivergeExport ex = diverge_pair_rows(rows);
    CHECK(ex.fail_why.empty());
    REQUIRE(ex.pairs.size() == 2);
    CHECK(ex.pairs[0].match == 1);
}

// Fixture builder: one capture node with explicit slices. Models the exact
// device capture output (slices keyed by mask+nstart); the online pairing
// path under test is production code.
static DivergeCapNode diverge_fixture_node(int index, int64_t n_tok, uint64_t h0,
                                           const std::vector<float>& f0, bool with_shifted,
                                           uint64_t h1, const std::vector<float>& f1) {
    DivergeCapNode nd;
    nd.index = index;
    nd.name = "attn_out";
    nd.op = 7;
    nd.op_name = "MUL_MAT";
    nd.ne[0] = 2560;
    nd.ne[1] = n_tok;
    nd.ne[2] = 1;
    nd.ne[3] = 1;
    nd.type = 0;
    DivergeCapNode::AxisSlice s0;
    s0.axes = (1 << 1);
    s0.nstart = 0;
    s0.hash = h0;
    s0.floats = f0;
    nd.slices.push_back(s0);
    if (with_shifted) {
        DivergeCapNode::AxisSlice s1;
        s1.axes = (1 << 1);
        s1.nstart = 1;
        s1.hash = h1;
        s1.floats = f1;
        nd.slices.push_back(s1);
    }
    return nd;
}

TEST_CASE("diverge: online pair emit carries both alignments") {
    DivergeCapNode n3 = diverge_fixture_node(62, 3, 0xaaa, {1.0f, 2.0f}, false, 0, {});
    DivergeCapNode n4 = diverge_fixture_node(62, 4, 0xbbb, {1.0f, 2.5f}, true, 0xccc, {9.0f, 9.5f});
    std::vector<DivergeTensorRow> out;
    diverge_emit_pair(0, n3, n4, &out);
    REQUIRE(out.size() == 2);
    CHECK(out[0].branch == "b3");
    CHECK(out[1].branch == "b4");
    CHECK(out[0].index == 62);
    CHECK(out[1].index == 62);
    CHECK(out[0].skip_why.empty());
    CHECK(out[1].skip_why.empty());
    CHECK(out[0].hash == 0xaaa);
    CHECK(out[1].hash == 0xbbb);
    CHECK(out[0].maxabs == doctest::Approx(0.5));
    CHECK(out[1].maxabs == doctest::Approx(0.5));
    CHECK(out[1].hash_alt == 0xccc);
    CHECK(out[0].hash_alt == 0xaaa); // B3 side repeats the unshifted slice
    CHECK(out[0].slice.empty());     // payload freed at emit
    CHECK(out[1].slice.empty());
}

TEST_CASE("diverge: online pair emit marks skips symmetrically") {
    // Static shapes on both sides.
    DivergeCapNode n3 = diverge_fixture_node(12, 3, 0, {}, false, 0, {});
    DivergeCapNode n4 = diverge_fixture_node(12, 3, 0, {}, false, 0, {});
    n3.ne[1] = n4.ne[1] = 3;
    n3.ne[0] = n4.ne[0] = 8192;
    std::vector<DivergeTensorRow> out;
    diverge_emit_pair(1, n3, n4, &out);
    REQUIRE(out.size() == 2);
    CHECK(out[0].skip_why == "static");
    CHECK(out[1].skip_why == "static");
    // Structural drift.
    n4.op = 8;
    out.clear();
    diverge_emit_pair(1, n3, n4, &out);
    REQUIRE(out.size() == 2);
    CHECK(out[0].skip_why == "structural");
    CHECK(out[1].skip_why == "structural");
}

TEST_CASE("diverge: narrow capture set is exactly the z-0 neighborhood") {
    CHECK(diverge_want_narrow("z-0"));
    CHECK(diverge_want_narrow("norm-0"));
    CHECK(diverge_want_narrow("attn_norm-0"));
    CHECK_FALSE(diverge_want_narrow("norm-1"));
    CHECK_FALSE(diverge_want_narrow("z-00"));
    CHECK_FALSE(diverge_want_narrow("Z-0"));
    CHECK_FALSE(diverge_want_narrow("attn_output-0"));
    CHECK_FALSE(diverge_want_narrow(""));
    CHECK_FALSE(diverge_want_narrow(nullptr));
}

TEST_CASE("diverge: tensor identity formats metadata without data reads") {
    const int64_t ne[4] = {4096, 3, 1, 1};
    CHECK(diverge_tensor_id("z-0", ne, "q4_k") == "z-0[q4_k 4096x3x1x1]");
    const int64_t nz[4] = {0, 0, 0, 0};
    CHECK(diverge_tensor_id(nullptr, nz, nullptr) == "-[? 0x0x0x0]");
}
