// Regression for the FLASH_ATTN_EXT op_params reader.
//
// ggml_tensor::op_params is int32_t[GGML_MAX_OP_PARAMS/sizeof(int32_t)], so an
// index selects a 32-bit SLOT (byte offset 4*i). ggml_flash_attn_ext() writes
//   float params[] = { scale, max_bias, logit_softcap };
//   ggml_set_op_params(result, params, sizeof(params));   -> bytes 0,4,8
// while ggml_flash_attn_ext_lse_out() stores a HOST POINTER at slot 8 (byte 32).
//
// The reader that used `op_params + N*sizeof(float)` therefore read:
//   max_bias      from byte 16 (KVARN_DOMAIN slot)
//   logit_softcap from byte 32 (the LSE host pointer)
// which made the native exact-tail path apply a pointer-derived softcap and
// produced the observed run-to-run red/green flip.
//
// This test drives the REAL reader (ggml_flash_attn_ext_get_alibi_params) with
// explicit non-zero max_bias and logit_softcap, with and without an attached
// LSE, plus the all-zero settings, and asserts the legacy expression fails.
#include "ggml.h"

#include <cmath>
#include "kvarn-stage-extract.h"
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>

static int g_fail = 0;
static void check(bool ok, const char * what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) { ++g_fail; }
}

static bool eq(float a, float b) {
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}


// ---- offline check of the SHARED extractor (no GPU) -----------------------
// Uses tests/kvarn-stage-extract.h -- the same implementation the destination
// test uses -- so a duplicate cannot diverge. Expectations come from the writer
// contract (n_slot = st->ne[2], row = ggml_row_size(type, head_dim)), and the
// observed metadata (elem=98304, nbytes=196608, offsets 0/512 -> 384 x 256).
static void run_extractor_selfcheck() {
    // Correct writer contract: n = cs.nbytes/sizeof(ggml_fp16_t); cs.vec.resize(n);
    // ggml_fp16_to_fp32_row(...) -> ONE FLOAT PER F16 ELEMENT; cs.offset = h*row.
    // So row_width = row_bytes/sizeof(fp16), derived below from offsets 0/512.
    const int group_dim = 128;                 // KVAR_N_GROUP (llama-kvarn.h)
    const int n_cells   = 384;                 // OBSERVED st->ne[2]
    const int row_width = 256;                 // OBSERVED (offset1-offset0)/sizeof(fp16)
    const size_t elem = 98304;                 // OBSERVED cs.vec size (n_bytes/2)
    const size_t nbytes = elem * sizeof(uint16_t);   // 196608

    // Contract from the WRITER's own offsets: cs.offset = h*row, one float per
    // F16 element, so row_bytes = offset(h1)-offset(h0) = 512 and
    // row_width = 512/2 = 256, n_cells = 98304/256 = 384. Nothing about the
    // model head width is assumed.
    const long long off0 = 0, off1 = 512;
    int gc = 0, gr = 0;
    check(kvarn_stage::stage_geometry_from_offsets(elem, nbytes, off0, off1, gc, gr)
          && gc == 384 && gr == 256,
          "geometry: writer offsets 0/512 close the observed contract "
          "(384 cells x row_width 256 = 98304)");
    int bad_c = 0, bad_r = 0;
    check(!kvarn_stage::stage_geometry_from_offsets(0, nbytes, off0, off1, bad_c, bad_r),
          "geometry: elem==0 -> refused");
    check(!kvarn_stage::stage_geometry_from_offsets(elem, nbytes + 2, off0, off1, bad_c, bad_r),
          "geometry: nbytes != elem*sizeof(fp16) -> refused");
    check(!kvarn_stage::stage_geometry_from_offsets(elem, nbytes, 512, 512, bad_c, bad_r),
          "geometry: zero stride (off1==off0) -> refused");
    check(!kvarn_stage::stage_geometry_from_offsets(elem, nbytes, 0, 511, bad_c, bad_r),
          "geometry: odd stride (511) -> refused");
    check(!kvarn_stage::stage_geometry_from_offsets(elem + 1, nbytes, off0, off1, bad_c, bad_r),
          "geometry: elem not divisible by row_width -> refused");

    // Unique per-row values so any row mix-up is visible.
    std::vector<float> snap(elem, 0.0f);
    for (int sc = 0; sc < n_cells; ++sc)
        for (int d = 0; d < row_width; ++d)
            snap[size_t(sc) * row_width + d] = float(sc * 1000 + (d % 1000));

    const int c0 = kvarn_stage::stage_cell_for_pos(0, group_dim, 4);   // group0 -> cell0
    const int c1 = kvarn_stage::stage_cell_for_pos(1, group_dim, 4);   // group0 -> cell1
    std::vector<float> r0, r1, rx; int st = -1;
    check(c0 == 0 && c1 == 1, "ring map: group0 pos0->cell0, pos1->cell1 (different cells)");
    check(kvarn_stage::extract_row(snap, elem, c0, row_width, r0, st)
          && st == kvarn_stage::NONZERO && r0.size() == size_t(row_width)
          && r0[0] == 0.0f && r0[row_width - 1] == float(row_width - 1),
          "extract: pos0 -> cell0, FULL row_width copied");
    check(kvarn_stage::extract_row(snap, elem, c1, row_width, r1, st)
          && st == kvarn_stage::NONZERO && r1[0] == 1000.0f,
          "extract: pos1 -> cell1, FULL row_width copied");
    check(!(r0 == r1), "extract: pos0 and pos1 select DIFFERENT rows (no aliasing)");
    check(kvarn_stage::extract_row(snap, elem, 2, row_width, rx, st)
          && rx[0] == 2000.0f, "extract: stage cell 2 selects the third packed row");
    check(!kvarn_stage::extract_row(snap, elem, n_cells, row_width, rx, st)
          && st == kvarn_stage::UNAVAILABLE,
          "extract: stage cell == n_cells is out of range -> unavailable");
    // Bounds: base + row_width <= elem.
    check(!kvarn_stage::extract_row(snap, /*elem=*/20, n_cells - 1, row_width, rx, st)
          && st == kvarn_stage::UNAVAILABLE,
          "extract: base+row_width <= elem enforced (cell383 needs 98560, elem=20)");
    std::vector<float> nanv = snap; nanv[5] = NAN;
    check(!kvarn_stage::extract_row(nanv, elem, c0, row_width, rx, st)
          && st == kvarn_stage::NONFINITE,
          "extract: non-finite reports nonfinite, never a fabricated zero");
    // Declared elem LARGER than the actual vector must be refused, not clamped.
    check(!kvarn_stage::extract_row(snap, /*elem=*/elem + 64, 0, row_width, rx, st)
          && st == kvarn_stage::UNAVAILABLE,
          "extract: declared elem > snap.size() is refused before indexing");
    check(!kvarn_stage::extract_row(snap, /*elem=*/0, 0, row_width, rx, st)
          && st == kvarn_stage::UNAVAILABLE, "extract: elem==0 is refused");
    check(kvarn_stage::stage_cell_for_pos(-1, group_dim, 4) == -1
          && kvarn_stage::stage_cell_for_pos(0, 0, 4) == -1,
          "ring map: negative pos / invalid group_dim refused BEFORE division");
    check(kvarn_stage::stage_cell_for_pos(200, group_dim, 0, /*ring_known=*/false) == -1
          && kvarn_stage::stage_cell_for_pos(200, group_dim, 0, /*ring_known=*/true) == -1,
          "ring map: unknown ring refuses non-group0 positions");
    std::vector<float> zv(elem, 0.0f);
    check(kvarn_stage::extract_row(zv, elem, c0, row_width, rx, st)
          && st == kvarn_stage::ZERO, "extract: all-zero row is valid-zero");
}


// Pairing rule: head0/head1 may be combined only when they are the SAME
// capture family (src/llama-context.cpp:8114 identity), differing only in
// `head=` and that head's expected `offset=`, and only with a fresh FA hit.
// Equal nbytes is NOT enough; differing identity is NOT proof of family.
static void run_capture_family_selfcheck() {
    const std::string base =
        "layer=3 kv=V head=0 pos=0 route=kvarn range=full name=f4kv_0 type=1 "
        "offset=0 nbytes=196608 elem=98304 shared=1 mapped=0 "
        "stored=f4kv_0:view_src/1 records=kvarn_v:1 backend=CUDA "
        "fa=7 seq=42 gen=1001";
    const auto with = [&base](const char * key, const char * val) {
        std::string o = base;
        const size_t p = o.find(key);
        if (p == std::string::npos) { return o; }
        size_t e = p + std::strlen(key);
        while (e < o.size() && o[e] != ' ') { ++e; }
        o.replace(p, e - p, std::string(key) + val);
        return o;
    };
    const std::string base1 = with(" head=", "1");
    std::string pair1 = base1;
    {   // head1 carries the stride: offset = 1 * row_bytes = 512
        const size_t p = pair1.find(" offset=");
        size_t e = p + std::strlen(" offset=");
        while (e < pair1.size() && isdigit((unsigned char) pair1[e])) { ++e; }
        pair1.replace(p, e - p, " offset=512");
    }
    int ha = -1, hb = -1;
    check(kvarn_stage::capture_same_family(base, pair1, &ha, &hb) && ha == 0 && hb == 1,
          "family: head0/1 same cohort, offset 0 vs 512 -> accepted (heads 0/1)");

    const auto reject = [&](const std::string & x, const char * tag) {
        int a = -1, b = -1;
        check(!kvarn_stage::capture_same_family(x, pair1, &a, &b), tag);
    };
    reject(with(" stored=", "other_leaf:self/0"), "family: different stored leaf -> rejected");
    reject(with(" route=", "tail"),              "family: different route -> rejected");
    reject(with(" gen=", "1002"),                "family: different clear-generation cohort -> rejected");
    reject(with(" seq=", "43"),                  "family: different context seq -> rejected");
    reject(with("layer=", "4"),                 "family: different layer -> rejected");
    reject(with(" kv=", "K"),                    "family: K instead of V -> rejected");
    reject(with(" backend=", "Vulkan"),          "family: different backend -> rejected");
    reject(with(" nbytes=", "196610"),           "family: different nbytes/shape -> rejected");
    {   // heads 0/0 (same head twice) -> rejected
        const std::string wrong = with(" head=", "0");
        const size_t p = wrong.find(" offset=");
        std::string w = wrong;
        size_t e = p + std::strlen(" offset=");
        while (e < w.size() && isdigit((unsigned char) w[e])) { ++e; }
        w.replace(p, e - p, " offset=512");
        int a = -1, b = -1;
        check(!kvarn_stage::capture_same_family(w, base, &a, &b),
              "family: heads 0/0 -> rejected");
    }
    {   // miss string -> rejected
        int a = -1, b = -1;
        const std::string miss = "absent layer=3 kv=V head=0 index=0 of 0";
        check(!kvarn_stage::capture_same_family(miss, pair1, &a, &b),
              "family: absent/miss capture -> rejected");
    }
    check(kvarn_stage::capture_fa_hits(base) == 7 && kvarn_stage::capture_fa_hits(" layer=3") == -1,
          "family: fa_hits parsed; missing fa -> -1");
    check(kvarn_stage::capture_fa_hits(base) > 0, "family: fresh FA hit required (fa>0)");
}

int main() {
    run_extractor_selfcheck();
    run_capture_family_selfcheck();

    ggml_init_params ip = { 16u * 1024u * 1024u, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { std::printf("FAIL: ggml_init\n"); return 1; }

    struct C { float scale, max_bias, softcap; const char * tag; };
    const C cases[] = {
        { 0.088388348f, 0.0f,        0.0f,      "zero max_bias and softcap" },
        { 0.088388348f, 0.25f,       0.0f,      "nonzero max_bias, zero softcap" },
        { 0.088388348f, 0.0f,        3.5f,      "zero max_bias, nonzero softcap" },
        { 0.088388348f, 1.75f,       6.25f,     "nonzero max_bias AND softcap" },
    };

    for (const C & c : cases) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        t->op = GGML_OP_FLASH_ATTN_EXT;
        const float params[3] = { c.scale, c.max_bias, c.softcap };
        // identical write path to ggml_flash_attn_ext(): memcpy into byte 0..11
        std::memcpy(t->op_params, params, sizeof(params));

        float s = -1.0f, mb = -1.0f, sc = -1.0f;
        ggml_flash_attn_ext_get_alibi_params(t, &s, &mb, &sc);
        check(eq(s, c.scale) && eq(mb, c.max_bias) && eq(sc, c.softcap), c.tag);

        // the exact expression that used to be in fattn-tail.cuh must NOT work
        float legacy_scale = 0, legacy_maxbias = 0, legacy_softcap = 0;
        std::memcpy(&legacy_scale,    t->op_params + 0 * sizeof(float), sizeof(float));
        std::memcpy(&legacy_maxbias,  t->op_params + 1 * sizeof(float), sizeof(float));
        std::memcpy(&legacy_softcap,  t->op_params + 2 * sizeof(float), sizeof(float));
        check(eq(legacy_scale, c.scale), "legacy reader still sees scale (byte 0)");
        if (c.max_bias != 0.0f || c.softcap != 0.0f) {
            check(!eq(legacy_maxbias, c.max_bias) || !eq(legacy_softcap, c.softcap),
                  "legacy '+N*sizeof(float)' reader must NOT reproduce the values");
        }

        // attach an LSE: the host pointer lives at slot 8 / byte 32 and must
        // not disturb bytes 0..11, and the shared reader must stay exact.
        ggml_tensor * lse = ggml_flash_attn_ext_lse_out(ctx, t);
        check(lse != nullptr && lse->op == GGML_OP_FLASH_ATTN_EXT_LSE, "lse_out attached");
        float s2 = -1, mb2 = -1, sc2 = -1;
        ggml_flash_attn_ext_get_alibi_params(t, &s2, &mb2, &sc2);
        check(eq(s2, c.scale) && eq(mb2, c.max_bias) && eq(sc2, c.softcap),
              "params survive LSE_PTR attachment (no overlap with bytes 0..11)");
        uintptr_t ptr = 0;
        std::memcpy(&ptr, &t->op_params[GGML_FLASH_ATTN_EXT_OP_PARAM_LSE_PTR], sizeof(ptr));
        check(ptr == (uintptr_t) lse, "LSE_PTR resolves to the attached lse tensor");
        check(GGML_FLASH_ATTN_EXT_OP_PARAM_LSE_PTR * (int) sizeof(int32_t) >= 12,
              "LSE_PTR slot cannot overlap the three float params");

        // and the legacy reader after attachment reads the pointer's low bits
        float legacy_after = 0;
        std::memcpy(&legacy_after, t->op_params + 2 * sizeof(float), sizeof(float));
        if (c.softcap == 0.0f) {
            check(!eq(legacy_after, c.softcap) || ((uintptr_t) legacy_after & 0xffffffffu) == 0,
                  "legacy reader after LSE no longer yields a clean 0.0f softcap");
        }
    }

    ggml_free(ctx);
    std::printf("test-flash-attn-ext-params: %s (%d failures)\n",
                g_fail == 0 ? "PASSED" : "FAILED", g_fail);
    return g_fail == 0 ? 0 : 1;
}
