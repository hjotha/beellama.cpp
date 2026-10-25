// Position-split ubatch boundary (plan §3.2, phase F2a).
//
// The attention range boundary P must not fall inside a prepared ubatch: the
// store of one ubatch would have to write two ranges. This gate covers the
// division of common ubatches at P and the fail-closed rejection of the
// protected recurrent window (the trailing 1 + n_rs_seq tokens that keep the
// rollback snapshots valid), which must happen *before* any state is mutated.
//
// Occupancy shapes required by the plan: P-129, P-128, P-1, P, P+1, P+128.

#include "llama-memory-hybrid.h"
#include "llama-position-split.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int g_checks = 0;
static int g_failures = 0;

static void check(bool ok, const std::string & what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

static void check_close(double a, double b, double tol, const std::string & what) {
    const double d = a > b ? a - b : b - a;
    check(d <= tol, what + " (" + std::to_string(a) + " vs " + std::to_string(b) + ")");
}

// Builds a single-sequence-set token ubatch whose positions start at pos0.
static llama_ubatch make_ubatch(uint32_t n_tokens, llama_pos pos0, uint32_t n_keep_tail) {
    llama_ubatch ub {};
    ub.b_equal_seqs = 1;
    ub.n_tokens     = n_tokens;
    ub.n_seq_tokens = n_tokens;
    ub.n_seqs       = 1;
    ub.n_seqs_unq   = 1;
    ub.n_pos        = 1;

    auto data = std::make_shared<llama_ubatch::data_t>();
    data->token.resize(n_tokens);
    data->pos.resize(n_tokens);
    data->n_seq_id.assign(n_tokens, 1);
    data->seq_id.resize(n_tokens);
    data->output.assign(n_tokens, 0);
    // Non-zero sequence ids: a zeroed or never-written pointer must not be able
    // to look like a pass (review 5, R1).
    const llama_seq_id kSeq = 7;
    data->seq_id_unq = { kSeq };
    data->seq_idx.assign(LLAMA_MAX_SEQ, -1);
    data->seq_idx[kSeq] = 0;
    data->seq_id_data.assign(n_tokens, kSeq);

    for (uint32_t i = 0; i < n_tokens; ++i) {
        data->token[i] = llama_token(1000 + i);
        data->pos[i]   = pos0 + llama_pos(i);
        // The trailing n_keep_tail tokens are the protected window.
        data->output[i] = (n_keep_tail > 0 && i + n_keep_tail >= n_tokens) ? 1 : 0;
        data->seq_id[i] = data->seq_id_data.data() + i;
    }

    ub.data       = data;
    ub.token      = data->token.data();
    ub.embd       = nullptr;
    ub.pos        = data->pos.data();
    ub.n_seq_id   = data->n_seq_id.data();
    ub.seq_id     = data->seq_id.data();
    ub.seq_id_unq = data->seq_id_unq.data();
    ub.seq_idx    = data->seq_idx.data();
    ub.output     = data->output.data();
    return ub;
}


// Component-major position layout exactly as the runtime fills it:
//   pos[i]                 -> sequence position of token i (component 0)
//   pos[n_tokens + i]      -> component 1
//   pos[2*n_tokens + i]    -> component 2  ...  (see llama-kv-cache.cpp:3031-3038)
// The production KV cache reads `ubatch.pos[i]`; every position-split site that
// instead indexed `pos[i * n_pos]` mis-classified tokens whenever n_pos > 1.
static llama_ubatch make_ubatch_npos4(uint32_t n_tokens, llama_pos pos0, uint32_t n_keep_tail) {
    const uint32_t n_pos = 4;
    llama_ubatch ub {};
    ub.b_equal_seqs = 1;
    ub.n_tokens     = n_tokens;
    ub.n_seq_tokens = n_tokens;
    ub.n_seqs       = 1;
    ub.n_seqs_unq   = 1;
    ub.n_pos        = n_pos;

    auto data = std::make_shared<llama_ubatch::data_t>();
    data->token.resize(n_tokens);
    data->pos.resize(size_t(n_tokens) * n_pos);
    data->n_seq_id.assign(n_tokens, 1);
    data->seq_id.resize(n_tokens);
    data->output.assign(n_tokens, 0);
    const llama_seq_id kSeq = 7;
    data->seq_id_unq = { kSeq };
    data->seq_idx.assign(LLAMA_MAX_SEQ, -1);
    data->seq_idx[kSeq] = 0;
    data->seq_id_data.assign(n_tokens, kSeq);

    for (uint32_t i = 0; i < n_tokens; ++i) {
        const llama_pos p = pos0 + llama_pos(i);
        data->token[i] = llama_token(1000 + i);
        // component 0 is the temporal/sequence position; the spatial components
        // differ, exactly as M-RoPE fills them, so an accessor that indexes
        // `pos[i * n_pos]` reads a spatial value instead of the sequence one.
        data->pos[i] = p;
        for (uint32_t c = 1; c < n_pos; ++c) {
            data->pos[size_t(c) * n_tokens + i] = p + llama_pos(1000 * c);
        }
        // pin the layout the KV cache relies on (llama-kv-cache.cpp:3031)
        if (i == 1) {
            data->pos[1] = p;   // pos[1] is token 1's sequence position
        }
        data->output[i] = (n_keep_tail > 0 && i + n_keep_tail >= n_tokens) ? 1 : 0;
        data->seq_id[i] = data->seq_id_data.data() + i;
    }

    ub.data       = data;
    ub.token      = data->token.data();
    ub.embd       = nullptr;
    ub.pos        = data->pos.data();
    ub.n_seq_id   = data->n_seq_id.data();
    ub.seq_id     = data->seq_id.data();
    ub.seq_id_unq = data->seq_id_unq.data();
    ub.seq_idx    = data->seq_idx.data();
    ub.output     = data->output.data();
    return ub;
}

static void check_ubatch_equal(const llama_ubatch & a, const llama_ubatch & b, const std::string & tag) {
    check(a.n_tokens == b.n_tokens, tag + ": n_tokens");
    check(a.n_seq_tokens == b.n_seq_tokens, tag + ": n_seq_tokens");
    check(a.n_seqs == b.n_seqs && a.n_seqs_unq == b.n_seqs_unq, tag + ": n_seqs");
    check(a.n_pos == b.n_pos, tag + ": n_pos");
    for (uint32_t i = 0; i < a.n_tokens && i < b.n_tokens; ++i) {
        check(a.token[i] == b.token[i], tag + ": token order");
        check(a.pos[i] == b.pos[i], tag + ": pos order");
        check(a.output[i] == b.output[i], tag + ": output flag");
        check(a.n_seq_id[i] == b.n_seq_id[i], tag + ": n_seq_id");
        for (int32_t s = 0; s < a.n_seq_id[i]; ++s) {
            check(a.seq_id[i][s] == b.seq_id[i][s], tag + ": seq_id");
        }
    }
}

int main() {
    const uint32_t P = 512;      // aligned boundary used by the gate
    const uint32_t NKEEP = 4;    // 1 + n_rs_seq with n_rs_seq = 3

    // 1) ubatches that do not cross P are passed through untouched.
    {
        std::vector<llama_ubatch> in;
        in.push_back(make_ubatch(64, 0, NKEEP));                 // [0,64)
        in.push_back(make_ubatch(64, P, NKEEP));                  // [P,P+64)
        std::vector<llama_ubatch> out;
        std::string err;
        check(llama_position_split::divide_ubatches_at_p(in, P, NKEEP, out, err),
              "non-crossing ubatches are accepted: " + err);
        check(out.size() == 2, "non-crossing ubatches keep their count");
        if (out.size() == 2) {
            check_ubatch_equal(in[0], out[0], "below P");
            check_ubatch_equal(in[1], out[1], "at/above P");
        }
    }

    // 2) a common ubatch crossing P is divided, preserving token order,
    //    positions and the output flags (plan §3.2, "divide common ubatches").
    {
        const uint32_t n = 300;
        std::vector<llama_ubatch> in;
        in.push_back(make_ubatch(n, 400, NKEEP));   // [400,700) crosses 512
        std::vector<llama_ubatch> out;
        std::string err;
        check(llama_position_split::divide_ubatches_at_p(in, P, NKEEP, out, err),
              "crossing ubatch is divided: " + err);
        check(out.size() == 2, "crossing ubatch yields two ubatches");
        if (out.size() == 2) {
            check(out[0].n_tokens == P - 400, "head holds [400,512)");
            check(out[1].n_tokens == 700 - P, "tail holds [512,700)");
            check(out[0].pos[0] == 400 && out[0].pos[out[0].n_tokens - 1] == P - 1,
                  "head positions are contiguous below P");
            check(out[1].pos[0] == P && out[1].pos[out[1].n_tokens - 1] == 699,
                  "tail positions start at P");
            check(out[0].n_seq_tokens == out[0].n_tokens, "head keeps one sequence set");
            check(out[1].n_seq_tokens == out[1].n_tokens, "tail keeps one sequence set");
            // The protected window is entirely in the tail here.
            uint32_t outputs_head = 0;
            for (uint32_t i = 0; i < out[0].n_tokens; ++i) {
                outputs_head += uint32_t(out[0].output[i] != 0);
            }
            check(outputs_head == 0, "no protected token moves into the head");
            // Token identity is preserved across the division.
            for (uint32_t i = 0; i < out[0].n_tokens; ++i) {
                check(out[0].token[i] == in[0].token[i], "head token identity");
            }
            for (uint32_t i = 0; i < out[1].n_tokens; ++i) {
                check(out[1].token[i] == in[0].token[out[0].n_tokens + i], "tail token identity");
            }
            // R1: the halves own their sequence ids. Real consumers dereference
            // these pointers (llama-memory-recurrent.cpp), so both halves must
            // resolve to the expected ids and stay valid after the source
            // ubatch is released and after the results are moved.
            const auto check_ids = [&](const llama_ubatch & ub, const std::string & tag) {
                for (uint32_t i = 0; i < ub.n_tokens; ++i) {
                    check(ub.seq_id[i] != nullptr, tag + ": seq_id pointer is not null");
                    if (ub.seq_id[i] == nullptr) {
                        return;
                    }
                    check(ub.n_seq_id[i] == 1, tag + ": n_seq_id");
                    for (int32_t k = 0; k < ub.n_seq_id[i]; ++k) {
                        check(ub.seq_id[i][k] == 7, tag + ": sequence id value");
                    }
                }
            };
            check_ids(out[0], "head");
            check_ids(out[1], "tail");
            {
                // Release the source before dereferencing the halves again.
                std::vector<llama_ubatch> moved;
                moved.push_back(std::move(out[0]));
                moved.push_back(std::move(out[1]));
                in.clear();
                in.shrink_to_fit();
                check_ids(moved[0], "head after source release");
                check_ids(moved[1], "tail after source release");
                out = std::move(moved);
            }
        }
    }

    // Optional fields stay absent when splitting token-only input; decision
    // order remains attached to each selected token, with independent storage.
    {
        auto source = make_ubatch(300, 400, NKEEP);
        source.data->decision_order.resize(source.n_tokens);
        for (uint32_t i = 0; i < source.n_tokens; ++i) {
            source.data->decision_order[i] = int32_t(17 + i * 3);
        }
        source.decision_order = source.data->decision_order.data();
        std::vector<llama_ubatch> out;
        std::string err;
        check(llama_position_split::divide_ubatches_at_p({source}, P, NKEEP, out, err),
                "decision-order token batch divides: " + err);
        uint32_t index = 0;
        for (const auto & half : out) {
            check(half.type == nullptr && half.embd == nullptr && half.embd_state == nullptr,
                    "split token batch keeps optional embedding/type pointers null");
            check(half.decision_order != nullptr, "split decision order is present");
            for (uint32_t i = 0; i < half.n_tokens; ++i, ++index) {
                check(half.decision_order && half.decision_order[i] == source.decision_order[index],
                        "split decision order matches original token");
            }
        }
        check(index == source.n_tokens, "decision-order split covers every token");
        std::vector<float> hidden(source.n_tokens, 0.5f);
        source.embd_state = hidden.data();
        check(!llama_position_split::divide_ubatches_at_p({source}, P, NKEEP, out, err),
                "cross-boundary hidden-state batch is rejected without guessed stride");
        check(out.empty() && !err.empty(), "hidden-state refusal publishes no partial batch");
    }

    // 3) the protected recurrent window crossing P is rejected with an
    //    explicit reason, before any mutation: nothing is produced.
    {
        std::vector<llama_ubatch> in;
        // window = last 4 tokens = [509,513) which straddles P = 512
        in.push_back(make_ubatch(12, 501, NKEEP));
        std::vector<llama_ubatch> out;
        std::string err;
        const bool ok = llama_position_split::divide_ubatches_at_p(in, P, NKEEP, out, err);
        check(!ok, "protected window crossing P is rejected");
        check(out.empty(), "rejection produces no ubatch (no partial state)");
        check(!err.empty(), "rejection carries an explicit reason");
        check(err.find("P") != std::string::npos, "reason mentions the boundary");
    }

    // 4) an ubatch that is entirely the protected window and crosses P is also
    //    rejected (dividing it would break the rollback snapshot).
    {
        std::vector<llama_ubatch> in;
        in.push_back(make_ubatch(4, 510, 4));   // whole ubatch is the window
        std::vector<llama_ubatch> out;
        std::string err;
        check(!llama_position_split::divide_ubatches_at_p(in, P, NKEEP, out, err),
              "ubatch inside the protected window is rejected");
        check(out.empty(), "rejection produces no ubatch");
    }

    // 5) plan occupancy shapes around P: P-129, P-128, P-1, P, P+1, P+128.
    //    Each shape is filled with whole ubatches below P plus one trailing
    //    ubatch that lands exactly on, before or after the boundary.
    {
        struct occupancy { const char * name; uint32_t total; };
        const occupancy shapes[] = {
            { "P-129", P - 129 },
            { "P-128", P - 128 },
            { "P-1",   P - 1   },
            { "P",     P       },
            { "P+1",   P + 1   },
            { "P+128", P + 128 },
        };
        const uint32_t ubatch = 64;
        for (const occupancy & occ : shapes) {
            std::vector<llama_ubatch> in;
            uint32_t filled = 0;
            while (filled + ubatch <= occ.total) {
                in.push_back(make_ubatch(ubatch, llama_pos(filled), NKEEP));
                filled += ubatch;
            }
            if (filled < occ.total) {
                in.push_back(make_ubatch(occ.total - filled, llama_pos(filled), NKEEP));
                filled = occ.total;
            }
            check(filled == occ.total, std::string(occ.name) + ": occupancy assembled");

            std::vector<llama_ubatch> out;
            std::string err;
            const bool ok = llama_position_split::divide_ubatches_at_p(in, P, NKEEP, out, err);

            // The protected window only rejects when it straddles P; with a
            // 64-token ubatch grid and NKEEP = 4 it never does, so the split
            // must succeed and no output ubatch may cross P.
            check(ok, std::string(occ.name) + ": accepted (" + err + ")");
            uint32_t total_out = 0;
            bool any_cross = false;
            for (const llama_ubatch & ub : out) {
                total_out += ub.n_tokens;
                const llama_pos lo = ub.pos[0];
                const llama_pos hi = ub.pos[ub.n_tokens - 1];
                if (lo < llama_pos(P) && hi >= llama_pos(P)) {
                    any_cross = true;
                }
            }
            check(!any_cross, std::string(occ.name) + ": no output ubatch crosses P");
            check(total_out == occ.total, std::string(occ.name) + ": token count preserved");
        }
    }

    // 5b) R2: a valid ubatch followed by a rejected one must leave no partial
    // output, because init_batch treats the helper as all-or-nothing.
    {
        std::vector<llama_ubatch> in;
        in.push_back(make_ubatch(64, 0, NKEEP));      // valid, below P
        in.push_back(make_ubatch(12, 501, NKEEP));    // rejected: window crosses P
        std::vector<llama_ubatch> out;
        std::string err;
        check(!llama_position_split::divide_ubatches_at_p(in, P, NKEEP, out, err),
              "R2: batch with a rejected ubatch fails");
        check(out.empty(), "R2: rejection after a valid prefix leaves no output");
        check(!err.empty(), "R2: rejection carries a reason");
    }

    // 5c) R2: a valid prefix followed by a rejection *after* the division point
    // must also leave no output.
    {
        std::vector<llama_ubatch> in;
        in.push_back(make_ubatch(300, 400, NKEEP));   // divided at P
        // positions [509,515): the protected window is the trailing 4 tokens
        // [511,515), which straddles P = 512 and must be rejected.
        in.push_back(make_ubatch(6, 509, NKEEP));
        std::vector<llama_ubatch> out;
        std::string err;
        check(!llama_position_split::divide_ubatches_at_p(in, P, NKEEP, out, err),
              "R2c: batch with a rejected ubatch after a divided one fails");
        check(out.empty(), "R2c: no partial output survives the rejection");
    }


    // 5d) F4: component-major positions (n_pos > 1) must split on the SAME
    // sequence position the KV cache stores (`pos[i]`, llama-kv-cache.cpp:3031).
    // Reading `pos[i * n_pos]` classifies tokens with the wrong position and
    // ships overflow tokens into the local band, which the allocator then
    // cannot satisfy (observed: 216 local tokens vs 214 free local cells).
    {
        const uint32_t n = 300;
        std::vector<llama_ubatch> in;
        in.push_back(make_ubatch_npos4(n, 400, NKEEP));   // positions [400,700)
        std::vector<llama_ubatch> out;
        std::string err;
        check(llama_position_split::divide_ubatches_at_p(in, P, NKEEP, out, err),
              "n_pos=4 crossing ubatch is divided: " + err);
        check(out.size() == 2, "n_pos=4 crossing ubatch yields two ubatches");
        if (out.size() == 2) {
            // Every token of the head must be strictly below P by the accessor
            // the allocator uses (pos[i]), and every tail token at/above P.
            uint32_t head_bad = 0, tail_bad = 0;
            for (uint32_t i = 0; i < out[0].n_tokens; ++i) {
                if (!(out[0].pos[i] >= 400 && out[0].pos[i] < llama_pos(P))) { ++head_bad; }
            }
            for (uint32_t i = 0; i < out[1].n_tokens; ++i) {
                if (!(out[1].pos[i] >= llama_pos(P) && out[1].pos[i] < 700)) { ++tail_bad; }
            }
            check(head_bad == 0, "n_pos=4 head holds only [400,P) by pos[i]");
            check(tail_bad == 0, "n_pos=4 tail holds only [P,700) by pos[i]");
            check(out[0].n_tokens == P - 400, "n_pos=4 head token count is P-400");
            check(out[1].n_tokens == 700 - P, "n_pos=4 tail token count is 700-P");
            // The local band must never receive more tokens than [0,P) can hold:
            // 400..P is P-400 tokens and they are the only local ones.
            uint32_t local_tokens = 0;
            for (uint32_t i = 0; i < n; ++i) {
                if (in[0].pos[i] < llama_pos(P)) { ++local_tokens; }
            }
            check(local_tokens == P - 400, "n_pos=4 local token count matches [400,P)");
            check(out[0].n_tokens == local_tokens,
                  "n_pos=4 divided head holds exactly the local tokens");
        }
    }


    // 5e) F4: tail_extent_queries -- the history-only extent helper. Before the
    // fix every graph used the WHOLE ubatch's q_max, so an overflow ubatch sized
    // mask/read-idxs/run_desc with tail_tokens + q_max while deliberately
    // binding no current segment (fattn-tail.cuh then asserted
    // mt->ne[0] <= kt->ne[1] + kt_current->ne[1]).
    {
        const int64_t q_max = 298;
        // non-P mode must not change legacy behaviour
        check(llama_position_split::tail_extent_queries(false, P, nullptr, 0, q_max) == q_max,
              "tail extent: non-P mode keeps q_max");
        // empty ubatch / no positions -> keep q_max (nothing to classify)
        check(llama_position_split::tail_extent_queries(true, P, nullptr, 0, q_max) == q_max,
              "tail extent: null positions keeps q_max");
        llama_pos none = 0;
        check(llama_position_split::tail_extent_queries(true, P, &none, 0, q_max) == q_max,
              "tail extent: empty ubatch keeps q_max");
        // P disabled inside P-mode plumbing
        check(llama_position_split::tail_extent_queries(true, 0, &none, 1, q_max) == q_max,
              "tail extent: P == 0 keeps q_max");
        // all-local ubatch -> number of local tokens (current segment IS bound)
        llama_pos loc[4] = { 0, 1, 2, 3 };
        check(llama_position_split::tail_extent_queries(true, P, loc, 4, q_max) == 4,
              "tail extent: local-only ubatch reports its token count");
        llama_pos loc2[256];
        for (int i = 0; i < 256; ++i) { loc2[i] = llama_pos(i); }
        check(llama_position_split::tail_extent_queries(true, P, loc2, 256, q_max) == 256,
              "tail extent: full local band reports P tokens");
        // overflow present -> history only (this is the case that asserted)
        llama_pos ovf[3] = { uint32_t(P), uint32_t(P) + 1, uint32_t(P) + 2 };
        check(llama_position_split::tail_extent_queries(true, P, ovf, 3, q_max) == 1,
              "tail extent: overflow ubatch falls back to the persistent tail");
        llama_pos mixed[2] = { llama_pos(P) - 1, llama_pos(P) };
        check(llama_position_split::tail_extent_queries(true, P, mixed, 2, q_max) == 1,
              "tail extent: a single overflow token makes the read history-only");
        // the pre-fix behaviour (always q_max) must NOT be reported for overflow
        check(llama_position_split::tail_extent_queries(true, P, ovf, 3, q_max) != q_max,
              "tail extent: overflow never reuses the whole-ubatch q_max");
    }

    // 6) P == 0 disables the split entirely (current behaviour).
    {
        std::vector<llama_ubatch> in;
        in.push_back(make_ubatch(300, 400, NKEEP));
        std::vector<llama_ubatch> out;
        std::string err;
        check(llama_position_split::divide_ubatches_at_p(in, 0, NKEEP, out, err) || !err.empty(),
              "P == 0 is a no-op or an explicit rejection");
    }

    // 7) the helper agrees with the plan contracts on the same inputs.
    {
        check(llama_position_split::ubatch_crosses_p(400, 300, P), "plan: [400,700) crosses 512");
        check(!llama_position_split::ubatch_crosses_p(400, 112, P), "plan: [400,512) does not cross");
        check(llama_position_split::validate_recurrent_window(509, 513, P).empty() == false,
              "plan: window [509,513) is rejected");
        check(llama_position_split::validate_recurrent_window(500, 505, P).empty(),
              "plan: window [500,505) is accepted");
        check_close(double(P % llama_position_split::kAlignTokens), 0.0, 0.0,
                    "plan: P is a multiple of 256");
    }

    std::printf("test-position-split-ubatch: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}