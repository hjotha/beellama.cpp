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
    llama_ubatch ub;
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
    data->seq_id_unq = { 0 };
    data->seq_idx.assign(LLAMA_MAX_SEQ, -1);
    data->seq_idx[0] = 0;
    data->seq_id_data = { 0 };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        data->token[i] = llama_token(1000 + i);
        data->pos[i]   = pos0 + llama_pos(i);
        // The trailing n_keep_tail tokens are the protected window.
        data->output[i] = (n_keep_tail > 0 && i + n_keep_tail >= n_tokens) ? 1 : 0;
        data->seq_id[i] = data->seq_id_data.data();
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
        }
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