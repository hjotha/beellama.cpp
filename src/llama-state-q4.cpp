#include "llama-state-q4.h"

#include "llama-hparams.h"
#include "llama.h"
#include "ggml.h"

#include <stdexcept>

namespace {

uint32_t read_u32(llama_state_q4_source & src) {
    uint32_t value = 0;
    src.read_raw(&value, sizeof(value));
    return value;
}

int32_t read_i32(llama_state_q4_source & src) {
    int32_t value = 0;
    src.read_raw(&value, sizeof(value));
    return value;
}

uint64_t read_u64(llama_state_q4_source & src) {
    uint64_t value = 0;
    src.read_raw(&value, sizeof(value));
    return value;
}

[[noreturn]] void fail(std::string & error, const std::string & message) {
    error = message;
    throw std::runtime_error(message);
}

uint64_t row_bytes(const llama_hparams & hparams, uint32_t il, bool value) {
    const uint32_t n_embd = value ? hparams.n_embd_v_gqa(il) : hparams.n_embd_k_gqa(il);
    return uint64_t(ggml_row_size(GGML_TYPE_Q4_0, n_embd));
}

} // namespace

bool llama_state_q4_parse(
        llama_state_q4_source & src,
        const llama_hparams & hparams,
        const std::vector<uint32_t> & attn_layers,
        bool has_cell_ext,
        llama_state_q4_info & out,
        std::string & error) {
    const uint32_t n_expected_tokens = out.n_tokens;
    out.layers.clear();
    try {
        const uint32_t n_stream = read_u32(src);
        if (n_stream != 1) {
            fail(error, "source state has more than one KV stream");
        }

        const uint32_t cell_count = read_u32(src);
        if (cell_count == 0) {
            fail(error, "source state holds no cells");
        }
        if (n_expected_tokens != 0 && cell_count != n_expected_tokens) {
            fail(error, "source state token count does not match its cells");
        }

        out.has_cell_ext = has_cell_ext;
        out.exts.resize(cell_count);
        uint32_t range_begin = cell_count;
        uint32_t range_end = 0;
        for (uint32_t i = 0; i < cell_count; ++i) {
            const llama_pos pos = read_i32(src);
            const uint32_t n_seq_id = read_u32(src);
            if (n_seq_id != 1) {
                fail(error, "source state cells hold multiple sequence owners");
            }
            if (out.has_cell_ext) {
                llama_kv_cell_ext ext;
                src.read_raw(&ext, sizeof(ext));
                out.exts[i] = ext;
            }
            read_i32(src); // sequence id: the destination sequence is chosen by the caller
            if (pos != llama_pos(i)) {
                fail(error, "source state is not a contiguous position prefix");
            }
            range_begin = std::min(range_begin, i);
            range_end = i + 1;
        }
        if (range_begin != 0 || range_end != cell_count) {
            fail(error, "source state cells are not a contiguous range");
        }

        const uint32_t v_trans = read_u32(src);
        if (v_trans != 0) {
            fail(error, "source state uses a transposed value layout");
        }
        const uint32_t n_layer = read_u32(src);

        if (n_layer != attn_layers.size()) {
            fail(error, "source state layer count does not match the attention cache");
        }

        out.layers.resize(n_layer);
        for (uint32_t i = 0; i < n_layer; ++i) {
            const int32_t k_type = read_i32(src);
            const uint64_t k_row = read_u64(src);
            const uint32_t il = attn_layers[i];
            const uint64_t expected = row_bytes(hparams, il, false);
            if (k_type != int32_t(GGML_TYPE_Q4_0) || k_row != expected) {
                fail(error, "source state key layout is not the expected q4_0 shape");
            }
            out.layers[i].il = il;
            out.layers[i].k_row_size = k_row;
            out.layers[i].k_data = src.tell();
            // A single contiguous cell range keeps every token row adjacent.
            src.seek(out.layers[i].k_data + k_row * cell_count);
        }

        for (uint32_t i = 0; i < n_layer; ++i) {
            const int32_t v_type = read_i32(src);
            const uint64_t v_row = read_u64(src);
            const uint32_t il = out.layers[i].il;
            const uint64_t expected = row_bytes(hparams, il, true);
            if (v_type != int32_t(GGML_TYPE_Q4_0) || v_row != expected) {
                fail(error, "source state value layout is not the expected q4_0 shape");
            }
            out.layers[i].v_row_size = v_row;
            out.layers[i].v_data = src.tell();
            src.seek(out.layers[i].v_data + v_row * cell_count);
        }

        if (out.n_tokens == 0) {
            out.n_tokens = cell_count;
        }
        return true;
    } catch (const std::exception &) {
        if (error.empty()) {
            error = "truncated or malformed sequence state";
        }
        out.layers.clear();
        out.exts.clear();
        out.n_tokens = 0;
        out.recr_offset = 0;
        out.recr_bytes = 0;
        return false;
    }
}
