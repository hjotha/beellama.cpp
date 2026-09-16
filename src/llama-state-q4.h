#pragma once

#include "llama.h"
#include "llama-kv-cells.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct llama_hparams;

// Seekable byte source for a native sequence-state stream. The conversion path
// accepts both the canonical file form (magic/version/token header, used by the
// router hand-off) and the in-memory form produced by state_seq_get_data.
struct llama_state_q4_source {
    virtual ~llama_state_q4_source() = default;

    virtual uint64_t size() const = 0;
    virtual uint64_t tell() const = 0;
    virtual void seek(uint64_t offset) = 0;
    virtual void read_raw(void * dst, size_t size) = 0;
};

// Parsed q4_0/q4_0 KV body. Rows stay in the source; only offsets are kept.
struct llama_state_q4_info {
    struct layer {
        uint32_t il = 0;
        uint64_t k_row_size = 0;
        uint64_t v_row_size = 0;
        uint64_t k_data = 0; // absolute offset of token 0's K row
        uint64_t v_data = 0; // absolute offset of token 0's V row
    };

    uint32_t n_tokens = 0;
    bool from_ram = false;
    bool has_cell_ext = false;
    std::vector<llama_token> tokens;
    // Per-token cell extension when the model stores one (M-RoPE spatial
    // position or PLE token hash). Copied verbatim into the converted state.
    std::vector<llama_kv_cell_ext> exts;
    std::vector<layer> layers;
    // Trailing non-attention memory state (hybrid recurrent/conv state). The
    // bytes in [recr_offset, recr_offset + recr_bytes) are preserved verbatim;
    // zero means the source carries only attention state.
    uint64_t recr_offset = 0;
    uint64_t recr_bytes = 0;
};

// Parses and validates the attention portion of a q4_0/q4_0 sequence state,
// starting after the outer file/RAM header. Accepts only the canonical
// single-stream contiguous-prefix form: one stream, one cell range covering
// cells 0..n-1 in order, positions 0..n-1 owned by a single sequence, both K
// and V stored as untransposed q4_0 rows with the exact row sizes of the given
// attention layers. `attn_layers` is the attention cache's layer order, in the
// order the state stores them. On failure returns false and fills `error`. The
// caller must set `out.n_tokens` to the bounded count from the outer header
// before calling; this prevents malformed body metadata from becoming an
// unbounded allocation.
LLAMA_API bool llama_state_q4_parse(
        llama_state_q4_source & src,
        const llama_hparams & hparams,
        const std::vector<uint32_t> & attn_layers,
        bool has_cell_ext,
        llama_state_q4_info & out,
        std::string & error);
