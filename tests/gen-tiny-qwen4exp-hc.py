#!/usr/bin/env python3
"""Generate a tiny Qwen4Exp (hyper-connections) target and a matching DFlash2 drafter.

The target mirrors the Qwen3.8-Flash-Next topology at toy width: two layers,
`hyper_connection.count = 4`, layer 0 is linear attention with PLE, layer 1 is
full attention with the QSA indexer, and both layers carry the hyper-connection
mixes. Every weight of *layer 1* is zero, so its attention/FFN blocks contribute
exactly zero and the residual streams written by layer 0 reach `h_nextn`
unchanged. That makes the HC mean of the exposed h_nextn rows an independent
oracle for `llama_get_embeddings_layer_inp(ctx, 1)`: the runtime tap must equal
the mean over the four streams, computed by the test in double precision.

The drafter is a plain DFlash2 (arch `dflash`): block_size 8, conv 2/16,
selector rank 8 / top_k 4, `target_layers = [1]` (the runtime taps a layer's
INPUT, matching the converter's HF id + 1 convention), and the 128-token
synthetic `test` vocab shared with the target. It deliberately has no token
embeddings or lm head: DFlash borrows both from the target through `ctx_other`.

Usage:
    python3 tests/gen-tiny-qwen4exp-hc.py <out_dir> [--d2d]

Writes <out_dir>/qwen4exp-hc-target.gguf and <out_dir>/dflash2-draft.gguf.
With --d2d, also writes a drafter that taps layers [1, 0] for CUDA coverage.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))
import gguf  # noqa: E402

N_VOCAB = 128
N_EMBD = 256
HC = 4
HC_LR = 8
N_LAYER = 2
N_FF = 384
N_FF_SHEXP = 192
N_EXPERT = 2

SSM_STATE = 128
SSM_GROUPS = 2
SSM_DT = 2
SSM_CONV = 4

IDX_HEADS = 64
IDX_DIM = 128
IDX_KPOOL = 4
IDX_TOP_K = 64

PLE_HEADS = 4
PLE_HEAD_DIM = 64
PLE_TABLE_ROWS = 512

DRAFT_N_LAYER = 2
DRAFT_FF = 384
DRAFT_BLOCK = 8
DRAFT_CONV = 2
DRAFT_CONV_GROUP = 16
DRAFT_RANK = 8
DRAFT_TOP_K = 4


def add_vocab(writer: gguf.GGUFWriter) -> None:
    """Synthetic 128-token vocab; `test` is the loader's fixture tokenizer."""
    tokens = [f"tok_{i}" for i in range(N_VOCAB)]
    writer.add_tokenizer_model("test")
    writer.add_token_list(tokens)
    writer.add_token_types([gguf.TokenType.NORMAL] * N_VOCAB)
    writer.add_token_scores([0.0] * N_VOCAB)
    writer.add_bos_token_id(0)
    writer.add_eos_token_id(1)
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)


def gen_target(path: Path) -> None:
    w = gguf.GGUFWriter(path, "qwen4exp")
    w.add_name("tiny-qwen4exp-hc")
    add_vocab(w)

    w.add_context_length(256)
    w.add_embedding_length(N_EMBD)
    w.add_embedding_length_out(HC * N_EMBD)
    w.add_block_count(N_LAYER)
    w.add_feed_forward_length(N_FF)
    w.add_expert_feed_forward_length(N_FF)
    w.add_expert_shared_feed_forward_length(N_FF_SHEXP)
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_EXPERT)

    head_dim = 128
    w.add_head_count(2)
    w.add_head_count_kv(2)
    w.add_key_length(head_dim)
    w.add_value_length(head_dim)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_causal_attention(True)

    # layer 0 linear (PLE), layer 1 full attention with the QSA indexer
    w.add_recurrent_layers([True, False])
    w.add_attention_compress_ratios([4, 4])
    w.add_indexer_head_count(IDX_HEADS)
    w.add_indexer_key_length(IDX_DIM)
    w.add_indexer_top_k(IDX_TOP_K)

    w.add_hyper_connection_count(HC)
    w.add_hyper_connection_low_rank(HC_LR)

    # PLE on the first layer, exactly like Qwen3.8-Flash-Next's ple_layer_ids
    w.add_ple_layers([0])
    w.add_ple_ngram_size(3)
    w.add_ple_heads_per_ngram(2)
    w.add_ple_conv_kernel(4)
    w.add_ple_eos_token_id(0)
    w.add_embedding_length_per_layer_input(PLE_HEAD_DIM)
    w.add_ple_layer_multipliers([1, 3, 5])
    w.add_ple_head_offsets([0, 128, 256, 384])
    w.add_ple_head_vocab_sizes([128, 128, 128, 128])

    w.add_rope_dimension_count(128)
    w.add_rope_dimension_sections([32, 32, 32, 32])
    w.add_rope_freq_base(10000.0)

    w.add_ssm_conv_kernel(SSM_CONV)
    w.add_ssm_inner_size(N_EMBD)
    w.add_ssm_state_size(SSM_STATE)
    w.add_ssm_time_step_rank(SSM_DT)
    w.add_ssm_group_count(SSM_GROUPS)
    w.add_nextn_predict_layers(0)

    rng = np.random.default_rng(20261121)

    def t(name: str, shape: tuple[int, ...], *, zero: bool = False, gamma: bool = False) -> None:
        a = np.zeros(shape, dtype=np.float32) if zero else (
            (1.0 + 0.01 * rng.standard_normal(shape)).astype(np.float32) if gamma else
            (0.05 * rng.standard_normal(shape)).astype(np.float32))
        w.add_tensor(name, a)

    # global tensors
    t("token_embd.weight", (N_VOCAB, N_EMBD))
    t("output.weight", (N_VOCAB, N_EMBD))
    t("per_layer_token_embd.weight", (PLE_TABLE_ROWS, PLE_HEAD_DIM))
    t("output_hc_norm.weight", (HC, N_EMBD), gamma=True)
    t("output_hc_down.weight", (HC_LR, HC * N_EMBD))
    t("output_hc_up.weight", (HC * N_EMBD, HC_LR))

    hc_dim = HC * N_EMBD

    def gen_moe(il: int, zero: bool) -> None:
        t(f"blk.{il}.ffn_gate_inp.weight", (N_EXPERT, N_EMBD), zero=zero)
        t(f"blk.{il}.ffn_down_exps.weight", (N_EXPERT, N_EMBD, N_FF), zero=zero)
        t(f"blk.{il}.ffn_gate_up_exps.weight", (N_EXPERT, 2 * N_FF, N_EMBD), zero=zero)
        t(f"blk.{il}.ffn_gate_inp_shexp.weight", (N_EMBD,), zero=zero)
        t(f"blk.{il}.ffn_gate_shexp.weight", (N_FF_SHEXP, N_EMBD), zero=zero)
        t(f"blk.{il}.ffn_up_shexp.weight", (N_FF_SHEXP, N_EMBD), zero=zero)
        t(f"blk.{il}.ffn_down_shexp.weight", (N_EMBD, N_FF_SHEXP), zero=zero)

    def gen_hc(il: int, zero: bool) -> None:
        t(f"blk.{il}.hc_attn_norm.weight", (HC, N_EMBD), zero=zero, gamma=True)
        t(f"blk.{il}.hc_attn_down.weight", (HC_LR, hc_dim), zero=zero)
        t(f"blk.{il}.hc_attn_up.weight", (hc_dim, HC_LR), zero=zero)
        t(f"blk.{il}.hc_attn_inject.weight", (HC, hc_dim), zero=zero)
        t(f"blk.{il}.hc_ffn_norm.weight", (HC, N_EMBD), zero=zero, gamma=True)
        t(f"blk.{il}.hc_ffn_down.weight", (HC_LR, hc_dim), zero=zero)
        t(f"blk.{il}.hc_ffn_up.weight", (hc_dim, HC_LR), zero=zero)
        t(f"blk.{il}.hc_ffn_inject.weight", (HC, hc_dim), zero=zero)

    # ---- layer 0: linear attention (GDN) + PLE, random ----
    key_dim = SSM_STATE * SSM_GROUPS
    value_dim = SSM_STATE * SSM_DT
    conv_dim = key_dim * 2 + value_dim
    t("blk.0.ssm_norm.weight", (SSM_STATE,), gamma=True)
    t("blk.0.attn_qkv.weight", (key_dim * 2 + value_dim, N_EMBD))
    t("blk.0.attn_gate.weight", (value_dim, N_EMBD))
    t("blk.0.ssm_conv1d.weight", (conv_dim, SSM_CONV))
    t("blk.0.ssm_dt.bias", (SSM_DT,))
    # negative real A keeps the scan stable for a short toy prompt
    w.add_tensor("blk.0.ssm_a", -np.exp(np.abs(rng.standard_normal(SSM_DT).astype(np.float32))))
    t("blk.0.ssm_alpha.weight", (SSM_DT, N_EMBD))
    t("blk.0.ssm_beta.weight", (SSM_DT, N_EMBD))
    t("blk.0.ssm_out.weight", (N_EMBD, value_dim))
    t("blk.0.ple_key.weight", (hc_dim, N_EMBD))
    t("blk.0.ple_value.weight", (N_EMBD, N_EMBD))
    t("blk.0.ple_norm_key.weight", (HC, N_EMBD), gamma=True)
    t("blk.0.ple_norm_query.weight", (HC, N_EMBD), gamma=True)
    t("blk.0.ple_norm_conv.weight", (HC, N_EMBD), gamma=True)
    t("blk.0.ple_conv1d.weight", (hc_dim, SSM_CONV))
    gen_hc(0, False)
    gen_moe(0, False)

    # ---- layer 1: full attention, every weight zero so the block is a no-op ----
    t("blk.1.attn_q.weight", (head_dim * 2 * 2, N_EMBD), zero=True)
    t("blk.1.attn_k.weight", (SSM_STATE * SSM_GROUPS, N_EMBD), zero=True)
    t("blk.1.attn_v.weight", (SSM_STATE * SSM_GROUPS, N_EMBD), zero=True)
    t("blk.1.attn_output.weight", (N_EMBD, head_dim * 2), zero=True)
    t("blk.1.attn_q_norm.weight", (head_dim,), zero=True)
    t("blk.1.attn_k_norm.weight", (head_dim,), zero=True)
    t("blk.1.indexer.q_proj.weight", (IDX_HEADS * IDX_DIM, N_EMBD), zero=True)
    t("blk.1.indexer.k_proj.weight", (IDX_DIM, N_EMBD), zero=True)
    t("blk.1.indexer.q_norm.weight", (IDX_DIM,), zero=True)
    t("blk.1.indexer.k_norm.weight", (IDX_DIM,), zero=True)
    gen_hc(1, True)
    gen_moe(1, True)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {path} (target: n_embd={N_EMBD}, hc={HC}, n_layer={N_LAYER})")


def gen_draft(path: Path, target_layers: tuple[int, ...] = (1,)) -> None:
    w = gguf.GGUFWriter(path, "dflash")
    w.add_name("tiny-dflash2-qwen4exp")
    add_vocab(w)
    # DFlash2 needs the target's mask token; the target itself does not use it
    w.add_mask_token_id(N_VOCAB - 1)

    w.add_context_length(256)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(DRAFT_FF)
    w.add_block_count(DRAFT_N_LAYER)
    head_dim = 128
    w.add_head_count(2)
    w.add_head_count_kv(2)
    w.add_key_length(head_dim)
    w.add_value_length(head_dim)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_causal_attention(False)
    w.add_rope_freq_base(10000.0)
    # M-RoPE draft: one rotary section, matching Qwen4Exp's M-RoPE positions
    w.add_rope_dimension_count(64)
    w.add_rope_dimension_sections([64, 0, 0, 0])

    w.add_block_size(DRAFT_BLOCK)
    w.add_conv_kernel_size(DRAFT_CONV)
    w.add_conv_group_size(DRAFT_CONV_GROUP)
    w.add_selector_rank(DRAFT_RANK)
    w.add_selector_top_k(DRAFT_TOP_K)
    # the runtime taps layer inputs, so index 1 captures layer 0's completed output
    w.add_target_layers(list(target_layers))

    rng = np.random.default_rng(20261122)

    def t(name: str, shape: tuple[int, ...], scale: float = 0.05) -> None:
        w.add_tensor(name, (scale * rng.standard_normal(shape)).astype(np.float32))

    n_feat = len(target_layers)
    t("fc.weight", (N_EMBD, n_feat * N_EMBD))
    t("enc.output_norm.weight", (N_EMBD,))
    t("output_norm.weight", (N_EMBD,))
    t("selector_hidden.weight", (DRAFT_RANK, N_EMBD))
    t("selector_predecessor.weight", (N_VOCAB, DRAFT_RANK))
    t("selector_successor.weight", (N_VOCAB, DRAFT_RANK))

    for i in range(DRAFT_N_LAYER):
        t(f"blk.{i}.attn_norm.weight", (N_EMBD,))
        t(f"blk.{i}.attn_q.weight", (head_dim * 2, N_EMBD))
        t(f"blk.{i}.attn_k.weight", (head_dim * 2, N_EMBD))
        t(f"blk.{i}.attn_v.weight", (head_dim * 2, N_EMBD))
        t(f"blk.{i}.attn_output.weight", (N_EMBD, head_dim * 2))
        t(f"blk.{i}.attn_q_norm.weight", (head_dim,))
        t(f"blk.{i}.attn_k_norm.weight", (head_dim,))
        t(f"blk.{i}.ffn_norm.weight", (N_EMBD,))
        t(f"blk.{i}.ffn_gate.weight", (DRAFT_FF, N_EMBD))
        t(f"blk.{i}.ffn_up.weight", (DRAFT_FF, N_EMBD))
        t(f"blk.{i}.ffn_down.weight", (N_EMBD, DRAFT_FF))
        t(f"blk.{i}.attn_conv_base", (2, DRAFT_CONV, N_EMBD))
        t(f"blk.{i}.attn_conv_proj.weight", (2 * DRAFT_CONV * (N_EMBD // DRAFT_CONV_GROUP), N_EMBD))
        t(f"blk.{i}.ffn_conv_base", (2, DRAFT_CONV, N_EMBD))
        t(f"blk.{i}.ffn_conv_proj.weight", (2 * DRAFT_CONV * (N_EMBD // DRAFT_CONV_GROUP), N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {path} (draft: n_embd={N_EMBD}, n_layer={DRAFT_N_LAYER}, "
          f"block={DRAFT_BLOCK}, target_layers={list(target_layers)})")


def main() -> None:
    if len(sys.argv) != 2 and sys.argv[2:] != ["--d2d"]:
        raise SystemExit(f"usage: {sys.argv[0]} <out_dir> [--d2d]")
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    gen_target(out / "qwen4exp-hc-target.gguf")
    gen_draft(out / "dflash2-draft.gguf")
    if len(sys.argv) == 3:
        gen_draft(out / "dflash2-d2d-draft.gguf", (1, 0))


if __name__ == "__main__":
    main()
