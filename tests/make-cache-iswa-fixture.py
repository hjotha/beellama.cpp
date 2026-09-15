#!/usr/bin/env python3
"""Small deterministic Gemma2 GGUF for the prompt-checkpoint iSWA regression."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))
from gguf import GGUFWriter

writer = GGUFWriter(sys.argv[1], "gemma2")
writer.add_name("cache-iswa-regression")
writer.add_context_length(512)
writer.add_embedding_length(32)
writer.add_block_count(2)
writer.add_feed_forward_length(64)
writer.add_head_count(2)
writer.add_head_count_kv(1)
writer.add_rope_dimension_count(16)
writer.add_layer_norm_rms_eps(1e-6)
writer.add_sliding_window(8)
writer.add_sliding_window_pattern(2)
writer.add_tokenizer_model("no_vocab")
writer.add_vocab_size(64)
rng = np.random.default_rng(7361)


def tensor(name, shape):
    values = rng.normal(0, 0.08, shape).astype(np.float32)
    writer.add_tensor(name + ".weight", values)


tensor("token_embd", (64, 32))
tensor("output_norm", (32,))
for layer in range(2):
    for name in ("attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm"):
        tensor(f"blk.{layer}.{name}", (32,))
    for name, shape in {
        "attn_q": (32, 32), "attn_k": (16, 32), "attn_v": (16, 32),
        "attn_output": (32, 32), "ffn_gate": (64, 32),
        "ffn_up": (64, 32), "ffn_down": (32, 64),
    }.items():
        tensor(f"blk.{layer}.{name}", shape)
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
