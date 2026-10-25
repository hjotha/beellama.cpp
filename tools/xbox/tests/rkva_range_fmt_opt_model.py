#!/usr/bin/env python3
"""Host model of the optimized RKVA F16 / Q8_0 decode kernel.

Emulates shaders/rkva/rkva_range_attn_gqa_opt.hlsl (RKVA_OPT_STORAGE 0 and 2)
plus rkva_range_reduce_opt.hlsl with the real Qwen35 shape (24 Q heads, 4 KV
heads, D=256; F16 rows of 512 bytes and Q8_0 rows of 8 x 34 bytes per KV head)
and checks:

* the natural unit mapping (unit u = dims 4u..4u+3) covers every dim once for
  32-, 64- and 128-lane waves;
* the dword-assembled F16 / Q8_0 unit loads equal an independent per-element
  decoder (numpy structured views of the raw bytes) bit for bit, including
  signed scales, int8 -128/127, f16 subnormals/zero/max, and never read
  outside the (row, KV head) record;
* attention against a float64 oracle and an emulation of the live259 generic
  reference kernel for every optimized mode/wave size: split coverage, causal
  limit, empty splits/waves, LSE=-inf when masked, direct output (split_k 1)
  and the split-K reducer.

The attention arithmetic emulation is shared with tests/rkva_range_q4_opt_model.py
(the kernels share their softmax/merge code); only decode and lane mapping are
format specific. CPU only: no GPU run and no performance claim. Run:
    python3 tests/rkva_range_fmt_opt_model.py [--quick]
"""
import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rkva_range_q4_opt_model as m4  # noqa: E402

N_HEAD, N_HEAD_KV, D = m4.N_HEAD, m4.N_HEAD_KV, m4.D
UNITS = 64
ROW_HEAD = {0: 512, 2: 272}
NAMES = {0: "f16", 2: "q8_0"}


def independent_decode(storage, buf, rows):
    """[rows, kv, D] float32 straight from the ggml layout (no shader arithmetic)."""
    if storage == 0:
        return np.frombuffer(buf, "<f2").astype(np.float32).reshape(rows, N_HEAD_KV, D)
    blocks = np.frombuffer(buf, np.dtype([("d", "<f2"), ("qs", "i1", 32)]))
    vals = blocks["qs"].astype(np.float32) * blocks["d"].astype(np.float32)[:, None]
    return vals.reshape(rows, N_HEAD_KV, D)


class Reader:
    """Opt shader load_unit with explicit dword loads and per-record bounds."""

    def __init__(self, buf):
        self.words = np.frombuffer(buf + b"\0" * ((-len(buf)) % 4), "<u4")

    def load(self, addr, lo, hi):
        assert addr % 4 == 0 and lo <= addr and addr + 4 <= hi, (addr, lo, hi)
        return int(self.words[addr // 4])

    @staticmethod
    def f16(bits):
        return np.float32(np.frombuffer(struct.pack("<H", bits & 0xFFFF), np.float16)[0])

    def load_unit(self, storage, base, u):
        lo, hi = base, base + ROW_HEAD[storage]
        if storage == 0:
            x = self.load(base + u * 8, lo, hi)
            y = self.load(base + u * 8 + 4, lo, hi)
            return np.array([self.f16(x), self.f16(x >> 16), self.f16(y), self.f16(y >> 16)],
                            np.float32)
        block = base + (u >> 3) * 34
        scale = self.f16(self.load(block & ~3, lo, hi) >> ((block & 3) * 8))
        quant = block + 2 + (u & 7) * 4
        aligned = quant & ~3
        shift = (quant & 3) * 8
        w_lo = self.load(aligned, lo, hi)
        w_hi = self.load(aligned + 4 if shift else aligned, lo, hi)
        word = ((w_lo >> shift) | (w_hi << (32 - shift))) & 0xFFFFFFFF if shift else w_lo
        out = []
        for i in range(4):
            byte = (word >> (8 * i)) & 0xFF
            q = byte - 256 if byte >= 128 else byte  # asint(word << (24-8i)) >> 24
            out.append(np.float32(q) * scale)
        return np.array(out, np.float32)


def shader_decode(storage, buf, rows):
    reader = Reader(buf)
    table = np.zeros((rows, N_HEAD_KV, D), np.float32)
    for r in range(rows):
        for h in range(N_HEAD_KV):
            base = (r * N_HEAD_KV + h) * ROW_HEAD[storage]
            for u in range(UNITS):
                table[r, h, 4 * u : 4 * u + 4] = reader.load_unit(storage, base, u)
    return table


def make_rows(storage, rng, rows, edge=False):
    n = rows * N_HEAD_KV * D
    if storage == 0:
        vals = rng.normal(0, 0.5, n).astype(np.float16)
        if edge:
            special = np.array([0.0, -0.0, 6.0e-8, -6.0e-8, 65504.0, -65504.0, 1e-5, -2.0],
                               np.float16)
            vals[: special.size] = special
        return vals.tobytes()
    n_blocks = n // 32
    blocks = np.zeros(n_blocks, np.dtype([("d", "<f2"), ("qs", "i1", 32)]))
    blocks["d"] = (rng.uniform(0.002, 0.05, n_blocks) * rng.choice([-1, 1], n_blocks)).astype(
        np.float16)
    blocks["qs"] = rng.integers(-128, 128, (n_blocks, 32), dtype=np.int8)
    if edge:
        blocks["qs"][0, :4] = [-128, 127, -1, 0]
        blocks["d"][0] = np.float16(-0.0)
        blocks["d"][1] = np.float16(6.0e-8)
        blocks["d"][2] = np.float16(-65504.0)
    return blocks.tobytes()


def check_lane_mapping():
    for wave in (32, 64, 128):
        covered = []
        for lane in range(wave):
            if lane < UNITS:
                covered.append(lane)
            if wave < UNITS and lane + wave < UNITS:
                covered.append(lane + wave)
        assert sorted(covered) == list(range(UNITS)), wave
        dims = sorted(4 * u + i for u in covered for i in range(4))
        assert dims == list(range(D)), wave


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the 28672-row case")
    ap.add_argument("--seed", type=int, default=20261019)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    check_lane_mapping()

    for storage in (0, 2):
        for edge in (False, True):
            buf = make_rows(storage, rng, 3, edge)
            a = shader_decode(storage, buf, 3)
            b = independent_decode(storage, buf, 3)
            assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), NAMES[storage]
        print(f"{NAMES[storage]}: unit mapping + dword loads bit-exact vs independent decode")

    scale = 1.0 / 16.0
    origin = 102400
    cases = [
        (0, 0, (1,), (32,)),
        (1, 0, (1, 2, 3), (32, 64, 128)),
        (2, 1, (1,), (64,)),
        (64, 63, (1, 2), (32, 64)),
        (257, 256, (1, 2, 3), (32, 64, 128)),
        (257, 99, (1, 2), (32, 64)),  # causal limit inside the range
        (300, -5, (1,), (32, 64)),  # query before origin: all masked
        (4096, 4095, (1, 2, 3), (32, 64, 128)),
    ]
    if not args.quick:
        cases.append((28672, 28671, (1,), (64,)))
    worst = 0.0
    for storage in (0, 2):
        for active, rel_pos, modes, waves in cases:
            rows = max(active, 1)
            k = independent_decode(storage, make_rows(storage, rng, rows), rows)
            v = independent_decode(storage, make_rows(storage, rng, rows), rows)
            q = rng.normal(0, 2, (N_HEAD, D)).astype(np.float32)
            pos = origin + rel_pos
            causal = 0 if pos < origin else pos - origin + 1
            limit = min(active, causal)
            o64, l64 = m4.oracle(q, k, v, limit, scale)
            r_out, r_lse = m4.run_reference_exact(q, k, v, active, limit, scale)
            re_o, re_l = m4.compare("reference", r_out, r_lse, o64, l64)
            for mode in modes:
                for wave in waves:
                    split_k, out, lse = m4.run_optimized(q, k, v, active, limit, scale, mode,
                                                         wave)
                    e_o, e_l = m4.compare("opt", out, lse, o64, l64)
                    d_o, _ = m4.compare("opt-vs-ref", out, lse, r_out, r_lse)
                    ok = e_o <= max(1e-5, 4 * re_o) and e_l <= max(1e-4, 4 * re_l)
                    worst = max(worst, e_o)
                    print(f"{NAMES[storage]:5s} active={active:5d} limit={limit:5d} mode={mode} "
                          f"wave={wave:3d} split_k={split_k:3d} direct={int(split_k == 1)} "
                          f"|O-oracle|={e_o:.2e} (ref {re_o:.2e}) |LSE-oracle|={e_l:.2e} "
                          f"(ref {re_l:.2e}) |O-ref|={d_o:.2e} {'OK' if ok else 'FAIL'}")
                    if not ok:
                        return 1
    print(f"rkva_range_fmt_opt_model: ok (worst |O-oracle| {worst:.2e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
