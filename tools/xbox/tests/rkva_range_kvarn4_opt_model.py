#!/usr/bin/env python3
"""Host model of the optimized native Bee KVarN4 range-attention kernel.

Three independent views of the same descriptor-v1 buffers (8960-byte KVarN4
records per 128-token group and physical head, rotated F16 stage slots, and
the optional original-domain F16 precision tail) with the real Qwen35 shape
(24 Q heads, 4 KV heads, D=256):

1. oracle: decodes records/stage/tail with numpy reshapes of the raw bytes,
   maps every body row to the original domain with the explicit orthonormal
   Sylvester Hadamard matrix H256/16 in float64 (Bee WHT256 == H2 (x) H128 /
   16) and runs plain float64 softmax attention with the original Q. This
   materialization exists only in this test, never in a kernel.
2. reference: float32 emulation of shaders/rkva/rkva_range_attn_kvarn4.hlsl
   (per-element load_native/load_tail address arithmetic, per-head row loop,
   live259 split policy, butterfly WHT, generic reducer).
3. optimized: float32 emulation of shaders/rkva/rkva_range_attn_kvarn4_opt.hlsl
   + rkva_range_reduce_opt.hlsl (GQA group per KV head, 8-aligned split rows,
   per-row path for stage/tail/edges, 8-token record tiles, wave-owned rows for
   32/64/128-lane waves, two-phase body/tail merge, rotated-domain partials and post-reduction WHT,
   direct output in the original domain for split_k == 1).

Every emulated load is a dword (or Load2/Load4) read that must stay inside the
region the descriptor allows (the record of that group/head, the stage slot,
the tail); unused buffer bytes are poisoned with 0xFF (F16 NaN). Decoded K/V
of both emulations must equal the oracle bit for bit before any attention
math. With --bee-fixtures (default: the range260 fixture directory when it
exists) the genuine Bee llama-kvarn.cpp record, dequant and WHT fixtures are
checked and the genuine records are placed in the attention buffers.

CPU only: no GPU run and no performance claim. Run:
    python3 tests/rkva_range_kvarn4_opt_model.py [--quick]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rkva_range_q4_opt_model as m4  # noqa: E402

N_HEAD, N_HEAD_KV, D = 24, 4, 256
GQA = N_HEAD // N_HEAD_KV
PHYS = N_HEAD_KV * 2
RECORD = 8960
ROW_BYTES = N_HEAD_KV * 256 * 2  # stage/tail bytes per token
STAGE_BYTES = 384 * ROW_BYTES
FLT_MAX = np.float32(3.402823466e38)
U = np.arange(64)
DEFAULT_BEE = Path(__file__).resolve().parents[3] / "build" / "xllama-kvarn4-range260-fixtures"
F32 = np.float32


# ---------------------------------------------------------------- policies ---


def kvarn4_split(active, mode):
    """Opt split policy: Q4 split count, rows rounded up to whole 8-token tiles."""
    k0 = m4.optimized_split_k(active, mode)
    rows = (active + k0 - 1) // k0
    rows = (rows + 7) & ~7
    k = (active + rows - 1) // rows if rows else 1
    return max(1, k), rows


def kvarn4_decode_split(active, mode):
    """Opt-in decode split policy: round rows up to whole 128-token record groups.
    Mode 1 (balanced): target_rows=512, max_splits=32.
    Mode 2 (fine):     target_rows=256, max_splits=64.
    Mode 3 (coarse):   keeps legacy kvarn4_split.
    """
    if active == 0:
        return 1, 0
    if mode == 3:
        return kvarn4_split(active, mode)
    target_rows = 256 if mode == 2 else 512
    max_splits = 64 if mode == 2 else 32
    wanted = (active + target_rows - 1) // target_rows
    k0 = max(1, min(max_splits, wanted))
    rows = (active + k0 - 1) // k0
    rows = (rows + 127) & ~127
    k = (active + rows - 1) // rows if rows else 1
    return max(1, k), rows


def kvarn4_mq_split(active, mode, q_tile, mq2_cap32=False):
    """Multi-query tile split policy for native KVarN4 (nq 2..9, q_tile 2 or 4):
    For q_tile == 2 or 4 in mode 1 (balanced) and mode 2 (fine):
      target_rows = 256 (2 record groups of 128 tokens),
      max_splits = 64 (mode fine) or 32 (mode balanced).
    Mode 3 (coarse) or invalid q_tile keeps kvarn4_split.
    rows is rounded up to whole kKvarnTile (8), and active == 0 returns 1, 0.
    """
    if active == 0:
        return 1, 0
    if q_tile not in (2, 4) or mode not in (1, 2):
        return kvarn4_split(active, mode)
    target_rows = 256
    max_splits = 64 if mode == 2 and not (mq2_cap32 and q_tile == 2) else 32
    wanted = (active + target_rows - 1) // target_rows
    k0 = max(1, min(max_splits, wanted))
    rows = (active + k0 - 1) // k0
    rows = (rows + 7) & ~7
    k = (active + rows - 1) // rows if rows else 1
    return max(1, k), rows


def reference_split(active):
    k = m4.reference_split_k(active)
    return k, (active + k - 1) // k


class Layout:
    """ring: physical ORIGINAL-F16 tail rows (128 for revisions 1/2, 256 for the
    revision-3 serial-prefix descriptor); the logical window stays ptt."""

    def __init__(self, origin, capacity, ptt, ring=128):
        assert origin % 128 == 0 and ptt in (0, 128) and ring in (128, 256)
        self.origin, self.capacity, self.ptt, self.ring = origin, capacity, ptt, ring
        self.groups = (capacity + 127) // 128
        self.records_bytes = self.groups * PHYS * RECORD
        self.tail_rows = ring if ptt else 0
        self.half = self.records_bytes + STAGE_BYTES + self.tail_rows * ROW_BYTES


# ------------------------------------------------------------------ buffers ---


def random_record(rng):
    nib = rng.integers(0, 256, 8192, dtype=np.uint8).tobytes()
    scale = (rng.uniform(0.01, 0.12, 128) * rng.choice([-1, 1], 128)).astype("<f2")
    zero = rng.uniform(-0.5, 0.5, 128).astype("<f2")
    other = (rng.uniform(0.5, 1.5, 128) * rng.choice([-1, 1], 128)).astype("<f2")
    return nib + scale.tobytes() + zero.tobytes() + other.tobytes()


def build_plane(rng, lay, active, bee_record=None):
    """One K or V plane. Only descriptor-live regions hold data; the rest is NaN."""
    buf = bytearray(b"\xff" * lay.half)
    for group in range(active // 128):  # eager records, including relative group 0
        for ph in range(PHYS):
            rec = bee_record if (bee_record is not None and ph % 3 == 0) else random_record(rng)
            off = (group * PHYS + ph) * RECORD
            buf[off : off + RECORD] = rec
    live_group = (active - 1) // 128
    staged_groups = set()
    if lay.origin == 0:
        staged_groups.add(0)
    if active % 128:
        staged_groups.add(live_group)
    for group in staged_groups:
        gg = lay.origin // 128 + group
        slot = 0 if gg == 0 else 1 + ((gg - 1) & 1)
        rows = 128 if group < live_group or active % 128 == 0 else active % 128
        data = rng.normal(0, 0.5, rows * N_HEAD_KV * 256).astype("<f2").tobytes()
        off = lay.records_bytes + slot * 128 * ROW_BYTES
        buf[off : off + len(data)] = data
    if lay.ptt:
        for row in range(active - min(active, lay.ptt), active):
            data = rng.normal(0, 0.5, N_HEAD_KV * 256).astype("<f2").tobytes()
            off = lay.records_bytes + STAGE_BYTES + (row % lay.ring) * ROW_BYTES
            buf[off : off + ROW_BYTES] = data
    return bytes(buf)


# ------------------------------------------------------------------- oracle ---


def decode_record(blob, value):
    """Independent record decode: [token, dim] float32 (K transposed to tokens)."""
    q = np.frombuffer(blob[:8192], np.uint8)
    quants = np.empty(16384, np.float32)
    quants[0::2], quants[1::2] = q & 15, q >> 4
    axes = np.frombuffer(blob[8192:], "<f2").astype(np.float32).reshape(3, 128)
    m = quants.reshape(128, 128)  # [row, col]; K: row=dim col=tok; V: row=tok col=dim
    with np.errstate(over="ignore", invalid="ignore"):
        x = (m * axes[0][:, None] + axes[1][:, None]).astype(np.float32) * axes[2][None, :]
    return x if value else x.T


def hadamard256():
    h = np.array([[1.0]])
    for _ in range(8):
        h = np.block([[h, h], [h, -h]])
    return h / 16.0


H256 = hadamard256()


def oracle_sources(lay, plane, active, value):
    """[active, kv, 256] float32 values in their storage domain + exact (tail) mask."""
    out = np.zeros((active, N_HEAD_KV, 256), np.float32)
    live_group = (active - 1) // 128
    tail_begin = active - min(active, lay.ptt)
    stage = np.frombuffer(plane[lay.records_bytes : lay.records_bytes + STAGE_BYTES], "<f2")
    stage = stage.astype(np.float32).reshape(384, N_HEAD_KV, 256)
    tail = None
    if lay.ptt:
        t0 = lay.records_bytes + STAGE_BYTES
        tail = np.frombuffer(plane[t0 : t0 + lay.ring * ROW_BYTES], "<f2").astype(np.float32)
        tail = tail.reshape(lay.ring, N_HEAD_KV, 256)
    decoded = {}
    for row in range(active):
        group, pos = divmod(row, 128)
        gg = lay.origin // 128 + group
        if row >= tail_begin:
            out[row] = tail[row % lay.ring]
        elif gg == 0 or (group == live_group and active % 128):
            slot = 0 if gg == 0 else 1 + ((gg - 1) % 2)
            out[row] = stage[slot * 128 + pos]
        else:
            for kv in range(N_HEAD_KV):
                for s in range(2):
                    key = (group, kv * 2 + s)
                    if key not in decoded:
                        off = (group * PHYS + kv * 2 + s) * RECORD
                        decoded[key] = decode_record(plane[off : off + RECORD], value)
                    out[row, kv, s * 128 : (s + 1) * 128] = decoded[key][pos]
    exact = np.arange(active) >= tail_begin
    return out, exact


def oracle_attention(q, k_src, v_src, exact, limit, scale):
    """float64 attention in the original domain (body rows mapped through H256)."""
    out = np.zeros((N_HEAD, D))
    lse = np.full(N_HEAD, -np.inf)
    if limit == 0:
        return out, lse
    k = k_src[:limit].astype(np.float64)
    v = v_src[:limit].astype(np.float64)
    body = ~exact[:limit]
    k[body] = k[body] @ H256  # H256 symmetric: row @ H == H @ row
    v[body] = v[body] @ H256
    for h in range(N_HEAD):
        kv = h // GQA
        s = (k[:, kv] @ q[h].astype(np.float64)) * scale
        mx = s.max()
        w = np.exp(s - mx)
        out[h] = (w @ v[:, kv]) / w.sum()
        lse[h] = mx + np.log(w.sum())
    return out, lse


# -------------------------------------------------------- shader primitives ---


def butterfly_wht(x):
    """Reference shader wht256 in float32 lane order; x [..., 256]."""
    t = np.array(x, np.float32).reshape(-1, 256).copy()
    lane = np.arange(128)
    for stride in (1, 2, 4, 8, 16, 32, 64):
        own0, other0 = t[:, lane], t[:, lane ^ stride]
        own1, other1 = t[:, lane + 128], t[:, (lane ^ stride) + 128]
        upper = (lane & stride) != 0
        t[:, lane] = np.where(upper, other0 - own0, own0 + other0)
        t[:, lane + 128] = np.where(upper, other1 - own1, own1 + other1)
    x0 = t[:, :128] * F32(0.08838834764831845)
    y0 = t[:, 128:] * F32(0.08838834764831845)
    out = np.concatenate([(x0 + y0) * F32(0.7071067811865475),
                          (x0 - y0) * F32(0.7071067811865475)], axis=1)
    return out.astype(np.float32).reshape(np.shape(x))


def merge_partial_states(partials):
    """Finite-state wave/split merge; only exactly zero denominator is empty.

    Validation precedes weighting, so even a split whose exponential weight
    underflows to zero cannot hide malformed participating metadata/numerators.
    """
    heads, width = partials[0][2].shape
    maximum = np.full(heads, -FLT_MAX, np.float32)
    total = np.zeros(heads, np.float32)
    numerator = np.zeros((heads, width), np.float32)
    bad = np.zeros(heads, bool)
    for mx, den, acc in partials:
        present = den != 0
        bad |= present & (~np.isfinite(den) | (den < 0) | ~np.isfinite(mx) |
                          ~np.all(np.isfinite(acc), axis=1))
        maximum = np.where(present & ~bad, np.maximum(maximum, mx), maximum)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        for mx, den, acc in partials:
            present = (den != 0) & ~bad
            weight = np.where(present, np.exp(mx - maximum), F32(0)).astype(np.float32)
            total = (total + np.where(present, den * weight, F32(0))).astype(np.float32)
            numerator = (numerator + np.where(present[:, None], acc * weight[:, None],
                                             F32(0))).astype(np.float32)
    bad |= ~np.isfinite(total) | (total < 0) | ~np.all(np.isfinite(numerator), axis=1)
    maximum[bad] = total[bad] = F32(np.nan)
    numerator[bad] = F32(np.nan)
    return maximum, total, numerator


def reduce_partials(partials, rotated=False):
    """Current optimized reducer, including R=H/16 after the split merge."""
    mx, den, acc = merge_partial_states(partials)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inv = np.where(den == 0, F32(0), F32(1) / den).astype(np.float32)
        out = (acc * inv[:, None]).astype(np.float32)
        lse = np.where(den == 0, F32(-np.inf), mx + np.log(den)).astype(np.float32)
        if rotated:
            out = butterfly_wht(out)
    bad = ((den != 0) & ~np.isfinite(lse)) | ~np.all(np.isfinite(out), axis=1)
    out[bad], lse[bad] = F32(np.nan), F32(np.nan)
    return out, lse


def f16(bits):
    return (np.asarray(bits, np.uint64) & 0xFFFF).astype(np.uint16).view(np.float16).astype(
        np.float32)


def decode_math(quant, scale, zero, other):
    # precise affine: float(q) * scale rounded, + zero rounded, then * other.
    with np.errstate(over="ignore", invalid="ignore"):
        affine = (np.asarray(quant, np.float32) * scale).astype(np.float32) + zero
        return (affine.astype(np.float32) * other).astype(np.float32)


class Plane:
    """Dword reads with region bounds: ('rec', group, head) / ('stage', slot) / ('tail',).

    window: the image the kernel is allowed to see (active_count for revisions
    1/2, the query's own prefix q_count for revision 3); a record of a group
    sealed after the window is an out-of-image read."""

    def __init__(self, buf, lay, active):
        self.w = np.frombuffer(buf, "<u4")
        self.lay, self.active, self.window = lay, active, active
        self.reads = 0

    def region(self, kind, a=0, b=0):
        lay = self.lay
        if kind == "rec":
            assert a < self.window // 128, "record group not covered by the image"
            lo = (a * PHYS + b) * RECORD
            return lo, lo + RECORD
        if kind == "stage":
            lo = lay.records_bytes + a * 128 * ROW_BYTES
            return lo, lo + 128 * ROW_BYTES
        lo = lay.records_bytes + STAGE_BYTES
        assert lay.ptt, "tail read without a precision tail"
        return lo, lo + lay.ring * ROW_BYTES

    def load(self, addr, lo, hi):
        addr = np.asarray(addr, np.int64)
        lo = np.asarray(lo, np.int64)
        hi = np.asarray(hi, np.int64)
        assert np.all(addr % 4 == 0) and np.all(addr >= lo) and np.all(addr + 4 <= hi), (
            "out-of-region read")
        self.reads += addr.size
        return self.w[addr // 4].astype(np.uint64)

    def half_at(self, addr, lo, hi):
        addr = np.asarray(addr, np.int64)
        return f16(self.load(addr & ~3, lo, hi) >> ((addr & 2) * 8).astype(np.uint64))


# Reference kernel per-element loads (rkva_range_attn_kvarn4.hlsl), lanes 0..127.
def ref_row(P, value, row, kv, active):
    lay = P.lay
    lane = np.arange(128)
    group, pos = row // 128, row & 127
    gg = lay.origin // 128 + group
    live_group = (active - 1) // 128 if active else 0
    out = np.zeros(256, np.float32)
    staged = gg == 0 or (group == live_group and (active & 127) != 0)
    for s in range(2):
        ph = kv * 2 + s
        if staged:
            slot = 0 if gg == 0 else 1 + ((gg - 1) & 1)
            element = ((slot * 128 + pos) * PHYS + ph) * 128 + lane
            lo, hi = P.region("stage", slot)
            out[s * 128 : (s + 1) * 128] = P.half_at(lay.records_bytes + element * 2, lo, hi)
            continue
        base = (group * PHYS + ph) * RECORD
        lo, hi = P.region("rec", group, ph)
        r, c = (pos, lane) if value else (lane, pos)
        index = r * 128 + c
        packed = P.load(base + (index // 8) * 4, lo, hi)
        quant = (packed >> ((index & 7) * 4).astype(np.uint64)) & 15
        scale = P.half_at(base + 8192 + np.asarray(r) * 2 + 0 * lane, lo, hi)
        zero = P.half_at(base + 8448 + np.asarray(r) * 2 + 0 * lane, lo, hi)
        other = P.half_at(base + 8704 + np.asarray(c) * 2 + 0 * lane, lo, hi)
        out[s * 128 : (s + 1) * 128] = decode_math(quant, scale, zero, other)
    return out


def ref_tail(P, row, kv):
    lay = P.lay
    lo, hi = P.region("tail")
    element = ((row % lay.ring) * N_HEAD_KV + kv) * 256 + np.arange(256)
    return P.half_at(lay.records_bytes + STAGE_BYTES + element * 2, lo, hi)


# Optimized kernel loads (rkva_range_attn_kvarn4_opt.hlsl), units 0..63.
def halves_from(words):
    """[..., n] dwords -> [..., 2n] halves (low first)."""
    words = np.asarray(words, np.uint64)
    return np.stack([f16(words), f16(words >> 16)], axis=-1).reshape(*words.shape[:-1], -1)


def opt_tail4(P, row, kv):
    lo, hi = P.region("tail")
    element = ((row % P.lay.ring) * N_HEAD_KV + kv) * 256 + U * 4
    a = P.lay.records_bytes + STAGE_BYTES + element * 2
    words = np.stack([P.load(a, lo, hi), P.load(a + 4, lo, hi)], -1)
    return halves_from(words).reshape(256)


def opt_body4(P, value, row, kv, active):
    lay = P.lay
    group, pos = row // 128, row & 127
    gg = lay.origin // 128 + group
    live_group = (active - 1) // 128 if active else 0
    if gg == 0 or (group == live_group and (active & 127) != 0):
        slot = 0 if gg == 0 else 1 + ((gg - 1) & 1)
        lo, hi = P.region("stage", slot)
        element = ((slot * 128 + pos) * N_HEAD_KV + kv) * 256 + U * 4
        a = lay.records_bytes + element * 2
        return halves_from(np.stack([P.load(a, lo, hi), P.load(a + 4, lo, hi)], -1)).reshape(256)
    dim = (U & 31) * 4
    ph = kv * 2 + (U >> 5)
    base = (group * PHYS + ph) * RECORD
    lo = base
    hi = base + RECORD
    assert group < P.window // 128
    out = np.zeros((64, 4), np.float32)
    if value:
        index = pos * 128 + dim
        word = P.load(base + (index // 8) * 4, lo, hi) >> ((index & 7) * 4).astype(np.uint64)
        scale = P.half_at(base + 8192 + pos * 2, lo, hi)
        zero = P.half_at(base + 8448 + pos * 2, lo, hi)
        other = halves_from(np.stack([P.load(base + 8704 + dim * 2, lo, hi),
                                      P.load(base + 8704 + dim * 2 + 4, lo, hi)], -1))
        for i in range(4):
            out[:, i] = decode_math((word >> np.uint64(4 * i)) & 15, scale, zero, other[:, i])
    else:
        scale = halves_from(np.stack([P.load(base + 8192 + dim * 2, lo, hi),
                                      P.load(base + 8192 + dim * 2 + 4, lo, hi)], -1))
        zero = halves_from(np.stack([P.load(base + 8448 + dim * 2, lo, hi),
                                     P.load(base + 8448 + dim * 2 + 4, lo, hi)], -1))
        other = P.half_at(base + 8704 + pos * 2, lo, hi)
        nib_base = base + (pos // 8) * 4
        for i in range(4):
            q = (P.load(nib_base + (dim + i) * 64, lo, hi) >> np.uint64((pos & 7) * 4)) & 15
            out[:, i] = decode_math(q, scale[:, i], zero[:, i], other)
    return out.reshape(256)


def opt_tile(P, value, t0, kv):
    """[8, 256] K or V of the aligned record tile t0..t0+7 (tile_scores / tile_value)."""
    group, pos0 = t0 // 128, t0 & 127
    assert pos0 % 8 == 0 and group < P.window // 128
    dim = (U & 31) * 4
    base = (group * PHYS + kv * 2 + (U >> 5)) * RECORD
    lo, hi = base, base + RECORD
    out = np.zeros((8, 64, 4), np.float32)
    if not value:
        scale = halves_from(np.stack([P.load(base + 8192 + dim * 2 + 4 * j, lo, hi)
                                      for j in range(2)], -1))
        zero = halves_from(np.stack([P.load(base + 8448 + dim * 2 + 4 * j, lo, hi)
                                     for j in range(2)], -1))
        other = halves_from(np.stack([P.load(base + 8704 + pos0 * 2 + 4 * j, lo, hi)
                                      for j in range(4)], -1))  # [64, 8]
        nib_base = base + (pos0 // 8) * 4
        words = [P.load(nib_base + (dim + i) * 64, lo, hi) for i in range(4)]
        for t in range(8):
            for i in range(4):
                q = (words[i] >> np.uint64(4 * t)) & 15
                out[t, :, i] = decode_math(q, scale[:, i], zero[:, i], other[:, t])
    else:
        scale = halves_from(np.stack([P.load(base + 8192 + pos0 * 2 + 4 * j, lo, hi)
                                      for j in range(4)], -1))  # [64, 8]
        zero = halves_from(np.stack([P.load(base + 8448 + pos0 * 2 + 4 * j, lo, hi)
                                     for j in range(4)], -1))
        other = halves_from(np.stack([P.load(base + 8704 + (U & 31) * 8 + 4 * j, lo, hi)
                                      for j in range(2)], -1))  # [64, 4]
        for t in range(8):
            index = (pos0 + t) * 128 + dim
            word = P.load(base + (index // 8) * 4, lo, hi) >> ((index & 7) * 4).astype(np.uint64)
            for i in range(4):
                q = (word >> np.uint64(4 * i)) & 15
                out[t, :, i] = decode_math(q, scale[:, t], zero[:, t], other[:, i])
    return out.reshape(8, 256)


# ---------------------------------------------------------- kernel emulation ---


def causal_limit(lay, active, pos):
    causal = 0 if pos < lay.origin else pos - lay.origin + 1
    return min(active, causal)


def run_reference(Kp, Vp, q, active, pos, scale, k_src, v_src, prefix=False):
    """rkva_range_attn_kvarn4.hlsl + rkva_range_reduce.hlsl.

    prefix=True emulates RKVA_SERIAL_PREFIX=1: window = q_count = limit (split
    policy still follows active_count)."""
    lay = Kp.lay
    limit = causal_limit(lay, active, pos)
    split_k, rows = reference_split(active)
    window = limit if prefix else active
    Kp.window = Vp.window = window
    tail_begin = window - min(window, lay.ptt)
    qrot = butterfly_wht(q)
    kh = np.repeat(np.arange(N_HEAD_KV), GQA)
    partials = []
    for split in range(split_k):
        b, e = min(limit, split * rows), min(limit, (split + 1) * rows)
        mx = np.full(N_HEAD, -FLT_MAX, np.float32)
        den = np.zeros(N_HEAD, np.float32)
        body = np.zeros((N_HEAD, D), np.float32)
        tail = np.zeros((N_HEAD, D), np.float32)
        for r in range(b, e):
            exact = r >= tail_begin
            if exact:
                k = np.stack([ref_tail(Kp, r, kv) for kv in range(N_HEAD_KV)])
                v = np.stack([ref_tail(Vp, r, kv) for kv in range(N_HEAD_KV)])
            else:
                k = np.stack([ref_row(Kp, False, r, kv, window) for kv in range(N_HEAD_KV)])
                v = np.stack([ref_row(Vp, True, r, kv, window) for kv in range(N_HEAD_KV)])
            assert np.array_equal(k.view(np.uint32), k_src[r].view(np.uint32)), ("ref K", r)
            assert np.array_equal(v.view(np.uint32), v_src[r].view(np.uint32)), ("ref V", r)
            qq = q if exact else qrot
            s = (np.einsum("hd,hd->h", qq, k[kh]) * F32(scale)).astype(np.float32)
            nm = np.maximum(mx, s)
            old = np.where(den == 0, F32(0), np.exp(mx - nm)).astype(np.float32)
            w = np.exp(s - nm).astype(np.float32)
            body = (body * old[:, None]).astype(np.float32)
            tail = (tail * old[:, None]).astype(np.float32)
            if exact:
                tail = (tail + w[:, None] * v[kh]).astype(np.float32)
            else:
                body = (body + w[:, None] * v[kh]).astype(np.float32)
            den = (den * old + w).astype(np.float32)
            mx = nm
        partials.append((mx, den, (butterfly_wht(body) + tail).astype(np.float32)))
    out, lse = m4.reduce_splits(partials)
    return split_k, out, lse


def run_optimized(
    Kp, Vp, q, active, pos, scale, mode, wave, k_src, v_src, prefix=False, decode_split=False, mq_tile=0,
    mq2_cap32=False
):
    """prefix=True emulates RKVA_SERIAL_PREFIX=1 (see run_reference)."""
    lay = Kp.lay
    limit = causal_limit(lay, active, pos)
    assert not (decode_split and mq_tile)
    split_k, rows = (
        kvarn4_decode_split(active, mode) if decode_split else
        kvarn4_mq_split(active, mode, mq_tile, mq2_cap32) if mq_tile else kvarn4_split(active, mode)
    )
    assert split_k <= 256 and rows % 8 == 0 and rows * split_k >= active
    n_waves = 128 // wave
    window = limit if prefix else active
    Kp.window = Vp.window = window
    tail_begin = window - min(window, lay.ptt)
    record_lo = 128 if lay.origin == 0 else 0
    record_hi = window & ~127
    tiles = 0
    partials = []
    for split in range(split_k):
        begin, end = min(limit, split * rows), min(limit, (split + 1) * rows)
        fast_lo = (max(begin, record_lo) + 7) & ~7
        fast_hi = min(end, record_hi, tail_begin) & ~7
        if fast_hi <= fast_lo:
            fast_lo = fast_hi = end
        mx = np.full(N_HEAD, -FLT_MAX, np.float32)
        den = np.zeros(N_HEAD, np.float32)
        acc = np.zeros((N_HEAD, D), np.float32)
        for kv in range(N_HEAD_KV):
            hs = slice(kv * GQA, (kv + 1) * GQA)
            qo = q[hs]
            qr = butterfly_wht(qo)
            covered = []
            states = []
            for w in range(n_waves):
                m_ = np.full(GQA, -FLT_MAX, np.float32)
                d_ = np.zeros(GQA, np.float32)
                bd = np.zeros((GQA, D), np.float32)
                tl = np.zeros((GQA, D), np.float32)
                for sb, se in ((begin, fast_lo), (fast_hi, end)):
                    for r in range(sb + w, se, n_waves):
                        exact = r >= tail_begin
                        if exact:
                            k, x, qa = opt_tail4(Kp, r, kv), opt_tail4(Vp, r, kv), qo
                        else:
                            k = opt_body4(Kp, False, r, kv, window)
                            x = opt_body4(Vp, True, r, kv, window)
                            qa = qr
                        assert np.array_equal(k.view(np.uint32), k_src[r, kv].view(np.uint32))
                        assert np.array_equal(x.view(np.uint32), v_src[r, kv].view(np.uint32))
                        s = (qa @ k * F32(scale)).astype(np.float32)
                        nm = np.maximum(m_, s)
                        old = np.where(d_ == 0, F32(0), np.exp(m_ - nm)).astype(np.float32)
                        wt = np.exp(s - nm).astype(np.float32)
                        bd = (bd * old[:, None]).astype(np.float32)
                        tl = (tl * old[:, None]).astype(np.float32)
                        if exact:
                            tl = (tl + wt[:, None] * x[None, :]).astype(np.float32)
                        else:
                            bd = (bd + wt[:, None] * x[None, :]).astype(np.float32)
                        d_ = (d_ * old + wt).astype(np.float32)
                        m_ = nm
                        covered.append(r)
                for t0 in range(fast_lo + 8 * w, fast_hi, 8 * n_waves):
                    k8, v8 = opt_tile(Kp, False, t0, kv), opt_tile(Vp, True, t0, kv)
                    assert not np.any(np.arange(t0, t0 + 8) >= tail_begin)
                    assert np.array_equal(k8.view(np.uint32), k_src[t0 : t0 + 8, kv].view(np.uint32))
                    assert np.array_equal(v8.view(np.uint32), v_src[t0 : t0 + 8, kv].view(np.uint32))
                    sc = (qr @ k8.T * F32(scale)).astype(np.float32)  # [GQA, 8]
                    nm = np.maximum(m_, sc.max(axis=1))
                    old = np.where(d_ == 0, F32(0), np.exp(m_ - nm)).astype(np.float32)
                    wt = np.exp(sc - nm[:, None]).astype(np.float32)
                    bd = (bd * old[:, None] + wt @ v8).astype(np.float32)
                    tl = (tl * old[:, None]).astype(np.float32)
                    d_ = (d_ * old + wt.sum(axis=1)).astype(np.float32)
                    m_ = nm
                    covered.extend(range(t0, t0 + 8))
                    tiles += 1
                states.append((m_, d_, bd, tl))
            assert sorted(covered) == list(range(begin, end)), "row coverage"
            mm, md, both = merge_partial_states([
                (m_, d_, np.concatenate((bd, tl), axis=1)) for m_, d_, bd, tl in states
            ])
            bm, tm = both[:, :D], both[:, D:]
            mx[hs], den[hs] = mm, md
            acc[hs] = ((butterfly_wht(bm) + tm) if split_k == 1 else
                       (bm + butterfly_wht(tm))).astype(np.float32)
        partials.append((mx, den, acc))
    out, lse = reduce_partials(partials, rotated=split_k != 1)
    return split_k, tiles, out, lse


# -------------------------------------------------------------------- checks ---


def check_bee(fixture_dir):
    raw = (fixture_dir / "bee-kvarn4-records.bin").read_bytes()
    deq = np.fromfile(fixture_dir / "bee-kvarn4-dequant.bin", "<f4").reshape(2, 128, 128)
    kd = decode_record(raw[:RECORD], value=False).T  # back to [row=dim, col=tok]
    vd = decode_record(raw[RECORD : 2 * RECORD], value=True)
    assert np.array_equal(kd, deq[0]) and np.array_equal(vd, deq[1]), "Bee record decode"
    pair = np.fromfile(fixture_dir / "bee-kvarn4-wht.bin", "<f4").reshape(2, 256)
    assert np.array_equal(butterfly_wht(pair[0]), pair[1]), "butterfly WHT vs Bee"
    hd = float(np.max(np.abs(H256 @ pair[0].astype(np.float64) - pair[1])))
    assert hd < 1e-5, hd
    # The involution the domain split relies on.
    assert np.allclose(H256 @ H256, np.eye(256), atol=1e-12)
    print(f"bee fixtures: record decode exact, butterfly WHT exact, |H256 - Bee WHT| {hd:.1e}")
    return raw[:RECORD], raw[RECORD : 2 * RECORD]


def check_policy():
    for mode in (1, 2, 3):
        for active in (0, 1, 7, 8, 9, 127, 128, 129, 1000, 4133, 28672, 131072):
            k, rows = kvarn4_split(active, mode)
            assert 1 <= k <= 256 and rows % 8 == 0 and rows * k >= active
            assert active == 0 or rows * (k - 1) < active


def check_wave_mapping():
    for wave in (32, 64, 128):
        units = []
        for lane in range(wave):
            if lane < 64:
                units.append(lane)
            if wave < 64 and lane + wave < 64:
                units.append(lane + wave)
        assert sorted(units) == list(range(64)), wave
        # unit -> (slice, record dims) covers each rotated dim exactly once
        dims = sorted((u >> 5) * 128 + (u & 31) * 4 + i for u in units for i in range(4))
        assert dims == list(range(256)), wave


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the 28672-row case")
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--bee-fixtures", type=Path, default=DEFAULT_BEE)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    check_policy()
    check_wave_mapping()
    bee = None
    if args.bee_fixtures and (args.bee_fixtures / "bee-kvarn4-records.bin").exists():
        bee = check_bee(args.bee_fixtures)
    else:
        print("bee fixtures: not found (synthetic records only)")

    scale = 1.0 / 16.0
    far = 102400
    W = (32, 64, 128)
    # (origin, active, ptt, query position relative to origin, modes, waves, bee)
    cases = [
        (far, 1, 0, 0, (1,), W, False),  # live stage only
        (far, 1, 128, 0, (1,), (64,), False),  # tail only
        (far, 127, 0, 126, (1, 2), W, False),
        (far, 128, 0, 127, (1, 2, 3), W, False),  # one complete record group: tiles
        (far, 128, 128, 127, (1,), (64,), False),  # all tail
        (far, 129, 128, 128, (1, 2), (32, 64), False),
        (far, 300, 0, 299, (1, 2, 3), W, False),
        (far, 300, 128, 299, (1, 2), (64,), False),
        (far, 384, 0, 383, (1, 2), (64,), True),  # genuine Bee records in the tiles
        (far, 1000, 128, 999, (1, 2, 3), W, True),
        (far, 1000, 128, 517, (1, 2), (32, 64), False),  # causal limit inside records
        (far, 1003, 0, 1002, (1, 3), (64,), False),
        (far, 500, 128, -3, (1,), (32, 64), False),  # query before origin: masked
        (0, 1, 0, 0, (1,), W, False),  # sink stage only
        (0, 128, 0, 127, (1,), (64,), False),  # complete sink group stays staged
        (0, 200, 128, 199, (1, 2), (32, 64), False),
        (0, 520, 0, 519, (1, 2, 3), W, True),
        (0, 520, 128, 300, (1,), (64,), False),
        (far, 4133, 128, 4132, (1, 2, 3), (64,), False),
        (far, 4133, 0, 4132, (1,), (32, 128), False),
    ]
    if not args.quick:
        cases.append((far, 28672, 128, 28671, (1,), (64,), False))
    worst = 0.0
    for origin, active, ptt, rel, modes, waves, use_bee in cases:
        lay = Layout(origin, max(active, 1) + 64, ptt)
        rec_k = bee[0] if (use_bee and bee) else None
        rec_v = bee[1] if (use_bee and bee) else None
        kbuf = build_plane(rng, lay, active, rec_k)
        vbuf = build_plane(rng, lay, active, rec_v)
        Kp, Vp = Plane(kbuf, lay, active), Plane(vbuf, lay, active)
        k_src, exact = oracle_sources(lay, kbuf, active, value=False)
        v_src, _ = oracle_sources(lay, vbuf, active, value=True)
        assert np.all(np.isfinite(k_src)) and np.all(np.isfinite(v_src)), "poison reached oracle"
        q = rng.normal(0, 1, (N_HEAD, D)).astype(np.float32)
        pos = origin + rel
        limit = causal_limit(lay, active, pos)
        o64, l64 = oracle_attention(q, k_src, v_src, exact, limit, scale)
        _, r_out, r_lse = run_reference(Kp, Vp, q, active, pos, scale, k_src, v_src)
        re_o, re_l = m4.compare("reference", r_out, r_lse, o64, l64)
        mag = max(1.0, float(np.max(np.abs(o64))))
        assert re_o <= 1e-4 * mag and re_l <= 1e-4 * max(1.0, float(np.nanmax(np.abs(
            np.where(np.isinf(l64), 0, l64))))), ("reference vs oracle", re_o, re_l)
        for mode in modes:
            for wave in waves:
                for dec_split in ([False, True] if (wave == 64 and mode in (1, 2)) else [False]):
                    split_k, tiles, out, lse = run_optimized(
                        Kp,
                        Vp,
                        q,
                        active,
                        pos,
                        scale,
                        mode,
                        wave,
                        k_src,
                        v_src,
                        decode_split=dec_split,
                    )
                    e_o, e_l = m4.compare("opt", out, lse, o64, l64)
                    d_o, _ = m4.compare("opt-vs-ref", out, lse, r_out, r_lse)
                    ok = e_o <= max(2e-5 * mag, 4 * re_o) and e_l <= max(2e-5, 4 * re_l)
                    worst = max(worst, e_o / mag)
                    dec_tag = " [decode_split]" if dec_split else ""
                    print(
                        f"origin={origin:6d} active={active:5d} ptt={ptt:3d} limit={limit:5d} "
                        f"mode={mode} wave={wave:3d} split_k={split_k:3d} tiles={tiles:5d}{dec_tag} "
                        f"bee={int(bool(use_bee and bee))} |O-oracle|={e_o:.2e} (ref {re_o:.2e}) "
                        f"|LSE-oracle|={e_l:.2e} (ref {re_l:.2e}) |O-ref|={d_o:.2e} "
                        f"{'OK' if ok else 'FAIL'}"
                    )
                    if not ok:
                        return 1
    print(f"rkva_range_kvarn4_opt_model: ok (worst relative |O-oracle| {worst:.2e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
