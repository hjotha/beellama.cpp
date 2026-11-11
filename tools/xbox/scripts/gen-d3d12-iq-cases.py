"""Generate IQ matmul conformance cases for the WARP harness.

The reference dequantizers below are ports of ggml/src/ggml-quants.c
(dequantize_row_iq2_s / iq3_xxs / iq3_s / iq4_xs). They are deliberately
independent of the C++ emulation in ggml_d3d12.cpp, so a match against the
DXIL kernels validates the shaders against ggml's semantics.

Case file (little-endian): magic "IQC1", n, k, ncols, row_bytes, weight bytes,
x floats, reference y floats. Weights are random valid block bytes with a
finite fp16 d: every index/sign/scale field of these formats spans its full
byte range, and IQ quantization needs an importance matrix, so random blocks
are the only way to cover the value space.

Usage:
    python3 scripts/gen-d3d12-iq-cases.py <out-dir>

Then, on a Windows host with the Windows SDK / VS2022:

    cl /EHsc /std:c++17 /O2 tests\\warp\\iq_harness.cpp /Fe:iq_harness.exe \
        d3d12.lib dxgi.lib
    iq_harness.exe iq2_s_t64.dxil case_00_iq2_s_17x2560_1.bin iq_tables.bin

The DXIL blobs come from scripts/compile-iq-shaders.py (shaders/generated).
"""

import argparse
import re
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
COMMON = ROOT / "ggml" / "src" / "ggml-common.h"

BLOCK = {"iq2_s": 82, "iq3_xxs": 98, "iq3_s": 110, "iq4_xs": 136}


def table(source, name):
    pattern = r"GGML_TABLE_BEGIN\([^,]+,\s*" + name + r",\s*\d+\)(.*?)GGML_TABLE_END\(\)"
    body = re.search(pattern, source, re.S).group(1)
    return [int(tok, 0) for tok in re.findall(r"-?0x[0-9a-fA-F]+|-?\d+", body)]


SOURCE = COMMON.read_text(encoding="utf-8")
IQ2S = table(SOURCE, "iq2s_grid")
IQ3XXS = table(SOURCE, "iq3xxs_grid")
IQ3S = table(SOURCE, "iq3s_grid")
KSIGNS = table(SOURCE, "ksigns_iq2xs")
KVAL = table(SOURCE, "kvalues_iq4nl")


def f16(raw):
    return struct.unpack("<e", raw)[0]


def u32(buf, off):
    return struct.unpack_from("<I", buf, off)[0]


def u16(buf, off):
    return struct.unpack_from("<H", buf, off)[0]


def dequant_iq2_s(block):
    d = f16(block[0:2])
    qs = block[2:66]
    qh = block[66:74]
    scales = block[74:82]
    signs = qs[32:64]
    out = []
    for ib in range(8):
        db0 = d * (0.5 + (scales[ib] & 0xF)) * 0.25
        db1 = d * (0.5 + (scales[ib] >> 4)) * 0.25
        for l in range(4):
            dl = db0 if l < 2 else db1
            idx = qs[ib * 4 + l] | ((qh[ib] << (8 - 2 * l)) & 0x300)
            grid = IQ2S[idx]
            s = signs[ib * 4 + l]
            for j in range(8):
                v = (grid >> (8 * j)) & 0xFF
                out.append(dl * v * (-1.0 if (s >> j) & 1 else 1.0))
    return out


def dequant_iq3_xxs(block):
    d = f16(block[0:2])
    qs = block[2:98]
    out = []
    for ib in range(8):
        aux = u32(block, 66 + 4 * ib)
        db = d * (0.5 + (aux >> 28)) * 0.5
        for l in range(4):
            signs = KSIGNS[(aux >> (7 * l)) & 127]
            g1 = IQ3XXS[qs[8 * ib + 2 * l]]
            g2 = IQ3XXS[qs[8 * ib + 2 * l + 1]]
            for j in range(4):
                out.append(db * ((g1 >> (8 * j)) & 0xFF) * (-1.0 if (signs >> j) & 1 else 1.0))
            for j in range(4):
                out.append(db * ((g2 >> (8 * j)) & 0xFF) * (-1.0 if (signs >> (j + 4)) & 1 else 1.0))
    return out


def dequant_iq3_s(block):
    d = f16(block[0:2])
    qs = block[2:66]
    qh = block[66:74]
    signs = block[74:106]
    scales = block[106:110]
    out = []
    for m in range(4):
        db1 = d * (1 + 2 * (scales[m] & 0xF))
        db2 = d * (1 + 2 * (scales[m] >> 4))
        for half in range(2):
            db = db1 if half == 0 else db2
            h = qh[2 * m + half]
            for l in range(4):
                g1 = IQ3S[qs[16 * m + 8 * half + 2 * l] | ((h << (8 - 2 * l)) & 256)]
                g2 = IQ3S[qs[16 * m + 8 * half + 2 * l + 1] | ((h << (7 - 2 * l)) & 256)]
                sg = signs[8 * m + 4 * half + l]
                for j in range(4):
                    out.append(db * ((g1 >> (8 * j)) & 0xFF) * (-1.0 if (sg >> j) & 1 else 1.0))
                for j in range(4):
                    out.append(db * ((g2 >> (8 * j)) & 0xFF) * (-1.0 if (sg >> (j + 4)) & 1 else 1.0))
    return out


def dequant_iq4_xs(block):
    d = f16(block[0:2])
    sh = u16(block, 2)
    sl = block[4:8]
    qs = block[8:136]
    out = []
    for ib in range(8):
        ls = ((sl[ib // 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4)
        dl = d * (ls - 32)
        for j in range(16):
            out.append(dl * KVAL[qs[ib * 16 + j] & 0xF])
        for j in range(16):
            out.append(dl * KVAL[qs[ib * 16 + j] >> 4])
    return out


DEQUANT = {
    "iq2_s": dequant_iq2_s,
    "iq3_xxs": dequant_iq3_xxs,
    "iq3_s": dequant_iq3_s,
    "iq4_xs": dequant_iq4_xs,
}


def rng_stream(seed, count):
    """xorshift32 bytes, independent of the C++ std::mt19937 generator."""
    state = seed & 0xFFFFFFFF or 0x9E3779B9
    out = bytearray(count)
    for i in range(count):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        out[i] = state & 0xFF
    return bytes(out)


def make_case(kind, n, k, ncols, seed):
    block_bytes = BLOCK[kind]
    nb = k // 256
    row_bytes = block_bytes * nb
    weights = bytearray(rng_stream(seed, row_bytes * n))
    d = struct.pack("<e", 0.0625 + (seed % 7) * 0.003)
    for r in range(n):
        for b in range(nb):
            off = r * row_bytes + b * block_bytes
            weights[off:off + 2] = d
    x = []
    state = (seed * 2654435761) & 0xFFFFFFFF or 1
    for _ in range(k * ncols):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        x.append(((state & 0xFFFF) / 32768.0) - 1.0)
    rows = []
    for r in range(n):
        row = []
        for b in range(nb):
            off = r * row_bytes + b * block_bytes
            row.extend(DEQUANT[kind](bytes(weights[off:off + block_bytes])))
        rows.append(row)
    y = []
    for c in range(ncols):
        for r in range(n):
            acc = 0.0
            for i in range(k):
                acc += rows[r][i] * x[c * k + i]
            y.append(acc)
    return bytes(weights), x, y, row_bytes


def write_case(path, kind, n, k, ncols, seed):
    weights, x, y, row_bytes = make_case(kind, n, k, ncols, seed)
    with open(path, "wb") as f:
        f.write(struct.pack("<5I", 0x31435149, n, k, ncols, row_bytes))
        f.write(weights)
        f.write(struct.pack("<%df" % len(x), *x))
        f.write(struct.pack("<%df" % len(y), *y))


def table_blob():
    blob = bytearray()
    for v in IQ2S:
        blob += struct.pack("<Q", v)
    for v in IQ3XXS:
        blob += struct.pack("<I", v)
    for v in IQ3S:
        blob += struct.pack("<I", v)
    blob += bytes(KSIGNS)
    for v in KVAL:
        blob += struct.pack("<f", float(v))
    if len(blob) != 11456:
        raise SystemExit("unexpected table blob size: %d" % len(blob))
    return bytes(blob)


CASES = [
    ("iq2_s", 17, 2560, 1, 11),
    ("iq3_xxs", 17, 2560, 1, 22),
    ("iq3_s", 17, 2560, 1, 33),
    ("iq4_xs", 17, 2560, 1, 44),
    # 9-block rows: 82*9/98*9/110*9 are 2-byte aligned, exercising ld32.
    ("iq2_s", 9, 2304, 1, 55),
    ("iq3_xxs", 9, 2304, 1, 66),
    ("iq3_s", 9, 2304, 1, 77),
    ("iq4_xs", 5, 2304, 1, 88),
    ("iq3_s", 5, 2304, 3, 99),
    # K=4096 selects the 128-thread blob in the backend.
    ("iq2_s", 9, 4096, 1, 111),
    ("iq3_xxs", 9, 4096, 1, 122),
    ("iq3_s", 9, 4096, 1, 133),
    ("iq4_xs", 9, 4096, 1, 144),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "iq_tables.bin").write_bytes(table_blob())
    for index, (kind, n, k, ncols, seed) in enumerate(CASES):
        name = "case_%02d_%s_%dx%d_%d.bin" % (index, kind, n, k, ncols)
        write_case(args.out / name, kind, n, k, ncols, seed)
        print(name)


if __name__ == "__main__":
    main()
