#!/usr/bin/env python3
"""Offline DXIL regression for the F16 wire's exact CPU-WHT query contract.

No GPU or socket is opened. DXC only disassembles the shipping header/blob.
The GPU serial/bulk probe remains the numerical OUT/LSE oracle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


def verify(assembly):
    if not re.search(r"define void @CSBulkUnpackQF16\(", assembly):
        raise ValueError("not the CSBulkUnpackQF16 entry point")
    arithmetic = [line.strip() for line in assembly.splitlines()
                  if re.search(r"= (?:fadd|fsub|fmul|fdiv)\b", line)]
    if any(re.search(r"\b(?:fast|reassoc|contract|afn)\b", line) for line in arithmetic):
        raise ValueError("F16 query WHT permits floating-point reassociation/contraction")
    if "FMad" in assembly or "fmuladd" in assembly:
        raise ValueError("F16 query WHT contains a contracted multiply/add")
    multiplies = [line for line in arithmetic if "= fmul " in line]
    # LLVM prints an exactly promoted F32 constant as its F64 hexadecimal bits.
    # Four H128 normalizations and two H2 normalizations serve the two lanes.
    h128 = sum("0x3FB6A09E60000000" in line for line in multiplies)
    h2 = sum("0x3FE6A09E60000000" in line for line in multiplies)
    if len(multiplies) != 6 or h128 != 4 or h2 != 2:
        raise ValueError("F16 query WHT lost separately rounded H128/H2 normalization")
    return dict(float_operations=len(arithmetic), h128_multiplications=h128,
                h2_multiplications=h2, relaxed_float_operations=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dxc", default="dxc", help="native DXC executable; disassembly only")
    parser.add_argument("--shader", type=Path, default=Path(__file__).resolve().parents[1] /
                        "shaders/generated/rkva_range_bulk_unpack_q_f16_dxil.h",
                        help="shipping generated header or compiled .dxil blob")
    args = parser.parse_args()
    data = args.shader.read_bytes()
    if args.shader.suffix == ".h":
        data = bytes(int(word, 16) for word in re.findall(r"0x([0-9a-fA-F]{2})", data.decode()))
    if not data.startswith(b"DXBC"):
        parser.exit(1, "FAIL: expected a DXIL container\n")
    try:
        with tempfile.TemporaryDirectory(prefix="rkva-f16-wht-") as directory:
            blob = Path(directory) / "unpack.dxil"
            blob.write_bytes(data)
            dumped = subprocess.run([args.dxc, "-dumpbin", str(blob)], check=True,
                                    capture_output=True, text=True)
        checks = verify(dumped.stdout)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        parser.exit(1, "FAIL: " + str(error) + "\n")
    print(json.dumps(dict(ok=True, shader=str(args.shader), sha256=hashlib.sha256(data).hexdigest(),
                         scope="offline compiled arithmetic contract; console parity still required",
                         **checks), indent=2))


if __name__ == "__main__":
    main()
