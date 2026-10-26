#!/usr/bin/env python3
"""Compile the GPU ceiling probe shaders into embedded DXIL headers.

Every target is a (source, entry point, shader model, defines) tuple. The DXIL
is byte-embedded the same way scripts/compile-rkva-range-shaders.sh embeds the
RKVA shaders, so the UWP app needs no runtime shader compiler.

Usage:
    # Local dxc on PATH (Windows build host):
    ./scripts/compile-gpupeak-shaders.py

    # Compile on the Windows box from a Linux checkout (isolated scratch dir):
    ./scripts/compile-gpupeak-shaders.py --ssh-host hjotha@192.168.1.193 \
        --remote-root 'C:/Users/hjotha/build/oc-xbox-ceiling-20261026'

    # Regenerate only some targets, or list them:
    ./scripts/compile-gpupeak-shaders.py --only fma_f32 empty
    ./scripts/compile-gpupeak-shaders.py --list

The remote mode copies nothing outside --remote-root and never builds the app;
it only runs dxc.exe on one source file at a time. A manifest
(shaders/generated/gpupeak-shaders.json) records source/header SHA256, the DXC
version string, the exact flags and the DXIL size for the report.
"""
import argparse
import base64
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHADERS = ROOT / "shaders"
GENERATED = SHADERS / "generated"
DEFAULT_DXC = r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe"

# name, source, entry, profile, extra dxc args, extra defines
TARGETS = [
    ("wave_prefix64", "gpupeak_wave_caps.hlsl", "CSWaveCaps", "cs_6_4", [], []),
    ("wave_multi65", "gpupeak_wave_caps.hlsl", "CSWaveCaps", "cs_6_5", [],
     ["GPUPK_WAVE_MULTI_PREFIX=1"]),
    ("wave_match65", "gpupeak_wave_caps.hlsl", "CSWaveCaps", "cs_6_5", [],
     ["GPUPK_WAVE_MATCH=1"]),
    ("wave_quad66", "gpupeak_wave_caps.hlsl", "CSWaveCaps", "cs_6_6", [],
     ["GPUPK_WAVE_QUAD=1"]),
    ("stream_read", "gpupeak_stream.hlsl", "CSRead", "cs_6_0", [], []),
    ("stream_write", "gpupeak_stream.hlsl", "CSWrite", "cs_6_0", [], []),
    ("stream_copy", "gpupeak_stream.hlsl", "CSCopy", "cs_6_0", [], []),
    ("fma_f32", "gpupeak_alu.hlsl", "CSFmaF32", "cs_6_0", [], ["GPUPK_ALU_F32=1"]),
    ("fma_f16", "gpupeak_alu.hlsl", "CSFmaF16", "cs_6_2", ["-enable-16bit-types"],
     ["GPUPK_ALU_F16=1"]),
    ("fma_f16_min16", "gpupeak_alu.hlsl", "CSFmaF16Min16", "cs_6_0", [],
     ["GPUPK_ALU_F16_MIN16=1"]),
    ("dot2_f16", "gpupeak_alu.hlsl", "CSDot2F16", "cs_6_4", ["-enable-16bit-types"],
     ["GPUPK_ALU_DOT2=1"]),
    ("dot4_i8", "gpupeak_alu.hlsl", "CSDot4I8", "cs_6_4", [], ["GPUPK_ALU_DOT4=1"]),
    ("latency_fma", "gpupeak_alu.hlsl", "CSLatencyFma", "cs_6_0", [], ["GPUPK_LATENCY_FMA=1"]),
    ("cu_probe_32", "gpupeak_alu.hlsl", "CSCuProbe", "cs_6_0", [], ["GPUPK_CU_PROBE=1"]),
    ("cu_probe_16", "gpupeak_alu.hlsl", "CSCuProbe", "cs_6_0", [],
     ["GPUPK_CU_PROBE=1", "GPUPK_CU_LDS_16=1"]),
    ("empty", "gpupeak_empty.hlsl", "CSEmpty", "cs_6_0", [], []),
]

ALLOWED_ONLY = {t[0] for t in TARGETS}


def dxc_args(entry, profile, flags, defines):
    args = ["-T", profile, "-E", entry]
    for define in defines:
        args += ["-D", define]
    return args + flags


def compile_local(dxc, source, args, out_dxil):
    cmd = [dxc] + args + [str(source), "-Fo", str(out_dxil)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit("dxc failed for %s" % source.name)
    return proc.stderr


def compile_remote(host, remote_root, dxc, source, args, out_dxil):
    """Copy one source to the Windows scratch dir, compile, copy the DXIL back."""
    remote_src = remote_root.rstrip("/") + "/ceiling.hlsl"
    remote_out = remote_root.rstrip("/") + "/ceiling.dxil"
    subprocess.run(["scp", "-q", str(source), "%s:%s" % (host, remote_src)], check=True)
    quote = lambda s: "'" + s.replace("'", "''") + "'"
    script = (
        "$ErrorActionPreference='Stop'; $dxc=" + quote(dxc) + "; $opts=@("
        + ",".join(quote(a) for a in args)
        + "); & $dxc @opts " + quote(remote_src) + " -Fo " + quote(remote_out)
        + "; exit $LASTEXITCODE"
    )
    encoded = base64.b64encode(script.encode("utf-16le")).decode()
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
         "powershell -NoProfile -EncodedCommand " + encoded],
        capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit("remote dxc failed for %s" % source.name)
    subprocess.run(["scp", "-q", "%s:%s" % (host, remote_out), str(out_dxil)], check=True)
    return proc.stdout + proc.stderr


def dxc_version(dxc, host, remote_root):
    if host:
        script = ("$ErrorActionPreference='Stop'; & " + "'" + dxc + "'" + " --version")
        encoded = base64.b64encode(script.encode("utf-16le")).decode()
        proc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
             "powershell -NoProfile -EncodedCommand " + encoded],
            capture_output=True, text=True)
        raw = proc.stdout + proc.stderr
    else:
        proc = subprocess.run([dxc, "--version"], capture_output=True, text=True)
        raw = proc.stdout + proc.stderr
    # Drop ssh's transport banner (PQ-key warning) so the manifest records only
    # the compiler identity.
    lines = [line.strip() for line in raw.splitlines()
             if line.strip() and "WARNING" not in line and "vulnerable" not in line
             and "openssh.com" not in line]
    return " ".join(lines)


def emit_header(out_header, name, data, source, flags):
    lines = [
        "// Generated by tools/xbox/scripts/compile-gpupeak-shaders.py; do not edit.",
        "// Target: " + name,
        "// Source: " + source.name,
        "// Source SHA256: " + hashlib.sha256(source.read_bytes()).hexdigest(),
        "// Compiler options: " + " ".join(flags),
        "#pragma once",
        "#include <cstddef>",
        "#include <cstdint>",
        "// " + str(len(data)) + " bytes DXIL",
        "// clang-format off",
        "static const uint8_t g_gpupeak_" + name + "_dxil[] = {",
    ]
    for offset in range(0, len(data), 12):
        lines.append("    " + ", ".join("0x%02x" % value for value in data[offset:offset + 12]) + ",")
    lines += [
        "};",
        "static const size_t g_gpupeak_" + name + "_dxil_size = " + str(len(data)) + ";",
        "// clang-format on",
        "",
    ]
    out_header.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dxc", default=None, help="path to dxc (default: dxc on PATH)")
    parser.add_argument("--dxc-version", default=None,
                        help="record this compiler provenance instead of invoking --version (wrappers)")
    parser.add_argument("--ssh-host", default=None,
                        help="compile on this Windows host instead of locally")
    parser.add_argument("--remote-root",
                        default="C:/Users/hjotha/build/oc-xbox-ceiling-20261026",
                        help="isolated scratch dir on the Windows host")
    parser.add_argument("--only", nargs="*", default=None, help="target names")
    parser.add_argument("--list", action="store_true", help="print targets and exit")
    parser.add_argument("--check", action="store_true",
                        help="verify every committed header matches its source hash and exit")
    args = parser.parse_args()

    if args.list:
        for name, source, entry, profile, flags, defines in TARGETS:
            print("%-14s %-22s %-14s %-6s %s" % (name, source, entry, profile,
                                                 " ".join(defines + flags)))
        return 0

    if args.check:
        stale = []
        for name, source_name, _entry, _profile, _flags, _defines in TARGETS:
            source = SHADERS / source_name
            header = GENERATED / ("gpupeak_%s_dxil.h" % name)
            want = hashlib.sha256(source.read_bytes()).hexdigest()
            got = None
            if header.exists():
                for line in header.read_text().splitlines():
                    if line.startswith("// Source SHA256: "):
                        got = line.split(": ", 1)[1].strip()
                        break
            if got != want:
                stale.append(name)
                print("%-14s STALE header source %s != %s" % (name, got, want))
            else:
                print("%-14s ok %s" % (name, want[:16]))
        if stale:
            sys.stderr.write("stale DXIL headers: " + ", ".join(stale) + "\n")
            return 1
        return 0

    only = ALLOWED_ONLY
    if args.only:
        unknown = sorted(set(args.only) - ALLOWED_ONLY)
        if unknown:
            raise SystemExit("unknown target(s): " + ", ".join(unknown))
        only = set(args.only)

    GENERATED.mkdir(parents=True, exist_ok=True)
    scratch = ROOT / ".gpupeak-dxil"
    scratch.mkdir(exist_ok=True)
    dxc = args.dxc or (DEFAULT_DXC if args.ssh_host else "dxc")
    version = args.dxc_version or dxc_version(dxc, args.ssh_host, args.remote_root)
    print("dxc: " + version.replace("\n", " | "))

    manifest = {"dxc_version": version, "ssh_host": args.ssh_host, "targets": []}
    manifest_path = GENERATED / "gpupeak-shaders.json"
    if manifest_path.exists():
        try:
            manifest["targets"] = [t for t in json.loads(manifest_path.read_text())["targets"]
                                   if t["name"] not in only]
        except (KeyError, ValueError):
            manifest["targets"] = []

    for name, source_name, entry, profile, flags, defines in TARGETS:
        if name not in only:
            continue
        source = SHADERS / source_name
        argv = dxc_args(entry, profile, flags, defines)
        out_dxil = scratch / (name + ".dxil")
        if args.ssh_host:
            compile_remote(args.ssh_host, args.remote_root, dxc, source, argv, out_dxil)
        else:
            compile_local(dxc, source, argv, out_dxil)
        data = out_dxil.read_bytes()
        header = GENERATED / ("gpupeak_%s_dxil.h" % name)
        emit_header(header, name, data, source, argv)
        manifest["targets"].append({
            "name": name,
            "source": source_name,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "entry": entry,
            "profile": profile,
            "defines": defines,
            "flags": flags,
            "dxil_bytes": len(data),
            "dxil_sha256": hashlib.sha256(data).hexdigest(),
            "header": header.name,
        })
        print("%-14s %6d bytes  %s" % (name, len(data), header.relative_to(ROOT)))

    manifest["targets"].sort(key=lambda t: t["name"])
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print("manifest: " + str(manifest_path.relative_to(ROOT)))
    shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
