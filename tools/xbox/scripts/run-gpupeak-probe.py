#!/usr/bin/env python3
"""Inert-by-default hardware run for the fixed gpupeak ceiling probe.

This is the B2 follow-up to tools/xbox/scripts/run-xbox-ceiling.py.  It runs
only the `gpupeak.flag` headless suite (no RKVA listener, no pass1 sweep),
fetches LocalState\\gpupeak-result.json and writes a combined JSON document with:

  * the raw probe document (per-row checksums, PSO errors, adapter)
  * the roofline from tests/rkva_ceiling_roofline.build_result()
  * explicit checks for the three B2 questions: did stream validation pass,
    was the relaxed min16float PSO accepted, and what clock does the measured
    FP32 peak imply.

The package must already be installed; this script never builds, signs or
installs it.  Without --execute it prints the plan and opens no connection.

Examples:
    # offline plan
    ./scripts/run-gpupeak-probe.py

    # real run inside the hardware window (env has XBOX_IP/USER/PASS)
    ./scripts/run-gpupeak-probe.py --execute --host 192.168.1.200 \
        --pfn <PackageFullName> --output .oc-artifacts/gpupeak-b2.json

    # reuse a fetched JSON without touching the console
    ./scripts/run-gpupeak-probe.py --execute --gpupeak-result gpupeak.json \
        --output .oc-artifacts/gpupeak-b2.json
"""

import argparse
import importlib.util
import json
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
XBOX_ROOT = HERE.parent
DEFAULT_OUTPUT = XBOX_ROOT / ".oc-artifacts" / "gpupeak-probe-result.json"


def load_orchestrator():
    """Reuse the deploy/flag helpers from run-xbox-ceiling.py.

    The filename contains a hyphen, so import it by path instead of duplicating
    the Device Portal command handling.
    """
    path = HERE / "run-xbox-ceiling.py"
    spec = importlib.util.spec_from_file_location("run_xbox_ceiling", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probe_checks(gpupeak, roofline):
    rows = gpupeak.get("rows") or []
    by_probe = {}
    for row in rows:
        by_probe.setdefault(str(row.get("probe")), []).append(row)

    def accepted(probe):
        return any(r.get("gpu_valid") and r.get("gpu_seconds", 0) > 0
                   for r in by_probe.get(probe, []))

    stream_rows = []
    for row in by_probe.get("read", []) + by_probe.get("copy", []):
        if row.get("checksum_checked"):
            stream_rows.append({
                "probe": row.get("probe"),
                "variant": row.get("variant"),
                "checksum_ok": bool(row.get("checksum_ok")),
                "checksum": row.get("checksum"),
                "expected_checksum": row.get("expected_checksum"),
                "error": row.get("error", ""),
            })
    return {
        "stream_checksum_verified": bool(roofline["stream_checksum"]["verified"]),
        "stream_validation_rows": stream_rows,
        "fma_f32_accepted": accepted("fma_f32"),
        "fma_f16_native_accepted": accepted("fma_f16"),
        "fma_f16_min16_accepted": accepted("fma_f16_min16"),
        "dot2_f16_accepted": accepted("dot2_f16"),
        "dot4_i8_accepted": accepted("dot4_i8"),
        "pso_errors": gpupeak.get("pso_errors") or [],
        "derived_shader_clock_mhz": roofline["clocks"]["derived_shader_clock_mhz"],
        "partial": bool(roofline.get("partial")),
        "warnings": roofline.get("warnings") or [],
    }


def plan_text(args):
    return json.dumps({
        "execute": False,
        "tool": "gpupeak only (no RKVA listener, no pass1 sweep)",
        "package": "pre-installed beellama-xbox package with gpupeak.flag",
        "requires": ["XBOX_IP/XBOX_USER/XBOX_PASS in the environment",
                     "console leased by the caller; no host benchmark running"],
        "device": {"host": args.host or "<XBOX_IP>",
                   "env": "<XBOX_USER>:<XBOX_PASS> from ~/.config/xllama/xbox-env"},
        "steps": [
            "delete stale gpupeak-result.json(.done)",
            "upload gpupeak.flag and optional gpupeak-opts.txt",
            "start-app and wait for gpupeak-result.json.done",
            "fetch gpupeak-result.json, stop-app",
            "build_result() -> roofline + B2 checks",
        ],
        "gpupeak_opts": args.gpupeak_opts or "(package default: 256 MiB, 5 iterations)",
        "reuse_json": args.gpupeak_result or None,
        "output": args.output,
        "check_hint": "stream_checksum_verified must be true; fma_f16_min16_accepted "
                      "answers whether the relaxed half PSO is accepted; "
                      "derived_shader_clock_mhz comes from the measured FP32 peak.",
        "note": "run with --execute inside the hardware window; package build/install is "
                "the reviewer's step, not this script's",
    }, indent=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--host", help="console IP (or XBOX_IP)")
    parser.add_argument("--pfn", help="package full name (default: deploy.sh pfn)")
    parser.add_argument("--expect-pfn", help="refuse a different installed package")
    parser.add_argument("--gpupeak-opts", help="local key=value file -> gpupeak-opts.txt")
    parser.add_argument("--gpupeak-result", help="reuse an already fetched JSON")
    parser.add_argument("--timeout", type=int, default=900, help="gpupeak wait seconds")
    parser.add_argument("--keep-app", action="store_true",
                        help="leave the app running after the probe")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    if not args.execute:
        print(plan_text(args))
        return 0

    ceiling = load_orchestrator()
    roof = ceiling.roof
    journal = ceiling.Journal()
    work = pathlib.Path(tempfile.mkdtemp(prefix="gpupeak-b2-"))
    result = {"schema": "xbox-gpupeak-probe-run-v1",
              "started_utc": ceiling.now(),
              "host": args.host or "<reused-json>",
              "package_full_name": args.pfn or ""}
    status = 0
    try:
        if args.gpupeak_result:
            document = json.loads(pathlib.Path(args.gpupeak_result).read_text(encoding="utf-8"))
            journal.note("gpupeak:reuse", path=args.gpupeak_result,
                         rows=len(document.get("rows") or []))
        else:
            busy = ceiling.busy_processes()
            if busy:
                raise SystemExit("refusing --execute: benchmark processes are active:\n  " +
                                 "\n  ".join(busy[:8]))
            env = ceiling.device_env(args)
            pfn = args.pfn
            if not pfn:
                proc = ceiling.deploy(journal, env, "pfn")
                pfn = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            if not pfn:
                raise SystemExit("could not resolve the installed package full name")
            if args.expect_pfn and pfn != args.expect_pfn:
                raise SystemExit("installed package %s != expected %s" % (pfn, args.expect_pfn))
            result["package_full_name"] = pfn
            journal.note("package", pfn=pfn)
            # run_gpupeak expects these attributes on the shared namespace.
            args.gpupeak_timeout = args.timeout
            document = ceiling.run_gpupeak(journal, args, env, pfn, work)
            if not args.keep_app:
                ceiling.deploy(journal, env, "stop-app", pfn, check=False)

        roofline = roof.build_result(document, {"points": []})
        result["gpupeak"] = document
        result["roofline"] = roofline
        result["checks"] = probe_checks(document, roofline)
        journal.note("checks", **{k: v for k, v in result["checks"].items()
                                  if k in ("stream_checksum_verified", "partial",
                                           "fma_f16_min16_accepted",
                                           "derived_shader_clock_mhz")})
    finally:
        result["journal"] = journal.entries
        result["commands"] = journal.commands
        result["finished_utc"] = ceiling.now()
        out = pathlib.Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        import shutil
        shutil.rmtree(work, ignore_errors=True)
        print("wrote %s" % out)
    return status


if __name__ == "__main__":
    sys.exit(main())
