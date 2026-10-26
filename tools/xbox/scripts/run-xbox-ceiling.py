#!/usr/bin/env python3
"""One-window Xbox Series X ceiling run: probes in, roofline out.

Orchestrates, in this order, on an already installed package (this script never
builds, signs or installs anything):

  1. guard: refuses to start while a host benchmark process is running.
  2. headless `gpupeak.flag` run -> LocalState\\gpupeak-result.json
     (optional `gpupeak-opts.txt` for the 1 GiB pass and iteration count).
  3. XAML run with `api.flag` -> RKVA listener on :50054
     -> tests/rkva_pass1_sweep.py --execute (production pass1 shapes).
  4. tests/rkva_ceiling_roofline.build_result() -> xbox-ceiling-result.json.

Inert by default: without --execute it prints the exact plan, the package it
would target and the commands it would run, and opens no connection.

Package prerequisite (installed by the reviewer, not by this script):
  beellama-xbox-<version>-llamacpp.msix with the `gpupeak` headless mode.
  `gpupeak.flag` is unknown to earlier packages: they simply ignore it, so the
  script refuses to wait forever and reports "flag not honoured" instead.
"""

import argparse
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
XBOX_ROOT = HERE.parent
TESTS = XBOX_ROOT / "tests"
sys.path.insert(0, str(TESTS))

import rkva_ceiling_roofline as roof  # noqa: E402

DEFAULT_RESULT = XBOX_ROOT / ".oc-artifacts" / "xbox-ceiling-result.json"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Journal:
    def __init__(self):
        self.entries = []
        self.commands = []

    def note(self, step, **fields):
        entry = {"at": now(), "step": step}
        entry.update(fields)
        self.entries.append(entry)
        print("[%s] %s%s" % (now(), step, "".join(
            " %s=%s" % (k, v) for k, v in fields.items())))

    def run(self, step, cmd, env=None, timeout=300, check=True):
        started = time.monotonic()
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        elapsed = time.monotonic() - started
        tail = (proc.stdout or "")[-2000:] + (proc.stderr or "")[-2000:]
        self.commands.append({"step": step, "argv": cmd, "returncode": proc.returncode,
                              "seconds": round(elapsed, 3), "output_tail": tail})
        if check and proc.returncode != 0:
            raise SystemExit("%s failed (%d): %s" % (step, proc.returncode, tail[-800:]))
        return proc


BUSY_EXECUTABLES = ("llama-server", "llama-bench", "llama-cli", "bench-rkva",
                    "test-rkva", "ninja", "cmake")
# Interpreter-like argv[0] values may carry a whole prompt on their command
# line (this agent's own session does), so only a real executable basename can
# make the guard fire.
BUSY_PREFIXES = ("bench-rkva", "test-rkva")


def busy_processes():
    """Host benchmark/toolchain processes, matched by executable name only.

    `pgrep -f` would match the task text carried by an agent or shell command
    line; this walks /proc and compares the basename of argv[0] instead, which
    is what the Codex benchmark and a build actually run as.
    """
    found = []
    proc_root = "/proc"
    if not os.path.isdir(proc_root):
        return found
    mine = os.getpid()
    for entry in os.listdir(proc_root):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == mine:
            continue
        try:
            with open(os.path.join(proc_root, entry, "cmdline"), "rb") as handle:
                raw = handle.read()
        except OSError:
            continue
        if not raw:
            continue
        argv0 = raw.split(b"\0", 1)[0].decode("utf-8", "replace")
        name = os.path.basename(argv0)
        if name in BUSY_EXECUTABLES or name.startswith(BUSY_PREFIXES):
            try:
                with open(os.path.join(proc_root, entry, "cmdline"), "rb") as handle:
                    full = handle.read().replace(b"\0", b" ").decode("utf-8", "replace")
            except OSError:
                full = name
            found.append("pid=%d %s" % (pid, full.strip()[:200]))
    return found


def deploy_path():
    return XBOX_ROOT / "scripts" / "deploy.sh"


def device_env(args):
    env = dict(os.environ)
    env.setdefault("XBOX_IP", args.host)
    missing = [k for k in ("XBOX_IP", "XBOX_USER", "XBOX_PASS") if not env.get(k)]
    if missing:
        raise SystemExit("device credentials missing: " + ", ".join(
            missing) + " (source ~/.config/xllama/xbox-env)")
    return env


def deploy(journal, env, *argv, check=True, timeout=300):
    return journal.run("deploy:" + argv[0], [str(deploy_path())] + [str(a) for a in argv],
                       env=env, timeout=timeout, check=check)


def localstate_exists(journal, env, pfn, name, timeout=120):
    """True when deploy.sh can fetch `name` from LocalState (bounded, one call)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / name
        proc = deploy(journal, env, "fetch-file", pfn, name, str(out), check=False,
                      timeout=timeout)
        return proc.returncode == 0 and out.exists() and out.stat().st_size > 0


def wait_for(journal, label, predicate, deadline_seconds, poll_seconds=3.0):
    started = time.monotonic()
    while time.monotonic() - started < deadline_seconds:
        if predicate():
            journal.note("wait:ok", label=label,
                         seconds=round(time.monotonic() - started, 1))
            return True
        time.sleep(poll_seconds)
    journal.note("wait:timeout", label=label, seconds=deadline_seconds)
    return False


def port_open(host, port, timeout=2.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def write_flag(directory, name, text=""):
    path = pathlib.Path(directory) / name
    path.write_text(text, encoding="utf-8")
    return path


def run_gpupeak(journal, args, env, pfn, work):
    if args.gpupeak_result:
        document = json.loads(pathlib.Path(args.gpupeak_result).read_text(encoding="utf-8"))
        journal.note("gpupeak:reuse", path=args.gpupeak_result,
                     rows=len(document.get("rows") or []))
        return document

    for stale in ("gpupeak-result.json", "gpupeak-result.json.done"):
        deploy(journal, env, "delete-file", pfn, stale, check=False)
    deploy(journal, env, "stop-app", pfn, check=False)

    flag = write_flag(work, "gpupeak.flag")
    deploy(journal, env, "upload-file", str(flag), pfn, "", "gpupeak.flag")
    if args.gpupeak_opts:
        opts = pathlib.Path(args.gpupeak_opts)
        deploy(journal, env, "upload-file", str(opts), pfn, "", "gpupeak-opts.txt")
    deploy(journal, env, "start-app", pfn)

    if not wait_for(journal, "gpupeak-result.json.done",
                    lambda: localstate_exists(journal, env, pfn, "gpupeak-result.json.done"),
                    args.gpupeak_timeout):
        deploy(journal, env, "stop-app", pfn, check=False)
        raise SystemExit("gpupeak did not produce a result: the installed package likely "
                         "predates the gpupeak headless mode")
    out = work / "gpupeak-result.json"
    deploy(journal, env, "fetch-file", pfn, "gpupeak-result.json", str(out))
    deploy(journal, env, "stop-app", pfn, check=False)
    document = json.loads(out.read_text(encoding="utf-8"))
    journal.note("gpupeak:result", rows=len(document.get("rows") or []),
                 d3d12_ran=document.get("d3d12_ran"))
    return document


def run_sweep(journal, args, env, pfn, work):
    deploy(journal, env, "stop-app", pfn, check=False)
    api_flag = write_flag(work, "api.flag")
    deploy(journal, env, "upload-file", str(api_flag), pfn, "", "api.flag")
    deploy(journal, env, "start-app", pfn)
    if not wait_for(journal, "rkva:%d" % args.port, lambda: port_open(args.host, args.port),
                    args.listen_timeout):
        deploy(journal, env, "stop-app", pfn, check=False)
        raise SystemExit("RKVA listener did not open on %s:%d" % (args.host, args.port))

    sweep_json = work / "pass1-sweep.json"
    cmd = [sys.executable, str(TESTS / "rkva_pass1_sweep.py"), "--execute",
           "--host", args.host, "--port", str(args.port),
           "--rows", args.rows, "--nq", args.nq,
           "--origin", str(args.origin), "--tail", str(args.tail),
           "--runs", str(args.runs), "--warmup", str(args.warmup),
           "--output", str(sweep_json), "--allow-busy"]
    journal.run("sweep", cmd, env=env, timeout=args.sweep_timeout)
    if not args.keep_app:
        deploy(journal, env, "stop-app", pfn, check=False)
    return json.loads(sweep_json.read_text(encoding="utf-8"))


def plan_text(args):
    env_hint = "<XBOX_USER>:<XBOX_PASS> from ~/.config/xllama/xbox-env"
    return json.dumps({
        "execute": False,
        "package": "beellama-xbox-<version>-llamacpp.msix (pre-installed; this script never builds)",
        "requires": ["gpupeak.flag headless mode", "api.flag RKVA listener on :50054",
                     "XBOX_IP/XBOX_USER/XBOX_PASS in the environment",
                     "console leased by the caller; no host benchmark running"],
        "device": {"host": args.host, "port": args.port, "env": env_hint},
        "phases": [
            {"name": "gpupeak", "flag": "gpupeak.flag",
             "localstate_out": ["gpupeak-result.json", "gpupeak-result.json.done"],
             "opts": args.gpupeak_opts or "(default plan: 256 MiB stream, 5 iterations)",
             "timeout_seconds": args.gpupeak_timeout},
            {"name": "pass1-sweep",
             "command": "python3 tools/xbox/tests/rkva_pass1_sweep.py --execute --host %s "
                        "--rows %s --nq %s --origin %d --tail %d --runs %d" %
                        (args.host, args.rows, args.nq, args.origin, args.tail, args.runs),
             "timeout_seconds": args.sweep_timeout}],
        "result": str(args.output),
        "busy_processes_now": busy_processes()[:8],
        "note": "run with --execute inside the hardware window; package build/install is "
                "the reviewer's step, not this script's",
    }, indent=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--host", default="192.168.1.200")
    parser.add_argument("--port", type=int, default=50054)
    parser.add_argument("--pfn", help="package full name (default: deploy.sh pfn)")
    parser.add_argument("--expect-pfn", help="refuse a different installed package")
    parser.add_argument("--gpupeak-opts", help="local key=value file -> gpupeak-opts.txt")
    parser.add_argument("--gpupeak-result", help="reuse an existing gpupeak JSON")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--rows", default="28672,16384,8192,4096")
    parser.add_argument("--nq", default="1,8,64,256")
    parser.add_argument("--origin", type=int, default=102400)
    parser.add_argument("--tail", type=int, default=128, choices=(0, 128))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--gpupeak-timeout", type=int, default=900)
    parser.add_argument("--listen-timeout", type=int, default=180)
    parser.add_argument("--sweep-timeout", type=int, default=3600)
    parser.add_argument("--keep-app", action="store_true",
                        help="leave the app running after the sweep (default: stop it)")
    parser.add_argument("--output", default=str(DEFAULT_RESULT))
    parser.add_argument("--journal", help="write the command journal here")
    args = parser.parse_args()

    if not args.execute:
        print(plan_text(args))
        return 0

    busy = busy_processes()
    if busy:
        raise SystemExit("refusing --execute: benchmark processes are active:\n  " +
                         "\n  ".join(busy[:8]))
    env = device_env(args)
    journal = Journal()
    journal.note("start", host=args.host, output=args.output)

    pfn = args.pfn
    if not pfn:
        proc = deploy(journal, env, "pfn")
        pfn = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if not pfn:
        raise SystemExit("could not resolve the installed package full name")
    if args.expect_pfn and pfn != args.expect_pfn:
        raise SystemExit("installed package %s != expected %s" % (pfn, args.expect_pfn))
    journal.note("package", pfn=pfn)

    work = pathlib.Path(tempfile.mkdtemp(prefix="xbox-ceiling-"))
    result = {"schema": "xbox-ceiling-run-v1", "started_utc": now(),
              "package_full_name": pfn, "host": args.host, "port": args.port}
    try:
        gpupeak = run_gpupeak(journal, args, env, pfn, work)
        result["gpupeak"] = gpupeak
        sweep = {"points": []}
        if not args.skip_sweep:
            sweep = run_sweep(journal, args, env, pfn, work)
            result["sweep"] = sweep
        roofline = roof.build_result(gpupeak, sweep)
        result["roofline"] = roofline
        ceiling = roofline["ceiling_measured"]
        journal.note("roofline",
                     bandwidth_gbs=round(ceiling["bandwidth_gbs"], 2),
                     fp32_tflops=round(ceiling["fp32_tflops"], 3),
                     fp16_tflops=round(ceiling["fp16_tflops"], 3),
                     dot4_i8_tops=round(ceiling["dot4_i8_tops"], 3),
                     dual_cu=roofline["compute_units"]["dual_cu"],
                     pass1_points=len(roofline["pass1_points"]))
        status = 0
    finally:
        result["journal"] = journal.entries
        result["commands"] = journal.commands
        result["finished_utc"] = now()
        out = pathlib.Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        if args.journal:
            pathlib.Path(args.journal).write_text(json.dumps(
                {"entries": journal.entries, "commands": journal.commands}, indent=1) + "\n",
                encoding="utf-8")
        print("wrote %s" % out)
        shutil.rmtree(work, ignore_errors=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
