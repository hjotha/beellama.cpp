#!/usr/bin/env python3
"""Monitor memory telemetry for an existing PID or an exact launched command.

This utility does not send requests or change GPU settings. Samples and summary
are local JSON artifacts. The campaign runner can also import
``TelemetryMonitor`` from ``campaign.py`` and control it around its own phases.

Examples:
  python3 tools/occupancy-f5/telemetry_monitor.py --pid 1234 --output run.jsonl
  python3 tools/occupancy-f5/telemetry_monitor.py --output run.jsonl --command -- ./server --config profile.ini
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from campaign import TelemetryMonitor, retained_growth_observation, summarize_telemetry


def pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pid", type=int, help="sample this existing process until it exits")
    source.add_argument("--command", nargs=argparse.REMAINDER,
                        help="launch exact argv in an owned process group; place argv after --")
    parser.add_argument("--output", required=True, type=Path, help="new telemetry JSONL path")
    parser.add_argument("--interval", type=float, default=1.0, help="sampling interval in seconds")
    parser.add_argument("--duration", type=float,
                        help="optional maximum duration; supported only with --pid")
    parser.add_argument("--nvidia-smi", default="nvidia-smi", help="nvidia-smi executable")
    parser.add_argument("--amd-sysfs-device", action="append", default=[],
                        help="explicit DRM cardN/device sysfs path; repeat per device")
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or not args.interval > 0 or (
        args.duration is not None and (not math.isfinite(args.duration) or not args.duration > 0)
    ):
        parser.error("interval and duration must be positive")
    if args.command is not None and args.duration is not None:
        parser.error("--duration applies only to --pid; launched commands run until exit or interrupt")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    command = args.command
    if command is not None and command and command[0] == "--":
        command = command[1:]
    if command is not None and not command:
        parser.error("--command needs an executable argv after --")

    process: subprocess.Popen[bytes] | None = None
    pid = args.pid
    if command is not None:
        try:
            process = subprocess.Popen(command, start_new_session=True)
        except OSError as exc:
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
            return 2
        pid = process.pid

    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = {"nvidia_smi_argv": [args.nvidia_smi],
              "amd_sysfs_devices": args.amd_sysfs_device}
    monitor = TelemetryMonitor(args.output, int(pid), config, args.interval).start()
    started = time.monotonic()
    interrupted = False
    try:
        while True:
            if args.duration is not None and time.monotonic() - started >= args.duration:
                break
            if process is not None:
                if process.poll() is not None:
                    break
            elif not pid_exists(int(pid)):
                break
            time.sleep(min(args.interval, 0.25))
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if interrupted and process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        metadata = monitor.stop()
    try:
        metadata["summary"] = summarize_telemetry(args.output)
        metadata["summary"]["retained_growth_observation"] = retained_growth_observation(args.output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        metadata["summary_error"] = f"{type(exc).__name__}: {exc}"
        metadata["complete"] = False
    metadata.update({"target_pid": pid, "command_argv": command,
                     "target_exit_code": process.returncode if process is not None else None})
    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"samples": str(args.output), "metadata": str(metadata_path),
                      "sample_count": metadata["sample_count"],
                      "complete": metadata["complete"]}, indent=2))
    if interrupted:
        return 130
    if process is not None and process.returncode not in (0, None):
        return process.returncode
    return 0 if metadata["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
