import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile


binary, summary = map(Path, sys.argv[1:])


def run(root, name, args, profile=False, lookahead=False):
    path = root / name
    env = dict(os.environ, GGML_RKVA_TRACE_FILE=str(path),
               LLAMA_PREFILL_PROFILE=str(int(profile)), LLAMA_PLE_PREFETCH=str(int(lookahead)),
               LLAMA_PLE_PREFETCH_TRACE="1")
    log = path.with_suffix(".log")
    with log.open("w") as out:
        result = subprocess.run([binary, *args, "-s", "12345"], env=env, stdout=out, stderr=subprocess.STDOUT)
    assert result.returncode == 0, log.read_text()
    trace = list(root.glob(name + ".*.jsonl"))
    assert trace
    events = [json.loads(line) for p in trace for line in p.read_text().splitlines()]
    assert not any(e.get("dropped") for e in events)
    return log, events


def summarize(*args):
    return json.loads(subprocess.check_output([sys.executable, summary, *args], text=True))


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    fixture = ["--test-ple-prefetch", str(root / "qwen4exp.gguf")]
    off, off_events = run(root, "off", fixture)
    on, on_events = run(root, "on", fixture, lookahead=True)
    for log in (off, on):
        assert "[PREFILL " not in log.read_text()
    assert not any(e.get("phase") == "strata_ple_lookahead" for e in off_events)
    ahead = [e for e in on_events if e.get("phase") == "strata_ple_lookahead"]
    # Two contiguous upcoming ubatches, repeated after restore, only with lazy mode on.
    assert [(e["first_pos"], e["nq"], e["value"]) for e in ahead] == [(24, 16, 56), (40, 9, 28)] * 2
    assert all(e["bytes"] > 0 for e in ahead)
    reads = [e for e in on_events if e.get("phase") == "strata_ple_prefetch"]
    assert {e["flags"] for e in reads} == {0, 1}

    mtp, _ = run(root, "profile", ["--test-mtp-qwen4exp"], profile=True)
    totals = summarize("--log", mtp)
    assert totals["PREP/0"]["nextn"] > 0
    assert totals["PREP/1"]["samples"] > 0
    assert totals["NEXTN/0"]["rows"] == totals["NEXTN/0"]["nq"]
    assert totals["NEXTN/0"]["bytes"] > 0 and totals["NEXTN/0"]["host"] == totals["NEXTN/0"]["samples"]
    assert all(totals[f"CLASS/{name}"]["nodes"] > 0 for name in ("gdn", "qsa_fa", "moe"))
    assert sum(totals[key]["cpu_us"] for key in totals if key.startswith("CLASS/")) > 0
    assert totals["SCHED/all"]["status"] == 0 and totals["SCHED/all"]["synchronized"] == 0
    assert totals["SYNC/0"]["queued"] >= 40
    trace = summarize("--trace", *root.glob("on.*.jsonl"))
    assert trace["strata_ple_lookahead/flags=0"]["samples"] == len(ahead)
    replay = root / "replay.jsonl"
    with replay.open("w") as out:
        for nq in (1, 2, 3):
            for phase, value in (("strata_verify_ubatch", 0), ("cuda_graph_decision", 10), ("cuda_local_span_gpu", 2)):
                out.write(json.dumps(dict(phase=phase, graph_run=nq, flags=0, nq=nq,
                                          value=value, gpu_us=23, begin_us=1, end_us=2)) + "\n")
    totals = summarize("--trace", replay)
    for nq in (1, 2, 3):
        assert totals[f"cuda_graph_decision/ctx=0/nq={nq}/value=10"]["samples"] == 1
        assert totals[f"cuda_local_span_gpu/ctx=0/nq={nq}/value=2"]["p95_gpu_us"] == 23
        assert totals[f"strata_verify_ubatch/ctx=0/nq={nq}"]["p50_host_us"] == 1


    with sqlite3.connect(root / "nsys.sqlite") as db:
        db.executescript("""
            CREATE TABLE StringIds(id INT, value TEXT);
            INSERT INTO StringIds VALUES (1, 'gated_delta_net'), (2, 'flash_attn'),
                (3, 'mul_mat_q'), (4, 'cudaEventSynchronize'), (5, 'normalization');
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INT, end INT, deviceId INT, demangledName INT);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
                (10, 30, 0, 1), (20, 40, 0, 2), (40, 50, 0, 3), (50, 60, 0, 5), (101, 130, 0, 1);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INT, end INT, deviceId INT, streamId INT, copyKind INT, bytes INT);
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (30, 45, 0, 9, 1, 256), (55, 75, 0, 10, 2, 64);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INT, end INT, nameId INT);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (50, 75, 4);
        """)
    gpu = summarize("--sqlite", root / "nsys.sqlite", "--start-ns", "0", "--end-ns", "100")
    assert gpu["timeline/0"] == {"kernel_busy_ns": 50, "copy_outside_kernel_ns": 15, "idle_ns": 35}
    assert gpu["copy/0/9/h2d"]["bytes"] == 256 and gpu["copy/0/10/d2h"]["bytes"] == 64
    assert gpu["runtime/cudaEventSynchronize"]["host_ns"] == 25
    assert gpu["kernel/0/gdn"]["samples"] == gpu["kernel/0/mmq_moe_candidates"]["samples"] == 1
