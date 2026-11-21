import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def run(root, enabled):
    env = os.environ.copy()
    env.update(
        GGML_RKVA_TRACE_FILE=str(root / ("ab-on" if enabled else "ab-off")),
        LLAMA_CUDA_VERIFY_TRACE=str(int(enabled)),
        LLAMA_MTP_HIDDEN_TRACE=str(int(enabled)),
        LLAMA_QSA_INDEXER_BOUNDS=str(int(enabled)),
        LLAMA_GDN_CONV_L2=str(int(enabled)),
        LLAMA_QSA_FA_TRACE="1",
        LLAMA_PLE_PREFETCH_TRACE="1",
        LLAMA_MTP_KV_STORE_ONLY="1",
    )
    subprocess.run([sys.argv[1], "--test-mtp-qwen4exp", "-s", "12345"], env=env, check=True)
    if enabled:
        subprocess.run([sys.argv[1], "--test-qsa-exact-tail", "-s", "12345"], env=env, check=True)
    files = list(root.glob(Path(env["GGML_RKVA_TRACE_FILE"]).name + "*.jsonl"))
    assert files, "no trace file"
    return [json.loads(line) for path in files for line in path.read_text().splitlines()]


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    off = run(root, False)
    on = run(root, True)


def phase(events, name):
    return [e for e in events if e.get("phase") == name]


for name in ["strata_verify_ubatch", "strata_nextn_read", "strata_nextn_wait", "strata_mtp_state_write"]:
    assert not phase(off, name), name
    assert phase(on, name), name

widths = phase(on, "strata_verify_ubatch")
assert {e["nq"] for e in widths} == {1, 2, 3}
graph_runs = {e["graph_run"] for e in phase(on, "graph_host")}
assert all(e["graph_run"] in graph_runs and e["end_us"] >= e["begin_us"] for e in widths)

for name in ["strata_nextn_read", "strata_mtp_state_write"]:
    assert all(e["nq"] > 0 and e["bytes"] % (4 * e["nq"]) == 0 and e["bytes"] > 0
               for e in phase(on, name))
assert any(e["flags"] & 2 == 0 for e in phase(on, "strata_nextn_read"))
assert any(e["flags"] & 2 != 0 for e in phase(on, "strata_nextn_read"))

off_inputs = phase(off, "strata_qsa_pool_input")
on_inputs = phase(on, "strata_qsa_pool_input")
assert off_inputs and all(e["flags"] == 0 for e in off_inputs)
assert any(e["flags"] == 1 for e in on_inputs)
assert all(e["bytes"] == 8 * e["nq"] for e in on_inputs if e["flags"] == 1)

ple = phase(on, "strata_ple_prefetch")
assert ple and all(e["flags"] == 0 and e["bytes"] == 0 and e["value"] >= e["nq"] for e in ple)
assert all(e.get("dropped", 0) == 0 for e in off + on)
print("Trace verified for widths 1/2/3, NextN reads/waits, MTP state writes, compact QSA and in-memory PLE")
