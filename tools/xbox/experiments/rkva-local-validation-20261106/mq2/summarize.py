"""Write a fresh CPU result summary without overwriting historical evidence."""
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import subprocess
import sys
import numpy as np
from output_paths import ROOT, RESULTS

names = ["gqa2_layout_results", "gqa2_policy_results", "mq2_masks_results",
         "mq2_results", "negative_controls_masks", "error_budget_results"]
data = {name: json.loads((RESULTS / (name + ".json")).read_text()) for name in names}
for name in names[:-1]:
    assert data[name]["success"], name
assert data["error_budget_results"]["official_gate_unchanged"]["ok"]
numeric = data["mq2_results"]
assert (numeric["numeric_cases"], numeric["numeric_runs"]) == (40, 104)
assert not numeric["contract"]
assert data["gqa2_layout_results"]["tests_run"] == 6
assert all(item["detected"] for item in numeric["negative_controls"])
assert len(numeric["negative_controls"]) == 1
summary = {
    "success": True,
    "record_kind": "fresh portable-package CPU validation",
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "full_attention_cpu_proposal": {k: v for k, v in numeric.items() if k != "runs"},
    "two_local_head_bitwise_comparisons": sum(
        r.get("bitwise_equal_to_heads6", False) for r in numeric["runs"]),
    "error_budget_replay": data["error_budget_results"],
    "gqa2_layout": data["gqa2_layout_results"],
    "gqa2_policy": data["gqa2_policy_results"],
    "hlsl_arithmetic_masks": data["mq2_masks_results"],
    "negative_controls_masks": data["negative_controls_masks"],
    "scope": "Historical bundled fixtures and CPU proposal, not current-main integration, HLSL compilation or GPU performance.",
}
artifacts = []
for p in sorted(ROOT.rglob("*")):
    rel = p.relative_to(ROOT)
    if p.is_file() and not any(part in (".build", "__pycache__", "results") for part in rel.parts):
        artifacts.append({"path": rel.as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
manifest = {
    "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
    "compiler": subprocess.check_output([os.environ.get("CXX", "g++"), "--version"], text=True).splitlines()[0],
    "command": "sh run_all.sh", "test_artifacts": artifacts,
    "provenance": "provenance.json", "scope": summary["scope"],
}
(RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"PASS: {numeric['numeric_cases']} cases / {numeric['numeric_runs']} numerical runs; "
      f"{summary['two_local_head_bitwise_comparisons']} two-head comparisons.")
print("Fresh results: .build/results/summary.json and manifest.json")
