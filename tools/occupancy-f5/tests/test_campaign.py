"""CPU-only fixtures for the F5 campaign runner; no server or HTTP calls."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "campaign.py"
spec = importlib.util.spec_from_file_location("f5_campaign", SCRIPT)
campaign = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = campaign
assert spec.loader is not None
spec.loader.exec_module(campaign)


class CampaignParsingTests(unittest.TestCase):
    def test_nonstreaming_http_final_json_and_server_samples(self):
        response = {
            "timings": {"prompt_ms": 100.0, "predicted_ms": 20.0,
                        "prompt_n": 12, "predicted_n": 4,
                        "f5_per_token": {"token_ms": [2.0, 4.0, 1.0, 3.0],
                                         "ttft_ms": 120.0, "truncated": False}},
        }
        parsed = campaign.parse_completion(json.dumps(response).encode(), "application/json")
        result = campaign.extract_timing(parsed)
        self.assertEqual(result["server_token_ms"], [2.0, 4.0, 1.0, 3.0])
        self.assertEqual(result["server_token_p95_ms_nearest_rank"], 4.0)
        self.assertEqual(result["timings"]["predicted_n"], 4)
        self.assertEqual(result["prefill_ms"], 100.0)
        self.assertEqual(result["ttft_ms"], 120.0)

    def test_sse_uses_final_server_payload_not_event_cadence(self):
        stream = (
            b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
            b'data: {"choices":[],"timings":{"prompt_ms":50,"predicted_ms":10,"prompt_n":6,"predicted_n":2,'
            b'"f5_per_token":{"token_ms":[7.0,3.0],"ttft_ms":62.5,"truncated":false}}}\n\n'
            b'data: [DONE]\n\n'
        )
        parsed = campaign.parse_completion(stream, "text/event-stream")
        result = campaign.extract_timing(parsed)
        self.assertEqual(result["server_token_p95_ms_nearest_rank"], 7.0)
        self.assertEqual(len(result["server_token_ms"]), 2)

    def test_missing_or_invalid_server_samples_are_not_synthesized(self):
        base = {"timings": {"prompt_ms": 1, "predicted_ms": 2,
                            "prompt_n": 1, "predicted_n": 1}}
        result = campaign.extract_timing(base)
        self.assertIsNone(result["server_token_ms"])
        self.assertIn("no f5_per_token", result["server_token_metrics_missing"])
        with self.assertRaisesRegex(ValueError, "nonempty finite"):
            campaign.p95_nearest_rank([])

    def test_measured_runs_require_32_non_ttft_samples_but_warmups_do_not(self):
        short = campaign.extract_timing({
            "timings": {"prompt_ms": 10, "predicted_ms": 2, "prompt_n": 2,
                        "predicted_n": 4,
                        "f5_per_token": {"token_ms": [1.0, 2.0, 3.0], "ttft_ms": 8}},
        })
        self.assertNotIn("server_token_metrics_missing",
                         campaign.enforce_measured_samples(dict(short), warmup=True))
        measured = campaign.enforce_measured_samples(dict(short), warmup=False)
        self.assertIn("requires at least 32", measured["server_token_metrics_missing"])

    def test_health_requires_200_and_retries_404(self):
        class FakeResponse:
            def __init__(self, status):
                self.status = status
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False

        class FakeProcess:
            def poll(self):
                return None

        with mock.patch.object(campaign.urllib.request, "urlopen",
                               side_effect=[FakeResponse(404), FakeResponse(200)]) as opened, \
             mock.patch.object(campaign.time, "sleep", return_value=None):
            campaign._health("http://127.0.0.1:8080/health", FakeProcess(),
                             campaign.time.monotonic() + 5)
        self.assertEqual(opened.call_count, 2)

    def test_alternating_schedule_has_both_warmups_then_five_pairs(self):
        rows = campaign.schedule()
        self.assertEqual(len(rows), 12)
        self.assertEqual([(r["arm"], r["warmup"]) for r in rows[:2]],
                         [("baseline", True), ("candidate", True)])
        measured = rows[2:]
        self.assertEqual([r["pair"] for r in measured], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        self.assertEqual([r["arm"] for r in measured], [
            "baseline", "candidate", "baseline", "candidate", "baseline",
            "candidate", "baseline", "candidate", "baseline", "candidate",
        ])
        for pair in range(5):
            self.assertEqual({r["arm"] for r in measured if r["pair"] == pair},
                             {"baseline", "candidate"})

    def test_memory_sample_reads_procfs_nvidia_and_amd_sysfs_fixtures(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            proc = root / "proc"
            (proc / "123").mkdir(parents=True)
            (proc / "123/status").write_text("Name:\ttest-server\nVmRSS:\t4096 kB\n")
            (proc / "meminfo").write_text(
                "MemAvailable: 2048 kB\nSwapTotal: 8192 kB\nSwapFree: 4096 kB\n")
            (proc / "vmstat").write_text("pswpin 3\npswpout 5\n")
            drm = root / "drm"
            device = drm / "card0/device"
            device.mkdir(parents=True)
            for name, value in {
                "mem_info_vram_used": "1000", "mem_info_vram_total": "4000",
                "mem_info_gtt_used": "2000", "mem_info_gtt_total": "8000",
            }.items():
                (device / name).write_text(value)

            def fake_command(argv, **kwargs):
                if any(arg.startswith("--query-gpu=") for arg in argv):
                    return subprocess.CompletedProcess(argv, 0, "2026/10/12 10:00:00, 128, 256\n", "")
                return subprocess.CompletedProcess(argv, 0, "123, 64\n", "")

            sample = campaign.sample_telemetry(
                123, {"nvidia_smi_argv": ["fake-nvidia-smi"]},
                proc_root=proc, drm_root=drm, command_runner=fake_command)
            self.assertEqual(sample["host"]["process_rss_bytes"], 4096 * 1024)
            self.assertEqual(sample["host"]["mem_available_bytes"], 2048 * 1024)
            self.assertEqual(sample["host"]["swap_in_pages"], 3)
            self.assertEqual(sample["cuda"]["devices"][0]["used_bytes"], 128 * 1024 * 1024)
            self.assertEqual(sample["cuda"]["server_process_vram_used_bytes"], 64 * 1024 * 1024)
            self.assertEqual(sample["vulkan"]["devices"][0]["gtt_used_bytes"], 2000)

    def test_monitor_persists_timestamped_samples_and_interval(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "telemetry.jsonl"
            def fixture_sampler(pid, config):
                return {"timestamp_utc": "2026-10-12T10:00:00+00:00", "monotonic_ns": 1,
                        "host": {"available": True, "process_rss_bytes": 1024,
                                 "mem_available_bytes": 4096, "swap_used_bytes": 0,
                                 "swap_in_pages": 0, "swap_out_pages": 0},
                        "cuda": {"available": False, "devices": []},
                        "vulkan": {"available": False}}
            monitor = campaign.TelemetryMonitor(path, 1, {}, 0.01, fixture_sampler).start()
            monitor.mark("retained_cycle_1")
            time.sleep(0.035)
            result = monitor.stop()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertGreaterEqual(result["sample_count"], 1)
            self.assertEqual(result["interval_seconds"], 0.01)
            self.assertEqual(len(rows), result["sample_count"])
            self.assertIn("timestamp_utc", rows[0])
            summary = campaign.summarize_telemetry(path)
            self.assertEqual(summary["host"]["process_rss_peak_bytes"], 1024)
            self.assertEqual(summary["host"]["mem_available_min_bytes"], 4096)
            self.assertIn("retained_cycle_1", [row.get("phase_marker") for row in rows])

    def test_oom_zero_requires_complete_logs_response_and_controlled_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            log = root / "server.log"
            response = root / "response.raw"
            log.write_text("server ready\n")
            response.write_text('{"ok":true}')
            complete = campaign.assess_oom_evidence(log, response, http_status=200, exit_code=0)
            self.assertTrue(complete["complete"])
            self.assertEqual(complete["oom_count"], 0)
            incomplete = campaign.assess_oom_evidence(log, response, http_status=None, exit_code=-9)
            self.assertFalse(incomplete["complete"])
            self.assertIsNone(incomplete["oom_count"])
            log.write_text("CUDA error: out of memory\n")
            observed = campaign.assess_oom_evidence(log, response, http_status=500, exit_code=-9)
            self.assertEqual(observed["oom_count"], 1)
            self.assertFalse(observed["complete"])
            from_error = campaign.assess_oom_evidence(
                root / "missing.log", response, http_status=None, exit_code=-9,
                runner_error="CUDA_ERROR_OUT_OF_MEMORY: allocation failed")
            self.assertEqual(from_error["oom_count"], 1)
            self.assertFalse(from_error["complete"])

    def test_drm_fdinfo_units_and_duplicate_client_fds_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as temp:
            fdinfo = Path(temp) / "123/fdinfo"
            fdinfo.mkdir(parents=True)
            first = (
                "pos:\t0\nflags:\t02000002\n"
                "drm-driver:\tamdgpu\ndrm-pdev:\t0000:03:00.0\ndrm-client-id:\t77\n"
                "drm-memory-vram:\t10 MiB\ndrm-memory-gtt:\t20 MiB\n"
                "drm-total-vram:\t12 MiB\ndrm-total-gtt:\t25 MiB\n"
            )
            # Same drm-client-id, different fd: values must be counted once.
            (fdinfo / "3").write_text(first)
            (fdinfo / "4").write_text(first)
            second = first.replace("client-id:\t77", "client-id:\t88").replace(
                "20 MiB", "30 MiB").replace("25 MiB", "35 MiB")
            (fdinfo / "5").write_text(second)
            result = campaign.sample_process_drm(123, Path(temp))
            self.assertTrue(result["complete"])
            self.assertEqual(result["fd_count"], 3)
            self.assertEqual(result["unique_client_count"], 2)
            self.assertEqual(result["duplicate_fd_count"], 1)
            self.assertEqual(result["memory_bytes_by_key"]["drm-memory-gtt"], 50 * 1024 * 1024)
            self.assertEqual(result["memory_bytes_by_key"]["drm-total-gtt"], 60 * 1024 * 1024)
            self.assertEqual(result["fd_records"][0]["raw"], first)

    def test_drm_fdinfo_permission_errors_mark_totals_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            fdinfo = Path(temp) / "123/fdinfo"
            fdinfo.mkdir(parents=True)
            (fdinfo / "3").write_text("drm-driver: amdgpu\ndrm-client-id: 7\ndrm-memory-gtt: 100 MiB\n")
            (fdinfo / "4").write_text("")
            def reader(path):
                if path.name == "4":
                    raise PermissionError("fixture permission denied")
                return path.read_text()
            result = campaign.sample_process_drm(123, Path(temp), fd_reader=reader)
            self.assertTrue(result["available"])
            self.assertFalse(result["complete"])
            self.assertIsNone(result["memory_bytes_by_key"]["drm-memory-gtt"])
            self.assertIn("PermissionError", result["read_errors"][0]["error"])

    def test_retained_growth_uses_pid_vram_and_deduplicated_process_gtt(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "telemetry.jsonl"
            first = {"phase_marker": "retained_cycle_1",
                     "cuda": {"server_process_vram_used_bytes": 100},
                     "vulkan": {"process": {"memory_bytes_by_key": {"drm-memory-gtt": 200}}}}
            last = {"phase_marker": "retained_cycle_10",
                    "cuda": {"server_process_vram_used_bytes": 300},
                    "vulkan": {"process": {"memory_bytes_by_key": {"drm-memory-gtt": 250}}}}
            path.write_text(json.dumps(first) + "\n" + json.dumps(last) + "\n")
            result = campaign.retained_growth_observation(path)
            self.assertEqual(result["status"], "observed")
            self.assertTrue(result["observation_only"])
            self.assertEqual(result["backends"]["cuda_process_vram"]["growth_bytes"], 200)
            self.assertEqual(result["backends"]["vulkan_process_gtt"]["growth_bytes"], 50)


if __name__ == "__main__":
    unittest.main()
