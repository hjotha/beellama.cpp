#!/usr/bin/env python3
"""Host tests for the layout-4/5 native-row profile correction.

Two independent checks:

1. Functional: compile the real, WinRT-free `rkva-range-profile.h` on the host,
   feed it the Samples that `finish_bulk_profile` composes for a Q8_0/Q4_0
   native-row envelope (and a KVarN4 envelope), and assert the resulting bucket
   keys, per-storage counters and additive JSON. This exercises the serializer
   and the shared `count_attention_sample` helper the engine now uses, so it is
   not a regex paraphrase of the source.
2. Structural: extract the exact function bodies of the edited engine and
   transport regions (brace matched) and assert the profiling lifecycle,
   per-segment field propagation and finalize-after-submit ordering.

Run:
    python3 tools/xbox/tests/test_rkva_native_rows_profile.py

The engine/transport sources themselves are UWP/D3D12 and cannot be compiled on
Linux; MSVC + Xbox validation of those edits remains a hardware task.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
XBOX = os.path.dirname(HERE)
RKVA = os.path.join(XBOX, "uwp", "rkva")
ENGINE_CPP = os.path.join(RKVA, "remote-attn-engine.cpp")
REMOTE_ATTN_CPP = os.path.join(RKVA, "remote-attn.cpp")
PROFILE_H = os.path.join(RKVA, "rkva-range-profile.h")


def read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def slice_between(text: str, start_anchor: str, end_anchor: str) -> str:
    """Return text[start_anchor:end_anchor] (first occurrence of end after start)."""
    start = text.index(start_anchor)
    end = text.index(end_anchor, start)
    return text[start:end]


def function_body(text: str, signature: str) -> str:
    """Return the brace-matched body of the first function containing signature."""
    start = text.index(signature)
    open_brace = text.index("{", start)
    depth = 0
    for index in range(open_brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace : index + 1]
    raise AssertionError(f"unbalanced braces after {signature}")


FIXTURE = r"""
#include "rkva-range-profile.h"

#include <cstdio>
#include <cstring>

using namespace rkva_rprof;

int main() {
    Recorder r;
    r.configure(64);

    // --- Layout-5 Q8_0 native-row envelope: upload + one reference attention.
    Sample up;
    up.kind = kUpload;
    up.storage = kStorageQ8;
    up.serial_prefix = 0;
    up.bulk_segment = 1;
    up.n_tokens = 256;
    up.rows = 256;
    up.bytes = 55296;
    up.cpu_ns[kCpuMapCopy] = 7000;
    up.cpu_ns[kCpuRecord] = 9000;
    up.gpu_ticks[0] = 100;
    up.gpu_valid = 1;
    r.record(up);
    ++r.counters.calls[kUpload];
    // Native layout-4/5 upload records one K and one V CopyBufferRegion.
    r.counters.copy_regions += bulk_upload_copy_regions(up.storage, true);

    Sample ref;
    ref.kind = kAttention;
    ref.storage = kStorageQ8;
    ref.serial_prefix = 0;
    ref.bulk_segment = 1;
    ref.n_tokens = 1;
    ref.rows = 4096;
    ref.split_k = 2;
    ref.groups = 8;
    ref.q_unpack_ns = 3000;
    ref.q_unpack_bytes = 6144;
    ref.cpu_ns[kCpuMapCopy] = 4000;
    ref.cpu_ns[kCpuRecord] = 5000;
    ref.cpu_ns[kCpuReadback] = 2000;
    ref.gpu_ticks[0] = 3000;
    ref.gpu_ticks[1] = 1000;
    ref.gpu_ticks[2] = 500;
    ref.gpu_valid = 1;
    r.record(ref);
    ++r.counters.calls[kAttention];
    ++r.counters.copy_regions; // attention readback copy
    count_attention_sample(r.counters, ref);

    // Q8_0 2-query tile (native rows): fmt_mq bucket, never a KVarN counter.
    Sample q8tile = ref;
    q8tile.optimized = 1;
    q8tile.q_tile = 2;
    q8tile.n_tokens = 2;
    r.record(q8tile);
    ++r.counters.calls[kAttention];
    ++r.counters.copy_regions;
    count_attention_sample(r.counters, q8tile);

    // Q4_0 4-query tile (native rows): q4_mq_calls[1].
    Sample q4 = q8tile;
    q4.storage = kStorageQ4;
    q4.q_tile = 4;
    r.record(q4);
    ++r.counters.calls[kAttention];
    ++r.counters.copy_regions;
    count_attention_sample(r.counters, q4);

    // KVarN4 serial-prefix 2-query tile: the only route allowed to move the
    // kvarn4_mq_* counters.
    Sample kv = q8tile;
    kv.storage = kStorageKvarn4;
    kv.serial_prefix = 1;
    kv.q_tile = 2;
    kv.q_unpack_ns = 0;
    kv.q_unpack_bytes = 0;
    r.record(kv);
    ++r.counters.calls[kAttention];
    ++r.counters.copy_regions;
    count_attention_sample(r.counters, kv);

    // Copy-region rule (storage-aware): native rows record K+V, KVarN keeps its
    // historical single region, query-only segments record none.
    if (bulk_upload_copy_regions(kStorageQ8, true) != 2) return 2;
    if (bulk_upload_copy_regions(kStorageQ4, true) != 2) return 3;
    if (bulk_upload_copy_regions(kStorageF16, true) != 1) return 4;
    if (bulk_upload_copy_regions(kStorageKvarn4, true) != 1) return 5;
    if (bulk_upload_copy_regions(kStorageQ8, false) != 0) return 6;

    // CPU Q expansion telemetry, exactly as finish_bulk_profile accumulates it.
    r.counters.cpu_q_unpack_calls = 1;
    r.counters.cpu_q_unpack_bytes = 6144;
    r.counters.cpu_q_unpack_ns = 3000;
    r.counters.native_rows_calls = 1;
    ++r.counters.bulk_calls;
    r.counters.bulk_segments = 1;
    r.counters.bulk_queries = 1;

    BulkEnvelopeSample envelope;
    envelope.cycle_id = 42;
    envelope.session_id = 7;
    envelope.layer = 3;
    envelope.wire = 4; // RKVA_WIRE_FP8
    envelope.storage = kStorageQ8;
    envelope.serial_prefix = 0;
    envelope.segment_count = 1;
    envelope.total_queries = 1;
    envelope.gpu_valid = 1;
    envelope.upload_bytes = 55296;
    envelope.query_bytes = 4096;
    envelope.output_bytes = 4096;
    envelope.q_unpack_ns = 3000;
    envelope.cpu_ns[kCpuMapCopy] = 4000;
    r.record_bulk(envelope);

    Header h;
    h.profiling = true;
    h.gpu_timestamps = true;
    h.gpu_timestamp_frequency = 1000000000ull;
    const std::string json = r.json(h);
    std::fputs(json.c_str(), stdout);
    return json.empty() ? 1 : 0;
}
"""


@unittest.skipUnless(shutil.which("g++"), "g++ is required for the profile fixture")
class ProfileJsonFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.mkdtemp(prefix="rkva-native-rows-")
        source = os.path.join(cls.tmp, "fixture.cpp")
        binary = os.path.join(cls.tmp, "fixture")
        with open(source, "w", encoding="utf-8") as handle:
            handle.write(FIXTURE)
        subprocess.run(
            ["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", RKVA,
             source, "-o", binary],
            check=True,
        )
        cls.json_text = subprocess.run(
            [binary], check=True, capture_output=True, text=True
        ).stdout
        cls.doc = json.loads(cls.json_text)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_native_row_buckets_use_real_storage(self) -> None:
        buckets = self.doc["buckets"]
        self.assertIn("upload/q8_0/nq129+/r1-256/warm", buckets)
        self.assertIn("attention/q8_0/reference/nq1/r1k-4k/warm", buckets)
        self.assertIn("attention/q8_0/optimized-q8_mq2_fine/nq2-16/r1k-4k/warm", buckets)
        self.assertIn("attention/q4_0/optimized-q4_mq4_balanced/nq2-16/r1k-4k/warm", buckets)
        self.assertIn("attention/kvarn4-prefix/optimized-kvarn4_mq2_fine/nq2-16/r1k-4k/warm",
                      buckets)
        # A native Q4_0/Q8_0 row segment is never reported as a KVarN prefix one.
        self.assertFalse(any("q8_0-prefix" in key or "q4_0-prefix" in key for key in buckets))

    def test_bucket_carries_cpu_q_unpack(self) -> None:
        bucket = self.doc["buckets"]["attention/q8_0/reference/nq1/r1k-4k/warm"]
        self.assertAlmostEqual(bucket["cpu_q_unpack_us"]["mean"], 3.0, places=3)
        self.assertAlmostEqual(bucket["q_unpack_bytes"]["mean"], 6144.0, places=3)
        # The KVarN bucket never claims a CPU Q expansion.
        kvarn = self.doc["buckets"][
            "attention/kvarn4-prefix/optimized-kvarn4_mq2_fine/nq2-16/r1k-4k/warm"]
        self.assertEqual(kvarn["cpu_q_unpack_us"]["n"], 1)
        self.assertAlmostEqual(kvarn["cpu_q_unpack_us"]["mean"], 0.0, places=3)

    def test_counters_route_isolation(self) -> None:
        counters = self.doc["counters"]
        # Q8 tile -> fmt_mq only; Q4 tile -> q4 counters; KVarN tile -> kvarn4.
        self.assertEqual(counters["optimized_q4_calls"], 1)
        self.assertEqual(counters["reference_q4_calls"], 0)
        optimized = self.doc["optimized_kernels"]
        self.assertEqual(optimized["q8_0"]["optimized_calls"], 1)
        self.assertEqual(optimized["q8_0"]["reference_calls"], 1)
        self.assertEqual(optimized["q4_0"]["optimized_calls"], 1)
        self.assertEqual(optimized["kvarn4"]["optimized_calls"], 1)
        fmt = self.doc["format_multi_query"]
        self.assertEqual(fmt["q8_0"]["mq2"]["calls"], 1)
        self.assertEqual(fmt["q8_0"]["mq4"]["calls"], 0)
        kvarn = self.doc["kvarn4_multi_query"]
        self.assertEqual(kvarn["mq2"]["calls"], 1)   # the single KVarN tile
        self.assertEqual(kvarn["mq4"]["calls"], 0)
        self.assertEqual(self.doc["kvarn4_decode"]["calls"], 0)
        self.assertEqual(self.doc["kvarn4_decode_rowlane"]["calls"], 0)
        q4_mq = self.doc["q4_multi_query"]
        self.assertEqual(q4_mq["mq2"]["calls"], 0)
        self.assertEqual(q4_mq["mq4"]["calls"], 1)

    def test_bulk_native_row_section_and_envelope_identity(self) -> None:
        bulk = self.doc["bulk"]
        self.assertEqual(bulk["native_rows_calls"], 1)
        self.assertEqual(bulk["native_rows_cpu_q_unpack"]["calls"], 1)
        self.assertEqual(bulk["native_rows_cpu_q_unpack"]["bytes"], 6144)
        self.assertAlmostEqual(bulk["native_rows_cpu_q_unpack"]["duration_us"], 3.0, places=3)
        self.assertAlmostEqual(bulk["timings"]["map_copy_cpu_us"]["mean"], 4.0, places=3)
        self.assertAlmostEqual(bulk["timings"]["engine_total_cpu_us"]["mean"], 7.0, places=3)
        # A native FP8 out wire must never be counted as a KVarN GPU/CPU codec.
        self.assertEqual(bulk["fp8_gpu_calls"], 0)
        self.assertEqual(bulk["fp8_cpu_fallback_calls"], 0)
        recent = bulk["recent_envelopes"]
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["cycle_id"], 42)
        self.assertEqual(recent[0]["storage"], 2)
        self.assertEqual(recent[0]["serial_prefix"], 0)

    def test_copy_region_accounting_rule(self) -> None:
        counters = self.doc["counters"]
        # One Q8 native upload contributes its two plane copies; attention
        # segments add their single readback copy each (4 samples in the fixture).
        self.assertEqual(counters["copy_regions"], 6)

    def test_schema_version_unchanged(self) -> None:
        self.assertEqual(self.doc["schema"], "rkva-range-profile-v1")


class EngineWiring(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = read(ENGINE_CPP)

    def test_native_rows_envelope_lifecycle(self) -> None:
        body = function_body(self.engine, "bool RemoteAttnEngine::attend_native_rows")
        self.assertIn("envelope.storage = uint8_t(storage);", body)
        self.assertIn("envelope.serial_prefix = 0;", body)
        self.assertIn("envelope.wire = segments.front().wire;", body)
        self.assertIn("envelope.cycle_id = cycle_id;", body)
        self.assertIn("profile_->bulk.active = true;", body)
        self.assertIn("profile_->bulk.segments.resize(segments.size());", body)
        self.assertIn("p.wire = uint8_t(seg.wire);", body)
        self.assertIn("p.has_upload = seg.append_rows ? 1 : 0;", body)
        self.assertIn("guard.done = true;", body)
        # Exactly the existing final submit; profiling never adds one.
        self.assertEqual(body.count("submit(error)"), 1)
        order = [
            body.index("profile_->bulk.active = true;"),
            body.index("bulk_recording_ = true;"),
            body.index("upload_range(id, layer"),
            body.index("attend_range(id, layer"),
            body.index("submit(error)"),
            body.index("finish_bulk_profile(envelope, ts_bulk);"),
        ]
        self.assertEqual(order, sorted(order))
        self.assertEqual(body.count("finish_bulk_profile(envelope, ts_bulk);"), 1)
        self.assertEqual(body.count("resolve_timestamps("), 1)
        self.assertIn("resolve_timestamps(0, total_slots);", body)
        self.assertNotIn("wait", body.replace("await", ""))

    def test_cycle_id_signatures_and_forwarding(self) -> None:
        header = read(os.path.join(RKVA, "remote-attn-engine.h"))
        for signature in ("attend_q4_bulk", "attend_q8_bulk"):
            decl = slice_between(header, f"bool {signature}(", ");")
            self.assertIn("uint64_t cycle_id", decl, f"{signature} lost the cycle_id parameter")
            self.assertNotIn("cycle_id = 0", decl, f"{signature} still defaults cycle_id")
        native = slice_between(header, "bool attend_native_rows(", ");")
        self.assertIn("uint64_t cycle_id", native)
        self.assertNotIn("cycle_id = 0", native)

        native_body = function_body(self.engine, "bool RemoteAttnEngine::attend_native_rows")
        self.assertIn("envelope.cycle_id = cycle_id;", native_body)
        q4 = function_body(self.engine, "bool RemoteAttnEngine::attend_q4_bulk")
        q8 = function_body(self.engine, "bool RemoteAttnEngine::attend_q8_bulk")
        self.assertIn("error, cycle_id);", q4)
        self.assertIn("error, cycle_id);", q8, "Q8 wrapper drops cycle_id")

        transport = read(REMOTE_ATTN_CPP)
        block = slice_between(
            transport,
            "const bool q8_rows = it->second.native_q8;",
            "        uint32_t wire = RKVA_WIRE_F32;",
        )
        self.assertEqual(block.count("&error, req.cycle_id)"), 3,
                         "Q4, Q8 CPU and Q8 GPU call sites must forward req.cycle_id")
        self.assertIn("attend_q8_bulk_fp8", block)

    def test_bulk_window_guard_armed_after_ownership(self) -> None:
        body = function_body(self.engine, "bool RemoteAttnEngine::attend_native_rows")
        self.assertEqual(body.count("BulkWindowGuard window{"), 1)
        refusal = body.index("bulk_recording_ ||")
        begin = body.index("if (!begin(error)) return false;")
        active = body.index("profile_->bulk.active = true;")
        guard = body.index("BulkWindowGuard window{")
        opened = body.index("bulk_recording_ = true;")
        # A refused call (already recording) returns before the guard exists, so
        # it cannot clear a foreign window; the owned window is armed only after
        # this call activates its own profiler/recording state.
        self.assertLess(refusal, begin)
        self.assertLess(begin, active)
        self.assertLess(active, guard)
        self.assertLess(guard, opened)

    def test_native_rows_records_cpu_q_unpack(self) -> None:
        body = function_body(self.engine, "bool RemoteAttnEngine::attend_native_rows")
        self.assertIn("p.q_unpack_ns += elapsed_ns(t_unpack", body)
        self.assertIn("p.q_unpack_bytes = elements", body)
        # The F32 wire is a memcpy passthrough and is excluded by the same guard.
        self.assertIn("(prof && seg.wire != RKVA_WIRE_F32)", body)

    def test_finish_bulk_profile_is_route_driven(self) -> None:
        body = function_body(self.engine, "void RemoteAttnEngine::finish_bulk_profile")
        self.assertIn("envelope.storage == rp::kStorageKvarn4", body)
        self.assertIn("kvarn_route", body)
        self.assertIn("++c.native_rows_calls;", body)
        self.assertIn("attn_sample.storage = seg.storage;", body)
        self.assertIn("attn_sample.serial_prefix = seg.serial_prefix;", body)
        self.assertIn("up_sample.storage = seg.storage;", body)
        self.assertIn("if (seg.has_upload)", body)
        self.assertIn("rp::count_attention_sample(c, attn_sample);", body)
        self.assertIn("attn_sample.q_unpack_ns = seg.q_unpack_ns;", body)
        self.assertIn("c.cpu_q_unpack_bytes += seg.q_unpack_bytes;", body)
        # Upload copy regions follow the real per-storage copy count.
        self.assertIn("rp::bulk_upload_copy_regions(seg.storage, seg.has_upload != 0);", body)
        self.assertEqual(body.count("c.copy_regions += 1;"), 1,
                         "only the attention readback copy is unconditional")
        # Finalize runs after the existing fence: no GPU command here.
        self.assertNotIn("timestamp(", body)
        self.assertNotIn("submit(", body)
        self.assertNotIn("up_sample.storage = 3;", body)
        self.assertNotIn("kvarn4_mq_calls", body)

    def test_upload_and_attention_fill_bulk_segments(self) -> None:
        upload = function_body(self.engine, "bool RemoteAttnEngine::upload_range")
        self.assertIn("const bool bulk_prof =", upload)
        self.assertIn("map_copy_upload_ns +=", upload)
        self.assertIn("seg.record_upload_ns +=", upload)
        self.assertIn("seg.upload_bytes = 2 * bytes;", upload)
        self.assertIn("seg.has_upload = 1;", upload)
        self.assertIn("kTsBulkUploadBegin", upload)
        self.assertIn("kTsBulkUploadEnd", upload)
        # The two plane copies the bulk accounting now charges for.
        self.assertEqual(upload.count("cmd_->CopyBufferRegion("), 2)

        attn = function_body(self.engine, "bool RemoteAttnEngine::attend_range")
        self.assertIn("seg.storage = uint8_t(s->range_storage);", attn)
        self.assertIn("seg.serial_prefix = prefix ? 1 : 0;", attn)
        self.assertIn("seg.optimized = 1;", attn)
        self.assertIn("seg.split_k = split_k;", attn)
        self.assertIn("seg.n_queries = n_tokens;", attn)

    def test_kvarn_callers_keep_explicit_route(self) -> None:
        body = function_body(self.engine, "bool RemoteAttnEngine::attend_kvarn_bulk")
        self.assertIn("envelope.storage = rp::kStorageKvarn4;", body)
        self.assertIn("envelope.serial_prefix = 1;", body)
        kvarn_upload = function_body(self.engine, "bool RemoteAttnEngine::upload_kvarn_range")
        self.assertIn("seg.storage = rp::kStorageKvarn4;", kvarn_upload)
        self.assertIn("seg.serial_prefix = 1;", kvarn_upload)
        # finish_profile_sample and finish_bulk_profile share one counter rule.
        self.assertEqual(self.engine.count("rp::count_attention_sample(c, sample);"), 1)


class GuardSemantics(unittest.TestCase):
    """Compiles the real BulkWindowGuard definition from the engine source.

    The engine itself is D3D12/UWP and cannot run here, but the guard is pure
    bookkeeping: extracting its exact definition and exercising it proves the
    owned-window cleanup without a stub copy.
    """

    @unittest.skipUnless(shutil.which("g++"), "g++ is required for the guard fixture")
    def test_owned_window_cleanup_and_foreign_window_preserved(self) -> None:
        engine = read(ENGINE_CPP)
        guard = "struct BulkWindowGuard " + function_body(engine, "struct BulkWindowGuard {")
        source = (
            "#include <cstdio>\n"
            "struct BulkSessionProfiling { bool active = false; };\n"
            + guard
            + ";\n"
            + r"""
int main() {
    // A refused call (already recording) never constructs the guard: the
    // foreign window keeps both flags.
    bool recording = true;
    BulkSessionProfiling foreign;
    foreign.active = true;
    if (!recording || !foreign.active) return 1;

    // The call that owns the window clears both on any exit.
    {
        BulkWindowGuard owned{&recording, &foreign};
    }
    if (recording || foreign.active) return 2;

    // Profiling-off windows own the recording flag only.
    bool recording_off = true;
    {
        BulkWindowGuard off{&recording_off, nullptr};
    }
    if (recording_off) return 3;
    std::puts("guard ok");
    return 0;
}
"""
        )
        tmp = tempfile.mkdtemp(prefix="rkva-guard-")
        try:
            path = os.path.join(tmp, "guard.cpp")
            binary = os.path.join(tmp, "guard")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(source)
            subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", path,
                            "-o", binary], check=True)
            out = subprocess.run([binary], check=True, capture_output=True, text=True).stdout
            self.assertIn("guard ok", out)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TransportTelemetry(unittest.TestCase):
    def setUp(self) -> None:
        self.text = read(REMOTE_ATTN_CPP)
        # The native-row branch ends where the KVarN bulk branch starts.
        self.block = slice_between(
            self.text,
            "const bool q8_rows = it->second.native_q8;",
            "        uint32_t wire = RKVA_WIRE_F32;",
        )

    def test_native_rows_out_pack_is_separate(self) -> None:
        block = self.block
        self.assertIn("state.native_rows_pack_calls", block, "pack counter missing")
        self.assertIn("state.native_rows_pack_bytes += total * out_elem;", block,
                      "pack bytes not per actual OUT payload")
        self.assertIn("state.native_rows_pack_us += elapsed_us(t_pack);", block,
                      "pack duration missing")
        # Counted after the existing engine call, never before it.
        self.assertLess(block.index("attend_q8_bulk("), block.index("state.native_rows_pack_calls"),
                        "pack telemetry precedes the engine call")
        self.assertLess(block.index("output.resize("), block.index("state.native_rows_pack_calls"),
                        "pack telemetry precedes the CPU pack")
        # It is not folded into the KVarN transcode/GPU-codec counters.
        self.assertNotIn("bulk_output_transcode_us", block, "pack folded into KVarN transcode")
        self.assertNotIn("fp8_cpu_fallback", block, "pack claims a KVarN fallback")

    def test_out_pack_wire_breakdown(self) -> None:
        for wire in ("fp8", "f16", "f32"):
            self.assertIn(f"state.native_rows_pack_{wire}_calls", self.block,
                          f"missing {wire} wire counter")

    def test_report_and_resets(self) -> None:
        self.assertIn("native_rows_cpu_out_pack", self.text, "transport report key missing")
        hello = slice_between(self.text, "case RKVA_HELLO: {", "case RKVA_CREATE_SESSION:")
        profile = slice_between(self.text, "case RKVA_RANGE_PROFILE: {",
                                "case RKVA_RANGE_UPLOAD_ATTN: {")
        for name, block in (("HELLO", hello), ("PROFILE RESET", profile)):
            self.assertIn("state.native_rows_pack_calls = ", block, f"{name} reset missing")
            self.assertIn("state.native_rows_pack_fp8_calls = ", block,
                          f"{name} wire reset missing")



class SharedCountersInHeader(unittest.TestCase):
    def test_header_declares_the_shared_rule_once(self) -> None:
        header = read(PROFILE_H)
        self.assertEqual(header.count("inline void count_attention_sample("), 1)
        self.assertEqual(header.count("inline uint32_t bulk_upload_copy_regions("), 1)
        self.assertIn("uint64_t native_rows_calls = 0;", header)
        self.assertIn("uint64_t cpu_q_unpack_calls = 0", header)
        self.assertIn("native_rows_cpu_q_unpack", header)
        self.assertIn('b.storage', header)


if __name__ == "__main__":
    unittest.main(verbosity=2)
