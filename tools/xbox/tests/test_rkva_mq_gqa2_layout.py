#!/usr/bin/env python3
"""CPU-only contract tests for the RKVA KVarN4 MQ two-local-head compile
experiment (`RKVA_MQ_HEADS_PER_GROUP == 2`).

This does NOT compile D3D12/DXIL. It exercises the C preprocessor and the
layout arithmetic that the shader relies on:

  * the shipped defaults (QT2 F32, QT4 F32, QT2 INT8) preprocess to the exact
    same token stream as the pinned base revision of the kernel;
  * the new wrapper keeps the entry point, cbuffer, resources and
    [numthreads(128,1,1)] contract;
  * the actual `groupshared` declarations evaluate to 12,384 B at six local
    heads and 4,128 B at two;
  * the proposed `DispatchX = 3 * n_head_kv` mapping covers every global head
    exactly once for every (nq, split) pair tested, and each global partial
    offset equals the shipped one;
  * the WHT halves and every local array stay in bounds at two and six heads;
  * `HEADS_PER_GROUP == 1`, `HEADS == 2 && QT == 4` and `HEADS == 2 && I8 == 1`
    are rejected by the preprocessor.

Pinned base: 8bad8e84e67e9fcd43d12f637962b87685e0fecc (override with the
RKVA_MQ_GQA2_BASE environment variable). Override the preprocessor with CPP.

Run: python3 tools/xbox/tests/test_rkva_mq_gqa2_layout.py
"""
import os
import hashlib
import importlib.util
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[3]
SHADER = ROOT / "tools/xbox/shaders/rkva/rkva_range_attn_kvarn4_mq.hlsl"
WRAPPER = SHADER.with_name("rkva_range_attn_kvarn4_mq_gqa2.hlsl")
GENERATED = ROOT / "tools/xbox/shaders/generated"
GQA2_HEADER = GENERATED / "rkva_range_attn_kvarn4_mq2_gqa2_dxil.h"
GQA2_DXIL_SHA256 = "b26dd85cb5512225dcb10333fc53a7a7b52676ccac2d480ed237fdcf92565a99"
GQA2_DXIL_BYTES = 47924
# Generated headers for the shipped variants must stay byte-identical; any
# intentional regeneration has to update these pins and the review record.
DEFAULT_HEADER_SHA256 = {
    "rkva_range_attn_kvarn4_mq2_dxil.h":
        "7119a24ed39ef605a1b47a63157daadc5a45a9e2da8a60cc10b1fe86f4b82857",
    "rkva_range_attn_kvarn4_mq4_dxil.h":
        "22e5f166d4c588d8dd635d826d962b67c527b2b25d062bd45aa42b6c35592662",
    "rkva_range_attn_kvarn4_mq2_i8_dxil.h":
        "887cbcc80a2c80db411991c282b09a437dbc38f518548bddd2f4ec536056a205",
}
# Embedded DXIL payloads, independent of the header metadata block. These are
# the real-DXC blobs from the supplied mq2/mq4/i8 artifacts.
DEFAULT_DXIL_SHA256 = {
    "rkva_range_attn_kvarn4_mq2_dxil.h":
        ("9d81c3beba4e36756c89c6bc70c5807cedb990be67fba683fc714ff7a1c5b431", 118452),
    "rkva_range_attn_kvarn4_mq4_dxil.h":
        ("ef2878458bd37b6f702fec19f36e674a836803d300915cf2604bfd801dfb48bb", 236896),
    "rkva_range_attn_kvarn4_mq2_i8_dxil.h":
        ("a2b66069596fe3b8111dd19d99b4b65a652b2b2e376f44014015ade13f2c2d3f", 140356),
}

BASE_SHA = os.environ.get(
    "RKVA_MQ_GQA2_BASE", "8bad8e84e67e9fcd43d12f637962b87685e0fecc"
)
TOKEN_RE = re.compile(r"\w+|[^\w\s]")
GS_RE = re.compile(r"groupshared\s+float\s+(gs_\w+)\s*\[([^\]]+)\]")
GS_EXPECTED = {
    "gs_max": 2 * 6 * 4,
    "gs_den": 2 * 6 * 4,
    "gs_vec": 2 * 6 * 256 * 4,
}
GS_EXPECTED_H2 = {
    "gs_max": 2 * 2 * 4,
    "gs_den": 2 * 2 * 4,
    "gs_vec": 2 * 2 * 256 * 4,
}


def preprocessor():
    """Return argv prefix for the C preprocessor."""
    explicit = os.environ.get("CPP")
    if explicit:
        path = shutil.which(explicit) or explicit
        if pathlib.Path(path).name == "cpp":
            return [path, "-P"]
        return [path, "-E", "-P", "-x", "c"]
    for name in ("cpp", "clang", "gcc"):
        path = shutil.which(name)
        if path:
            if pathlib.Path(path).name == "cpp":
                return [path, "-P"]
            return [path, "-E", "-P", "-x", "c"]
    raise RuntimeError("no C preprocessor found; set CPP")


def preprocess(text=None, defines=(), path=None):
    """Preprocess a source string or file. Returns (rc, stdout, stderr)."""
    cmd = preprocessor() + list(defines)
    if path is not None:
        return subprocess.run(
            cmd + [str(path)], capture_output=True, text=True
        )
    with tempfile.TemporaryDirectory(prefix="rkva-mq-gqa2-") as tmp:
        src = pathlib.Path(tmp) / "probe.hlsl"
        src.write_text(text)
        return subprocess.run(cmd + [str(src)], capture_output=True, text=True)


def tokens(text):
    return TOKEN_RE.findall(text)


def base_source():
    rel = SHADER.relative_to(ROOT).as_posix()
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{BASE_SHA}:{rel}"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"git show {BASE_SHA}:{rel} failed: {proc.stderr.strip()}"
        )
    return proc.stdout


def gs_bytes(pp_text):
    """Evaluate the real groupshared declaration sizes in a preprocessed file."""
    found = {}
    for name, expr in GS_RE.findall(pp_text):
        if not re.fullmatch(r"[0-9+\-*/() ]+", expr):
            raise AssertionError(f"unexpected groupshared size expression: {expr!r}")
        count = eval(expr, {"__builtins__": {}}, {})
        if not isinstance(count, int):
            raise AssertionError(f"non-integer groupshared size: {expr!r}")
        found[name] = count * 4
    return found, sum(found.values())


def contract(text):
    block = re.search(
        r"cbuffer Params : register\(b0\) \{.*?"
        r"RWByteAddressBuffer output : register\(u1\);",
        text,
        re.S,
    )
    signature = re.search(r"void\s+CSRangeAttnKvarn4Mq\s*\([^)]*\)", text)
    if block is None or signature is None:
        raise AssertionError("entry/cbuffer/resource declaration not found")
    return tokens(block.group(0)), tokens(signature.group(0))


DEFAULT_VARIANTS = (
    ("qt2_f32", ("-DRKVA_MQ_QT=2",)),
    ("qt4_f32", ("-DRKVA_MQ_QT=4",)),
    ("qt2_i8", ("-DRKVA_MQ_QT=2", "-DRKVA_MQ_I8=1")),
)


class Gqa2CompileContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = base_source()

    def test_default_preprocess_is_token_identical_to_base(self):
        for label, defines in DEFAULT_VARIANTS:
            with self.subTest(variant=label):
                ref = preprocess(self.base, defines)
                cur = preprocess(path=SHADER, defines=defines)
                self.assertEqual(ref.returncode, 0, ref.stderr)
                self.assertEqual(cur.returncode, 0, cur.stderr)
                self.assertEqual(
                    tokens(cur.stdout),
                    tokens(ref.stdout),
                    f"{label} defaults changed the preprocessed token stream",
                )

    def test_wrapper_keeps_entry_cbuffer_resources_and_threads(self):
        ref = preprocess(self.base, ("-DRKVA_MQ_QT=2",))
        cur = preprocess(path=WRAPPER)
        self.assertEqual(ref.returncode, 0, ref.stderr)
        self.assertEqual(cur.returncode, 0, cur.stderr)
        self.assertEqual(contract(cur.stdout), contract(ref.stdout))
        flat = " ".join(tokens(cur.stdout))
        self.assertIn("[ numthreads ( 128 , 1 , 1 ) ]", flat)
        # The wrapper really selected the two-local-head path.
        self.assertIn("uint head_groups = 6 / 2 ;", flat)
        self.assertIn("head_base = 2 * ( group . x % head_groups ) ;", flat)

    def test_groupshared_bytes_follow_the_definitions(self):
        ref = preprocess(self.base, ("-DRKVA_MQ_QT=2",))
        cur = preprocess(path=WRAPPER)
        self.assertEqual(ref.returncode, 0, ref.stderr)
        self.assertEqual(cur.returncode, 0, cur.stderr)

        ref_parts, ref_total = gs_bytes(ref.stdout)
        cur_parts, cur_total = gs_bytes(cur.stdout)
        self.assertEqual(ref_parts, GS_EXPECTED)
        self.assertEqual(ref_total, 12384)
        self.assertEqual(cur_parts, GS_EXPECTED_H2)
        self.assertEqual(cur_total, 4128)

    def test_two_head_dispatch_covers_every_global_head_once(self):
        gqa, lh, kv = 6, 2, 4
        groups_per_kv, n_head = gqa // lh, gqa * kv
        dim, partial = 256, 258

        seen = {}
        for group_x in range(groups_per_kv * kv):
            kv_head = group_x // groups_per_kv
            head_base = lh * (group_x % groups_per_kv)
            self.assertLess(kv_head, kv)
            for local in range(lh):
                head = kv_head * gqa + head_base + local
                self.assertEqual(head, 2 * group_x + local)
                self.assertLessEqual(head_base + local, gqa - 1)
                self.assertNotIn(head, seen, f"head {head} written twice")
                seen[head] = (group_x, local)
        self.assertEqual(sorted(seen), list(range(n_head)))

        def offset(nq, split, qi, head):
            return ((split * (nq * n_head) + qi * n_head + head) * partial) * 4

        for nq in (2, 3, 5, 8, 9):
            for split in (1, 2, 17, 64):
                for qi in range(nq):
                    shipped = {offset(nq, split, qi, h): [h] for h in range(n_head)}
                    candidate = {}
                    for group_x in range(groups_per_kv * kv):
                        kv_head = group_x // groups_per_kv
                        head_base = lh * (group_x % groups_per_kv)
                        for local in range(lh):
                            head = kv_head * gqa + head_base + local
                            candidate.setdefault(offset(nq, split, qi, head), []).append(head)
                    self.assertEqual(candidate, shipped)
                    self.assertEqual(sum(len(v) for v in candidate.values()), n_head)

    def test_local_arrays_wht_and_rotation_bounds(self):
        # (local heads, elements per thread, WHT half) from the two definitions.
        for lh, elems, wht_half in ((6, 12, 1536), (2, 4, 512)):
            with self.subTest(local_heads=lh):
                dim, threads = 256, 128
                self.assertEqual(elems * threads, lh * dim)
                self.assertEqual(wht_half, lh * dim)
                self.assertEqual(elems, 2 * lh)
                # One lane writes g*DIM + tid and +128, tid < 128.
                last_write = (lh - 1) * dim + (threads - 1) + 128
                self.assertEqual(last_write, wht_half - 1)
                self.assertLess(last_write, wht_half)
                # Ping-pong halves are 0 and RKVA_WHT_HALF; dst max is total-1.
                self.assertLess(wht_half + last_write, 2 * wht_half)
                for qt in (2, 4):
                    self.assertLess((qt - 1) * lh + (lh - 1), qt * lh)  # body/q_lo
                self.assertLess((2 - 1) * lh + (lh - 1), 2 * lh)  # pair/tail_pair
                self.assertLess((2 - 1) * lh + (lh - 1), 2 * lh)  # wave slot
                self.assertLess(elems // 2 - 1, lh)  # merged_body g == k/2

    def test_bad_configurations_fail_preprocess(self):
        cases = (
            ("heads1", "#define RKVA_MQ_HEADS_PER_GROUP 1\n",
             "supports 2 or 6 only"),
            ("qt4_heads2", "#define RKVA_MQ_HEADS_PER_GROUP 2\n"
                           "#define RKVA_MQ_QT 4\n",
             "supports RKVA_MQ_QT == 2 only"),
            ("i8_heads2", "#define RKVA_MQ_HEADS_PER_GROUP 2\n"
                          "#define RKVA_MQ_I8 1\n",
             "requires RKVA_MQ_I8 == 0"),
        )
        for label, preamble, message in cases:
            with self.subTest(case=label):
                probe = preamble + f'#include "{SHADER.as_posix()}"\n'
                proc = preprocess(probe)
                self.assertNotEqual(proc.returncode, 0, proc.stdout)
                self.assertIn(message, proc.stderr)


def embedded_header(path):
    text = path.read_text()
    match = re.search(r"static const uint8_t g_(rkva_range_\w+_dxil)\[\] = \{(.*?)\};",
                      text, re.S)
    if match is None:
        raise AssertionError(f"no embedded DXIL array in {path}")
    values = [int(value, 16) for value in re.findall(r"0x([0-9a-fA-F]{2})", match.group(2))]
    return match.group(1), bytes(values), text


class GeneratedDxilTest(unittest.TestCase):
    def test_candidate_header_embeds_the_real_dxil(self):
        symbol, data, text = embedded_header(GQA2_HEADER)
        self.assertEqual(symbol, "rkva_range_attn_kvarn4_mq2_gqa2_dxil")
        self.assertEqual(len(data), GQA2_DXIL_BYTES)
        self.assertEqual(hashlib.sha256(data).hexdigest(), GQA2_DXIL_SHA256)
        self.assertIn("// Target specialization: attn_kvarn4_mq2_gqa2", text)
        self.assertIn(
            "// Source SHA256: " + hashlib.sha256(WRAPPER.read_bytes()).hexdigest(), text)
        self.assertIn(
            '// Included "rkva_range_attn_kvarn4_mq.hlsl" SHA256: ' +
            hashlib.sha256(SHADER.read_bytes()).hexdigest(), text)

    def test_project_renderer_reproduces_the_candidate_header(self):
        renderer_path = ROOT / "tools/xbox/scripts/render-rkva-dxil-header.py"
        spec = importlib.util.spec_from_file_location("rkva_dxil_render", renderer_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _, data, text = embedded_header(GQA2_HEADER)
        with tempfile.TemporaryDirectory(prefix="rkva-gqa2-header-") as temporary:
            binary = pathlib.Path(temporary) / "gqa2.dxil"
            binary.write_bytes(data)
            rendered, rendered_data = module.render(
                binary, WRAPPER, "attn_kvarn4_mq2_gqa2", "cs_6_0", "")
        self.assertEqual(rendered_data, data)
        self.assertEqual(rendered, text)

    def test_shipped_generated_headers_are_unchanged(self):
        for name, expected in DEFAULT_HEADER_SHA256.items():
            with self.subTest(header=name):
                path = GENERATED / name
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_shipped_dxil_arrays_are_unchanged(self):
        for name, (expected, size) in DEFAULT_DXIL_SHA256.items():
            with self.subTest(header=name):
                _, data, _ = embedded_header(GENERATED / name)
                self.assertEqual(len(data), size)
                self.assertEqual(hashlib.sha256(data).hexdigest(), expected)

    def test_shipped_headers_record_current_source_provenance(self):
        i8_wrapper = SHADER.with_name("rkva_range_attn_kvarn4_mq_i8.hlsl")
        cases = (
            ("rkva_range_attn_kvarn4_mq2_dxil.h", SHADER, None),
            ("rkva_range_attn_kvarn4_mq4_dxil.h", SHADER, None),
            ("rkva_range_attn_kvarn4_mq2_i8_dxil.h", i8_wrapper, SHADER),
        )
        for name, source, included in cases:
            with self.subTest(header=name):
                text = (GENERATED / name).read_text()
                self.assertIn(
                    "// Source SHA256: " + hashlib.sha256(source.read_bytes()).hexdigest(),
                    text)
                if included is not None:
                    self.assertIn(
                        '// Included "rkva_range_attn_kvarn4_mq.hlsl" SHA256: ' +
                        hashlib.sha256(included.read_bytes()).hexdigest(), text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
