#!/usr/bin/env python3
"""Source regressions and scalar Wave64 model for the low4/SAW rowlane kernels.

This is NOT DXC compilation, DXIL validation, or GPU parity. It deliberately
rejects conditional query broadcasts even if a compiler might flatten them.

  python tests/test-rkva-rowlane-uniform.py
  python tests/test-rkva-rowlane-uniform.py --revision HEAD  # old source fails

Only the generic low4 template and the SAW branch are covered, not legacy
q4_opt/gqa_opt or every kernel which includes the generic template.
"""
import argparse
import hashlib
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ("tools/xbox/shaders/rkva/rkva_decode_rowlane_template.hlsli", "ROWLANE_ENTRY", 1),
    ("tools/xbox/shaders/rkva/rkva_range_attn_rowlane_matrix.hlsl",
     "CSRangeAttnSawInt4DecodeRowlane", 2),
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def strip_comments(text):
    # Preserve offsets/newlines for useful source diagnostics.
    return re.sub(r"/\*.*?\*/|//[^\n]*", lambda m: re.sub(r"[^\n]", " ", m[0]),
                  text, flags=re.S)


def paired(text, start, left, right):
    require(text[start] == left, "malformed delimiter")
    depth = 1
    for end in range(start + 1, len(text)):
        depth += (text[end] == left) - (text[end] == right)
        if depth == 0:
            return end + 1
    raise AssertionError("unbalanced source delimiters")


def body(text, name):
    text = strip_comments(text)
    match = re.search(r"\bvoid\s+" + re.escape(name) + r"\s*\(", text)
    require(match is not None, "missing kernel " + name)
    begin = text.index("{", paired(text, match.end() - 1, "(", ")"))
    return text[begin:paired(text, begin, "{", "}")]


def statement_end(text, start):
    while text[start].isspace():
        start += 1
    while text[start] == "[":
        start = paired(text, start, "[", "]")
        while text[start].isspace():
            start += 1
    if text[start] == "{":
        return paired(text, start, "{", "}")
    control = re.match(r"(?:if|for|while|switch)\s*\(", text[start:])
    if control:
        condition_end = paired(text, start + control.end() - 1, "(", ")")
        end = statement_end(text, condition_end)
        trailing = re.match(r"\s*else\b", text[end:])
        return statement_end(text, end + trailing.end()) if trailing else end
    # Semicolons inside calls/array indexing do not terminate a statement.
    i = start
    while i < len(text):
        if text[i] in "([":
            i = paired(text, i, text[i], ")" if text[i] == "(" else "]")
        elif text[i] == ";":
            return i + 1
        else:
            i += 1
    raise AssertionError("unterminated statement")


def control_scopes(text):
    result = []
    for match in re.finditer(r"\b(if|for|while|switch)\s*\(", text):
        stop = paired(text, match.end() - 1, "(", ")")
        end = statement_end(text, stop)
        result.append((stop, end, match[1], re.sub(r"\s+", "", text[match.end():stop-1])))
        if match[1] == "if":
            trailing = re.match(r"\s*else\b", text[end:])
            if trailing:
                begin = end + trailing.end()
                result.append((begin, statement_end(text, begin), "else", ""))
    return result


def validate_source(text, name, expected_broadcasts):
    text = body(text, name)
    scopes = control_scopes(text)
    ancestors = lambda offset: [s for s in scopes if s[0] <= offset < s[1]]
    broadcasts = list(re.finditer(r"\bWaveReadLaneAt\s*\(\s*q0\s*\[", text))
    require(len(broadcasts) == expected_broadcasts, "unexpected/missing Q broadcasts")
    for match in broadcasts:
        parents = ancestors(match.start())
        require(not any(s[2] in ("if", "else", "switch") for s in parents),
                "Q broadcast is inside conditional control flow (inactive source lane risk)")
        require(not any(re.search(r"\b(?:row_valid|lane|my_row|u)\b", s[3])
                        for s in parents if s[2] in ("for", "while")),
                "Q broadcast loop is lane-dependent")
        require(any(s[2] == "for" and "tile_row0<end" in s[3] for s in parents),
                "Q broadcast missing uniform tile loop")
    qk_end = broadcasts[-1].end()
    require(not re.search(r"\b(?:break|continue|return|discard)\b", text[:qk_end]),
            "early lane exit before Q broadcasts requires a new uniformity proof")

    key_loads = list(re.finditer(r"(?:ROWLANE_LOAD\s*\(\s*keys\b|keys\s*\.\s*Load\d*\s*\()", text))
    require(bool(key_loads), "missing K loads")
    for match in key_loads:
        require(any(s[2] == "if" and s[3] == "row_valid" for s in ancestors(match.start())),
                "K load is not protected by if (row_valid)")

    require(re.search(r"rows_in_tile\s*=\s*min\(64u,\s*\(end\s*>\s*tile_row0\)\s*\?\s*\(end\s*-\s*tile_row0\)\s*:\s*0u\)", text),
            "PV row count is not bounded by tile/end")
    value_loads = list(re.finditer(r"(?:ROWLANE_LOAD\s*\(\s*values\b|values\s*\.\s*Load\d*\s*\()", text))
    require(bool(value_loads), "missing V loads")
    for match in value_loads:
        parents = ancestors(match.start())
        if "Load2" in match[0]:
            require(any(s[2] == "if" and s[3] == "row_valid" for s in parents),
                    "V metadata load is not protected by if (row_valid)")
        else:
            require(any(s[2] == "for" and "r<rows_in_tile" in s[3] for s in parents),
                    "V row load is not protected by the bounded PV loop")
    require(re.search(r"float\s+weight\s*=\s*row_valid\s*\?", text),
            "invalid rows must retain zero softmax weight")
    return text


def mutation_checks(source, name, count):
    """Ensure the structural checker detects branch and bounds regressions."""
    mutations = (
        re.sub(r"(float4\s+q_(?:val|lo)\s*=\s*WaveReadLaneAt[^;]+;)",
               r"if (row_valid) { \1 }", source, count=1),
        re.sub(r"if\s*\(row_valid\)", "if (true)", source),
        source.replace("r < rows_in_tile", "r < 64u"),
    )
    for index, changed in enumerate(mutations):
        require(changed != source, "negative fixture did not mutate source")
        try:
            validate_source(changed, name, count)
        except AssertionError:
            continue
        raise AssertionError("checker accepted unsafe mutation " + str(index))


def header_hashes(read_bytes):
    shader_dir = "tools/xbox/shaders/rkva/"
    source_hash = hashlib.sha256(read_bytes(shader_dir + "rkva_range_attn_rowlane_matrix.hlsl")).hexdigest()
    for quant in ("q4_1", "iq4_nl", "saw_int4"):
        path = "tools/xbox/shaders/generated/rkva_range_attn_" + quant + "_decode_rowlane_dxil.h"
        text = read_bytes(path).decode()
        match = re.search(r"^// Source SHA256: ([0-9a-f]{64})$", text, re.M)
        require(match is not None and match[1] == source_hash, path + ": stale source hash metadata")
        includes = re.findall(r'^// Included "([^"\n]+)" SHA256: ([0-9a-f]{64})$', text, re.M)
        require(any(name == "rkva_decode_rowlane_template.hlsli" for name, _ in includes),
                path + ": missing template hash metadata")
        for name, expected in includes:
            require("/" not in name and "\\" not in name, "unexpected include path")
            actual = hashlib.sha256(read_bytes(shader_dir + name)).hexdigest()
            require(actual == expected, path + ": stale include hash metadata: " + name)
        print("PASS generated-header source/include hash metadata:", quant)


def model_wave64():
    """Model source-lane activity, row loads, and split online softmax in FP64.

    One scalar per Q quarter suffices for source-lane provenance. This does not
    model quantization, the real float4 decoder, compiler execution, or GPU math.
    """
    q = [0.01 * (lane + 1) for lane in range(64)]
    scores = [sum(q[u] * (((row * 13 + u * 7) % 31 - 15) / 31.0)
                  for u in range(64)) / 8.0 for row in range(257)]
    vals = [[((row * 5 + d * 3) % 23 - 11) / 11.0 for d in range(4)]
            for row in range(257)]
    cases = 0
    unsafe_old_tiles = 0
    observed_tail_sizes = set()

    def summary(rows):
        if not rows:
            return (-math.inf, 0.0, [0.0] * 4)
        peak = max(scores[row] for row in rows)
        weights = [math.exp(scores[row] - peak) for row in rows]
        return (peak, sum(weights), [sum(w * vals[row][d] for row, w in zip(rows, weights))
                                    for d in range(4)])

    def merge(states):
        live = [state for state in states if state[1]]
        if not live:
            return (-math.inf, 0.0, [0.0] * 4)
        peak = max(state[0] for state in live)
        factors = [math.exp(state[0] - peak) for state in live]
        return (peak, sum(s[1] * f for s, f in zip(live, factors)),
                [sum(s[2][d] * f for s, f in zip(live, factors)) for d in range(4)])

    for count in (1, 63, 64, 65, 127, 128, 129, 257):
        for causal_count in (0, 1, 63, 64, 65, count):
            limit = min(count, causal_count)
            for split_rows in (64, 65, 127, 256):
                for n_waves in (2, 4):
                    parts = []
                    loaded = []
                    for split in range((count + split_rows - 1) // split_rows):
                        begin = min(limit, split * split_rows)
                        end = min(limit, (split + 1) * split_rows)
                        wave_states = []
                        for wave in range(n_waves):
                            tiles = []
                            for tile in range(begin + wave * 64, end, n_waves * 64):
                                valid = min(64, end - tile)
                                observed_tail_sizes.add(valid)
                                # Old row_valid branch activates only lanes with rows.
                                old_active = set(range(valid))
                                if any(source not in old_active for source in range(64)):
                                    unsafe_old_tiles += 1
                                # Corrected Q broadcasts occur after reconvergence.
                                new_active = set(range(64))
                                for source in range(64):
                                    require(source in new_active, "inactive Q source in corrected model")
                                rows = list(range(tile, tile + valid))
                                require(all(begin <= row < end <= count for row in rows), "out-of-bounds K/V row")
                                loaded.extend(rows)
                                tiles.append(summary(rows))
                            wave_states.append(merge(tiles))
                        parts.append(merge(wave_states))
                    require(sorted(loaded) == list(range(limit)), "lost/duplicated row across wave/split tails")
                    candidate, reference = merge(parts), summary(list(range(limit)))
                    if limit:
                        got = [x / candidate[1] for x in candidate[2]]
                        expected = [x / reference[1] for x in reference[2]]
                        require(all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                                    for a, b in zip(got, expected)), "online split softmax output mismatch")
                        require(math.isclose(candidate[0] + math.log(candidate[1]),
                                             reference[0] + math.log(reference[1]),
                                             rel_tol=1e-12, abs_tol=1e-12), "split LSE mismatch")
                    else:
                        require(candidate == reference, "empty causal range mismatch")
                    cases += 1
    require({1, 63, 64}.issubset(observed_tail_sizes), "missing boundary coverage")
    require(unsafe_old_tiles > 0, "model did not expose old inactive-source hazard")
    return cases, unsafe_old_tiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", help="read both shader sources using git show REV:path")
    args = parser.parse_args()
    def read_bytes(path):
        if args.revision:
            return subprocess.check_output(["git", "show", args.revision + ":" + path], cwd=ROOT)
        return (ROOT / path).read_bytes()

    failures = []
    for path, name, count in SOURCES:
        source = read_bytes(path).decode()
        try:
            validate_source(source, name, count)
            mutation_checks(source, name, count)
            print("PASS source uniformity/bounds and negative mutations:", path)
        except AssertionError as exc:
            failures.append(path + ": " + str(exc))
    try:
        header_hashes(read_bytes)
    except AssertionError as exc:
        failures.append(str(exc))
    cases, hazards = model_wave64()
    print(f"PASS scalar Wave64 model: {cases} causal/split cases; {hazards} old partial tiles expose inactive Q source risk")
    print("Scope: source/model only; no DXC, DXIL, GPU parity, or performance qualification.")
    for error in failures:
        print("FAIL", error, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
