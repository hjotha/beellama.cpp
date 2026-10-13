#!/usr/bin/env python3
"""F4 actual-server comparator (binding-gated).

Compares the TWO full-logit captures taken by the build-f4 TEST-ONLY post-decode
hook in the SAME real llama-server code path:

  producer : PURE KV, cold, prefix evaluated from scratch
  target   : position_split destination, prefix imported from the immutable disk
             snapshot, then the SAME next token decoded

Nothing is compared unless every binding that makes the two phases
interchangeable matches EXACTLY. A mismatch is reported as a REFUSAL, never as a
numerical result, so a phase difference can never be mislabelled as state
divergence (or vice versa).

Canonical gates are the F4 ones from tests/test-position-split-real-pair.cpp:
  NRMSE(O) <= 1e-3, max|O| <= 1e-2, max|LSE| <= 1e-2
with the canonical NRMSE definition
  sqrt(mean(d^2)) / max(sqrt(mean(ref^2)), 1e-6)
"""

import argparse
import hashlib
import json
import math
import os
import struct
import sys

import numpy as np

GATE_NRMSE_O = 1e-3
GATE_MAX_ABS_O = 1e-2
GATE_MAX_ABS_LSE = 1e-2


def load(run, phase, tokens_n, log):
    stem = os.path.join(run, "cap.%s.n%d" % (phase, tokens_n))
    try:
        with open(stem + ".json") as fh:
            meta = json.load(fh)
    except Exception as exc:                                  # noqa: BLE001
        print("REFUSED: no capture for phase=%s tokens_n=%d (%s)" % (phase, tokens_n, exc))
        print("        expected %s.json -- the phase never reached its fixed checkpoint"
              % stem)
        return None
    logits = np.fromfile(stem + ".logits", dtype="<f4")
    toks = np.fromfile(stem + ".tokens", dtype="<i4")
    meta["_stem"] = stem
    meta["_logits"] = logits
    meta["_tokens"] = toks
    if log:
        print("capture %-8s tokens_n=%s input_pos=%s output_pos=%s seq=%s vocab=%s "
              "layout_hash=%s cache=%s/%s attempt=%s"
              % (phase, meta.get("tokens_n"), meta.get("input_pos"), meta.get("output_pos"),
                 str(meta.get("tokens_seq_hash"))[:12], meta.get("n_vocab"),
                 str(meta.get("layout_hash"))[:12], meta.get("cache_source"),
                 meta.get("cache_reason"), meta.get("attempt")))
    return meta


def refuse(msg, log):
    print("REFUSED: " + msg)
    if log:
        with open(log, "a") as fh:
            fh.write("REFUSED: " + msg + "\n")
    print("VERDICT: NOT COMPARED (phase mismatch; no numerical claim is made)")
    return 2


def nrmse_of(got, ref):
    if got.shape != ref.shape or got.size == 0:
        return float("inf")
    d = got.astype(np.float64) - ref
    se = float(np.sum(d * d))
    sr = float(np.sum(ref.astype(np.float64) ** 2))
    return math.sqrt(se / got.size) / max(math.sqrt(sr / got.size), 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--tokens-n", type=int, required=True)
    ap.add_argument("--expect-cache-source", required=True)
    ap.add_argument("--require-target-cache-n", type=int, default=None)
    ap.add_argument("--require-baseline-cache-n", type=int, default=None)
    ap.add_argument("--require-evaluated-suffix", type=int, default=None)
    ap.add_argument("--producer-counts", default="")
    ap.add_argument("--target-counts", default="")
    ap.add_argument("--producer-phase", default="producer")
    ap.add_argument("--target-phase", default="target")
    ap.add_argument("--log", default="")
    args = ap.parse_args()

    a = load(args.run, args.producer_phase, args.tokens_n, args.log)
    if a is None:
        return 2
    b = load(args.run, args.target_phase, args.tokens_n, args.log)
    if b is None:
        return 2

    # --- bindings that MUST be identical -----------------------------------
    same = [
        ("run_id", a.get("run_id"), b.get("run_id")),
        ("build_id", a.get("build_id"), b.get("build_id")),
        ("model_fingerprint", a.get("model_fingerprint"), b.get("model_fingerprint")),
        ("config_hash", a.get("config_hash"), b.get("config_hash")),
        ("tokens_n", a.get("tokens_n"), b.get("tokens_n")),
        ("tokens_seq_hash", a.get("tokens_seq_hash"), b.get("tokens_seq_hash")),
        ("input_pos", a.get("input_pos"), b.get("input_pos")),
        ("output_pos", a.get("output_pos"), b.get("output_pos")),
        ("n_ctx", a.get("n_ctx"), b.get("n_ctx")),
        ("n_vocab", a.get("n_vocab"), b.get("n_vocab")),
        ("logits_count", a.get("logits_count"), b.get("logits_count")),
    ]
    for name, x, y in same:
        if x != y or x in (None, ""):
            return refuse("%s differs: producer=%r target=%r" % (name, x, y), args.log)

    # --- bindings that MUST differ (proves the phases are what they claim) --
    if a.get("phase") == b.get("phase"):
        return refuse("both captures claim phase=%r" % a.get("phase"), args.log)
    if a.get("layout_hash") == b.get("layout_hash"):
        return refuse("producer and target share the same KV layout (%s); the target "
                      "is not a position_split destination" % a.get("layout"), args.log)

    # --- the evaluated token stream must be independently identical --------
    for m in (a, b):
        if m["_tokens"].size != args.tokens_n:
            return refuse("%s: token file holds %d ids, expected %d"
                          % (m.get("phase"), m["_tokens"].size, args.tokens_n), args.log)
    if not np.array_equal(a["_tokens"], b["_tokens"]):
        d = int(np.nonzero(a["_tokens"] != b["_tokens"])[0][0])
        return refuse("token streams differ at index %d (producer=%d target=%d) -- the "
                      "'same fixed next token' precondition does not hold"
                      % (d, a["_tokens"][d], b["_tokens"][d]), args.log)
    nxt = int(a["_tokens"][-1])
    print("fixed next token evaluated by both phases: id=%d (position %d)"
          % (nxt, a["input_pos"]))

    # --- the target must genuinely have imported the prefix from disk -------
    if b.get("cache_source") != args.expect_cache_source:
        return refuse("target capture is not a disk import (cache_source=%r cache_reason=%r)"
                      % (b.get("cache_source"), b.get("cache_reason")), args.log)
    if a.get("cache_source") == args.expect_cache_source:
        return refuse("producer capture also claims cache_source=%r; the producer must "
                      "evaluate the prefix itself" % a.get("cache_source"), args.log)

    for m in (a, b):
        if not m.get("finite"):
            return refuse("%s capture contains non-finite logits" % m.get("phase"), args.log)

    # --- the HTTP-side token counters must back the capture bindings --------
    def counts(path):
        if not path:
            return None
        try:
            with open(path) as fh:
                line = [l for l in fh if l.strip()][-1]
            return json.loads(line)
        except Exception:                                            # noqa: BLE001
            return None
    pc, tc = counts(args.producer_counts), counts(args.target_counts)
    if pc is None or tc is None:
        return refuse("missing per-phase token counters (%s / %s)"
                      % (args.producer_counts, args.target_counts), args.log)
    if args.require_baseline_cache_n is not None and pc.get("cache_n") != args.require_baseline_cache_n:
        return refuse("baseline cache_n=%r, required %s (the baseline must be cold)"
                      % (pc.get("cache_n"), args.require_baseline_cache_n), args.log)
    if args.require_target_cache_n is not None and tc.get("cache_n") != args.require_target_cache_n:
        return refuse("target cache_n=%r, required %s (the imported prefix)"
                      % (tc.get("cache_n"), args.require_target_cache_n), args.log)
    if args.require_evaluated_suffix is not None:
        for lbl, cc, m in (("baseline", pc, a), ("target", tc, b)):
            cn = cc.get("cache_n")
            suffix = (m.get("tokens_n") - cn) if (cn is not None and m.get("tokens_n")) else None
            if suffix is None:
                return refuse("%s evaluated suffix is unknown (cache_n=%r)"
                              % (lbl, cn), args.log)
            if lbl == "target" and suffix != args.require_evaluated_suffix:
                return refuse("target evaluated suffix=%s, required %s"
                              % (suffix, args.require_evaluated_suffix), args.log)
    print("counters: %s tokens_n=%s cache_n=%s | %s tokens_n=%s cache_n=%s "
          "evaluated_suffix=%s" % (a.get("phase"), a.get("tokens_n"), pc.get("cache_n"),
                                   b.get("phase"), b.get("tokens_n"), tc.get("cache_n"),
                                   b.get("tokens_n") - tc.get("cache_n")))

    # --- canonical full-logit gates ----------------------------------------
    ref = a["_logits"].astype(np.float64)
    got = b["_logits"].astype(np.float64)
    if ref.shape != got.shape or ref.size == 0:
        return refuse("logit vectors differ in shape: %s vs %s" % (ref.shape, got.shape),
                      args.log)

    nr = nrmse_of(got, ref)
    md = float(np.max(np.abs(got - ref)))
    lse = abs(float(b.get("lse_vocab", float("nan"))) - float(a.get("lse_vocab", float("nan"))))
    top_a = int(np.argmax(ref))
    top_b = int(np.argmax(got))

    ok_o = nr <= GATE_NRMSE_O and md <= GATE_MAX_ABS_O
    ok_lse = lse <= GATE_MAX_ABS_LSE
    verdict = "PASS" if (ok_o and ok_lse) else "FAIL"

    print("")
    print("  producer: n_params=%s n_ctx=%s pos_max=%s state_bytes=%s cache=%s/%s"
          % (a.get("n_params"), a.get("n_ctx"), a.get("pos_max"), a.get("state_bytes"),
             a.get("cache_source"), a.get("cache_reason")))
    print("  target  : n_params=%s n_ctx=%s pos_max=%s state_bytes=%s cache=%s/%s"
          % (b.get("n_params"), b.get("n_ctx"), b.get("pos_max"), b.get("state_bytes"),
             b.get("cache_source"), b.get("cache_reason")))
    print("  layout  : producer=%s" % a.get("layout"))
    print("            target  =%s" % b.get("layout"))
    print("  gates   : NRMSE(O)=%.6e (<= %.0e) %s | max|O|=%.6e (<= %.0e) %s | "
          "|dLSE|=%.6e (<= %.0e) %s"
          % (nr, GATE_NRMSE_O, "ok" if nr <= GATE_NRMSE_O else "FAIL",
             md, GATE_MAX_ABS_O, "ok" if md <= GATE_MAX_ABS_O else "FAIL",
             lse, GATE_MAX_ABS_LSE, "ok" if ok_lse else "FAIL"))
    print("  argmax  : producer=%d target=%d %s" % (top_a, top_b,
                                                     "same" if top_a == top_b else "DIVERGED"))
    print("")
    print("VERDICT: %s" % verdict)
    if args.log:
        with open(args.log, "a") as fh:
            fh.write("  gates   : NRMSE(O)=%.6e max|O|=%.6e |dLSE|=%.6e argmax %d/%d\n"
                     % (nr, md, lse, top_a, top_b))
            fh.write("VERDICT: %s\n" % verdict)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())