#!/usr/bin/env python3
"""F4 branch driver: ONE test process PER BRANCH, strictly sequential.

Real isolation without relying on llama_memory_clear() or on a partial restore:
each branch runs in its own process with exactly one live context, exports the
K/V head slices captured from scheduler-executed consumer tensors plus its
single proof-token full logits, and exits. The process exit returns the VRAM
before the next branch starts. Branches never run in parallel.

Gates are asserted in order:
  1. every branch has a context and executed consumer samples, else FIXTURE ERROR
  2. A == A2 (executed K/V before + logits)  -> determinism
  3. A == D  (executed K/V after restore + logits) -> exact restore before decode
  4. only then B (live mutation) and C (inactive mutation) are interpreted
"""

import os
import struct
import subprocess
import sys
import tempfile

BIN = "/home/hjotha/beellama.cpp/build-optimized/bin/test-kv-mixed-destination-intact"
SPOOL = os.environ.get("F4_TEST_SPOOL", "/home/hjotha/f4-spool")


DRIVER_LAYER = int(os.environ.get("F4_DRIVER_LAYER", "3"))


def run_branch(branch, n, tensor="k_stage", head=0, mutate=0, restore=0, live=1, layer=None):
    """Run ONE branch in its own process. Returns (meta, flags, vecs) or raises."""
    if layer is None:
        layer = DRIVER_LAYER
    env = dict(os.environ)
    env.update({
        "F4_TEST_BRANCH": branch,
        "F4_TEST_N": str(n),
        "F4_TEST_TENSOR": tensor,
        "F4_TEST_HEAD": str(head),
        "F4_TEST_MUTATE": str(mutate),
        "F4_TEST_RESTORE": str(restore),
        "F4_TEST_LIVE": str(live),
        "F4_TEST_LAYER": str(layer),
        "F4_TRACE_CAPTURE": "1",
        "F4_CAPTURE_LAYERS": str(layer),
        "F4_TEST_GPU": "1",
        "F4_TEST_SPOOL": SPOOL,
    })
    fd, outp = tempfile.mkstemp(prefix="f4branch-", suffix=".bin")
    os.close(fd)
    env["F4_TEST_OUT"] = outp
    try:
        proc = subprocess.run([BIN], env=env, capture_output=True, text=True, timeout=2400)
        line = None
        for ln in proc.stdout.splitlines():
            if ln.startswith("SINGLEBRANCH "):
                line = ln
        if line is None:
            tail = "\n".join(proc.stderr.strip().splitlines()[-12:])
            raise RuntimeError(
                f"branch {branch} n={n} produced NO SINGLEBRANCH line "
                f"(exit={proc.returncode})\n{tail}")
        meta = {}
        for tok in line.split()[1:]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                meta[k] = v
        data = open(outp, "rb").read()
    finally:
        try:
            os.unlink(outp)
        except OSError:
            pass
    off = 0

    def rmeta():
        nonlocal off
        (kl,) = struct.unpack_from("<I", data, off); off += 4
        k = data[off:off + kl].decode(); off += kl
        (vl,) = struct.unpack_from("<I", data, off); off += 4
        v = data[off:off + vl].decode(); off += vl
        return k, v

    def rvec():
        nonlocal off
        (n,) = struct.unpack_from("<Q", data, off); off += 8
        v = list(struct.unpack_from(f"<{n}f", data, off)) if n else []
        off += 4 * n
        return v

    # exactly nine string pairs are written before the flags
    for _ in range(9):
        k, v = rmeta()
        meta[k] = v
    fl = struct.unpack_from("<5I", data, off); off += 20
    names = ["consumer_before", "consumer_after", "wrote", "restored", "decoded"]
    flags = dict(zip(names, [bool(x) for x in fl]))
    vecs = {
        "before": [rvec() for _ in range(4)],   # k0 k1 v0 v1
        "after":  [rvec() for _ in range(4)],
        "logits": rvec(),
    }
    routes = {}
    if off < len(data):
        (nr,) = struct.unpack_from("<Q", data, off); off += 8
        for _ in range(nr):
            (rl,) = struct.unpack_from("<I", data, off); off += 4
            label = data[off:off + rl].decode(); off += rl
            routes[label] = rvec()
    return meta, flags, vecs, routes


def diff(a, b):
    """Max abs difference, or None when a vector is absent (fixture error)."""
    if not a or not b or len(a) != len(b):
        return None
    return max(abs(x - y) for x, y in zip(a, b))


def worst(pairs):
    """Max over diffs; None (absent) dominates any numeric value."""
    ds = [diff(x, y) for x, y in pairs]
    if any(d is None for d in ds):
        return None
    return max(ds)


def first_diffs(a, b, limit=6):
    """Indices (and values) where two equal-length vectors first differ."""
    out = []
    if not a or not b or len(a) != len(b):
        return out
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            out.append((i, x, y))
            if len(out) >= limit:
                break
    return out


# Below-P local live state lives in the F16 KVarN stage, not in sealed I8
# records and not in the unoccupied remote Q4 range. Mutate that source here.
# Below-P local live state lives in the F16 KVarN stage, not in sealed I8
# records and not in the unoccupied remote Q4 range. Mutate that source here.
FIELD = [("k_stage", 0, "K.h0"), ("k_stage", 1, "K.h1"),
         ("v_stage", 0, "V.h0"), ("v_stage", 1, "V.h1")]
# At/above P with occupied overflow: mutate the Q4 companion storage itself at
# a genuinely occupied overflow position (pos >= P). The remote target must
# move ONLY the split_overflow head; split_local must be identical.
FIELD_REMOTE = [("remote_k", 0, "K.h0"), ("remote_k", 1, "K.h1"),
                ("remote_v", 0, "V.h0"), ("remote_v", 1, "V.h1")]


def main():
    ns = [int(x) for x in (sys.argv[1:] or ["1", "127", "128", "129"])]
    failures = []
    for n in ns:
        print(f"\n===== n={n} =====", flush=True)
        # ---- gate 1: every branch must have a context and a working consumer ----
        A = run_branch("A", n, "k_records", 0, 0, 0)
        print(f"  A   {A[0].get('pos')} cb={int(A[1]['consumer_before'])} "
              f"ca={int(A[1]['consumer_after'])} dec={int(A[1]['decoded'])}", flush=True)
        if not A[1]["consumer_before"] or not A[1]["decoded"]:
            failures.append(f"n={n}: fixture error: A consumer/decode unavailable "
                            f"(cb={A[1]['consumer_before']} dec={A[1]['decoded']})")
            print("  FIXTURE ERROR -> B/C not interpreted", flush=True)
            continue

        above = n >= int(A[0].get("P", 0))

        # ---- gate 2: determinism A == A2 (prefill AND proof decode) ----
        # At/above P the proof decode is dual-range; its samples live in the
        # labeled route section, compared in gate 3/4. Here compare prefills.
        A2 = run_branch("A2", n, "k_stage", 0, 0, 0)
        m = worst(zip(A[2]["before"], A2[2]["before"]))
        dl = diff(A[2]["logits"], A2[2]["logits"])
        if above:
            ok2 = m == 0.0 and dl == 0.0
            ma = "(routes, see gate 3/4)"
        else:
            ma = worst(zip(A[2]["after"], A2[2]["after"]))
            ok2 = m == 0.0 and ma == 0.0 and dl == 0.0
        print(f"  A2  cb={int(A2[1]['consumer_before'])} max_mat={m} max_mat_after={ma} "
              f"max_logits={dl} -> A=A2 {'PASS' if ok2 else 'FAIL'}", flush=True)
        if not ok2:
            failures.append(f"n={n}: A=A2 failed (max_mat={m} max_mat_after={ma} "
                            f"max_logits={dl}); B/C not interpreted")

        # ---- gate 3: exact restore before any decode, A == D ----
        # Both ran the SAME proof decode; only D touched state (then restored).
        # Comparing A_before (prefill) with D_after (post-decode) is wrong: the
        # decode legitimately stores the proof token. Compare post-decode states.
        # At/above P the decode is dual-range: compare per-route samples.
        D = run_branch("D", n, "k_stage", 0, 1, 1)
        if above:
            labels = sorted(A[3].keys())
            if not labels or set(labels) != set(D[3].keys()):
                ok3 = False
                m = None
                print(f"  D   route labels A={sorted(A[3].keys())} "
                      f"D={sorted(D[3].keys())} -> A=D FAIL", flush=True)
            else:
                m = worst((A[3][lb], D[3][lb]) for lb in labels)
                dl = diff(A[2]["logits"], D[2]["logits"])
                ok3 = m == 0.0 and dl == 0.0 and D[1]["restored"]
        else:
            m = worst(zip(A[2]["after"], D[2]["after"]))
            dl = diff(A[2]["logits"], D[2]["logits"])
            ok3 = m == 0.0 and dl == 0.0 and D[1]["restored"]
        dl = dl if 'dl' in dir() else None
        print(f"  D   wrote={int(D[1]['wrote'])} restored={int(D[1]['restored'])} "
              f"max_mat(A_after,D_after)={m} max_logits={dl} -> A=D "
              f"{'PASS' if ok3 else 'FAIL'}", flush=True)
        if not ok3:
            if not above:
                for i, (x, y) in enumerate(zip(A[2]["after"], D[2]["after"])):
                    fd = first_diffs(x, y)
                    if fd:
                        print(f"  D field[{i}] first diffs: {fd}", flush=True)
            failures.append(f"n={n}: A=D failed (max_mat={m} max_logits={dl} "
                            f"restored={D[1]['restored']}); B/C not interpreted")

        if failures and any(f.startswith(f"n={n}:") for f in failures):
            print("  -> gates failed, B/C NOT interpreted for this n", flush=True)
            continue

        # ---- gate 4: only now interpret B (live) and C (inactive) ----
        # Within-branch before/after can never be "only target differs": the
        # proof decode itself stores the proof token in every head. Compare
        # each mutated branch's POST-DECODE state against A's post-decode
        # state: same decode, so the mutation is the only possible difference.
        # At/above P the decode is dual-range: compare per-route samples. The
        # local stage mutation must move ONLY the split_local target head; the
        # split_overflow samples must be identical everywhere.
        for tensor, head, name in FIELD:
            B = run_branch("B", n, tensor, head, 1, 0, 1)
            C = run_branch("C", n, tensor, head, 1, 0, 0)
            if not (B[1]["consumer_before"] and B[1]["wrote"]):
                failures.append(f"n={n} B {name}: consumer/write unavailable")
                continue
            if above:
                kv = "V" if tensor.startswith("v_") else "K"
                tlabel = f"{kv}.h{head}.split_local"
                blabels = sorted(B[3].keys())
                if set(blabels) != set(A[3].keys()) or tlabel not in B[3]:
                    failures.append(f"n={n} B {name}: route labels {blabels} "
                                    f"vs A {sorted(A[3].keys())}")
                    continue
                db = {lb: diff(A[3][lb], B[3][lb]) for lb in blabels}
                if None in db.values():
                    failures.append(f"n={n} B {name}: vector absent {db}")
                    continue
                live_delta = db[tlabel]
                neighbors = [v for k, v in db.items() if k != tlabel]
                okB = live_delta is not None and live_delta > 0.0 and all(x == 0.0 for x in neighbors)
                clabels = sorted(C[3].keys())
                if set(clabels) != set(A[3].keys()):
                    failures.append(f"n={n} C {name}: route labels {clabels}")
                    continue
                dc = [diff(A[3][lb], C[3][lb]) for lb in clabels]
                okC = None not in dc and all(x == 0.0 for x in dc)
                # Remote round (occupied overflow): same gate shape, target on
                # split_overflow. Only meaningful when PREFILL already occupies
                # overflow (n-1 >= P): at n == P the proof decode is the first
                # overflow token, so there is nothing to mutate yet -> SKIP,
                # never a pass, never a fail.
                if n <= int(A[0].get("P", 0)):
                    print(f"  B/C {name}.remote: SKIP (prefill max pos {n-1} < "
                          f"P, overflow unoccupied)", flush=True)
                    continue
                rtensor = tensor.replace("k_stage", "remote_k").replace("v_stage", "remote_v")
                rname = name + ".remote"
                RB = run_branch("B", n, rtensor, head, 1, 0, 1)
                RC = run_branch("C", n, rtensor, head, 1, 0, 0)
                if not (RB[1]["consumer_before"] and RB[1]["wrote"]):
                    failures.append(f"n={n} B {rname}: consumer/write unavailable")
                    continue
                tpos = int(RB[0].get("tpos", -1))
                if tpos < int(A[0].get("P", 0)):
                    failures.append(f"n={n} B {rname}: target pos {tpos} not >= P "
                                    f"(remote unoccupied, gate invalid)")
                    continue
                rtlabel = f"{kv}.h{head}.split_overflow"
                rblabels = sorted(RB[3].keys())
                if set(rblabels) != set(A[3].keys()) or rtlabel not in RB[3]:
                    failures.append(f"n={n} B {rname}: route labels {rblabels}")
                    continue
                rdb = {lb: diff(A[3][lb], RB[3][lb]) for lb in rblabels}
                if None in rdb.values():
                    failures.append(f"n={n} B {rname}: vector absent")
                    continue
                rlive = rdb[rtlabel]
                rneigh = [v for k, v in rdb.items() if k != rtlabel]
                okRB = rlive is not None and rlive > 0.0 and all(x == 0.0 for x in rneigh)
                rclabels = sorted(RC[3].keys())
                if set(rclabels) != set(A[3].keys()):
                    failures.append(f"n={n} C {rname}: route labels {rclabels}")
                    continue
                rdc = [diff(A[3][lb], RC[3][lb]) for lb in rclabels]
                okRC = None not in rdc and all(x == 0.0 for x in rdc)
                rdlb = diff(A[2]["logits"], RB[2]["logits"])
                rdlc = diff(A[2]["logits"], RC[2]["logits"])
                print(f"  B/C {rname}@pos={tpos}: B target_delta={rlive} "
                      f"neighbors={rneigh} -> {'PASS' if okRB else 'FAIL'} | "
                      f"C all_delta={rdc} -> {'PASS' if okRC else 'FAIL'} | "
                      f"logits B={rdlb} C={rdlc}", flush=True)
                if not okRB:
                    failures.append(f"n={n} B {rname}: remote mutation delta={rlive} "
                                    f"neighbors={rneigh}")
                if not okRC:
                    failures.append(f"n={n} C {rname}: remote inactive changed state {rdc}")
                if okRC and rdlc not in (None, 0.0):
                    failures.append(f"n={n} C {rname}: remote inactive changed logits by {rdlc}")
            else:
                target = next(i for i, f in enumerate(FIELD) if f[2] == name)
                db = [diff(A[2]["after"][i], B[2]["after"][i]) for i in range(4)]
                if None in db:
                    failures.append(f"n={n} B {name}: vector absent {db}")
                    continue
                live_delta = db[target]
                neighbors = [db[i] for i in range(4) if i != target]
                okB = live_delta is not None and live_delta > 0.0 and all(x == 0.0 for x in neighbors)
                dc = [diff(A[2]["after"][i], C[2]["after"][i]) for i in range(4)]
                okC = None not in dc and all(x == 0.0 for x in dc)
            dlb = diff(A[2]["logits"], B[2]["logits"])
            dlc = diff(A[2]["logits"], C[2]["logits"])
            print(f"  B/C {name}: B target_delta={live_delta} neighbors={neighbors} "
                  f"-> {'PASS' if okB else 'FAIL'} | C all_delta={dc} "
                  f"-> {'PASS' if okC else 'FAIL'} | logits B={dlb} C={dlc}", flush=True)
            if not okB:
                failures.append(f"n={n} B {name}: live mutation delta={live_delta} "
                                f"neighbors={neighbors} (expected finite change, only target)")
            if not okC:
                failures.append(f"n={n} C {name}: inactive mutation changed live state {dc}")
            if okC and dlc not in (None, 0.0):
                failures.append(f"n={n} C {name}: inactive mutation changed logits by {dlc}")

    print("\n===== SUMMARY =====")
    if failures:
        for f in failures:
            print(f"FAIL {f}")
        print(f"{len(failures)} failures")
        return 1
    print("all branch gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
