#!/usr/bin/env bash
# F4: run one position-split frontier case under compute-sanitizer.
#
# usage: run-sanitizer.sh <P> <C> <n_tokens> <decode> <outdir> [tool] [port]
#   tool defaults to memcheck; initcheck is also supported.
#
# Contract enforced by this runner (absence of a report is NOT a pass):
#   * the sanitizer process is started with job control ON and is never disowned,
#     so its exit status and its full report are observable;
#   * the TARGET llama-server is identified by walking this script's own process
#     tree BEFORE any signal is sent, and only that PID receives SIGTERM/SIGKILL;
#   * the sanitizer is then WAITED on so it can flush and print its summary;
#   * a run is green only if: the sanitizer exited 0, its summary line is
#     PRESENT with a zero error count, the HTTP request returned 200, and the
#     expected number of tokens was evaluated. Missing summary, nonzero CUDA or
#     initcheck errors, timeout, or a forced cleanup without a report are all
#     inconclusive/failures.
#
# Exit codes: 0 green, 2 startup timeout, 3 bad HTTP, 4 sanitizer error count,
# 5 missing summary, 6 timeout waiting for the sanitizer, 7 forced cleanup.
set -u

P="$1"; C="$2"; N="$3"; DECODE="$4"; OUT="$5"; TOOL="${6:-memcheck}"
PORT="${7:-20700}"
ROOT=/home/hjotha/beellama.cpp
SERVER="$ROOT/build-optimized/bin/llama-server"
MODEL=/home/hjotha/models/Qwen3.5-4B-Q4_K_M.gguf
SAN=/opt/cuda/bin/compute-sanitizer
TAG="P${P}-C${C}-n${N}-d${DECODE}-${TOOL}"

mkdir -p "$OUT"
LOG="$OUT/$TAG.log"
SUM="$OUT/$TAG.summary"

if [ ! -x "$SAN" ]; then
    echo "BLOCKED: compute-sanitizer not found at $SAN" | tee "$SUM"
    exit 127
fi

export LLAMA_KV_POSITION_SPLIT_P="$P"
export TMPDIR=/home/hjotha/occupancy-f4-spool
export CUDA_LAUNCH_BLOCKING=1

set -m   # job control on: the background job is a real job we can wait on

"$SAN" --tool "$TOOL" --target-processes application-only --error-exitcode 99 \
    "$SERVER" --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --device CUDA0 -ngl 99 --split-mode none --main-gpu 0 \
    --ctx-size "$C" --parallel 1 --batch-size 256 --ubatch-size 256 \
    --cache-type-k kvarn4 --cache-type-v kvarn4 \
    --kv-tail-tokens 0 --kv-tail-type f16 --flash-attn on \
    --cache-prompt --fit off --no-ui --slots --no-warmup --no-context-shift \
    --remote-attn vulkan:0 --remote-attn-layers 8 \
    --remote-attn-cache-type-k q4_0 --remote-attn-cache-type-v q4_0 \
    --spec-type none > "$LOG" 2>&1 &
SAN_PID=$!
echo "$SAN_PID" > "$OUT/$TAG.pid"

# Walk OUR OWN tree to find the target llama-server PID. Only this PID is ever
# signalled; nothing here matches by process name.
find_target() {
    local frontier="$SAN_PID" kids next
    for _ in 1 2 3 4 5; do
        next=""
        for p in $frontier; do
            kids=$(pgrep -P "$p" 2>/dev/null || true)
            next="$next $kids"
        done
        [ -z "${next// /}" ] && break
        for p in $next; do
            if [ -r "/proc/$p/comm" ] && [ "$(cat "/proc/$p/comm" 2>/dev/null)" = "llama-server" ]; then
                echo "$p"; return 0
            fi
        done
        frontier="$next"
    done
    return 1
}

# wait for readiness (the sanitizer serializes kernels, so startup is slow)
READY=0
for _ in $(seq 1 900); do
    if ! kill -0 "$SAN_PID" 2>/dev/null; then break; fi
    code=$(curl -s -o /dev/null -m 3 -w '%{http_code}' "http://127.0.0.1:${PORT}/health" || true)
    if [ "$code" = "200" ]; then READY=1; break; fi
    sleep 2
done

HTTP_CODE=000
EVAL_TOKENS=0
if [ "$READY" = "1" ]; then
    python3 - "$N" "$PORT" "$DECODE" > "$OUT/$TAG.request" 2>&1 <<'PYREQ'
import json, sys, urllib.request
n_tok, port, n_predict = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
base = "http://127.0.0.1:%d" % port

def post(path, payload, timeout):
    req = urllib.request.Request(base + path,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode() or "{}")

text = " ".join("w%d" % i for i in range(n_tok * 2))
_, tok = post("/tokenize", {"content": text}, 180)
ids = tok["tokens"][:n_tok]
st, body = post("/completion",
                {"prompt": ids, "n_predict": n_predict, "temperature": 0.0,
                 "cache_prompt": True, "seed": 42}, 1200)
print("http=%d tokens_evaluated=%d expected_prompt=%d" % (
    st, body.get("tokens_evaluated", 0), n_tok))
PYREQ
    REQ_LINE=$(cat "$OUT/$TAG.request")
    echo "request: $REQ_LINE" | tee -a "$LOG"
    HTTP_CODE=$(echo "$REQ_LINE" | sed -n 's/.*http=\([0-9][0-9]*\).*/\1/p')
    EVAL_TOKENS=$(echo "$REQ_LINE" | sed -n 's/.*tokens_evaluated=\([0-9][0-9]*\).*/\1/p')
    [ -n "$HTTP_CODE" ] || HTTP_CODE=000
    [ -n "$EVAL_TOKENS" ] || EVAL_TOKENS=0
else
    echo "server never became ready" | tee -a "$LOG"
fi

# Graceful shutdown of ONLY the target, then WAIT for the sanitizer to report.
TARGET_PID=$(find_target || true)
FORCED=0
if [ -n "${TARGET_PID:-}" ] && kill -0 "$TARGET_PID" 2>/dev/null; then
    echo "target_pid=$TARGET_PID SIGTERM" | tee -a "$LOG"
    kill -TERM "$TARGET_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do
        kill -0 "$TARGET_PID" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$TARGET_PID" 2>/dev/null; then
        echo "target did not exit on SIGTERM; sending SIGKILL, cleanup forced" | tee -a "$LOG"
        kill -KILL "$TARGET_PID" 2>/dev/null || true
        FORCED=1
    fi
else
    echo "target already gone, pid=${TARGET_PID:-none}" | tee -a "$LOG"
fi

# Wait for the sanitizer so it flushes and prints the full report.
SAN_EXIT=124
for _ in $(seq 1 180); do
    if ! kill -0 "$SAN_PID" 2>/dev/null; then
        wait "$SAN_PID" 2>/dev/null
        SAN_EXIT=$?
        break
    fi
    sleep 2
done
if [ "$SAN_EXIT" = "124" ]; then
    echo "sanitizer did not exit; timeout waiting for report" | tee -a "$LOG"
fi

# Parse the report. grep -c prints 0 AND exits nonzero when there is no match,
# which previously produced a duplicated bogus "0" line; count explicitly.
SUMMARY_LINE=$(grep -E "ERROR SUMMARY" "$LOG" | tail -1 || true)
ERR_REPORTED=0
ERR_COUNT=""
if [ -n "$SUMMARY_LINE" ]; then
    ERR_COUNT=$(echo "$SUMMARY_LINE" | sed -n 's/.*ERROR SUMMARY: *\([0-9][0-9]*\).*/\1/p')
    [ -n "$ERR_COUNT" ] && ERR_REPORTED=1
fi
CUDA_ERR=$(grep -cE "cudaError|CUDA error" "$LOG" || true)
INIT_ERR=$(grep -cE "Uninitialized|uninitialized" "$LOG" || true)
INVALID=$(grep -cE "Invalid __global__|Invalid read|Invalid write|Invalid __shared__" "$LOG" || true)
LEAKED=$(grep -cE "Leaked [0-9]" "$LOG" || true)
LEFTOVER=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')

RC=0
REASON=""
if [ "$SAN_EXIT" = "124" ]; then RC=6; REASON="sanitizer timeout"; fi
if [ "$READY" != "1" ]; then RC=2; REASON="startup timeout"; fi
if [ "$HTTP_CODE" != "200" ]; then RC=3; REASON="http $HTTP_CODE"; fi
if [ "$EVAL_TOKENS" != "$N" ]; then RC=3; REASON="tokens_evaluated=$EVAL_TOKENS expected=$N"; fi
if [ "$ERR_REPORTED" != "1" ]; then RC=5; REASON="sanitizer summary missing"; fi
if [ "$ERR_REPORTED" = "1" ] && [ "${ERR_COUNT:-1}" != "0" ]; then RC=4; REASON="sanitizer reported ${ERR_COUNT} errors"; fi
if [ "$SAN_EXIT" = "99" ]; then RC=4; REASON="sanitizer --error-exitcode 99"; fi
if [ "$INVALID" != "0" ]; then RC=4; REASON="invalid accesses=$INVALID"; fi
if [ "$CUDA_ERR" != "0" ]; then RC=4; REASON="cuda errors=$CUDA_ERR"; fi
if [ "$INIT_ERR" != "0" ]; then RC=4; REASON="initcheck findings=$INIT_ERR"; fi
if [ "$FORCED" = "1" ] && [ "$ERR_REPORTED" != "1" ]; then RC=7; REASON="forced cleanup without report"; fi

{
    echo "tool=$TOOL"
    echo "P=$P C=$C n_tokens=$N decode=$DECODE port=$PORT"
    echo "sanitizer_pid=$SAN_PID sanitizer_exit=$SAN_EXIT target_pid=${TARGET_PID:-none} forced_cleanup=$FORCED"
    echo "http=$HTTP_CODE tokens_evaluated=$EVAL_TOKENS"
    echo "sanitizer_summary=${SUMMARY_LINE:-MISSING}"
    echo "error_count=${ERR_COUNT:-unknown} error_summary_present=$ERR_REPORTED"
    echo "invalid_accesses=$INVALID cuda_errors=$CUDA_ERR initcheck_findings=$INIT_ERR leaked=$LEAKED"
    echo "leftover_gpu_mib=$LEFTOVER"
    echo "verdict=$RC reason=${REASON:-green}"
    echo "log=$LOG"
} | tee "$SUM"

exit "$RC"
