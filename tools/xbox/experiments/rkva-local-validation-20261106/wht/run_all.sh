#!/usr/bin/env bash
# CPU-only source model. Never builds/deploys an Xbox artifact or edits production.
set -euo pipefail
export LC_ALL=C
unset PYTHONOPTIMIZE
cd "$(dirname "$0")"
ROOT="$PWD"
OUT="$ROOT/.build"
# A failed rerun must not leave an earlier successful result visible.
mkdir -p "$OUT"
rm -f -- "$OUT/results.json" "$OUT/cpu_results.json" "$OUT/structure_results.json" \
    "$OUT/sanitized_results.json" "$OUT/source_hashes.json"
PYTHON="${PYTHON:-python3}"
CXX="${CXX:-g++}"
WHT_DETECT_LEAKS="${WHT_DETECT_LEAKS:-0}"
export CXX WHT_DETECT_LEAKS
case "$WHT_DETECT_LEAKS" in 0|1) ;; *) echo 'WHT_DETECT_LEAKS must be 0 or 1' >&2; exit 2;; esac
for tool in "$PYTHON" "$CXX" patch cmp; do command -v "$tool" >/dev/null || { echo "Missing dependency: $tool" >&2; exit 2; }; done
"$PYTHON" -c 'import numpy'
"$PYTHON" "$ROOT/build_prototype.py" "$OUT" > "$OUT/build_prototype.log"
"$PYTHON" "$ROOT/generate_cpu_kernel.py" "$OUT"
flags=(-std=c++20 -Wall -Wextra -Werror -fno-fast-math -ffp-contract=off -frounding-math -I "$OUT")
"$CXX" "${flags[@]}" -O2 "$ROOT/validate.cpp" -o "$OUT/validate"
(cd "$OUT" && ./validate > cpu_results.json)
"$PYTHON" "$ROOT/validate_structure.py" "$OUT" > "$OUT/structure.log"
"$CXX" "${flags[@]}" -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer "$ROOT/validate.cpp" -o "$OUT/validate_sanitized"
(cd "$OUT" && ASAN_OPTIONS="detect_leaks=$WHT_DETECT_LEAKS" UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ./validate_sanitized > sanitized_results.json)
cmp "$OUT/cpu_results.json" "$OUT/sanitized_results.json"
"$PYTHON" "$ROOT/summarize_results.py" "$OUT"
echo 'Detailed local results: .build/results.json'
