#!/usr/bin/env bash
# Compile only archived research fixtures. Never replace shipping headers.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DXC="${DXC:-dxc}"
OUT="$HERE/.build/dxil"
command -v "$DXC" >/dev/null 2>&1 || { echo "DXC unavailable: $DXC" >&2; exit 2; }
mkdir -p "$OUT"
"$DXC" --version > "$OUT/dxc-version.txt"
: > "$OUT/commands.log"
compile() {
    local name="$1" entry="$2" source="$3"
    shift 3
    local cmd=("$DXC" -T cs_6_0 -E "$entry" "$@" "$source"
        -Fo "$OUT/$name.dxil" -Fc "$OUT/$name.ll")
    printf '%q ' "${cmd[@]}" >> "$OUT/commands.log"
    printf '\n' >> "$OUT/commands.log"
    "${cmd[@]}"
}
for rowlane in 0 1; do
    compile "qrot-baseline-rowlane-$rowlane" CSRangeAttnKvarn4Decode \
        "$HERE/qrot/source-main/rkva_range_attn_kvarn4_opt.hlsl" \
        -D RKVA_SERIAL_PREFIX=1 -D RKVA_DECODE_OPT=1 -D "RKVA_DECODE_ROWLANE=$rowlane"
    for reuse in 0 1; do
        compile "qrot-reuse-$reuse-rowlane-$rowlane" CSRangeAttnKvarn4Decode \
            "$HERE/qrot/prototype/rkva_range_attn_kvarn4_opt.hlsl" \
            -D RKVA_SERIAL_PREFIX=1 -D RKVA_DECODE_OPT=1 \
            -D "RKVA_DECODE_ROWLANE=$rowlane" -D "RKVA_DECODE_QROT_REUSE=$reuse"
    done
done
compile wht-baseline CSRangeReduceOpt "$HERE/wht/baseline.hlsl" -D RKVA_REDUCE_WHT=1
compile wht-pingpong CSRangeReduceOpt "$HERE/wht/rkva_range_reduce_opt_pingpong.hlsl" -D RKVA_REDUCE_WHT=1
echo "DXIL/disassembly written to $OUT. Compilation is not GPU qualification."
