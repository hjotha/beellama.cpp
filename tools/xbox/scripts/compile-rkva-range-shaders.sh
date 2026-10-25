#!/usr/bin/env bash
# Compile only the RKVA v2 positional attention shaders and embed their DXIL.
# Usage: compile-rkva-range-shaders.sh [target ...]   (default: every target)
# Passing targets regenerates only those headers, so the others stay
# byte-identical. DXC defaults to `dxc` on PATH; DXC=/path/to/dxc overrides.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DXC="${DXC:-dxc}"
ALL_TARGETS=(attn attn_q4 attn_q8 attn_kvarn4 reduce attn_q4_opt reduce_opt reduce_kvarn4
    attn_f16_opt attn_q8_opt attn_kvarn4_opt attn_kvarn4_prefix attn_kvarn4_opt_prefix
    attn_q4_mq2 attn_q4_mq4 attn_f16_mq2 attn_f16_mq4 attn_q8_mq2 attn_q8_mq4
    attn_kvarn4_mq2 attn_kvarn4_mq4 bulk_unpack_q_f16 bulk_pack_out_f16 bulk_unpack_q_fp8 bulk_pack_out_fp8 attn_kvarn4_decode_prefix)
if [[ $# -gt 0 ]]; then
    TARGETS=("$@")
else
    TARGETS=("${ALL_TARGETS[@]}")
fi
for target in "${TARGETS[@]}"; do
    known=0
    for candidate in "${ALL_TARGETS[@]}"; do
        if [[ "$candidate" == "$target" ]]; then known=1; fi
    done
    if [[ $known -eq 0 ]]; then
        echo "unknown target: $target" >&2
        exit 2
    fi
    entry=CSRangeAttn
    if [[ "$target" == reduce ]]; then entry=CSRangeReduce; fi
    source="$ROOT/shaders/rkva/rkva_range_attn.hlsl"
    defines=(-D RKVA_RANGE_STORAGE=0)
    if [[ "$target" == attn_q4 ]]; then defines=(-D RKVA_RANGE_STORAGE=1); fi
    if [[ "$target" == attn_q8 ]]; then defines=(-D RKVA_RANGE_STORAGE=2); fi
    if [[ "$target" == attn_kvarn4 ]]; then
        source="$ROOT/shaders/rkva/rkva_range_attn_kvarn4.hlsl"
        defines=()
    fi
    if [[ "$target" == reduce ]]; then
        source="$ROOT/shaders/rkva/rkva_range_reduce.hlsl"
        defines=()
    fi
    if [[ "$target" == attn_q4_opt ]]; then
        entry=CSRangeAttnQ4Opt
        source="$ROOT/shaders/rkva/rkva_range_attn_q4_opt.hlsl"
        defines=()
    fi
    if [[ "$target" == reduce_opt ]]; then
        entry=CSRangeReduceOpt
        source="$ROOT/shaders/rkva/rkva_range_reduce_opt.hlsl"
        defines=()
    fi
    if [[ "$target" == reduce_kvarn4 ]]; then
        entry=CSRangeReduceOpt
        source="$ROOT/shaders/rkva/rkva_range_reduce_opt.hlsl"
        defines=(-D RKVA_REDUCE_WHT=1)
    fi
    if [[ "$target" == attn_f16_opt || "$target" == attn_q8_opt ]]; then
        entry=CSRangeAttnGqaOpt
        source="$ROOT/shaders/rkva/rkva_range_attn_gqa_opt.hlsl"
        defines=(-D RKVA_OPT_STORAGE=0)
        if [[ "$target" == attn_q8_opt ]]; then defines=(-D RKVA_OPT_STORAGE=2); fi
    fi
    if [[ "$target" == attn_kvarn4_opt ]]; then
        entry=CSRangeAttnKvarn4Opt
        source="$ROOT/shaders/rkva/rkva_range_attn_kvarn4_opt.hlsl"
        defines=()
    fi
    # Descriptor revision 3 (serial query prefix, 256-row ORIGINAL-F16 tail ring):
    # same sources, separate PSOs; the legacy targets above stay byte-identical.
    if [[ "$target" == attn_kvarn4_prefix ]]; then
        source="$ROOT/shaders/rkva/rkva_range_attn_kvarn4.hlsl"
        defines=(-D RKVA_SERIAL_PREFIX=1)
    fi
    if [[ "$target" == attn_kvarn4_opt_prefix ]]; then
        entry=CSRangeAttnKvarn4Opt
        source="$ROOT/shaders/rkva/rkva_range_attn_kvarn4_opt.hlsl"
        defines=(-D RKVA_SERIAL_PREFIX=1)
    fi
    if [[ "$target" == attn_kvarn4_decode_prefix ]]; then
        entry=CSRangeAttnKvarn4Decode
        source="$ROOT/shaders/rkva/rkva_range_attn_kvarn4_opt.hlsl"
        defines=(-D RKVA_SERIAL_PREFIX=1 -D RKVA_DECODE_OPT=1)
    fi
    # Q4_0 multi-query tile (nq 2..9): RKVA_MQ_QT queries share each K/V dequantization.
    if [[ "$target" == attn_q4_mq2 || "$target" == attn_q4_mq4 ]]; then
        entry=CSRangeAttnQ4Mq
        source="$ROOT/shaders/rkva/rkva_range_attn_q4_mq.hlsl"
        defines=(-D RKVA_MQ_QT="${target#attn_q4_mq}")
    fi
    # F16 / Q8_0 multi-query tile (nq 2..9): same algorithm, storage-specific K/V decode.
    if [[ "$target" =~ ^attn_(f16|q8)_mq([24])$ ]]; then
        entry=CSRangeAttnFmtMq
        source="$ROOT/shaders/rkva/rkva_range_attn_fmt_mq.hlsl"
        storage=0
        if [[ "${BASH_REMATCH[1]}" == q8 ]]; then storage=2; fi
        defines=(-D RKVA_OPT_STORAGE="$storage" -D RKVA_MQ_QT="${BASH_REMATCH[2]}")
    fi
    # Native KVarN4 descriptor-3 multi-query tile (nq 2..9): 2 or 4 queries share
    # each record/stage/tail K/V decode; every query keeps its own serial prefix.
    if [[ "$target" == attn_kvarn4_mq2 || "$target" == attn_kvarn4_mq4 ]]; then
        entry=CSRangeAttnKvarn4Mq
        source="$ROOT/shaders/rkva/rkva_range_attn_kvarn4_mq.hlsl"
        defines=(-D RKVA_MQ_QT="${target#attn_kvarn4_mq}")
    fi
    if [[ "$target" == bulk_unpack_q_f16 ]]; then
        entry=CSBulkUnpackQF16
        source="$ROOT/shaders/rkva/rkva_range_bulk_f16.hlsl"
        defines=()
    fi
    if [[ "$target" == bulk_pack_out_f16 ]]; then
        entry=CSBulkPackOutF16
        source="$ROOT/shaders/rkva/rkva_range_bulk_f16.hlsl"
        defines=()
    fi
    if [[ "$target" == bulk_unpack_q_fp8 ]]; then
        entry=CSBulkUnpackQFP8
        source="$ROOT/shaders/rkva/rkva_range_bulk_f16.hlsl"
        defines=()
    fi
    if [[ "$target" == bulk_pack_out_fp8 ]]; then
        entry=CSBulkPackOutFP8
        source="$ROOT/shaders/rkva/rkva_range_bulk_f16.hlsl"
        defines=()
    fi
    out="$ROOT/shaders/generated/rkva_range_${target}_dxil.h"
    tmp="$(mktemp)"
    trap 'rm -f "$tmp"' EXIT
    "$DXC" -T cs_6_0 -E "$entry" "${defines[@]}" "$source" -Fo "$tmp"
    python3 - "$tmp" "$out" "$source" "$target" <<'PY'
import hashlib
import pathlib
import sys
binary, output, source, target = sys.argv[1:]
data = pathlib.Path(binary).read_bytes()
lines = [
    '// Generated by scripts/compile-rkva-range-shaders.sh; do not edit.',
    '// Target specialization: ' + target,
    '// Source SHA256: ' + hashlib.sha256(pathlib.Path(source).read_bytes()).hexdigest(),
    '#pragma once', '#include <cstdint>', '// clang-format off',
    'static const uint8_t g_rkva_range_' + target + '_dxil[] = {',
]
for offset in range(0, len(data), 12):
    lines.append('    ' + ', '.join(f'0x{value:02x}' for value in data[offset:offset+12]) + ',')
lines += ['};', '// clang-format on', '']
pathlib.Path(output).write_text('\n'.join(lines))
print(output, len(data), 'bytes')
PY
    rm -f "$tmp"
    trap - EXIT
done
