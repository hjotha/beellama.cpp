#!/usr/bin/env bash
# Remote DXC wrapper for compiling shaders on Windows build host 192.168.1.193.
# Usage: dxc-remote-193.sh [dxc arguments...]
set -euo pipefail

REMOTE_HOST="hjotha@192.168.1.193"
REMOTE_DIR="C:/Users/hjotha/build/rkva-shaders"
DXC_EXE="C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/dxc.exe"

# Synchronize local shader dir to remote directory once or on demand
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHADER_DIR="$(cd "$SCRIPT_DIR/../shaders/rkva" && pwd)"

# Find the source file and output file from arguments
args=()
source_file=""
output_file=""
skip_next=0

for i in "$@"; do
    if [[ $skip_next -eq 1 ]]; then
        output_file="$i"
        skip_next=0
        continue
    fi
    if [[ "$i" == "-Fo" ]]; then
        skip_next=1
        continue
    fi
    if [[ "$i" == *.hlsl || "$i" == *.hlsli ]]; then
        source_file="$i"
    else
        args+=("$i")
    fi
done

if [[ -z "$source_file" || -z "$output_file" ]]; then
    echo "Error: source file or output file (-Fo) missing" >&2
    exit 1
fi

src_basename="$(basename "$source_file")"
remote_src="$REMOTE_DIR/$src_basename"
remote_out="$REMOTE_DIR/out_$(date +%s%N).dxil"

# Copy source directory files to remote (only changed files if possible, or direct scp)
scp -q "$SHADER_DIR"/* "$REMOTE_HOST:$REMOTE_DIR/"

# Execute remote dxc
remote_cmd="& '$DXC_EXE' ${args[*]} '$remote_src' -Fo '$remote_out'"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" "powershell -NoProfile -Command \"$remote_cmd\""

# Copy back the output dxil
scp -q "$REMOTE_HOST:$remote_out" "$output_file"

# Cleanup remote temporary output
ssh -o BatchMode=yes "$REMOTE_HOST" "powershell -NoProfile -Command \"Remove-Item -Force '$remote_out' -ErrorAction SilentlyContinue\""
