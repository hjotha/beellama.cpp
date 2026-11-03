#!/bin/bash
set -euo pipefail

echo "==> Waiting for llama-server slot 0 to become idle..."
while true; do
    RESP=$(curl -s "http://127.0.0.1:8090/slots?model=qwen-3.8-27b" || true)
    if [[ -z "$RESP" ]]; then
        echo "Server not responding, proceeding..."
        break
    fi
    IS_PROC=$(echo "$RESP" | jq -r '.[0].is_processing // false' 2>/dev/null || echo "false")
    if [[ "$IS_PROC" == "false" ]]; then
        echo "Slot 0 is idle! Proceeding with restart..."
        break
    fi
    N_DEC=$(echo "$RESP" | jq -r '.[0].next_token[0].n_decoded // 0' 2>/dev/null || echo "0")
    N_REM=$(echo "$RESP" | jq -r '.[0].next_token[0].n_remain // 0' 2>/dev/null || echo "0")
    PROG=$(echo "$RESP" | jq -r '.[0].n_prompt_tokens_processed // 0' 2>/dev/null || echo "0")
    TOTAL=$(echo "$RESP" | jq -r '.[0].n_prompt_tokens // 0' 2>/dev/null || echo "0")
    echo "Processing active: prompt=$PROG/$TOTAL, decoded=$N_DEC, remain=$N_REM. Waiting..."
    sleep 3
done

echo "==> Restarting llama-server-root.service..."
echo amdk6x | sudo -S systemctl restart llama-server-root.service

echo "==> Waiting for llama-server-root to initialize..."
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:8090/health 2>/dev/null | grep -q "ok"; then
        echo "==> Server is healthy on port 8090!"
        break
    fi
    sleep 2
done

echo "==> Checking loaded model configuration in journal..."
journalctl -u llama-server-root.service -n 50 --no-pager
