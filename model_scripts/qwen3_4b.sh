#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${BENCH_MODEL_PATH:?Set models.qwen3_4b.env.BENCH_MODEL_PATH in YAML}"
[[ -d "$MODEL_PATH" ]] || { echo "Missing local model directory: $MODEL_PATH" >&2; exit 2; }
# Generic BF16 / TP1 recipe; add only options validated for the target HCU/image.
# Thinking is explicitly controlled by the accuracy/performance request settings.
exec python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" --dtype bfloat16 --tp-size 1 \
    --host "${BENCH_HOST:-127.0.0.1}" --port "${BENCH_PORT:-30000}" \
    --trust-remote-code --reasoning-parser qwen3
