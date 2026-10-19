#!/usr/bin/env bash
set -euo pipefail

# Copy to model_a.local.sh and use your validated model-specific command/env.
MODEL_PATH=/YOUR_LOCAL_MODEL
if [[ "$MODEL_PATH" == /YOUR_* ]]; then
    echo 'Edit MODEL_PATH and the model-specific launch options in this script.' >&2
    exit 2
fi
# Keep HIP_VISIBLE_DEVICES from Bench; TP must fit the assigned HCUs.
# Do not start Docker, nohup, background jobs, or stop other services here.
exec python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" --tp 8 \
    --host "${BENCH_HOST:-127.0.0.1}" --port "${BENCH_PORT:-30000}"
