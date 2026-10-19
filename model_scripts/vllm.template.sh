#!/usr/bin/env bash
set -euo pipefail

# Copy to model_b.local.sh. Quantization/parallelism/environment are model-specific.
MODEL_PATH=/YOUR_LOCAL_MODEL
if [[ "$MODEL_PATH" == /YOUR_* ]]; then
    echo 'Edit MODEL_PATH and the model-specific launch options in this script.' >&2
    exit 2
fi
exec vllm serve "$MODEL_PATH" --tensor-parallel-size 8 \
    --host "${BENCH_HOST:-127.0.0.1}" --port "${BENCH_PORT:-30000}"
