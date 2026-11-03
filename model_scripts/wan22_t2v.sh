#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${BENCH_MODEL_PATH:?Set the local Wan Diffusers directory}"
PROFILE="${BENCH_WAN_PROFILE:?Expected tp1 or tp4}"
[[ -f "$MODEL_PATH/model_index.json" ]] || { echo "Missing Wan model_index.json: $MODEL_PATH" >&2; exit 2; }
command -v sglang >/dev/null || { echo 'Missing SGLang diffusion CLI' >&2; exit 2; }

case "$PROFILE" in
  tp1) HCUS=1; FSDP=false; LAYERWISE=true ;;
  tp4) HCUS=4; FSDP=true; LAYERWISE=false ;;
  *) echo "Unsupported Wan profile: $PROFILE" >&2; exit 2 ;;
esac

# The reference sheet used two known warmup interfaces. Other interfaces fail explicitly.
HELP="$(sglang serve --help 2>&1)"
if [[ "$HELP" == *'--warmup-mode'* ]]; then
  WARMUP=(--warmup-mode server)
elif grep -Eq -- '(^|[[:space:],])--warmup([[:space:],=]|$)' <<< "$HELP"; then
  WARMUP=(--warmup)
else
  echo 'Unsupported diffusion warmup interface; inspect sglang serve --help' >&2
  exit 2
fi

export GPU_MAX_HW_QUEUES=3
# Do not silently add the Cache-DiT optimization from the separate BW1100 example.
export SGLANG_CACHE_DIT_ENABLED=false
ARGS=(serve --model-type diffusion --model-path "$MODEL_PATH"
  --host "${BENCH_HOST:-127.0.0.1}" --port "${BENCH_PORT:-30005}"
  --pin-cpu-memory --num-gpus "$HCUS"
  --attention-backend sla_attn --attention-backend-config sla_topk=0.25
  --dit-cpu-offload false --text-encoder-cpu-offload true --vae-cpu-offload false
  --dit-layerwise-offload "$LAYERWISE" --use-fsdp-inference "$FSDP"
  --ulysses-degree "$HCUS" --ring-degree 1 "${WARMUP[@]}"
  --warmup-resolutions 1280x720 --warmup-steps 1
  --output-path "${BENCH_OUTPUT_DIR:?}/videos")
[[ "$PROFILE" != tp1 ]] || ARGS+=(--performance-mode memory)
exec sglang "${ARGS[@]}"
