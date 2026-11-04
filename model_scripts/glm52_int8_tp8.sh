#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${BENCH_MODEL_PATH:?Set models.glm52_int8.env.BENCH_MODEL_PATH in YAML}"
[[ -d "$MODEL_PATH" ]] || { echo "Missing local model directory: $MODEL_PATH" >&2; exit 2; }
IFS=',' read -r -a HCUS <<< "${HIP_VISIBLE_DEVICES:?Assign eight HCUs in YAML}"
[[ ${#HCUS[@]} -eq 8 ]] || { echo 'GLM-5.2 INT8 TP8 requires exactly eight assigned HCUs.' >&2; exit 2; }
declare -A SEEN=()
for hcu in "${HCUS[@]}"; do
    [[ -n "$hcu" && -z "${SEEN[$hcu]+set}" ]] || { echo 'HCU assignments must be nonempty and unique.' >&2; exit 2; }
    SEEN[$hcu]=1
done
for name in NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME; do
    iface="${!name:-}"
    [[ -n "$iface" && -d "/sys/class/net/$iface" ]] || { echo "$name must name an existing interface; set vars.iface in YAML." >&2; exit 2; }
done

# HYGON cookbook: GLM-5.2 Channel INT8, IFB BW1100 8x, SGLang 0.5.12.
# Platform NICs are inherited from YAML, not hard-coded in this recipe.
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=0
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0

exec sglang serve \
    --model-path "$MODEL_PATH" \
    --host "${BENCH_HOST:-127.0.0.1}" --port "${BENCH_PORT:-30000}" \
    --trust-remote-code --tp-size 8 \
    --nsa-prefill-backend flashmla_auto --nsa-decode-backend flashmla_kv \
    --dtype bfloat16 --dist-timeout 10000 --watchdog-timeout 3600 \
    --page-size 64 --kv-cache-dtype fp8_e4m3 --chunked-prefill-size 16384 \
    --cuda-graph-max-bs 16 --max-running-requests 32 \
    --reasoning-parser glm45 --tool-call-parser glm47 \
    --speculative-algorithm EAGLE --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --quantization slimquant_marlin --mem-fraction-static 0.9
