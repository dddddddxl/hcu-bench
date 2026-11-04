#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# HCU Bench integration changes; original user-supplied sources retained.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
# Review native check diagnostics before executing the perf script.
python3 run_gemm_cases_legacy.py --dtype fp32 --mode perf "$@"
python3 run_gemm_cases_legacy.py --dtype fp16 --mode perf "$@"
python3 run_gemm_cases_legacy.py --dtype fp64 --mode perf "$@"
python3 run_gemm_cases_legacy.py --dtype complex64 --mode perf "$@"
python3 run_gemm_cases_legacy.py --dtype complex128 --mode perf "$@"
