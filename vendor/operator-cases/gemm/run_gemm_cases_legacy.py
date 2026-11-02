#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# HCU Bench integration changes; original user-supplied sources retained.
"""Explicit rocBLAS 4.3.0 entry point for the shared GEMM cases."""
import sys
sys.dont_write_bytecode = True
from run_gemm_cases import entrypoint

if __name__ == "__main__":
    sys.exit(entrypoint("legacy"))
