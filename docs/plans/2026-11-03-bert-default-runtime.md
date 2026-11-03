# BERT Default Runtime Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow the BERT FP16 recipe to run with the prepared environment's rocBLAS, preserving strict optional tuned-library execution and recording the distinction.

**Architecture:** An empty rocblas_dir selects runtime_default and leaves library search settings unchanged. An explicitly supplied directory selects custom_directory and retains existence, nonempty-tree, shared-library and fingerprint checks; it must never silently fall back. The label records configuration and does not certify that a library is tuned. Model input/dtype/batch validation and three-round measurement requirements remain intact. Results identify the mode so custom-library historical baselines are not confused with runtime-default results.

**Tech Stack:** Existing MIGraphX native perf wrapper, Python standard library, YAML templates, existing onnx/psutil environment.

## Tasks

1. Add regression tests for default BERT runtime, explicit missing/empty tuning directories, preservation of LD_LIBRARY_PATH, and distinct report provenance. Use CPU substitutes only for wrapper tests and label them as such.
2. Make the directory optional, update the manifest/default configuration and add a BERT-only in-container template. Preserve all existing model and measurement checks.
3. Document runtime-default versus custom-tuned use, model preparation requirements and the fresh MIGraphX image workflow. No package/library replacement and no silent model conversion.
4. Validate dependency checks, actual ONNX input/batch checks, and the public BERT execution/report pipeline with explicitly labeled CPU native-process fixtures. Run Linux CI. Keep hardware inference/performance validation separate and do not claim that CPU fixtures prove BERT GPU support.
5. User clarified that the deliverable is a complete repository workflow for colleagues using their own compatible nodes. Submit the PR after portable startup and regression checks pass; document required hardware/runtime/model prerequisites and the absence of real DCU BERT inference/performance validation.
