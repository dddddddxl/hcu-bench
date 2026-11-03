# BERT Default Runtime Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow the BERT FP16 recipe to run with the prepared environment's rocBLAS, preserving strict optional tuned-library execution and recording the distinction.

**Architecture:** An empty rocblas_dir selects runtime_default and leaves library search settings unchanged. An explicitly supplied directory selects custom_tuned and retains existence, nonempty-tree, shared-library and fingerprint checks; it must never silently fall back. Model input/dtype/batch validation and three-round measurement requirements remain intact. Results identify the effective mode so custom-tuned historical baselines are not confused with runtime-default results.

**Tech Stack:** Existing MIGraphX native perf wrapper, Python standard library, YAML templates, existing onnx/psutil environment.

## Tasks

1. Add regression tests for default BERT runtime, explicit missing/empty tuning directories, preservation of LD_LIBRARY_PATH, and distinct report provenance. Use CPU substitutes only for wrapper tests and label them as such.
2. Make the directory optional, update the manifest/default configuration and add a BERT-only in-container template. Preserve all existing model and measurement checks.
3. Document runtime-default versus custom-tuned use, model preparation requirements and the fresh MIGraphX image workflow. No package/library replacement and no silent model conversion.
4. Run focused tests, then execute an actual BERT ONNX on an available DCU environment at batch 1/32, three rounds × 20 iterations where supported. Keep logs, full software/model provenance and cleanup evidence.
5. Submit a PR only after the requested validation is satisfied; if the node/model is unavailable, retain local changes and clearly report the missing hardware verification instead of claiming CPU fixtures prove BERT runs.
