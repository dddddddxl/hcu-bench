# MIGraphX Dependency Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the ResNet50 recipe usable from a fresh, explicitly documented MIGraphX image and diagnose missing native dependencies before measurement.

**Architecture:** Use the user-approved specific-image delivery option rather than inventing a universal Torch-image installer. Pin the verified SourceFind MIGraphX 5.2.0 image, discover an existing DTK driver when PATH is incomplete, and provide a target-side CPU-only dependency check. Explicit binary overrides remain strict and installed runtime libraries are never replaced.

**Tech Stack:** Existing Python 3.10+ CLI, standard library, MIGraphX native driver, numactl, existing onnx/psutil optional dependency group.

## Tasks

1. Add tests for PATH discovery, /opt/dtk/bin fallback, explicit-path failure, missing-library diagnostics, required perf flags and missing Python dependencies. Preserve static bench check behaviour.
2. Implement testpacks/migraphx/dependencies.py and check.py. Reuse discovery/library validation in the run wrapper and keep measurements unchanged. Run targeted unittest regressions.
3. Add a ResNet50-only in-container configuration and document a fresh-container workflow with the verified image tag/digest, required native tools, Python preparation, ONNX model constraints and dependency check command. Update the managed template's image reference.
4. Verify the new image from scratch without GPU access: binary/version, ldd, perf help and Python dependency reporting. Complete Python setup in an isolated task container and, if an appropriate model and idle GPU are available, execute the real wrapper.
5. Run appropriate contract tests, capture actual validation and limitations, review the diff, commit and open a PR against current main. No models, binaries, credentials or private installation archives are committed.
