# MIGraphX dependency delivery validation — 2026-11-02

This change addresses ResNet50 users whose execution image does not expose migraphx-driver. The selected delivery is a documented, versioned vendor image plus Python dependency setup, not automatic installation of a native SDK into arbitrary Torch images.

## Fresh-image evidence

New isolated containers were created from the existing local image, without inheriting another user's container filesystem:

- Tag: `harbor.sourcefind.cn:5443/dcu/admin/base/migraphx:5.2.0-ubuntu22.04-dtk26.04-py3.11`.
- Registry manifest digest: `sha256:bf6e6a8d8d7d124dbba7cf49eb598415d0e10f6bdb5901eb2d8199ee2deea360`.
- Local Docker image ID: `sha256:12847869eec65e0015f70063f4f8589ba3e007a7f297d06a8e9a61a94a29a527`.
- Ubuntu 22.04, Python 3.11, DTK-26.04-rc4, `/opt/dtk` resolving to `/opt/dtk-26.04`.
- `/opt/dtk/bin/migraphx-driver --version`: MIGraphX 5.2.0; perf help exposes FP16, INT8 and opt-compile flags.
- `/usr/bin/numactl` is present; driver ldd has no missing shared libraries.
- PyYAML 6.0.3 and psutil 7.2.2 are included, **onnx is not included**.

The driver is already in the image's native DTK tree; it is not provided by pip extras or the new repository scripts. Registry pull access/authentication must be supplied by the user; the image was locally present, so this validation did not establish anonymous registry download access.

## Reproduction

All checks used new --init / --network none containers and a separate task directory on the authorised Zhengzhou node. Host hyhal was mounted read-only for native runtime loading. No GPU devices or ONNX model were passed to these containers.

1. Base Python: `check.py --recipe resnet50_fp16` returned exit 2 and ready=false, identifying missing onnx while preserving successful native driver/library/flag evidence.
2. A separate --system-site-packages virtual environment was created. Linux/Python 3.11 onnx wheels were downloaded on the control machine, copied into the task directory, and `pip install --no-index --find-links=/work/wheels --no-build-isolation -e /work/repo[migraphx]` completed without modifying the vendor image or another user's container.
3. FP16 and INT8 dependency checks returned exit 0 and ready=true. FP16 was checked with PATH limited to `/usr/bin:/bin`, confirming standard DTK-path discovery works without a manual PATH export.
4. Effective Python packages were onnx 1.17.0, numpy 1.25.0, protobuf 6.33.6, psutil 7.2.2 and PyYAML 6.0.3.

Missing native ldd libraries and native runtime-loader errors remain failures; discovery does not silently switch away from an explicit binary override or replace shared libraries. No native SDK/package installer was run.

## Automated verification and limits

- Linux/Python 3.11: **270 tests collected, 269 passed, 1 skipped** (optional installed EvalScope HTTP integration).
- New dependency tests: **12 passed** on Windows/Python 3.13, covering PATH/fallback/strict overrides, missing native libraries and Python packages, flag requirements, zero-exit loader errors and preflight exit/report behaviour.
- An additional full Windows suite attempt encountered failures in existing E2E foreground-service fixtures and Bash entrypoint encoding/path handling. It is not reported as passing; repository CI targets Linux, and the full Linux run above passed.
- Native driver/version/help and dependency readiness were verified only. No ResNet50/BERT inference throughput, numerical accuracy, GPU mapping or arbitrary Torch-image compatibility is claimed by this run. Models remain user-supplied.
- Only task-owned containers were stopped and source, wheel and check/test reports were retained in the task directory.
