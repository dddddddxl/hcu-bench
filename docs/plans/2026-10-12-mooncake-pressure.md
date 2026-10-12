# Mooncake Pressure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a real RDMA/VRAM transfer_engine_bench test pack, validate it on the Zhengzhou cluster, and submit a PR.

**Architecture:** Reuse the command adapter, concurrent node launch and owned-process cleanup. A target-side wrapper coordinates fresh endpoints and completion through an explicitly supplied shared directory, namespaced by run/case; no controller networking API or password handling is added. Initiators report native completion measurements; targets report coordination only, never fabricated bandwidth.

**Tech Stack:** Python 3.10+, existing psutil native helpers, YAML manifests, externally installed Mooncake 0.3.12+ and RDMA/DCU stack.

## Design decisions

- Support pair, fan-in, disjoint pairs and ring. Six cross-check pairs can be configured as six two-node tests; no implicit node discovery.
- Require explicit per-node IPs, selected NICs, GPU visibility and a shared coordination directory. Ports come from each newly launched target's log.
- Use native initiator duration, startup and completion deadlines plus an outer framework timeout. Do not set framework duration_s, which could terminate before final metrics/cleanup.
- Require a positive finite final throughput, positive batch count and a completed requested measurement window. Preserve reported units; do not reinterpret GiB/s as GB/s.
- Optional min_bandwidth_gbps is explicit and applies to each initiator, with a separate performance_judgement; correctness remains not_checked. Historical 130 GB/s from a different cluster is not a universal default.
- Targets always stop after their incoming initiators complete. Cancellation/error terminates only owned native processes; unconfirmed cleanup uses reserved exit 125.
- Record package/binary/wrapper hashes, flags, visibility, NIC state and raw logs. Never use name-wide pkill, install into another user's container, or change NIC/driver settings.

## Task 1: Native command, topology and parser contracts

Create `testpacks/mooncake/run.py` and `tests/test_mooncake.py`. Test dynamic endpoint parsing, pair/fan-in/pairs/ring routing, throughput units, malformed/nonfinite/empty results, and version/argument validation. Run `python -m unittest discover -s tests -p test_mooncake.py -v` before and after implementation.

## Task 2: Coordinated lifecycle

Implement atomic run/case-scoped shared records and exclusive per-rank joins. Start local targets, wait for fresh endpoint records, run local initiators, validate final completion, publish completion only after cleanup, and wait for incoming initiators before stopping targets. Add CPU native-process integration tests for success, missing endpoints, early native failures, threshold failures, peer configuration mismatch and target cleanup.

## Task 3: Manifest and examples

Create `testpacks/mooncake/manifest.yaml`, `configs/mooncake.template.yaml` and `docs/mooncake.md`. Test planner rendering and topology-specific settings; explain shared-directory requirements, native versus framework duration, version restrictions and device mapping. Update README/native-tests documentation without calling placeholders hardware coverage.

## Task 4: Framework regression

Run the full existing and new unittest suite. Verify real tests reject empty/missing measurements and do not silently accept low bandwidth. Review diff and clean process/file boundaries.

## Task 5: Zhengzhou validation

Inspect authorised nodes read-only, select idle devices and active RDMA NICs, inspect available image versions, then deploy the same source to an isolated task directory. Run preflight, a short two-node native test and a longer pressure window where resources permit. Archive actual software versions, topology, parameters, reports, logs and cleanup evidence. If hardware prerequisites are unavailable, record the exact limitation and do not call CPU/TCP/self-node checks cross-node RDMA validation.

## Task 6: Reviewable delivery

Commit source/tests/docs and sanitised validation findings. Push a feature branch (fork if upstream is not writable), create and attach a PR. PR description must distinguish implementation, local tests and actual hardware outcomes.
