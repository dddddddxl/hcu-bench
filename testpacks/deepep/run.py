"""Wrap the supplied low-latency test without editing its code or environment."""
import argparse
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from testpacks.native_common import CleanupError, NativeProcess, cancellation, cleanup_failure, emit, sha256, versions, write_json

NUMBER = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
RANK = r"\[rank\s+(\d+)\]\s*"
COMBINED = re.compile(RANK + r"Dispatch \+ combine bandwidth:\s*" + NUMBER + r"\s*GB/s,\s*avg_t=" + NUMBER + r"\s*us,\s*min_t=" + NUMBER + r"\s*us,\s*max_t=" + NUMBER + r"\s*us")
SEPARATE = re.compile(RANK + r"Dispatch bandwidth( \(total\))?:\s*" + NUMBER + r"\s*GB/s,\s*avg_t=" + NUMBER + r"\s*us\s*\|\s*Combine bandwidth(?: \(total\))?:\s*" + NUMBER + r"\s*GB/s,\s*avg_t=" + NUMBER + r"\s*us")
SPLIT = re.compile(RANK + r"Dispatch send/recv time:\s*" + NUMBER + r"\s*\+\s*" + NUMBER + r"\s*us\s*\|\s*Combine send/recv time:\s*" + NUMBER + r"\s*\+\s*" + NUMBER + r"\s*us")


def metric(name, value, unit, statistic):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Native measurement must be finite and positive")
    return {"name": name, "value": value, "unit": unit, "statistic": statistic}


def parse_line(line):
    samples = []
    for match in COMBINED.finditer(line):
        rank, bandwidth, avg, minimum, maximum = match.groups()
        if not float(minimum) <= float(avg) <= float(maximum):
            raise ValueError("Native min/avg/max latency ordering is invalid")
        samples.append({"native_rank": int(rank), "metrics": [
            metric("dispatch_combine_effective_bandwidth", bandwidth, "GB/s", "reported"),
            metric("dispatch_combine_latency", avg, "us", "mean"),
            metric("dispatch_combine_latency", minimum, "us", "min"),
            metric("dispatch_combine_latency", maximum, "us", "max")]})
    for match in SEPARATE.finditer(line):
        rank, total, dispatch_bw, dispatch_t, combine_bw, combine_t = match.groups()
        suffix = "_total" if total else ""
        samples.append({"native_rank": int(rank), "metrics": [
            metric("dispatch_bandwidth" + suffix, dispatch_bw, "GB/s", "reported"),
            metric("dispatch_latency" + suffix, dispatch_t, "us", "mean"),
            metric("combine_bandwidth" + suffix, combine_bw, "GB/s", "reported"),
            metric("combine_latency" + suffix, combine_t, "us", "mean")]})
    for match in SPLIT.finditer(line):
        rank, ds, dr, cs, cr = match.groups()
        samples.append({"native_rank": int(rank), "metrics": [
            metric("dispatch_send_latency", ds, "us", "reported"), metric("dispatch_recv_latency", dr, "us", "reported"),
            metric("combine_send_latency", cs, "us", "reported"), metric("combine_recv_latency", cr, "us", "reported")]})
    return samples


def command_for(args, cycle):
    port = args.master_port + cycle
    if port > 65535:
        raise ValueError("Fresh rendezvous port range exhausted")
    return [sys.executable, "-u", "-m", "torch.distributed.run", "--nproc-per-node=1",
            f"--nnodes={args.nnodes}", f"--node-rank={args.node_rank}", f"--master-addr={args.master_addr}",
            f"--master-port={port}", str(Path(args.test_script).resolve()), "--pressure-test",
            "--num-processes", str(args.num_processes), "--num-tokens", str(args.num_tokens),
            "--hidden", str(args.hidden), "--num-topk", str(args.num_topk), "--num-experts", str(args.num_experts)]


def parser():
    cli = argparse.ArgumentParser(description="Native DeepEP low-latency pressure wrapper")
    cli.add_argument("--test-script", required=True)
    cli.add_argument("--output-dir", required=True)
    cli.add_argument("--node-rank", type=int, required=True)
    cli.add_argument("--nnodes", type=int, required=True)
    cli.add_argument("--master-addr", required=True)
    cli.add_argument("--master-port", type=int, default=12345)
    cli.add_argument("--num-processes", type=int, required=True)
    cli.add_argument("--num-tokens", type=int, default=128)
    cli.add_argument("--hidden", type=int, default=7168)
    cli.add_argument("--num-topk", type=int, default=8)
    cli.add_argument("--num-experts", type=int, default=256)
    cli.add_argument("--check-only", action="store_true")
    return cli


def preflight(args, env=None):
    env = os.environ if env is None else env
    if args.nnodes < 1 or not 0 <= args.node_rank < args.nnodes:
        raise ValueError("Invalid node count/rank")
    for name in ("num_processes", "num_tokens", "hidden", "num_topk", "num_experts"):
        if getattr(args, name) <= 0:
            raise ValueError(name + " must be positive")
    if args.num_experts % (args.nnodes * args.num_processes) or args.num_topk > args.num_experts:
        raise ValueError("Experts must divide the actual GPU world size, and topk must not exceed experts")
    if not 1 <= args.master_port <= 65535:
        raise ValueError("Invalid master port")
    script = Path(args.test_script)
    if not script.is_file() or not script.with_name("utils.py").is_file():
        raise ValueError("Supply native test_low_latency.py and its sibling utils.py")
    for name in ("torch", "deep_ep", "psutil"):
        if importlib.util.find_spec(name) is None:
            raise RuntimeError("Target dependency is missing: " + name)
    # MNVL/XDP settings are transport-specific, not mandatory on IPC/IB nodes.
    required = {"ROCSHMEM_HEAP_SIZE", "HIP_BUFFER_EXTRA_SIZE"}
    if any(not env.get(name) for name in required) or env.get("DEEP_EP_NORMAL_MNVL"):
        raise ValueError("Configure low-latency ROCSHMEM/HIP env and unset DEEP_EP_NORMAL_MNVL explicitly")
    return {"native_script": str(script.resolve()), "native_script_sha256": sha256(script),
            "utils_sha256": sha256(script.with_name("utils.py")), "packages": versions(["torch", "deep-ep", "psutil"]),
            "gpu_world_size": args.nnodes * args.num_processes,
            "communication_env": {key: value for key, value in env.items() if key.startswith(("ROCSHMEM_", "NCCL_", "GLOO_", "HIP_", "DEEP_EP_"))},
            "unset_env": ["DEEP_EP_NORMAL_MNVL"]}


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    native_env = dict(os.environ)
    native_env.pop("DEEP_EP_NORMAL_MNVL", None)
    evidence = preflight(args, native_env)
    write_json(output / "environment.json", evidence)
    if args.check_only:
        print("DeepEP preflight OK; no GPU workload started", flush=True)
        return 0
    stopped = cancellation()
    cycle = 0
    sample_id = 0
    summary = {"completed_cycles": 0, "valid_samples": 0, "invalid_samples": 0, "status": "running"}
    write_json(output / "summary.json", summary)
    with (output / "samples.jsonl").open("w", encoding="utf-8") as samples:
        while not stopped.is_set():
            cycle_start_samples = summary["valid_samples"]
            command = command_for(args, cycle)
            write_json(output / "command.json", {"cycle": cycle, "argv": command, "environment": evidence})

            def record(line):
                nonlocal sample_id
                try:
                    parsed = parse_line(line)
                except ValueError as exc:
                    parsed = [{"measurement_status": "invalid", "metrics": [], "reason": str(exc)}]
                    summary["invalid_samples"] += 1
                for sample in parsed:
                    rank = sample.get("native_rank")
                    if rank is not None and not (args.node_rank * args.num_processes <= rank < (args.node_rank + 1) * args.num_processes):
                        raise ValueError("Native rank falls outside this Bench node's GPU allocation")
                    payload = {"measurement_status": "measured", "correctness": "not_checked", "sample_id": sample_id,
                               "cycle": cycle, "timing_method": "native_test_report", **sample}
                    samples.write(json.dumps(payload, allow_nan=False) + "\n")
                    samples.flush()
                    emit(payload)
                    sample_id += 1
                    summary["valid_samples"] += payload["measurement_status"] == "measured"
                if parsed:
                    write_json(output / "summary.json", summary)

            process = NativeProcess(command, str(Path(args.test_script).resolve().parent), native_env,
                                    output / f"native_cycle_{cycle:04d}.log", record)
            try:
                code = process.wait(stopped)
            finally:
                process.close()
            if stopped.is_set():
                summary["status"] = "interrupted"
                write_json(output / "summary.json", summary)
                return 130
            if code != 0:
                summary.update(status="failed", native_returncode=code)
                write_json(output / "summary.json", summary)
                return 1
            if summary["valid_samples"] == cycle_start_samples:
                raise RuntimeError("Native pressure process exited without parseable performance data")
            summary["completed_cycles"] += 1
            cycle += 1
            write_json(output / "summary.json", summary)
            print(f"Native pressure cycle complete; starting cycle {cycle} on a fresh rendezvous port", flush=True)
    return 130


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return run(args)
    except CleanupError as exc:
        return cleanup_failure(args.output_dir, exc)
    except Exception as exc:
        print("[deepep error] " + str(exc), flush=True)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "error.json", {"error": str(exc), "status": "failed"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
