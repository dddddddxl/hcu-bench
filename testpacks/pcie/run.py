"""Finite, single-node rocm-bandwidth-test pair sweeps; no library changes."""
import argparse
import csv
from datetime import datetime
import itertools
import math
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_common import NativeProcess, CleanupError, cancellation, cleanup_failure, emit, sha256, write_json, prepared_binary

KINDS = ("cpu_to_gpu", "gpu_to_cpu", "gpu_to_gpu", "cpu_gpu_bidir", "gpu_gpu_bidir")
FIELDS = ("kind", "source", "target", "size_bytes", "avg_gbps", "peak_gbps")


def ids(text):
    values = text.split(",")
    if not values or any(not re.fullmatch(r"[0-9]+", value) for value in values):
        raise ValueError("Device IDs must be comma-separated nonnegative integers")
    result = [int(value) for value in values]
    if len(set(result)) != len(result):
        raise ValueError("Duplicate device IDs")
    return result


def validate(args, env):
    if args.nnodes != 1 or args.node_rank != 0:
        raise ValueError("PCIe currently supports one node per case")
    cpus, gpus, reserved = ids(args.cpu_devices), ids(args.gpu_devices), ids(args.gpu_ids)
    if set(cpus) & set(gpus) or len(gpus) < 2 or len(gpus) != len(reserved):
        raise ValueError("Choose distinct CPU/HCU Device IDs and at least two HCUs matching nodes.hcus count")
    if not 1 <= args.size_mib <= 4096 or not 1 <= args.iters <= 10000:
        raise ValueError("Invalid buffer size or iteration count")
    if env.get("HIP_VISIBLE_DEVICES") not in (None, "", args.gpu_ids):
        raise ValueError("HIP_VISIBLE_DEVICES conflicts with nodes.hcus")
    if env.get("ROCR_VISIBLE_DEVICES") or env.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Remove ROCR/CUDA masks and recheck native Device IDs")
    # These can change the parser contract, device enumeration or timing method.
    for key in env:
        if key.startswith(("ROCM_BW_", "ROCR_BW_", "ROCM_SKIP_")) and key != "ROCM_BW_ITER_CNT":
            if env[key]:
                raise ValueError(f"Unsupported native override: {key}")
    if env.get("ROCM_BW_ITER_CNT") not in (None, "", str(args.iters)):
        raise ValueError("ROCM_BW_ITER_CNT conflicts with iters")
    return cpus, gpus


def parse_devices(text):
    devices = {}
    for match in re.finditer(r"Device Index:\s*(\d+)\s*(.*?)(?=Device Index:|\Z)", text, re.S):
        index, body = int(match[1]), match[2]
        kind = re.search(r"Device Type:\s*(CPU|GPU)\b", body)
        if index in devices or not kind:
            raise ValueError("Ambiguous native device listing")
        pools = len(re.findall(r"Allocatable Memory Size \(KB\):", body))
        if pools != 1:
            raise ValueError("Ambiguous native memory pools: require exactly one pool per Device Index; use the bundled tool")
        devices[index] = {"type": kind[1], "pool_count": pools}
        for field, pattern in (("name", r"Device Name:\s*([^\r\n]+)"),
                               ("bdf", r"Device\s+BDF:\s*([^\r\n]+)"),
                               ("uuid", r"Device UUID:\s*([^\r\n]+)")):
            value = re.search(pattern, body)
            if value:
                devices[index][field] = value[1].strip()
    if not devices:
        raise ValueError("No devices found in native -e output")
    return devices


def pairs(cpus, gpus):
    for cpu, gpu in itertools.product(cpus, gpus):
        yield "cpu_to_gpu", cpu, gpu
        yield "gpu_to_cpu", gpu, cpu
        yield "cpu_gpu_bidir", cpu, gpu
    for src, dst in itertools.permutations(gpus, 2):
        yield "gpu_to_gpu", src, dst
    for src, dst in itertools.combinations(gpus, 2):
        yield "gpu_gpu_bidir", src, dst


def command(binary, kind, src, dst, size_mib):
    direction = ["-b", f"{src},{dst}"] if kind.endswith("bidir") else ["-s", str(src), "-d", str(dst)]
    return [binary, *direction, "-m", str(size_mib)]


def parse_result(text, kind, src, dst, size_mib):
    if re.search(r"\b(error|failed|failure|fatal|nan|inf)\b|HSA_STATUS_ERROR|segmentation fault|assertion", text, re.I):
        raise ValueError("Native output contains an error; inspect raw log")
    direction = "Bidirectional" if kind.endswith("bidir") else "Unidirectional"
    if re.findall(r"(Unidirectional|Bidirectional) Benchmark Result", text) != [direction]:
        raise ValueError("Missing or unexpected native benchmark mode")
    types = {"cpu_to_gpu": ("Cpu", "Gpu"), "gpu_to_cpu": ("Gpu", "Cpu"),
             "cpu_gpu_bidir": ("Cpu", "Gpu"), "gpu_to_gpu": ("Gpu", "Gpu"), "gpu_gpu_bidir": ("Gpu", "Gpu")}
    for role, device, expected_type in zip(("Src", "Dst"), (src, dst), types[kind]):
        found = re.findall(rf"{role} Device Id:\s*(\d+)\s+{role} Device Type:\s*(\w+)", text)
        if found != [(str(device), expected_type)]:
            raise ValueError("Native result device or direction mismatch")
    header = r"Data Size\s+Avg Time\(us\)\s+Avg BW\(GB/s\)\s+Min Time\(us\)\s+Peak BW\(GB/s\)"
    headers = list(re.finditer(header, text))
    if len(headers) != 1:
        raise ValueError("Unsupported native table header; expected Avg and Peak GB/s")
    rows = re.findall(r"^\s*(\d+)\s+(B|KB|MB|GB)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$", text[headers[0].end():], re.M)
    if len(rows) != 1:
        raise ValueError("Expected exactly one buffer-size row")
    size, unit, avg_time, avg, min_time, peak = rows[0]
    if int(size) * {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}[unit] != size_mib * 1024**2:
        raise ValueError("Native buffer size differs from requested size")
    avg_time, avg, min_time, peak = map(float, (avg_time, avg, min_time, peak))
    if not all(math.isfinite(v) and v > 0 for v in (avg_time, avg, min_time, peak)) or avg > peak or min_time > avg_time:
        raise ValueError("Invalid bandwidth or timing values")
    # Both directions already contribute to the native bidirectional result.
    return dict(kind=kind, source=src, target=dst, size_bytes=size_mib * 1024**2, avg_gbps=avg, peak_gbps=peak)


def export(out, rows):
    for name in ("measurements", *KINDS):
        with (out / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(row for row in rows if name == "measurements" or row["kind"] == name)


def publish(row):
    prefix = f"{row['kind']}_dev{row['source']}_dev{row['target']}_{row['size_bytes']}B"
    emit({"measurement_status": "measured", "correctness": "not_checked", **row,
          "metrics": [{"name": prefix + "_" + stat, "value": row[field], "unit": "GB/s", "statistic": statistic}
                      for stat, field, statistic in (("avg", "avg_gbps", "reported_mean"), ("peak", "peak_gbps", "reported_peak"))]})


def execute(argv, name, out, env, stopped, execution):
    if stopped.is_set():
        raise RuntimeError("Cancelled before native launch")
    entry = dict(argv=argv, log=f"{name}.log", started_at=datetime.now().astimezone().isoformat(), returncode=None)
    execution.append(entry)
    write_json(out / "execution.json", execution)
    process = NativeProcess(argv, out, env, out / entry["log"])
    try:
        process.wait(stopped)
    finally:
        process.close()
        entry.update(returncode=process.proc.returncode, finished_at=datetime.now().astimezone().isoformat(),
                     cancelled=stopped.is_set(), termination_actions=process.root_termination_actions)
        write_json(out / "execution.json", execution)
    if stopped.is_set() or process.root_termination_actions or process.proc.returncode != 0:
        raise RuntimeError(f"Native command {name} failed or was cancelled; exit={process.proc.returncode}")
    return (out / entry["log"]).read_text(encoding="utf-8")


def run(args, stopped):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, execution = [], []
    try:
        prepared_binary(args)
        env = dict(os.environ)
        cpus, gpus = validate(args, env)
        binary = Path(args.binary).resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError("Supply an existing executable rocm-bandwidth-test; no automatic installation")
        env["ROCM_BW_ITER_CNT"] = str(args.iters)
        write_json(out / "plan.json", {"params": vars(args), "binary": str(binary), "binary_sha256": sha256(binary),
                   "wrapper_sha256": sha256(__file__), "size_bytes": args.size_mib * 1024**2,
                   "variables": {key: value for key, value in env.items()
                                 if key.startswith(("ROCM_BW_", "ROCR_BW_", "ROCM_SKIP_"))
                                 or key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}})
        devices = parse_devices(execute([str(binary), "-e"], "devices", out, env, stopped, execution))
        for selected, kind in ((cpus, "CPU"), (gpus, "GPU")):
            if any(device not in devices or devices[device]["type"] != kind for device in selected):
                raise ValueError("Configured native device IDs do not match the enumerated CPU/HCU types")
        write_json(out / "devices.json", devices)
        execute([str(binary), "-t"], "topology", out, env, stopped, execution)
        for kind, src, dst in pairs(cpus, gpus):
            name = f"{kind}_{src}_{dst}"
            log = execute(command(str(binary), kind, src, dst, args.size_mib), name, out, env, stopped, execution)
            row = parse_result(log, kind, src, dst, args.size_mib)
            rows.append(row)
            export(out, rows)
            publish(row)
        if stopped.is_set():
            raise RuntimeError("Cancelled before sweep completion")
        write_json(out / "assessment.json", {"complete": True, "paths": len(rows), "correctness": "not_checked"})
        return 0
    except CleanupError as exc:
        return cleanup_failure(out, exc)
    except Exception as exc:
        write_json(out / "assessment.json", {"complete": False, "completed_paths": len(rows),
                   "correctness": "not_checked", "error": str(exc)})
        emit({"measurement_status": "invalid", "correctness": "not_checked", "metrics": [], "reason": str(exc)})
        print(f"[PCIe error] {exc}", flush=True)
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--cpu-devices", required=True)
    parser.add_argument("--hcu-devices", "--gpu-devices", dest="gpu_devices", required=True)
    parser.add_argument("--size-mib", type=int, default=512)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--hcu-ids", "--gpu-ids", dest="gpu_ids", required=True)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    return run(parser.parse_args(), cancellation())


if __name__ == "__main__":
    raise SystemExit(main())
