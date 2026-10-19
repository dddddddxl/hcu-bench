"""Serial single-HCU bandwidth presets from dcu-hipbenchmark main-staging-1.0."""
import argparse
import csv
from datetime import datetime
import math
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_common import NativeProcess, CleanupError, cancellation, cleanup_failure, emit, sha256, write_json, prepared_binary

# (native case, wave count, LDS argument in KiB); preserve the two upstream tables.
PRESETS = {
    "zd1": (
        ("l1_read", 4, 0), ("l2_read", 4, 0), ("l2_write", 4, 0),
        ("lds_read", 4, 0), ("lds_write", 4, 0),
        ("memory_readx4_x1", 4, 0), ("memory_write", 4, 0), ("memory_copy", 4, 0),
        ("memory_to_ldsx1", 4, 4), ("lds_to_memory", 4, 4),
        ("matrix_read", 4, 0), ("matrix_write", 4, 0), ("matrix_copy", 4, 0),
        ("mapping_matrix_write", 4, 0), ("tiling_read", 4, 0),
    ),
    "zd2": (
        ("memory_readx4_x1", 8, 0), ("memory_write", 8, 0), ("memory_copy", 8, 0),
        ("matrix_readx4_x4", 4, 0), ("matrix_writex4_x4", 4, 0), ("matrix_copy", 4, 0),
        ("memory_readx4_x4", 4, 0), ("mapping_matrix_writex4_x4", 4, 0), ("tiling_read_2w", 4, 0),
    ),
}
CYCLE_CASES = {"l1_read", "l2_read", "l2_write", "lds_read", "lds_write"}
FIELDS = ("gpu_id", "preset", "test", "category", "wave", "lds_arg_kib", "array_size",
          "model_mhz", "timing", "bandwidth_gbps")


def ids(text):
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", text):
        raise ValueError("HCU IDs must be comma-separated nonnegative integers")
    values = list(map(int, text.split(",")))
    if len(set(values)) != len(values):
        raise ValueError("Duplicate HCU IDs")
    return values


def validate(args, env):
    if args.nnodes != 1 or args.node_rank != 0:
        raise ValueError("Memory bandwidth supports one node per case")
    reserved = ids(args.gpu_ids)
    selected = ids(args.devices) if args.devices else reserved
    if not set(selected) <= set(reserved):
        raise ValueError("devices must be a subset of nodes.hcus")
    if args.presets not in ("both", *PRESETS):
        raise ValueError("presets must be both, zd1 or zd2")
    if not 2 <= args.model_mhz <= 10000 or not 1 <= args.array_size <= 2147483647:
        raise ValueError("Invalid MODEL MHz or array element count")
    if env.get("HIP_VISIBLE_DEVICES") not in (None, "", args.gpu_ids):
        raise ValueError("HIP_VISIBLE_DEVICES conflicts with nodes.hcus")
    for key in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        if env.get(key):
            raise ValueError(f"Remove conflicting HCU mask {key}; use nodes.hcus/devices")
    if env.get("MODEL") not in (None, "", str(args.model_mhz)):
        raise ValueError("MODEL conflicts with model_mhz; configure it in YAML")
    return selected, list(PRESETS) if args.presets == "both" else [args.presets]


def command(binary, test, wave, lds, array_size):
    argv = [str(binary), "bandwidth", test, "--arraysize", str(array_size)]
    if wave != 4:
        argv += ["-w", str(wave)]
    if lds:
        argv += ["--lds", str(lds)]
    return argv + ["--clean"]


def parse_bandwidth(text):
    # --clean prints one mean value only. Do not accept help/errors plus a number.
    value = text.strip()
    if not re.fullmatch(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", value):
        raise ValueError("Expected exactly one --clean bandwidth value; inspect native log")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Native bandwidth must be positive and finite")
    return result


def export(out, rows):
    with (out / "measurements.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def publish(row):
    name = (f"memory_gpu{row['gpu_id']}_{row['preset']}_{row['test']}_w{row['wave']}"
            f"_lds{row['lds_arg_kib']}_n{row['array_size']}_model{row['model_mhz']}")
    emit({"measurement_status": "measured", "correctness": "not_checked", **row,
          "metrics": [{"name": name, "value": row["bandwidth_gbps"], "unit": "GB/s",
                       "statistic": "reported_mean"}]})


def execute(argv, name, out, env, stopped, execution):
    if stopped.is_set():
        raise RuntimeError("Cancelled before native launch")
    entry = dict(argv=argv, log=f"{name}.log", started_at=datetime.now().astimezone().isoformat(),
                 returncode=None, env={key: env[key] for key in ("HIP_VISIBLE_DEVICES", "MODEL")})
    execution.append(entry)
    write_json(out / "execution.json", execution)
    process = NativeProcess(argv, out, env, out / entry["log"])
    try:
        process.wait(stopped)
    finally:
        try:
            process.close()
        finally:
            entry.update(returncode=process.proc.returncode, finished_at=datetime.now().astimezone().isoformat(),
                         cancelled=stopped.is_set(), termination_actions=process.root_termination_actions)
            write_json(out / "execution.json", execution)
    if stopped.is_set() or process.root_termination_actions or process.proc.returncode != 0:
        raise RuntimeError(f"Native command {name} failed or was cancelled; exit={process.proc.returncode}")
    return parse_bandwidth((out / entry["log"]).read_text(encoding="utf-8"))


def run(args, stopped):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, execution = [], []
    try:
        prepared_binary(args)
        env = dict(os.environ)
        devices, presets = validate(args, env)
        binary = Path(args.binary).resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError("hcu-stream executable missing; use binary=auto through bench.sh to build it")
        env["MODEL"] = str(args.model_mhz)
        expected = len(devices) * sum(len(PRESETS[p]) for p in presets)
        write_json(out / "plan.json", {"params": vars(args), "devices": devices, "presets": presets,
                   "expected_measurements": expected, "source_contract": "dcu-hipbenchmark main-staging-1.0",
                   "binary": str(binary), "binary_sha256": sha256(binary), "wrapper_sha256": sha256(__file__),
                   "model_is_clock_setting": False, "native_logical_device": 0,
                   "variables": {key: value for key, value in env.items()
                                 if key in ("MODEL", "UCLK", "HIP_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION",
                                            "HSA_DISABLE_L1_CACHE", "HSA_DISABLE_L2_CACHE")}})
        for gpu in devices:
            env["HIP_VISIBLE_DEVICES"] = str(gpu)
            for preset in presets:
                for test, wave, lds in PRESETS[preset]:
                    name = f"gpu{gpu}_{preset}_{test}"
                    value = execute(command(binary, test, wave, lds, args.array_size), name, out, env, stopped, execution)
                    category = "cache" if test.startswith(("l1_", "l2_")) else "lds" if test in CYCLE_CASES else "memory"
                    row = dict(gpu_id=gpu, preset=preset, test=test, category=category, wave=wave, lds_arg_kib=lds,
                               array_size=args.array_size, model_mhz=args.model_mhz,
                               timing="cycles_with_model" if test in CYCLE_CASES else "host_sync",
                               bandwidth_gbps=value)
                    rows.append(row)
                    export(out, rows)
                    publish(row)
        if stopped.is_set():
            raise RuntimeError("Cancelled before all presets completed")
        write_json(out / "assessment.json", {"complete": True, "measurements": len(rows),
                   "expected_measurements": expected, "correctness": "not_checked"})
        return 0
    except CleanupError as exc:
        return cleanup_failure(out, exc)
    except Exception as exc:
        write_json(out / "assessment.json", {"complete": False, "completed_measurements": len(rows),
                   "correctness": "not_checked", "error": str(exc)})
        emit({"measurement_status": "invalid", "correctness": "not_checked", "metrics": [], "reason": str(exc)})
        print(f"[Memory bandwidth error] {exc}", flush=True)
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--hcu-ids", "--gpu-ids", dest="gpu_ids", required=True)
    parser.add_argument("--devices", default="")
    parser.add_argument("--presets", default="both", choices=("both", *PRESETS))
    parser.add_argument("--model-mhz", type=int, default=1400)
    parser.add_argument("--array-size", type=int, default=104857600)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    return run(parser.parse_args(), cancellation())


if __name__ == "__main__":
    raise SystemExit(main())
