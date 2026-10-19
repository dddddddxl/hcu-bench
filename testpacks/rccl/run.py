"""Run single-host RCCL through Bench, reusing the strict wheel-pilot parser."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "testpacks"))
sys.path.insert(0, str(ROOT / "experimental/wheel-pilot/rccl-plugin/src"))
from native_common import NativeProcess, CleanupError, cancellation, cleanup_failure, emit, sha256, write_json, prepared_binary
from hcubench_communication_rccl import RcclAllReduce


class LocalAllReduce(RcclAllReduce):
    def plan(self, params):
        plan = super().plan(params)
        # Never let an inherited MPI allocation launch tests on other hosts.
        plan["argv"][1:1] = ["--host", "localhost"]
        plan["execution_owner"] = "hcu-bench"
        plan["notes"] = ["Single-host launch; full native validation required"]
        return plan


def validate(params, gpu_ids, nnodes, node_rank, env):
    if nnodes != 1 or node_rank != 0:
        raise ValueError("RCCL currently supports one node only")
    for key, rule in RcclAllReduce.schema.items():
        value = params[key]
        if rule["type"] == "integer":
            if type(value) is not int or not rule["minimum"] <= value <= rule["maximum"]:
                raise ValueError(f"Invalid {key}: outside supported range")
        elif not isinstance(value, str) or ("enum" in rule and value not in rule["enum"]):
            raise ValueError(f"Invalid {key}")
    LocalAllReduce().validate(params)
    devices = gpu_ids.split(",")
    if len(devices) != params["ranks"] or len(set(devices)) != len(devices) or not all(d.isdigit() for d in devices):
        raise ValueError("ranks must equal the number of distinct configured HCU IDs")
    if env.get("HIP_VISIBLE_DEVICES") not in (None, "", gpu_ids):
        raise ValueError("HIP_VISIBLE_DEVICES conflicts with configured HCU IDs")
    if env.get("ROCR_VISIBLE_DEVICES") or env.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Remove secondary ROCR/CUDA visibility masks; use configured HCU IDs only")
    for key in ("NCCL_TESTS_SPLIT", "NCCL_TESTS_SPLIT_MASK", "NCCL_TESTS_DEVICE"):
        if env.get(key):
            raise ValueError(f"Unsupported environment override: {key}")


def capture(argv, env):
    result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=30, check=True)
    return result.stdout


def environment(params, env):
    binary = Path(params["binary"])
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("RCCL binary is missing or not executable; supply a prebuilt MPI-enabled all_reduce_perf")
    if not shutil.which("mpirun", path=env.get("PATH")):
        raise ValueError("Open MPI mpirun is not on PATH")
    libraries = capture(["ldd", str(binary)], env)
    rccl = None
    for line in libraries.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0].startswith("librccl.so") and fields[1] == "=>" and fields[2].startswith("/"):
            rccl = fields[2]
    if not rccl:
        raise ValueError("Cannot resolve the RCCL shared library from ldd")
    return dict(hostname=platform.node(), binary_sha256=sha256(binary),
                mpi_version=capture(["mpirun", "--version"], env),
                dtk_version=Path(env.get("ROCM_PATH", "/opt/dtk"), ".dtk_version").read_text(),
                rccl_library=rccl, rccl_library_sha256=sha256(rccl), dynamic_libraries=libraries,
                variables={k: v for k, v in env.items() if k.startswith(("NCCL_", "RCCL_", "HIP_", "ROCR_", "UCX_", "OMPI_"))
                           or k in ("LD_LIBRARY_PATH", "ROCM_PATH")})


def publish(assessment):
    valid = assessment["evidence_valid"] and assessment["correctness"]["passed"]
    if not valid:
        emit({"measurement_status": "invalid", "correctness": "not_checked", "metrics": [],
              "reason": "; ".join(assessment["issues"])})
        return 2
    for row in assessment["measurement"]["rows"]:
        # The common reporter groups by metric name, not sample extra fields.
        prefix = f"allreduce_{row['size_bytes']}B_{row['mode']}"
        emit({"measurement_status": "measured", "correctness": "passed", "size_bytes": row["size_bytes"],
              "mode": row["mode"], "timing_method": "native_iteration_aggregate", "metrics": [
                  {"name": prefix + "_" + name, "value": row[field], "unit": unit, "statistic": "reported_mean"}
                  for name, field, unit in (("latency", "time_us", "us"), ("algbw", "algbw_GBps", "GB/s"),
                                            ("busbw", "busbw_GBps", "GB/s"))]})
    return 0


def run(args, stopped):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    proc = None
    try:
        prepared_binary(args)
        params = {key: getattr(args, key) for key in RcclAllReduce.schema}
        env = dict(os.environ)
        validate(params, args.gpu_ids, args.nnodes, args.node_rank, env)
        env["HIP_VISIBLE_DEVICES"] = args.gpu_ids
        plugin = LocalAllReduce()
        plan = plugin.plan(params)
        identity = {"params": params, "argv": plan["argv"], "wrapper_sha256": sha256(__file__),
                    "parser_sha256": sha256(ROOT / "experimental/wheel-pilot/rccl-plugin/src/hcubench_communication_rccl/__init__.py")}
        config_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        meta = {"schema_version": 1, "config_hash": config_hash, "argv": plan["argv"],
                "environment": environment(params, env), "started_at": datetime.now().astimezone().isoformat(),
                "returncode": None, "stream_layout": "stdout.log contains merged stdout/stderr; stderr.log is intentionally empty"}
        write_json(out / "plan.json", {**plan, **identity, "config_hash": config_hash})
        write_json(out / "execution.json", meta)
        (out / "stderr.log").write_text("", encoding="utf-8")
        if stopped.is_set():
            raise RuntimeError("Cancelled before RCCL launch")
        proc = NativeProcess(plan["argv"], out, env, out / "stdout.log")
        try:
            proc.wait(stopped)
        finally:
            proc.close()
        meta.update(returncode=proc.proc.returncode, finished_at=datetime.now().astimezone().isoformat(),
                    cancelled=stopped.is_set(), root_termination_actions=proc.root_termination_actions,
                    sha256={name: sha256(out / name) for name in ("stdout.log", "stderr.log")})
        write_json(out / "execution.json", meta)
        assessment = plugin.assess(out, params, config_hash)
        if stopped.is_set() or proc.root_termination_actions:
            assessment["evidence_valid"] = False
            assessment["correctness"]["passed"] = False
            assessment["issues"].append("cancelled_or_terminated")
        write_json(out / "assessment.json", assessment)
        plugin.export_csv(assessment["measurement"], out / "measurements.csv")
        return publish(assessment)
    except CleanupError as exc:
        return cleanup_failure(out, exc)
    except Exception as exc:
        write_json(out / "error.json", {"error": str(exc)})
        print(f"[RCCL error] {exc}", flush=True)
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key, rule in RcclAllReduce.schema.items():
        parser.add_argument("--" + key.replace("_", "-"), type=int if rule["type"] == "integer" else str,
                            default=rule["default"] if key != "binary" else None, required=key == "binary")
    parser.add_argument("--hcu-ids", "--gpu-ids", dest="gpu_ids", required=True)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    return run(parser.parse_args(), cancellation())


if __name__ == "__main__":
    raise SystemExit(main())
