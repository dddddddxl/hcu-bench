#!/usr/bin/env python3
"""Example USER runner. Not included in either wheel; does not manage nodes/containers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from datetime import datetime, timezone


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(argv):
    r = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    if r.returncode:
        raise RuntimeError(f"Environment query failed: {argv}: {r.stderr}")
    return r.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--environment", required=True, help="JSON with host/image/topology evidence collected by the user")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    argv = plan["argv"]
    # Plans are executable user input. Review plan.json before running this example.
    if not isinstance(argv, list) or not argv or any(not isinstance(a, str) for a in argv):
        raise ValueError("Plan argv must be a list of strings")
    devices = os.environ.get("HIP_VISIBLE_DEVICES", "").split(",")
    ranks = plan["resolved"]["params"]["ranks"]
    if len(devices) != ranks or len(set(devices)) != ranks or not all(d.isdigit() for d in devices):
        raise ValueError("Set HIP_VISIBLE_DEVICES to one distinct physical card per rank")
    binary = plan["test_argv"][0]
    ldd = capture(["ldd", binary])
    libs = {}
    for line in ldd.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[1] == "=>" and fields[2].startswith("/"):
            libs[fields[0]] = fields[2]
    rccl = next((path for name, path in libs.items() if name.startswith("librccl.so")), None)
    if rccl is None:
        raise ValueError("Cannot identify resolved RCCL shared library")
    env = json.loads(Path(args.environment).read_text(encoding="utf-8"))
    env.update(hostname=platform.node(), binary_sha256=sha(binary), mpi_version=capture([argv[0], "--version"]),
               mpi_executable=shutil.which(argv[0]), dtk_version=Path("/opt/dtk/.dtk_version").read_text(),
               rccl_library=rccl, rccl_library_sha256=sha(rccl), dynamic_libraries=ldd,
               variables={k:v for k,v in os.environ.items() if k.startswith(("NCCL_", "RCCL_", "HIP_", "ROCR_", "UCX_", "OMPI_")) or k in ("LD_LIBRARY_PATH", "ROCM_PATH")})
    bundle = Path(args.bundle)
    bundle.mkdir(parents=True, exist_ok=False)
    meta = {"schema_version": 1, "config_hash": plan["config_hash"], "argv": argv, "environment": env,
            "started_at": datetime.now(timezone.utc).isoformat(), "returncode": None}
    (bundle / "execution.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with (bundle / "stdout.log").open("wb") as stdout, (bundle / "stderr.log").open("wb") as stderr:
        try:
            meta["returncode"] = subprocess.run(argv, stdout=stdout, stderr=stderr).returncode
        except OSError as exc:
            meta["launch_error"] = str(exc)
    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    meta["sha256"] = {n:sha(bundle/n) for n in ("stdout.log", "stderr.log")}
    (bundle / "execution.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"bundle":str(bundle), "returncode":meta["returncode"]}))
    return 0 if meta["returncode"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
