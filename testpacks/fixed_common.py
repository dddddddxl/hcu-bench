"""Ownership, receipts and device checks shared by fixed benchmark recipes."""
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
import threading

from native_common import NativeProcess, write_json


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("Expected positive finite native measurement")
    return number


def single_device(args, env):
    if args.nnodes != 1 or args.node_rank != 0 or not re.fullmatch(r"[0-9]+", args.hcu_ids):
        raise ValueError("Fixed recipes require one node and exactly one assigned HCU")
    if env.get("HIP_VISIBLE_DEVICES") not in (None, "", args.hcu_ids):
        raise ValueError("HIP_VISIBLE_DEVICES conflicts with nodes.hcus")
    for key in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        if env.get(key):
            raise ValueError(f"Remove secondary visibility mask {key}; use nodes.hcus")
    env["HIP_VISIBLE_DEVICES"] = args.hcu_ids
    # Python -O would remove the native stress assertions.
    if env.get("PYTHONOPTIMIZE") not in (None, "", "0"):
        raise ValueError("PYTHONOPTIMIZE would disable native repeatability assertions")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def environment_record(env):
    prefixes = ("HIP_", "HSA_", "ROCR_", "CUDA_", "GPU_", "FA_", "PA_", "FLASH_",
                "MIGRAPHX_", "ROCBLAS_", "MIOPEN_", "OMP_", "BENCH_")
    return {k: v for k, v in env.items() if k.startswith(prefixes) or k in
            ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONOPTIMIZE")}


def execute(argv, name, out, env, stopped, executions, timeout_s):
    """Only completed, unforced exit zero can be parsed into a measurement."""
    positive(timeout_s)
    if stopped.is_set():
        raise RuntimeError("Cancelled before native launch")
    entry = dict(argv=list(map(str, argv)), log=f"{name}.log", env=environment_record(env),
                 started_at=datetime.now(timezone.utc).isoformat(), returncode=None,
                 timeout_s=timeout_s, cleanup="pending")
    executions.append(entry)
    write_json(out / "execution.json", executions)
    expired = threading.Event()
    timer = threading.Timer(timeout_s, expired.set)
    timer.daemon = True

    class Stop:
        def is_set(self):
            return expired.is_set() or stopped.is_set()

        def wait(self, seconds):
            stopped.wait(seconds)
            return self.is_set()

    process = None
    timer.start()
    try:
        process = NativeProcess(entry["argv"], out, env, out / entry["log"], echo=False)
        try:
            process.wait(Stop())
        finally:
            process.close()
        entry["cleanup"] = "confirmed"
    finally:
        timer.cancel()
        entry.update(finished_at=datetime.now(timezone.utc).isoformat(),
                     timed_out=expired.is_set(), cancelled=stopped.is_set())
        if process is not None:
            entry.update(returncode=process.proc.returncode,
                         termination_actions=process.root_termination_actions)
        else:
            entry["cleanup"] = "not_started"
        write_json(out / "execution.json", executions)
    if entry["timed_out"] or entry["cancelled"] or entry["termination_actions"] or entry["returncode"] != 0:
        raise RuntimeError(f"{name}: incomplete native execution; inspect {entry['log']} and execution.json")
    return (out / entry["log"]).read_text(encoding="utf-8")


def claim_output(path):
    out = Path(path).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # A second invocation may not overwrite a partial/completed run or reuse its cache.
    with (out / "fixed-recipe.receipt").open("x", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()}\nstarted={datetime.now(timezone.utc).isoformat()}\n")
    return out
