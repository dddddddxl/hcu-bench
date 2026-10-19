"""Prepare the pinned HTTP client in an isolated cache, never in the serving Python."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from hcu_bench.agent import acquire_locks
from testpacks.native_common import NativeProcess, cancellation, sha256, write_json, CleanupError, cleanup_failure

VERSION = "1.13.0"
SOURCE_SHA256 = "e0a57e9a0b4ed5716e5c7a7cad4747d5cf1831de2a9e129d9b4894a8fb0d217d"
VERIFY = ("import importlib.metadata as m; assert m.version('evalscope') == '" + VERSION +
          "'; from evalscope.perf.main import run_perf_benchmark")


def execute(command, output, stopped):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    log = output / f"client-prepare-{len(list(output.glob('client-prepare-*.log'))):02d}.log"
    process = NativeProcess(command, str(output), env, log)
    try:
        code = process.wait(stopped)
        if stopped.is_set():
            raise InterruptedError("Client preparation cancelled")
        if code != 0:
            raise RuntimeError("Performance client preparation failed; see " + log.name)
    finally:
        process.close()


def prepare(session, output, stopped):
    requested = {task["params"].get("python", "auto") for task in session["tasks"] if task["suite"] == "e2e_perf"}
    interpreters = {}
    if "auto" in requested:
        source = ROOT / "vendor/evalscope" / f"evalscope-{VERSION}.tar.gz"
        if sha256(source) != SOURCE_SHA256:
            raise ValueError("Bundled EvalScope source checksum mismatch")
        build = json.loads(os.environ.get("HCU_BENCH_BUILD_CONFIG", "{}"))
        base = Path(build.get("cache_dir") or Path.home() / ".cache/hcu-bench").resolve()
        cache = base / f"evalscope-{VERSION}-{sys.platform}-py{sys.version_info.major}.{sys.version_info.minor}"
        cache.mkdir(parents=True, exist_ok=True)
        locks = acquire_locks(str(cache), ["client-install"])
        try:
            python = cache / ("venv/Scripts/python.exe" if os.name == "nt" else "venv/bin/python")
            marker = cache / "ready.json"
            if not marker.is_file():
                execute([sys.executable, "-m", "venv", str(cache / "venv")], output, stopped)
                print("Preparing isolated EvalScope client; Python dependencies require a package index or configured wheelhouse.", flush=True)
                execute([str(python), "-m", "pip", "install", "--disable-pip-version-check",
                         f"evalscope[perf] @ {source.as_uri()}"], output, stopped)
                execute([str(python), "-c", VERIFY], output, stopped)
                write_json(marker, {"source_sha256": SOURCE_SHA256, "version": VERSION})
            else:
                if json.loads(marker.read_text())["source_sha256"] != SOURCE_SHA256:
                    raise ValueError("Client cache source mismatch; choose a fresh build.cache_dir")
                execute([str(python), "-c", VERIFY], output, stopped)
            interpreters["auto"] = str(python)
        finally:
            for lock in locks:
                lock.close()
    for python in requested - {"auto"}:
        execute([python, "-c", VERIFY], output, stopped)
        interpreters[python] = python
    for python in sorted(set(interpreters.values())):
        execute([python, "-m", "pip", "freeze"], output, stopped)
    write_json(output / "e2e-client.json", {"version": VERSION, "interpreters": interpreters})


def main():
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--session-json", required=True)
    cli.add_argument("--output-dir", required=True)
    args = cli.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        prepare(json.loads(args.session_json), output, cancellation())
        return 0
    except CleanupError as exc:
        return cleanup_failure(output, exc)
    except InterruptedError:
        return 130
    except Exception as exc:
        print("[client prepare error] " + str(exc), flush=True)
        write_json(output / "prepare-error.json", {"error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
