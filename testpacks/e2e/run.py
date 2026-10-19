"""One model service, followed by selected accuracy and throughput tasks."""
import argparse
import importlib.metadata
import itertools
import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from hcu_bench.e2e_config import task_params, validate_models
from hcu_bench.results import parse_sample
from testpacks.native_common import CleanupError, NativeProcess, cancellation, cleanup_failure, emit, sha256, versions, write_json
from testpacks.sglang_accuracy.run import ensure_free_port


def script_path(value):
    path = (ROOT / value).resolve()
    if not path.is_file():
        raise ValueError("Missing launch/hook script in the target environment: " + str(path))
    return path


def bash_command(path):
    bash = shutil.which("bash")
    if os.name == "nt":
        candidate = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
        if candidate.is_file():
            bash = str(candidate)
    if not bash:
        raise ValueError("Model launch scripts require bash")
    return [bash, path.as_posix()]


def check_alive(server, stopped):
    if stopped.is_set():
        raise InterruptedError("Model session cancelled")
    if server.proc.poll() is not None:
        raise RuntimeError("Owned model server exited; launch script must run the server in the foreground using exec")


def discover(server, model, url, stopped):
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + model.get("startup_timeout_s", 1800)
    while time.monotonic() < deadline:
        check_alive(server, stopped)
        try:
            timeout = min(2, max(0.1, deadline - time.monotonic()))
            with opener.open(url + model.get("health_path", "/health"), timeout=timeout) as response:
                if response.status != 200:
                    raise ValueError("Health endpoint did not return 200")
            with opener.open(url + "/v1/models", timeout=timeout) as response:
                data = json.load(response)["data"]
            requested = model.get("model_name")
            matches = [row for row in data if isinstance(row, dict) and isinstance(row.get("id"), str)
                       and (not requested or row["id"] == requested)]
            if len(matches) != 1:
                raise ValueError("Set models.<name>.model_name to one exact /v1/models ID")
            check_alive(server, stopped)
            return matches[0]
        except (URLError, OSError, TimeoutError):
            stopped.wait(0.25)
    raise TimeoutError("Model readiness deadline exceeded")


def resolve_tokenizer(model, served):
    candidates = [model["tokenizer_path"]] if model.get("tokenizer_path") else [served.get("root", ""), served["id"]]
    for candidate in candidates:
        if candidate and (ROOT / candidate).is_dir():
            return str((ROOT / candidate).resolve())
    raise ValueError("Cannot locate a local tokenizer from /v1/models; set models.<name>.tokenizer_path")


def run_child(command, cwd, env, log, server, stopped, timeout, on_line=None):
    if timeout <= 0:
        raise TimeoutError("Client/hook task timed out")
    check_alive(server, stopped)
    child = NativeProcess(command, str(cwd), env, log, on_line, echo=on_line is None)
    deadline = time.monotonic() + timeout
    try:
        while child.proc.poll() is None:
            check_alive(server, stopped)
            if time.monotonic() >= deadline:
                raise TimeoutError("Client/hook task timed out")
            if child.errors:
                raise RuntimeError("Client output failed: " + child.errors[0])
            stopped.wait(0.1)
        check_alive(server, stopped)
        return child.proc.returncode
    finally:
        child.close()


def workloads(task, offset):
    if task["suite"] == "e2e_acc":
        return [None]
    params = task["params"]
    points = []
    for (input_tokens, output_tokens), concurrency, repeat in itertools.product(
            params["shapes"], params["concurrency"], range(params["repeats"])):
        points.append(dict(input_tokens=input_tokens, output_tokens=output_tokens, concurrency=concurrency,
                           repeat=repeat, dataset_offset=offset))
        offset += params["requests"] + params["warmup_requests"] + 1
    return points


def run(session, output, stopped=None):
    stopped = stopped if stopped is not None else cancellation()
    output.mkdir(parents=True, exist_ok=True)
    model = session["model"]
    validate_models({session["model_id"]: model})
    launch = script_path(model["start_script"])
    hook = script_path(model["before_perf_script"]) if model.get("before_perf_script") else None
    for task in session["tasks"]:
        task["params"] = task_params(task["suite"], task["params"])
    prepared = json.loads((output / "e2e-client.json").read_text()) if (output / "e2e-client.json").is_file() else {"interpreters": {}}
    for task in session["tasks"]:
        if task["suite"] == "e2e_perf" and task["params"]["python"] not in prepared["interpreters"]:
            raise ValueError("Performance client is not prepared; run through bench.sh")
    host, port = "127.0.0.1", model.get("port", 30000)
    env = {**os.environ, **model.get("env", {}), "BENCH_HOST": host, "BENCH_PORT": str(port),
           "BENCH_MODEL_ID": session["model_id"], "BENCH_OUTPUT_DIR": str(output), "BENCH_ROOT": str(ROOT),
           "PYTHONIOENCODING": "utf-8"}
    url = f"http://{host}:{port}"
    ensure_free_port(SimpleNamespace(host=host, port=port))
    evidence = dict(model_id=session["model_id"], start_script=str(launch), script_sha256=sha256(launch),
                    url=url, hcus=env.get("HIP_VISIBLE_DEVICES", ""), packages=versions(["sglang", "vllm", "torch"]),
                    before_perf_sha256=sha256(hook) if hook else None)
    write_json(output / "session.json", session)
    write_json(output / "environment.json", evidence)
    server = NativeProcess(bash_command(launch), str(launch.parent), env, output / "server.log")
    statuses, failed, offset = [], False, 0
    try:
        served = discover(server, model, url, stopped)
        evidence["served_model"] = served
        write_json(output / "environment.json", evidence)
        for task in session["tasks"]:
            task_dir = output / task["test_id"]
            task_dir.mkdir()
            deadline = time.monotonic() + task["timeout_s"]
            try:
                check_alive(server, stopped)
                tokenizer = resolve_tokenizer(model, served) if task["suite"] == "e2e_perf" else ""
                if hook and task["suite"] == "e2e_perf":
                    code = run_child(bash_command(hook), hook.parent, env, task_dir / "before-perf.log",
                                     server, stopped, max(0, deadline - time.monotonic()))
                    if code:
                        raise RuntimeError("before_perf_script failed")
                task_failed = False
                points = workloads(task, offset)
                if points[0] is not None:
                    offset = points[-1]["dataset_offset"] + task["params"]["requests"] + task["params"]["warmup_requests"] + 1
                for index, point in enumerate(points):
                    point_dir = task_dir if point is None else task_dir / f"point-{index:04d}"
                    point_dir.mkdir(exist_ok=True)
                    config = dict(suite=task["suite"], params=task["params"], workload=point, url=url,
                                  served_model=served["id"], tokenizer_path=tokenizer)
                    write_json(point_dir / "client-config.json", config)
                    samples = []

                    def on_line(line):
                        sample = parse_sample(line)
                        if sample is None:
                            print(line, flush=True)
                            return
                        sample.update(e2e_task=task["test_id"], e2e_suite=task["suite"], model_id=session["model_id"],
                                      task_params=task["params"], workload=point, serving=evidence)
                        samples.append(sample)
                        emit(sample)

                    python = sys.executable if point is None else prepared["interpreters"][task["params"]["python"]]
                    command = [python, "-u", str(ROOT / "testpacks/e2e/client.py"), "--config",
                               str(point_dir / "client-config.json"), "--output-dir", str(point_dir)]
                    code = run_child(command, point_dir, env, point_dir / "client.log", server, stopped,
                                     max(0, deadline - time.monotonic()), on_line)
                    success = code == 0 and len(samples) == 1 and samples[0]["measurement_status"] == "measured" and samples[0]["correctness"] != "failed"
                    statuses.append(dict(test_id=task["test_id"], suite=task["suite"], workload=point,
                                         status="completed" if success else "failed", returncode=code))
                    task_failed |= not success
                    if not samples:
                        emit(dict(measurement_status="invalid", correctness="not_checked", metrics=[],
                                  reason="Client exited without a result", e2e_task=task["test_id"], e2e_suite=task["suite"],
                                  model_id=session["model_id"], task_params=task["params"], workload=point))
                    write_json(output / "tasks.json", statuses)
                    if not success and session["failure_policy"] == "stop":
                        break
                failed |= task_failed
            except (CleanupError, InterruptedError):
                raise
            except Exception as exc:
                failed = True
                statuses.append(dict(test_id=task["test_id"], suite=task["suite"], status="failed", error=str(exc)))
                emit(dict(measurement_status="invalid", correctness="not_checked", metrics=[], reason=str(exc),
                          e2e_task=task["test_id"], e2e_suite=task["suite"], model_id=session["model_id"], task_params=task["params"]))
                write_json(output / "tasks.json", statuses)
            if failed and session["failure_policy"] == "stop":
                break
            check_alive(server, stopped)
        return 1 if failed else 0
    finally:
        server.close()
        # A leftover listener must not be mistaken for the next model's service.
        try:
            listener = socket.create_connection((host, port), timeout=0.5)
        except (ConnectionRefusedError, OSError):
            pass
        else:
            listener.close()
            raise CleanupError("Serving port is still occupied after owned-process cleanup")


def main():
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--session-json", required=True)
    cli.add_argument("--output-dir", required=True)
    args = cli.parse_args()
    output = Path(args.output_dir)
    try:
        return run(json.loads(args.session_json), output)
    except CleanupError as exc:
        return cleanup_failure(output, exc)
    except InterruptedError:
        return 130
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "error.json", {"error": str(exc)})
        print("[e2e error] " + str(exc), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
