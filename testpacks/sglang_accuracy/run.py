"""Managed single-node serving plus the v0.5.12 HCU GSM8K evaluation engine."""
import argparse
import importlib.metadata
import importlib.util
import inspect
import ipaddress
import json
import math
import os
import re
import socket
import sys
import time
from pathlib import Path
from string import Template
from types import SimpleNamespace
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from testpacks.native_common import CleanupError, NativeProcess, cancellation, cleanup_failure, emit, sha256, versions, write_json

SGLANG_REFERENCE = "6c946e92fa4ec1755c8a2a8f8800bd2e6795213e"


def parser():
    cli = argparse.ArgumentParser(description="SGLang 0.5.12 GSM8K accuracy smoke test")
    cli.add_argument("--profile", required=True)
    cli.add_argument("--model-path", required=True)
    cli.add_argument("--data-path", required=True)
    cli.add_argument("--ep-config", required=True)
    cli.add_argument("--output-dir", required=True)
    cli.add_argument("--node-rank", type=int, default=0)
    cli.add_argument("--nnodes", type=int, default=1)
    cli.add_argument("--host", default="127.0.0.1")
    cli.add_argument("--port", type=int, default=30000)
    cli.add_argument("--startup-timeout", type=float, default=1800)
    cli.add_argument("--num-examples", type=int, default=100, help="0 means all examples after the few-shot prefix")
    cli.add_argument("--num-shots", type=int, default=5)
    cli.add_argument("--num-threads", type=int, default=128)
    cli.add_argument("--max-tokens", type=int, default=2048)
    cli.add_argument("--api", choices=("chat", "completion"), default="chat")
    cli.add_argument("--min-score", default="", help="Empty means report-only, not an accuracy pass")
    cli.add_argument("--check-only", action="store_true")
    return cli


def score_threshold(value):
    if value == "":
        return None
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError("min_score must be in [0, 1], or empty for report-only")
    return number


def validate_dataset(path, num_examples, num_shots):
    count = 0
    with Path(path).open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            row = json.loads(line)
            if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip() for key in ("question", "answer")):
                raise ValueError(f"Invalid GSM8K row {number}: question and answer strings required")
            count += 1
    available = count - num_shots
    if available < 1 or num_examples > available:
        raise ValueError("Dataset has insufficient rows after excluding the few-shot examples")
    return num_examples or available


def profile_for(args):
    profile = json.loads(Path(args.profile).read_text(encoding="utf-8-sig"))
    if not isinstance(profile, dict) or profile.get("version") != 1:
        raise ValueError("Serving profile must be a version 1 JSON object")
    context = {"model_path": args.model_path, "ep_config": args.ep_config, "host": args.host, "port": args.port}
    command = profile.get("server_argv")
    env = profile.get("env", {})
    unset = profile.get("unset_env", [])
    if not isinstance(command, list) or not command or not all(isinstance(item, str) and "\x00" not in item for item in command):
        raise ValueError("server_argv must be a nonempty argv list")
    if not isinstance(env, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in env.items()):
        raise ValueError("Serving env must contain string values")
    if not isinstance(unset, list) or not all(isinstance(key, str) for key in unset):
        raise ValueError("unset_env must be a list of environment variable names")
    rendered = [Template(argument).substitute(context) for argument in command]
    child_env = {**os.environ, **{key: Template(value).substitute(context) for key, value in env.items()}}
    for key in unset:
        child_env.pop(key, None)
    if any("YOUR_" in value for value in rendered + list(env.values())):
        raise ValueError("Serving profile still contains YOUR_* environment/path placeholders")
    expected_gpus = profile.get("visible_gpus")
    if expected_gpus is not None:
        if not isinstance(expected_gpus, int) or isinstance(expected_gpus, bool) or expected_gpus < 1:
            raise ValueError("Profile visible_gpus must be a positive integer")
        visible = child_env.get("HIP_VISIBLE_DEVICES", "").split(",")
        if any(not item.strip() for item in visible) or len(visible) != expected_gpus:
            raise ValueError("Explicit HIP_VISIBLE_DEVICES must match the recipe's GPU count")
    if profile.get("require_memlock_unlimited", False):
        import resource

        if resource.getrlimit(resource.RLIMIT_MEMLOCK)[0] != resource.RLIM_INFINITY:
            raise ValueError("Recipe requires unlimited memlock; provide it through the container/start hook")
    return profile, rendered, child_env


def preflight(args):
    if args.nnodes != 1 or args.node_rank != 0:
        raise ValueError("This simple accuracy pack manages one serving node; do not launch it on every rank")
    address = ipaddress.ip_address(args.host)
    if not (address.is_loopback or address.is_unspecified):
        raise ValueError("Managed smoke serving binds a loopback/unspecified local address only")
    if not 1 <= args.port <= 65535 or not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        raise ValueError("Invalid serving port/startup timeout")
    if args.num_examples < 0 or args.num_shots < 0 or args.num_threads < 1 or args.max_tokens < 1:
        raise ValueError("Invalid GSM8K sampling settings")
    score_threshold(args.min_score)
    if not Path(args.model_path).is_dir() or not Path(args.model_path, "config.json").is_file():
        raise ValueError("Supply an existing local model directory with config.json; no automatic model download")
    for path in (args.data_path, args.ep_config, args.profile):
        if not Path(path).is_file():
            raise ValueError("Required local file is missing: " + path)
    ep_config = json.loads(Path(args.ep_config).read_text(encoding="utf-8-sig"))
    if not isinstance(ep_config, dict) or not {"normal_dispatch", "normal_combine"} <= set(ep_config):
        raise ValueError("DeepEP config must contain normal_dispatch and normal_combine")
    count = validate_dataset(args.data_path, args.num_examples, args.num_shots)
    if importlib.util.find_spec("psutil") is None:
        raise RuntimeError("Target dependency psutil is missing")
    version = importlib.metadata.version("sglang")
    if not re.match(r"^0\.5\.12(?:$|[.+a-z-])", version):
        raise ValueError("This pack targets SGLang 0.5.12; installed version is " + version)
    from sglang.test.run_eval import run_eval_once
    from sglang.test.simple_eval_gsm8k import GSM8KEval

    for function in (run_eval_once, GSM8KEval):
        if not inspect.getsourcefile(function):
            raise RuntimeError("Cannot identify the installed evaluation source")
    return {"packages": versions(["sglang", "torch", "deep-ep", "psutil"]), "sglang_reference_commit": SGLANG_REFERENCE,
            "data_path": str(Path(args.data_path).resolve()), "dataset_sha256": sha256(args.data_path),
            "num_examples": count, "num_shots": args.num_shots, "api": args.api, "max_tokens": args.max_tokens,
            "model_path": args.model_path, "model_config_sha256": sha256(Path(args.model_path) / "config.json"),
            "profile_sha256": sha256(args.profile), "ep_config_sha256": sha256(args.ep_config),
            "evaluation_sources": {str(inspect.getsourcefile(function)): sha256(inspect.getsourcefile(function)) for function in (run_eval_once, GSM8KEval)}}


def local_url(args):
    address = ipaddress.ip_address(args.host)
    host = ("::1" if address.version == 6 else "127.0.0.1") if address.is_unspecified else args.host
    return f"http://{'[' + host + ']' if ':' in host else host}:{args.port}"


def ensure_free_port(args):
    family = socket.AF_INET6 if ":" in args.host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.bind((args.host, args.port))


def wait_ready(process, url, timeout, stopped):
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while not stopped.is_set() and time.monotonic() < deadline:
        if process.proc.poll() is not None:
            raise RuntimeError("Owned SGLang server exited before readiness")
        try:
            with opener.open(Request(url + "/health"), timeout=min(2, max(0.05, deadline - time.monotonic()))) as response:
                if response.status == 200 and process.proc.poll() is None:
                    return
        except (URLError, OSError, TimeoutError):
            pass
        stopped.wait(0.25)
    if stopped.is_set():
        raise InterruptedError("Serving startup was cancelled")
    raise TimeoutError("Owned SGLang server readiness deadline exceeded")


def evaluate(args, evidence, url, output):
    from sglang.test.run_eval import run_eval_once
    from sglang.test.simple_eval_common import make_report
    from sglang.test.simple_eval_gsm8k import GSM8KEval

    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",localhost,127.0.0.1,::1"
    evaluator = GSM8KEval(num_examples=evidence["num_examples"], num_threads=args.num_threads,
                         num_shots=args.num_shots, data_path=args.data_path)
    settings = SimpleNamespace(model=args.model_path, max_tokens=args.max_tokens, api=args.api,
                               temperature=0.0, top_p=1.0, chat_template_kwargs={})
    result, latency, sampler = run_eval_once(settings, url + "/v1", evaluator)
    (output / "answers.json").write_text(json.dumps(result.convos, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    (output / "report.html").write_text(make_report(result), encoding="utf-8")
    payload = normalize_result(result, latency, args, evidence)
    payload["served_model"] = sampler.model
    write_json(output / "accuracy.json", payload)
    emit(payload)
    return 1 if payload["correctness"] == "failed" else 0


def normalize_result(result, latency, args, evidence):
    if len(result.convos) != evidence["num_examples"]:
        raise ValueError("Evaluation returned fewer answers than the configured sample count")
    score = float(result.score)
    if not math.isfinite(score) or not 0 <= score <= 1 or not math.isfinite(latency) or latency < 0:
        raise ValueError("Invalid native accuracy score or evaluation latency")
    empty = sum(not convo or not str(convo[-1].get("content", "")).strip() for convo in result.convos)
    if empty == evidence["num_examples"]:
        raise ValueError("All responses are empty; do not report this as a completed accuracy measurement")
    threshold = score_threshold(args.min_score)
    correctness = "not_checked" if threshold is None else "passed" if score >= threshold else "failed"
    return {"measurement_status": "measured", "correctness": correctness, "threshold": threshold,
            "accuracy_judgement": "report_only" if threshold is None else correctness,
            "dataset": "gsm8k", "evidence": evidence,
            "metrics": [{"name": "gsm8k_accuracy", "value": score, "unit": "ratio", "statistic": "mean"},
                        {"name": "gsm8k_eval_latency", "value": latency, "unit": "s", "statistic": "total"},
                        {"name": "gsm8k_evaluated_examples", "value": evidence["num_examples"], "unit": "examples", "statistic": "count"},
                        {"name": "gsm8k_empty_response_rate", "value": empty / evidence["num_examples"], "unit": "ratio", "statistic": "mean"}]}


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    profile, command, env = profile_for(args)
    evidence = preflight(args)
    evidence["communication_env"] = {key: value for key, value in env.items() if key.startswith(("ROCSHMEM_", "NCCL_", "GLOO_", "HIP_"))}
    write_json(output / "environment.json", evidence)
    write_json(output / "command.json", {"argv": command, "profile": profile})
    if args.check_only:
        print("SGLang accuracy preflight OK; no server started", flush=True)
        return 0
    ensure_free_port(args)
    stopped = cancellation()
    process = NativeProcess(command, str(output), env, output / "server.log")
    try:
        wait_ready(process, local_url(args), args.startup_timeout, stopped)
        if stopped.is_set():
            return 130
        code = evaluate(args, evidence, local_url(args), output)
        if process.proc.poll() is not None:
            raise RuntimeError("Owned SGLang server exited during evaluation")
        return code
    finally:
        process.close()


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return run(args)
    except CleanupError as exc:
        return cleanup_failure(args.output_dir, exc)
    except Exception as exc:
        print("[accuracy error] " + str(exc), flush=True)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "error.json", {"error": str(exc), "status": "failed"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
