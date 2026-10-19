"""Expand selected model tasks into one owned serving session per deployment."""
import hashlib
import json
from pathlib import Path

from .models import ID_PATTERN, keys, positive, require

ACC_DEFAULTS = dict(dataset="gsm8k", data_path="auto", num_examples=100, num_shots=5,
                    num_threads=128, max_tokens=2048, api="chat", min_score="")
PERF_DEFAULTS = dict(shapes=[[128, 128], [1024, 1024], [4096, 512]], concurrency=[1, 8, 32],
                     requests=256, warmup_requests=32, repeats=3, seed=42, request_timeout_s=600,
                     extra_body={}, python="auto")


def validate_models(models):
    require(isinstance(models, dict), "models must be a mapping")
    for name, model in models.items():
        require(isinstance(name, str) and bool(ID_PATTERN.fullmatch(name)), f"Invalid model ID: {name}")
        keys(model, {"start_script", "tokenizer_path", "model_name", "port", "startup_timeout_s",
                     "nodes", "container", "env", "health_path", "before_perf_script"}, f"model {name}")
        for field in ("start_script", "tokenizer_path", "model_name", "health_path", "before_perf_script"):
            value = model.get(field, "/health" if field == "health_path" else "")
            require(isinstance(value, str) and "\x00" not in value, f"{name}.{field} must be a string without NUL")
        require(bool(model.get("start_script")), f"{name}.start_script is required")
        require(model.get("health_path", "/health").startswith("/"), f"{name}.health_path must start with /")
        port = model.get("port", 30000)
        require(type(port) is int and 1 <= port <= 65535, f"{name}: invalid port")
        positive(model.get("startup_timeout_s", 1800), f"{name}.startup_timeout_s")
        env = model.get("env", {})
        require(isinstance(env, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()),
                f"{name}.env must contain strings")


def task_params(suite, supplied):
    require(suite in ("e2e_acc", "e2e_perf"), "Unknown end-to-end task suite")
    defaults = ACC_DEFAULTS if suite == "e2e_acc" else PERF_DEFAULTS
    keys(supplied, set(defaults), suite + " params")
    params = {**defaults, **supplied}
    integer_fields = (("num_examples", "num_shots", "num_threads", "max_tokens") if suite == "e2e_acc"
                      else ("requests", "warmup_requests", "repeats", "seed", "request_timeout_s"))
    for field in integer_fields:
        minimum = 0 if field in ("num_examples", "num_shots", "warmup_requests", "seed") else 1
        require(type(params[field]) is int and params[field] >= minimum, f"{suite}.{field} must be an integer >= {minimum}")
    if suite == "e2e_acc":
        require(params["dataset"] == "gsm8k", "e2e_acc currently supports gsm8k only")
        require(params["api"] in ("chat", "completion"), "api must be chat or completion")
        require(isinstance(params["data_path"], str) and params["data_path"], "data_path must be a path or auto")
        threshold = params["min_score"]
        require(isinstance(threshold, str), "min_score must be a string or empty")
        if threshold:
            try:
                score = float(threshold)
            except ValueError:
                score = -1
            require(0 <= score <= 1, "min_score must be in [0, 1]")
    else:
        shapes, concurrency = params["shapes"], params["concurrency"]
        require(isinstance(shapes, list) and shapes and all(isinstance(s, list) and len(s) == 2
                and all(type(n) is int and n > 0 for n in s) for s in shapes), "shapes must be [[input_tokens, output_tokens], ...]")
        require(isinstance(concurrency, list) and concurrency and all(type(n) is int and n > 0 for n in concurrency),
                "concurrency must be positive integers")
        require(len(concurrency) == len(set(concurrency)) and len(shapes) == len({tuple(s) for s in shapes}), "Duplicate performance combinations")
        require(params["requests"] >= max(concurrency), "requests must be >= maximum concurrency")
        require(len(shapes) * len(concurrency) * params["repeats"] <= 4096, "Performance sweep exceeds 4096 points")
        positive(params["request_timeout_s"], "request_timeout_s")
        require(isinstance(params["python"], str) and params["python"], "python must be auto or a prepared client interpreter")
        require(isinstance(params["extra_body"], dict), "extra_body must be an object")
        require(not set(params["extra_body"]) & {"model", "messages", "prompt", "stream", "max_tokens", "min_tokens", "max_completion_tokens", "n"},
                "extra_body cannot override model, prompts, stream, token limits or number of completions")
    return params


def expand_model_tests(config, selected):
    result, sessions = [], {}
    manifest = Path(__file__).resolve().parents[2] / "testpacks/e2e/manifest.yaml"
    for test_id, test in config["tests"].items():
        if not test.get("enabled", True) or test["suite"] not in selected:
            continue
        if "models" not in test:
            result.append((test_id, test))
            continue
        params = task_params(test["suite"], test.get("params", {}))
        bench_root = config.get("vars", {}).get("bench_root")
        require(isinstance(bench_root, str) and bool(bench_root), "Model sessions require vars.bench_root (repository path in the target environment)")
        for name in test["models"]:
            model = config["models"][name]
            nodes = model.get("nodes", test.get("nodes"))
            require(isinstance(nodes, list) and len(nodes) == 1 and all(isinstance(n, str) and n in config["nodes"] for n in nodes),
                    f"{name}: model sessions support at most 1 serving node; choose one known node")
            container = model.get("container", test.get("container"))
            require(container is None or isinstance(container, str), f"{name}: container must name a profile")
            env = {**test.get("env", {}), **model.get("env", {})}
            identity = json.dumps([name, nodes, container, env], sort_keys=True)
            if identity not in sessions:
                suffix = hashlib.sha256(identity.encode()).hexdigest()[:8]
                session_id = "model_" + name[:55] + "_" + suffix
                require(session_id not in config["tests"], f"Reserved generated test ID: {session_id}")
                # The adapter renders env once; do not reapply raw placeholders in the wrapper.
                deployment = {key: value for key, value in model.items() if key != "env"}
                session = dict(model_id=name, model=deployment, tasks=[], failure_policy=config.get("failure_policy", "continue"))
                entry = dict(suite=test["suite"], adapter="command", pack=str(manifest), nodes=nodes,
                             container=container, env=env, timeout_s=model.get("startup_timeout_s", 1800) + 15,
                             params=dict(bench_root=bench_root, session=session))
                sessions[identity] = entry
                result.append((session_id, entry))
            entry = sessions[identity]
            entry["timeout_s"] += test["timeout_s"]
            tasks = entry["params"]["session"]["tasks"]
            tasks.append(dict(test_id=test_id, suite=test["suite"], params=params, timeout_s=test["timeout_s"]))
            tasks.sort(key=lambda task: task["suite"] != "e2e_acc")
            if len({task["suite"] for task in tasks}) > 1:
                entry["suite"] = "e2e"
    return result
