import copy
import re
from pathlib import Path

import yaml

from .models import BenchError, ID_PATTERN, SUITES, argv, keys, positive, require


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        require(isinstance(key, str), "Configuration keys must be strings")
        require(key not in result, f"Duplicate configuration key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def load_document(path: Path) -> dict:
    try:
        content = path.read_text(encoding="utf-8-sig")
        document = yaml.load(content, Loader=UniqueLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise BenchError(f"Cannot read {path}: {exc}") from exc
    require(isinstance(document, dict), f"{path} must contain a mapping")
    return document


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    result = copy.deepcopy(config)
    for override in overrides:
        require("=" in override, f"Override must be PATH=VALUE: {override}")
        path, value = override.split("=", 1)
        parts = path.split(".")
        parameter_override = len(parts) == 4 and parts[0] == "tests" and parts[2] == "params"
        if parameter_override and isinstance(result.get("tests"), dict) and parts[1] in result["tests"]:
            result["tests"][parts[1]].setdefault("params", {})
        target = result
        for part in parts[:-1]:
            require(isinstance(target, dict) and part in target, f"Unknown override path: {path}")
            target = target[part]
        require(isinstance(target, dict) and (parts[-1] in target or parameter_override), f"Unknown override path: {path}")
        try:
            target[parts[-1]] = yaml.load(value, Loader=UniqueLoader)
        except yaml.YAMLError as exc:
            raise BenchError(f"Invalid override {path}: {exc}") from exc
    return result


def resolve_variables(config: dict) -> dict:
    """Expand only vars.*, leaving worker placeholders for the execution stage."""
    variables = config.get("vars", {})
    require(isinstance(variables, dict), "vars must be a mapping")
    require(all(isinstance(name, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name)
                for name in variables), "vars names must be ASCII identifiers")
    pattern = re.compile(r"\$\{vars\.([A-Za-z][A-Za-z0-9_]*)\}")
    resolved, active = {}, set()

    def variable(name):
        require(name in variables, f"Unknown configuration variable: vars.{name}")
        require(name not in active, f"Circular configuration variable: vars.{name}")
        if name not in resolved:
            active.add(name)
            resolved[name] = expand(variables[name])
            active.remove(name)
        return copy.deepcopy(resolved[name])

    def expand(value):
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if not isinstance(value, str):
            require(value is None or isinstance(value, (bool, int, float)), "Unsupported configuration value type")
            return value
        matches = list(pattern.finditer(value))
        if len(matches) == 1 and matches[0].group() == value:
            return variable(matches[0].group(1))

        def replace(match):
            item = variable(match.group(1))
            require(isinstance(item, (str, int, float, bool)),
                    f"vars.{match.group(1)} must be scalar inside a string")
            return str(item)

        result = pattern.sub(replace, value)
        require("${vars." not in result, f"Invalid configuration variable reference: {value}")
        return result

    for name in variables:
        variable(name)
    return {key: copy.deepcopy(resolved) if key == "vars" else expand(value) for key, value in config.items()}


def validate_config(data: dict) -> None:
    keys(data, {"version", "name", "output_dir", "failure_policy", "nodes", "containers", "tests", "max_cases", "vars"}, "config")
    require(data.get("version") == 1, "Configuration version must be 1")
    require(isinstance(data.get("name"), str) and data["name"], "name is required")
    require(data.get("failure_policy", "continue") in ("continue", "stop"), "failure_policy must be continue or stop")
    require(isinstance(data.get("output_dir", "runs"), str), "output_dir must be a path")
    require(isinstance(data.get("nodes"), dict) and data["nodes"], "nodes are required")
    require(isinstance(data.get("tests"), dict) and data["tests"], "tests are required")
    require(isinstance(data.get("containers", {}), dict), "containers must be a mapping")
    maximum = data.get("max_cases", 10000)
    require(isinstance(maximum, int) and not isinstance(maximum, bool) and maximum > 0, "max_cases must be a positive integer")
    for name, node in data["nodes"].items():
        require(bool(ID_PATTERN.fullmatch(name)), f"Invalid node ID: {name}")
        keys(node, {"executor", "host", "user", "port", "identity_file", "python", "work_root", "gpus", "env", "connect_timeout_s"}, f"node {name}")
        require(node.get("executor") in ("local", "ssh"), f"{name}: executor must be local or ssh")
        gpus = node.get("gpus", [])
        require(isinstance(gpus, list) and all(isinstance(gpu, (int, str)) and not isinstance(gpu, bool) for gpu in gpus), f"{name}: gpus must be a list")
        require(len({str(gpu) for gpu in gpus}) == len(gpus), f"{name}: duplicate GPUs")
        env = node.get("env", {})
        require(isinstance(env, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()), f"{name}: env values must be strings")
        if node["executor"] == "ssh":
            host = node.get("host")
            require(isinstance(host, str) and bool(host) and not host.startswith("-") and not any(c.isspace() for c in host), f"{name}: host is required (IP or SSH config alias)")
            require(isinstance(node.get("work_root"), str) and node["work_root"].startswith("/"), f"{name}: work_root must be an absolute Linux path")
            argv([node.get("python", "python3")], f"{name}.python")
            for key in ("user", "identity_file"):
                if key in node:
                    require(isinstance(node[key], str) and bool(node[key]) and not node[key].startswith("-"), f"Invalid {name}.{key}")
            port = node.get("port", 22)
            require(isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535, f"Invalid SSH port on {name}")
            require(positive(node.get("connect_timeout_s", 10), f"{name}.connect_timeout_s") >= 1, f"{name}.connect_timeout_s must be at least 1 second")
    for name, container in data.get("containers", {}).items():
        require(bool(ID_PATTERN.fullmatch(name)), f"Invalid container profile: {name}")
        keys(container, {"name", "mode", "work_root", "start", "check", "stop", "hook_timeout_s", "exec_prefix"}, f"container {name}")
        require(container.get("mode") in ("existing", "managed"), f"{name}: explicitly select existing or managed")
        require(isinstance(container.get("name"), str) and bool(container["name"]), f"{name}: container name is required")
        require(isinstance(container.get("work_root"), str) and container["work_root"].startswith("/"), f"{name}: writable work_root inside the container must be an absolute Linux path")
        argv(container.get("check"), f"{name}.check")
        positive(container.get("hook_timeout_s", 60), f"{name}.hook_timeout_s")
        if "exec_prefix" in container:
            argv(container["exec_prefix"], f"{name}.exec_prefix")
        if container["mode"] == "managed":
            argv(container.get("start"), f"{name}.start")
            argv(container.get("stop"), f"{name}.stop")
            require("${run_id}" in container["name"] and "${case_id}" in container["name"] and "${node_id}" in container["name"], f"{name}: managed name must include ${{run_id}}, ${{case_id}}, ${{node_id}} to avoid claiming another container")
        else:
            require("start" not in container and "stop" not in container, f"{name}: existing containers must not have start/stop hooks")
    for test_id, test in data["tests"].items():
        require(bool(ID_PATTERN.fullmatch(test_id)), f"Invalid test ID: {test_id}")
        keys(test, {"suite", "adapter", "enabled", "nodes", "pack", "params", "matrix", "repeats", "timeout_s", "duration_s", "container", "env"}, f"test {test_id}")
        require(test.get("suite") in SUITES, f"{test_id}: unknown suite")
        require(isinstance(test.get("enabled", True), bool), f"{test_id}: enabled must be boolean")
        positive(test.get("timeout_s"), f"{test_id}.timeout_s")
        if test.get("duration_s") is not None:
            positive(test["duration_s"], f"{test_id}.duration_s")
            require(test["duration_s"] < test["timeout_s"], f"{test_id}: duration_s must be less than timeout_s")
        require(isinstance(test.get("params", {}), dict), f"{test_id}: params must be a mapping")
        env = test.get("env", {})
        require(isinstance(env, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()), f"{test_id}: env values must be strings")
        require(isinstance(test.get("matrix", {}), dict), f"{test_id}: matrix must be a mapping")
        repeat = test.get("repeats", 1)
        require(isinstance(repeat, int) and not isinstance(repeat, bool) and repeat > 0, f"{test_id}: repeats must be a positive integer")
