import hashlib
import itertools
import json
from pathlib import Path

from .config import apply_overrides, load_document, resolve_variables, validate_config
from .models import Case, Plan, SUITES, require
from .registry import get_pack, resolve_params


def make_plan(path: str, selection: str = "all", overrides: list[str] | None = None) -> Plan:
    source = Path(path).resolve()
    config = resolve_variables(apply_overrides(load_document(source), overrides or []))
    validate_config(config)
    selected = list(SUITES) if selection == "all" else selection.split(",")
    require(bool(selected) and all(suite in SUITES for suite in selected) and len(selected) == len(set(selected)), f"Invalid suite selection: {selection}")
    cases = []
    case_ids = set()
    for test_id, test in config["tests"].items():
        if not test.get("enabled", True) or test.get("suite") not in selected:
            continue
        pack = get_pack(test, source.parent)
        nodes = test.get("nodes")
        require(isinstance(nodes, list) and nodes and all(isinstance(node, str) and node in config["nodes"] for node in nodes), f"{test_id}: choose known nodes")
        require(len(nodes) == len(set(nodes)), f"{test_id}: duplicate nodes")
        if test["adapter"] == "mock":
            require(all(config["nodes"][node]["executor"] == "local" for node in nodes), "mock runs only on local nodes; it does not represent remote hardware")
        container = test.get("container")
        require(container is None or container in config.get("containers", {}), f"{test_id}: unknown container profile")
        matrix = test.get("matrix", {})
        count = test.get("repeats", 1)
        for name, values in matrix.items():
            require(isinstance(values, list) and values, f"{test_id}.{name}: matrix values must be a non-empty list")
            count *= len(values)
        require(count + len(cases) <= config.get("max_cases", 10000), "Parameter matrix exceeds max_cases; use selection/chunks, not a million-row matrix")
        combinations = itertools.product(*matrix.values()) if matrix else [()]
        for values in combinations:
            params = resolve_params(pack, {**test.get("params", {}), **dict(zip(matrix, values))})
            for repeat in range(test.get("repeats", 1)):
                identity = {"test": test_id, "params": params, "repeat": repeat, "nodes": nodes}
                digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:10]
                case_id = f"{test_id}-{digest}"
                require(case_id not in case_ids, "Duplicate matrix values produce the same case ID")
                case_ids.add(case_id)
                cases.append(Case(f"{test_id}-{digest}", test_id, test["suite"], test["adapter"], nodes, params, repeat,
                                  float(test["timeout_s"]), test.get("duration_s"), container, pack, test.get("env", {})))
    require(bool(cases), "No enabled tests match this selection")
    nodes = {name: dict(value) for name, value in config["nodes"].items()}
    for node in nodes.values():
        if "identity_file" in node:
            node["identity_file"] = str((source.parent / node["identity_file"]).expanduser().resolve())
    output_dir = str((source.parent / config.get("output_dir", "runs")).resolve())
    return Plan(config["name"], str(source), output_dir, config.get("failure_policy", "continue"), nodes,
                config.get("containers", {}), cases, config)
