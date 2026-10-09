from pathlib import Path

from .config import load_document
from .models import SUITES, argv, keys, positive, require

MOCK_PACK = {
    "version": 1, "id": "mock", "suite": "any", "result_protocol": "bench-jsonl-v1",
    "requires_metrics": True,
    "parameters": {
        "payload_bytes": {"type": "integer", "default": 1024, "min": 1},
        "samples": {"type": "integer", "default": 3, "min": 1, "max": 10000},
        "delay_s": {"type": "number", "default": 0.01, "min": 0},
        "fail": {"type": "boolean", "default": False},
        "empty": {"type": "boolean", "default": False},
    },
}


def get_pack(test: dict, base: Path) -> dict:
    adapter = test.get("adapter")
    require(adapter in ("mock", "command"), "adapter must be mock or command")
    require(test.get("suite") in SUITES, f"Unknown suite: {test.get('suite')}")
    if adapter == "mock":
        require("pack" not in test and not test.get("container"), "mock must not use a real test pack or container")
        return dict(MOCK_PACK)
    require(isinstance(test.get("pack"), str), "command adapter requires a test-pack manifest path")
    pack_path = (base / test["pack"]).resolve()
    pack = load_document(pack_path)
    keys(pack, {"version", "id", "suite", "parameters", "command", "cwd", "env", "result_protocol", "requires_metrics", "artifacts", "artifact_limit_mb"}, "test pack")
    require(pack.get("version") == 1 and pack.get("suite") == test["suite"], "Test-pack version or suite mismatch")
    require(isinstance(pack.get("id"), str) and bool(pack["id"]), "Test-pack id is required")
    argv(pack.get("command"), "test-pack command")
    require(isinstance(pack.get("cwd"), str), "Test-pack cwd is required; use ${case_dir} for a scratch directory")
    require(pack.get("result_protocol") in ("bench-jsonl-v1", "none"), "Unsupported result_protocol")
    require(isinstance(pack.get("requires_metrics"), bool), "Test-pack requires_metrics must be explicitly true or false")
    require(not pack["requires_metrics"] or pack["result_protocol"] != "none", "requires_metrics cannot use protocol none")
    require(isinstance(pack.get("env", {}), dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in pack.get("env", {}).items()), "Test-pack env values must be strings")
    pack["manifest_path"] = str(pack_path)
    artifacts = pack.get("artifacts", [])
    require(isinstance(artifacts, list) and all(isinstance(path, str) and path for path in artifacts), "artifacts must be a list of declared path/glob strings")
    positive(pack.get("artifact_limit_mb", 100), "artifact_limit_mb")
    return pack


def resolve_params(pack: dict, supplied: dict) -> dict:
    schema = pack.get("parameters", {})
    require(isinstance(schema, dict), "Test-pack parameters must be a mapping")
    require(not set(supplied) - set(schema), f"Unsupported test parameters: {sorted(set(supplied) - set(schema))}")
    result = {}
    types = {"integer": int, "number": (int, float), "string": str, "boolean": bool}
    for name, definition in schema.items():
        require(isinstance(name, str) and name.isidentifier(), f"Invalid parameter name: {name}")
        keys(definition, {"type", "default", "required", "min", "max", "choices", "description"}, f"parameter {name}")
        require(definition.get("type") in types, f"Unknown type for {name}")
        for bound in ("min", "max"):
            if bound in definition:
                require(definition["type"] in ("integer", "number"), f"{bound} only applies to numeric parameter {name}")
                require(isinstance(definition[bound], (int, float)) and not isinstance(definition[bound], bool), f"Invalid {name}.{bound}")
        if "choices" in definition:
            require(isinstance(definition["choices"], list), f"{name}.choices must be a list")
        if name in supplied:
            value = supplied[name]
        elif "default" in definition:
            value = definition["default"]
        else:
            require(not definition.get("required", False), f"Required parameter missing: {name}")
            continue
        expected = definition["type"]
        require(isinstance(value, types[expected]) and not (expected in ("number", "integer") and isinstance(value, bool)), f"{name} must have type {expected}")
        if expected in ("number", "integer"):
            import math
            require(math.isfinite(value), f"{name} must be finite")
        if "min" in definition:
            require(value >= definition["min"], f"{name} must be >= {definition['min']}")
        if "max" in definition:
            require(value <= definition["max"], f"{name} must be <= {definition['max']}")
        if "choices" in definition:
            require(value in definition["choices"], f"Unsupported value for {name}: {value}")
        result[name] = value
    return result
