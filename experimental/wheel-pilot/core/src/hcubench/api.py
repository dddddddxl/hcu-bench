import hashlib
import importlib.metadata as metadata
import inspect
import json
import math
import statistics
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROTOCOL = "hcubench.operator.v1"
EXTERNAL_PROTOCOL = "hcubench.external.v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def discover():
    found = {}
    for ep in metadata.entry_points(group="hcubench.plugins"):
        if ep.name in found:
            raise ValueError(f"Duplicate plugin: {ep.name}")
        found[ep.name] = ep
    return found


def plugin(name):
    ep = discover().get(name)
    if ep is None:
        raise ValueError(f"Plugin not installed: {name}")
    obj = ep.load()()
    if obj.protocol not in (PROTOCOL, EXTERNAL_PROTOCOL):
        raise ValueError(f"Unsupported plugin protocol: {obj.protocol}")
    return obj


def describe(name):
    obj = plugin(name)
    return {"test": name, "protocol": obj.protocol, "description": obj.description, "parameters": obj.schema}


def resolve(config):
    required = {"schema_version", "test", "params"}
    if not isinstance(config, dict) or not required <= set(config) or set(config) - required - {"source"}:
        raise ValueError("Config requires schema_version, test, params; optional source")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Unsupported config schema_version")
    if not isinstance(config["test"], str):
        raise ValueError("test must be a string")
    obj = plugin(config["test"])
    raw = config["params"]
    if not isinstance(raw, dict) or set(raw) - set(obj.schema):
        raise ValueError("Unknown parameter(s) or invalid params mapping")
    params = {}
    for name, rule in obj.schema.items():
        value = raw.get(name, rule["default"])
        expected = {"integer": int, "number": (int, float), "string": str}[rule["type"]]
        if isinstance(value, bool) or not isinstance(value, expected):
            raise ValueError(f"{name}: expected {rule['type']}")
        if isinstance(value, (int, float)):
            if not math.isfinite(value) or value < rule.get("minimum", -math.inf) or value > rule.get("maximum", math.inf):
                raise ValueError(f"{name}: outside supported range")
        if "enum" in rule and value not in rule["enum"]:
            raise ValueError(f"{name}: choose {rule['enum']}")
        params[name] = value
    obj.validate(params)
    resolved = {"schema_version": 1, "test": config["test"], "params": params}
    if "source" in config:
        if not hasattr(obj, "validate_source"):
            raise ValueError("This plugin does not support dataset provenance")
        obj.validate_source(config["source"], params)
        resolved["source"] = config["source"]
    return obj, resolved


def identities(config, obj):
    params = config["params"]
    case = {k: v for k, v in params.items() if obj.schema[k]["identity"]}
    # Content hash replaces installation path; implementation belongs to config identity.
    source = Path(inspect.getfile(type(obj))).read_bytes()
    implementation = {"plugin_version": obj.version, "source_sha256": hashlib.sha256(source).hexdigest()}
    if hasattr(obj, "implementation_files"):
        implementation["files"] = {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in obj.implementation_files}
    core_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    logical = {"protocol": obj.protocol, "test": config["test"], "case": case}
    effective = {k: v for k, v in params.items() if k != "device" and not obj.schema[k].get("location", False)}
    return {"case_id": digest(logical), "config_hash": digest({"logical": logical, "params": effective, "source": config.get("source"), "implementation": implementation, "core_sha256": core_sha}), "implementation": implementation, "core_sha256": core_sha}


def check(config):
    obj, resolved = resolve(config)
    return {"resolved": resolved, **identities(resolved, obj), "environment": obj.preflight(resolved["params"])}


def validate_payload(correctness, measurement, expected_samples):
    if not isinstance(correctness, dict) or type(correctness.get("passed")) is not bool:
        raise ValueError("Plugin omitted correctness result")
    if not correctness.get("reference") or not correctness.get("tolerance"):
        raise ValueError("Plugin omitted reference or tolerance")
    if not correctness["passed"]:
        raise ValueError("Correctness failed; performance cannot be accepted")
    if measurement.get("unit") != "ms" or measurement.get("sample_kind") != "single_invocation":
        raise ValueError("Expected raw single-invocation samples in ms")
    values = measurement.get("samples_ms")
    if not isinstance(values, list) or len(values) != expected_samples or not values:
        raise ValueError("Raw sample count mismatch")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Invalid raw timing sample")
    if not measurement.get("boundary") or not measurement.get("method"):
        raise ValueError("Timing boundary/method missing")


def summarize(values):
    ordered = sorted(values)
    p10 = ordered[math.ceil(.10 * len(values)) - 1]
    p90 = ordered[math.ceil(.90 * len(values)) - 1]
    p50 = statistics.median(values)
    spread = (p90 - p10) / p50
    return {"count": len(values), "mean_ms": statistics.mean(values), "median_ms": p50, "p50_ms": p50, "p10_ms": p10, "p90_ms": p90, "p95_ms": ordered[math.ceil(.95 * len(values)) - 1], "p95_method": "nearest_rank", "percentile_method": "P10/P90/P95 nearest_rank; P50 median", "spread_p90_p10_over_p50": spread, "stable_under_5pct": spread <= .05, "min_ms": ordered[0], "max_ms": ordered[-1], "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run(config, output):
    obj, resolved = resolve(config)
    if obj.protocol != PROTOCOL:
        raise ValueError("External tests use plan + import-result; launch the test with your own runner")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10]
    target = Path(output).resolve() / run_id
    target.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "run_id": run_id, **identities(resolved, obj), "resolved": resolved, "started_at": datetime.now(timezone.utc).isoformat(), "execution": "running", "evidence_valid": False, "acceptance": "pilot_only", "performance_regression_passed": None, "warnings": ["Pilot tolerances are not team-approved acceptance thresholds", "No exclusive device lease or continuous interference monitor"]}
    write_json(target / "config.json", resolved)
    try:
        result["environment"] = obj.preflight(resolved["params"])
        result["environment_fingerprint"] = digest(result["environment"])
        correctness, measurement = obj.execute(resolved["params"])
        write_json(target / "correctness.json", correctness)
        write_json(target / "measurement.json", measurement)
        result["correctness"] = correctness
        validate_payload(correctness, measurement, resolved["params"]["samples"])
        result["statistics"] = summarize(measurement["samples_ms"])
        result["measurement_stability"] = "stable" if result["statistics"]["stable_under_5pct"] else "unstable"
        if "source" in resolved:
            historical_ns = resolved["source"]["reference_time_ns"]
            result["historical_comparison"] = {"reference_time_ns": historical_ns, "measured_p50_ns": result["statistics"]["p50_ms"] * 1e6, "relative_difference": result["statistics"]["p50_ms"] * 1e6 / historical_ns - 1, "comparable": False, "regression_passed": None, "reason": "Historical software stack and timing procedure are incomplete; source hardware label is not verified equivalent to this device"}
        result["execution"] = "completed"
        result["evidence_valid"] = True
    except Exception as exc:
        result["execution"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        (target / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    result["artifacts"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(target.iterdir()) if p.is_file()}
    write_json(target / "result.json", result)
    render_report(result, target / "report.html")
    return target, result


def plan(config):
    obj, resolved = resolve(config)
    if obj.protocol != EXTERNAL_PROTOCOL:
        raise ValueError("This plugin does not support external execution plans")
    return {"schema_version": 1, "resolved": resolved, **identities(resolved, obj), **obj.plan(resolved["params"])}


def import_result(config, bundle, output):
    """Snapshot external evidence before parsing; never launch a subprocess."""
    obj, resolved = resolve(config)
    if obj.protocol != EXTERNAL_PROTOCOL:
        raise ValueError("This plugin does not support external result import")
    target = Path(output).resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10])
    target.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "run_id": target.name, **identities(resolved, obj), "resolved": resolved,
              "execution": "unknown", "evidence_valid": False, "acceptance": "pilot_only",
              "performance_regression_passed": None, "measurement_stability": "not_assessed",
              "warnings": ["External evidence is user-supplied, not an authenticated execution attestation", "Iteration aggregates do not provide per-invocation percentiles", "No exclusive device lease or continuous interference monitor"]}
    write_json(target / "config.json", resolved)
    raw = target / "raw"
    raw.mkdir()
    try:
        for name in ("execution.json", "stdout.log", "stderr.log"):
            source = Path(bundle) / name
            if source.is_symlink() or not source.is_file() or source.stat().st_size > 64 * 1024 * 1024:
                raise ValueError(f"Invalid or oversized evidence file: {name}")
            (raw / name).write_bytes(source.read_bytes())
        assessment = obj.assess(raw, resolved["params"], result["config_hash"])
        result.update(assessment)
        result["environment_fingerprint"] = digest(result.get("environment", {}))
        write_json(target / "correctness.json", result["correctness"])
        write_json(target / "measurement.json", result["measurement"])
        obj.export_csv(result["measurement"], target / "results.csv")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["evidence_valid"] = False
        (target / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
    result["imported_at"] = datetime.now(timezone.utc).isoformat()
    result["artifacts"] = {p.relative_to(target).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(target.rglob("*")) if p.is_file()}
    write_json(target / "result.json", result)
    render_report(result, target / "report.html")
    return target, result


def render_report(result, path):
    import html
    if result.get("measurement", {}).get("sample_kind") == "iteration_aggregate":
        rows = result["measurement"]["rows"]
        cells = "".join("<tr>" + "".join(f"<td>{html.escape(str(row[k]))}</td>" for k in ("size_bytes", "mode", "time_us", "algbw_GBps", "busbw_GBps", "wrong")) + "</tr>" for row in rows)
        issues = html.escape(json.dumps(result.get("issues", []), ensure_ascii=False))
        path.write_text("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>HCU Bench RCCL</title><style>body{max-width:1100px;margin:40px auto;font-family:system-ui;color:#182133;background:#f6f7f9}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px;text-align:right;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style><h1>HCU Bench · RCCL AllReduce</h1>"
                        + f"<p>执行：{html.escape(result['execution'])} · 证据有效：{result['evidence_valid']} · 工程试点</p>"
                        + "<p>按消息大小分别展示两种缓冲区模式。耗时为 rccl-tests 迭代汇总；无逐次样本，不计算 P50/P99。尚未判定性能达标。</p>"
                        + f"<p>校验问题：{issues}</p><table><tr><th>Bytes</th><th>模式</th><th>耗时 μs</th><th>algbw GB/s</th><th>busbw GB/s</th><th>错误数</th></tr>{cells}</table>"
                        + "<p>原始日志：raw/stdout.log、raw/stderr.log；执行记录：raw/execution.json；明细：results.csv。</p></html>", encoding="utf-8")
        return
    text = html.escape(json.dumps(result, indent=2, ensure_ascii=False))
    path.write_text("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>HCU Bench pilot</title><style>body{max-width:1000px;margin:40px auto;font-family:system-ui;background:#f6f7f9;color:#182133}pre{white-space:pre-wrap;background:white;padding:24px;border:1px solid #ddd}h1{font-size:26px}</style><h1>HCU Bench · 单次测试证据</h1><p>工程试点结果；不代表正式性能达标或回归结论。逐次样本见同目录 measurement.json。</p><pre>" + text + "</pre></html>", encoding="utf-8")
