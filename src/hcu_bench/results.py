import json
import math

from .models import BenchError, require

PREFIX = "BENCH_RESULT "


def parse_sample(line: str):
    if not line.startswith(PREFIX):
        return None
    try:
        value = json.loads(line[len(PREFIX):])
    except ValueError as exc:
        raise BenchError(f"Malformed benchmark JSON: {exc}") from exc
    require(isinstance(value, dict), "Benchmark result must be a mapping")
    require(value.get("measurement_status") in ("measured", "invalid", "skipped", "simulated"), "Result measurement_status is required")
    require(value.get("correctness") in ("passed", "failed", "not_checked"), "Result correctness must be explicit")
    metrics = value.get("metrics", [])
    require(isinstance(metrics, list), "metrics must be a list")
    if value["measurement_status"] in ("measured", "simulated"):
        require(bool(metrics), "Measured result contains no metrics")
    for metric in metrics:
        require(isinstance(metric, dict), "Metric must be a mapping")
        for field in ("name", "unit", "statistic"):
            require(isinstance(metric.get(field), str) and bool(metric[field]), f"Metric {field} is required")
        number = metric.get("value")
        require(isinstance(number, (int, float)) and not isinstance(number, bool) and math.isfinite(number), "Metric value must be finite numeric data")
    return value
