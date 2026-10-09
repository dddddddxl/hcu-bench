import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

SUITES = ("rccl", "deepep", "mooncake", "operators", "e2e")
ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,79}$")


class BenchError(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BenchError(message)


def keys(data: dict, allowed: set[str], label: str) -> None:
    require(isinstance(data, dict), f"{label} must be a mapping")
    extra = set(data) - allowed
    require(not extra, f"Unknown {label} keys: {sorted(extra)}")


def positive(value: Any, label: str, zero: bool = False) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} must be numeric")
    require(math.isfinite(value) and (value >= 0 if zero else value > 0), f"Invalid {label}: {value}")
    return float(value)


def argv(value: Any, label: str) -> list[str]:
    require(isinstance(value, list) and bool(value), f"{label} must be a non-empty argv list, not a shell string")
    require(all(isinstance(item, str) and '\x00' not in item for item in value), f"{label} must contain strings without NUL")
    return value


@dataclass
class Case:
    id: str
    test_id: str
    suite: str
    adapter: str
    nodes: list[str]
    params: dict[str, Any]
    repeat: int
    timeout_s: float
    duration_s: float | None
    container: str | None
    pack: dict
    env: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Plan:
    name: str
    source: str
    output_dir: str
    failure_policy: str
    nodes: dict
    containers: dict
    cases: list[Case]
    resolved: dict
    questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        result = asdict(self)
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "Plan":
        return cls(**{**data, "cases": [Case(**case) for case in data["cases"]]})
