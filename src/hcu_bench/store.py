import json
import os
import platform
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path

from . import __version__
from .models import BenchError, Plan, require


def now():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def write_json(path: Path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


class RunStore:
    def __init__(self, directory):
        self.path = Path(directory).resolve()
        self.lock = threading.RLock()

    @classmethod
    def create(cls, plan: Plan):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        store = cls(Path(plan.output_dir) / run_id)
        store.path.mkdir(parents=True)
        (store.path / "logs").mkdir()
        (store.path / "raw").mkdir()
        write_json(store.path / "plan.json", plan.to_dict())
        write_json(store.path / "resolved.json", plan.resolved)
        write_json(store.path / "environment.json", {"bench_version": __version__, "controller_platform": platform.platform(),
                                                    "controller_python": sys.version, "targets": {}})
        write_json(store.path / "state.json", {"run_id": run_id, "name": plan.name, "status": "queued", "created_at": now(),
                                             "updated_at": now(), "cases": {case.id: {"status": "pending", "test_id": case.test_id,
                                             "suite": case.suite, "params": case.params, "repeat": case.repeat,
                                             "simulated": case.adapter == "mock"} for case in plan.cases}})
        return store

    def read(self, name):
        try:
            return json.loads((self.path / name).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BenchError(f"Cannot read run {self.path}: {exc}") from exc

    def state(self, **updates):
        with self.lock:
            value = self.read("state.json")
            value.update(updates, updated_at=now())
            write_json(self.path / "state.json", value)

    def case(self, case_id, **updates):
        with self.lock:
            value = self.read("state.json")
            value["cases"][case_id].update(updates, updated_at=now())
            value["updated_at"] = now()
            write_json(self.path / "state.json", value)

    def append(self, name, value):
        with self.lock:
            with (self.path / name).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

    def environment(self, key, value):
        with self.lock:
            data = self.read("environment.json")
            data["targets"][key] = value
            write_json(self.path / "environment.json", data)

    def stopped(self):
        return (self.path / "STOP").exists()


def open_run(path: str) -> RunStore:
    store = RunStore(path)
    require((store.path / "state.json").is_file(), f"Not a run directory: {store.path}")
    return store
