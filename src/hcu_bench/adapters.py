import json
import posixpath
import sys
from pathlib import Path
from typing import Protocol

from .containers import HookContainerLauncher, render, render_argv
from .models import Case, Plan, require


class Adapter(Protocol):
    def build(self, plan: Plan, case: Case, node_id: str, run_dir: Path) -> dict: ...


def context_for(plan, case, node_id, run_dir):
    node = plan.nodes[node_id]
    run_id = run_dir.name
    if node["executor"] == "local":
        root = str(run_dir.parent)
        case_dir = str(run_dir / "work" / case.id / node_id)
    else:
        root = node["work_root"]
        case_dir = posixpath.join(root, run_id, case.id, node_id)
    host_case_dir = case_dir
    if case.container:
        case_dir = posixpath.join(plan.containers[case.container]["work_root"], run_id, case.id, node_id)
    result = {"run_id": run_id, "case_id": case.id, "test_id": case.test_id, "node_id": node_id,
            "node_rank": case.nodes.index(node_id), "nnodes": len(case.nodes), "case_dir": case_dir,
            "host_case_dir": host_case_dir,
            "run_dir": str(run_dir), "gpu_ids": ",".join(str(gpu) for gpu in node.get("gpus", [])),
            "python": node.get("python", sys.executable if node["executor"] == "local" else "python3"),
            "lock_root": root, **{"p_" + key: value for key, value in case.params.items()}}
    if node["executor"] == "local" and not case.container and "manifest_path" in case.pack:
        result["pack_dir"] = str(Path(case.pack["manifest_path"]).parent)
    return result


class CommandAdapter:
    def build(self, plan, case, node_id, run_dir):
        node = plan.nodes[node_id]
        context = context_for(plan, case, node_id, run_dir)
        command = render_argv(case.pack["command"], context)
        cwd = render(case.pack["cwd"], context)
        if node["executor"] == "local" and not case.container and not Path(cwd).is_absolute():
            cwd = str((Path(case.pack["manifest_path"]).parent / cwd).resolve())
        if (node["executor"] == "ssh" or case.container) and not cwd.startswith("/"):
            require(False, "SSH/container cwd must be an absolute Linux path")
        env = {key: render(value, context) for key, value in {**node.get("env", {}), **case.pack.get("env", {}), **case.env}.items()}
        spec = {"argv": command, "cwd": cwd, "env": env, "work_dir": context["host_case_dir"],
                "lock_root": context["lock_root"], "resources": ["gpu:" + str(gpu) for gpu in node.get("gpus", [])] or ["node"],
                "gpus": node.get("gpus", []), "timeout_s": case.timeout_s, "duration_s": case.duration_s,
                "simulated": False, "wrapped": False}
        spec.update(artifacts=[render(path, context) for path in case.pack.get("artifacts", [])],
                    artifact_limit_mb=case.pack.get("artifact_limit_mb", 100), display={"argv": command, "cwd": cwd, "env": env})
        if case.container:
            container = HookContainerLauncher().build(plan.containers[case.container], context, spec)
            spec.update(argv=container.pop("argv"), cwd=context["host_case_dir"], env={}, wrapped=True,
                        duration_s=None, container=container, artifacts=[])
            # The inner worker owns the workload deadline; host allows its shutdown grace period.
            spec["timeout_s"] += 15
        return spec


class MockAdapter:
    def build(self, plan, case, node_id, run_dir):
        context = context_for(plan, case, node_id, run_dir)
        return {"argv": [context["python"], "-u", str(Path(__file__).with_name("mock_workload.py")), json.dumps(case.params)],
                "cwd": context["case_dir"], "env": {}, "work_dir": context["case_dir"], "lock_root": context["lock_root"],
                "resources": ["mock:" + node_id], "gpus": [], "timeout_s": case.timeout_s,
                "duration_s": case.duration_s, "simulated": True, "wrapped": False}


def for_case(case):
    return MockAdapter() if case.adapter == "mock" else CommandAdapter()


def describe_spec(spec):
    return {**spec.get("display", {"argv": spec["argv"], "cwd": spec["cwd"], "env": spec.get("env", {})}),
            "timeout_s": spec["timeout_s"], "duration_s": spec.get("duration_s"),
            "container": spec.get("container"), "simulated": spec["simulated"]}
