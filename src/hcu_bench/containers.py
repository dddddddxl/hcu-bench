import base64
import json
import re
from string import Template
from typing import Protocol

from .executors.base import AGENT_PATH
from .models import BenchError, require


def render(value: str, context: dict) -> str:
    try:
        return Template(value).substitute(context)
    except (KeyError, ValueError) as exc:
        raise BenchError(f"Invalid or unavailable template variable in {value!r}: {exc}") from exc


def render_argv(command: list[str], context: dict) -> list[str]:
    return [render(argument, context) for argument in command]


class ContainerLauncher(Protocol):
    def build(self, profile: dict, context: dict, workload: dict) -> dict: ...


class HookContainerLauncher:
    """Keep image, mounts, devices and privileges entirely in user-supplied hooks."""

    def build(self, profile, context, workload):
        name = render(profile["name"], context)
        require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)), "Invalid rendered Docker container name")
        context = {**context, "container_name": name}
        hooks = {key: render_argv(profile[key], context) for key in ("start", "check", "stop") if key in profile}
        inner = {**workload, "work_dir": context["case_dir"],
                 "lock_root": profile["work_root"], "resources": [], "container": None, "wrapped": False}
        payload = base64.b64encode(json.dumps(inner).encode()).decode()
        prefix = render_argv(profile.get("exec_prefix", ["docker", "exec", "-i", "${container_name}", "python3", "-u"]), context)
        require("-it" not in prefix and "-t" not in prefix, "Container exec must not allocate a TTY")
        return {"name": name, "mode": profile["mode"], "hook_timeout_s": float(profile.get("hook_timeout_s", 60)),
                **hooks, "argv": prefix + ["-c", AGENT_PATH.read_text(encoding="utf-8"), payload]}
