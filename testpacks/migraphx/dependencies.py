"""Discover prepared MIGraphX tools and diagnose native runtime dependencies."""
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

DRIVER = "migraphx-driver"
DRIVER_LOCATIONS = ("/opt/dtk/bin/migraphx-driver", "/opt/rocm/bin/migraphx-driver")
RECIPES = ("resnet50_fp16", "resnet50_int8", "bert_large_fp16")


def executable(value, env=None):
    env = os.environ if env is None else env
    path = shutil.which(value, path=env.get("PATH", os.defpath))
    # Only the default driver name gets discovery. Explicit overrides never
    # silently select a different installation or recover from a broken one.
    if not path and value == DRIVER:
        for candidate in DRIVER_LOCATIONS:
            if Path(candidate).is_file() and os.access(candidate, os.X_OK):
                path = candidate
                break
    if not path:
        detail = (" Install/use the prepared MIGraphX image documented in docs/migraphx.md; "
                  "pip install '.[migraphx]' provides Python dependencies, not the native driver.") if value == DRIVER else ""
        raise ValueError(f"Prepared executable is missing: {value}.{detail}")
    return str(Path(path).resolve())


def shared_libraries(binary, env):
    with Path(binary).open("rb") as source:
        elf = source.read(4) == b"\x7fELF"
    if not sys.platform.startswith("linux") or not elf:
        return {"checked": False, "reason": "Linux ELF dependency check does not apply"}
    ldd = shutil.which("ldd", path=env.get("PATH", os.defpath))
    if not ldd and Path("/usr/bin/ldd").is_file():
        ldd = "/usr/bin/ldd"
    if not ldd:
        raise ValueError("ldd is required to check the native MIGraphX shared libraries")
    result = subprocess.run([ldd, binary], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace", timeout=15)
    missing = re.findall(r"^\s*(\S+)\s+=>\s+not found\s*$", result.stdout, re.MULTILINE)
    if missing:
        raise ValueError(f"Native libraries missing for {binary}: {', '.join(missing)}. "
                         "Use the matching MIGraphX/DTK runtime; see docs/migraphx.md")
    if result.returncode:
        raise ValueError(f"Cannot inspect native dependencies for {binary}: {result.stdout.strip()}")
    return {"checked": True, "ldd": str(Path(ldd).resolve()), "output": result.stdout}


def check(recipe="resnet50_fp16", binary=DRIVER, numactl="numactl", env=None):
    """Inspect dependencies without loading an ONNX model or launching perf."""
    env = dict(os.environ if env is None else env)
    result = {"schema_version": 1, "ready": False, "recipe": recipe, "errors": [],
              "scope": "dependencies only; no GPU/model performance or numerical validation",
              "python_dependencies": {name: importlib.util.find_spec(name) is not None
                                      for name in ("yaml", "psutil", "onnx")}}
    missing = [name for name, present in result["python_dependencies"].items() if not present]
    if missing:
        result["errors"].append("Missing Python dependencies: " + ", ".join(missing) + "; install '.[migraphx]' in the chosen Python environment")
    try:
        if recipe not in RECIPES:
            raise ValueError("Unknown MIGraphX recipe")
        result["binary"] = executable(binary, env)
        result["numactl"] = executable(numactl, env)
        result["libraries"] = shared_libraries(result["binary"], env)
        version = subprocess.run([result["binary"], "--version"], env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, errors="replace", timeout=15)
        if version.returncode:
            raise ValueError("MIGraphX version probe failed: " + version.stdout.strip())
        result["version"] = version.stdout.strip()
        loader_error = re.search(r"(?im)^(?:load (?:hsa )?library error|open hsa lib:.*error|open hydmilib:.*error).*$", version.stdout)
        if loader_error:
            raise ValueError("MIGraphX runtime loader failed: " + loader_error[0] +
                             "; check the matching DTK runtime and read-only host hyhal mount")
        help_result = subprocess.run([result["binary"], "perf", "--help"], env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, errors="replace", timeout=15)
        flags = set(re.findall(r"--[a-zA-Z0-9_-]+", help_result.stdout))
        required = {"--onnx", "--batch", "--gpu", "--int8" if recipe == "resnet50_int8" else "--fp16"}
        if recipe == "resnet50_fp16":
            required.add("--opt-compile")
        if recipe == "bert_large_fp16":
            required.add("--fill1")
        if help_result.returncode not in (0, 1) or required - flags:
            raise ValueError(f"MIGraphX perf help is incompatible (exit {help_result.returncode}); required flags missing: " + ", ".join(sorted(required - flags)))
        result["required_flags"] = sorted(required)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result["errors"].append(str(exc))
    result["ready"] = not result["errors"]
    return result
