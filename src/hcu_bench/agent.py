"""Standard-library-only worker, also shipped to Linux targets over SSH."""
import base64
import json
import os
import platform
import signal
import socket
import glob
import hashlib
import subprocess
import sys
import threading
import time
from pathlib import Path
from datetime import datetime

OUTPUT_LOCK = threading.Lock()
CANCEL = threading.Event()


def emit(kind, **fields):
    if kind == "log":
        fields["timestamp"] = datetime.now().astimezone().isoformat(timespec="milliseconds")
    with OUTPUT_LOCK:
        print(json.dumps({"kind": kind, **fields}, ensure_ascii=True), flush=True)


def listen():
    try:
        for line in sys.stdin:
            if line.strip() == "cancel":
                CANCEL.set()
                return
    finally:
        CANCEL.set()


def kill_group(proc):
    try:
        if os.name == "nt":
            if proc.poll() is None:
                killed = subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL, timeout=10, check=False)
                return killed.returncode == 0 or proc.poll() is not None
        else:
            os.killpg(proc.pid, signal.SIGTERM)
            time.sleep(0.15)
            os.killpg(proc.pid, signal.SIGKILL)
        return True
    except ProcessLookupError:
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def stage(command, cwd, env, timeout_s, duration_s=None, wrapped=False, ignore_cancel=False, capture=False):
    proc = subprocess.Popen(command, cwd=cwd, env={**os.environ, **env}, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1,
                            start_new_session=os.name != "nt",
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
    inner = []
    output = []
    reader_errors = []

    def read_output():
        try:
            for line in proc.stdout:
                text = line.rstrip("\r\n")
                if capture:
                    if len(output) >= 1000:
                        raise RuntimeError("Hook output exceeded the capture limit")
                    output.append(text)
                if wrapped:
                    try:
                        event = json.loads(text)
                    except (ValueError, TypeError):
                        emit("log", text=text)
                        continue
                    if event.get("kind") == "result":
                        inner.append(event)
                    elif event.get("kind") == "log":
                        emit("log", text=event.get("text", ""))
                    elif event.get("kind") == "environment":
                        emit("container_environment", data=event.get("data", {}))
                    elif event.get("kind", "").startswith("artifact_"):
                        emit(event["kind"], **{key: value for key, value in event.items() if key != "kind"})
                    else:
                        emit("log", text=text)
                else:
                    emit("log", text=text)
        except Exception as exc:
            reader_errors.append(str(exc))

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    started = time.monotonic()
    status = "completed"
    group_clean = True
    while proc.poll() is None:
        elapsed = time.monotonic() - started
        if not ignore_cancel and CANCEL.is_set():
            status = "cancelled"
        elif duration_s is not None and elapsed >= duration_s:
            status = "duration_reached"
        elif elapsed >= timeout_s:
            status = "timed_out"
        else:
            time.sleep(0.05)
            continue
        if wrapped:
            try:
                proc.stdin.write("cancel\n")
                proc.stdin.flush()
                proc.wait(timeout=8)
            except (OSError, subprocess.TimeoutExpired):
                group_clean = kill_group(proc)
        else:
            group_clean = kill_group(proc)
        break
    try:
        proc.wait(timeout=12)
    except subprocess.TimeoutExpired:
        kill_group(proc)
        return {"status": "failed", "returncode": None, "cleanup": "unconfirmed",
                "error": "Owned process did not exit after termination", "elapsed_s": time.monotonic() - started}
    # A test shell may have left children after its own exit; terminate this session only.
    if os.name != "nt":
        group_clean = kill_group(proc) and group_clean
    proc.stdin.close()
    reader.join(timeout=3)
    clean = group_clean and not reader.is_alive() and not reader_errors
    if wrapped:
        clean = clean and bool(inner) and inner[-1].get("cleanup") == "confirmed"
        if inner and status == "completed":
            status = inner[-1]["status"]
    if status == "completed" and proc.returncode != 0:
        status = "failed"
    if proc.returncode == 125:
        clean = False
    return {"status": status, "returncode": proc.returncode,
            "elapsed_s": time.monotonic() - started, "cleanup": "confirmed" if clean else "unconfirmed",
            **({"output": "\n".join(output)} if capture else {})}


def acquire_locks(root, resources):
    import hashlib
    handles = []
    directory = Path(root) / ".locks"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        for resource in sorted(set(resources)):
            name = hashlib.sha256(resource.encode()).hexdigest() + ".lock"
            handle = open(directory / name, "a+b")
            handles.append(handle)
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handles
    except Exception:
        for handle in handles:
            handle.close()
        raise RuntimeError("Resource is locked by another Bench task; no workload started")


def collect_artifacts(spec):
    remaining = int(spec.get("artifact_limit_mb", 100) * 1024 * 1024)
    seen = set()
    for pattern in spec.get("artifacts", []):
        if not os.path.isabs(pattern):
            pattern = os.path.join(spec["cwd"], pattern)
        matched = False
        for path in glob.iglob(pattern, recursive=True):
            if not os.path.isfile(path) or os.path.islink(path):
                continue
            matched = True
            if path in seen:
                continue
            seen.add(path)
            try:
                size = os.path.getsize(path)
                if size > remaining:
                    emit("artifact_warning", source=path, reason="Artifact transfer byte budget exceeded")
                    continue
                file_id = hashlib.sha256(path.encode()).hexdigest()[:20]
                emit("artifact_start", file_id=file_id, name=os.path.basename(path), source=path, size=size)
                digest = hashlib.sha256()
                sent = 0
                with open(path, "rb") as handle:
                    while True:
                        chunk = handle.read(min(65536, remaining + 1))
                        if not chunk:
                            break
                        if len(chunk) > remaining:
                            raise RuntimeError("Artifact grew beyond the transfer byte budget")
                        digest.update(chunk)
                        sent += len(chunk)
                        remaining -= len(chunk)
                        emit("artifact_chunk", file_id=file_id, data=base64.b64encode(chunk).decode())
                emit("artifact_end", file_id=file_id, size=sent, sha256=digest.hexdigest())
            except Exception as exc:
                emit("artifact_warning", source=path, reason=str(exc))
        if not matched:
            emit("artifact_warning", source=pattern, reason="Declared report path has no regular-file matches")


def run(spec):
    threading.Thread(target=listen, daemon=True).start()
    emit("environment", data={"hostname": socket.gethostname(), "platform": platform.platform(),
                              "python": sys.version, "executable": sys.executable,
                              "selected_gpus": spec.get("gpus", []), "simulated": spec.get("simulated", False)})
    work_dir = Path(spec["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    leases = []
    container = spec.get("container")
    start_attempted = False
    cleanup = "confirmed"
    result = {"status": "failed", "returncode": None, "elapsed_s": 0}
    try:
        leases = acquire_locks(spec["lock_root"], spec.get("resources", []))
        if CANCEL.is_set():
            result["status"] = "cancelled"
            return result
        if container:
            limit = container["hook_timeout_s"]
            if container["mode"] == "managed":
                exists = stage(["docker", "container", "ls", "--all", "--quiet", "--filter", "name=" + container["name"]],
                               str(work_dir), {}, limit, capture=True)
                if exists["cleanup"] != "confirmed":
                    cleanup = "unconfirmed"
                if exists["status"] != "completed" or exists["cleanup"] != "confirmed":
                    raise RuntimeError("Cannot establish whether the container name is unused; no start hook called")
                if exists.get("output", "").strip():
                    raise RuntimeError("Managed container name already exists; refusing to claim or stop it")
                # Set before the hook so partially-created containers are cleaned after start failure.
                start_attempted = True
                started = stage(container["start"], str(work_dir), {}, limit)
                if started["cleanup"] != "confirmed":
                    cleanup = "unconfirmed"
                if started["status"] != "completed" or started["cleanup"] != "confirmed":
                    raise RuntimeError("Container start hook failed: " + started["status"])
            checked = stage(container["check"], str(work_dir), {}, limit)
            if checked["cleanup"] != "confirmed":
                cleanup = "unconfirmed"
            if checked["status"] != "completed" or checked["cleanup"] != "confirmed":
                raise RuntimeError("Container check hook failed: " + checked["status"])
        result = stage(spec["argv"], spec["cwd"], spec.get("env", {}), spec["timeout_s"],
                       spec.get("duration_s"), spec.get("wrapped", False))
        cleanup = result["cleanup"]
    except Exception as exc:
        emit("log", text="[agent error] " + str(exc))
        result["error"] = str(exc)
    finally:
        if container and start_attempted:
            try:
                stopped = stage(container["stop"], str(work_dir), {}, container["hook_timeout_s"], ignore_cancel=True)
                still_running = stage(container["check"], str(work_dir), {}, container["hook_timeout_s"], ignore_cancel=True)
                if (stopped["status"] != "completed" or stopped["cleanup"] != "confirmed"
                        or still_running["returncode"] != 1 or still_running["cleanup"] != "confirmed"):
                    cleanup = "unconfirmed"
                    emit("log", text="[cleanup warning] owned container stop could not be confirmed")
            except Exception as exc:
                cleanup = "unconfirmed"
                emit("log", text="[cleanup warning] " + str(exc))
        for lease in leases:
            lease.close()
        result["cleanup"] = cleanup
        collect_artifacts(spec)
    return result


def main():
    try:
        # The local controller assigns its Windows job before sending this first line.
        if len(sys.argv) > 1:
            spec = json.loads(base64.b64decode(sys.argv[1]))
        else:
            spec = json.loads(sys.stdin.readline())
        result = run(spec)
        emit("result", **result)
        return 0 if result["status"] in ("completed", "duration_reached") else 1
    except Exception as exc:
        emit("result", status="failed", returncode=None, elapsed_s=0, cleanup="unconfirmed", error=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
