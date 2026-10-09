"""Small target-side helpers; no Torch/SGLang imports on the controller."""
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import threading
from pathlib import Path


class CleanupError(RuntimeError):
    pass


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def emit(value):
    print("BENCH_RESULT " + json.dumps(value, ensure_ascii=True, allow_nan=False), flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def versions(names):
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def cancellation():
    stopped = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stopped.set())
    return stopped


class NativeProcess:
    """Track owned descendants by process identity; never kill by program name."""

    def __init__(self, command, cwd, env, log_path, on_line=None):
        import psutil

        self.psutil = psutil
        self.log = Path(log_path).open("w", encoding="utf-8")
        self.stopping = threading.Event()
        self.errors = []
        self.owned = set()
        self.lock = threading.Lock()
        try:
            # Inherit the Bench worker's group as a fallback if this wrapper is killed.
            self.proc = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, encoding="utf-8", errors="replace", bufsize=1)
        except Exception:
            self.log.close()
            raise
        try:
            self.root = psutil.Process(self.proc.pid)
            self.owned.add(self.root)
        except psutil.NoSuchProcess:
            self.root = None

        def read():
            try:
                for line in self.proc.stdout:
                    self.log.write(line)
                    self.log.flush()
                    print(line.rstrip("\r\n"), flush=True)
                    if on_line:
                        on_line(line.rstrip("\r\n"))
            except Exception as exc:
                self.errors.append(str(exc))

        def track():
            while not self.stopping.wait(0.1):
                self.snapshot()

        self.reader = threading.Thread(target=read, daemon=True)
        self.tracker = threading.Thread(target=track, daemon=True)
        self.reader.start()
        self.tracker.start()

    def snapshot(self):
        if self.root is None:
            return
        try:
            children = self.root.children(recursive=True)
        except self.psutil.NoSuchProcess:
            return
        except self.psutil.Error as exc:
            self.errors.append(str(exc))
            return
        with self.lock:
            self.owned.update(children)

    def wait(self, stopped):
        while self.proc.poll() is None and not stopped.wait(0.1):
            if self.errors:
                raise RuntimeError("Native output/tracking failed: " + self.errors[0])
        return self.proc.poll()

    def close(self):
        self.snapshot()
        self.stopping.set()
        self.tracker.join(timeout=2)
        with self.lock:
            owned = list(self.owned)
        owned.sort(key=lambda process: process == self.root)
        for process in owned:
            try:
                process.terminate()
            except self.psutil.NoSuchProcess:
                pass
            except self.psutil.Error as exc:
                self.errors.append(str(exc))
        _, alive = self.psutil.wait_procs(owned, timeout=3)
        for process in alive:
            try:
                process.kill()
            except self.psutil.NoSuchProcess:
                pass
            except self.psutil.Error as exc:
                self.errors.append(str(exc))
        _, alive = self.psutil.wait_procs(alive, timeout=2)
        self.proc.poll()
        self.reader.join(timeout=2)
        if not self.reader.is_alive():
            self.proc.stdout.close()
            self.log.close()
        live = []
        for process in alive:
            try:
                if process.is_running() and process.status() not in (self.psutil.STATUS_ZOMBIE, self.psutil.STATUS_DEAD):
                    live.append(process)
            except self.psutil.NoSuchProcess:
                pass
        if live or self.reader.is_alive() or self.errors:
            raise CleanupError("Owned native process cleanup/output could not be confirmed")


def cleanup_failure(output_dir, reason):
    write_json(Path(output_dir) / "cleanup.json", {"cleanup": "unconfirmed", "reason": str(reason)})
    print("[cleanup warning] " + str(reason), flush=True)
    # The outer agent must not interpret a wrapper exit as confirmed cleanup.
    return 125
