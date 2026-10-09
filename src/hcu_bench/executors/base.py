import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Protocol

from ..models import BenchError

AGENT_PATH = Path(__file__).resolve().parents[1] / "agent.py"


class Executor(Protocol):
    def execute(self, spec: dict, cancelled, on_event) -> dict: ...


class WindowsJob:
    def __init__(self, proc):
        import ctypes
        from ctypes import wintypes

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel = kernel
        self.handle = kernel.CreateJobObjectW(None, None)
        info = Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000  # Kill only this job's descendants on close.
        if not self.handle or not kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)) or not kernel.AssignProcessToJobObject(self.handle, wintypes.HANDLE(int(proc._handle))):
            self.close()
            raise BenchError("Cannot create an isolated Windows process job; refusing unsafe execution")

    def close(self):
        if getattr(self, "handle", None):
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class AgentExecutor:
    def __init__(self, node):
        self.node = node

    def launch_argv(self, spec):
        raise NotImplementedError

    def execute(self, spec, cancelled, on_event):
        process = subprocess.Popen(self.launch_argv(spec), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                   start_new_session=os.name != "nt",
                                   creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
        job = None
        final = []
        reader_errors = []

        def read():
            try:
                for line in process.stdout:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        event = {"kind": "log", "text": line.rstrip("\r\n")}
                    if not isinstance(event, dict):
                        event = {"kind": "log", "text": line.rstrip("\r\n")}
                    if event.get("kind") == "result":
                        final.append(event)
                    on_event(event)
            except Exception as exc:
                reader_errors.append(str(exc))

        reader = threading.Thread(target=read, daemon=True)
        try:
            if os.name == "nt":
                job = WindowsJob(process)
            # Keep workload parameters off SSH argv and below Windows command-line limits.
            process.stdin.write(json.dumps(spec) + "\n")
            process.stdin.flush()
            reader.start()
            started = time.monotonic()
            sent_cancel = False
            hook_budget = 5 * spec.get("container", {}).get("hook_timeout_s", 0) if spec.get("container") else 0
            hard_limit = spec["timeout_s"] + hook_budget + self.node.get("connect_timeout_s", 10) + 30 + (120 if spec.get("artifacts") or spec.get("wrapped") else 0)
            while process.poll() is None:
                if cancelled() and not sent_cancel:
                    try:
                        process.stdin.write("cancel\n")
                        process.stdin.flush()
                    except OSError:
                        pass
                    sent_cancel = True
                if time.monotonic() - started > hard_limit:
                    process.kill()
                    process.wait(timeout=10)
                    break
                time.sleep(0.05)
            reader.join(timeout=5)
            expected_code = 0 if final and final[-1].get("status") in ("completed", "duration_reached") else 1
            if reader.is_alive() or reader_errors or not final or process.returncode != expected_code:
                return {"status": "failed", "returncode": process.returncode, "cleanup": "unconfirmed",
                        "error": "Transport ended without a confirmed worker result", "elapsed_s": time.monotonic() - started}
            return final[-1]
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
            if job:
                job.close()
            process.stdin.close()
            process.stdout.close()
