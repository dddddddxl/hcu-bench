"""Build bundled native tools in the target environment, without downloads/install."""
import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from testpacks.native_common import CleanupError, NativeProcess, cancellation, cleanup_failure, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {"rccl": "rccl-tests", "pcie": "rocm-bandwidth-test", "memory": "hcu-stream", "mooncake": "mooncake"}


def tree_hash(directory):
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*"), key=lambda item: item.relative_to(directory).as_posix()):
        if path.is_symlink():
            raise ValueError("Bundled source must not contain symlinks: " + str(path))
        if path.is_file():
            digest.update(path.relative_to(directory).as_posix().encode())
            digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def executable(value):
    found = shutil.which(str(value))
    if not found:
        raise ValueError("Required executable not found: " + str(value))
    return str(Path(found).resolve())


def fingerprint_file(path):
    path = Path(path)
    if not path.is_file():
        return None
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def settings_for(suite, settings):
    if sys.platform != "linux":
        raise ValueError("Native tool preparation requires a Linux execution environment")
    dtk = Path(settings.get("dtk_root") or "/opt/dtk").resolve()
    mpi = Path(settings.get("mpi_home") or "/opt/mpi").resolve()
    if not dtk.is_dir():
        raise ValueError("DTK is not present; set build.dtk_root to the image's existing DTK")
    arch = settings.get("arch", "auto")
    if suite in ("memory", "rccl"):
        if arch in ("", "auto"):
            probe = subprocess.run([executable(dtk / "bin/rocminfo")], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, timeout=30, check=True)
            arches = set(re.findall(r"\bName:\s+(gfx[0-9a-f]+)\b", probe.stdout))
            if len(arches) != 1:
                raise ValueError("Cannot identify one HCU architecture; set build.arch explicitly")
            arch = arches.pop()
        if not re.fullmatch(r"gfx[0-9a-f]+", arch):
            raise ValueError("build.arch must be a gfx architecture name")
    else:
        arch = "host"
    arch_num = settings.get("arch_num")
    if suite == "memory" and arch_num is None:
        if arch == "gfx92a":
            arch_num = 9210
        elif arch[3:].isdigit():
            arch_num = int(arch[3:])
        else:
            raise ValueError("Set build.arch_num for this hcu-stream architecture")
    default_compiler = dtk / "bin/hipcc" if suite == "rccl" else dtk / "bin/aicc" if suite == "memory" else "c++"
    compiler = executable(settings.get("compiler") or default_compiler)
    jobs = settings.get("jobs", 2)
    if type(jobs) is not int or jobs < 1:
        raise ValueError("build.jobs must be a positive integer")
    compiler_version = subprocess.check_output([compiler, "--version"], stderr=subprocess.STDOUT, text=True, timeout=30)
    files = [dtk / ".dtk_version", Path("/etc/os-release")]
    libraries = []
    for directory in (dtk / "lib", dtk / "rccl/lib", mpi / "lib"):
        libraries += [fingerprint_file(path) for path in sorted(directory.glob("*.so*"))]
    return {"dtk_root": str(dtk), "mpi_home": str(mpi), "arch": arch, "arch_num": arch_num,
            "compiler": compiler, "compiler_version": compiler_version, "compiler_file": fingerprint_file(compiler),
            "jobs": jobs, "platform": platform.platform(), "libc": list(platform.libc_ver()),
            "runtime": {str(path): sha256(path) for path in files if path.is_file()},
            "libraries": libraries, "recipe_sha256": sha256(__file__)}


def runtime_env(settings):
    dtk, mpi = settings["dtk_root"], settings["mpi_home"]
    return {"ROCM_PATH": dtk,
            "PATH": os.pathsep.join([mpi + "/bin", dtk + "/bin", dtk + "/hip/bin/hipify", os.environ.get("PATH", "")]),
            "LD_LIBRARY_PATH": os.pathsep.join([mpi + "/lib", dtk + "/rccl/lib", dtk + "/lib", os.environ.get("LD_LIBRARY_PATH", "")])}


def commands_for(suite, source, build, settings):
    dtk, mpi = settings["dtk_root"], settings["mpi_home"]
    compiler, arch, jobs = settings["compiler"], settings["arch"], str(settings["jobs"])
    if suite == "rccl":
        flags = shlex.join(["-std=c++14", "-O3", "--offload-arch=" + arch,
                            "--gpu-max-threads-per-block=512", "-DMPI_SUPPORT",
                            "-I" + mpi + "/include", "-I" + dtk + "/rccl/include", "-I" + dtk + "/include/hip"])
        binary = build / "all_reduce_perf"
        return [["make", "-C", str(source / "src"), str(binary), "BUILDDIR=" + str(build),
                 "ROCM_PATH=" + dtk, "MPI=1", "MPI_HOME=" + mpi, "NCCL_HOME=" + dtk + "/rccl",
                 "CUSTOM_RCCL_LIB=" + dtk + "/rccl/lib", "HIPCC=" + compiler, "HIPCUFLAGS=" + flags,
                 "GIT_REV=ae3e635", "GIT_BRANCH=vendored", "GIT_DIFF=", "-j" + jobs]], binary
    if suite == "memory":
        return [["make", "-C", str(source), "HIPCC=" + compiler, "ROCM_PATH=" + dtk,
                 "OFFLOAD_ARCH=--offload-arch=" + arch, "TARGET_NUM=" + str(settings["arch_num"]), "-j" + jobs]], source / "hcu-stream"
    common = ["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
              "-DCMAKE_CXX_COMPILER=" + compiler, "-DCMAKE_PREFIX_PATH=" + dtk]
    if suite == "pcie":
        return [common, ["cmake", "--build", str(build), "--target", "rocm-bandwidth-test", "-j", jobs]], build / "rocm-bandwidth-test"
    common[2] = str(source / "mooncake-transfer-engine")
    common += ["-DUSE_HYGON=ON", "-DDTK_ROOT=" + dtk, "-DBUILD_UNIT_TESTS=OFF", "-DBUILD_EXAMPLES=ON",
               "-DBUILD_BENCHMARK=OFF", "-DUSE_TENT=OFF", "-DUSE_ETCD=OFF", "-DUSE_HTTP=OFF",
               "-DUSE_REDIS=OFF", "-DWITH_METRICS=OFF", "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
               "-DFETCHCONTENT_SOURCE_DIR_YALANTINGLIBS=" + str(source / "extern/yalantinglibs")]
    return [common, ["cmake", "--build", str(build), "--target", "transfer_engine_bench", "-j", jobs]], build / "example/transfer_engine_bench"


def cache_record(directory, identity):
    try:
        record = json.loads((directory / "ready.json").read_text())
        binary = Path(record["binary"])
        if record["identity"] == identity and binary.is_file() and os.access(binary, os.X_OK) and sha256(binary) == record["binary_sha256"]:
            return record
    except (OSError, ValueError, KeyError):
        pass
    return None


def prepare(suite, binary, output, settings, stopped):
    if binary != "auto":
        # Explicit overrides never fall back silently to a different executable.
        path = executable(binary)
        return {"binary": path, "binary_sha256": sha256(path), "source": "explicit", "env": {}}
    source = ROOT / "vendor" / SOURCES[suite]
    manifest = json.loads((ROOT / "vendor/sources.json").read_text())
    expected = manifest[SOURCES[suite]]["tree_sha256"]
    if tree_hash(source) != expected:
        raise ValueError("Bundled source checksum mismatch: " + str(source))
    build_settings = settings_for(suite, settings)
    identity = {"suite": suite, "source_sha256": expected, "settings": build_settings}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    cache = Path(settings.get("cache_dir") or ROOT / ".cache/native").expanduser().resolve()
    directory = cache / suite / key
    directory.mkdir(parents=True, exist_ok=True)
    import fcntl

    with (directory / "build.lock").open("a") as lock:
        while not stopped.is_set():
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                stopped.wait(0.2)
        if stopped.is_set():
            raise InterruptedError("Preparation cancelled while waiting for build lock")
        record = cache_record(directory, identity)
        if record:
            print("[prepare] cache hit: " + record["binary"], flush=True)
            return {**record, "cache_hit": True, "env": runtime_env(build_settings)}
        # Each attempt keeps its own directory: failed/partial builds cannot become cache hits.
        import tempfile
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=directory))
        copied = attempt / "source"
        shutil.copytree(source, copied)
        commands, target = commands_for(suite, copied, attempt / "build", build_settings)
        env = {**os.environ, **runtime_env(build_settings)}
        write_json(output / "prepare-commands.json", {"identity": identity, "commands": commands})
        for index, command in enumerate(commands):
            if stopped.is_set():
                raise InterruptedError("Preparation cancelled")
            executable(command[0])
            print("[prepare] " + shlex.join(command), flush=True)
            process = NativeProcess(command, str(copied), env, output / ("prepare-%02d.log" % index))
            try:
                code = process.wait(stopped)
                if stopped.is_set():
                    raise InterruptedError("Preparation cancelled")
                if code != 0:
                    raise RuntimeError("Native build failed (exit %s); see prepare-%02d.log" % (code, index))
            finally:
                process.close()
        if not target.is_file() or not os.access(target, os.X_OK):
            raise RuntimeError("Build did not produce the expected executable: " + str(target))
        record = {"identity": identity, "binary": str(target), "binary_sha256": sha256(target),
                  "source": "bundled", "source_version": manifest[SOURCES[suite]]["version"], "cache_hit": False}
        write_json(directory / "ready.json", record)
        return {**record, "env": runtime_env(build_settings)}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("suite", choices=SOURCES)
    cli.add_argument("--binary", default="auto")
    cli.add_argument("--output-dir", required=True)
    cli.add_argument("--rendezvous", default="")
    cli.add_argument("--node-rank", type=int, default=0)
    cli.add_argument("--nnodes", type=int, default=1)
    args = cli.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stopped = cancellation()
    rendezvous = None
    claimed = False
    try:
        if args.rendezvous:
            if not 0 <= args.node_rank < args.nnodes:
                raise ValueError("node_rank must be within nnodes")
            rendezvous = Path(args.rendezvous)
            rendezvous.mkdir(parents=True, exist_ok=True)
            # Run/case IDs isolate this preparation from earlier tests.
            with (rendezvous / ("rank-%d.claim" % args.node_rank)).open("x"):
                pass
            claimed = True
        record = prepare(args.suite, args.binary, output, json.loads(os.environ.get("HCU_BENCH_BUILD_CONFIG", "{}")), stopped)
        write_json(output / "prepared.json", record)
        if rendezvous:
            write_json(rendezvous / ("rank-%d.ready.json" % args.node_rank), {"ready": True})
            while not stopped.is_set():
                if list(rendezvous.glob("*.error.json")):
                    raise RuntimeError("A peer failed native preparation; no Mooncake workload started")
                if all((rendezvous / ("rank-%d.ready.json" % rank)).is_file() for rank in range(args.nnodes)):
                    break
                stopped.wait(0.2)
            if stopped.is_set():
                raise InterruptedError("Preparation barrier cancelled")
        return 0
    except CleanupError as exc:
        if claimed:
            write_json(rendezvous / ("rank-%d.error.json" % args.node_rank), {"error": str(exc)})
        return cleanup_failure(output, exc)
    except Exception as exc:
        write_json(output / "prepare-error.json", {"phase": "prepare", "error": str(exc)})
        if claimed:
            write_json(rendezvous / ("rank-%d.error.json" % args.node_rank), {"error": str(exc)})
        print("[prepare error] " + str(exc), flush=True)
        return 130 if isinstance(exc, InterruptedError) else 1


if __name__ == "__main__":
    raise SystemExit(main())
