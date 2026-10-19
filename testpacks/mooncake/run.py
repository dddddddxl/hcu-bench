"""Run native RDMA/VRAM pressure tests with fresh, shared-directory rendezvous."""
import argparse
import hashlib
import importlib.util
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from testpacks.native_common import CleanupError, NativeProcess, cancellation, cleanup_failure, emit, sha256, versions, write_json, prepared_binary

ENDPOINT = re.compile(r"listening on\s+([0-9.]+):(\d+)\b")
COMPLETION = re.compile(r"Test completed:\s*duration\s+([^,\s]+),\s*batch count\s+(\d+),\s*throughput\s+([^\s]+)\s+(GiB/s|GB/s)\b")
NATIVE_ERROR = re.compile(r"^(?:[EF]\d{4}\s|.*\b(?:ERROR|FATAL)\b)")


def parser():
    cli = argparse.ArgumentParser(description="Mooncake 0.3.12+ P2PHANDSHAKE RDMA/VRAM pressure test")
    cli.add_argument("--binary", default="transfer_engine_bench")
    cli.add_argument("--output-dir", required=True)
    cli.add_argument("--shared-dir", required=True)
    cli.add_argument("--run-id", required=True)
    cli.add_argument("--case-id", required=True)
    cli.add_argument("--node-rank", type=int, required=True)
    cli.add_argument("--nnodes", type=int, required=True)
    cli.add_argument("--addresses", required=True, help="Comma-separated IPv4 addresses in test nodes order")
    cli.add_argument("--topology", choices=("pair", "fan_in", "pairs", "ring"), default="pair")
    cli.add_argument("--duration", type=int, default=180, help="Native initiator measurement seconds")
    cli.add_argument("--startup-timeout", type=float, default=90)
    cli.add_argument("--completion-grace", type=float, default=120)
    cli.add_argument("--hcu-id", "--gpu-id", dest="gpu_id", type=int, default=0, help="Visible HCU index; -1 explicitly uses all visible HCUs")
    cli.add_argument("--buffer-size", type=int, default=1073741824)
    cli.add_argument("--threads", type=int, default=12)
    cli.add_argument("--batch-size", type=int, default=128)
    cli.add_argument("--block-size", type=int, default=65536)
    cli.add_argument("--operation", choices=("read", "write"), default="read")
    cli.add_argument("--expected-version", default="")
    cli.add_argument("--min-bandwidth-gbps", default="", help="Optional decimal GB/s threshold per initiator")
    cli.add_argument("--check-only", action="store_true")
    return cli


def routes_for(topology, count):
    """Return initiator -> target node ranks; every configured node participates."""
    if count < 2:
        raise ValueError("Mooncake requires at least two node ranks")
    if topology == "pair":
        if count != 2:
            raise ValueError("pair requires exactly two nodes (target first, initiator second)")
        return {1: 0}
    if topology == "fan_in":
        return {rank: 0 for rank in range(1, count)}
    if topology == "pairs":
        if count % 2:
            raise ValueError("pairs requires an even number of nodes")
        return {rank: rank - 1 for rank in range(1, count, 2)}
    if topology == "ring":
        return {rank: (rank + 1) % count for rank in range(count)}
    raise ValueError("Unknown topology")


def threshold_for(value):
    if value == "":
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("min_bandwidth_gbps must be positive and finite, or empty")
    return number


def validate_args(args):
    routes = routes_for(args.topology, args.nnodes)
    if not 0 <= args.node_rank < args.nnodes:
        raise ValueError("Invalid node rank")
    addresses = args.addresses.split(",")
    if len(addresses) != args.nnodes or len(set(addresses)) != args.nnodes:
        raise ValueError("Supply one distinct IPv4 address per node, in test nodes order")
    for address in addresses:
        parsed = ipaddress.ip_address(address)
        if parsed.version != 4 or parsed.is_loopback or parsed.is_unspecified or parsed.is_multicast:
            raise ValueError("Use explicit non-loopback IPv4 addresses reachable by peers")
    for name in ("run_id", "case_id"):
        value = getattr(args, name)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,160}", value):
            raise ValueError(name + " must be a safe run/case identifier")
    if not Path(args.shared_dir).is_absolute():
        raise ValueError("shared_dir must be an explicit absolute path shared by all nodes")
    for name in ("duration", "startup_timeout", "completion_grace", "buffer_size", "threads", "batch_size", "block_size"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(name + " must be positive and finite")
    if args.gpu_id < -1 or args.block_size > args.buffer_size:
        raise ValueError("Invalid HCU index or block_size exceeds buffer_size")
    required_buffer = args.block_size * args.batch_size * args.threads
    if args.buffer_size < required_buffer:
        raise ValueError(f"buffer_size must be >= block_size * batch_size * threads = {required_buffer} bytes; "
                         "refusing native buffer auto-expansion")
    threshold_for(args.min_bandwidth_gbps)
    return addresses, routes


def parse_completion(line):
    match = COMPLETION.search(line)
    if not match:
        if "Test completed:" in line:
            raise ValueError("Unrecognised native completion line; cannot interpret throughput units")
        return None
    duration, count, throughput, unit = match.groups()
    duration, throughput, count = float(duration), float(throughput), int(count)
    if not all(math.isfinite(v) and v > 0 for v in (duration, throughput)) or count <= 0:
        raise ValueError("Native completion requires positive finite duration/throughput and batch count")
    return {"duration_s": duration, "batch_count": count, "throughput": throughput, "unit": unit,
            "throughput_gbps": throughput * (1.073741824 if unit == "GiB/s" else 1.0)}


def preflight(args):
    validate_args(args)
    if importlib.util.find_spec("psutil") is None:
        raise RuntimeError("Target dependency psutil is missing")
    import psutil
    packages = versions(["mooncake-transfer-engine", "psutil"])
    prepared = getattr(args, "prepared", {})
    if prepared.get("source") == "bundled":
        # The native executable is built from the pinned source, not the installed wheel.
        packages["mooncake-transfer-engine"] = prepared["source_version"]
    version = packages["mooncake-transfer-engine"]
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version or "")
    if not match or tuple(map(int, match.groups())) < (0, 3, 12):
        raise ValueError("This pack requires installed mooncake-transfer-engine 0.3.12+; older P2PHANDSHAKE builds are outside its contract")
    if args.expected_version and version != args.expected_version:
        raise ValueError("Mooncake distribution does not match expected_version")
    binary = shutil.which(args.binary)
    if not binary:
        raise ValueError("transfer_engine_bench binary is unavailable: " + args.binary)
    help_result = subprocess.run([binary, "--helpfull"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace", timeout=10)
    supported = set(re.findall(r"^\s*-(\w+)\s+\(", help_result.stdout, re.MULTILINE))
    required = {"mode", "protocol", "metadata_server", "local_server_name", "segment_id", "gpu_id", "use_vram",
                "auto_discovery", "duration", "buffer_size", "threads", "batch_size", "block_size", "operation"}
    if help_result.returncode not in (0, 1) or not required <= supported:
        raise ValueError("Native binary lacks the required RDMA/VRAM flags: " + ",".join(sorted(required - supported)))
    visible = os.environ.get("HIP_VISIBLE_DEVICES", "").split(",")
    if any(not re.fullmatch(r"\d+", item) for item in visible) or len(set(visible)) != len(visible):
        raise ValueError("Set explicit, distinct HIP_VISIBLE_DEVICES for the allowed physical HCUs")
    if args.gpu_id >= len(visible):
        raise ValueError("hcu_id is outside the explicitly visible HCUs")
    nics = os.environ.get("MC_TE_FILTERS", "").split(",")
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", nic) for nic in nics) or len(set(nics)) != len(nics):
        raise ValueError("Set explicit distinct RDMA NIC names in MC_TE_FILTERS")
    nic_state = {}
    for nic in nics:
        port = Path("/sys/class/infiniband") / nic / "ports" / "1"
        state = (port / "state").read_text().strip()
        physical = (port / "phys_state").read_text().strip()
        if not state.startswith("4:") or not physical.startswith("5:"):
            raise ValueError(f"RDMA NIC {nic} is not ACTIVE/LINK_UP: {state}, {physical}")
        nic_state[nic] = {"state": state, "physical_state": physical, "rate": (port / "rate").read_text().strip()}
    for process in psutil.process_iter(["name", "cmdline"]):
        try:
            name = process.info["name"] or ""
            first = (process.info["cmdline"] or [""])[0]
            if name.startswith("transfer_engine_ben") or Path(first).name == Path(binary).name:
                raise RuntimeError("An existing transfer_engine_bench process is present; do not stop other users' workloads")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {"packages": packages, "binary": binary, "binary_sha256": sha256(binary),
            "wrapper_sha256": sha256(__file__), "nic_state": nic_state,
            "supported_flags": sorted(supported), "help_sha256": hashlib.sha256(help_result.stdout.encode()).hexdigest(),
            "communication_env": {key: value for key, value in os.environ.items()
                                  if key.startswith(("MC_", "MOONCAKE_", "HIP_", "ROCR_"))},
            "visible_gpu_count": len(visible), "correctness": "not_checked"}


def native_command(args, role, address, segment=None):
    command = [args.binary, "--mode=" + role, "--auto_discovery", "--protocol=rdma",
               "--metadata_server=P2PHANDSHAKE", "--use_vram=true", f"--gpu_id={args.gpu_id}",
               "--local_server_name=" + address, f"--buffer_size={args.buffer_size}",
               f"--threads={args.threads}", f"--batch_size={args.batch_size}", f"--block_size={args.block_size}",
               f"--duration={args.duration}", "--operation=" + args.operation]
    if segment is not None:
        command.append("--segment_id=" + segment)
    return command


def stop_target(process, output):
    """Confirm that a passive target stopped because of our requested shutdown."""
    before = process.proc.poll()
    try:
        process.close()
    finally:
        after = process.proc.poll()
        actions = list(process.root_termination_actions)
        write_json(output / "target_exit.json", {"before_shutdown_returncode": before, "returncode": after,
                   "owned_termination_actions": actions})
    allowed = {0}
    if os.name == "nt":
        if actions:
            allowed.add(int(signal.SIGTERM))  # psutil's Windows TerminateProcess status.
    else:
        if "terminate" in actions:
            allowed.add(-signal.SIGTERM)
        if "kill" in actions:
            allowed.add(-signal.SIGKILL)
    if before is not None or not actions or after not in allowed:
        raise RuntimeError(f"Native target shutdown failed: before={before}, exit={after}, owned_actions={actions}")


class Rendezvous:
    """Atomic records on a deliberately configured shared filesystem, never old fixed ports."""
    def __init__(self, args, signature, stopped):
        self.args, self.signature, self.stopped = args, signature, stopped
        self.root = Path(args.shared_dir) / args.run_id / args.case_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = {"run_id": args.run_id, "case_id": args.case_id, "signature": signature}
        with self.path("join", args.node_rank).open("x", encoding="utf-8") as handle:
            json.dump({**self.identity, "rank": args.node_rank}, handle)

    def path(self, kind, rank):
        return self.root / f"{kind}-{rank}.json"

    def publish(self, kind, rank, **data):
        path = self.path(kind, rank)
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        temporary.write_text(json.dumps({**self.identity, "rank": rank, **data}, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)

    def read(self, kind, rank):
        try:
            value = json.loads(self.path(kind, rank).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            # Join is exclusively created; another node can briefly see its incomplete write.
            if kind == "join":
                return None
            raise
        if not isinstance(value, dict) or any(value.get(key) != item for key, item in self.identity.items()) or value.get("rank") != rank:
            raise ValueError("Shared rendezvous identity/configuration mismatch")
        return value

    def check(self):
        if self.stopped.is_set():
            raise InterruptedError("Mooncake test cancelled")
        for rank in range(self.args.nnodes):
            failure = self.read("failed", rank)
            if failure:
                raise RuntimeError(f"Peer {rank} failed: {failure.get('reason')}")

    def wait(self, predicate, timeout, health=lambda: None):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.check()
            health()
            value = predicate()
            if value:
                return value
            self.stopped.wait(0.1)
        raise TimeoutError("Mooncake rendezvous/completion deadline exceeded")


def run(args):
    addresses, routes = validate_args(args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prepared_binary(args)
    evidence = preflight(args)
    write_json(output / "environment.json", evidence)
    if args.check_only:
        print("Mooncake preflight OK; no native workload or shared rendezvous started", flush=True)
        return 0
    args.binary = evidence["binary"]
    evidence["native_parameters"] = {key: getattr(args, key) for key in ("addresses", "topology", "duration", "gpu_id",
                                    "buffer_size", "threads", "batch_size", "block_size", "operation")}
    evidence["applied_env"] = {"MC_ENABLE_DEST_DEVICE_AFFINITY": "1", "VLLM_HOST_IP": addresses[args.node_rank],
                               "initiator_only": {"MOONCAKE_DISABLE_HIP_DMABUF": "0"}}
    write_json(output / "environment.json", evidence)
    stopped = cancellation()
    definition = {key: getattr(args, key) for key in ("addresses", "topology", "duration", "startup_timeout", "completion_grace",
                  "gpu_id", "buffer_size", "threads", "batch_size", "block_size", "operation", "min_bandwidth_gbps")}
    definition["version"] = evidence["packages"]["mooncake-transfer-engine"]
    definition["binary_sha256"] = evidence.get("binary_sha256")
    definition["wrapper_sha256"] = evidence.get("wrapper_sha256")
    signature = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
    rendezvous = Rendezvous(args, signature, stopped)
    target = initiator = None
    errors, completions, endpoints = [], [], []
    incoming = [rank for rank, destination in routes.items() if destination == args.node_rank]
    commands = []

    def health():
        if errors:
            raise RuntimeError(errors[0])
        for role, process in (("target", target), ("initiator", initiator)):
            if process and process.errors:
                raise RuntimeError("Native output/tracking error: " + process.errors[0])
            if role == "target" and process and process.proc.poll() is not None:
                raise RuntimeError("Native target exited before its initiators completed")

    def target_line(line):
        if NATIVE_ERROR.search(line):
            errors.append("Native target error: " + line)
        match = ENDPOINT.search(line)
        if match:
            ip, port = match.groups()
            if ip not in (addresses[args.node_rank], "0.0.0.0") or not 1 <= int(port) <= 65535:
                errors.append("Native target advertised an unexpected endpoint")
            else:
                endpoints.append(f"{addresses[args.node_rank]}:{int(port)}")

    def initiator_line(line):
        if NATIVE_ERROR.search(line):
            errors.append("Native initiator error: " + line)
        try:
            sample = parse_completion(line)
            if sample:
                completions.append(sample)
        except ValueError as exc:
            errors.append(str(exc))

    def launch(role, segment=None):
        command = native_command(args, role, addresses[args.node_rank], segment)
        commands.append({"role": role, "argv": command})
        write_json(output / "commands.json", commands)
        env = {**os.environ, "MC_ENABLE_DEST_DEVICE_AFFINITY": "1", "VLLM_HOST_IP": addresses[args.node_rank]}
        if role == "initiator":
            env["MOONCAKE_DISABLE_HIP_DMABUF"] = "0"
        return NativeProcess(command, str(output), env, output / (role + ".log"), target_line if role == "target" else initiator_line)

    try:
        rendezvous.wait(lambda: all(rendezvous.read("join", rank) for rank in range(args.nnodes)), args.startup_timeout)
        if incoming:
            target = launch("target")
            endpoint = rendezvous.wait(lambda: endpoints and endpoints[0], args.startup_timeout, health)
            rendezvous.publish("endpoint", args.node_rank, endpoint=endpoint)
            write_json(output / "target_endpoint.json", {"endpoint": endpoint})
        if args.node_rank in routes:
            destination = routes[args.node_rank]
            record = rendezvous.wait(lambda: rendezvous.read("endpoint", destination), args.startup_timeout, health)
            endpoint = record.get("endpoint", "")
            if not re.fullmatch(re.escape(addresses[destination]) + r":\d+", endpoint) or not 1 <= int(endpoint.rsplit(":", 1)[1]) <= 65535:
                raise ValueError("Peer endpoint does not match the configured target address")
            initiator = launch("initiator", endpoint)
            deadline = time.monotonic() + args.duration + args.completion_grace
            while initiator.proc.poll() is None:
                rendezvous.check()
                health()
                if time.monotonic() >= deadline:
                    raise TimeoutError("Native initiator exceeded duration plus completion_grace")
                stopped.wait(0.1)
            code = initiator.proc.returncode
            completed_process = initiator
            initiator = None
            completed_process.close()  # Drain output and confirm cleanup before publishing completion.
            health()
            if code != 0 or len(completions) != 1:
                raise RuntimeError(f"Native initiator requires exit 0 and exactly one final result (exit={code}, results={len(completions)})")
            measurement = completions[0]
            if measurement["duration_s"] + max(0.1, args.duration * 0.01) < args.duration:
                raise ValueError("Native measurement ended before the requested duration")
            threshold = threshold_for(args.min_bandwidth_gbps)
            judgement = "report_only" if threshold is None else "passed" if measurement["throughput_gbps"] >= threshold else "failed"
            sample = {"measurement_status": "measured", "correctness": "not_checked", "sample_id": 0,
                      "role": "initiator", "target_rank": destination, "target_endpoint": endpoint,
                      "topology": args.topology, "performance_judgement": judgement, "min_bandwidth_gbps": threshold,
                      "timing_method": "native_test_completed", "metrics": [
                          {"name": "transfer_throughput", "value": measurement["throughput"], "unit": measurement["unit"], "statistic": "reported"},
                          {"name": "transfer_duration", "value": measurement["duration_s"], "unit": "s", "statistic": "total"},
                          {"name": "transfer_batch_count", "value": measurement["batch_count"], "unit": "batches", "statistic": "count"}]}
            write_json(output / "measurement.json", sample)
            emit(sample)
            if judgement == "failed":
                raise RuntimeError("Measured throughput is below the configured min_bandwidth_gbps")
            rendezvous.publish("done", args.node_rank, status="completed")
        if incoming:
            rendezvous.wait(lambda: all(rendezvous.read("done", rank) for rank in incoming),
                            args.startup_timeout + args.duration + args.completion_grace, health)
            completed_process = target
            target = None
            stop_target(completed_process, output)
            health()
            rendezvous.publish("target_done", args.node_rank, status="completed")
        # Initiator throughput alone does not prove the target survived teardown.
        # All ranks must observe successful target cleanup before claiming completion.
        rendezvous.wait(lambda: all(rendezvous.read("target_done", rank) for rank in set(routes.values())),
                        args.startup_timeout + args.completion_grace, health)
        if args.node_rank not in routes:
            emit({"measurement_status": "skipped", "correctness": "not_checked", "metrics": [], "role": "target",
                  "reason": "Passive target coordination completed; throughput is measured by initiators"})
        write_json(output / "summary.json", {"status": "completed", "topology": args.topology,
                   "node_rank": args.node_rank, "initiator": args.node_rank in routes,
                   "incoming_initiators": incoming, "correctness": "not_checked"})
        return 0
    except Exception as exc:
        rendezvous.publish("failed", args.node_rank, reason=str(exc))
        raise
    finally:
        cleanup_errors = []
        for process in (initiator, target):
            if process is not None:
                try:
                    process.close()
                except CleanupError as exc:
                    cleanup_errors.append(str(exc))
        write_json(output / "cleanup.json", {"cleanup": "unconfirmed" if cleanup_errors else "confirmed", "errors": cleanup_errors})
        if cleanup_errors:
            raise CleanupError("; ".join(cleanup_errors))


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return run(args)
    except CleanupError as exc:
        return cleanup_failure(args.output_dir, exc)
    except Exception as exc:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "error.json", {"status": "failed", "error": str(exc)})
        print("[mooncake error] " + str(exc), flush=True)
        return 130 if isinstance(exc, InterruptedError) else 1


if __name__ == "__main__":
    raise SystemExit(main())
