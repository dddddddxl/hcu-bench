import concurrent.futures
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import yaml

from hcu_bench.cli import preview
from hcu_bench.planner import make_plan
from hcu_bench.models import BenchError
from testpacks.native_common import CleanupError
from testpacks.mooncake import run as mooncake

ROOT = Path(__file__).resolve().parents[1]


def args_for(root, rank=0, count=2, topology="pair", **overrides):
    args = mooncake.parser().parse_args([
        "--output-dir", str(root / f"node-{rank}"), "--shared-dir", str(root / "shared"),
        "--run-id", "test-run", "--case-id", "test-case", "--node-rank", str(rank), "--nnodes", str(count),
        "--addresses", ",".join(f"10.20.1.{n + 1}" for n in range(count)), "--topology", topology,
        "--duration", "1", "--startup-timeout", "2", "--completion-grace", "2"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class MooncakeContracts(unittest.TestCase):
    def test_topology_routes(self):
        self.assertEqual(mooncake.routes_for("pair", 2), {1: 0})
        self.assertEqual(mooncake.routes_for("fan_in", 4), {1: 0, 2: 0, 3: 0})
        self.assertEqual(mooncake.routes_for("pairs", 4), {1: 0, 3: 2})
        self.assertEqual(mooncake.routes_for("ring", 4), {0: 1, 1: 2, 2: 3, 3: 0})

    def test_invalid_topology_sizes(self):
        for topology, count in (("pair", 1), ("pair", 4), ("pairs", 3), ("ring", 1), ("unknown", 2)):
            with self.subTest(topology=topology, count=count), self.assertRaises(ValueError):
                mooncake.routes_for(topology, count)

    def test_native_final_units_and_counts(self):
        value = mooncake.parse_completion("I1012 log] Test completed: duration 180.02, batch count 3016282, throughput 140.54 GB/s")
        self.assertEqual(value["batch_count"], 3016282)
        self.assertEqual(value["throughput_gbps"], 140.54)
        value = mooncake.parse_completion("Test completed: duration 1.02, batch count 10, throughput 100 GiB/s")
        self.assertAlmostEqual(value["throughput_gbps"], 107.3741824)
        self.assertEqual(value["unit"], "GiB/s")
        self.assertIsNone(mooncake.parse_completion("Registered 8 VRAM buffers"))

    def test_invalid_final_never_becomes_a_measurement(self):
        for duration, count, throughput, unit in (("nan", 1, "1", "GB/s"), ("1", 1, "inf", "GB/s"),
                ("1", 0, "1", "GB/s"), ("0", 1, "1", "GB/s"), ("1", 1, "0", "GB/s"), ("1", 1, "1", "Gb/s")):
            with self.subTest(duration=duration, count=count, throughput=throughput, unit=unit), self.assertRaises(ValueError):
                mooncake.parse_completion(f"Test completed: duration {duration}, batch count {count}, throughput {throughput} {unit}")

    def test_explicit_threshold(self):
        self.assertIsNone(mooncake.threshold_for(""))
        self.assertEqual(mooncake.threshold_for("130"), 130)
        for value in ("0", "-1", "nan", "inf", "bad"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                mooncake.threshold_for(value)

    def test_args_reject_unsafe_identity_and_invalid_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for overrides in ({"run_id": "../other"}, {"case_id": ".."}, {"shared_dir": "relative"},
                    {"addresses": "127.0.0.1,10.20.1.2"}, {"addresses": "10.20.1.1,10.20.1.1"},
                    {"gpu_id": -2}, {"block_size": 1073741825}, {"startup_timeout": float("nan")}, {"duration": 0}):
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    mooncake.validate_args(args_for(root, **overrides))

    def test_commands_use_explicit_vram_dynamic_endpoint_and_native_duration(self):
        args = args_for(Path(tempfile.gettempdir()))
        command = mooncake.native_command(args, "initiator", "10.20.1.2", "10.20.1.1:43123")
        self.assertIn("--segment_id=10.20.1.1:43123", command)
        self.assertIn("--protocol=rdma", command)
        self.assertIn("--use_vram=true", command)
        self.assertIn("--duration=1", command)
        self.assertNotIn("--segment_id=", mooncake.native_command(args, "target", "10.20.1.1"))

    def test_manifest_and_template_plan(self):
        plan = make_plan(str(ROOT / "configs/mooncake.template.yaml"), "mooncake")
        rendered = preview(plan)
        self.assertEqual(len(plan.cases), 1)
        self.assertIsNone(plan.cases[0].duration_s)
        self.assertEqual(plan.cases[0].params["duration"], 180)
        self.assertFalse(plan.cases[0].pack["requires_metrics"])
        argv = rendered["cases"][0]["commands"]["target"]["argv"]
        self.assertIn("--run-id", argv)
        self.assertIn("--node-rank", argv)
        self.assertIn("--shared-dir", argv)

    def test_framework_duration_cannot_truncate_native_final_results(self):
        document = yaml.safe_load((ROOT / "configs/mooncake.template.yaml").read_text(encoding="utf-8"))
        document["tests"]["transfer_pair"]["pack"] = str(ROOT / "testpacks/mooncake/manifest.yaml")
        document["tests"]["transfer_pair"]["duration_s"] = 30
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.yaml"
            path.write_text(yaml.safe_dump(document), encoding="utf-8")
            with self.assertRaisesRegex(BenchError, "forbids framework duration_s"):
                make_plan(str(path))

    def test_exclusive_join_and_mismatched_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = mooncake.Rendezvous(args_for(root), "one", threading.Event())
            with self.assertRaises(FileExistsError):
                mooncake.Rendezvous(args_for(root), "one", threading.Event())
            mooncake.Rendezvous(args_for(root, rank=1), "two", threading.Event())
            with self.assertRaisesRegex(ValueError, "mismatch"):
                first.read("join", 1)

    def test_cancelled_shared_wait_does_not_start_native_work(self):
        with tempfile.TemporaryDirectory() as directory:
            stopped = threading.Event()
            rendezvous = mooncake.Rendezvous(args_for(Path(directory)), "one", stopped)
            stopped.set()
            with self.assertRaises(InterruptedError):
                rendezvous.wait(lambda: None, 1)

    def test_preflight_rejects_old_version_before_starting_work(self):
        with patch.object(mooncake, "versions", return_value={"mooncake-transfer-engine": "0.3.10.post1+das", "psutil": "7"}):
            with self.assertRaisesRegex(ValueError, "0.3.12"):
                mooncake.preflight(args_for(Path(tempfile.gettempdir())))

    def test_unconfirmed_cleanup_uses_reserved_failure_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            argv = ["--output-dir", str(output), "--shared-dir", str(root / "shared"), "--run-id", "run",
                    "--case-id", "case", "--node-rank", "0", "--nnodes", "2", "--addresses", "10.20.1.1,10.20.1.2"]
            with patch.object(mooncake, "run", side_effect=CleanupError("test cleanup failure")), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(mooncake.main(argv), 125)
            self.assertEqual(json.loads((output / "cleanup.json").read_text())["cleanup"], "unconfirmed")

    def test_preflight_checks_native_flags_visibility_active_nics_and_busy_processes(self):
        flags = ("mode protocol metadata_server local_server_name segment_id gpu_id use_vram auto_discovery "
                 "duration buffer_size threads batch_size block_size operation").split()
        help_result = subprocess.CompletedProcess([], 1, "\n".join(f" -{flag} (help)" for flag in flags))
        def sysfs(path, *values, **kwargs):
            return {"state": "4: ACTIVE", "phys_state": "5: LinkUp", "rate": "200 Gb/sec"}[path.name]
        with patch.dict(os.environ, {"HIP_VISIBLE_DEVICES": "3", "MC_TE_FILTERS": "mlx5_0"}), \
             patch.object(mooncake, "versions", return_value={"mooncake-transfer-engine": "0.3.13.post1+test", "psutil": "7"}), \
             patch.object(mooncake.shutil, "which", return_value="/bin/transfer_engine_bench"), \
             patch.object(mooncake.subprocess, "run", return_value=help_result), \
             patch.object(mooncake, "sha256", return_value="hash"), \
             patch.object(Path, "read_text", sysfs), patch("psutil.process_iter", return_value=[]):
            value = mooncake.preflight(args_for(Path(tempfile.gettempdir())))
            self.assertEqual(value["visible_gpu_count"], 1)
            self.assertIn("mlx5_0", value["nic_state"])
            with self.assertRaisesRegex(ValueError, "outside"):
                mooncake.preflight(args_for(Path(tempfile.gettempdir()), gpu_id=1))
            help_result.stdout = " -mode (help)"
            with self.assertRaisesRegex(ValueError, "flags"):
                mooncake.preflight(args_for(Path(tempfile.gettempdir())))


FAKE_NATIVE = r'''
import os,sys,time
from pathlib import Path
flags = dict(value[2:].split('=',1) for value in sys.argv[1:] if value.startswith('--') and '=' in value)
Path(flags['mode']+'.pid').write_text(str(os.getpid()))
if flags['mode']=='target':
    port=43100+int(flags['local_server_name'].rsplit('.',1)[1])
    print('Transfer Engine RPC using P2P handshake, listening on '+flags['local_server_name']+':'+str(port),flush=True)
    while True: time.sleep(0.1)
else:
    if os.environ.get('TEST_MOONCAKE_CASE')=='missing': sys.exit(0)
    if os.environ.get('TEST_MOONCAKE_CASE')=='hang':
        while True: time.sleep(0.1)
    if flags.get('operation')=='write':
        print('E1012 12:00:00 Native transfer failed',flush=True)
        sys.exit(7)
    ip,port=flags['segment_id'].split(':')
    assert int(port)==43100+int(ip.rsplit('.',1)[1]), 'stale/fixed port'
    duration='0.1' if os.environ.get('TEST_MOONCAKE_CASE')=='early' else flags['duration']
    print('Test completed: duration '+duration+', batch count 100, throughput 140.54 GB/s',flush=True)
'''


class MooncakeNativeLifecycle(unittest.TestCase):
    """CPU substitute exercises real process ownership/rendezvous, not RDMA performance."""
    def execute(self, root, count=2, topology="pair", **overrides):
        native = root / "fake_native.py"
        native.write_text(FAKE_NATIVE, encoding="utf-8")
        original = mooncake.native_command
        args = [args_for(root, rank, count, topology, **overrides) for rank in range(count)]
        evidence = {"binary": sys.executable, "packages": {"mooncake-transfer-engine": "0.3.13.post1+test"}}
        emitted = []

        def call(value):
            try:
                return mooncake.run(value)
            except Exception as exc:
                return exc

        with patch.object(mooncake, "preflight", return_value=evidence), \
             patch.object(mooncake, "cancellation", side_effect=threading.Event), \
             patch.object(mooncake, "emit", side_effect=emitted.append), \
             patch.object(mooncake, "native_command", side_effect=lambda *values: [sys.executable, str(native), *original(*values)[1:]]), \
             contextlib.redirect_stdout(io.StringIO()), concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
            results = list(pool.map(call, args))
        return results, emitted

    def assert_clean(self, root):
        import psutil
        for pid_file in root.glob("node-*/*.pid"):
            self.assertFalse(psutil.pid_exists(int(pid_file.read_text())), str(pid_file))
        for cleanup in root.glob("node-*/cleanup.json"):
            self.assertEqual(json.loads(cleanup.read_text())["cleanup"], "confirmed")

    def test_pair_uses_fresh_port_and_passive_target_has_no_fake_throughput(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, emitted = self.execute(root)
            self.assertEqual(results, [0, 0])
            measured = [row for row in emitted if row["measurement_status"] == "measured"]
            self.assertEqual(len(measured), 1)
            self.assertEqual(measured[0]["target_endpoint"], "10.20.1.1:43101")
            self.assertEqual(measured[0]["performance_judgement"], "report_only")
            self.assertEqual(measured[0]["correctness"], "not_checked")
            self.assertTrue(any(row["measurement_status"] == "skipped" and not row["metrics"] for row in emitted))
            self.assert_clean(root)

    def test_ring_stops_both_roles_on_every_node(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, emitted = self.execute(root, 4, "ring")
            self.assertEqual(results, [0] * 4)
            self.assertEqual(len([row for row in emitted if row["measurement_status"] == "measured"]), 4)
            self.assert_clean(root)

    def test_fan_in_and_independent_pairs(self):
        for topology, expected in (("fan_in", 3), ("pairs", 2)):
            with self.subTest(topology=topology), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                results, emitted = self.execute(root, 4, topology)
                self.assertEqual(results, [0] * 4)
                self.assertEqual(len([row for row in emitted if row["measurement_status"] == "measured"]), expected)
                self.assert_clean(root)

    def test_low_bandwidth_is_a_failure_without_claiming_numeric_correctness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, emitted = self.execute(root, min_bandwidth_gbps="150")
            self.assertTrue(all(isinstance(result, Exception) for result in results))
            self.assertEqual(emitted[0]["performance_judgement"], "failed")
            self.assertEqual(emitted[0]["correctness"], "not_checked")
            self.assert_clean(root)

    def test_explicit_threshold_can_pass_without_marking_correctness_passed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, emitted = self.execute(root, min_bandwidth_gbps="130")
            self.assertEqual(results, [0, 0])
            measurement = next(row for row in emitted if row["measurement_status"] == "measured")
            self.assertEqual(measurement["performance_judgement"], "passed")
            self.assertEqual(measurement["correctness"], "not_checked")
            self.assert_clean(root)

    def test_native_error_fails_peers_and_leaves_unrelated_process_alive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
            try:
                results, emitted = self.execute(root, operation="write")
                self.assertTrue(all(isinstance(result, Exception) for result in results))
                self.assertFalse(emitted)
                self.assertIsNone(unrelated.poll())
                self.assert_clean(root)
            finally:
                unrelated.terminate()
                unrelated.wait(timeout=5)

    def test_missing_peer_times_out_without_starting_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = {"binary": sys.executable, "packages": {"mooncake-transfer-engine": "0.3.13.post1+test"}}
            with patch.object(mooncake, "preflight", return_value=evidence), patch.object(mooncake, "cancellation", side_effect=threading.Event), \
                 patch.object(mooncake, "NativeProcess") as process:
                with self.assertRaises(TimeoutError):
                    mooncake.run(args_for(root, startup_timeout=0.15))
                process.assert_not_called()

    def test_empty_measurement_and_early_window_fail_and_clean_up(self):
        for scenario in ("missing", "early"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory, \
                 patch.dict(os.environ, {"TEST_MOONCAKE_CASE": scenario}):
                root = Path(directory)
                results, emitted = self.execute(root)
                self.assertTrue(all(isinstance(result, Exception) for result in results))
                self.assertFalse(emitted)
                self.assert_clean(root)

    def test_hung_initiator_has_independent_deadline_and_cleans_target(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"TEST_MOONCAKE_CASE": "hang"}):
            root = Path(directory)
            results, emitted = self.execute(root, completion_grace=0.1)
            self.assertTrue(all(isinstance(result, Exception) for result in results))
            self.assertFalse(emitted)
            self.assert_clean(root)


if __name__ == "__main__":
    unittest.main()
