"""PCIe contracts with synthetic subprocesses; never hardware performance data."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml
from hcu_bench.models import BenchError
from hcu_bench.planner import make_plan
from hcu_bench.results import parse_sample
from testpacks.pcie import run as pcie

ROOT = Path(__file__).resolve().parents[1]
DEVICES = """RocmBandwidthTest Version: 2.6.0
Device Index: 0
  Device Type: CPU
  Device Name: fixture CPU
  Allocatable Memory Size (KB): 1024
Device Index: 2
  Device Type: GPU
  Device Name: fixture GPU
  Device  BDF: 2a:0.0
  Device UUID: GPU-fixture-a
  Allocatable Memory Size (KB): 1024
Device Index: 3
  Device Type: GPU
  Device Name: fixture GPU
  Device  BDF: 2d:0.0
  Device UUID: GPU-fixture-b
  Allocatable Memory Size (KB): 1024
"""


def log_for(kind="cpu_to_gpu", src=0, dst=2, size=512):
    direction = "Bidirectional" if kind.endswith("bidir") else "Unidirectional"
    a, b = ("Cpu" if src == 0 else "Gpu"), ("Cpu" if dst == 0 else "Gpu")
    return f"""RocmBandwidthTest Version: 2.6.0
================ {direction} Benchmark Result ================
================ Src Device Id: {src} Src Device Type: {a} ================
================ Dst Device Id: {dst} Dst Device Type: {b} ================
Data Size      Avg Time(us)   Avg BW(GB/s)   Min Time(us)   Peak BW(GB/s)
{size} MB          20000.123       26.844         19999.999       26.845
"""


class PcieTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.args = SimpleNamespace(binary=sys.executable, cpu_devices="0", gpu_devices="2,3", gpu_ids="0,1",
                                    size_mib=512, iters=20, nnodes=1, node_rank=0, output_dir=str(self.root))

    def test_device_listing_and_strict_selection(self):
        devices = pcie.parse_devices(DEVICES)
        self.assertEqual(devices[2]["bdf"], "2a:0.0")
        self.assertEqual(devices[3]["uuid"], "GPU-fixture-b")
        for text in ("", "Usage: ...", DEVICES + DEVICES, DEVICES.replace("Device Type: CPU", "Device Type: OTHER")):
            with self.assertRaises(ValueError):
                pcie.parse_devices(text)

    def test_rejects_missing_or_duplicate_native_pools(self):
        marker = "  Allocatable Memory Size (KB): 1024\n"
        for text in (DEVICES.replace(marker, "", 1), DEVICES.replace(marker, marker * 2, 1)):
            with self.assertRaisesRegex(ValueError, "Ambiguous native memory pools"):
                pcie.parse_devices(text)

    def test_five_paths_without_self_or_duplicate_bidir(self):
        paths = list(pcie.pairs([0, 1], list(range(2, 10))))
        self.assertEqual(len(paths), 132)
        self.assertEqual(len(set(paths)), 132)
        self.assertEqual({p[0] for p in paths}, set(pcie.KINDS))
        self.assertTrue(all(src != dst for _, src, dst in paths))
        self.assertEqual(sum(kind == "gpu_to_gpu" for kind, _, _ in paths), 56)
        self.assertEqual(sum(kind == "gpu_gpu_bidir" for kind, _, _ in paths), 28)
        self.assertEqual(pcie.command("rbt", "cpu_gpu_bidir", 0, 2, 512), ["rbt", "-b", "0,2", "-m", "512"])
        self.assertEqual(pcie.command("rbt", "gpu_to_cpu", 2, 0, 64), ["rbt", "-s", "2", "-d", "0", "-m", "64"])

    def test_avg_and_peak_separate_no_bidir_doubling(self):
        for kind, src, dst in pcie.pairs([0], [2, 3]):
            row = pcie.parse_result(log_for(kind, src, dst), kind, src, dst, 512)
            self.assertEqual(row["size_bytes"], 536870912)
            self.assertEqual(row["avg_gbps"], 26.844)
            self.assertEqual(row["peak_gbps"], 26.845)

    def test_actual_smoke_output_shape(self):
        text = """..
          RocmBandwidthTest Version: 2.6.0
          Launch Command is: rocm-bandwidth-test -b 1,7 -m 64
================    Bidirectional Benchmark Result    ================
================ Src Device Id: 1 Src Device Type: Cpu ================
================ Dst Device Id: 7 Dst Device Type: Gpu ================
Data Size      Avg Time(us)   Avg BW(GB/s)   Min Time(us)   Peak BW(GB/s)
64 MB          2577.798       52.067         2575.358       52.116
"""
        row = pcie.parse_result(text, "cpu_gpu_bidir", 1, 7, 64)
        self.assertEqual(row["avg_gbps"], 52.067)
        self.assertEqual(row["peak_gbps"], 52.116)

    def test_parser_rejects_help_wrong_size_direction_units_errors(self):
        text = log_for()
        invalid = ("", "Usage: ...", text.replace("512 MB", "64 MB"), text.replace("GB/s", "Gb/s"),
                   text.replace("Dst Device Id: 2", "Dst Device Id: 3"), text.replace("Unidirectional", "Bidirectional"),
                   text.replace("Src Device Type: Cpu", "Src Device Type: Gpu"), text + text,
                   text.replace("26.844", "nan"), text.replace("26.844", "-1"), text.replace("26.844", "99"),
                   text + "\nHSA_STATUS_ERROR: fault", text + "\nValidation failed", text + "\nSegmentation fault")
        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                pcie.parse_result(candidate, "cpu_to_gpu", 0, 2, 512)

    def test_validate_inputs_and_environment(self):
        self.assertEqual(pcie.validate(self.args, {}), ([0], [2, 3]))
        for attr, value in (("nnodes", 2), ("node_rank", 1), ("gpu_devices", "2,2"), ("gpu_devices", "0,2"),
                            ("gpu_devices", "2"), ("cpu_devices", ""), ("cpu_devices", "-1"),
                            ("gpu_ids", "0,1,2"), ("size_mib", 0), ("iters", 0)):
            with self.subTest(attr=attr, value=value), self.assertRaises(ValueError):
                pcie.validate(SimpleNamespace(**{**vars(self.args), attr: value}), {})
        for env in ({"HIP_VISIBLE_DEVICES": "2,3"}, {"ROCR_VISIBLE_DEVICES": "0,1"},
                    {"CUDA_VISIBLE_DEVICES": "0,1"}, {"ROCM_BW_ITER_CNT": "99"},
                    {"ROCM_BW_DEFAULT_RUN": "1"}, {"ROCM_SKIP_CPU_FINE_GRAINED_POOL": "1"}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                pcie.validate(self.args, env)

    def execute(self, fail_at=None, help_only=False, cancel=False, cleanup_error=False, stop=None):
        stopped = stop or threading.Event()
        original = pcie.NativeProcess
        launched = []

        def launch(argv, cwd, env, path):
            launched.append(argv)
            self.assertEqual(env["ROCM_BW_ITER_CNT"], "20")
            if argv[-1] == "-e":
                text = DEVICES
            elif argv[-1] == "-t":
                text = "fixture topology"
            else:
                name = Path(path).stem
                kind, src, dst = name.rsplit("_", 2)
                text = "Usage: ..." if help_only else log_for(kind, int(src), int(dst))
            code = 7 if len(launched) == fail_at else 0
            script = f"import time,sys; print({text!r}, flush=True); time.sleep({60 if cancel else 0}); sys.exit({code})"
            process = original([sys.executable, "-u", "-c", script], cwd, env, path)
            if cancel:
                timer = threading.Timer(0.2, stopped.set)
                timer.start()
                self.addCleanup(timer.join)
            if cleanup_error:
                close = process.close
                def fail():
                    close()
                    raise pcie.CleanupError("fixture cleanup failure")
                process.close = fail
            return process

        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("ROCM_BW_", "ROCR_BW_", "ROCM_SKIP_"))
               and k not in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}
        output = io.StringIO()
        with patch.dict(os.environ, env, clear=True), patch.object(pcie, "NativeProcess", side_effect=launch), \
                contextlib.redirect_stdout(output):
            code = pcie.run(self.args, stopped)
        samples = [parse_sample(line) for line in output.getvalue().splitlines() if line.startswith("BENCH_RESULT ")]
        return code, samples, launched

    def test_complete_run_all_paths_artifacts_and_unique_metrics(self):
        code, samples, launched = self.execute()
        self.assertEqual(code, 0)
        self.assertEqual(len(launched), 11)
        self.assertEqual(len(samples), 9)
        names = [m["name"] for s in samples for m in s["metrics"]]
        self.assertEqual(len(set(names)), 18)
        self.assertTrue(all(s["correctness"] == "not_checked" for s in samples))
        self.assertTrue(json.loads((self.root / "assessment.json").read_text())["complete"])
        for kind in pcie.KINDS:
            with (self.root / f"{kind}.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
                self.assertTrue(rows)
                self.assertTrue(all(row["kind"] == kind for row in rows))
        executions = json.loads((self.root / "execution.json").read_text())
        self.assertTrue(all(e["returncode"] == 0 for e in executions))

    def test_nonzero_exit_keeps_partial_data_and_fails(self):
        code, samples, launched = self.execute(fail_at=4)
        self.assertEqual(code, 2)
        self.assertEqual(len(launched), 4)
        self.assertEqual([s["measurement_status"] for s in samples], ["measured", "invalid"])
        self.assertFalse(json.loads((self.root / "assessment.json").read_text())["complete"])
        self.assertTrue((self.root / "gpu_to_cpu_2_0.log").exists())

    def test_help_with_zero_exit_is_not_a_measurement(self):
        code, samples, _ = self.execute(help_only=True)
        self.assertEqual(code, 2)
        self.assertEqual([s["measurement_status"] for s in samples], ["invalid"])

    def test_device_mismatch_stops_before_bandwidth(self):
        self.args.gpu_devices = "2,9"
        code, samples, launched = self.execute()
        self.assertEqual(code, 2)
        self.assertEqual(len(launched), 1)
        self.assertEqual(samples[0]["metrics"], [])

    def test_cancel_and_unconfirmed_cleanup(self):
        code, samples, _ = self.execute(cancel=True)
        self.assertEqual(code, 2)
        self.assertEqual(samples[0]["measurement_status"], "invalid")
        code, samples, _ = self.execute(cleanup_error=True)
        self.assertEqual(code, 125)
        self.assertEqual(samples, [])
        self.assertTrue((self.root / "cleanup.json").exists())

    def test_cancel_before_start(self):
        stopped = threading.Event()
        stopped.set()
        code, _, launched = self.execute(stop=stopped)
        self.assertEqual(code, 2)
        self.assertEqual(launched, [])

    def test_templates_selection_and_constraints(self):
        for template in ("in-container.template.yaml", "managed-container.template.yaml", "pcie.template.yaml"):
            plan = make_plan(str(ROOT / "configs" / template), "pcie")
            self.assertEqual(len(plan.cases), 1)
            self.assertEqual(plan.cases[0].pack["id"], "pcie.bandwidth")
            self.assertEqual(plan.cases[0].params["size_mib"], 512)
        data = yaml.safe_load((ROOT / "configs/pcie.template.yaml").read_text(encoding="utf-8"))
        test = data["tests"]["pcie_bandwidth"]
        test["pack"] = str(ROOT / "testpacks/pcie/manifest.yaml")
        path = self.root / "bench.yaml"
        for key, value, message in (("duration_s", 60, "duration_control"), ("nodes", ["node0", "node1"], "at most 1")):
            bad = {**data, "tests": {"pcie_bandwidth": {**test, key: value}},
                   "nodes": {**data["nodes"], "node1": {"executor": "local"}}}
            path.write_text(yaml.safe_dump(bad), encoding="utf-8")
            with self.assertRaisesRegex(BenchError, message):
                make_plan(str(path), "pcie")


if __name__ == "__main__":
    unittest.main()
