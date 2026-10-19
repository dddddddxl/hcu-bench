"""Memory pack contracts using synthetic processes, not GPU performance data."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml
from hcu_bench.adapters import for_case
from hcu_bench.models import BenchError
from hcu_bench.planner import make_plan
from hcu_bench.report import build_report
from hcu_bench.results import parse_sample
from hcu_bench.runner import run_plan
from hcu_bench.store import RunStore
from testpacks.memory import run as memory

ROOT = Path(__file__).resolve().parents[1]


class MemoryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.args = SimpleNamespace(binary=sys.executable, gpu_ids="1,3", devices="", presets="both",
                                    model_mhz=1400, array_size=104857600, nnodes=1, node_rank=0,
                                    output_dir=str(self.root))

    def test_presets_match_native_tables_and_flags(self):
        self.assertEqual([case[0] for case in memory.PRESETS["zd1"]], [
            "l1_read", "l2_read", "l2_write", "lds_read", "lds_write", "memory_readx4_x1",
            "memory_write", "memory_copy", "memory_to_ldsx1", "lds_to_memory", "matrix_read",
            "matrix_write", "matrix_copy", "mapping_matrix_write", "tiling_read"])
        self.assertEqual([case[0] for case in memory.PRESETS["zd2"]], [
            "memory_readx4_x1", "memory_write", "memory_copy", "matrix_readx4_x4",
            "matrix_writex4_x4", "matrix_copy", "memory_readx4_x4", "mapping_matrix_writex4_x4", "tiling_read_2w"])
        for preset, cases in memory.PRESETS.items():
            for test, wave, lds in cases:
                self.assertEqual(wave, 8 if preset == "zd2" and test in
                                 ("memory_readx4_x1", "memory_write", "memory_copy") else 4)
                self.assertEqual(lds, 4 if test in ("memory_to_ldsx1", "lds_to_memory") else 0)
        self.assertEqual(memory.command("hcu-stream", "memory_copy", 8, 0, 104857600),
                         ["hcu-stream", "bandwidth", "memory_copy", "--arraysize", "104857600", "-w", "8", "--clean"])
        self.assertEqual(memory.command("hcu-stream", "memory_to_ldsx1", 4, 4, 1024),
                         ["hcu-stream", "bandwidth", "memory_to_ldsx1", "--arraysize", "1024", "--lds", "4", "--clean"])

    def test_native_clean_output_format(self):
        # Format observed in YY85 logs; these are parser inputs, not new results.
        for text, value in (("10742.7\n", 10742.7), ("983.961\n", 983.961), ("20839\n", 20839),
                            (" 9.83e2\r\n", 983), (".5", .5)):
            self.assertEqual(memory.parse_bandwidth(text), value)

    def test_reject_help_errors_multiple_values_and_nonfinite(self):
        for text in ("", "Usage: hcu-stream", "0", "-1", "nan", "NaN", "inf", "1e999", "1e-999",
                     "983.961\n984.0", "ERROR\n983.961", "Validation failed\n1", "1 GB/s", "1 junk"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                memory.parse_bandwidth(text)

    def test_validate_allocation_and_frequency(self):
        self.assertEqual(memory.validate(self.args, {}), ([1, 3], ["zd1", "zd2"]))
        args = SimpleNamespace(**{**vars(self.args), "devices": "3", "presets": "zd2"})
        self.assertEqual(memory.validate(args, {"HIP_VISIBLE_DEVICES": "1,3", "MODEL": "1400"}), ([3], ["zd2"]))
        for field, value in (("nnodes", 2), ("node_rank", 1), ("devices", "0"), ("devices", "1,1"),
                             ("devices", "-1"), ("gpu_ids", ""), ("gpu_ids", "1,1"), ("presets", "unknown"),
                             ("model_mhz", 1), ("model_mhz", 10001), ("array_size", 0), ("array_size", 2147483648)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                memory.validate(SimpleNamespace(**{**vars(self.args), field: value}), {})
        for key, value in (("HIP_VISIBLE_DEVICES", "0"), ("ROCR_VISIBLE_DEVICES", "1,3"),
                           ("CUDA_VISIBLE_DEVICES", "0"), ("GPU_DEVICE_ORDINAL", "1"), ("MODEL", "1280")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                memory.validate(self.args, {key: value})

    def execute(self, fail_at=None, output="983.961", cancel=False, cleanup_error=False, stop=None):
        stopped = stop or threading.Event()
        original = memory.NativeProcess
        launched = []

        def launch(argv, cwd, env, path):
            launched.append((argv, dict(env)))
            code = 7 if len(launched) == fail_at else 0
            script = f"import time,sys; print({output!r}, flush=True); time.sleep({60 if cancel else 0}); sys.exit({code})"
            process = original([sys.executable, "-u", "-c", script], cwd, env, path)
            if cancel:
                timer = threading.Timer(0.25, stopped.set)
                timer.start()
                self.addCleanup(timer.join)
            if cleanup_error:
                close = process.close
                def fail():
                    close()
                    raise memory.CleanupError("fixture cleanup failure")
                process.close = fail
            return process

        env = {key: value for key, value in os.environ.items() if key not in
               ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "MODEL")}
        captured = io.StringIO()
        with patch.dict(os.environ, env, clear=True), patch.object(memory, "NativeProcess", side_effect=launch), \
                contextlib.redirect_stdout(captured):
            code = memory.run(self.args, stopped)
        samples = [parse_sample(line) for line in captured.getvalue().splitlines() if line.startswith("BENCH_RESULT ")]
        return code, samples, launched

    def test_full_serial_sweep_and_artifacts(self):
        code, samples, launched = self.execute()
        self.assertEqual(code, 0)
        self.assertEqual(len(launched), 48)
        self.assertEqual([env["HIP_VISIBLE_DEVICES"] for _, env in launched], ["1"] * 24 + ["3"] * 24)
        self.assertTrue(all(env["MODEL"] == "1400" for _, env in launched))
        self.assertEqual(len(samples), 48)
        self.assertEqual(len({sample["metrics"][0]["name"] for sample in samples}), 48)
        self.assertTrue(all(sample["correctness"] == "not_checked" for sample in samples))
        self.assertTrue(all(sample["metrics"][0]["statistic"] == "reported_mean" for sample in samples))
        self.assertEqual(sum(s["timing"] == "cycles_with_model" for s in samples), 10)
        with (self.root / "measurements.csv").open(encoding="utf-8-sig") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 48)
        plan = json.loads((self.root / "plan.json").read_text())
        self.assertEqual(plan["expected_measurements"], 48)
        self.assertFalse(plan["model_is_clock_setting"])
        self.assertEqual(plan["native_logical_device"], 0)
        self.assertEqual(len(plan["binary_sha256"]), 64)
        execution = json.loads((self.root / "execution.json").read_text())
        self.assertEqual(len(execution), 48)
        self.assertTrue(all(item["returncode"] == 0 and not item["termination_actions"] for item in execution))
        self.assertTrue(json.loads((self.root / "assessment.json").read_text())["complete"])

    def test_subset_preset_and_custom_frequency_size(self):
        self.args.devices, self.args.presets = "3", "zd2"
        self.args.model_mhz, self.args.array_size = 1280, 1024
        code, samples, launched = self.execute()
        self.assertEqual(code, 0)
        self.assertEqual(len(launched), 9)
        self.assertTrue(all(env["HIP_VISIBLE_DEVICES"] == "3" and env["MODEL"] == "1280" for _, env in launched))
        self.assertTrue(all(argv[argv.index("--arraysize") + 1] == "1024" for argv, _ in launched))
        self.assertTrue(all(s["model_mhz"] == 1280 and s["array_size"] == 1024 for s in samples))

    def test_native_failure_preserves_partial_data(self):
        code, samples, launched = self.execute(fail_at=3)
        self.assertEqual(code, 2)
        self.assertEqual(len(launched), 3)
        self.assertEqual([s["measurement_status"] for s in samples], ["measured", "measured", "invalid"])
        assessment = json.loads((self.root / "assessment.json").read_text())
        self.assertFalse(assessment["complete"])
        self.assertEqual(assessment["completed_measurements"], 2)
        self.assertTrue((self.root / "gpu1_zd1_l2_write.log").exists())

    def test_bad_output_with_zero_exit_fails(self):
        code, samples, launched = self.execute(output="Usage: hcu-stream")
        self.assertEqual(code, 2)
        self.assertEqual(len(launched), 1)
        self.assertEqual(samples[0]["measurement_status"], "invalid")

    def test_missing_binary_does_not_launch(self):
        self.args.binary = str(self.root / "absent")
        code, samples, launched = self.execute()
        self.assertEqual(code, 2)
        self.assertEqual(launched, [])
        self.assertEqual(samples[0]["metrics"], [])

    def test_cancel_before_and_during_run(self):
        stopped = threading.Event()
        stopped.set()
        code, _, launched = self.execute(stop=stopped)
        self.assertEqual(code, 2)
        self.assertEqual(launched, [])
        code, samples, launched = self.execute(cancel=True)
        self.assertEqual(code, 2)
        self.assertEqual(len(launched), 1)
        self.assertEqual(samples[0]["measurement_status"], "invalid")

    def test_cleanup_failure_stops_plan_contract(self):
        code, samples, launched = self.execute(cleanup_error=True)
        self.assertEqual(code, 125)
        self.assertEqual(len(launched), 1)
        self.assertEqual(samples, [])
        self.assertTrue((self.root / "cleanup.json").exists())

    def test_templates_all_selection_and_commands(self):
        for name in ("in-container.template.yaml", "managed-container.template.yaml", "memory.template.yaml"):
            plan = make_plan(str(ROOT / "configs" / name), "memory")
            self.assertEqual(len(plan.cases), 1)
            case = plan.cases[0]
            self.assertEqual(case.pack["id"], "memory.bandwidth")
            self.assertEqual(case.params["model_mhz"], 1400)
            spec = for_case(case).build(plan, case, "node0", self.root / "run")
            argv = spec["display"]["argv"]
            self.assertIn("--devices", argv)
            self.assertEqual(argv[argv.index("--devices") + 1], "" if name == "memory.template.yaml" else "0")
            self.assertFalse(any("${" in part for part in argv))
            plan = make_plan(str(ROOT / "configs" / name), "all")
            self.assertIn("memory", [case.suite for case in plan.cases])
            if name != "memory.template.yaml":
                self.assertEqual([case.suite for case in plan.cases], ["rccl", "pcie", "memory", "deepep", "e2e"])

    def test_planner_rejects_duration_multinode_and_bad_params(self):
        data = yaml.safe_load((ROOT / "configs/memory.template.yaml").read_text(encoding="utf-8"))
        test = data["tests"]["memory_bandwidth"]
        test["pack"] = str(ROOT / "testpacks/memory/manifest.yaml")
        path = self.root / "bench.yaml"
        for key, value, message in (("duration_s", 60, "duration_control"), ("nodes", ["node0", "node1"], "at most 1"),
                                   ("params", {**test["params"], "presets": "unknown"}, "presets"),
                                   ("params", {**test["params"], "model_mhz": 1}, "model_mhz")):
            bad = {**data, "tests": {"memory_bandwidth": {**test, key: value}},
                   "nodes": {**data["nodes"], "node1": {"executor": "local"}}}
            path.write_text(yaml.safe_dump(bad), encoding="utf-8")
            with self.assertRaisesRegex(BenchError, message):
                make_plan(str(path), "memory")

    def test_framework_report_and_native_artifact_collection(self):
        # Replace only the hardware subprocess, keeping the real wrapper/worker/report path.
        launcher = self.root / "synthetic_memory.py"
        launcher.write_text(
            "import runpy,sys\n"
            f"sys.path.insert(0, {str(ROOT / 'testpacks')!r})\n"
            "import native_common\n"
            "native = native_common.NativeProcess\n"
            "def launch(argv, cwd, env, log_path):\n"
            "    return native([sys.executable, '-c', 'print(123.456)'], cwd, env, log_path)\n"
            "native_common.NativeProcess = launch\n"
            f"runpy.run_path({str(ROOT / 'testpacks/memory/run.py')!r}, run_name='__main__')\n",
            encoding="utf-8")
        pack = yaml.safe_load((ROOT / "testpacks/memory/manifest.yaml").read_text())
        pack["command"][0], pack["command"][2] = sys.executable, str(launcher)
        (self.root / "pack.yaml").write_text(yaml.safe_dump(pack), encoding="utf-8")
        data = {"version": 1, "name": "synthetic-memory-contract", "output_dir": "runs",
                "nodes": {"local": {"executor": "local", "gpus": [1],
                                    "env": {"HIP_VISIBLE_DEVICES": "${gpu_ids}", "MODEL": "1400",
                                            "CUDA_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "", "GPU_DEVICE_ORDINAL": ""}}},
                "tests": {"memory": {"suite": "memory", "adapter": "command", "pack": "pack.yaml",
                                     "nodes": ["local"], "timeout_s": 60,
                                     "params": {"bench_root": str(ROOT), "binary": sys.executable}}}}
        config = self.root / "bench.yaml"
        config.write_text(yaml.safe_dump(data), encoding="utf-8")
        plan = make_plan(str(config), "all")
        store = RunStore.create(plan)
        with contextlib.redirect_stdout(io.StringIO()):
            code = run_plan(plan, store, quiet=True)
        self.assertEqual(code, 0)
        report = build_report(store)
        self.assertEqual(len(report["metrics"]), 24)
        self.assertTrue(all(m["suite"] == "memory" and m["mean_of_reported_values"] == 123.456
                            and m["count"] == 1 for m in report["metrics"]))
        records = [json.loads(line) for line in (store.path / "artifacts.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(records and all(item["status"] == "collected" for item in records))
        archived = list((store.path / "raw").rglob("*-measurements.csv"))
        self.assertEqual(len(archived), 1)
        with archived[0].open(encoding="utf-8-sig") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 24)
        self.assertEqual(len(list((store.path / "raw").rglob("*-gpu*_zd*.log"))), 24)


if __name__ == "__main__":
    unittest.main()
