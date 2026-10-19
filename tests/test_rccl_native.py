"""Synthetic runner/CLI contracts, not hardware performance evidence."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
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
from testpacks.rccl import run as rccl

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
LOG = """# nThread 1 nGpus 1 minBytes 8192 maxBytes 16384 step: 2(factor) warmup iters: 20 iters: 50 agg iters: 1 validation: 1 graph: 0
rccl-tests: Version skill-bundle:archive-ae3e635
#   Rank 0 Pid 100 on fixture device 0 [0000:e7:00.0] fixture-HCU
#   Rank 1 Pid 101 on fixture device 1 [0000:e8:00.0] fixture-HCU
# out-of-place in-place
# size count type redop root time algbw busbw #wrong time algbw busbw #wrong
# (B) (elements) (us) (GB/s) (GB/s) (us) (GB/s) (GB/s)
8192 4096 half sum -1 10.0 0.82 0.82 0 9.0 0.91 0.91 0
16384 8192 half sum -1 11.0 1.49 1.49 0 10.0 1.64 1.64 0
# Out of bounds values : 0 OK
# Avg bus bandwidth : 1.215
"""


class RcclNativeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.params = {key: rule["default"] for key, rule in rccl.RcclAllReduce.schema.items()}
        self.params["max_bytes"] = 16384
        self.args = SimpleNamespace(**self.params, gpu_ids="6,7", nnodes=1, node_rank=0, output_dir=str(self.root))
        self.env = dict(hostname="fixture", binary_sha256="a" * 64, rccl_library_sha256="b" * 64,
                        mpi_version="fixture", dtk_version="fixture", variables={"HIP_VISIBLE_DEVICES": "6,7"})

    def execute(self, log=LOG, code=0, stderr="", stopped=None, close_error=False, cancel_running=False):
        original = rccl.NativeProcess
        stop = stopped or threading.Event()
        def launch(argv, cwd, env, path):
            self.assertEqual(env["HIP_VISIBLE_DEVICES"], "6,7")
            self.assertEqual(argv[1:3], ["--host", "localhost"])
            script = f"import sys,time; print({log!r}); print({stderr!r}, file=sys.stderr); time.sleep({60 if cancel_running else 0}); sys.exit({code})"
            process = original([sys.executable, "-u", "-c", script], cwd, env, path)
            if cancel_running:
                timer = threading.Timer(0.2, stop.set)
                timer.start()
                self.addCleanup(timer.join)
            if close_error:
                close = process.close
                def fail():
                    close()
                    raise rccl.CleanupError("fixture cleanup failure")
                process.close = fail
            return process
        output = io.StringIO()
        base_env = {key: value for key, value in os.environ.items()
                    if key not in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                                   "NCCL_TESTS_SPLIT", "NCCL_TESTS_SPLIT_MASK", "NCCL_TESTS_DEVICE")}
        with patch.dict(os.environ, base_env, clear=True), patch.object(rccl, "environment", return_value=self.env), \
                patch.object(rccl, "NativeProcess", side_effect=launch), contextlib.redirect_stdout(output):
            result = rccl.run(self.args, stop)
        samples = [parse_sample(line) for line in output.getvalue().splitlines() if line.startswith("BENCH_RESULT ")]
        return result, samples

    def test_complete_run_metrics_and_artifacts(self):
        code, samples = self.execute()
        self.assertEqual(code, 0)
        self.assertEqual(len(samples), 4)
        names = [metric["name"] for sample in samples for metric in sample["metrics"]]
        self.assertEqual(len(set(names)), 12)
        self.assertTrue(all(s["correctness"] == "passed" for s in samples))
        meta = json.loads((self.root / "execution.json").read_text())
        self.assertEqual(meta["returncode"], 0)
        self.assertIn("merged", meta["stream_layout"])
        self.assertEqual(meta["sha256"]["stdout.log"], rccl.sha256(self.root / "stdout.log"))
        self.assertTrue((self.root / "measurements.csv").is_file())

    def test_native_failure_never_publishes_valid_metrics(self):
        for kwargs in ({"code": 7}, {"log": ""}, {"log": LOG.replace("0 9.0", "1 9.0")},
                       {"log": LOG.split("# Out of bounds")[0]},
                       {"stderr": "Launch params (512, 1, 1) are larger than launch bounds (256)"}):
            with self.subTest(kwargs=kwargs):
                code, samples = self.execute(**kwargs)
                self.assertEqual(code, 2)
                self.assertEqual(samples[0]["measurement_status"], "invalid")
                self.assertEqual(samples[0]["metrics"], [])
                self.assertFalse(json.loads((self.root / "assessment.json").read_text())["evidence_valid"])

    def test_cleanup_failure_uses_reserved_exit_code(self):
        code, samples = self.execute(close_error=True)
        self.assertEqual(code, 125)
        self.assertEqual(samples, [])
        self.assertEqual(json.loads((self.root / "cleanup.json").read_text())["cleanup"], "unconfirmed")

    def test_cancelled_before_start_does_not_launch(self):
        stop = threading.Event()
        stop.set()
        code, samples = self.execute(stopped=stop)
        self.assertEqual(code, 2)
        self.assertEqual(samples, [])
        self.assertFalse((self.root / "stdout.log").exists())

    def test_cancelled_running_process_keeps_logs_without_pass(self):
        code, samples = self.execute(cancel_running=True)
        self.assertEqual(code, 2)
        self.assertEqual(samples[0]["measurement_status"], "invalid")
        meta = json.loads((self.root / "execution.json").read_text())
        self.assertTrue(meta["cancelled"])
        self.assertTrue(meta["root_termination_actions"])
        self.assertTrue((self.root / "stdout.log").is_file())

    def test_validation_rejects_ambiguous_devices_and_overrides(self):
        for env in ({"HIP_VISIBLE_DEVICES": "0,1"}, {"ROCR_VISIBLE_DEVICES": "6,7"},
                    {"CUDA_VISIBLE_DEVICES": "0,1"}, {"NCCL_TESTS_SPLIT": "1"}):
            with self.assertRaises(ValueError):
                rccl.validate(self.params, "6,7", 1, 0, env)
        for ids in ("6,6", "6", "0,1,2", "a,b"):
            with self.assertRaises(ValueError):
                rccl.validate(self.params, ids, 1, 0, {})
        for key, value in (("step_factor", 1), ("max_bytes", 9000), ("ranks", 1), ("iters", 0)):
            with self.assertRaises(ValueError):
                rccl.validate({**self.params, key: value}, "6,7", 1, 0, {})
        with self.assertRaises(ValueError):
            rccl.validate(self.params, "6,7", 2, 0, {})

    def test_all_templates_include_real_rccl_first(self):
        for template in ("in-container.template.yaml", "managed-container.template.yaml"):
            plan = make_plan(str(ROOT / "configs" / template))
            self.assertEqual([case.suite for case in plan.cases], ["rccl", "pcie", "memory", "deepep", "e2e"])
            self.assertTrue(all(case.adapter == "command" for case in plan.cases))
            self.assertEqual(plan.cases[0].pack["id"], "rccl.allreduce")

    def test_planner_rejects_multiple_nodes_and_pressure_duration(self):
        data = yaml.safe_load((ROOT / "configs/in-container.template.yaml").read_text(encoding="utf-8"))
        case = data["tests"]["rccl_allreduce"]
        case["pack"] = str(ROOT / "testpacks/rccl/manifest.yaml")
        config = self.root / "bench.yaml"
        case["duration_s"] = 10
        config.write_text(yaml.safe_dump(data), encoding="utf-8")
        with self.assertRaisesRegex(BenchError, "duration_control"):
            make_plan(str(config), "rccl")
        del case["duration_s"]
        data["nodes"]["node1"] = {"executor": "local"}
        case["nodes"] = ["node0", "node1"]
        config.write_text(yaml.safe_dump(data), encoding="utf-8")
        with self.assertRaisesRegex(BenchError, "at most 1"):
            make_plan(str(config), "rccl")


@unittest.skipUnless(os.name == "posix" and BASH, "Native Bash entrypoint requires POSIX")
class BashEntrypointTests(unittest.TestCase):
    def test_default_explicit_selection_and_missing_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(ROOT / "bench.sh", root / "bench.sh")
            (root / "configs").mkdir()
            config = root / "configs/bench.yaml"
            config.write_text("fixture")
            module = root / "src/hcu_bench"
            module.mkdir(parents=True)
            (module / "__init__.py").write_text("")
            (module / "__main__.py").write_text("import json,sys; print(json.dumps(sys.argv[1:]))")
            env = {**os.environ, "BENCH_PYTHON": sys.executable}
            for args, suite in ((["all"], "all"), ([], "all"), (["rccl"], "rccl"), ([str(config), "all"], "all")):
                result = subprocess.run([BASH, str(root / "bench.sh"), *args], cwd=root.parent, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                actual = json.loads(result.stdout)
                self.assertEqual(actual[:2], ["run", "-c"])
                self.assertEqual(Path(actual[2]).resolve(), config.resolve())
                self.assertEqual(actual[3:], ["--suite", suite])
            config.unlink()
            result = subprocess.run([BASH, str(root / "bench.sh"), "all"], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("First copy", result.stderr)
