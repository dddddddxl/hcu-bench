"""BERT runtime-selection/report regressions using CPU fixtures, not model evidence."""
import contextlib
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

from testpacks.migraphx import run as model
from hcu_bench.planner import make_plan

ROOT = Path(__file__).resolve().parents[1]


class BertRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        path = self.root / "fixture.onnx"
        path.write_bytes(b"CPU wrapper fixture; ONNX inspection explicitly mocked")
        self.args = SimpleNamespace(recipe="bert_large_fp16", batch=1, numa_node=0,
                                    model=str(path), binary=sys.executable, numactl=sys.executable,
                                    rocblas_dir="", hcu_ids="0", nnodes=1, node_rank=0,
                                    round_timeout_s=10, output_dir=str(self.root / "output"))

    def test_default_bert_does_not_require_external_tuning_or_change_library_path(self):
        with patch.object(model, "shared_libraries", return_value={"checked": True}):
            env = model.validate(self.args, {"LD_LIBRARY_PATH": "/prepared/runtime/lib"})
        self.assertEqual(env["LD_LIBRARY_PATH"], "/prepared/runtime/lib")
        self.assertEqual(self.args.rocblas_dir, "")

    def test_explicit_missing_directory_still_fails(self):
        self.args.rocblas_dir = str(self.root / "missing")
        with self.assertRaisesRegex(ValueError, "rocblas_dir"):
            model.validate(self.args, {})

    def test_custom_directory_is_preferred_without_mutating_caller_environment(self):
        lib = self.root / "custom/lib"
        lib.mkdir(parents=True)
        self.args.rocblas_dir = str(lib.parent)
        original = {"LD_LIBRARY_PATH": "/prepared/runtime/lib"}
        with patch.object(model, "shared_libraries", return_value={"checked": True}):
            env = model.validate(self.args, dict(original))
        self.assertEqual(env["LD_LIBRARY_PATH"], str(lib.resolve()) + os.pathsep + original["LD_LIBRARY_PATH"])
        self.assertEqual(original["LD_LIBRARY_PATH"], "/prepared/runtime/lib")

    def run_fixture(self):
        inputs = {name: {"dtype": "int64", "shape": [1, 384]} for name in
                  ("input_ids", "attention_mask", "token_type_ids")}
        with patch.object(model, "inspect_model", return_value=inputs), \
             patch.object(model, "shared_libraries", return_value={"checked": True}), \
             patch.object(model, "execute", return_value="Rate: 100 inf/s\nTotal time: 10 ms/batch\n") as execute, \
             contextlib.redirect_stdout(io.StringIO()):
            code = model.run(self.args, threading.Event())
        output = Path(self.args.output_dir)
        return code, output, execute.call_count

    def test_runtime_default_reports_three_rounds_and_explicit_library_mode(self):
        code, output, count = self.run_fixture()
        self.assertEqual(code, 0)
        self.assertEqual(count, 3)
        plan = json.loads((output / "plan.json").read_text())
        result = json.loads((output / "measurements.json").read_text())
        self.assertEqual(plan["rocblas_mode"], "runtime_default")
        self.assertIsNone(plan["tuning_library"])
        self.assertEqual(result["rocblas_mode"], "runtime_default")
        self.assertEqual(result["correctness"], "not_checked")

    def test_custom_mode_retains_library_fingerprints(self):
        lib = self.root / "custom/lib"
        lib.mkdir(parents=True)
        (lib / "fixture.so").write_bytes(b"CPU fixture, not a usable rocBLAS library")
        self.args.rocblas_dir = str(lib.parent)
        code, output, count = self.run_fixture()
        self.assertEqual((code, count), (0, 3))
        plan = json.loads((output / "plan.json").read_text())
        result = json.loads((output / "measurements.json").read_text())
        self.assertEqual(plan["rocblas_mode"], "custom_directory")
        self.assertEqual(result["rocblas_mode"], "custom_directory")
        self.assertEqual(plan["tuning_library"][0]["sha256"], model.sha256(lib / "fixture.so"))

    def test_explicit_empty_library_is_not_silently_replaced_by_default(self):
        lib = self.root / "custom/lib"
        lib.mkdir(parents=True)
        self.args.rocblas_dir = str(lib.parent)
        code, output, count = self.run_fixture()
        self.assertEqual(code, 2)
        self.assertEqual(count, 0)
        self.assertFalse((output / "measurements.json").exists())
        self.assertIn("empty", json.loads((output / "assessment.json").read_text())["error"])

    def test_bert_template_has_two_batches_and_no_required_override(self):
        plan = make_plan(str(ROOT / "configs/bert.template.yaml"), "migraphx", ["tests.bert_large.enabled=true"])
        self.assertEqual(len(plan.cases), 2)
        self.assertEqual({case.params["batch"] for case in plan.cases}, {1, 32})
        self.assertTrue(all(case.params["recipe"] == "bert_large_fp16" for case in plan.cases))
        self.assertTrue(all(case.params["rocblas_dir"] == "" for case in plan.cases))


if __name__ == "__main__":
    unittest.main()
