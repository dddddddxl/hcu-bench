"""Portable BERT startup checks; CPU fixture metrics are never hardware evidence."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from testpacks.migraphx import check as cli

ROOT = Path(__file__).resolve().parents[1]
HAS_ONNX = importlib.util.find_spec("onnx") is not None


def save_metadata_fixture(path, batch="batch", dtype=None):
    import onnx
    from onnx import helper, TensorProto
    # This graph only tests the input contract, not a BERT network or its accuracy.
    inputs = [helper.make_tensor_value_info(name, dtype or TensorProto.INT64, [batch, 384])
              for name in ("input_ids", "attention_mask", "token_type_ids")]
    graph = helper.make_graph([], "CPU metadata fixture, not a BERT model", inputs, [])
    onnx.save(helper.make_model(graph), path)


@unittest.skipUnless(HAS_ONNX, "ONNX is required for input-contract startup checks")
class BertModelPreflight(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.model = self.root / "input contract fixture.onnx"

    def invoke(self, batches=(1, 32), dependency_errors=()):
        output = self.root / "preflight.json"
        dependencies = {"ready": not dependency_errors, "errors": list(dependency_errors)}
        with patch.object(cli, "check", return_value=dependencies), contextlib.redirect_stdout(io.StringIO()):
            code = cli.main(["--recipe", "bert_large_fp16", "--model", str(self.model),
                             "--batch", *map(str, batches), "--output", str(output)])
        return code, json.loads(output.read_text(encoding="utf-8"))

    def test_dynamic_input_contract_is_checked_for_both_batches(self):
        save_metadata_fixture(self.model)
        code, result = self.invoke()
        self.assertEqual(code, 0)
        self.assertTrue(result["ready"])
        self.assertEqual(result["model"]["inputs_by_batch"]["32"]["input_ids"]["shape"], [32, 384])

    def test_static_batch_mismatch_is_not_reported_ready(self):
        save_metadata_fixture(self.model, batch=1)
        code, result = self.invoke()
        self.assertEqual(code, 2)
        self.assertFalse(result["ready"])
        self.assertIn("batch 32", result["errors"][0])
        self.assertEqual(self.invoke(batches=(1,))[0], 0)

    def test_missing_model_and_missing_native_dependency_are_both_reported(self):
        code, result = self.invoke(batches=(1,), dependency_errors=("missing native driver",))
        self.assertEqual(code, 2)
        self.assertEqual(len(result["errors"]), 2)
        self.assertEqual(result["errors"][0], "missing native driver")
        self.assertIn("ONNX validation failed", result["errors"][1])

    def test_unsupported_dtype_is_rejected_before_inference(self):
        from onnx import TensorProto
        save_metadata_fixture(self.model, dtype=TensorProto.INT32)
        code, result = self.invoke(batches=(1,))
        self.assertEqual(code, 2)
        self.assertIn("dtype/rank", result["errors"][0])


@unittest.skipUnless(sys.platform.startswith("linux") and HAS_ONNX and shutil.which("bash"),
                     "Linux, Bash and ONNX are required for native-process startup fixtures")
class BertPublicEntrypoint(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="bert startup fixture ")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.model = self.root / "input contract fixture.onnx"
        save_metadata_fixture(self.model)
        self.driver = self.root / "native-driver-cpu-fixture"
        self.driver.write_text(
            f"#!{sys.executable}\n"
            "import os, sys\n"
            "a = sys.argv[1:]\n"
            "assert a[0:3] == ['perf', '-n', '20']\n"
            "assert '--gpu' in a and '--fp16' in a\n"
            "assert [a[i+1] for i,v in enumerate(a) if v == '--fill1'] == ['input_ids','attention_mask','token_type_ids']\n"
            "assert os.path.isfile(a[a.index('--onnx')+1])\n"
            "assert os.environ['HIP_VISIBLE_DEVICES'] == '0'\n"
            "if os.environ.get('BERT_FIXTURE_FAIL'): raise SystemExit(7)\n"
            "batch = int(a[a.index('--batch')+1])\n"
            "print('CPU startup fixture, not HCU/BERT performance evidence')\n"
            "print(f'Rate: {batch*100} inf/s\\nTotal time: 10 ms/batch')\n", encoding="utf-8")
        self.numactl = self.root / "numactl-cpu-fixture"
        self.numactl.write_text(
            f"#!{sys.executable}\nimport os, sys\n"
            "assert sys.argv[1:3] == ['--cpubind=0', '--membind=0']\n"
            "os.execv(sys.argv[3], sys.argv[3:])\n", encoding="utf-8")
        self.driver.chmod(0o755)
        self.numactl.chmod(0o755)

    def invoke(self, fail=False):
        config = yaml.safe_load((ROOT / "configs/bert.template.yaml").read_text(encoding="utf-8"))
        config["name"] = "CPU BERT startup fixture, not hardware data"
        config["output_dir"] = str(self.root / "results")
        config["vars"].update(bench_root=str(ROOT), model=str(self.model))
        test = config["tests"]["bert_large"]
        test["enabled"] = True
        test["pack"] = str(ROOT / "testpacks/migraphx/manifest.yaml")
        test["params"].update(binary=str(self.driver), numactl=str(self.numactl))
        if fail:
            test["matrix"]["batch"] = [1]
            test["env"] = {"BERT_FIXTURE_FAIL": "1"}
        path = self.root / "bert.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        env = {key: value for key, value in os.environ.items() if key not in
               ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "PYTHONOPTIMIZE")}
        env.update(BENCH_PYTHON=sys.executable, PATH=str(Path(sys.executable).parent) + os.pathsep + env["PATH"])
        result = subprocess.run(["bash", str(ROOT / "bench.sh"), str(path), "migraphx", "--quiet"],
                                cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=60)
        runs = [path for path in (self.root / "results").iterdir() if (path / "state.json").is_file()]
        self.assertEqual(len(runs), 1, result.stdout)
        return result, runs[0]

    def test_public_template_executes_both_batches_and_collects_reports(self):
        result, run = self.invoke()
        self.assertEqual(result.returncode, 0, result.stdout)
        report = json.loads((run / "summary.json").read_text())
        self.assertEqual(report["case_counts"], {"completed": 2})
        self.assertTrue((run / "metrics.csv").is_file())
        self.assertTrue((run / "summary.txt").is_file())
        plans = [json.loads(path.read_text()) for path in run.rglob("plan.json")]
        plans = [plan for plan in plans if "rocblas_mode" in plan]
        self.assertEqual({plan["params"]["batch"] for plan in plans}, {1, 32})
        self.assertTrue(all(plan["rocblas_mode"] == "runtime_default" for plan in plans))
        executions = [json.loads(path.read_text()) for path in run.rglob("execution.json")]
        self.assertTrue(executions)
        self.assertTrue(all(len(items) == 3 and all(item["returncode"] == 0 for item in items) for items in executions))

    def test_failed_native_process_does_not_publish_successful_performance(self):
        result, run = self.invoke(fail=True)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        report = json.loads((run / "summary.json").read_text())
        self.assertEqual(report["case_counts"], {"failed": 1})
        self.assertEqual(report["metrics"], [])
        self.assertFalse(list(run.rglob("measurements.json")))


if __name__ == "__main__":
    unittest.main()
