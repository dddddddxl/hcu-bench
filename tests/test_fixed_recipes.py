"""CPU fixtures for fixed recipes; these results are not HCU performance evidence."""
import contextlib
import ast
import csv
import importlib.util
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

from hcu_bench.cli import preview
from hcu_bench.models import BenchError
from hcu_bench.planner import make_plan
from hcu_bench.results import parse_sample
from hcu_bench.report import build_report
from hcu_bench.runner import run_plan
from hcu_bench.store import RunStore
from testpacks.operators import run as op
from testpacks.operators import probe
from testpacks.migraphx import run as model
import fixed_common

ROOT = Path(__file__).resolve().parents[1]


def pa_log():
    return "\n".join(f"bs {b}, qhead 24, kvhead 2, seqlen_k {s} headdim 128 128 dtype torch.float16\n"
                     "Performance: 0.100 ms, 12.00 TFLOPS, 150.0 GB/s" for b, s in sorted(op.PA_GRID))


class FixedTemplates(unittest.TestCase):
    def test_image_selection_and_case_catalog_are_static(self):
        with patch("subprocess.Popen", side_effect=AssertionError("static check started a process")):
            for config, count, suite in (("operators", 6, "operators"), ("migraphx", 6, "e2e_perf")):
                overrides = ["vars.image=registry.example/test@sha256:fixture"]
                if config == "operators":
                    overrides += ["tests.operator_stress.enabled=true"]
                plan = make_plan(str(ROOT / f"configs/{config}.template.yaml"), overrides=overrides)
                self.assertEqual(len(plan.cases), count)
                self.assertEqual({c.suite for c in plan.cases}, {suite})
                rendered = json.dumps(preview(plan))
                self.assertIn("registry.example/test@sha256:fixture", rendered)
                self.assertNotIn("42.228.", rendered)
                self.assertNotIn("script-session", rendered)
                self.assertTrue(all(c.pack["duration_control"] == "native" for c in plan.cases))

    def test_rejects_unfixed_params_and_framework_duration(self):
        config = str(ROOT / "configs/migraphx.template.yaml")
        for override in ("tests.resnet50.matrix.batch=[2]", "tests.resnet50.matrix.recipe=[resnet50_fp32]",
                         "tests.resnet50.duration_s=10", "tests.resnet50.params.rounds=1"):
            with self.subTest(override=override), self.assertRaises(BenchError):
                make_plan(config, overrides=[override])

    def test_native_source_hashes_and_safe_adaptations(self):
        provenance = json.loads((op.ROOT / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(len(provenance["files"]), 7)
        for record in provenance["files"]:
            self.assertEqual(op.sha256(op.ROOT / "native" / record["file"]), record["bundled_sha256"])
            ast.parse((op.ROOT / "native" / record["file"]).read_text(encoding="utf-8"))
        pressure = (op.ROOT / "native/fa_fwd_pressure.py").read_text(encoding="utf-8")
        self.assertIn('os.environ["BENCH_FA_CACHE_DIR"]', pressure)
        self.assertIn("pressure_count = 1000000", pressure)
        self.assertNotIn(" = eval(", pressure)


class FixedParsers(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_fixed_operator_commands(self):
        for recipe in op.RECIPES:
            self.assertEqual(op.command(recipe)[:2], [sys.executable, "-u"])
        self.assertEqual(op.command("fa_fp16_stress")[3:],
                         ["1", "52", "8192", "128", "128", "true", "fp16", "13", "[-1,-1]", "false", "--whl"])
        self.assertEqual(op.command("fa_bf16_stress")[9], "bfloat16")
        self.assertEqual(op.command("pa_decode")[3:], ["--whl"])
        self.assertEqual(op.command("fa_fp16_forward")[3:], [])

    def test_stress_requires_completion_without_claiming_accuracy(self):
        text = "CHECK NaN: PASS\nCHECK Inf: PASS\nPressure Test: 1000000/1000000 \x1b[32mPASS\x1b[0m\n"
        for recipe in ("blas_fp16_stress", "fa_fp16_stress", "fa_bf16_stress"):
            row, = op.parse(recipe, text, self.root)
            self.assertEqual(row["repeatability"], "passed")
            self.assertEqual(row["correctness"], "not_checked")
            self.assertEqual(row["metrics"][0]["value"], 1000000)
        for bad in ("PASS", text.replace("1000000/", "999999/"), text.replace("Inf: PASS", "Inf: FAIL"),
                    text + "Traceback (most recent call last)", text.replace("CHECK NaN: PASS", "")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                op.parse("fa_fp16_stress", bad, self.root)

    def test_pa_requires_all_40_distinct_shapes_and_ignores_summary(self):
        rows = op.parse("pa_decode", pa_log() + "\nTotal Statistics:\nPerformance: summary", self.root)
        self.assertEqual(len(rows), 40)
        self.assertEqual(len({m["name"] for r in rows for m in r["metrics"]}), 120)
        for bad in (pa_log().split("\n", 2)[2], pa_log() + "\n" + pa_log(),
                    pa_log().replace("qhead 24", "qhead 16", 1),
                    pa_log().replace("0.100 ms", "0.000 ms", 1),
                    pa_log().replace("150.0 GB/s", "nan GB/s", 1)):
            with self.subTest(bad=bad[:80]), self.assertRaises(ValueError):
                op.parse("pa_decode", bad, self.root)

    def test_fa_fp16_requires_one_positive_pair(self):
        text = "Time: 2.5\nPerformance: \x1b[35m500.0\x1b[0m TFLOPS\n"
        row, = op.parse("fa_fp16_forward", text, self.root)
        self.assertEqual([m["value"] for m in row["metrics"]], [2.5, 500])
        for bad in (text * 2, "Time: nan\nPerformance: 10 TFLOPS", text.replace("2.5", "0"), ""):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                op.parse("fa_fp16_forward", bad, self.root)

    def test_fp8_csv_and_filter_statistic(self):
        row = dict(op="mha_fixlen_fwd", dtype="fp8_e4m3", output_dtype="fp16", batch_size=8,
                   seqlen_q=16384, seqlen_kv=16384, num_heads_q=16, num_heads_kv=2,
                   dim_qk=128, dim_v=128, is_causal=1, window_size_left=-1, window_size_right=0,
                   repeats=20, iterations=5, source="/prepared/extension.so", softmax_scale=.088,
                   softcap=0, samples_used=3, raw_tflops=95, tflops=100, mean_ms=2)
        path = self.root / "fp8.csv"
        def save(value):
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=value)
                writer.writeheader()
                writer.writerow(value)
        save(row)
        result, = op.parse("fa_fp8_forward", "100.00 TFLOPS, 2.0000 ms", self.root)
        self.assertEqual(result["metrics"][0]["statistic"], "filtered_mean")
        for key, value in (("samples_used", 0), ("samples_used", 5), ("batch_size", 1), ("raw_tflops", "nan")):
            save({**row, key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                op.parse("fa_fp8_forward", "100.00 TFLOPS, 2.0000 ms", self.root)
        save(row)
        with self.assertRaises(ValueError):
            op.parse("fa_fp8_forward", "100.00 TFLOPS, 3.0000 ms", self.root)

    def test_single_device_and_conflicting_overrides(self):
        args = SimpleNamespace(hcu_ids="6", nnodes=1, node_rank=0)
        env = op.environment("fa_fp16_stress", args, self.root)
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "6")
        self.assertEqual(env["FA_FWD_NO_MLS"], "1")
        for update in ({"HIP_VISIBLE_DEVICES": "0"}, {"ROCR_VISIBLE_DEVICES": "6"}, {"PYTHONOPTIMIZE": "1"}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                fixed_common.single_device(args, update)
        for key in ("FA_DEBUG", "PA_USE_CUDAGRAPH_BENCH", "PA_PARTITION_SIZE"):
            with patch.dict(os.environ, {key: ""}, clear=True), self.assertRaises(ValueError):
                op.environment("fa_fp16_stress", args, self.root)
        args.hcu_ids = "6,7"
        with self.assertRaises(ValueError):
            fixed_common.single_device(args, {})

    def test_probe_rejects_wrong_backend_or_multiple_visible_devices(self):
        record = 'BENCH_DEVICE {"visible_count": 1, "hip_version": "fixture"}'
        self.assertEqual(op.parse_probe("Import warning\n" + record)["visible_count"], 1)
        for text in ("", record + "\n" + record, record.replace('count": 1', 'count": 2')):
            with self.assertRaises(ValueError):
                op.parse_probe(text)
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                               get_device_properties=lambda _: SimpleNamespace(name="fixture", total_memory=1))
        torch = SimpleNamespace(version=SimpleNamespace(hip="fixture"), __version__="fixture", cuda=cuda)
        with patch.dict(sys.modules, {"torch": torch}):
            self.assertEqual(probe.inspect("blas_fp16_stress")["visible_count"], 1)
            cuda.device_count = lambda: 2
            with self.assertRaisesRegex(RuntimeError, "Exactly one"):
                probe.inspect("blas_fp16_stress")
            cuda.device_count = lambda: 1
            torch.version.hip = None
            with self.assertRaisesRegex(RuntimeError, "HIP"):
                probe.inspect("blas_fp16_stress")
            torch.version.hip = "fixture"
            with patch.object(Path, "exists", return_value=True), self.assertRaisesRegex(RuntimeError, "CUDA branch"):
                probe.inspect("pa_decode")

    def test_model_command_contracts(self):
        args = SimpleNamespace(numactl="numactl", numa_node=3, binary="migraphx-driver",
                               model="/models/with space.onnx", batch=32)
        for recipe in model.RECIPES:
            args.recipe = recipe
            argv = model.command(args)
            self.assertEqual(argv[:7], ["numactl", "--cpubind=3", "--membind=3", "migraphx-driver", "perf", "-n", "20"])
            self.assertEqual(argv[argv.index("--batch") + 1], "32")
            self.assertEqual("--opt-compile" in argv, recipe == "resnet50_fp16")
            self.assertEqual("--int8" in argv, recipe == "resnet50_int8")
            self.assertEqual(argv.count("--fill1"), 3 if recipe == "bert_large_fp16" else 0)
            if recipe == "bert_large_fp16":
                self.assertEqual(argv[9:15], ["--fill1", "input_ids", "--fill1", "attention_mask", "--fill1", "token_type_ids"])

    def test_model_parse_native_units_not_iteration_sum(self):
        for batch, rate, ms in ((1, 500, 2), (32, 4000, 8)):
            text = f"Batch size: {batch}\nRate: {rate} inferences/sec\nTotal time: {ms}ms (Min: 1ms, Max: 10ms)\n"
            result = model.parse(text, batch)
            self.assertEqual(result["latency_ms_batch"], ms)
            self.assertEqual(result["throughput_inf_s"], rate)
        text = "Rate: 500 inf/s\nTotal time: 2 ms/batch"
        self.assertEqual(model.parse(text, 1)["throughput_inf_s"], 500)
        for bad in (text * 2, text.replace("500", "nan"), text.replace("2 ms", "0 ms"),
                    text.replace("500", "1e999"), text.replace("2 ms", "40 ms"), "Rate: 500 inf/s"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                model.parse(bad, 1)


class FixedExecution(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.onnx = self.root / "model.onnx"
        self.onnx.write_bytes(b"fixture; shape inspection mocked for subprocess tests")
        self.args = SimpleNamespace(recipe="resnet50_fp16", batch=1, numa_node=0, model=str(self.onnx),
                                    binary=sys.executable, numactl=sys.executable, rocblas_dir="",
                                    round_timeout_s=10, hcu_ids="0", nnodes=1, node_rank=0,
                                    output_dir=str(self.root / "out"))

    def invoke(self, codes, stopped=None):
        commands = [[sys.executable, "-u", "-c", code] for code in codes]
        # The first call records the planned command, subsequent calls launch each round.
        sequence = [commands[0], *commands]
        capture = io.StringIO()
        with patch.object(model, "command", side_effect=sequence), patch.object(model, "inspect_model", return_value={}), \
                patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(capture):
            code = model.run(self.args, stopped or threading.Event())
        samples = [parse_sample(line) for line in capture.getvalue().splitlines() if line.startswith("BENCH_RESULT ")]
        return code, samples

    def test_complete_three_round_median_and_receipts(self):
        code, samples = self.invoke([f"print('Rate: {rate} inf/s\\nTotal time: {1000/rate} ms/batch')" for rate in (100, 1000, 200)])
        self.assertEqual(code, 0)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["metrics"][0]["value"], 200)
        self.assertEqual(samples[0]["metrics"][1]["value"], 5)
        self.assertEqual(samples[0]["correctness"], "not_checked")
        out = Path(self.args.output_dir)
        execution = json.loads((out / "execution.json").read_text())
        self.assertEqual(len(execution), 3)
        self.assertTrue(all(e["cleanup"] == "confirmed" and not e["termination_actions"] for e in execution))
        self.assertTrue((out / "rounds.csv").is_file())
        self.assertTrue((out / "round-3.log").is_file())
        with self.assertRaises(FileExistsError):
            self.invoke(["print('duplicate')"])

    def test_failed_second_round_never_publishes_partial_median(self):
        code, samples = self.invoke(["print('Rate: 500 inf/s\\nTotal time: 2 ms/batch')", "raise SystemExit(7)"])
        self.assertEqual(code, 2)
        self.assertEqual([s["measurement_status"] for s in samples], ["invalid"])
        out = Path(self.args.output_dir)
        self.assertEqual(len(json.loads((out / "rounds.json").read_text())), 1)
        self.assertFalse((out / "measurements.json").exists())

    def test_timeout_after_plausible_output_is_invalid_and_cleaned(self):
        self.args.round_timeout_s = .3
        code, samples = self.invoke(["import time; print('Rate: 500 inf/s\\nTotal time: 2 ms/batch'); time.sleep(30)"])
        self.assertEqual(code, 2)
        self.assertEqual(samples[0]["measurement_status"], "invalid")
        execution = json.loads((Path(self.args.output_dir) / "execution.json").read_text())[0]
        self.assertTrue(execution["timed_out"])
        self.assertEqual(execution["cleanup"], "confirmed")
        self.assertTrue(execution["termination_actions"])

    def test_pre_cancel_never_launches(self):
        stopped = threading.Event()
        stopped.set()
        with patch("subprocess.Popen", side_effect=AssertionError("must not launch")):
            code, samples = self.invoke(["pass"], stopped)
        self.assertEqual(code, 2)
        self.assertEqual(samples[0]["measurement_status"], "invalid")

    def test_uncertain_cleanup_uses_reserved_exit(self):
        with patch.object(model, "execute", side_effect=model.CleanupError("fixture cleanup uncertain")):
            code, samples = self.invoke(["pass"])
        self.assertEqual(code, 125)
        self.assertEqual(samples, [])
        self.assertEqual(json.loads((Path(self.args.output_dir) / "cleanup.json").read_text())["cleanup"], "unconfirmed")

    def test_bert_library_environment_and_hashes(self):
        self.args.recipe = "bert_large_fp16"
        with self.assertRaisesRegex(ValueError, "rocblas_dir"):
            model.validate(self.args, {})
        lib = self.root / "rocblas/lib"
        lib.mkdir(parents=True)
        (lib / "fixture.so").write_bytes(b"library fixture")
        self.args.rocblas_dir = str(lib.parent)
        env = model.validate(self.args, {"LD_LIBRARY_PATH": "/original"})
        self.assertEqual(env["LD_LIBRARY_PATH"], str(lib.resolve()) + os.pathsep + "/original")
        files = model.tuning_fingerprint(str(lib.parent), threading.Event())
        self.assertEqual(files[0]["sha256"], model.sha256(lib / "fixture.so"))

    def test_operator_wrapper_publishes_only_complete_success(self):
        for index, code in enumerate((
                "print('Pressure Test: 1000000/1000000 PASS')",
                "print('Pressure Test: 999999/1000000 PASS')",
                "print('Pressure Test: 1000000/1000000 PASS'); raise SystemExit(7)")):
            args = SimpleNamespace(recipe="blas_fp16_stress", native_timeout_s=10, hcu_ids="0",
                                   nnodes=1, node_rank=0, output_dir=str(self.root / f"op-{index}"))
            capture = io.StringIO()
            with patch.object(op, "command", return_value=[sys.executable, "-u", "-c", code]), \
                    patch.object(op, "probe_command", return_value=[sys.executable, "-c",
                        "print('BENCH_DEVICE {\"visible_count\": 1, \"hip_version\": \"fixture\"}')"]), \
                    patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(capture):
                result = op.run(args, threading.Event())
            samples = [parse_sample(line) for line in capture.getvalue().splitlines() if line.startswith("BENCH_RESULT ")]
            self.assertEqual(result, 0 if index == 0 else 2)
            self.assertEqual(samples[0]["measurement_status"], "measured" if index == 0 else "invalid")
            self.assertEqual(samples[0]["correctness"], "not_checked")

    def test_existing_framework_reports_direct_model_pack_without_service(self):
        fixture = self.root / "fixture.py"
        fixture.write_text(
            "import os,sys\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from testpacks.migraphx import run as m\n"
            "m.inspect_model=lambda args: {}\n"
            "m.command=lambda args: [sys.executable, '-c', \"print('Rate: 500 inf/s\\\\nTotal time: 2 ms/batch')\"]\n"
            "raise SystemExit(m.main())\n", encoding="utf-8")
        manifest = yaml.safe_load((ROOT / "testpacks/migraphx/manifest.yaml").read_text(encoding="utf-8"))
        manifest["command"][:3] = [sys.executable, "-u", str(fixture)]
        pack = self.root / "manifest.yaml"
        pack.write_text(yaml.safe_dump(manifest), encoding="utf-8")
        data = dict(version=1, name="cpu-fixture-not-hardware", output_dir="runs",
                    nodes={"node": dict(executor="local", hcus=[0])},
                    tests={"fixed": dict(suite="e2e_perf", adapter="command", pack=str(pack),
                       nodes=["node"], timeout_s=45, params=dict(bench_root=str(ROOT), recipe="resnet50_fp16",
                       batch=1, model=str(self.onnx), numa_node=0, binary=sys.executable, numactl=sys.executable))})
        config = self.root / "config.yaml"
        config.write_text(yaml.safe_dump(data), encoding="utf-8")
        plan = make_plan(str(config))
        store = RunStore.create(plan)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(run_plan(plan, store, quiet=True), 0)
        report = build_report(store)
        self.assertEqual(report["case_counts"], {"completed": 1})
        self.assertEqual(report["metrics"][0]["source_statistic"], "median_of_3_rounds")
        self.assertEqual(report["metrics"][0]["mean_of_reported_values"], 500)
        # Wrapper files survive transport to the controller's artifact directory.
        self.assertTrue(list(store.path.rglob("rounds.csv")))


@unittest.skipUnless(importlib.util.find_spec("onnx"), "optional ONNX input metadata checks require onnx")
class ModelInputs(unittest.TestCase):
    def test_shapes_types_batch_and_external_weights(self):
        import onnx
        from onnx import TensorProto, helper
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            args = SimpleNamespace(recipe="resnet50_fp16", batch=32, model=str(path))
            def save(inputs, initializers=()):
                graph = helper.make_graph([], "fixture", inputs, [], initializer=list(initializers))
                onnx.save(helper.make_model(graph), path)
            save([helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 224, 224])])
            self.assertEqual(model.inspect_model(args)["input"]["shape"], [32, 3, 224, 224])
            for shape in ([1, 3, 224, 224], ["batch", 3, 256, 256]):
                save([helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)])
                with self.assertRaises(ValueError):
                    model.inspect_model(args)
            args.recipe = "bert_large_fp16"
            names = ["input_ids", "attention_mask", "token_type_ids"]
            save([helper.make_tensor_value_info(n, TensorProto.INT64, ["batch", 384]) for n in names])
            self.assertEqual(set(model.inspect_model(args)), set(names))
            save([helper.make_tensor_value_info(n, TensorProto.INT32, ["batch", 384]) for n in names])
            with self.assertRaises(ValueError):
                model.inspect_model(args)
            tensor = TensorProto(name="external", data_type=TensorProto.FLOAT, dims=[1], data_location=TensorProto.EXTERNAL)
            entry = tensor.external_data.add()
            entry.key, entry.value = "location", "missing.data"
            save([helper.make_tensor_value_info(n, TensorProto.INT64, ["batch", 384]) for n in names], [tensor])
            with self.assertRaisesRegex(ValueError, "External ONNX"):
                model.inspect_model(args)


if __name__ == "__main__":
    unittest.main()
