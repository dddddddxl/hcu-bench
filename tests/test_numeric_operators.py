"""Conv/GEMM parsing and execution contracts; fixtures are not hardware results."""
import contextlib
import io
import json
import sys
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml
from hcu_bench.planner import make_plan
from hcu_bench.runner import run_plan
from hcu_bench.store import RunStore
from hcu_bench.report import build_report
from testpacks.operators.numeric import run as numeric

ROOT = Path(__file__).resolve().parents[1]
GEMM = "transA,transB,M,N,K,lda,ldb,ldc,rocblas-Gflops,us,norm_error_1,norm_error_2\nN,N,65,33,33,128,128,128,123.5,17.2,0,0\n"


class NumericTests(unittest.TestCase):
    def test_complex_scalars_and_explicit_native_ld_adjustments(self):
        text = 'transA,transB,M,N,K,lda,ldb,ldc,alpha,beta,rocblas-Gflops,us\nC,N,65,33,33,128,128,128,(-0.5,0.5),(-0.5,0.5),10,2\n'
        self.assertEqual(numeric.parse_gemm(text, True, '')['native']['alpha'], '(-0.5,0.5)')
        expected = dict(dtype='fp32', function='gemm', transA='N', transB='N', M='65', N='33', K='33',
                        lda='32', ldb='128', ldc='128')
        text = GEMM
        with self.assertRaises(ValueError):
            numeric.parse_gemm(text, True, '', expected)
        text = 'rocblas-bench INFO: lda < min_lda, setting lda = 128\n' + text
        result = numeric.parse_gemm(text, True, '', expected)
        self.assertEqual(result['native_adjustments']['lda'], dict(requested='32', native='128'))

    def test_native_csv_coverage(self):
        for dtype in ("fp32", "fp16"):
            args = SimpleNamespace(recipe="conv", dtype=dtype, conv_suite="all", direction="all", group="all", case_id="")
            _, cases = numeric.selected(args, numeric.native_module("conv"), "/bin/driver")
            self.assertEqual(len(cases), 28)
            self.assertEqual({c["direction"] for c in cases}, {"1", "2", "4"})
        for dtype, count in dict(fp32=2207, fp64=819, fp16=27, complex64=27, complex128=27).items():
            args = SimpleNamespace(recipe="gemm", dtype=dtype, conv_suite="all", direction="all", group="all",
                                   case_id="", warmup=0, iters=0, gemm_profile="new", max_norm_error="")
            _, cases = numeric.selected(args, numeric.native_module("gemm"), "/bin/bench")
            self.assertEqual(len(cases), count)
            self.assertTrue({c["row"]["group"] for c in cases} <= {"general", "smoke"})

    def test_conv_verdicts_and_native_timer(self):
        for direction, label in numeric.VERIFY_LABEL.items():
            verdict = numeric.parse_conv(f"{label} Verifies OK on CPU reference (0 < 1.5e-6)", direction, True)
            self.assertEqual(verdict["reference"], "CPU")
            timer = f"GPU Kernel Time {numeric.CONV_LABEL[direction]}. Elapsed: 2.500 ms (average)"
            self.assertEqual(numeric.parse_conv(timer, direction, False)["latency_ms"], 2.5)
            for bad in ("", timer + "\n" + timer, timer.replace("2.500", "nan")):
                with self.assertRaises(ValueError):
                    numeric.parse_conv(bad, direction, False)
        with self.assertRaises(numeric.VerificationError):
            numeric.parse_conv("Forward Convolution FAILED: 0.1 > 1e-6", "1", True)
        with self.assertRaises(ValueError):
            numeric.parse_conv("Everything looks good; exit 0", "1", True)

    def test_gemm_native_unit_check_and_optional_norm_threshold(self):
        self.assertEqual(numeric.parse_gemm(GEMM, True, "")["tolerance"], None)
        nonzero = GEMM.replace(",17.2,0,0", ",17.2,0.0001,0.0002")
        self.assertIsNone(numeric.parse_gemm(nonzero, True, "")["tolerance"])
        self.assertEqual(numeric.parse_gemm(nonzero, True, "0.001")["tolerance"], .001)
        with self.assertRaises(numeric.VerificationError):
            numeric.parse_gemm(nonzero, True, "0.00001")
        for text in ("", GEMM + GEMM, GEMM.replace(",17.2,0,0", ",nan,0,0"), GEMM + "\nERROR: mismatch"):
            with self.assertRaises(ValueError):
                numeric.parse_gemm(text, True, "")
        self.assertEqual(numeric.parse_gemm("rocblas-Gflops,us\n10,1\n", True, "")["norm_errors"], {})
        with self.assertRaises(ValueError):
            numeric.parse_gemm("rocblas-Gflops,us\n10,1\n", True, "0")

    def test_gemm_unit_failure_cannot_be_overridden_by_norm_tolerance(self):
        text = ("unit.hpp:159: Failure\nExpected equality of these values:\n"
                "  float(hCPU[i + idx])\n    Which is: -0.2800293\n"
                "  float(hGPU[i + idx])\n    Which is: -0.27978516\n"
                + GEMM.replace(",17.2,0,0", ",17.2,6.13645e-05,6.13645e-05"))
        for check, threshold in ((True, ""), (True, "0.0001"), (False, "")):
            with self.subTest(check=check, threshold=threshold), self.assertRaisesRegex(
                    numeric.VerificationError, "unit/equality check failed"):
                numeric.parse_gemm(text, check, threshold)
        code, commands, result = self.run_fixture(text, threshold="0.0001")
        self.assertEqual(code, 2)
        self.assertEqual(len(commands), 3)  # CLI/device probes and check; no performance rounds.
        self.assertEqual(result["results"][0]["correctness"], "failed")
        self.assertEqual(result["results"][0]["metrics"], [])

    def run_fixture(self, check, profile="new", cli_error=False, threshold=""):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(recipe="gemm", dtype="fp16", mode="perf", group="smoke", case_id="dtk-smoke-59-0001",
                device="auto", driver="auto", rocblas_bench="auto", conv_suite="all", direction="all",
                warmup=10, iters=100, repeats=3, native_timeout_s=60, max_norm_error=threshold, hcu_ids="0,1",
                nnodes=1, node_rank=0, output_dir=folder, gemm_profile=profile)
            commands = []
            def execute(argv, name, out, env, stopped, executions, timeout_s):
                commands.append(argv)
                self.assertEqual(env["HIP_VISIBLE_DEVICES"], "0")
                if name == "cli-probe":
                    report = Path(out) / "cli" / "fixture" / "cli_check.json"
                    report.parent.mkdir(parents=True)
                    report.write_text(json.dumps(dict(profile=profile, rocblas_version="fixture",
                        check_semantics="fixture only", cli_compatible=not cli_error)))
                    if cli_error:
                        raise RuntimeError("cli-probe: incomplete native execution")
                    return "CLI fixture"
                if name == "device-probe":
                    return 'BENCH_DEVICE {"visible_count":1,"pci_bus":"0000:01:00.0"}'
                return check if name.endswith("check") else GEMM
            with patch.object(numeric, "executable", return_value=__file__), patch.object(numeric, "execute", side_effect=execute), \
                    patch.dict(numeric.os.environ, {"HIP_VISIBLE_DEVICES": "0,1"}), contextlib.redirect_stdout(io.StringIO()):
                code = numeric.run(args, threading.Event())
            return code, commands, json.loads((Path(folder) / "assessment.json").read_text())

    def test_perf_runs_check_before_three_native_rounds(self):
        code, commands, result = self.run_fixture(GEMM)
        self.assertEqual(code, 0)
        self.assertEqual(len(commands), 6)
        self.assertIn("--verify", commands[2])
        self.assertEqual(commands[2][commands[2].index("--verify") + 1], "0")
        self.assertTrue(result["complete"])
        self.assertEqual(result["results"][0]["metrics"][0]["statistic"], "median_of_native_means")

    def test_additional_norm_failure_never_publishes_perf(self):
        code, commands, result = self.run_fixture(GEMM.replace(",17.2,0,0", ",17.2,0.1,0.1"), threshold="0.001")
        self.assertEqual(code, 2)
        self.assertEqual(len(commands), 3)
        self.assertEqual(result["results"][0]["metrics"], [])
        self.assertEqual(result["results"][0]["correctness"], "failed")

    def test_template_has_seven_dtype_sweeps(self):
        plan = make_plan(str(ROOT / "configs/numeric-operators.template.yaml"), "operators",
                         ["tests.gemm.enabled=true"])
        self.assertEqual(len(plan.cases), 7)
        self.assertEqual({(c.params["recipe"], c.params["dtype"]) for c in plan.cases},
                         {("conv", "fp32"), ("conv", "fp16"), ("gemm", "fp32"), ("gemm", "fp16"), ("gemm", "fp64"), ("gemm", "complex64"), ("gemm", "complex128")})

    def test_wrong_shape_is_not_published(self):
        code, commands, result = self.run_fixture(GEMM.replace("N,N,65,", "N,N,2048,"))
        self.assertEqual(code, 2)
        self.assertEqual(len(commands), 3)
        self.assertIn("requested M", result["results"][0]["reason"])
        self.assertEqual(result["results"][0]["metrics"], [])

    def test_framework_report_and_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            launcher = root / "fixture.py"
            launcher.write_text(
                "import sys\n"
                f"sys.path.insert(0, {str(ROOT)!r})\n"
                "from testpacks.operators.numeric import run as numeric\n"
                "from native_common import write_json\n"
                "def execute(argv, name, out, env, stopped, executions, timeout_s):\n"
                "    if name == 'cli-probe':\n"
                "        (out / 'cli' / 'fixture').mkdir(parents=True)\n"
                "        write_json(out / 'cli' / 'fixture' / 'cli_check.json', dict(profile='new', rocblas_version='fixture', check_semantics='fixture only'))\n"
                "        return 'CLI fixture'\n"
                f"    text = 'BENCH_DEVICE {{\"visible_count\":1,\"pci_bus\":\"0000:01:00.0\"}}' if name == 'device-probe' else {GEMM!r}\n"
                "    (out / (name + '.log')).write_text(text)\n"
                "    return text\n"
                "numeric.execute = execute\n"
                "numeric.executable = lambda value, recipe: sys.executable\n"
                "raise SystemExit(numeric.main())\n", encoding="utf-8")
            pack = yaml.safe_load((ROOT / "testpacks/operators/numeric/manifest.yaml").read_text())
            pack["command"][0], pack["command"][2] = sys.executable, str(launcher)
            (root / "pack.yaml").write_text(yaml.safe_dump(pack), encoding="utf-8")
            data = dict(version=1, name="synthetic-numeric-contract", output_dir="runs",
                        nodes=dict(local=dict(executor="local", hcus=[0], env=dict(HIP_VISIBLE_DEVICES="0"))),
                        tests=dict(gemm=dict(suite="operators", adapter="command", pack="pack.yaml", nodes=["local"],
                                   timeout_s=60, params=dict(bench_root=str(ROOT), recipe="gemm", case_id="dtk-smoke-59-0001"))))
            config = root / "bench.yaml"
            config.write_text(yaml.safe_dump(data), encoding="utf-8")
            plan = make_plan(str(config))
            store = RunStore.create(plan)
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_plan(plan, store, quiet=True)
            self.assertEqual(code, 0)
            report = build_report(store)
            self.assertEqual(len(report["metrics"]), 2)
            self.assertTrue(all(m["suite"] == "operators" for m in report["metrics"]))
            self.assertEqual(len(list((store.path / "raw").rglob("*-measurements.csv"))), 1)
            records = [json.loads(line) for line in (store.path / "artifacts.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(records and all(r["status"] == "collected" for r in records))

    def test_incompatible_cli_stops_before_device_or_gemm(self):
        code, commands, result = self.run_fixture(GEMM, cli_error=True)
        self.assertEqual(code, 2)
        self.assertEqual(len(commands), 1)
        self.assertIn("check_gemm_env.py", commands[0][1])
        self.assertIn("cli-probe", result["error"])

    def test_explicit_legacy_preserves_verify_and_profile(self):
        code, commands, result = self.run_fixture(GEMM, profile="legacy")
        self.assertEqual(code, 0)
        self.assertIn("legacy", commands[0])
        for argv in commands[2:]:
            self.assertNotIn("--test", argv)
            self.assertNotIn("--use_hipblaslt", argv)
            self.assertIn("--verify", argv)
        self.assertEqual(result["results"][0]["gemm_profile"], "legacy")

    def test_profile_is_forwarded_by_manifest(self):
        plan = make_plan(str(ROOT / "configs/numeric-operators.template.yaml"), "operators",
                         ["tests.conv.enabled=false", "tests.gemm.enabled=true", "tests.gemm.params.gemm_profile=legacy"])
        self.assertTrue(all(c.params["gemm_profile"] == "legacy" for c in plan.cases))
