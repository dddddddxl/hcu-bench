"""Dependency-only fixtures; no model, native GPU work or package installation."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from testpacks.migraphx import dependencies as dep
from testpacks.migraphx import check as cli
from hcu_bench.planner import make_plan

ROOT = Path(__file__).resolve().parents[1]


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.driver = self.root / "migraphx-driver"
        self.driver.write_bytes(b"fixture")
        self.driver.chmod(0o755)

    def test_path_driver_precedes_dtk_fallback(self):
        with patch.object(dep.shutil, "which", return_value=str(self.driver)):
            self.assertEqual(dep.executable(dep.DRIVER), str(self.driver.resolve()))

    def test_prepared_dtk_driver_found_when_path_is_incomplete(self):
        with patch.object(dep.shutil, "which", return_value=None), patch.object(dep, "DRIVER_LOCATIONS", (str(self.driver),)):
            self.assertEqual(dep.executable(dep.DRIVER, {"PATH": ""}), str(self.driver.resolve()))

    def test_explicit_override_never_falls_back(self):
        with patch.object(dep.shutil, "which", return_value=None), patch.object(dep, "DRIVER_LOCATIONS", (str(self.driver),)):
            with self.assertRaisesRegex(ValueError, "missing"):
                dep.executable("/chosen/missing-driver")

    def test_missing_driver_explains_native_dependency_and_reference_image(self):
        with patch.object(dep.shutil, "which", return_value=None), patch.object(dep, "DRIVER_LOCATIONS", ()):
            with self.assertRaisesRegex(ValueError, "Python dependencies, not the native driver"):
                dep.executable(dep.DRIVER)

    def test_resnet_template_selects_only_four_resnet_cases(self):
        plan = make_plan(str(ROOT / "configs/resnet50.template.yaml"), "migraphx")
        self.assertEqual(len(plan.cases), 4)
        self.assertEqual({case.params["recipe"] for case in plan.cases}, {"resnet50_fp16", "resnet50_int8"})
        self.assertTrue(all(case.container is None for case in plan.cases))


class DependencyCheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.driver = self.root / "migraphx-driver"
        self.driver.write_bytes(b"\x7fELFfixture")
        self.help = "--onnx --batch --gpu --fp16 --int8 --opt-compile --fill1"
        self.ldd = "libmigraphx.so => /opt/dtk/lib/libmigraphx.so (0x1)\n"
        self.help_exit = 0
        self.version = "MIGraphX Version: 5.2.0"
        self.commands = []

    def command(self, argv, **kwargs):
        self.commands.append(argv)
        if argv[0] == "/fixture/ldd":
            return subprocess.CompletedProcess(argv, 0, self.ldd)
        if argv[1:] == ["--version"]:
            return subprocess.CompletedProcess(argv, 0, self.version)
        self.assertEqual(argv[1:], ["perf", "--help"])
        return subprocess.CompletedProcess(argv, self.help_exit, self.help)

    def inspect(self, **kwargs):
        def which(name, **options):
            return {dep.DRIVER: str(self.driver), "numactl": "/fixture/numactl", "ldd": "/fixture/ldd"}.get(name)
        with patch.object(dep.shutil, "which", side_effect=which), \
             patch.object(dep.sys, "platform", "linux"), \
             patch.object(dep.importlib.util, "find_spec", return_value=object()), \
             patch.object(dep.subprocess, "run", side_effect=self.command):
            return dep.check(**kwargs)

    def test_ready_environment_checks_flags_and_libraries_without_model_run(self):
        result = self.inspect()
        self.assertTrue(result["ready"], result)
        self.assertTrue(result["libraries"]["checked"])
        self.assertIn("--opt-compile", result["required_flags"])
        self.assertEqual(len(self.commands), 3)

    def test_missing_shared_libraries_fail_before_driver_execution(self):
        self.ldd = "libhipnn.so => not found\nliborigami.so.1 => not found\n"
        result = self.inspect()
        self.assertFalse(result["ready"])
        self.assertIn("libhipnn.so", result["errors"][0])
        self.assertEqual(len(self.commands), 1)

    def test_fp16_requires_opt_compile_but_int8_does_not(self):
        self.help = self.help.replace("--opt-compile", "")
        self.assertFalse(self.inspect()["ready"])
        self.assertTrue(self.inspect(recipe="resnet50_int8")["ready"])

    def test_failed_help_does_not_pass_even_with_flag_text(self):
        self.help_exit = 127
        self.assertFalse(self.inspect()["ready"])

    def test_runtime_loader_error_cannot_pass_with_zero_exit(self):
        self.version = "load hsa library error\nMIGraphX Version: 5.2.0"
        result = self.inspect()
        self.assertFalse(result["ready"])
        self.assertIn("runtime loader failed", result["errors"][0])

    def test_missing_python_dependency_does_not_mask_native_discovery(self):
        with patch.object(dep.importlib.util, "find_spec", side_effect=lambda name: None if name == "onnx" else object()), \
             patch.object(dep, "executable", side_effect=[str(self.driver), "/fixture/numactl"]), \
             patch.object(dep, "shared_libraries", return_value={"checked": True}), \
             patch.object(dep.subprocess, "run", side_effect=self.command):
            result = dep.check()
        self.assertFalse(result["ready"])
        self.assertIn("Missing Python dependencies: onnx", result["errors"][0])
        self.assertEqual(result["version"], "MIGraphX Version: 5.2.0")

    def test_check_entrypoint_saves_report_and_returns_failure(self):
        report = {"ready": False, "errors": ["missing driver"]}
        output = self.root / "check.json"
        with patch.object(cli, "check", return_value=report), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["--output", str(output)]), 2)
        self.assertEqual(json.loads(output.read_text()), report)


if __name__ == "__main__":
    unittest.main()
