"""Default selection contracts only; no hardware performance claims."""
from pathlib import Path
import unittest
from unittest.mock import patch

import yaml

from hcu_bench.cli import preview
from hcu_bench.planner import make_plan

ROOT = Path(__file__).resolve().parents[1]


class SglangDefaultsTests(unittest.TestCase):
    def test_rccl_inherits_fixed_16g_without_yaml_bounds(self):
        for name in ("in-container.template.yaml", "managed-container.template.yaml",
                     "rccl.template.yaml", "nmz6.sglang-smoke.yaml"):
            with self.subTest(config=name):
                path = ROOT / "configs" / name
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                for test in data["tests"].values():
                    if test["suite"] == "rccl":
                        self.assertNotIn("min_bytes", test["params"])
                        self.assertNotIn("max_bytes", test["params"])
                with patch("subprocess.Popen", side_effect=AssertionError("static only")):
                    plan = make_plan(str(path), "rccl")
                    self.assertEqual(len(plan.cases), 10)
                    self.assertEqual(preview(plan)["case_count"], 10)
                for case in plan.cases:
                    self.assertEqual(case.params["min_bytes"], 16 * 1024 ** 3)
                    self.assertEqual(case.params["max_bytes"], 16 * 1024 ** 3)

    def test_incompatible_native_cases_are_opt_in(self):
        for name in ("in-container.template.yaml", "managed-container.template.yaml",
                     "memory.template.yaml", "migraphx.template.yaml", "resnet50.template.yaml",
                     "bert.template.yaml", "operators.template.yaml", "numeric-operators.template.yaml",
                     "nmz6.sglang-smoke.yaml"):
            with self.subTest(config=name):
                data = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
                for test in data["tests"].values():
                    recipe = test.get("params", {}).get("recipe")
                    if test["suite"] in ("memory", "migraphx") or recipe in ("gemm", "fa_fp8_forward"):
                        self.assertIs(test["enabled"], False)
                    self.assertNotIn("fa_fp8_forward", test.get("matrix", {}).get("recipe", []))

    def test_combined_templates_skip_disabled_tests_but_allow_opt_in(self):
        for name in ("in-container.template.yaml", "managed-container.template.yaml"):
            path = str(ROOT / "configs" / name)
            plan = make_plan(path)
            self.assertEqual(len(plan.cases), 16)
            self.assertNotIn("memory", {case.suite for case in plan.cases})
            self.assertFalse(any(case.params.get("recipe") == "gemm" for case in plan.cases))
            enabled = make_plan(path, overrides=["tests.memory_bandwidth.enabled=true", "tests.gemm.enabled=true"])
            self.assertEqual(len(enabled.cases), 20)
            self.assertEqual(sum(case.suite == "memory" for case in enabled.cases), 1)
            self.assertEqual(sum(case.params.get("recipe") == "gemm" for case in enabled.cases), 3)

    def test_nmz6_keeps_supported_attention_and_short_stress(self):
        plan = make_plan(str(ROOT / "configs/nmz6.sglang-smoke.yaml"))
        self.assertEqual(len(plan.cases), 21)
        self.assertTrue(all(case.adapter == "command" for case in plan.cases))
        recipes = {case.params.get("recipe") for case in plan.cases}
        self.assertTrue({"pa_decode", "fa_fp16_forward"} <= recipes)
        self.assertNotIn("fa_fp8_forward", recipes)
        self.assertTrue(all(case.params["stress_iterations"] == 100
                            for case in plan.cases if case.test_id == "operator_stress_smoke"))
        self.assertFalse({"memory", "migraphx", "mooncake"} & {case.suite for case in plan.cases})


if __name__ == "__main__":
    unittest.main()
