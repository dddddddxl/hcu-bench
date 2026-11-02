# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# HCU Bench integration changes; original user-supplied sources retained.
"""Profile regressions against native CLI captures and the original dry-run commands."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
sys.dont_write_bytecode = True
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import check_gemm_env as cli
import run_gemm_cases as suite


def probes(version):
    data = json.loads((ROOT / "validation" / ("rocblas-%s-native-cli.json" % version)).read_text(encoding="utf-8"))
    return {r["argv"][-1]: r for r in data["results"] if r["argv"][-1] in ("--version", "--help")}


class Profiles(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(cases=ROOT / "gemm_cases.csv", group="all", case_id=None,
                                    bench="./bin/rocblas-bench", profile="new", warmup=10, iters=100)
        self.rows = suite.load_cases(self.args)

    def test_profiles_match_their_native_interfaces(self):
        for profile, version in (("new", "5.2.0"), ("legacy", "4.3.0")):
            with self.subTest(profile=profile):
                report = cli.assess_interface(probes(version), cli.required_options(profile), profile)
                self.assertTrue(report["cli_compatible"], report["errors"])
                self.assertEqual(report["missing_options"], [])

    def test_crossed_profiles_reject_changed_verify_semantics(self):
        for profile, version in (("new", "4.3.0"), ("legacy", "5.2.0")):
            with self.subTest(profile=profile):
                report = cli.assess_interface(probes(version), cli.required_options(profile), profile)
                self.assertFalse(report["cli_compatible"])
                self.assertTrue(any("requires rocBLAS" in e for e in report["errors"]))
        old_with_new = cli.assess_interface(probes("4.3.0"), cli.required_options("new"), "new")
        self.assertEqual(old_with_new["missing_options"], ["--test", "--use_hipblaslt"])

    def test_new_recipe_matches_original_diagnostic_commands(self):
        reference = json.loads((ROOT / "validation" / "new_recipe_reference.json").read_text(encoding="utf-8"))
        for self.args.dtype in suite.TYPES:
            for self.args.mode in ("check", "perf"):
                rounds = 1 if self.args.mode == "check" else 3
                actual = [suite.command(self.args, row)[1:] for row in self.rows for _ in range(rounds)]
                self.assertEqual(actual, reference["commands"][self.args.dtype + "-" + self.args.mode])

    def test_legacy_retains_every_shared_setting(self):
        for self.args.dtype in suite.TYPES:
            for self.args.mode in ("check", "perf"):
                for row in self.rows:
                    self.args.profile = "new"
                    argv = suite.command(self.args, row)
                    new = dict(zip(argv[1::2], argv[2::2]))
                    self.args.profile = "legacy"
                    argv = suite.command(self.args, row)
                    old = dict(zip(argv[1::2], argv[2::2]))
                    self.assertEqual(set(new) - set(old), {"--test", "--use_hipblaslt"})
                    self.assertEqual(set(old) - set(new), set())
                    self.assertEqual(old, {key: new[key] for key in old})
                    self.assertEqual(old["--verify"], "1" if self.args.mode == "check" else "0")

    def test_default_and_legacy_entry_points(self):
        options = ["--dtype", "fp16", "--mode", "check", "--case-id", "square-1024", "--dry-run"]
        default = subprocess.run([sys.executable, str(ROOT / "run_gemm_cases.py")] + options,
                                 capture_output=True, text=True, check=True).stdout
        explicit = subprocess.run([sys.executable, str(ROOT / "run_gemm_cases.py"), "--profile", "legacy"] + options,
                                  capture_output=True, text=True, check=True).stdout
        legacy = subprocess.run([sys.executable, str(ROOT / "run_gemm_cases_legacy.py")] + options,
                                capture_output=True, text=True, check=True).stdout
        self.assertTrue(default.startswith("Profile new;"))
        self.assertIn("--test 1", default)
        self.assertTrue(legacy.startswith("Profile legacy;"))
        self.assertEqual(explicit, legacy)
        self.assertNotIn("--test", legacy)
        self.assertNotIn("--use_hipblaslt", legacy)

    def test_published_raw_commands_match_the_runner(self):
        for self.args.profile in ("new", "legacy"):
            for self.args.dtype in suite.TYPES:
                for self.args.mode in ("check", "perf"):
                    path = ROOT / "commands" / "raw" / ("%s_%s_%s.txt" % (self.args.profile, self.args.dtype, self.args.mode))
                    actual = [shlex.split(line)[1:] for line in path.read_text(encoding="utf-8").splitlines()
                              if line.startswith("HIP_VISIBLE_DEVICES=")]
                    rounds = 1 if self.args.mode == "check" else 3
                    expected = [suite.command(self.args, row) for row in self.rows for _ in range(rounds)]
                    self.assertEqual(actual, expected)
                    self.assertEqual(len(actual), 14 * rounds)


if __name__ == "__main__":
    unittest.main()
