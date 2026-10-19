"""Public templates and Bash entrypoint contracts; never launch GPU workloads."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from hcu_bench.cli import preview
from hcu_bench.config import load_document
from hcu_bench.models import BenchError, SUITES
from hcu_bench.planner import make_plan

ROOT = Path(__file__).resolve().parents[1]


class PublicTemplateTests(unittest.TestCase):
    def test_every_config_previews_without_starting_processes(self):
        for path in sorted((ROOT / "configs").glob("*.yaml")):
            if path.name == "bench.yaml":
                continue
            with self.subTest(config=path.name), patch("subprocess.Popen", side_effect=AssertionError("static only")):
                plan = make_plan(str(path))
                rendered = preview(plan)
                self.assertEqual(rendered["case_count"], len(plan.cases))
                for case in rendered["cases"]:
                    for spec in case["commands"].values():
                        self.assertFalse(any("${" in arg for arg in spec["argv"]))

    def test_remote_template_uses_native_deepep_on_both_nodes(self):
        plan = make_plan(str(ROOT / "configs/remote.template.yaml"))
        case = plan.cases[0]
        self.assertEqual(case.pack["id"], "deepep.low-latency-pressure")
        self.assertEqual(case.nodes, ["node0", "node1"])
        self.assertEqual(case.duration_s, 600)
        rendered = preview(plan)["cases"][0]["commands"]
        for rank, node in enumerate(case.nodes):
            argv = rendered[node]["argv"]
            self.assertIn("/YOUR_CONTAINER_BENCH_REPO/testpacks/deepep/run.py", argv)
            self.assertEqual(argv[argv.index("--nnodes") + 1], "2")
            self.assertEqual(argv[argv.index("--node-rank") + 1], str(rank))
            self.assertEqual(argv[argv.index("--num-processes") + 1], "4")

    def test_multi_model_template_keeps_models_and_profiles_separate(self):
        plan = make_plan(str(ROOT / "configs/e2e-multi-model.template.yaml"), "e2e")
        self.assertEqual([c.test_id for c in plan.cases], ["model_a_gsm8k", "model_b_gsm8k"])
        for field in ("model_path", "serving", "port"):
            self.assertNotEqual(plan.cases[0].params[field], plan.cases[1].params[field])
        self.assertTrue(all(c.nodes == ["node0"] and c.params["min_score"] == "" for c in plan.cases))

    def test_accuracy_rejects_multinode_and_framework_duration_before_launch(self):
        config = str(ROOT / "configs/in-container.template.yaml")
        # Add valid configuration fields so validation exercises the manifest limits.
        data = load_document(Path(config))
        data["nodes"]["node1"] = dict(data["nodes"]["node0"])
        test = data["tests"]["sglang_gsm8k"]
        test["pack"] = str(ROOT / "testpacks/sglang_accuracy/manifest.yaml")
        data["tests"] = {"sglang_gsm8k": test}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "test.yaml"
            for changes, message in (({"nodes": ["node0", "node1"]}, "at most 1"),
                                     ({"duration_s": 10}, "duration_control")):
                changed = {**data, "tests": {"sglang_gsm8k": {**test, **changes}}}
                path.write_text(yaml.safe_dump(changed), encoding="utf-8")
                with self.subTest(changes=changes), self.assertRaisesRegex(BenchError, message):
                    make_plan(str(path), "e2e")

    def test_readme_relative_file_links_exist(self):
        for document in (ROOT / "README.md", ROOT / "docs/configuration.md"):
            for target in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
                if "://" not in target and not target.startswith("#"):
                    with self.subTest(document=document.name, target=target):
                        self.assertTrue((document.parent / target.split("#")[0]).is_file())


class BashEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.bash = shutil.which("bash")
        if not self.bash and os.name == "nt":
            candidate = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
            if candidate.is_file():
                self.bash = str(candidate)
        if not self.bash:
            self.skipTest("Bash is required for launcher integration tests")
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repo = self.root / "repo with spaces"
        (self.repo / "configs").mkdir(parents=True)
        shutil.copyfile(ROOT / "bench.sh", self.repo / "bench.sh")
        self.env = {**os.environ, "BENCH_PYTHON": Path(sys.executable).as_posix(),
                    "PYTHONPATH": str(ROOT / "src")}

    def invoke(self, *args):
        return subprocess.run([self.bash, (self.repo / "bench.sh").as_posix(), *args],
                              cwd=self.root, env=self.env, capture_output=True,
                              text=True, encoding="utf-8", timeout=60)

    def test_default_yaml_all_runs_every_configured_demo_suite_from_other_cwd(self):
        shutil.copyfile(ROOT / "configs/demo.yaml", self.repo / "configs/bench.yaml")
        result = self.invoke("all", "--quiet")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        reports = list((self.repo / "runs").glob("*/summary.json"))
        self.assertEqual(len(reports), 1)
        summary = json.loads(reports[0].read_text(encoding="utf-8"))
        self.assertEqual({m["suite"] for m in summary["metrics"]}, set(SUITES))
        self.assertTrue(all(m["simulated"] for m in summary["metrics"]))
        self.assertEqual(summary["case_counts"], {"completed": 8})

    def test_explicit_yaml_and_selected_suites(self):
        config = self.repo / "configs/custom node.yaml"
        shutil.copyfile(ROOT / "configs/demo.yaml", config)
        result = self.invoke(config.as_posix(), "pcie,memory", "--quiet")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = next((self.repo / "runs").glob("*/summary.json"))
        summary = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual({m["suite"] for m in summary["metrics"]}, {"pcie", "memory"})

    def test_missing_default_config_does_not_start_run(self):
        result = self.invoke("all")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Config not found", result.stderr)
        self.assertFalse((self.repo / "runs").exists())


if __name__ == "__main__":
    unittest.main()
