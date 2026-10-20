"""Preparation contracts with CPU-only fixtures; no DTK builds or HCU workloads."""
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from hcu_bench import agent
from hcu_bench.adapters import CommandAdapter
from hcu_bench.config import load_document, validate_config
from hcu_bench.executors.base import AGENT_PATH, AgentExecutor
from hcu_bench.models import BenchError
from hcu_bench.planner import make_plan
from hcu_bench.registry import resolve_params
from testpacks import prepare
from testpacks.native_common import prepared_binary, sha256, write_json
from testpacks.sglang_accuracy import run as accuracy

ROOT = Path(__file__).resolve().parents[1]


class BundledSourceTests(unittest.TestCase):
    def test_all_bundled_trees_match_recorded_checksums(self):
        manifest = json.loads((ROOT / "vendor/sources.json").read_text())
        self.assertEqual(set(manifest), {*prepare.SOURCES.values(), "deepep-tests", "gsm8k", "evalscope"})
        for name, entry in manifest.items():
            with self.subTest(source=name):
                self.assertEqual(prepare.tree_hash(ROOT / "vendor" / name), entry["tree_sha256"])
        self.assertEqual(accuracy.validate_dataset(ROOT / "vendor/gsm8k/test.jsonl", 0, 0), 1319)

    def test_source_hash_uses_case_sensitive_portable_order(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("a", "B"):
                (root / name).write_bytes(name.encode())
            digest = hashlib.sha256()
            for name in ("B", "a"):
                digest.update(name.encode())
                digest.update(hashlib.sha256(name.encode()).digest())
            self.assertEqual(prepare.tree_hash(root), digest.hexdigest())

    def test_pcie_local_headers_and_version_input_are_bundled(self):
        import re
        source = ROOT / "vendor/rocm-bandwidth-test"
        for path in [*source.glob("*.cpp"), *source.glob("*.hpp")]:
            for header in re.findall(r'#include\s+"([^"]+)"', path.read_text()):
                if header in ("hsa/hsa.h", "hsa/hsa_ext_amd.h"):
                    continue  # Runtime SDK headers come from the existing DTK.
                self.assertTrue((source / header).is_file(), "%s needs %s" % (path.name, header))
        self.assertTrue((source / "rocm_bandwidth_test.hpp").is_file())

    def test_build_recipes_use_local_sources_and_expected_architecture(self):
        settings = {"dtk_root": "/opt/dtk", "mpi_home": "/opt/mpi", "compiler": "/opt/dtk/bin/aicc",
                    "arch": "gfx92a", "arch_num": 9210, "jobs": 2}
        memory, _ = prepare.commands_for("memory", Path("/src"), Path("/build"), settings)
        self.assertIn("TARGET_NUM=9210", memory[0])
        self.assertIn("OFFLOAD_ARCH=--offload-arch=gfx92a", memory[0])
        for suite in prepare.SOURCES:
            commands, _ = prepare.commands_for(suite, Path("/src"), Path("/build"), settings)
            self.assertTrue(all(c[0] in ("make", "cmake") for c in commands))
        commands, _ = prepare.commands_for("mooncake", Path("/src"), Path("/build"), settings)
        self.assertIn("-DFETCHCONTENT_FULLY_DISCONNECTED=ON", commands[0])
        self.assertTrue(any(c.startswith("-DFETCHCONTENT_SOURCE_DIR_YALANTINGLIBS=") for c in commands[0]))


class CacheTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        source = self.root / "vendor/hcu-stream"
        source.mkdir(parents=True)
        (source / "source.txt").write_text("CPU fixture", encoding="utf-8")
        write_json(self.root / "vendor/sources.json", {"hcu-stream": {
            "tree_sha256": prepare.tree_hash(source), "version": "test-only"}})
        self.settings = {"dtk_root": "/test/dtk", "mpi_home": "/test/mpi", "arch": "gfx92a"}
        for target in (patch.object(prepare, "ROOT", self.root),
                       patch.object(prepare, "settings_for", return_value=self.settings)):
            target.start()
            self.addCleanup(target.stop)
        if os.name == "nt":
            # Windows tests validate cache logic, not Linux flock implementation.
            target = patch.dict(sys.modules, {"fcntl": SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=lambda *args: None)})
            target.start()
            self.addCleanup(target.stop)
        self.output = self.root / "output"
        self.output.mkdir()
        self.stopped = threading.Event()

    def build(self):
        return prepare.prepare("memory", "auto", self.output, {}, self.stopped)

    def recipe(self, suite, source, build, settings):
        target = source / "fixture.exe"
        code = "import shutil; shutil.copy2(%r, %r)" % (sys.executable, str(target))
        return [[sys.executable, "-c", code]], target

    def test_first_build_cache_hit_and_corrupt_binary_rebuild(self):
        with patch.object(prepare, "commands_for", side_effect=self.recipe) as commands, contextlib.redirect_stdout(io.StringIO()):
            first = self.build()
            self.assertFalse(first["cache_hit"])
            second = self.build()
            self.assertTrue(second["cache_hit"])
            self.assertEqual(commands.call_count, 1)
            Path(first["binary"]).write_bytes(b"corrupt")
            third = self.build()
            self.assertFalse(third["cache_hit"])
            self.assertNotEqual(first["binary"], third["binary"])
            self.assertEqual(commands.call_count, 2)
            self.assertEqual((self.root / "vendor/hcu-stream/source.txt").read_text(), "CPU fixture")

    def test_architecture_change_uses_separate_cache(self):
        with patch.object(prepare, "commands_for", side_effect=self.recipe), contextlib.redirect_stdout(io.StringIO()):
            first = self.build()
            self.settings["arch"] = "gfx938"
            second = self.build()
            self.assertNotEqual(first["binary"], second["binary"])

    def test_failed_build_never_creates_ready_record(self):
        with patch.object(prepare, "commands_for", return_value=([[sys.executable, "-c", "raise SystemExit(7)"]], self.root / "absent")), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "exit 7"):
            self.build()
        self.assertFalse(list(self.root.rglob("ready.json")))
        self.assertTrue((self.output / "prepare-00.log").exists())

    def test_bad_source_and_cancelled_build_do_not_execute(self):
        with patch.object(prepare, "NativeProcess") as launched:
            self.stopped.set()
            with self.assertRaises(InterruptedError):
                self.build()
            (self.root / "vendor/hcu-stream/source.txt").write_text("changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                self.build()
            launched.assert_not_called()

    def test_explicit_binary_does_not_build_or_fall_back(self):
        with patch.object(prepare, "settings_for", side_effect=AssertionError("must not build")):
            record = prepare.prepare("memory", sys.executable, self.output, {}, self.stopped)
            self.assertEqual(record["source"], "explicit")
            with self.assertRaisesRegex(ValueError, "not found"):
                prepare.prepare("memory", str(self.root / "absent"), self.output, {}, self.stopped)

    def test_wrapper_checks_prepared_binary_digest(self):
        args = SimpleNamespace(binary="auto", output_dir=str(self.output))
        write_json(self.output / "prepared.json", {"binary": sys.executable, "binary_sha256": "wrong", "env": {}})
        with self.assertRaises(ValueError):
            prepared_binary(args)


class PreparationStageTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = temp.name
        self.spec = {"work_dir": root, "lock_root": root, "resources": [], "cwd": root,
                     "prepare": ["prepare"], "prepare_timeout_s": 90, "argv": ["workload"],
                     "timeout_s": 10, "duration_s": 5}
        agent.CANCEL.clear()
        for target in (patch.object(agent, "listen"), patch.object(agent, "emit"), patch.object(agent, "collect_artifacts")):
            target.start()
            self.addCleanup(target.stop)

    def test_workload_timer_excludes_preparation(self):
        result = {"status": "completed", "cleanup": "confirmed", "returncode": 0}
        with patch.object(agent, "stage", return_value=dict(result)) as stage:
            self.assertEqual(agent.run(self.spec)["status"], "completed")
        self.assertEqual(stage.call_args_list[0].args[3:], (90,))
        self.assertTrue(stage.call_args_list[0].kwargs["cooperative"])
        self.assertEqual(stage.call_args_list[1].args[3:], (10, 5, False))

    def test_forced_preparation_shutdown_cannot_confirm_child_cleanup(self):
        process = Mock(pid=12345, returncode=130)
        for forced in (False, True):
            process.wait.side_effect = subprocess.TimeoutExpired("compiler", 8) if forced else None
            with self.subTest(forced=forced), patch.object(agent.os, "name", "posix"), \
                    patch.object(agent.os, "killpg", create=True), patch.object(agent, "kill_group") as kill:
                self.assertEqual(agent.stop_cooperative(process), not forced)
                self.assertEqual(kill.called, forced)
        process.wait.side_effect = None
        process.returncode = -15
        with patch.object(agent.os, "name", "posix"), patch.object(agent.os, "killpg", create=True):
            self.assertFalse(agent.stop_cooperative(process))

    def test_failed_timed_out_cancelled_or_unclean_prepare_skips_workload(self):
        for status, cleanup in (("failed", "confirmed"), ("timed_out", "confirmed"), ("cancelled", "confirmed"), ("completed", "unconfirmed")):
            with self.subTest(status=status, cleanup=cleanup), patch.object(agent, "stage", return_value={"status": status, "cleanup": cleanup}) as stage:
                result = agent.run(self.spec)
                self.assertEqual(result["phase"], "prepare")
                self.assertEqual(result["cleanup"], cleanup)
                stage.assert_called_once()

    def test_missing_workload_after_successful_prepare_is_not_completed(self):
        with patch.object(agent, "stage", side_effect=[{"status": "completed", "cleanup": "confirmed"}, FileNotFoundError("missing workload")]):
            self.assertEqual(agent.run(self.spec)["status"], "failed")

    def test_container_prepares_inside_only_and_allows_build_budget(self):
        plan = make_plan(str(ROOT / "configs/managed-container.template.yaml"), "memory")
        case = plan.cases[0]
        outer = CommandAdapter().build(plan, case, case.nodes[0], Path(self.spec["work_dir"]) / "run")
        inner = json.loads(base64.b64decode(outer["argv"][-1]))
        self.assertNotIn("prepare", outer)
        self.assertIn("prepare", inner)
        self.assertEqual(inner["prepare_timeout_s"], 3600)
        self.assertEqual(outer["timeout_s"], case.timeout_s + 3600 + 15)


class InlineInputsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def args(self, **kwargs):
        args = accuracy.parser().parse_args(["--model-path", "/models/flash", "--output-dir", str(self.root)])
        vars(args).update(kwargs)
        return args

    def test_auto_inputs_and_yaml_overrides_materialize(self):
        serving = {"env": {"NCCL_IB_HCA": "nic0", "NCCL_SOCKET_IFNAME": "eth0", "GLOO_SOCKET_IFNAME": "eth0"},
                   "require_memlock_unlimited": False, "visible_hcus": 2,
                   "server_argv": ["server", "${model_path}", "${port}"]}
        args = self.args(serving_json=json.dumps(serving), ep_settings_json=json.dumps({"normal_dispatch": {}, "normal_combine": {}}))
        accuracy.materialize_inputs(args, self.root)
        with patch.dict(os.environ, {"HIP_VISIBLE_DEVICES": "0,1"}):
            profile, command, env = accuracy.profile_for(args)
        self.assertEqual(command, ["server", "/models/flash", "30000"])
        self.assertEqual(env["NCCL_IB_HCA"], "nic0")
        self.assertIn("GPU_MAX_HW_QUEUES", env)
        self.assertEqual(profile["visible_hcus"], 2)
        self.assertEqual(Path(args.data_path), ROOT / "vendor/gsm8k/test.jsonl")
        self.assertTrue(Path(args.ep_config).is_file())

    def test_unknown_network_and_conflicting_ep_fail_clearly(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "Configure NCCL_IB_HCA"):
            accuracy.materialize_inputs(self.args(), self.root)
        with self.assertRaisesRegex(ValueError, "not both"):
            accuracy.materialize_inputs(self.args(ep_config="external.json", ep_settings_json='{"normal_dispatch": {}}'), self.root)

    def test_legacy_external_profile_remains_unchanged(self):
        profile = self.root / "external.json"
        profile.write_text('{"version":1,"server_argv":["server"],"visible_gpus":1}')
        before = profile.read_bytes()
        args = self.args(profile=str(profile))
        accuracy.materialize_inputs(args, self.root)
        self.assertEqual(profile.read_bytes(), before)
        with patch.dict(os.environ, {"HIP_VISIBLE_DEVICES": "0"}):
            self.assertEqual(accuracy.profile_for(args)[1], ["server"])

    def test_hcu_parameter_aliases_remain_compatible(self):
        pack = {"parameters": {"hcu_id": {"type": "integer", "default": 0}}, "parameter_aliases": {"gpu_id": "hcu_id"}}
        self.assertEqual(resolve_params(pack, {"gpu_id": 3}), {"hcu_id": 3})
        with self.assertRaisesRegex(BenchError, "not both"):
            resolve_params(pack, {"gpu_id": 3, "hcu_id": 3})

    def test_nested_serving_placeholders_survive_outer_command_render(self):
        serving = {"server_argv": ["server", "${model_path}", "${port}"]}
        plan = make_plan(str(ROOT / "configs/deepep-sglang.template.yaml"), "e2e_acc",
                         ["tests.sglang_gsm8k.params.serving=" + json.dumps(serving),
                          "tests.sglang_gsm8k.container=null"])
        case = plan.cases[0]
        spec = CommandAdapter().build(plan, case, case.nodes[0], self.root / "run")
        self.assertEqual(json.loads(spec["argv"][spec["argv"].index("--serving-json") + 1]), serving)
        self.assertEqual(spec["resources"], ["gpu:" + str(n) for n in range(8)])


class PreparationBarrierTests(unittest.TestCase):
    def test_two_nodes_wait_for_preparation_and_peer_failure_is_propagated(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                rendezvous = root / "shared"
                def command(rank, binary):
                    return [sys.executable, str(ROOT / "testpacks/prepare.py"), "mooncake", "--binary", binary,
                            "--output-dir", str(root / str(rank)), "--rendezvous", str(rendezvous),
                            "--node-rank", str(rank), "--nnodes", "2"]
                first = subprocess.Popen(command(0, sys.executable), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + 15
                    while not (rendezvous / "rank-0.ready.json").exists() and time.monotonic() < deadline:
                        time.sleep(.05)
                    self.assertTrue((rendezvous / "rank-0.ready.json").is_file())
                    self.assertIsNone(first.poll())
                    second = subprocess.run(command(1, str(root / "missing") if fail else sys.executable), capture_output=True, timeout=15)
                    first.communicate(timeout=15)
                    self.assertEqual(first.returncode, 1 if fail else 0)
                    self.assertEqual(second.returncode, 1 if fail else 0)
                finally:
                    if first.poll() is None:
                        first.kill()
                    first.communicate()


class PreparationIntegrationTests(unittest.TestCase):
    def test_real_agent_keeps_prepare_logs_out_of_metrics_and_stops_on_failure(self):
        class Transport(AgentExecutor):
            def launch_argv(self, spec):
                return [sys.executable, "-u", "-c", AGENT_PATH.read_text(encoding="utf-8")]

        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as root:
                events = []
                code = "print('BENCH_RESULT preparation is not a measurement'); raise SystemExit(%d)" % (7 if failure else 0)
                spec = {"work_dir": root, "lock_root": root, "resources": [], "cwd": root,
                        "prepare": [sys.executable, "-c", code], "prepare_timeout_s": 3,
                        "argv": [sys.executable, "-c", "print('workload reached')"], "timeout_s": 3}
                result = Transport({"executor": "local"}).execute(spec, lambda: False, events.append)
                self.assertEqual(result["status"], "failed" if failure else "completed")
                lines = [event["text"] for event in events if event["kind"] == "log"]
                self.assertFalse(any(line.startswith("BENCH_RESULT") for line in lines))
                self.assertEqual("workload reached" in lines, not failure)
                if failure:
                    self.assertEqual(result["phase"], "prepare")

    def test_duplicate_hcu_configuration_names_rejected(self):
        data = load_document(ROOT / "configs/demo.yaml")
        next(iter(data["nodes"].values())).update(gpus=[0], hcus=[0])
        with self.assertRaisesRegex(BenchError, "not both"):
            validate_config(data)


if __name__ == "__main__":
    unittest.main()
