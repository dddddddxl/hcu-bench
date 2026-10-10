import base64
import contextlib
import io
import json
import os
import shlex
import sys
import tempfile
import threading
import time
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from hcu_bench import agent
from hcu_bench.adapters import for_case
from hcu_bench.cli import main, preview
from hcu_bench.config import load_document
from hcu_bench.containers import HookContainerLauncher, render_argv
from hcu_bench.executors.base import AGENT_PATH
from hcu_bench.executors.ssh import SSHExecutor
from hcu_bench.models import BenchError, Plan
from hcu_bench.planner import make_plan
from hcu_bench.report import build_report
from hcu_bench.results import parse_sample
from hcu_bench.runner import run_plan
from hcu_bench.store import RunStore

ROOT = Path(__file__).resolve().parents[1]


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = {"version": 1, "name": "test", "output_dir": "runs", "nodes": {"local": {"executor": "local"}},
                     "tests": {"demo": {"suite": "deepep", "adapter": "mock", "nodes": ["local"], "timeout_s": 5, "params": {}}}}

    def plan(self, suite="all", overrides=None):
        path = self.root / "config.yaml"
        path.write_text(yaml.safe_dump(self.data, sort_keys=False), encoding="utf-8")
        return make_plan(str(path), suite, overrides)

    def execute(self):
        plan = self.plan()
        store = RunStore.create(plan)
        code = run_plan(plan, store, quiet=True)
        summary = build_report(store)
        return code, store, summary

    def command_pack(self, script, requires=True, artifacts=None):
        path = self.root / "script.py"
        path.write_text(script, encoding="utf-8")
        pack = {"version": 1, "id": "test.command", "suite": "deepep", "parameters": {},
                "command": [sys.executable, "-u", str(path), "${node_rank}"], "cwd": "${case_dir}",
                "result_protocol": "bench-jsonl-v1", "requires_metrics": requires}
        if artifacts is not None:
            pack["artifacts"] = artifacts
        (self.root / "pack.yaml").write_text(yaml.safe_dump(pack), encoding="utf-8")
        self.data["tests"]["demo"].update(adapter="command", pack="pack.yaml")


class ConfigurationTests(Fixture):
    def test_demo_has_five_suites(self):
        plan = make_plan(str(ROOT / "configs/demo.yaml"))
        self.assertEqual(len(plan.cases), 6)
        self.assertEqual({case.suite for case in plan.cases}, {"rccl", "deepep", "mooncake", "operators", "e2e"})

    def test_suite_selection(self):
        plan = make_plan(str(ROOT / "configs/demo.yaml"), "deepep,mooncake")
        self.assertEqual(len(plan.cases), 2)

    def test_matrix_and_repeats(self):
        self.data["tests"]["demo"].update(matrix={"payload_bytes": [1, 2, 3]}, repeats=2)
        plan = self.plan()
        self.assertEqual(len({case.id for case in plan.cases}), 6)

    def test_reject_duplicate_matrix(self):
        self.data["tests"]["demo"]["matrix"] = {"payload_bytes": [1, 1]}
        with self.assertRaises(BenchError):
            self.plan()

    def test_reject_large_matrix_before_expanding(self):
        self.data["max_cases"] = 2
        self.data["tests"]["demo"]["repeats"] = 3
        with self.assertRaises(BenchError):
            self.plan()

    def test_override_typed_parameter(self):
        self.data["tests"]["demo"]["params"] = {"samples": 2}
        self.assertEqual(self.plan(overrides=["tests.demo.params.samples=8"]).cases[0].params["samples"], 8)

    def test_config_variables_preserve_types_and_worker_placeholders(self):
        self.data["vars"] = {"workspace": "scratch", "results": "${vars.workspace}/results", "gpus": [0, 2], "samples": 2}
        self.data["output_dir"] = "${vars.results}"
        self.data["nodes"]["local"].update(gpus="${vars.gpus}", env={"HIP_VISIBLE_DEVICES": "${gpu_ids}"})
        self.data["tests"]["demo"]["params"] = {"samples": "${vars.samples}"}
        plan = self.plan(overrides=["vars.workspace=other", "vars.samples=4"])
        self.assertEqual(plan.output_dir, str(self.root / "other/results"))
        self.assertEqual(plan.nodes["local"]["gpus"], [0, 2])
        self.assertEqual(plan.nodes["local"]["env"]["HIP_VISIBLE_DEVICES"], "${gpu_ids}")
        self.assertEqual(plan.cases[0].params["samples"], 4)
        self.assertEqual(plan.resolved["vars"]["results"], "other/results")

    def test_config_variables_are_expanded_in_hooks_without_shell_splitting(self):
        self.command_pack("print('example')", requires=False)
        self.data["vars"] = {"image": "registry/image:tag", "workspace": "/a path; literal"}
        self.data["containers"] = {"image": {
            "mode": "managed", "name": "bench-${run_id}-${case_id}-${node_id}", "work_root": "/bench",
            "start": ["start", "${container_name}", "${vars.image}", "${vars.workspace}"],
            "check": ["check", "${container_name}"], "stop": ["stop", "${container_name}"]}}
        self.data["tests"]["demo"]["container"] = "image"
        plan = self.plan()
        spec = for_case(plan.cases[0]).build(plan, plan.cases[0], "local", self.root / "run")
        self.assertEqual(spec["container"]["start"][2:], ["registry/image:tag", "/a path; literal"])

    def test_config_variables_reject_unknown_cycles_and_invalid_types(self):
        for variables in ({"a": "${vars.missing}"}, {"a": "${vars.b}", "b": "${vars.a}"},
                          {"a": "${vars.a}"}, {"invalid-name": "x"}, {"a": [1, 2], "b": "path/${vars.a}"}):
            with self.subTest(variables=variables), self.assertRaises(BenchError):
                self.data["vars"] = variables
                self.plan()

    def test_config_variables_reject_malformed_references(self):
        self.data["output_dir"] = "${vars.bad-name}"
        with self.assertRaises(BenchError):
            self.plan()

    def test_simple_templates_build_with_different_container_modes(self):
        for name, container in (("in-container.template.yaml", None), ("managed-container.template.yaml", "test_image")):
            with self.subTest(name=name):
                plan = make_plan(str(ROOT / "configs" / name))
                self.assertEqual([case.suite for case in plan.cases], ["deepep", "e2e"])
                self.assertTrue(all(case.container == container for case in plan.cases))
                self.assertEqual(plan.cases[0].duration_s, 600)
                self.assertEqual(plan.nodes["node0"]["gpus"], list(range(8)))
                with patch("subprocess.Popen", side_effect=AssertionError("preview must not execute")):
                    self.assertEqual(preview(plan)["case_count"], 2)

    def test_variable_based_configuration_executes(self):
        self.data["vars"] = {"samples": 1, "output": "variable-results"}
        self.data["output_dir"] = "${vars.output}"
        self.data["tests"]["demo"]["params"] = {"samples": "${vars.samples}"}
        code, store, _ = self.execute()
        self.assertEqual(code, 0)
        self.assertEqual(store.path.parent, self.root / "variable-results")
        self.assertEqual(len((store.path / "results.jsonl").read_text().splitlines()), 1)

    def test_reject_unknown_override(self):
        with self.assertRaises(BenchError):
            self.plan(overrides=["tests.demo.wrong=3"])

    def test_override_parameter_default_without_repeating_it_in_config(self):
        self.assertEqual(self.plan(overrides=["tests.demo.params.samples=8"]).cases[0].params["samples"], 8)

    def test_override_unknown_parameter_still_rejected(self):
        with self.assertRaises(BenchError):
            self.plan(overrides=["tests.demo.params.unsupported=8"])

    def test_reject_unknown_parameter(self):
        self.data["tests"]["demo"]["params"] = {"made_up_flag": 1}
        with self.assertRaises(BenchError):
            self.plan()

    def test_reject_boolean_as_integer(self):
        self.data["tests"]["demo"]["params"] = {"samples": True}
        with self.assertRaises(BenchError):
            self.plan()

    def test_reject_unknown_config_key(self):
        self.data["nnodse"] = 8
        with self.assertRaises(BenchError):
            self.plan()

    def test_reject_duplicate_yaml_keys(self):
        path = self.root / "duplicate.yaml"
        path.write_text("version: 1\nversion: 2\n", encoding="utf-8")
        with self.assertRaises(BenchError):
            load_document(path)

    def test_plan_does_not_execute_or_create_run(self):
        plan = self.plan()
        with patch("subprocess.Popen", side_effect=AssertionError("must not execute")):
            self.assertEqual(preview(plan)["case_count"], 1)
        self.assertFalse(Path(plan.output_dir).exists())

    def test_roundtrip_plan(self):
        plan = self.plan()
        self.assertEqual(Plan.from_dict(plan.to_dict()).to_dict(), plan.to_dict())

    def test_argv_substitution_does_not_split_or_run_shell(self):
        self.assertEqual(render_argv(["echo", "${p_path}"], {"p_path": "a b; $(touch bad)"}), ["echo", "a b; $(touch bad)"])

    def test_container_directories_do_not_assume_a_mount(self):
        self.command_pack("print('example')", artifacts=["${case_dir}/report.json"])
        self.data["containers"] = {"image": {"name": "existing", "mode": "existing", "work_root": "/container/runs",
                                               "check": ["bash", "/host/check.sh", "${container_name}"]}}
        self.data["tests"]["demo"]["container"] = "image"
        plan = self.plan()
        spec = for_case(plan.cases[0]).build(plan, plan.cases[0], "local", self.root / "run")
        inner = json.loads(base64.b64decode(spec["argv"][-1]))
        self.assertTrue(inner["work_dir"].startswith("/container/runs/run/"))
        self.assertEqual(inner["cwd"], inner["work_dir"])
        self.assertEqual(inner["artifacts"], [inner["work_dir"] + "/report.json"])
        self.assertNotEqual(spec["work_dir"], inner["work_dir"])
        self.assertEqual(spec["cwd"], spec["work_dir"])

    def test_container_requires_explicit_writable_directory(self):
        self.data["containers"] = {"image": {"name": "existing", "mode": "existing", "check": ["check"]}}
        with self.assertRaises(BenchError):
            self.plan()

    def test_ssh_connect_timeout_cannot_round_down_to_zero(self):
        self.data["nodes"]["remote"] = {"executor": "ssh", "host": "example.invalid", "work_root": "/bench", "connect_timeout_s": 0.5}
        with self.assertRaises(BenchError):
            self.plan()

    def test_missing_template_is_error(self):
        with self.assertRaises(BenchError):
            render_argv(["${unknown}"], {})

    def test_ssh_build_preserves_argv(self):
        spec = {"argv": ["python3", "a b.py", "x; echo broken"], "cwd": "/work space"}
        command = SSHExecutor({"executor": "ssh", "host": "my-ssh-alias", "user": "someone"}).launch_argv(spec)
        remote = shlex.split(command[-1])
        self.assertEqual(remote[:3], ["python3", "-u", "-c"])
        self.assertEqual(remote[-1], AGENT_PATH.read_text(encoding="utf-8"))
        self.assertNotIn("x; echo broken", command[-1])
        self.assertLess(len(subprocess.list2cmdline(command)), 30000)
        self.assertIn("BatchMode=yes", command)
        self.assertNotIn("StrictHostKeyChecking=no", command)


class RuntimeTests(Fixture):
    def test_mock_runs_and_marks_simulated(self):
        code, store, summary = self.execute()
        self.assertEqual(code, 0)
        self.assertTrue(summary["metrics"][0]["simulated"])
        sample = json.loads((store.path / "results.jsonl").read_text().splitlines()[0])
        self.assertEqual(sample["correctness"], "not_checked")
        self.assertIn("[SIMULATED", (store.path / "summary.txt").read_text())

    def test_failure_continues_to_next_case(self):
        self.data["tests"]["demo"]["params"] = {"fail": True}
        self.data["tests"]["next"] = {"suite": "rccl", "adapter": "mock", "nodes": ["local"], "timeout_s": 5}
        code, store, _ = self.execute()
        statuses = [case["status"] for case in store.read("state.json")["cases"].values()]
        self.assertEqual((code, statuses), (1, ["failed", "completed"]))

    def test_stop_policy_skips_next_case(self):
        self.data["failure_policy"] = "stop"
        self.data["tests"]["demo"]["params"] = {"fail": True}
        self.data["tests"]["next"] = {"suite": "rccl", "adapter": "mock", "nodes": ["local"], "timeout_s": 5}
        _, store, _ = self.execute()
        self.assertEqual([case["status"] for case in store.read("state.json")["cases"].values()], ["failed", "not_run"])

    def test_empty_profiler_is_invalid_not_success(self):
        self.data["tests"]["demo"]["params"] = {"empty": True}
        code, store, _ = self.execute()
        self.assertEqual(code, 1)
        self.assertEqual(next(iter(store.read("state.json")["cases"].values()))["status"], "invalid")

    def test_timeout_is_failure(self):
        self.data["tests"]["demo"].update(params={"delay_s": 1, "samples": 5}, timeout_s=0.2)
        code, store, _ = self.execute()
        self.assertEqual(code, 1)
        self.assertEqual(next(iter(store.read("state.json")["cases"].values()))["status"], "timed_out")

    def test_duration_is_planned_finish_not_correctness(self):
        self.data["tests"]["demo"].update(params={"delay_s": 0.01, "samples": 1000}, duration_s=0.3)
        code, store, _ = self.execute()
        self.assertEqual(code, 0)
        self.assertEqual(next(iter(store.read("state.json")["cases"].values()))["status"], "duration_reached")

    def test_stop_request_cancels_running_task(self):
        self.data["tests"]["demo"].update(params={"delay_s": 1, "samples": 1000})
        plan = self.plan()
        store = RunStore.create(plan)
        timer = threading.Timer(0.4, lambda: (store.path / "STOP").touch())
        timer.start()
        self.addCleanup(timer.join)
        self.assertEqual(run_plan(plan, store, quiet=True), 130)

    def test_invalid_sample_warns_but_valid_sample_retained(self):
        valid = {"measurement_status": "measured", "correctness": "not_checked", "metrics": [{"name": "latency", "value": 2, "unit": "us", "statistic": "mean"}]}
        self.command_pack("print('BENCH_RESULT broken')\nprint(" + repr("BENCH_RESULT " + json.dumps(valid)) + ")\n")
        code, store, summary = self.execute()
        self.assertEqual(code, 0)
        self.assertEqual(summary["excluded_invalid_or_failed_samples"], 1)
        case = next(iter(store.read("state.json")["cases"].values()))
        self.assertEqual(case["nodes"]["local"]["invalid_samples"], 1)

    def test_one_failed_rank_cancels_peers(self):
        self.command_pack("import sys,time\nif sys.argv[1]=='0':\n time.sleep(.2)\n sys.exit(3)\ntime.sleep(20)\n", requires=False)
        self.data["nodes"]["other"] = {"executor": "local", "gpus": [1]}
        self.data["nodes"]["local"]["gpus"] = [0]
        self.data["tests"]["demo"]["nodes"] = ["local", "other"]
        started = time.monotonic()
        _, store, _ = self.execute()
        self.assertLess(time.monotonic() - started, 8)
        nodes = next(iter(store.read("state.json")["cases"].values()))["nodes"]
        self.assertEqual(nodes["local"]["status"], "failed")
        self.assertEqual(nodes["other"]["status"], "cancelled")

    def test_native_artifact_is_copied_and_checksummed(self):
        self.command_pack("from pathlib import Path\nPath('report.json').write_text('native report')\n", requires=False, artifacts=["report.json"])
        code, store, _ = self.execute()
        self.assertEqual(code, 0)
        records = [json.loads(line) for line in (store.path / "artifacts.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(records[0]["status"], "collected")
        self.assertEqual(Path(records[0]["path"]).read_text(), "native report")

    def test_missing_artifact_is_warning_only(self):
        self.command_pack("print('done')", requires=False, artifacts=["absent.json"])
        code, store, _ = self.execute()
        self.assertEqual(code, 0)
        node = next(iter(store.read("state.json")["cases"].values()))["nodes"]["local"]
        self.assertEqual(node["artifact_status"], "warnings")

    def test_cleanup_unconfirmed_stops_remaining_plan(self):
        self.data["tests"]["next"] = {"suite": "rccl", "adapter": "mock", "nodes": ["local"], "timeout_s": 5}
        with patch("hcu_bench.executors.local.LocalExecutor.execute", return_value={"status": "failed", "cleanup": "unconfirmed"}):
            _, store, _ = self.execute()
        self.assertEqual([case["status"] for case in store.read("state.json")["cases"].values()], ["cleanup_unconfirmed", "not_run"])

    def test_ssh_agent_stdin_protocol_without_live_host(self):
        from hcu_bench.executors.base import AgentExecutor

        class TransportHarness(AgentExecutor):
            def launch_argv(self, spec):
                return [sys.executable, "-u", "-c", AGENT_PATH.read_text(encoding="utf-8")]

        plan = self.plan()
        store = RunStore.create(plan)
        spec = for_case(plan.cases[0]).build(plan, plan.cases[0], "local", store.path)
        events = []
        result = TransportHarness({"executor": "ssh"}).execute(spec, lambda: False, events.append)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(any(event["kind"] == "log" and "BENCH_RESULT" in event["text"] for event in events))

    def test_artifact_byte_limit_warns_without_failing_execution(self):
        self.command_pack("from pathlib import Path\nPath('report.bin').write_bytes(b'x'*2000)\n", requires=False, artifacts=["report.bin"])
        pack_path = self.root / "pack.yaml"
        pack = load_document(pack_path)
        pack["artifact_limit_mb"] = 0.001
        pack_path.write_text(yaml.safe_dump(pack), encoding="utf-8")
        code, store, _ = self.execute()
        self.assertEqual(code, 0)
        self.assertEqual(next(iter(store.read("state.json")["cases"].values()))["nodes"]["local"]["artifact_status"], "warnings")

    def test_framework_identity_cannot_be_overridden_by_test(self):
        result = {"suite": "fake", "node": "fake", "simulated": True, "measurement_status": "measured", "correctness": "not_checked",
                  "metrics": [{"name": "x", "value": 1, "unit": "us", "statistic": "mean"}]}
        self.command_pack("print(" + repr("BENCH_RESULT " + json.dumps(result)) + ")")
        _, store, _ = self.execute()
        sample = json.loads((store.path / "results.jsonl").read_text().splitlines()[0])
        self.assertEqual((sample["suite"], sample["node"], sample["simulated"]), ("deepep", "local", False))


class ProtocolTests(unittest.TestCase):
    def test_non_result_log_ignored(self):
        self.assertIsNone(parse_sample("native log line"))

    def test_nan_rejected(self):
        with self.assertRaises(BenchError):
            parse_sample('BENCH_RESULT {"measurement_status":"measured","correctness":"not_checked","metrics":[{"name":"x","unit":"us","statistic":"mean","value":NaN}]}')

    def test_container_hooks_are_rendered_not_executed(self):
        profile = {"name": "existing", "mode": "existing", "work_root": "/workspace/runs", "check": ["docker", "inspect", "${container_name}"]}
        context = {"run_id": "run", "case_id": "case", "node_id": "node", "case_dir": "/workspace/runs/run/case/node"}
        workload = {"argv": ["python3", "test.py"], "cwd": "/tests"}
        with patch("subprocess.Popen", side_effect=AssertionError("must not execute")):
            rendered = HookContainerLauncher().build(profile, context, workload)
        self.assertEqual(rendered["check"], ["docker", "inspect", "existing"])
        self.assertNotIn("stop", rendered)

    def test_existing_container_is_not_stopped(self):
        spec = {"work_dir": tempfile.gettempdir(), "lock_root": tempfile.gettempdir(), "resources": [], "cwd": tempfile.gettempdir(),
                "argv": ["unused"], "timeout_s": 1, "container": {"name": "existing", "mode": "existing", "check": ["check"], "hook_timeout_s": 1}}
        agent.CANCEL.clear()
        with patch.object(agent, "listen"), patch.object(agent, "emit"), patch.object(agent, "collect_artifacts"), patch.object(agent, "stage", return_value={"status": "completed", "returncode": 0, "cleanup": "confirmed"}) as stage:
            self.assertEqual(agent.run(spec)["status"], "completed")
        self.assertEqual(stage.call_count, 2)

    def test_managed_container_name_collision_does_not_stop_it(self):
        spec = {"work_dir": tempfile.gettempdir(), "lock_root": tempfile.gettempdir(), "resources": [], "cwd": tempfile.gettempdir(),
                "argv": ["unused"], "timeout_s": 1, "container": {"name": "already-owned", "mode": "managed", "start": ["start"], "stop": ["stop"], "check": ["check"], "hook_timeout_s": 1}}
        agent.CANCEL.clear()
        with patch.object(agent, "listen"), patch.object(agent, "emit"), patch.object(agent, "collect_artifacts"), patch.object(agent, "stage", return_value={"status": "completed", "returncode": 0, "cleanup": "confirmed", "output": "existing-id"}) as stage:
            self.assertEqual(agent.run(spec)["status"], "failed")
        self.assertEqual(stage.call_count, 1)

    def test_docker_inspection_failure_never_attempts_start_or_stop(self):
        spec = {"work_dir": tempfile.gettempdir(), "lock_root": tempfile.gettempdir(), "resources": [], "cwd": tempfile.gettempdir(),
                "argv": ["unused"], "timeout_s": 1, "container": {"name": "new", "mode": "managed", "start": ["start"], "stop": ["stop"], "check": ["check"], "hook_timeout_s": 1}}
        agent.CANCEL.clear()
        with patch.object(agent, "listen"), patch.object(agent, "emit"), patch.object(agent, "collect_artifacts"), patch.object(agent, "stage", return_value={"status": "failed", "returncode": 1, "cleanup": "confirmed"}) as stage:
            self.assertEqual(agent.run(spec)["status"], "failed")
        self.assertEqual(stage.call_count, 1)

    def test_container_check_error_does_not_confirm_cleanup(self):
        spec = {"work_dir": tempfile.gettempdir(), "lock_root": tempfile.gettempdir(), "resources": [], "cwd": tempfile.gettempdir(),
                "argv": ["unused"], "timeout_s": 1, "container": {"name": "new", "mode": "managed", "start": ["start"], "stop": ["stop"], "check": ["check"], "hook_timeout_s": 1}}
        ok = {"status": "completed", "returncode": 0, "cleanup": "confirmed", "output": ""}
        error = {"status": "failed", "returncode": 2, "cleanup": "confirmed"}
        agent.CANCEL.clear()
        with patch.object(agent, "listen"), patch.object(agent, "emit"), patch.object(agent, "collect_artifacts"), patch.object(agent, "stage", side_effect=[ok, ok, ok, ok, ok, error]):
            self.assertEqual(agent.run(spec)["cleanup"], "unconfirmed")

    def test_process_that_does_not_exit_cannot_confirm_cleanup(self):
        agent.CANCEL.clear()
        with patch("subprocess.Popen") as launch, patch.object(agent, "kill_group", return_value=False):
            proc = launch.return_value
            proc.poll.return_value = None
            proc.stdout = []
            proc.wait.side_effect = subprocess.TimeoutExpired("owned-process", 12)
            result = agent.stage(["unused"], tempfile.gettempdir(), {}, timeout_s=0)
        self.assertEqual(result["cleanup"], "unconfirmed")
        self.assertEqual(result["status"], "failed")

    def test_cli_list(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["list"]), 0)
        self.assertIn("mooncake", output.getvalue())


if __name__ == "__main__":
    unittest.main()
