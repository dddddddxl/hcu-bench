"""Model pipelines and metric contracts; no HCU or model weights required."""
import contextlib
import copy
import io
import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from hcu_bench.e2e_config import task_params, validate_models
from hcu_bench.adapters import CommandAdapter
from hcu_bench.e2e_report import build_e2e_report
from hcu_bench.models import BenchError
from hcu_bench.planner import make_plan
from hcu_bench.report import build_report
from hcu_bench.runner import run_plan
from hcu_bench.store import RunStore
from testpacks.e2e import client, prepare, run
from testpacks.native_common import CleanupError, NativeProcess

ROOT = Path(__file__).resolve().parents[1]


class PlanningTests(unittest.TestCase):
    def plan(self, selection="all", changes=None):
        data = yaml.safe_load((ROOT / "configs/e2e-multi-model.template.yaml").read_text(encoding="utf-8"))
        if changes:
            changes(data)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.yaml"
            path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
            return make_plan(str(path), selection)

    def test_group_all_and_overlapping_selectors_do_not_duplicate_tasks(self):
        for selection in ("e2e", "all", "e2e,e2e_acc", "e2e_perf,e2e_acc"):
            with self.subTest(selection=selection):
                plan = self.plan(selection)
                self.assertEqual(len(plan.cases), 2)
                for case in plan.cases:
                    tasks = case.params["session"]["tasks"]
                    self.assertEqual([t["suite"] for t in tasks], ["e2e_acc", "e2e_perf"])
                    self.assertGreater(case.timeout_s, sum(t["timeout_s"] for t in tasks))

    def test_each_leaf_only_schedules_that_task(self):
        for suite in ("e2e_acc", "e2e_perf"):
            plan = self.plan(suite)
            self.assertEqual(len(plan.cases), 2)
            for case in plan.cases:
                self.assertEqual(case.suite, suite)
                self.assertEqual([t["suite"] for t in case.params["session"]["tasks"]], [suite])

    def test_disabled_task_is_not_added_to_session(self):
        plan = self.plan(changes=lambda d: d["tests"]["accuracy"].update(enabled=False))
        self.assertTrue(all(c.suite == "e2e_perf" for c in plan.cases))

    def test_different_environments_create_separate_sessions(self):
        plan = self.plan(changes=lambda d: d["tests"]["performance"].update(env={"CACHE_MODE": "off"}))
        self.assertEqual(len(plan.cases), 4)

    def test_model_environment_uses_rendered_hcu_ids(self):
        plan = self.plan(changes=lambda d: d["models"]["model_a"].update(env={"HIP_VISIBLE_DEVICES": "${hcu_ids}"}))
        case = plan.cases[0]
        spec = CommandAdapter().build(plan, case, "node0", ROOT / "runs/fixture")
        self.assertEqual(spec["env"]["HIP_VISIBLE_DEVICES"], "0,1,2,3,4,5,6,7")
        self.assertNotIn("env", case.params["session"]["model"])

    def test_model_node_override_and_missing_model(self):
        def change(data):
            data["nodes"]["node1"] = copy.deepcopy(data["nodes"]["node0"])
            data["models"]["model_b"]["nodes"] = ["node1"]
        self.assertEqual([c.nodes for c in self.plan(changes=change).cases], [["node0"], ["node1"]])
        with self.assertRaisesRegex(BenchError, "known models"):
            self.plan(changes=lambda d: d["models"].pop("model_a"))

    def test_legacy_suite_is_accuracy_only(self):
        plan = make_plan(str(ROOT / "configs/deepep-sglang.template.yaml"), "e2e",
                         ["tests.sglang_gsm8k.suite=e2e"])
        self.assertEqual([c.suite for c in plan.cases], ["e2e_acc"])

    def test_invalid_parameters_fail_before_launch(self):
        for params in ({"concurrency": [0]}, {"requests": 1, "concurrency": [2]}, {"shapes": [[0, 8]]},
                       {"concurrency": [1, 1]}, {"repeats": False}, {"request_timeout_s": 0.5},
                       {"extra_body": {"stream": False}}, {"unknown": 2}):
            with self.subTest(params=params), self.assertRaises(BenchError):
                task_params("e2e_perf", params)
        with self.assertRaises(BenchError):
            task_params("unknown", {})
        with self.assertRaises(BenchError):
            validate_models({1: {"start_script": "a.sh"}})

    def test_runner_attributes_leaf_results_and_rejects_unknown_model(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = self.plan()
            plan.output_dir = folder
            plan.nodes = {"node0": {"executor": "local", "python": sys.executable}}
            for index, case in enumerate(plan.cases):
                session = case.params["session"]
                payloads = [dict(measurement_status="measured", correctness="not_checked", e2e_task=t["test_id"],
                                 e2e_suite=t["suite"], model_id=session["model_id"] if index == 0 else "wrong-model",
                                 metrics=[dict(name="fixture", value=1, unit="count", statistic="mean")]) for t in session["tasks"]]
                case.pack = copy.deepcopy(case.pack)
                case.pack.pop("prepare", None)
                case.pack["artifacts"] = []
                case.pack["command"] = [sys.executable, "-u", "-c", "import json;\nfor p in " + repr(payloads) + ": print('BENCH_RESULT ' + json.dumps(p))"]
            store = RunStore.create(plan)
            self.assertEqual(run_plan(plan, store, quiet=True), 1)
            report = build_report(store)
            self.assertEqual(report["case_counts"], {"completed": 1, "invalid": 1})
            samples = [json.loads(line) for line in (store.path / "results.jsonl").read_text().splitlines()]
            self.assertEqual([s["suite"] for s in samples[:2]], ["e2e_acc", "e2e_perf"])
            self.assertTrue(all(s["params"]["model_id"] == "model_a" for s in samples[:2]))
            self.assertTrue(all(s["measurement_status"] == "invalid" for s in samples[2:]))
            self.assertIn("not_measured", (store.path / "e2e.csv").read_text(encoding="utf-8-sig"))


class MetricTests(unittest.TestCase):
    def native(self):
        summary = {key: 1.0 for key in client.SUMMARY_FIELDS}
        summary.update({"Total Requests": 4, "Success Requests": 4, "Failed Requests": 0})
        percentiles = [{"Percentiles": p, "TTFT (ms)": 3.0, "TPOT (ms)": 2.0, "ITL (ms)": 2.0, "Latency (s)": .5}
                       for p in ("50%", "95%", "99%")]
        return {"parallel_1_number_4": {"metrics": summary, "percentiles": percentiles}}

    def normalize(self, data):
        return client.normalize_perf(data, {"params": {"requests": 4}, "workload": {"repeat": 0}})

    def test_metrics_preserve_units_and_percentiles(self):
        result = self.normalize(self.native())
        self.assertEqual(result["measurement_status"], "measured")
        metrics = {(m["name"], m["statistic"]): (m["value"], m["unit"]) for m in result["metrics"]}
        self.assertEqual(metrics[("ttft", "p99")], (3, "ms"))
        self.assertEqual(metrics[("e2e_latency", "p95")], (.5, "s"))
        self.assertEqual(result["correctness"], "not_checked")

    def test_failed_or_missing_requests_are_invalid(self):
        for success, failure in ((3, 1), (3, 0)):
            data = self.native()
            data["parallel_1_number_4"]["metrics"].update({"Total Requests": success + failure,
                                                        "Success Requests": success, "Failed Requests": failure})
            self.assertEqual(self.normalize(data)["measurement_status"], "invalid")

    def test_missing_nonfinite_or_inconsistent_metrics_are_rejected(self):
        for key, value in (("Avg TTFT (ms)", float("nan")), ("Total Requests", 9), ("Avg ITL (ms)", None)):
            data = self.native()
            data["parallel_1_number_4"]["metrics"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.normalize(data)

    def test_transposed_percentiles(self):
        data = self.native()
        rows = data["parallel_1_number_4"]["percentiles"]
        data["parallel_1_number_4"]["percentiles"] = {k: [row[k] for row in rows] for k in rows[0]}
        self.assertEqual(self.normalize(data)["measurement_status"], "measured")

    def test_offsets_and_repeats(self):
        params = task_params("e2e_perf", {"shapes": [[16, 8], [32, 8]], "concurrency": [1, 2],
                                          "requests": 4, "warmup_requests": 1, "repeats": 2})
        points = run.workloads({"suite": "e2e_perf", "params": params}, 0)
        self.assertEqual(len(points), 8)
        self.assertEqual([p["dataset_offset"] for p in points], list(range(0, 48, 6)))


class SessionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.launch = self.root / "start model.sh"
        self.launch.write_text("#!/usr/bin/env bash\nexec " + shlex.quote(Path(sys.executable).as_posix()) + " -u " +
                               shlex.quote((ROOT / "tests/fixtures/e2e_server.py").as_posix()) + "\n", encoding="utf-8")
        if not run.bash_command(self.launch)[0]:
            self.skipTest("Requires bash")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.session = {"model_id": "model_a", "model": {"start_script": str(self.launch), "port": self.port,
                         "tokenizer_path": str(self.root), "startup_timeout_s": 5}, "failure_policy": "continue",
                        "tasks": [{"test_id": "acc", "suite": "e2e_acc", "params": {}, "timeout_s": 5},
                                  {"test_id": "perf", "suite": "e2e_perf", "timeout_s": 5,
                                   "params": {"shapes": [[16, 8]], "concurrency": [1, 2], "repeats": 1,
                                              "requests": 4, "warmup_requests": 1}}]}
        self.output = self.root / "output"
        self.output.mkdir()
        (self.output / "e2e-client.json").write_text(json.dumps({"interpreters": {"auto": sys.executable}}))
        self.calls = []
        self.fail_accuracy = False

    def fake_client(self, command, cwd, env, log, server, stopped, timeout, on_line=None):
        self.assertIsNone(server.proc.poll())
        if on_line is None:
            self.calls.append(("hook", server.proc.pid))
            return 0
        config = json.loads((Path(cwd) / "client-config.json").read_text())
        self.calls.append((config["suite"], server.proc.pid))
        if self.fail_accuracy and config["suite"] == "e2e_acc":
            return 1
        on_line("BENCH_RESULT " + json.dumps({"measurement_status": "measured", "correctness": "not_checked",
                "metrics": [{"name": "fixture_metric", "value": 1, "unit": "count", "statistic": "mean"}]}))
        return 0

    def invoke(self):
        with contextlib.redirect_stdout(io.StringIO()), patch.object(run, "run_child", side_effect=self.fake_client):
            return run.run(self.session, self.output, threading.Event())

    def assert_port_closed(self):
        with self.assertRaises(OSError):
            socket.create_connection(("127.0.0.1", self.port), timeout=.2)

    def test_same_service_for_all_tasks_and_cleanup_before_next_model(self):
        self.assertEqual(self.invoke(), 0)
        self.assertEqual([c[0] for c in self.calls], ["e2e_acc", "e2e_perf", "e2e_perf"])
        self.assertEqual(len({c[1] for c in self.calls}), 1)
        self.assert_port_closed()
        first_output = self.output
        self.output = self.root / "second"
        self.output.mkdir()
        (self.output / "e2e-client.json").write_bytes((first_output / "e2e-client.json").read_bytes())
        self.session["model_id"] = "model_b"
        self.assertEqual(self.invoke(), 0)
        self.assert_port_closed()

    def test_failure_policy_continue_and_stop(self):
        self.fail_accuracy = True
        for policy, expected in (("continue", 3), ("stop", 1)):
            self.output = self.root / policy
            self.output.mkdir()
            (self.output / "e2e-client.json").write_text(json.dumps({"interpreters": {"auto": sys.executable}}))
            self.session["failure_policy"] = policy
            self.calls.clear()
            self.assertEqual(self.invoke(), 1)
            self.assertEqual(len(self.calls), expected)
            self.assert_port_closed()

    def test_optional_hook_runs_once_before_performance(self):
        self.session["model"]["before_perf_script"] = str(self.launch)
        self.assertEqual(self.invoke(), 0)
        self.assertEqual([c[0] for c in self.calls], ["e2e_acc", "hook", "e2e_perf", "e2e_perf"])

    def test_missing_tokenizer_fails_and_preserves_accuracy(self):
        self.session["model"]["tokenizer_path"] = str(self.root / "missing")
        self.assertEqual(self.invoke(), 1)
        self.assertEqual([c[0] for c in self.calls], ["e2e_acc"])
        self.assert_port_closed()

    def test_occupied_port_is_not_reused_or_killed(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", self.port))
            sock.listen()
            with patch.object(run, "NativeProcess") as launched, self.assertRaises(OSError):
                self.invoke()
            launched.assert_not_called()
            self.assertEqual(sock.getsockname()[1], self.port)

    def test_early_exit_is_not_ready(self):
        self.launch.write_text("exit 7\n")
        with self.assertRaisesRegex(RuntimeError, "foreground"):
            self.invoke()
        self.assert_port_closed()

    def test_cancelled_session_cleans_server(self):
        stopped = threading.Event()
        stopped.set()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(InterruptedError):
            run.run(self.session, self.output, stopped)
        self.assert_port_closed()

    def test_uncertain_cleanup_is_not_ordinary_failure(self):
        close = NativeProcess.close
        def unconfirmed(process):
            close(process)
            raise CleanupError("fixture cleanup unconfirmed")
        with patch.object(NativeProcess, "close", unconfirmed), self.assertRaises(CleanupError):
            self.invoke()

    def test_child_timeout_cleans_its_process(self):
        processes = []
        def capture(*args, **kwargs):
            process = NativeProcess(*args, **kwargs)
            processes.append(process)
            return process
        server = SimpleNamespace(proc=SimpleNamespace(poll=lambda: None))
        with patch.object(run, "NativeProcess", side_effect=capture), self.assertRaises(TimeoutError):
            run.run_child([sys.executable, "-c", "import time; time.sleep(60)"], self.root, os.environ,
                          self.root / "child.log", server, threading.Event(), .1)
        self.assertIsNotNone(processes[0].proc.poll())

    @unittest.skipUnless(os.environ.get("HCU_BENCH_TEST_PERF_PYTHON"), "Optional installed EvalScope HTTP integration")
    def test_real_evalscope_client_against_cpu_streaming_service(self):
        python = os.environ["HCU_BENCH_TEST_PERF_PYTHON"]
        tokenizer = self.root / "tokenizer"
        code = """import sys
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
vocab = {'[UNK]': 0, 'hello': 1, **{'token' + str(i): i + 2 for i in range(1000)}}
native = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
native.pre_tokenizer = pre_tokenizers.Whitespace()
template = "{% for message in messages %}{{ message['content'] + ' ' }}{% endfor %}"
PreTrainedTokenizerFast(tokenizer_object=native, unk_token='[UNK]', model_max_length=8192,
                       chat_template=template).save_pretrained(sys.argv[1])
"""
        made = subprocess.run([python, "-c", code, str(tokenizer)], capture_output=True, text=True, encoding="utf-8", timeout=60)
        self.assertEqual(made.returncode, 0, made.stderr)
        self.session["tasks"] = [self.session["tasks"][1]]
        self.session["tasks"][0]["timeout_s"] = 120
        self.session["tasks"][0]["params"]["concurrency"] = [1]
        self.session["model"]["tokenizer_path"] = str(tokenizer)
        (self.output / "e2e-client.json").write_text(json.dumps({"interpreters": {"auto": python}}))
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            status = run.run(self.session, self.output, threading.Event())
        self.assertEqual(status, 0, stream.getvalue())
        sample = json.loads((self.output / "perf/point-0000/performance.json").read_text())
        self.assertEqual(sample["measurement_status"], "measured")
        self.assertTrue(list(self.output.rglob("*.db")))
        self.assertTrue(list(self.output.rglob("*.html")))
        self.assert_port_closed()


class PreparationAndReportTests(unittest.TestCase):
    def test_accuracy_only_never_installs_performance_client(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(prepare, "execute") as execute:
            prepare.prepare({"tasks": [{"suite": "e2e_acc", "params": {}}]}, Path(folder), threading.Event())
            execute.assert_not_called()
            self.assertEqual(json.loads((Path(folder) / "e2e-client.json").read_text())["interpreters"], {})

    def test_explicit_client_checks_version_without_install(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(prepare, "execute") as execute:
            prepare.prepare({"tasks": [{"suite": "e2e_perf", "params": {"python": "/client/python"}}]},
                            Path(folder), threading.Event())
            self.assertTrue(all("install" not in call.args[0] for call in execute.call_args_list))
            self.assertEqual(execute.call_args_list[0].args[0][-1], prepare.VERIFY)

    def test_report_keeps_missing_task_and_ignores_incomplete_jsonl(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sample = dict(e2e_task="acc", case_id="c", test_id="acc", model_id="m", suite="e2e_acc",
                          measurement_status="measured", correctness="not_checked", metrics=[])
            (root / "results.jsonl").write_text(json.dumps(sample) + '\n{"partial":', encoding="utf-8")
            state = {"cases": {"c": {"status": "failed", "params": {"session": {"model_id": "m", "tasks": [
                {"test_id": "acc", "suite": "e2e_acc"}, {"test_id": "perf", "suite": "e2e_perf"}]}}}}}
            build_e2e_report(SimpleNamespace(path=root), state)
            text = (root / "e2e.csv").read_text(encoding="utf-8-sig")
            self.assertIn("not_measured", text)
            self.assertIn("e2e_acc", text)
            self.assertIn("e2e_perf", text)


if __name__ == "__main__":
    unittest.main()
