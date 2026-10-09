import contextlib
import importlib.util
import io
import json
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import yaml

from hcu_bench import agent
from hcu_bench.planner import make_plan
from hcu_bench.results import parse_sample
from hcu_bench.runner import run_plan
from hcu_bench.store import RunStore
from testpacks import native_common
from testpacks.deepep import run as deepep
from testpacks.sglang_accuracy import run as accuracy

ROOT = Path(__file__).resolve().parents[1]


class DeepEPParserTests(unittest.TestCase):
    def test_combined_metrics_and_source_statistics(self):
        sample, = deepep.parse_line("[rank 3] Dispatch + combine bandwidth: 70.20 GB/s, avg_t=313.9 us, min_t=310.1 us, max_t=320.0 us")
        self.assertEqual(sample["native_rank"], 3)
        self.assertEqual([item["statistic"] for item in sample["metrics"]], ["reported", "mean", "min", "max"])
        self.assertEqual(sample["metrics"][0]["value"], 70.2)

    def test_merged_rank_lines_are_not_lost(self):
        text = "[rank 2] Dispatch + combine bandwidth: 65.53 GB/s, avg_t=336.46 us, min_t=323.53 us, max_t=383.85 us"
        text += "[rank 1] Dispatch + combine bandwidth: 65.61 GB/s, avg_t=336.04 us, min_t=324.97 us, max_t=380.01 us"
        self.assertEqual([sample["native_rank"] for sample in deepep.parse_line(text)], [2, 1])

    def test_stage_bandwidth_total_has_distinct_names(self):
        sample, = deepep.parse_line("[rank 0] Dispatch bandwidth (total): 90.10 GB/s, avg_t=80.0 us | Combine bandwidth (total): 99.10 GB/s, avg_t=140.0 us")
        self.assertEqual(sample["metrics"][0]["name"], "dispatch_bandwidth_total")

    def test_send_recv_components(self):
        sample, = deepep.parse_line("[rank 2] Dispatch send/recv time: 55.81 + 19.22 us | Combine send/recv time: 109.23 + 28.95 us")
        self.assertEqual(len(sample["metrics"]), 4)

    def test_non_measurement_lines_are_ignored(self):
        self.assertEqual(deepep.parse_line("deep_ep initialization done"), [])

    def test_invalid_latency_order_rejected(self):
        with self.assertRaises(ValueError):
            deepep.parse_line("[rank 0] Dispatch + combine bandwidth: 80 GB/s, avg_t=100 us, min_t=110 us, max_t=120 us")

    def test_torchrun_spawns_one_launcher_not_one_per_gpu(self):
        args = deepep.parser().parse_args(["--test-script", "native.py", "--output-dir", "out", "--node-rank", "1", "--nnodes", "2",
                                          "--master-addr", "example", "--num-processes", "8"])
        command = deepep.command_for(args, 2)
        self.assertIn("--nproc-per-node=1", command)
        self.assertIn("--master-port=12347", command)
        self.assertIn("--pressure-test", command)
        self.assertEqual(command[command.index("--num-processes") + 1], "8")

    def test_port_range_does_not_wrap_into_another_job(self):
        args = SimpleNamespace(master_port=65535)
        with self.assertRaises(ValueError):
            deepep.command_for(args, 1)

    def test_world_size_validation_is_gpu_world_size(self):
        args = deepep.parser().parse_args(["--test-script", "missing.py", "--output-dir", "out", "--node-rank", "0", "--nnodes", "6",
                                          "--master-addr", "example", "--num-processes", "8"])
        with self.assertRaisesRegex(ValueError, "Experts"):
            deepep.preflight(args)


class AccuracyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def dataset(self, count=110):
        path = self.root / "gsm8k.jsonl"
        path.write_text("\n".join(json.dumps({"question": f"Question {i}", "answer": "#### 1"}) for i in range(count)) + "\n", encoding="utf-8")
        return path

    def test_sample_count_excludes_shots_and_never_silently_truncates(self):
        path = self.dataset()
        self.assertEqual(accuracy.validate_dataset(path, 100, 5), 100)
        self.assertEqual(accuracy.validate_dataset(path, 0, 5), 105)
        with self.assertRaises(ValueError):
            accuracy.validate_dataset(path, 106, 5)

    def test_malformed_dataset_fails_preflight(self):
        path = self.root / "bad.jsonl"
        path.write_text('{"question":"x"}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            accuracy.validate_dataset(path, 1, 0)

    def test_report_only_score_is_not_a_correctness_pass(self):
        args = SimpleNamespace(min_score="")
        result = SimpleNamespace(score=0.92, convos=[[{"content": "answer"}]] * 100)
        payload = accuracy.normalize_result(result, 1.25, args, {"num_examples": 100})
        self.assertEqual(payload["correctness"], "not_checked")
        self.assertEqual(payload["accuracy_judgement"], "report_only")
        self.assertIsNotNone(parse_sample("BENCH_RESULT " + json.dumps(payload)))

    def test_optional_threshold_marks_failure_not_invalid_measurement(self):
        result = SimpleNamespace(score=0.80, convos=[[{"content": "answer"}]] * 100)
        payload = accuracy.normalize_result(result, 1.25, SimpleNamespace(min_score="0.90"), {"num_examples": 100})
        self.assertEqual((payload["measurement_status"], payload["correctness"]), ("measured", "failed"))

    def test_all_empty_answers_are_not_success(self):
        result = SimpleNamespace(score=0.0, convos=[[{"content": ""}]] * 100)
        with self.assertRaises(ValueError):
            accuracy.normalize_result(result, 1.0, SimpleNamespace(min_score=""), {"num_examples": 100})

    def test_nan_threshold_and_missing_answers_rejected(self):
        with self.assertRaises(ValueError):
            accuracy.score_threshold("NaN")
        with self.assertRaises(ValueError):
            accuracy.normalize_result(SimpleNamespace(score=1, convos=[]), 1, SimpleNamespace(min_score=""), {"num_examples": 100})

    def test_accuracy_cannot_launch_one_server_per_cluster_rank(self):
        with self.assertRaisesRegex(ValueError, "one serving node"):
            accuracy.preflight(SimpleNamespace(nnodes=2, node_rank=0))

    def test_occupied_port_is_not_reused_or_stopped(self):
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            with self.assertRaises(OSError):
                accuracy.ensure_free_port(SimpleNamespace(host="127.0.0.1", port=occupied.getsockname()[1]))

    def test_reference_profile_requires_real_nic_values(self):
        args = SimpleNamespace(profile=str(ROOT / "testpacks/sglang_accuracy/flash-int8-bw1100.template.json"),
                               model_path="/model", ep_config="/ep.json", host="127.0.0.1", port=30000)
        with self.assertRaisesRegex(ValueError, "YOUR_"):
            accuracy.profile_for(args)

    def test_argv_model_path_is_not_shell_interpreted(self):
        path = self.root / "profile.json"
        path.write_text(json.dumps({"version": 1, "server_argv": ["sglang", "serve", "--model-path", "${model_path}"]}), encoding="utf-8")
        args = SimpleNamespace(profile=str(path), model_path="/a b; echo bad", ep_config="/ep.json", host="127.0.0.1", port=30000)
        _, command, _ = accuracy.profile_for(args)
        self.assertEqual(command[-1], args.model_path)

    def test_recipe_gpu_count_cannot_silently_use_a_different_allocation(self):
        path = self.root / "profile.json"
        path.write_text(json.dumps({"version": 1, "server_argv": ["sglang", "serve"], "visible_gpus": 8, "env": {"HIP_VISIBLE_DEVICES": "0,1"}}), encoding="utf-8")
        args = SimpleNamespace(profile=str(path), model_path="/model", ep_config="/ep.json", host="127.0.0.1", port=30000)
        with self.assertRaisesRegex(ValueError, "GPU count"):
            accuracy.profile_for(args)

    def test_readiness_success_and_timeout(self):
        class Health(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Health)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            process = SimpleNamespace(proc=SimpleNamespace(poll=lambda: None))
            accuracy.wait_ready(process, f"http://127.0.0.1:{server.server_port}", 1, threading.Event())
            with self.assertRaises(TimeoutError):
                accuracy.wait_ready(process, "http://127.0.0.1:1", 0.01, threading.Event())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_dead_server_not_mistaken_for_ready_service(self):
        process = SimpleNamespace(proc=SimpleNamespace(poll=lambda: 1))
        with self.assertRaisesRegex(RuntimeError, "exited"):
            accuracy.wait_ready(process, "http://127.0.0.1:1", 1, threading.Event())

    def test_end_to_end_accuracy_wrapper_keeps_reports_and_cleans_server(self):
        output = self.root / "output"
        evidence = {"num_examples": 100, "test_fixture": True}
        args = SimpleNamespace(output_dir=str(output), check_only=False, port=30000, host="127.0.0.1", startup_timeout=1)
        profile = {"version": 1}
        with patch.object(accuracy, "preflight", return_value=evidence), patch.object(accuracy, "profile_for", return_value=(profile, ["fake-server"], {})), \
                patch.object(accuracy, "ensure_free_port"), patch.object(accuracy, "wait_ready"), patch.object(accuracy, "evaluate", return_value=0) as evaluate, \
                patch.object(accuracy, "NativeProcess") as launch:
            launch.return_value.proc.poll.return_value = None
            self.assertEqual(accuracy.run(args), 0)
        launch.return_value.close.assert_called_once()
        evaluate.assert_called_once()
        self.assertTrue((output / "command.json").exists())

    def test_accuracy_error_still_cleans_its_server(self):
        args = SimpleNamespace(output_dir=str(self.root), check_only=False, port=30000, host="127.0.0.1", startup_timeout=1)
        with patch.object(accuracy, "preflight", return_value={}), patch.object(accuracy, "profile_for", return_value=({}, ["fake"], {})), \
                patch.object(accuracy, "ensure_free_port"), patch.object(accuracy, "wait_ready", side_effect=TimeoutError()), patch.object(accuracy, "NativeProcess") as launch:
            with self.assertRaises(TimeoutError):
                accuracy.run(args)
        launch.return_value.close.assert_called_once()


class NativeLifecycleTests(unittest.TestCase):
    def test_real_pressure_wrapper_streams_then_stops_without_claiming_correctness(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            args = deepep.parser().parse_args(["--test-script", str(root / "native.py"), "--output-dir", str(root), "--node-rank", "0", "--nnodes", "1",
                                              "--master-addr", "localhost", "--num-processes", "1"])
            stopped = threading.Event()
            timer = threading.Timer(0.5, stopped.set)
            script = "import time; print('[rank 0] Dispatch + combine bandwidth: 10 GB/s, avg_t=3 us, min_t=2 us, max_t=4 us', flush=True); time.sleep(0.05)"
            output = io.StringIO()
            timer.start()
            try:
                with contextlib.redirect_stdout(output), patch.object(deepep, "preflight", return_value={"test_fixture": True}), \
                        patch.object(deepep, "cancellation", return_value=stopped), patch.object(deepep, "command_for", return_value=[sys.executable, "-u", "-c", script]):
                    self.assertEqual(deepep.run(args), 130)
            finally:
                timer.cancel()
                timer.join()
            samples = [json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()]
            self.assertGreater(len(samples), 0)
            self.assertTrue(all(sample["correctness"] == "not_checked" for sample in samples))
            self.assertEqual(json.loads((root / "summary.json").read_text())["status"], "interrupted")

    def test_real_pressure_failure_is_not_restarted(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            args = deepep.parser().parse_args(["--test-script", str(root / "native.py"), "--output-dir", str(root), "--node-rank", "0", "--nnodes", "1",
                                              "--master-addr", "localhost", "--num-processes", "1"])
            script = "print('[rank 0] Dispatch + combine bandwidth: 10 GB/s, avg_t=3 us, min_t=2 us, max_t=4 us', flush=True); raise SystemExit(1)"
            with contextlib.redirect_stdout(io.StringIO()), patch.object(deepep, "preflight", return_value={"test_fixture": True}), \
                    patch.object(deepep, "command_for", return_value=[sys.executable, "-u", "-c", script]) as command:
                self.assertEqual(deepep.run(args), 1)
            self.assertEqual(command.call_count, 1)
            self.assertEqual(json.loads((root / "summary.json").read_text())["status"], "failed")

    def test_real_managed_http_service_and_original_accuracy_wrapper_contract(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            server = root / "server.py"
            server.write_text("from http.server import BaseHTTPRequestHandler,HTTPServer\nimport sys\n"
                              "class H(BaseHTTPRequestHandler):\n def do_GET(self):\n  self.send_response(200)\n  self.end_headers()\n  self.wfile.write(b'{}')\n def log_message(self,*args): pass\n"
                              "print('CPU test fixture server',flush=True)\nHTTPServer(('127.0.0.1',int(sys.argv[1])),H).serve_forever()\n", encoding="utf-8")
            profile = root / "profile.json"
            profile.write_text(json.dumps({"version": 1, "server_argv": [sys.executable, "-u", str(server), "${port}"]}), encoding="utf-8")
            args = accuracy.parser().parse_args(["--profile", str(profile), "--model-path", str(root / "model"), "--data-path", str(root / "data"),
                                                "--ep-config", str(root / "ep"), "--output-dir", str(root / "output"), "--port", str(port), "--startup-timeout", "5"])
            modules = {name: ModuleType(name) for name in ("sglang", "sglang.test", "sglang.test.run_eval", "sglang.test.simple_eval_common", "sglang.test.simple_eval_gsm8k")}
            modules["sglang"].__path__ = []
            modules["sglang.test"].__path__ = []
            requested = []

            class FakeEval:
                def __init__(self, **kwargs):
                    requested.append(kwargs)

            def evaluate(settings, url, evaluator):
                with accuracy.build_opener(accuracy.ProxyHandler({})).open(url + "/models", timeout=2):
                    pass
                return SimpleNamespace(score=0.42, convos=[[{"content": "CPU fixture answer"}]] * 100), 0.1, SimpleNamespace(model=settings.model)

            modules["sglang.test.run_eval"].run_eval_once = evaluate
            modules["sglang.test.simple_eval_common"].make_report = lambda result: "<html>CPU fixture only</html>"
            modules["sglang.test.simple_eval_gsm8k"].GSM8KEval = FakeEval
            owned = []

            def launch(*args, **kwargs):
                process = native_common.NativeProcess(*args, **kwargs)
                owned.append(process)
                return process

            with patch.dict(sys.modules, modules), patch.object(accuracy, "preflight", return_value={"num_examples": 100, "test_fixture": True}), \
                    patch.object(accuracy, "NativeProcess", side_effect=launch), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(accuracy.run(args), 0)
            report = json.loads((root / "output/accuracy.json").read_text())
            self.assertEqual(report["correctness"], "not_checked")
            self.assertEqual(requested[0]["num_shots"], 5)
            self.assertTrue((root / "output/answers.json").exists())
            self.assertTrue((root / "output/report.html").exists())
            self.assertIsNotNone(owned[0].proc.poll())

    def test_real_subprocess_output_is_streamed_and_owned_child_is_stopped(self):
        if importlib.util.find_spec("psutil") is None:
            self.skipTest("Install .[native] for target process-tree integration tests")
        import psutil

        with tempfile.TemporaryDirectory() as root:
            seen = []
            script = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print(p.pid, flush=True); time.sleep(60)"
            with contextlib.redirect_stdout(io.StringIO()):
                process = native_common.NativeProcess([sys.executable, "-u", "-c", script], root, dict(__import__("os").environ), Path(root) / "native.log", seen.append)
                for _ in range(100):
                    if seen:
                        break
                    threading.Event().wait(0.02)
                process.snapshot()
                process.close()
            self.assertTrue(seen)
            self.assertFalse(psutil.pid_exists(int(seen[0])))
            self.assertIn(seen[0], (Path(root) / "native.log").read_text())

    def test_reserved_cleanup_failure_stops_remaining_plan(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            pack = {"version": 1, "id": "cleanup", "suite": "deepep", "parameters": {}, "command": [sys.executable, "-c", "raise SystemExit(125)"],
                    "cwd": "${case_dir}", "result_protocol": "none", "requires_metrics": False}
            (root / "pack.yaml").write_text(yaml.safe_dump(pack), encoding="utf-8")
            config = {"version": 1, "name": "cleanup", "nodes": {"local": {"executor": "local"}},
                      "tests": {"native": {"suite": "deepep", "adapter": "command", "nodes": ["local"], "pack": "pack.yaml", "timeout_s": 5},
                                "after": {"suite": "e2e", "adapter": "mock", "nodes": ["local"], "timeout_s": 5}}}
            (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            plan = make_plan(str(root / "config.yaml"))
            store = RunStore.create(plan)
            self.assertEqual(run_plan(plan, store, quiet=True), 1)
            statuses = [case["status"] for case in store.read("state.json")["cases"].values()]
            self.assertEqual(statuses, ["cleanup_unconfirmed", "not_run"])

    def test_two_native_cases_plan_has_confirmed_defaults(self):
        plan = make_plan(str(ROOT / "configs/deepep-sglang.template.yaml"))
        self.assertEqual([case.suite for case in plan.cases], ["deepep", "e2e"])
        self.assertEqual(plan.cases[0].duration_s, 600)
        self.assertEqual(plan.cases[1].params["num_examples"], 100)
        self.assertEqual(plan.cases[1].params["min_score"], "")


if __name__ == "__main__":
    unittest.main()
