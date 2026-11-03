"""Wan HTTP/collection CPU fixtures, never video-model or HCU performance evidence."""
import csv
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import yaml

from hcu_bench.cli import preview
from hcu_bench.models import BenchError
from hcu_bench.planner import make_plan
from testpacks.native_common import CleanupError
from testpacks.wan import run as wan

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = dict(prompt="CPU HTTP fixture, not a video benchmark", size="1280x720", num_frames=81,
               fps=16, num_inference_steps=40, seed=42, guidance_scale=4.0)
SERVER = r'''
import json, os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
mode = os.environ.get("WAN_FIXTURE_MODE", "ok")
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def send(self, value):
        data = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        if self.path == "/health": return self.send({"status": "ok"})
        if self.path == "/openapi.json":
            return self.send({"paths": {"/v1/videos": {}, "/v1/videos/{video_id}": {}},
                "components": {"schemas": {"Body_create_video": {"properties": {"prompt": {"type": "string"}}}}}})
        if mode == "async": return self.send(self.server.pending)
        self.send({"id": self.path.rsplit("/", 1)[-1], "status": "in_progress"})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.count = getattr(self.server, "count", 0) + 1
        job_id = "video_fixture_" + str(self.server.count)
        if mode == "failed": return self.send({"id": job_id, "status": "failed", "error": {"message": "native fixture failure"}})
        if mode == "timeout": return self.send({"id": job_id, "status": "queued"})
        native = {"request_id": "wrong" if mode == "wrong_id" else job_id, "steps": [
            {"name": "DenoisingStage", "duration_ms": 2000}, {"name": "DecodingStage", "duration_ms": 300}]}
        Path(body["perf_dump_path"]).write_text(json.dumps(native))
        output = Path(body["output_path"]) / "CPU_fixture_not_a_video.mp4"
        output.write_bytes(b"CPU fixture, not video content")
        print("[DenoisingStage] average time per step: 0.0500 seconds", flush=True)
        print("Pixel data generated successfully in 2.50 seconds", flush=True)
        if mode == "duplicate": print("Pixel data generated successfully in 2.50 seconds", flush=True)
        result = {"id": job_id, "status": "completed", "file_path": str(output)}
        if mode != "missing_api": result["inference_time_s"] = 2.3
        self.server.pending = result
        self.send({"id": job_id, "status": "queued"} if mode == "async" else result)
HTTPServer(("127.0.0.1", int(os.environ["BENCH_PORT"])), Handler).serve_forever()
'''


class WanPlanningTests(unittest.TestCase):
    def test_wan_is_perf_not_accuracy_and_does_not_prepare_evalscope(self):
        path = str(ROOT / "configs/wan.template.yaml")
        for selection in ("all", "e2e", "e2e_perf"):
            with patch("subprocess.Popen", side_effect=AssertionError("static only")):
                plan = make_plan(path, selection)
                self.assertEqual(len(plan.cases), 2)
                self.assertEqual({case.suite for case in plan.cases}, {"e2e_perf"})
                self.assertEqual([len(plan.nodes[case.nodes[0]]["gpus"]) for case in plan.cases], [1, 4])
                self.assertTrue(all("prepare" not in case.pack for case in plan.cases))
                self.assertEqual(preview(plan)["case_count"], 2)
        with self.assertRaisesRegex(BenchError, "No enabled tests"):
            make_plan(path, "e2e_acc")

    def test_combined_templates_opt_in_wan_without_gsm8k_task(self):
        for name in ("in-container", "managed-container"):
            path = str(ROOT / f"configs/{name}.template.yaml")
            plan = make_plan(path, "e2e", ["tests.wan_tp1.enabled=true", "tests.wan_tp4.enabled=true"])
            self.assertEqual(len(plan.cases), 4)
            videos = [case for case in plan.cases if case.pack["id"] == "e2e_perf.wan-video"]
            self.assertEqual(len(videos), 2)
            self.assertTrue(all("session" not in case.params for case in videos))
            self.assertTrue(all(case.params["request"]["num_frames"] == 81 for case in videos))
            self.assertTrue(all(case.container == ("test_image" if name == "managed-container" else None) for case in videos))


@unittest.skipUnless(os.name == "posix", "Wan serving processes require Linux/POSIX")
class WanRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "model_index.json").write_text("{}")
        self.server = self.root / "server.py"
        self.server.write_text(SERVER, encoding="utf-8")
        self.launch = self.root / "start.sh"
        self.launch.write_text(f"#!/bin/bash\nexec {sys.executable} -u {self.server}\n", encoding="utf-8")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]

    def invoke(self, mode="ok"):
        output_dir = self.root / "results" / mode
        data = dict(version=1, name="CPU-Wan-fixture-not-HCU-data", output_dir=str(output_dir),
                    nodes={"local": {"executor": "local", "hcus": [0]}}, tests={
                        "wan_fixture": dict(suite="e2e_perf", adapter="command", pack=str(ROOT / "testpacks/wan/manifest.yaml"),
                            nodes=["local"], timeout_s=30, env={"WAN_FIXTURE_MODE": mode}, params=dict(bench_root=str(ROOT),
                            start_script=str(self.launch), model_path=str(self.model), profile="tp1", request=PAYLOAD,
                            port=self.port, warmup_requests=1, requests=2, startup_timeout_s=5, request_timeout_s=1))})
        config = self.root / "bench.yaml"
        config.write_text(yaml.safe_dump(data), encoding="utf-8")
        env = {key: value for key, value in os.environ.items() if key not in
               ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")}
        env["BENCH_PYTHON"] = sys.executable
        result = subprocess.run(["bash", str(ROOT / "bench.sh"), str(config), "e2e_perf", "--quiet"],
                                cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=45)
        runs = [p for p in output_dir.iterdir() if (p / "state.json").is_file()]
        self.assertEqual(len(runs), 1, result.stdout)
        run = runs[0]
        self.assertFalse(self.port_open(), "Owned CPU fixture server was not cleaned up")
        return result, run

    def port_open(self):
        try:
            connection = socket.create_connection(("127.0.0.1", self.port), timeout=0.2)
        except OSError:
            return False
        connection.close()
        return True

    def test_complete_public_entry_collects_reports_and_excludes_warmup(self):
        result, run = self.invoke()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(json.loads((run / "summary.json").read_text())["case_counts"], {"completed": 1})
        samples = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines()]
        self.assertEqual(len(samples), 2)
        self.assertTrue(all(s["correctness"] == "not_checked" for s in samples))
        metrics = {m["name"]: m["value"] for m in samples[0]["metrics"]}
        self.assertEqual(metrics["pixel_total_time"], 2.5)
        self.assertEqual(metrics["denoise_time"], 2.0)
        self.assertEqual(metrics["denoise_step_time"], 0.05)
        self.assertEqual(metrics["decode_time"], 0.3)
        self.assertEqual(metrics["api_inference_time"], 2.3)
        self.assertFalse(any("throughput" in key or "token" in key for key in metrics))
        with (run / "e2e.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["suite"] == "e2e_perf" for row in rows))
        self.assertIn("pixel_total_time_value (s)", (run / "e2e.md").read_text())
        artifacts = [json.loads(line) for line in (run / "artifacts.jsonl").read_text().splitlines()]
        self.assertTrue(any("native.json" in record["path"] for record in artifacts if record["status"] == "collected"))
        self.assertTrue(any("server.log" in record["path"] for record in artifacts if record["status"] == "collected"))

    def test_missing_api_timing_is_warned_not_replaced(self):
        result, run = self.invoke("missing_api")
        self.assertEqual(result.returncode, 0, result.stdout)
        samples = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines()]
        self.assertTrue(all(s["warnings"] for s in samples))
        self.assertTrue(all("api_inference_time" not in {m["name"] for m in s["metrics"]} for s in samples))

    def test_async_video_jobs_are_polled_to_completion(self):
        result, run = self.invoke("async")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(json.loads((run / "summary.json").read_text())["case_counts"], {"completed": 1})

    def test_launch_script_preserves_both_reference_profiles_and_warmup_interfaces(self):
        binary = self.root / "sglang"
        binary.write_text(f"#!{sys.executable}\nimport json,os,sys\n"
                          "print(os.environ['WAN_HELP'] if sys.argv[1:] == ['serve', '--help'] else json.dumps(sys.argv[1:]))\n")
        binary.chmod(0o755)
        for profile in ("tp1", "tp4"):
            for help_text, warmup in (("--warmup-mode MODE", "--warmup-mode"), ("--warmup [WARMUP]", "--warmup")):
                env = {**os.environ, "PATH": str(self.root) + os.pathsep + os.environ["PATH"],
                       "BENCH_MODEL_PATH": str(self.model), "BENCH_OUTPUT_DIR": str(self.root),
                       "BENCH_WAN_PROFILE": profile, "WAN_HELP": help_text}
                result = subprocess.run(["bash", str(ROOT / "model_scripts/wan22_t2v.sh")],
                                        env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                argv = json.loads(result.stdout)
                self.assertIn(warmup, argv)
                self.assertEqual(argv[argv.index("--num-gpus") + 1], "1" if profile == "tp1" else "4")
                self.assertEqual(argv[argv.index("--use-fsdp-inference") + 1], "false" if profile == "tp1" else "true")
                self.assertEqual(argv[argv.index("--dit-layerwise-offload") + 1], "true" if profile == "tp1" else "false")
                self.assertEqual("--performance-mode" in argv, profile == "tp1")

    def test_launch_script_rejects_unrecognized_warmup(self):
        binary = self.root / "sglang"
        binary.write_text("#!/bin/bash\nprintf '%s\\n' '--warmup-steps STEPS'\n")
        binary.chmod(0o755)
        env = {**os.environ, "PATH": str(self.root) + os.pathsep + os.environ["PATH"],
               "BENCH_MODEL_PATH": str(self.model), "BENCH_OUTPUT_DIR": str(self.root), "BENCH_WAN_PROFILE": "tp1"}
        result = subprocess.run(["bash", str(ROOT / "model_scripts/wan22_t2v.sh")], env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unsupported diffusion warmup interface", result.stderr)

    def test_failure_wrong_request_id_and_ambiguous_logs_never_publish_metrics(self):
        for mode in ("failed", "wrong_id", "duplicate", "timeout"):
            with self.subTest(mode=mode):
                result, run = self.invoke(mode)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertEqual(json.loads((run / "summary.json").read_text())["case_counts"], {"failed": 1})
                samples = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines()]
                self.assertTrue(samples)
                self.assertTrue(all(s["measurement_status"] == "invalid" and not s["metrics"] for s in samples))
                self.assertIn("failed", (run / "e2e.csv").read_text())

    def test_validation_precedes_model_launch(self):
        args = wan.parser().parse_args(["--start-script", str(self.launch), "--model-path", str(self.model),
              "--profile", "tp1", "--request-json", json.dumps(PAYLOAD), "--hcu-ids", "0", "--output-dir", str(self.root / "out")])
        wan.validate(args, PAYLOAD, {})
        for payload in ({}, {**PAYLOAD, "seed": True}, {**PAYLOAD, "num_frames": 0},
                        {**PAYLOAD, "perf_dump_path": "/foreign/report.json"}, {**PAYLOAD, "input_reference": "other.png"}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                wan.validate(args, payload, {})
        args.profile = "tp4"
        with self.assertRaisesRegex(ValueError, "exactly 4"):
            wan.validate(args, PAYLOAD, {})

    def test_cancelled_before_launch_never_starts_server(self):
        args = wan.parser().parse_args(["--start-script", str(self.launch), "--model-path", str(self.model),
              "--profile", "tp1", "--request-json", json.dumps(PAYLOAD), "--hcu-ids", "0",
              "--port", str(self.port), "--output-dir", str(self.root / "cancelled")])
        stopped = threading.Event()
        stopped.set()
        with patch.dict(os.environ, {"HIP_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "", "GPU_DEVICE_ORDINAL": ""}), \
                patch.object(wan, "NativeProcess", side_effect=AssertionError("cancelled before launch")) as launch:
            with self.assertRaises(InterruptedError):
                wan.run(args, stopped)
            launch.assert_not_called()
        self.assertFalse(self.port_open())

    def test_cleanup_error_keeps_reserved_exit_code(self):
        argv = ["wan", "--start-script", str(self.launch), "--model-path", str(self.model), "--profile", "tp1",
                "--request-json", json.dumps(PAYLOAD), "--hcu-ids", "0", "--output-dir", str(self.root)]
        with patch.object(sys, "argv", argv), patch.object(wan, "run", side_effect=CleanupError("CPU fixture cleanup failure")):
            self.assertEqual(wan.main(), 125)
        self.assertEqual(json.loads((self.root / "cleanup.json").read_text())["cleanup"], "unconfirmed")
