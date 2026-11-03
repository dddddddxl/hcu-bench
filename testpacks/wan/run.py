"""Owned Wan diffusion service and serial video requests; no LLM client dependencies."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from testpacks.e2e.run import bash_command, check_alive, script_path
from testpacks.native_common import CleanupError, NativeProcess, cancellation, cleanup_failure, emit, sha256, versions, write_json
from testpacks.sglang_accuracy.run import ensure_free_port

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER = r"([0-9]+(?:\.[0-9]+)?)"
PIXEL = re.compile(r"Pixel data generated successfully in " + NUMBER + r" seconds")
STEP = re.compile(r"average time per step: " + NUMBER + r" seconds")
RESERVED = {"model", "output_path", "output_file_name", "perf_dump_path", "n", "num_outputs_per_prompt",
            "input_reference", "reference_url", "image_path"}
REQUEST_FIELDS = {"prompt", "size", "num_frames", "fps", "num_inference_steps", "seed", "negative_prompt",
                  "guidance_scale", "guidance_scale_2"}


def positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Expected a positive finite timing")
    return float(value)


def validate(args, payload, env):
    ids = args.hcu_ids.split(",")
    expected = {"tp1": 1, "tp4": 4}[args.profile]
    if args.nnodes != 1 or args.node_rank != 0 or len(ids) != expected or len(set(ids)) != expected or not all(i.isdigit() for i in ids):
        raise ValueError(f"{args.profile} requires one node and exactly {expected} assigned HCUs")
    if env.get("HIP_VISIBLE_DEVICES") not in (None, "", args.hcu_ids):
        raise ValueError("HIP_VISIBLE_DEVICES conflicts with assigned HCUs")
    if any(env.get(k) for k in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")):
        raise ValueError("Remove secondary device visibility masks; configure nodes.hcus")
    if not (Path(args.model_path) / "model_index.json").is_file():
        raise ValueError("Missing local Wan Diffusers model_index.json; weights are not downloaded")
    if not isinstance(payload, dict) or RESERVED & payload.keys():
        raise ValueError("Wan request must be an object without output/model overrides or image inputs")
    if payload.keys() - REQUEST_FIELDS:
        raise ValueError("Unsupported Wan request fields: " + str(sorted(payload.keys() - REQUEST_FIELDS)))
    if not isinstance(payload.get("prompt"), str) or not payload["prompt"].strip() or "YOUR_" in payload["prompt"]:
        raise ValueError("Set the actual Wan prompt; the reference sheet did not specify one")
    if not re.fullmatch(r"[1-9][0-9]*x[1-9][0-9]*", payload.get("size", "")):
        raise ValueError("Set explicit video size as WIDTHxHEIGHT")
    for key in ("num_frames", "fps", "num_inference_steps"):
        if type(payload.get(key)) is not int or payload[key] <= 0:
            raise ValueError("Set an explicit positive integer: " + key)
    if type(payload.get("seed")) is not int or payload["seed"] < 0:
        raise ValueError("Set an explicit nonnegative seed")
    for key in ("guidance_scale", "guidance_scale_2"):
        if key in payload:
            positive(payload[key])
    if not 0 <= args.warmup_requests <= 100 or not 1 <= args.requests <= 100:
        raise ValueError("Invalid warmup/request count")
    positive(args.startup_timeout_s)
    positive(args.request_timeout_s)


def request_json(opener, url, deadline, body=None):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Wan HTTP request deadline exceeded")
    request = Request(url, data=None if body is None else json.dumps(body, allow_nan=False).encode(),
                      headers={"Content-Type": "application/json"})
    with opener.open(request, timeout=min(10, remaining)) as response:
        return json.load(response)


def wait_ready(server, base, stopped, timeout):
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        check_alive(server, stopped)
        try:
            request_json(opener, base + "/health", deadline)
            schema = request_json(opener, base + "/openapi.json", deadline)
            paths = schema.get("paths", {})
            if "/v1/videos" not in paths or "/v1/videos/{video_id}" not in paths:
                raise ValueError("Unsupported diffusion video API interface")
            # This endpoint parses JSON manually; its OpenAPI body is multipart, not VideoGenerationsRequest.
            return opener, schema
        except HTTPError as exc:
            if exc.code not in (502, 503):
                raise ValueError(f"Diffusion API preflight returned HTTP {exc.code}") from exc
        except (URLError, OSError, TimeoutError):
            pass
        stopped.wait(0.25)
    raise TimeoutError("Wan service readiness deadline exceeded")


def generate(opener, base, body, server, stopped, timeout):
    deadline = time.monotonic() + timeout
    started = time.monotonic()
    check_alive(server, stopped)
    result = request_json(opener, base + "/v1/videos", deadline, body)
    job_id = result.get("id")
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise ValueError("Video API returned an invalid job id")
    while result.get("status") != "completed":
        check_alive(server, stopped)
        if result.get("status") not in ("queued", "in_progress") or result.get("error"):
            raise RuntimeError("Video generation did not complete: " + json.dumps(result))
        if stopped.wait(0.25):
            raise InterruptedError("Wan request cancelled")
        result = request_json(opener, base + "/v1/videos/" + quote(job_id, safe=""), deadline)
        if result.get("id") != job_id:
            raise ValueError("Video API returned a different job id")
    check_alive(server, stopped)
    if result.get("error") or not (result.get("file_path") or result.get("url")):
        raise ValueError("Completed video job has an error or no output reference")
    return result, time.monotonic() - started


def parse_timings(record, pixel, step):
    native = json.loads(Path(record["perf_path"]).read_text(encoding="utf-8"))
    if native.get("request_id") != record["response"]["id"]:
        raise ValueError("Native timing report does not match the video job id")
    stages = native.get("steps")
    if not isinstance(stages, list):
        raise ValueError("Unsupported native performance report: missing stage list")
    values = {}
    for stage, name in (("DenoisingStage", "denoise_time"), ("DecodingStage", "decode_time")):
        matches = [item for item in stages if item.get("name") == stage]
        if len(matches) != 1:
            raise ValueError("Expected exactly one native timing for " + stage)
        values[name] = positive(matches[0]["duration_ms"]) / 1000
    values.update(pixel_total_time=positive(float(pixel)), denoise_step_time=positive(float(step)),
                  client_elapsed_time=positive(record["client_elapsed_s"]))
    api = record["response"].get("inference_time_s")
    if api is not None:
        values["api_inference_time"] = positive(api)
    return values


def run(args, stopped=None):
    stopped = stopped if stopped is not None else cancellation()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.request_json)
    env = dict(os.environ)
    validate(args, payload, env)
    launch = script_path(args.start_script)
    for folder in ("requests", "videos", "native-perf"):
        (output / folder).mkdir(exist_ok=True)
    env.update(HIP_VISIBLE_DEVICES=args.hcu_ids, BENCH_HOST="127.0.0.1", BENCH_PORT=str(args.port),
               BENCH_ROOT=str(ROOT), BENCH_MODEL_PATH=args.model_path, BENCH_WAN_PROFILE=args.profile,
               BENCH_OUTPUT_DIR=str(output), SGLANG_PERF_LOG_DIR=str(output / "native-perf"))
    ensure_free_port(args)
    evidence = dict(model_path=args.model_path, profile=args.profile, hcus=args.hcu_ids,
                    start_script=str(launch), start_script_sha256=sha256(launch),
                    packages=versions(["sglang", "torch"]), request=payload,
                    warmup_requests=args.warmup_requests, requests=args.requests)
    write_json(output / "environment.json", evidence)
    with (output / "wan-session.receipt").open("x", encoding="utf-8") as receipt:
        receipt.write("Owned Wan session; do not reuse this output directory\n")
    if stopped.is_set():
        raise InterruptedError("Wan session cancelled before model launch")
    server = NativeProcess(bash_command(launch), str(launch.parent), env, output / "server.log")
    records = []
    try:
        base = f"http://127.0.0.1:{args.port}"
        opener, schema = wait_ready(server, base, stopped, args.startup_timeout_s)
        write_json(output / "api-schema.json", schema)
        for i in range(args.warmup_requests + args.requests):
            check_alive(server, stopped)
            perf_path = output / "requests" / f"request-{i:04d}-native.json"
            body = {**payload, "model": args.model_path, "output_path": str(output / "videos"), "perf_dump_path": str(perf_path)}
            write_json(output / "requests" / f"request-{i:04d}.json", body)
            result, elapsed = generate(opener, base, body, server, stopped, args.request_timeout_s)
            if result["id"] in {record["response"]["id"] for record in records}:
                raise ValueError("Video API reused a completed request id")
            record = dict(index=i, warmup=i < args.warmup_requests, response=result,
                          client_elapsed_s=elapsed, perf_path=str(perf_path))
            records.append(record)
            write_json(output / "requests.json", records)
    finally:
        server.close()
        try:
            listener = socket.create_connection(("127.0.0.1", args.port), timeout=0.5)
        except OSError:
            pass
        else:
            listener.close()
            raise CleanupError("Wan serving port remains occupied after cleanup")
    text = ANSI.sub("", (output / "server.log").read_text(encoding="utf-8"))
    pixels, steps = PIXEL.findall(text), STEP.findall(text)
    if len(pixels) != len(records) or len(steps) != len(records):
        raise ValueError("Native Pixel/step log count differs from serial requests; refusing ambiguous timing attribution")
    rows = []
    for record, pixel, step in zip(records, pixels, steps):
        if record["warmup"]:
            continue
        values = parse_timings(record, pixel, step)
        row = dict(profile=args.profile, repeat=record["index"] - args.warmup_requests, **values)
        rows.append(row)
    with (output / "measurements.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "measurements.json", rows)
    if stopped.is_set():
        raise InterruptedError("Wan session cancelled before publishing results")
    for row in rows:
        emit(dict(measurement_status="measured", correctness="not_checked", e2e_task="wan_" + args.profile,
                  e2e_suite="e2e_perf", model_id="Wan2.2-T2V-A14B-Diffusers", workload={**payload, "profile": args.profile, "repeat": row["repeat"]},
                  metrics=[dict(name=name, value=value, unit="s", statistic="value") for name, value in row.items() if name not in ("profile", "repeat")],
                  reason="" if "api_inference_time" in row else "API inference_time_s absent; client elapsed is not a substitute",
                  warnings=[] if "api_inference_time" in row else ["API inference_time_s absent; client elapsed is not a substitute"]))
    return 0


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("start_script", "model_path", "profile", "request_json", "hcu_ids", "output_dir"):
        cli.add_argument("--" + name.replace("_", "-"), required=True)
    for name, default in (("warmup_requests", 1), ("requests", 1), ("port", 30005), ("startup_timeout_s", 1800),
                          ("request_timeout_s", 7200), ("nnodes", 1), ("node_rank", 0)):
        cli.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    cli.set_defaults(host="127.0.0.1")
    return cli


def main():
    args = parser().parse_args()
    try:
        return run(args)
    except CleanupError as exc:
        return cleanup_failure(args.output_dir, exc)
    except Exception as exc:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "error.json", {"error": str(exc)})
        emit(dict(measurement_status="invalid", correctness="not_checked", metrics=[], reason=str(exc),
                  e2e_task="wan_" + args.profile, e2e_suite="e2e_perf", model_id="Wan2.2-T2V-A14B-Diffusers",
                  workload={"profile": args.profile}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
