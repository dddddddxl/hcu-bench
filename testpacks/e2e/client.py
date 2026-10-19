"""One client task, run separately from the owned model server."""
import argparse
import importlib.metadata
import json
import math
import os
import re
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from testpacks.native_common import emit, sha256, write_json
from testpacks.sglang_accuracy import run as accuracy
from testpacks.e2e.prepare import VERSION


def accuracy_task(config, output):
    version = importlib.metadata.version("sglang")
    if not re.match(r"^0\.5\.12(?:$|[.+a-z-])", version):
        raise ValueError("Accuracy client requires SGLang 0.5.12; the server may use another OpenAI-compatible framework")
    params = dict(config["params"])
    params.pop("dataset")
    path = params["data_path"]
    path = ROOT / "vendor/gsm8k/test.jsonl" if path == "auto" else ROOT / path
    params["data_path"] = str(path)
    args = SimpleNamespace(**params, model_path=config["served_model"])
    evidence = dict(num_examples=accuracy.validate_dataset(path, args.num_examples, args.num_shots),
                    dataset_sha256=sha256(path), num_shots=args.num_shots, client_version=version)
    return accuracy.evaluate(args, evidence, config["url"], output)


SUMMARY_FIELDS = {
    "Test Duration (s)": ("duration", "s", "total"),
    "Total Requests": ("requests", "requests", "count"),
    "Success Requests": ("successful_requests", "requests", "count"),
    "Failed Requests": ("failed_requests", "requests", "count"),
    "Req Throughput (req/s)": ("request_throughput", "req/s", "mean"),
    "Output Throughput (tok/s)": ("output_throughput", "tok/s", "mean"),
    "Total Throughput (tok/s)": ("total_throughput", "tok/s", "mean"),
    "Avg Input Tokens": ("actual_input_tokens", "tokens", "mean"),
    "Avg Output Tokens": ("actual_output_tokens", "tokens", "mean"),
    "Avg TTFT (ms)": ("ttft", "ms", "mean"),
    "Avg TPOT (ms)": ("tpot", "ms", "mean"),
    "Avg ITL (ms)": ("itl", "ms", "mean"),
    "Avg Latency (s)": ("e2e_latency", "s", "mean"),
}


def normalize_perf(result, config):
    if not isinstance(result, dict) or len(result) != 1:
        raise ValueError("Expected exactly one EvalScope workload result")
    native = next(iter(result.values()))
    summary, percentiles = native["metrics"], native["percentiles"]
    if hasattr(summary, "to_dict"):
        summary = summary.to_dict()
    if hasattr(percentiles, "to_list"):
        percentiles = percentiles.to_list()
    # 1.13.0 run_one_benchmark returns alias-keyed dictionaries.
    if isinstance(percentiles, dict) and "Percentiles" in percentiles:
        percentiles = [{k: v[i] for k, v in percentiles.items()} for i in range(len(percentiles["Percentiles"]))]
    metrics = []

    def metric(value, name, unit, statistic):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Missing/invalid native metric: {name}/{statistic}")
        metrics.append(dict(name=name, unit=unit, statistic=statistic, value=value))

    for key, fields in SUMMARY_FIELDS.items():
        metric(summary.get(key), *fields)
    if summary["Success Requests"] + summary["Failed Requests"] != summary["Total Requests"]:
        raise ValueError("Native request counts disagree")
    complete = (summary["Total Requests"] == config["params"]["requests"] and summary["Failed Requests"] == 0
                and summary["Success Requests"] > 0 and summary["Avg Output Tokens"] > 0)
    if not isinstance(percentiles, list):
        raise ValueError("Unsupported native percentile schema")
    for percentile in ("50%", "95%", "99%"):
        row = next((r for r in percentiles if r.get("Percentiles") == percentile), {})
        for key, name, unit in (("TTFT (ms)", "ttft", "ms"), ("TPOT (ms)", "tpot", "ms"),
                                ("ITL (ms)", "itl", "ms"), ("Latency (s)", "e2e_latency", "s")):
            metric(row.get(key), name, unit, "p" + percentile[:-1])
    return dict(measurement_status="measured" if complete else "invalid", correctness="not_checked",
                reason="" if complete else "Failed, missing or empty responses; not a complete throughput measurement",
                metrics=metrics, workload=config["workload"], client_version=VERSION,
                cache_policy="unique_prompt_offsets; server cache settings are controlled by the launch script")


def performance_task(config, output):
    if importlib.metadata.version("evalscope") != VERSION:
        raise ValueError("EvalScope version mismatch")
    point, params = config["workload"], config["params"]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from evalscope.perf.main import run_perf_benchmark
    native_args = dict(model=config["served_model"], url=config["url"] + "/v1/chat/completions", api="openai",
                       tokenizer_path=config["tokenizer_path"], dataset="random", stream=True,
                       parallel=[point["concurrency"]], number=[params["requests"]], rate=-1,
                       min_prompt_length=point["input_tokens"], max_prompt_length=point["input_tokens"],
                       min_tokens=point["output_tokens"], max_tokens=point["output_tokens"],
                       prefix_length=0, seed=params["seed"], dataset_offset=point["dataset_offset"],
                       warmup_num=params["warmup_requests"], total_timeout=params["request_timeout_s"],
                       extra_args=params["extra_body"], outputs_dir=str(output), name="native", no_timestamp=True)
    write_json(output / "native-config.json", native_args)
    result = run_perf_benchmark(native_args)
    def serializable(value):
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if hasattr(value, "to_list"):
            return value.to_list()
        if hasattr(value, "model_dump"):
            return value.model_dump(by_alias=True)
        raise TypeError(type(value).__name__)
    # Keep the untouched native result even when validation below rejects it.
    (output / "native-result.json").write_text(json.dumps(result, default=serializable, ensure_ascii=True, indent=2), encoding="utf-8")
    payload = normalize_perf(result, config)
    write_json(output / "performance.json", payload)
    emit(payload)
    return 0 if payload["measurement_status"] == "measured" else 1


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", required=True)
    cli.add_argument("--output-dir", required=True)
    args = cli.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",localhost,127.0.0.1,::1"
        return (accuracy_task if config["suite"] == "e2e_acc" else performance_task)(config, output)
    except Exception as exc:
        write_json(output / "error.json", {"error": str(exc)})
        traceback.print_exc()
        print("[e2e client error] " + str(exc), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
