"""Fixed single-HCU operator recipes from the supplied benchmark collection."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_common import CleanupError, cancellation, cleanup_failure, emit, sha256, versions, write_json
from fixed_common import claim_output, environment_record, execute, positive, single_device

ROOT = Path(__file__).resolve().parent
STRESS_COUNT = 1_000_000
FA_ARGS = ["1", "52", "8192", "128", "128", "true", "fp16", "13", "[-1,-1]", "false", "--whl"]
RECIPES = {
    "blas_fp16_stress": ("blas_pressure.py", [], {}),
    "fa_fp16_stress": ("fa_fwd_pressure.py", FA_ARGS, {"FA_FWD_NO_MLS": "1"}),
    "fa_bf16_stress": ("fa_fwd_pressure.py", [*FA_ARGS[:6], "bfloat16", *FA_ARGS[7:]], {"FA_FWD_NO_MLS": "1"}),
    "pa_decode": ("prefix_decode_benchmark.py", ["--whl"], {"PA_NO_MLS": "1"}),
    "fa_fp16_forward": ("fa_fwd_mfu.py", [], {}),
    "fa_fp8_forward": ("bench_fp8_fixlen.py", ["--whl", "--csv", "--output", "fp8.csv"], {"FLASH_ATTENTION_HG_92A_MODE": "fp8"}),
}
PA_GRID = {(b, s) for b in (1, 2, 4, 8, 16, 32, 64, 128) for s in (2048, 3072, 3584, 4096, 5000)}
NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def command(recipe):
    script, argv, _ = RECIPES[recipe]
    return [sys.executable, "-u", str(ROOT / "native" / script), *argv]


def probe_command(recipe):
    return [sys.executable, "-u", str(ROOT / "probe.py"), recipe]


def parse_probe(text):
    records = [json.loads(line[len("BENCH_DEVICE "):]) for line in text.splitlines()
               if line.startswith("BENCH_DEVICE ")]
    if len(records) != 1 or records[0].get("visible_count") != 1 or not records[0].get("hip_version"):
        raise ValueError("Missing/invalid single-device HIP probe result")
    return records[0]


def environment(recipe, args, out):
    env = single_device(args, dict(os.environ))
    required = RECIPES[recipe][2]
    # These switches alter the documented operation, stress loop or timer.
    for key in ("FA_DEBUG", "PA_USE_CUDAGRAPH_BENCH", "PA_NO_SPLITKV", "PA_PARTITION_SIZE"):
        if key in env:
            raise ValueError(f"{key} changes the fixed recipe; remove it")
    for key in ("FA_FWD_NO_MLS", "PA_NO_MLS", "FLASH_ATTENTION_HG_92A_MODE"):
        if key in env and env[key] != required.get(key):
            raise ValueError(f"Conflicting fixed-recipe environment: {key}")
    env.update(required)
    env["BENCH_FA_CACHE_DIR"] = str(out / "fa-cache")
    if (out / "fa-cache").exists():
        raise ValueError("Refusing a pre-existing FA cache")
    return env


def metric(name, value, unit, statistic="native_reported"):
    return dict(name=name, value=positive(value), unit=unit, statistic=statistic)


def parse(recipe, text, out):
    text = ANSI.sub("", text)
    if "Traceback (most recent call last)" in text or re.search(r"\bFAIL(?:ED)?\b", text):
        raise ValueError("Native failure marker; inspect native.log")
    if recipe.endswith("_stress"):
        completed = re.findall(r"Pressure Test: (\d+)/(\d+)\s+PASS\s*$", text)
        if completed != [(str(STRESS_COUNT), str(STRESS_COUNT))]:
            raise ValueError("Stress requires all 1000000 iterations and terminal PASS")
        if recipe.startswith("fa_"):
            for check in ("NaN", "Inf"):
                if len(re.findall(rf"CHECK {check}:\s+PASS", text)) != 1:
                    raise ValueError(f"Missing FA {check} check")
        return [dict(repeatability="passed", correctness="not_checked",
                     correctness_scope="native equality to first result, not independent numerical accuracy",
                     metrics=[metric("completed_iterations", STRESS_COUNT, "iterations", "count")])]
    if recipe == "fa_fp16_forward":
        times = re.findall(rf"^Time:\s*({NUMBER})\s*$", text, re.M)
        rates = re.findall(rf"^Performance:\s*({NUMBER})\s+TFLOPS\s*$", text, re.M)
        if len(times) != 1 or len(rates) != 1:
            raise ValueError("Expected one FA FP16 time and throughput pair")
        return [dict(timing_method="triton.testing.do_bench_cudagraph", metrics=[
            metric("latency", times[0], "ms"), metric("throughput", rates[0], "TFLOPS")])]
    if recipe == "pa_decode":
        return parse_pa(text)
    if recipe == "fa_fp8_forward":
        return parse_fp8(text, out / "fp8.csv")
    raise ValueError("Unknown recipe")


def parse_pa(text):
    rows, seen, pending = [], set(), None
    shape = re.compile(r"^bs (\d+), qhead 24, kvhead 2, seqlen_k (\d+) headdim 128 128 dtype torch.float16$")
    perf = re.compile(rf"^Performance:\s*({NUMBER}) ms,\s*({NUMBER}) TFLOPS,\s*({NUMBER}) GB/s$")
    for line in text.splitlines():
        line = line.strip()
        if line == "Total Statistics:":
            break  # Human summary repeats the same measurements.
        if line.startswith("bs "):
            match = shape.fullmatch(line)
            if pending is not None or match is None:
                raise ValueError("Unexpected/unfinished PA shape")
            pending = tuple(map(int, match.groups()))
            if pending not in PA_GRID or pending in seen:
                raise ValueError("Unexpected or duplicate PA grid point")
        elif line.startswith("Performance:"):
            match = perf.fullmatch(line)
            if pending is None or match is None:
                raise ValueError("PA timing without the expected shape")
            latency, tflops, bandwidth = match.groups()
            b, s = pending
            suffix = f"b{b}_s{s}"
            rows.append(dict(batch=b, kv_length=s, query_length=1, q_heads=24, kv_heads=2,
                             dim_qk=128, dim_v=128, dtype="fp16", block_size=128,
                             timing_method="triton.testing.do_bench", metrics=[
                metric(f"latency_{suffix}", latency, "ms"),
                metric(f"throughput_{suffix}", tflops, "TFLOPS"),
                metric(f"bandwidth_{suffix}", bandwidth, "GB/s")]))
            seen.add(pending)
            pending = None
    if seen != PA_GRID or pending is not None:
        raise ValueError(f"Incomplete PA grid: {len(seen)}/40")
    return rows


def parse_fp8(text, path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError("Expected one FP8 CSV result")
    row = rows[0]
    expected = dict(op="mha_fixlen_fwd", dtype="fp8_e4m3", output_dtype="fp16", batch_size="8",
                    seqlen_q="16384", seqlen_kv="16384", num_heads_q="16", num_heads_kv="2",
                    dim_qk="128", dim_v="128", is_causal="1", window_size_left="-1",
                    window_size_right="0", repeats="20", iterations="5")
    if any(row.get(k) != v for k, v in expected.items()) or not row.get("source"):
        raise ValueError("FP8 CSV does not match the fixed recipe")
    if float(row["softmax_scale"]) != .088 or float(row["softcap"]) != 0 or not 1 <= int(row["samples_used"]) <= 4:
        raise ValueError("Invalid FP8 parameters/sample count")
    rate, latency = positive(row["tflops"]), positive(row["mean_ms"])
    summary = re.findall(rf"^({NUMBER}) TFLOPS, ({NUMBER}) ms$", text, re.M)
    if len(summary) != 1 or abs(float(summary[0][0]) - rate) > .0051 or abs(float(summary[0][1]) - latency) > .000051:
        raise ValueError("FP8 stdout/CSV mismatch")
    return [dict(native=row, timing_method="triton.testing.do_bench",
                 filter_policy="drop TFLOPS below 0.8 mean, then above 1.25 new mean; native fallback",
                 metrics=[metric("throughput", rate, "TFLOPS", "filtered_mean"),
                          metric("latency", latency, "ms", "derived_from_filtered_throughput"),
                          metric("raw_throughput", row["raw_tflops"], "TFLOPS", "mean")])]


def run(args, stopped):
    out = claim_output(args.output_dir)
    executions = []
    try:
        if args.recipe not in RECIPES:
            raise ValueError("Unknown fixed operator recipe")
        env = environment(args.recipe, args, out)
        provenance = json.loads((ROOT / "provenance.json").read_text(encoding="utf-8"))
        for record in provenance["files"]:
            if sha256(ROOT / "native" / record["file"]) != record["bundled_sha256"]:
                raise ValueError(f"Bundled script changed: {record['file']}")
        write_json(out / "plan.json", dict(params=vars(args), argv=command(args.recipe),
                   provenance=provenance, environment=environment_record(env),
                   wrapper_sha256=sha256(__file__),
                   probe_sha256=sha256(ROOT / "probe.py"),
                   loader_sha256=sha256(ROOT / "native/_flash_attn_build_loader.py"),
                   versions=versions(["torch", "triton", "flash-attn", "numpy", "einops"])))
        hardware = execute(probe_command(args.recipe), "device-probe", out, env, stopped, executions,
                           min(60, args.native_timeout_s))
        write_json(out / "device.json", parse_probe(hardware))
        text = execute(command(args.recipe), "native", out, env, stopped, executions, args.native_timeout_s)
        rows = parse(args.recipe, text, out)
        if stopped.is_set():
            raise RuntimeError("Cancelled before publishing complete results")
        for row in rows:
            row.update(measurement_status="measured", correctness="not_checked", recipe=args.recipe)
        write_json(out / "measurements.json", rows)
        write_json(out / "assessment.json", dict(complete=True, correctness="not_checked", measurements=len(rows)))
        for row in rows:
            emit(row)
        return 0
    except CleanupError as exc:
        return cleanup_failure(out, exc)
    except Exception as exc:
        write_json(out / "assessment.json", dict(complete=False, correctness="not_checked", error=str(exc)))
        emit(dict(measurement_status="invalid", correctness="not_checked", metrics=[], reason=str(exc)))
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, choices=RECIPES)
    parser.add_argument("--native-timeout-s", type=int, default=1800)
    parser.add_argument("--hcu-ids", required=True)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    return run(parser.parse_args(), cancellation())


if __name__ == "__main__":
    raise SystemExit(main())
