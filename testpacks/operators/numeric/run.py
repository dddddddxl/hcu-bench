"""Run the supplied Conv/GEMM cases with existing DTK tools, never install libraries."""
import argparse
import csv
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "testpacks")]
from native_common import CleanupError, cancellation, cleanup_failure, emit, sha256, write_json
from fixed_common import claim_output, environment_record, execute, single_device
from testpacks.prepare import tree_hash

SOURCE = ROOT / "vendor/operator-cases"
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
CONV_LABEL = {"1": "Forward Conv", "2": "Backward Data Conv", "4": "Backward Weights Conv"}
VERIFY_LABEL = {"1": "Forward Convolution", "2": "Backward Convolution Data", "4": "Backward Convolution Weights"}


class VerificationError(ValueError):
    pass


def clean(text):
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    if re.search(r"\b(?:FAILED|FATAL|ERROR|ASSERTION|HSA_STATUS_ERROR)\b|Traceback \(most recent", text, re.I):
        raise ValueError("Native error marker; inspect the original log")
    return text


def finite(value, positive=False):
    number = float(value)
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError("Invalid native numeric result")
    return number


def parse_conv(text, direction, check):
    if check and re.search(r"Convolution.*\bFAILED\b", text):
        raise VerificationError("MIOpen numerical verification failed")
    text = clean(text)
    if check:
        pattern = re.escape(VERIFY_LABEL[direction]) + rf" Verifies OK on (CPU|GPU) reference\s*\(({NUMBER})\s*<\s*({NUMBER})\)"
        matches = re.findall(pattern, text)
        if len(matches) != 1:
            raise ValueError("Missing/ambiguous MIOpen verification verdict")
        reference, error, tolerance = matches[0]
        error, tolerance = finite(error), finite(tolerance, True)
        if error > tolerance:
            raise VerificationError("MIOpen error exceeds its reported tolerance")
        return dict(reference=reference, norm_error=error, tolerance=tolerance)
    pattern = rf"GPU Kernel Time {re.escape(CONV_LABEL[direction])}\. Elapsed:\s*({NUMBER})\s+ms\s*\(average\)"
    matches = re.findall(pattern, text)
    if len(matches) != 1:
        raise ValueError("Expected one native average kernel time for the selected direction")
    return dict(latency_ms=finite(matches[0], True))


def parse_gemm(text, check, threshold, expected=None):
    text = clean(text)
    if re.search(r"^.*:\d+:\s*Failure\b", text, re.M):
        raise VerificationError("rocblas-bench unit/equality check failed; norm tolerance cannot override it")
    lines = text.splitlines()
    records = []
    for index, line in enumerate(lines[:-1]):
        header = [cell.strip() for cell in next(csv.reader([line]))]
        if "rocblas-Gflops" not in header or "us" not in header:
            continue
        values = [cell.strip() for cell in next(csv.reader([lines[index + 1]]))]
        if len(header) != len(values):
            # Native complex scalars can contain unquoted '(real,imag)' pairs.
            values = [cell.strip() for cell in re.split(r",(?![^()]*\))", lines[index + 1])]
        if len(header) != len(set(header)) or len(header) != len(values):
            raise ValueError("Invalid rocblas-bench CSV record")
        records.append(dict(zip(header, values)))
    if len(records) != 1:
        raise ValueError("Expected exactly one rocblas-bench measurement row")
    row = records[0]
    if expected is not None:
        fields = ["M", "N", "K", "lda", "ldb", "ldc", "transA", "transB"]
        if "batched" in expected["function"]:
            fields += ["batch_count"]
        if "strided" in expected["function"]:
            fields += ["stride_a", "stride_b", "stride_c"]
        for field in fields:
            if row.get(field) != expected[field]:
                adjusted = row.get(field, "")
                notice = rf"rocblas-bench INFO: {field} .*setting {field} = {re.escape(adjusted)}\b"
                if expected.get("dtype") and field in ("lda", "ldb", "ldc", "stride_c", "batch_count") and re.search(notice, text):
                    raise ValueError(f"Native adjusted original DTK {field} from {expected[field]} to {adjusted}; measurement not accepted")
                raise ValueError("rocblas-bench result does not match requested " + field)
    result = dict(latency_us=finite(row["us"], True), gflops=finite(row["rocblas-Gflops"], True), native=row)
    if check:
        errors = {key: finite(value) for key, value in row.items() if re.fullmatch(r"norm_error(?:_\d+)?", key)}
        bound = None if threshold == "" else finite(threshold)
        if bound is not None:
            if not errors:
                raise ValueError("Additional norm check requested but no norm errors were reported")
            if any(value > bound for value in errors.values()):
                raise VerificationError("rocblas-bench norm error exceeds configured threshold")
        result.update(norm_errors=errors, tolerance=bound, reference="rocblas-bench native CPU comparison",
                      verification="native_comparison" if bound is None else "native_comparison_and_norm")
    return result


def native_module(recipe):
    path = SOURCE / recipe / f"run_{recipe}_cases.py"
    spec = importlib.util.spec_from_file_location("bench_" + recipe + "_cases", path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def executable(value, recipe):
    if value == "auto":
        dtk = Path(os.environ.get("ROCM_PATH", "/opt/dtk"))
        candidates = ([str(dtk / "bin/MIOpenDriver"), str(dtk / "miopen/bin/MIOpenDriver"), "MIOpenDriver"] if recipe == "conv" else
                      [str(dtk / "lib/rocblas/benchmark_tool/rocblas-bench"), str(dtk / "bin/rocblas-bench"),
                       str(dtk / "rocblas/bin/rocblas-bench"), "rocblas-bench"])
    else:
        candidates = [value]
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return str(Path(found).resolve())
    raise ValueError("Required existing native tool not found: " + ", ".join(candidates))


def selected(args, module, binary):
    if args.recipe == "conv":
        if args.dtype not in ("fp32", "fp16") or args.group != "all":
            raise ValueError("Conv supports fp32/fp16 and no GEMM group filter")
        native = SimpleNamespace(suite=args.conv_suite, direction=args.direction, dtype=args.dtype, driver=binary)
        cases, _ = module.load_cases(native)
        cases = [dict(id=case_id, argv=argv, direction=case_id.split("-F")[1].split("-")[0])
                 for case_id, argv in cases if not args.case_id or case_id == args.case_id]
    else:
        if args.conv_suite != "all" or args.direction != "all":
            raise ValueError("GEMM does not accept Conv filters")
        native = SimpleNamespace(cases=SOURCE / "gemm/gemm_cases.csv", group=args.group, case_id=args.case_id,
                                 bench=Path(binary), dtype=args.dtype, mode="check", profile=args.gemm_profile,
                                 warmup=args.warmup or None, iters=args.iters or None, max_norm_error=args.max_norm_error)
        cases = [dict(id=row["case_id"], row=row) for row in module.load_cases(native)]
    if not cases:
        raise ValueError("No selected native cases")
    return native, cases


def run(args, stopped):
    out = claim_output(args.output_dir)
    executions, results, failed = [], [], False
    deadline = time.monotonic() + args.native_timeout_s
    try:
        if min(args.warmup, args.iters, args.repeats) < 0 or args.native_timeout_s < 1:
            raise ValueError("Iteration overrides must be nonnegative and timeout positive")
        if args.recipe == "conv":
            args.warmup, args.iters = args.warmup or 10, args.iters or 100
        args.repeats = args.repeats or (3 if args.recipe == "conv" else 1)
        if args.max_norm_error != "":
            finite(args.max_norm_error)
        env = dict(os.environ)
        reserved = args.hcu_ids.split(",")
        if not reserved or len(set(reserved)) != len(reserved) or any(not re.fullmatch(r"[0-9]+", v) for v in reserved):
            raise ValueError("Invalid reserved HCU IDs")
        if env.get("HIP_VISIBLE_DEVICES") not in (None, "", args.hcu_ids):
            raise ValueError("HIP_VISIBLE_DEVICES conflicts with reserved HCUs")
        device = reserved[0] if args.device == "auto" else args.device
        if device not in reserved:
            raise ValueError("Numeric operator device must belong to the assigned HCUs")
        env["HIP_VISIBLE_DEVICES"] = device
        env = single_device(SimpleNamespace(hcu_ids=device, nnodes=args.nnodes, node_rank=args.node_rank), env)
        manifest = json.loads((ROOT / "vendor/sources.json").read_text())
        if tree_hash(SOURCE) != manifest["operator-cases"]["tree_sha256"]:
            raise ValueError("Bundled operator case checksum mismatch")
        binary = executable(args.driver if args.recipe == "conv" else args.rocblas_bench, args.recipe)
        module = native_module(args.recipe)
        native, cases = selected(args, module, binary)
        cli = None
        if args.recipe == "gemm":
            try:
                execute([sys.executable, str(SOURCE / "gemm/check_gemm_env.py"), "--bench", binary,
                         "--profile", args.gemm_profile, "--output", str(out / "cli")],
                        "cli-probe", out, env, stopped, executions, min(65, args.native_timeout_s))
            finally:
                # Keep failed preflight evidence in the pack's top-level artifact glob.
                for path in (out / "cli").glob("*/cli_check.json"):
                    cli = json.loads(path.read_text(encoding="utf-8"))
                    write_json(out / "cli_check.json", cli)
        probe = execute([sys.executable, str(Path(__file__).with_name("probe.py"))], "device-probe", out,
                        env, stopped, executions, min(60, args.native_timeout_s))
        devices = [json.loads(line[len("BENCH_DEVICE "):]) for line in probe.splitlines() if line.startswith("BENCH_DEVICE ")]
        if len(devices) != 1 or devices[0].get("visible_count") != 1 or not devices[0].get("pci_bus"):
            raise ValueError("Missing or ambiguous HIP device identity")
        write_json(out / "device.json", devices[0])
        write_json(out / "plan.json", dict(params=vars(args), cases=cases, binary=binary,
                   binary_sha256=sha256(binary), wrapper_sha256=sha256(__file__), source=manifest["operator-cases"],
                   environment=environment_record(env), cli=cli))
        for case in cases:
            case_id = case["id"]
            def launch(argv, stage):
                if stopped.is_set() or time.monotonic() >= deadline:
                    raise InterruptedError("Operator sweep cancelled or timed out")
                return execute(argv, case_id + "-" + stage, out, env, stopped, executions,
                               max(0.001, deadline - time.monotonic()))
            try:
                if stopped.is_set() or time.monotonic() >= deadline:
                    raise InterruptedError("Operator case sweep cancelled or timed out")
                check = None
                if args.recipe == "conv":
                    check = parse_conv(launch(case["argv"] + ["-V", "1", "-i", "1", "-t", "1"], "check"), case["direction"], True)
                elif args.mode == "check" or args.max_norm_error != "":
                    native.mode = "check"
                    check = parse_gemm(launch(module.command(native, case["row"]), "check"), True, args.max_norm_error, case["row"])
                row = dict(recipe=args.recipe, case_id=case_id, dtype=args.dtype, mode=args.mode,
                           correctness="passed" if check is not None else "not_checked", measurement_status="measured")
                if check is not None:
                    row["check"] = check
                if cli is not None:
                    row.update(gemm_profile=cli["profile"], rocblas_version=cli["rocblas_version"],
                               check_semantics=cli["check_semantics"])
                if args.mode == "check":
                    row["metrics"] = [dict(name=f"{case_id}_checked", value=1, unit="cases", statistic="count")]
                else:
                    samples = []
                    if args.recipe == "conv":
                        launch(case["argv"] + ["-V", "0", "-i", str(args.warmup), "-t", "1"], "warmup")
                    for repeat in range(args.repeats):
                        if args.recipe == "conv":
                            text = launch(case["argv"] + ["-V", "0", "-i", str(args.iters), "-t", "1"], f"run{repeat + 1}")
                            samples.append(parse_conv(text, case["direction"], False))
                        else:
                            native.mode = "perf"
                            samples.append(parse_gemm(launch(module.command(native, case["row"]), f"run{repeat + 1}"), False, "", case["row"]))
                    row["samples"] = samples
                    fields = [("latency_ms", "ms")] if args.recipe == "conv" else [("latency_us", "us"), ("gflops", "GFLOPS")]
                    row["metrics"] = [dict(name=f"{case_id}_{field}", value=statistics.median(s[field] for s in samples),
                                           unit=unit, statistic="native_mean" if args.repeats == 1 else "median_of_native_means") for field, unit in fields]
                results.append(row)
                emit(row)
            except CleanupError:
                raise
            except Exception as exc:
                failed = True
                row = dict(recipe=args.recipe, case_id=case_id, dtype=args.dtype, mode=args.mode,
                           correctness="failed" if isinstance(exc, VerificationError) else "not_checked",
                           measurement_status="invalid", metrics=[], reason=str(exc))
                results.append(row)
                emit(row)
                if stopped.is_set() or time.monotonic() >= deadline:
                    break
            write_json(out / "measurements.json", results)
        write_json(out / "measurements.json", results)
        write_json(out / "assessment.json", dict(complete=not failed, expected_cases=len(cases), results=results))
        with (out / "measurements.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["case_id", "dtype", "mode", "status", "metric", "value", "unit", "statistic"])
            for row in results:
                for metric in row["metrics"]:
                    writer.writerow([row["case_id"], row["dtype"], row["mode"], row["measurement_status"],
                                     metric["name"], metric["value"], metric["unit"], metric["statistic"]])
        return 2 if failed else 0
    except CleanupError as exc:
        return cleanup_failure(out, exc)
    except Exception as exc:
        write_json(out / "assessment.json", dict(complete=False, error=str(exc)))
        emit(dict(measurement_status="invalid", correctness="not_checked", metrics=[], reason=str(exc)))
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=("conv", "gemm"), required=True)
    parser.add_argument("--dtype", choices=("fp32", "fp16", "fp64", "complex64", "complex128"), default="fp16")
    parser.add_argument("--mode", choices=("check", "perf"), default="perf")
    parser.add_argument("--driver", default="auto")
    parser.add_argument("--rocblas-bench", default="auto")
    parser.add_argument("--gemm-profile", choices=("new", "legacy"), default="new")
    parser.add_argument("--group", default="all")
    parser.add_argument("--conv-suite", choices=("all", "conv2d", "conv3d"), default="all")
    parser.add_argument("--direction", choices=("all", "fwd", "dgrad", "wgrad"), default="all")
    parser.add_argument("--case-id", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-norm-error", default="")
    for key, default in (("warmup", 0), ("iters", 0), ("repeats", 0), ("native-timeout-s", 7200)):
        parser.add_argument("--" + key, type=int, default=default)
    parser.add_argument("--hcu-ids", required=True)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    return run(parser.parse_args(), cancellation())


if __name__ == "__main__":
    raise SystemExit(main())
