#!/usr/bin/env python3
"""Run gemm_cases.csv with rocblas-bench. Requires Python 3.8+ standard library.

Default profile: new (rocBLAS 5.2.0); legacy profile: rocBLAS 4.3.0.
Inspect native validation diagnostics; process exit codes alone are not a
correctness verdict. Perf mode requires prior successful checks.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
sys.dont_write_bytecode = True
import tempfile

ROOT = Path(__file__).resolve().parent
TYPES = {"fp32": "f32_r", "fp16": "f16_r", "fp64": "f64_r",
         "complex64": "f32_c", "complex128": "f64_c"}
INITIALIZATION = "rand_int"


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def load_cases(args):
    with args.cases.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["dtype"] == args.dtype]
    rows = [r for r in rows if (args.group == "all" or r["group"] == args.group)
            and (not args.case_id or r["case_id"] == args.case_id)]
    if not rows:
        raise ValueError("No matching cases")
    ids = set()
    for r in rows:
        if r["case_id"] in ids or not r["case_id"].replace("-", "").isalnum():
            raise ValueError("Invalid or duplicate case ID")
        ids.add(r["case_id"])
        m, n, k = (int(r[x]) for x in ("M", "N", "K"))
        ta, tb = r["transA"], r["transB"]
        if min(m, n, k) <= 0 or ta not in ("N", "T", "C") or tb not in ("N", "T", "C"):
            raise ValueError("Invalid shape or transpose: " + r["case_id"])
        if r["function"] not in ("gemm", "gemm_batched", "gemm_strided_batched", "gemm_ex", "gemm_ex_batched", "gemm_ex_strided_batched"):
            raise ValueError("Unsupported function")
        if int(r["batch_count"]) < 1:
            raise ValueError("Invalid batch count")
    return rows


def command(args, r):
    extended = "_ex" in r["function"]
    function = r["function"]
    function = {"gemm_ex_batched": "gemm_batched_ex", "gemm_ex_strided_batched": "gemm_strided_batched_ex"}.get(function, function)
    cmd = [str(args.bench), "-f", function, "-r", TYPES[args.dtype]]
    if extended:
        for name in ("a", "b", "c", "d"):
            cmd.extend(["--" + name + "_type", r.get(name + "_type", TYPES[args.dtype])])
        cmd.extend(["--compute_type", r.get("compute_type", "f32_r"), "--ldd", r["ldd"]])
    for column, flag in (("M", "-m"), ("N", "-n"), ("K", "-k"),
                         ("transA", "--transposeA"), ("transB", "--transposeB"),
                         ("lda", "--lda"), ("ldb", "--ldb"), ("ldc", "--ldc"),
                         ("alpha", "--alpha"), ("beta", "--beta")):
        cmd.extend([flag, r[column]])
    if args.dtype in ("complex64", "complex128"):
        cmd.extend(["--alphai", r["alphai"], "--betai", r["betai"]])
    if "batched" in function:
        cmd.extend(["--batch_count", r["batch_count"]])
    if "strided" in function:
        for column in ("stride_a", "stride_b", "stride_c"):
            cmd.extend(["--" + column, r[column]])
        if extended:
            cmd.extend(["--stride_d", r["stride_d"]])
    cmd.extend(["--device", "0", "--math_mode", "0"])
    if args.profile == "new":
        cmd.extend(["--use_hipblaslt", r["use_hipblaslt"]])
    cmd.extend(["--flush_batch_count", "1", "--initialization", r.get("initialization", INITIALIZATION)])
    if args.mode == "check":
        if args.profile == "new":
            cmd.extend(["--test", "1"])
        verify = "1" if args.profile == "legacy" or getattr(args, "max_norm_error", "") != "" else "0"
        cmd.extend(["--verify", verify, "--cold_iters", r.get("cold_iters", "2"), "--iters", r.get("check_iters", "10")])
    else:
        if args.profile == "new":
            cmd.extend(["--test", "0"])
        cmd.extend(["--verify", "0", "--cold_iters", str(args.warmup or r["cold_iters"]), "--iters", str(args.iters or r["check_iters"])])
    return cmd


def main(profile=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if profile is None:
        parser.add_argument("--profile", choices=("new", "legacy"), default="new")
    else:
        parser.set_defaults(profile=profile)
    parser.add_argument("--dtype", required=True, choices=TYPES)
    parser.add_argument("--mode", required=True, choices=("check", "perf"))
    parser.add_argument("--cases", type=Path, default=ROOT / "gemm_cases.csv")
    parser.add_argument("--max-norm-error", default="", help="optional additional norm check threshold")
    parser.add_argument("--bench", type=Path, default=ROOT / "bin" / "rocblas-bench")
    parser.add_argument("--group", default="all", choices=("all", "general", "smoke"))
    parser.add_argument("--case-id")
    parser.add_argument("--warmup", type=positive_int, default=None)
    parser.add_argument("--iters", type=positive_int, default=None)
    parser.add_argument("--repeats", type=positive_int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "logs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.bench = args.bench.resolve()
    rows = load_cases(args)
    rounds = 1 if args.mode == "check" else args.repeats
    print("Profile %s; selected %d cases; %d round(s) per case" % (args.profile, len(rows), rounds))
    if args.dry_run:
        for row in rows:
            for i in range(1, rounds + 1):
                print("# %s %s round%d\n%s" % (row["case_id"], args.dtype, i, shlex.join(command(args, row))))
        return 0
    if not args.bench.is_file() or not os.access(args.bench, os.X_OK):
        parser.error("rocblas-bench is missing or not executable: " + str(args.bench))
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = "%s_%s_%s_%s_" % (args.profile, args.mode, args.dtype, datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=str(args.output)))
    from check_gemm_env import assess_interface, probe_cli
    required = {option for row in rows for option in command(args, row)[1::2]}
    probes = probe_cli(args.bench)
    cli_report = assess_interface(probes, required, args.profile)
    cli_report.update(probes=probes, gpu_tested=False)
    (run_dir / "cli_check.json").write_text(json.dumps(cli_report, indent=2), encoding="utf-8")
    if cli_report["errors"]:
        for error in cli_report["errors"]:
            print("INCOMPATIBLE: " + error, file=sys.stderr)
        print("CLI evidence: " + str(run_dir / "cli_check.json"), file=sys.stderr)
        return 2
    metadata = {
        "started_at": datetime.now().astimezone().isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "case_count": len(rows),
        "rocblas_version": cli_report["rocblas_version"],
        "check_semantics": cli_report["check_semantics"],
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "environment": {key: value for key, value in os.environ.items()
                        if key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")
                        or key.startswith(("ROCBLAS_", "TENSILE_"))},
        "correctness_status": "Review check logs; process exit codes are not correctness verdicts",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    failed = 0
    with (run_dir / "execution.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("case_id", "dtype", "stage", "return_code", "log_file", "command"))
        for row in rows:
            for i in range(1, rounds + 1):
                stage = "check" if args.mode == "check" else "run%d" % i
                cmd = command(args, row)
                log_path = run_dir / (row["case_id"] + "_" + args.dtype + "_" + stage + ".log")
                print("Running %s %s -> %s" % (row["case_id"], stage, log_path), flush=True)
                with log_path.open("w", encoding="utf-8") as log:
                    log.write("Command: " + shlex.join(cmd) + "\n")
                    log.flush()
                    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
                writer.writerow((row["case_id"], args.dtype, stage, result.returncode, log_path.name, shlex.join(cmd)))
                handle.flush()
                if result.returncode != 0:
                    failed += 1
                    print("Process failed; remaining rounds of this case skipped", flush=True)
                    break
    print("Logs: %s\nCases with process errors: %d" % (run_dir, failed))
    print("Inspect check diagnostics; for perf read rocblas-Gflops and us from run logs.")
    return 1 if failed else 0


def entrypoint(profile=None):
    try:
        return main(profile)
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(entrypoint())
