#!/usr/bin/env python3
"""Run the accompanying Conv2D/Conv3D CSV cases with MIOpenDriver.

Requires Python 3.8+ standard library only. Run check mode and review validation
messages before using perf mode. A zero process exit code is not a correctness
verdict. CSV count is source metadata, not the number of benchmark iterations.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime


ROOT = Path(__file__).resolve().parent
FIELDS_2D = (
    ("n", "-n"), ("c", "-c"), ("H", "-H"), ("W", "-W"),
    ("k", "-k"), ("y", "-y"), ("x", "-x"),
    ("p", "-p"), ("q", "-q"), ("u", "-u"), ("v", "-v"),
    ("l", "-l"), ("j", "-j"), ("g", "-g"), ("F", "-F"),
)
FIELDS_3D = (
    ("in_d", "--in_d"), ("fil_d", "--fil_d"),
    ("pad_d", "--pad_d"), ("conv_stride_d", "--conv_stride_d"),
    ("dilation_d", "--dilation_d"),
)
DIRECTIONS = {"fwd": "1", "dgrad": "2", "wgrad": "4"}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def load_cases(args):
    cases, hashes = [], {}
    suites = ("conv2d", "conv3d") if args.suite == "all" else (args.suite,)
    for suite in suites:
        path = ROOT / (suite + ".csv")
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), 1):
                direction = row["F"].strip()
                if direction not in DIRECTIONS.values():
                    raise ValueError("Unsupported direction in %s row %d" % (path, row_number))
                if args.direction != "all" and direction != DIRECTIONS[args.direction]:
                    continue
                cmd = [args.driver, "conv" if args.dtype == "fp32" else "convfp16"]
                fields = FIELDS_2D + (FIELDS_3D if suite == "conv3d" else ())
                for column, flag in fields:
                    cmd.extend([flag, str(int(row[column]))])
                layout = "NCDHW" if suite == "conv3d" else "NCHW"
                cmd.extend(["--spatial_dim", "3" if suite == "conv3d" else "2", "-m", "conv", "-b", "0"])
                for flag in ("--in_layout", "--fil_layout", "--out_layout"):
                    cmd.extend([flag, layout])
                case_id = "%s-row%03d-F%s-%s" % (suite, row_number, direction, args.dtype)
                cases.append((case_id, cmd))
    if not cases:
        raise ValueError("No cases selected")
    return cases, hashes


def stages(args):
    if args.mode == "check":
        return [("check", ["-V", "1", "-i", "1", "-t", "1"])]
    result = [("warmup", ["-V", "0", "-i", str(args.warmup), "-t", "1"])]
    result.extend(("run%d" % i, ["-V", "0", "-i", str(args.iters), "-t", "1"])
                  for i in range(1, args.repeats + 1))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", required=True, choices=("fp32", "fp16"))
    parser.add_argument("--mode", required=True, choices=("check", "perf"))
    parser.add_argument("--suite", choices=("all", "conv2d", "conv3d"), default="all")
    parser.add_argument("--direction", choices=("all", "fwd", "dgrad", "wgrad"), default="all")
    parser.add_argument("--driver", default="/opt/dtk/bin/MIOpenDriver")
    parser.add_argument("--iters", type=positive_int, default=100)
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "logs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them or writing logs")
    args = parser.parse_args()
    cases, hashes = load_cases(args)
    run_stages = stages(args)
    print("Selected %d cases; %d command(s) per case" % (len(cases), len(run_stages)))
    if args.dry_run:
        for case_id, base_cmd in cases:
            for stage, suffix in run_stages:
                print("# %s %s\n%s" % (case_id, stage, shlex.join(base_cmd + suffix)))
        return 0

    if not os.path.isfile(args.driver) or not os.access(args.driver, os.X_OK):
        parser.error("MIOpenDriver is missing or not executable: " + args.driver)
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = "%s_%s_%s_" % (args.mode, args.dtype, datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=str(args.output)))
    metadata = {
        "started_at": datetime.now().astimezone().isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "case_count": len(cases), "input_sha256": hashes,
        "environment": {key: value for key, value in os.environ.items()
                        if key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
                        or key.startswith("MIOPEN_")},
        "correctness_status": "Review check logs manually; exit codes only describe process execution",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    failed = 0
    with (run_dir / "execution.csv").open("w", encoding="utf-8", newline="") as summary:
        writer = csv.writer(summary)
        writer.writerow(("case_id", "stage", "return_code", "log_file", "command"))
        for case_id, base_cmd in cases:
            for stage, suffix in run_stages:
                cmd = base_cmd + suffix
                log_path = run_dir / (case_id + "_" + stage + ".log")
                print("Running %s %s -> %s" % (case_id, stage, log_path), flush=True)
                with log_path.open("w", encoding="utf-8") as log:
                    log.write("Command: " + shlex.join(cmd) + "\n")
                    log.flush()
                    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
                writer.writerow((case_id, stage, result.returncode, log_path.name, shlex.join(cmd)))
                summary.flush()
                if result.returncode != 0:
                    failed += 1
                    print("Command failed (exit %d); remaining stages for this case skipped" % result.returncode, flush=True)
                    break
    print("Logs: %s\nCases with process errors: %d" % (run_dir, failed))
    print("Check mode: inspect verification messages. Perf mode: read GPU Kernel Time from run1/run2/run3 logs.")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        sys.exit(1)
