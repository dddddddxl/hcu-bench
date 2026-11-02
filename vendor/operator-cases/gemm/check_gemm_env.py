#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# HCU Bench integration changes; original user-supplied sources retained.
"""Check one GEMM CLI profile without running GEMM. Default: new (5.2.0)."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
sys.dont_write_bytecode = True
import tempfile
from types import SimpleNamespace

import run_gemm_cases as suite

PROFILES = {
    "new": ("5.2.0", "--test 1 tolerance/equality check plus --verify 1 norm check"),
    "legacy": ("4.3.0", "--verify 1 GPU vs CPU validation; legacy native diagnostics"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, default=suite.ROOT / "bin" / "rocblas-bench")
    parser.add_argument("--output", type=Path, default=suite.ROOT / "logs")
    parser.add_argument("--profile", choices=PROFILES, default="new")
    args = parser.parse_args()
    args.bench = args.bench.resolve()
    required = required_options(args.profile)
    probes = probe_cli(args.bench)
    report = assess_interface(probes, required, args.profile)
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = "cli_%s_%s_" % (args.profile, datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=str(args.output)))
    report.update(bench=str(args.bench), probes=probes, gpu_tested=False)
    (out_dir / "cli_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(probes["--version"]["stdout"] + probes["--version"]["stderr"], end="")
    print("CLI evidence: " + str(out_dir / "cli_check.json"))
    if report["errors"]:
        for error in report["errors"]:
            print("INCOMPATIBLE: " + error, file=sys.stderr)
        print("Use the matching complete DTK environment for profile %s; see RUNNING.txt." % args.profile,
              file=sys.stderr)
        return 2
    print("CLI version and options match profile %s. GPU correctness and performance are NOT checked." % args.profile)
    return 0


def probe_cli(bench):
    if not bench.is_file() or not os.access(bench, os.X_OK):
        raise ValueError("rocblas-bench is missing or not executable. Prepare bin/rocblas-bench from the SAME DTK image first.")
    probes = {}
    for option in ("--version", "--help"):
        result = subprocess.run([str(bench), option], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=30)
        probes[option] = {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    return probes


def required_options(profile="new"):
    # Generate requirements from the runner so both entry points use the same recipe.
    args = SimpleNamespace(cases=suite.ROOT / "gemm_cases.csv", group="all", case_id=None,
                           bench="rocblas-bench", warmup=10, iters=100, profile=profile)
    rows = suite.load_cases(args)
    required = set()
    for args.dtype in suite.TYPES:
        for args.mode in ("check", "perf"):
            for row in rows:
                required.update(suite.command(args, row)[1::2])
    return required


def assess_interface(probes, required, profile="new"):
    errors = []
    for option, result in probes.items():
        if result["exit_code"] != 0:
            errors.append("%s exited with %s" % (option, result["exit_code"]))
    version_text = probes["--version"]["stdout"] + probes["--version"]["stderr"]
    version_match = re.search(r"rocBLAS version:\s*(\S+)", version_text)
    if not version_match:
        errors.append("rocBLAS version could not be identified")
    version = version_match.group(1) if version_match else None
    expected, semantics = PROFILES[profile]
    # --verify changed meaning between these releases; do not silently switch profiles.
    if version is not None and version != expected and not version.startswith(expected + "."):
        errors.append("Profile %s requires rocBLAS %s; detected %s" % (profile, expected, version))
    help_text = probes["--help"]["stdout"] + probes["--help"]["stderr"]
    supported = set(re.findall(r"(?:^|\|)[ \t]*(--?[A-Za-z][A-Za-z0-9_]*)", help_text, re.MULTILINE))
    missing = sorted(required - supported)
    if missing:
        errors.append("Missing required CLI options: " + ", ".join(missing))
    return {"profile": profile, "check_semantics": semantics, "rocblas_version": version,
            "required_options": sorted(required), "missing_options": missing,
            "errors": errors, "cli_compatible": not errors}


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        print("CLI CHECK ERROR: " + str(exc), file=sys.stderr)
        sys.exit(1)
