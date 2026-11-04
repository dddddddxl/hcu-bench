# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# HCU Bench integration changes; original user-supplied sources retained.
"""Regenerate both GEMM command sets from the runner and shared CSV."""
import shlex
import sys
sys.dont_write_bytecode = True
from types import SimpleNamespace

import run_gemm_cases as suite


def main():
    directory = suite.ROOT / "commands"
    raw_directory = directory / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(cases=suite.ROOT / "gemm_cases.csv", group="all", case_id=None,
                           bench="./bin/rocblas-bench", warmup=None, iters=None)
    for args.profile in ("new", "legacy"):
        runner = "run_gemm_cases.py" if args.profile == "new" else "run_gemm_cases_legacy.py"
        for args.mode in ("check", "perf"):
            shell_lines = ["#!/usr/bin/env bash",
                           "# Copyright (c) 2026 Hygon Information Technology Co., Ltd.",
                           "# HCU Bench integration changes; original user-supplied sources retained.",
                           "set -euo pipefail",
                           'cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."',
                           'export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"',
                           "# Review native check diagnostics before executing the perf script."]
            for args.dtype in suite.TYPES:
                rows = suite.load_cases(args)
                shell_lines.append('python3 %s --dtype %s --mode %s "$@"' % (runner, args.dtype, args.mode))
                lines = ["# Profile: %s; dtype: %s; mode: %s" % (args.profile, args.dtype, args.mode),
                         "# Command reference: run from the package root after DTK setup.",
                         "# Prefer commands/%s_%s.sh for logged execution and automatic CLI checks." % (args.profile, args.mode)]
                for row in rows:
                    rounds = 1
                    for i in range(1, rounds + 1):
                        lines.extend(["# %s round%d" % (row["case_id"], i),
                                      "HIP_VISIBLE_DEVICES=0 " + shlex.join(suite.command(args, row))])
                name = "%s_%s_%s.txt" % (args.profile, args.dtype, args.mode)
                with (raw_directory / name).open("w", encoding="utf-8", newline="\n") as handle:
                    handle.write("\n".join(lines) + "\n")
            with (directory / ("%s_%s.sh" % (args.profile, args.mode))).open("w", encoding="utf-8", newline="\n") as handle:
                handle.write("\n".join(shell_lines) + "\n")
    print("Generated 4 batch scripts and %d raw command references." % (len(suite.TYPES) * 4))


if __name__ == "__main__":
    main()
