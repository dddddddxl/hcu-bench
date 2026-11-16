# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

from .models import Plan
from .report import build_report
from .runner import run_plan
from .store import RunStore


def main():
    store = RunStore(Path(sys.argv[1]))
    try:
        return run_plan(Plan.from_dict(store.read("plan.json")), store, quiet=True)
    finally:
        build_report(store)
        print(f"Report: {store.path / 'summary.txt'}", flush=True)
        print(f"Markdown report: {store.path / 'summary.md'}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
