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


if __name__ == "__main__":
    raise SystemExit(main())
