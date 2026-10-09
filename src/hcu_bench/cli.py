import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import __version__
from .adapters import describe_spec, for_case
from .models import BenchError, SUITES
from .planner import make_plan
from .report import build_report
from .runner import run_plan
from .store import RunStore, open_run


def parser():
    cli = argparse.ArgumentParser(prog="hcu-bench", description="Unified CLI benchmark orchestration (v0.1)")
    cli.add_argument("--version", action="version", version=__version__)
    commands = cli.add_subparsers(dest="action", required=True)
    commands.add_parser("list", help="List suite slots and available adapter types")
    for name in ("check", "plan", "run"):
        child = commands.add_parser(name)
        child.add_argument("-c", "--config", required=True)
        child.add_argument("--suite", default="all", help="all, or comma-separated suite names")
        child.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")
        if name == "plan":
            child.add_argument("--json", action="store_true")
        if name == "run":
            child.add_argument("--detach", action="store_true")
            child.add_argument("--quiet", action="store_true")
    for name in ("status", "stop", "report"):
        child = commands.add_parser(name)
        child.add_argument("run_dir", help="Run directory printed by run")
    return cli


def preview(plan):
    # Preview uses a reserved placeholder; it never creates directories, SSH sessions or containers.
    run_dir = Path(plan.output_dir) / "PREVIEW"
    return {"name": plan.name, "failure_policy": plan.failure_policy, "output_dir": plan.output_dir,
            "case_count": len(plan.cases), "cases": [{**case.to_dict(), "commands": {
            node: describe_spec(for_case(case).build(plan, case, node, run_dir)) for node in case.nodes}} for case in plan.cases]}


def dispatch(args):
    if args.action == "list":
        for suite in SUITES:
            print(f"{suite:12} mock (simulation only), command (external test-pack manifest)")
        print("Executors: local, ssh (Linux + key/ssh-agent/SSH config)")
        print("Containers: existing, managed; user-supplied start/check/stop hooks")
        return 0
    if args.action in ("check", "plan", "run"):
        plan = make_plan(args.config, args.suite, args.set)
        rendered = preview(plan)
        if args.action == "check":
            print(f"Configuration OK: {len(plan.cases)} cases. Static check only; no workloads started.")
            return 0
        if args.action == "plan":
            if args.json:
                print(json.dumps(rendered, ensure_ascii=False, indent=2))
            else:
                print(f"{plan.name}: {len(plan.cases)} cases, serial, failure_policy={plan.failure_policy}")
                for case in rendered["cases"]:
                    print(f"\n{case['id']}  suite={case['suite']}  nodes={','.join(case['nodes'])}  repeat={case['repeat']}")
                    print(f"  params={case['params']}  timeout={case['timeout_s']}s  duration={case['duration_s']}")
                    for node, command in case["commands"].items():
                        print(f"  {node}: {json.dumps(command['argv'], ensure_ascii=False)}")
                        if command.get("container"):
                            print(f"  container={json.dumps(command['container'], ensure_ascii=False)}")
            return 0
        store = RunStore.create(plan)
        print(f"Run directory: {store.path}", flush=True)
        if args.detach:
            with (store.path / "worker.log").open("w", encoding="utf-8") as output:
                flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
                worker = subprocess.Popen([sys.executable, "-m", "hcu_bench.worker", str(store.path)], stdin=subprocess.DEVNULL,
                                          stdout=output, stderr=subprocess.STDOUT, start_new_session=os.name != "nt", creationflags=flags)
            print(f"Worker PID: {worker.pid}; use status/stop with the run directory.")
            return 0
        try:
            return run_plan(plan, store, args.quiet)
        finally:
            build_report(store)
            print(f"Report: {store.path / 'summary.txt'}", flush=True)
    store = open_run(args.run_dir)
    if args.action == "status":
        print(json.dumps(store.read("state.json"), ensure_ascii=False, indent=2))
    elif args.action == "stop":
        state = store.read("state.json")
        if state["status"] not in ("running", "queued"):
            print(f"Already terminal: {state['status']}")
        else:
            (store.path / "STOP").touch()
            print("Cancellation requested. Wait for status to become terminal; no unrelated PID is killed.")
    else:
        build_report(store)
        print((store.path / "summary.txt").read_text(encoding="utf-8"))
    return 0


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    try:
        return dispatch(parser().parse_args(argv))
    except (BenchError, OSError) as exc:
        print(f"hcu-bench: {exc}", file=sys.stderr)
        return 2
