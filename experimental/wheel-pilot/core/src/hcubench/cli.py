import argparse
import json
import importlib.metadata as metadata
import sys
from pathlib import Path
from . import api


def main():
    parser = argparse.ArgumentParser(description="HCU Bench single-case wheel pilot (no scheduler)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    sub.add_parser("describe").add_argument("test")
    imp = sub.add_parser("import-csv", help="Convert selected dataset rows; never executes tests")
    imp.add_argument("adapter")
    imp.add_argument("csv")
    imp.add_argument("--contract", required=True, help="Dataset README defining the input contract")
    imp.add_argument("--rows", required=True, help="Comma-separated 1-based data row numbers, header excluded")
    imp.add_argument("--output", required=True)
    init = sub.add_parser("init")
    init.add_argument("test")
    init.add_argument("--output", required=True)
    init.add_argument("--interactive", action="store_true")
    for name in ("check", "run"):
        p = sub.add_parser(name)
        p.add_argument("config")
        if name == "run":
            p.add_argument("--output", required=True)
    sub.add_parser("report").add_argument("directory")
    p = sub.add_parser("plan", help="Write an external execution plan; does not launch tests")
    p.add_argument("config")
    p.add_argument("--output", required=True)
    p = sub.add_parser("import-result", help="Validate externally collected logs and execution metadata")
    p.add_argument("config")
    p.add_argument("--bundle", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "list":
            value = sorted(api.discover())
        elif args.command == "describe":
            value = api.describe(args.test)
        elif args.command == "import-csv":
            entries = list(metadata.entry_points(group="hcubench.datasets", name=args.adapter))
            if len(entries) != 1:
                raise ValueError("Dataset adapter must have exactly one installed entry point")
            rows = [int(v) for v in args.rows.split(",")]
            value = entries[0].load()(args.csv, args.contract, rows, args.output)
        elif args.command == "init":
            spec = api.describe(args.test)
            params = {}
            for name, rule in spec["parameters"].items():
                value = rule["default"]
                if args.interactive:
                    answer = input(f"{name}: {rule['description']} [{value}] ").strip()
                    if answer:
                        value = {"integer": int, "number": float, "string": str}[rule["type"]](answer)
                params[name] = value
            config = {"schema_version": 1, "test": args.test, "params": params}
            api.resolve(config)
            with Path(args.output).open("x", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            value = {"config": str(Path(args.output).resolve())}
        elif args.command == "report":
            directory = Path(args.directory)
            result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            api.render_report(result, directory / "report.html")
            value = {"report": str((directory / "report.html").resolve())}
        else:
            config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
            if args.command == "check":
                value = api.check(config)
            elif args.command == "plan":
                value = api.plan(config)
                with Path(args.output).open("x", encoding="utf-8") as f:
                    json.dump(value, f, ensure_ascii=False, indent=2)
            elif args.command == "import-result":
                directory, result = api.import_result(config, args.bundle, args.output)
                print(json.dumps({"directory": str(directory), "execution": result["execution"], "evidence_valid": result["evidence_valid"], "issues": result.get("issues"), "error": result.get("error")}, ensure_ascii=False, indent=2))
                return 0 if result["evidence_valid"] else 2
            else:
                directory, result = api.run(config, args.output)
                value = {"directory": str(directory), "execution": result["execution"], "evidence_valid": result["evidence_valid"], "statistics": result.get("statistics"), "error": result.get("error")}
                print(json.dumps(value, ensure_ascii=False, indent=2))
                return 0 if result["evidence_valid"] else 2
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
