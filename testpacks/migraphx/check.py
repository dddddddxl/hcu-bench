"""Target-container dependency preflight; never executes a model benchmark."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from testpacks.migraphx.dependencies import DRIVER, RECIPES, check


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=RECIPES, default="resnet50_fp16")
    parser.add_argument("--binary", default=DRIVER)
    parser.add_argument("--numactl", default="numactl")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = check(args.recipe, args.binary, args.numactl)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
