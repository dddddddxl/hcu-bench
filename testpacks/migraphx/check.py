"""Target-container dependency preflight; never executes a model benchmark."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from testpacks.migraphx.dependencies import DRIVER, RECIPES, check


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=RECIPES, default="resnet50_fp16")
    parser.add_argument("--binary", default=DRIVER)
    parser.add_argument("--numactl", default="numactl")
    parser.add_argument("--model", type=Path, help="Optionally inspect the local ONNX input contract without running inference")
    parser.add_argument("--batch", type=int, choices=(1, 32), nargs="+", default=[1, 32],
                        help="Batches to verify when --model is provided (default: 1 32)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = check(args.recipe, args.binary, args.numactl)
    if args.model is not None:
        from testpacks.migraphx.run import inspect_model
        result["scope"] = "dependencies and ONNX input metadata only; no GPU inference, performance or numerical validation"
        result["model"] = {"path": str(args.model.resolve()), "inputs_by_batch": {}}
        for batch in dict.fromkeys(args.batch):
            try:
                inputs = inspect_model(SimpleNamespace(recipe=args.recipe, model=str(args.model), batch=batch))
                result["model"]["inputs_by_batch"][str(batch)] = inputs
            except Exception as exc:
                result["errors"].append(f"ONNX validation failed for batch {batch}: {exc}")
        result["ready"] = not result["errors"]
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
