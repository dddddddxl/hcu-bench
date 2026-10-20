"""Three independent MIGraphX perf rounds, twenty native iterations per round."""
import argparse
import csv
import os
from pathlib import Path
import re
import shutil
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_common import CleanupError, cancellation, cleanup_failure, emit, sha256, versions, write_json
from fixed_common import claim_output, environment_record, execute, positive, single_device

RECIPES = ("resnet50_fp16", "resnet50_int8", "bert_large_fp16")
ROUNDS, ITERATIONS = 3, 20
NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def executable(value):
    path = shutil.which(value)
    if not path:
        raise ValueError(f"Prepared executable is missing: {value}")
    return str(Path(path).resolve())


def command(args):
    argv = [args.numactl, f"--cpubind={args.numa_node}", f"--membind={args.numa_node}",
            args.binary, "perf", "-n", str(ITERATIONS), "--onnx", args.model]
    if args.recipe == "bert_large_fp16":
        argv += ["--fill1", "input_ids", "--fill1", "attention_mask", "--fill1", "token_type_ids",
                 "--batch", str(args.batch), "--gpu", "--fp16"]
    else:
        argv += ["--batch", str(args.batch)]
        argv += ["--fp16", "--opt-compile"] if args.recipe == "resnet50_fp16" else ["--int8"]
        argv += ["--gpu"]
    return argv


def validate(args, env):
    if args.recipe not in RECIPES or args.batch not in (1, 32) or args.numa_node < 0:
        raise ValueError("Unsupported fixed model recipe/batch/NUMA node")
    positive(args.round_timeout_s)
    env = single_device(args, env)
    args.model = str(Path(args.model).resolve())
    if not Path(args.model).is_file():
        raise ValueError("Local ONNX model missing; no model is downloaded")
    args.binary, args.numactl = executable(args.binary), executable(args.numactl)
    if args.recipe == "bert_large_fp16":
        if not args.rocblas_dir or not (Path(args.rocblas_dir) / "lib").is_dir():
            raise ValueError("BERT requires rocblas_dir with the matching prepared lib directory")
        lib = str((Path(args.rocblas_dir) / "lib").resolve())
        env["LD_LIBRARY_PATH"] = lib + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    elif args.rocblas_dir:
        raise ValueError("rocblas_dir is only part of the BERT recipe")
    return env


def inspect_model(args):
    try:
        import onnx
    except ImportError as exc:
        raise ValueError("Prepared environment needs onnx to verify fixed input shapes/types; no packages are installed") from exc
    model = onnx.load(args.model, load_external_data=False)
    initializers = {v.name for v in model.graph.initializer}
    inputs = {v.name: v.type.tensor_type for v in model.graph.input if v.name not in initializers}
    bert = args.recipe == "bert_large_fp16"
    if bert and set(inputs) != {"input_ids", "attention_mask", "token_type_ids"}:
        raise ValueError("BERT requires exactly input_ids, attention_mask and token_type_ids")
    if not bert and len(inputs) != 1:
        raise ValueError("ResNet50 requires one tensor input")
    expected_tail, expected_type = ([384], 7) if bert else ([3, 224, 224], 1)
    metadata = {}
    for name, tensor in inputs.items():
        dims = list(tensor.shape.dim)
        if tensor.elem_type != expected_type or len(dims) != len(expected_tail) + 1:
            raise ValueError(f"Unexpected ONNX dtype/rank: {name}")
        if [d.dim_value for d in dims[1:]] != expected_tail:
            raise ValueError(f"Unexpected fixed ONNX input dimensions: {name}")
        # --batch supplies dynamic batch size; it does not prove a static model was reshaped.
        if dims[0].HasField("dim_value") and dims[0].dim_value != args.batch:
            raise ValueError(f"Static ONNX batch does not match requested batch {args.batch}: {name}")
        metadata[name] = dict(dtype="int64" if bert else "float32", shape=[args.batch, *expected_tail],
                              source_batch=dims[0].dim_param or dims[0].dim_value)
    # External weights need their own evidence; reject them rather than fingerprint only the graph.
    from onnx.external_data_helper import _get_all_tensors
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in _get_all_tensors(model)):
        raise ValueError("External ONNX weights are unsupported by this fixed pack; provide a self-contained ONNX model")
    return metadata


def tuning_fingerprint(directory, stopped):
    files = []
    root = Path(directory) / "lib"
    for path in sorted(root.rglob("*")):
        if stopped.is_set():
            raise RuntimeError("Cancelled while fingerprinting tuning library")
        if path.is_symlink() and path.is_dir():
            raise ValueError("Tuning library directory symlinks cannot be fully fingerprinted; use a self-contained lib tree")
        if path.is_file():
            files.append(dict(path=path.relative_to(root).as_posix(), size=path.stat().st_size, sha256=sha256(path)))
    if not files:
        raise ValueError("rocBLAS tuning lib directory is empty")
    return files


def parse(text, batch):
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    rates, times = [], []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Rate:"):
            match = re.fullmatch(rf"Rate:\s*({NUMBER})\s*(?:inferences/sec|inf/s)", line)
            if not match:
                raise ValueError("Unsupported MIGraphX Rate format")
            rates.append(positive(match[1]))
        elif line.startswith("Total time:"):
            match = re.fullmatch(rf"Total time:\s*({NUMBER})\s*ms(?:/batch)?(?:\s+\([^\n]*\))?", line)
            if not match:
                raise ValueError("Unsupported MIGraphX Total time format")
            times.append(positive(match[1]))
        elif line.startswith("Batch size:") and line != f"Batch size: {batch}":
            raise ValueError("Native MIGraphX batch differs from request")
    if len(rates) != 1 or len(times) != 1:
        raise ValueError("Expected exactly one native Rate and Total time per round")
    # MIGraphX program::perf_report emits rate = batch * 1000 / total_time.
    # Total time is one batch's latency, not the sum of all twenty iterations.
    if abs(rates[0] * times[0] / (batch * 1000) - 1) > .02:
        raise ValueError("Native Rate/Total time units or batch are inconsistent")
    return dict(throughput_inf_s=rates[0], latency_ms_batch=times[0])


def run(args, stopped):
    out = claim_output(args.output_dir)
    rows, executions = [], []
    try:
        env = validate(args, dict(os.environ))
        inputs = inspect_model(args)
        tuning = tuning_fingerprint(args.rocblas_dir, stopped) if args.recipe == "bert_large_fp16" else None
        write_json(out / "plan.json", dict(params=vars(args), argv=command(args), inputs=inputs,
                   rounds=ROUNDS, iterations_per_round=ITERATIONS,
                   input_policy="fill1" if args.recipe == "bert_large_fp16" else "native generated",
                   model_sha256=sha256(args.model), binary_sha256=sha256(args.binary),
                   numactl_sha256=sha256(args.numactl), wrapper_sha256=sha256(__file__),
                   tuning_library=tuning, environment=environment_record(env),
                   versions=versions(["onnx", "migraphx"]), correctness="not_checked"))
        for index in range(1, ROUNDS + 1):
            text = execute(command(args), f"round-{index}", out, env, stopped, executions, args.round_timeout_s)
            rows.append(dict(round=index, **parse(text, args.batch)))
            write_json(out / "rounds.json", rows)
            with (out / "rounds.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("round", "throughput_inf_s", "latency_ms_batch"))
                writer.writeheader()
                writer.writerows(rows)
        if stopped.is_set():
            raise RuntimeError("Cancelled before publishing all three rounds")
        result = dict(measurement_status="measured", correctness="not_checked", recipe=args.recipe,
                      batch=args.batch, rounds=rows, accuracy_scope="performance only; no numerical/SQuAD accuracy",
                      metrics=[dict(name="throughput", unit="inf/s", statistic="median_of_3_rounds",
                                    value=statistics.median(r["throughput_inf_s"] for r in rows)),
                               dict(name="batch_latency", unit="ms/batch", statistic="median_of_3_rounds",
                                    value=statistics.median(r["latency_ms_batch"] for r in rows))])
        write_json(out / "measurements.json", result)
        write_json(out / "assessment.json", dict(complete=True, completed_rounds=len(rows), correctness="not_checked"))
        emit(result)
        return 0
    except CleanupError as exc:
        return cleanup_failure(out, exc)
    except Exception as exc:
        write_json(out / "assessment.json", dict(complete=False, completed_rounds=len(rows), correctness="not_checked", error=str(exc)))
        emit(dict(measurement_status="invalid", correctness="not_checked", metrics=[], reason=str(exc)))
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=RECIPES, required=True)
    parser.add_argument("--batch", type=int, choices=(1, 32), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--numa-node", type=int, required=True)
    parser.add_argument("--rocblas-dir", default="")
    parser.add_argument("--binary", default="migraphx-driver")
    parser.add_argument("--numactl", default="numactl")
    parser.add_argument("--round-timeout-s", type=int, default=600)
    parser.add_argument("--hcu-ids", required=True)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    return run(parser.parse_args(), cancellation())


if __name__ == "__main__":
    raise SystemExit(main())
