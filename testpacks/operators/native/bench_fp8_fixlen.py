import argparse
import csv
from pathlib import Path

from _flash_attn_build_loader import load_flash_attn_extension

import numpy as np
import torch
import triton


def load_extension(args):
    return load_flash_attn_extension(use_whl=args.whl, extension_so=args.so)


def make_inputs(args, device):
    h_kv = args.h // args.g
    q = torch.randn(args.bs, args.s, args.h, args.d, device=device, dtype=torch.float16) * 0.5
    k = torch.randn(args.bs, args.s, h_kv, args.d, device=device, dtype=torch.float16) * 0.5
    v = torch.randn(args.bs, args.s, h_kv, args.dv, device=device, dtype=torch.float16) * 0.5
    q = q.to(torch.float8_e4m3fn)
    k = k.to(torch.float8_e4m3fn)
    v = v.to(torch.float8_e4m3fn)
    q_descale = torch.ones((args.bs, args.h), device=device, dtype=torch.float32)
    k_descale = torch.ones((args.bs, h_kv), device=device, dtype=torch.float32)
    v_descale = torch.ones((args.bs, h_kv), device=device, dtype=torch.float32)
    return q, k, v, q_descale, k_descale, v_descale


def flops(args):
    useful = args.s * args.s
    if args.causal:
        useful *= 0.5
    return 2 * args.bs * args.h * useful * args.d + 2 * args.bs * args.h * useful * args.dv


def run_once(flash_attn, tensors, args):
    q, k, v, q_descale, k_descale, v_descale = tensors
    return flash_attn.fwd(
        q,
        k,
        v,
        None,
        None,
        0.0,
        args.softmax_scale,
        args.causal,
        args.window_left,
        args.window_right,
        args.softcap,
        False,
        None,
        False,
        q_descale,
        k_descale,
        v_descale,
        None,
        0.0,
    )[0]


def filter_perf(samples):
    perf = np.asarray(samples, dtype=np.float64)
    if perf.size == 0:
        return perf

    mean = perf.mean()
    perf = np.delete(perf, np.where(perf < (0.8 * mean)))
    if perf.size == 0:
        return np.asarray(samples, dtype=np.float64)

    mean = perf.mean()
    perf = np.delete(perf, np.where(perf > (1.25 * mean)))
    return perf if perf.size else np.asarray(samples, dtype=np.float64)


def bench(flash_attn, args, device):
    flop = flops(args)
    perf = []
    for i in range(args.iterations):
        torch.cuda.empty_cache()
        tensors = make_inputs(args, device)

        def call():
            run_once(flash_attn, tensors, args)

        t = triton.testing.do_bench(call, rep=args.repeats, return_mode="mean") / 1e3
        if i > 0:
            perf.append(flop / t / 1.0e12)
        torch.cuda.synchronize()
        del tensors

    if not perf:
        raise RuntimeError("--iterations must be at least 2")

    used = filter_perf(perf)
    tflops = float(used.mean())
    mean_ms = float(flop / (tflops * 1.0e12) * 1000.0)
    return {
        "raw_tflops": float(np.asarray(perf, dtype=np.float64).mean()),
        "mean_ms": mean_ms,
        "min_tflops": float(used.min()),
        "max_tflops": float(used.max()),
        "std_tflops": float(used.std()),
        "samples_used": int(used.size),
        "tflops": tflops,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--whl", action="store_true", help="load flash_attn_cuda from installed flash_attn wheel")
    parser.add_argument("--so", default=None, help="path to flash_attn_2_cuda*.so for source-tree testing")
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--output", default="mha_fixlen_fp8_fwd_results.csv")
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--h", type=int, default=16)
    parser.add_argument("--g", type=int, default=8)
    parser.add_argument("--s", type=int, default=16384)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--dv", type=int, default=128)
    parser.add_argument("--softmax-scale", type=float, default=0.088)
    parser.add_argument("--no-causal", dest="causal", action="store_false")
    parser.add_argument("--window-left", type=int, default=-1)
    parser.add_argument("--window-right", type=int, default=0)
    parser.add_argument("--softcap", type=float, default=0.0)
    parser.add_argument("--bf16-output", action="store_true")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.h % args.g != 0:
        raise ValueError("--h must be divisible by --g")
    if args.whl and args.so:
        raise ValueError("--whl and --so cannot be used together")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device is not available")

    flash_attn, source = load_extension(args)
    torch.manual_seed(args.seed)
    stats = bench(flash_attn, args, "cuda")

    row = {
        "op": "mha_fixlen_fwd",
        "dtype": "fp8_e4m3",
        "out": "none",
        "output_dtype": "bf16" if args.bf16_output else "fp16",
        "batch_size": args.bs,
        "seqlen_q": args.s,
        "num_heads_q": args.h,
        "seqlen_kv": args.s,
        "num_heads_kv": args.h // args.g,
        "dim_qk": args.d,
        "dim_v": args.dv,
        "softmax_scale": args.softmax_scale,
        "is_causal": int(args.causal),
        "window_size_left": args.window_left,
        "window_size_right": args.window_right,
        "softcap": args.softcap,
        "repeats": args.repeats,
        "iterations": args.iterations,
        "source": str(source),
        **stats,
    }

    print(f"{row['tflops']:.2f} TFLOPS, {row['mean_ms']:.4f} ms")

    if args.csv:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
