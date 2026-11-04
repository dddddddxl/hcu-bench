# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Generate the unchanged DTK general and GEMM smoke catalog."""
import csv
import hashlib
import json
from pathlib import Path
from dtk_cases import source_cases

ROOT = Path(__file__).resolve().parent
fields = ("case_id", "group", "function", "dtype", "M", "N", "K", "transA", "transB",
          "lda", "ldb", "ldc", "ldd", "alpha", "beta", "alphai", "betai", "batch_count",
          "stride_a", "stride_b", "stride_c", "stride_d", "a_type", "b_type", "c_type", "d_type",
          "compute_type", "initialization", "cold_iters", "check_iters", "use_hipblaslt",
          "source", "source_test", "source_index", "source_case")
types = {"f32_r": "fp32", "f64_r": "fp64", "f16_r": "fp16", "f32_c": "complex64", "f64_c": "complex128"}
rows = []
for case in source_cases():
    group = "general" if case["source_file"] == "rocblas_general.yaml" else "smoke"
    row = {key: case[key] for key in fields if key in case}
    row.update(case_id=f"dtk-{group}-{case['source_index']:02}-{case['source_case']:04}",
               group=group, dtype=types[case["a_type"]], source=case["source_file"], check_iters=case["iters"])
    rows.append(row)
with (ROOT / "gemm_cases.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
provenance = {
    "source_directory": "/public/dtk/autotest_platform/dcc_cp/rocblas",
    "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                      for name in ("rocblas_general.yaml", "rocblas_common.yaml", "rocblas_smoke.yaml", "rocblas_gentest.py")},
    "cases": {"general": sum(r["group"] == "general" for r in rows),
              "gemm_smoke": sum(r["group"] == "smoke" for r in rows), "total": len(rows)},
    "scope": "Only retained DTK general and GEMM smoke. No added shapes or precision variants. Not all BLAS or gtest branches.",
    "settings": "Original per-case initialization, types, scalars, shapes, strides and iteration counts. New profile uses unit_check=1, norm_check=0 and use_hipblaslt=-1.",
}
(ROOT / "gemm_sources.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8", newline="\n")
print("Generated %d unchanged DTK general/GEMM smoke entries" % len(rows))
