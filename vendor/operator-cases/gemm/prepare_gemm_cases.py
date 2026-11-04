"""Build the small, traceable GEMM starter suite from local reference files."""
import csv
import hashlib
import json
from pathlib import Path
import re

from run_gemm_cases import INITIALIZATION

ROOT = Path(__file__).resolve().parent
source_path = ROOT / "rocblas_general.yaml"
lines = source_path.read_text(encoding="utf-8-sig").splitlines()
groups = {}
anchor = None
for line_number, line in enumerate(lines, 1):
    match = re.match(r"^\S.*: &(\w+)\s*$", line)
    if match:
        anchor = match.group(1)
    match = re.match(r"^\s*-\s*\{\s*M:\s*(.*?)\}\s*$", line)
    if anchor and match:
        values = {}
        for entry in ("M: " + match.group(1)).split(","):
            key, value = entry.split(":", 1)
            values[key.strip()] = value.strip()
        groups.setdefault(anchor, []).append((values, line_number))

rows = []


def add(case_id, group, function, data, source, line_number):
    row = dict(data)
    row.update(case_id=case_id, group=group, function=function,
               ldd=data["ldc"], alpha="1", beta="0",
               source=source, source_line=line_number)
    row.setdefault("batch_count", "1")
    if function == "gemm_strided_batched":
        row["stride_d"] = row["stride_c"]
    rows.append(row)


for size in (1024, 2048, 4096, 8192):
    data = {key: str(size) for key in ("M", "N", "K", "lda", "ldb", "ldc")}
    data.update(transA="N", transB="N")
    add("square-%d" % size, "square", "gemm", data, "added_square_sweep", "")

selections = [
    ("dlrm-01", "dlrm", "sgemm_dlrm2_sizes", 257, 1024, 4096, "N", "N", 1),
    ("dlrm-02", "dlrm", "sgemm_dlrm2_sizes", 3200, 1024, 2048, "N", "N", 1),
    ("dlrm-03", "dlrm", "sgemm_dlrm2_sizes", 4096, 4096, 1024, "N", "T", 1),
    ("bert-01", "bert", "sgemm_bert_big_sizes", 1024, 3072, 1024, "N", "N", 1),
    ("bert-02", "bert", "sgemm_bert_big_sizes", 1024, 1024, 3072, "N", "T", 1),
    ("bert-03", "bert", "sgemm_bert_big_sizes", 1024, 2048, 3072, "T", "N", 1),
    ("transformer-01", "transformer", "sgemm_transformer_sizes", 2048, 1139, 512, "N", "N", 1),
    ("transformer-02", "transformer", "sgemm_transformer_sizes", 2048, 1184, 512, "N", "N", 1),
    ("batch-01", "batched", "sgemm_sb_bert_big_sizes", 64, 512, 512, "N", "N", 16),
    ("batch-02", "batched", "sgemm_sb_bert_big_sizes", 512, 512, 64, "T", "N", 64),
]
for case_id, group, anchor, m, n, k, ta, tb, batch in selections:
    matches = [(d, ln) for d, ln in groups[anchor]
               if (int(d["M"]), int(d["N"]), int(d["K"]), d["transA"], d["transB"], int(d.get("batch_count", 1)))
               == (m, n, k, ta, tb, batch)]
    if len(matches) != 1:
        raise ValueError("Selection is not unique: " + case_id)
    data, line_number = matches[0]
    function = "gemm_strided_batched" if batch > 1 else "gemm"
    add(case_id, group, function, data, "rocblas_general.yaml:" + anchor, line_number)

fields = ("case_id", "group", "function", "M", "N", "K", "transA", "transB",
          "lda", "ldb", "ldc", "ldd", "alpha", "beta", "batch_count",
          "stride_a", "stride_b", "stride_c", "stride_d", "source", "source_line")
with (ROOT / "gemm_cases.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

provenance = {
    "source_host_alias": "nmz36", "source_hostname": "github-nmz2",
    "source_directory": "/public/dtk/autotest_platform/dcc_cp/rocblas",
    "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                      for p in ("rocblas_general.yaml", "rocblas_common.yaml")},
    "cases": {"total": 14, "added_square_sweep": 4, "selected_source_cases": 10},
    "selected_scalars": "Selected source Tests bind alpha_1_beta_0; alpha=1 and beta=0 retained.",
    "dtype_scope": "Selected model cases were SGEMM. FP16/BF16 variants with FP32 accumulation are new extensions.",
    "fixed_settings": {"initialization": INITIALIZATION, "math_mode": 0, "use_hipblaslt": 0, "flush_batch_count": 1},
    "scope": "Starter suite, not the full YAML workload and not established as the nightly workload.",
    "verification": "CLI help and version inspected in megatron_test; no GPU benchmark executed.",
}
with (ROOT / "gemm_sources.json").open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(json.dumps(provenance, indent=2))
print("Generated %d cases: 4 added square cases and 10 selected source cases" % len(rows))
