# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Expand the retained DTK YAML with its original generator; generation only."""
from pathlib import Path

import yaml
import rocblas_gentest as native

ROOT = Path(__file__).resolve().parent


def source_cases():
    cases = []
    for filename in ("rocblas_general.yaml", "rocblas_smoke.yaml"):
        text = "\n".join(line for path in (ROOT / "rocblas_common.yaml", ROOT / filename)
                         for line in path.read_text(encoding="utf-8-sig").splitlines()
                         if line.strip() not in ("---", "...") and not line.startswith("include:"))
        doc = yaml.safe_load(text)
        native.param.update(dict_lists_to_expand=doc["Dictionary lists to expand"],
                            lists_to_not_expand=doc["Lists to not expand"], Functions=doc.get("Functions", {}))
        for index, test in enumerate(doc["Tests"], 1):
            if not test["name"].startswith(("sgemm", "dgemm", "gemm")):
                continue
            ordinal = [0]

            def collect(values):
                case = dict(values)
                native.setdefaults(case)
                ordinal[0] += 1
                case.update(source_file=filename, source_test=test["name"],
                            source_index=index, source_case=ordinal[0])
                cases.append(case)

            native.generate({**doc["Defaults"], **test}, collect)
    return cases
