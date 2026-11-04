"""Bench adapter for the wheel-only recipe; the supplied archive omitted this helper."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re


INTERFACE_REPORT = "fa-fp8-interface.json"
# This is the positional contract of bench_fp8_fixlen.run_once, including all
# three descales. A 16-argument binding cannot represent this contract.
FP8_ARGUMENT_TYPES = (
    "tensor", "tensor", "tensor", "tensor?", "tensor?", "float", "float",
    "bool", "int", "int", "float", "bool", "generator?", "bool",
    "tensor", "tensor", "tensor", "tensor?", "float",
)


def inspect_fp8_interface(extension):
    """Inspect pybind's exported signature without calling fwd or allocating tensors."""
    fwd = getattr(extension, "fwd", None)
    doc = getattr(fwd, "__doc__", "") or ""
    signatures = []
    aliases = {"torch.Tensor": "tensor", "torch.Tensor|None": "tensor?",
               "torch.Generator|None": "generator?", "SupportsFloat": "float",
               "SupportsInt": "int", "float": "float", "int": "int", "bool": "bool"}
    for signature in re.findall(r"^\s*(?:\d+\.\s*)?fwd\((.*)\)\s*->", doc, re.M):
        normalized = signature.replace("typing.", "").replace(" ", "")
        normalized = re.sub(r"Optional\[([^\[\]]+)\]", r"\1|None", normalized)
        types = [aliases.get(arg.partition(":")[2].split("=", 1)[0], "unknown")
                 for arg in normalized.split(",") if arg]
        signatures.append(dict(signature="fwd(" + signature + ")", argument_types=types,
                               argument_count=len(types)))
    compatible = callable(fwd) and any(
        len(s["argument_types"]) == len(FP8_ARGUMENT_TYPES)
        and all(actual == expected or (expected == "tensor" and actual == "tensor?")
                for actual, expected in zip(s["argument_types"], FP8_ARGUMENT_TYPES))
        for s in signatures)
    source = getattr(extension, "__file__", None)
    digest = None
    if source:
        fingerprint = hashlib.sha256()
        with Path(source).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                fingerprint.update(chunk)
        digest = fingerprint.hexdigest()
    counts = sorted({s["argument_count"] for s in signatures})
    reason = ("FA FP8 fwd matches the 19-argument recipe interface; kernel support and correctness are not checked"
              if compatible else
              f"FA FP8 interface incompatible: fwd exposes argument counts {counts}; expected 19 positional "
              "arguments with q_descale, k_descale and v_descale at positions 15-17 and matching types. "
              "Use a prepared FlashAttention wheel exporting this FP8 interface. No arguments are dropped.")
    return dict(interface_compatible=compatible, reason=reason, extension_path=source,
                extension_sha256=digest,
                expected_argument_types=list(FP8_ARGUMENT_TYPES), exported_signatures=signatures,
                fwd_doc=doc)


def inspect_installed_fp8(output=INTERFACE_REPORT):
    versions = {}
    for name in ("flash-attn", "torch"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    extension = None
    try:
        from flash_attn.flash_attn_interface import flash_attn_cuda
        extension = flash_attn_cuda
        report = inspect_fp8_interface(extension)
    except (ImportError, OSError) as exc:
        report = dict(interface_compatible=False, reason=f"FA FP8 extension unavailable: {exc}",
                      extension_path=None, extension_sha256=None, exported_signatures=[])
    report.update(package_versions=versions, check_scope="interface_only", correctness="not_checked")
    Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return extension, report


def load_flash_attn_extension(*, use_whl=False, extension_so=None):
    if not use_whl or extension_so is not None:
        raise ValueError("This fixed recipe supports only --whl with the prepared flash_attn wheel")
    flash_attn_cuda, report = inspect_installed_fp8()
    if not report["interface_compatible"]:
        raise RuntimeError(report["reason"] + f" Evidence: {INTERFACE_REPORT}")
    return flash_attn_cuda, flash_attn_cuda.__file__


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the installed FA FP8 interface without running a benchmark")
    parser.add_argument("--output", default=INTERFACE_REPORT)
    args = parser.parse_args()
    _, report = inspect_installed_fp8(args.output)
    print(report["reason"] + f"; evidence: {args.output}")
    raise SystemExit(0 if report["interface_compatible"] else 2)
