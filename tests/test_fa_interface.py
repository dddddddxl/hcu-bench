# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU interface/dispatch fixtures; no FlashAttention performance is measured."""
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hcu_bench.results import parse_sample
from testpacks.operators import run as op
from testpacks.operators.native import _flash_attn_build_loader as loader

ROOT = Path(__file__).resolve().parents[1]
# Exported binding from the reported incompatible wheel, without input tensor dumps.
FWD16 = "fwd(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: torch.Tensor | None, arg4: torch.Tensor | None, arg5: typing.SupportsFloat, arg6: typing.SupportsFloat, arg7: bool, arg8: typing.SupportsInt, arg9: typing.SupportsInt, arg10: typing.SupportsFloat, arg11: bool, arg12: torch.Generator | None, arg13: bool, arg14: torch.Tensor | None, arg15: typing.SupportsFloat) -> list[torch.Tensor]"
FWD19 = "fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor | None, alibi: torch.Tensor | None, dropout: float, scale: float, causal: bool, wl: int, wr: int, softcap: float, return_softmax: bool, gen: torch.Generator | None, bhsd: bool, q_descale: torch.Tensor | None, k_descale: torch.Tensor | None, v_descale: torch.Tensor | None, sink: torch.Tensor | None, sink_scale: float) -> list[torch.Tensor]"


class FlashAttentionInterfaceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.module = self.root / "extension.py"
        self.module.write_text("# CPU interface fixture\n", encoding="utf-8")

    def extension(self, doc):
        def fwd(*args):
            self.fail("Interface inspection called the benchmark kernel")
        fwd.__doc__ = doc
        return SimpleNamespace(fwd=fwd, __file__=str(self.module))

    def test_reported_16_argument_interface_is_incompatible(self):
        report = loader.inspect_fp8_interface(self.extension(FWD16))
        self.assertFalse(report["interface_compatible"])
        self.assertEqual(report["exported_signatures"][0]["argument_count"], 16)
        self.assertIn("expected 19", report["reason"])
        self.assertIn("q_descale", report["reason"])
        self.assertEqual(report["extension_sha256"], hashlib.sha256(self.module.read_bytes()).hexdigest())

    def test_compatible_optional_and_overloaded_signatures(self):
        for doc in (FWD19, FWD19.replace("torch.Tensor | None", "typing.Optional[torch.Tensor]"),
                    "fwd(*args, **kwargs)\nOverloaded function.\n1. " + FWD16 + "\n2. " + FWD19):
            with self.subTest(doc=doc[:60]):
                self.assertTrue(loader.inspect_fp8_interface(self.extension(doc))["interface_compatible"])

    def test_unknown_or_wrong_types_are_not_guessed_from_arity(self):
        for doc in (None, "fwd(*args, **kwargs)", FWD19.replace("v_descale: torch.Tensor | None", "v_descale: float")):
            with self.subTest(doc=doc):
                self.assertFalse(loader.inspect_fp8_interface(self.extension(doc))["interface_compatible"])
        self.assertFalse(loader.inspect_fp8_interface(SimpleNamespace())["interface_compatible"])

    def test_contract_preserves_the_native_descale_arguments(self):
        tree = ast.parse((op.ROOT / "native/bench_fp8_fixlen.py").read_text(encoding="utf-8"))
        call, = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "fwd"]
        self.assertEqual(len(call.args), 19)
        self.assertEqual([n.id for n in call.args[14:17]], ["q_descale", "k_descale", "v_descale"])

    def install_fixture(self, doc):
        package = self.root / "flash_attn"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "flash_attn_interface.py").write_text("from . import extension as flash_attn_cuda\n", encoding="utf-8")
        (package / "extension.py").write_text(
            "def fwd(*args):\n    raise AssertionError('preflight must not launch fwd')\n"
            f"fwd.__doc__ = {doc!r}\n", encoding="utf-8")
        # The real device probe runs against a CPU-only device API fixture.
        (self.root / "torch.py").write_text(
            "from types import SimpleNamespace\n"
            "__version__ = 'cpu-fixture'\nversion = SimpleNamespace(hip='fixture')\n"
            "cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, "
            "get_device_properties=lambda _: SimpleNamespace(name='fixture', total_memory=1))\n",
            encoding="utf-8")

    def test_standalone_inspection_preserves_evidence_and_never_calls_fwd(self):
        self.install_fixture(FWD16)
        result = subprocess.run([sys.executable, str(Path(loader.__file__).resolve()), "--output", "interface.json"],
                                cwd=self.root, env={**os.environ, "PYTHONPATH": str(self.root)},
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        report = json.loads((self.root / "interface.json").read_text(encoding="utf-8"))
        self.assertFalse(report["interface_compatible"])
        self.assertEqual(report["check_scope"], "interface_only")
        self.assertEqual(report["correctness"], "not_checked")
        self.assertIn("flash-attn", report["package_versions"])

    def test_unloadable_extension_is_reported_as_unavailable(self):
        self.install_fixture(FWD19)
        (self.root / "flash_attn/flash_attn_interface.py").write_text(
            "raise ImportError('fixture extension has an unresolved symbol')\n", encoding="utf-8")
        result = subprocess.run([sys.executable, str(Path(loader.__file__).resolve())], cwd=self.root,
                                env={**os.environ, "PYTHONPATH": str(self.root)}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        report = json.loads((self.root / loader.INTERFACE_REPORT).read_text(encoding="utf-8"))
        self.assertFalse(report["interface_compatible"])
        self.assertIn("unresolved symbol", report["reason"])

    def test_standalone_loader_refuses_incompatible_interface(self):
        self.install_fixture(FWD16)
        command = (f"import sys; sys.path.insert(0, {str(Path(loader.__file__).resolve().parent)!r}); "
                   "from _flash_attn_build_loader import load_flash_attn_extension; "
                   "load_flash_attn_extension(use_whl=True)")
        result = subprocess.run([sys.executable, "-c", command], cwd=self.root,
                                env={**os.environ, "PYTHONPATH": str(self.root)}, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected 19", result.stderr)
        self.assertNotIn("preflight must not launch fwd", result.stderr)

    def invoke_operator(self, doc):
        self.install_fixture(doc)
        marker = self.root / "native-started"
        command = [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch(); raise SystemExit(7)"]
        args = SimpleNamespace(recipe="fa_fp8_forward", native_timeout_s=10, hcu_ids="0", nnodes=1,
                               node_rank=0, output_dir=str(self.root / "out"))
        capture = io.StringIO()
        with patch.dict(os.environ, {"PYTHONPATH": str(self.root)}), \
                patch.object(op, "command", return_value=command), contextlib.redirect_stdout(capture):
            code = op.run(args, threading.Event())
        samples = [parse_sample(line) for line in capture.getvalue().splitlines() if line.startswith("BENCH_RESULT ")]
        return code, marker.exists(), samples, Path(args.output_dir)

    def test_incompatible_interface_skips_before_native_launch(self):
        code, launched, samples, out = self.invoke_operator(FWD16)
        self.assertEqual(code, 0)
        self.assertFalse(launched)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["measurement_status"], "skipped")
        self.assertTrue(samples[0]["skip_case"])
        self.assertEqual(samples[0]["metrics"], [])
        self.assertEqual(samples[0]["correctness"], "not_checked")
        self.assertFalse((out / "measurements.json").exists())
        self.assertFalse((out / "native.log").exists())
        self.assertTrue((out / loader.INTERFACE_REPORT).exists())
        self.assertEqual(json.loads((out / "assessment.json").read_text())["status"], "skipped")
        self.assertEqual(len(json.loads((out / "execution.json").read_text())), 1)

    def test_compatible_interface_routes_to_native_and_runtime_errors_remain_failures(self):
        code, launched, samples, out = self.invoke_operator(FWD19)
        self.assertEqual(code, 2)
        self.assertTrue(launched)
        self.assertEqual(samples[0]["measurement_status"], "invalid")
        self.assertNotIn("skip_case", samples[0])
        self.assertEqual(len(json.loads((out / "execution.json").read_text())), 2)
