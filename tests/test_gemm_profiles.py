# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Run the supplied GEMM profile regressions as part of normal repository CI."""
from pathlib import Path
import subprocess
import sys
import unittest


class BundledGemmProfiles(unittest.TestCase):
    def test_native_cli_captures_and_generated_commands(self):
        root = Path(__file__).resolve().parents[1] / "vendor/operator-cases/gemm"
        result = subprocess.run(
            [sys.executable, "-B", "-m", "unittest", "discover", "-s", "validation", "-v"],
            cwd=root, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
