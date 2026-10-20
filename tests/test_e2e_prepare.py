"""Isolated client installation commands; no package downloads or HCU usage."""
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from testpacks.e2e import prepare as client


class ClientPrepareTests(unittest.TestCase):
    def test_auto_client_does_not_require_ensurepip_or_modify_serving_python(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "output"
            output.mkdir()
            calls = []
            session = {"tasks": [{"suite": "e2e_perf", "params": {"python": "auto"}}]}
            with patch.dict(os.environ, {"HCU_BENCH_BUILD_CONFIG": json.dumps({"cache_dir": folder})}), \
                    patch.object(client, "acquire_locks", return_value=[]), \
                    patch.object(client, "execute", side_effect=lambda argv, *args: calls.append(argv)):
                client.prepare(session, output, threading.Event())
                self.assertIn("--without-pip", calls[0])
                install = next(argv for argv in calls if "install" in argv)
                self.assertEqual(install[:4], [client.sys.executable, "-m", "pip", "--python"])
                self.assertTrue(str(install[4]).startswith(str(root)))
                self.assertEqual(calls[-1][-1], "freeze")
                calls.clear()
                client.prepare(session, output, threading.Event())
                self.assertFalse(any("install" in argv for argv in calls))

    def test_explicit_client_is_verified_without_installing(self):
        with tempfile.TemporaryDirectory() as folder:
            calls = []
            with patch.object(client, "execute", side_effect=lambda argv, *args: calls.append(argv)):
                client.prepare({"tasks": [{"suite": "e2e_perf", "params": {"python": "/client/python"}}]},
                               Path(folder), threading.Event())
            self.assertEqual(calls[-1], ["/client/python", "-m", "pip", "freeze"])
            self.assertFalse(any("install" in argv for argv in calls))
