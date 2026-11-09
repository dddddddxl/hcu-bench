"""CPU contracts only; these fixtures are not fresh HCU measurements."""
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hcu_bench.models import BenchError
from hcu_bench.planner import make_plan
from testpacks.gemmpower import run

ROOT = Path(__file__).resolve().parents[1]


class GemmpowerContracts(unittest.TestCase):
    def test_default_main_and_standalone_are_disabled(self):
        with patch('subprocess.Popen', side_effect=AssertionError('static only')):
            main = make_plan(str(ROOT / 'configs/bench.template.yaml'))
            self.assertFalse(any(c.test_id == 'gemmpower' for c in main.cases))
            with self.assertRaisesRegex(BenchError, 'No enabled tests'):
                make_plan(str(ROOT / 'configs/gemmpower.template.yaml'))
            plan = make_plan(str(ROOT / 'configs/gemmpower.template.yaml'), 'operators', ['tests.gemmpower.enabled=true'])
        self.assertEqual(len(plan.cases), 1)
        self.assertEqual(plan.cases[0].params['iterations'], 4)
        self.assertNotIn('duration_s', plan.cases[0].params)

    def test_architecture_guard_is_fixed_and_fails_closed(self):
        self.assertTrue(run.compatible(dict(architecture='gfx938:sramecc+:xnack-', visible_count=1)))
        self.assertFalse(run.compatible(dict(architecture='gfx92a', visible_count=1)))
        for device in ({}, dict(architecture='gfx938', visible_count=2), dict(architecture='unknown', visible_count=1)):
            with self.assertRaises(ValueError):
                run.compatible(device)

    def test_bundled_asset_hashes_match(self):
        for name, expected in run.ASSETS.items():
            self.assertEqual(run.sha256(run.ROOT / 'native' / name), expected)
        self.assertTrue((run.ROOT / 'native/LICENSE').is_file())

    def test_native_logs_must_complete_all_iterations_without_errors(self):
        text = 'f8f8sgemm[NT]\n' + '\n'.join('iterator: ' + str(i) for i in range(4))
        run.parse(text, 4)
        for bad in (text.replace('iterator: 3', ''), text + '\nFailure', text.replace('f8f8sgemm[NT]', ''), text + '\niterator: 4'):
            with self.assertRaises(ValueError):
                run.parse(bad, 4)

    def test_incompatible_architecture_never_copies_or_executes_native_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(output_dir=folder, iterations=4, native_timeout_s=30,
                                   hcu_ids='0', nnodes=1, node_rank=0)
            calls, results = [], []

            def execute(argv, name, *rest):
                calls.append(name)
                if name == 'device-probe':
                    return json.dumps(dict(architecture='gfx92a', pci='0000:2d:00.0', visible_count=1))
                if name != 'probe-build':
                    raise AssertionError('Native loader must not start')
                return ''

            with patch.object(run, 'compiler', return_value='hipcc'), patch.object(run, 'execute', side_effect=execute), \
                 patch.object(run, 'emit', side_effect=results.append), patch.object(run.shutil, 'copyfile', side_effect=AssertionError('No loader copy')):
                self.assertEqual(run.run(args, threading.Event()), 0)
            self.assertEqual(calls, ['probe-build', 'device-probe'])
            self.assertEqual(results[-1]['measurement_status'], 'skipped')
            self.assertEqual(results[-1]['metrics'], [])

    def test_no_power_or_device_override_commands(self):
        source = (run.ROOT / 'run.py').read_text()
        self.assertIn('HSA_OVERRIDE_', source)
        self.assertNotIn('hipModuleLoad', (run.ROOT / 'probe.cpp').read_text())
        self.assertNotIn('--setpower', source)
        self.assertNotIn('--setsclk', source)

    def test_complete_native_exit_required_and_private_loader_copy(self):
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                args = SimpleNamespace(output_dir=folder, iterations=4, native_timeout_s=30,
                                       hcu_ids='0', nnodes=1, node_rank=0)
                calls, results = [], []

                def execute(argv, name, *rest):
                    calls.append((name, argv))
                    if name == 'device-probe':
                        return json.dumps(dict(architecture='gfx938', pci='0000:30:00.0', visible_count=1))
                    if name == 'native':
                        if failure:
                            raise RuntimeError('fixture timeout/nonzero native exit')
                        return 'f8f8sgemm[NT]\n' + '\n'.join('iterator: ' + str(i) for i in range(4))
                    return ''

                with patch.object(run, 'compiler', return_value='hipcc'), patch.object(run, 'execute', side_effect=execute), \
                     patch.object(run, 'emit', side_effect=results.append), patch.object(run, 'sample', return_value={'values': {}}):
                    self.assertEqual(run.run(args, threading.Event()), 2 if failure else 0)
                self.assertEqual([n for n, _ in calls], ['probe-build', 'device-probe', 'native'])
                self.assertEqual(Path(calls[-1][1][0]).parent, Path(folder))
                self.assertEqual(results[-1]['correctness'], 'not_checked')
                self.assertEqual(bool(results[-1]['metrics']), not failure)
