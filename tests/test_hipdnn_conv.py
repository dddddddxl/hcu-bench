# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""DTK script preservation and native test_conv result contracts."""
from collections import Counter
import hashlib
from pathlib import Path
from types import SimpleNamespace
import contextlib
import io
import json
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from hcu_bench.planner import make_plan
from testpacks.operators.hipdnn_conv import run as hipdnn


class HipdnnConvTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('tar'), 'source extraction requires tar')
    def test_missing_installed_binary_builds_bundled_original_source(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            args = SimpleNamespace(binary='auto', source_root='', build_jobs=2)
            calls = []
            def launch(argv, name):
                calls.append(argv)
                if name == 'source-extract':
                    subprocess.run(argv, check=True, capture_output=True)
            with patch.object(hipdnn.shutil, 'which', return_value=None):
                binary = hipdnn.resolve_binary(args, out, launch)
            source = out / 'native-source'
            build = out / 'native-build'
            self.assertEqual(calls, [
                ['tar', '-xzf', str(hipdnn.ROOT / 'vendor/hipdnn-api-test/dtk-api-test-source.tar.gz'),
                 '-C', str(source), '--strip-components=1', '--no-same-owner'],
                ['cmake', '-S', str(source), '-B', str(build), '-G', 'Ninja'],
                ['cmake', '--build', str(build), '--target', 'test_conv', '-j', '2'],
            ])
            self.assertEqual(binary, str(build / 'bin/test_conv'))
            inputs = json.loads((out / 'build-inputs.json').read_text())
            self.assertEqual(inputs['files'][str(source / 'cpp/test_conv/test_conv.cpp')],
                             '49334673eb1f2709919d5277232b19cbdd1305feed63ca82c5a52932c8a44935')

    def test_native_commands_do_not_require_a_nonexistent_help_option(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / 'test_conv'
            binary.write_bytes(b'contract fixture; not a GPU executable')
            args = SimpleNamespace(output_dir=str(root / 'out'), native_timeout_s=60,
                                   binary=str(binary), source_root='', build_jobs=2,
                                   case_id='dtk-conv-001', hcu_ids='0', nnodes=1, node_rank=0)
            calls = []
            def execute(argv, name, out, env, stopped, executions, timeout_s):
                calls.append(argv)
                self.assertNotIn('--help', argv)
                if name == 'device-probe':
                    return 'BENCH_DEVICE {"visible_count":1,"pci_bus":"0000:01:00.0"}'
                original = hipdnn.load_cases(hipdnn.SOURCE / 'test_conv.sh')[0]
                self.assertEqual(argv[1:], original['argv'])
                self.assertEqual(env['MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32'], '0')
                return 'fprop avg times: 1.25 ms\nCPU validation result: successful\n'
            with patch.object(hipdnn, 'execute', side_effect=execute), \
                    patch.object(hipdnn, 'resolve_binary', return_value=str(binary)), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(hipdnn.run(args, threading.Event()), 0)
            self.assertEqual(len(calls), 2)
            self.assertTrue(json.loads((root / 'out/assessment.json').read_text())['complete'])

    def test_original_script_and_ordered_environment(self):
        script = hipdnn.SOURCE / 'test_conv.sh'
        self.assertEqual(hashlib.sha256(script.read_bytes().split(b"\n", 2)[2]).hexdigest(),
                         '9711f3a21218eb32e16d2fbe758ae13b5affaefb476966e3dda4b02884b14c9c')
        cases = hipdnn.load_cases(script)
        self.assertEqual(len(cases), 68)
        self.assertEqual({c['direction'] for c in cases}, {'1', '2', '4'})
        self.assertEqual({c['dtype'] for c in cases}, {'fp16', 'fp32', 'bfp16', 'tf32', 'int8'})
        for case in cases:
            self.assertIn('--cpu-validation', case['argv'])
            self.assertEqual(case['env']['MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32'],
                             '1' if case['dtype'] == 'tf32' else '0')
            self.assertNotIn('--iters', case['argv'])
        self.assertEqual(len({c['case_id'] for c in cases}), 68)

    def test_only_cpu_verified_native_times_are_accepted(self):
        for direction, label in hipdnn.LABELS.items():
            text = 'Graph validation successful\n%s avg times: 1.25 ms\nCPU validation result: successful\n' % label
            self.assertEqual(hipdnn.parse_result(text, direction, 'fp16')['latency_ms'], 1.25)
            with self.assertRaises(hipdnn.VerificationError):
                hipdnn.parse_result(text.replace('result: successful', 'result: failed'), direction, 'fp16')
            for bad in (text.replace('CPU validation result: successful', ''), text + text,
                        text.replace('1.25', '0'), text + 'hipDNN Frontend Error: no engine',
                        text.replace('CPU validation result: successful', 'CPU validation is not supported for int8 input type.')):
                with self.assertRaises(ValueError):
                    hipdnn.parse_result(bad, direction, 'fp16')
            unsupported = text.replace('CPU validation result: successful',
                                       'CPU validation is not supported for int8 input type.')
            result = hipdnn.parse_result(unsupported, direction, 'int8')
            self.assertEqual(result['latency_ms'], 1.25)
            self.assertEqual(result['correctness'], 'not_checked')
            self.assertEqual(result['validation_status'], 'native_not_supported')

    def test_config_has_one_unmodified_original_sweep(self):
        config = Path(__file__).resolve().parents[1] / 'configs/cases/hipdnn-conv.yaml'
        plan = make_plan(str(config))
        self.assertEqual(len(plan.cases), 1)
        self.assertEqual(plan.cases[0].params['case_id'], '')
        self.assertFalse(any(k in plan.cases[0].params for k in ('warmup', 'iters', 'repeats')))
