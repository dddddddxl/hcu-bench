# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""DTK script preservation and native test_conv result contracts."""
from collections import Counter
import hashlib
from pathlib import Path
import unittest

from hcu_bench.planner import make_plan
from testpacks.operators.hipdnn_conv import run as hipdnn


class HipdnnConvTests(unittest.TestCase):
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
            self.assertEqual(hipdnn.parse_result(text, direction), 1.25)
            with self.assertRaises(hipdnn.VerificationError):
                hipdnn.parse_result(text.replace('result: successful', 'result: failed'), direction)
            for bad in (text.replace('CPU validation result: successful', ''), text + text,
                        text.replace('1.25', '0'), text + 'hipDNN Frontend Error: no engine',
                        text.replace('CPU validation result: successful', 'CPU validation is not supported for int8 input type.')):
                with self.assertRaises(ValueError):
                    hipdnn.parse_result(bad, direction)

    def test_config_has_one_unmodified_original_sweep(self):
        config = Path(__file__).resolve().parents[1] / 'configs/cases/hipdnn-conv.yaml'
        plan = make_plan(str(config))
        self.assertEqual(len(plan.cases), 1)
        self.assertEqual(plan.cases[0].params['case_id'], '')
        self.assertFalse(any(k in plan.cases[0].params for k in ('warmup', 'iters', 'repeats')))
