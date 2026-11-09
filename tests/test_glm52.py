"""GLM launch/configuration contracts; stub CLI only, no HCU allocation."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from hcu_bench.adapters import CommandAdapter
from hcu_bench.e2e_config import performance_grid
from hcu_bench.planner import make_plan

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'model_scripts/glm52_int8_tp8.sh'


class GLMConfigurationTests(unittest.TestCase):
    def test_main_adds_glm_only_to_eight_card_performance(self):
        main = ROOT / 'configs/bench.template.yaml'
        accuracy = make_plan(str(main), 'e2e_acc')
        self.assertEqual([c.params['session']['model_id'] for c in accuracy.cases], ['qwen3_4b', 'qwen3_8b'])
        performance = make_plan(str(main), 'e2e_perf', ['vars.iface=lo'])
        glm = next(c for c in performance.cases if c.params.get('session', {}).get('model_id') == 'glm52_int8')
        self.assertEqual(glm.nodes, ['multi'])
        self.assertEqual(len(performance.nodes['multi']['gpus']), 8)
        self.assertEqual([t['suite'] for t in glm.params['session']['tasks']], ['e2e_perf'])
        spec = CommandAdapter().build(performance, glm, 'multi', ROOT / 'unused-run')
        self.assertEqual(spec['env']['HIP_VISIBLE_DEVICES'], '0,1,2,3,4,5,6,7')
        self.assertEqual(spec['env']['NCCL_SOCKET_IFNAME'], 'lo')
        self.assertEqual(spec['env']['GLOO_SOCKET_IFNAME'], 'lo')
        self.assertNotIn('NCCL_IB_DISABLE', spec['env'])
        params = glm.params['session']['tasks'][0]['params']
        self.assertEqual(params['extra_body'], {})
        self.assertEqual(len(performance_grid(params)), 31)

    def test_deepep_and_glm_reference_common_interface_without_sharing_ib_policy(self):
        plan = make_plan(str(ROOT / 'configs/bench.template.yaml'), 'deepep,e2e_perf', ['vars.iface=lo'])
        for case in plan.cases:
            if case.suite == 'deepep' or case.params.get('session', {}).get('model_id') == 'glm52_int8':
                spec = CommandAdapter().build(plan, case, case.nodes[0], ROOT / 'unused-run')
                self.assertEqual(spec['env']['NCCL_SOCKET_IFNAME'], 'lo')
                self.assertEqual(spec['env']['GLOO_SOCKET_IFNAME'], 'lo')
                if case.suite == 'deepep':
                    self.assertEqual(spec['env']['NCCL_IB_DISABLE'], '1')

    def test_standalone_matches_main_glm_performance(self):
        standalone = make_plan(str(ROOT / 'configs/glm52-perf.template.yaml'))
        self.assertEqual(len(standalone.cases), 1)
        main = make_plan(str(ROOT / 'configs/bench.template.yaml'), 'e2e_perf')
        glm = next(c for c in main.cases if c.params.get('session', {}).get('model_id') == 'glm52_int8')
        self.assertEqual(standalone.cases[0].params['session']['tasks'], glm.params['session']['tasks'])


class GLMLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        cli = self.root / 'sglang'
        cli.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                       'print(json.dumps(dict(argv=sys.argv[1:], env={k:os.environ.get(k) for k in '
                       '["NCCL_SOCKET_IFNAME","GLOO_SOCKET_IFNAME","SGLANG_USE_FP8_W8A8_MOE","SGLANG_ENABLE_SPEC_V2"]})))\n')
        cli.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.root) + os.pathsep + os.environ['PATH'],
                        BENCH_MODEL_PATH=str(self.root), HIP_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
                        NCCL_SOCKET_IFNAME='lo', GLOO_SOCKET_IFNAME='lo', BENCH_HOST='127.0.0.1', BENCH_PORT='31914')

    def launch(self):
        return subprocess.run(['bash', str(SCRIPT)], env=self.env, text=True, capture_output=True, timeout=10)

    def test_recipe_preserves_tp8_quantization_speculation_and_inherited_network(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        argv = data['argv']
        self.assertEqual(argv[0], 'serve')
        for name, expected in {'--tp-size': '8', '--quantization': 'slimquant_marlin', '--port': '31914',
                               '--reasoning-parser': 'glm45', '--tool-call-parser': 'glm47',
                               '--speculative-algorithm': 'EAGLE', '--mem-fraction-static': '0.9',
                               '--nsa-prefill-backend': 'flashmla_auto', '--nsa-decode-backend': 'flashmla_kv',
                               '--max-running-requests': '32'}.items():
            self.assertEqual(argv[argv.index(name) + 1], expected)
        self.assertEqual(data['env']['NCCL_SOCKET_IFNAME'], 'lo')
        self.assertEqual(data['env']['GLOO_SOCKET_IFNAME'], 'lo')
        self.assertEqual(data['env']['SGLANG_USE_FP8_W8A8_MOE'], '0')

    def test_invalid_hcu_allocation_fails_before_cli(self):
        for ids in ('0', '0,1,2,3,4,5,6,6', '0,1,2,3,4,5,6,'):
            with self.subTest(ids=ids):
                self.env['HIP_VISIBLE_DEVICES'] = ids
                result = self.launch()
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, '')

    def test_invalid_interface_and_missing_local_model_fail_before_cli(self):
        self.env['NCCL_SOCKET_IFNAME'] = 'YOUR_ACTIVE_INTERFACE'
        self.assertEqual(self.launch().returncode, 2)
        self.env['NCCL_SOCKET_IFNAME'] = 'lo'
        self.env['BENCH_MODEL_PATH'] = str(self.root / 'missing')
        self.assertEqual(self.launch().returncode, 2)


if __name__ == '__main__':
    unittest.main()
