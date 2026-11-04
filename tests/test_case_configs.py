"""Public main/case YAML selection contracts; no hardware workloads."""
from pathlib import Path
import unittest
from unittest.mock import patch

from hcu_bench.cli import preview
from hcu_bench.config import load_document
from hcu_bench.models import BenchError
from hcu_bench.planner import make_plan

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / 'configs/bench.template.yaml'


class CaseConfigurationTests(unittest.TestCase):
    def test_readme_preserves_configuration_and_suite_report_sections(self):
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        parameters = readme.split('### 测试块字段', 1)[1].split('## Suite 参数说明', 1)[0]
        self.assertIn('`enabled`', parameters)
        self.assertIn('`params`', parameters)
        self.assertNotIn('summary.txt', parameters)
        reports = readme.split('## 结果与日志', 1)[1].split('## 常用命令', 1)[0]
        for content in ('summary.md', 'report-details/suite-*.csv', 'docs/reports.md', 'bash bench.sh report'):
            self.assertIn(content, reports)
        self.assertIn('默认禁用（disabled）', readme)

    def test_main_has_only_common_variables_and_no_optional_cases(self):
        document = load_document(MAIN)
        self.assertEqual(set(document['vars']), {'bench_root', 'workspace', 'hcus', 'single_hcus'})
        with patch('subprocess.Popen', side_effect=AssertionError('static only')):
            plan = make_plan(str(MAIN))
        self.assertEqual(len(plan.cases), 18)
        self.assertFalse({'memory', 'migraphx', 'mooncake'} & {c.suite for c in plan.cases})
        self.assertFalse(any(c.pack.get('id') == 'e2e_perf.wan-video' for c in plan.cases))
        self.assertFalse(any(c.params.get('recipe') in ('gemm', 'fa_fp8_forward') for c in plan.cases))

    def test_rccl_bounds_are_inherited_and_rank_count_matches_allocation(self):
        document = load_document(MAIN)
        for test in document['tests'].values():
            if test['suite'] == 'rccl':
                self.assertNotIn('min_bytes', test['params'])
                self.assertNotIn('max_bytes', test['params'])
                self.assertEqual(test['params']['ranks'], len(document['vars']['hcus']))
        plan = make_plan(str(MAIN), 'rccl')
        self.assertEqual(len(plan.cases), 10)
        for case in plan.cases:
            self.assertEqual(case.params['min_bytes'], 16 * 1024 ** 3)
            self.assertEqual(case.params['max_bytes'], 16 * 1024 ** 3)

    def test_main_preserves_full_qwen_accuracy_and_performance_contracts(self):
        plan = make_plan(str(MAIN), 'e2e')
        self.assertEqual(len(plan.cases), 2)
        for case in plan.cases:
            tasks = case.params['session']['tasks']
            self.assertEqual([t['suite'] for t in tasks], ['e2e_acc', 'e2e_perf'])
            accuracy, performance = (t['params'] for t in tasks)
            self.assertEqual((accuracy['num_examples'], accuracy['num_shots']), (0, 5))
            self.assertEqual(accuracy['few_shot_path'], 'auto')
            self.assertEqual(accuracy['min_score'], '')
            self.assertEqual(sum(len(w['concurrency']) for w in performance['workloads']), 31)
            self.assertEqual((performance['requests'], performance['warmup_requests'], performance['repeats']), (256, 32, 1))

    def test_independent_cases_preview_without_starting_processes(self):
        expected = {
            'memory': ('memory', 1), 'migraphx-resnet50': ('migraphx', 4),
            'migraphx-bert': ('migraphx', 2), 'gemm': ('operators', 3),
            'fa-fp8': ('operators', 1), 'operator-stress': ('operators', 3),
            'mooncake': ('mooncake', 1), 'wan': ('e2e_perf', 1),
        }
        self.assertEqual({p.stem for p in (ROOT / 'configs/cases').glob('*.yaml')}, set(expected))
        for name, (suite, count) in expected.items():
            with self.subTest(case=name), patch('subprocess.Popen', side_effect=AssertionError('static only')):
                path = ROOT / 'configs/cases' / (name + '.yaml')
                self.assertTrue(all(t['enabled'] for t in load_document(path)['tests'].values()))
                plan = make_plan(str(path))
                self.assertEqual(len(plan.cases), count)
                self.assertEqual({c.suite for c in plan.cases}, {suite})
                rendered = preview(plan)
                self.assertEqual(rendered['case_count'], count)
                for case in rendered['cases']:
                    for command in case['commands'].values():
                        self.assertFalse(any('${' in arg for arg in command['argv']))

    def test_optional_case_selection_does_not_implicitly_import_files(self):
        with self.assertRaisesRegex(BenchError, 'No enabled tests'):
            make_plan(str(MAIN), 'memory')
        self.assertEqual(len(make_plan(str(ROOT / 'configs/cases/memory.yaml')).cases), 1)

    def test_short_stress_and_strict_gemm_defaults_are_preserved(self):
        stress = make_plan(str(ROOT / 'configs/cases/operator-stress.yaml'))
        self.assertTrue(all(c.params['stress_iterations'] == 100 for c in stress.cases))
        gemm = make_plan(str(ROOT / 'configs/cases/gemm.yaml'))
        self.assertTrue(all(c.params['max_norm_error'] == '' for c in gemm.cases))
        self.assertTrue(all(c.params['mode'] == 'perf' for c in gemm.cases))

    def test_wan_is_tp4_and_mooncake_is_a_true_two_node_case(self):
        wan = make_plan(str(ROOT / 'configs/cases/wan.yaml')).cases[0]
        self.assertEqual(wan.params['profile'], 'tp4')
        self.assertEqual(wan.params['warmup_requests'], 1)
        self.assertEqual(wan.params['requests'], 1)
        mooncake = make_plan(str(ROOT / 'configs/cases/mooncake.yaml')).cases[0]
        self.assertEqual(mooncake.nodes, ['target', 'initiator'])


if __name__ == '__main__':
    unittest.main()
