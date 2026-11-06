"""CPU training contract fixtures, not hardware performance evidence."""
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hcu_bench.cli import preview
from hcu_bench.models import BenchError
from hcu_bench.planner import make_plan
from testpacks.training import run
from testpacks.training.throughput_entry import TrainingWindow

ROOT = Path(__file__).resolve().parents[1]


def args(root):
    script = root / 'train.py'
    script.write_text('# CPU contract fixture, not a training implementation\n')
    return SimpleNamespace(train_script=str(script), script_sha256='', dataset=str(root / 'images'),
                           mode='smoke', epochs=1, batch_size=8, workers=0, precision='fp32',
                           seed=1, native_timeout_s=30, warmup_steps=10, measured_steps=50,
                           hcu_ids='0', nnodes=1, node_rank=0,
                           output_dir=str(root / 'output'))


def log(epochs=1):
    return '\n'.join('Epoch: [{}][0/1]\tLoss 6.9e+00 (6.9e+00)\nTest: [0/1] Loss 6.0e+00 (6.0e+00)\n * Acc@1 0.000 Acc@5 6.250'.format(i)
                     for i in range(epochs))


class TrainingContracts(unittest.TestCase):
    def test_plan_training_is_independent_of_inference(self):
        config = str(ROOT / 'configs/resnet50-training.template.yaml')
        with patch('subprocess.Popen', side_effect=AssertionError('static preview started a process')):
            plan = make_plan(config, 'training')
            self.assertEqual(len(plan.cases), 1)
            self.assertEqual(plan.cases[0].params['epochs'], 1)
            self.assertEqual(plan.cases[0].params['mode'], 'throughput')
            self.assertEqual(plan.cases[0].pack['max_nodes'], 1)
            text = json.dumps(preview(plan))
            self.assertIn('training/run.py', text)
            self.assertNotIn('migraphx-driver', text)
        for selection in ('migraphx', 'e2e'):
            with self.assertRaisesRegex(BenchError, 'No enabled tests'):
                make_plan(config, selection)
        for override in ('tests.resnet50_pretraining.duration_s=10',
                         'tests.resnet50_pretraining.params.precision=fp8',
                         'tests.resnet50_pretraining.params.epochs=0',
                         'tests.resnet50_pretraining.params.pretrained=true'):
            with self.subTest(override=override), self.assertRaises(BenchError):
                make_plan(config, 'training', [override])

    def test_source_pin_limits_and_single_card(self):
        with tempfile.TemporaryDirectory() as folder:
            a = args(Path(folder))
            self.assertEqual(run.validate(a), run.sha256(a.train_script))
            for name, value in [('epochs', 2), ('batch_size', 65), ('seed', True), ('script_sha256', '0' * 64)]:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    run.validate(SimpleNamespace(**{**vars(a), name: value}))
            for value in ('0,1', '-1'):
                with self.assertRaises(ValueError):
                    run.single_device(SimpleNamespace(**{**vars(a), 'hcu_ids': value}), {})

    def test_command_no_download_no_implicit_amp_conversion(self):
        with tempfile.TemporaryDirectory() as folder:
            a = args(Path(folder)); out = Path(folder) / 'output'
            argv = run.command(a, {'effective_root': '/real/images'}, out)
            self.assertNotIn('--pretrained', argv)
            self.assertNotIn('--amp', argv)
            self.assertNotIn('numactl', argv)
            a.precision = 'amp'
            argv = run.command(a, {'effective_root': '/real/images'}, out)
            self.assertEqual(argv[argv.index('--opt-level') + 1], 'O1')
            replay = run.command(a, {'effective_root': '/real/images'}, out, out / 'own-checkpoint')
            self.assertIn('--evaluate', replay)
            self.assertIn('--resume', replay)

    def test_logs_reject_partial_nonfinite_and_bad_accuracy(self):
        self.assertEqual(run.parse(log(), 1)['validation'][0]['top5_percent'], 6.25)
        self.assertEqual(len(run.parse(log(2), 2)['validation']), 2)
        for text in (log().replace('6.9e+00', 'nan'), log().replace('6.250', 'inf'),
                     log().replace('6.0e+00', 'nan'),
                     log().replace('0.000', '101.000'), log().split(' * Acc')[0],
                     log().replace('Epoch: [0]', 'Epoch: [1]')):
            with self.subTest(text=text), self.assertRaises(ValueError):
                run.parse(text, 1)
        with self.assertRaises(ValueError):
            run.parse(log(), 2)

    def test_real_subset_policy_does_not_modify_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'images'; out = Path(folder) / 'output'; out.mkdir()
            for split, count in [('train', 8), ('val', 2)]:
                for i in range(1000):
                    directory = root / split / ('n{:08d}'.format(i)); directory.mkdir(parents=True)
                    if i < 8:
                        for j in range(count):
                            (directory / ('{:03d}.JPEG'.format(j))).write_bytes(b'CPU copying fixture, not a JPEG')
            record = run.dataset(root, 'smoke', out)
            self.assertEqual(len(record['files']), 80)
            self.assertEqual(record['train_images'], 64)
            self.assertEqual(record['validation_images'], 16)
            self.assertEqual(record['selected_classes'], record['source_classes'][:8])
            self.assertEqual(len(list(root.rglob('*.JPEG'))), 80)
            self.assertEqual(run.dataset(root, 'full', out)['effective_root'], str(root.resolve()))
            (root / 'val' / 'n00000999').rmdir()
            with self.assertRaisesRegex(ValueError, '1000'):
                run.dataset(root, 'full', out)

    def test_throughput_window_synchronizes_boundaries_not_each_step(self):
        times = iter([10.0, 12.0])
        sync = []
        window = TrainingWindow(32, 10, 50, lambda: sync.append(True), lambda: next(times))
        optimizer = object()
        for _ in range(59):
            window.step(optimizer, (), {})
        self.assertEqual(len(sync), 1)
        with self.assertRaises(ValueError):
            window.result()
        window.step(optimizer, (), {})
        result = window.result()
        self.assertEqual(len(sync), 2)
        self.assertEqual(result['measured_images'], 1600)
        self.assertEqual(result['images_per_second'], 800)
        window.step(optimizer, (), {})
        with self.assertRaises(ValueError):
            window.result()
        with self.assertRaises(ValueError):
            window.step(object(), (), {})

    def test_throughput_rejects_unbounded_or_unsupported_modes(self):
        with tempfile.TemporaryDirectory() as folder:
            a = args(Path(folder)); a.mode = 'throughput'; a.batch_size = 32
            run.validate(a)
            for name, value in [('epochs', 90), ('precision', 'amp'), ('measured_steps', 0),
                                ('warmup_steps', 0), ('batch_size', 2)]:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    run.validate(SimpleNamespace(**{**vars(a), name: value}))

    def test_throughput_real_subset_keeps_every_original_class_and_exact_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'images'; out = Path(folder) / 'output'; out.mkdir()
            for split, count in [('train', 2), ('val', 1)]:
                for i in range(1000):
                    directory = root / split / ('n{:08d}'.format(i)); directory.mkdir(parents=True)
                    for j in range(count):
                        (directory / ('{:03d}.JPEG'.format(j))).write_bytes(b'CPU copying fixture, not image data')
            record = run.dataset(root, 'throughput', out)
            self.assertEqual(record['selected_classes'], record['source_classes'])
            self.assertEqual((record['train_images'], record['validation_images']), (1920, 1000))
            self.assertEqual(len(record['files']), 2920)
            self.assertEqual(len(list((out / 'throughput-data/train').glob('*/*.jpeg'))), 60 * 32)

    def test_success_requires_checkpoint_and_matching_reload(self):
        for failure in ('none', 'missing_checkpoint', 'reload_mismatch', 'native_failed', 'inspect_failed'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                a = args(Path(folder)); captured = []

                def execute(argv, name, out, *rest):
                    if name == 'train':
                        if failure == 'native_failed':
                            raise RuntimeError('fixture native failure')
                        if failure != 'missing_checkpoint':
                            path = out / 'checkpoints/0/checkpoint_0.pth.tar'; path.parent.mkdir()
                            path.write_bytes(b'CPU fixture, not actual checkpoint')
                        return log()
                    if name == 'checkpoint-inspect':
                        run.write_json(out / 'checkpoint.json', {'complete': failure != 'inspect_failed', 'epoch': 1})
                        return ''
                    return 'Test: [0/1] Loss 6.0e+00 (6.0e+00)\n * Acc@1 0.000 Acc@5 ' + ('12.500' if failure == 'reload_mismatch' else '6.250')

                with patch.object(run, 'dataset', return_value={'effective_root': '/fixture', 'coverage': 'CPU fixture'}), \
                     patch.object(run, 'execute', side_effect=execute), \
                     patch.object(run, 'emit', side_effect=captured.append), \
                     patch.object(run, 'versions', return_value={}):
                    rc = run.run(a, threading.Event())
                self.assertEqual(rc, 0 if failure == 'none' else 2)
                self.assertEqual(captured[-1]['correctness'], 'not_checked')
                if failure == 'none':
                    self.assertEqual(captured[-1]['training_convergence'], 'not_assessed')
                    self.assertTrue(captured[-1]['checkpoint_reload_matches'])
                else:
                    self.assertEqual(captured[-1]['measurement_status'], 'invalid')
                    self.assertEqual(captured[-1]['metrics'], [])

    def test_throughput_does_not_publish_partial_or_missing_window(self):
        for failure in ('missing', 'partial', 'nonfinite', 'none'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                a = args(Path(folder)); a.mode = 'throughput'; a.batch_size = 32
                captured = []

                def execute(argv, name, out, *rest):
                    if name == 'train':
                        checkpoint = out / 'checkpoints/0/checkpoint_0.pth.tar'
                        checkpoint.parent.mkdir(); checkpoint.write_bytes(b'CPU fixture')
                        if failure != 'missing':
                            run.write_json(out / 'throughput-window.json', dict(
                                complete=True, total_updates=59 if failure == 'partial' else 60,
                                warmup_steps=10, measured_steps=50, measured_images=1600,
                                elapsed_s=float('inf') if failure == 'nonfinite' else 2,
                                images_per_second=800))
                        return log()
                    if name == 'checkpoint-inspect':
                        run.write_json(out / 'checkpoint.json', {'complete': True, 'epoch': 1})
                        return ''
                    return 'Test: [0/1] Loss 6.0e+00 (6.0e+00)\n * Acc@1 0.000 Acc@5 6.250'

                with patch.object(run, 'dataset', return_value={'effective_root': '/fixture', 'coverage': 'CPU fixture'}), \
                     patch.object(run, 'execute', side_effect=execute), \
                     patch.object(run, 'emit', side_effect=captured.append), \
                     patch.object(run, 'versions', return_value={}):
                    rc = run.run(a, threading.Event())
                self.assertEqual(rc, 0 if failure == 'none' else 2)
                if failure == 'none':
                    metrics = {m['name']: m['value'] for m in captured[-1]['metrics']}
                    self.assertEqual(metrics['training_throughput'], 800)
                    self.assertEqual(metrics['training_step_time'], 40)
                else:
                    self.assertEqual(captured[-1]['metrics'], [])


if __name__ == '__main__':
    unittest.main()
