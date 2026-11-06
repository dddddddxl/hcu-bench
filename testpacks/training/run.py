"""Run the supplied, unchanged ImageNet training entry point on one assigned HCU."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_common import CleanupError, cancellation, cleanup_failure, emit, sha256, versions, write_json
from fixed_common import claim_output, environment_record, execute, single_device

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def validate(args):
    if args.mode not in ('full', 'smoke', 'throughput') or args.precision not in ('fp32', 'amp'):
        raise ValueError('Unsupported training mode or precision')
    for name, minimum, maximum in [('epochs', 1, 1000), ('batch_size', 2, 65536),
                                   ('workers', 0, 128), ('seed', 0, 2147483647),
                                   ('native_timeout_s', 1, 2592000),
                                   ('warmup_steps', 1, 100), ('measured_steps', 1, 500)]:
        value = getattr(args, name)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError('Invalid ' + name)
    if args.mode == 'smoke' and (args.epochs != 1 or args.batch_size > 64):
        raise ValueError('Smoke requires one epoch and batch_size <= 64; use full explicitly otherwise')
    if args.mode == 'throughput':
        images = args.batch_size * (args.warmup_steps + args.measured_steps)
        if args.epochs != 1 or args.batch_size > 64 or not 1000 <= images <= 32768:
            raise ValueError('Throughput requires one epoch, batch_size <= 64 and 1000..32768 training images')
        if args.precision != 'fp32':
            raise ValueError('Throughput currently supports measured FP32 SGD updates only')
    script = Path(args.train_script).resolve()
    if not script.is_file() or script.suffix != '.py':
        raise ValueError('Prepared external train.py is required; no training source is downloaded')
    actual = sha256(script)
    if args.script_sha256 and (not re.fullmatch('[0-9a-f]{64}', args.script_sha256) or actual != args.script_sha256):
        raise ValueError('Training source SHA256 differs from the explicitly pinned input')
    args.train_script = str(script)
    return actual


def dataset(root, mode, out, batch_size=32, warmup_steps=10, measured_steps=50):
    root = Path(root).resolve()
    classes = {}
    for split in ('train', 'val'):
        directory = root / split
        if not directory.is_dir():
            raise ValueError('ImageFolder train/val directory missing: ' + str(directory))
        classes[split] = sorted(p.name for p in directory.iterdir() if p.is_dir())
    if len(classes['train']) != 1000 or classes['train'] != classes['val']:
        raise ValueError('Expected matching 1000 ImageNet class directories in train and val')
    record = dict(source_root=str(root), source_classes=classes['train'], mode=mode,
                  input_policy='real ImageFolder images; no generated data or pretrained weights')
    if mode == 'full':
        record.update(effective_root=str(root), coverage='configured epochs over full ImageFolder dataset')
    else:
        selected = classes['train'] if mode == 'throughput' else classes['train'][:8]
        entries = []
        target = out / ('throughput-data' if mode == 'throughput' else 'smoke-data')
        train_images = batch_size * (warmup_steps + measured_steps) if mode == 'throughput' else 64
        validation_images = 1000 if mode == 'throughput' else 16
        for split, total in [('train', train_images), ('val', validation_images)]:
            for class_index, name in enumerate(selected):
                count = total // len(selected) + (class_index < total % len(selected))
                directory = root / split / name
                if mode == 'throughput':
                    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in EXTENSIONS)
                else:
                    files = sorted(p for p in directory.rglob('*')
                                   if p.is_file() and p.suffix.lower() in EXTENSIONS)
                if len(files) < count:
                    raise ValueError('Insufficient real images for smoke: ' + split + '/' + name)
                dest = target / split / name
                dest.mkdir(parents=True)
                for index, source in enumerate(files[:count]):
                    output = dest / (str(index) + source.suffix.lower())
                    shutil.copyfile(source, output)
                    entries.append(dict(source=str(source.relative_to(root)),
                                        copy=str(output.relative_to(out)), bytes=output.stat().st_size,
                                        sha256=sha256(output)))
        record.update(effective_root=str(target), selected_classes=selected, files=entries,
                      train_images=train_images, validation_images=validation_images,
                      coverage=('throughput only: bounded real-image subset over all 1000 original classes'
                                if mode == 'throughput' else
                                'smoke only: first 8 classes retain their original 0..7 class indices'))
    write_json(out / 'dataset.json', record)
    return record


def command(args, data, out, checkpoint=None):
    argv = [sys.executable, '-u', args.train_script, '--arch', 'resnet50', '--gpu', '0',
            '--world-size', '1', '--rank', '0', '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size), '--workers', str(args.workers),
            '--seed', str(args.seed), '--print-freq', '1', '--lr', '0.1',
            '--momentum', '0.9', '--weight-decay', '0.0001',
            '--save-path', str(out / 'checkpoints')]
    if args.precision == 'amp':
        argv += ['--amp', '--opt-level', 'O1', '--loss-scale', 'dynamic']
    if checkpoint:
        argv += ['--evaluate', '--resume', str(checkpoint)]
    return argv + [data['effective_root']]


def parse(text, epochs, evaluation=False):
    results = []
    seen_epochs = set()
    losses = []
    validation_losses = []
    for line in text.splitlines():
        if re.match(r'\s*\*\s+Acc@1\b', line):
            match = re.fullmatch(r'\s*\*\s+Acc@1\s+(\S+)\s+Acc@5\s+(\S+)\s*', line)
            if not match:
                raise ValueError('Malformed validation summary')
            top1, top5 = map(float, match.groups())
            if not all(math.isfinite(x) for x in (top1, top5)) or not 0 <= top1 <= top5 <= 100:
                raise ValueError('Invalid validation accuracy')
            if not validation_losses:
                raise ValueError('Missing validation batch loss')
            results.append(dict(top1_percent=top1, top5_percent=top5,
                                validation_loss=validation_losses[-1]))
            validation_losses = []
        if line.startswith('Test:'):
            loss = re.search(r'\bLoss\s+(\S+)\s+\(([^)]+)\)', line)
            if not loss:
                raise ValueError('Malformed validation progress')
            values = [float(x) for x in loss.groups()]
            if not all(math.isfinite(x) and x >= 0 for x in values):
                raise ValueError('Nonfinite or negative validation loss')
            validation_losses.append(values[-1])
        if line.startswith('Epoch:'):
            epoch = re.match(r'Epoch:\s*\[(\d+)\]', line)
            loss = re.search(r'\bLoss\s+(\S+)\s+\(([^)]+)\)', line)
            if not epoch or not loss:
                raise ValueError('Malformed training progress')
            values = [float(x) for x in loss.groups()]
            if not all(math.isfinite(x) and x >= 0 for x in values):
                raise ValueError('Nonfinite or negative training loss')
            losses.append(values[-1])
            seen_epochs.add(int(epoch[1]))
    expected = 1 if evaluation else epochs
    if len(results) != expected or (not evaluation and (seen_epochs != set(range(epochs)) or not losses)):
        raise ValueError('Missing training epochs or validation summaries; partial logs are not complete')
    return dict(validation=results, last_reported_train_loss=losses[-1] if losses else None)


def run(args, stopped):
    out = claim_output(args.output_dir)
    executions = []
    try:
        source_hash = validate(args)
        env = single_device(args, dict(os.environ))
        data = dataset(args.dataset, args.mode, out, args.batch_size, args.warmup_steps, args.measured_steps)
        (out / 'checkpoints').mkdir()
        argv = command(args, data, out)
        if args.mode == 'throughput':
            argv = [sys.executable, '-u', str(Path(__file__).with_name('throughput_entry.py')),
                    '--train-script', args.train_script, '--batch-size', str(args.batch_size),
                    '--warmup-steps', str(args.warmup_steps), '--measured-steps', str(args.measured_steps),
                    '--output', str(out / 'throughput-window.json'), '--'] + argv[3:]
        write_json(out / 'plan.json', dict(params=vars(args), argv=argv, source_sha256=source_hash,
                   wrapper_sha256=sha256(__file__), versions=versions(['torch', 'torchvision', 'apex']),
                   environment=environment_record(env), pretrained=False, distributed=False,
                   recipe=dict(optimizer='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001,
                               lr_schedule='native 0.1 decay every 30 epochs', crop_size=224),
                   correctness='not_checked'))
        start = time.monotonic()
        text = execute(argv, 'train', out, env, stopped, executions, args.native_timeout_s)
        training_elapsed = time.monotonic() - start
        trained = parse(text, args.epochs)
        throughput = None
        if args.mode == 'throughput':
            throughput = json.loads((out / 'throughput-window.json').read_text())
            expected_steps = args.warmup_steps + args.measured_steps
            if (throughput.get('complete') is not True or throughput.get('total_updates') != expected_steps
                    or throughput.get('measured_steps') != args.measured_steps
                    or throughput.get('warmup_steps') != args.warmup_steps
                    or throughput.get('measured_images') != args.batch_size * args.measured_steps
                    or not math.isfinite(throughput['elapsed_s']) or throughput['elapsed_s'] <= 0
                    or not math.isfinite(throughput['images_per_second']) or throughput['images_per_second'] <= 0):
                raise ValueError('Missing, partial or invalid synchronized training throughput window')
        checkpoints = [out / 'checkpoints' / '0' / ('checkpoint_' + str(i) + '.pth.tar')
                       for i in range(args.epochs)]
        if not all(p.is_file() and p.stat().st_size > 0 for p in checkpoints):
            raise ValueError('Expected completed-epoch checkpoints are missing')
        latest = checkpoints[-1]
        inspect = [sys.executable, '-u', str(Path(__file__).with_name('inspect_checkpoint.py')),
                   '--checkpoint', str(latest), '--epochs', str(args.epochs),
                   '--seed', str(args.seed), '--output', str(out / 'checkpoint.json')]
        execute(inspect, 'checkpoint-inspect', out, env, stopped, executions, 120)
        checkpoint = json.loads((out / 'checkpoint.json').read_text())
        if checkpoint.get('complete') is not True or checkpoint.get('epoch') != args.epochs:
            raise ValueError('Checkpoint structural/finite-state checks failed')
        reloaded_text = execute(command(args, data, out, latest), 'reload-validation', out,
                                env, stopped, executions, args.native_timeout_s)
        reloaded = parse(reloaded_text, 1, evaluation=True)['validation'][0]
        if reloaded != trained['validation'][-1]:
            raise ValueError('Reloaded checkpoint validation differs from saved model validation')
        if sha256(args.train_script) != source_hash:
            raise ValueError('Training source changed during execution')
        if stopped.is_set():
            raise RuntimeError('Cancelled before publishing training results')
        result = dict(measurement_status='measured', correctness='not_checked',
                      recipe='resnet50_imagenet_from_scratch', mode=args.mode, precision=args.precision,
                      epochs=args.epochs, pretrained=False, training_convergence='not_assessed',
                      validation_scope=data['coverage'], checkpoint_reload_matches=True,
                      validation=trained['validation'], metrics=[
                          dict(name='validation_top1', value=reloaded['top1_percent'], unit='%', statistic='last_epoch'),
                          dict(name='validation_top5', value=reloaded['top5_percent'], unit='%', statistic='last_epoch'),
                          dict(name='validation_loss', value=reloaded['validation_loss'], unit='loss', statistic='last_reported_average'),
                          dict(name='train_loss', value=trained['last_reported_train_loss'], unit='loss', statistic='last_reported_average'),
                          dict(name='training_job_elapsed', value=training_elapsed, unit='s', statistic='wall_time_including_validation_and_checkpoint')])
        if throughput is not None:
            result['throughput'] = throughput
            result['metrics'] += [
                dict(name='training_throughput', value=throughput['images_per_second'], unit='images/s', statistic='synchronized_window'),
                dict(name='training_step_time', value=throughput['elapsed_s'] * 1000 / args.measured_steps,
                     unit='ms', statistic='mean_over_measured_steps')]
        write_json(out / 'measurements.json', result)
        write_json(out / 'assessment.json', dict(complete=True, correctness='not_checked',
                   training_convergence='not_assessed', checkpoint_reload_matches=True, mode=args.mode))
        emit(result)
        return 0
    except CleanupError as exc:
        return cleanup_failure(out, exc)
    except Exception as exc:
        write_json(out / 'assessment.json', dict(complete=False, correctness='not_checked', error=str(exc)))
        emit(dict(measurement_status='invalid', correctness='not_checked', metrics=[], reason=str(exc)))
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-script', required=True)
    parser.add_argument('--script-sha256', default='')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--mode', choices=('full', 'smoke', 'throughput'), default='throughput')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--warmup-steps', type=int, default=10)
    parser.add_argument('--measured-steps', type=int, default=50)
    parser.add_argument('--precision', choices=('fp32', 'amp'), default='fp32')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--native-timeout-s', type=int, default=300)
    parser.add_argument('--hcu-ids', required=True)
    parser.add_argument('--nnodes', type=int, default=1)
    parser.add_argument('--node-rank', type=int, default=0)
    parser.add_argument('--output-dir', required=True)
    return run(parser.parse_args(), cancellation())


if __name__ == '__main__':
    raise SystemExit(main())
