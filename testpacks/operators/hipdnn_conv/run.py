# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Run the retained DTK test_conv.sh commands without changing native parameters."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'testpacks')]
from native_common import CleanupError, cancellation, cleanup_failure, emit, sha256, write_json
from fixed_common import claim_output, environment_record, execute, single_device
from testpacks.prepare import tree_hash

SOURCE = ROOT / 'vendor/operator-cases/hipdnn'
LABELS = {'1': 'fprop', '2': 'dgrad', '4': 'wgrad'}


def run(args, stopped):
    out = claim_output(args.output_dir)
    executions, results = [], []
    deadline = time.monotonic() + args.native_timeout_s
    try:
        env = single_device(args, dict(os.environ))
        expected = json.loads((ROOT / 'vendor/sources.json').read_text())['operator-cases']['tree_sha256']
        if tree_hash(SOURCE.parent) != expected:
            raise ValueError('Bundled operator source checksum mismatch')
        cases = load_cases(SOURCE / 'test_conv.sh')
        if args.case_id:
            cases = [case for case in cases if case['case_id'] == args.case_id]
        if not cases:
            raise ValueError('No matching original DTK test_conv command')
        def launch(argv, name, run_env=env):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or stopped.is_set():
                raise InterruptedError('DTK test_conv run cancelled or timed out')
            return execute(argv, name, out, run_env, stopped, executions, remaining)
        binary = resolve_binary(args, out, launch)
        # The original program has no --help; argument errors are native case failures.
        probe = launch([sys.executable, str(Path(__file__).parents[1] / 'numeric/probe.py')], 'device-probe')
        devices = [json.loads(line.split('BENCH_DEVICE ', 1)[1]) for line in probe.splitlines() if line.startswith('BENCH_DEVICE ')]
        if len(devices) != 1 or devices[0].get('visible_count') != 1:
            raise ValueError('Missing single-device identity')
        write_json(out / 'plan.json', dict(binary=binary, binary_sha256=sha256(binary),
                   script_sha256=sha256(SOURCE / 'test_conv.sh'), params=vars(args), cases=cases,
                   device=devices[0], environment=environment_record(env),
                   timing='Native test_conv event timer; no added iterations or repeats'))
        for case in cases:
            row = dict(case_id=case['case_id'], dtype=case['dtype'], direction=case['direction'],
                       recipe='hipdnn_conv', correctness='not_checked', measurement_status='invalid', metrics=[])
            try:
                text = launch([binary] + case['argv'], case['case_id'], {**env, **case['env']})
                native = parse_result(text, case['direction'], case['dtype'])
                row.update(correctness=native['correctness'], validation_status=native['validation_status'],
                           measurement_status='measured',
                           metrics=[dict(name=case['case_id'] + '_latency_ms', value=native['latency_ms'],
                                         unit='ms', statistic='native_mean')])
            except CleanupError:
                raise
            except Exception as exc:
                row['reason'] = str(exc)
                if isinstance(exc, VerificationError):
                    row['correctness'] = 'failed'
            results.append(row)
            emit(row)
            write_json(out / 'measurements.json', results)
            if stopped.is_set() or time.monotonic() >= deadline:
                break
        complete = len(results) == len(cases) and all(r['measurement_status'] == 'measured' for r in results)
        write_json(out / 'assessment.json', dict(complete=complete,
                   correctness_complete=complete and all(r['correctness'] == 'passed' for r in results),
                   expected_cases=len(cases), results=results))
        return 0 if complete else 2
    except CleanupError as exc:
        return cleanup_failure(out, exc)
    except Exception as exc:
        write_json(out / 'assessment.json', dict(complete=False, reason=str(exc), results=results))
        emit(dict(recipe='hipdnn_conv', correctness='not_checked', measurement_status='invalid', metrics=[], reason=str(exc)))
        return 2


def load_cases(path):
    cases, env = [], {}
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        words = shlex.split(line, comments=True)
        if not words:
            continue
        if words[0] == 'export':
            key, value = words[1].split('=', 1)
            env[key] = value
            continue
        if words[0] != 'test_conv':
            raise ValueError('Unexpected DTK script command at line ' + str(number))
        args = words[1:]
        if '--cpu-validation' not in args:
            raise ValueError('Original CPU validation flag missing')
        cases.append(dict(case_id='dtk-conv-%03d' % (len(cases) + 1), argv=args,
                          env=dict(env), source_line=number, dtype=args[args.index('-d') + 1],
                          direction=args[args.index('-F') + 1]))
    return cases


def resolve_binary(args, out, launch):
    if args.binary != 'auto':
        found = shutil.which(args.binary)
        if not found:
            raise ValueError('Explicit test_conv binary missing: ' + args.binary)
        return str(Path(found).resolve())
    if not args.source_root:
        for name in ('test_conv', '/opt/dtk/bin/test_conv'):
            found = shutil.which(name)
            if found:
                return str(Path(found).resolve())
        raise ValueError('test_conv is not installed; set source_root to the original DTK hipdnn/api_test directory to build it')
    source = Path(args.source_root).resolve()
    files = [source / 'CMakeLists.txt', source / 'cpp/test_conv/test_conv.cpp', source / 'cpp/utils.hpp']
    write_json(out / 'build-inputs.json', dict(source_root=str(source), files={str(p): sha256(p) for p in files}))
    build = out / 'native-build'
    commands = [['cmake', '-S', str(source), '-B', str(build), '-G', 'Ninja'],
                ['cmake', '--build', str(build), '--target', 'test_conv', '-j', str(args.build_jobs)]]
    write_json(out / 'build-commands.json', commands)
    for index, argv in enumerate(commands):
        launch(argv, 'build-%d' % index)
    return str(build / 'bin/test_conv')


class VerificationError(ValueError):
    pass


def parse_result(text, direction, dtype):
    if re.search(r'CPU validation result:\s*failed\b', text):
        raise VerificationError('Native CPU validation failed')
    unsupported = 'CPU validation is not supported for int8 input type.' in text
    if unsupported and dtype != 'int8':
        raise ValueError('Native validation dtype does not match the original case')
    if re.search(r'\b(?:HIP Error:|hipDNN(?: Frontend)? Error:|Exception:|Unsupported data type:)', text, re.I):
        raise ValueError('Native test_conv error; inspect original log')
    if not unsupported and len(re.findall(r'CPU validation result:\s*successful\b', text)) != 1:
        raise ValueError('Missing or ambiguous native CPU validation verdict')
    times = re.findall(r'\b' + LABELS[direction] + r' avg times:\s*([0-9.eE+-]+)\s+ms\b', text)
    if len(times) != 1:
        raise ValueError('Missing or ambiguous native timing')
    value = float(times[0])
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Invalid native timing')
    return dict(latency_ms=value, correctness='not_checked' if unsupported else 'passed',
                validation_status='native_not_supported' if unsupported else 'native_cpu_passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='auto')
    parser.add_argument('--source-root', default='')
    parser.add_argument('--case-id', default='')
    parser.add_argument('--build-jobs', type=int, default=2)
    parser.add_argument('--native-timeout-s', type=int, default=14400)
    parser.add_argument('--hcu-ids', required=True)
    parser.add_argument('--nnodes', type=int, default=1)
    parser.add_argument('--node-rank', type=int, default=0)
    parser.add_argument('--output-dir', required=True)
    return run(parser.parse_args(), cancellation())


if __name__ == '__main__':
    raise SystemExit(main())
