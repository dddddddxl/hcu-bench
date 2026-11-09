"""Opt-in, hash-pinned gfx938 EDPP finite-loop smoke; never changes device settings."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_common import CleanupError, cancellation, cleanup_failure, emit, sha256, write_json
from fixed_common import claim_output, environment_record, execute, single_device

ROOT = Path(__file__).resolve().parent
ASSETS = {
    'gemmPower_edpp': 'c1950ca0108223034a545c52fcf6dce07f733b3e9a729462fb304ec83a6252eb',
    'fp8_nmz_edpp.co': '67d005b2ab4dfdf3c2f2742f32637e403121fecb3644f1b76da0d459858bdc66',
}


def compatible(device):
    arch = device.get('architecture', '').split(':')[0]
    if not re.fullmatch(r'gfx[0-9a-f]+', arch) or device.get('visible_count') != 1:
        raise ValueError('Invalid HIP device probe result')
    return arch == 'gfx938'


def compiler():
    for name in ['hipcc', '/opt/dtk/hip/bin/hipcc', '/opt/dtk/bin/hipcc', '/opt/rocm/bin/hipcc']:
        found = shutil.which(name)
        if found:
            return found
    raise ValueError('Prepared HIP compiler/headers required for read-only architecture probe')


def parse(text, iterations):
    if 'f8f8sgemm[NT]' not in text or re.search(r'(?i)\b(error|failed|failure|mismatch|nan|inf)\b', text):
        raise ValueError('Native GEMM signature missing or native error reported')
    observed = [int(x) for x in re.findall(r'^iterator:\s*(\d+)\s*$', text, re.M)]
    if observed != list(range(iterations)):
        raise ValueError('Incomplete native iterations; refusing partial success')


def sample(pci):
    if not re.fullmatch(r'[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]', pci):
        raise ValueError('Invalid PCI identity')
    base = Path('/sys/bus/pci/devices') / pci.lower()
    values = {}
    for pattern in ['gpu_busy_percent', 'mem_info_vram_used', 'hwmon/hwmon*/power*_average',
                    'hwmon/hwmon*/temp*_input', 'ras/*err_count']:
        for p in base.glob(pattern):
            try:
                values[str(p.relative_to(base))] = p.read_text().strip()
            except OSError:
                pass
    return dict(time=time.time(), pci=pci, values=values)


def run(args, stopped):
    out = claim_output(args.output_dir)
    executions = []
    try:
        if not 1 <= args.iterations <= 1000 or not 1 <= args.native_timeout_s <= 3600:
            raise ValueError('Invalid finite-loop limits')
        env = single_device(args, dict(os.environ))
        if any(k.startswith('HSA_OVERRIDE_') and v for k, v in env.items()) or env.get('HSA_VISIBLE_DEVICES'):
            raise ValueError('Architecture overrides and secondary visibility masks are not allowed')
        env.update(OMP_NUM_THREADS='4', OMP_WAIT_POLICY='PASSIVE')
        launch = lambda argv, name, limit: execute(argv, name, out, env, stopped, executions, limit)
        launch([compiler(), str(ROOT / 'probe.cpp'), '-o', str(out / 'device-probe')], 'probe-build', 120)
        device = json.loads(launch([str(out / 'device-probe')], 'device-probe', 30).strip())
        write_json(out / 'device.json', device)
        if not compatible(device):
            reason = 'Bundled EDPP loader/co support gfx938 only; actual=' + device['architecture']
            write_json(out / 'assessment.json', dict(complete=False, status='skipped', reason=reason, native_started=False))
            emit(dict(measurement_status='skipped', skip_case=True, correctness='not_checked', metrics=[], reason=reason))
            return 0
        for name, expected in ASSETS.items():
            if sha256(ROOT / 'native' / name) != expected:
                raise ValueError('Bundled asset SHA256 mismatch: ' + name)
        binary = out / 'gemmPower_edpp'
        shutil.copyfile(ROOT / 'native/gemmPower_edpp', binary)
        binary.chmod(0o700)
        co = ROOT / 'native/fp8_nmz_edpp.co'
        argv = [str(binary), '-m', '2048', '-n', '2048', '-k', '12032', '-g', '22', '-t', '1',
                '-f', str(co), '-w', '-l', str(args.iterations), '-vs']
        write_json(out / 'plan.json', dict(argv=argv, device=device, assets=ASSETS,
                   upstream='HYGON-AI/hcu-dcgm@b3d37387be4264fa25325922d01dd056ef999a64',
                   environment=environment_record(env), correctness='not_checked', mode='finite_smoke'))
        before = sample(device['pci'])
        write_json(out / 'telemetry-before.json', before)
        finished = threading.Event()
        rows, warnings = [], []

        def collect():
            while not finished.wait(0.5):
                try:
                    rows.append(sample(device['pci']))
                except Exception as exc:
                    warnings.append(str(exc))
                    return

        collector = threading.Thread(target=collect, daemon=True)
        collector.start()
        start = time.monotonic()
        try:
            text = launch(argv, 'native', args.native_timeout_s)
        finally:
            elapsed = time.monotonic() - start
            finished.set(); collector.join(timeout=2)
            if collector.is_alive():
                warnings.append('Telemetry read did not finish promptly; sampling is incomplete')
            (out / 'telemetry.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in list(rows)))
            write_json(out / 'telemetry-after.json', sample(device['pci']))
        parse(text, args.iterations)
        if stopped.is_set():
            raise RuntimeError('Cancelled before publishing complete finite smoke')
        result = dict(measurement_status='measured', correctness='not_checked', recipe='gemmpower_edpp',
                      mode='finite_smoke', iterations=args.iterations, architecture=device['architecture'],
                      telemetry_samples=len(rows), telemetry_warnings=warnings,
                      scope='native finite-loop completion only; not peak/steady power or MMAC reproduction',
                      metrics=[dict(name='native_job_elapsed', value=elapsed, unit='s', statistic='wall_time_including_CPU_reference')])
        write_json(out / 'measurements.json', result)
        write_json(out / 'assessment.json', dict(complete=True, correctness='not_checked', mode='finite_smoke'))
        emit(result)
        return 0
    except CleanupError as exc:
        return cleanup_failure(out, exc)
    except Exception as exc:
        write_json(out / 'assessment.json', dict(complete=False, error=str(exc)))
        emit(dict(measurement_status='invalid', correctness='not_checked', metrics=[], reason=str(exc)))
        return 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iterations', type=int, default=4)
    p.add_argument('--native-timeout-s', type=int, default=30)
    p.add_argument('--hcu-ids', required=True)
    p.add_argument('--nnodes', type=int, default=1)
    p.add_argument('--node-rank', type=int, default=0)
    p.add_argument('--output-dir', required=True)
    return run(p.parse_args(), cancellation())


if __name__ == '__main__':
    raise SystemExit(main())
