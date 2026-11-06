"""Observe native SGD updates without replacing the supplied training implementation."""
import argparse
import math
from pathlib import Path
import runpy
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_common import write_json


class TrainingWindow:
    def __init__(self, batch_size, warmup_steps, measured_steps, synchronize, clock=time.perf_counter):
        if any(type(x) is not int or x <= 0 for x in (batch_size, warmup_steps, measured_steps)):
            raise ValueError('Positive integer batch size and step counts required')
        self.batch_size, self.warmup_steps, self.measured_steps = batch_size, warmup_steps, measured_steps
        self.synchronize, self.clock = synchronize, clock
        self.updates = 0
        self.optimizer = None
        self.started = self.elapsed = None

    def step(self, optimizer, args, kwargs):
        if self.optimizer is None:
            self.optimizer = optimizer
        if optimizer is not self.optimizer:
            raise ValueError('Only one optimizer is supported by the throughput window')
        self.updates += 1
        if self.updates == self.warmup_steps:
            self.synchronize()
            self.started = self.clock()
        elif self.updates == self.warmup_steps + self.measured_steps:
            self.synchronize()
            self.elapsed = self.clock() - self.started

    def result(self):
        if (self.updates != self.warmup_steps + self.measured_steps or self.elapsed is None
                or not math.isfinite(self.elapsed) or self.elapsed <= 0):
            raise ValueError('Incomplete or invalid synchronized training window')
        images = self.batch_size * self.measured_steps
        return dict(complete=True, total_updates=self.updates, batch_size=self.batch_size,
                    warmup_steps=self.warmup_steps, measured_steps=self.measured_steps,
                    measured_images=images, elapsed_s=self.elapsed, images_per_second=images / self.elapsed,
                    scope='single-HCU real-image training: data loading, augmentation, H2D, forward, loss, backward and SGD',
                    excluded='initialization, warmup updates, validation and checkpoint IO',
                    timing='perf_counter, device synchronize after last warmup and last measured optimizer update',
                    synchronization='window boundaries only; not per-step', correctness='not_checked')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-script', required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--warmup-steps', type=int, required=True)
    parser.add_argument('--measured-steps', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('train_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one assigned HCU must be visible')
    window = TrainingWindow(args.batch_size, args.warmup_steps, args.measured_steps, torch.cuda.synchronize)

    def observe(optimizer, positional, keywords):
        if not isinstance(optimizer, torch.optim.SGD):
            raise ValueError('Native throughput recipe requires SGD')
        window.step(optimizer, positional, keywords)

    handle = register_optimizer_step_post_hook(observe)
    original_argv = sys.argv
    script = str(Path(args.train_script).resolve())
    sys.argv = [script] + (args.train_args[1:] if args.train_args[:1] == ['--'] else args.train_args)
    try:
        runpy.run_path(script, run_name='__main__')
        write_json(args.output, window.result())
    finally:
        handle.remove()
        sys.argv = original_argv


if __name__ == '__main__':
    main()
