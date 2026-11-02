import torch
import sys
import os

a = torch.randn((4096, 4096), dtype=torch.float16, device="cuda")
b = torch.randn((4096, 4096), dtype=torch.float16, device="cuda")

c = torch.matmul(a, b)

pressure_count = int(os.environ.get("BENCH_STRESS_ITERATIONS", "1000000"))
if not 1 <= pressure_count <= 1000000:
    raise ValueError("BENCH_STRESS_ITERATIONS must be from 1 to 1000000")
for p in range(pressure_count):
    __c = torch.matmul(a.clone(), b.clone())
    sys.stdout.write("\rPressure Test: {}/{}".format(p + 1, pressure_count))
    assert torch.equal(c, __c), "unstable"
print(" \x1b[32mPASS\x1b[0m")
