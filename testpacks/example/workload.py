import json
import sys
from pathlib import Path
from time import perf_counter_ns

size, samples = int(sys.argv[1]), int(sys.argv[2])
results = []
for index in range(samples):
    started = perf_counter_ns()
    value = sum(range(size))
    elapsed = perf_counter_ns() - started
    result = {"measurement_status": "measured", "correctness": "passed" if value == size * (size - 1) // 2 else "failed",
              "sample_id": index, "metrics": [{"name": "example_cpu_sum", "value": elapsed, "unit": "ns", "statistic": "single_iteration"}]}
    results.append(result)
    print("BENCH_RESULT " + json.dumps(result), flush=True)
Path(sys.argv[3]).write_text(json.dumps(results), encoding="utf-8")
