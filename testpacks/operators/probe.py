"""Target-only read-only probe; never allocate benchmark tensors or change device settings."""
import json
from pathlib import Path
import sys


def inspect(recipe):
    import torch
    if not getattr(torch.version, "hip", None) or not torch.cuda.is_available():
        raise RuntimeError("Fixed HCU recipes require a working HIP PyTorch runtime")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one HCU must be visible inside the native process")
    if recipe == "pa_decode":
        # The supplied script chooses its backend by these paths. Do not silently
        # enter its NVIDIA branch (which uses page size 16 instead of 128).
        if Path("/usr/local/cuda/bin/nvcc").exists():
            raise RuntimeError("PA native script would select its CUDA branch; use the matching HCU image")
        if not (Path("/opt/dtk").exists() or Path("/opt/rocm/llvm/bin/clang").exists()):
            raise RuntimeError("PA native script cannot identify its prepared HIP toolkit")
    device = torch.cuda.get_device_properties(0)
    return dict(torch_version=torch.__version__, hip_version=torch.version.hip,
                logical_device=0, visible_count=1, name=device.name,
                total_memory=device.total_memory,
                architecture=getattr(device, "gcnArchName", None),
                uuid=str(getattr(device, "uuid", "unavailable")))


if __name__ == "__main__":
    print("BENCH_DEVICE " + json.dumps(inspect(sys.argv[1]), allow_nan=False))
