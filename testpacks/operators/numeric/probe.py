"""Read HIP device identity without Torch or benchmark allocations."""
import ctypes
import ctypes.util
import json

hip = ctypes.CDLL(ctypes.util.find_library("amdhip64") or "libamdhip64.so")
count = ctypes.c_int()
if hip.hipGetDeviceCount(ctypes.byref(count)) != 0 or count.value != 1:
    raise RuntimeError("Exactly one working HIP device must be visible")
bus = ctypes.create_string_buffer(64)
if hip.hipDeviceGetPCIBusId(bus, len(bus), 0) != 0:
    raise RuntimeError("Cannot identify the selected HIP device PCI address")
print("BENCH_DEVICE " + json.dumps(dict(visible_count=count.value, logical_device=0, pci_bus=bus.value.decode())))
