# DCU-hipbenchmark

A tool for hardware **performance** measurements on DCU (memory bandwidth, latency, related compute patterns).

## Features

1. **Bandwidth tests**  
   Memory read/write, cache read/write, LDS read/write, matrix-related patterns, and type-specific operations under configurable strides and wave counts.

2. **Latency tests**  
   Instruction-level timing for memory, cache, and LDS paths.

3. **Other workloads**  
   GEMM-style memory, element-wise, FA/PA-style cases, L2/TCC helpers, dispatch latency, and optional scene binaries under `scene/`.

## Repository layout

| Path | Role |
|------|------|
| `main.cpp` | CLI (`argparse`) and dispatch to testcase modules |
| `include/` | Shared headers: `testcase.hpp`, `device.hpp`, `output.hpp`, `argparse/`, type helpers, GEMM helpers |
| `src/memory/` | Bandwidth-heavy implementations (`BWtestcase.cpp`, L2/TCC, GEMM/FA/PA, element-wise) |
| `src/latency/` | Latency and kernel-dispatch testcase hosts |
| `src/compute/` | TFLOPs / GEMM peak style hosts |
| `src/UnitTest/` | VGPR-related unit paths |
| `kernel/` | HIP device code (headers included by host translation units) |
| `cmake/hcu_stream_sources.cmake` | **Single list** of `.cpp` sources for the main executable (extend here when adding modules) |
| `test_script/` | Shell/Python helpers for batch / BMZ workflows |
| `scene/` | Optional larger workloads (e.g. flash-attn) with their own `CMakeLists.txt` |

Adding a new testcase: update `cmake/hcu_stream_sources.cmake` only if you introduce a **new** `.cpp` file; otherwise follow the steps in **How to add code** below.

## Requirements

ROCm (or compatible HIP toolchain), matching driver, and a device supported by your `-DGPU_ARCH` / `-DGPU_ARCH_NUM` settings.

## Build

### Option 1: CMake

```bash
mkdir build && cd build
```

Single architecture:

```bash
cmake -DGPU_ARCH=gfx936 -DGPU_ARCH_NUM=936 ..
cmake --build .
```

The default binary name is **`hcu-stream`**, output under `build/bin/` (or your generator’s equivalent).

Multi-architecture (one binary per ISA):

```bash
cmake -DMULTI_ARCH_BUILD=ON ..
cmake --build .
```

Optional fat binary (all ISAs in one executable, when supported by your toolchain):

```bash
cmake -DMULTI_ARCH_BUILD=ON -DBUILD_FATBINARY=ON ..
cmake --build .
```

### Option 2: Makefile

```bash
make
```

(Adjust `Makefile` if your install path differs from `/opt/rocm`.)

## Code style

- **Formatting**: run `clang-format` using the repo-root `.clang-format` (LLVM-based, 2-space indent, 100 columns). Example: `clang-format -i main.cpp src/**/*.cpp`.
- **Editor defaults**: `.editorconfig` sets UTF-8, LF, and trim-trailing-whitespace for C/C++/CMake/shell/Python.

## Usage

After a CMake build, run for example:

```bash
./bin/hcu-stream -h
./bin/hcu-stream bandwidth -h
./bin/hcu-stream bandwidth -l
```

Global options:

- `-o`, `--output` — output path (see help for current limitations)
- `--verbose` — more detailed logging
- `-v`, `--version` — print build timestamp

Subcommands include `bandwidth`, `latency`, `tflops`, `l2_rw`, `tcc_read`, `elewise`, `fa`, `pa`, `gemm`, `kernel_dispatch`, `VGPR_TEST`.

### Bandwidth subcommand notes

- `-d`, `--div` — crop divisor for matrix-style cases  
- `-s`, `--arraysize` — linear buffer size hint  
- `-w`, `--wave` — wavefront / wave count  
- **`-g`, `--device`** — GPU index (short `-g` avoids clashing with `-d` for `--div`)

Example:

```bash
./bin/hcu-stream -o - bandwidth memory_copy -g 0
```

## How to add code

Typical flow for a **new memory bandwidth** case:

1. Declare the member in `include/testcase.hpp` (or the appropriate testcase base).
2. Implement the host-side driver in `src/memory/BWtestcase.cpp` and register it in the case map.
3. Put device kernels in `kernel/memoryBandwidth/<name>.hpp` (or split by concern if the file grows large).
4. If you add a **new** `.cpp` translation unit, append it to `cmake/hcu_stream_sources.cmake` and re-run CMake.

Keep kernel indexing and launch grid (`gridDim` / `blockDim`) consistent; mismatches are a common source of out-of-bounds errors.

## Shell scripts (`test_script/`, `BMZ_peak.sh`, …)

Most scripts call the benchmark binary as `./run` for historical reasons. The CMake/Makefile product is **`hcu-stream`**. Use either:

- `export HCUBENCH_BIN=./hcu-stream` (see `BMZ_peak.sh` for the same pattern), or  
- `ln -s hcu-stream run` in the repo root after build.

## BMZ scripts (optional)

```bash
python3 test_script/merge_patterns_to_csv.py
```

Default GC / memory frequencies and CU count can be adjusted in the script’s prompts. Outputs such as `pattern1_output.txt` merge into `all_pattern.csv`. Remove stale `pattern*_output.txt` before a clean run if needed.
