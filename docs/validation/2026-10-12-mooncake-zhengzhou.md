# Zhengzhou Mooncake validation — 2026-10-12

The new pack was exercised through the real hcu-bench SSH executor on two BW150 nodes. Each node used one explicitly selected idle HCU and one ACTIVE/LINK_UP `mlx5_0` NIC rated at 200 Gb/s. This is single-HCU/single-NIC validation, not the original handbook's eight-HCU/eight-NIC reference topology.

## Environment

- Mooncake distribution: `0.3.13.post1+dtk2604.2609222101.g466e70` on both nodes.
- Image: `sha256:965d44ef052951f18e74090af9663f5cbeba40b902775276800b5841527141ee`.
- Python 3.10.12; PyYAML 6.0.3; psutil 7.2.2.
- Image PyTorch distribution: `2.11.0+dtk2604.2609291046.ge12687`; Torch is not the transfer test driver.
- Recorded DTK label: `dtk26041-72cu-0911`; driver `6.4.0-V1.7.0`.
- Dedicated test containers used host networking, DCU/RDMA device passthrough, unlimited memlock, explicit HIP visibility and a shared task directory. No packages, NIC settings or drivers were changed on hosts or other users' containers.
- Hosts had Python 3.6.8. An explicit SSH interpreter shim invoked Python inside our own prepared container, without a Docker socket mount or host Python installation.
- Peer IP reachability, native flag availability and selected NIC states passed checks; native logs confirmed VRAM allocation and RDMA ready ACKs.

## Actual measurements and expected outcomes

All tests used P2PHANDSHAKE, automatic discovery restricted by `MC_TE_FILTERS=mlx5_0`, logical HCU 0, 12 submission threads, batch size 128 and block size 65,536 bytes. Short tests used a 256 MiB buffer; the 180-second test used 1 GiB. Values are the native final aggregate throughput, not per-iteration percentiles.

| Run ID | Topology / operation | Native window | Throughput | Outcome |
| --- | --- | --- | --- | --- |
| `20261012_203835_4a1cf7dd` | Pair / read | 10.01 s | 24.46 GB/s | completed, report-only |
| `20261012_204401_c1785c50` | Pair / read | 180.01 s | 24.50 GB/s | completed, report-only |
| `20261012_204916_43e8575e` | Pair / read, deliberate 100 GB/s floor | 5.01 s | 24.46 GB/s | expected failure, CLI exit 1, measurement retained |
| `20261012_205037_6a8e7fb9` | Two-node ring / read | 10.01 s each | 24.46 GB/s in each direction | completed, both roles cleaned on both nodes |
| `20261012_205448_71515c40` | Pair / write, explicit 20 GB/s floor | 10.01 s | 24.45 GB/s | completed, performance_judgement=passed |

The 100 and 20 GB/s settings exercise failure and pass paths; they are not universal acceptance criteria. All measurements remain `correctness=not_checked`.

The target endpoint was read from each newly started native target's logs; the pair and ring tests used fresh run/case-scoped shared records. No fixed or cached segment port was used. Each framework report retained original native logs, parameters, commands, package/binary/wrapper hashes, measurements and cleanup records. Native `Test completed` lines preceded the successful initiator done records; passive targets produced no fabricated bandwidth.

## Software tests and cleanup

- **105 unittest tests passed** on Windows / Python 3.13.
- **105 unittest tests passed** in the Linux image / Python 3.10 with `--init`, including CPU substitute integration for four-node fan-in, disjoint pairs and ring, invalid/missing measurements, independent hung-initiator deadlines, threshold outcomes and unrelated-process preservation.
- The first Linux suite in a `sleep` PID-1 container exposed an existing test's assumption that an orphan child's PID immediately disappears: the exited child remained a zombie. Running the suite in a separate `--init` test container passed without changing/skipping tests. This is why the deployment guide recommends an init/subreaper for long-lived containers.
- All native benchmark processes were absent from both dedicated pressure containers after runs, and framework node cleanup was confirmed. Only containers created for this task were stopped; the task directory and reports were preserved.

## Coverage limits

Two-node pair, two-node ring, read/write and explicit throughput gates were tested on hardware. Four-node fan-in, disjoint pairs, four-node ring and all six pair combinations are implemented/tested with CPU substitutes but were not validated on four physical nodes in this run. Other HCU had existing workloads; this is not a claim of an exclusive whole-node/NIC environment or a complete long-term stability assessment. No model inference, payload correctness or RCCL validation is claimed.
