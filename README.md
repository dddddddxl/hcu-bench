# HCU Bench

HCU Bench 是面向海光 GPU 测试的命令行工具，将通信、PCIe、显存带宽和模型精度测试统一到一份 YAML 中。准备好测试环境后，修改配置即可单项运行、组合运行或一键执行：

```bash
bash bench.sh rccl          # 单个 suite
bash bench.sh pcie,memory   # 多个 suite
bash bench.sh all           # YAML 中全部启用项
```

工具负责参数展开、节点调度、进程与容器生命周期、日志归档和结果汇总。原生测试工具、模型、数据集和测试镜像由使用者提供，不自动安装或升级。

## 支持的测试

| Suite | 当前功能 | 节点范围 | 主要结果 |
| --- | --- | --- | --- |
| `rccl` | MPI AllReduce sum，支持消息大小扫描、预热、迭代次数和数据类型 | 单机 2–8 卡 | 延迟、algbw、busbw、原生正确性检查 |
| `pcie` | CPU→GPU、GPU→CPU、GPU→GPU、CPU↔GPU、GPU↔GPU 五类路径；默认 512 MiB | 单机，逐对串行 | 各路径 Avg BW、Peak BW，五类 CSV |
| `memory` | hcu-stream 的 zd1 / zd2 共 24 项，覆盖显存、L1/L2 缓存和 LDS | 单机，指定卡逐卡串行 | 每卡、每用例的 GB/s |
| `deepep` | 原生低延迟压测，默认 600 秒；支持 token、hidden、top-k、expert 数 | 单机 / 多机 | 有效带宽、分阶段带宽和延迟 |
| `mooncake` | RDMA / VRAM 压测，支持 pair、fan-in、pairs、ring | 多机，至少两节点 | 吞吐、原生日志、可选吞吐下限判断 |
| `e2e` | SGLang 0.5.12 GSM8K 精度测试；自启动服务、评测、清理；默认 100 题、5-shot | 单节点服务 | 分数、评测耗时、回答及原生报告 |
| `operators` | 扩展入口和模拟用例 | 由接入的测试包决定 | **尚未内置真实算子测试** |

`e2e` 当前提供 DeepSeek-V4-Flash Channel INT8 / BW1100 8 卡参考 profile，不是通用推理性能测试，也不支持多机模型服务编排。其他模型需要单独准备并验证启动 profile。

“已接入”表示可由统一入口调用，不表示任意卡型、镜像或原生工具版本都已验证。PCIe 和 memory 已覆盖解析与编排回归，新的统一入口仍需目标 GPU 验证；原生测试成功不等于整套 `all` 已验收。

## 快速开始

以下是在**已有 Linux 测试容器内**运行的推荐流程，不需要 Docker socket。

### 1. 获取代码并确认依赖

```bash
git clone https://github.com/dddddddxl/hcu-bench.git
cd hcu-bench
python3 -c 'import sys, yaml, psutil; assert sys.version_info >= (3, 10); print("Python dependencies OK")'
```

运行入口需要 Python 3.10+ 和 PyYAML；内置原生包装器还需要 psutil。满足依赖时可以直接从仓库运行，无需安装 HCU Bench。

| 测试 | 执行环境需要提前准备 |
| --- | --- |
| RCCL | 已编译的 MPI 版 `all_reduce_perf`、Open MPI、匹配的 DTK/RCCL |
| PCIe | `rocm-bandwidth-test`，当前解析契约为 2.6.0 的逐对输出 |
| memory | 针对目标 GPU 架构编译的 main-staging-1.0 `hcu-stream` |
| DeepEP | 匹配镜像接口的 `test_low_latency.py`、同目录 `utils.py`、Torch/DeepEP |
| Mooncake | `transfer_engine_bench`、RDMA 环境、各节点共同可写的共享目录 |
| e2e | SGLang 0.5.12、本地模型、GSM8K JSONL、服务 profile、EP config |

原生源码和二进制不随本仓库分发。缺少依赖时应先准备环境，不在待验镜像中直接升级软件来绕过错误。控制机或开发环境的安装方式见文末。

### 2. 复制并修改 YAML

```bash
cp configs/in-container.template.yaml configs/bench.yaml
```

编辑 `configs/bench.yaml`。模板带中文注释，主要修改以下内容，路径均使用**容器内绝对路径**：

| 配置位置 | 填写内容 |
| --- | --- |
| `vars.workspace` / `vars.bench_root` | 可写输出目录 / 当前仓库目录 |
| `vars.gpus` | 本次允许使用的卡号；先确认无其他业务占用 |
| `vars.rccl_binary` / `vars.rccl_ranks` | RCCL 工具路径 / 进程数，进程数等于卡数 |
| `vars.pcie_binary` / `vars.pcie_*_devices` | PCIe 工具路径及其 CPU/GPU Device ID |
| `vars.memory_binary` / `vars.memory_gpus` | hcu-stream 路径 / 测试卡，必须属于 `gpus` |
| `vars.deepep_script` / `vars.iface` / `vars.gid` | DeepEP 脚本、实际网卡和 GID |
| `vars.model` / `dataset` / `profile` / `ep_config` | e2e 的本地模型、数据集、服务启动参数和 EP 配置 |

只填写需要运行的 suite；不运行的测试块设为 `enabled: false`，不要删除公共变量定义。模板中的 `YOUR_*` 是占位符，不是可直接使用的配置。

PCIe 的 Device ID 必须先在同一执行环境运行 `rocm-bandwidth-test -e` 获取，并根据 PCI 地址核对；**它不是 HCU 卡号**。显存带宽与 PCIe 是两种测试，不能互换二进制或设备编号。

e2e 的启动命令存放在 JSON profile 的 `server_argv` 中，环境变量存放在 `env` 中。复制 [Flash INT8 profile](testpacks/sglang_accuracy/flash-int8-bw1100.template.json) 后修改实际网卡和服务参数，再将路径填入 YAML；不要把整条 shell 命令塞进 YAML 的模型路径。配置示例见 [参数与模型配置](docs/configuration.md)。

### 3. 检查并运行

```bash
bash bench.sh check -c configs/bench.yaml --suite all
bash bench.sh plan -c configs/bench.yaml --suite all
bash bench.sh all
```

`check` 检查配置和参数，`plan` 展示实际命令；二者都不会连接节点、启动容器或占用 GPU，也不证明环境已经可运行。原生工具、模型和依赖在实际执行时检查。

综合模板按 **RCCL → PCIe → memory → DeepEP → e2e** 串行运行。**`all` 只运行当前 YAML 中启用的测试**，不自动加入 Mooncake、算子或任何模拟项。要加入 Mooncake，将其模板中的节点、容器配置和测试块合并到同一 YAML。

`configs/bench.yaml` 是默认配置路径，已被 Git 忽略。升级仓库不会覆盖个人配置，也不会自动增加新 suite；新增测试块需从最新模板补入。

## 常用命令与参数

```bash
bash bench.sh list                         # 查看 suite 和接入范围
bash bench.sh rccl
bash bench.sh pcie
bash bench.sh memory
bash bench.sh deepep
bash bench.sh e2e
bash bench.sh deepep,e2e                   # 仍按 YAML 中的顺序执行
bash bench.sh configs/another-node.yaml all

# 参数仅对本次运行生效
bash bench.sh deepep --set vars.duration_s=1200
bash bench.sh pcie --set tests.pcie_bandwidth.params.size_mib=64
bash bench.sh memory --set tests.memory_bandwidth.params.presets=zd2

# 后台执行；RUN_ID 使用启动时打印的真实路径
bash bench.sh all --detach
bash bench.sh status /work/bench-output/results/RUN_ID
bash bench.sh stop /work/bench-output/results/RUN_ID
bash bench.sh report /work/bench-output/results/RUN_ID
```

| 范围 | 常调参数 |
| --- | --- |
| 公共 | `enabled`、`nodes`、`repeats`、`timeout_s`、`failure_policy` |
| RCCL | `ranks`、`min_bytes`、`max_bytes`、`step_factor`、`datatype`、`warmup_iters`、`iters` |
| PCIe | `cpu_devices`、`gpu_devices`、`size_mib`、`iters` |
| memory | `devices`、`presets`、`model_mhz`、`array_size` |
| DeepEP | `duration_s`（测试块）、`num_processes`、`num_tokens`、`hidden`、`num_topk`、`num_experts` |
| Mooncake | `topology`、`duration`、`buffer_size`、`threads`、`batch_size`、`block_size`、`operation` |
| e2e | `profile`、`model_path`、`num_examples`、`num_shots`、`num_threads`、`max_tokens`、`min_score` |

参数通常写在测试块的 `params` 中；字段、范围和默认值以各测试包的 `manifest.yaml` 为准。`matrix` 可展开参数组合，`repeats` 可重复整个用例，见 [配置说明](docs/configuration.md)。

DeepEP 的 `duration_s` 是计划窗口，必须小于故障时限 `timeout_s`。Mooncake 使用原生 `params.duration`；RCCL、PCIe、memory、e2e 按完整用例结束，不能设置框架 `duration_s`。memory 的 `model_mhz` 仅影响部分用例的带宽换算，**不会锁定硬件频率**。

## 其他运行方式

| 场景 | 模板 | 说明 |
| --- | --- | --- |
| 已有容器内运行 | [in-container.template.yaml](configs/in-container.template.yaml) | 最短使用路径，不需要宿主机 Docker 权限 |
| 宿主机自动创建容器 | [managed-container.template.yaml](configs/managed-container.template.yaml) | 由用户脚本提供启动、检查、停止接口 |
| 控制机通过 SSH 调度 | [remote.template.yaml](configs/remote.template.yaml) | 双节点原生 DeepEP 示例 |
| 仅 PCIe / 仅 memory | [pcie.template.yaml](configs/pcie.template.yaml) / [memory.template.yaml](configs/memory.template.yaml) | 无需填写模型和通信测试输入 |
| Mooncake 多机 | [mooncake.template.yaml](configs/mooncake.template.yaml) | 使用已准备的容器，共享目录须在所有节点可写 |
| 无 GPU 演示 | [demo.yaml](configs/demo.yaml) | 全部为模拟数据，不能作为性能结果 |

自动建容器模式的控制端需要 Python 3.10+、PyYAML 和 Docker。模板保留以下脚本接口，设备、权限、挂载和镜像由使用者明确配置：

```text
bash container.sh start NAME IMAGE HOST_WORKSPACE CONTAINER_WORKSPACE
bash container.sh check NAME
bash container.sh stop NAME
```

修改模板后同样执行 `bash bench.sh all`。工具按用例创建并清理自己的容器，不接管已有同名容器。也支持 `mode: existing` 复用明确指定的容器，不删除它；详细接口见 [容器运行说明](docs/quick-run.md)。

多节点使用已有 SSH 密钥、ssh-agent 或 SSH config，不保存密码。测试代码、模型及依赖须预先部署到目标环境。每个测试独立配置 `nodes`，因此同一 YAML 可以让 RCCL/PCIe/memory/e2e 使用一个节点、DeepEP/Mooncake 使用多个节点；**不能仅给 e2e 增加节点就获得多机推理支持**。

多个模型可写成多个 `suite: e2e` 测试块，各自指定模型和启动 profile，顺序评测。完整示例见 [多模型配置](docs/configuration.md#多个模型顺序评测)。

## 结果与失败处理

每次运行生成独立的 `RUN_ID`，控制台打印输出位置。容器内模板默认写入 `vars.workspace/results/RUN_ID`；SSH 模式的统一报告保存在控制机的 `output_dir`。

| 文件或目录 | 内容 |
| --- | --- |
| `summary.txt` / `summary.json` | 执行状态和分组指标汇总 |
| `metrics.csv` | 逐条指标，含 suite、节点、参数、单位、正确性及模拟标记 |
| `state.json` / `results.jsonl` | 用例状态 / 标准结果记录 |
| `logs/` | 各节点执行日志 |
| `raw/` / `artifacts.jsonl` | 收回的原生报告 / 归档状态 |
| 配置、命令和环境快照 | 本次运行的复测依据 |

日志使用执行环境本地时间及偏移。不同路径、参数、单位和原生统计口径分开汇总，Avg BW 不当成 Peak BW，DeepEP effective bandwidth 不当成 RCCL busbw。

- 普通失败默认清理后继续下一项；`failure_policy: stop` 可立即停止。清理无法确认时，无论策略如何都停止后续测试。
- 达到计划时长不等于正确性通过；超时、原生断言、空结果和异常退出不会被当作成功。
- PCIe、memory 默认只报告带宽；e2e 的 `min_score: ""` 只报告分数。有明确标准后才能启用阈值判断。
- 原生报告缺失或归档失败会单独警告，保留已采集结果；查看汇总时也要检查用例状态和归档状态。
- GPU 资源锁只协调本工具，不能代替业务占用检查。工具不改频率，不清理其他用户的进程或容器。

## 开发与测试包接入

需要新建控制端或开发环境时：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[native]'
python -m unittest discover -s tests -v
bash bench.sh configs/demo.yaml all
```

这不是待验镜像的自动安装步骤。模拟输出标记为 `SIMULATED`，单元测试不依赖 GPU，不能替代硬件验证。Bash 入口优先使用仓库 `.venv/bin/python`，也可通过 `BENCH_PYTHON=/path/to/python` 指定已有解释器。Windows 可作为控制端运行 `hcu-bench` CLI，原生 GPU 测试目标为 Linux。

接入新测试只需编写 manifest 和必要的包装脚本，声明参数、启动命令、指标及原生报告路径，不必修改调度框架。见 [测试包接口](docs/testpacks.md)。

模块说明：[RCCL](docs/rccl.md) · [PCIe](docs/pcie.md) · [显存带宽](docs/memory.md) · [DeepEP / GSM8K](docs/native-tests.md) · [Mooncake](docs/mooncake.md)。
