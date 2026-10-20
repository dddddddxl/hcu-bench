# HCU Bench

HCU Bench 是面向海光 HCU 测试的命令行工具，将通信、PCIe、显存带宽、算子压测与性能、模型精度和推理吞吐测试统一编排。测试参数写在 YAML；服务类模型的部署参数放在各自的启动脚本中，MIGraphX 固定模型直接配置 ONNX 路径：

```bash
bash bench.sh rccl          # 单个 suite
bash bench.sh pcie,memory   # 多个 suite
bash bench.sh all           # YAML 中全部启用项
```

**测试源码和包装器随仓库提供；需要编译的原生 C/C++ 工具首次运行自动构建，之后复用缓存。** 算子、DeepEP 和 MIGraphX 包使用执行环境中已准备的框架或二进制。工具负责参数展开、节点调度、进程与容器生命周期、日志归档和结果汇总。驱动、DTK、框架运行库和模型权重仍由执行环境提供，不自动安装或升级。

## 支持的测试

| Suite | 当前功能 | 节点范围 | 主要结果 |
| --- | --- | --- | --- |
| `rccl` | MPI AllReduce sum，支持消息大小扫描、预热、迭代次数和数据类型 | 单机 2–8 卡 | 延迟、algbw、busbw、原生正确性检查 |
| `pcie` | CPU→HCU、HCU→CPU、HCU→HCU、CPU↔HCU、HCU↔HCU 五类路径；默认 512 MiB | 单机，逐对串行 | 各路径 Avg BW、Peak BW，五类 CSV |
| `memory` | hcu-stream 的 zd1 / zd2 共 24 项，覆盖显存、L1/L2 缓存和 LDS | 单机，指定卡逐卡串行 | 每卡、每用例的 GB/s |
| `deepep` | 原生低延迟压测，默认 600 秒；支持 token、hidden、top-k、expert 数 | 单机 / 多机 | 有效带宽、分阶段带宽和延迟 |
| `mooncake` | RDMA / VRAM 压测，支持 pair、fan-in、pairs、ring | 多机，至少两节点 | 吞吐、原生日志、可选吞吐下限判断 |
| `e2e_acc` | SGLang 0.5.12 GSM8K 评测接口，默认 100 题、5-shot | 单节点服务 | 分数、评测耗时、回答及原生报告 |
| `e2e_perf` | EvalScope 1.13.0 流式请求压测；MIGraphX ResNet50 FP16/INT8、BERT-large FP16 固定推理用例 | 服务压测为单节点；MIGraphX 为单机单卡 | 服务吞吐/延迟；MIGraphX 三轮吞吐中位数和 ms/batch |
| `operators` | BLAS FP16、FA FP16/BF16 重复性压测；PA decode、FA FP16/FP8 前向性能 | 单机单卡 | 完成迭代数、TFLOPS、延迟、PA 带宽；保留原始日志 |

`e2e` 是 `e2e_acc` + `e2e_perf` 的分组。对于服务类模型，每个模型启动一次，先运行选中的精度项，再运行吞吐项，清理后换下一个模型。支持 OpenAI 兼容服务，附 SGLang / vLLM 启动脚本模板；不支持多机模型服务编排。具体模型能否启动仍取决于镜像、权重和启动参数。

MIGraphX 使用直接命令测试包，无需 HTTP 服务或 `models` 服务配置。固定用例见 [算子测试](docs/operators.md) 和 [ResNet50 / BERT](docs/migraphx.md)；镜像均由 `vars.image` 选择。Conv 和通用 GEMM 暂未接入（BLAS 固定矩阵压测是独立用例）。百万次压测默认关闭，需明确安排测试窗口后开启。

“已接入”表示可由统一入口调用，不表示任意卡型、镜像或原生工具版本都已验证。PCIe、memory、固定算子和 MIGraphX 包已覆盖解析与编排回归；固定算子与 MIGraphX 的新入口尚未完成目标 HCU 实测，原生测试成功也不等于整套 `all` 已验收。

## 快速开始

以下是在**已有 Linux 测试容器内**运行的推荐流程，不需要 Docker socket。

### 1. 获取代码

```bash
git clone https://github.com/dddddddxl/hcu-bench.git
cd hcu-bench
python3 -c 'import sys, yaml, psutil; assert sys.version_info >= (3, 10); print("Python dependencies OK")'
```

运行入口需要 Python 3.10+ 和 PyYAML；内置原生包装器还需要 psutil。满足依赖时可以直接从仓库运行，无需安装 HCU Bench。

| 测试 | 仓库已提供 | 镜像仍需具备 |
| --- | --- | --- |
| RCCL | 固定版本 rccl-tests 源码，自动编译 MPI AllReduce | make、HIP 编译器、Open MPI、DTK/RCCL 开发库 |
| PCIe | rocm-bandwidth-test 2.6.0 逐对输出版本源码，自动编译 | CMake、C++ 编译器、HSA 开发库 |
| memory | main-staging-1.0 hcu-stream 源码，自动按架构编译 | make、匹配架构的 AICC/DTK |
| DeepEP | `test_low_latency.py`、`utils.py`，包含已验证的 11 项 handle 接口适配 | 匹配接口的 Torch/DeepEP、通信环境 |
| Mooncake | Hygon Transfer Engine、pybind11、yalantinglibs 源码，自动编译 | CMake、C++20 编译器、HIP、RDMA/numa/gflags/glog/jsoncpp/yaml-cpp 开发库、Python 开发头文件；两端共享可写目录 |
| e2e_acc | GSM8K test JSONL、评测包装器、模型启动脚本模板 | 客户端使用执行环境中的 SGLang 0.5.12；另需模型服务框架、本地权重 |
| e2e_perf | EvalScope 1.13.0 源码包、隔离客户端自动准备、模型启动脚本模板 | Python 3.10+ / venv / pip、模型服务框架、本地权重和 tokenizer；首次准备需可访问 Python 包源或已配置 wheelhouse |
| operators | 固定算子脚本、参数目录、解析器和来源 SHA256 | 匹配脚本 ABI 的 Torch、Triton、FlashAttention、NumPy、einops；FA FP16 性能使用原脚本的本地 `.so` 路径 |
| MIGraphX e2e_perf | ResNet50 / BERT 命令、三轮中位数、输入检查和指纹记录 | MIGraphX driver、numactl、onnx、psutil、本地 ONNX；BERT 另需匹配的 rocBLAS 调优库 |

以上是运行镜像的基础依赖，不是要求另找测试代码。原生构建源码版本、校验与授权说明见 [vendor](vendor/README.md)，固定算子脚本来源见 [算子测试](docs/operators.md)。原生 C/C++ 构建不联网下载源码、不安装系统软件包；EvalScope 服务压测首次会在独立缓存虚拟环境安装客户端 Python 依赖，不改服务环境的 DTK、Torch 或 SGLang。MIGraphX 固定模型包不自动安装依赖。依赖不全或准备失败会保留错误，不自动换库。

### 2. 复制并修改 YAML

```bash
cp configs/in-container.template.yaml configs/bench.yaml
```

编辑 `configs/bench.yaml`。模板带中文注释，主要修改以下内容，路径均使用**容器内绝对路径**：

| 配置位置 | 填写内容 |
| --- | --- |
| `vars.workspace` / `vars.bench_root` | 可写输出目录 / 当前仓库目录 |
| `vars.hcus` | 本次允许使用的卡号；先确认无其他业务占用 |
| `vars.rccl_ranks` | RCCL 进程数，等于分配卡数 |
| `vars.pcie_*_devices` | PCIe 工具的 CPU/HCU Device ID，不是 HIP 卡号 |
| `vars.memory_hcus` | 显存测试卡号，必须属于 `hcus` |
| `vars.iface` / `vars.ib_hca` / `vars.gid` | 实际通信网卡、RDMA 设备和 GID |
| `vars.model_script` / `models` | 模型启动脚本路径；模型路径、TP、量化、网卡等部署参数写在脚本内 |
| `build` | 已有 DTK/MPI 路径、架构、构建缓存；默认探测 HCU 架构 |

只填写需要运行的 suite；不运行的测试块设为 `enabled: false`，不要删除公共变量定义。模板中的 `YOUR_*` 是占位符，不是可直接使用的配置。

PCIe 的 Device ID 必须先在同一执行环境运行 `rocm-bandwidth-test -e` 获取，并根据 PCI 地址核对；**它不是 HCU 卡号**。显存带宽与 PCIe 是两种测试，不能互换二进制或设备编号。

模板中的 `*_binary`、`deepep_script`、`dataset` 默认保持 `auto`，无需另找测试源码。已有工具也可填写绝对路径，工具不会替换无效的显式路径。

运行服务类模型测试时，复制一份 [SGLang 模板](model_scripts/sglang.template.sh) 或 [vLLM 模板](model_scripts/vllm.template.sh)，写入已经验证的部署命令，再把路径填入 YAML：

```bash
cp model_scripts/sglang.template.sh model_scripts/model_a.local.sh
# 编辑 model_a.local.sh：权重路径、TP、量化、EP 配置、网卡等
# 在 bench.yaml 中填写 vars.model_script 或 models.<名称>.start_script
```

脚本最后用 `exec` 前台启动服务，监听 `$BENCH_HOST:$BENCH_PORT`，不要加 `nohup` 或 `&`。`*.local.sh` 和 `configs/bench.yaml` 都被 Git 忽略。具体字段见 [端到端测试](docs/e2e.md)。

### 3. 检查并运行

```bash
bash bench.sh check -c configs/bench.yaml --suite all
bash bench.sh plan -c configs/bench.yaml --suite all
bash bench.sh all
```

`check` 检查配置和参数，`plan` 展示实际命令；二者都不会连接节点、启动容器或占用 HCU，也不证明环境已经可运行。原生工具、模型和依赖在实际执行时检查。

综合模板按 **RCCL → PCIe → memory → DeepEP → 模型流水线** 串行运行。**`all` 只运行当前 YAML 中启用的测试**，不会重复运行 e2e 分组，也不自动加入 Mooncake、算子或任何模拟项。要加入 Mooncake，将其模板中的节点、容器配置和测试块合并到同一 YAML。

`configs/bench.yaml` 是默认配置路径，已被 Git 忽略。升级仓库不会覆盖个人配置，也不会自动增加新 suite；新增测试块需从最新模板补入。

### 首次编译与缓存

运行 RCCL、PCIe、memory、Mooncake 时，先校验源码，再在执行环境的缓存目录编译。源码、架构、编译器或已记录的运行库信息变化时会使用新缓存；失败构建不会标记为可用。DeepEP / e2e_acc 不编译推理框架；EvalScope 客户端也缓存于 `build.cache_dir`，首次安装成功后复用。固定算子和 MIGraphX 包不自动安装或重编译框架。

`build.timeout_s` 默认 3600 秒，独立于测试时限；`build.jobs` 默认 2，`build.cache_dir` 指定可写缓存目录。自动建容器时应把缓存放在持久挂载目录，避免每个容器重复编译。Mooncake 会等待所有节点准备好后再开始负载。

memory 默认使用已有 AICC，YY/gfx92a 自动对应 `arch_num: 9210`。**此前 gfx938 + DTK 26.04 的原生编译仍有兼容问题；自动编译不代表已修复该问题。** 新构建链路经过本地软件回归，尚不能视为所有镜像的 HCU 实测通过。

## 固定算子与 ResNet50 / BERT

这两类测试使用独立模板，可以选择不同镜像。模板默认在 Docker 宿主机执行，需要填写容器启动、检查、停止 hook；在已有容器内运行时，删除 `containers` 和测试块中的 `container` 字段，并填写容器内路径。详见 [算子用例](docs/operators.md) 和 [模型用例](docs/migraphx.md)。

```bash
# 算子性能：PA decode、FA FP16、FA FP8，共 3 项。
cp configs/operators.template.yaml configs/bench.yaml
# 编辑 vars.image、容器 hook、仓库路径和唯一目标卡。
bash bench.sh check -c configs/bench.yaml --suite operators
bash bench.sh plan -c configs/bench.yaml --suite operators
bash bench.sh operators --set vars.image=YOUR_APPROVED_OPERATOR_IMAGE
```

BLAS FP16、FA FP16/BF16 三项压力用例另在 `operator_stress` 中，默认关闭。安排测试窗口后才设为 `enabled: true`；必须完成原生 100 万次循环，超时或中断不算完成。逐次比较首次结果只证明重复一致性，不能代替独立数值精度验证。

```bash
# 模型性能：ResNet50 FP16/INT8、BERT-large FP16，各测试 batch=1、32。
cp configs/migraphx.template.yaml configs/bench.yaml
# 编辑 vars.image、模型路径、NUMA、BERT rocBLAS 调优库及容器 hook。
bash bench.sh check -c configs/bench.yaml --suite e2e_perf
bash bench.sh plan -c configs/bench.yaml --suite e2e_perf
bash bench.sh e2e_perf --set vars.image=YOUR_APPROVED_MIGRAPHX_IMAGE
```

两段示例应分别配置运行；第二次复制会替换 `configs/bench.yaml`。镜像可填写 tag 或 digest，`vars.image` 会传给启动 hook。MIGraphX 共 6 个组合，每组 3 轮、每轮 20 次，保留 Rate、Total time 和三轮中位数；ResNet50 仅 FP16 添加 `--opt-compile`。BERT 使用三个 INT64 `[batch,384]` 输入并填充为 1。本用例只测性能，不评估模型数值精度或 SQuAD 准确率。

## 常用命令与参数

```bash
bash bench.sh list                         # 查看 suite 和接入范围
bash bench.sh rccl
bash bench.sh pcie
bash bench.sh memory
bash bench.sh deepep
bash bench.sh operators                   # 当前 YAML 中启用的算子项
bash bench.sh e2e_acc                     # 仅精度
bash bench.sh e2e_perf                    # 当前 YAML 中的服务吞吐或 MIGraphX 固定模型
bash bench.sh e2e                         # 精度 + 性能；服务类任务共用模型服务
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
| PCIe | `cpu_devices`、`hcu_devices`、`size_mib`、`iters` |
| memory | `devices`、`presets`、`model_mhz`、`array_size` |
| DeepEP | `duration_s`（测试块）、`num_processes`、`num_tokens`、`hidden`、`num_topk`、`num_experts` |
| Mooncake | `topology`、`duration`、`buffer_size`、`threads`、`batch_size`、`block_size`、`operation` |
| operators | `recipe`、`native_timeout_s`；固定 shape / 循环次数不开放任意修改 |
| 服务类模型部署 | `models.<名称>.start_script`、`port`、`startup_timeout_s`；其余部署参数在启动脚本内 |
| e2e_acc | `models`、`num_examples`、`num_shots`、`num_threads`、`max_tokens`、`min_score` |
| EvalScope e2e_perf | `models`、`shapes`、`concurrency`、`requests`、`warmup_requests`、`repeats`、`seed`、`extra_body` |
| MIGraphX e2e_perf | `recipe`、`batch`（1 / 32）、`model`、`numa_node`、`rocblas_dir`、`round_timeout_s` |

参数通常写在测试块的 `params` 中；原生包字段以 `manifest.yaml` 为准。普通测试可用 `matrix` / 外层 `repeats`；引用 `models` 的测试不使用这两个字段，吞吐扫描使用 `params.shapes`、`params.concurrency`、`params.repeats`，见 [端到端测试](docs/e2e.md)。

DeepEP 的 `duration_s` 是计划窗口，必须小于故障时限 `timeout_s`。Mooncake 使用原生 `params.duration`；RCCL、PCIe、memory、固定算子和 e2e 按完整用例结束，不能设置框架 `duration_s`。memory 的 `model_mhz` 仅影响部分用例的带宽换算，**不会锁定硬件频率**。

## 其他运行方式

| 场景 | 模板 | 说明 |
| --- | --- | --- |
| 已有容器内运行 | [in-container.template.yaml](configs/in-container.template.yaml) | 最短使用路径，不需要宿主机 Docker 权限 |
| 宿主机自动创建容器 | [managed-container.template.yaml](configs/managed-container.template.yaml) | 由用户脚本提供启动、检查、停止接口 |
| 控制机通过 SSH 调度 | [remote.template.yaml](configs/remote.template.yaml) | 双节点原生 DeepEP 示例 |
| 仅 PCIe / 仅 memory | [pcie.template.yaml](configs/pcie.template.yaml) / [memory.template.yaml](configs/memory.template.yaml) | 无需填写模型和通信测试输入 |
| Mooncake 多机 | [mooncake.template.yaml](configs/mooncake.template.yaml) | 使用已准备的容器，共享目录须在所有节点可写 |
| 多模型精度 / 吞吐 | [e2e-multi-model.template.yaml](configs/e2e-multi-model.template.yaml) | 每个模型单独启动脚本，精度和吞吐串行共用服务 |
| 固定算子压测 / 性能 | [operators.template.yaml](configs/operators.template.yaml) | 单机单卡；3 项性能默认开启，3 项百万次压测默认关闭；镜像可选 |
| ResNet50 / BERT 性能 | [migraphx.template.yaml](configs/migraphx.template.yaml) | MIGraphX 原生推理，6 个组合，每组 3 轮；镜像可选 |
| 无 HCU 演示 | [demo.yaml](configs/demo.yaml) | 全部为模拟数据，不能作为性能结果 |

自动建容器模式的控制端需要 Python 3.10+、PyYAML 和 Docker。模板保留以下脚本接口，设备、权限、挂载和镜像由使用者明确配置：

```text
bash container.sh start NAME IMAGE HOST_WORKSPACE CONTAINER_WORKSPACE
bash container.sh check NAME
bash container.sh stop NAME
```

修改模板后同样执行 `bash bench.sh all`。工具按用例创建并清理自己的容器，不接管已有同名容器。也支持 `mode: existing` 复用明确指定的容器，不删除它；详细接口见 [容器运行说明](docs/quick-run.md)。

多节点使用已有 SSH 密钥、ssh-agent 或 SSH config，不保存密码。将这一个完整仓库部署或共享挂载到目标环境，原生工具在各节点自动准备；模型和运行库仍需在目标可见。每个测试独立配置 `nodes`，因此同一 YAML 可以让 RCCL/PCIe/memory/e2e 使用一个节点、DeepEP/Mooncake 使用多个节点；**不能仅给 e2e 增加节点就获得多机推理支持**。

多模型写在顶层 `models`，精度与吞吐测试各自用 `models: [model_a, model_b]` 引用。同一模型、节点、容器及环境配置会合并为一次服务会话；改变部署配置会分开启动。旧 `suite: e2e` / profile 配置仍按精度测试解释，不会自动加吞吐负载。完整示例见 [端到端测试](docs/e2e.md)。

文档和新 YAML 字段统一使用 HCU。旧 `gpus`、`gpu_devices`、`gpu_id` 仍有兼容别名，不能和对应新字段同时填写。原生环境变量、命令参数、源码及历史结果字段保留原名，例如 `GPU_MAX_HW_QUEUES`，不可机械改名。

## 结果与失败处理

每次运行生成独立的 `RUN_ID`，控制台打印输出位置。容器内模板默认写入 `vars.workspace/results/RUN_ID`；SSH 模式的统一报告保存在控制机的 `output_dir`。

| 文件或目录 | 内容 |
| --- | --- |
| `summary.txt` / `summary.json` | 执行状态和分组指标汇总 |
| `metrics.csv` | 逐条指标，含 suite、节点、参数、单位、正确性及模拟标记 |
| `e2e.csv` / `e2e.md` | 服务类模型流水线专用汇总，每个工作负载和轮次单独一行；保留失败及未出结果项 |
| `state.json` / `results.jsonl` | 用例状态 / 标准结果记录 |
| `logs/` | 各节点执行日志 |
| `raw/` / `artifacts.jsonl` | 收回的原生报告 / 归档状态 |
| 配置、命令和环境快照 | 本次运行的复测依据 |

日志使用执行环境本地时间及偏移。不同路径、参数、单位和原生统计口径分开汇总，Avg BW 不当成 Peak BW，DeepEP effective bandwidth 不当成 RCCL busbw。

固定算子和 MIGraphX 使用通用 `summary` / `metrics.csv` 汇总，原始日志与测量 JSON 归档到 `raw/`。MIGraphX 另保留 `rounds.json/csv`，指标统计口径为 `median_of_3_rounds`；如设置外层 `repeats`，框架汇总的是多组三轮中位数的均值。

- 普通失败默认清理后继续下一项；`failure_policy: stop` 可立即停止。清理无法确认时，无论策略如何都停止后续测试。
- 达到计划时长不等于正确性通过；超时、原生断言、空结果和异常退出不会被当作成功。
- PCIe、memory 默认只报告带宽；e2e_acc 的 `min_score: ""` 只报告分数。EvalScope 要求正式请求全部成功，MIGraphX 要求三轮全部完成且指标完整；两者均不把流程成功当作模型正确性通过，也不设通用吞吐门槛。
- 原生报告缺失或归档失败会单独警告，保留已采集结果；查看汇总时也要检查用例状态和归档状态。
- HCU 资源锁只协调本工具，不能代替业务占用检查。工具不改频率，不清理其他用户的进程或容器。

## 开发与测试包接入

需要新建控制端或开发环境时：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[native]'
python -m unittest discover -s tests -v
bash bench.sh configs/demo.yaml all
```

这不是待验镜像的自动安装步骤。模拟输出标记为 `SIMULATED`，单元测试不依赖 HCU，不能替代硬件验证。Bash 入口优先使用仓库 `.venv/bin/python`，也可通过 `BENCH_PYTHON=/path/to/python` 指定已有解释器。Windows 可作为控制端运行 `hcu-bench` CLI，原生 HCU 测试目标为 Linux。

接入新测试只需编写 manifest 和必要的包装脚本，声明参数、启动命令、指标及原生报告路径，不必修改调度框架。见 [测试包接口](docs/testpacks.md)。

模块说明：[RCCL](docs/rccl.md) · [PCIe](docs/pcie.md) · [显存带宽](docs/memory.md) · [DeepEP / 旧版 GSM8K 配置](docs/native-tests.md) · [固定算子](docs/operators.md) · [ResNet50 / BERT](docs/migraphx.md) · [端到端精度与吞吐](docs/e2e.md) · [Mooncake](docs/mooncake.md)。
