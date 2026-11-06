# HCU Bench

HCU Bench 是 Linux 下的命令行测试工具，统一执行通信、带宽、算子和模型测试。测试参数通过 YAML 配置，模型部署命令由各自的启动脚本管理。

```bash
bash bench.sh all           # 执行主配置中启用的测试
bash bench.sh rccl          # 仅执行 RCCL
bash bench.sh pcie,deepep   # 选择多个 suite
bash bench.sh e2e           # 模型精度 + LLM 性能
bash bench.sh training      # ResNet50 单卡训练吞吐短测
```

## 支持的测试

主配置适用于已有 SGLang 容器，包含以下项目。实际执行仍需满足各项目的环境要求；缺少依赖时记录失败原因，不自动替换运行库。

| Suite | 项目 | 收集结果 |
| --- | --- | --- |
| `rccl` | 单机10类通信 | 延迟、algbw、busbw、原生正确性检查 |
| `pcie` | CPU→HCU、HCU→CPU、HCU→HCU、CPU↔HCU、HCU↔HCU | 各路径平均/峰值带宽 |
| `operators` | Conv、hipDNN Conv、GEMM、PA decode、FA FP16/FP8前向性能 | 原生校验状态、耗时、计算性能或带宽；FP8接口不兼容时跳过 |
| `deepep` | 低延迟通信压测，默认10分钟 | 原生带宽、延迟、错误 |
| `e2e_acc` | Qwen3-4B/8B，GSM8K全量1319题，训练集5-shot，非思考 | 精度分数、回答及原生报告 |
| `e2e_perf` | Qwen3-4B/8B及GLM-5.2 INT8 TP8，多长度、多并发流式性能 | 输出吞吐、首token延迟等 |

`e2e` 为 `e2e_acc` 和 `e2e_perf` 的组合入口。每个模型按服务启动、精度评测、性能测试、服务清理的顺序执行，完成后切换至下一模型。

### 独立测例

独立配置保留原有 suite 名称，可单项复测。GLM-5.2、FA FP8、GEMM、hipDNN Conv已纳入主模板，复制新主模板后使用 `all` 即会执行；其余独立项目不自动加入主计划。已有个人YAML不会随模板更新自动增加测试块。

**显存带宽（`memory`）和 MIGraphX（`migraphx`）默认禁用（disabled）**，不随主配置的 `bash bench.sh all` 执行。主模板不包含这两个测试块；独立配置中的 `enabled: true` 仅在显式选择该配置时生效。SGLang 镜像不保证具备这两项所需的工具链和原生依赖。

| 配置文件 | Suite | 内容及前提 |
| --- | --- | --- |
| `configs/cases/memory.yaml` | `memory` | zd1/zd2共24项；需目标架构兼容的 hcu-stream/AICC |
| `configs/cases/migraphx-resnet50.yaml` | `migraphx` | ResNet50 FP16/INT8；需 MIGraphX driver 和本地 ONNX |
| `configs/cases/migraphx-bert.yaml` | `migraphx` | BERT-large FP16；需 MIGraphX driver、本地 ONNX、兼容的原生库 |
| `configs/cases/hipdnn-conv.yaml` | `operators` | 主模板已启用；DTK 原始 test_conv.sh 68 条命令；自带原始源码，复用已有 test_conv 或自动编译；需配套 DTK/hipDNN SDK，见[编译与运行指南](docs/hipdnn-conv.md) |
| `configs/cases/gemm.yaml` | `operators` | 主模板已启用；需匹配 rocblas-bench，默认原生性能计时，不代表正确性通过 |
| `configs/cases/fa-fp8.yaml` | `operators` | 主模板已启用；匹配19参数接口才测，不匹配记skipped |
| `configs/cases/operator-stress.yaml` | `operators` | BLAS FP16、FA FP16/BF16重复性压力；可100次短测或100万次正式压测 |
| `configs/cases/mooncake.yaml` | `mooncake` | 至少两节点、RDMA、共享可写目录 |
| `configs/cases/wan.yaml` | `e2e_perf` | Wan2.2文生视频；不使用LLM token吞吐口径 |
| `configs/resnet50-training.template.yaml` | `training` | ResNet50 ImageNet从头训练；需外部训练源码、真实数据及匹配的Torch/TorchVision/Apex |

每份独立配置仅包含该测例所需的公共信息和测试参数，测试块默认 `enabled: true`，通过指定配置文件执行。

```bash
bash bench.sh configs/cases/memory.yaml all
bash bench.sh configs/cases/migraphx-resnet50.yaml all
bash bench.sh configs/cases/wan.yaml all
```

默认 `bash bench.sh all` 仅读取 `configs/bench.yaml`，不扫描 `cases/`。新主模板包含27个执行用例：RCCL 10类、PCIe、Conv两种精度、GEMM五种精度、hipDNN Conv、PA、FA FP16/FP8、DeepEP、ResNet50训练吞吐、两个Qwen模型会话和GLM性能会话。模型会话分别包含其精度/性能子任务，不能把27个执行用例理解为27个底层shape或请求。

新增其他独立项目时，将对应测试块加入主配置的 `tests`，并补充所需的节点、模型或容器定义。迁移测试块时，须按新 YAML 所在目录调整 `pack` 相对路径。

主配置未包含 memory 测试块时，`bash bench.sh memory` 不会自动加载独立配置，须显式指定文件路径。配置选择沿用现有 CLI 规则。

### 显存带宽与 MIGraphX 使用方法

在对应的兼容容器中，修改下列独立 YAML 后，从仓库根目录执行。两项均使用文件内声明的路径和卡号，不继承个人 `configs/bench.yaml`。

**显存带宽**：编辑 `configs/cases/memory.yaml`，设置 `vars.bench_root`（容器内仓库路径）、`vars.workspace`（可写输出目录）、`vars.hcus`（目标卡）以及 `build.dtk_root`。`params.binary: auto` 使用已有兼容工具链编译内置 hcu-stream；已有二进制时可改为其绝对路径。`params.presets: both` 执行 zd1/zd2 共24项，多卡逐卡串行。

```bash
bash bench.sh configs/cases/memory.yaml all
```

**MIGraphX**：编辑 ResNet50 或 BERT 配置中的 `vars.bench_root`、`vars.workspace`、`vars.hcus`，并设置 `params.model`（本地 ONNX 文件）、`params.binary`（可用的 `migraphx-driver` 命令或绝对路径）、`params.numa_node`（目标卡对应的 NUMA 节点）。执行环境须提前提供兼容的 MIGraphX 原生程序和动态库，以及 numactl、onnx、psutil；本工具不通过安装 driver 来补齐运行库。

```bash
# ResNet50：FP16/INT8，batch 1/32
bash bench.sh configs/cases/migraphx-resnet50.yaml all

# BERT-large：FP16，batch 1/32
bash bench.sh configs/cases/migraphx-bert.yaml all
```

BERT 的 `params.rocblas_dir: ""` 使用镜像原生库；仅在需要外置兼容库时指定包含 `lib/` 的目录。MIGraphX 每组参数执行三轮，报告吞吐及耗时中位数，不包含模型精度评测。

结果分别写入所选 YAML 的 `${vars.workspace}/results/RUN_ID`，状态与原因查看 `summary.txt`，性能指标查看 `metrics.csv`。详细环境要求见[显存带宽说明](docs/memory.md)和[MIGraphX说明](docs/migraphx.md)。

### ResNet50 从头训练

`training` 是独立suite，不属于MIGraphX推理或LLM的 `e2e`。新主模板默认 `enabled: true`，
加入 `all`；填写 `vars.train_script` 和 `vars.imagenet` 并使用匹配的Torch/TorchVision/Apex环境。
也可单独使用 [ResNet50训练模板](configs/resnet50-training.template.yaml)，填写只读训练源码、
本地ImageNet、可写输出目录及唯一获分配的HCU后执行：

```bash
bash bench.sh configs/resnet50-training.template.yaml training
```

默认 `mode: throughput`、FP32、batch32、workers4、预热10步+计时50步，随机初始化，
不下载预训练权重。单卡真实图像训练吞吐以同步墙钟窗口报告images/s，含加载/增强、前向、
反向及SGD，不含初始化、预热、验证和存盘；仍保留有限loss、checkpoint及重载检查。
各取少量真实图片覆盖1000类，不跑90轮；单次原生执行超时300秒、整体900秒上限，不要求跑满。
完整训练需显式设置 `mode: full`、轮数及长超时；小子集分数不能称为全量精度或收敛验收。
训练源码由用户提供并记录SHA256，不自动替换Apex或框架，详见[训练说明](docs/training.md)。

## 快速开始

### 1. 获取代码与检查依赖

在已准备的 Linux 测试容器中执行：

```bash
git clone https://github.com/dddddddxl/hcu-bench.git
cd hcu-bench
python3 -c 'import sys, yaml, psutil; assert sys.version_info >= (3, 10); print("OK")'
```

运行入口需要 Python 3.10+、PyYAML、psutil。依赖已满足时可直接运行，无需额外创建虚拟环境或执行 `pip install -e`。

仓库提供测试源码和包装脚本，RCCL/PCIe/memory/Mooncake 首次运行时按需编译。驱动、DTK、模型框架、原生运行库和权重由镜像及挂载目录提供，不自动安装或升级。LLM 性能测试首次在独立缓存环境准备 EvalScope 客户端依赖，需要可用包源或 wheelhouse；此过程不升级模型服务环境。

### 2. 配置测试方案

复制主模板为个人配置：

```bash
cp configs/bench.template.yaml configs/bench.yaml
```

主模板见 [bench.template.yaml](configs/bench.template.yaml)，独立测例见 [configs/cases](configs/cases/README.md)。原有 `in-container.template.yaml`、managed/SSH及各专用模板保留兼容；已有个人配置不会自动迁移或覆盖。

主配置按以下顺序组织：

1. **通用配置**：仓库目录、结果目录、HCU 分配和编译环境。
2. **模型部署**：Qwen3-4B/8B及GLM-5.2权重、启动脚本、端口，仅 e2e 使用。
3. **Suite 配置**：RCCL、PCIe、算子、DeepEP、模型精度和模型性能，各项目参数分别配置。

仅执行 RCCL 时，无需配置模型权重、PCIe 设备编号或 DeepEP 网卡。未执行的测试块可设为 `enabled: false`；使用 `all` 时，所有启用项目均须配置完整。`YOUR_*` 为待替换的占位符，不具有自动发现功能。

所有运行路径均使用**当前执行容器内的绝对路径**，并确保对应目录已挂载且可访问。

### 3. 检查与执行

```bash
bash bench.sh check -c configs/bench.yaml --suite all
bash bench.sh plan -c configs/bench.yaml --suite all
bash bench.sh all
```

`check` 检查配置，`plan` 展示执行计划；两者均不占用 HCU 或启动模型，也不验证原生依赖的完整可用性。测试前应通过 `hy-smi` 和 `hy-smi --showpids` 核对资源占用，瞬时利用率为0不能单独作为空闲依据。

## 通用配置

| 配置项 | 说明与示例 | 调整条件 |
| --- | --- | --- |
| `name` | 本轮方案名称，例如 `nmz6-bench` | 便于区分报告，可自行命名 |
| `vars.bench_root` | 仓库绝对路径，在仓库根目录执行 `pwd` 获取 | 仓库路径或容器挂载变化时修改 |
| `vars.workspace` | 可写输出目录，建议持久挂载，例如 `/workspace/bench-output` | 换平台时核对，用于保存结果和缓存 |
| `vars.hcus` | 多卡测试允许使用的编号列表，例如 `[0,1,2,3]` | 必须核对实际可见设备和业务占用 |
| `vars.single_hcus` | 单卡测试所用编号，例如 `[0]` | 仅分配一张卡，且属于 `vars.hcus` 范围 |
| `vars.iface` | 平台实际socket网卡，例如 `ens65f0np0`；通过 `ip -br addr` 核对 | DeepEP和GLM分别引用，换平台只改这一处；不是RDMA HCA名或GID |
| `failure_policy` | `continue` 或 `stop` | 普通失败清理后继续，或立即停止 |
| `build.dtk_root` | 实际DTK安装根目录 | 核对 `/opt/dtk` 是否存在且对应当前环境 |
| `build.mpi_home` | Open MPI安装根目录 | RCCL需要；可从 `command -v mpirun` 初步定位，再核对实际安装目录 |
| `build.arch` | `auto` 或实际架构名，如 `gfx938` | 存在多架构或无法唯一识别时显式指定；可用 `rocminfo` 核对 |
| `build.cache_dir` | 可写原生构建/客户端缓存目录 | 默认引用 workspace，须具有写权限 |
| `build.jobs` | 首次编译并行任务数，默认2 | 用于调整编译CPU负载，与卡数无关 |
| `build.timeout_s` | 编译准备超时，默认3600秒 | 首次编译耗时较长时调整 |
| `output_dir` | 统一结果目录 | 默认 `${vars.workspace}/results`，通常无需调整 |
| `nodes.*.executor` | 已有容器内执行用 `local` | SSH控制机调度时另用远程模板 |
| `nodes.*.hcus` | 引用分配的卡列表 | 通常不改引用；不是PCIe工具的Device ID |
| `HIP_VISIBLE_DEVICES` | 保留 `${hcu_ids}` | 由框架根据该用例的卡分配展开 |

本模板的卡号以当前容器执行环境为准。若容器仅透传部分物理卡，应通过PCI地址或UUID核对与宿主机设备的映射关系。调整卡数时，须同步修改 RCCL `ranks` 和 DeepEP `num_processes`。

启用GLM-5.2时，`vars.hcus` 必须分配8张空闲卡；不足8卡时应将 `tests.glm52_throughput.enabled` 改为false，不自动降低TP。`vars.iface` 只被需要的测试块/模型引用，不会把DeepEP的IB策略全局应用到模型服务。

### 测试块字段

| 字段 | 说明 |
| --- | --- |
| 测试名，如 `pcie_bandwidth` | 日志和命令覆盖使用的ID；不是suite名 |
| `enabled` | 是否纳入执行；即使指定suite，false项也不运行 |
| `suite` | 命令行选择名，通常不改 |
| `adapter: command` / `pack` | 真实测试包及其manifest，通常不改 |
| `nodes` | 选择前面声明的节点/卡组；单卡用single，多卡用multi |
| `timeout_s` | 测试故障时限，单位秒；测试完成后立即结束，不要求持续至超时 |
| 外层 `repeats` | 整个测试块的独立重复次数，适用于普通测试 |
| `params` | 本测试的原生参数 |
| `matrix` | 将列出的参数组合展开成多个用例 |
| `env` | 只给当前测试设置的环境变量，值写字符串 |

`pack` 相对于 YAML 所在目录解析：`configs/bench.yaml` 使用 `../testpacks/...`，`configs/cases/*.yaml` 使用 `../../testpacks/...`。同一文件中的顶层 `tests:` 和 `models:` 均不得重复定义。

## Suite 参数说明

### RCCL

配置位置：`tests.rccl_collectives`。默认执行10类通信，每类使用固定 **16 GiB** 消息、half、20次预热和50次测量。YAML 省略消息大小参数，继承仓库默认值。

| 参数 | 取值与说明 |
| --- | --- |
| `binary` | 默认 `auto`，按collective编译对应工具；显式指定的程序须与collective匹配 |
| `ranks` | 等于分配HCU数，一进程一卡；当前2–8卡 |
| `datatype` | `half` / `float` / `double`，对比时保持一致 |
| `warmup_iters` / `iters` | 预热和正式迭代次数，不是秒数 |
| `matrix.collective` | 待执行的通信项列表，可按测试范围删减 |

`min_bytes` / `max_bytes` 仅保留注释。已有个人配置若指定了64 MiB或1 GiB，须移除这两个覆盖值才能继承16 GiB。显存不足时保留原始错误，不自动降低消息大小或将降级结果判为通过。

该suite目前用于单机通信，不代表网卡带宽测试。通信环境变量在本测试块的 `env` 中配置，与DeepEP的网卡设置分别管理。

### PCIe

配置位置：`tests.pcie_bandwidth.params`。换平台时须重新枚举并核对 `cpu_devices`、`hcu_devices`，其余参数按测试口径调整。

```bash
/实际工具路径/rocm-bandwidth-test -e
/实际工具路径/rocm-bandwidth-test -t
hy-smi
```

设备枚举须使用本轮实际运行的工具版本，并在同一容器内通过PCI地址核对映射关系。工具缺失时可使用仓库内置源码的准备入口构建，无需另行获取源码；不同工具版本的设备编号不能直接复用。

| 参数 | 取值与说明 |
| --- | --- |
| `binary` | `auto`编译内置工具，或已核对接口的程序绝对路径 |
| `cpu_devices` | 例如 `"0,1"`，是工具枚举的CPU内存代理，不是CPU核心号 |
| `hcu_devices` | 例如 `"2,3,4,5"`，是同一工具的HCU Device ID，不是卡号 |
| `size_mib` | 默认512，单次拷贝512 MiB；不是整轮只传512 MiB |
| `iters` | 默认20，原生内部迭代次数 |
| 外层 `repeats` | 默认1，整套路径复测次数 |

上述编号仅用于演示格式，实际取值以目标环境的枚举结果为准。五类路径完整扫描至少需要两张HCU；平均值和峰值分别保存，双向带宽已包含两个方向之和，不再乘2。

### operators：Conv、hipDNN、GEMM、PA、FA

| 测试块 | 参数 | 取值与说明 |
| --- | --- | --- |
| `conv` | `mode` | `check`只校验；`perf`先校验再测性能 |
| `conv` | `driver` | `auto`查找镜像的MIOpenDriver；不自动安装原生库 |
| `conv` | `conv_suite` | `all` / `conv2d` / `conv3d` |
| `conv` | `direction` | `all` / `fwd` / `dgrad` / `wgrad` |
| `conv` | `matrix.dtype` | 默认 `[fp32, fp16]` |
| `conv` | `warmup` / `iters` / `params.repeats` | 每用例预热、正式迭代、性能轮数，默认10/100/3 |
| `conv` | `native_timeout_s` | 原生执行时限；外层timeout另留清理余量 |
| `pa_decode` | `recipe` | 固定 `pa_decode`；形状由当前固定配方定义 |
| `fa_fp16_forward` | `recipe` | 固定 `fa_fp16_forward` |
| `fa_fp8_forward` | `recipe` | 固定 `fa_fp8_forward`；预检实际扩展，不兼容则跳过且无性能指标 |
| `gemm` | `gemm_profile` / `rocblas_bench` | 默认new，对应rocBLAS 5.2.0接口；legacy对应4.3.0，工具须已存在 |
| `gemm` | `mode` / `max_norm_error` | 默认perf、空阈值：原生计时、正确性not_checked；check或非空阈值显式启用校验 |
| `gemm` | `group` / `case_id` / `matrix.dtype` | 默认all、空ID、五种原始精度，共3107条；可用smoke或原始ID筛选 |
| `hipdnn_conv` | `binary` / `source_root` | 默认优先复用已有test_conv，未安装时自动编译随包原始源码；可显式指定程序或外部源码目录 |
| `hipdnn_conv` | `native_timeout_s` | 默认14400秒，外层14520秒；68条原始调用，不改原生校验或迭代 |
| PA/FA | `native_timeout_s` | 默认1800秒；不是迭代次数 |

PA/FA使用固定配方，暂不支持任意形状配置。FP8已加入主模板；不兼容接口的skipped不计作性能完成或正确性通过。依赖缺失、内核失败、超时仍保留实际状态，不自动安装原生库。hipDNN默认使用随包原始源码编译；镜像需具备匹配的DTK/hipDNN SDK。

百万次重复性压力仍为独立测例，不随默认all执行：BLAS FP16、FA FP16、FA BF16。`configs/cases/operator-stress.yaml` 默认 `stress_iterations: 100`、`native_timeout_s: 1800`、外层 `timeout_s: 1920`；100次仅验证流程。正式长压将次数改为1000000，并按独占窗口同时增大两个超时。该次数不影响固定FA性能配方，也不是DeepEP的600秒压测。

### DeepEP

配置位置：`tests.deepep_pressure`。负载参数通过 `params` 配置，通信环境变量通过本测试块的 `env` 配置。

| 参数 | 取值与说明 |
| --- | --- |
| `duration_s` | 默认600秒，计划压测窗口，含初始化 |
| `timeout_s` | 故障时限，必须大于duration_s |
| `test_script` | `auto`使用内置测试，不代表安装或升级DeepEP |
| `num_processes` | 每节点使用的进程数，等于分配HCU数 |
| `master_addr` | 单机 `127.0.0.1`；多机填首节点可达地址，并配置SSH节点 |
| `master_port` | 未占用端口，示例29450 |
| `num_tokens` | 每卡token数，默认128 |
| `hidden` | hidden维度，默认7168 |
| `num_topk` | 每token选择的expert数，默认8 |
| `num_experts` | expert总数，默认256；须满足原生实现的整除条件 |
| `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` | 实际可用socket网卡，两项通常填同一接口 |
| `ROCSHMEM_IB_GID_INDEX` | 实际RDMA设备、端口所用的GID index，写字符串 |

使用以下命令核对网络设备与GID信息：

```bash
ip -br addr
ibv_devices
ibdev2netdev
show_gids
```

`ibdev2netdev`、`show_gids` 并非所有镜像均已提供；缺失时应通过宿主机或sysfs核对。socket网卡可能用于管理网络，RDMA设备可能对应其他网络，两者名称不必相同。GID index须与目标RDMA设备、端口、地址类型及原生运行库要求匹配，不预设为0或3。多HCA选择须按目标ROCSHMEM接口确认，`NCCL_IB_HCA` 并非所有运行库通用的设备选择参数。

`ROCSHMEM_HEAP_SIZE`、`HIP_BUFFER_EXTRA_SIZE`、`NCCL_IB_DISABLE` 默认保留已确认的单机配方；其中 `NCCL_IB_DISABLE` 仅作用于NCCL，不表示关闭ROCSHMEM的RDMA。

内置DeepEP测试启动前生成 `api_contract.json`，识别已支持的11/12字段结构；worker复核接口指纹，原生正确性短测通过后才进入压力阶段。未知或歧义接口明确拒绝，不保证任意未来版本均可适配，也不会绕过原生正确性检查。

### e2e：模型启动脚本

模型部署在顶层 `models` 中定义，由精度和性能测试共同引用，无需重复配置部署命令。

| 参数 | 取值与说明 |
| --- | --- |
| 模型名，如 `qwen3_4b` | 自定义ID，用于选模型和报告 |
| `start_script` | 启动脚本路径；相对路径以执行环境的bench_root为基准 |
| `env.BENCH_MODEL_PATH` | 默认4B/8B脚本读取的权重路径；自有脚本按自身约定读取 |
| `tokenizer_path` | 配套本地tokenizer目录；模板显式配置，避免服务返回的模型名无法解析为本地路径 |
| `port` | 空闲服务端口；模型串行可复用 |
| `startup_timeout_s` | 等服务就绪的最长秒数，默认1800 |

默认4B/8B脚本采用通用BF16、TP1配置。新增模型可在自有脚本中管理权重、量化、TP、网络和后端参数，YAML仅声明 `start_script`；客户端无法通过服务获取可用tokenizer时，另行配置 `tokenizer_path`。

```yaml
models:
  new_model:
    start_script: model_scripts/new_model.local.sh
    # tokenizer_path: /models/new-model
```

启动脚本须前台运行，使用 `exec` 启动服务，并读取 `$BENCH_HOST`、`$BENCH_PORT`，不得使用 `nohup` 或 `&` 后台化。上述模型定义应加入现有 `models`，不得重复定义顶层键。

### e2e_acc：模型精度

| 参数 | 默认值与说明 |
| --- | --- |
| `models` | 主模板为 `[qwen3_4b, qwen3_8b]`，不包含GLM；也可只选一个。`all`会包含所有已声明模型，须明确是否适用于精度 |
| `data_path` | `auto`明确指仓库 `vendor/gsm8k/test.jsonl` |
| `few_shot_path` | `auto`明确指 `vendor/gsm8k/train-5shot.jsonl`，不从测试集扣除5题 |
| `num_examples` | 0=全量1319题；功能短测可填100 |
| `num_shots` | 5，使用训练示例 |
| `num_threads` | 128，客户端线程数；不是HCU数或TP |
| `max_tokens` | 2048，每题输出上限，会影响结果可比性 |
| `chat_template_kwargs.enable_thinking` | false；是否生效也取决于模型模板和服务支持 |
| `min_score` | 空字符串只报告分数；有验收标准再填 `"0.90"` 等比例 |
| `timeout_s` | 每模型精度任务的故障时限，默认14400秒 |

历史vLLM结果不直接作为当前SGLang的通过阈值。自定义JSONL须包含 `question`、`answer` 字符串，训练提示与评分数据应分开提供。

### e2e_perf：LLM吞吐

| 参数 | 默认值与说明 |
| --- | --- |
| `models` | 同精度，可单独指定模型列表 |
| `python` | `auto`准备隔离EvalScope客户端；也可填已准备的客户端解释器路径 |
| `workloads[].input_tokens` | 目标输入token长度 |
| `workloads[].output_tokens` | 目标输出token长度；保留实际长度，不假定每次等长 |
| `workloads[].concurrency` | 此长度组合的并发档位列表 |
| `requests` | 每个点256正式请求，必须不小于最大并发 |
| `warmup_requests` | 每点32预热，不计入正式性能 |
| `params.repeats` | 每个点重复1轮，不是模型服务启动次数 |
| `seed` | 42，随机数据种子 |
| `request_timeout_s` | 每条请求最长1800秒 |
| `extra_body` | 附加API字段，默认关闭thinking；不可覆盖框架管理的模型、提示和token上限字段 |
| `timeout_s` | 每模型性能任务最多172800秒，即48小时，不是必须运行48小时 |

默认采用月英参考表的4组长度及并发组合，共31个测试点。每模型包含7936次正式请求和992次预热请求，完整两模型测试预计耗时较长。功能验证可将 `workloads` 缩减为单个测试点，并减少正式及预热请求数；短测结果不能代替完整扫描。

### e2e_perf：GLM-5.2 INT8 八卡

主模板的 `models.glm52_int8` 和 `tests.glm52_throughput` 已启用，`all` / `e2e_perf` 会在Qwen之后运行GLM；`e2e_acc` 仍只测Qwen。GLM使用 `multi` 节点的8张卡，不能沿用Qwen的单卡分配。

修改GLM的 `BENCH_MODEL_PATH` 和 `tokenizer_path` 为本地权重目录。启动脚本 `model_scripts/glm52_int8_tp8.sh` 采用官方BW1100 / SGLang0.5.12的INT8、TP8、EAGLE配方；其他平台/镜像须先验证后使用，不自动安装依赖或下载权重。

网卡在 `vars.iface` 配置一次，模型的 `env.NCCL_SOCKET_IFNAME` / `env.GLOO_SOCKET_IFNAME` 分别引用，脚本只检查和继承，不覆盖为某台机器的固定名称。官方TP8配方不额外要求DeepEP的HCA/GID配置；如平台需指定其他通信环境变量，可在该模型的 `env` 中设置。

GLM默认保留模型自身的思考模式，`extra_body: {}`，不继承Qwen的 `enable_thinking: false`。报告记录实际输入/输出token长度；不同思考模式的性能不可直接对比。其独立性能块也采用31点扫描；官方服务的 `max-running-requests=32`，客户端并发64/128将产生排队，需连同延迟阅读。

只验证GLM时，可修改[独立性能模板](configs/glm52-perf.template.yaml)后执行：

```bash
bash bench.sh configs/glm52-perf.template.yaml all
```

小规模验证可把该模板的 `workloads` 改成128/128与1024/128、并发1/8，`requests` 改32、`warmup_requests` 改4、`repeats` 保持1；仅4个正式性能点，不作为完整31点验收。详见[GLM-5.2说明](docs/glm52.md)。

## 独立测例配置

独立测例沿用主配置的通用字段，仅保留各自所需的输入，无需配置无关的LLM模型或DeepEP网卡。

| 测例 | 必须核对 | 主要可调参数 |
| --- | --- | --- |
| memory | 目标卡、兼容AICC/DTK或已编译hcu-stream | `binary`、`devices`、`presets`（zd1/zd2/both）、`model_mhz`、`array_size`、轮数 |
| MIGraphX ResNet50 | 单卡、原生driver、ONNX、NUMA节点 | `model`、`numa_node`、`matrix.recipe`（fp16/int8）、`matrix.batch`（1/32）、`round_timeout_s` |
| MIGraphX BERT | 单卡、原生driver、BERT ONNX、原生库兼容 | `model`、`numa_node`、`rocblas_dir`、`matrix.batch`（1/32）、`round_timeout_s` |
| GEMM | rocblas-bench接口与原生unit_check | `gemm_profile`（new/legacy）、`rocblas_bench`、`group`、`case_id`、`matrix.dtype`、`mode`、`max_norm_error`、预热/迭代/轮次 |
| FA FP8 | 匹配的Torch/FA FP8扩展 | 固定 `recipe: fa_fp8_forward`，`native_timeout_s` |
| 算子压力 | 单卡空闲窗口、BLAS/FA依赖 | `stress_iterations`=100短测或1000000正式；3个recipe，原生及外层超时 |
| Mooncake | 至少两节点、RDMA、双方可写且内容相同的共享目录 | `nodes`、`addresses`、`shared_dir`、`topology`、`duration`、`hcu_id`、`min_bandwidth_gbps` |
| Wan | 本地Diffusers权重、可用扩散服务、1或4卡 | `start_script`、`model_path`、`profile`、`request`、预热/正式请求数和各层超时 |

- memory的 `devices: ""` 表示使用已分配的全部卡，逐卡串行；非空字符串可填其子集。`model_mhz` 是原脚本换算参数，不锁频；`array_size` 是元素数，不是字节。
- MIGraphX的 `numa_node` 须按目标卡PCI拓扑对应的实际NUMA节点配置，不预设为0。BERT的 `rocblas_dir: ""` 使用镜像原生库，仅在明确要求外置兼容库时指定目录。每个batch组合内部执行三轮并报告中位数，不包含模型精度评测。
- GEMM的 `new` 对应当前5.2.0接口，`legacy` 对应4.3.0接口，实际版本应通过原生工具确认。默认perf沿用原生benchmark计时，不追加正确性检查，结果为 `not_checked`；check启用原生比较，非空 `max_norm_error` 追加norm检查。原生unit失败仍拒收，不因进程exit 0判为通过。默认仅DTK原始general + GEMM smoke，共3,107条；五种精度及筛选见[算子说明](docs/operators.md)。
- 独立压力测例默认100次，标记为smoke，不属于100万次压力验收。正式压力将 `stress_iterations` 改为1000000，并同步增大原生及外层超时；超时或中断不判为完成。
- Mooncake的 `addresses` 须按 `nodes` 的顺序配置。`duration` 为原生测试时长，单位秒，不使用框架 `duration_s`；带宽阈值为空字符串时仅报告结果。该测例至少需要两台节点。
- Wan的 `request` 包含prompt、size、num_frames、fps、num_inference_steps、seed、guidance_scale；初版采用1280×720、81帧、16 FPS、40步，1次预热和1次正式生成。TP4已完成实机流程验证，TP1尚未实测；视频质量未评估，不使用LLM吞吐或精度指标。

## 结果与日志

每轮启动时通过 `Run directory` 输出结果目录，默认路径为 `${vars.workspace}/results/RUN_ID`。

| 文件 | 用途 |
| --- | --- |
| `summary.txt` / `summary.md` | 按suite展示执行概览、关键指标、正确性、失败/取消原因及日志入口 |
| `summary.json` | 保留原有机器可读的分组指标汇总 |
| `report-details/suite-*.csv` | 对应suite的全部可展示指标，含参数、节点、轮次和原生统计口径 |
| `metrics.csv` | 所有suite的逐条标准指标 |
| `e2e.md` / `e2e.csv` | 各模型精度和性能；Wan按视频指标单列 |
| `state.json` / `results.jsonl` | 原始执行状态 / 标准结果记录 |
| `logs/测试ID/节点.log` | 分测试、分节点日志；模型会话按其生成ID保存 |
| `raw/` / `artifacts.jsonl` | 原生报告及单独的归档状态 |

### 结果阅读顺序

1. 先查看 `summary.md`（表格）或 `summary.txt`（纯文本），确认整体状态和各suite是否实际执行、是否完成，以及是否发生取消或归档警告。
2. 在对应suite章节查看关键指标；形状、rank或测试点较多时，摘要展示前12条，全部指标通过该节的suite CSV链接查看。模型会话按精度/性能子任务展示，但执行总数按父会话计一次；子任务无结果不视为完成。模型原始分组另见 `e2e.md`，所有逐条指标见 `metrics.csv`。
3. 只有需要定位失败、核对统计来源或分析异常时，再打开对应测试ID的日志及 `raw/` 原生报告。

| 测试 | 优先关注的指标 | 解释 |
| --- | --- | --- |
| RCCL | `algbw`、`busbw`、latency、原生wrong | GB/s、GB/s、微秒；algbw与busbw分别比较，不能互换 |
| PCIe | `avg_gbps`、`peak_gbps`及源/目标路径 | 原生CSV字段名虽为gbps，单位为GB/s；双向数值已含两个方向 |
| memory | 配方名称与带宽 | 单位GB/s，显存、缓存和LDS用例分别比较 |
| DeepEP | effective bandwidth、dispatch/combine带宽与延迟 | 按指标名称、rank和参数区分；计划到时不等于完整稳定性通过 |
| Conv/GEMM | 原生校验、norm_error、latency及计算性能 | 正确性未确认时，check日志中的速度不能作为正式性能结果 |
| PA/FA | 带宽或TFLOPS；压力任务的迭代数和完成状态 | 固定性能配方与重复性压力是不同测试 |
| MIGraphX | 三轮吞吐中位数、ms/batch | 仅代表固定模型性能，不表示模型精度已验证 |
| 模型精度 | GSM8K分数、评分题数、shot数、输出上限 | 比较时必须统一数据集与评测口径 |
| LLM性能 | 输出tokens/s、TTFT、TPOT、成功请求数 | 吞吐通常越高越好、延迟通常越低越好；只比较同长度/并发/轮次 |
| Wan | Pixel、Denoise、单步、Decode、API耗时 | 单位秒，计时边界不同，不能用LLM tokens/s解释 |

`measurement_status` 表示是否获得有效测量，`correctness` 表示数值正确性结论，两者分别判断。`not_checked` 不等于失败，也不等于正确性通过。不同卡型、镜像、消息大小、模型、精度或并发的结果不直接合并为一个平均值。

RCCL的消息大小、原地/非原地分开展示；PCIe只在同一配置、节点和轮次下展示不同路径的Avg/Peak范围，不计算跨路径均值。LLM吞吐与Wan耗时分别列表，百分位数不跨轮平均。模拟数据保留显著标记，不作为HCU实测结果。

错误原因仅来自结构化结果和执行状态。若只有非零退出码，报告会注明原因未确认并提供对应日志，不自动推断OOM或库版本问题；独占状态、外部CI归属等仍须结合资源监控记录判断。归档状态独立列出，不覆盖已采性能。

已有结果可重新生成报告，无需重跑测试：

```bash
bash bench.sh report /实际结果目录/RUN_ID
```

重新生成只读取该轮已有结果，更新报告及指标导出，不修改配置、测试状态、原生结果或日志。`summary.json` 和 `metrics.csv` 保持原有格式；新版展示生成失败时明确告警，并回退保留逐条指标摘要。完整说明见[结果报告](docs/reports.md)。

MIGraphX使用通用suite报告，不写入服务类模型的 `e2e.csv`。原始 `rounds.json/csv` 保留三轮结果，指标口径为 `median_of_3_rounds`；外层 `repeats` 分别展示，原有 `summary.json` 的分组统计契约不变。

普通失败默认在清理后继续执行；无法确认清理完成时，停止剩余计划。原生报告缺失或归档失败单独告警，不覆盖已采集的性能结果。`completed` 不代表正确性验证通过，短测不作为完整验收依据。

资源锁仅协调HCU Bench自身，不阻止CI或其他用户使用HCU，也不自动终止外部任务。

## 常用命令

```bash
bash bench.sh e2e_acc
bash bench.sh e2e_perf
bash bench.sh deepep --set tests.deepep_pressure.duration_s=1200
bash bench.sh pcie --set tests.pcie_bandwidth.repeats=3
bash bench.sh all --detach
bash bench.sh status /实际输出目录/RUN_ID
bash bench.sh stop /实际输出目录/RUN_ID
```

`--set` 的测试ID必须与当前配置相同。增大DeepEP计划时长时，外层 `timeout_s` 仍需更大；命令行覆盖不修改原YAML。

宿主机自动创建容器和SSH多节点调度使用对应独立模板及start/check/stop脚本接口，与本文的容器内执行方式分开配置。容器权限和挂载由启动脚本明确声明，配置文件不保存密码。每个测试可独立选择节点；当前e2e模型服务仅支持单节点，增加节点名称不能直接实现多机推理。

个人 `configs/bench.yaml` 不随模板更新自动覆盖。测试源码和内置数据随仓库提供，无需从其他仓库另行获取测试脚本；模型权重及环境依赖仍按各测例要求准备。

详细说明：[配置结构](docs/configuration.md) · [容器与远程执行](docs/quick-run.md) · [RCCL](docs/rccl.md) · [PCIe](docs/pcie.md) · [显存带宽](docs/memory.md) · [算子](docs/operators.md) · [MIGraphX](docs/migraphx.md) · [端到端](docs/e2e.md) · [Mooncake](docs/mooncake.md) · [Wan](docs/wan.md)。
