# YAML 配置说明

所有 suite 可以放在同一 YAML 中。公共路径和卡号放在 `vars`；执行位置放在 `nodes`；每个测试的参数放在 `tests.<测试名>.params`。

## 配置结构

| 字段 | 用途 |
| --- | --- |
| `version: 1` | 配置格式版本 |
| `name` | 测试方案名称 |
| `vars` | 可复用的值，通过 `${vars.名称}` 引用 |
| `build` | 首次原生编译配置：DTK/MPI 路径、架构、缓存与编译时限 |
| `output_dir` | 控制端报告目录；相对路径以 YAML 所在目录为基准 |
| `nodes` | 本地或 SSH 节点，声明卡号、工作目录和环境变量 |
| `containers` | 可选；复用已有容器或调用用户脚本创建容器 |
| `tests` | 按顺序执行的测试块，同一个 suite 可以有多个测试块 |

以一个显存带宽测试块为例，下面是要放到 `tests` 下的配置片段：

```yaml
tests:
  memory_bandwidth:                     # 测试名，用于报告和 --set，不是 suite 名。
    suite: memory
    adapter: command                   # 真实测试包；mock 只用于模拟。
    pack: ../testpacks/memory/manifest.yaml
    nodes: [node0]
    enabled: true                      # false 时不会被 all 或单项选择运行。
    timeout_s: 1800
    repeats: 1
    params:
      bench_root: "${vars.bench_root}"
      binary: "${vars.memory_binary}"
      devices: "0"
      presets: both
      model_mhz: 1400
      array_size: 104857600
```

`pack` 相对于 YAML 所在目录解析；本仓库模板应保留在 `configs/`。`bench_root`、原生工具、模型和数据路径则属于实际执行环境：直接运行时是当前容器或主机，SSH/Docker 模式是目标节点或目标容器。框架不会自动把这些文件复制过去。

`${vars.hcus}` 作为整个字段使用时保留列表类型，嵌入路径时则进行字符串替换。`${hcu_ids}`、`${node_rank}`、`${nnodes}` 等由执行阶段展开。配置不会执行 shell 表达式，也不会将 `$HOME` 自动展开成路径。

环境变量值使用字符串，例如 `ROCSHMEM_IB_GID_INDEX: "0"`。不要把 SSH 密码或访问令牌写入配置。

## 参数修改与扫描

长期配置直接改 YAML；临时参数用 `--set`，不修改原文件：

```bash
bash bench.sh deepep --set tests.deepep_pressure.params.num_tokens=256
bash bench.sh pcie --set tests.pcie_bandwidth.repeats=3
bash bench.sh e2e --set 'tests.sglang_gsm8k.params.min_score="0.90"'
```

命令中的测试名要与当前 YAML 对应。`min_score` 是字符串；命令行覆盖时保留内层引号。同理，字符串类型的设备参数应写成 `--set 'tests.memory_bandwidth.params.devices="0,1"'`。

需要扫描多组参数时，在对应测试块下增加：

```yaml
matrix:
  num_tokens: [128, 256, 512]
repeats: 3
```

这会生成 3 个 token 配置，每个运行 3 轮。多个 matrix 字段按笛卡尔积组合；每轮独立归档。不要将整个算子形状数据集展开成百万项 matrix，数据文件路径应交给原生测试包处理。

`timeout_s` 是正式测试的故障时限，不是测量迭代次数。首次编译另用 `build.timeout_s`，不扣减测试窗口。DeepEP 可使用测试块的 `duration_s`；Mooncake 使用 `params.duration`。RCCL、PCIe、memory 和 e2e 必须执行完各自的有限用例，不接受框架 `duration_s`。

## 自动准备

RCCL、PCIe、memory、Mooncake 的 `params.binary: auto` 从仓库源码构建并缓存；填具体路径则复用已有程序。DeepEP 的 `test_script: auto` 使用内置测试，e2e 的 `data_path`、`profile`、`ep_config` 均可保持 `auto`。

```yaml
build:
  dtk_root: /opt/dtk
  mpi_home: /opt/mpi
  arch: auto                # 无法唯一识别时显式写 gfx92a / gfx938 等。
  # arch_num: 9210          # hcu-stream 的数字宏；gfx92a 默认映射为 9210。
  # compiler: /opt/dtk/bin/aicc  # 可选覆盖；通常保留各 suite 默认编译器。
  cache_dir: "${vars.workspace}/native-cache"
  jobs: 2
  timeout_s: 3600
```

默认 RCCL 使用 hipcc，memory 使用 aicc，PCIe/Mooncake 使用 c++。所有路径属于目标执行环境，构建只写缓存和本次报告，不改源码、驱动或镜像软件包。编译错误保留原始输出；缺少开发库不自动联网安装。

## 单机与多机混用

控制机仅启动一次 Bench。各测试块的 `nodes` 分别选择执行节点，不要求整个 YAML 使用同样的节点数。

例如准备 `node0` 和 `node1` 后，各测试块可这样分配：

| 测试块 | `nodes` | 行为 |
| --- | --- | --- |
| RCCL / PCIe / memory | `[node0]` | 在 node0 上完成单机测试 |
| DeepEP | `[node0, node1]` | 同一个用例在两节点协同执行，列表顺序决定 node_rank |
| Mooncake pair | `[node0, node1]` | 两节点协同，另需地址列表和共享可写目录 |
| e2e | `[node0]` | 单节点启动模型服务并评测 |

方案的用例之间串行，一个多节点用例内部同时启动各节点。仅在单机测试中写 `nodes: [node0, node1]` 不会变成“每节点独立复测”；这类需求应复制测试块，分别选择一个节点。当前 RCCL、PCIe、memory、e2e 均限制每用例一个节点。

双节点 DeepEP 的完整模板是 [remote.template.yaml](../configs/remote.template.yaml)。填写各节点的 SSH 信息、网卡/GID、容器脚本，以及 node0 可被另一节点访问的 `master_addr`。两端使用相同原生 tests、参数、每节点进程数和兼容镜像；每节点 HCU 数应与 `num_processes` 相同。

```bash
bash bench.sh check -c configs/bench.yaml --suite deepep
bash bench.sh plan -c configs/bench.yaml --suite deepep
bash bench.sh deepep
```

SSH 只使用已有免密配置；`check/plan` 不会验证连通性或启动负载。容器启动接口及复用方式见 [运行说明](quick-run.md)。

## 多个模型顺序评测

每个模型写成一个独立 `suite: e2e` 测试块，分别指定 `model_path` 和 `serving`。运行 `bash bench.sh e2e` 会按 YAML 顺序评测所有启用的 e2e 测试块；每个模型的服务在该用例结束时清理，报告按测试名分开。

```bash
cp configs/e2e-multi-model.template.yaml configs/bench.yaml
# 编辑模型路径、卡号、网卡及各模型 serving 参数
bash bench.sh check -c configs/bench.yaml --suite e2e
bash bench.sh e2e
```

启动参数直接写在 YAML 的 `params` 下，无需另外制作 JSON：

```yaml
params:
  bench_root: "${vars.bench_root}"
  model_path: "${vars.model}"
  profile: auto
  data_path: auto
  ep_config: auto
  serving:
    visible_hcus: 8
    env:
      NCCL_IB_HCA: "${vars.ib_hca}"
      NCCL_SOCKET_IFNAME: "${vars.iface}"
      GLOO_SOCKET_IFNAME: "${vars.iface}"
    # server_argv: [...]   # 填完整 argv 时替换内置启动命令；不是 shell 字符串。
    # unset_env: [...]     # 替换内置列表，按模型列出不需要的环境变量。
  # ep_settings:          # 有自定义需求时填完整 normal_dispatch/normal_combine。
  #   normal_dispatch: {...}
  #   normal_combine: {...}
```

`profile: auto` 以仓库的 Flash INT8 8 卡配方为基础，`serving.env` 合并环境，其余 `serving` 字段替换同名字段。不同模型不能盲用该配方，替换启动命令时也应检查继承的环境和 EP 设置。仍可用 `profile: /path/profile.json` 兼容已有配置；`ep_settings` 与外部 `ep_config` 不能同时指定。

`server_argv` 是参数数组，例如 `[sglang, serve, --model-path, "${model_path}", --tp, "8", --host, "${host}", --port, "${port}"]`。它支持 `${model_path}`、`${ep_config}`、`${host}`、`${port}`，由 e2e 包装器展开。YAML 内也可引用 `${vars.xxx}`；外部 JSON 不展开 YAML 变量。内置 profile 的网卡取 YAML 环境或显式 `serving.env`，不会猜测；外部 JSON 中已写死的环境只有 `serving.env` 能覆盖。

当前评测接口固定为 SGLang 0.5.12 的 GSM8K；仓库自带官方 test 集 1319 条。自定义数据每行须含非空 `question` 和 `answer` 字符串。默认前 5 条作为提示样例，随后 100 条评分，至少需要 105 条。`num_examples: 0` 表示评测提示样例之后的全部数据。每次生成的实际 profile/EP 与评测信息记录在该用例的报告目录。

多模型模板只演示编排，不保证任意模型都兼容 Flash INT8 配方。每个模型都需经过验证的服务参数和 EP 配置。当前不支持多机模型服务、其他评测数据集或通用推理吞吐测试。
