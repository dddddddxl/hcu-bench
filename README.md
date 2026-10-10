# HCU Bench

一个 YAML 驱动的 CLI 测试工具。改配置、选 suite，自动执行测试、保存日志和汇总结果，不需要平台或网页。

| 模块 | 当前状态 |
| --- | --- |
| DeepEP | 原生低延迟压测，默认 10 分钟，输出带宽和延迟 |
| e2e | SGLang 0.5.12 / DeepSeek-V4-Flash INT8，GSM8K 100 题、5-shot，默认只报告分数 |
| RCCL / Mooncake / 算子 | 框架入口和模拟用例已预留，真实测试包待接入 |

## 获取代码

```bash
git clone https://github.com/dddddddxl/hcu-bench.git
cd hcu-bench
```

可在宿主机 clone 后挂载进容器，也可在容器内 clone。下面选择一种运行方式即可。

仓库直接提供以下 YAML 模板，两种容器模板已逐项添加中文注释：

| 模板 | 用途 |
| --- | --- |
| [in-container.template.yaml](configs/in-container.template.yaml) | 在自己已创建的容器内运行，优先用这份 |
| [managed-container.template.yaml](configs/managed-container.template.yaml) | 宿主机调用你的脚本，自动创建/清理容器 |
| [remote.template.yaml](configs/remote.template.yaml) | 控制机通过 SSH 调度 Linux 节点 |
| [demo.yaml](configs/demo.yaml) | 无 GPU 的模拟测试 |

## 方式一：使用自己创建的容器

你先按节点环境创建容器，挂载代码、原生测试、模型和数据。后面的命令在**容器内的 hcu-bench 仓库目录**执行；不需要 Docker 命令或 Docker socket。

运行环境需已有 Python 3.10+、PyYAML、psutil，以及测试所需的 Torch/DeepEP 或 SGLang。工具不会自动安装软件、升级包或修改 DTK。已有这些依赖时，不必 pip 安装本仓库。

### 1. 准备 YAML

```bash
cp configs/in-container.template.yaml configs/bench.yaml
```

编辑 `configs/bench.yaml` 顶部 `vars`，替换你需要运行的测试中的 `YOUR_*`。所有路径使用**容器内路径**。例如：

```yaml
vars:
  workspace: /work/bench-output
  bench_root: /work/hcu-bench
  deepep_script: /work/tests/test_low_latency.py
  model: /models/DeepSeek-V4-Flash-Channel-INT8-w8a8
  dataset: /datasets/gsm8k.jsonl
  profile: /work/flash-int8.profile.json
  ep_config: /work/ep_config.json
  iface: YOUR_ACTIVE_INTERFACE
  gid: "YOUR_GID_INDEX"
  gpus: [0, 1, 2, 3, 4, 5, 6, 7]
  num_processes: 8
  duration_s: 600
```

这只是 `vars` 区域的示例，不要用它覆盖整个 YAML。工作目录等公共值通过 `${vars.xxx}` 引用，改一次即可；卡号和进程数要匹配实际资源。

- **DeepEP**：准备与镜像 wheel 接口兼容的 `test_low_latency.py` 和同目录 `utils.py`。只选 deepep 时，精度项的输入路径可以暂不填写。
- **精度**：准备已有模型、question/answer 格式的 GSM8K JSONL、服务 JSON profile 和 EP JSON config。服务 profile 可从 [Flash INT8 模板](testpacks/sglang_accuracy/flash-int8-bw1100.template.json)复制后修改；部署配方与 EP 配置来源见 [真实测试说明](docs/native-tests.md)。YAML 的网卡不会自动改写外部服务 profile，两处需保持一致。
- 先确认 GPU 无其他业务占用。工具不下载模型、不转换未提供的数据、不猜网卡/GID，也不自动修补原生测试代码。

`configs/bench.yaml` 被 Git 忽略，个人节点配置不会随普通 git add 提交。

### 2. 检查并运行

```bash
bash bench.sh check -c configs/bench.yaml --suite deepep
bash bench.sh deepep      # 只跑低延迟压测
bash bench.sh e2e         # 只跑精度测试
bash bench.sh all         # 按 YAML 顺序跑全部启用项
```

上面三个运行命令是不同选择，不需要全部执行。日志自动保存，不用手动设置 ROOT/NAME、串联 docker exec 或 tee。结束后只清理本次测试进程，**不会删除你自己创建的容器**。

## 方式二：自动创建容器

在具备 Python 3.10+、PyYAML 和 Docker 的 **Linux 宿主机控制环境**中，选择另一份模板：

```bash
cp configs/managed-container.template.yaml configs/bench.yaml
```

修改顶部 `vars`：镜像、宿主机/容器目录、容器内测试输入，以及你的 `container_hook` 脚本路径。模板约定脚本接口为：

```text
bash container.sh start NAME IMAGE HOST_WORKSPACE CONTAINER_WORKSPACE
bash container.sh check NAME
bash container.sh stop NAME
```

设备、挂载、权限及空闲检查由你的脚本明确提供。也可以修改 YAML 的 start/check/stop argv 接入现有脚本；返回码和安全约定见 [容器接口说明](docs/quick-run.md#自动创建容器)。

填好后仍用同样的命令：

```bash
bash bench.sh check -c configs/bench.yaml --suite all
bash bench.sh deepep
bash bench.sh all
```

工具按用例创建独立容器、执行、归档、清理；不接管已有同名容器。没有提供实际启动脚本时，这份模板不能直接运行。宿主机只有 Python 3.6 时，使用方式一，或通过已有控制机和 [SSH 配置](configs/remote.template.yaml)调度，不给测试镜像擅自安装软件。

## 参数与后台运行

```bash
bash bench.sh plan -c configs/bench.yaml --suite all
bash bench.sh deepep,e2e
bash bench.sh configs/another-node.yaml deepep
bash bench.sh deepep --set vars.duration_s=1200
bash bench.sh deepep --set tests.deepep_pressure.params.num_tokens=256
bash bench.sh all --detach
bash bench.sh status /work/bench-output/results/RUN_ID
bash bench.sh stop /work/bench-output/results/RUN_ID
bash bench.sh report /work/bench-output/results/RUN_ID
```

`--set` 只覆盖本次运行；长期参数直接改 YAML。参数矩阵用 `matrix`，重复次数用 `repeats`。压测时长须小于 `timeout_s`；模板故障时限为 3600 秒，长压测也要增加它。`check/plan` 只是静态检查，不连接节点、创建容器或启动负载。

`BENCH_PYTHON=/path/to/python bash bench.sh ...` 可选择已有解释器。默认优先仓库 `.venv/bin/python`，否则用 python3。SSH 使用密钥、ssh-agent 或 SSH config，不存密码。

## 日志与结果

每次运行生成独立 `RUN_ID`。方式一默认位于 `vars.workspace/results/RUN_ID`，方式二位于宿主机配置的输出目录；控制台会打印路径。

主要产物是 `summary.txt` / `summary.json` / `metrics.csv`、`state.json`、`results.jsonl`、`logs/`、`raw/`，另有配置、命令和环境快照。日志带执行环境本地时间及偏移，原始输出和测试包声明的原生报告均保留。

- 普通失败默认清理后继续下一项；`failure_policy: stop` 可改为停止。清理无法确认时，停止剩余计划。
- 达到 `duration_s` 是计划结束，保留数据，不推断完整正确性通过；超过 `timeout_s` 是故障超时。
- 不吞掉原生正确性断言、SIGABRT/SIGKILL 或 profiler 崩溃；空性能结果不是成功。
- 精度默认 `min_score: ""`，只报告分数；没有明确阈值时不称为“精度达标”。DeepEP effective bandwidth 不是 RCCL busbw。

## 验证与开发

已在一个 BW1100 单机 8 卡环境验证：适配原生 handle 索引后，DeepEP 达到 10 分钟窗口；GSM8K 100 题得分 98%，仅报告分数。存在性能尖峰，不能推断所有镜像/节点或性能稳定性已验收。新版 Bash 入口及框架免密 SSH/managed Docker 接口仍需对应真实环境验证。

本地 82 项测试通过。开发环境安装、测试及无 GPU 模拟运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[native]'
python -m unittest discover -s tests -v
bash bench.sh configs/demo.yaml all
```

开发安装不要求在待验镜像里执行。Windows 使用 `.venv\Scripts\python.exe` 和 `bench.cmd`。模拟数据明确标为 SIMULATED，不是 GPU 性能。

更多内容：[简化运行](docs/quick-run.md) · [真实测试说明](docs/native-tests.md) · [同事测试包接入规范](docs/testpacks.md)。
