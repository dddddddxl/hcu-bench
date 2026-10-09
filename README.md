# HCU Bench 0.1

一个 CLI 工具，集成 RCCL、DeepEP、Mooncake、算子和端到端测试入口。没有网页、服务端平台或数据库，核心只依赖 Python 3.10+ 和 PyYAML。

初版实现框架、模拟测试和通用脚本执行。五类真实 GPU/通信测试由后续提供的测试包接入，不包含假装可用的原生测试实现。

## 直接试用

Windows 安装，PowerShell 进入本目录后：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

当前开发机器已经安装，可直接运行：

```powershell
.\bench.cmd list
.\bench.cmd check -c configs/demo.yaml
.\bench.cmd plan -c configs/demo.yaml --suite deepep,mooncake
.\bench.cmd run -c configs/demo.yaml --suite all
.\bench.cmd run -c configs/local-command.yaml
```

`demo.yaml` 跑五类模拟入口，所有数据明确标为 SIMULATED；`local-command.yaml` 实际运行一个很小的 CPU 求和示例，并拷回原生报告，用来验证外部脚本接入流程，不是算子性能基线。

Linux 安装与运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
hcu-bench run -c configs/demo.yaml --suite all
```

## 常用命令

```bash
hcu-bench run -c configs/demo.yaml --suite deepep
hcu-bench run -c configs/demo.yaml --suite rccl,deepep,mooncake
hcu-bench run -c configs/demo.yaml --suite all --detach
hcu-bench run -c configs/demo.yaml --set tests.deepep_demo.params.samples=10
hcu-bench status runs/RUN_ID
hcu-bench stop runs/RUN_ID
hcu-bench report runs/RUN_ID
```

`--set` 覆盖已有配置键，也可覆盖 manifest 声明的测试参数默认值；未知测试参数会被拒绝。值按 YAML 类型解析。批量参数写在测试项的 `matrix`，重复次数写 `repeats`；计划按配置顺序展开。`plan --json` 输出机器可读计划，`check/plan` 不连接节点、不创建容器、不执行负载。

## 核心文件

```text
src/hcu_bench/
  cli.py                  命令入口
  config.py / models.py   配置与数据结构
  planner.py / registry.py 参数校验、矩阵、测试包能力
  adapters.py             mock / 外部测试脚本
  containers.py           容器命令接口
  executors/              Local / SSH 执行器
  agent.py                本地、SSH、容器内的轻量执行 worker
  runner.py               任务状态、多节点启动、失败策略
  store.py / results.py   运行档案、标准结果
  artifacts.py / report.py 原生报告传回、汇总
```

各类测试共用一个 command adapter；区别放在测试包 manifest，不为五个空模块复制五份执行代码。接入规范见 [测试包接口](docs/testpacks.md)。

## 已确认的行为

- 控制机通过 SSH 并发启动同一用例的各节点；用例之间默认串行。
- SSH 目标是 Linux，使用密钥、ssh-agent 或 SSH config；不存密码，不关闭 host-key 校验。首次访问的主机应先人工确认 SSH 指纹。
- 普通测试失败：清理本轮后继续下一项；`failure_policy: stop` 可改为停止。
- `duration_s` 达到计划时长后结束压测，记为 `duration_reached`；`timeout_s` 是故障时限，记为 `timed_out`。已有数据保留，但不推断整项正确性通过。
- 空性能结果记为 `invalid`。单条无效测量警告并跳过，原始输出保留；正确性失败不能被当作 profiler 警告跳过。
- 分布式用例任一关键节点失败，取消该用例的其他节点。后续用例可继续；清理无法确认则停止整个剩余计划。
- 容器支持 existing/managed。只对本次 managed 启动尝试调用 stop，不停止复用容器；已存在的同名 managed 容器拒绝接管。
- 原生报告按 manifest 声明路径传回。缺失、超出传输限额或传输失败只警告，单独保存归档状态。
- 不安装测试依赖，不更改 DTK、镜像、网卡、设备权限或超节点环境变量。

## 日志与结果

每次执行生成独立 `runs/RUN_ID`，包含配置快照、计划、环境信息、任务状态、标准结果、各节点日志、原生报告和汇总。`.log` 每行采用测试节点宿主机的本地时间，带时区偏移；控制机产生的框架警告使用控制机本地时间。`.raw.log` 保留原始内容。原生报告的大小与 SHA256 保存到 `artifacts.jsonl`。

`summary.txt/summary.json/metrics.csv` 按测试、参数、节点、单位和原始统计口径分组。均值明确为“已报告数值的均值”，不把 mean 改叫 p50，不混合不同 shape，也不将 RCCL busbw 与 DeepEP effective bandwidth 当成同一指标。失败任务中的已采数据附带 case_status。

资源锁只协调 Bench 启动的任务，不等于自动判断节点是否空闲。实际 GPU 绑定和业务预检由 manifest/env/测试脚本明确指定；不会猜 HIP/CUDA 的可见设备配置。

## 初版边界与验证

后台运行、状态查询、取消、计划时长、报告归档已实现。跨用例并行、断点恢复、自动分发测试代码、模型部署、原生 RCCL/DeepEP/Mooncake/xpu-perf/vLLM 解析器未接入。服务端/客户端就绪流程由接入脚本负责。

本地 Windows/Python 3.10 的测试覆盖配置、矩阵、参数、执行、取消、多 rank 协同和报告归档；SSH 命令及容器生命周期使用协议/模拟测试验证，真实 Linux SSH/Docker 和 GPU 验收仍需目标节点。

```bash
python -m unittest discover -s tests -v
```
