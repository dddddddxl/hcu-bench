# 简化运行

一次准备环境和输入，之后改 YAML、选 suite 即可。不需要每次设置 ROOT/NAME、手动启动测试进程、手动 tee 或创建日志目录。

## 自己创建容器

把 Bench 仓库和原生测试/模型/数据挂进你自己创建的容器，在容器里使用现有 Python 3.10+、PyYAML 和测试依赖。工具不安装软件，不改 DTK；容器内不需要 Docker 命令或 Docker socket。

在容器内的 Bench 仓库目录，首次准备：

```bash
cp configs/in-container.template.yaml configs/bench.yaml
```

修改 `configs/bench.yaml` 顶部 `vars`：可写工作目录、容器内 Bench 路径、原生脚本、模型、数据、服务 profile、EP 配置、实际网卡/GID、卡号、压测时长。其余字段引用 `${vars.xxx}`，无需反复填写同一路径。`configs/bench.yaml` 被 Git 忽略，不提交节点私有配置。

随后在容器里执行：

```bash
bash bench.sh deepep              # 只跑低延迟压测
bash bench.sh e2e                 # 只跑精度测试
bash bench.sh all                 # 按 YAML 顺序跑全部启用项
bash bench.sh deepep,e2e          # 选择多个 suite
```

suite 顺序仍按 YAML，而不是逗号参数顺序。当前真实模块为 deepep/e2e，其他 suite 需先接入测试包。

退出测试会保留用户自己的容器，只清理本次测试的进程。所有输出归档到 `vars.workspace/results/RUN_ID`，控制台会打印具体目录。

## 自动创建容器

在已准备好 Python 3.10+ / PyYAML / Docker 的 Linux 宿主机控制环境中，首次准备：

```bash
cp configs/managed-container.template.yaml configs/bench.yaml
```

填顶部 `vars`：镜像、宿主机/容器工作目录、容器内测试路径，以及你的容器脚本 `container_hook`。启动还是相同命令：

```bash
bash bench.sh deepep
bash bench.sh all
```

沿用原有 managed 生命周期：每个测试用例创建一个独立容器，结束后清理它；用例间默认串行。不会将用户现有的同名容器当作自己的容器。不会自动猜设备、挂载、权限、网卡或镜像依赖。

模板用一个用户脚本提供三个动作。接口是模板约定，可以直接修改 YAML 的 start/check/stop argv 来接现有脚本，不要求重写为这三个动作：

```text
bash container.sh start NAME IMAGE HOST_WORKSPACE CONTAINER_WORKSPACE
bash container.sh check NAME
bash container.sh stop NAME
```

- start：拒绝接管已有同名容器，创建唯一指定 NAME，完成后返回；正确挂载代码/输入和可写工作目录。不后台启动 Bench 本身，工具负责执行测试。
- check：运行中返回 0；不存在或已停止返回 1；Docker 无法访问等检查错误返回其他非零值。后者不能假装清理成功。
- stop：只清理本次创建的 NAME，完成返回 0；无法确认所有权或清理失败返回非零。需要能处理 start 中途失败留下的现场。

当前没有擅自添加通用 docker run 配方；真实脚本仍由节点环境提供。模板填好并接入该脚本后，工具自动调用它，不再手动 docker exec / run_node.sh。

宿主机只有 Python 3.6 时，这个宿主机控制模式不能直接运行 Bench。可选择上面的容器内模式，或使用已准备好依赖的控制机和现有 SSH executor；不默认给测试容器挂载宿主机 Docker socket。

## 检查、覆盖参数

```bash
bash bench.sh check -c configs/bench.yaml --suite all
bash bench.sh plan -c configs/bench.yaml --suite deepep
bash bench.sh configs/another-node.yaml deepep
bash bench.sh deepep --set vars.duration_s=1200
bash bench.sh deepep --set tests.deepep_pressure.params.num_tokens=256
bash bench.sh all --detach
bash bench.sh status /ABSOLUTE_PATH/results/RUN_ID
bash bench.sh stop /ABSOLUTE_PATH/results/RUN_ID
```

check/plan 只做静态检查，不连节点或启动工作负载。原生依赖/文件/接口在真实测试入口进行检查。duration_s 必须小于 timeout_s；模板故障时限为 3600 秒，延长到超过该时限时也要改 timeout_s。

`BENCH_PYTHON=/path/to/python bash bench.sh ...` 可选择已有解释器。默认优先仓库 `.venv/bin/python`，否则使用 python3；入口会检查版本和 PyYAML，不运行 pip。

`vars` 只做显式配置引用：全值 `${vars.gpus}` 保留列表/数字类型；字符串中的 `${vars.workspace}` 拼接路径；支持引用其他 vars，未知变量及循环引用报错。不会展开 shell 环境变量或执行命令。`${gpu_ids}`、`${run_id}` 等原有运行时变量仍由 worker 阶段处理。

## 不会隐藏的准备条件

模型/数据/原生脚本必须已存在；GSM8K 仍需 question/answer JSONL，服务参数仍在 JSON profile 中，EP config 为已有 JSON。工具不下载模型、安装依赖、改动原生脚本或自动适配 wheel 接口。修改 YAML 的网卡不会自动改写外部服务 profile，两处必须对应同一实际环境。

原生断言、SIGABRT/SIGKILL、空结果都会如实记录；达到压测时长是计划结束，不推断完整正确性通过。GPU 锁不代表其他业务空闲，启动前的业务占用检查仍需人工或你的启动脚本负责。
