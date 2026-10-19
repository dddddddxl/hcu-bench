# 简化运行

一次准备环境和输入，之后改 YAML、选 suite 即可。不需要每次设置 ROOT/NAME、手动启动测试进程、手动 tee 或创建日志目录。

## 自己创建容器

把完整 Bench 仓库和本地模型挂进自己创建的容器，在容器里使用现有 Python 3.10+、PyYAML、编译器及运行库。原生测试源码和 GSM8K 已随仓库提供，首次自动编译，不安装软件包、不改 DTK；容器内不需要 Docker 命令或 Docker socket。

内置原生包装器还需要 psutil。各 suite 的输入要求和功能范围见 [README](../README.md)，参数、混合节点及多模型示例见 [配置说明](configuration.md)。

在容器内的 Bench 仓库目录，首次准备：

```bash
cp configs/in-container.template.yaml configs/bench.yaml
```

修改 `configs/bench.yaml` 顶部 `vars`：可写工作目录、容器内 Bench 路径、模型、实际网卡/GID、卡号和压测时长。工具、脚本、数据、profile、EP 路径可保持 `auto`，启动参数可在 YAML 的 `serving` 修改。`configs/bench.yaml` 被 Git 忽略，不提交节点私有配置。

随后在容器里执行：

```bash
bash bench.sh rccl                # 只跑单机 MPI AllReduce
bash bench.sh deepep              # 只跑低延迟压测
bash bench.sh pcie                # 只跑五类 PCIe 带宽路径
bash bench.sh memory              # 只跑单卡显存/缓存/LDS 带宽
bash bench.sh e2e                 # 只跑精度测试
bash bench.sh all                 # 按 YAML 顺序跑全部启用项
bash bench.sh deepep,e2e          # 选择多个 suite
```

suite 顺序仍按 YAML，而不是逗号参数顺序。综合容器模板包含 RCCL、PCIe、memory、DeepEP、e2e；Mooncake 见单独的双机模板。PCIe 需先核对原生工具的 Device ID，见 [PCIe 带宽测试](pcie.md)；memory 使用已有 hcu-stream，选卡后逐卡串行，见 [显存带宽测试](memory.md)。

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

仓库不提供通用 docker run 配方，启动脚本由节点环境提供。模板填好并接入该脚本后，工具自动调用它，无需手动 docker exec。

### 从宿主机或 SSH 复用已有容器

将测试的 `container` 指向一个 `mode: existing` 配置，指定固定容器名、容器内可写 `work_root` 和 `check` 命令。删除 managed 模式的 `start` / `stop` 字段；existing 模式禁止这两项。`check` 仍须遵守上述返回码约定。

默认通过 `docker exec -i` 在容器内执行 Python worker，不停止或删除该容器。完整 existing 示例见 [Mooncake 模板](../configs/mooncake.template.yaml)；接口同样可用于单机 suite。

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

`vars` 只做显式配置引用：全值 `${vars.hcus}` 保留列表/数字类型；字符串中的 `${vars.workspace}` 拼接路径；支持引用其他 vars，未知变量及循环引用报错。不会展开 shell 环境变量或执行命令。`${hcu_ids}`、`${run_id}` 等原有运行时变量仍由 worker 阶段处理。

## 不会隐藏的准备条件

模型权重和匹配的框架运行库必须已存在。源码在目标环境自动编译，缺少编译器或开发库会报告准备失败，不下载模型、不安装依赖、不自动猜测 wheel 接口。内置 profile 的网卡取 YAML 配置；外部 JSON 中写死的环境需用 `serving.env` 显式覆盖。

原生断言、SIGABRT/SIGKILL、空结果都会如实记录；达到压测时长是计划结束，不推断完整正确性通过。HCU 锁不代表其他业务空闲，启动前的业务占用检查仍需人工或你的启动脚本负责。
