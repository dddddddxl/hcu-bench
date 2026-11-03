# ResNet50 / BERT 固定推理用例

来源为用户补充的飞书模型页面配方。两种模型均通过 MIGraphX 原生 `perf` 入口执行，使用独立的 `migraphx` suite，不属于 `e2e` 分组，不启动 HTTP 服务、不使用 `models` 服务字段。

首次使用、提示缺少 `migraphx-driver` 时，按 [专用镜像与新容器准备步骤](migraphx-setup.md) 操作。已提供包含 MIGraphX 5.2.0 / DTK 26.04 原生工具的镜像 tag/digest；仍需安装本工具的 Python extras。任意 Torch 镜像并不保证具备此运行环境。

BERT 的默认路径使用当前运行环境的 rocBLAS，不要求另找外置调优库；仅在明确复现特定外置库基线时设置 `rocblas_dir`。BERT 单独运行及输入要求见 [BERT 配方](bert.md)。

| recipe | ONNX 输入 | batch | 精度 / 编译参数 |
| --- | --- | --- | --- |
| `resnet50_fp16` | `[batch,3,224,224]` FP32 | 1、32 | `--fp16 --opt-compile --gpu` |
| `resnet50_int8` | `[batch,3,224,224]` FP32 | 1、32 | `--int8 --gpu`，不添加 `--opt-compile` |
| `bert_large_fp16` | input_ids / attention_mask / token_type_ids，均为 INT64 `[batch,384]` | 1、32 | `--gpu --fp16`，三个输入均 `--fill1` |

每个组合 **3 次独立进程，每次 `-n 20`**，共 6 组、18 次调用。保留三轮 Rate（inf/s）和 Total time（ms/batch），计算三轮吞吐与单批延迟各自的中位数。

## 配置与执行

```bash
cp configs/migraphx.template.yaml configs/bench.yaml
# 编辑镜像、模型/调优库路径、NUMA、唯一目标卡和容器 hook。
bash bench.sh check -c configs/bench.yaml --suite migraphx
bash bench.sh plan -c configs/bench.yaml --suite migraphx
bash bench.sh migraphx --set vars.image=YOUR_APPROVED_MIGRAPHX_IMAGE
```

镜像可填 tag 或 digest，沿用用户提供的 start/check/stop [容器接口](testpacks.md)，不固定厂商地址。hook 的 start 接收 `NAME IMAGE HOST_WORKSPACE CONTAINER_WORKSPACE`；负责把代码、模型、hyhal、BERT 调优库只读挂载，把输出目录设为可写，只透传经 BDF/UUID 核对的目标卡，记录实际镜像 ID。建议按原配方使用 `--network none`。工具不下载模型、安装驱动或替换库。

已有容器内运行：删除模板的 `containers` 及测试块的 `container`，用 `executor: local`；所有路径填写容器内路径。SSH 模式见 [配置说明](configuration.md)。执行前检查占用；共享 CI 主机需要宿主 CI 守卫，CI 优先。

| 配置 | 含义 |
| --- | --- |
| `vars.image` | 本次已准备、兼容目标环境的 MIGraphX 镜像 |
| `vars.resnet50_model` | `resnet50-torch-fp32.onnx` 的容器内路径 |
| `vars.bert_model` | `bert-large-uncased-squad-v2-opt-mha-384.onnx` 的容器内路径 |
| `vars.numa_node` | 目标卡对应的 CPU / memory NUMA 节点 |
| `vars.rocblas_dir` | 可选 BERT 外置库目录；空表示使用当前运行环境，非空必须含非空 `lib` |
| `vars.hcus` | 经核实的单个执行环境内 HIP 编号；单卡隔离容器通常为 `[0]` |
| `params.binary` / `params.numactl` | 默认从 PATH 查找，也可填已有可执行文件路径 |
| `params.round_timeout_s` | 每轮故障时限，默认 600 秒；外层默认 2100 秒 |

环境需已有 Python 3.10+、PyYAML、psutil、onnx、numactl、migraphx-driver；FP16 ResNet50 需要支持 `--opt-compile`。仓库提供可选 Python 依赖组 `.[migraphx]` 供镜像准备阶段使用，运行时不安装依赖。

`.[migraphx]` 不安装原生 MIGraphX/DTK。目标容器中可运行 `python3 testpacks/migraphx/check.py --recipe resnet50_fp16` 先检查包、可执行文件、共享库与 help flags；它与框架的静态 `bench check` 不同，但也不替代 GPU/模型实测。执行脚本会自动发现 PATH 外的标准 DTK/ROCm driver，并在开始测量前检查缺失的动态库；显式 binary 覆盖保持严格。

执行前用 ONNX 元数据验证输入名、类型和形状；动态 batch 使用请求值，静态 batch 必须与配置一致。只接受自包含 ONNX，外部权重会明确报错，避免仅记录图文件哈希而漏掉权重。BERT 仅在指定非空 `rocblas_dir` 时，向本次子进程的 `LD_LIBRARY_PATH` 前置 `<rocblas_dir>/lib`；未指定则保持环境原样，不修改宿主配置。显式指定但缺失或为空的目录仍失败，不静默退回默认库。`lib` 内的目录软链接会因指纹覆盖不完整而报错，应提供完整的目录树；文件软链接记录目标内容哈希。

## 原生命令

```bash
# 每条命令按 BS=1、32 分别执行三轮，wrapper 自动记录日志。
numactl --cpubind="$NUMA_NODE" --membind="$NUMA_NODE" \
  migraphx-driver perf -n 20 --onnx "$RESNET50_MODEL" --batch "$BS" --fp16 --opt-compile --gpu
numactl --cpubind="$NUMA_NODE" --membind="$NUMA_NODE" \
  migraphx-driver perf -n 20 --onnx "$RESNET50_MODEL" --batch "$BS" --int8 --gpu
numactl --cpubind="$NUMA_NODE" --membind="$NUMA_NODE" \
  migraphx-driver perf -n 20 --onnx "$BERT_MODEL" \
  --fill1 input_ids --fill1 attention_mask --fill1 token_type_ids --batch "$BS" --gpu --fp16
```

## 结果与边界

每个 case 保留 `plan.json`（实际 argv、环境、模型/二进制 SHA256、BERT 调优库文件指纹、输入形状）、`execution.json`、`round-1.log` 至 `round-3.log`、`rounds.json/csv`、`measurements.json` 和 `assessment.json`。

BERT 的 plan、测量和 assessment 另记 `rocblas_mode`：`runtime_default` 或 `custom_directory`。默认库结果不能直接当成历史外置调优库配方的性能验收；`custom_directory` 只表明显式配置了该目录，不证明它适配当前硬件或带来加速。

结果以 `suite: migraphx` 收入 `summary.json`、`summary.txt` 和 `metrics.csv`，原生文件归档到 `raw/`，不写入服务类模型的 `e2e.csv`。将测试块及所需节点、容器配置合并到 `configs/bench.yaml` 后，`bash bench.sh all` 会运行其中启用的 MIGraphX 用例。

解析器同时检查 Rate 和 Total time 的唯一性、正有限值及 `Rate × Total time ≈ batch × 1000`。MIGraphX 的 `Total time` 是每 batch 的时间，不能再除以 20；格式依据上游 `program::perf_report`。未知输出格式保留原日志并报无效，不猜单位。

只有三轮全部正常结束且结果完整才输出 `median_of_3_rounds`。任一轮失败、取消或超时都不发布部分中位数。框架报告的 `source_statistic` 保留该名称；如再设外层 `repeats`，汇总是多组三轮中位数的均值，不是把所有轮次重新求中位数。

所有结果标为 `correctness: not_checked`。本测试不验证 ResNet50 FP16/INT8 数值精度，不评估 BERT SQuAD 准确率。比较性能须保持硬件、驱动、实际镜像、模型、batch、编译参数及 BERT 调优库相同。
