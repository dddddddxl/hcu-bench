# ResNet50 / BERT 固定推理用例

来源为用户补充的飞书模型页面配方。两种模型均通过 MIGraphX 原生 `perf` 入口执行，使用 `e2e_perf` suite，不启动 HTTP 服务、不使用 `models` 服务字段。

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
bash bench.sh check -c configs/bench.yaml --suite e2e_perf
bash bench.sh plan -c configs/bench.yaml --suite e2e_perf
bash bench.sh e2e_perf --set vars.image=YOUR_APPROVED_MIGRAPHX_IMAGE
```

镜像可填 tag 或 digest，沿用用户提供的 start/check/stop [容器接口](testpacks.md)，不固定厂商地址。hook 的 start 接收 `NAME IMAGE HOST_WORKSPACE CONTAINER_WORKSPACE`；负责把代码、模型、hyhal、BERT 调优库只读挂载，把输出目录设为可写，只透传经 BDF/UUID 核对的目标卡，记录实际镜像 ID。建议按原配方使用 `--network none`。工具不下载模型、安装驱动或替换库。

已有容器内运行：删除模板的 `containers` 及测试块的 `container`，用 `executor: local`；所有路径填写容器内路径。SSH 模式见 [配置说明](configuration.md)。执行前检查占用；共享 CI 主机需要宿主 CI 守卫，CI 优先。

| 配置 | 含义 |
| --- | --- |
| `vars.image` | 本次已准备、兼容目标环境的 MIGraphX 镜像 |
| `vars.resnet50_model` | `resnet50-torch-fp32.onnx` 的容器内路径 |
| `vars.bert_model` | `bert-large-uncased-squad-v2-opt-mha-384.onnx` 的容器内路径 |
| `vars.numa_node` | 目标卡对应的 CPU / memory NUMA 节点 |
| `vars.rocblas_dir` | BERT 匹配的调优库目录，必须含非空 `lib` |
| `vars.hcus` | 经核实的单个执行环境内 HIP 编号；单卡隔离容器通常为 `[0]` |
| `params.binary` / `params.numactl` | 默认从 PATH 查找，也可填已有可执行文件路径 |
| `params.round_timeout_s` | 每轮故障时限，默认 600 秒；外层默认 2100 秒 |

环境需已有 Python 3.10+、PyYAML、psutil、onnx、numactl、migraphx-driver；FP16 ResNet50 需要支持 `--opt-compile`。仓库提供可选 Python 依赖组 `.[migraphx]` 供镜像准备阶段使用，运行时不安装依赖。

执行前用 ONNX 元数据验证输入名、类型和形状；动态 batch 使用请求值，静态 batch 必须与配置一致。只接受自包含 ONNX，外部权重会明确报错，避免仅记录图文件哈希而漏掉权重。BERT 只在本次子进程中向 `LD_LIBRARY_PATH` 前置 `<rocblas_dir>/lib`，不修改宿主配置。`lib` 内的目录软链接会因指纹覆盖不完整而报错，应提供完整的调优库目录树；文件软链接记录目标内容哈希。

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

解析器同时检查 Rate 和 Total time 的唯一性、正有限值及 `Rate × Total time ≈ batch × 1000`。MIGraphX 的 `Total time` 是每 batch 的时间，不能再除以 20；格式依据上游 `program::perf_report`。未知输出格式保留原日志并报无效，不猜单位。

只有三轮全部正常结束且结果完整才输出 `median_of_3_rounds`。任一轮失败、取消或超时都不发布部分中位数。框架报告的 `source_statistic` 保留该名称；如再设外层 `repeats`，汇总是多组三轮中位数的均值，不是把所有轮次重新求中位数。

所有结果标为 `correctness: not_checked`。本测试不验证 ResNet50 FP16/INT8 数值精度，不评估 BERT SQuAD 准确率。比较性能须保持硬件、驱动、实际镜像、模型、batch、编译参数及 BERT 调优库相同。
