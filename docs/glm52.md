# GLM-5.2 INT8 八卡性能

模型为 `GLM-5.2-Channel-INT8-w8a8`，单节点TP8，归入 `e2e_perf`，不加入GSM8K精度计划。启动参数参考[HYGON官方BW1100八卡SGLang0.5.12配方](https://github.com/HYGON-AI/inference-cookbook-das/blob/main/docs/model-deployment/sglang/glm-5.2.md#glm-52-channel-int8-w8a8-ifb-bw1100-8x-sglang-0512)。

## 配置

主模板 `configs/bench.template.yaml` 已包含GLM模型和性能测试块。复制/修改主配置后，`bash bench.sh all` 或 `bash bench.sh e2e_perf` 会运行Qwen及GLM；已有个人YAML不会自动增加测例。

| 配置 | 设置 |
| --- | --- |
| `vars.bench_root` / `vars.workspace` | 当前容器内仓库绝对路径、持久可写输出目录 |
| `vars.hcus` | 8张已确认空闲的卡；GLM不能使用单卡节点 |
| `vars.iface` | `ip -br addr` 查到的实际socket网卡名，不填mlx5设备名或GID |
| `models.glm52_int8.env.BENCH_MODEL_PATH` | 完整本地INT8权重目录 |
| `models.glm52_int8.tokenizer_path` | 同一模型的本地tokenizer目录 |
| `models.glm52_int8.port` | 空闲服务端口，默认30904 |
| `startup_timeout_s` | 最长3600秒等待服务就绪，不是正式性能计时 |
| `tests.glm52_throughput.params` | 长度/并发、正式/预热请求数、轮数及超时 |

socket网卡通过模型环境变量引用 `vars.iface`，启动脚本不硬编码网卡。DeepEP引用相同的平台网卡字段，但其GID和IB禁用等策略仍限定在DeepEP测试块，不传播到GLM。该官方TP8模式没有启用DeepEP专家并行，不擅自增加HCA/GID参数。

模型专用环境变量、量化、TP及后端保留在启动脚本。脚本验证本地目录、8个不重复卡号和现有网卡，然后通过 `exec sglang serve` 前台运行，由框架管理服务清理。不自动安装/升级SGLang、DTK、Torch或量化库，不在运行时下载模型。

## 单独复测

修改 `configs/glm52-perf.template.yaml`，它只包含GLM性能任务：

```bash
bash bench.sh configs/glm52-perf.template.yaml all
```

模板默认31点，每点256正式请求、32预热、1轮，与主配置一致。官方服务允许最多32个运行请求；客户端并发64/128将排队，不擅自更改官方服务参数。

小规模功能验证可使用：

```yaml
      workloads:
        - {input_tokens: 128, output_tokens: 128, concurrency: [1, 8]}
        - {input_tokens: 1024, output_tokens: 128, concurrency: [1, 8]}
      requests: 32
      warmup_requests: 4
      repeats: 1
      extra_body: {}
```

GLM保留模型默认思考模式，不假设与Qwen的非思考模式等价；正式数据必须连同实际输出长度、模式及启动参数比较。性能完成不表示模型精度已验证。启动和模型加载不计入正式请求吞吐。

## 结果

优先查看本轮 `summary.md` 的 `e2e_perf` 章节。逐点指标包括成功/失败请求数、输出tokens/s、TTFT、TPOT及实际输入/输出长度；完整统计见该节suite CSV、`metrics.csv` 和 `e2e.csv`。启动失败时查看模型会话日志及原生 `server.log`，不能把服务启动或权重加载误当性能结果。

本测例需要目标镜像已具备对应模型架构、slimquant_marlin、FlashMLA和EAGLE接口。配方接入不代表任意SGLang镜像均兼容；真实验证状态以该轮报告为准。

## 已完成验证

2026-11-04，BW1101八卡，标准镜像标签 `0.5.12-ubuntu22.04-dtk2604-py3.10-20260909-0006`，按上述4点小规模配方完成。每点32个正式请求全部成功，实际输入/输出长度均与目标一致，模型默认思考模式未显式覆盖。

| 输入/输出token | 并发 | 输出吞吐 tok/s | TTFT均值 ms | TPOT均值 ms |
| --- | --- | --- | --- | --- |
| 128/128 | 1 | 63.5258 | 209.03 | 14.22 |
| 128/128 | 8 | 180.3552 | 460.25 | 33.70 |
| 1024/128 | 1 | 52.4480 | 383.90 | 16.19 |
| 1024/128 | 8 | 131.5271 | 987.84 | 43.50 |

数据来自运行 `20261104_111420_8a9715a0`，是整套TP8服务吞吐，不是单卡数值。仅证明该环境4点流程完成，不代表完整31点扫描、精度或稳定性验收。未修改DTK/Torch、频率或宿主机NUMA；执行期间未检测到外部正显存占用，结束后服务进程已清理。出现1条未生成JSONL文件的归档告警，标准结果、原生JSON/CSV和日志均已保存。
