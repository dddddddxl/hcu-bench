# 端到端精度与吞吐测试

## 运行入口

```bash
cp configs/e2e-multi-model.template.yaml configs/bench.yaml
cp model_scripts/sglang.template.sh model_scripts/model_a.local.sh
cp model_scripts/vllm.template.sh model_scripts/model_b.local.sh
# 修改两个启动脚本和 bench.yaml 的路径、卡号及测试参数
bash bench.sh check -c configs/bench.yaml --suite e2e
bash bench.sh e2e_acc       # 只测精度
bash bench.sh e2e_perf      # 只测吞吐
bash bench.sh e2e           # 两项都测
```

不需要多个模型时，删除 `model_b` 及测试块中的对应引用。集成到其他 suite 时，把 `models` 和两个测试块合并进同一份 YAML，入口仍为 `bash bench.sh all`。

`e2e` 是选择分组，不是新增的一轮测试。两项都启用时执行：模型 A 启动 → 精度 → 吞吐全部组合 → 清理 → 模型 B。只选择其中一项，就不会启动另一项。模型按所选测试块首次引用的顺序运行，同一会话内始终先精度后吞吐。

## 启动脚本

权重路径、量化、TP/DP/EP、上下文长度、显存比例、网卡和框架环境变量都写在模型自己的脚本中。例如：

```bash
#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH=/models/your-model
# 在这里添加该模型已验证的环境变量和启动参数
exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --tp 8 \
  --host "$BENCH_HOST" --port "$BENCH_PORT"
```

这是接口示例，不是任意模型都能用的部署配方。需要 EP 配置的模型在脚本里指定自己的配置文件。服务必须前台运行，最后使用 `exec`；不要启动 Docker、使用 `nohup` 或把服务放到后台。脚本应保留 Bench 传入的 `HIP_VISIBLE_DEVICES`，只使用 YAML 分配的 HCU。

Bench 注入 `BENCH_HOST`（当前为 127.0.0.1）、`BENCH_PORT`、`BENCH_MODEL_ID`、`BENCH_OUTPUT_DIR` 和 `BENCH_ROOT`。本版客户端和服务位于同一执行环境，不支持给定外部 URL 或多机服务启动。

模型配置只保留编排所需字段：

```yaml
models:
  model_a:
    start_script: model_scripts/model_a.local.sh
    port: 30000
    startup_timeout_s: 1800
    # tokenizer_path: /models/your-model
    # model_name: served-name
    # health_path: /health
    # before_perf_script: model_scripts/reset-cache.local.sh
    # nodes: [node0]
    # container: existing_image
```

脚本路径相对**执行环境的仓库根目录**解析，也可以使用绝对路径，不相对 YAML。`vars.bench_root` 必须指向目标仓库。SSH 或容器模式需要该仓库、脚本及权重在目标可见。

服务就绪后读取 `/v1/models` 获取模型名；返回多个模型时必须填 `model_name`。吞吐客户端需要本地 tokenizer，优先使用 `tokenizer_path`，未填则尝试接口中的 `root` / `id` 本地目录，不下载权重。自定义服务健康地址可修改 `health_path`。

当前随机 chat 压测还要求 tokenizer 提供可用的 `chat_template`，用于计算输入长度。仅提供自定义 encoding 脚本、没有 Jinja 聊天模板的模型（例如部分 DeepSeek-V4 权重包）不能直接套用这条吞吐路径，需要先适配与服务一致的计数方式；工具会保留原生报错，不自动填入不匹配的模板。此限制不影响原来的 GSM8K 精度接口。

`models.<名称>.nodes` / `container` 可覆盖测试块的执行位置；每个会话仅支持一个节点。相同模型在不同节点、容器或不同测试环境变量下会分开启动，不会错误复用服务。

## 测试参数

```yaml
tests:
  accuracy:
    suite: e2e_acc
    models: [model_a, model_b]
    nodes: [node0]
    timeout_s: 5400
    params:
      data_path: auto
      num_examples: 100
      num_shots: 5
      num_threads: 128
      max_tokens: 2048
      min_score: ""
  throughput:
    suite: e2e_perf
    models: [model_a, model_b]
    nodes: [node0]
    timeout_s: 14400
    params:
      shapes: [[128, 128], [1024, 1024], [4096, 512]]
      concurrency: [1, 8, 32]
      requests: 256
      warmup_requests: 32
      repeats: 3
      seed: 42
      request_timeout_s: 600
      extra_body: {}
      python: auto
```

精度仍使用执行环境中 SGLang 0.5.12 的 GSM8K 评测接口，默认前 5 题作提示，随后 100 题评分。`num_examples: 0` 使用提示之外全部样本；`min_score: ""` 只报分数，非空字符串如 `"0.90"` 才判定阈值。只安装 vLLM 的镜像不保证具备这个精度客户端，可以先只跑 `e2e_perf`。

吞吐使用固定版 EvalScope 1.13.0 的 OpenAI 流式 chat 接口：

| 参数 | 含义 |
| --- | --- |
| `shapes` | 每对为目标输入 / 输出 token 数，不是字节或字符长度 |
| `concurrency` | 每个测试点的最大并发；闭环请求，不按固定 QPS 发送 |
| `requests` | 每个测试点的正式请求数，必须不少于最大并发 |
| `warmup_requests` | 每点额外预热请求数；不计入正式指标 |
| `repeats` | 每个长度 × 并发组合的测量轮次，不重复加载模型 |
| `seed` | 随机输入种子；不同测试点使用不同 dataset offset |
| `request_timeout_s` | 单个请求的超时，正整数秒 |
| `extra_body` | 该测试的附加请求字段，如框架支持的 `ignore_eos` 或思考模式；不覆盖模型名、prompt、stream、token 上限 |
| `python` | `auto` 准备隔离客户端；也可填已安装 EvalScope 1.13.0 的解释器路径，只检查，不往该解释器安装 |

示例产生每模型 3 × 3 × 3 = 27 个吞吐测试点。测试块的 `timeout_s` 是一个模型完成该测试全部点的总预算，模型启动另算 `startup_timeout_s`，客户端安装另算 `build.timeout_s`。模型测试不用外层 `matrix`、`repeats`、`duration_s`、`pack` 或 `adapter`。

随机输入长度在 tokenizer 层生成，chat 模板、服务端计数及提前 EOS 可能使实际长度不同。结果同时记录目标长度与实际平均 token 数；对比性能时检查两者，以及缓存、思考模式、流式返回策略是否一致。`min_tokens` 等请求字段也需要目标服务支持。

Bench 不默认关闭模型缓存，也不声称不同随机 prompt 一定完全没有共享前缀。需要固定缓存策略时在启动脚本配置；需要精度后清理缓存时提供 `before_perf_script`，它在吞吐测试开始前执行一次，非零退出视为该项失败。不提供通用清缓存命令，避免误改服务行为。

## 首次准备与结果

EvalScope 源码随仓库提供，校验后在 `build.cache_dir` 下创建独立 venv。首次用 pip 安装 Python 依赖，需网络包源或管理员配置好的 `PIP_NO_INDEX` / `PIP_FIND_LINKS`；这些依赖 wheel 不随仓库带齐，不能声称完全离线。无需另拉测试代码，不向服务环境安装或升级 DTK、Torch、SGLang。每次记录客户端版本和依赖清单，后续复用缓存。

统一报告仍有 `summary.json`、`metrics.csv` 和 `results.jsonl`，另外生成：

- `e2e.csv`：每个模型、测试类型、长度、并发和轮次一行，含成功/失败数、输出/总 tokens/s、requests/s、平均及 P50/P95/P99 TTFT/TPOT/ITL/端到端延迟。
- `e2e.md`：便于阅读的关键指标表；不把多轮 P99 再平均成一个 P99。
- 原生归档：服务日志、实际启动脚本摘要、客户端参数、任务状态、EvalScope 原始 JSON/SQLite/HTML、精度回答。

TTFT/TPOT/ITL 单位为 ms，端到端请求延迟单位为 s。吞吐是正式测量窗口上的结果，不是所有卡各自相加；`total_throughput` 包括输入和输出，不能与纯输出 tokens/s 混用。

正式请求未全部成功、结果为空或缺少必要指标时，记录为无效/失败，不给出伪成功结论；仍保留原始结果。普通失败默认清理后继续；`failure_policy: stop` 停止后续项。服务退出、超时或取消都执行本次进程清理，清理无法确认则停止整个计划。模型会话失败时，已完成的子项指标仍保留，会话状态与测量状态分别列出。

本次新增流水线以本地 CPU 测试验证编排和 HTTP 客户端，尚未完成真实 HCU 多模型性能验证；启动脚本模板不是经过目标模型实测的配方。本机 Windows 客户端验证使用同版本官方 wheel；源码包自动安装尚需在目标 Linux 环境复核。

## 旧配置

旧的 `suite: e2e` 被解释为 `e2e_acc`；`testpacks/sglang_accuracy/manifest.yaml`、`profile`、`serving` 和 `ep_settings` 继续可用。新配置优先用上面的启动脚本方式。旧配置不会自动增加 `e2e_perf`，需要从模板加入新的测试块。
