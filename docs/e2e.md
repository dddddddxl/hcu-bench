# 端到端精度与吞吐测试

本页说明语言模型的精度/吞吐流水线。Wan 文生视频也属于 `e2e_perf`，但使用独立测试包和参数，见 [Wan 测试](wan.md)；不使用下面的 tokenizer、token 长度和 EvalScope 客户端。

## 运行入口

```bash
cp configs/e2e-multi-model.template.yaml configs/bench.yaml
# 在bench.yaml填写工作目录、仓库目录、两份本地权重路径和可用单卡
bash bench.sh check -c configs/bench.yaml --suite e2e
bash bench.sh e2e_acc       # 只测精度
bash bench.sh e2e_perf      # 只测吞吐
bash bench.sh e2e           # 两项都测
```

默认模型是Qwen3-4B、Qwen3-8B，使用仓库中的两份通用BF16、TP1启动脚本，串行复用单卡。权重不下载，路径填入`vars.qwen3_4b_path`、`vars.qwen3_8b_path`。具体attention backend、显存比例等优化按目标节点验证后写入对应启动脚本；默认不强塞硬件专用开关，不改宿主机设置。

两项测试默认`models: all`，按顶层`models`的声明顺序运行全部模型。新增模型只要在`models`下添加`start_script`，不用再改两份引用列表；也可继续写`models: [qwen3_4b]`只选指定模型。集成其他suite时，把模型和两个测试块放进同一YAML，入口仍为`bash bench.sh all`。旧个人配置不会自动迁移。

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
  qwen3_4b:
    start_script: model_scripts/qwen3_4b.sh
    env:
      BENCH_MODEL_PATH: /models/Qwen3-4B
    port: 30000
    startup_timeout_s: 1800
    # tokenizer_path: /models/your-model
    # model_name: served-name
    # health_path: /health
    # before_perf_script: model_scripts/reset-cache.local.sh
    # nodes: [node0]
    # container: existing_image
  qwen3_8b:
    start_script: model_scripts/qwen3_8b.sh
    env:
      BENCH_MODEL_PATH: /models/Qwen3-8B
  another_model:               # 新模型只加这一段，启动命令放进自己的脚本。
    start_script: model_scripts/another_model.local.sh
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
    models: all
    nodes: [node0]
    timeout_s: 14400
    params:
      data_path: auto
      few_shot_path: auto
      num_examples: 0
      num_shots: 5
      num_threads: 128
      max_tokens: 2048
      chat_template_kwargs: {enable_thinking: false}
      min_score: ""
  throughput:
    suite: e2e_perf
    models: all
    nodes: [node0]
    timeout_s: 172800
    params:
      workloads:
        - {input_tokens: 128, output_tokens: 128, concurrency: [1, 4, 8, 16, 32, 64, 128]}
        - {input_tokens: 1024, output_tokens: 1024, concurrency: [1, 2, 4, 8, 16, 32, 64, 128]}
        - {input_tokens: 2048, output_tokens: 128, concurrency: [1, 2, 4, 8, 16, 32, 64, 128]}
        - {input_tokens: 3500, output_tokens: 1000, concurrency: [1, 2, 4, 8, 16, 32, 64, 128]}
      requests: 256
      warmup_requests: 32
      repeats: 1
      seed: 42
      request_timeout_s: 1800
      extra_body: {chat_template_kwargs: {enable_thinking: false}}
      python: auto
```

精度使用执行环境中SGLang 0.5.12的GSM8K评测接口。新版模板默认`data_path: auto`、`few_shot_path: auto`、`num_examples: 0`：训练集前5题作为提示，test集1319题全部评分，不会少评5题；两份数据及许可随仓库提供，无需另拉代码或数据。可填自有`few_shot_path`，若提示问题与评分集重叠则拒绝运行。`num_examples`可选评分数量，`0`表示全量。

为了兼容旧配置，未设置`few_shot_path`或设为空时仍沿用“同一文件前num_shots题作提示、剩余题评分”，此时内置test集5-shot的全量只有1314题。原生模型/评测默认值不偷偷改变，推荐从新版模板迁移。`min_score: ""`只报分数，不把参考表的vLLM历史成绩当成SGLang通过阈值。

默认精度温度0、top_p=1、输出上限2048，精度和性能都显式请求`chat_template_kwargs.enable_thinking: false`。`--reasoning-parser qwen3`是服务解析设置，不等于关闭思考。新增模型沿用所选测试参数；其服务不支持对应请求字段时应调整YAML，不自动猜测模式。只安装vLLM的镜像不保证具备SGLang精度客户端，可先只跑`e2e_perf`。

吞吐使用固定版 EvalScope 1.13.0 的 OpenAI 流式 chat 接口：

| 参数 | 含义 |
| --- | --- |
| `shapes` | 每对为目标输入 / 输出 token 数，不是字节或字符长度 |
| `concurrency` | 每个测试点的最大并发；闭环请求，不按固定 QPS 发送 |
| `workloads` | 每个长度单独指定并发列表，与`shapes`/`concurrency`二选一，避免无意补出表中没有的测试点 |
| `requests` | 每个测试点的正式请求数，必须不少于最大并发 |
| `warmup_requests` | 每点额外预热请求数；不计入正式指标 |
| `repeats` | 每个长度 × 并发组合的测量轮次，不重复加载模型 |
| `seed` | 随机输入种子；不同测试点使用不同 dataset offset |
| `request_timeout_s` | 单个请求的超时，正整数秒 |
| `extra_body` | 该测试的附加请求字段，如框架支持的 `ignore_eos` 或思考模式；不覆盖模型名、prompt、stream、token 上限 |
| `python` | `auto` 准备隔离客户端；也可填已安装 EvalScope 1.13.0 的解释器路径，只检查，不往该解释器安装 |

默认性能组合来自[月英性能数据的sglang llm表](https://kcnm6g5dkw5p.feishu.cn/sheets/PcLIsjCPOhMLRMtGVi9cCYRDnse?sheet=IpWVtW)（读取revision 2000）：BF16、TP1，每模型31点，不含并发256，128/128不含并发2。表没有给请求数、预热、轮数及EvalScope版本；256正式请求、32预热、1轮、固定客户端1.13.0是本次确认的配置，不宣称与原表原始命令完全一致。

两个默认模型共62个性能点、15872个正式请求，完整扫描可能需要数小时以上，应安排足够的空闲窗口。模板的性能总故障时限是每模型48小时，并非要求连续压测48小时；单请求故障时限1800秒。可按环境调整，不影响请求数和测试组合。测试块`timeout_s`是该模型完成全部测试点的总预算，模型启动另算`startup_timeout_s`，客户端安装另算`build.timeout_s`。模型测试不用外层`matrix`、`repeats`、`duration_s`、`pack`或`adapter`。

随机输入长度在 tokenizer 层生成，chat 模板、服务端计数及提前 EOS 可能使实际长度不同。结果同时记录目标长度与实际平均 token 数；对比性能时检查两者，以及缓存、思考模式、流式返回策略是否一致。`min_tokens` 等请求字段也需要目标服务支持。

Bench 不默认关闭模型缓存，也不声称不同随机 prompt 一定完全没有共享前缀。需要固定缓存策略时在启动脚本配置；需要精度后清理缓存时提供 `before_perf_script`，它在吞吐测试开始前执行一次，非零退出视为该项失败。不提供通用清缓存命令，避免误改服务行为。

## 首次准备与结果

EvalScope 源码随仓库提供，校验后在 `build.cache_dir` 下创建独立 venv。通过执行端现有 pip 22.3+ 的 `--python` 参数安装到该隔离环境，不依赖镜像自带 ensurepip。首次安装 Python 依赖需网络包源或管理员配置好的 `PIP_NO_INDEX` / `PIP_FIND_LINKS`；这些依赖 wheel 不随仓库带齐，不能声称完全离线。无需另拉测试代码，不向服务环境安装或升级 DTK、Torch、SGLang。每次记录客户端版本和依赖清单，后续复用缓存。

统一报告仍有 `summary.json`、`metrics.csv` 和 `results.jsonl`，另外生成：

- `e2e.csv`：每个模型、测试类型、长度、并发和轮次一行，含成功/失败数、输出/总 tokens/s、requests/s、平均及 P50/P95/P99 TTFT/TPOT/ITL/端到端延迟。
- `e2e.md`：便于阅读的关键指标表；不把多轮 P99 再平均成一个 P99。
- 原生归档：服务日志、实际启动脚本摘要、客户端参数、任务状态、EvalScope 原始 JSON/SQLite/HTML、精度回答。

TTFT/TPOT/ITL 单位为 ms，端到端请求延迟单位为 s。吞吐是正式测量窗口上的结果，不是所有卡各自相加；`total_throughput` 包括输入和输出，不能与纯输出 tokens/s 混用。

正式请求未全部成功、结果为空或缺少必要指标时，记录为无效/失败，不给出伪成功结论；仍保留原始结果。普通失败默认清理后继续；`failure_policy: stop` 停止后续项。服务退出、超时或取消都执行本次进程清理，清理无法确认则停止整个计划。模型会话失败时，已完成的子项指标仍保留，会话状态与测量状态分别列出。

新默认模型和全量参数需在目标镜像验证。软件回归、原生评测器数据切分检查、HTTP客户端检查不等于已取得两模型1319题和31点完整实测结果；通用启动脚本也不宣称复现原表的硬件优化配置。部署参考见[Qwen3启动文档](https://github.com/HYGON-AI/inference-cookbook-das/blob/main/docs/model-deployment/sglang/qwen3.md)。

## 旧配置

旧的 `suite: e2e` 被解释为 `e2e_acc`；`testpacks/sglang_accuracy/manifest.yaml`、`profile`、`serving` 和 `ep_settings` 继续可用。新配置优先用上面的启动脚本方式。旧配置不会自动增加 `e2e_perf`，需要从模板加入新的测试块。
