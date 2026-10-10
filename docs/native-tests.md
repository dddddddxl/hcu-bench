# 两项真实测试的编排

入口已经实现，但尚未在 HCU 节点验收。`configs/deepep-sglang.template.yaml` 按顺序执行：10 分钟 DeepEP 低延迟压测 → DeepSeek-V4-Flash INT8 的 GSM8K 100 题、5-shot。模板含节点/路径/网卡占位，不能直接当作生产配置运行。

```bash
hcu-bench check -c configs/deepep-sglang.template.yaml
hcu-bench plan -c configs/deepep-sglang.template.yaml
hcu-bench run -c configs/your-env.yaml --suite all
hcu-bench run -c configs/your-env.yaml --suite deepep
hcu-bench run -c configs/your-env.yaml --suite e2e
```

`check/plan` 仍是控制机静态检查，不验证远程依赖或路径。每个真实入口还支持 `--check-only`，在目标镜像中验证依赖/原生文件，且不会启动压测或模型服务。框架不安装镜像内依赖，不下载模型，不改 DTK，不停止其他业务。

## 部署准备

- 控制机按 README 安装 Bench；目标镜像需要现有 Torch/DeepEP 或 SGLang，以及 `psutil`，用于按进程身份清理本次子进程。Bench 的 `[native]` 是可选依赖，不会由框架自动安装。
- 将同一提交的 `testpacks/` 及相关文件放入各目标的 `bench_root`。初版仍不自动分发代码；容器启动脚本负责挂载，宿主机与容器目录分别配置。
- 提供容器 start/check/stop 脚本或切换 existing。确认 GPU 空闲、设备绑定、共享内存、memlock、驱动、网卡和 GID；工具不会猜实际网卡、拓扑或清理其他服务。
- 示例的 8 卡与 INT8 参数来自 BW1100 配方，不代表任意节点都能使用。若是 BW1000/其他硬件，先按节点环境确认配方，不能只改卡数。

## DeepEP 低延迟压测

原生入口由 `test_script` 指定，保留其 sibling `utils.py`，包装器不复制或改动原生测试。启动方式保持 `torchrun --nproc-per-node=1`，再由原生脚本 `--num-processes` 在每节点 spawn GPU workers；实际 GPU world size 为节点数 × num_processes，不能把 torchrun worker 数再次设为卡数。

默认 600 秒由 Bench `duration_s` 控制。原生 `--pressure-test` 如果提前成功结束，包装器重新启动；每次完成周期使用 `master_port + cycle`，避免连接上一周期的 TCP store，网络需允许这个端口范围。所有节点必须使用相同配置和原生代码，并由控制机同时启动。多机只扩展此测试的 nodes；精度测试仍只启动一个服务节点。

只重启成功完成的周期。SIGKILL、SIGABRT、正确性断言或 profiler 崩溃不会自动吞掉重试；控制机取消其他节点并保留日志。包装器只对可解析但数值无效的性能行记 invalid，不修改原生 profiler 的断言。低延迟子进程明确 unset DEEP_EP_NORMAL_MNVL；其余通信变量由配置提供。

预检要求显式提供 `ROCSHMEM_HEAP_SIZE` 和 `HIP_BUFFER_EXTRA_SIZE`。模板里的 `ROCSHMEM_IPC_MNVL` / `ROCSHMEM_GDR_DISABLE_XDP` 是超节点配置示例，不是所有硬件的必填项；IPC/IB 节点应依据实际镜像的后端选择配置，不能直接套用旧超节点环境。

指标保留 native_rank/cycle：dispatch+combine effective bandwidth、mean/min/max latency、dispatch/combine bandwidth、send/recv latency。`(total)` 分阶段指标使用独立名称，不和普通分阶段指标混合。不把这些指标称为 RCCL busbw；停止时长也不表示整项正确性通过。

原生脚本/工具 SHA256、软件版本、通信环境、每周期原始日志和 samples.jsonl 自动归档。压力日志没有识别到性能数据时不能算成功。

## SGLang GSM8K

服务启动 argv/env 放在独立 JSON profile 中；目前只有 Flash INT8、SGLang 0.5.12、BW1100 8 卡参考模板。模型必须已经在本地，不自动拉取远端代码或权重；配方的 trust-remote-code 由你确认模型来源后使用。网卡 YOUR_* 必须替换。

包装器检查模型 config、DeepEP config、数据格式、SGLang 0.5.12 版本和原生评测接口。参考配方要求显式绑定 8 张卡、容器内 memlock 为 unlimited，不满足时先报错，不擅自改宿主机权限。端口已占用就报错，不复用或杀掉现有服务。只启动自己的前台服务，等待 `/health`，运行评测，然后清理自己的子进程；服务退出/全空响应/样本不足都是执行失败，不伪装成精度结果。

复用该分支的 `GSM8KEval` 与 `run_eval_once`，未复制评分算法。按 cookbook HCU 用例使用 chat、temperature=0、top_p=1、max_tokens=2048；这些参数可配置。数据集前 5 条作为 few-shot，不计入后续 100 条评分。`num_examples: 0` 表示所有剩余题目。

`min_score: ""` 表示只报告真实分数，correctness 为 not_checked，不称为“精度通过”。以后设置字符串阈值，例如 `"0.90"` 才启用阈值判断；不套用 HCU Qwen 用例的 0.88。报告包含准确率、评测耗时、实际题数和空响应率，附模型 config/数据集/评测源码哈希、软件版本、回答 JSON、原生 HTML 和服务日志。

## 参考来源

- [DeepSeek-V4 部署配方，固定文档版本](https://github.com/HYGON-AI/inference-cookbook-das/blob/89cc09d13325a5fdb420c8e52c1148f902f3c6ae/docs/model-deployment/sglang/deepseek-v4.md)，选择 Flash Channel INT8 / BW1100 8x / 0.5.12，不混入 0.5.18 或 PD 配方。
- [0.5.12 HCU cookbook GSM8K 用例](https://github.com/HYGON-AI/sglang-das/blob/6c946e92fa4ec1755c8a2a8f8800bd2e6795213e/test/registered/hcu/accuracy/bw1100/test_cookbook_text_gsm8k_eval_hcu.py)。参考其 100 题、5-shot、chat 的评测调用，不复制模型表或阈值。
- [0.5.12 HCU 通用 GSM8K 用例](https://github.com/HYGON-AI/sglang-das/blob/6c946e92fa4ec1755c8a2a8f8800bd2e6795213e/test/registered/hcu/accuracy/bw1100/test_gsm8k_eval_hcu.py)，该用例原模型为 Qwen，不能直接改模型名就假定阈值也适用。

参考提交固定用于解释接口，不代表目标镜像一定包含同一套评测实现。实际评测源码哈希单独记录；若接口缺失或版本不符，明确失败，不偷偷升级镜像。

本地测试包含标准结果解析、原生子进程树、服务就绪/占用端口/失败清理，以及 CPU 替身的压测和评测入口联调；替身数据不代表真实 GPU 性能或模型精度。完整测试需在开发环境安装可选 `[native]` 依赖，测试不会安装 Torch 或 SGLang。
