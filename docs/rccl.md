# RCCL 统一测试入口

RCCL 已通过 `testpacks/rccl/manifest.yaml` 接入主框架，无需安装实验 wheel。
从两份容器模板复制为 `configs/bench.yaml`，填写顶部变量后执行：

```bash
bash bench.sh rccl       # 单独测试
bash bench.sh all        # 与 YAML 中其余启用项串行执行
```

已有个人 YAML 不会自动升级。加入模板中的 `rccl_binary`、`rccl_ranks` 和
`tests.rccl_allreduce`；自动容器模式还需 `container: test_image`。
远程 SSH 也使用同一个测试包，`bench_root` 和 `binary` 填目标执行环境路径。

## 前提与参数

- 当前支持单节点、2–8 卡 AllReduce sum，每个 MPI 进程一线程一卡；不是多机或 AllToAll。
- `hcus` 明确列出本次空闲卡号，`ranks` 必须等于其数量；不另叠加 ROCR/CUDA 可见卡掩码。
- 镜像须已有 Python 3.10+、psutil、Open MPI、DTK/RCCL；控制端还需 PyYAML。
- `binary: auto` 使用仓库内 rccl-tests `ae3e635`，首次自动编译 MPI 版 `all_reduce_perf`，之后复用缓存。
  也可显式指定已有二进制。构建使用镜像已有 make、hipcc、Open MPI 和 DTK/RCCL，不下载或升级库。
- 默认 8 卡、half/sum、固定 16 GiB 消息，等价于 `mpirun -np 8 all_reduce_perf -b 16G -e 16G -f 2 -g 1` 的卡数和消息配置。
  YAML 的 `min_bytes`、`max_bytes` 均为 `17179869184`；原生 `G` 按 1024³ 字节换算，这是每个 rank 的消息大小，不是八卡合计。
  保留预热 20 次、测量 50 次及原生数据校验；一次产生 out-of-place、in-place 两条测量记录。
  参数允许设置到 16 GiB；调整范围时 `max_bytes` 必须落在倍增序列上。完整有限轮次测试不接受 `duration_s`；超时由 `timeout_s` 控制并记为失败。
- MPI 显式使用 `--host localhost:<ranks>`，为本机提供与进程数一致的 slot；默认 `localhost:8`，不会继承集群分配去其他节点启动进程。

## 结果与失败

复用独立试点的严格解析器，检查完整参数、全部 size/mode、不同卡的 rank 映射、退出码、
日志哈希、环境指纹、两种模式的 wrong 数和正确性页尾。内核 launch-bounds 警告也视为失败。
空结果、截断、校验失败、取消或超时均不会作为成功数据输出。

`metrics.csv` 的指标名包含字节数和模式，例如 `allreduce_17179869184B_out_of_place_busbw`。
不同消息大小、in-place/out-of-place 不混合平均；latency 为原生迭代平均值，不伪造 P99。
RCCL busbw 不能直接当成 DeepEP effective bandwidth。

原始产物随框架归档：`plan.json`、`execution.json`、`assessment.json`、`measurements.csv`、
`stdout.log` 和 `stderr.log`。为沿用原生进程跟踪，stdout.log 合并保存 stdout/stderr，
stderr.log 为空占位；execution.json 明确记录这一布局，警告不会被丢弃。
普通失败按 `failure_policy` 继续或停止；进程清理无法确认时退出码 125，停止剩余计划。

独立 [wheel 试点](../experimental/wheel-pilot/README.md) 的外部导入功能保持不变。
