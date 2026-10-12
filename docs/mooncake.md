# Mooncake RDMA / VRAM 压测

本测试包调用已有的 `transfer_engine_bench`，测试 GPU 显存 buffer 间的 RDMA read/write 吞吐，不启动模型。默认只报告性能；数值正确性始终为 `not_checked`。不将 Mooncake 吞吐称为 RCCL busbw，也不套用其他集群的 130 GB/s 作为默认标准。

## 准备条件

- 所有目标节点使用 Mooncake transfer-engine **0.3.12+ 的同一完整 distribution 版本**。本包不支持旧 0.3.10 P2PHANDSHAKE 路径，不自动升级或安装依赖。
- 已有 Python 3.10+、psutil、支持 GPU VRAM 的 `transfer_engine_bench`，以及可用的 DCU/DTK/RDMA 环境。预检会读取 native help 检查所需 flags。
- 每节点显式配置可用的 `HIP_VISIBLE_DEVICES`、`MC_TE_FILTERS` 和 peer 可达的 IPv4；NIC 必须 ACTIVE / LINK_UP。运行前自行确认选中的 GPU/NIC 未被业务占用，包不会重置设备或停止其他用户进程。
- 每节点都有同一提交的仓库。`shared_dir` 必须是所有节点/容器能读写**同一份内容**的共享目录，例如 NFS；仅填写相同路径但内容不共享不能工作。
- 容器需提供 DCU、RDMA 设备、足够 memlock 和 peer 可达网络；设备、挂载和权限由你现有的创建脚本配置。

## 双机执行

复制 `configs/mooncake.template.yaml` 为本地 `configs/bench.yaml`，填写 SSH alias/IP、现有自有容器名称、代码/共享目录、卡号、NIC 和地址。`nodes: [target, initiator]` 与 `addresses` 的 IP 顺序必须一致。

```bash
bash bench.sh check -c configs/bench.yaml --suite mooncake
bash bench.sh plan -c configs/bench.yaml --suite mooncake
bash bench.sh mooncake
bash bench.sh mooncake --set tests.transfer_pair.params.duration=600
bash bench.sh mooncake --detach
```

框架 `check/plan` 是静态检查。镜像内可以对每个 rank 执行目标侧预检：

```bash
HIP_VISIBLE_DEVICES=0 MC_TE_FILTERS=mlx5_0 python3 testpacks/mooncake/run.py \
  --output-dir /shared/check/node0 --shared-dir /shared/rendezvous \
  --run-id check --case-id pair --node-rank 0 --nnodes 2 \
  --addresses 10.0.0.1,10.0.0.2 --check-only
```

替换真实地址、路径和 GPU/NIC。此命令只读取环境、保存预检证据，不启动 native 负载或创建 rendezvous。IP 可达性、实际显存注册和握手仍需真实运行验证。

宿主机 Python 过旧时，也可让 SSH executor 的 `nodes.<node>.python` 指向一个已有自有容器的解释器入口脚本：

```sh
#!/bin/sh
exec docker exec -i YOUR_OWN_CONTAINER python3 "$@"
```

此时 agent 已在容器内，测试不要再设置 `container`，`work_root`、`bench_root`、`shared_dir` 都使用容器内路径；每节点的 `work_root` 仍应独立。脚本需在各宿主机可执行，控制机仍使用 SSH 密钥。它不需要向宿主机安装 Python，也不需要给容器挂 Docker socket。自建长驻容器建议 `--init`，使 orphan 子进程退出后能被及时回收。

## 生命周期与超时

1. 各节点预检后，在 `shared_dir/run_id/case_id` 创建独占 rank join，核对拓扑、native 参数和 Mooncake 版本。
2. 启动本节点所需的 target；轮询它本次日志里的 `listening on IP:port`。endpoint 通过原子 JSON 发布；不写死或复用上次端口。
3. initiator 等待自己 target 的 endpoint，启动 native 压测。默认 `duration=180` 秒，使用原生 `--duration`，得到完整的最终结果后才能结束。
4. 每个 initiator 必须退出 0、产生恰好一条有效 `Test completed`、完成请求的测量时间窗。缺失/重复结果、非正或非有限值、native ERROR/FATAL、错误退出和超时均失败。
5. initiator 清理确认后发布 done；target 等待所有打向它的 initiator 完成后停止。环形节点同时拥有两种角色。任何失败/取消都只清理本次 owned 进程；清理无法确认返回 125，框架停止后续用例。

**不要设置框架 `duration_s`。** 本 manifest 声明 `duration_control: native`，planner 会拒绝该组合，防止在最终吞吐输出前按时间强制结束却被当作完成。native `duration` 只算传输窗口，不含启动、注册和清理；`startup_timeout` 限制 join/endpoint 等待，`duration + completion_grace` 限制 initiator 总等待。外层 `timeout_s` 应更大，例如 180 秒测量使用 600 秒外层故障时限，长压测需同步调大。

不会按进程名称执行 pkill。预检发现本容器命名空间内已有 native benchmark 时会拒绝启动；跨用户/容器资源协调仍依赖节点管理和共享 Bench 锁。共享 NFS 上各节点的 host `work_root` 应不同，避免独立节点的相同 GPU 编号争用同一个锁文件。

## 参数与拓扑

| 参数 | 默认 / 含义 |
| --- | --- |
| `topology` | pair；target 在前、initiator 在后，恰好两节点 |
| `duration` | 180 秒，native initiator 测量窗口 |
| `gpu_id` | 0，显式可见卡中的逻辑编号；-1 使用全部可见卡 |
| `buffer_size` | 1 GiB，分配语义由记录的 native 版本决定；多卡/环形需另核显存余量 |
| `threads / batch_size / block_size` | 12 / 128 / 65536 字节 |
| `operation` | read，可选 write |
| `startup_timeout / completion_grace` | 90 / 120 秒 |
| `expected_version` | 空；可填写完整安装版本做额外精确匹配 |
| `min_bandwidth_gbps` | 空，只报告；填字符串数值后逐 initiator 判断 decimal GB/s 下限 |

多拓扑均复用同一 manifest：

- `fan_in`：rank 0 为 target，其余 rank 同时打向它。
- `pairs`：节点数量必须为偶数，rank 1→0、3→2 等独立配对同时运行。
- `ring`：每个 rank 同时运行 target 与 initiator，initiator 指向下一 rank，最后一台指向 rank 0。需为双角色 buffer 预留显存。
- 四节点六个交叉对：配置六项 `topology: pair` 测试，每项仅选对应两个节点并填写对应两个地址。框架顺序执行六项；不声称自动实现跨用例的三波并行。

节点列表、IP 顺序、卡号和 NIC 要一起配置。`matrix` 可以扫描 block/buffer/threads，`repeats` 重复独立测试；每次 run/case 的 endpoint 隔离。调优用的 `MC_SLICE_SIZE` 可在 YAML env 中显式提供。

## 结果与判定

initiator 上报 native `transfer_throughput`（保留 GB/s 或 GiB/s）、`transfer_duration`、`transfer_batch_count`。如配置阈值，将 GiB/s 明确换算为 decimal GB/s 后比较；低于阈值产生真实 measured 样本、`performance_judgement=failed` 并非零退出，不将性能回退伪装成数值错误。未设置阈值则 `report_only`。

passive target 不产生吞吐指标，上报 skipped coordination 记录；本包因此 `requires_metrics: false`，但包装器自身对所有 initiator 严格要求最终结果。每个 initiator 的吞吐独立保存，多对一不自动求和或把不同时间窗口当作一个聚合测试。

框架保存 `summary.txt/json`、`metrics.csv`、`results.jsonl`；`raw/` 收集各节点的 `environment.json`、`commands.json`、`target_endpoint.json`、`measurement.json`、`summary.json`、`cleanup.json`、`error.json` 和 target/initiator 原始日志。环境包含完整包版本、二进制/包装器 SHA256、可见设备、NIC 状态和参数，便于复现。

只有所有参与节点成功完成且清理确认，整项才完成。仍需根据本集群实际卡数、NIC 速率、块大小和协议建立基线；单卡单 NIC 的结果不能使用八卡八 NIC 历史阈值解释。
