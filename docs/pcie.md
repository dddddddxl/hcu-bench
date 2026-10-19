# PCIe 带宽测试

使用镜像中已有的 `rocm-bandwidth-test`，单机逐对运行五类路径：CPU→GPU、GPU→CPU、GPU→GPU、CPU↔GPU、GPU↔GPU。默认每次拷贝 512 MiB，原生迭代参数 `ROCM_BW_ITER_CNT=20`，同时记录 Avg BW 和 Peak BW。不安装工具、不编译、不改 DTK 或频率。

## 配置和运行

```bash
cp configs/pcie.template.yaml configs/bench.yaml
# 编辑 bench.yaml：工作目录、仓库路径、工具路径和设备编号
bash bench.sh check -c configs/bench.yaml --suite pcie
bash bench.sh pcie
```

使用综合容器模板时，已包含 `pcie_bandwidth`，`bash bench.sh all` 会按 YAML 顺序执行 RCCL、PCIe、memory、DeepEP、e2e。旧的个人 YAML 不会自动更新：从模板补入 PCIe 的 vars 和测试块。也支持既有 SSH/managed 容器接口，一个 PCIe 用例只选择一个节点；多台机器分别配置用例，不做跨机 PCIe。

先在**实际执行测试的容器**中运行：

```bash
/opt/dtk/bin/rocm-bandwidth-test -e
/opt/dtk/bin/rocm-bandwidth-test -t
hy-smi
hy-smi --showpids
```

`cpu_devices` / `gpu_devices` 是原生工具 Device ID，以逗号分隔的字符串填写。`nodes.gpus` 是 Bench 资源卡号，两者不是同一套编号。例如 1.85 上 HCU0～7 对应 Device 2～9，但其他节点不一定如此。必须根据 PCI 地址/UUID 核对映射，确保选中设备都是本次允许使用的卡。程序校验编号存在、CPU/GPU 类型和数量，不猜测 HCU 编号到 HSA 内存代理的映射；`HIP_VISIBLE_DEVICES` 不能代替原生工具的路径选择。存在 ROCR/CUDA 二次可见性掩码时会拒绝执行，须重新核对编号。

默认至少选择两张 GPU，以便覆盖五类路径。有其他业务时先协调，不以瞬时利用率 0 判定空闲。Bench 资源锁只协调本工具，不能阻止其他用户启动负载。

可修改的参数：

| 参数 | 含义 |
| --- | --- |
| `binary` | 已有原生工具的绝对路径 |
| `cpu_devices` | CPU 内存代理 Device ID，例如 `"0,1"` |
| `gpu_devices` | GPU Device ID，例如 `"2,3"`；须对应 `nodes.gpus` |
| `size_mib` | 单次拷贝大小，默认 512 MiB，范围 1～4096 |
| `iters` | 原生内部迭代参数，默认 20 |
| `repeats` | 测试块的独立重复次数，默认 1，可改成 3 |
| `timeout_s` | 整个路径扫描的故障时限，默认 1800 秒；不是压测时长 |

```bash
bash bench.sh pcie --set tests.pcie_bandwidth.params.size_mib=64
bash bench.sh pcie --set tests.pcie_bandwidth.repeats=3
```

本包不接受 `duration_s`。每轮串行遍历全部配置路径，CPU/GPU 单向正反分别测，GPU 单向覆盖有序对，双向覆盖无序对。2 个 CPU 代理、8 张 GPU 共 132 条命令，不包含 GPU 自拷贝，也不是八卡同时拷贝的并发总吞吐测试。

## 原生命令和结果

```bash
rocm-bandwidth-test -s SRC -d DST -m 512
rocm-bandwidth-test -b SRC,DST -m 512
```

不使用 `-a/-A` 的默认缓冲区大小，也不将 `-m` 与全矩阵模式混用。解析契约来自 2.6.0 逐对输出：检查方向、源/目标类型及编号、大小、Avg/Peak 列和单位。格式变化、空输出、仅打印帮助、错误信息或非零退出均记为失败，保留原始日志，不猜数值。

每个用例归档：

- `plan.json`：参数、工具及包装脚本 SHA256、相关环境变量。
- `devices.log` / `devices.json`、`topology.log`：本轮枚举与拓扑。
- 每条路径的 `.log`，以及记录命令、时间、退出码的 `execution.json`。
- `measurements.csv` 和五张分类 CSV：`cpu_to_gpu.csv`、`gpu_to_cpu.csv`、`gpu_to_gpu.csv`、`cpu_gpu_bidir.csv`、`gpu_gpu_bidir.csv`。均为源/目标逐行明细，包含 `size_bytes`、`avg_gbps`、`peak_gbps`。
- `assessment.json`：是否完成全部路径，不将性能测试标成正确性验证通过。

框架的 `metrics.csv` / `summary.json` 按路径、大小及 Avg/Peak 分开统计；三轮重复时分别统计原生平均值与峰值，不跨路径混算。双向结果已经是两个方向之和，不再乘 2。遇到中途失败，已完成路径的数据保留，但用例状态是失败，不代表全量完成。

正确性固定标为 `not_checked`，无默认性能门槛。超时/取消不算通过；进程清理无法确认时使用框架的停止机制，不能继续下一项。单独的 `-v` 校验和长时间稳定性测试不在本包范围。

## 验证范围

解析与命令规则参考 1.85 节点的 2.6.0 原生日志，并有合成进程测试覆盖。新接入的统一 PCIe suite 仍需真实节点跑一轮；本地回归不代表 GPU 实测通过。
