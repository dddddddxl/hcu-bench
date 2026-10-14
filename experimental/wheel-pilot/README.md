# RCCL external-result wheel pilot

这个独立试点提供 `hcubench-core` 与 `hcubench-communication-rccl` 两个 Python wheel，统一测试配置、执行计划、证据检查和结果报告。用户负责 MPI、节点、容器、设备选择和超时回收。

新命令是 `hcubench`。原有 `hcu-bench` / `bench.sh` 不变，原 RCCL 模拟 suite 不会因此变成真实测试。此目录保留 experimental 定位；源码和接口检查不能代替硬件性能验收。

## 安装

需要 Python 3.10+。在仓库根目录执行：

```bash
python -m pip install ./experimental/wheel-pilot/core ./experimental/wheel-pilot/rccl-plugin
hcubench list
hcubench describe communication.rccl_allreduce
```

离线交付可先构建 wheel：

```bash
python -m pip wheel --no-deps ./experimental/wheel-pilot/core ./experimental/wheel-pilot/rccl-plugin --wheel-dir ./wheelhouse
# 在目标 Python 环境中安装
python -m pip install --no-index --find-links ./wheelhouse hcubench-core==0.1.0a3 hcubench-communication-rccl==0.1.0a3
```

wheel 不包含 MPI、DTK、RCCL 或 rccl-tests 二进制，不安装或替换 Torch。

## 测试与导入

1. 在测试环境准备 MPI-enabled `all_reduce_perf` 和兼容的动态库。参考实现为 [ROCm/rccl-tests](https://github.com/ROCm/rccl-tests)，当前解析器对应 `ae3e635` 的输出格式；这里没有复制或分发原生源码。
2. 复制 `examples/rccl/allreduce.json` 到工作目录，修改 `binary` 为实际绝对路径。默认单机 2 个 rank、每 rank 一线程一卡，half/sum，8 KiB～64 MiB、步长 2、预热 20 次、迭代 50 次。
3. 校验参数并生成计划：

```bash
hcubench check allreduce.json
hcubench plan allreduce.json --output plan.json
```

`check` 只检查配置，不证明节点、库或设备状态正常。计划中的 argv 是使用者需要审阅的可执行输入；core 不启动它。

4. 使用自己的 Runner，或复制 `examples/rccl/collect.py` 到工作目录使用。此脚本未打入 wheel，它会启动实际 MPI 子进程、保存 stdout/stderr/退出码并采集环境。它假定 DTK 版本文件位于 `/opt/dtk/.dtk_version`；其他布局需由 Runner 适配。示例不负责超时、取消、节点管理或容器清理。

准备 `host-environment.json`，填入真实宿主、镜像和物理卡信息：

```json
{
  "host_name": "replace-with-actual-host",
  "image_id": "replace-with-actual-image-digest",
  "physical_indices": [0, 1]
}
```

检查两张卡空闲，确保 `mpirun`、动态库和参数与配置对应，然后执行：

```bash
export HIP_VISIBLE_DEVICES=0,1
python collect.py plan.json --bundle first-bundle --environment host-environment.json
hcubench import-result allreduce.json --bundle first-bundle --output results
```

新运行使用新的计划文件和 bundle 目录，示例拒绝覆盖既有目录。即使测试进程失败，也可以导入已有 bundle 保存失败状态。

5. 打开 `results/<run-id>/report.html`。同目录包含 `result.json`、`measurement.json`、`correctness.json`、`results.csv` 和 `raw/` 原始证据。导入、报告和参数描述可在无 HCU 的环境运行。

```python
import json
import hcubench

with open('allreduce.json', encoding='utf-8') as f:
    config = json.load(f)
directory, result = hcubench.import_result(config, 'first-bundle', 'results')
assert result['evidence_valid'], result.get('issues', result.get('error'))
```

## 插件与结果合同

插件通过 `hcubench.plugins` entry point 注册，使用 `hcubench.external.v1`，提供 `schema`、`validate`、`preflight`、`plan`、`assess` 和 `export_csv`。已安装插件是受信任 Python 代码，不是沙箱。core 同时保留 `hcubench.operator.v1` 接口，但本 PR 没有打包算子实现。

外部 bundle 必须含 `execution.json`、`stdout.log`、`stderr.log`。execution 字段：

| 字段 | 约束 |
| --- | --- |
| schema_version | 1 |
| config_hash | 来自当前计划 |
| argv | 实际参数列表；首版要求与计划完整一致 |
| returncode | 真实整数退出码，缺失则执行状态 unknown |
| started_at / finished_at | 实际执行时间 |
| environment | hostname、MPI/DTK 版本、测试二进制和 RCCL 库 SHA256、通信环境变量 |
| sha256 | 两份原始日志的 SHA256 |

`case_id` 包含逻辑测试参数；预热/迭代进入 `config_hash`，二进制安装路径不进入身份。实际二进制/库哈希及环境变量保存在 `environment_fingerprint`。单靠 case_id 不能证明两次测量可比。用户提交的日志和元数据不是可信执行证明，哈希只用于一致性检查。

目前支持单机 2～8 rank、AllReduce/sum、half/float/double，两种缓冲区模式分别存储。数据为 `iteration_aggregate`，保留原生 us、algbw、busbw 和错误数，不伪造逐次样本或 P50/P99。busbw 是原生集合通信公式折算值，不是硬件链路计数器。

接入同时检查实际参数、rank/不同 PCI 设备、表头单位、预期消息点和模式完整性、错误数、结束汇总、返回码、命令和文件哈希。communicator split、未知格式、日志缺行、非有限值及已知 kernel launch-bounds 告警会被拒绝。失败时仍保留原始日志和已解析结果。

状态分开保存执行、正确性、证据有效性；`acceptance=pilot_only`，没有性能阈值或正式性能通过结论。未收集逐次样本，因此稳定性为 not_assessed。未知 stderr 诊断仍需要人工检查。

## 验证状态与已知限制

单机双卡曾产生完整 14 个消息点 × 两种模式，但原生校验 kernel 出现 launch-bounds 告警，即使退出码和 wrong 都为 0，也只作为诊断数据。增加 `--gpu-max-threads-per-block=512` 后的新二进制已编译，尚待空闲设备完成硬件复测；目前没有接受为有效测量的真实运行。

此编译参数针对已观察到的 DTK/原生校验 kernel 组合，其他编译器需要先检查参数支持和实际 launch bound。OpenMPI 5 单机共享内存组件名称为 `sm`；不同 MPI 版本应由外部 Runner 适配。试点尚未验证多机/RDMA，也未接入其他 collective。

原型的完整本地合同套件曾在 Windows/Linux 各通过 28 项，其中 16 项属于未纳入本 PR 的算子试点；本 PR 的 RCCL 套件独立执行：

```bash
python -m unittest discover -s experimental/wheel-pilot/tests -v
```

测试使用明确标注的合成日志验证拒绝路径，不能替代真实硬件验收。公开仓库不包含本地节点信息、原始运行日志、第三方原生源码或二进制。
