# PCIe 接入验证（2026-10-15）

- 入口：`bash bench.sh pcie`；综合模板的 `all` 包含 PCIe。
- 默认：512 MiB，`ROCM_BW_ITER_CNT=20`，单轮扫描五类路径。
- 13 项 PCIe 专项测试通过，覆盖设备/参数检查、方向和大小解析、Avg/Peak 分离、五类路径覆盖、非零退出、空结果/帮助文本、取消和清理失败。
- 本地 Windows 全仓最终回归：133 项，131 通过，2 项 POSIX 专用测试跳过。
- 中途一次全仓运行出现原有 Mooncake ring 进程退出测试偶发失败（`exit=15, owned_actions=[]`），未修改该模块；随后完整重跑通过。该平台时序问题仍需留意，不能将重跑通过理解为已修复。
- 另用 1.85 已归档的五类原生 64 MiB 逐对日志检查解析，平均/峰值及双向合计均一致。旧日志解析不是新 512 MiB suite 的硬件验证。
- 新 PCIe suite 尚未在真实 HCU 上执行；容器内工具、设备映射和空闲情况仍须在目标节点确认。没有新增独立正确性或长时间稳定性测试。

执行命令：

```text
.venv/Scripts/python.exe -m unittest discover -s tests -p test_pcie.py -v
.venv/Scripts/python.exe -m unittest discover -s tests -q
.venv/Scripts/python.exe -m hcu_bench check -c configs/in-container.template.yaml --suite all
```
