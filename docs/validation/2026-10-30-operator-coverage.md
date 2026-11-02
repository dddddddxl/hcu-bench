# Conv/GEMM 与 RCCL 扩展验证

## 接入范围

- `operators` 增加 Conv2D/Conv3D：28 个 CSV 用例，fp32/fp16 共 56 组。
- `operators` 增加通用 GEMM：14 个 CSV 用例，fp32/fp16/bf16 共 42 组。
- `rccl` 从 AllReduce 扩展到 10 个单机 MPI 通信项；默认参数保持兼容。
- 原始 Conv/GEMM 附件随仓库提供，已获提供者公开分发授权；来源与 SHA256 见 `vendor/sources.json`。

## 判定规则

Conv 使用 MIOpen 明确的校验结论及其报告的容差。
GEMM 默认仅零 norm error 判通过；非零但未配置阈值时记未确认，跳过性能。
每项性能测量必须先完成同一配置的原生检查，不以进程退出码替代正确性。

RCCL 检查参数、消息尺寸、rank/BDF、wrong 数、页尾及日志指纹。
AllToAll/AllToAllv/SendRecv 的 in-place 为原生 `N/A`，只汇总经校验的 out-of-place；其他通信项保留两种模式。

## 软件验证

以下是无 HCU 的接口回归，不是实测性能：

95 项定向回归中通过 94 项、跳过 1 项 POSIX 入口；独立 RCCL 插件 12 项回归通过。

- 模板展开、原始 CSV 覆盖、命令生成、严格校验、错误输出拒绝。
- GEMM 校验未确认时不运行性能；结果尺寸不匹配时不发布指标。
- RCCL 十类原生表格布局、尺寸对齐、root 和失败页尾。
- 统一编排到报告、原生 CSV/日志归档；原 AllReduce 独立插件兼容。
- 所有 8 份 vendored 源码目录校验和通过。

本机进程超时/计划结束/同伴取消的三项旧测试未通过；未修改的 main 基线 `ab90682` 在同一环境中也复现。这三项不作为本次新增功能的通过依据，也未为本机环境修改 Linux 进程管理逻辑。

## 实机状态

2026-10-30 检查授权节点时均发现 CI 任务，未启动任何 HCU 负载，也未进入或修改他人容器。
新增 Conv/GEMM、九类新增 RCCL 的目标镜像兼容性及实际性能尚待空闲节点验证。
软件回归通过不能替代这一步；不得将本记录用于宣称整套 `all` 已验收。

后续实测见 [2026-11-02 BW1000 验证](2026-11-02-bw1000-operator-coverage.md)。

## 运行

已有 Linux 容器内，从 `configs/numeric-operators.template.yaml` 或 `configs/rccl.template.yaml` 复制为 `configs/bench.yaml`，填写目录、分配卡号及已有工具路径：

```bash
bash bench.sh operators
bash bench.sh rccl
bash bench.sh all
```

综合容器模板已经包含新增测试块；旧的个人 YAML 不会自动增加测试项。
