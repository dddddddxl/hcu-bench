# ResNet50 单卡训练流程验证

## 结论

2026-11-06 在 nmz3 的单张 BW1100/gfx938 上，FP32 真实 ImageNet 子集训练、
验证、checkpoint 保存、CPU 状态检查及重载验证均完成，原生三个阶段退出码均为 0，
清理 confirmed。**仅验证流程可运行，不是完整 ImageNet 精度、收敛或性能验收。**

验证 loss 明显偏大，最后一版约 `3.0777e11`，训练质量不能判为合格。
未通过调整学习率、删除日志或放宽 NaN/Inf 检查隐藏该现象。
原因尚未定位；小数据和小 batch 沿用原配方 lr0.1 可能影响训练稳定性，但尚无对照实验。

## 环境与输入

- 仓库基线：`dc5fff3ae40ef1c6df9291021caa2400734961c8`，叠加本次 training 包。
- 用户确认使用从头训练，不加载预训练权重；原 `train.py` 保持不变。
- 原脚本 SHA256：`ef9f17962c7a2f0a77a78e28f7b5c68275e4a935b97efbd857a8e3c1db914f9d`。
- 镜像：`42.228.13.241:5000/dcu/admin/base/pytorch:2.7.1-ubuntu22.04-dtk26.04-py3.10`。
- image ID：`sha256:964e8e47533cd2cd686ec310e0cf9a37fccaf95c7636f195b5867bf9f61eedfc`。
- Torch 2.7.1、TorchVision 0.22.0、Apex 1.5.0，均使用镜像已有版本，未升级框架。
- 物理 HCU2 / PCI `0000:18:00.0` / `renderD130`，容器内逻辑 HCU0。
- 容器仅开放选中 render 和 KFD，8 CPU、12GiB 内存、2GiB shm、无网络、cap-drop ALL。
- 数据及训练源码只读，输出独立可写；不修改共享数据、宿主 NUMA、频率或功耗限制。

## 范围与结果

固定 `smoke`、1 epoch、batch8、workers2、FP32、seed1、SGD lr0.1，
从前 8 类各取 8 张训练和 2 张验证图，合计 64/16 张真实图片，保留原标签，仍为 1000 类输出。
不是全量数据；分数不能和标准 ImageNet 验收分数比较。

| 项目 | 首轮 | 最终版本 |
| --- | --- | --- |
| run ID | `20261106_150124_7dc0de05` | `20261106_150827_4ceda3ba` |
| 状态 | completed / cleanup confirmed | completed / cleanup confirmed |
| Top1 / Top5 | 12.5% / 62.5% | 12.5% / 62.5% |
| 训练最后报告平均 loss | 41.352 | 42.461 |
| 验证最后报告平均 loss | 约 6.2460e12 | 3.0777e11 |
| 训练进程墙钟时间 | 14.240s | 13.420s |
| checkpoint 重载验证 | 一致 | Top1、Top5、打印的 loss 均一致 |

时间包含验证和存盘，不是纯训练吞吐。两次 loss 不同，不宣称固定 seed 保证逐位可复现。
最终版本补充了验证 loss 的 NaN/Inf 拒绝和结果记录，随后重新完成 CPU 回归与单卡短测。

最终 checkpoint 为 204827427 字节，SHA256：
`85910419fdb0b7457deec53d2c82eaef3f747527483317ceef833a4d0969724c`。
CPU 检查确认 epoch1、严格匹配 ResNet50 state_dict、320 个模型张量及 161 个优化器状态均有限，
首层权重不同于按同一 seed 构造的初始状态。
这些是流程和结构检查，不是独立数值基准；`correctness=not_checked`、`training_convergence=not_assessed`。

## 回归与边界

- 最终 Linux CPU 回归：363 项，355 通过、8 项可选 ONNX/EvalScope 依赖跳过；新增 6 项 training 合同测试包含在内，不重复累加。
- Windows 全套回归未通过，存在缺 Bash、Windows 路径和进程清理相关失败；不宣称跨平台全绿。新增 6 项合同测试在 Windows 通过。
- AMP、多卡、全量数据和 90 epoch 均未执行。本记录为早期流程smoke，不是同步窗口吞吐结果；后续用户要求将独立的训练吞吐短测加入默认all。
- 三阶段退出 0 后确认无自有显存计算残留，再停止只剩 sleep 的专用容器；容器停止的 137 不代表训练退出码。
- 未创建定时任务，未自动合入 main。

## 证据

原始日志、标准结果、实际 YAML、原脚本、部署代码、镜像身份及清理回执已归档。
证据包 `resnet50-training-evidence.tar.gz` 共 2169 个文件，逐文件大小与 SHA256 全部核对，
部署的 8 个关键代码/配置文件与本地工作树相同。
归档 SHA256：`bd1b81c38b38d034dec311d488fec0d4954c5c733dc46eadff21b063c02f9f6f`。
不包含 ImageNet 图片和大 checkpoint，图片来源逐文件清单及 checkpoint 大小/哈希保留，权重留在节点本次 case 中。

使用方式和完整配置见 [training 说明](../training.md)。
