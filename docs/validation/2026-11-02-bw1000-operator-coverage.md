# BW1000 新增用例验证

测试代码：`b41fd4d`。本记录只覆盖 Conv/GEMM 与 RCCL 新增入口，不代表其他 suite 已完成同轮验收。

## 环境

- BW1000_H ×8，gfx936，Driver 6.4.2-V2.1.0。
- 标准 SGLang 镜像：`0.5.12-ubuntu22.04-dtk2604-py3.10-20260909-0006`。
- DTK-26.04-rc4、Open MPI 5.0.3、rocBLAS 4.3.0.ef408db7-dirty。
- 自有容器带 `--init`，只读挂载宿主 `/opt/hyhal`；未替换运行库、修改频率或操作他人容器。

## 软件回归

Linux 容器内执行 `PYTHONPATH=src:tests:. python3 -m unittest discover -s tests -q`：244 项中通过 243、跳过 1。
此前 Windows 本机出现的三项旧进程生命周期测试失败，在此 Linux 环境没有复现。

## RCCL

十类通信的两卡冒烟和八卡标准配置均完成，原生校验通过。
八卡参数：half、8 KiB–64 MiB、倍增2、预热20、正式50；每个 MPI 进程一卡。
14 个尺寸共 238 条经原生校验的 size/mode 样本进入标准结果；AllToAll/AllToAllv/SendRecv 仅汇总 out-of-place，其余保留两种模式。

64 MiB out-of-place 示例，使用原生迭代平均值：

| 通信项 | 延迟(us) | algbw(GB/s) | busbw(GB/s) |
| --- | ---: | ---: | ---: |
| AllReduce | 1017.7 | 65.94 | 115.40 |
| AllGather | 527.5 | 127.22 | 111.32 |
| ReduceScatter | 545.6 | 122.99 | 107.62 |
| AllToAll | 606.9 | 110.58 | 96.76 |

Open MPI 5 的本机控制通信配置为 ob1、self/sm，并排除当前不需要的 hcoll；不使用无 HCA 时 HCOLL 初始化报错的冒烟轮次作标准性能结论。algbw 与 busbw 口径不同，这些数值不作为 PCIe 带宽或跨机器验收峰值。

## Conv

FP32、FP16 各完成 Conv2D 13组、Conv3D 15组，合计56组，全部原生校验通过并收集到3轮计时；完整 Conv 轮次状态 completed，控制器退出0。
每组先执行 MIOpen 原生验证，再预热10次、执行3轮×100次；记录三轮原生 GPU Kernel Time 平均值的中位数。
该镜像实际输出 GPU reference 校验结果，保留其 norm error 和原生容差，不冒称 CPU reference。

## GEMM

工具不支持附件要求的 `--test` 和 `--use_hipblaslt`，在数值计算前退出。
本轮经确认记为工具版本不兼容并跳过，不发布 GEMM 性能，也不将其记为 HCU 数值错误。
未删除参数、放宽 norm error 阈值或更换库来绕过错误。

## 中断与收集

八卡 RCCL 完成后，早期监测脚本遇到一个短暂、归属无法确认的 KFD 条目，安全停止了后续 Conv；该轮不记作整个计划通过。已完成的 RCCL 结果保留，Conv 全集另起新轮次。
随后按测试安排改为连续确认外部活进程有实际 HCU 显存占用才正常停止测试，不因 CI Worker、新容器或无具体 HCU Index 的短暂条目直接停容器。
原始日志、命令、标准结果、三轮计时和环境信息均保留在控制端本地归档。
已校验成功项的366个归档文件，文件大小与 SHA256 全部匹配，并导出56行 Conv 及238行 RCCL 明细 CSV。测试结束后原生进程正常退出，自有容器保留为空闲状态。
