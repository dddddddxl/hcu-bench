# 内置训练入口验证

## 范围

2026-11-09，基于main的f616059，新增仓库自有训练循环和分类模型配方注册表。
使用已安装TorchVision ResNet50 weights=None，不复制原modelzoo训练代码、不下载权重或数据。
FP32不依赖Apex；旧外部train_script仅保留显式兼容，不是默认要求。

部署包没有reference目录，实跑YAML没有train_script或script_sha256，只有内置recipe及数据配置。
实际结果为training_backend=builtin，证明不是误用旧外部脚本。

## 环境与实测

- nmz3单卡BW1100/gfx938，物理HCU2，PCI0000:18:00.0，renderD130。
- 镜像42.228.13.241:5000/dcu/admin/base/pytorch:2.7.1-ubuntu22.04-dtk26.04-py3.10。
- image ID sha256:964e8e47533cd2cd686ec310e0cf9a37fccaf95c7636f195b5867bf9f61eedfc。
- 自有容器8CPU/12GiB/2GiB shm、无网络、cap-drop ALL，代码/数据/hyhal只读，输出单独可写。
- FP32、batch32、workers4、seed1、预热10步+计时50步；1920张训练/1000张验证真实图像，覆盖原1000类。
- run ID `20261109_103028_b2f3356a`，completed，三个执行阶段退出码0/0/0、cleanup均confirmed。
- 同步计时窗口8.761950秒，1600张正式图片：**182.6078 images/s，175.2390 ms/步**。
- 训练进程21.525秒，含验证与存盘；控制器完整流程131.269秒，含数据准备及检查，不计作吞吐窗口。
- 训练loss8.5442、验证loss22.91，Top1/Top5=0.1%/0.6%，重载打印指标一致。
- checkpoint204827228字节，SHA256 a2d6e475da99d16f607e8a46b7a470d68699098a860af7049b148a33f9006a67。
- 320个模型张量、161个优化器状态有限，首层权重已改变，strict state_dict检查通过。

实现及后端选项不同，不将此数字直接与旧外部入口吞吐比较；没有认定旧loss根因已修复。
correctness=not_checked、training_convergence=not_assessed，小子集精度不是ImageNet收敛验收。
AMP、多卡、90轮和其他模型未执行，其他模型需要配方/测试接入后另行验证。

## 回归和清理

Linux全套371项：363通过、8个可选ONNX/EvalScope依赖跳过。
首次无GPU的CPU容器未挂载hyhal，新增真实CPU更新测试因缺librocm_smi64失败；保留日志，
只读挂载宿主已有/opt/hyhal后完整重测通过，未改库、安装依赖或跳过该测试。
本地training合同13项中12通过、1项因无Torch跳过，主/独立配置10项通过。
README顶部支持表已明确列出training；默认all仍是27个父用例，仅硬件验证本训练项，不重跑整套all。
启动前两次核实选中卡空闲，守卫检查外部存活正显存PID、温度和RAS；不干预其他任务、频率或NUMA。
终态无自有正显存进程，仅剩sleep后停止专用容器；容器shutdown的137不是训练退出码。
原生日志、配置、部署源码、标准结果、输入/权重哈希、首次失败与重测日志及清理回执归档；不打包数据图片或大权重。
