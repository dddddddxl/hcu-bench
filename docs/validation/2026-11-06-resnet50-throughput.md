# ResNet50 默认训练吞吐短测

## 本轮结论

按用户要求，以短测观察单卡训练吞吐，不执行90轮训练。
2026-11-06 在 nmz3 单张 BW1100/gfx938 上，默认FP32、batch32、workers4、
预热10步+正式50步完成：**183.558 images/s，平均174.331 ms/步**。
完整受保护验证流程约80.49秒；该总时长包含数据准备、验证、checkpoint检查和重载，不用于计算吞吐。
仅验证该环境该规模，不代表其他平台或完整ImageNet训练性能、精度、收敛通过。

## 输入与计时

- 仓库基线 `dc5fff3ae40ef1c6df9291021caa2400734961c8`，叠加本次 training 包与默认all配置。
- 原train.py保持不变，SHA256 `ef9f17962c7a2f0a77a78e28f7b5c68275e4a935b97efbd857a8e3c1db914f9d`。
- 镜像 `42.228.13.241:5000/dcu/admin/base/pytorch:2.7.1-ubuntu22.04-dtk26.04-py3.10`。
- image ID `sha256:964e8e47533cd2cd686ec310e0cf9a37fccaf95c7636f195b5867bf9f61eedfc`。
- Torch2.7.1、TorchVision0.22.0、Apex1.5.0，未安装或替换框架、SDK、原生库。
- 物理HCU2 / PCI `0000:18:00.0` / renderD130，容器内逻辑HCU0。
- 8 CPU、12GiB内存、2GiB shm、OMP_NUM_THREADS=4、OMP_WAIT_POLICY=PASSIVE。
- ResNet50随机初始化、FP32、SGD lr0.1/momentum0.9/weight_decay1e-4、224裁剪、seed1。
- 真实ImageNet原1000类，每类1或2张训练图，共1920张，正好60个完整batch；每类1张验证图，共1000张。标签不变，不用FakeData、不加载预训练权重。

PyTorch optimizer post-hook只观察更新，不替换原训练循环或SGD数值实现。
第10次更新完成后同步设备并记录perf_counter，第60次完成后再次同步并记录结束。
正式计时8.716569秒，图片数1600，吞吐=1600/8.716569。
窗口包含真实加载/增强、H2D、前向、loss/精度统计、反向、SGD和原日志开销；
排除初始化、预热更新、验证和存盘，不直接使用原日志Time的四舍五入值。
仅在首尾额外同步，未逐步添加同步；比较时需保持数据、CPU/worker、batch与精度一致。

## 标准结果

- run ID `20261106_160944_bc28f324`。
- 标准状态completed，原训练、CPU checkpoint检查、原生重载验证退出码0/0/0，cleanup均confirmed。
- 观察到恰好60次优化器更新，无部分窗口、超时或计划终止伪装成功。
- 训练进程墙钟36.836秒，含原生验证与存盘，不是计时窗口长度。
- 训练最后报告平均loss8.1667，验证loss56.921，均有限。
- 小子集Top1=0.1%、Top5=0.4%，重载Top1/Top5/打印loss一致；不是完整ImageNet准确率。
- 模型320个张量及优化器161个状态均有限，首层权重相对seed初始化已改变。
- checkpoint204827427字节，SHA256 `272c52a1c2430bf693400c2d7cce7d04f4db1b663a00ea9adb5e5112c2a416ae`。
- `correctness=not_checked`、`training_convergence=not_assessed`；不因loss有限或有吞吐就宣称训练质量合格。

早期8类、batch8流程smoke的验证loss很大，证据仍保留在[早期记录](2026-11-06-resnet50-training.md)。
该轮改变了数据覆盖和batch，不能据此认定旧loss根因已修复，也未把旧耗时当新吞吐。

## 回归、默认配置与清理

- Linux全套368项：360通过、8项可选ONNX/EvalScope依赖跳过；新增training和主模板测试包含在内，不重复累加。
- Windows本轮training合同10项、主/独立配置10项通过，未宣称Windows全套通过。
- 新主模板 `resnet50_pretraining.enabled: true`、suite training，加入all，共27个父用例；单独training仅选1项，禁用训练块后恢复26项。
- 默认native_timeout_s300秒、外层900秒是故障上限，不是计划运行时长；缺数据、外部源码、依赖或执行失败明确失败，不自动降级/静默跳过。
- 90轮、AMP、多卡、完整数据均未执行；90轮需要显式full及长超时配置。
- 只映射选中卡，源代码和数据只读，自有输出可写；启动前两次空闲检查，运行期守卫检查外部存活PID正显存与RAS/温度。
- 不干预其他用户进程、频率、功耗或宿主NUMA。结束后无自有正显存进程，容器仅剩sleep后停止；该容器停止137不是训练退出码。
- 没有创建定时任务，也未运行整套all的实机复测。

证据包 `resnet50-throughput-evidence.tar.gz`，8167236字节、2128文件，包含实际YAML、
标准结果、原生日志、同步窗口JSON、输入来源/哈希、部署代码、CPU回归和清理回执。
SHA256 `8a22a5a85c46be85f888960d03025fad74e64144eabbbe75f777e83c895fe1a7`。
不含ImageNet图片和大checkpoint，来源清单及权重大小/哈希保留，权重留在自有运行目录。
