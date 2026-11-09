# 内置训练测试

`training`与MIGraphX推理及LLM的e2e分开。主模板默认启用ResNet50训练吞吐并纳入all，
已有个人YAML不会自动迁移或覆盖。

## 配好数据就能跑

训练循环已内置在 `testpacks/training/train.py`，网络结构调用已安装的TorchVision。
无需外部train.py或模型权重，默认FP32不导入Apex。
环境须已有匹配的Torch/TorchVision、Python3.10+、PyYAML、psutil及明确分配的单张HCU。
不自动下载数据、权重、框架或SDK。

ImageNet预先解包为 `train/<1000类>/图片` 和 `val/<相同1000类>/图片`。
主模板填写vars.imagenet，独立模板填写vars.dataset；通用仓库路径、输出目录和卡号仍须符合节点：

```bash
cp configs/resnet50-training.template.yaml configs/bench.yaml
# 编辑vars.bench_root、vars.workspace、vars.dataset和nodes.node0.hcus。
bash bench.sh check -c configs/bench.yaml --suite training
bash bench.sh plan -c configs/bench.yaml --suite training
bash bench.sh training
```

默认recipe为resnet50_imagenet，train_script为auto（可省略），明确选择内置程序，不搜索用户目录。

## 默认吞吐短测

单卡FP32、batch32、workers4、seed1、1 epoch，10步预热+50步计时。
从原1000类各取1或2张训练图，共1920张，正好60个完整batch；每类1张验证图，共1000张。
图片直接位于类别目录，复制到自有输出并记录来源/大小/SHA256，标签不变，不修改共享数据。
不是FakeData或完整ImageNet精度/收敛验收。

TorchVision weights=None随机初始化，224随机裁剪/翻转、ImageNet归一化、交叉熵及SGD，
lr0.1/momentum0.9/weight_decay1e-4。验证Resize256/CenterCrop224。
固定seed、独立数据生成器、deterministic=True、benchmark=False，不保证跨SDK逐位相同。
这是本仓库自有实现，不是原modelzoo程序的逐字副本或原性能复现，不能直接对比旧外部入口吞吐。

第10次SGD更新后同步设备并启动perf_counter，第60次更新后同步并停止，仅首尾额外同步。
窗口包含数据加载/增强、H2D、前向、loss/精度统计、反向、SGD及日志开销；
排除初始化、预热、验证和存盘，报告training_throughput（images/s）和training_step_time（平均ms/步）。
窗口结束后仍须完成验证、checkpoint保存、CPU安全加载检查及重载验证，才发布指标。
NaN/Inf、失败、部分窗口、缺checkpoint或重载不一致均失败；清理不明返回125并阻断后续。
默认原生300秒、整体900秒是故障上限，不要求跑满；缺数据/依赖明确失败，不静默降级。
跨平台比较须固定batch、worker、数据/存储、CPU配额与精度。

## 完整训练与兼容入口

完整训练显式设置mode full、epochs90、batch64、workers6、native_timeout_s604800及外层timeout_s1209900。
每30轮学习率衰减0.1，每轮存盘，需充分磁盘；1000类目录检查不等于整个数据集无损验证。
smoke保留早期64/16张流程检查，要求1 epoch和batch<=64，不输出正式吞吐窗口。

旧配置显式提供train_script绝对路径时仍支持原modelzoo CLI/日志/checkpoint合同及script_sha256校验。
不支持任意同名程序；用户主动提供才执行。外部full/smoke的AMP仍是其Apex O1/dynamic，
内置目前只支持FP32，不隐式更换AMP实现。默认模板不再声明外部源码路径。

## 扩展其他训练测例

共享层负责卡号、超时/取消/进程树清理、数据来源、同步吞吐、结果/日志及checkpoint生命周期。
同类ImageNet分类模型在testpacks/training/recipes.py注册architecture和权重变化检查参数，
更新manifest的recipe choices，新增YAML测试块和合同测试；模型构造、训练/验证、checkpoint检查共同读取配方。
目前仅支持resnet50_imagenet，不宣称其他模型已验证。
检测或语言模型等不同数据/优化器/协议应增加独立training pack及配置，复用进程和结果协议，
不要强套ImageFolder和SGD检查。suite仍叫training，测试块及recipe区分具体测例。

## 结果边界

模型/优化器有限、权重已变、严格state_dict和重载Top1/Top5/打印loss一致只是结构与流程检查。
correctness=not_checked，training_convergence=not_assessed；有限但偏大loss原样报告，不判收敛合格。
JSON、日志和哈希自动收集，大checkpoint留在目标case的checkpoints/0，默认不上传。
历史[早期smoke](validation/2026-11-06-resnet50-training.md)和[外部吞吐](validation/2026-11-06-resnet50-throughput.md)
不能当成新版内置入口的实测结果。
