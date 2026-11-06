# ResNet50 ImageNet 从头训练

`training` 是独立 suite，与 MIGraphX ResNet50 推理、LLM 精度和吞吐分开。
目前只支持单节点、明确分配的单张 HCU。新主模板已默认启用训练吞吐短测并纳入 `all`，
已有个人 YAML 不自动迁移或覆盖；需复制新的测试块和两个输入路径，或用独立模板。

## 输入和依赖

本包复用已准备的海光 modelzoo `resnet50-pytorch/train.py`，不执行其写死卡号、
NUMA、MPI、数据路径的 shell 启动器。上游参考：
https://developer.hpccube.com/codes/modelzoo/resnet50-pytorch 。
检查的用户副本没有单独许可证文件，因此不把它复制分发到本仓库；
通过 `train_script` 提供已审核的外部源码，推荐同时填写 `script_sha256`。
本包适配其CLI、日志和checkpoint合同，不支持任意名字为train.py的程序。

环境须已有匹配的Torch、TorchVision、Apex、psutil、PyYAML。原脚本即使FP32也会
在HCU可用时导入Apex，不自动安装或替换框架。
数据是已解包的真实ImageNet ImageFolder：`train/<1000类>/图片`、`val/<相同1000类>/图片`。
不下载数据或预训练权重，不用FakeData，不传 `--pretrained`。

## 配置和执行

```bash
cp configs/resnet50-training.template.yaml configs/bench.yaml
# 填写仓库、可写输出、外部train.py、ImageNet路径和获分配的唯一HCU。
bash bench.sh check -c configs/bench.yaml --suite training
bash bench.sh plan -c configs/bench.yaml --suite training
bash bench.sh training
```

## 默认训练吞吐短测

默认 `mode: throughput`、1 epoch、FP32、batch32、workers4、seed1，10步预热+50步正式更新。
从原1000类分别取1或2张训练图片，合计1920张（正好60个完整batch），验证每类1张、共1000张。
各类图片须直接位于对应类目录；数据复制到自有输出，保留原1000类索引，来源/大小/哈希全部记录。
这不是完整ImageNet评测，不据此判断收敛。

使用PyTorch optimizer post-hook观察原脚本的SGD更新，不替换训练循环、优化器或计算内核。
第10次更新后同步设备并启动高精度墙钟，第60次更新后同步并结束。
窗口只在首尾同步，不逐步额外同步；包含加载、增强、H2D、前向、loss/精度统计、反向、SGD及原脚本日志开销。
`training_throughput` 为1600张正式图片/同步窗口秒数，单位images/s；`training_step_time`为窗口平均每步毫秒。
不使用原日志四舍五入的Time求和，不把验证、存盘或整个进程启动时间当训练吞吐。
计时完毕仍须完成原生验证、保存checkpoint、CPU检查与重载验证，才发布正式指标。
多/少一步、原生失败、NaN/Inf、缺checkpoint、重载不一致均不能发布成功吞吐。
该模式目前仅支持FP32，训练图片数限制1000..32768、batch<=64、1 epoch，避免意外跑长训。

默认单次原生执行300秒，外层900秒故障上限；正常不要求跑满。慢平台超时如实失败，不将部分窗口当完整结果。
图片复制/扫描、运行环境和存储会影响整体耗时；跨平台比较必须保持batch、workers、数据布局、CPU配额及精度一致。
主模板缺输入或依赖时明确失败，不自动降级为FakeData或跳过。
实现依据PyTorch 2.7.1的[optimizer post-hook](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/optim/optimizer.py)
与[设备同步接口](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html)。

## 显式完整训练和流程 smoke

需要完整训练时显式设置 `mode: full`、`epochs: 90`、`batch_size: 64`、`workers: 6`、
`native_timeout_s: 604800`、外层 `timeout_s: 1209900`，不要仅将mode改full而保留短测超时。
原始正式方案为ResNet50随机初始化、90 epoch、batch64、workers6、SGD，lr0.1、momentum0.9、
weight_decay1e-4，每30 epoch原生衰减0.1。沿用原脚本的随机裁剪224/水平翻转，
验证Resize256/CenterCrop224及ImageNet归一化。
`fp32`不开AMP；`amp`使用原脚本Apex O1/dynamic，不等于纯FP16，也不替换为原生AMP。

显式短测：

```bash
bash bench.sh configs/bench.yaml training \
  --set tests.resnet50_pretraining.params.mode=smoke \
  --set tests.resnet50_pretraining.params.epochs=1 \
  --set tests.resnet50_pretraining.params.batch_size=8 \
  --set tests.resnet50_pretraining.params.workers=2 \
  --set tests.resnet50_pretraining.params.native_timeout_s=900 \
  --set tests.resnet50_pretraining.timeout_s=2100
```

smoke从1000类前8类各复制8张训练、2张验证图片到本次输出，共64/16张真实图片，
保持这些类原有0..7标签，模型仍输出1000类。清单记录来源、大小和SHA256，不改共享数据。
仅允许1 epoch、batch<=64。该分数不是完整ImageNet准确率或90轮收敛结果。
full不筛选图片；只检查1000类目录不等于已证明整个数据集完整无损。

## 完成条件与结果

每次独立输出，运行原训练脚本，检查每轮验证输出、有限loss及每轮checkpoint存在。
在CPU安全加载本次checkpoint，严格匹配ResNet50结构、epoch、有限模型/SGD状态及初始化后权重变化；
再以原脚本 `--evaluate --resume` 重载最后checkpoint，Top1/Top5及打印的验证loss须与存盘前一致。
超时、取消、非零退出、缺checkpoint、重载不一致均失败；清理不明返回125并停止后续测试。

报告含Top1/Top5、训练及验证最后报告的平均loss、训练进程墙钟时间（含验证和存盘，不是纯计算吞吐）。
训练与验证出现NaN/Inf loss均失败；有限但很大的loss仍原样报告，不能据此宣称训练质量合格。
此前8类小子集流程smoke已出现很大的验证loss，证据见[短测记录](validation/2026-11-06-resnet50-training.md)，
不把那次进程耗时换算为新版同步窗口吞吐。
新版10+50步同步窗口的实机证据见[吞吐短测记录](validation/2026-11-06-resnet50-throughput.md)。
数值正确性仍为 `not_checked`，上述状态检查不是独立算子数值基准。
训练收敛为 `not_assessed`，不将小数据分数包装成收敛通过，也无默认精度门槛。

JSON和原始日志自动收集，checkpoint保留在目标case的 `checkpoints/0/`，默认不上传巨大权重。
`checkpoint.json`记录最后权重大小和SHA256。原脚本每轮存盘并可能保留best副本，正式90轮需充分磁盘。
源码和数据只读挂载，输出单独可写，仅操作明确分配的卡。
