# 随包 Conv 源码自动编译快验

本轮验证从空工作目录通过 hcu-bench 自动准备 `test_conv`，不提供外部 `source_root`，不复用先前编译的程序。使用已有 DTK 26.10 环境，未安装或替换库、SDK 或工具链。

## 运行方式

```bash
source /opt/dtk/env.sh
bash /bench/bench.sh /bench/configs/cases/hipdnn-conv.yaml all \
  --set vars.bench_root=/bench \
  --set vars.workspace=/work \
  --set tests.hipdnn_conv.params.case_id=dtk-conv-020 \
  --set tests.hipdnn_conv.params.native_timeout_s=900 \
  --set tests.hipdnn_conv.timeout_s=960
```

仓库只读挂载，构建与结果目录可写。容器中未安装 `test_conv`；默认 `binary: auto`、`source_root: ""` 触发随包源码路径。执行顺序为校验归档哈希、解包、原始 CMake 配置、`cmake --build ... --target test_conv -j 2`、HIP 设备探测、原始测试命令。

## 实测结果

- run_id：`20261105_120940_b55d932c`。
- 容器内恰好可见一张 HIP 设备，BDF `0000:bf:00.0`。
- 源码解包、CMake 配置和编译均成功。
- `dtk-conv-020`：FP32、NCHW、二维分组卷积前向，保留原始 shape、CPU 校验和计时循环。
- 原生日志：`CPU validation result: successful`；`fprop avg times: 0.255815 ms`。
- 从启动容器内命令到退出共约 55.8 秒，包含编译和执行。
- 自有测试容器已删除，完整身份清理回执记录 `removed=true`、`remaining=[]`。

本次在用户允许的共享环境中验证功能和计时可获取性，耗时不用于性能对比。只有这一条代表项得到本轮 CPU 校验和原生计时；不表示全部 68 条、其他精度/方向或其他 DTK 版本已通过。

## 输入与证据标识

- 原始源码归档 SHA256：`eb7f2788e880ac988f7a24d5ff43a92e06de904a997acfa615ca531a960300d8`。
- 本次生成的 `test_conv` SHA256：`4f5e79c57ac732af55d0d3337abd80cc5fd89b98c5bbfe4587b6e713f44562ba8`。
- 执行镜像 ID：`sha256:a566cf05ed0fda70a05aacef9282795ee81ff47255f162733c5dbb31950aabfd`。
- 完整本地证据归档 SHA256：`cb8c5406e4bc7305af2c9d49ea8bb3789c8fb50d99c5a4697cbd4012d1944fc1`，包含 check/plan/run、构建命令及日志、原生校验/计时、设备与环境记录、容器身份和清理证明。

首次自动解包曾因归档原始属主不能在受限容器中恢复而失败，已通过 `tar --no-same-owner` 修复；上述成功运行来自新的空工作目录。
