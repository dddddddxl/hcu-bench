# DTK 原始用例接入与流程验证

本轮目标是按保留的 DTK YAML 和 `test_conv.sh` 走通工具流程，暂不做性能对比。测试环境为 gfx938、已有 DTK 26.10 镜像，rocBLAS 实测版本为 `5.2.0.678dfb584-dirty`。所有操作仅使用自有容器和已分配的单卡。

## 配方与执行结果

| 项目 | 来源覆盖 | 本轮真实执行结果 |
| --- | --- | --- |
| GEMM general | 原始 2,972 条 | 未做全量实测；单独探测的 1 条被原生工具自动调整参数，不接受为原始参数的测量 |
| GEMM smoke | 原始 135 条，五种精度各 27 条 | 分两轮完成 135 条原生校验和性能记录，未出现数值失败；不是一次完整 `all` 验收 |
| hipDNN `test_conv` | 原始 68 条调用 | 前 4 条通过 CPU 校验并产生原生计时；第 5 条在整轮时限到期时未完成，后 63 条未执行 |

GEMM 保留原始尺寸、类型、转置、实虚标量、stride、初始化和逐项迭代数。新版默认 `unit_check=1/norm_check=0`、`use_hipblaslt=-1`，性能直接保留一轮原生均值；不再使用自加的方阵、精度变体、10/100/3 计时默认值或额外零 norm 判定。保留的 general/GEMM smoke 没有 BF16，不自行增加。

`test_conv` 用原始 CMakeLists 和现有 SDK 成功编译、链接，构建类型保持空值，没有切换 Release 或删除原有编译警告选项。保留脚本顺序、环境导出、CPU 校验和原生计时循环；封装没有另加预热、迭代或重复次数。

已完成的 `test_conv` 条目为：

- `dtk-conv-001`：FP16，二维前向。
- `dtk-conv-002`：FP32，二维前向。
- `dtk-conv-003`：BF16，二维前向。
- `dtk-conv-004`：BF16，三维前向。

## 未接受、未完成及原生限制

- `dtk-general-04-0002` 原配置为 `M=33, lda=32`。native bench 明确输出将 lda 调成 33；源码配置保持原样，封装拒绝将调整后的结果作为原始参数的测量。这里不推断该条目是来源笔误还是非法参数检查用途。
- `dtk-conv-005` 没有完成 CPU 校验。整轮 API 任务的 950 秒诊断预算在此到期，该条目实际仅剩约 22 秒；进程以 SIGTERM 结束且清理确认。没有将原生计时行单独作为通过依据，也没有将此事判成数值不匹配。产品配置的默认整轮上限为 14,400 秒，本轮没有跑满全部 68 条。
- 原始程序不支持 `--help`，返回未知参数错误，而且打印的部分 padding 拼写与解析器不同。接入代码已去掉这项错误的预检假设，直接运行原始命令并保留实际参数错误。
- 原生源码明确不支持 INT8 CPU 校验，对应 `dtk-conv-025/026`；本轮尚未执行到它们。工具保留条目并明确标记校验未支持，不因 exit 0 记为通过。

## 执行边界与证据

第一轮运行 `20261104_142715_b50530c8` 在宿主设备状态查询超时后被守卫清理，留下 101 条完整 smoke 结果。第二轮 `20261104_150718_4baf84f1` 仅继续未完成的 smoke 条目，另执行 general 探测和原始 API 脚本。两轮自有容器均有完整身份及删除证明，没有拼成一次 `all` 成功。

第二轮在观察到主机空闲后，将自有容器 CPU 配额从 4 核提高至 32 核以完成原生 CPU 参考计算，保存了调整回执。原始测量参数未改变，但本记录不作为跨轮性能对比。宿主 CI/设备守卫的无设备容器清理自测和运行日志单独保存。

最终解析规则在上述真实日志上回放：135 条 smoke 和 4 条 API 结果可接受，自动调整 lda 的 general 条目和没有 CPU 校验终态的 API 条目均不接受。该回放没有再次执行 GPU。原始回执、编译输入/二进制哈希、check/plan/run 日志、逐项结果和归档均保留，未覆盖旧记录。Linux CI 仅验证控制与解析契约。

## 复现入口

先填写配置中的仓库路径、可写工作区及已分配的空闲卡，在匹配的完整 DTK 环境执行：

```bash
# 原始 general + GEMM smoke，完整扫描耗时较长。
bash bench.sh configs/cases/gemm.yaml all

# 仅原始 GEMM smoke。
bash bench.sh configs/cases/gemm.yaml all --set tests.gemm.params.group=smoke

# 原始 hipDNN 68 条；已有 test_conv 可填写 binary，否则提供原始源码根目录。
bash bench.sh configs/cases/hipdnn-conv.yaml all \
  --set tests.hipdnn_conv.params.source_root=/PATH/hipdnn/api_test
```

本记录未验收完整 general、全部 68 条 API、反向方向或全部精度组合，亦不宣称与 DTK 全部 nightly/gtest 分支等价。
