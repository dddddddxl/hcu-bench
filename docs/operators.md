# 固定算子用例

来源为用户提供的 `hcu-bench用例收集.pdf` 和 `yy_script` 脚本目录。脚本已放入 `testpacks/operators/native`，原始 SHA256、仓库版本 SHA256 和适配记录在 `testpacks/operators/provenance.json`。Conv 和通用 GEMM 不在本次范围。

## 配置与执行

```bash
cp configs/operators.template.yaml configs/bench.yaml
# 编辑镜像、容器 hook、仓库路径及唯一目标卡。
bash bench.sh check -c configs/bench.yaml --suite operators
bash bench.sh plan -c configs/bench.yaml --suite operators
bash bench.sh operators --set vars.image=YOUR_APPROVED_OPERATOR_IMAGE
```

`vars.image` 作为 argv 传入用户容器启动 hook，支持 tag 或 digest；不拉取、升级或写死镜像。沿用 [容器接口](testpacks.md)：hook 负责透传唯一目标卡、只读挂载代码和运行库、保存实际镜像 ID、确认卡空闲及清理自有容器。共享 CI 主机必须由宿主侧守卫优先让出 CI，不可只看利用率为 0。

在已有容器内执行时，删除 `containers` 和测试块中的 `container` 字段，保留 `executor: local`，把 `bench_root` 填成容器内仓库路径。SSH 运行可将 node 改为 `executor: ssh` 并填写 host、work_root。`nodes.hcus` 必须只有一个目标执行环境内的 HIP 编号，并用 BDF/UUID 核实物理卡对应关系。

## BLAS FP16 压力测试

`recipe: blas_fp16_stress`

```bash
HIP_VISIBLE_DEVICES=0 python blas_pressure.py
```

4096×4096 FP16 矩阵；完整 1,000,000 次；逐次 `torch.equal` 对照第一次结果。

## FA FP16 / BF16 压力测试

`recipe: fa_fp16_stress` / `fa_bf16_stress`

```bash
HIP_VISIBLE_DEVICES=0 FA_FWD_NO_MLS=1 python fa_fwd_pressure.py \
  1 52 8192 128 128 true fp16 13 '[-1,-1]' false --whl
HIP_VISIBLE_DEVICES=0 FA_FWD_NO_MLS=1 python fa_fwd_pressure.py \
  1 52 8192 128 128 true bfloat16 13 '[-1,-1]' false --whl
```

完整 1,000,000 次，检查输出与 LSE 对首次结果的重复一致性、初始 NaN/Inf 检查。原脚本打印 golden 差异但没有容差验收，不能据此声称数值精度通过。

压力测试块默认 `enabled: false`。安排独占测试窗口后启用：

```bash
bash bench.sh operators --set tests.operator_performance.enabled=false \
  --set tests.operator_stress.enabled=true
```

`native_timeout_s` 为故障上限，外层 `timeout_s` 应额外留出清理时间。不缩短百万次循环，不支持框架 `duration_s`；未跑完、超时或取消均无有效完成指标。`PYTHONOPTIMIZE` 不得禁用原生 assert。

## PA decode 性能

`recipe: pa_decode`

```bash
HIP_VISIBLE_DEVICES=0 PA_NO_MLS=1 python prefix_decode_benchmark.py --whl
```

batch=1/2/4/8/16/32/64/128，KV 长度=2048/3072/3584/4096/5000，共 40 点；Q 长度 1，Q/KV heads=24/2，head dim=128，FP16，page block=128，causal。每个点保留延迟、TFLOPS、GB/s；只解析逐点结果，不重复计入末尾汇总。

## FA FP16 前向性能

`recipe: fa_fp16_forward`

```bash
HIP_VISIBLE_DEVICES=0 python fa_fwd_mfu.py
```

batch=1、heads=32、seq=32768、dim=128、groups=8、causal；使用 `triton.testing.do_bench_cudagraph`。保留原脚本的 `/usr/local/lib/python3.10/dist-packages/flash_attn/lib/libflash_attention.so` 和 `fwd_bshd` 接口；镜像缺文件或 ABI 不匹配时明确失败，不自动换成另一套 wheel 路径。

## FA FP8 前向性能

`recipe: fa_fp8_forward`

```bash
HIP_VISIBLE_DEVICES=0 FLASH_ATTENTION_HG_92A_MODE=fp8 \
  python bench_fp8_fixlen.py --whl --csv --output fp8.csv
```

batch=8、heads=16、groups=8、seq=16384、dim=128、FP8 e4m3、scale=0.088、causal、window=(-1,0)、seed=123；iterations=5，第一轮舍弃，`do_bench(rep=20, return_mode="mean")`。沿用原脚本先剔除低于均值 0.8 倍、再剔除高于新均值 1.25 倍的样本及其 fallback。结果分别标记原始吞吐均值、筛选后均值、由筛选后吞吐换算的耗时，不冒充未筛选中位数。

## 脚本适配与结果

- FA 压测的 `/fa_debug` 改为 wrapper 必填的 `BENCH_FA_CACHE_DIR`，每次运行唯一；独立调用该脚本时也必须设置一个新的可写目录。缓存不进入归档，结果日志保留。
- 字面量窗口/序列参数由 `eval` 改为 `ast.literal_eval`；按仓库约定统一 LF，PA 的 GB18030 注释转为文件声明的 UTF-8。
- 源目录缺失 `_flash_attn_build_loader.py`；补充的适配器仅加载已安装 wheel 的 `flash_attn_cuda`，不支持猜测 `.so` 或重编译替代实现。FP8 仍要求原脚本的 19 参数 `fwd` ABI。
- PA 的 `do_bench` 计时方式保持不变；拒绝 `FA_DEBUG`、`PA_USE_CUDAGRAPH_BENCH` 等会改变固定配方的开关。

先在相同子进程环境只读检查 HIP PyTorch 与唯一可见设备，保存 `device.json` 和 `device-probe.log`。PA 原脚本按工具链路径选择分支；如果会误入 CUDA 的 page=16 分支则提前失败，防止把它记作 HCU page=128 的固定用例。

每个 case 保留 `plan.json`（脚本哈希、环境、版本、命令）、`execution.json`（时间、退出码、中断及清理）、`native.log`、`measurements.json` 和 `assessment.json`；FP8 另有 `fp8.csv`。全流程使用 `correctness: not_checked`，压力用例另记录 `repeatability: passed`。指标完整且进程正常结束才发布测量值；清理无法确认时返回保留码 125。

接入回归只证明命令、解析、编排与失败处理；实际性能和镜像 ABI 仍需在空闲目标 HCU 上验证。
