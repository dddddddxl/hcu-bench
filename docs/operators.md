# 固定算子用例

固定配方来自用户提供的 `hcu-bench用例收集.pdf` 和 `yy_script`，保留在 `testpacks/operators/native`。新增 Conv/GEMM 来自飞书测试指导的附件，原始脚本、CSV 和来源文件保存在 `vendor/operator-cases`；校验和见 `vendor/sources.json`。均属于 `operators` suite。

## Conv / GEMM

在已有 Linux 容器内复制 `configs/numeric-operators.template.yaml` 为 `configs/bench.yaml`，填写仓库、输出目录及允许使用的卡，然后执行 `bash bench.sh operators`。综合容器模板也已包含这两项；已有个人 YAML 需要手动合并对应测试块。

GEMM 默认关闭：当前标准 SGLang 镜像的原生接口与配方不匹配。准备好对应 profile 的工具及数值标准后，将 `tests.gemm.enabled` 改为 `true`。PA/FA 模板中的 FP8 也独立为 `fa_fp8_performance`，默认关闭；PA、FA FP16 保持启用。

| recipe | 精度 | 用例范围 | 原生工具 |
| --- | --- | --- | --- |
| `conv` | fp32 / fp16 | Conv2D 13 项、Conv3D 15 项，含普通/分组/Depthwise、前向/dgrad/wgrad；共 56 组精度配置 | 镜像已有 MIOpenDriver |
| `gemm` | fp32 / fp16 / fp64 / complex64 / complex128 | DTK 原始 general 2,972 条 + GEMM smoke 135 条，无自行扩展 | 镜像已有 rocblas-bench |

`mode: check` 只校验；`mode: perf` 对每个用例先执行原生校验，通过后才计时。Conv 使用 MIOpen 的 `Verifies OK` 和原生容差。新版 GEMM 默认与 DTK common 对齐：`unit_check=1`、`norm_check=0`，对应 `--test 1 --verify 0`，不再额外要求 norm 为零。原生逐元素校验失败始终阻止该项计时。`max_norm_error: ""` 表示仅使用原生校验；填写非空阈值时追加 `--verify 1` 并要求有限 norm 不超过该值，不能覆盖 unit Failure。

Conv 的现有 MIOpenDriver 路径仍为 10 次预热、每轮 100 次、3 轮，取三轮中位数；这不是 DTK 原命令原样执行。GEMM 默认只执行一轮原生计时，保留来源 cold_iters=2、iters=10（部分 HPL 为 5），不额外取三轮；记录原生 us 和 rocblas-Gflops。显式设置 warmup/iters/repeats 是参数覆盖，结果对比前需核对。

可调参数：`dtype`、`mode`、`device`、`warmup`、`iters`、`repeats`、`native_timeout_s`；Conv 可调 `conv_suite`、`direction`，GEMM 可调 `group`（all/general/smoke），两者可用 `case_id` 选择单项。`device: auto` 使用本次已分配的第一张卡，不寻找或抢占其他卡；可显式指定已分配集合中的某一张。每个原生进程只可见一张卡，并记录 HIP 逻辑编号及 PCI 地址。

`driver` / `rocblas_bench` 默认 `auto`，在现有 DTK 常规路径中查找，也可填写绝对路径。不下载安装工具，不修改 DTK 或频率。失败用例保留原始日志、命令和原因，其他用例继续；清理无法确认时停止整个计划。

GEMM 默认 `gemm_profile: new`，要求 rocBLAS **5.2.0**，正确性阶段默认使用 `--test 1 --verify 0`，并显式设置 `--use_hipblaslt -1`。旧版 **4.3.0** 必须选择 `gemm_profile: legacy`，仅发送该版本支持的 `--verify 1`，不发送 `--test` 和 `--use_hipblaslt`。两版校验语义分别记录，不自动回退，不把旧版一个检查称为新版两个检查均已通过。

两版 GEMM 的正确性与性能阶段均默认 `--initialization rand_int`；来源中 HPL 显式指定的 `trig_float` 覆盖默认值，与保留的 DTK `rocblas_common.yaml` 默认初始化一致；FP16 输入输出、FP32 累加按 smoke 原配置保留。原生 `unit.hpp:行号: Failure` 始终判为数值校验失败，配置的 norm 阈值不能覆盖该失败，也不会继续该项性能测试。

### 与 DTK 来源配置的对照

下表对照随包保留的 common/general YAML 默认值，不代表 DTK 团队实际 nightly 命令的全部覆盖参数。比较外部结果前还需核对实际命令和完整 DTK 环境。

| 项目 | DTK 来源配置 | 当前 GEMM 封装及影响 |
| --- | --- | --- |
| 输入初始化 | common 默认 `rand_int`，HPL 显式 `trig_float` | 保留默认及逐项覆盖；正确性与性能使用相同初始化 |
| 算子后端 | common 默认 `use_hipblaslt: -1`，自动选择 | 新版对齐为 `--use_hipblaslt -1`，由完整原生库选择；旧版没有该参数，不发送 |
| 预热与计时 | common 默认 `cold_iters=2`、`iters=10` | 正确性与性能都保留来源 2/10（部分 HPL 为 2/5）；性能默认一轮，直接保留原生均值 |
| 数值判定 | common 默认 `unit_check=1`、`norm_check=0` | 新版默认对齐，norm 阈值仅为可选追加检查。旧版保留其原生 `--verify 1`，不宣称两个版本接口完全相同 |
| 用例覆盖 | general 全量 + smoke 中 GEMM | 仅 2,972 + 135 条原始配置全部展开；已移除自加方阵和精度变体，不是 DTK 全部 BLAS/nightly |

来源 YAML 通过随包保留的原生 `rocblas_gentest.py` 展开为 `gemm_cases.csv`；运行时仅需标准库。保留尺寸、转置（含复数共轭）、leading dimension、stride、batch、实虚标量和类型。原生 bench 对 LD/输出 stride 的显式调整会记录为 `native_adjustments`，不静默修改来源。`math_mode=0`、`flush_batch_count=1` 和 C/D 同址方式沿用默认值。这里仍采用 bench 的调用路径；不是直接运行 rocblas-test/gtest，不能宣称其所有 pointer-mode/API 分支均已覆盖。

默认共 3,107 条原始配置：FP32 2,207、FP64 819、FP16 27、complex64 27、complex128 27。原始 general/GEMM smoke 没有 BF16，不额外添加。完整扫描可能超过默认超时，应按资源设置 `native_timeout_s` 与外层 `timeout_s`；超时和筛选运行不能称为完整扫描通过。

每次执行先检查原生版本和 CLI，不匹配立即失败并保留原始输出。新版默认不依赖 norm 字段；必须有与计划尺寸一致的原生记录，且无原生错误或 unit Failure。独立附件脚本只记录原生结果，hcu-bench 封装负责检查后才进入性能阶段。

现场已核验 CLI 的新版环境是 `10.16.1.152:5000/hcu/base/dtk:26.10-ubuntu22.04-py3.12`；旧版可使用其完整的 DTK 26.04 环境。工具与库应来自同一环境，不混拷二进制。若原文件为0644，复制到任务自有目录并 `chmod u+x` 后，用 `rocblas_bench` 指向可执行副本。**CLI匹配不代替目标机器上的数值与性能实测。**

```bash
# 默认新版
bash bench.sh operators --set tests.conv.enabled=false --set tests.gemm.enabled=true --set tests.gemm.params.gemm_profile=new
# 显式旧版
bash bench.sh operators --set tests.conv.enabled=false --set tests.gemm.enabled=true --set tests.gemm.params.gemm_profile=legacy
```

独立附件入口保存在 `vendor/operator-cases/gemm`：默认 `run_gemm_cases.py`，旧版 `run_gemm_cases_legacy.py`；`commands/new_check.sh`、`new_perf.sh`、`legacy_check.sh`、`legacy_perf.sh` 分别覆盖五种来源精度。完整说明见 [RUNNING.txt](../vendor/operator-cases/gemm/RUNNING.txt)。独立脚本的perf需人工先审核check日志；上面的 hcu-bench perf入口会自动先check，再决定是否计时。

```bash
bash bench.sh operators --set tests.conv.params.mode=check --set tests.gemm.params.mode=check
bash bench.sh operators --set tests.conv.enabled=false --set tests.gemm.enabled=true --set tests.gemm.params.group=smoke
```

## DTK hipDNN 原始 test_conv

独立配置 [configs/cases/hipdnn-conv.yaml](../configs/cases/hipdnn-conv.yaml) 执行随包原样保留的 `test_conv.sh` 68 条命令，属于 `operators` suite，与上面的 MIOpenDriver 路径分开。原脚本的 FP32/FP16/BF16（参数原文 `bfp16`）/TF32/INT8、布局、前后 padding、方向和 TF32 环境开关全部保留。封装没有增加迭代、预热或轮次；当前来源 C++ 原生代码执行一次准备调用，再用 HIP event 计时 100 次并输出平均毫秒数。

已有完整 DTK 环境可用时填写 `binary`，或将 `test_conv` 放入 PATH。若尚未编译，可填写原始 `hipdnn/api_test` 的 `source_root`；工具用原始 CMakeLists 与已有 CMake/Ninja/SDK 构建 `test_conv`，不下载依赖、不安装库、不删 `-Werror` 或替换实现。构建并行数只影响编译调度。缺少 SDK、CLI 不兼容或编译失败都会保留原始错误并失败。

```bash
bash bench.sh configs/cases/hipdnn-conv.yaml all
```

`case_id` 默认空，执行全部 68 条；指定 `dtk-conv-001` 仅用于诊断，不能称整套通过。日志必须有明确的 `CPU validation result: successful` 和对应方向的原生计时；`Graph validation successful` 仅表示图构建，不能代替数值校验。原生 `CPU validation result: failed` 即失败。来源代码对 INT8 明确不支持 CPU 校验，这类项保留原始提示并标记未验收，不因 exit 0 放行性能。失败项继续记录，最终不会把部分成功报成整套成功。保存逐项原生命令/环境、二进制与构建输入哈希、编译及运行日志。

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
  --set tests.conv.enabled=false --set tests.gemm.enabled=false \
  --set tests.operator_stress.enabled=true
```

`stress_iterations` 默认 `1000000`，可设为 `100` 先检查脚本能否运行。少于100万次的结果标记 `stress_mode: smoke`、`full_stress_completed: false`，不能当作完整压力测试通过。一致性检查和FA的NaN/Inf检查保持不变，必须完成指定次数并出现原生终态PASS才有完成指标。

```yaml
  operator_stress:
    enabled: true
    # 其余字段沿用模板；本例只显示需要修改的部分。
    params:
      stress_iterations: 100
```

只执行三项100次短测（当前YAML需要包含这些测试块）：

```bash
bash bench.sh operators --set tests.operator_performance.enabled=false \
  --set tests.conv.enabled=false --set tests.gemm.enabled=false \
  --set tests.operator_stress.enabled=true \
  --set tests.operator_stress.params.stress_iterations=100
```

`native_timeout_s` 为故障上限，外层 `timeout_s` 应额外留出清理时间。不支持框架 `duration_s`；未跑完、超时或取消均无有效完成指标。`PYTHONOPTIMIZE` 不得禁用原生 assert。独立调用原生脚本时，可用 `BENCH_STRESS_ITERATIONS=100` 选择短测；框架执行必须通过YAML参数选择，不允许冲突的环境变量偷偷缩短循环。

2026-11-02在BW1000 / gfx936、标准SGLang 0.5.12 / DTK26.04镜像验证：BLAS FP16、FA FP16、FA BF16各100次均完成，原生重复一致性检查通过，两项FA的NaN/Inf检查通过。结果均为smoke，不代表百万次压力或独立数值精度验收通过。

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

每个固定 PA/FA/BLAS case 保留 `plan.json`（脚本哈希、环境、版本、命令）、`execution.json`（时间、退出码、中断及清理）、`native.log`、`measurements.json` 和 `assessment.json`；FP8 另有 `fp8.csv`。这部分固定脚本使用 `correctness: not_checked`，压力用例另记录 `repeatability: passed`；Conv/GEMM 使用前述独立校验规则。指标完整且进程正常结束才发布测量值；清理无法确认时返回保留码 125。

接入回归只证明命令、解析、编排与失败处理；实际性能和镜像 ABI 仍需在空闲目标 HCU 上验证。
