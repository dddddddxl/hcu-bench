# 显存带宽测试

`memory` 调用已有的 **dcu-hipbenchmark main-staging-1.0 / hcu-stream**，执行 `run_pattern.sh peak zd1`、`peak zd2` 对应的 24 项用例。它测的是单卡显存、缓存和 LDS 带宽，不是 PCIe，也不是多卡通信带宽。

## 准备与运行

在已有容器的 hcu-bench 仓库目录执行：

```bash
cp configs/memory.template.yaml configs/bench.yaml
```

改 YAML 中的工作目录、仓库路径、`memory_binary` 和 `gpus`，然后：

```bash
bash bench.sh check -c configs/bench.yaml --suite memory
bash bench.sh memory
```

`check` 只检查配置，不会检查远端二进制、占卡或启动负载。运行前自行确认所选卡空闲、编译架构与设备匹配。Python 3.10+、PyYAML、psutil 及原生运行库须已有，不会自动安装。

综合容器模板已加入 memory，填写 `memory_binary`、`memory_gpus` 后，`bash bench.sh all` 会按 YAML 顺序一起运行。旧配置不会被覆盖，需补入这两个变量和 `memory_bandwidth` 测试块；暂不测可设置 `enabled: false`。SSH 和自动创建容器仍使用现有接口，输入路径必须在执行环境内可见。

### 尚未编译时

原生源码由使用者准备，仓库不分发该压缩包或二进制。YY / gfx92a 已验证的编译方式是在原生源码根目录执行：

```bash
export ROCM_PATH=/opt/dtk
make -B HIPCC=/opt/dtk/bin/aicc \
  OFFLOAD_ARCH=--offload-arch=gfx92a TARGET_NUM=9210
```

YY85 这次环境中，默认 `hipcc` 指向的 DCC 不识别部分指令，改用镜像已有 AICC 后原源码编译通过。这里是该环境的已验证配方，不代表其他卡型也应使用 gfx92a/9210。工具不会自动编译、换库或改频率；将编译得到的 **`hcu-stream`** 路径填入 YAML 即可。

gfx938 在现有 DTK 26.04 镜像上的试编译尚未通过，分别遇到 AICC 指令操作数和 HIPCC 内建函数接口错误；当前不能宣称已支持该组合。不要只替换架构编号就沿用 gfx92a 的二进制或配方，需要原生测试包提供兼容实现。

## 可调参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `nodes.node0.gpus` | 模板 `[0]` | 允许使用的卡号，先确认空闲 |
| `params.devices` | `""` | 空字符串测上述全部卡；`"1,3"` 只测其中两张，不能超出分配范围 |
| `params.presets` | `both` | 两组共 24 项；也可单选 `zd1` / `zd2` |
| `params.model_mhz` | `1400` | 传入原生 `MODEL` 的频率换算参数，不是锁频命令 |
| `params.array_size` | `104857600` | `--arraysize` 元素数，等于原生默认 10240×10240；不是字节，部分矩阵/缓存用例不使用此参数 |
| `repeats` | `1` | 独立复测轮数，不是 kernel 内部迭代数 |
| `timeout_s` | `1800` | 整轮故障时限，覆盖全部指定卡；不配置 `duration_s` |

每次仅向子进程暴露一张卡，按配置顺序串行执行。原生进程使用可见设备 0，结果记录的是配置中的卡号。例如 `devices: "1,3"` 先用 `HIP_VISIBLE_DEVICES=1`，再用 `3`。不要叠加 ROCR/CUDA/GPU_DEVICE_ORDINAL 掩码；容器内卡号应先与实际设备核对。

## 用例与数据

| 组 | 按执行顺序列出用例 |
| --- | --- |
| zd1，15 项 | l1_read、l2_read、l2_write、lds_read、lds_write、memory_readx4_x1、memory_write、memory_copy、memory_to_ldsx1 `--lds 4`、lds_to_memory `--lds 4`、matrix_read、matrix_write、matrix_copy、mapping_matrix_write、tiling_read |
| zd2，9 项 | memory_readx4_x1 `-w 8`、memory_write `-w 8`、memory_copy `-w 8`、matrix_readx4_x4、matrix_writex4_x4、matrix_copy、memory_readx4_x4、mapping_matrix_writex4_x4、tiling_read_2w |

包装器逐条调用 `hcu-stream bandwidth 用例 --arraysize 元素数 [用例参数] --clean`，不直接运行整段 shell 脚本，确保中途失败不会被末尾的成功输出掩盖。除显式参数外，保留该版本的原生默认设置：默认 wave=4，不启用 `--useEvent`；kernel 内部迭代由原生代码决定。

- **单位是 GB/s，保存原生平均带宽**。脚本名字虽有 peak，这里不把均值叫作峰值；`--clean` 只简化输出，不代表清缓存。
- L1/L2/LDS 五项使用周期数与 `MODEL` 换算。默认 1400 沿用 zd1 原脚本；需要与 1280 MHz 口径比较时，应先确认双方参数再改 `model_mhz`。这不会设置硬件频率，也不会自动修正历史结果。
- 其余 19 项按原生计时结果记录，不能把五项缓存/LDS 带宽和显存带宽混在一起求平均。`memory_copy` 计读写总流量，不等于 PCIe 双向传输。
- 只报告测量值，不自动套用性能门槛、不宣称正确性通过。不同卡、preset、用例及换算参数的指标单独记录，zd1/zd2 的同名项不会合并。

## 结果与失败处理

统一报告中可查看指标；原生产物按测试包声明归档：

| 文件 | 内容 |
| --- | --- |
| `measurements.csv` | 每卡、每组、每项的 GB/s，以及 wave、LDS 参数、元素数、MODEL 和计时方式 |
| `gpu*_zd*.log` | 各用例原始输出 |
| `plan.json` / `execution.json` | 参数、二进制哈希、实际命令、单卡掩码、时间及返回码 |
| `assessment.json` | 是否完整执行；正确性始终单独标记 `not_checked` |

非零退出、帮助文本、空输出、多个数值、NaN/Inf 或错误信息都不会算作有效结果。当前扫描失败后保留已完成用例的数据，整体记失败；清理确认后由方案的 `failure_policy` 决定是否继续其他 suite，清理失败则停止剩余计划。

原生 24 项此前已在 YY85 单卡跑通。新统一入口的本地测试使用合成子进程验证编排、解析和失败处理，不能代替真实 GPU 复测。
