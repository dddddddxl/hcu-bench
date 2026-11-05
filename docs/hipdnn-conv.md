# hipDNN test_conv：源码准备、编译与运行

`configs/cases/hipdnn-conv.yaml` 调用 DTK 的原始 `test_conv` 程序。hcu-bench 保留了原始 68 条测试命令，但**没有捆绑 `api_test` C++ 源码或预编译的 `test_conv`**，也不会自动下载它们。安装了 DTK/hipDNN 库，不代表镜像中同时存在这个测试程序或它的源码。

需要先准备以下任意一种输入：

- 与当前 DTK/hipDNN 环境匹配的 `test_conv` 可执行文件；或
- 从 DTK 源码提供方取得的完整 `hipdnn/api_test` 源码目录，用当前环境的 SDK 构建。

下面的命令均在测试容器内执行。`BENCH_ROOT`、`WORKSPACE`、`API_TEST_SOURCE` 和 `TEST_CONV_BIN` 都必须是容器可见路径。

## 1. 准备源码和已有工具链

完整源码树至少包含下面这些位置；不能只复制 `test_conv.cpp`，因为原始 CMake 配置还引用其他 API 测试的源文件：

```text
api_test/
├── CMakeLists.txt
└── cpp/
    ├── utils.hpp
    ├── test_conv/test_conv.cpp
    └── ... 原始 CMakeLists.txt 引用的其余源码
```

若收到的归档顶层目录名为 `api_test`，可解包到自己的可写目录：

```bash
mkdir -p /work/dtk-reference
tar -xzf dtk-api-test-source.tar.gz -C /work/dtk-reference
export API_TEST_SOURCE=/work/dtk-reference/api_test
```

归档由源码提供方交付，文件名仅为示例，不是仓库自带附件或公共下载地址。若已解包或挂载源码，直接把 `API_TEST_SOURCE` 设为实际目录。

准备当前环境：

```bash
source /opt/dtk/env.sh
export BENCH_ROOT=/PATH/hcu-bench
export WORKSPACE=/PATH/writable-workspace
mkdir -p "$WORKSPACE"

cmake --version
ninja --version
test -f "$API_TEST_SOURCE/CMakeLists.txt"
test -f "$API_TEST_SOURCE/cpp/test_conv/test_conv.cpp"
test -f "$API_TEST_SOURCE/cpp/utils.hpp"
```

原始 CMakeLists 要求 CMake 3.25.2 或更高版本、C++17、已有 HIP 工具链，以及匹配的 `hipdnn_frontend`、`hipdnn_test_sdk` 配置和 `hipdnn_data_sdk` 头文件。`source /opt/dtk/env.sh` 后，原始构建文件使用该 DTK 的 `ROCM_PATH` 和编译器。只有运行库或 MIOpenDriver，不能代替这些开发依赖。缺少时需要取得配套 SDK，而不是在此流程中临时升级或替换库。

通过 `bench.sh` 运行还需要已准备好的 Python 3.10+ 和 PyYAML 环境；若默认 Python 不符合要求，设置 `BENCH_PYTHON` 指向已有环境。

宿主机上存在源码，不表示已有容器能访问它。创建自己的测试容器时，可把源码只读挂载到容器内，例如加入 `-v /HOST/api_test:/dtk-api-test:ro`，随后使用 `API_TEST_SOURCE=/dtk-api-test`；可写构建和结果目录另行挂载。已有容器则需将源码复制到其自有工作目录，或重建自有容器并添加挂载。

## 2. 由 hcu-bench 自动编译并快测

没有预编译程序时，保持 `binary: auto`，填写 `source_root`。以下选择原始 `dtk-conv-020`（FP32 二维分组前向）作单项确认，未修改它的尺寸、CPU 校验或计时循环：

```bash
cd "$BENCH_ROOT"
bash bench.sh configs/cases/hipdnn-conv.yaml all \
  --set "vars.bench_root=$BENCH_ROOT" \
  --set "vars.workspace=$WORKSPACE" \
  --set "tests.hipdnn_conv.params.source_root=$API_TEST_SOURCE" \
  --set tests.hipdnn_conv.params.build_jobs=2 \
  --set tests.hipdnn_conv.params.case_id=dtk-conv-020
```

封装在该次运行的 `<case_dir>/native-build` 下构建一次，再复用生成的 `bin/test_conv` 执行选中的命令。`source_root` 非空且 `binary` 为 `auto` 时走此构建路径；它不会把源码目录猜成一个已安装的二进制。

## 3. 手动编译后复用二进制

若希望先单独检查编译过程，或让后续多次测试复用同一个程序，执行与封装一致的原始构建命令：

```bash
cmake -S "$API_TEST_SOURCE" -B "$WORKSPACE/test-conv-build" -G Ninja
cmake --build "$WORKSPACE/test-conv-build" --target test_conv -j 2

export TEST_CONV_BIN="$WORKSPACE/test-conv-build/bin/test_conv"
test -x "$TEST_CONV_BIN"
sha256sum "$TEST_CONV_BIN"
ldd "$TEST_CONV_BIN"
```

保留原始 CMake 配置，不额外指定 Release、优化参数，不删除 `-Werror` 或替换测试实现。构建目录需可写，源码目录可以只读。原始程序不支持 `--help`，不要用它作为可用性检查。

在相同的 DTK 环境中显式指定生成的二进制：

```bash
cd "$BENCH_ROOT"
bash bench.sh configs/cases/hipdnn-conv.yaml all \
  --set "vars.bench_root=$BENCH_ROOT" \
  --set "vars.workspace=$WORKSPACE" \
  --set "tests.hipdnn_conv.params.binary=$TEST_CONV_BIN" \
  --set tests.hipdnn_conv.params.case_id=dtk-conv-020
```

这条路径不再编译，也不需要填写 `source_root`。更换 DTK、SDK 或运行环境后应重新核对依赖，不能仅凭文件名相同就跨环境复用旧程序。

## 4. 查看结果与执行完整集合

所选单卡需要已分配、空闲，并正确映射到容器内 HIP 设备。示例配置使用逻辑编号 `0`，须与实际分配一致。

FP32 代表项的原始日志应同时包含 `CPU validation result: successful` 和 `fprop avg times: ... ms`。只有图构建成功或进程退出码为 0，不代表数值校验通过。计时和校验记录位于 `$WORKSPACE/results/<run_id>`，包括原生命令、构建输入/二进制哈希以及日志。

确认代表项后，移除 `case_id` 覆盖即可执行原始 68 条命令。它不是每条编译一次，但部分原生 CPU 参考计算耗时较长；抽测成功不能称为 68 条全部通过。INT8 的原生 CPU 校验不受支持，具体结果语义见 [算子说明](operators.md#dtk-hipdnn-原始-test_conv)。
