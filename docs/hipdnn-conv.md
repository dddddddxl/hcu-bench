# hipDNN test_conv：源码准备、编译与运行

`configs/cases/hipdnn-conv.yaml` 调用 DTK 的原始 `test_conv` 程序，保留原始 68 条测试命令及参数。仓库已捆绑完整的原始 C++ 构建输入的完整归档 [`vendor/hipdnn-api-test`](../vendor/hipdnn-api-test)，不再需要另找源码或手动复制程序。

`binary: auto`、`source_root: ""` 是默认配置：先查找 PATH 或 `/opt/dtk/bin/test_conv` 中已有的程序；未找到时校验随包源码哈希，再自动通过原始 CMakeLists 编译 `test_conv` 并执行测试。显式设置 `binary` 可复用指定程序；显式设置 `source_root` 可覆盖随包源码。

DTK 运行库、编译工具和配套 hipDNN SDK 由执行镜像提供，不随仓库重新分发。仓库不自动下载、安装或替换它们。

## 1. 准备已有 DTK 环境

下面的命令均在测试容器内执行。仓库和可写工作目录必须在容器内可见：

```bash
source /opt/dtk/env.sh
export BENCH_ROOT=/PATH/hcu-bench
export WORKSPACE=/PATH/writable-workspace
mkdir -p "$WORKSPACE"

cmake --version
ninja --version
```

原始 CMakeLists 要求 CMake 3.25.2 或更高版本、C++17、tar、已有 HIP 工具链，以及匹配的 `hipdnn_frontend`、`hipdnn_test_sdk` 配置和 `hipdnn_data_sdk` 头文件。`source /opt/dtk/env.sh` 后，原始构建文件使用该 DTK 的 `ROCM_PATH` 和编译器。只有运行库或 MIOpenDriver，不能代替这些开发依赖。缺少时需要取得配套 SDK，而不是在此流程中临时升级或替换库。

通过 `bench.sh` 运行还需要已准备好的 Python 3.10+ 和 PyYAML 环境；若默认 Python 不符合要求，设置 `BENCH_PYTHON` 指向已有环境。

宿主机克隆了仓库，不表示已有容器能访问它。创建自己的测试容器时，将完整仓库只读挂载到容器内，例如加入 `-v /HOST/hcu-bench:/bench:ro`，随后设置 `BENCH_ROOT=/bench`；可写构建和结果目录另行挂载。

## 2. 由 hcu-bench 自动编译并快测

保持默认 `binary: auto` 和空 `source_root` 即可使用随包源码自动编译（已有程序时优先复用）。以下选择原始 `dtk-conv-020`（FP32 二维分组前向）作单项确认，未修改它的尺寸、CPU 校验或计时循环：

```bash
cd "$BENCH_ROOT"
bash bench.sh configs/cases/hipdnn-conv.yaml all \
  --set "vars.bench_root=$BENCH_ROOT" \
  --set "vars.workspace=$WORKSPACE" \
  --set tests.hipdnn_conv.params.build_jobs=2 \
  --set tests.hipdnn_conv.params.case_id=dtk-conv-020
```

封装先将随包归档解压到该次运行的 `<case_dir>/native-source`，再在 `<case_dir>/native-build` 下构建一次，再复用生成的 `bin/test_conv` 执行选中的命令。本次构建只选择 `test_conv` 目标，完整源码树中的其他 API 不会被一并构建或测试。完整 CMake/cpp 树被保留，是因为原始 CMake 配置还引用了其他源文件；只复制 `test_conv.cpp` 不能完成原始配置。

## 3. 手动编译后复用二进制

若希望先单独检查编译过程，或让后续多次测试复用同一个程序，执行与封装一致的原始构建命令：

```bash
mkdir -p "$WORKSPACE/conv-source"
tar -xzf "$BENCH_ROOT/vendor/hipdnn-api-test/dtk-api-test-source.tar.gz" \
  -C "$WORKSPACE/conv-source" --strip-components=1 --no-same-owner
export API_TEST_SOURCE="$WORKSPACE/conv-source"
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

若使用其他匹配版本的原始 `api_test` 源码，在自动编译命令中添加 `--set "tests.hipdnn_conv.params.source_root=/CONTAINER/PATH/api_test"`。填写外部目录时需保留完整 CMake/cpp 树，并保证容器可以读取。

## 4. 查看结果与执行完整集合

所选单卡需要已分配、空闲，并正确映射到容器内 HIP 设备。示例配置使用逻辑编号 `0`，须与实际分配一致。

FP32 代表项的原始日志应同时包含 `CPU validation result: successful` 和 `fprop avg times: ... ms`。只有图构建成功或进程退出码为 0，不代表数值校验通过。计时和校验记录位于 `$WORKSPACE/results/<run_id>`，包括原生命令、构建输入/二进制哈希以及日志。

确认代表项后，移除 `case_id` 覆盖即可执行原始 68 条命令。它不是每条编译一次，但部分原生 CPU 参考计算耗时较长；抽测成功不能称为 68 条全部通过。INT8 的原生 CPU 校验不受支持，具体结果语义见 [算子说明](operators.md#dtk-hipdnn-原始-test_conv)。

本次从随包源码自动编译并完成代表项原生校验/计时的证据见 [自动编译快验记录](validation/2026-11-05-conv-autobuild.md)。
