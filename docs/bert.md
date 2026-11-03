# BERT-large FP16：运行环境默认库与外置库

## 原来的阻碍

此前 BERT 用例无条件要求外部 `rocblas_dir/lib`，综合模板还默认填写 `/opt/rocblas-install`。即使 MIGraphX 原生环境和模型都齐全，只要这个外置目录没有准备，也会在开始推理前报错。这是固定配方施加的额外前提，不应让它阻止使用当前运行环境执行 BERT。

现在 `rocblas_dir: ""` 表示 `runtime_default`，使用当前环境的库搜索设置，不新增 LD_LIBRARY_PATH。已有环境仍需包含 MIGraphX 依赖的 rocBLAS；并不是删除了原生库依赖。指定非空目录表示 `custom_directory`，保留目录、非空内容、共享库和指纹检查；无效时明确失败。

## 模型合同

模型由用户提供，不会自动下载或把别的模型改名。当前配方要求：

- 自包含 ONNX，典型文件名 `bert-large-uncased-squad-v2-opt-mha-384.onnx`。
- 输入恰好为 `input_ids`、`attention_mask`、`token_type_ids`。
- 三个输入均为 INT64 `[batch,384]`。
- 动态 batch 可分别测 1 和 32；静态 batch 必须与本次配置一致。只有 batch=1 模型时把模板的 `matrix.batch` 改成 `[1]`，不能假定改 CLI 参数已改写静态模型。

输入用 native `--fill1` 填充，执行 `--gpu --fp16`，不添加 ResNet50 专用的 `--opt-compile`。三次独立进程，每次 -n 20；所有轮次完整后才发布吞吐/单批耗时中位数。这里只测性能，不验证 SQuAD 或数值准确率。

## 单独运行 BERT

按 [MIGraphX 新容器准备](migraphx-setup.md) 获得原生环境和 Python 依赖，并挂载已有模型后，在容器内：

```bash
PY=/work/venv/bin/python
"$PY" testpacks/migraphx/check.py --recipe bert_large_fp16 --output /work/bert-dependencies.json
# 编辑 configs/bert.template.yaml 的 vars.model、vars.numa_node、vars.hcus 和 batch。
# 默认 rocblas_dir 为空。
BENCH_PYTHON="$PY" bash bench.sh plan -c configs/bert.template.yaml --suite migraphx
BENCH_PYTHON="$PY" bash bench.sh configs/bert.template.yaml migraphx
```

预检检查 native 环境和 `--fill1`/`--fp16` 等 flags，不加载模型，也不表示 GPU 实测通过。测量结果保留在 `/work/results`，每个 case 的原始日志为 `round-1.log` 至 `round-3.log`。

## 需要复现外置库基线时

将已确认兼容目标硬件、DTK、MIGraphX 的目录挂载进容器，显式设置：

```yaml
vars:
  rocblas_dir: /tuning/rocblas-install
```

其中必须存在非空 `/tuning/rocblas-install/lib`。显式错误路径和空目录不会被忽略；不会自动安装、替换库或更改系统文件。请核对 `plan.json` 的环境、库文件 SHA256、模型 SHA256、镜像和 `rocblas_mode` 后再比较结果。默认库测量与外置库测量应分别保存，目录标签本身不能证明性能达标。
