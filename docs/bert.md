# BERT-large FP16：运行环境默认库与外置库

## 运行前准备

仓库提供测试代码、独立配置、Python 依赖声明和 [原生环境镜像/新容器步骤](migraphx-setup.md)。同事可在自己的兼容 Linux/DCU 节点使用，无需访问开发者的节点、个人目录或额外调优库。需准备：

- 与节点驱动兼容的 MIGraphX 原生环境；按上面的镜像步骤安装 `.[migraphx]` Python 依赖。仅有 Torch 不代表已具备 `migraphx-driver`。
- 满足下列输入要求的本地 ONNX 模型；模型权重作为测试输入挂载，不打入 Git。文件名是示例，不依赖固定个人路径。
- 已核对的空闲目标卡、对应 NUMA 节点和可写结果目录。参考容器只开放已选中的单卡，容器内逻辑卡号为 0。

现在 `rocblas_dir: ""` 表示 `runtime_default`，使用当前环境的库搜索设置，不新增 LD_LIBRARY_PATH。已有环境仍需包含 MIGraphX 依赖的 rocBLAS；并不是删除了原生库依赖。指定非空目录表示 `custom_directory`，保留目录、非空内容、共享库和指纹检查；无效时明确失败。

## 模型输入要求

模型由用户提供，不会自动下载或把别的模型改名。当前配方要求：

- 自包含 ONNX，典型文件名 `bert-large-uncased-squad-v2-opt-mha-384.onnx`。
- 输入恰好为 `input_ids`、`attention_mask`、`token_type_ids`。
- 三个输入均为 INT64 `[batch,384]`。
- 动态 batch 可分别测 1 和 32；静态 batch 必须与本次配置一致。只有 batch=1 模型时把模板的 `matrix.batch` 改成 `[1]`，不能假定改 CLI 参数已改写静态模型。

输入用 native `--fill1` 填充，执行 `--gpu --fp16`，不添加 ResNet50 专用的 `--opt-compile`。三次独立进程，每次 -n 20；所有轮次完整后才发布吞吐/单批耗时中位数。这里只测性能，不验证 SQuAD 或数值准确率。

## 单独运行 BERT

按 [MIGraphX 新容器准备](migraphx-setup.md) 获得原生环境和 Python 依赖，并挂载已有模型后，在仓库根目录执行。模型路径可替换；默认 batch 为 1、32，修改模板的 NUMA 和设备配置以匹配当前节点：

```bash
source /work/venv/bin/activate
PY=/work/venv/bin/python
MODEL=/models/bert-large-uncased-squad-v2-opt-mha-384.onnx
"$PY" testpacks/migraphx/check.py --recipe bert_large_fp16 \
  --model "$MODEL" --batch 1 32 --output /work/bert-preflight.json
# 修改 configs/bert.template.yaml 的 vars.numa_node、vars.hcus 和 batch。
# rocblas_dir 默认留空；通过参数指定本次代码和模型位置。
BENCH_PYTHON="$PY" bash bench.sh run -c configs/bert.template.yaml --suite migraphx \
  --set "vars.bench_root=$PWD" --set "vars.model=$MODEL"
```

预检输出 `ready: true`、退出 0 后再运行；退出 2 时先查看 `errors`，可定位缺 Python 包、原生库、driver flags、模型文件或输入形状/类型不匹配。指定 `--model` 会读取真实 ONNX 的输入元数据并检查请求的所有 batch，不进行 GPU 推理；不指定时仍只检查依赖。静态 batch=1 模型预检改用 `--batch 1`，同时把模板的 `matrix.batch` 改成 `[1]`。

每次运行生成独立结果目录 `/work/results/<run_id>`，终端打印路径。`summary.txt` / `summary.json` 汇总 case 状态，`metrics.csv` 记录吞吐和耗时。原始 `round-1.log` 至 `round-3.log`、执行退出码、模型/程序指纹与测量文件保留在 case 目录和收集的 artifacts 内；失败的进程不发布成功性能结果。预检通过也不保证任意模型图都能由当前 MIGraphX 编译，实际推理失败时检查原始日志。

要查看命令而不开始执行，把上述 `run` 改成 `plan`。模型、芯片、驱动/镜像、batch 和库模式均匹配时才比较吞吐基线。

## 验证范围

仓库 CI 检查真实 ONNX 输入元数据，并以明确标注的 CPU driver/NUMA 替身验证公共 `bench.sh` 入口、batch 1/32 的三轮调用、退出码处理及报告收集。这些检查验证启动链路，不是 BERT 网络推理或 DCU 性能结果。当前修复未完成真实 DCU BERT 推理验收；设备/模型兼容性与性能需在使用者的目标环境实测。

## 需要复现外置库基线时

将已确认兼容目标硬件、DTK、MIGraphX 的目录挂载进容器，显式设置：

```yaml
vars:
  rocblas_dir: /tuning/rocblas-install
```

其中必须存在非空 `/tuning/rocblas-install/lib`。显式错误路径和空目录不会被忽略；不会自动安装、替换库或更改系统文件。请核对 `plan.json` 的环境、库文件 SHA256、模型 SHA256、镜像和 `rocblas_mode` 后再比较结果。默认库测量与外置库测量应分别保存，目录标签本身不能证明性能达标。
