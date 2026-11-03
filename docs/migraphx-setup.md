# 从新容器准备 ResNet50 / MIGraphX 环境

`migraphx-driver` 是 MIGraphX 原生工具，不随 Torch 或本仓库的 Python extras 安装。为了避免用户自行拼装 DTK/MIGraphX 动态库，本配方提供一个已检查的专用镜像；不要求复用别人的旧容器。

## 已检查的镜像

```bash
# 可读标签：
IMAGE_TAG=harbor.sourcefind.cn:5443/dcu/admin/base/migraphx:5.2.0-ubuntu22.04-dtk26.04-py3.11
# 固定本次检查的 registry manifest，建议实际使用 digest：
IMAGE=harbor.sourcefind.cn:5443/dcu/admin/base/migraphx@sha256:bf6e6a8d8d7d124dbba7cf49eb598415d0e10f6bdb5901eb2d8199ee2deea360
docker pull "$IMAGE"
```

访问需要能连通 SourceFind Harbor；如要求认证，先用本人账号执行 `docker login harbor.sourcefind.cn:5443`，不要把密码写入 YAML/Git。本仓库不镜像分发厂商二进制。

2026-11-02 在该镜像全新创建的容器内确认：

| 项目 | 结果 |
| --- | --- |
| 系统 / Python | Ubuntu 22.04 / Python 3.11 |
| DTK | `/opt/dtk` → `/opt/dtk-26.04`，标签 `DTK-26.04-rc4` |
| 原生 driver | `/opt/dtk/bin/migraphx-driver`，MIGraphX 5.2.0 |
| `numactl` | `/usr/bin/numactl` |
| 原生依赖 | `ldd` 无 `not found`，perf help 支持 `--fp16`、`--int8`、`--opt-compile` |
| Python 依赖 | 已有 PyYAML、psutil；**仍需补 onnx** |

这解释了原配方为何要求“容器内提供 migraphx-driver”：工具来自选用的原生运行环境，不是从 benchmark Python 代码生成。这里的检查只证明工具与参数依赖，不等于在所有 DCU/驱动上做过性能验收。

## 1. 准备 Python wheel（联网阶段，无需 DCU）

在仓库根目录的 Linux 宿主机执行。模型需已存在，挂载的模型目录只读。

```bash
REPO_DIR=$(pwd)
MODEL_DIR=/YOUR_MODELS
HYHAL_DIR=/opt/hyhal                 # 按主机实际运行库目录设置
RENDER_NODE=/dev/dri/renderD128      # 须先用 BDF/UUID 核对目标卡
NUMA_NODE=0                         # 目标卡对应的 CPU/memory NUMA
WORK=$(mktemp -d "${TMPDIR:-/tmp}/resnet50-perf.XXXXXXXX")
WHEELS_DIR="$WORK/wheels"
mkdir -p "$WHEELS_DIR"

# 这个临时准备容器允许联网；使用镜像自己的 Python 下载匹配 Linux/Python 3.11 的 wheel。
docker run --rm -v "$WHEELS_DIR":/wheels --entrypoint python3 "$IMAGE" \
  -m pip download --only-binary=:all: -d /wheels \
  'onnx>=1.16,<2' 'psutil>=5.9,<8' 'PyYAML>=6,<7' 'setuptools>=61' wheel
```

离线机器可以从其他能联网的机器传入匹配的 wheel 目录；不能拿 Windows wheel 直接装进 Linux 镜像。`.[migraphx]` 只补 Python 依赖，原生 driver/DTK 已由上述镜像提供。

## 2. 新建测试容器

继续在同一宿主机终端执行。只透传已确认空闲的目标 render 节点，测试容器禁网。仓库挂载可写，以便 editable 安装生成本工具的元数据；模型和主机运行库只读。

```bash
NAME="resnet50_perf_$(date +%s)"
DEVICES=(--device=/dev/kfd --device="$RENDER_NODE")
if [ -e /dev/mkfd ]; then DEVICES+=(--device=/dev/mkfd); fi
docker run -d --init --name "$NAME" --network none \
  "${DEVICES[@]}" --group-add video \
  --cap-add SYS_NICE --cap-add IPC_LOCK --ulimit memlock=-1 --shm-size=8g \
  -e NUMA_NODE="$NUMA_NODE" \
  -v "$HYHAL_DIR":/opt/hyhal:ro -v "$MODEL_DIR":/models:ro \
  -v "$REPO_DIR":/workspace/hcu-bench -v "$WORK":/work \
  -v "$WHEELS_DIR":/wheels:ro -w /workspace/hcu-bench \
  --entrypoint /bin/bash "$IMAGE" \
  -c 'mkdir -p /var/log/hylog; exec sleep infinity'
docker exec -it "$NAME" bash
```

## 3. 容器内安装 Python 依赖并预检

```bash
python3 -m venv --system-site-packages /work/venv
source /work/venv/bin/activate
PY=/work/venv/bin/python
"$PY" -m pip install --no-index --find-links=/wheels --no-build-isolation -e '.[migraphx]'
"$PY" testpacks/migraphx/check.py --recipe resnet50_fp16 --output /work/migraphx-check.json
```

激活虚拟环境是必要步骤：测试包使用 PATH 中的 `python3`，只设置 `BENCH_PYTHON` 会选择 CLI 解释器，不会改变测试子进程的解释器。另开终端后应再次激活同一环境。

预检返回 0 且 JSON `ready: true` 才表示依赖检查通过；2 表示失败。它检查 Python 包、driver/numactl 路径、共享库以及配方要求的 help flags，不加载模型、不分配 GPU、不声称正确性或性能通过。之后再核对 DCU/驱动、设备映射、NUMA 和 ONNX 输入。

driver 默认先从 PATH 查找，再查 `/opt/dtk/bin/migraphx-driver`、`/opt/rocm/bin/migraphx-driver`。显式 `params.binary` 不会被替换为别的版本；找到的 driver 缺库会直接报具体库名。脚本不会自动补写 LD_LIBRARY_PATH 或替换运行库。

## 4. 运行 ResNet50

在容器内复制只包含 ResNet50 的模板，修改模型路径、NUMA 和逻辑卡号：

```bash
cp configs/resnet50.template.yaml configs/bench.yaml
# 编辑 paths/NUMA/hcus，并将 tests.resnet50.enabled 改为 true；标准 SGLang 镜像默认关闭此项。
# 修改 vars.model、vars.numa_node、vars.hcus；确认隔离后映射的逻辑设备编号。
BENCH_PYTHON="$PY" bash bench.sh check -c configs/bench.yaml --suite migraphx
BENCH_PYTHON="$PY" bash bench.sh migraphx
```

默认 FP16/INT8 × batch 1/32，每个组合三轮，每轮 -n 20。模型应是符合输入合同的自包含 ONNX；动态 batch 可运行两种 batch，静态 batch 必须与请求一致，例如静态 batch=1 时先把 `matrix.batch` 改成 `[1]`。结果保留在 `/work/results`，详见 [性能口径](migraphx.md)。本步骤不要求 Torch 参与推理，也不自动下载/导出模型。

同一原生环境也可使用 [BERT 独立配置](bert.md)。该配置默认使用当前环境的 rocBLAS，不要求另行准备外置调优库；仍须提供满足 BERT 输入合同的实际 ONNX 模型。

退出容器后在原宿主机终端 `docker stop "$NAME"`，确认无需保留容器时可执行 `docker rm "$NAME"`。宿主机 `$WORK` 内的 wheel、预检报告、虚拟环境和测量结果仍保留。
