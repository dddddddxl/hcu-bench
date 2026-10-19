# 随仓库提供的测试源码

这里只保存原生测试及构建所需源码，不包含驱动、DTK、Torch、模型权重或机器上的运行日志。
固定提交、来源及目录校验和见 [sources.json](sources.json)，第三方许可证保留在对应目录。
不使用 Git submodule，普通 `git clone` 即可得到这些文件。

| 目录 | 用途 | 范围 |
| --- | --- | --- |
| rccl-tests | MPI AllReduce | src、verifiable 及构建文件，NVIDIA/AMD 原许可证 |
| rocm-bandwidth-test | CPU↔HCU、HCU↔HCU | ROCm 6.4.2 源码提交中的原生 2.6.0；不是新版重写项目的同名标签 |
| hcu-stream | zd1/zd2 24 项 | 用户提供的 main-staging-1.0，含 main.cpp 引用的内核；未移除编译失败的用例 |
| deepep-tests | DeepEP 低延迟压测 | 用户提供的原生测试和 utils；使用已验证 daily 的 11 字段 handle 接口 |
| mooncake | RDMA/VRAM 传输 | HYGON-AI 固定提交的 transfer-engine、common、构建文件，及固定版 pybind11/yalantinglibs |
| gsm8k | 简单精度测试 | 官方 test.jsonl，MIT；沿用评测器的前 5 条 few-shot、后 100 条评分方式 |

## 来源与修改

- hcu-stream 和 DeepEP 测试由提供方于 2026-10-19 明确授权公开分发。提供的压缩包没有这两部分的根许可证，不另行声称它们采用 MIT/Apache；已有版权说明和 GPU-stream 子项目许可证均保留。
- DeepEP 的唯一接口适配：读取接收布局前断言 `len(handle) == 11`，随后从 `handle[6]` 读取布局，而不是原来的 `handle[4]`。没有删除正确性检查。其他 wheel 接口不保证兼容，可在 YAML 指定另一份已验证的测试脚本。
- 源码导入时统一文本换行为 LF。构建在缓存副本里进行，不修改这里的源码。
- 月英 gfx92a 的 hcu-stream 使用 AICC 和数值架构宏 9210。已有 gfx938/标准 DTK 组合存在原生编译错误，自动编译会保留报错，不跳过内核或偷偷换库。
- Mooncake 构建只开启 Hygon/HIP 的原生 transfer-engine 示例，不构建 Store、EP、TENT 和单元测试；yalantinglibs 从本目录读取，禁用联网获取。系统 C/C++ 开发库仍来自执行镜像。
- 更新源码后使用 `python tools/vendor_checksums.py` 更新清单，并提交来源变更和回归结果。

HCU 是本工具的统一称呼；第三方源码中的原始名称、版权、API 和命令参数不作文字替换。
