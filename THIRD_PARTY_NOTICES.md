# Third-Party Notices

HCU Bench 的原创编排框架采用 Apache-2.0。以下随仓组件不由根许可证重新授权，
原始版权、许可证、NOTICE 及其适用条款须保留。本文是来源索引，不代替原许可证。
准确版本、来源 URL、归档 SHA-256、修改说明和目录校验和见 [vendor/sources.json](vendor/sources.json)。

| 本地路径 | 来源/固定版本 | 许可文件或当前核验状态 |
| --- | --- | --- |
| `vendor/rccl-tests` | ROCm/rccl-tests，`ae3e6357cb3514d5649b3b5dfaa80cb2d2af3532` | [原始 LICENSE.txt](vendor/rccl-tests/LICENSE.txt)，保留 NVIDIA/AMD 声明 |
| `vendor/rocm-bandwidth-test` | ROCm/rocm_bandwidth_test，`84b8ddd2686be9bd3e438126b44e6bb10d94d522` | [原始 LICENSE.txt](vendor/rocm-bandwidth-test/LICENSE.txt)；修改见来源登记 |
| `vendor/mooncake` | HYGON-AI/mooncake-das，`237c9167cc8778ffbbc44460f53d590822a602eb` | [LICENSE-APACHE](vendor/mooncake/LICENSE-APACHE) |
| `vendor/mooncake/extern/pybind11` | pybind/pybind11，`58c382a8e3d7081364d2f5c62e7f429f0412743b` | [原始 LICENSE](vendor/mooncake/extern/pybind11/LICENSE)；需按全文核对，不因嵌套于 Mooncake 而改为 Apache-2.0 |
| `vendor/mooncake/extern/yalantinglibs` | alibaba/yalantinglibs，`7801bc9ad9021781f15217552214e325a1cf7373` | [原始 LICENSE](vendor/mooncake/extern/yalantinglibs/LICENSE) |
| `vendor/evalscope` | EvalScope 1.13.0，来源归档校验和见登记 | [原始 LICENSE](vendor/evalscope/LICENSE)；含源码归档，核验须包含归档内部内容 |
| `vendor/gsm8k` | openai/grade-school-math，`3101c7d5072418e28b9008a6636bde82a006892c` | [原始 LICENSE](vendor/gsm8k/LICENSE)；包括测试集及训练集前5条 |
| `vendor/operator-cases` | 提供方附件，版本及归档校验和见登记 | 部分材料未附根许可证；登记中记录了公开分发授权，须核实授权主体、范围及证据，不自动认定 MIT/Apache-2.0 |
| `vendor/hcu-stream` | 提供方 `main-staging-1.0`，目录校验和见登记 | 外层工程未附根许可证；[GPU-stream 子项目许可证](vendor/hcu-stream/GPU-stream/LICENSE)不代表整个目录的许可；公开分发授权待核验 |
| `vendor/deepep-tests` | 提供方 `tests (3).zip`，目录校验和及修改说明见登记 | 未登记完整根许可证；公开分发授权及实际来源需核验 |
| `vendor/hipdnn-api-test` | DTK api_test 固定归档，SHA-256 见登记 | 未附根许可证；保留逐文件 MIT SPDX/版权声明；须核验全部归档内容及分发权限，用户要求打包不等于权利人授权 |

## 发布前核验

- 对照固定来源核验许可证、版权和修改说明，不删除真实原权利人声明。
- 未明确许可或仅有提供方授权记录的内容，须由维护者补齐可核查的权利证明并完成人工审核。
- MIT/BSD 第三方文件保持原声明；HYGON 对第三方的贡献按经核实的修改范围另行登记，不能给未修改第三方源码批量添加 HYGON 版权。
- 所有归档内部源码与嵌套依赖同样属于发布检查范围。模型权重、驱动和 SDK 未由本文授予分发许可。
