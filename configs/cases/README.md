# 独立测例配置

此目录的配置不会由主配置自动加载。每个文件仅包含对应测例，选择文件后直接执行其中启用的项目：

FA FP8、GEMM、hipDNN Conv及Wan TP4已显式写入主模板，默认all执行；此处保留其独立复测入口。memory、MIGraphX、长压力等仍需单独选择。

```bash
bash bench.sh configs/cases/memory.yaml all
bash bench.sh configs/cases/migraphx-resnet50.yaml all
bash bench.sh configs/cases/migraphx-bert.yaml all
bash bench.sh configs/cases/gemm.yaml all
bash bench.sh configs/cases/hipdnn-conv.yaml all
bash bench.sh configs/cases/fa-fp8.yaml all
bash bench.sh configs/cases/operator-stress.yaml all
bash bench.sh configs/cases/mooncake.yaml all
bash bench.sh configs/cases/wan.yaml all
```

执行前修改该文件的公共路径、HCU分配和专用输入，具体说明见[主README](../../README.md)。模板具有 `YOUR_*` 占位符，未配置的文件不能直接作为节点运行配置。

- 除Mooncake外，默认在已有测试容器内运行，不需要Docker socket。
- Mooncake需要至少两台节点；示例通过SSH调用两端已有Docker容器，使用密钥或SSH配置，不保存密码。
- 独立压力测试默认为100次smoke。改为100万次正式压力时，须同步调整原生和框架超时。
- Wan默认TP4；单卡TP1须同时修改HCU列表和profile，TP1尚未完成实机验证。
- GEMM默认perf沿用原生benchmark计时，correctness=not_checked；check显式启用原生比较，非空max_norm_error追加norm检查。hipDNN test_conv使用原脚本与原生校验，不额外设置迭代或预热。
- FA FP8在实际容器内检查已安装扩展的19参数接口；匹配才测，不匹配明确记为skipped并继续，不计作性能通过。

如需加入主计划，合并所需测试块和节点定义至 `configs/bench.yaml`。复制测试块后，`pack` 从 `../../testpacks/...` 改为 `../testpacks/...`。仅增加suite名称不会自动导入本目录文件。
