# Wan2.2 文生视频性能

Wan 属于 `e2e_perf`，`e2e` 和 `all` 也会选中已启用的 Wan 项。它是独立视频测试包，不使用 EvalScope、tokenizer、GSM8K，也不会因 `models: all` 被送进语言模型精度测试。只支持单机单卡或四卡，按 YAML 顺序串行执行。

## 测试配置

依据[月英性能表的 Wan 页](https://kcnm6g5dkw5p.feishu.cn/sheets/PcLIsjCPOhMLRMtGVi9cCYRDnse?sheet=OlX3DI)的 HCU 部署命令提供两个 profile：

| profile | 卡数 | FSDP | text encoder CPU offload | DiT layerwise offload | Ulysses / ring | performance mode |
| --- | --- | --- | --- | --- | --- | --- |
| tp1 | 1 | false | true | true | 1 / 1 | memory |
| tp4 | 4 | true | true | false | 4 / 1 | 原生默认 |

两项均使用 `sla_attn`、`sla_topk=0.25`、`GPU_MAX_HW_QUEUES=3`，DiT 整体 CPU offload 和 VAE CPU offload 关闭；服务端初始化预热为 1280×720、1 步。表中的 tp1/tp4 是配置标签，不将它解释成语言模型 TP。不启用单独 BW1100 优化示例的 Cache-DiT，不改变宿主机频率或 NUMA 设置。

表里没有给出客户端 prompt、帧数、步数、seed、预热和正式次数。**按用户确认，初版采用[官方 Wan 指南](https://github.com/HYGON-AI/inference-cookbook-das/blob/main/docs/model-deployment/sglang/wan2.2-t2v.md)的示例 prompt、1280×720、81 帧、16 FPS、40 步、seed 42、guidance_scale 4.0，配置 1 次预热、1 次正式生成。** 不宣称与原表完全同口径。官方示例中的 embedded_guidance_scale 不在已核对的 API schema 中，因此不发送，也不擅自改名为其他 guidance 字段；需要该设置时先核实目标接口。

## 使用

```bash
cp configs/wan.template.yaml configs/bench.yaml
# 填仓库、输出、模型路径及卡号；模型目录须有 model_index.json，不自动下载权重。
# vars.request 已有官方格式初值，可调整；专用模板默认两项都开，可将不测项 enabled 改 false。
bash bench.sh plan --suite e2e_perf
bash bench.sh e2e_perf
```

部署参数放在 `model_scripts/wan22_t2v.sh`，YAML 的 `start_script` 可改为自己的脚本；最后必须 `exec` 前台启动，读取 `BENCH_MODEL_PATH`、`BENCH_HOST`、`BENCH_PORT`、`BENCH_OUTPUT_DIR` 和已分配的 `HIP_VISIBLE_DEVICES`。表中的物理卡 6 不会硬编码到脚本。自动容器/SSH 沿用已有节点及容器接口。

将专用模板中两个 Wan 测试块、节点和相关变量合入个人 `bench.yaml`，启用后 `bash bench.sh all` 即可包含它们。保留 Qwen 流水线的 `models` 配置即可，两类模型不会混用启动流程。

## 依赖与结果

镜像须已有支持 `sglang serve --model-type diffusion` 的 SGLang，以及模型对应的扩散依赖与 SLA 后端。启动脚本只识别表里出现的 `--warmup-mode server` / `--warmup` 两种接口；不认识就报错。服务就绪后检查视频 API 路由；每个请求必须生成可核对 id 的原生 `perf_dump_path` 报告，否则失败。该接口手动解析 JSON，OpenAPI 中只有 multipart 请求体，不能以不存在的 JSON schema 判定它不支持视频。不自动安装扩散库或回退到另一种推理算法。

接口及原生 JSON 解析依据 [sglang-das 6c946e9 的 Video API](https://github.com/HYGON-AI/sglang-das/blob/6c946e92fa4ec1755c8a2a8f8800bd2e6795213e/python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py)和[性能报告定义](https://github.com/HYGON-AI/sglang-das/blob/6c946e92fa4ec1755c8a2a8f8800bd2e6795213e/python/sglang/multimodal_gen/runtime/utils/perf_logger.py)。版本接口不匹配会保留原始错误，不能保证任意标有 0.5.12 的镜像都适用。

| 输出指标 | 来源 |
| --- | --- |
| Pixel 总耗时 | 本次自有服务日志 `Pixel data generated successfully in ... seconds` |
| Denoise / Decode 耗时 | 每个请求原生性能 JSON 的 DenoisingStage / DecodingStage，毫秒转秒 |
| 单步耗时 | 原生日志 `average time per step`，不以总耗时除以配置步数代替 |
| API 推理耗时 | 视频结果的 `inference_time_s`；缺失时留空并警告，不补 0 |
| 客户端耗时 | 提交至轮询 completed 的墙钟时间；独立记录，不冒充 API 推理耗时 |

逐个提交并轮询至 completed，原生报告的 request_id 必须与视频 id 匹配。预热请求不发布性能指标；Pixel / 单步日志条数与完成请求数不符、阶段缺失、非有限时间、HTTP 失败或生成失败都不能记为成功。

每个 profile 独立启动、清理自有服务，保留 server.log、请求/响应、原生阶段 JSON、measurements.csv 和本次生成视频；通过现有归档接口收集，超出归档上限只警告并单列归档状态。汇总进入 metrics.csv、e2e.csv 和 e2e.md，单位为秒，不产生 tok/s 或精度通过结论。

目前为接口接入与 CPU 替身验证，尚无本适配器的 HCU 实测结果。
