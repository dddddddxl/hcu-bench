# 测试包与容器接口

一个测试包只需 manifest 和已有测试入口/包装脚本。框架负责生命周期，同事负责原生参数、正确性检查和指标解析。

## 最小测试包

```yaml
version: 1
id: deepep.low-latency
suite: deepep
parameters:
  tokens: {type: integer, required: true, min: 1}
command: [python3, -u, /workspace/run_test.py, --tokens, "${p_tokens}", --node-rank, "${node_rank}"]
cwd: /workspace/tests
env: {}
result_protocol: bench-jsonl-v1
requires_metrics: true
artifacts: ["${case_dir}/reports/*.json", "${case_dir}/reports/*.csv"]
artifact_limit_mb: 100
```

五类 suite 是 `rccl/deepep/mooncake/operators/e2e`。`command` 必须是 argv 数组；需要 shell 时显式写 `[bash, -lc, '...']`，优先直接指向脚本。参数必须声明类型和支持范围，未知参数不透传。

可用模板变量：`run_id/case_id/test_id/node_id/node_rank/nnodes/case_dir/host_case_dir/run_dir/gpu_ids/python` 以及 `p_参数名`。本地非容器测试还可用 `pack_dir`。`${run_dir}` 是控制机路径，`${host_case_dir}` 是目标宿主机目录；使用容器时 `${case_dir}` 是容器内目录，否则与宿主机目录相同。`${python}` 为宿主机解释器配置，容器内解释器应在测试入口命令中明确填写。Shell 原生 `$VAR` 需写成 `$$VAR`，避免被模板处理。

输出文件建议落在 `${case_dir}`，避免不同运行覆盖同一个报告。`${gpu_ids}` 不会自动绑定设备；按实际测试工具设置 env 或设备参数。算子数据集路径可作为普通 string 参数，不将百万行 CSV 展开成框架参数矩阵。

## 标准结果

包装脚本逐轮向 stdout 输出一行 `BENCH_RESULT ` 后跟 JSON：

```json
{"measurement_status":"measured","correctness":"not_checked","sample_id":0,"metrics":[{"name":"dispatch_combine_effective_bandwidth","value":70.2,"unit":"GB/s","statistic":"mean"},{"name":"dispatch_combine_latency","value":313.9,"unit":"us","statistic":"mean"}]}
```

普通原生输出仍保留为日志。允许额外保存 provider、timing_method、source_file/source_row、reference_time_ns 等字段；框架的 suite/node/case/params 身份不能被测试输出覆盖。

- `measurement_status`：measured / invalid / skipped；mock 另用 simulated。
- `correctness`：passed / failed / not_checked，不根据进程退出码猜测。
- 无效 profiler 轮次输出 invalid、空 metrics 和 reason；不要伪造耗时或吞吐。
- 原生结果存在但尚未提供解析器时，用 `result_protocol: none`、`requires_metrics: false`，只记录执行完成和原生报告，不称为性能测试通过。
- 包装脚本不能确认本次子进程清理时，保留证据并使用保留退出码 125；框架标记 cleanup_unconfirmed，停止后续用例。
- `artifacts` 支持文件或 glob，仅普通文件传回，软链接不收集。默认每节点每用例总限额 100 MiB，可在 manifest 修改；文件分块传输并校验 SHA256，失败保留 `.partial` 和警告。

## 容器启动预留接口

```yaml
containers:
  test_image:
    mode: managed
    work_root: /workspace/bench-runs
    name: "bench-${run_id}-${case_id}-${node_id}"
    start: [bash, /public/dxl/start_container.sh, "${container_name}"]
    check: [bash, /public/dxl/check_container_running.sh, "${container_name}"]
    stop: [bash, /public/dxl/stop_container.sh, "${container_name}"]
    hook_timeout_s: 120
    exec_prefix: [docker, exec, -i, "${container_name}", python3, -u]
```

start/check/stop 在目标宿主机执行。start 必须新建指定名称的容器，不能偷换成复用其他容器。check 退出码约定：0 表示正在运行，1 表示已停止或不存在，其他值表示检查失败。stop 结束本次自有容器，随后 check 必须为 1；检查失败不能当成清理成功。框架不自动拼接镜像、设备、权限和挂载。

`work_root` 必须显式指定容器内可写目录；宿主机目录与容器内目录分别配置，不假定挂载关系。报告由容器内 worker 读取并传回，不要求报告目录挂载到宿主机。

复用容器设置 `mode: existing`、固定 name、work_root、check，可设置 exec_prefix；禁止 start/stop。Docker exec 不分配 TTY，保留 `-i` 用于取消控制。exec_prefix 必须以可运行 Python 的入口结尾，框架随后追加 `-c` 和轻量 worker；不向镜像安装包或复制 DTK。

自有容器名称包含 run/case/node，避免误认其他容器。工作进程只清理自己创建的进程组；SSH 断连后 worker 会尝试退出，无法确认时标记 cleanup_unconfirmed，不继续后续测试。

参考模板：`configs/remote.template.yaml` 与 `testpacks/deepep/manifest.template.yaml`。其中 YOUR_* 和 wrapper 入口必须由你提供并确认，模板不是可以直接跑的 DeepEP 测试。
