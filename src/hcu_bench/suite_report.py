"""Human-readable views of recorded results; never infer outcomes from log text."""
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import unicodedata
import uuid

LABELS = {"completed": "完成", "duration_reached": "达到计划时长", "failed": "失败",
          "cancelled": "取消", "not_run": "未执行", "pending": "等待", "running": "运行中",
          "queued": "排队", "timed_out": "超时", "invalid": "结果无效",
          "cleanup_unconfirmed": "清理未确认", "passed": "通过", "not_checked": "未校验"}
ORDER = ("rccl", "pcie", "memory", "operators", "migraphx", "deepep", "mooncake", "e2e_acc", "e2e_perf")
LIMIT = 12


def text(value):
    return str(value).replace("\r", " ").replace("\n", " ")


def number(value):
    return "无结果" if value is None else f"{value:.6g}"


def atomic_text(path, content):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Document:
    def __init__(self):
        self.markdown, self.plain = [], []
        self.case_labels = {}

    def case_label(self, case_id):
        return self.case_labels[case_id]

    def heading(self, title, level=2):
        self.markdown += ["#" * level + " " + title, ""]
        self.plain += [title, "=" * min(72, len(title) * 2), ""]

    def paragraph(self, value):
        self.markdown += [value, ""]
        self.plain += [value, ""]

    def table(self, headers, rows):
        rows = [[text(cell) for cell in row] for row in rows]
        if not rows:
            return
        self.markdown += ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        self.markdown += ["| " + " | ".join(c.replace("|", "\\|") for c in row) + " |" for row in rows] + [""]
        def width(value):
            return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in value)
        widths = [max(width(row[i]) for row in [headers] + rows) for i in range(len(headers))]
        self.plain += ["  ".join(c + " " * (widths[i] - width(c)) for i, c in enumerate(row)) for row in [headers] + rows] + [""]

    def link(self, label, path):
        target = Path(path).as_posix()
        self.markdown += [f"[{label}]({target})", ""]
        self.plain += [f"{label}：{target}", ""]


def usable(sample):
    return sample.get("measurement_status") in ("measured", "simulated") and sample.get("correctness") != "failed"


def value(sample, name=None, statistic=None, suffix=None, unit=None):
    matches = [m for m in sample.get("metrics", []) if (name is None or m["name"] == name)
               and (statistic is None or m["statistic"] == statistic)
               and (suffix is None or m["name"].endswith(suffix))
               and (unit is None or m["unit"] == unit)]
    return matches[0]["value"] if len(matches) == 1 else None


def entries_for(state, samples):
    entries = {}
    for case_id, case in state["cases"].items():
        session = case.get("params", {}).get("session")
        tasks = session["tasks"] if session else [dict(test_id=case["test_id"], suite=case["suite"])]
        for task in tasks:
            key = (case_id, task["suite"], task["test_id"])
            entries[key] = dict(case_id=case_id, case=case, suite=task["suite"], test_id=task["test_id"],
                                model=session["model_id"] if session else "", session=bool(session), samples=[])
    for sample in samples:
        key = (sample["case_id"], sample["suite"], sample["test_id"])
        if key not in entries:
            entries[key] = dict(case_id=key[0], case=state["cases"][key[0]], suite=key[1], test_id=key[2],
                                model=sample.get("model_id", ""), session=False, samples=[])
        entries[key]["samples"].append(sample)
    return list(entries.values())


def reason(entry):
    case = entry["case"]
    messages = [case[k] for k in ("reason", "error") if case.get(k)]
    messages += [s["reason"] for s in entry["samples"] if s.get("reason")]
    for node, result in case.get("nodes", {}).items():
        messages += [f"{node}: {result[k]}" for k in ("error", "reason") if result.get(k)]
        if result.get("returncode") not in (None, 0):
            messages.append(f"{node}: returncode={result['returncode']}")
        if result.get("phase"):
            messages.append(f"{node}: phase={result['phase']}")
        if result.get("cleanup") not in (None, "confirmed"):
            messages.append(f"{node}: cleanup={result['cleanup']}")
    if entry["session"] and not entry["samples"]:
        messages.insert(0, "该子任务无结果；父模型会话状态不作为子任务通过依据")
    if case["status"] == "duration_reached":
        messages.append("计划终止不等于压力循环自然结束或完整稳定性通过")
    if case["status"] not in ("completed", "pending", "running", "queued") and not any(case.get(k) for k in ("reason", "error")):
        messages.append("具体原因未确认，见日志；不从退出码推断OOM或接口错误")
    return text("；".join(dict.fromkeys(map(str, messages)))) or "—"


def export_details(path, entries):
    fields = ("case_id", "test_id", "node", "repeat", "model", "case_status", "measurement_status", "correctness",
              "simulated", "native_rank", "cycle", "sample_id", "observed_at", "params", "workload", "sample_context",
              "name", "value", "unit", "statistic")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            for sample in entry["samples"]:
                if not usable(sample):
                    continue
                for metric in sample["metrics"]:
                    writer.writerow({**{k: sample.get(k, "") for k in ("case_id", "test_id", "node", "repeat", "measurement_status", "correctness", "simulated", "native_rank", "cycle", "sample_id", "observed_at")},
                                     "model": sample.get("model_id", ""), "case_status": entry["case"]["status"],
                                     "params": json.dumps(sample.get("params", {}), sort_keys=True, ensure_ascii=False),
                                     "workload": json.dumps(sample.get("workload"), sort_keys=True, ensure_ascii=False),
                                     "sample_context": json.dumps({k: v for k, v in sample.items() if k not in ("metrics", "params", "workload")}, sort_keys=True, ensure_ascii=False),
                                     **{k: metric[k] for k in ("name", "value", "unit", "statistic")}})


def render_metrics(doc, suite, samples):
    # Keep every measurement's node/repeat/configuration identity; never average rows here.
    if suite == "rccl":
        rows = [[s.get("collective", s["test_id"]), doc.case_label(s["case_id"]), s["node"], s.get("repeat", 0),
                 s.get("size_bytes", "未知"), s.get("mode", "未知"), number(value(s, suffix="_algbw", unit="GB/s")),
                 number(value(s, suffix="_busbw", unit="GB/s")), number(value(s, suffix="_latency", unit="us"))] for s in samples]
        doc.paragraph("algbw / busbw单位GB/s，延迟单位us；两个带宽不能互换，不代表网卡/RDMA带宽。原地/非原地分别列出。")
        doc.table(["通信项", "用例ID", "节点", "轮次", "消息字节数", "模式", "algbw", "busbw", "延迟"], rows)
    elif suite == "pcie":
        groups = defaultdict(list)
        for s in samples:
            groups[(s["case_id"], s["node"], s.get("repeat", 0), json.dumps(s.get("params", {}), sort_keys=True),
                    bool(s.get("simulated") or s.get("measurement_status") == "simulated"), s.get("kind", "未知"), s.get("size_bytes"))].append(s)
        labels = {"cpu_to_gpu": "CPU→HCU", "gpu_to_cpu": "HCU→CPU", "gpu_to_gpu": "HCU→HCU", "cpu_gpu_bidir": "CPU↔HCU", "gpu_gpu_bidir": "HCU↔HCU"}
        rows = []
        for (case_id, node, repeat, _, simulated, kind, size), group in groups.items():
            avg, peak = [value(s, suffix="_avg", unit="GB/s") for s in group], [value(s, suffix="_peak", unit="GB/s") for s in group]
            span = lambda values: "无完整结果" if None in values else f"{min(values):.3f}–{max(values):.3f}"
            rows.append([doc.case_label(case_id), node, repeat, "模拟" if simulated else "实测", labels.get(kind, kind), size, len(group), span(avg), span(peak)])
        doc.paragraph("单位GB/s。范围是同一配置下不同路径的最小值至最大值，不是多轮均值或并发总吞吐；双向已包含两方向。未自动核实独占条件，性能适用性须结合资源记录判断。")
        doc.table(["用例", "节点", "轮次", "数据类型", "路径", "消息字节数", "路径数", "Avg范围", "Peak范围"], rows)
    elif suite == "e2e_perf":
        llm = [s for s in samples if value(s, "output_throughput", "mean", unit="tok/s") is not None]
        wan = [s for s in samples if value(s, "pixel_total_time", "value", unit="s") is not None]
        if llm:
            by_model = defaultdict(list)
            for s in llm:
                by_model[(s.get("model_id", "未知"), s["case_id"], s["node"], s.get("repeat", 0))].append(s)
            doc.paragraph("输入/输出为目标token长度，实际长度及P99见明细。吞吐tok/s，TTFT/TPOT为均值ms；每模型/节点/外层轮次分表，不合并。")
            for (model, case_id, node, repeat), group in by_model.items():
                doc.heading(f"{model} / {doc.case_label(case_id)} / {node} / 外层轮次{repeat}", 3)
                rows = []
                for s in group:
                    w = s.get("workload") or {}
                    rows.append([f"{w.get('input_tokens', '?')}/{w.get('output_tokens', '?')}", w.get("concurrency", "未知"),
                                 w.get("repeat", "未知"), number(value(s, "output_throughput", "mean", unit="tok/s")),
                                 number(value(s, "ttft", "mean", unit="ms")), number(value(s, "tpot", "mean", unit="ms")),
                                 f"{number(value(s, 'successful_requests'))}/{number(value(s, 'failed_requests'))}"])
                doc.table(["输入/输出", "并发", "测量轮次", "输出吞吐", "TTFT", "TPOT", "成功/失败"], rows[:LIMIT])
        if wan:
            rows = []
            for s in wan:
                w = s.get("workload") or {}
                rows.append([s.get("model_id", "未知"), doc.case_label(s["case_id"]), s["node"], s.get("repeat", 0),
                             f"{w.get('profile', '?')}/{w.get('size', '?')}/{w.get('num_frames', '?')}帧/{w.get('num_inference_steps', '?')}步",
                             *[number(value(s, k, "value", unit="s")) for k in ("pixel_total_time", "denoise_time", "denoise_step_time", "decode_time", "api_inference_time")]])
            doc.paragraph("Wan各项耗时单位秒，计时边界不同；不使用LLM token吞吐。")
            doc.table(["模型", "用例ID", "节点", "轮次", "视频配置", "Pixel", "Denoise", "单步", "Decode", "API"], rows[:LIMIT])
        other = [s for s in samples if s not in llm and s not in wan]
        if other:
            generic_metrics(doc, other)
    elif suite == "e2e_acc":
        rows = []
        for s in samples:
            p = s.get("task_params", s.get("params", {}))
            accuracy = value(s, "gsm8k_accuracy", "mean", unit="ratio")
            rows.append([s.get("model_id", "未知"), doc.case_label(s["case_id"]), s["node"], s.get("repeat", 0),
                         number(value(s, "gsm8k_evaluated_examples")), p.get("num_shots", "未知"),
                         "无结果" if accuracy is None else f"{accuracy * 100:.2f}%", p.get("max_tokens", "未知"),
                         p.get("chat_template_kwargs", {}).get("enable_thinking", "未记录"), s.get("accuracy_judgement", "未记录")])
        doc.paragraph("分数、评分题数及提示口径分别记录；未记录的thinking开关不推断为false。只报告分数不等于通过某个验收阈值。")
        doc.table(["模型", "用例ID", "节点", "轮次", "评分题数", "shot", "分数", "输出上限", "thinking", "判定策略"], rows[:LIMIT])
    else:
        generic_metrics(doc, samples)
    doc.paragraph("表格仅展示关键指标或前12条明细，完整参数、各rank、所有形状/测试点及统计口径见本suite CSV；缺失值不填0。")


def generic_metrics(doc, samples):
    rows = []
    for s in samples:
        point = ";".join(f"{k}={s[k]}" for k in ("native_rank", "cycle", "sample_id", "gpu_id", "preset", "test", "wave", "lds_arg_kib") if k in s)
        for m in s.get("metrics", []):
            rows.append([doc.case_label(s["case_id"]), s["node"], s.get("repeat", 0), point or "—", m["name"], number(m["value"]), m["unit"], m["statistic"]])
    doc.table(["用例", "节点", "轮次", "测量位置/配方", "指标", "数值", "单位", "统计口径"], rows[:LIMIT])


def write_suite_report(store, state, samples):
    entries = entries_for(state, samples)
    groups = defaultdict(list)
    for entry in entries:
        groups[entry["suite"]].append(entry)
    doc = Document()
    doc.case_labels = {key: f"C{index + 1:02}" for index, key in enumerate(state["cases"])}
    doc.heading("HCU Bench 测试结果", 1)
    doc.paragraph(f"运行编号：{state['run_id']}；方案：{state.get('name', '未记录')}；整体状态：{LABELS.get(state['status'], state['status'])}。")
    doc.paragraph(f"开始：{state.get('started_at', '未开始')}；结束：{state.get('finished_at', '未结束')}。时间使用宿主机记录。")
    counts = Counter(c["status"] for c in state["cases"].values())
    doc.paragraph(f"执行用例总数：{len(state['cases'])}；" + " / ".join(f"{LABELS.get(k, k)}{v}项" for k, v in counts.items()))
    doc.paragraph("模型会话总数按会话计一次，下方按精度/性能子任务分别展示，suite行数不能直接相加作为总数。完成不等于正确性通过；无测量不填0；轮次从0起。")
    if state.get("error"):
        doc.paragraph("运行错误：" + text(state["error"]))
    if any(s.get("simulated") or s.get("measurement_status") == "simulated" for s in samples) or any(c.get("simulated") for c in state["cases"].values()):
        doc.paragraph("[SIMULATED - NOT HARDWARE DATA] 存在模拟用例；模拟指标不作为真实性能。各行模拟标记见suite明细。")
    doc.heading("执行概览")
    suites = sorted(groups, key=lambda s: (ORDER.index(s) if s in ORDER else len(ORDER), s))
    rows = []
    for suite in suites:
        group = groups[suite]
        statuses = Counter(e["case"]["status"] for e in group)
        count = sum(usable(s) for e in group for s in e["samples"])
        rows.append([suite, " / ".join(f"{LABELS.get(k, k)}{v}" for k, v in statuses.items()),
                     count, sum(not usable(s) for e in group for s in e["samples"]),
                     "父会话状态；见子任务测量" if any(e["session"] for e in group) else "执行状态不代替验收结论"])
    doc.table(["Suite", "执行/父会话状态", "可展示样本", "无效/未测量样本", "说明"], rows)
    details = store.path / "report-details"
    details.mkdir(exist_ok=True)
    for index, suite in enumerate(suites):
        group = groups[suite]
        doc.heading(suite)
        rows = []
        for e in group:
            ss = e["samples"]
            correction = Counter(s.get("correctness", "未记录") for s in ss)
            p = e["case"].get("params", {})
            label = e["model"] or p.get("collective") or p.get("recipe") or e["test_id"]
            label += "".join(f" / {k}={p[k]}" for k in ("dtype", "batch") if k in p)
            rows.append([label, doc.case_label(e["case_id"]), LABELS.get(e["case"]["status"], e["case"]["status"]),
                         sum(usable(s) for s in ss), "/".join(f"{LABELS.get(k, k)}{v}" for k, v in correction.items()) or "无结果", reason(e)])
        doc.table(["项目/模型", "用例", "执行/父会话状态", "可展示样本", "正确性(样本数)", "原因/说明"], rows)
        configurations = defaultdict(list)
        for e in group:
            p = e["case"].get("params", {})
            keys = ("ranks", "datatype", "warmup_iters", "iters", "size_mib", "recipe", "dtype", "batch", "presets",
                    "model_mhz", "num_processes", "num_tokens", "hidden", "num_topk", "num_experts", "stress_iterations")
            settings = "; ".join(f"{key}={p[key]}" for key in keys if key in p)
            if settings:
                configurations[settings].append(doc.case_label(e["case_id"]))
        for settings, labels in configurations.items():
            doc.paragraph(f"{'、'.join(labels)} 配置：{settings}")
        usable_samples = [s for e in group for s in e["samples"] if usable(s)]
        if usable_samples:
            if suite in ("rccl", "pcie", "e2e_acc", "e2e_perf"):
                render_metrics(doc, suite, usable_samples)
            else:
                for e in group:
                    selected = [s for s in e["samples"] if usable(s)]
                    if selected:
                        doc.heading(f"{doc.case_label(e['case_id'])} 关键指标", 3)
                        render_metrics(doc, suite, selected)
        else:
            doc.paragraph("未获得可展示的性能/精度结果；不以进程退出或依赖准备作为测量完成。")
        target = details / f"suite-{index + 1}.csv"
        export_details(target, group)
        doc.link(f"{suite}完整指标和参数", target.relative_to(store.path))
        for e in group:
            for node in e["case"].get("nodes", {}):
                # Derive paths within this run, not remote container absolute paths.
                path = Path("logs") / e["case_id"] / f"{node}.log"
                if (store.path / path).is_file():
                    doc.link(f"{doc.case_label(e['case_id'])}/{node}日志", path)
    doc.heading("未进入本次执行计划的配置")
    resolved = store.path / "resolved.json"
    if resolved.is_file():
        config = json.loads(resolved.read_text(encoding="utf-8"))
        planned = {e["test_id"] for e in entries}
        rows = [[key, t.get("suite", "未知"), "配置禁用" if t.get("enabled") is False else "未进入本次计划（不等于依赖失败）"]
                for key, t in config.get("tests", {}).items() if key not in planned]
        doc.table(["测试块", "Suite", "状态"], rows)
        if not rows:
            doc.paragraph("未发现计划外测试块；未声明的suite不能据此认定已测试。")
    else:
        doc.paragraph("未保存resolved.json，无法确认哪些配置被禁用；不推断未声明的suite状态。")
    doc.heading("归档与阅读说明")
    doc.table(["用例", "原始用例ID"], [[label, key] for key, label in doc.case_labels.items()])
    archives = Counter(r.get("artifact_status", "未记录") for c in state["cases"].values() for r in c.get("nodes", {}).values())
    doc.paragraph("节点归档状态：" + (" / ".join(f"{k}={v}" for k, v in archives.items()) or "未记录"))
    doc.paragraph("归档告警不覆盖已采性能；详见artifacts.jsonl。达到计划时长不等于自然完成压力测试；取消/失败时已采数据不自动成为正式验收结果。")
    doc.link("完整标准指标", "metrics.csv")
    if (store.path / "e2e.md").is_file():
        doc.link("模型原始分组报告", "e2e.md")
    doc.link("原始执行状态", "state.json")
    atomic_text(store.path / "summary.md", "\n".join(doc.markdown) + "\n")
    atomic_text(store.path / "summary.txt", "\n".join(doc.plain) + "\n")
