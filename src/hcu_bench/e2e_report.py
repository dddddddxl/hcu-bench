"""One row per model/task/workload; never average percentiles across runs."""
import csv
import json


def build_e2e_report(store, state):
    rows, seen = [], set()
    path = store.path / "results.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
            if not line.endswith("\n"):
                continue
            sample = json.loads(line)
            if not sample.get("e2e_task"):
                continue
            case_id, test_id = sample["case_id"], sample["test_id"]
            seen.add((case_id, test_id))
            point = sample.get("workload") or {}
            row = dict(model=sample["model_id"], suite=sample["suite"], test_id=test_id, case_id=case_id,
                       case_status=state["cases"][case_id]["status"],
                       measurement_status=sample["measurement_status"], correctness=sample["correctness"],
                       reason=sample.get("reason", ""), **point)
            for metric in sample["metrics"]:
                row[f"{metric['name']}_{metric['statistic']} ({metric['unit']})"] = metric["value"]
            rows.append(row)
    for case_id, case in state["cases"].items():
        session = case.get("params", {}).get("session")
        if not session:
            continue
        for task in session["tasks"]:
            if (case_id, task["test_id"]) not in seen:
                rows.append(dict(model=session["model_id"], suite=task["suite"], test_id=task["test_id"],
                                 case_id=case_id, case_status=case["status"], measurement_status="not_measured",
                                 reason="No result: inspect session state and logs"))
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (store.path / "e2e.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    columns = ["model", "suite", "input_tokens", "output_tokens", "concurrency", "repeat", "measurement_status",
               "gsm8k_accuracy_mean (ratio)", "output_throughput_mean (tok/s)", "ttft_p99 (ms)", "case_status"]
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["# End-to-End Results", "", "Separate rows per workload/repeat; see e2e.csv for all metrics and failures.", "",
             "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines += ["| " + " | ".join(cell(row.get(key, "")) for key in columns) + " |" for row in rows]
    (store.path / "e2e.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
