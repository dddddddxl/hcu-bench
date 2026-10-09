import csv
import json
from collections import Counter

from .store import RunStore, write_json


def build_report(store: RunStore) -> dict:
    state = store.read("state.json")
    groups = {}
    skipped = 0
    samples = store.path / "results.jsonl"
    with (store.path / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as destination:
        fields = ["suite", "test_id", "case_id", "case_status", "node", "repeat", "params", "name", "unit", "statistic", "value", "correctness", "simulated", "observed_at"]
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        if samples.exists():
            with samples.open(encoding="utf-8") as source:
                for line in source:
                    if not line.endswith("\n"):
                        continue
                    sample = json.loads(line)
                    sample["case_status"] = state["cases"][sample["case_id"]]["status"]
                    if sample["measurement_status"] not in ("measured", "simulated") or sample["correctness"] == "failed":
                        skipped += 1
                        continue
                    for metric in sample["metrics"]:
                        params = json.dumps(sample["params"], sort_keys=True)
                        identity = (sample["suite"], sample["test_id"], params, sample["node"], metric["name"], metric["unit"], metric["statistic"], sample["simulated"])
                        group = groups.setdefault(identity, {"suite": identity[0], "test_id": identity[1], "params": sample["params"], "node": identity[3],
                            "name": identity[4], "unit": identity[5], "source_statistic": identity[6], "simulated": identity[7],
                            "case_statuses": [], "count": 0, "mean_of_reported_values": 0, "min": metric["value"], "max": metric["value"]})
                        if sample["case_status"] not in group["case_statuses"]:
                            group["case_statuses"].append(sample["case_status"])
                        group["count"] += 1
                        group["mean_of_reported_values"] += (metric["value"] - group["mean_of_reported_values"]) / group["count"]
                        group["min"] = min(group["min"], metric["value"])
                        group["max"] = max(group["max"], metric["value"])
                        writer.writerow({**{key: sample.get(key, "") for key in fields if key not in ("params", "name", "unit", "statistic", "value")},
                                         "params": params, **{key: metric[key] for key in ("name", "unit", "statistic", "value")}})
    counts = dict(Counter(case["status"] for case in state["cases"].values()))
    summary = {"run_id": state["run_id"], "status": state["status"], "case_counts": counts,
               "excluded_invalid_or_failed_samples": skipped, "metrics": list(groups.values())}
    write_json(store.path / "summary.json", summary)
    lines = [f"Run: {state['run_id']}", f"Status: {state['status']}", f"Cases: {counts}",
             "Values below are grouped by test, parameters, node, unit and source statistic."]
    for group in groups.values():
        prefix = "[SIMULATED - NOT HARDWARE DATA] " if group["simulated"] else ""
        lines.append(f"{prefix}{group['test_id']}/{group['node']} {group['name']}: reported-value mean={group['mean_of_reported_values']:.6g} {group['unit']}, range={group['min']:.6g}..{group['max']:.6g}, n={group['count']}, source={group['source_statistic']}, case_status={group['case_statuses']}, params={group['params']}")
    (store.path / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
