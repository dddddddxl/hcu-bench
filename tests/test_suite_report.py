import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hcu_bench.report import build_report
from hcu_bench.store import RunStore, write_json


class SuiteReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        self.store = RunStore(self.path)
        self.state = dict(run_id="fixture", name="fixture", status="completed", cases={})
        self.samples = []

    def case(self, case_id, suite="rccl", status="completed", **updates):
        self.state["cases"][case_id] = dict(suite=suite, test_id=case_id, status=status, params={}, **updates)

    def sample(self, case_id, name="test", value=1, unit="GB/s", statistic="mean", **updates):
        case = self.state["cases"][case_id]
        result = dict(case_id=case_id, test_id=case["test_id"], suite=case["suite"], node="n0", repeat=0,
                      params=case["params"], measurement_status="measured", correctness="not_checked", simulated=False,
                      metrics=[dict(name=name, value=value, unit=unit, statistic=statistic)])
        result.update(updates)
        self.samples.append(result)
        return result

    def report(self, tail=""):
        write_json(self.path / "state.json", self.state)
        (self.path / "results.jsonl").write_text("".join(json.dumps(s) + "\n" for s in self.samples) + tail, encoding="utf-8")
        result = build_report(self.store)
        self.md = (self.path / "summary.md").read_text(encoding="utf-8")
        self.txt = (self.path / "summary.txt").read_text(encoding="utf-8")
        return result

    def detail(self, index=1):
        with (self.path / "report-details" / f"suite-{index}.csv").open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_empty_and_all_disabled_configuration(self):
        write_json(self.path / "resolved.json", dict(tests=dict(memory=dict(suite="memory", enabled=False))))
        self.report()
        self.assertIn("执行用例总数：0", self.md)
        self.assertIn("配置禁用", self.md)
        self.assertNotIn("memory完整指标", self.md)

    def test_invalid_correctness_failure_and_simulation_never_become_real_perf(self):
        self.case("bad", suite="operators", status="failed")
        self.sample("bad", value=999, correctness="failed", reason="numerical mismatch")
        self.sample("bad", value=888, measurement_status="invalid", reason="ABI mismatch")
        self.case("mock", suite="memory", simulated=True)
        self.sample("mock", value=123, measurement_status="simulated", simulated=True)
        result = self.report()
        self.assertEqual(result["excluded_invalid_or_failed_samples"], 2)
        self.assertNotIn("999", self.md)
        self.assertNotIn("888", self.md)
        self.assertIn("numerical mismatch", self.md)
        self.assertNotIn("具体原因未确认", self.md)
        self.assertIn("[SIMULATED - NOT HARDWARE DATA]", self.txt)
        self.assertEqual(len(self.detail()), 1)  # memory sorts before operators
        self.assertEqual(self.detail()[0]["simulated"], "True")

    def test_failed_case_preserves_partial_measurement_and_exit_reason(self):
        self.case("failed", suite="operators", status="failed", nodes={"n0": dict(returncode=7, cleanup="confirmed")})
        self.sample("failed", value=42)
        self.report()
        self.assertIn("returncode=7", self.md)
        self.assertIn("具体原因未确认", self.md)
        self.assertIn("42", self.md)
        self.assertEqual(self.detail()[0]["case_status"], "failed")

    def test_duration_reached_cleanup_and_archiving_are_separate(self):
        self.case("pressure", suite="deepep", status="duration_reached", nodes={"n0": dict(returncode=-9, cleanup="confirmed", artifact_status="warnings")})
        self.sample("pressure", value=44, correctness="passed")
        self.report()
        self.assertIn("达到计划时长", self.md)
        self.assertIn("完整稳定性", self.md)
        self.assertIn("warnings=1", self.md)
        self.assertEqual(self.detail()[0]["value"], "44")

    def test_rccl_modes_units_sizes_repeats_and_nodes_do_not_merge(self):
        self.case("rccl")
        for i, (size, mode, unit, node) in enumerate([(64, "out_of_place", "GB/s", "n0"),
                                                   (1024, "in_place", "GB/s", "n1")]):
            self.sample("rccl", f"x_{mode}_algbw", i + 2, unit, "reported_mean", node=node, repeat=i,
                        size_bytes=size, mode=mode, collective="all_reduce", params=dict(min_bytes=size))
        self.report()
        rows = self.detail()
        self.assertEqual([r["node"] for r in rows], ["n0", "n1"])
        self.assertEqual([r["repeat"] for r in rows], ["0", "1"])
        self.assertIn("out_of_place", self.md)
        self.assertIn("in_place", self.md)
        self.assertIn("无结果", self.md)  # busbw was not reported, not zero

    def test_pcie_ranges_only_within_identical_configuration_and_simulation(self):
        self.case("pcie", suite="pcie")
        for size, avg, simulated in [(512, 20, False), (512, 30, False), (1024, 40, False), (512, 99, True)]:
            self.sample("pcie", size_bytes=size, kind="cpu_to_gpu", params=dict(size_mib=size), simulated=simulated,
                        metrics=[dict(name="path_avg", value=avg, unit="GB/s", statistic="reported_mean"),
                                 dict(name="path_peak", value=avg + 1, unit="GB/s", statistic="reported_peak")])
        self.report()
        self.assertIn("20.000–30.000", self.md)
        self.assertIn("40.000–40.000", self.md)
        self.assertIn("99.000–99.000", self.md)
        self.assertNotIn("20.000–99.000", self.md)

    def test_session_leaf_missing_never_inherits_completed_measurement(self):
        tasks = [dict(test_id="accuracy", suite="e2e_acc"), dict(test_id="perf", suite="e2e_perf")]
        self.case("session", suite="e2e")
        self.state["cases"]["session"]["params"] = dict(session=dict(model_id="model", tasks=tasks))
        self.sample("session", "gsm8k_accuracy", .5, "ratio", "mean", suite="e2e_acc", test_id="accuracy",
                    model_id="model", e2e_task="accuracy", e2e_suite="e2e_acc", params={}, task_params=dict(num_shots=5))
        result = self.report()
        self.assertEqual(result["case_counts"], {"completed": 1})
        self.assertIn("## e2e_acc", self.md)
        self.assertIn("## e2e_perf", self.md)
        self.assertIn("50.00%", self.md)
        self.assertIn("该子任务无结果", self.md)
        self.assertIn("未获得可展示", self.md)

    def test_llm_and_wan_are_different_tables_with_no_percentile_averaging(self):
        self.case("perf", suite="e2e_perf")
        for concurrency, p99 in [(1, 111), (8, 222)]:
            self.sample("perf", model_id="llm", workload=dict(input_tokens=128, output_tokens=128, concurrency=concurrency, repeat=0),
                        metrics=[dict(name="output_throughput", value=20 * concurrency, unit="tok/s", statistic="mean"),
                                 dict(name="ttft", value=10, unit="ms", statistic="mean"),
                                 dict(name="ttft", value=p99, unit="ms", statistic="p99")])
        self.case("wan", suite="e2e_perf")
        self.sample("wan", "pixel_total_time", 490.37, "s", "value", model_id="wan", workload=dict(profile="tp4", num_frames=81))
        self.report()
        self.assertIn("输出吞吐", self.md)
        self.assertIn("Wan各项耗时", self.md)
        self.assertNotIn("166.5", self.md)
        self.assertEqual([r["value"] for r in self.detail() if r["statistic"] == "p99"], ["111", "222"])

    def test_full_metrics_retained_when_preview_is_truncated(self):
        self.case("conv", suite="operators")
        for index in range(28):
            self.sample("conv", f"shape{index}_latency", index + .25, "ms", "median_of_native_means", correctness="passed")
        self.report()
        self.assertEqual(len(self.detail()), 28)
        self.assertIn("前12条", self.md)
        self.assertNotIn("shape27_latency", self.md)
        self.assertIn("shape11_latency", self.md)

    def test_partial_tail_ignored_and_unknown_metric_preserved(self):
        self.case("custom", suite="memory")
        self.sample("custom", "custom_metric", 3.14159, "widgets", "custom_stat")
        self.report(tail='{"unfinished":')
        self.assertIn("custom_metric", self.md)
        self.assertEqual(self.detail()[0]["statistic"], "custom_stat")

    def test_archived_remote_log_paths_are_not_used_as_local_links(self):
        self.case("case", suite="operators", nodes={"n0": dict(log="/remote/elsewhere.log", cleanup="confirmed")})
        log = self.path / "logs/case/n0.log"
        log.parent.mkdir(parents=True)
        log.touch()
        self.report()
        self.assertIn("](logs/case/n0.log)", self.md)
        self.assertNotIn("/remote/elsewhere", self.md)

    def test_presentation_failure_keeps_structured_metrics_and_fallback(self):
        self.case("case", suite="operators")
        self.sample("case", "latency", 5, "ms")
        with patch("hcu_bench.report.write_suite_report", side_effect=RuntimeError("fixture-render-error")), contextlib.redirect_stderr(io.StringIO()) as error:
            result = self.report()
        self.assertIn("fixture-render-error", error.getvalue())
        self.assertIn("fixture-render-error", self.md)
        self.assertIn("latency", self.txt)
        self.assertEqual(result["metrics"][0]["mean_of_reported_values"], 5)
        self.assertTrue((self.path / "metrics.csv").is_file())

    def test_native_rank_cycle_and_extra_metric_fields_survive_export(self):
        self.case("pressure", suite="deepep")
        sample = self.sample("pressure", native_rank=3, cycle=7, sample_id=9)
        sample["metrics"][0]["native_annotation"] = "extra field is permitted"
        self.report()
        row = self.detail()[0]
        self.assertEqual((row["native_rank"], row["cycle"], row["sample_id"]), ("3", "7", "9"))
        self.assertIn("native_rank=3", self.md)

    def test_known_metric_with_different_unit_is_not_silently_relabelled(self):
        self.case("rccl")
        self.sample("rccl", "x_algbw", 123, "MiB/s", collective="all_reduce")
        self.report()
        self.assertNotIn("| 123 |", self.md)
        self.assertEqual(self.detail()[0]["unit"], "MiB/s")

    def test_each_model_has_preview_even_when_first_model_has_many_points(self):
        for model in ("model1", "model2"):
            self.case(model, suite="e2e_perf")
            for concurrency in range(1, 15):
                self.sample(model, "output_throughput", concurrency, "tok/s", model_id=model,
                            workload=dict(input_tokens=128, output_tokens=128, concurrency=concurrency, repeat=0))
        self.report()
        self.assertIn("### model1", self.md)
        self.assertIn("### model2", self.md)
        self.assertEqual(len(self.detail()), 28)


if __name__ == "__main__":
    unittest.main()
