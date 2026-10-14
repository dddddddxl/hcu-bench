"""Synthetic contract cases, never hardware/performance acceptance evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from hcubench import api


class RcclTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = {"schema_version":1, "test":"communication.rccl_allreduce", "params":{"max_bytes":16384}}
        self.obj, self.resolved = api.resolve(self.config)
        self.plan = api.plan(self.config)
        self.log = """# nThread 1 nGpus 1 minBytes 8192 maxBytes 16384 step: 2(factor) warmup iters: 20 iters: 50 agg iters: 1 validation: 1 graph: 0
rccl-tests: Version skill-bundle:archive-ae3e635
#   Rank 0 Pid 100 on fixture device 0 [0000:e7:00.0] fixture-HCU
#   Rank 1 Pid 101 on fixture device 1 [0000:e8:00.0] fixture-HCU
# out-of-place in-place
# size count type redop root time algbw busbw #wrong time algbw busbw #wrong
# (B) (elements) (us) (GB/s) (GB/s) (us) (GB/s) (GB/s)
8192 4096 half sum -1 10.0 0.82 0.82 0 9.0 0.91 0.91 0
16384 8192 half sum -1 11.0 1.49 1.49 0 10.0 1.64 1.64 0
# Out of bounds values : 0 OK
# Avg bus bandwidth : 1.215
"""
        self.meta = {"schema_version":1, "config_hash":self.plan["config_hash"], "argv":self.plan["argv"], "returncode":0, "started_at":"2026-10-13T00:00:00Z", "finished_at":"2026-10-13T00:00:01Z", "environment":{"hostname":"fixture", "binary_sha256":"a"*64, "rccl_library_sha256":"b"*64, "mpi_version":"fixture", "dtk_version":"fixture", "variables":{"HIP_VISIBLE_DEVICES":"6,7"}}}

    def assess(self, log=None, mutate=None, stderr=""):
        bundle = self.root / "bundle"
        bundle.mkdir(exist_ok=True)
        text = self.log if log is None else log
        (bundle / "stdout.log").write_text(text, encoding="utf-8")
        (bundle / "stderr.log").write_text(stderr, encoding="utf-8")
        meta = copy.deepcopy(self.meta)
        meta["sha256"] = {n:hashlib.sha256((bundle/n).read_bytes()).hexdigest() for n in ("stdout.log", "stderr.log")}
        if mutate:
            mutate(meta)
        (bundle / "execution.json").write_text(json.dumps(meta), encoding="utf-8")
        return self.obj.assess(bundle, self.resolved["params"], self.plan["config_hash"])

    def test_complete_both_modes_and_no_fake_percentiles(self):
        result = self.assess()
        self.assertTrue(result["evidence_valid"], result["issues"])
        self.assertTrue(result["correctness"]["passed"])
        self.assertEqual(len(result["measurement"]["rows"]), 4)
        self.assertEqual(result["measurement"]["sample_kind"], "iteration_aggregate")
        self.assertNotIn("samples_ms", result["measurement"])

    def test_truncated_missing_duplicate_and_unknown_dialect(self):
        for text in (self.log.split("# Out of bounds")[0], self.log.replace("16384 8192 half sum -1 11.0 1.49 1.49 0 10.0 1.64 1.64 0\n", ""), self.log + "8192 4096 half sum -1 10 1 1 0 10 1 1 0\n", self.log.replace("time algbw", "cputime algbw"), self.log.replace("(us)", "(ms)"), self.log.replace("10.0 1.64 1.64 0", "10.0 1.64")):
            with self.subTest(text=text):
                self.assertFalse(self.assess(text)["evidence_valid"])

    def test_errors_in_either_mode_and_nonfinite_values(self):
        for token in ("1", "1e+06", "0*", "N/A"):
            for old in ("0 9.0", "0\n16384"):
                text = self.log.replace(old, old.replace("0", token, 1))
                self.assertFalse(self.assess(text)["evidence_valid"])
        for token in ("nan", "inf", "-1", "0"):
            self.assertFalse(self.assess(self.log.replace("10.0 0.82", token+" 0.82"))["evidence_valid"])

    def test_runtime_parameter_and_device_mismatch(self):
        for before, after in (("validation: 1", "validation: 0"), ("iters: 50", "iters: 51"), ("Rank 1", "Rank 0"), ("0000:e8:00.0", "0000:e7:00.0"), ("on fixture device 1", "on other device 1"), ("4096 half", "4000 half"), ("half sum", "half max")):
            self.assertFalse(self.assess(self.log.replace(before, after))["evidence_valid"])

    def test_exit_code_hash_command_and_environment(self):
        mutations = [lambda m:m.update(returncode=7), lambda m:m.update(returncode=None), lambda m:m.update(returncode=False), lambda m:m.update(config_hash="bad"), lambda m:m.update(argv=["echo", "fake"]), lambda m:m.update(environment={}), lambda m:m["sha256"].update({"stdout.log":"bad"})]
        for mutation in mutations:
            self.assertFalse(self.assess(mutate=mutation)["evidence_valid"])

    def test_scientific_notation(self):
        result = self.assess(self.log.replace("10.0 0.82", "1e1 8.2e-1").replace("0 9.0", "0e0 9.0"))
        self.assertTrue(result["evidence_valid"], result["issues"])

    def test_communicator_split_override_rejected(self):
        result = self.assess(mutate=lambda m:m["environment"]["variables"].update(NCCL_TESTS_SPLIT_MASK="1"))
        self.assertFalse(result["evidence_valid"])

    def test_runtime_launch_violation_even_with_zero_wrong(self):
        result = self.assess(stderr="Launch params (512, 1, 1) are larger than launch bounds (256) for kernel prepareInput2")
        self.assertFalse(result["evidence_valid"])
        self.assertFalse(result["correctness"]["passed"])

    def test_path_independence_and_sampling_identity(self):
        first = api.identities(self.resolved, self.obj)
        self.resolved["params"]["binary"] = "/another/all_reduce_perf"
        self.assertEqual(first, api.identities(self.resolved, self.obj))
        self.resolved["params"]["iters"] += 1
        changed = api.identities(self.resolved, self.obj)
        self.assertEqual(first["case_id"], changed["case_id"])
        self.assertNotEqual(first["config_hash"], changed["config_hash"])

    def test_external_plugin_cannot_be_run_by_core(self):
        with self.assertRaisesRegex(ValueError, "own runner"):
            api.run(self.config, self.root / "results")

    def test_import_failure_preserves_raw_and_report(self):
        self.assess(mutate=lambda m:m.update(returncode=3))
        target, result = api.import_result(self.config, self.root / "bundle", self.root / "results")
        self.assertEqual(result["execution"], "failed")
        self.assertFalse(result["evidence_valid"])
        for name in ("raw/stdout.log", "raw/stderr.log", "raw/execution.json", "measurement.json", "results.csv", "report.html"):
            self.assertTrue((target/name).is_file())

    def test_parameter_validation(self):
        for key,value in (("max_bytes",9000),("ranks",1),("binary","YOUR_PATH"),("operation","max"),("warmup_iters",0)):
            config = copy.deepcopy(self.config)
            config["params"][key] = value
            with self.assertRaises(ValueError):
                api.resolve(config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
