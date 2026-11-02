"""Validate collective-specific native row counts without weakening AllReduce."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from testpacks.rccl import run as rccl
from testpacks import prepare
from hcu_bench.planner import make_plan
from test_rccl_native import LOG

ROOT = Path(__file__).resolve().parents[1]


class CollectiveTests(unittest.TestCase):
    def params(self, collective):
        p = {k: v["default"] for k, v in rccl.RcclAllReduce.schema.items()}
        p.update(binary=f"/work/{collective}_perf", ranks=2, min_bytes=8192, max_bytes=16384)
        return p

    def test_all_collectives_have_matching_build_and_local_launch(self):
        for collective in prepare.RCCL_COLLECTIVES:
            p = self.params(collective)
            plugin = rccl.LocalCollective(collective)
            plugin.validate(p)
            argv = plugin.plan(p)["argv"]
            self.assertEqual(argv[1:3], ["--host", "localhost:2"])
            commands, target = prepare.commands_for("rccl", Path("/src"), Path("/build"), dict(
                dtk_root="/opt/dtk", mpi_home="/opt/mpi", compiler="hipcc", arch="gfx936", jobs=2, rccl_collective=collective))
            self.assertEqual(target.name, collective + "_perf")
            self.assertIn(str(target), commands[0])

    def assess(self, collective, corrupt=False):
        p = self.params(collective)
        plugin = rccl.LocalCollective(collective)
        factor = p["ranks"] if collective in ("all_gather", "reduce_scatter", "alltoall", "alltoallv", "gather", "scatter") else 1
        op = "sum" if collective in ("all_reduce", "reduce_scatter", "reduce", "sendrecv") else "" if collective == "alltoallv" else "none"
        root = "0" if collective in ("broadcast", "reduce", "gather", "scatter") else "-1"
        text = LOG
        for size in (8192, 16384):
            text = text.replace(f"{size} {size // 2} half sum -1", f"{size} {size // 2 // factor} half {op} {root}")
        if collective in ("alltoall", "alltoallv", "sendrecv"):
            text = "\n".join(line.rsplit(" ", 1)[0] + " N/A" if line[:1].isdigit() else line
                             for line in text.splitlines()) + "\n"
        if corrupt:
            text = text.replace("# Out of bounds values : 0 OK", "# Out of bounds values : 1 FAILED")
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / "stdout.log").write_text(text)
            (out / "stderr.log").write_text("")
            meta = dict(schema_version=1, config_hash="fixture", argv=plugin.plan(p)["argv"], returncode=0,
                        started_at="start", finished_at="finish", environment=dict(hostname="fixture",
                        binary_sha256="a" * 64, rccl_library_sha256="b" * 64, mpi_version="MPI", dtk_version="DTK",
                        variables={"HIP_VISIBLE_DEVICES": "0,1"}), sha256={name:hashlib.sha256((out/name).read_bytes()).hexdigest()
                                                                         for name in ("stdout.log", "stderr.log")})
            (out / "execution.json").write_text(json.dumps(meta))
            return plugin.assess(out, p, "fixture")

    def test_all_native_layouts_require_complete_rows_and_checks(self):
        for collective in prepare.RCCL_COLLECTIVES:
            with self.subTest(collective=collective):
                good = self.assess(collective)
                self.assertTrue(good["evidence_valid"], good["issues"])
                self.assertTrue(good["correctness"]["passed"])
                if collective in ("alltoall", "alltoallv", "sendrecv"):
                    self.assertEqual({r["mode"] for r in good["measurement"]["rows"]}, {"out_of_place"})
                self.assertFalse(self.assess(collective, True)["correctness"]["passed"])

    def test_wrong_element_counts_or_root_are_rejected(self):
        p = self.params("all_gather")
        plugin = rccl.LocalCollective("all_gather")
        self.assertFalse(plugin.row_semantics(p, ["8192", "4096", "half", "none", "-1"], 8192, 4096, 2))
        with self.assertRaises(ValueError):
            rccl.LocalCollective("broadcast", 2).validate(p)
        p["min_bytes"] = p["max_bytes"] = 10
        with self.assertRaises(ValueError):
            plugin.validate(p)

    def test_public_template_contains_ten_collectives(self):
        plan = make_plan(str(ROOT / "configs/rccl.template.yaml"), "rccl")
        self.assertEqual({c.params["collective"] for c in plan.cases}, set(prepare.RCCL_COLLECTIVES))
        self.assertEqual(len(plan.cases), 10)
        for case in plan.cases:
            self.assertEqual((case.params["ranks"], case.params["min_bytes"], case.params["max_bytes"]),
                             (8, 17179869184, 17179869184))
