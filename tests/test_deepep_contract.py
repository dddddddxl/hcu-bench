import json
import contextlib
import io
import os
import tempfile
import unittest
import threading
from pathlib import Path
from unittest.mock import patch

from testpacks.deepep import api_contract as api
from testpacks.deepep import run


class BufferFixture:
    def __init__(self, group, **kwargs):
        pass

    @staticmethod
    def get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts, num_topk=8):
        pass

    def low_latency_dispatch(self, x, topk_idx, topk_weight, num_max_dispatch_tokens_per_rank,
                             num_experts, **kwargs):
        handle = (packed_recv_src_info, packed_recv_weights_info, recv_token_unique_counter,
                  recv_token_meta, recv_node_topk_froward_rank_mask, recv_node_tokens,
                  packed_recv_layout_range, num_max_dispatch_tokens_per_rank, x.size(1),
                  num_experts, int(dispatch_mode), int(combine_mode))
        return None, None, handle, None, None

    def low_latency_combine(self, x, topk_idx, topk_weights, handle, **kwargs):
        (src_info, weight_info, recv_token_unique_counter, recv_token_meta,
         recv_node_topk_froward_rank_mask, recv_node_tokens, layout_range,
         num_max_dispatch_tokens_per_rank, hidden, num_experts, dispatch_mode, combine_mode) = handle

    def get_next_low_latency_combine_buffer(self, handle):
        pass


class ContractTests(unittest.TestCase):
    def sources(self):
        import inspect
        return (inspect.getsource(BufferFixture.low_latency_dispatch),
                inspect.getsource(BufferFixture.low_latency_combine))

    def test_new_and_old_route_contracts(self):
        dispatch, combine = self.sources()
        new = api.describe_layout(dispatch, combine)
        self.assertEqual(new["handle_size"], 12)
        self.assertEqual(new["handle_fields"]["packed_recv_layout_range"], 6)
        old = api.describe_layout(dispatch.replace("int(dispatch_mode), int(combine_mode)", "int(route_mode)"),
                                  combine.replace("dispatch_mode, combine_mode", "route_mode"))
        self.assertEqual(old["handle_size"], 11)

    def test_reordered_named_fields_are_discovered_not_guessed(self):
        dispatch, combine = self.sources()
        dispatch = dispatch.replace("packed_recv_src_info, packed_recv_weights_info",
                                    "packed_recv_weights_info, packed_recv_src_info")
        combine = combine.replace("src_info, weight_info", "weight_info, src_info")
        contract = api.describe_layout(dispatch, combine)
        self.assertEqual(contract["handle_fields"]["packed_recv_src_info"], 1)
        self.assertEqual(api.handle_field(contract, tuple(range(12)), "packed_recv_src_info"), 1)

    def test_same_length_with_changed_semantics_is_rejected(self):
        dispatch, combine = self.sources()
        with self.assertRaisesRegex(ValueError, "unknown"):
            api.describe_layout(dispatch.replace("packed_recv_layout_range", "new_metadata"), combine)
        with self.assertRaisesRegex(ValueError, "combine"):
            api.describe_layout(dispatch, combine.replace("src_info, weight_info", "weight_info, src_info"))

    def test_ambiguous_or_duplicate_layout_rejected(self):
        dispatch, combine = self.sources()
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            api.describe_layout(dispatch + "\n    handle = ()\n", combine)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            api.describe_layout(dispatch.replace("int(dispatch_mode)", "int(combine_mode)"), combine)

    def test_runtime_length_and_changed_contract_rejected(self):
        contract = api.probe(BufferFixture)
        with self.assertRaisesRegex(ValueError, "Runtime"):
            api.handle_field(contract, tuple(range(11)), "packed_recv_src_info")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "api_contract.json"
            path.write_text(json.dumps(contract), encoding="utf-8")
            with patch.dict(os.environ, {api.ENV_KEY: str(path)}):
                self.assertEqual(api.load_contract(BufferFixture), contract)
                contract["handle_fields"]["packed_recv_src_info"] = 9
                path.write_text(json.dumps(contract), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "changed"):
                    api.load_contract(BufferFixture)

    def test_unknown_required_argument_rejected(self):
        def changed(self, x, topk_idx, topk_weight, num_max_dispatch_tokens_per_rank,
                    num_experts, required_new_option, **kwargs):
            pass
        with patch.object(BufferFixture, "low_latency_dispatch", changed):
            with self.assertRaises(TypeError):
                api.probe(BufferFixture)

    def test_smoke_has_no_pressure_flag(self):
        args = run.parser().parse_args(["--test-script", "native.py", "--output-dir", "out",
            "--node-rank", "0", "--nnodes", "1", "--master-addr", "127.0.0.1", "--num-processes", "8"])
        self.assertNotIn("--pressure-test", run.command_for(args, 0, pressure=False))
        self.assertIn("--pressure-test", run.command_for(args, 1))

    def test_failed_smoke_never_starts_pressure(self):
        with tempfile.TemporaryDirectory() as directory:
            args = run.parser().parse_args(["--test-script", "auto", "--output-dir", directory,
                "--node-rank", "0", "--nnodes", "1", "--master-addr", "127.0.0.1", "--num-processes", "8"])
            with patch.object(run, "preflight", return_value={}), \
                    patch.object(run, "installed_contract", return_value={"handle_size": 12}), \
                    patch.object(run, "cancellation", return_value=threading.Event()), \
                    patch.object(run, "NativeProcess") as process, contextlib.redirect_stdout(io.StringIO()):
                process.return_value.wait.return_value = 1
                self.assertEqual(run.run(args), 1)
                self.assertEqual(process.call_count, 1)
                self.assertNotIn("--pressure-test", process.call_args.args[0])
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertEqual(summary["smoke_status"], "failed")
            self.assertEqual(summary["valid_samples"], 0)


if __name__ == "__main__":
    unittest.main()
