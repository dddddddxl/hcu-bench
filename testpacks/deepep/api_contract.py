"""Describe installed DAS low-latency interfaces without launching HCU kernels."""
import ast
import hashlib
import inspect
import json
import os
import textwrap
from importlib.metadata import version
from pathlib import Path

ENV_KEY = "HCU_BENCH_DEEPEP_CONTRACT"
METHODS = ("__init__", "get_low_latency_rdma_size_hint", "low_latency_dispatch",
           "low_latency_combine", "get_next_low_latency_combine_buffer")
FIELDS = {
    "packed_recv_src_info": "src_info",
    "packed_recv_weights_info": "weight_info",
    "recv_token_unique_counter": "recv_token_unique_counter",
    "recv_token_meta": "recv_token_meta",
    "recv_node_topk_froward_rank_mask": "recv_node_topk_froward_rank_mask",
    "recv_node_tokens": "recv_node_tokens",
    "packed_recv_layout_range": "layout_range",
    "num_max_dispatch_tokens_per_rank": "num_max_dispatch_tokens_per_rank",
    "x.size(1)": "hidden",
    "num_experts": "num_experts",
}
ROUTES = ({"int(route_mode)": "route_mode"},
          {"int(dispatch_mode)": "dispatch_mode", "int(combine_mode)": "combine_mode"})


def digest(source):
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def describe_layout(dispatch_source, combine_source):
    """Match named fields and the receiver's unpacking, not tuple length alone."""
    tree = ast.parse(textwrap.dedent(dispatch_source))
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "handle" for t in n.targets)]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Tuple):
        raise ValueError("Unsupported DeepEP interface: dispatch handle construction is ambiguous")
    function = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if function is None or assignments[0] not in function.body:
        raise ValueError("Unsupported DeepEP interface: conditional handle construction")
    returned = [n for n in ast.walk(tree) if isinstance(n, ast.Return)]
    if not returned or any(not isinstance(n.value, ast.Tuple) or len(n.value.elts) != 5
                           or not isinstance(n.value.elts[2], ast.Name)
                           or n.value.elts[2].id != "handle" for n in returned):
        raise ValueError("Unsupported DeepEP interface: dispatch return contract changed")
    expressions = [ast.unparse(n) for n in assignments[0].value.elts]
    route = next((r for r in ROUTES if set(expressions) == set(FIELDS) | set(r)), None)
    if route is None or len(expressions) != len(set(expressions)):
        raise ValueError("Unsupported DeepEP interface: unknown or duplicate handle fields")
    names = {**FIELDS, **route}
    expected = [names[e] for e in expressions]
    combine = ast.parse(textwrap.dedent(combine_source))
    unpackings = [n.targets[0] for n in ast.walk(combine) if isinstance(n, ast.Assign)
                 and isinstance(n.value, ast.Name) and n.value.id == "handle"
                 and len(n.targets) == 1 and isinstance(n.targets[0], ast.Tuple)]
    if not any([t.id if isinstance(t, ast.Name) else None for t in n.elts] == expected
               for n in unpackings):
        raise ValueError("Unsupported DeepEP interface: combine handle layout does not match dispatch")
    return {"profile": "das_weighted_ll", "handle_size": len(expressions),
            "handle_expressions": expressions,
            "handle_fields": {name: expressions.index(name) for name in FIELDS}}


def probe(buffer_class):
    sources = {name: textwrap.dedent(inspect.getsource(getattr(buffer_class, name)))
               for name in METHODS}
    # Bind the exact calls made by the bundled test, including optional keyword use.
    calls = {
        "__init__": ((None, None), dict(num_rdma_bytes=0, low_latency_mode=True,
            num_qps_per_rank=1, allow_nvlink_for_low_latency_mode=True,
            explicitly_destroy=True, allow_mnnvl=False, enable_combine_overlap=False)),
        "get_low_latency_rdma_size_hint": ((128, 7168, 8, 256), dict(num_topk=8)),
        "low_latency_dispatch": ((None, None, None, None, 128, 256), dict(
            quant_type=0, quant_group_size=0, fp8_round_scale=False,
            async_finish=False, return_recv_hook=False, x_scales=None)),
        "low_latency_combine": ((None, None, None, None, None), dict(
            packed_recv_count=None, comp_signal=None, block_m=64, threshold=10,
            num_sms=3, use_logfmt=False, zero_copy=False, async_finish=False,
            return_recv_hook=False, out=None)),
        "get_next_low_latency_combine_buffer": ((None, None), {}),
    }
    signatures = {}
    for name, (args, kwargs) in calls.items():
        signature = inspect.signature(getattr(buffer_class, name))
        signature.bind(*args, **kwargs)
        signatures[name] = str(signature)
    prefixes = {
        "__init__": ["self", "group"],
        "low_latency_dispatch": ["self", "x", "topk_idx", "topk_weight",
                                 "num_max_dispatch_tokens_per_rank", "num_experts"],
        "low_latency_combine": ["self", "x", "topk_idx", "topk_weights", "handle"],
        "get_low_latency_rdma_size_hint": ["num_max_dispatch_tokens_per_rank", "hidden", "num_ranks", "num_experts"],
        "get_next_low_latency_combine_buffer": ["self", "handle"],
    }
    for name, prefix in prefixes.items():
        if list(inspect.signature(getattr(buffer_class, name)).parameters)[:len(prefix)] != prefix:
            raise ValueError("Unsupported DeepEP interface: positional arguments changed: " + name)
    contract = describe_layout(sources["low_latency_dispatch"], sources["low_latency_combine"])
    return {"schema_version": 1, **contract, "signatures": signatures,
            "source_sha256": {name: digest(source) for name, source in sources.items()}}


def installed_contract():
    import deep_ep

    return {**probe(deep_ep.Buffer), "package_version": version("deep-ep"),
            "package_file": str(Path(deep_ep.__file__).resolve())}


def load_contract(buffer_class):
    path = os.environ.get(ENV_KEY)
    if not path:
        raise ValueError("Run the bundled DeepEP test through bench.sh to generate its API contract")
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    actual = probe(buffer_class)
    if any(contract.get(key) != value for key, value in actual.items()):
        raise ValueError("DeepEP API contract changed between preflight and worker startup")
    return actual


def handle_field(contract, handle, field):
    if not isinstance(handle, tuple) or len(handle) != contract["handle_size"]:
        raise ValueError("Runtime DeepEP handle differs from its probed contract")
    return handle[contract["handle_fields"][field]]
