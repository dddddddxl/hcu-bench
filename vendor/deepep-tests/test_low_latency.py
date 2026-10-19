import argparse
import random
import torch
import torch.distributed as dist
from functools import partial
from typing import Literal, Set
from typing import Literal, Set, Optional
import deep_ep
from utils import init_dist, bench, bench_kineto, calc_diff, hash_tensor, per_token_cast_pg_back, per_token_cast_pc_back, deepep_per_token_quantize, per_token_cast_back_generic


def simulate_failure_and_skip(rank: int, api: Literal["dispatch", "combine", "clean"], expected_masked_ranks: Set[int]):
    # Simulates rank failure when the rank first calls the corresponding communication API
    failed_api_ranks = {
        # API -> rank to fail (rank fails when it first calls the corresponding communication API)
        'dispatch': 1,
        'combine': 3,
        'clean': 5
    }
    if rank in expected_masked_ranks:
        # Rank already failed
        return True
    if api in failed_api_ranks.keys():
        expected_masked_ranks.add(failed_api_ranks[api])
        if failed_api_ranks[api] == rank:
            print(f"Rank {rank} failed when first calling {api} communication API, exit...", flush=True)
            return True
    return False


def query_mask_buffer_and_check(api: Literal["dispatch", "combine", "clean"], buffer: deep_ep.Buffer, mask_status: torch.Tensor,
                                expected_masked_ranks: Set[int]):
    buffer.low_latency_query_mask_buffer(mask_status)
    assert set(mask_status.nonzero().squeeze(-1).tolist()) == expected_masked_ranks


def ceil_div(a, b):
    return (a + b - 1) // b


def is_buffer_mode_enabled(method_name: str, num_ranks: int) -> bool:
    """Treat modes unsupported by an older DeepEP wheel as disabled."""
    checker = getattr(deep_ep.Buffer, method_name, None)
    return bool(checker(num_ranks)) if checker is not None else False

def test_main_moriep_layered(num_tokens: int,
              hidden: int,
              num_experts: int,
              num_topk: int,
              rank: int,
              num_ranks: int,
              group: dist.ProcessGroup,
              buffer: deep_ep.Buffer,
              quant_type: int,
              group_size: int,
              seed: int = 0):

    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    rank_offset = 128
    assert num_ranks - rank_offset < 257, 'Too many ranks (exceeding test precision limit)'

    # 构造测试数据
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * (rank - rank_offset)
    x[:, -128:] = torch.arange(num_tokens, device='cuda').to(torch.bfloat16).view(-1, 1)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='cuda').abs()

    # 动态执行量化
    x_quant, x_scales = deepep_per_token_quantize(x, quant_type, group_size)
    x_quant = x_quant.contiguous()

    if rank == 0:
        print(f"\n--- [Testing Mode: {quant_type} | Group Size: {group_size}] ---", flush=True)
        print("x_quant shape & dtype: ", x_quant.size(), x_quant.dtype, flush=True)
        if x_scales is not None:
            print("x_scales shape & dtype: ", x_scales.size(), x_scales.dtype, flush=True)
        else:
            print("x_scales is None (BF16 Mode)", flush=True)

    packed_recv_x, packed_recv_count, handle, event, hook = \
        buffer.low_latency_dispatch(
            x=x_quant,
            topk_idx=topk_idx,
            topk_weight=topk_weights,
            num_max_dispatch_tokens_per_rank=num_tokens,
            num_experts=num_experts,
            quant_type=quant_type, quant_group_size=0,
            async_finish=False,
            return_recv_hook=False,
            x_scales=x_scales # 量化因子，BF16下传入 None
        )

    # 聚合全局路由拓扑以供校验
    all_topk_idx = torch.empty((num_ranks, num_tokens, num_topk), dtype=topk_idx.dtype, device='cuda')
    dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=group)

    # 记录本地专家 验证失败的次数
    local_errors = 0

    if isinstance(packed_recv_x, tuple):
        # =================================== 采用 主数据对比的方式确保分发的 Token 数据 及 Scale 数据 正确
        # 收集 全Rank 量化后的 token 数据
        all_token_quant = torch.empty((num_ranks, num_tokens, hidden), dtype=x_quant.dtype, device='cuda')
        dist.all_gather_into_tensor(all_token_quant, x_quant, group=group)

        # 收集 全Rank 量化后的 scale 数据
        all_token_scale = torch.empty((num_ranks, num_tokens, x_scales.size()[1]), dtype=x_scales.dtype, device='cuda')
        dist.all_gather_into_tensor(all_token_scale, x_scales, group=group)

        # 遍历验证 本地专家
        for i in range(num_local_experts):
            expert_id = rank * num_local_experts + i   # 获取 本地专家的 全局专家索引
            
            # 获取 当前本地专家 Dispatch 接收到的 token 数据 及 Scale 数据
            recv_x_quant = packed_recv_x[0][i].contiguous()
            recv_x_scale = packed_recv_x[1][i].contiguous()
            # 获取 当前本地专家 Dispatch 接收到的 token 总数 及 每个token 的 源索引
            recv_count, recv_src_info = packed_recv_count[i], handle[0][i]

            # 判断 当前本地专家 接收 的Token 总数 是否和 全局 topk_ids 对的上
            num_valid_tokens = recv_count.item()
            expected_tokens = (all_topk_idx == expert_id).sum().item()
            if num_valid_tokens != expected_tokens:
                print(f"❌ [Rank {rank}] token count mismatch! expert_id={expert_id} "
                    f"recv={num_valid_tokens} expect={expected_tokens}", flush=True)
            assert num_valid_tokens == expected_tokens, f'{num_valid_tokens} != {expected_tokens}'

            # 当前本地专家 不接收token 则不需要验证 token 和 scale 数据是否正确
            if num_valid_tokens == 0:
                continue

            # 截取有效区间
            recv_x_quant = recv_x_quant[:num_valid_tokens]
            recv_x_scale = recv_x_scale[:num_valid_tokens]
            recv_src_info = recv_src_info[:num_valid_tokens]  

            # 获取 接收 Token 对应的 rank 索引 和 rank 上的 token 索引， 用于获取 当前token 来自 哪个 Rank 上的哪个专家
            src_info_long = recv_src_info.to(torch.int64)
            recv_src_rank = torch.div(src_info_long, num_tokens, rounding_mode='floor')
            recv_src_token_idx = src_info_long % num_tokens

            # 根据  rank 索引 和 rank 上的 token 索引 获取 期望的 token 数据和 scale 数据
            expect_x_quant = all_token_quant[recv_src_rank, recv_src_token_idx, :].contiguous()
            expect_x_scale = all_token_scale[recv_src_rank, recv_src_token_idx, :].contiguous()
            # 确保接收到的数据 是 连续状态
            recv_x_quant = recv_x_quant.contiguous()
            recv_x_scale = recv_x_scale.contiguous()

            # 校验 1：量化特征矩阵比对
            if recv_x_quant.shape != expect_x_quant.shape:
                print(f"❌ [Rank {rank}] Shape mismatch on expert {expert_id}! Recv {recv_x_quant.shape}, Expect {expect_x_quant.shape}", flush=True)
                local_errors += 1
            elif not torch.equal(recv_x_quant, expect_x_quant):
                # 如果形状一样但内容还错，计算最大绝对差值
                max_diff = (recv_x_quant.float() - expect_x_quant.float()).abs().max().item()
                print(f"❌ [Rank {rank}] Feature data mismatch on expert {expert_id}! Max diff: {max_diff}", flush=True)
                local_errors += 1

            # 校验 2：Scale 缩放因子比对
            if recv_x_scale.shape != expect_x_scale.shape:
                print(f"❌ [Rank {rank}] Scale shape mismatch on expert {expert_id}! Recv {recv_x_scale.shape}, Expect {expect_x_scale.shape}", flush=True)
                local_errors += 1
            elif not torch.equal(recv_x_scale, expect_x_scale):
                max_diff = (recv_x_scale - expect_x_scale).abs().max().item()
                print(f"❌ [Rank {rank}] Scale data mismatch on expert {expert_id}! Max diff: {max_diff}", flush=True)
                local_errors += 1

    else:
        # =================================== 采用 主数据对比的方式确保分发的 Token 数据 及 Scale 数据 正确
        # 收集 全Rank 全量的  token 数据
        all_token_bf16 = torch.empty((num_ranks, num_tokens, hidden), dtype=x_quant.dtype, device='cuda')
        dist.all_gather_into_tensor(all_token_bf16, x_quant, group=group)

        for i in range(num_local_experts):
            expert_id = rank * num_local_experts + i

            # 获取 当前本地专家 Dispatch 接收到的 token 数据
            recv_x_bf16 = packed_recv_x[i].contiguous()
            # 获取 当前本地专家 Dispatch 接收到的 token 总数 及 每个token 的 源索引
            recv_count, recv_src_info = packed_recv_count[i], handle[0][i]

            # 判断 当前本地专家 接收 的Token 总数 是否和 全局 topk_ids 对的上
            num_valid_tokens = recv_count.item()
            expected_tokens = (all_topk_idx == expert_id).sum().item()
            if num_valid_tokens != expected_tokens:
                print(f"❌ [Rank {rank}] token count mismatch! expert_id={expert_id} "
                    f"recv={num_valid_tokens} expect={expected_tokens}", flush=True)
            assert num_valid_tokens == expected_tokens, f'{num_valid_tokens} != {expected_tokens}'

            # 当前本地专家 不接收token 则不需要验证 token 和 scale 数据是否正确
            if num_valid_tokens == 0:
                continue

            # 截取有效区间
            recv_x_bf16 = recv_x_bf16[:num_valid_tokens]
            recv_src_info = recv_src_info[:num_valid_tokens]

            # 获取 接收 Token 对应的 rank 索引 和 rank 上的 token 索引， 用于获取 当前token 来自 哪个 Rank 上的哪个专家
            src_info_long = recv_src_info.to(torch.int64)
            recv_src_rank = torch.div(src_info_long, num_tokens, rounding_mode='floor')
            recv_src_token_idx = src_info_long % num_tokens

            # 根据  rank 索引 和 rank 上的 token 索引 获取 期望的 token 数据和 scale 数据
            expect_x_bf16 = all_token_bf16[recv_src_rank, recv_src_token_idx, :].contiguous()
            # 确保接收到的数据 是 连续状态
            recv_x_bf16 = recv_x_bf16.contiguous()

            # 校验 1：全量 比对
            if recv_x_bf16.shape != expect_x_bf16.shape:
                print(f"❌ [Rank {rank}] Shape mismatch on expert {expert_id}! Recv {recv_x_quant.shape}, Expect {expect_x_quant.shape}", flush=True)
                local_errors += 1
            elif not torch.equal(recv_x_bf16, expect_x_bf16):
                # 如果形状一样但内容还错，计算最大绝对差值
                max_diff = (recv_x_bf16.float() - expect_x_bf16.float()).abs().max().item()
                print(f"❌ [Rank {rank}] Feature data mismatch on expert {expert_id}! Max diff: {max_diff}", flush=True)
                local_errors += 1

    # 聚合全网错误计数，避免单个 Rank 挂掉导致整体全局断言静默 Hang 死
    total_errors = torch.tensor([local_errors], dtype=torch.int32, device='cuda')
    dist.all_reduce(total_errors, op=dist.ReduceOp.SUM, group=group)

    if total_errors.item() == 0:
        if rank == 0:
            print("✅ 全网 Dispatch 正确性验证成功！！！", flush=True)
    else:
        if rank == 0:
            print(f"❌ 全网验证失败，累计发现错误次数: {total_errors.item()}", flush=True)
        sys.exit(1) # 发现错误主动退出，不带病运行后续代码

    torch.cuda.synchronize()
    dist.barrier()

    print(f"✅ {quant_type} Dispatch 正确性验证成功！！！", flush=True)
    # ==========================================
    # 执行 combine 阶段
    # ==========================================
    if isinstance(packed_recv_x, tuple):
        flat_recv_x = packed_recv_x[0].view(-1, hidden)
        flat_recv_scales = packed_recv_x[1].view(-1, x_scales.size(-1) if x_scales is not None else 1) if packed_recv_x[1] is not None else None
    else:
        flat_recv_x = packed_recv_x.view(-1, hidden)
        flat_recv_scales = None

    simulated_gemm_x = per_token_cast_back_generic(
        flat_recv_x,
        flat_recv_scales,
        quant_type,
        group_size
    ).view(packed_recv_x[0].shape if isinstance(packed_recv_x, tuple) else packed_recv_x.shape)

    combined_x, event, hook = buffer.low_latency_combine(
                                                        simulated_gemm_x,
                                                        topk_idx,
                                                        topk_weights,
                                                        handle
                                                        )
    torch.cuda.synchronize()

    diff = calc_diff(x * topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1), combined_x)
    assert torch.isnan(combined_x).sum().item() == 0

    # 动态设定精度误差阈值
    if quant_type == 2:
        threshold = 1e-2
    elif quant_type == 4:
        threshold = 3e-2
    elif quant_type == 1:
        threshold = 9e-4
    else:
        threshold = 1e-5

    assert diff < threshold, f'Error: rank={rank} diff={diff} exceeds threshold {threshold}'

    # 让每个 rank 都输出自己的 diff 信息
    print(f"[Rank {rank}] ✅ {quant_type} Combine 验证成功! 局部 diff: {diff:.6f}", flush=True)

    # 带宽测试
    def test_func(zero_copy: bool, use_fp8: bool, return_recv_hook: bool):
        recv_x, recv_count, handle, event, hook = \
                        buffer.low_latency_dispatch(
                            x=x_quant,
                            topk_idx=topk_idx,
                            topk_weight=topk_weights,
                            num_max_dispatch_tokens_per_rank=num_tokens,
                            num_experts=num_experts,
                            quant_type=quant_type, quant_group_size=0,
                            async_finish=False,
                            return_recv_hook=False,
                            x_scales=x_scales  # 量化因子，BF16下传入 None
                        )

        combined_x, event, hook = buffer.low_latency_combine(
                                                            simulated_gemm_x,
                                                            topk_idx,
                                                            topk_weights,
                                                            handle
                                                            )
    # FP8 quantization is not yet supported.
    bench_use_fp8 = False
    if quant_type != 0:
        bench_use_fp8 = True

    # Calculate bandwidth
    num_fp8_bytes, num_bf16_bytes = (hidden + hidden / 128 * 4 + 16), hidden * 2
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += (num_fp8_bytes if bench_use_fp8 else num_bf16_bytes) * num_selections
        num_combine_comm_bytes += num_bf16_bytes * num_selections
    print(f'[rank {rank}] num_dispatch_comm_bytes {num_dispatch_comm_bytes} num_combine_comm_bytes {num_combine_comm_bytes}')

    # Dispatch + combine testing
    avg_t, min_t, max_t = bench(partial(test_func, zero_copy=False, use_fp8=bench_use_fp8, return_recv_hook=False))
    print(
        f'[rank {rank}] Dispatch + combine bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / avg_t:.2f} GB/s, '
        f'avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us',
        flush=True)

    # Separate profiling
    group.barrier()

    kernel_dispatch = ["EpDispatchCopyToStaging", "EpDispatchInterNodeV1KernelLowLatency", "EpDispatchConvertDispatchOutput"]
    kernel_combine =  ["CombineConvertCombineInput", "CombineSyncBarrier", "EpCombineInterNodeV1KernelLowLatency", "CombineAll"]

    kernel_names = kernel_dispatch + kernel_combine


    timings = list(
        bench_kineto(
            partial(test_func, zero_copy=False, use_fp8=False, return_recv_hook=False),
            kernel_names=tuple(kernel_names),
            barrier_comm_profiling=True,
            suppress_kineto_output=True
        ))
    dispatch_copy = timings[0]
    dispatch_rdma_ipc = timings[1]
    dispatch_covert = timings[2]

    dispatch_total = dispatch_copy + dispatch_rdma_ipc + dispatch_covert

    combine_convert = timings[3]
    combine_barry = timings[4]
    combine_rdma_ipc = timings[5]
    combine_last_reduce = timings[6]

    combine_total = combine_convert + combine_barry + combine_rdma_ipc + combine_last_reduce

    print(
        f'[rank {rank}] Dispatch bandwidth (total): {num_dispatch_comm_bytes / 1e9 / dispatch_total:.2f} GB/s, '
        f'avg_t={dispatch_total * 1e6:.2f} us | '
        f'Combine bandwidth (total): {num_combine_comm_bytes / 1e9 / combine_total:.2f} GB/s, '
        f'avg_t={combine_total * 1e6:.2f} us',
        flush=True)

def test_main(num_tokens: int,
              hidden: int,
              num_experts: int,
              num_topk: int,
              rank: int,
              num_ranks: int,
              group: dist.ProcessGroup,
              buffer: deep_ep.Buffer,
              enable_combine_overlap: bool = False,
              use_logfmt: bool = False,
              seed: int = 0):
    enable_ll_layered_opt = is_buffer_mode_enabled('is_ll_layered_opt_enabled', num_ranks)
    enable_ll_layered = is_buffer_mode_enabled('is_ll_layered_enabled', num_ranks)
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)
    if rank == 0:
        print(f"enable_ll_layered_opt={enable_ll_layered_opt}, enable_ll_layered={enable_ll_layered}, enable_combine_overlap={enable_combine_overlap}, use_logfmt={use_logfmt}")

    assert not (use_logfmt and enable_combine_overlap), \
        "use_logfmt=True and enable_combine_overlap conflict"
    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    # NOTES: the integers greater than 256 exceed the BF16 precision limit
    rank_offset = 128
    assert num_ranks - rank_offset < 257, 'Too many ranks (exceeding test precision limit)'

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * (rank - rank_offset)
    x[:, -128:] = torch.arange(num_tokens, device='cuda').to(torch.bfloat16).view(-1, 1)
    x_list = [x]
    for _ in range(4 if use_logfmt else 0):
        # NOTES: make more LogFMT casts and also with some BF16
        x_list.append(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * 0.5 * random.random())
    # NOTES: the last one is for performance testing
    # Most of the values in the perf case is lower than the threshold, casting most channels
    x_list.append(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * 0.1)

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda').abs()

    # Randomly mask some positions
    for _ in range(10):
        topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1

    all_topk_idx = torch.empty((num_ranks, num_tokens, num_topk), dtype=topk_idx.dtype, device='cuda')
    dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=group)

    # For failure simulation and shrink testing
    mask_status = torch.zeros((num_ranks,), dtype=torch.int, device='cuda')
    expected_masked_ranks = set()

    # Check dispatch correctness
    do_check = True
    hash_value, num_times = 0, 0
    for x_i, current_x in enumerate(x_list):
        for return_recv_hook in (False, True):
            if enable_combine_overlap and (not return_recv_hook):  # return_recv_hook 为False 时，不能启用 overlop
                continue

            for quant_type in (0, 1, 2, 3,): # 0: 不量化, 1: int8, 2: FP8_E4M3, 3: FP8_UE8M0 (仅支持round_scale=True), 4: FP8_E5M2
                dispatch_use_quant = quant_type > 0
                for fp8_round_scale in (False, True) if (quant_type != 3 and not enable_ll_layered_opt) else (True,):
                    for quant_group_size in (0, 128,):
                        if quant_type == 3 and (fp8_round_scale == False or quant_group_size == 0):
                            continue

                        num_times += 1
                        for _ in range((num_times % 2) + 1):
                            packed_recv_x, packed_recv_count, handle, event, hook = \
                                buffer.low_latency_dispatch(current_x, topk_idx, topk_weights, num_tokens, num_experts,
                                                            quant_type=quant_type, fp8_round_scale=fp8_round_scale, quant_group_size=quant_group_size,
                                                            async_finish=not return_recv_hook, return_recv_hook=return_recv_hook)
                            hook() if return_recv_hook else event.current_stream_wait()
                        packed_recv_x = (packed_recv_x[0], packed_recv_x[1].contiguous()) if dispatch_use_quant else packed_recv_x
                        if not dispatch_use_quant:
                            simulated_gemm_x = packed_recv_x.clone()
                        elif quant_group_size == 0:
                            simulated_gemm_x = per_token_cast_pc_back(packed_recv_x[0].view(-1, hidden), packed_recv_x[1].reshape(-1)).view(packed_recv_x[0].shape)
                        elif quant_group_size == 128:
                            simulated_gemm_x = per_token_cast_pg_back(packed_recv_x[0].view(-1, hidden), packed_recv_x[1].view(-1, hidden // 128)).view(packed_recv_x[0].shape)
                        for i in range(num_local_experts if do_check else 0):
                            expert_id = rank * num_local_experts + i
                            if not dispatch_use_quant:
                                recv_x = packed_recv_x[i]
                            elif quant_group_size == 0:
                                recv_x = per_token_cast_pc_back(packed_recv_x[0][i], packed_recv_x[1][i])
                            elif quant_group_size == 128:
                                recv_x = per_token_cast_pg_back(packed_recv_x[0][i], packed_recv_x[1][i])
                            
                            # Verified against this daily wheel's 11-field dispatch handle.
                            assert len(handle) == 11, f'Unexpected dispatch handle length: {len(handle)}'
                            recv_count, recv_src_info, recv_layout_range = packed_recv_count[i], handle[0][i], handle[6][i]

                            # Check expert indices
                            int_mask = (2 ** 32) - 1
                            num_valid_tokens = recv_count.item()
                            assert num_valid_tokens == (
                                    recv_layout_range
                                    & int_mask).sum().item(), f'{num_valid_tokens} != {recv_layout_range & int_mask}.sum().item()'
                            assert num_valid_tokens == (all_topk_idx == expert_id).sum(dim=[1, 2])[mask_status == 0].sum().item(
                            ), f'{num_valid_tokens} != {(all_topk_idx == expert_id).sum(dim=[1, 2])[mask_status == 0].sum().item()}'

                            if num_valid_tokens == 0:
                                continue
                            # Check received data
                            if current_x is x:
                                recv_x = recv_x[:num_valid_tokens]
                                recv_x_amin = recv_x[:, :-128].amin(dim=-1)
                                recv_x_amax = recv_x[:, :-128].amax(dim=-1)

                                if enable_combine_overlap:
                                    recv_src_info = recv_src_info[:num_valid_tokens] & int_mask  # 掩掉多余的信息
                                else:
                                    recv_src_info = recv_src_info[:num_valid_tokens]

                                assert torch.equal(recv_x_amin, recv_x_amax)

                                if dispatch_use_quant and not enable_ll_layered_opt:
                                    assert calc_diff(recv_x[:, -1], recv_src_info.view(-1)) < 0.007

                                assert torch.equal(recv_x_amin, recv_x[:, :-128].amax(dim=-1))
                                if quant_group_size != 0:
                                    if not enable_ll_layered_opt:
                                        if fp8_round_scale:
                                            assert calc_diff(recv_x[:, -1], recv_src_info.view(-1)) < 0.007
                                        else:
                                            assert (recv_x[:, -128:] - recv_src_info.view(-1, 1) % num_tokens).sum().item() == 0
                                    for j in range(num_ranks):
                                        begin_idx, count = (recv_layout_range[j] >> 32).item(), (recv_layout_range[j] & int_mask).item()
                                        if not fp8_round_scale:
                                            assert (recv_x_amin == j - rank_offset).sum().item() == (all_topk_idx[j] == expert_id).sum().item()
                                            assert (recv_x[begin_idx:begin_idx + count, :-128] - j + rank_offset).sum().item() == 0

                            if dispatch_use_quant:
                                hash_value ^= hash_tensor(packed_recv_x[0][i, :num_valid_tokens])
                                hash_value ^= hash_tensor(packed_recv_x[1][i, :num_valid_tokens])
                            else:
                                hash_value ^= hash_tensor(packed_recv_x[i, :num_valid_tokens])

                        # Check combine correctness
                        for zero_copy in (False,) if (use_logfmt or enable_ll_layered_opt) else (False, True,):
                            if zero_copy:
                                buffer.get_next_low_latency_combine_buffer(handle)[:, :, :] = simulated_gemm_x
                            out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
                            if enable_combine_overlap:
                                block_m, threshold, num_sms = 64, 10, 3
                                total_num_per_expert = ceil_div(num_tokens * num_ranks, block_m)  # 每个本地专家 总的信号数
                                comp_signal = torch.zeros(num_local_experts * total_num_per_expert, dtype=torch.int32, device='cuda')

                                for i in range(num_local_experts):
                                    vaild_num = ceil_div(packed_recv_count[i], block_m)
                                    comp_signal[i * total_num_per_expert:i * total_num_per_expert + vaild_num] = threshold
                                combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x,
                                                                                     topk_idx,
                                                                                     topk_weights,
                                                                                     handle,
                                                                                     packed_recv_count=packed_recv_count,
                                                                                     comp_signal=comp_signal,
                                                                                     block_m=block_m,
                                                                                     threshold=threshold,
                                                                                     num_sms=num_sms,
                                                                                     async_finish=not return_recv_hook,
                                                                                     zero_copy=zero_copy,
                                                                                     return_recv_hook=return_recv_hook,
                                                                                     out=out)
                            else:
                                combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x,
                                                                                     topk_idx,
                                                                                     topk_weights,
                                                                                     handle,
                                                                                     use_logfmt=use_logfmt,
                                                                                     async_finish=not return_recv_hook,
                                                                                     zero_copy=zero_copy,
                                                                                     return_recv_hook=return_recv_hook,
                                                                                     out=out)

                            hook() if return_recv_hook else event.current_stream_wait()
                            if do_check:
                                diff = calc_diff(current_x * topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1), combined_x)
                                assert torch.isnan(combined_x).sum().item() == 0
                                # if not fp8_round_scale:
                                assert diff < (9e-4 if dispatch_use_quant else 1e-5), f'Error: diff={diff}, dispatch_use_quant={dispatch_use_quant}, zero_copy={zero_copy}'
                                hash_value ^= hash_tensor(combined_x)

                        if rank == 0:
                            print(f"data:{x_i}, return_recv_hook:{return_recv_hook}, quant_type:{quant_type}, ",
                                  f"fp8_round_scale:{fp8_round_scale}, quant_group_size:{quant_group_size} pass")

    print("deep_ep 全部正确性测试完成")
    if enable_combine_overlap:
        return hash_value

    # noinspection PyShadowingNames
    def large_gemm_with_hook(hook):
        mat_0 = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
        mat_1 = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
        mat_0 @ mat_1
        mat_0 @ mat_1
        hook()

    # noinspection PyShadowingNames
    def test_func(return_recv_hook: bool):
        recv_x, recv_count, handle, event, hook = \
            buffer.low_latency_dispatch(current_x, topk_idx, topk_weights, num_tokens, num_experts,
                                        quant_type=2, quant_group_size=128,
                                        async_finish=False, return_recv_hook=return_recv_hook)
        large_gemm_with_hook(hook) if return_recv_hook else None
        combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x,
                                                             topk_idx,
                                                             topk_weights,
                                                             handle,
                                                             use_logfmt=use_logfmt,
                                                             return_recv_hook=return_recv_hook)
        large_gemm_with_hook(hook) if return_recv_hook else None

    # Calculate bandwidth
    num_fp8_bytes, num_bf16_bytes = (hidden + hidden / 128 * 4 + 16), hidden * 2
    num_logfmt10_bytes = hidden * 10 / 8 + hidden / 128 * 4
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += num_fp8_bytes * num_selections
        num_combine_comm_bytes += num_bf16_bytes * num_selections

    # Dispatch + combine testing
    avg_t, min_t, max_t = bench(partial(test_func, return_recv_hook=False))
    print(f'[rank {rank}] Dispatch + combine bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / avg_t:.2f} GB/s, '
          f'avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us',
          flush=True)

    # Separate profiling
    for return_recv_hook in (False, True):
        group.barrier()
        dispatch_t, combine_t = bench_kineto(partial(test_func, return_recv_hook=return_recv_hook),
                                             kernel_names=('dispatch', 'combine'),
                                             barrier_comm_profiling=True,
                                             suppress_kineto_output=True,
                                             num_kernels_per_period=2 if return_recv_hook else 1)
        if not return_recv_hook:
            print(f'[rank {rank}] Dispatch bandwidth: {num_dispatch_comm_bytes / 1e9 / dispatch_t:.2f} GB/s, avg_t={dispatch_t * 1e6:.2f} us | '
                  f'Combine bandwidth: {num_combine_comm_bytes / 1e9 / combine_t:.2f} GB/s, avg_t={combine_t * 1e6:.2f} us',
                  flush=True)
        else:
            print(f'[rank {rank}] Dispatch send/recv time: {dispatch_t[0] * 1e6:.2f} + {dispatch_t[1] * 1e6:.2f} us | '
                  f'Combine send/recv time: {combine_t[0] * 1e6:.2f} + {combine_t[1] * 1e6:.2f} us',
                  flush=True)
    return hash_value


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    enable_combine_overlap = args.enable_combine_overlap

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts, num_topk=num_topk)
    if local_rank == 0:
        print(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...', flush=True)
    buffer = deep_ep.Buffer(group,
                            num_rdma_bytes=num_rdma_bytes,
                            low_latency_mode=True,
                            num_qps_per_rank=num_experts // num_ranks,
                            allow_nvlink_for_low_latency_mode=not args.disable_nvlink,
                            explicitly_destroy=True,
                            allow_mnnvl=args.allow_mnnvl,
                            enable_combine_overlap=enable_combine_overlap,
                            )
    print("deep_ep 初始化完成")

    if is_buffer_mode_enabled('is_ll_layered_enabled', num_ranks):
        quant_type = args.quant_type
        group_size = hidden if args.group_size == 0 else args.group_size

        # 根据不同的量化格式类型映射 C++ 后端配置
        if quant_type == 0:
            torch_type = torch.bfloat16
            token_size = 2
            scale_dim = 0
            scale_type_size = 0
        elif quant_type == 1:
            torch_type = torch.int8
            token_size = 1
            scale_dim = hidden // group_size
            scale_type_size = 4
        elif quant_type == 2:
            torch_type = torch.float8_e4m3fn
            token_size = 1
            scale_dim = hidden // group_size
            scale_type_size = 4
        elif quant_type == 4:
            torch_type = torch.float8_e5m2
            token_size = 1
            scale_dim = hidden // group_size
            scale_type_size = 4
        else:
            raise ValueError(f"Unknown quant type: {quant_type}")

        if rank == 0:
            print(f"[Rank {rank}] OP 实例化完成，量化模式配置为: {quant_type}", flush=True)

        test_main_moriep_layered(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer, quant_type, group_size, seed=1)

        # Destroy the buffer runtime and communication group
        buffer.destroy()
        dist.barrier()
        dist.destroy_process_group()
        return

    test_main(num_tokens,
              hidden,
              num_experts,
              num_topk,
              rank,
              num_ranks,
              group,
              buffer,
              use_logfmt=args.use_logfmt,
              enable_combine_overlap=enable_combine_overlap,
              seed=1)

    do_pressure_test = args.pressure_test
    for seed in range(int(100) if do_pressure_test else 0):
        if local_rank == 0:
            print(f'Testing with seed {seed} ...', flush=True)
        ref_hash = test_main(num_tokens,
                             hidden,
                             num_experts,
                             num_topk,
                             rank,
                             num_ranks,
                             group,
                             buffer,
                             use_logfmt=args.use_logfmt,
                             enable_combine_overlap=enable_combine_overlap,
                             seed=seed)
        for _ in range(20):
            assert test_main(num_tokens,
                             hidden,
                             num_experts,
                             num_topk,
                             rank,
                             num_ranks,
                             group,
                             buffer,
                             use_logfmt=args.use_logfmt,
                             enable_combine_overlap=enable_combine_overlap,
                             seed=seed) == ref_hash, f'Error: seed={seed}'

    # Destroy the buffer runtime and communication group
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    # TODO: you may modify NUMA binding for less CPU overhead
    # TODO: buggy with `num_tokens=512`
    parser = argparse.ArgumentParser(description='Test low-latency EP kernels')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-tokens', type=int, default=128, help='Number of tokens (default: 128)')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden dimension size (default: 7168)')
    parser.add_argument('--num-topk', type=int, default=8, help='Number of top-k experts (default: 8)')
    parser.add_argument('--num-experts', type=int, default=288, help='Number of experts (default: 288)')
    parser.add_argument('--allow-mnnvl', action="store_true", help='Allow MNNVL for communication')
    parser.add_argument('--disable-nvlink', action='store_true', help='Whether to disable NVLink for testing')
    parser.add_argument("--pressure-test", action='store_true', help='Whether to do pressure test')
    parser.add_argument("--shrink-test", action='store_true', help='Whether to simulate failure and test shrink mode')
    parser.add_argument('--use-logfmt', action='store_true', help='Whether to test LogFMT combine')
    parser.add_argument("--enable-combine-overlap", action='store_true', help='Enable GEMM-compute/communication overlap in the combine phase')
    parser.add_argument('--quant-type', type=int, default=0, choices=[0, 1, 2, 4], help="'0 -> bf16', '1->int8', '2->fp8_e4m3', '3->fp8_e8m0', '4->fp8_e5m2'")
    parser.add_argument('--group-size', type=int, default=0, help='Group size for quantization. Set to 0 to auto-match hidden_dim (Per-Token layout)')

    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
