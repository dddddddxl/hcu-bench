# -*- coding: utf-8 -*-

from flash_attention_utils import *
if __name__ == '__main__':

    # 选择 SDK
    if (use_cuda_toolkits):
        from vllm.vllm_flash_attn import flash_attn_varlen_func
    elif (use_rocm_toolkits):
        if (args.whl):
            from flash_attn.flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache
            from flash_attn.flash_attn_interface import flash_attn_cuda
        else:
            from flash_attn.flash_attn_interface import (
                flash_attn_varlen_func,
                flash_attn_with_kvcache,
                flash_attn_cuda,
                hg_flash_attn_varlen_func,
            )

    # 记录数据
    statistics = []

    # 遍历多 batch
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:

        # 遍历常用的 seqlen
        for used_seqlen_kv in [2048, 3072, 3584, 4096, 5000]:

            # 得到 Q 的长度
            seqlen_q = [random.randint(1, 1) for i in range(batch_size)]
            seqlen_q_sum = sum(seqlen_q)
            max_seqlen_q = max(seqlen_q)
            cu_seqlens_q = numpy.array([0] + numpy.cumsum(seqlen_q).tolist()).astype("int32")
            cu_seqlens_q = torch.from_numpy(cu_seqlens_q)

            # 得到 KV 的长度
            cache_seqlens  = [random.randint(used_seqlen_kv, used_seqlen_kv) for i in range(batch_size)]

            # 指定分页块的大小, nvidia 64, ours 128
            page_block_size = 16 if (use_cuda_toolkits) else args.block_size

            # 根据分页块大小计算实际需要的页表的大小
            max_seqlen_k = max(cache_seqlens)
            seqlen_kv_real_required_page = [math.ceil(it / page_block_size) for it in cache_seqlens]
            seqlen_kv_real_required_page_sum = sum(seqlen_kv_real_required_page)
            # 默认按照最大 seqlenkv 的来分配
            seqlen_kv_max_required_page = math.ceil(max_seqlen_k / page_block_size)
            seqlen_kv_max_required_page_total = batch_size * seqlen_kv_max_required_page
            # 打乱页表
            shuffle = True
            if (shuffle):
                block_random = torch.randperm(seqlen_kv_max_required_page_total, dtype=torch.int32, device="cuda")
            else:
                block_random = torch.arange(seqlen_kv_max_required_page_total , dtype=torch.int32)
            page_table    = []
            seq_block_incre = 0
            for i in range(batch_size):
                blocks_pad = [0] * seqlen_kv_max_required_page
                if (shuffle):
                    blocks_pad[:seqlen_kv_real_required_page[i]] = block_random[seq_block_incre: seq_block_incre + seqlen_kv_real_required_page[i]].cpu().tolist()
                    seq_block_incre += seqlen_kv_real_required_page[i]
                else:
                    blocks_pad = block_random[seq_block_incre: seq_block_incre + seqlen_kv_max_required_page].cpu().tolist()
                    seq_block_incre += seqlen_kv_max_required_page
                page_table.append(torch.IntTensor(blocks_pad))
            page_table = torch.stack(page_table).contiguous().to("cuda")
            # 创建基本参数
            head_dim_qk   = 128
            head_dim_v    = 128
            num_heads     = 24
            num_heads_kv  = 2
            infer_dtype   = torch.float16 # deepseek 默认使用 bfloat16 推理
            if (args.bf16): infer_dtype = torch.bfloat16 # 除非命令行指定用 fp16, 不受 args.dtype 影响
            softmax_scale = 1.0 / math.sqrt(head_dim_qk)
            causal        = True
            window_size   = (-1, -1)
            alibi_slopes  = None
            softcap       = 0.0
            fa_version    = 2
            q_descale     = torch.ones((batch_size), dtype=infer_dtype, device="cuda")
            k_descale     = torch.ones((batch_size), dtype=infer_dtype, device="cuda")
            v_descale     = torch.ones((batch_size), dtype=infer_dtype, device="cuda")

            # 打印基本信息
            basic_info = (batch_size, num_heads, num_heads_kv, used_seqlen_kv, head_dim_qk, head_dim_v, infer_dtype)
            print("bs {}, qhead {}, kvhead {}, seqlen_k {} headdim {} {} dtype {}".format(*basic_info))

            # 创建输入张量
            if (args.pad):
                query_origin_tensor = torch.randn((seqlen_q_sum, num_heads + 16, head_dim_qk), dtype=infer_dtype, device="cuda")
                q = query_origin_tensor[:, :num_heads]
            else:
                q = torch.randn((seqlen_q_sum, num_heads, head_dim_qk), dtype=infer_dtype, device="cuda")
            k_cache   = torch.randn((seqlen_kv_max_required_page_total, page_block_size, num_heads_kv, head_dim_qk), device="cuda", dtype=infer_dtype)
            v_cache = torch.randn((seqlen_kv_max_required_page_total, page_block_size, num_heads_kv, head_dim_v), device="cuda", dtype=infer_dtype)
            vllm_golden  = None
            cu_seqlens_q = cu_seqlens_q.to(q.device)
            cache_seqlens    = torch.from_numpy(numpy.array(cache_seqlens).astype("int32")).to(q.device)

            # 先从 kvcache 中还原出 key 和 value
            key_original   = []
            value_original = []
            for b in range(batch_size):
                # 获取页表索引
                index = page_table[b]
                # 获取实际的索引
                max_page_blocks = math.ceil(cache_seqlens[b] / page_block_size)
                actual_index = index[:max_page_blocks]
                # 根据该页表索引获取当前 seqlenkv 的内容
                key_content = k_cache[actual_index]
                # reshape 回去
                key_content = key_content.view(-1, num_heads_kv, head_dim_qk)[:cache_seqlens[b]].contiguous()
                # 同理
                value_content = v_cache[actual_index].view(-1, num_heads_kv, head_dim_v)[:cache_seqlens[b]].contiguous()
                key_original.append(key_content)
                value_original.append(value_content)

            # 同理还原出 query 的内容
            query_original = []
            cum_q = 0
            for b in range(batch_size):
                query_len = cu_seqlens_q[b + 1] - cu_seqlens_q[b]
                query_content = q[cum_q: cum_q + query_len]
                query_original.append(query_content.contiguous())
                cum_q += query_len

            # 重新实现 self-attention
            golden = []
            golden_lse = []
            golden_max = []
            for b in range(batch_size):
                tmp_output, lse, scores_max, scores_sum = scaled_dot_product_attention(query_original[b], key_original[b], value_original[b], num_heads, num_heads_kv, is_causal=causal, USE_CPU=args.cpu, transpose_io=True)
                golden.append(tmp_output)
                golden_lse.append(lse)
                golden_max.append(scores_max)
            golden = torch.cat(golden, dim=0)
            golden_lse = torch.cat(golden_lse, dim=-1)
            golden_max = torch.cat(golden_max, dim=-1)
            if (args.detail):
                # 展示一下输入数据
                print("--------------------------------------------------------------------------------------------")
                print("q: ", q.shape, q.dtype, q.is_contiguous(), q.stride())
                print("k_cache: ", k_cache.shape, k_cache.dtype, k_cache.is_contiguous(), k_cache.stride())
                print("v_cache: ", v_cache.shape, v_cache.dtype, v_cache.is_contiguous(), v_cache.stride())
                print("cu_seqlens_q: ", cu_seqlens_q.shape, cu_seqlens_q.dtype, cu_seqlens_q.is_contiguous())
                print("cu_seqlens_q: ", cu_seqlens_q)
                print("max_seqlen_q: ", max_seqlen_q)
                print("cache_seqlens: ", cache_seqlens)
                print("max_seqlen_k: ", max_seqlen_k)
                print("softmax_scale: ", softmax_scale)
                print("causal: ", causal)
                print("window_size: ", window_size)
                print("alibi_slopes: ", alibi_slopes)
                print("page_table: ", page_table.shape, page_table.dtype, page_table.is_contiguous(), page_table.stride())
                print("page_table: ", page_table)
                print("softcap: ", softcap)
                print("fa_version: ", fa_version)
                print("q_descale: ", q_descale.shape, q_descale.dtype, q_descale.tolist())
                print("k_descale: ", k_descale.shape, k_descale.dtype, k_descale.tolist())
                print("v_descale: ", v_descale.shape, v_descale.dtype, v_descale.tolist())

                print("--------------------------------------------------------------------------------------------")
                print("golden: ", golden.shape)
                print("golden_lse: ", golden_lse.shape)
                print("--------------------------------------------------------------------------------------------")
            if (True):
                fa_output, fa_lse = flash_attn_cuda.hg_prefix_decode_varlen_fwd(
                    q,
                    k_cache,
                    v_cache,
                    None, # out_
                    cu_seqlens_q,
                    None, # cu_seqlens_k
                    cache_seqlens,
                    alibi_slopes,
                    page_table,
                    max_seqlen_q,
                    0,
                    0.0, # dropout
                    softmax_scale,
                    False, # zero_tensors
                    causal,
                    window_size[0],
                    window_size[1],
                    softcap,
                    True, # return_softmax_lse,
                    1,
                    None, # q_descale
                    None, # k_descale
                    None, # v_descale
                    None, # s_aux
                    infer_dtype == torch.bfloat16,
                )
            else:
                fa_output, fa_lse, *rest = flash_attn_with_kvcache(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cache_seqlens,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    softcap=softcap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=True,
                )
            torch.cuda.synchronize()
            if (vllm_golden is not None):
                # 检查保存流程是否有错误
                cal_diff(fa_output, vllm_golden, "check")

            if (args.detail):
                print("fa_output: ", fa_output.shape)
                if (fa_lse is not None): print("fa_lse: ", fa_lse.shape)
            # 检验精度如何
            cal_diff(golden, fa_output, "accuracy", False)
            if (fa_lse is not None): cal_diff(golden_lse, fa_lse, "softmax_lse", False)
            print("--------------------------------------------------------------------------------------------")

            # benchmark 性能数据
            import triton
            def benchmark_prefix_prefill():
                _ = flash_attn_cuda.hg_prefix_decode_varlen_fwd(
                    q,
                    k_cache,
                    v_cache,
                    None, # out_
                    cu_seqlens_q,
                    None, # cu_seqlens_k
                    cache_seqlens,
                    alibi_slopes,
                    page_table,
                    max_seqlen_q,
                    0,
                    0.0, # dropout
                    softmax_scale,
                    False, # zero_tensors
                    causal,
                    window_size[0],
                    window_size[1],
                    softcap,
                    True, # return_softmax_lse,
                    1,
                    None, # q_descale
                    None, # k_descale
                    None, # v_descale
                    None, # s_aux
                    False,
                )
            import triton
            bench_fn = triton.testing.do_bench_cudagraph if os.getenv("PA_USE_CUDAGRAPH_BENCH") is not None else triton.testing.do_bench
            t = bench_fn(benchmark_prefix_prefill)
            FLOPS = float(0)
            BYTES = float(0)
            for b in range(batch_size):
                batch_seqlen_q = cu_seqlens_q[b + 1] - cu_seqlens_q[b]
                batch_seqlen_k = cache_seqlens[b]
                undo_flops = batch_seqlen_q * batch_seqlen_q / 2 if (causal) else 0
                qk_flops = num_heads * (batch_seqlen_q * batch_seqlen_k - undo_flops) * head_dim_qk * 2
                pv_flops = num_heads * (batch_seqlen_q * batch_seqlen_k - undo_flops) * head_dim_v * 2
                FLOPS += qk_flops + pv_flops
                q_load = batch_seqlen_q * num_heads * head_dim_qk
                k_load = batch_seqlen_k * num_heads_kv * head_dim_qk # k load not only once
                v_load = batch_seqlen_k * num_heads_kv * head_dim_v
                BYTES += (q_load + k_load + v_load) * (torch.finfo(infer_dtype).bits // 8) # ignore storation ?
            compute_bandwidth = BYTES / 10 ** 6 / t
            print(f"Performance: {t:.3f} ms, \x1b[35m{FLOPS / 10 ** 9 / t:.2f}\x1b[0m TFLOPS, \x1b[35m{compute_bandwidth:.1f}\x1b[0m GB/s")

            # 记录当前数据
            statistics.append((compute_bandwidth, t * 1e3, *basic_info))

    print("Total Statistics:")
    for stat_item in statistics:
        print("\x1b[35m{:.1f}\x1b[0m {:.1f} : {} {} {} {} {} {} {}".format(*stat_item))
    print("ALL: ")
    ultimate = []
    for stat_item in statistics:
        print("{:.1f}".format(stat_item[0]))
        ultimate.append(stat_item[0])
    print("\n\n{:.1f}".format(max(ultimate)))
