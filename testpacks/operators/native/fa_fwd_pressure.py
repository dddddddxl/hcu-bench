import ast
from flash_attention_utils import *


# 多头注意力
batch_size    = 1
nhead         = 32
seq_len       = 32768
head_dim_qk   = 128
head_dim_v    = 128
causal_mask   = True
use_bfloat16  = False
suffix        = ".float16"
groups        = 8
window_size_left  = 0
window_size_right = 0
is_local          = False
use_alibi         = False

try:
    args = sys.argv
    if (len(args) > 1): batch_size = int(args[1])
    if (len(args) > 2): nhead      = int(args[2])
    if (len(args) > 3): seq_len    = int(args[3])
    if (len(args) > 4): head_dim_qk = int(args[4])
    if (len(args) > 5): head_dim_v  = int(args[5])
    if (len(args) > 6 and args[6] != "true"):
        causal_mask = False
    if (len(args) > 7 and args[7] == "bfloat16"):
        use_bfloat16 = True
        suffix   = ".bfloat16"
    if (len(args) > 8):
        groups = int(args[8])
        assert groups >= 1, "groups must >= 1"
    if (len(args) > 9):
        window_size = ast.literal_eval(args[9])
        window_size_left  = window_size[0]
        window_size_right = window_size[1]
        is_local = bool(not (window_size_left < 0 and window_size_right < 0))
    if (len(args) > 10):
        use_alibi = bool(args[10] == "true")
except:
    print("example: python generate.py 1 1 192 128 true fp16 1 [3,0]")

if (args[-1] == "--whl"):
    import flash_attn
    from flash_attn.flash_attn_interface import flash_attn_cuda as flash_attn_2_cuda
    print("\x1b[33mload from {}\x1b[0m".format(flash_attn_2_cuda.__file__))
else:
    # 加载动态库(模块名必须匹配共享库导出的 PyInit_flash_attn_hg_cuda)
    path_to_so = "/usr/local/lib/python3.10/dist-packages/flash_attn/lib/libflash_attention.so"
    print("\x1b[33mload from {}\x1b[0m".format(path_to_so))
    spec = importlib.util.spec_from_file_location("flash_attn_hg_cuda", path_to_so)
    flash_attn_2_cuda = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(flash_attn_2_cuda)

print("batch_size: ", batch_size)
print("nhead: ",      nhead)
print("seq_len: ",    seq_len)
print("head_dim_qk: ",  head_dim_qk)
print("head_dim_v: ",   head_dim_v)
print("causal_mask: ", causal_mask)
print("use_bfloat16: ", use_bfloat16)
print("groups: ", groups)
print("window_size_left: ", window_size_left)
print("window_size_right: ", window_size_right)
print("is_local: ", is_local)
print("use_alibi: ", use_alibi)
input_dtype  = torch.float16 if (not use_bfloat16) else torch.bfloat16
softmax_type = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    fa_test_dir = os.environ["BENCH_FA_CACHE_DIR"]
    os.makedirs(fa_test_dir, exist_ok=True)
    problem_id_name = "{}_{}_{}_{}_{}_{}_{}_{}.pt".format(batch_size, nhead, seq_len, head_dim_qk, head_dim_v, causal_mask, groups, "float16" if (not use_bfloat16) else "bfloat16")
    problem_id = os.path.join(fa_test_dir, problem_id_name)
    print("problem_id: ", problem_id)

    if (not os.path.exists(problem_id)):
        print("generating {} from scratch".format(problem_id))
        # Q, K, V
        r1 = -1.0
        r2 = 1.0
        nhead_k = int(nhead / groups)
        Q = ((r1 - r2) * torch.rand((batch_size, nhead, seq_len, head_dim_qk)) + r2).to(input_dtype).to(device)
        K = ((r1 - r2) * torch.rand((batch_size, nhead_k, seq_len, head_dim_qk)) + r2).to(input_dtype).to(device)
        V = ((r1 - r2) * torch.rand((batch_size, nhead_k, seq_len, head_dim_v)) + r2).to(input_dtype).to(device)
        print("Q : ", Q.shape, Q.dtype)
        print("K : ", K.shape, K.dtype)
        print("V : ", V.shape, V.dtype)

        #######################################################################################################################
        if (not use_alibi):
            split_count = K.shape[1]
            Q_groups = torch.chunk(Q, split_count, dim=1)
            K_groups = torch.chunk(K, split_count, dim=1)
            V_groups = torch.chunk(V, split_count, dim=1)
            O   = []
            lse = []
            alibi_slope = []
            S_max = []
            S_sum = []
            for i, (_Q, _K, _V) in enumerate(zip(Q_groups, K_groups, V_groups)):
                _O, _lse, _alibi_slope, _s_max, _s_sum = standard_attention(
                    _Q, _K.repeat(1, groups, 1, 1), _V.repeat(1, groups, 1, 1),
                    causal_mask=causal_mask,
                    is_local=is_local,
                    use_alibi=use_alibi
                )
                O.append(_O)
                lse.append(_lse)
                alibi_slope.append(_alibi_slope)
                S_max.append(_s_max)
                S_sum.append(_s_sum)
            O   = torch.concatenate(O, dim=1)
            golden_lse = torch.concatenate(lse, dim=1)
            S_max = torch.concatenate(S_max, dim=1)
            S_sum = torch.concatenate(S_sum, dim=1)
            alibi_slope = torch.concatenate(alibi_slope, dim=1) if (use_alibi) else None
        else:
            O, golden_lse, alibi_slope, S_max, S_sum = standard_attention(Q, K, V)
        print("O : ", O.shape, O.dtype)
        #######################################################################################################################
        torch.cuda.empty_cache()
        # 把输入, 输出都转置一下, 得到 bshd 的输入 layout
        query_layer = Q.permute(0, 2, 1, 3).contiguous()
        key_layer   = K.permute(0, 2, 1, 3).contiguous()
        value_layer = V.permute(0, 2, 1, 3).contiguous()
        golden      = O.contiguous()
        torch.save((query_layer, key_layer, value_layer, golden, golden_lse), problem_id)
        torch.cuda.empty_cache()
    else:
        # 不计算直接加载测试
        (query_layer, key_layer, value_layer, golden, golden_lse) = torch.load(problem_id)
        print("load Q,K,V from {}".format(problem_id))

    # 获取环境变量
    USE_BSHD = bool(os.getenv("USE_BSHD") is not None)
    USE_SBHD = bool(os.getenv("USE_SBHD") is not None)

    # 根据环境变量决定 layout 变换操作
    if (USE_BSHD):
        USE_SBHD = False
    elif (USE_SBHD):
        # bshd -> sbhd
        query_layer = query_layer.transpose(1, 2).contiguous()
        key_layer   = key_layer.transpose(1, 2).contiguous()
        value_layer = value_layer.transpose(1, 2).contiguous()
    else:
        # 保持 bshd: wheel 的 fwd 期望 bshd 输入(bhsd=True 那条路径结果不正确),
        # 本地 hg 库对应 fwd_bshd。
        pass
    print("USE_BSHD: ", USE_BSHD)
    print("USE_SBHD: ", USE_SBHD)
    print("input shape: ", query_layer.shape, key_layer.shape, value_layer.shape)
    print("input dtype: ", query_layer.dtype, key_layer.dtype, value_layer.dtype)
    print("input stride: ", query_layer.stride(), key_layer.stride(), value_layer.stride())
    torch.cuda.synchronize()

    # 决定调用哪个接口
    # wheel(flash_attn_2_cuda) 只有通用 fwd, 用第 14 个参数 bhsd 指定 layout;
    # 本地 hg 库按 layout 提供 fwd_bshd / fwd_bhsd。
    IS_WHEEL = not hasattr(flash_attn_2_cuda, "fwd_bshd")
    if (USE_SBHD):
        raise NotImplementedError("fwd_sbhd_fuse is not inplemented yet")
    elif (IS_WHEEL):
        fwd_fixlen_func = flash_attn_2_cuda.fwd
    else:
        fwd_fixlen_func = flash_attn_2_cuda.fwd_bshd

    # 统一构造前向入参 (与当前 flash_attn 的 fwd 签名一致, 共 16 个):
    #   q, k, v, out_, alibi_slopes, dropout, softmax_scale, causal,
    #   window_left, window_right, softcap, return_softmax, gen,
    #   bhsd, s_aux, <float>
    # 注意第 14 个参数是 layout 标志: 上面的张量已按 layout 变换过,
    # 非 USE_BSHD 时张量是 bhsd, 所以传 (not USE_BSHD)。
    def build_input_params(q, k, v):
        if (IS_WHEEL):
            # fwd(q,k,v,out,alibi,dropout,scale,causal,wl,wr,softcap,return_softmax,
            #     gen, bhsd, s_aux, <float>)
            return (q, k, v, None, None, 0.0, 1.0 / math.sqrt(head_dim_qk),
                    causal_mask, -1, -1, 0.0, False, None, False, None, 0.0)
        else:
            # fwd_bshd(q,k,v,out,alibi,dropout,scale,causal,wl,wr,softcap,
            #          return_softmax, gen, q_descale, k_descale, v_descale)
            return (q, k, v, None, None, 0.0, 1.0 / math.sqrt(head_dim_qk),
                    causal_mask, -1, -1, 0.0, False, None, None, None, None)

    input_params = build_input_params(query_layer, key_layer, value_layer)
    outputs = fwd_fixlen_func(*input_params)
    torch.cuda.synchronize()
    fa_output   = outputs[0]
    softmax_lse = outputs[5]

    # 对输出做变换: 输出为 bshd, .pt 里的 golden 为 bhsd
    if (USE_SBHD):
        golden = golden.permute(2, 0, 1, 3).contiguous()
        golden_lse = golden_lse.transpose(1, 2).contiguous()
    else:
        golden = golden.transpose(1, 2).contiguous()
        # 输出/内核的 softmax_lse 是 (b, h, s), 与 .pt 里的 golden_lse 同序, 只需去掉尾维
        if (golden_lse.dim() == 4):
            golden_lse = golden_lse.squeeze(-1)
    golden_lse = golden_lse.reshape(softmax_lse.shape).to(softmax_lse.device)

    # 计算正确性情况
    abs_diff = torch.abs(fa_output - golden)
    lse_abs_diff  = torch.abs(softmax_lse - golden_lse)
    print("Output: \x1b[35m{:.12f}\x1b[0m  |  \x1b[35m{:.12f}\x1b[0m  |  [{:.2f}, {:.2f}]  {}".format(abs_diff.mean(), abs_diff.max(), fa_output.min().item(), fa_output.max().item(), fa_output.shape))
    print("LSE   : \x1b[35m{:.12f}\x1b[0m  |  \x1b[35m{:.12f}\x1b[0m  |  [{:.2f}, {:.2f}]  {} ".format(lse_abs_diff.mean(), lse_abs_diff.max(), softmax_lse.min().item(), softmax_lse.max().item(), softmax_lse.shape))

    # 检查是否有异常值
    occur_nan_in_output = torch.any(torch.isnan(fa_output))
    occur_inf_in_output = torch.any(torch.isinf(fa_output))
    occur_nan_in_lse    = torch.any(torch.isnan(softmax_lse))
    occur_inf_in_lse    = torch.any(torch.isinf(softmax_lse))
    print("CHECK NaN: ", get_status_str(not (occur_nan_in_output or occur_nan_in_lse)))
    print("CHECK Inf: ", get_status_str(not (occur_inf_in_output or occur_inf_in_lse)))

    # 测试性能
    if (os.getenv("FA_DEBUG") is None):
        cost_time = []
        for __iter in range(10):
            t = benchmark_forward(fwd_fixlen_func, *input_params, repeats=1, verbose=False) / 1e6
            if (__iter > 0): cost_time.append(t)
        cost_time = numpy.array(cost_time)
        cost_time_mean = cost_time.mean()
        cost_time = numpy.delete(cost_time, numpy.where(cost_time < (0.8 * cost_time_mean))) # 去除突发低数据
        cost_time_mean = cost_time.mean()
        # 统计计算量
        batch_size  = query_layer.shape[0]
        num_heads   = query_layer.shape[1]
        seqlen_q    = query_layer.shape[2]
        seqlen_k    = key_layer.shape[2]
        head_dim_qk = query_layer.shape[3]
        if (not USE_SBHD):
            seqlen_q, num_heads = num_heads, seqlen_q
            seqlen_k = key_layer.shape[1]
        qk_flops = 2 * batch_size * num_heads * seqlen_q * seqlen_k * head_dim_qk
        pv_flops = 2 * batch_size * num_heads * seqlen_q * seqlen_k * head_dim_v
        flops_count = qk_flops + pv_flops # get_fa_flops(batch_size, seqlen_q, num_heads, head_dim_qk, True, "fwd")
        if (causal_mask): flops_count = flops_count / 2
        tflops = flops_count / (cost_time_mean * 1e12)
        print("FLOPS: ", flops_count)
        print("Performance: \x1b[35m{:.1f}\x1b[0m TFLOPS".format(tflops))

        # 压力测试
        pressure_count = 1000000
        for p in range(pressure_count):
            torch.cuda.empty_cache()
            pressure_query_layer = query_layer.clone()
            pressure_key_layer   = key_layer.clone()
            pressure_value_layer = value_layer.clone()
            outputs = fwd_fixlen_func(
                *build_input_params(pressure_query_layer, pressure_key_layer, pressure_value_layer)
            )
            torch.cuda.synchronize()
            pressure_fa_output   = outputs[0]
            pressure_softmax_lse = outputs[5]
            assert torch.equal(pressure_fa_output, fa_output), "Unstable"
            assert torch.equal(pressure_softmax_lse, softmax_lse), "Unstable"
            pressure_query_layer.fill_(0)
            pressure_key_layer.fill_(0)
            pressure_value_layer.fill_(0)
            del pressure_query_layer, pressure_key_layer, pressure_value_layer
            sys.stdout.write("\rPressure Test: {}/{}".format(p + 1, pressure_count))
        print(" \x1b[32mPASS\x1b[0m")
