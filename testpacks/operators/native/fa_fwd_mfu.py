from fa_fwd_fixlen import *

query_layer = torch.randn((batch_size, seq_len, nhead, head_dim_qk), dtype=input_dtype, device=device)
key_layer   = torch.randn((batch_size, seq_len, int(nhead / groups), head_dim_v), dtype=input_dtype, device=device)
value_layer = torch.randn((batch_size, seq_len, int(nhead / groups), head_dim_v), dtype=input_dtype, device=device)

# input_params = (
#     query_layer,
#     key_layer,
#     value_layer,
#     None,
#     None,
#     0.0,
#     1.0 / math.sqrt(head_dim_qk),
#     causal_mask,
#     -1,
#     -1,
#     0.0,
#     False,
#     None,
#     False,
#     None,
#     None,
#     None,
#     None,
#     0.0
# )
input_params = (
    query_layer,
    key_layer,
    value_layer,
    None,
    None,
    0.0,
    1.0 / math.sqrt(head_dim_qk),
    causal_mask,
    -1,
    -1,
    0.0,
    False,
    None,
    None,
    None,
    None
)
output, output_lse, *rest_extend = flash_attn_2_cuda.fwd_bshd(*input_params)
print("output: ", output.shape)


def benchmark_fwd_mfu():
    output, output_lse, *rest_extend = flash_attn_2_cuda.fwd_bshd(*input_params)
import triton
cost_time_mean = triton.testing.do_bench_cudagraph(benchmark_fwd_mfu)
print("Time: {}".format(cost_time_mean))

# 统计计算量
batch_size  = query_layer.shape[0]
num_heads   = query_layer.shape[1]
seqlen_q    = query_layer.shape[2]
seqlen_k    = key_layer.shape[2]
head_dim_qk = query_layer.shape[3]
seqlen_q, num_heads = num_heads, seqlen_q
seqlen_k = key_layer.shape[1]
qk_flops = 2 * batch_size * num_heads * seqlen_q * seqlen_k * head_dim_qk
pv_flops = 2 * batch_size * num_heads * seqlen_q * seqlen_k * head_dim_v
flops_count = qk_flops + pv_flops # get_fa_flops(batch_size, seqlen_q, num_heads, head_dim_qk, True, "fwd")
if (causal_mask): flops_count = flops_count / 2
tflops = flops_count / (cost_time_mean * 1e9)
print("FLOPS: ", flops_count)
print("Performance: \x1b[35m{:.1f}\x1b[0m TFLOPS".format(tflops))
