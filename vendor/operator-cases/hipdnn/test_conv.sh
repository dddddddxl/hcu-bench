# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Integration metadata only; the following DTK command payload is unchanged.
export MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32=0
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 --pre-pad-h 2 --post-pad-h 1 --pre-pad-w 2 --post-pad-w 1 -u 1 -v 1 -l 1 -j 1 -g 1 -d fp16 -L NCHW -F 1
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 --pre-pad-h 2 --post-pad-h 1 --pre-pad-w 2 --post-pad-w 1 -u 1 -v 1 -l 1 -j 1 -g 1 -d fp32 -L NCHW -F 1
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 --pre-pad-h 2 --post-pad-h 1 --pre-pad-w 2 --post-pad-w 1 -u 1 -v 1 -l 1 -j 1 -g 1 -d bfp16 -L NCHW -F 1
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pre-pad-h 2 --post-pad-h 1 --pre-pad-w 2 --post-pad-w 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 1 -d bfp16 -L NCHW -F 1
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pre-pad-h 2 --post-pad-h 1 --pre-pad-w 2 --post-pad-w 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 1 -d fp16 -L NCHW -F 1
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pre-pad-h 2 --post-pad-h 1 --pre-pad-w 2 --post-pad-w 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 1 -d fp32 -L NCHW -F 1
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 --pre-pad-h 2 --post-pad-h 1 --pre-pad-w 2 --post-pad-w 1 -u 1 -v 1 -l 1 -j 1 -g 1 -d fp32 -L NCHW -F 2
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 --pre-pad-h 2 --post-pad-h 1 --pre-pad-w 2 --post-pad-w 1 -u 1 -v 1 -l 1 -j 1 -g 1 -d fp16 -L NCHW -F 4
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 --pre-pad-h 2 --post-pad-h 1 --pre-pad-w 2 --post-pad-w 1 -u 1 -v 1 -l 1 -j 1 -g 1 -d fp32 -L NCHW -F 4
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 --pre-pad-h 2 --post-pad-h 1 --pre-pad-w 2 --post-pad-w 1 -u 1 -v 1 -l 1 -j 1 -g 1 -d bfp16 -L NCHW -F 4
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pre-pad-h 2 --post-pad-h 1 --pre-pad-w 2 --post-pad-w 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 1 -d bfp16 -L NCHW -F 4
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 64 -W 64 -k 8 --fil_d 3 -y 3 -x 3 --pre-pad-h 2 --post-pad-h 1 --pre-pad-w 2 --post-pad-w 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 1 -d fp16 -L NCHW -F 4
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pre-pad-h 2 --post-pad-h 1 --pre-pad-w 2 --post-pad-w 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 1 -d fp32 -L NCHW -F 4
export MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32=1
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 --pre-pad-h 2 --post-pad-h 1 --pre-pad-w 2 --post-pad-w 1 -u 1 -v 1 -l 1 -j 1 -g 1 -d tf32 -L NCHW -F 1
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pre-pad-h 2 --post-pad-h 1 --pre-pad-w 2 --post-pad-w 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 1 -d tf32 -L NCHW -F 1
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 --pre-pad-h 2 --post-pad-h 1 --pre-pad-w 2 --post-pad-w 1 -u 1 -v 1 -l 1 -j 1 -g 1 -d tf32 -L NCHW -F 2
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 --pre-pad-h 2 --post-pad-h 1 --pre-pad-w 2 --post-pad-w 1 -u 1 -v 1 -l 1 -j 1 -g 1 -d tf32 -L NCHW -F 4
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pre-pad-h 2 --post-pad-h 1 --pre-pad-w 2 --post-pad-w 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 1 -d tf32 -L NCHW -F 4

export MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32=0
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -d fp16 -L NCHW -F 1
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -d fp32 -L NCHW -F 1
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -d bfp16 -L NCHW -F 1
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -d fp16 -L NHWC -F 1
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -d fp32 -L NHWC -F 1
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -d bfp16 -L NHWC -F 1
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -d int8 -L NCHW -F 1
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -d int8 -L NHWC -F 1
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 2 -d bfp16 -L NCHW -F 1
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 2 -d fp16 -L NCHW -F 1
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 2 -d fp32 -L NCHW -F 1
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 2 -d bfp16 -L NHWC -F 1
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 2 -d fp16 -L NHWC -F 1
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 2 -d fp32 -L NHWC -F 1
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -d fp16 -L NCHW -F 2
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -d fp32 -L NCHW -F 2
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -d bfp16 -L NCHW -F 2
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -d fp16 -L NHWC -F 2
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -d fp32 -L NHWC -F 2
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -d bfp16 -L NHWC -F 2
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 2 -d bfp16 -L NCHW -F 2
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 2 -d fp16 -L NCHW -F 2
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 2 -d fp32 -L NCHW -F 2
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 2 -d bfp16 -L NHWC -F 2
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 2 -d fp16 -L NHWC -F 2
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 2 -d fp32 -L NHWC -F 2
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -d fp16 -L NCHW -F 4
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -d fp32 -L NCHW -F 4
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -d bfp16 -L NCHW -F 4
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -d fp16 -L NHWC -F 4
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -d fp32 -L NHWC -F 4
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -d bfp16 -L NHWC -F 4
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 2 -d bfp16 -L NCHW -F 4
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 64 -W 64 -k 8 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 2 -d fp16 -L NCHW -F 4
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 2 -d fp32 -L NCHW -F 4
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 2 -d bfp16 -L NHWC -F 4
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 18 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 2 -d fp16 -L NHWC -F 4
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 2 -d fp32 -L NHWC -F 4
export MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32=1
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -d tf32 -L NCHW -F 1
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -d tf32 -L NHWC -F 1
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 2 -d tf32 -L NCHW -F 1
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 2 -d tf32 -L NHWC -F 1
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -d tf32 -L NCHW -F 2
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -d tf32 -L NHWC -F 2
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 2 -d tf32 -L NCHW -F 2
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 2 -d tf32 -L NHWC -F 2
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -d tf32 -L NCHW -F 4
test_conv --cpu-validation -n 64 -c 128 -H 28 -W 28 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 4 -d tf32 -L NHWC -F 4
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 2 -d tf32 -L NCHW -F 4
test_conv --cpu-validation -n 2 -c 8 --spatial_dim 3 --in_d 4 -H 182 -W 182 -k 8 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 -g 2 -d tf32 -L NHWC -F 4
