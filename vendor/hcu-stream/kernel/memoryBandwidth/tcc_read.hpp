#pragma once
#include "hip/hip_runtime.h"
__global__ void stride_read_kernel(const __int128_t *a, __int128_t *b,
                                   uint32_t width) {
  const int i = (blockDim.y * blockIdx.x + threadIdx.y) * width + threadIdx.x;
  __int128_t value = a[i];
  // 此处并不会发生实际写出，仅为了不让编译器优化掉读指令
  if (value == 123456) {
    b[i] = value;
  }
}