#pragma once
#include "hip/hip_runtime.h"

__global__ void L2_read_kernel(const __int128_t *a, __int128_t *b, int iter,
                               int parts) {
  __int128_t sum = (__int128_t)0;
  // #pragma unroll
  for (int i = 0; i < iter; i++) {
    const int idx =
        (parts * i + (blockIdx.x % parts)) * blockDim.x + threadIdx.x;
    sum += a[idx];
  }

  if (sum == 123456) {
    b[threadIdx.x] = sum;
  }
}
