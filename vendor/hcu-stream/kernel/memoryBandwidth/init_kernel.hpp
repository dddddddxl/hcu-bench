#pragma once
#include "hip/hip_runtime.h"

inline __global__ void init_kernel(__int128_t *a, __int128_t initA) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] = initA;
}

inline __global__ void init_kernel(int *a, int initA) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] = initA;
}

