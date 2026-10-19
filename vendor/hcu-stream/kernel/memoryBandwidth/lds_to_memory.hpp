#pragma once
#include "hip/hip_runtime.h"
__global__ void lds_to_memory_kernel(__int128_t *array) {
  __shared__ __int128_t sharedTmp[256];
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  array[threadId] = sharedTmp[threadIdx.x];
  // avoid compiler optimizations
  if (array[0] == 123) {
    sharedTmp[0] = array[0];
  }
}

__global__ void lds_to_memory_x4_kernel(__int128_t *array) {
  __shared__ __int128_t sharedTmp[256];
  int threadId = blockIdx.x * blockDim.x * 4 + threadIdx.x % 64;
  array[threadId] = sharedTmp[threadIdx.x];
  array[threadId + 1] = sharedTmp[threadIdx.x];
  array[threadId + 2] = sharedTmp[threadIdx.x];
  array[threadId + 2] = sharedTmp[threadIdx.x];
  // avoid compiler optimizations
  if (array[0] == 123) {
    sharedTmp[0] = array[0];
  }
}
