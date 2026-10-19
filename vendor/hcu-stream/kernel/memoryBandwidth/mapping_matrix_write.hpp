#pragma once
#include "hip/hip_runtime.h"

__global__ void mapping_matrix_write_kernel(__int128_t *a, __int128_t initA,
                                            const uint32_t width) {
  int WORK_GROUP_MAPPING = 8;
  uint64_t wgSerial = blockIdx.x + (blockIdx.y % WORK_GROUP_MAPPING) *
                                       gridDim.x; // within block
  uint32_t block = blockIdx.y / WORK_GROUP_MAPPING;
  uint32_t blockRemainder =
      (blockIdx.y < gridDim.y - (gridDim.y % WORK_GROUP_MAPPING))
          ? 8
          : gridDim.y % WORK_GROUP_MAPPING;
  int block_x = wgSerial / blockRemainder;
  int block_y = wgSerial % blockRemainder + block * WORK_GROUP_MAPPING;
  const int i = (blockDim.y * block_y + threadIdx.y) * width +
                block_x * blockDim.x + threadIdx.x;
  a[i] = initA;
}
__global__ void mapping_matrix_write_kernelx4_x4(__int128_t *a, __int128_t initA,
                                            const uint32_t width) {
  int WORK_GROUP_MAPPING = 8;
  uint64_t wgSerial = blockIdx.x + (blockIdx.y % WORK_GROUP_MAPPING) *
                                       gridDim.x; // within block
  uint32_t block = blockIdx.y / WORK_GROUP_MAPPING;
  uint32_t blockRemainder =
      (blockIdx.y < gridDim.y - (gridDim.y % WORK_GROUP_MAPPING))
          ? 8
          : gridDim.y % WORK_GROUP_MAPPING;
  int block_x = wgSerial / blockRemainder;
  int block_y = wgSerial % blockRemainder + block * WORK_GROUP_MAPPING;
  int row = blockDim.y * block_y + threadIdx.y;
  int col = (blockDim.x * block_x) * 4;
  const int i = row * width + col  + threadIdx.x;
  a[i] = initA;
  a[i + 64] = initA;
  a[i + 128] = initA;
  a[i + 192] = initA;
}

