#pragma once
#include "hip/hip_runtime.h"

__global__ void matrix_copy_kernel(const __int128_t *a, __int128_t *b,
                                   const uint32_t width) {
  const int i = (blockDim.y * blockIdx.y + threadIdx.y) * width +
                blockDim.x * blockIdx.x + threadIdx.x;
  b[i] = a[i];
}

__global__ void matrix_write_kernel(__int128_t *a, __int128_t initA,
                                    const uint32_t width) {
  const int i = (blockDim.y * blockIdx.y + threadIdx.y) * width +
                blockDim.x * blockIdx.x + threadIdx.x;
  a[i] = initA;
}

__global__ void matrix_read_kernel(const __int128_t *a, __int128_t *b,
                                   const uint32_t width) {
  const int i = (blockDim.y * blockIdx.y + threadIdx.y) * width +
                blockDim.x * blockIdx.x + threadIdx.x;
  __int128_t value = a[i];
  // 此处并不会发生实际写出，仅为了不让编译器优化掉读指令
  if (value == 123456) {
    b[i] = value;
  }
}

// inst continue
__global__ void matrix_read_x4_kernel(const __int128_t *a, __int128_t *b,
                                   const uint32_t width) {
  const int i = (blockDim.y * blockIdx.y + threadIdx.y) * width +
                blockDim.x * blockIdx.x * 4 + threadIdx.x * 4;
  __int128_t value = a[i];
  __int128_t value1 = a[i+1];
  __int128_t value2 = a[i+2];
  __int128_t value3 = a[i+3];
  // 此处并不会发生实际写出，仅为了不让编译器优化掉读指令
  if (value == 123456) {
    b[i] = value;
  }
  if (value1 == 123456) {
    b[i+1] = value1;
  }
  if (value2 == 123456) {
    b[i+2] = value2;
  }
  if (value3 == 123456) {
    b[i+3] = value3;
  }
}
__global__ void matrix_copy_x4_kernel(const __int128_t *a, __int128_t *b,
    const uint32_t width) {
  const int i = (blockDim.y * blockIdx.y + threadIdx.y) * width +
  blockDim.x * blockIdx.x * 4 + threadIdx.x * 4;
  b[i] = a[i];
  b[i + 1] = a[i + 1];
  b[i + 2] = a[i + 2];
  b[i + 3] = a[i + 3];
}
__global__ void matrix_write_x4_kernel(__int128_t *a, __int128_t initA,
   const uint32_t width) {
  const int i = (blockDim.y * blockIdx.y + threadIdx.y) * width +
  blockDim.x * blockIdx.x * 4 + threadIdx.x * 4;
  a[i] = initA;
  a[i + 1] = initA;
  a[i + 2] = initA;
  a[i + 3] = initA;
}

__global__ void matrix_readx4_x2_kernel(const __int128_t *a, __int128_t *b,
  const uint32_t width) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = (blockDim.x * blockIdx.x) * 2;
  const int i = row * width + col  + threadIdx.x;
  __int128_t value = a[i];
  __int128_t value1 = a[i + 64];

  // 此处并不会发生实际写出，仅为了不让编译器优化掉读指令
  if (value == 123456 || value1 == 123456) {
    b[i] = value;
    b[i + 64] = value1;
  }
}

// thread continue
__global__ void matrix_readx4_x4_kernel(const __int128_t *a, __int128_t *b,
  const uint32_t width) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = (blockDim.x * blockIdx.x) * 4;
  const int i = row * width + col  + threadIdx.x;
  __int128_t value = a[i];
  __int128_t value1 = a[i + 64];
  __int128_t value2 = a[i + 128];
  __int128_t value3 = a[i + 192];
  // 此处并不会发生实际写出，仅为了不让编译器优化掉读指令
  if (value == 123456 || value1 == 123456 || value2 == 123456 || value3 == 123456) {
    b[i] = value;
    b[i + 64] = value1;
    b[i + 128] = value2;
    b[i + 192] = value3;
  }
}

__global__ void matrix_writex4_x2_kernel(__int128_t *a, __int128_t initA,
  const uint32_t width) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x * 2;
  const int i = row * width + col  + threadIdx.x;
  a[i] = initA;
  a[i + 64] = initA;

}
__global__ void matrix_writex4_x4_kernel(__int128_t *a, __int128_t initA,
  const uint32_t width) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x * 4;
  const int i = row * width + col  + threadIdx.x;
  a[i] = initA;
  a[i + 64] = initA;
  a[i + 128] = initA;
  a[i + 192] = initA;

}
__global__ void matrix_copyx4_x4_kernel(const __int128_t *a, __int128_t *b,
  const uint32_t width) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x * 4;
  const int i = row * width + col  + threadIdx.x;
  b[i] = a[i];
  b[i + 64] = a[i + 64];
  b[i + 128] = a[i + 128];
  b[i + 192] = a[i + 192];

}

// design for anshi
// inst continue
__global__ void matrix_readx4_multi_kernel(const __int128_t *a, __int128_t *b,
  const uint32_t width, int inst_count) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = (blockDim.x * blockIdx.x) * inst_count;
  const int i = row * width + col  + threadIdx.x;
  // error: unsupported dynamic alloca, may be use templeate <int insst_count>
  __int128_t value = a[i];
  __int128_t value1 = a[i + 64];
  __int128_t value2 = a[i + 128];
  __int128_t value3 = a[i + 192];
  __int128_t value4 = a[i + 256];
  __int128_t value5 = a[i + 320];
  __int128_t value6 = a[i + 384];
  __int128_t value7 = a[i + 448];
  // 此处并不会发生实际写出，仅为了不让编译器优化掉读指令
  if (value == 123456 || value1 == 123456 || value2 == 123456 || value3 == 123456 ||
    value4 == 123456 || value5 == 123456 || value6 == 123456 || value7 == 123456 ) {
    b[i] = value;
    b[i + 64] = value1;
    b[i + 128] = value2;
    b[i + 192] = value3;
    b[i + 256] = value4;
    b[i + 320] = value5;
    b[i + 384] = value6;
    b[i + 448] = value7;
  }
}

__global__ void matrix_writex4_multi_kernel(__int128_t *a, __int128_t initA,
  const uint32_t width, int inst_count) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x * inst_count;
  const int i = row * width + col  + threadIdx.x;
  // for (int j = 0; j < inst_count; j++) {
  //   a[i + j * 64] = initA;
  // }
  a[i] = initA;
  a[i + 64] = initA;
  a[i + 128] = initA;
  a[i + 192] = initA;
  a[i + 256] = initA;
  a[i + 320] = initA;
  a[i + 384] = initA;
  a[i + 448] = initA;

}

__global__ void matrix_copyx4_multi_kernel(const __int128_t *a, __int128_t *b,
  const uint32_t width, int inst_count) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x * inst_count;
  const int i = row * width + col  + threadIdx.x;
  // for (int j = 0; j < inst_count; j++) {
  //   b[i + j * 64] = a[i + j * 64];
  // }
  b[i] = a[i];
  b[i + 64] = a[i + 64];
  b[i + 128] = a[i + 128];
  b[i + 192] = a[i + 192];
  b[i + 256] = a[i + 256];
  b[i + 320] = a[i + 320];
  b[i + 384] = a[i + 384];
  b[i + 448] = a[i + 448];
}
