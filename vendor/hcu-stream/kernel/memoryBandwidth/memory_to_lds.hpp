#pragma once
#include "hip/hip_runtime.h"

__global__ void memory_to_ldsx1_kernel(int *array) {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  int *addr = array + (threadId / 64) * 64 * 4 + threadIdx.x % 64;
  int data0, data1, data2, data3;
  int lds_offset;

  asm volatile("v_readfirstlane_b32 %0, %1 \n"
               "s_mov_b32 m0, %0 \n"
               : "=s"(lds_offset)
               : "v"(threadIdx.x * 4 * 4)
               :);

  asm volatile("global_load_dword %0, %4, off lds\n"
               "global_load_dword %1, %4, off offset:0x100 lds\n"
               "global_load_dword %2, %4, off offset:0x200 lds\n"
               "global_load_dword %3, %4, off offset:0x300 lds\n"
               : "={v20}"(data0), "={v21}"(data1), "={v22}"(data2),
                 "={v23}"(data3)
               : "v"(addr)
               :);
}
#if HIP_ARCH >= 936
__global__ void memory_to_ldsx4_kernel(__int128_t *array) {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  __int128_t *addr = array + threadId;
  __int128_t data;
  int lds_offset;

  asm volatile("v_readfirstlane_b32 %0, %1 \n"
               "s_mov_b32 m0, %0 \n"
               : "=s"(lds_offset)
               : "v"(threadIdx.x * 4 * 4)
               :);

  asm volatile("global_load_dwordx4 %0, %1, off lds\n"
               : "={v[20:23]}"(data)
               : "v"(addr)
               :);
}



__global__ void memory_to_ldsx4_x4_kernel(__int128_t *array) {

  int threadId = blockIdx.x * blockDim.x * 4 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  __int128_t *waveAddr = array + waveId * 1024/16 + threadId;
  __int128_t data[4];
  __int128_t *addr_0  = waveAddr + 64 * 0;
  __int128_t *addr_1  = waveAddr + 64 * 4;
  __int128_t *addr_2  = waveAddr + 64 * 8;
  __int128_t *addr_3  = waveAddr + 64 * 12;
  int lds_offset;

  asm volatile("v_readfirstlane_b32 %0, %1 \n"
               "s_mov_b32 m0, %0 \n"
               : "=s"(lds_offset)
               : "v"(threadIdx.x * 4 * 4)
               :);

  asm volatile(
    "global_load_dwordx4 %0, %1, off lds \n"
    : "=v"(data[0])
    : "v"(addr_0)
    :
  );
  asm volatile(
  "global_load_dwordx4 %0, %1, off lds\n"
    : "=v"(data[1])
    : "v"(addr_1)
    :
    );
  asm volatile(
    "global_load_dwordx4 %0, %1, off lds\n"
    : "=v"(data[2])
    : "v"(addr_2)
    :
  );
  asm volatile(
    "global_load_dwordx4 %0, %1, off lds\n"
    : "=v"(data[3])
    : "v"(addr_3)
    :
  );
}
#endif