#pragma once
#include <chrono>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>

using half_t = _Float16;
#if HIP_ARCH >= 936
__global__ void paTest(int *array, int height, int weight, int pad) {
  int data = 0;
  HIP_DYNAMIC_SHARED(uint16_t, sharedTmp);
  int lds_offset;
  int waveId = (blockIdx.x * blockDim.x + threadIdx.x) / 64;
  // wave中线程行和列序
  int thread_row = (threadIdx.x % 64) / 16;
  int thread_col = (threadIdx.x % 64) % 16;
  // 分块的行和列序
  int block_row = waveId % (height / 4);
  int block_col = waveId / (height / 4);

  int *wave_addr = array + block_row * (weight + pad) * 4 + block_col * 16;
  int *thread_addr = wave_addr + thread_row * (weight + pad) + thread_col;

  asm volatile("v_readfirstlane_b32 %0, %1 \n"
               "s_mov_b32 m0, %0 \n"
               : "=s"(lds_offset)
               : "v"(threadIdx.x * 4 * 4)
               :);
  // uint *addr = (uint *)thread_addr;
  asm volatile("global_load_dword %0, %1, off lds \n"
               "s_waitcnt vmcnt(0) \n"
               :
               : "v"(data), "v"(thread_addr)
               :);
}
#endif