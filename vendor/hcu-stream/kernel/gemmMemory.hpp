#pragma once
#include <chrono>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>

#if HIP_ARCH >= 936
__global__ void gemmTest(__int128_t *array, int height, int weight) {
  __int128_t data = 0;
  int waveId = (blockIdx.x * blockDim.x + threadIdx.x) / 64;
  // wave中线程行和列序
  int thread_row = (threadIdx.x % 64) / 16;
  int thread_col = (threadIdx.x % 64) % 16;
  // 分块的行和列序  s
  int block_row = waveId / (weight / 16);
  int block_col = waveId % (weight / 16);

  __int128_t *wave_addr = array + block_row * weight * 4 + block_col * 16;
  __int128_t *thread_addr = wave_addr + thread_row * weight + thread_col;
#if HIP_ARCH == 936
  asm volatile("global_load_dwordx4 %0, %1, off lds \n"
               "s_waitcnt vmcnt(0) \n"
               :
               : "v"(data), "v"(thread_addr)
               :);
#endif
}
#endif
