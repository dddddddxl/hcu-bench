#pragma once
#include "hip/hip_runtime.h"

__global__ void latency_l1_read_kernel(uint64_t *out, uint32_t *inputArray) {
  uint32_t idx = threadIdx.x;
  uint32_t *addrIndex = inputArray + idx;
  uint64_t start = 0, stop = 0;
  uint32_t result;

  // warm up
  asm volatile("global_load_dword %0, %1, off \n"
               "s_waitcnt vmcnt(0) \n"
               : "={v20}"(result)
               : "v"(addrIndex));

  asm volatile("s_memtime %0 \n"
               "s_waitcnt lgkmcnt(0) \n"
               : "=s"(start)::"memory");

  for (int i = 0; i < 10000; ++i) {
    asm volatile("global_load_dword %0, %1, off \n"
                 "s_waitcnt vmcnt(0) \n"
                 : "={v21}"(result)
                 : "v"(addrIndex));
  }

  asm volatile("s_memtime %0 \n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=s"(stop)::"memory");

  out[idx] = stop - start;
}

__global__ void latency_l2_read_kernel(uint64_t *out, uint32_t *inputArray) {
  uint32_t idx = threadIdx.x;
  uint32_t *addrIndex = inputArray + idx;
  uint64_t start = 0, stop = 0;
  uint32_t result;

  // warm up
  asm volatile("global_load_dword %0, %1, off glc\n"
               "s_waitcnt vmcnt(0) \n"
               : "={v20}"(result)
               : "v"(addrIndex));

  asm volatile("s_memtime %0 \n"
               "s_waitcnt lgkmcnt(0) \n"
               : "=s"(start)::"memory");

  for (int i = 0; i < 10000; ++i) {
    asm volatile("global_load_dword %0, %1, off glc \n"
                 "s_waitcnt vmcnt(0) \n"
                 : "={v21}"(result)
                 : "v"(addrIndex));
  }

  asm volatile("s_memtime %0 \n"
               "s_waitcnt lgkmcnt(0) \n"
               : "=s"(stop)::"memory");

  out[idx] = stop - start;
}

__global__ void latency_l2_write_kernel(uint64_t *out, uint32_t *outputArray) {
  uint64_t start = 0, stop = 0;
  uint32_t *dst = outputArray + threadIdx.x;
  uint32_t result;

  // warm up
  asm volatile("global_load_dword %0, %1, off \n"
               "s_waitcnt vmcnt(0) \n"
               : "={v10}"(result)
               : "v"(dst));

  asm volatile("s_memtime %0\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "={s[6:7]}"(start)
               :);

  for (int i = 0; i < 10000; ++i) {
    asm volatile("global_store_dword %0, v10, off glc\n"
                 "s_waitcnt vmcnt(0) \n"
                 :
                 : "v"(dst));
  }

  asm volatile("s_memtime %0\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "={s[8:9]}"(stop)
               :);

  out[threadIdx.x] = stop - start;
}
