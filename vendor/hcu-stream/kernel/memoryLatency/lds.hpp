#pragma once
#include "hip/hip_runtime.h"
#include "params.h"

__global__ void latency_lds_read_kernel(uint64_t *out) {
  HIP_DYNAMIC_SHARED(int, sharedTmp);
  // init the lds
  sharedTmp[threadIdx.x] = threadIdx.x;
  __syncthreads();
  int ldsAddr = threadIdx.x * 4;
  uint64_t start = 0, stop = 0;
  int result;

  // start timing
  asm volatile("s_memtime %0 \n"
               "s_waitcnt lgkmcnt(0) \n"
               : "=s"(start)::"memory");

  for (int i = 0; i < lds_read_iter; ++i) {
    asm volatile("ds_read_b32 %0, %1 \n"
                 "s_waitcnt lgkmcnt(0) \n"
                 : "=v"(result)
                 : "v"(ldsAddr));
  }

  // stop timing
  asm volatile("s_memtime %0 \n"
               "s_waitcnt lgkmcnt(0) \n"
               : "=s"(stop)::"memory");

  out[threadIdx.x] = stop - start;
}

__global__ void latency_lds_write_kernel(uint64_t *out) {
  HIP_DYNAMIC_SHARED(int, sharedTmp);
  int value = threadIdx.x;
  int ldsAddr = threadIdx.x * 4;
  uint64_t start = 0, stop = 0;

  // start timing
  asm volatile("s_memtime %0 \n"
               "s_waitcnt lgkmcnt(0) \n"
               : "=s"(start)::"memory");

  for (int i = 0; i < 10000; ++i) {
    asm volatile("ds_write_b32 %0 ,%1\n"
                 "s_waitcnt lgkmcnt(0)\n"
                 :
                 : "v"(ldsAddr), "v"(value)
                 :);
  }

  // stop timing
  asm volatile("s_memtime %0 \n"
               "s_waitcnt lgkmcnt(0) \n"
               : "=s"(stop)::"memory");

  out[threadIdx.x] = stop - start;
}

__global__ void latency_lds_read_b16_kernel(uint64_t *out) {
  __shared__ uint16_t lds[LDS_TOTAL_SIZE];
  uint32_t uid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t start = 0, stop = 0;
  if(uid == 0){
      uint16_t value = 0;
      uint32_t offset = 0;

    // start timing
    asm volatile("s_memtime %0 \n"
                "s_waitcnt lgkmcnt(0) \n"
                : "=s"(start)::"memory");
    for(uint32_t i = 0; i < LDS_TOTAL_SIZE; ++i) {
        asm volatile("ds_read_u16 %0 %1\n"
                    "s_waitcnt lgkmcnt(0)\n"
                    : "=v" (value)
                    : "v" (offset));
        // offset += sizeof(uint16_t); // offset bytes
    }
    // stop timing
    asm volatile("s_memtime %0 \n"
                "s_waitcnt lgkmcnt(0) \n"
                : "=s"(stop)::"memory");

    out[threadIdx.x] = stop - start;
  }

}

__global__ void latency_lds_read_b32_kernel(uint64_t *out) {
  __shared__ uint32_t lds[LDS_TOTAL_SIZE];
  uint32_t uid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t start = 0, stop = 0;
  if(uid == 0){
      uint32_t value = 0;
      uint32_t offset = 0;

    // start timing
    asm volatile("s_memtime %0 \n"
                "s_waitcnt lgkmcnt(0) \n"
                : "=s"(start)::"memory");
    for(uint32_t i = 0; i < LDS_TOTAL_SIZE; ++i) {
      asm volatile("ds_read_b32 %0 %1\n"
                  "s_waitcnt lgkmcnt(0)\n"
                  : "=v" (value)
                  : "v" (offset));
      // offset += sizeof(uint32_t); // offset bytes
    }
    // stop timing
    asm volatile("s_memtime %0 \n"
                "s_waitcnt lgkmcnt(0) \n"
                : "=s"(stop)::"memory");

    out[threadIdx.x] = stop - start;
  }

}

__global__ void latency_lds_read_b64_kernel(uint64_t *out) {
  __shared__ uint64_t lds[LDS_TOTAL_SIZE];
  
  uint32_t uid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t start = 0, stop = 0;
  if(uid == 0){
    uint64_t value = 0;
    uint32_t offset = 0;

    // start timing
    asm volatile("s_memtime %0 \n"
               "s_waitcnt lgkmcnt(0) \n"
               : "=s"(start)::"memory");
    for(uint32_t i = 0; i < LDS_TOTAL_SIZE; ++i) {
      asm volatile("ds_read_b64 %0 %1\n"
                   "s_waitcnt lgkmcnt(0)\n"
                   : "=v"(value)
                   : "v"(offset));
      // offset += sizeof(uint64_t); // offset bytes
    }
    // stop timing
    asm volatile("s_memtime %0 \n"
                "s_waitcnt lgkmcnt(0) \n"
                : "=s"(stop)::"memory");

    out[threadIdx.x] = stop - start;
  }

}

__global__ void latency_lds_read_b128_kernel(uint64_t *out) {
  __shared__ __int128_t lds[LDS_TOTAL_SIZE];
  uint32_t uid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t start = 0, stop = 0;

  if (uid == 0) {
    __int128_t value = 0;
    uint32_t offset = 0;
    // start timing
    asm volatile("s_memtime %0 \n"
                  "s_waitcnt lgkmcnt(0) \n"
                  : "=s"(start)::"memory");

    for(uint32_t i = 0; i < LDS_TOTAL_SIZE; ++i) {
      asm volatile("ds_read_b128 %0 %1\n"
                   "s_waitcnt lgkmcnt(0)\n"
                   : "=v"(value)
                   : "v"(offset));
      // offset += sizeof(__int128_t); // offset bytes
    }

    // stop timing
    asm volatile("s_memtime %0 \n"
                  "s_waitcnt lgkmcnt(0) \n"
                  : "=s"(stop)::"memory");

    out[threadIdx.x] = stop - start;
  }
}