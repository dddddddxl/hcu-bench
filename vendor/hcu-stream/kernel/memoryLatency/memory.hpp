
#pragma once
#include "hip/hip_runtime.h"
#include "params.h"

__global__ void latency_memory_read_kernel(uint64_t *out,
                                           uint32_t *inputArray) {
  uint32_t *addrIndex = inputArray + threadIdx.x;
  uint64_t start = 0, stop = 0;
  uint32_t result;

  asm volatile("s_memtime %0 \n"
               "s_waitcnt lgkmcnt(0) \n"
               : "=s"(start)::"memory");

  asm volatile("global_load_dword %0, %1, off \n"
               "s_waitcnt vmcnt(0) \n"
               : "={v21}"(result)
               : "v"(addrIndex));

  asm volatile("s_memtime %0 \n"
               "s_waitcnt lgkmcnt(0) \n"
               : "=s"(stop)::"memory");

  out[threadIdx.x] = stop - start;
}

__global__ void latency_memory_write_kernel(uint64_t *out,
                                            uint32_t *outputArray) {
  uint64_t start = 0, stop = 0;
  uint32_t *dst = outputArray + threadIdx.x;
  uint32_t result;

  asm volatile("global_load_dword %0, %1, off \n"
               "s_waitcnt vmcnt(0) \n"
               : "={v10}"(result)
               : "v"(dst));

  asm volatile("s_memtime %0\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "={s[6:7]}"(start)
               :);

  asm volatile("global_store_dword %0, v10, off\n"
               "s_waitcnt vmcnt(0) \n"
               :
               : "v"(dst));

  asm volatile("s_memtime %0\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "={s[8:9]}"(stop)
               :);

  out[threadIdx.x] = stop - start;
}

__global__ void latency_memory_to_ldsx1_kernel(uint64_t *out, uint32_t *Array) {
  HIP_DYNAMIC_SHARED(int, sharedTmp);
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = thread_id * 4;
  int ldsAddrPerWave;
  // init the lds
  sharedTmp[threadIdx.x] = -1;
  __syncthreads();
  uint64_t start = 0, stop = 0;

  asm volatile("v_readfirstlane_b32 %0,%1 \n"
               "s_mov_b32 m0, %0 \n"
               : "=s"(ldsAddrPerWave)
               : "v"(offset)
               :);

  asm volatile("s_memtime %0\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "={s[6:7]}"(start)
               :);

  asm volatile("global_load_dword %0, %1, %2, offset:0, lds \n"
               "s_waitcnt vmcnt(0) \n"
               :
               : "v"(offset), "v"(offset), "s"(Array)
               :);

  asm volatile("s_memtime %0\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "={s[8:9]}"(stop)
               :);

  out[threadIdx.x] = stop - start;
}
#if HIP_ARCH >= 936
__global__ void latency_memory_to_ldsx4_kernel(uint64_t *out, uint32_t *Array) {
  typedef int int4_ __attribute__((ext_vector_type(4)));
  HIP_DYNAMIC_SHARED(int, sharedTmp);
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = thread_id * 4 * 4;
  int ldsAddrPerWave;
  int4_ vdst;
  // init the lds
  sharedTmp[threadIdx.x] = -1;
  __syncthreads();
  uint64_t start = 0, stop = 0;

  asm volatile("v_readfirstlane_b32 %0,%1 \n"
               "s_mov_b32 m0, %0 \n"
               : "=s"(ldsAddrPerWave)
               : "v"(offset)
               :);

  asm volatile("s_memtime %0\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "={s[6:7]}"(start)
               :);

  asm volatile("global_load_dwordx4 %0, %1, %2, offset:0, lds \n"
               "s_waitcnt vmcnt(0) \n"
               :
               : "v"(vdst), "v"(offset), "s"(Array)
               :);

  asm volatile("s_memtime %0\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "={s[8:9]}"(stop)
               :);

  out[threadIdx.x] = stop - start;
}
#endif
__global__ void latency_lds_to_memory_kernel(uint64_t *out, uint32_t *Array) {
  HIP_DYNAMIC_SHARED(int, sharedTmp);
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = thread_id * 4;
  int ldsAddrPerWave;
  // init the lds
  sharedTmp[threadIdx.x] = 123;
  __syncthreads();
  uint64_t start = 0, stop = 0;

  uint2 bufDes0; // just a placeholder
  uint bufDes1;
  uint bufDes2;
  uint dword2 = 0x80000000;
  uint dword3 = 0x00800000; // see "bandwidth_lds_direct_to_memory" for details

  uint stride = 0x4;
  uint64_t dword0_dword1 = (uint64_t)stride << 48 | (uint64_t)Array;

  asm volatile("s_mov_b64 %0, %3 \n"
               "s_mov_b32 %1, %4 \n"
               "s_mov_b32 %2, %5 \n"
               : "={s[20:21]}"(bufDes0), "={s22}"(bufDes1), "={s23}"(bufDes2)
               : "s"(dword0_dword1), "s"(dword2), "s"(dword3)
               :);

  asm volatile("v_readfirstlane_b32 %0,%1 \n"
               "s_mov_b32 m0, %0 \n"
               : "=s"(ldsAddrPerWave)
               : "v"(offset)
               :);

  asm volatile("s_memtime %0\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "={s[6:7]}"(start)
               :);
  asm volatile("buffer_store_lds_dword s[20:23], 0, offset:0, lds \n"
               "s_waitcnt vmcnt(0) \n"
               :
               :);

  asm volatile("s_memtime %0\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "={s[8:9]}"(stop)
               :);

  out[threadIdx.x] = stop - start;
}

__global__ void latency_global_load_dwordx1_kernel(uint32_t* src, uint64_t *out) {
  uint32_t uid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t start = 0, stop = 0;
  if(uid == 0){
    uint32_t value = 0;
    uint64_t addr = (uint64_t)src;
    uint32_t step = WIDTH * sizeof(uint32_t);

    asm volatile("s_memtime %0 \n"
      "s_waitcnt lgkmcnt(0) \n"
      : "=s"(start)::"memory");
    for(uint32_t i = 0; i < HEIGHT; ++i) {
      asm volatile("global_load_dword %0 %1 off glc slc\n"
                    "s_waitcnt vmcnt(0)\n"
                    : "=v" (value)
                    : "v" (addr));
      addr += step; // offset bytes
    }
    asm volatile("s_memtime %0 \n"
      "s_waitcnt lgkmcnt(0) \n"
      : "=s"(stop)::"memory");
    out[threadIdx.x] = stop - start;
  }
}

__global__ void latency_global_load_dwordx2_kernel(uint64_t* src, uint64_t *out) {
  uint32_t uid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t start = 0, stop = 0;
  if(uid == 0){
    uint64_t value = 0;
    uint64_t addr = (uint64_t)src;
    uint32_t step = WIDTH * sizeof(uint64_t);

    asm volatile("s_memtime %0 \n"
      "s_waitcnt lgkmcnt(0) \n"
      : "=s"(start)::"memory");
    for(uint32_t i = 0; i < HEIGHT; ++i) {
      asm volatile("global_load_dwordx2 %0 %1 off glc slc\n"
                  "s_waitcnt vmcnt(0)\n"
                  : "=v" (value)
                  : "v" (addr));
      addr += step; // offset bytes
    }
    asm volatile("s_memtime %0 \n"
      "s_waitcnt lgkmcnt(0) \n"
      : "=s"(stop)::"memory");
    out[threadIdx.x] = stop - start;
  }
}

__global__ void latency_global_load_dwordx4_kernel(__int128_t* src, uint64_t *out) {
  uint32_t uid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t start = 0, stop = 0;
  if(uid == 0){
    __int128_t value = 0;
    uint64_t addr = (uint64_t)src;
    uint32_t step = WIDTH * sizeof(__int128_t);

    asm volatile("s_memtime %0 \n"
      "s_waitcnt lgkmcnt(0) \n"
      : "=s"(start)::"memory");
    for(uint32_t i = 0; i < HEIGHT; ++i) {
      asm volatile("global_load_dwordx4 %0 %1 off glc slc\n"
                    "s_waitcnt vmcnt(0)\n"
                    : "=v" (value)
                    : "v" (addr));
      addr += step; // offset bytes
    }
    asm volatile("s_memtime %0 \n"
      "s_waitcnt lgkmcnt(0) \n"
      : "=s"(stop)::"memory");
    out[threadIdx.x] = stop - start;
  }
}
