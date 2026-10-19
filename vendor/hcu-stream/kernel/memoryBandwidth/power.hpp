#pragma once
#include "hip/hip_runtime.h"
#include <hiprand_kernel.h>
#define __private__ __attribute__((address_space(5)))

using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int32_t)))) int32_t;
using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int32_t)))) int32_t;
using int8x4_t = __attribute__((__vector_size__(4 * sizeof(uint8_t)))) int8_t;
#if HIP_ARCH == 936
extern "C" __global__ void powerTest_GC_HBM(int8_t *A, int8_t *B,
                                            int32_t *C, int32_t *D, __int128_t *array,
                                            int num, hiprandState_t* states, int cnt) {
  if (blockIdx.x < 80) {
        int block_id = blockIdx.y * gridDim.x + blockIdx.x;
    // int thread_id = block_id * blockDim.x + threadIdx.x;
    int row = threadIdx.x % 16;
    int col = threadIdx.x / 16;
    int offset;
    // uint64_t s_offset = blockIdx.x * 2 * 4;
    int32x2_t a, b;
    int32x4_t c;

    offset = row * 32 + col * 8;
    a[0] = reinterpret_cast<int32_t>(
        int8x4_t{A[offset], A[offset + 1], A[offset + 2], A[offset + 3]});
    a[1] = reinterpret_cast<int32_t>(
        int8x4_t{A[offset + 4], A[offset + 5], A[offset + 6], A[offset + 7]});
    b[0] = reinterpret_cast<int32_t>(
        int8x4_t{B[offset], B[offset + 1], B[offset + 2], B[offset + 3]});
    b[1] = reinterpret_cast<int32_t>(
        int8x4_t{B[offset + 4], B[offset + 5], B[offset + 6], B[offset + 7]});

    offset = row + col * 16;

#pragma unroll
    for (int i = 0; i < 4; i++)
      c[i] = C[offset + 64 * i];

    __builtin_amdgcn_sched_barrier(0);
    for (int k = 0; k < 10000; ++k){
    for (int i = 0; i<800000000; ++i)
      for (int j = 0; j < 6400; ++j)
        c = __builtin_hcu_mmac_i32_16x16x32_i8(a, b, c);
    }
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int i = 0; i < 4; i++)
      D[offset + 64 * i] = c[i];
  }
  
  else{
    asm volatile("s_setprio 1");
    int idx = (blockIdx.x - 80) * blockDim.x + threadIdx.x;
    int offset = threadIdx.x   % 64 * 16;
    hiprandState_t localState = states[idx];
    // __device__ __forceinline__   uint32x4_t prepare_for_buffer_load(__uint128_t* ptr) {
    int32x4_t res;
    *(uint64_t*)&res = reinterpret_cast<uint64_t>(array);
    res[2] = 0x80000000;
    res[3] = 0x00020000;
    // return res;
    // }
    __int128_t *array2 = array + num * 256 * 160;
    __int128_t base = (blockIdx.x - 80) * num * 4 * 1024;
    int32x4_t res2;
    *(uint64_t*)&res2 = reinterpret_cast<uint64_t>(array);
    res2[2] = 0x80000000;
    res2[3] = 0x00020000;
    int index = 0;
    for (int j = 0; j < 10000 ; j++) {
      int high = hiprand(&localState);
      int32x4_t data = {high, high, high, high};
      for (int i = 0; i < num; ++i) {
          index = i * 1024 + base;
          asm volatile("buffer_store_dwordx4 %0, %1, %2, %3, offen\n"
          :
          : "v"(data), "v"(offset), "{s[20:23]}"(res2), "s"(index)
          :);
      }
      for (int i = 0; i < num; ++i) {
          index = i * 1024 + base;
          asm volatile(
          "buffer_load_dwordx4 %0, %1, %2, %3 offen \n"
          : "={v[20:23]}"(data)
          : "v"(offset), "{s[20:23]}"(res2), "s"(index)
          :);
          asm volatile("buffer_store_dwordx4 %0, %1, %2, %3, offen\n"
          :
          : "v"(data), "v"(offset), "{s[24:27]}"(res), "s"(index)
          :);
      }
    }



    // // __device__ __forceinline__  uint32x4_t prepare_for_buffer_load(__uint128_t* ptr) {
    // int32x4_t res;
    // *(uint64_t*)&res = reinterpret_cast<uint64_t>(array);
    // res[2] = 0x80000000;
    // res[3] = 0x00020000;
    // // return res;
    // // }

    // for (int j = 0; j < 100; j++) {
    //   for (int i = 0; i < num; i++) {
    //     int high = hiprand(&localState);
    //     int32x4_t data = {high, 0, high, 9};
    //     asm volatile("buffer_store_dwordx4 %0, %1, %2, %3, offen\n"
    //     :
    //     : "v"(data), "v"(offset), "{s[20:23]}"(res), "s"(num * 1024)
    //     :);
    //   }
    // }

  }

}

__global__ void powerTest_HBM(__int128_t *array, int num, hiprandState_t *states) {
 	   // int waveId = threadIdx.x / 64;
	   // int idx = blockIdx.x * blockDim.x + threadIdx.x;
   
	   // hiprandState_t localState = states[idx];
   
	   // __int128_t array1[6520];
	   // __private__ __int128_t* array2 = (__private__ __int128_t*) array1;
	   // __int128_t base = blockIdx.x * blockDim.x * num + waveId * num * 64 + threadIdx.x % 64;
	   // for (int j = 0; j < 100; j++) {
	   //    int high = hiprand(&localState);
	   //    for (int i = 0; i < num; i++) {
	   //          array2[i] = high;
	   //    }
	   //    for (int i = 0; i < num; i++) {
	   //          // __int128_t index = i * 64;
	   //          __int128_t index = base + i * 64;
	   //          // array = array + index;
	   //          array[index] = array2[i];
	   //    }
	   // }
	   int idx = blockIdx.x * blockDim.x + threadIdx.x;
	   int offset = threadIdx.x   % 64 * 16;
	   hiprandState_t localState = states[idx];
	   // __device__ __forceinline__   uint32x4_t prepare_for_buffer_load(__uint128_t* ptr) {
	   int32x4_t res;
	   *(uint64_t*)&res = reinterpret_cast<uint64_t>(array);
	   res[2] = 0x80000000;
	   res[3] = 0x00020000;
	   // return res;
	   // }
	   __int128_t *array2 = array + num * 256 * 160;
	   __int128_t base = blockIdx.x * num * 4 * 1024;
	   int32x4_t res2;
	   *(uint64_t*)&res2 = reinterpret_cast<uint64_t>(array);
	   res2[2] = 0x80000000;
	   res2[3] = 0x00020000;
	   int index = 0;
	   for (int j = 0; j < 100; j++) {
	      int high = hiprand(&localState);
	      int32x4_t data = {high, high, high, high};
	      for   (int i = 0; i < num; ++i) {
	         index = i * 1024 + base;
	         asm volatile("buffer_store_dwordx4 %0, %1, %2, %3, offen\n"
	         :
	         : "v"(data), "v"(offset), "{s[20:23]}"(res2), "s"(index)
	         :);
	      }
	     for (int i = 0; i < 6520; ++i) {
	         index = i * 1024 + base;
	         asm volatile(
	         "buffer_load_dwordx4 %0, %1, %2, %3 offen \n"
	         : "={v[20:23]}"(data)
	         : "v"(offset), "{s[20:23]}"(res2), "s"(index)
	         :);
	         asm volatile("buffer_store_dwordx4 %0, %1, %2, %3, offen\n"
	         :
	         : "v"(data), "v"(offset), "{s[24:27]}"(res), "s"(index)
	         :);
	      }
     }
}


__global__ void powerTest_GC(int8_t *A, int8_t *B,
                                            int32_t *C, int32_t *D) {
    int block_id = blockIdx.y * gridDim.x + blockIdx.x;
    int thread_id = block_id * blockDim.x + threadIdx.x;
    int row = threadIdx.x % 16;
    int col = threadIdx.x / 16;
    int offset;
    uint64_t s_offset = blockIdx.x * 2 * 4;
    int32x2_t a, b;
    int32x4_t c;

    offset = row * 32 + col * 8;
    a[0] = reinterpret_cast<int32_t>(
        int8x4_t{A[offset], A[offset + 1], A[offset + 2], A[offset + 3]});
    a[1] = reinterpret_cast<int32_t>(
        int8x4_t{A[offset + 4], A[offset + 5], A[offset + 6], A[offset + 7]});
    b[0] = reinterpret_cast<int32_t>(
        int8x4_t{B[offset], B[offset + 1], B[offset + 2], B[offset + 3]});
    b[1] = reinterpret_cast<int32_t>(
        int8x4_t{B[offset + 4], B[offset + 5], B[offset + 6], B[offset + 7]});

    offset = row + col * 16;

#pragma unroll
    for (int i = 0; i < 4; i++)
      c[i] = C[offset + 64 * i];

    __builtin_amdgcn_sched_barrier(0);
    for (int i = 0; i < 80000000; ++i)
      for (int j = 0; j < 6400; ++j)
        c = __builtin_hcu_mmac_i32_16x16x32_i8(a, b, c);
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int i = 0; i < 4; i++)
      D[offset + 64 * i] = c[i];
}
#endif
