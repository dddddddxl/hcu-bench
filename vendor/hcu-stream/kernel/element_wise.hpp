#pragma once
#include "elementwiseHelper/common.h"
#include "elementwiseHelper/offsetcalculator.hpp"
#include <chrono>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define FLOAT2(value) (reinterpret_cast<float2 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define HALF4(value) (reinterpret_cast<half4 *>(&(value))[0])
// address continuity
__global__ void element_wise_con_kernel(_Float16 *array, int size) {
  int threadId = blockIdx.x * blockDim.x * 4 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  if ((waveId * 64 + threadId + 64 *12) > size) {
    return;
  }
  _Float16 *waveAddr = array + waveId * 1024/16 + threadId;
  _Float16 data[4];
  _Float16 *addr_0  = waveAddr + 64 * 0;
  _Float16 *addr_1  = waveAddr + 64 * 4;
  _Float16 *addr_2  = waveAddr + 64 * 8;
  _Float16 *addr_3  = waveAddr + 64 * 12;

  asm volatile(
    "global_load_ushort %0, %1, off offset:0x0\n"
    : "=v"(data[0])
    : "v"(addr_0)
    :
  );
  asm volatile(
  "global_load_ushort %0, %1, off offset:0x0\n"
    : "=v"(data[1])
    : "v"(addr_1)
    :
    );
  asm volatile(
    "global_load_ushort %0, %1, off offset:0x0\n"
    : "=v"(data[2])
    : "v"(addr_2)
    :
  );
  asm volatile(
    "global_load_ushort %0, %1, off offset:0x0\n"
    : "=v"(data[3])
    : "v"(addr_3)
    :
  );
  data[0] = _Float16(threadIdx.x);
  data[1] = _Float16(threadIdx.x + 1);
  data[2] = _Float16(threadIdx.x + 2);
  data[3] = _Float16(threadIdx.x + 3);


  asm volatile(
    "global_store_short %0, %1, off offset:0x0\n"
    : "=v"(addr_0)
    : "v"(data[0])
    :
  );
  asm volatile(
  "global_store_short %0, %1, off offset:0x0\n"
    : "=v"(addr_1)
    : "v"(data[1])
    :
    );
  asm volatile(
    "global_store_short %0, %1, off offset:0x0\n"
    : "=v"(addr_2)
    : "v"(data[2])
    :
  );
  asm volatile(
    "global_store_short %0, %1, off offset:0x0\n"
    : "=v"(addr_3)
    : "v"(data[3])
    :
  );

}

// -------------------------------------- FP32
// --------------------------------------
__global__ void elementwise_add_f32_kernel(float *a, float *b, float *c,
                                           int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    c[idx] = a[idx] + b[idx];
}

__global__ void elementwise_add_f32x2_kernel(float *a, float *b, float *c,
                                             int N) {
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    float2 reg_a = FLOAT2(a[idx]);
    float2 reg_b = FLOAT2(b[idx]);
    float2 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    FLOAT2(c[idx]) = reg_c;
  }
}
__global__ void elementwise_add_f32x4_kernel(float *a, float *b, float *c,
                                             int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_b = FLOAT4(b[idx]);
    float4 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    FLOAT4(c[idx]) = reg_c;
  }
}

// -------------------------------------- FP16
// --------------------------------------
__global__ void elementwise_add_f16_kernel(half *a, half *b, half *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    c[idx] = __hadd(a[idx], b[idx]);
}

__global__ void elementwise_add_f16x2_kernel(half *a, half *b, half *c, int N) {
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    half2 reg_a = HALF2(a[idx]);
    half2 reg_b = HALF2(b[idx]);
    HALF2(c[idx]) = __hadd2(reg_a, reg_b);
  }
}

__global__ void elementwise_add_f16x4_kernel(half *a, half *b, half *c, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    half4 reg_a = HALF4(a[idx]);
    half4 reg_b = HALF4(b[idx]);
    half4 reg_c;
    reg_c.x = __hadd2(reg_a.x, reg_b.x);
    reg_c.y = __hadd2(reg_a.y, reg_b.y);
    HALF4(c[idx]) = reg_c;
  }
}

template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};

template <typename DT, typename offset_caculator>
__global__ void elementwise_add(DT *input1, DT *input2, DT *output,
                                int element_cnt, int loop_cnt,
                                offset_caculator off_calc) {
  int block_off = blockIdx.x * blockDim.x * loop_cnt;
  // printf("Thread[%d]: blockIdx.x =%d\n", (int)threadIdx.x, (int)blockIdx.x);

  for (int i = 0; i < loop_cnt; i++) {
    int loop_off = block_off + blockDim.x * i;

    int idx = threadIdx.x + loop_off;
    if (idx < element_cnt) {
      int offset = off_calc.get(idx);
      // if (blockIdx.x < 10) {
      //   printf("Thread[%d]: idx = %d, offset = %d\n", (int)threadIdx.x, idx,
      //          offset);
      // }
      DT *in1 = input1 + offset;
      DT *in2 = input2 + offset;
      DT *out = output + offset;

      *out = (*in1) + (*in2);
    }
  }
}

template <int vect_size, typename DT, typename offset_caculator>
__global__ void elementwise_add_vect(DT *input1, DT *input2, DT *output,
                                     int element_cnt, int loop_cnt,
                                     offset_caculator off_calc) {
  using vec_t = aligned_vector<DT, vect_size>;

  int block_off = blockIdx.x * blockDim.x * loop_cnt;

  int loop = loop_cnt / vect_size;

  for (int i = 0; i < loop; i++) {
    int loop_off = block_off + blockDim.x * vect_size * i;

    int idx = threadIdx.x * vect_size + loop_off;
    if (idx < element_cnt) {
      int offset = off_calc.get(idx);
      // if (blockIdx.x == 16) {
      //     printf("Thread[%d]: idx = %d, offset = %d\n", (int)threadIdx.x,
      //     idx, offset);
      // }
      vec_t *in1 = reinterpret_cast<vec_t *>(input1 + offset);
      vec_t *in2 = reinterpret_cast<vec_t *>(input2 + offset);
      vec_t *out = reinterpret_cast<vec_t *>(output + offset);
      vec_t v;
      for (int j = 0; j < vect_size; j++) {
        v.val[j] = in1[0].val[j] + in2[0].val[j];
      }

      out[0] = v;
    }
  }
}
