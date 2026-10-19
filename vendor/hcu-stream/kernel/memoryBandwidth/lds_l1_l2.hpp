#pragma once
#include "hip/hip_runtime.h"
using float4_type = __attribute__((__vector_size__(4 * sizeof(float)))) float;
typedef float float4_ __attribute__((ext_vector_type(4)));
__global__ void l1_read_kernel(uint64_t *startClk, uint64_t *stopClk,
                               int *inputArray, int *outputArray,
                               int repeat_times) {

  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int *addrIdx0 = inputArray + idx * 4;
  // int *addrIdx1 = outputArray + idx * 4;
  // registers to avoid compiler optimization
  uint4 data0;
  uint4 data1;

  // populate l1 cache
  // load from memory
  asm volatile(
      // "buffer_wbinvl1 \n"
      "global_load_dwordx4 %0, %1, off \n"
      "s_waitcnt vmcnt(0) \n"
      : "={v[24:27]}"(data0)
      : "v"(addrIdx0)
      : "memory");

  // start timing
  uint64_t start = 0, stop = 0;
  asm volatile("s_barrier \n"
               "s_memtime %0 \n"
               "s_waitcnt lgkmcnt(0) \n"
               : "=s"(start)::"memory");

  for (int i = 0; i < repeat_times / 63; ++i) {
#pragma unroll
    for (int j = 0; j < 63; ++j) {
      asm volatile("global_load_dwordx4 %0, %1, off \n"
                   : "={v[20:23]}"(data1)
                   : "v"(addrIdx0)
                   : "memory");
    }
    asm volatile("s_barrier");
  }
  asm volatile("s_waitcnt vmcnt(0) \n" ::);

  // stop timing
  asm volatile("s_barrier \n"
               "s_memtime %0 \n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=s"(stop)::"memory");

  startClk[idx] = start;
  stopClk[idx] = stop;
}

__global__ void l2_read_kernel(uint64_t *startClk, uint64_t *stopClk,
                                float4_type *inputArray, int repeat_times) {
  uint32_t uid = blockIdx.x * blockDim.x + threadIdx.x;
  float4_type *inputArray_addrIdx = inputArray + uid;
  float4_type data;

  // warm up l2 cache
  asm volatile("global_load_dwordx4 %0, %1, off glc\n"
               "s_waitcnt vmcnt(0)\n"
               : "=v"(data)
               : "v"(inputArray_addrIdx)
               : "memory");

  // start timing
  uint64_t start, stop = 0;
  asm volatile("s_barrier \n"
               "s_memtime %0 \n"
               "s_waitcnt lgkmcnt(0) \n"
               : "=s"(start)::"memory");

  for (int i = 0; i < repeat_times; ++i) {
    asm volatile("global_load_dwordx4 %0, %1, off glc\n"
                 : "={v[24:27]}"(data)
                 : "v"(inputArray_addrIdx)
                 : "memory");
    asm volatile("s_barrier");
  }
  asm volatile("s_waitcnt vmcnt(0) \n" :::);

  // stop timing
  asm volatile("s_barrier \n"
               "s_memtime %0 \n"
               "s_waitcnt lgkmcnt(0) \n"
               : "=s"(stop)::"memory");

  startClk[uid] = start;
  stopClk[uid] = stop;
}

__global__ void l2_write_kernel(uint64_t *startClk, uint64_t *stopClk,
                                float4_type *dsink, int repeat_times) {
  // block and thread index
  uint32_t tidX = threadIdx.x, tidY = threadIdx.y;
  uint32_t uid =
      blockIdx.x * blockDim.x * blockDim.y + tidY * blockDim.x + tidX;
  float4_type *outputArray_addrIdx = dsink + uid;

  // warm up l2 cache
  float4_type result;
  asm volatile("global_load_dwordx4 %0, %1 off glc\n"
               "s_waitcnt vmcnt(0)\n"
               : "=v"(result)
               : "v"(dsink + uid)
               : "memory");

  // value to write
  float4_type offset;
  offset[0] = uid * 16;
  offset[1] = uid * 16 + 4;
  offset[2] = uid * 16 + 8;
  offset[3] = uid * 16 + 12;

  // start timing
  uint64_t start, stop = 0;
  asm volatile("s_barrier \n"
               "s_memtime %0\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=s"(start)
               :);

  for (int i = 0; i < repeat_times; ++i) {
    asm volatile("global_store_dwordx4 %0, %1 off glc\n"
                 :
                 : "v"(outputArray_addrIdx), "v"(offset)
                 : "memory");
    asm volatile("s_barrier");
  }
  asm volatile("s_waitcnt vmcnt(0) \n" :::);

  // stop timing
  asm volatile("s_barrier \n"
               "s_memtime %0\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=s"(stop)
               :);

  // store the result
  startClk[uid] = start;
  stopClk[uid] = stop;
}

__global__ void lds_read_kernel(uint64_t *startClk, uint64_t *stopClk,
                                /*int *outArray,*/ int repeat_times) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int offset =
      threadIdx.x / 64 * 1024 + threadIdx.x % 64 * 4; // avoid lds bank conflict
  int ldsOffset = offset * 4;

  int val0, val1, val2, val3;
  val0 = val1 = val2 = val3 = threadIdx.x;
  int4 result;
  asm volatile("v_mov_b32 %0, %4 \n"
               "v_mov_b32 %1, %5 \n"
               "v_mov_b32 %2, %6 \n"
               "v_mov_b32 %3, %7 \n"
               : "={v20}"(result.x), "={v21}"(result.y), "={v22}"(result.z),
                 "={v23}"(result.w)
               : "v"(val0), "v"(val1), "v"(val2), "v"(val3)
               :);

  asm volatile(
      // "s_mov_b32 m0, 0x0 \n"
      "ds_write_b128 %0, v[20:23] \n"
      "ds_write_b128 %0, v[20:23] offset:0x400 \n"
      "ds_write_b128 %0, v[20:23] offset:0x800 \n"
      "ds_write_b128 %0, v[20:23] offset:0xc00 \n"
      "s_waitcnt lgkmcnt(0) \n"
      :
      : "v"(ldsOffset));

  uint64_t start = 0, stop = 0;
  int4 data0, data1, data2, data3;

  // start timing
  asm volatile("s_barrier \n"
               "s_memtime %0 \n"
               "s_waitcnt lgkmcnt(0) \n"
               : "=s"(start)::"memory");

  for (int i = 0; i < repeat_times; ++i) {
    asm volatile("ds_read_b128 %0, %4 \n"
                 "ds_read_b128 %1, %4 offset:0x400 \n"
                 "ds_read_b128 %2, %4 offset:0x800 \n"
                 "ds_read_b128 %3, %4 offset:0xc00 \n"
                 "s_waitcnt lgkmcnt(0) \n"
                 : "={v[20:23]}"(data0), "={v[24:27]}"(data1),
                   "={v[28:31]}"(data2), "={v[32:35]}"(data3)
                 : "v"(ldsOffset));
  }

  // stop timing
  asm volatile("s_barrier \n"
               "s_memtime %0 \n"
               "s_waitcnt lgkmcnt(0) \n"
               : "=s"(stop)::"memory");

  // write time and data back to memory
  startClk[idx] = start;
  stopClk[idx] = stop;
}

__global__ void lds_add_u32_kernel(uint64_t *startClk, uint64_t *stopClk,
  /*int *outArray,*/ int repeat_times) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = threadIdx.x / 64 * 1024 + threadIdx.x % 64; // avoid lds bank conflict
  int ldsOffset = offset * 4;
  int addvalue = 1;
  int val0, val1, val2, val3;
  val0 = val1 = val2 = val3 = threadIdx.x;
  int4 result;
  asm volatile("v_mov_b32 %0, %4 \n"
  "v_mov_b32 %1, %5 \n"
  "v_mov_b32 %2, %6 \n"
  "v_mov_b32 %3, %7 \n"
  : "={v20}"(result.x), "={v21}"(result.y), "={v22}"(result.z),
  "={v23}"(result.w)
  : "v"(val0), "v"(val1), "v"(val2), "v"(val3)
  :);

  // asm volatile(
  // // "s_mov_b32 m0, 0x0 \n"
  // "ds_write_b128 %0, v[20:23] \n"
  // "ds_write_b128 %0, v[20:23] offset:0x400 \n"
  // "ds_write_b128 %0, v[20:23] offset:0x800 \n"
  // "ds_write_b128 %0, v[20:23] offset:0xc00 \n"
  // "s_waitcnt lgkmcnt(0) \n"
  // :
  // : "v"(ldsOffset));

  uint64_t start = 0, stop = 0;
  uint data[16];

  // start timing
  asm volatile("s_barrier \n"
  "s_memtime %0 \n"
  "s_waitcnt lgkmcnt(0) \n"
  : "=s"(start)::"memory");

  for (int i = 0; i < repeat_times; ++i) {
  asm volatile(
  "ds_add_rtn_u32 %0, %16, %17\n"
  "ds_add_rtn_u32 %1, %16, %17 offset:0x100\n"
  "ds_add_rtn_u32 %2, %16, %17 offset:0x200\n"
  "ds_add_rtn_u32 %3, %16, %17 offset:0x300\n"
  "ds_add_rtn_u32 %4, %16, %17 offset:0x400\n"
  "ds_add_rtn_u32 %5, %16, %17 offset:0x500\n"
  "ds_add_rtn_u32 %6, %16, %17 offset:0x600\n"
  "ds_add_rtn_u32 %7, %16, %17 offset:0x700\n"
  "ds_add_rtn_u32 %8, %16, %17 offset:0x800\n"
  "ds_add_rtn_u32 %9, %16, %17 offset:0x900\n"
  "ds_add_rtn_u32 %10, %16, %17 offset:0xa00\n"
  "ds_add_rtn_u32 %11, %16, %17 offset:0xb00\n"
  "ds_add_rtn_u32 %12, %16, %17 offset:0xc00\n"
  "ds_add_rtn_u32 %13, %16, %17 offset:0xd00\n"
  "ds_add_rtn_u32 %14, %16, %17 offset:0xe00\n"
  "ds_add_rtn_u32 %15, %16, %17 offset:0xf00\n"
  "s_waitcnt lgkmcnt(0) \n"
  : "={v20}"(data[0]), "={v21}"(data[1]), "={v22}"(data[2]), "={v23}"(data[3]),
  "={v24}"(data[4]), "={v25}"(data[5]),"={v26}"(data[6]), "={v27}"(data[7]),
  "={v28}"(data[8]), "={v29}"(data[9]),"={v30}"(data[10]), "={v31}"(data[11]),
  "={v32}"(data[12]), "={v23}"(data[13]),"={v34}"(data[14]), "={v35}"(data[15])
  : "v"(ldsOffset), "v"(addvalue));
  }

  // stop timing
  asm volatile("s_barrier \n"
  "s_memtime %0 \n"
  "s_waitcnt lgkmcnt(0) \n"
  : "=s"(stop)::"memory");

  // write time and data back to memory
  startClk[idx] = start;
  stopClk[idx] = stop;
}


// __global__ void ds_add_u32kernel(int* dst, int iters) {
//     __shared__ int smem[LDS_TOTAL_SIZE / sizeof(int)];
//     int acc = 0;
//     for (int idx = 0; idx < FIX_ITERS; idx++) {
//         for (int i = 0; i < iters; i ++) {
//             // acc += smem[i * WARP_SIZE + threadIdx.x];
//           acc += atomicAdd(smem + (i * 64 + threadIdx.x), 1);
//         }
//     }
//     if (acc == 123456) {
//         dst[threadIdx.x] = acc;
//     }
// }


__global__ void lds_write_write(uint64_t *startClk, uint64_t *stopClk,
                                /*int *outArray,*/ int repeat_times) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = (threadIdx.x / 64) * 64 * 4 * 4 +
               threadIdx.x % 64 * 4; // avoid lds bank conflict
  int ldsOffset = offset * 4;
  uint64_t start, stop;

  int val0, val1, val2, val3;
  val0 = val1 = val2 = val3 = threadIdx.x;
  int4 result;
  asm volatile("v_mov_b32 %0, %4 \n"
               "v_mov_b32 %1, %5 \n"
               "v_mov_b32 %2, %6 \n"
               "v_mov_b32 %3, %7 \n"
               : "={v20}"(result.x), "={v21}"(result.y), "={v22}"(result.z),
                 "={v23}"(result.w)
               : "v"(val0), "v"(val1), "v"(val2), "v"(val3)
               :);

  // start timing
  asm volatile("s_barrier \n"
               "s_memtime %0 \n"
               "s_waitcnt lgkmcnt(0) \n"
               : "=s"(start)::"memory");

  for (int i = 0; i < repeat_times; ++i) {
    asm volatile("ds_write_b128 %0, v[20:23] \n"
                 "ds_write_b128 %0, v[20:23] offset:0x400 \n"
                 "ds_write_b128 %0, v[20:23] offset:0x800 \n"
                 "ds_write_b128 %0, v[20:23] offset:0xc00 \n"
                 "s_waitcnt lgkmcnt(0) \n"
                 "s_barrier \n"
                 :
                 : "v"(ldsOffset));
  }

  // stop timing
  asm volatile("s_barrier \n"
               "s_memtime %0 \n"
               "s_waitcnt lgkmcnt(0) \n"
               : "=s"(stop)::"memory");

  // write time and data back to memory
  startClk[idx] = start;
  stopClk[idx] = stop;
}

#if HIP_ARCH >= 928
__global__ void lds_b32_matrix_read_kernel(float *dst, int iters,
                                           const int fix_iters) {
  int row = threadIdx.x / 8;
  int col = threadIdx.x % 8;
  int offset = (row * 32 + col * 4) * sizeof(float);
  constexpr int step = 32 * 8 * sizeof(float);
  float4_ value;
  float acc = 0;
  for (int idx = 0; idx < fix_iters; idx++) {
    for (int i = 0; i < iters; i++) {
      asm volatile("ds_read_m32x8_b32 %0, %1 offset:0\n\t"
                   : "+v"(value), "+v"(offset));
      offset += step;
      acc += value.x;
    }
  }
  if (acc == 123456) {
    dst[threadIdx.x] = acc;
  }
}

__global__ void lds_b16_matrix_read_kernel(float *dst, int iters,
                                           const int fix_iters) {
  int row = threadIdx.x / 4;
  int col = threadIdx.x % 4;
  int offset = (row * 32 + col * 8) * sizeof(_Float16);
  constexpr int step = 32 * 16 * sizeof(_Float16);
  float4_ value;
  float acc = 0;
  for (int idx = 0; idx < fix_iters; idx++) {
    for (int i = 0; i < iters; i++) {
      asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t"
                   : "+v"(value), "+v"(offset));
      offset += step;
      acc += value.x;
    }
  }
  if (acc == 123456) {
    dst[threadIdx.x] = acc;
  }
}

__global__ void lds_b16_alt_matrix_read_kernel(float *dst, int iters,
                                               const int fix_iters) {
  int row = threadIdx.x / 4;
  int col = threadIdx.x % 4;
  int offset = (row * 32 + col * 8) * sizeof(_Float16);
  constexpr int step = 32 * 16 * sizeof(_Float16);
  float4_ value;
  float acc = 0;
  for (int idx = 0; idx < fix_iters; idx++) {
    for (int i = 0; i < iters; i++) {
      asm volatile("ds_read_m32x16_b16_alt %0, %1 offset:0\n\t"
                   : "+v"(value), "+v"(offset));
      offset += step;
      acc += value.x;
    }
  }
  if (acc == 123456) {
    dst[threadIdx.x] = acc;
  }
}

__global__ void lds_b8_matrix_read_kernel(float *dst, int iters,
                                          const int fix_iters) {
  int row = threadIdx.x / 2;
  int col = threadIdx.x % 2;
  int offset = (row * 32 + col * 16) * sizeof(char);
  constexpr int step = 32 * 32 * sizeof(char);
  float4_ value;
  float acc = 0;
  for (int idx = 0; idx < fix_iters; idx++) {
    for (int i = 0; i < iters; i++) {
      asm volatile("ds_read_m32x32_b8 %0, %1 offset:0\n\t"
                   : "+v"(value), "+v"(offset));
      offset += step;
      acc += value.x;
    }
  }
  if (acc == 123456) {
    dst[threadIdx.x] = acc;
  }
}
#endif

template <typename T>
__global__ void lds_read4x_kernel(T* dst, int iters, const int fix_iters, const int warp_size) {
    extern __shared__ T smem[];
    T acc = 0;
    for (int idx = 0; idx < fix_iters; idx++) {
        for (int i = 0; i < iters; i ++) {
            acc += smem[i * warp_size + threadIdx.x];
        }
    }
    if (acc == 123456) {
        dst[threadIdx.x] = acc;
    }
}

template <typename T>
__global__ void lds_write4x_kernel(T* dst, int iters, const int fix_iters, const int warp_size) {
    extern __shared__ T smem[];
    for (int idx = 0; idx < fix_iters; idx++) {
        for (int i = 0; i < iters; i ++) {
            smem[i * warp_size + threadIdx.x] = threadIdx.x;
        }
    }
    if (iters == 123456) {
        dst[threadIdx.x] = smem[threadIdx.x];
    }
}
