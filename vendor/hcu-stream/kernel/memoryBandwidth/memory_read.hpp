#pragma once
#include "hip/hip_runtime.h"

//-----------------------------------------------------------------------------
//-------------------------- global_load_dwordx1 ------------------------------
//-----------------------------------------------------------------------------
__global__ void memory_readx1_x1_kernel(int *array) {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  int *addr = array + threadId;
  int data;

  asm volatile("global_load_dword %0, %1, off \n"
               : "={v20}"(data)
               : "v"(addr)
               :);
}
__global__ void memory_readx1_x4_kernel(int *array) {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  int *addr = array + threadId / 64 * 64 * 4 + threadIdx.x % 64;
  int data0, data1, data2, data3;

  asm volatile("global_load_dword %0, %4, off \n"
               "global_load_dword %1, %4, off offset:0x100 \n"
               "global_load_dword %2, %4, off offset:0x200 \n"
               "global_load_dword %3, %4, off offset:0x300 \n"
               : "={v20}"(data0), "={v21}"(data1), "={v22}"(data2),
                 "={v23}"(data3)
               : "v"(addr)
               :);
}
__global__ void memory_readx1_x8192_kernel(int *array) {
  unsigned int waveId = (blockIdx.x * blockDim.x + threadIdx.x) / 64;
  unsigned int threadId = threadIdx.x % 64;
  int data[8192];
  int *addr_0;
  // #pragma unroll
  for (unsigned int i = 0; i < 8192; i++) {
    addr_0 = array + waveId * 64 + threadId + i * 128 * 64 * 4;
    asm volatile("global_load_dword %0, %1, off\n"
                 : "={v20}"(data[i])
                 : "v"(addr_0)
                 :);
  }
}
__global__ void memory_readx1_x8192_80cu_kernel(int *array) {
  unsigned int waveId = (blockIdx.x * blockDim.x + threadIdx.x) / 64;
  unsigned int threadId = threadIdx.x % 64;
  int data[8192];
  int *addr_0;
  // #pragma unroll
  for (unsigned int i = 0; i < 8192; i++) {
    addr_0 = array + waveId * 64 + threadId + i * 80 * 64 * 4;
    asm volatile("global_load_dword %0, %1, off\n"
                 : "={v20}"(data[i])
                 : "v"(addr_0)
                 :);
  }
}

//-----------------------------------------------------------------------------
//-------------------------- global_load_dwordx2 ------------------------------
//-----------------------------------------------------------------------------
__global__ void memory_readx2_x1_kernel(__int64_t *array) {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  __int64_t *waveAddr = array + threadId;
  __int64_t data;
  __int64_t *addr_0 = waveAddr;

  asm volatile("global_load_dwordx2 %0, %1, off \n"
               : "={v[20:21]}"(data)
               : "v"(addr_0)
               :);
}
__global__ void memory_readx2_x2_kernel(__int64_t *array, int wave) {
  int threadId = blockIdx.x * blockDim.x * 2 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  __int64_t *waveAddr = array + waveId * 512 / 8 + threadId;
  __int64_t data[2];
  __int64_t *addr_0 = waveAddr + 64 * 0;
  __int64_t *addr_1 = waveAddr + 64 * wave * 1;

  asm volatile("global_load_dwordx2 %0, %1, off \n"
               : "={v[20:21]}"(data[0])
               : "v"(addr_0)
               :);
  asm volatile("global_load_dwordx2 %0, %1, off\n"
               : "={v[20:21]}"(data[1])
               : "v"(addr_1)
               :);
}


__global__ void memory_readx2_x4_kernel(__int64_t *array) {
  int threadId = blockIdx.x * blockDim.x * 4 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  __int64_t *waveAddr = array + waveId * 4 * 512 / 8 + threadId;
  __int64_t data[4];
  __int64_t *addr_0 = waveAddr + 64 * 0;

  asm volatile("global_load_dwordx2 %0, %1, off \n"
               : "={v[20:21]}"(data[0])
               : "v"(addr_0)
               :);
  asm volatile("global_load_dwordx2 %0, %1, off offset:0x200\n"
               : "={v[20:21]}"(data[1])
               : "v"(addr_0)
               :);

  asm volatile("global_load_dwordx2 %0, %1, off offset:0x400\n"
               : "={v[20:21]}"(data[2])
               : "v"(addr_0)
               :);
  asm volatile("global_load_dwordx2 %0, %1, off offset:0x600\n"
               : "={v[20:21]}"(data[3])
               : "v"(addr_0)
               :);
}

__global__ void memory_readx2_x4096_kernel(__int64_t *array) {
  unsigned int waveId = (blockIdx.x * blockDim.x + threadIdx.x) / 64;
  unsigned int threadId = threadIdx.x % 64;
  __int64_t data[4096];
  __int64_t *addr_0;
  // #pragma unroll
  // unsigned long array_size = 128 * 1024 * 1024;
  for (unsigned int i = 0; i < 4096; i++) {
    addr_0 = array + waveId * 64 + threadId + i * 128 * 64 * 4;
    asm volatile("global_load_dwordx2 %0, %1, off\n"
                 : "={v[20:21]}"(data[i])
                 : "v"(addr_0)
                 :);
  }
  // for (unsigned int i = 2048; i < 4096; i++) {
  //   addr_0 = array + waveId * 64 + threadId + i * 128 * 64 * 4;
  //   asm volatile(
  //   "global_load_dwordx2 %0, %1, off\n"
  //   : "=v"(data[i])
  //   : "v"(addr_0)
  //   :
  //   );
  // }
}
__global__ void memory_readx2_x4096_80cu_kernel(__int64_t *array) {
  unsigned int waveId = (blockIdx.x * blockDim.x + threadIdx.x) / 64;
  unsigned int threadId = threadIdx.x % 64;
  __int64_t data[4096];
  __int64_t *addr_0;
  for (unsigned int i = 0; i < 4096; i++) {
    addr_0 = array + waveId * 64 + threadId + i * 80 * 64 * 4;
    asm volatile("global_load_dwordx2 %0, %1, off\n"
                 : "={v[20:21]}"(data[i])
                 : "v"(addr_0)
                 :);
  }
}

//-----------------------------------------------------------------------------
//-------------------------- global_load_dwordx4 ------------------------------
//-----------------------------------------------------------------------------
__global__ void memory_readx4_x1_kernel(__int128_t *array) {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  __int128_t *addr = array + threadId;
  __int128_t data;

  asm volatile("global_load_dwordx4 %0, %1, off \n"
               : "={v[20:23]}"(data)
               : "v"(addr)
               :);
}
__global__ void memory_readx4_x2_kernel(__int128_t *array, int wave) {
  int threadId = blockIdx.x * blockDim.x * 2 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  __int128_t *waveAddr = array + waveId * 1024 / 16 + threadId;
  __int128_t data[2];
  __int128_t *addr_0 = waveAddr + 64 * 0;
  __int128_t *addr_1 = waveAddr + 64 * wave * 1;

  asm volatile("global_load_dwordx4 %0, %1, off\n"
               : "=v"(data[0])
               : "v"(addr_0)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off\n"
               : "=v"(data[1])
               : "v"(addr_1)
               :);
}
// __global__ void memory_readx4_x4_kernel(__int128_t *array) {
//   int threadId = blockIdx.x * blockDim.x * 4 + threadIdx.x % 64;
//   int waveId = threadIdx.x / 64;
//   __int128_t *waveAddr = array + waveId * 4 * 1024 / 16 + threadId;
//   __int128_t data[4];
//   __int128_t *addr_0 = waveAddr + 64 * 0;

//   asm volatile("global_load_dwordx4 %0, %1, off offset:0x0\n"
//                : "=v"(data[0])
//                : "v"(addr_0)
//                :);
//   asm volatile("global_load_dwordx4 %0, %1, off offset:0x400\n"
//                : "=v"(data[1])
//                : "v"(addr_0)
//                :);
//   asm volatile("global_load_dwordx4 %0, %1, off offset:0x800\n"
//                : "=v"(data[2])
//                : "v"(addr_0)
//                :);
//   asm volatile("global_load_dwordx4 %0, %1, off offset:0xc00\n"
//                : "=v"(data[3])
//                : "v"(addr_0)
//                :);
// }

__global__ void memory_readx4_x4_kernel(__int128_t *array) {
  int threadId = blockIdx.x * blockDim.x * 4 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  __int128_t *waveAddr = array + waveId * 1024/16 + threadId;
  __int128_t data[4];
  __int128_t *addr_0  = waveAddr + 64 * 0;
  __int128_t *addr_1  = waveAddr + 64 * 4;
  __int128_t *addr_2  = waveAddr + 64 * 8;
  __int128_t *addr_3  = waveAddr + 64 * 12;

  asm volatile(
    "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[0])
    : "v"(addr_0)
    :
  );
  asm volatile(
  "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[1])
    : "v"(addr_1)
    :
    );
  asm volatile(
    "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[2])
    : "v"(addr_2)
    :
  );
  asm volatile(
    "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[3])
    : "v"(addr_3)
    :
  );

}
// wave_continue
__global__ void memory_readx4_x8_wave_continue_kernel(__int128_t *array) {
  int threadId = blockIdx.x * blockDim.x * 8 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  __int128_t *waveAddr = array + waveId * 1024/16 + threadId;
  __int128_t data[8];
  __int128_t *addr_0  = waveAddr + 64 * 0;
  __int128_t *addr_1  = waveAddr + 64 * 4;
  __int128_t *addr_2  = waveAddr + 64 * 8;
  __int128_t *addr_3  = waveAddr + 64 * 12;
  __int128_t *addr_4  = waveAddr + 64 * 16;
  __int128_t *addr_5  = waveAddr + 64 * 20;
  __int128_t *addr_6  = waveAddr + 64 * 24;
  __int128_t *addr_7  = waveAddr + 64 * 28;

  asm volatile(
    "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[0])
    : "v"(addr_0)
    :
  );
  asm volatile(
  "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[1])
    : "v"(addr_1)
    :
    );
  asm volatile(
    "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[2])
    : "v"(addr_2)
    :
  );
  asm volatile(
    "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[3])
    : "v"(addr_3)
    :
  );
  asm volatile(
    "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[4])
    : "v"(addr_4)
    :
  );
  asm volatile(
  "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[5])
    : "v"(addr_5)
    :
    );
  asm volatile(
    "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[6])
    : "v"(addr_6)
    :
  );
  asm volatile(
    "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[7])
    : "v"(addr_7)
    :
  );

}

__global__ void memory_readx4_x8_wave_continue_8wave_kernel(__int128_t *array) {
  int threadId = blockIdx.x * blockDim.x * 8 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  __int128_t *waveAddr = array + waveId * 1024/16 + threadId;
  __int128_t data[8];
  __int128_t *addr_0  = waveAddr + 64 * 0;
  __int128_t *addr_1  = waveAddr + 64 * 8;
  __int128_t *addr_2  = waveAddr + 64 * 16;
  __int128_t *addr_3  = waveAddr + 64 * 24;
  __int128_t *addr_4  = waveAddr + 64 * 32;
  __int128_t *addr_5  = waveAddr + 64 * 40;
  __int128_t *addr_6  = waveAddr + 64 * 48;
  __int128_t *addr_7  = waveAddr + 64 * 56;

  asm volatile(
    "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[0])
    : "v"(addr_0)
    :
  );
  asm volatile(
  "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[1])
    : "v"(addr_1)
    :
    );
  asm volatile(
    "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[2])
    : "v"(addr_2)
    :
  );
  asm volatile(
    "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[3])
    : "v"(addr_3)
    :
  );
  asm volatile(
    "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[4])
    : "v"(addr_4)
    :
  );
  asm volatile(
  "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[5])
    : "v"(addr_5)
    :
    );
  asm volatile(
    "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[6])
    : "v"(addr_6)
    :
  );
  asm volatile(
    "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[7])
    : "v"(addr_7)
    :
  );

}

__global__ void memory_readx4_x8_wave_continue_16wave_kernel(__int128_t *array) {
  int threadId = blockIdx.x * blockDim.x * 8 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  __int128_t *waveAddr = array + waveId * 1024/16 + threadId;
  __int128_t data[8];
  __int128_t *addr_0  = waveAddr + 64 * 0;
  __int128_t *addr_1  = waveAddr + 64 * 16;
  __int128_t *addr_2  = waveAddr + 64 * 32;
  __int128_t *addr_3  = waveAddr + 64 * 48;
  __int128_t *addr_4  = waveAddr + 64 * 64;
  __int128_t *addr_5  = waveAddr + 64 * 80;
  __int128_t *addr_6  = waveAddr + 64 * 96;
  __int128_t *addr_7  = waveAddr + 64 * 112;

  asm volatile(
    "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[0])
    : "v"(addr_0)
    :
  );
  asm volatile(
  "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[1])
    : "v"(addr_1)
    :
    );
  asm volatile(
    "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[2])
    : "v"(addr_2)
    :
  );
  asm volatile(
    "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[3])
    : "v"(addr_3)
    :
  );
  asm volatile(
    "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[4])
    : "v"(addr_4)
    :
  );
  asm volatile(
  "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[5])
    : "v"(addr_5)
    :
    );
  asm volatile(
    "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[6])
    : "v"(addr_6)
    :
  );
  asm volatile(
    "global_load_dwordx4 %0, %1, off offset:0x0\n"
    : "=v"(data[7])
    : "v"(addr_7)
    :
  );

}

// inst continue
__global__ void memory_readx4_x8_inst_continue_kernel(__int128_t *array) {
  int threadId = blockIdx.x * blockDim.x * 8 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  __int128_t *waveAddr = array + waveId * 8 * 1024 / 16 + threadId;
  __int128_t data[8];
  __int128_t *addr_0 = waveAddr + 64 * 0;

  asm volatile("global_load_dwordx4 %0, %1, off offset:0x0\n"
               : "=v"(data[0])
               : "v"(addr_0)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x400\n"
               : "=v"(data[1])
               : "v"(addr_0)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x800\n"
               : "=v"(data[2])
               : "v"(addr_0)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0xc00\n"
               : "=v"(data[3])
               : "v"(addr_0)
               :);
  __int128_t *addr_4 = waveAddr + 64 * 4;
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x0\n"
               : "=v"(data[4])
               : "v"(addr_4)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x400\n"
               : "=v"(data[5])
               : "v"(addr_4)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x800\n"
               : "=v"(data[6])
               : "v"(addr_4)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0xc00\n"
               : "=v"(data[7])
               : "v"(addr_4)
               :);
}

__global__ void memory_readx4_x2048_kernel(__int128_t *array) {
  unsigned int waveId = (blockIdx.x * blockDim.x + threadIdx.x) / 64;
  unsigned int threadId = threadIdx.x % 64;
  __int128_t data[2048];
  __int128_t *addr_0;
  // #pragma unroll
  for (unsigned int i = 0; i < 2048; i++) {
    // unsigned int index = waveId * 64 + threadId + i * (128 * 64 * 4);
    // if(array[index] == 100) {
    //   array[0] = 1;
    // }
    addr_0 = array + waveId * 64 + threadId + i * 128 * 64 * 4;
    // unsigned long index = (waveId * 1024 + threadId * 16 + i * 512 * 1024) /
    // 16; printf("%d,waveID:%d, threadID:%d: %lx\n", i, waveId, threadId,
    // index); addr_0  = array + (waveId * 1024 + threadId * 16 + i * 512 *
    // 1024) / 16; printf("%d,waveID:%d, threadID:%d: %lx\n", i, waveId,
    // threadId, addr_0);
    asm volatile("global_load_dwordx4 %0, %1, off\n"
                 : "=v"(data[i])
                 : "v"(addr_0)
                 :);
  }
}
__global__ void memory_readx4_x2048_80cu_kernel(__int128_t *array) {
  unsigned int waveId = (blockIdx.x * blockDim.x + threadIdx.x) / 64;
  unsigned int threadId = threadIdx.x % 64;
  __int128_t data[2048];
  __int128_t *addr_0;
  // #pragma unroll
  for (unsigned int i = 0; i < 2048; i++) {
    addr_0 = array + waveId * 64 + threadId + i * 80 * 64 * 4;
    asm volatile("global_load_dwordx4 %0, %1, off\n"
                 : "=v"(data[i])
                 : "v"(addr_0)
                 :);
  }
}

__global__ void memory_readx4_x16_kernel(__int128_t *array) {
  int threadId = blockIdx.x * blockDim.x * 16 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  __int128_t *waveAddr = array + waveId * 16 * 1024 / 16 + threadId;
  __int128_t data[16];
  __int128_t *addr_0 = waveAddr + 64 * 0;

  asm volatile("global_load_dwordx4 %0, %1, off offset:0x0\n"
               : "=v"(data[0])
               : "v"(addr_0)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x400\n"
               : "=v"(data[1])
               : "v"(addr_0)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x800\n"
               : "=v"(data[2])
               : "v"(addr_0)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0xc00\n"
               : "=v"(data[3])
               : "v"(addr_0)
               :);
  __int128_t *addr_4 = waveAddr + 64 * 4;
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x0\n"
               : "=v"(data[4])
               : "v"(addr_4)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x400\n"
               : "=v"(data[5])
               : "v"(addr_4)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x800\n"
               : "=v"(data[6])
               : "v"(addr_4)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0xc00\n"
               : "=v"(data[7])
               : "v"(addr_4)
               :);
    __int128_t *addr_8 = waveAddr + 64 * 8;
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x0\n"
               : "=v"(data[8])
               : "v"(addr_8)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x400\n"
               : "=v"(data[9])
               : "v"(addr_8)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x800\n"
               : "=v"(data[10])
               : "v"(addr_8)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0xc00\n"
               : "=v"(data[11])
               : "v"(addr_8)
               :);
  __int128_t *addr_12 = waveAddr + 64 * 12;
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x0\n"
               : "=v"(data[12])
               : "v"(addr_12)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x400\n"
               : "=v"(data[13])
               : "v"(addr_12)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x800\n"
               : "=v"(data[14])
               : "v"(addr_12)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0xc00\n"
               : "=v"(data[15])
               : "v"(addr_12)
               :);
}

__global__ void __launch_bounds__(256, 1)
    memory_readx4_x64_kernel(__int128_t *array) {
  int threadId = blockIdx.x * blockDim.x * 64 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  __int128_t *waveAddr = array + waveId * 64 * 1024 / 16 + threadId;
  __int128_t data[64];
  __int128_t *addr_0 = waveAddr + 64 * 0;
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x0\n"
               : "=v"(data[0])
               : "v"(addr_0)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x400\n"
               : "=v"(data[1])
               : "v"(addr_0)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x800\n"
               : "=v"(data[2])
               : "v"(addr_0)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0xc00\n"
               : "=v"(data[3])
               : "v"(addr_0)
               :);
  __int128_t *addr_4 = waveAddr + 64 * 4;
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x0\n"
               : "=v"(data[4])
               : "v"(addr_4)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x400\n"
               : "=v"(data[5])
               : "v"(addr_4)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x800\n"
               : "=v"(data[6])
               : "v"(addr_4)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0xc00\n"
               : "=v"(data[7])
               : "v"(addr_4)
               :);
  __int128_t *addr_8 = waveAddr + 64 * 8;
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x0\n"
               : "=v"(data[8])
               : "v"(addr_8)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x400\n"
               : "=v"(data[9])
               : "v"(addr_8)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x800\n"
               : "=v"(data[10])
               : "v"(addr_8)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0xc00\n"
               : "=v"(data[11])
               : "v"(addr_8)
               :);
  __int128_t *addr_12 = waveAddr + 64 * 12;
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x0\n"
               : "=v"(data[12])
               : "v"(addr_12)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x400\n"
               : "=v"(data[13])
               : "v"(addr_12)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x800\n"
               : "=v"(data[14])
               : "v"(addr_12)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0xc00\n"
               : "=v"(data[15])
               : "v"(addr_12)
               :);
  __int128_t *addr_16 = waveAddr + 64 * 16;
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x0\n"
               : "=v"(data[16])
               : "v"(addr_16)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x400\n"
               : "=v"(data[17])
               : "v"(addr_16)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x800\n"
               : "=v"(data[18])
               : "v"(addr_16)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0xc00\n"
               : "=v"(data[19])
               : "v"(addr_16)
               :);
  __int128_t *addr_20 = waveAddr + 64 * 20;
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x0\n"
               : "=v"(data[20])
               : "v"(addr_20)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x400\n"
               : "=v"(data[21])
               : "v"(addr_20)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x800\n"
               : "=v"(data[22])
               : "v"(addr_20)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0xc00\n"
               : "=v"(data[23])
               : "v"(addr_20)
               :);
  __int128_t *addr_24 = waveAddr + 64 * 24;
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x0\n"
               : "=v"(data[24])
               : "v"(addr_24)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x400\n"
               : "=v"(data[25])
               : "v"(addr_24)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x800\n"
               : "=v"(data[26])
               : "v"(addr_24)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0xc00\n"
               : "=v"(data[27])
               : "v"(addr_24)
               :);
  __int128_t *addr_28 = waveAddr + 64 * 28;
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x0\n"
               : "=v"(data[28])
               : "v"(addr_28)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x400\n"
               : "=v"(data[29])
               : "v"(addr_28)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x800\n"
               : "=v"(data[30])
               : "v"(addr_28)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0xc00\n"
               : "=v"(data[31])
               : "v"(addr_28)
               :);
  __int128_t *addr_32 = waveAddr + 64 * 32;
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x0\n"
               : "=v"(data[32])
               : "v"(addr_32)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x400\n"
               : "=v"(data[33])
               : "v"(addr_32)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x800\n"
               : "=v"(data[34])
               : "v"(addr_32)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0xc00\n"
               : "=v"(data[35])
               : "v"(addr_32)
               :);
  __int128_t *addr_36 = waveAddr + 64 * 36;
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x0\n"
               : "=v"(data[36])
               : "v"(addr_36)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x400\n"
               : "=v"(data[37])
               : "v"(addr_36)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x800\n"
               : "=v"(data[38])
               : "v"(addr_36)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0xc00\n"
               : "=v"(data[39])
               : "v"(addr_36)
               :);
  __int128_t *addr_40 = waveAddr + 64 * 40;
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x0\n"
               : "=v"(data[40])
               : "v"(addr_40)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x400\n"
               : "=v"(data[41])
               : "v"(addr_40)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x800\n"
               : "=v"(data[42])
               : "v"(addr_40)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0xc00\n"
               : "=v"(data[43])
               : "v"(addr_40)
               :);
  __int128_t *addr_44 = waveAddr + 64 * 44;
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x0\n"
               : "=v"(data[44])
               : "v"(addr_44)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x400\n"
               : "=v"(data[45])
               : "v"(addr_44)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x800\n"
               : "=v"(data[46])
               : "v"(addr_44)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0xc00\n"
               : "=v"(data[47])
               : "v"(addr_44)
               :);
  __int128_t *addr_48 = waveAddr + 64 * 48;
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x0\n"
               : "=v"(data[48])
               : "v"(addr_48)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x400\n"
               : "=v"(data[49])
               : "v"(addr_48)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x800\n"
               : "=v"(data[50])
               : "v"(addr_48)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0xc00\n"
               : "=v"(data[51])
               : "v"(addr_48)
               :);
  __int128_t *addr_52 = waveAddr + 64 * 52;
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x0\n"
               : "=v"(data[52])
               : "v"(addr_52)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x400\n"
               : "=v"(data[53])
               : "v"(addr_52)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x800\n"
               : "=v"(data[54])
               : "v"(addr_52)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0xc00\n"
               : "=v"(data[55])
               : "v"(addr_52)
               :);
  __int128_t *addr_56 = waveAddr + 64 * 56;
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x0\n"
               : "=v"(data[56])
               : "v"(addr_56)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x400\n"
               : "=v"(data[57])
               : "v"(addr_56)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x800\n"
               : "=v"(data[58])
               : "v"(addr_56)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0xc00\n"
               : "=v"(data[59])
               : "v"(addr_56)
               :);
  __int128_t *addr_60 = waveAddr + 64 * 60;
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x0\n"
               : "=v"(data[60])
               : "v"(addr_60)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x400\n"
               : "=v"(data[61])
               : "v"(addr_60)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0x800\n"
               : "=v"(data[62])
               : "v"(addr_60)
               :);
  asm volatile("global_load_dwordx4 %0, %1, off offset:0xc00\n"
               : "=v"(data[63])
               : "v"(addr_60)
               :);
}


#if HIP_ARCH >= 936
// BPS x4_x8
// wave_continue
__global__ void memory_readx4_x8_wave_continue_bps_kernel(__int128_t *array) {
  int threadId = blockIdx.x * blockDim.x * 8 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  int v = threadIdx.x; // per thread 16Byte
  int ldsAddr0, ldsAddr1, ldsAddr2, ldsAddr3, ldsAddr4, ldsAddr5, ldsAddr6, ldsAddr7;
  int ldsAddrPerWave;
  __int128_t *waveAddr = array + waveId * 1024/16 + threadId;
  __int128_t data[8];
  __int128_t *addr_0  = waveAddr + 64 * 0;
  __int128_t *addr_1  = waveAddr + 64 * 4;
  __int128_t *addr_2  = waveAddr + 64 * 8;
  __int128_t *addr_3  = waveAddr + 64 * 12;
  __int128_t *addr_4  = waveAddr + 64 * 16;
  __int128_t *addr_5  = waveAddr + 64 * 20;
  __int128_t *addr_6  = waveAddr + 64 * 24;
  __int128_t *addr_7  = waveAddr + 64 * 28;
 

  asm volatile("v_readfirstlane_b32 %0,%9\t\n"
              "s_lshl_b32 m0, %0, 4\t\n"
              "v_lshlrev_b32_e32 %1, 4, %9\t\n"
              "v_add_u32 %2, 4096, %1\t\n"
              "v_add_u32 %3, 4096, %2\t\n"
              "v_add_u32 %4, 4096, %3\t\n"
              "v_add_u32 %5, 4096, %4\t\n"
              "v_add_u32 %6, 4096, %5\t\n"
              "v_add_u32 %7, 4096, %6\t\n"
              "v_add_u32 %8, 4096, %7\t\n"
    : "=s"(ldsAddrPerWave), "+v"(ldsAddr0), "+v"(ldsAddr1), "+v"(ldsAddr2), "+v"(ldsAddr3), "+v"(ldsAddr4), "+v"(ldsAddr5), "+v"(ldsAddr6), "+v"(ldsAddr7)
    : "v"(v)
    :);

  asm volatile(
    "global_load_dwordx4 %0, %8, off lds\t\n"
    "s_add_u32 m0, m0, 0x1000 \n"
    "s_nop 0 \n"
    "global_load_dwordx4 %1, %9, off lds\t\n"
    "s_add_u32 m0, m0, 0x1000 \n"
    "s_nop 0 \n"
    "global_load_dwordx4 %2, %10, off lds\t\n"
    "s_add_u32 m0, m0, 0x1000 \n"
    "s_nop 0 \n"
    "global_load_dwordx4 %3, %11, off lds\t\n"
    "s_add_u32 m0, m0, 0x1000 \n"
    "s_nop 0 \n"
    "global_load_dwordx4 %4, %12, off lds\t\n"
    "s_add_u32 m0, m0, 0x1000 \n"
    "s_nop 0 \n"
    "global_load_dwordx4 %5, %13, off lds\t\n"
    "s_add_u32 m0, m0, 0x1000 \n"
    "s_nop 0 \n"
    "global_load_dwordx4 %6, %14, off lds\t\n"
    "s_add_u32 m0, m0, 0x1000 \n"
    "s_nop 0 \n"
    "global_load_dwordx4 %7, %15, off lds\t\n"
    "s_waitcnt vmcnt(7)\t\n"
    "ds_read_b128 %0, %16\t\n"
    "ds_read_b128 %1, %16\t\n"
    "ds_read_b128 %2, %16\t\n"
    "ds_read_b128 %3, %16\t\n"
    "ds_read_b128 %4, %16\t\n"
    "ds_read_b128 %5, %16\t\n"
    "ds_read_b128 %6, %16\t\n"
    "ds_read_b128 %7, %16\t\n"
    : "=v"(data[0]), "=v"(data[1]), "=v"(data[2]), "=v"(data[3]), "=v"(data[4]), "=v"(data[5]), "=v"(data[6]), "=v"(data[7])
    : "v"(addr_0), "v"(addr_1), "v"(addr_2), "v"(addr_3), "v"(addr_4), "v"(addr_5), "v"(addr_6), "v"(addr_7),
      "v"(ldsAddr0), "v"(ldsAddr1), "v"(ldsAddr2), "v"(ldsAddr3), "v"(ldsAddr4), "v"(ldsAddr5), "v"(ldsAddr6), "v"(ldsAddr7)
    :
  );
}

__global__ void memory_readx4_x8_wave_continue_8wave_bps_kernel(__int128_t *array) {

  int threadId = blockIdx.x * blockDim.x * 8 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  int v = threadIdx.x; // per thread 16Byte
  int ldsAddr0, ldsAddr1, ldsAddr2, ldsAddr3, ldsAddr4, ldsAddr5, ldsAddr6, ldsAddr7;
  int ldsAddrPerWave;
  __int128_t *waveAddr = array + waveId * 1024/16 + threadId;
  __int128_t data[8];
  __int128_t *addr_0  = waveAddr + 64 * 0;
  __int128_t *addr_1  = waveAddr + 64 * 8;
  __int128_t *addr_2  = waveAddr + 64 * 16;
  __int128_t *addr_3  = waveAddr + 64 * 24;
  __int128_t *addr_4  = waveAddr + 64 * 32;
  __int128_t *addr_5  = waveAddr + 64 * 40;
  __int128_t *addr_6  = waveAddr + 64 * 48;
  __int128_t *addr_7  = waveAddr + 64 * 56;

  asm volatile("v_readfirstlane_b32 %0,%2\t\n"
    "s_lshl_b32 m0, %0, 4\t\n"
    "v_lshlrev_b32_e32 %1, 4, %2\t\n"
  : "=s"(ldsAddrPerWave), "+v"(ldsAddr0)
  : "v"(v)
  :);

  asm volatile(
    "global_load_dwordx4 %0, %8, off lds\t\n"
    "s_add_u32 m0, m0, 0x2000 \n"
    "s_nop 0 \n"
    "global_load_dwordx4 %1, %9, off lds\t\n"
    "s_add_u32 m0, m0, 0x2000 \n"
    "s_nop 0 \n"
    "global_load_dwordx4 %2, %10, off lds\t\n"
    "s_add_u32 m0, m0, 0x2000 \n"
    "s_nop 0 \n"
    "global_load_dwordx4 %3, %11, off lds\t\n"
    "s_add_u32 m0, m0, 0x2000 \n"
    "s_nop 0 \n"
    "global_load_dwordx4 %4, %12, off lds\t\n"
    "s_add_u32 m0, m0, 0x2000 \n"
    "s_nop 0 \n"
    "global_load_dwordx4 %5, %13, off lds\t\n"
    "s_add_u32 m0, m0, 0x2000 \n"
    "s_nop 0 \n"
    "global_load_dwordx4 %6, %14, off lds\t\n"
    "s_add_u32 m0, m0, 0x2000 \n"
    "s_nop 0 \n"
    "global_load_dwordx4 %7, %15, off lds\t\n"
    "s_waitcnt vmcnt(7)\t\n"
    "ds_read_b128 %0, %16\t\n"
    "ds_read_b128 %1, %16\t\n"
    "ds_read_b128 %2, %16\t\n"
    "ds_read_b128 %3, %16\t\n"
    "ds_read_b128 %4, %16\t\n"
    "ds_read_b128 %5, %16\t\n"
    "ds_read_b128 %6, %16\t\n"
    "ds_read_b128 %7, %16\t\n"
    : "=v"(data[0]), "=v"(data[1]), "=v"(data[2]), "=v"(data[3]), "=v"(data[4]), "=v"(data[5]), "=v"(data[6]), "=v"(data[7])
    : "v"(addr_0), "v"(addr_1), "v"(addr_2), "v"(addr_3), "v"(addr_4), "v"(addr_5), "v"(addr_6), "v"(addr_7),
    "v"(ldsAddr0)
    :
  );
}

__global__ void memory_readx4_x8_wave_continue_16wave_bps_kernel(__int128_t *array) {
  int threadId = blockIdx.x * blockDim.x * 8 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  int v = threadIdx.x; // per thread 16Byte
  int baseAddr;
  int ldsAddr0, ldsAddr1, ldsAddr2, ldsAddr3, ldsAddr4, ldsAddr5, ldsAddr6, ldsAddr7;
  int ldsAddrPerWave;
  __int128_t *waveAddr = array + waveId * 1024/16 + threadId;
  __int128_t data[8];
  __int128_t *addr_0  = waveAddr + 64 * 0;
  __int128_t *addr_1  = waveAddr + 64 * 16;
  __int128_t *addr_2  = waveAddr + 64 * 32;
  __int128_t *addr_3  = waveAddr + 64 * 48;
  __int128_t *addr_4  = waveAddr + 64 * 64;
  __int128_t *addr_5  = waveAddr + 64 * 80;
  __int128_t *addr_6  = waveAddr + 64 * 96;
  __int128_t *addr_7  = waveAddr + 64 * 112;

  asm volatile("v_readfirstlane_b32 %0,%3\t\n"
              "s_lshl_b32 m0, %0, 4\t\n"
              "s_lshl_b32 %2, %0, 4\t\n"
              "v_lshlrev_b32_e32 %1, 4, %3\t\n"
        : "=s"(ldsAddrPerWave), "+v"(ldsAddr0), "=s"(baseAddr)
        : "v"(v)
  :);

  asm volatile(
    "global_load_dwordx4 %0, %8, off lds\t\n"
    "s_add_u32 m0, m0, 0x4000 \n"
    "s_nop 0 \n"
    "global_load_dwordx4 %1, %9, off lds\t\n"
    "s_add_u32 m0, m0, 0x4000 \n"
    "s_nop 0 \n"
    "global_load_dwordx4 %2, %10, off lds\t\n"
    "s_add_u32 m0, m0, 0x4000 \n"
    "s_nop 0 \n"
    "global_load_dwordx4 %3, %11, off lds\t\n"
    "s_add_u32 m0, m0, 0x4000 \n"
    "s_nop 0 \n"
    "global_load_dwordx4 %4, %12, off lds\t\n"
    "s_add_u32 m0, m0, 0x4000 \n"
    "s_nop 0 \n"
    "global_load_dwordx4 %5, %13, off lds\t\n"
    "s_add_u32 m0, m0, 0x4000 \n"
    "s_nop 0 \n"
    "global_load_dwordx4 %6, %14, off lds\t\n"
    "s_add_u32 m0, m0, 0x4000 \n"
    "s_nop 0 \n"
    "global_load_dwordx4 %7, %15, off lds\t\n"
    "s_waitcnt vmcnt(7)\t\n"
    "ds_read_b128 %0, %16\t\n"
    "ds_read_b128 %1, %16\t\n"
    "ds_read_b128 %2, %16\t\n"
    "ds_read_b128 %3, %16\t\n"
    "ds_read_b128 %4, %16\t\n"
    "ds_read_b128 %5, %16\t\n"
    "ds_read_b128 %6, %16\t\n"
    "ds_read_b128 %7, %16\t\n"
    : "=v"(data[0]), "=v"(data[1]), "=v"(data[2]), "=v"(data[3]), "=v"(data[4]), "=v"(data[5]), "=v"(data[6]), "=v"(data[7])
    : "v"(addr_0), "v"(addr_1), "v"(addr_2), "v"(addr_3), "v"(addr_4), "v"(addr_5), "v"(addr_6), "v"(addr_7),
    "v"(ldsAddr0)
    :
  );
}
#endif