#pragma once
#include "hip/hip_runtime.h"

__global__ void memory_write_kernel(__int128_t *array) {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  __int128_t *addr = array + threadId;
  __int128_t data = 123456;

  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(addr), "v"(data)
               : "memory");
}

__global__ void memory_writex2_x1_kernel(__int64_t *array) {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  __int64_t *addr = array + threadId;
  __int64_t data = 123456;

  asm volatile("global_store_dwordx2 %0, %1, off glc\n"
               :
               : "v"(addr), "v"(data)
               : "memory");
}

__global__ void memory_writex1_x1_kernel(int *array) {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  int *addr = array + threadId;
  int data = 123456;

  asm volatile("global_store_dword %0, %1, off glc\n"
               :
               : "v"(addr), "v"(data)
               : "memory");
}


__global__ void memory_writex2_x2_kernel(__int64_t *array, int wave) {
  int threadId = blockIdx.x * blockDim.x * 2 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  __int64_t *waveAddr = array + waveId * 1024/16 + threadId;
  __int64_t data[4] = {123455, 12345, 1232, 12};
  __int64_t *addr_0  = waveAddr + 64 * 0;
  __int64_t *addr_1  = waveAddr + 64 * wave * 1;

  asm volatile("global_store_dwordx2 %0, %1, off glc\n"
               :
               : "v"(addr_0), "v"(data[0])
               : "memory");

  asm volatile("global_store_dwordx2 %0, %1, off glc\n"
               :
               : "v"(addr_1), "v"(data[1])
               : "memory");

}


__global__ void memory_writex2_x4_kernel(__int64_t *array) {
  int threadId = blockIdx.x * blockDim.x * 4 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  __int64_t *waveAddr = array + waveId * 1024/16 + threadId;
  __int64_t data[4] = {123455, 12345, 1232, 12};
  __int64_t *addr_0  = waveAddr + 64 * 0;
  __int64_t *addr_1  = waveAddr + 64 * 4;
  __int64_t *addr_2  = waveAddr + 64 * 8;
  __int64_t *addr_3  = waveAddr + 64 * 12;

  asm volatile("global_store_dwordx2 %0, %1, off glc\n"
               :
               : "v"(addr_0), "v"(data[0])
               : "memory");

  asm volatile("global_store_dwordx2 %0, %1, off glc\n"
               :
               : "v"(addr_1), "v"(data[1])
               : "memory");

  asm volatile("global_store_dwordx2 %0, %1, off glc\n"
              :
              : "v"(addr_2), "v"(data[2])
              : "memory");

  asm volatile("global_store_dwordx2 %0, %1, off glc\n"
              :
              : "v"(addr_3), "v"(data[3])
              : "memory");

}


__global__ void memory_writex1_x4_kernel(int *array) {
  int threadId = blockIdx.x * blockDim.x * 4 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  int *waveAddr = array + waveId * 1024/16 + threadId;
  int data[4] = {123455, 12345, 1232, 12};
  int *addr_0  = waveAddr + 64 * 0;
  int *addr_1  = waveAddr + 64 * 4;
  int *addr_2  = waveAddr + 64 * 8;
  int *addr_3  = waveAddr + 64 * 12;

  asm volatile("global_store_dword %0, %1, off glc\n"
               :
               : "v"(addr_0), "v"(data[0])
               : "memory");

  asm volatile("global_store_dword %0, %1, off glc\n"
               :
               : "v"(addr_1), "v"(data[1])
               : "memory");

  asm volatile("global_store_dword %0, %1, off glc\n"
              :
              : "v"(addr_2), "v"(data[2])
              : "memory");

  asm volatile("global_store_dword %0, %1, off glc\n"
              :
              : "v"(addr_3), "v"(data[3])
              : "memory");

}

__global__ void memory_writex4_x2_kernel(__int128_t *array, int wave) {
  int threadId = blockIdx.x * blockDim.x * 2 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  __int128_t *waveAddr = array + waveId * 1024/16 + threadId;
  __int128_t data[4] = {123455, 12345, 1232, 12};
  __int128_t *addr_0  = waveAddr + 64 * 0;
  __int128_t *addr_1  = waveAddr + 64 * wave * 1;



  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(addr_0), "v"(data[0])
               : "memory");

  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(addr_1), "v"(data[1])
               : "memory");
}

__global__ void memory_writex4_x4_kernel(__int128_t *array, int wave) {
  int threadId = blockIdx.x * blockDim.x * 4 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  __int128_t *waveAddr = array + waveId * 1024/16 + threadId;
  __int128_t data[4] = {123455, 12345, 1232, 12};
  __int128_t *addr_0  = waveAddr + 64 * 0;
  __int128_t *addr_1  = waveAddr + 64 * wave * 1;
  __int128_t *addr_2  = waveAddr + 64 * wave * 2;
  __int128_t *addr_3  = waveAddr + 64 * wave * 3;



  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(addr_0), "v"(data[0])
               : "memory");

  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(addr_1), "v"(data[1])
               : "memory");

  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
              :
              : "v"(addr_2), "v"(data[2])
              : "memory");

  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
              :
              : "v"(addr_3), "v"(data[3])
              : "memory");

}
// wave_continue
__global__ void memory_writex4_x8_wave_continue_kernel(__int128_t *array) {
  int threadId = blockIdx.x * blockDim.x * 8 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  __int128_t *waveAddr = array + waveId * 1024/16 + threadId;
  __int128_t data[8] = {123455, 12345, 1232, 12, 32, 43, 34, 54};
  __int128_t *addr_0  = waveAddr + 64 * 0;
  __int128_t *addr_1  = waveAddr + 64 * 4;
  __int128_t *addr_2  = waveAddr + 64 * 8;
  __int128_t *addr_3  = waveAddr + 64 * 12;
  __int128_t *addr_4  = waveAddr + 64 * 16;
  __int128_t *addr_5  = waveAddr + 64 * 20;
  __int128_t *addr_6  = waveAddr + 64 * 24;
  __int128_t *addr_7  = waveAddr + 64 * 28;

  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(addr_0), "v"(data[0])
               : "memory");

  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(addr_1), "v"(data[1])
               : "memory");

  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
              :
              : "v"(addr_2), "v"(data[2])
              : "memory");

  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
              :
              : "v"(addr_3), "v"(data[3])
              : "memory");

  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
              :
              : "v"(addr_4), "v"(data[4])
              : "memory");
 
   asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
                :
                : "v"(addr_5), "v"(data[5])
                : "memory");
 
   asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(addr_6), "v"(data[6])
               : "memory");
 
   asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(addr_7), "v"(data[7])
               : "memory");

}
__global__ void memory_writex4_x8_wave_continue_8wave_kernel(__int128_t *array) {
  int threadId = blockIdx.x * blockDim.x * 8 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  __int128_t *waveAddr = array + waveId * 1024/16 + threadId;
  __int128_t data[8] = {123455, 12345, 1232, 12, 32, 43, 34, 54};
  __int128_t *addr_0  = waveAddr + 64 * 0;
  __int128_t *addr_1  = waveAddr + 64 * 8;
  __int128_t *addr_2  = waveAddr + 64 * 16;
  __int128_t *addr_3  = waveAddr + 64 * 24;
  __int128_t *addr_4  = waveAddr + 64 * 32;
  __int128_t *addr_5  = waveAddr + 64 * 40;
  __int128_t *addr_6  = waveAddr + 64 * 48;
  __int128_t *addr_7  = waveAddr + 64 * 56;

  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(addr_0), "v"(data[0])
               : "memory");

  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(addr_1), "v"(data[1])
               : "memory");

  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
              :
              : "v"(addr_2), "v"(data[2])
              : "memory");

  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
              :
              : "v"(addr_3), "v"(data[3])
              : "memory");

  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
              :
              : "v"(addr_4), "v"(data[4])
              : "memory");
 
   asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
                :
                : "v"(addr_5), "v"(data[5])
                : "memory");
 
   asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(addr_6), "v"(data[6])
               : "memory");
 
   asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(addr_7), "v"(data[7])
               : "memory");

}
__global__ void memory_writex4_x8_wave_continue_16wave_kernel(__int128_t *array) {
  int threadId = blockIdx.x * blockDim.x * 8 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  __int128_t *waveAddr = array + waveId * 1024/16 + threadId;
  __int128_t data[8] = {123455, 12345, 1232, 12, 32, 43, 34, 54};
  __int128_t *addr_0  = waveAddr + 64 * 0;
  __int128_t *addr_1  = waveAddr + 64 * 16;
  __int128_t *addr_2  = waveAddr + 64 * 32;
  __int128_t *addr_3  = waveAddr + 64 * 48;
  __int128_t *addr_4  = waveAddr + 64 * 64;
  __int128_t *addr_5  = waveAddr + 64 * 80;
  __int128_t *addr_6  = waveAddr + 64 * 96;
  __int128_t *addr_7  = waveAddr + 64 * 112;

  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(addr_0), "v"(data[0])
               : "memory");

  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(addr_1), "v"(data[1])
               : "memory");

  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
              :
              : "v"(addr_2), "v"(data[2])
              : "memory");

  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
              :
              : "v"(addr_3), "v"(data[3])
              : "memory");

  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
              :
              : "v"(addr_4), "v"(data[4])
              : "memory");
 
   asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
                :
                : "v"(addr_5), "v"(data[5])
                : "memory");
 
   asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(addr_6), "v"(data[6])
               : "memory");
 
   asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(addr_7), "v"(data[7])
               : "memory");

}
// inst continue
__global__ void memory_writex4_x8_inst_continue_kernel(__int128_t *array) {
  int threadId = blockIdx.x * blockDim.x * 8 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  __int128_t *waveAddr = array + waveId * 1024/16 * 8 + threadId;
  __int128_t data[8] = {123455, 12345, 1232, 12, 32, 43, 34, 54};
  __int128_t *addr_0  = waveAddr + 64 * 0;
  __int128_t *addr_1  = waveAddr + 64 * 1;
  __int128_t *addr_2  = waveAddr + 64 * 2;
  __int128_t *addr_3  = waveAddr + 64 * 3;
  __int128_t *addr_4  = waveAddr + 64 * 4;
  __int128_t *addr_5  = waveAddr + 64 * 5;
  __int128_t *addr_6  = waveAddr + 64 * 6;
  __int128_t *addr_7  = waveAddr + 64 * 7;

  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(addr_0), "v"(data[0])
               : "memory");

  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(addr_1), "v"(data[1])
               : "memory");

  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
              :
              : "v"(addr_2), "v"(data[2])
              : "memory");

  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
              :
              : "v"(addr_3), "v"(data[3])
              : "memory");

  asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
              :
              : "v"(addr_4), "v"(data[4])
              : "memory");
 
   asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
                :
                : "v"(addr_5), "v"(data[5])
                : "memory");
 
   asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(addr_6), "v"(data[6])
               : "memory");
 
   asm volatile("global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(addr_7), "v"(data[7])
               : "memory");
}

