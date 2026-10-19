#pragma once
#include "hip/hip_runtime.h"

__global__ void memory_copy_kernel(__int128_t *a, __int128_t *b) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  b[idx] = a[idx];
}
__global__ void memory_copyx2_x1_kernel(double *a, double *b) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  b[idx] = a[idx];
}


__global__ void memory_copy_x2_kernel(__int128_t *a, __int128_t *b) {
  int threadId = blockDim.x * blockIdx.x * 2 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  int idx = waveId * 64 * 2 + threadId;
  b[idx] = a[idx];
  b[idx + 64] = a[idx + 64];
}

// same wave ： Inst0 1 2 3 continue
__global__ void memory_copyx4_x4_kernel(__int128_t *a, __int128_t *b) {
  int threadId = blockDim.x * blockIdx.x * 4 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  int idx = waveId * 64 * 4 + threadId;
  b[idx] = a[idx];
  b[idx + 64] = a[idx + 64];
  b[idx + 128] = a[idx + 128];
  b[idx + 192] = a[idx + 192];
}

// same Inst ： wave0 1 2 3 continue
__global__ void memory_copyx4_x4_wave_continue_kernel(__int128_t *a, __int128_t *b) {
  int threadId = blockDim.x * blockIdx.x * 4 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  int idx = waveId * 64 + threadId;
  b[idx] = a[idx];
  b[idx + 64 * 4] = a[idx + 64 * 4];
  b[idx + 64 * 8] = a[idx + 64 * 8];
  b[idx + 64 * 12] = a[idx + 64 * 12];
}


// same wave ： Inst0 1 2 3 continue
__global__ void memory_copyx4_x8_inst_continue_kernel(__int128_t *a, __int128_t *b) {
  int threadId = blockDim.x * blockIdx.x * 8 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  int idx = waveId * 64 * 8 + threadId;
  b[idx] = a[idx];
  b[idx + 64] = a[idx + 64];
  b[idx + 128] = a[idx + 128];
  b[idx + 192] = a[idx + 192];
  b[idx + 256] = a[idx + 256];
  b[idx + 320] = a[idx + 320];
  b[idx + 384] = a[idx + 384];
  b[idx + 448] = a[idx + 448];
}

// same Inst ： wave0 1 2 3 continue
__global__ void memory_copyx4_x8_wave_continue_kernel(__int128_t *a, __int128_t *b) {
  int threadId = blockDim.x * blockIdx.x * 8 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  int idx = waveId * 64 + threadId;
  b[idx] = a[idx];
  b[idx + 64 * 4] = a[idx + 64 * 4];
  b[idx + 64 * 8] = a[idx + 64 * 8];
  b[idx + 64 * 12] = a[idx + 64 * 12];
  b[idx + 64 * 16] = a[idx + 64 * 16];
  b[idx + 64 * 20] = a[idx + 64 * 20];
  b[idx + 64 * 24] = a[idx + 64 * 24];
  b[idx + 64 * 28] = a[idx + 64 * 28];
}

__global__ void memory_copyx4_x8_wave_continue_8wave_kernel(__int128_t *a, __int128_t *b) {
  int threadId = blockDim.x * blockIdx.x * 8 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  int idx = waveId * 64 + threadId;
  b[idx] = a[idx];
  b[idx + 64 * 8] = a[idx + 64 * 8];
  b[idx + 64 * 16] = a[idx + 64 * 16];
  b[idx + 64 * 24] = a[idx + 64 * 24];
  b[idx + 64 * 32] = a[idx + 64 * 32];
  b[idx + 64 * 40] = a[idx + 64 * 40];
  b[idx + 64 * 48] = a[idx + 64 * 48];
  b[idx + 64 * 56] = a[idx + 64 * 56];
}

__global__ void memory_copyx4_x8_wave_continue_16wave_kernel(__int128_t *a, __int128_t *b) {
  int threadId = blockDim.x * blockIdx.x * 8 + threadIdx.x % 64;
  int waveId = threadIdx.x / 64;
  int idx = waveId * 64 + threadId;
  b[idx] = a[idx];
  b[idx + 64 * 16] = a[idx + 64 * 16];
  b[idx + 64 * 32] = a[idx + 64 * 32];
  b[idx + 64 * 48] = a[idx + 64 * 48];
  b[idx + 64 * 64] = a[idx + 64 * 64];
  b[idx + 64 * 80] = a[idx + 64 * 80];
  b[idx + 64 * 96] = a[idx + 64 * 96];
  b[idx + 64 * 112] = a[idx + 64 * 112];
}

__global__ void memory_copyx4_x8_wave_continue_asm_kernel(__int128_t *array, __int128_t *b) {
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

  __int128_t *waveAddr1 = b + waveId * 1024/16 + threadId;

  __int128_t *b_addr_0  = waveAddr1 + 64 * 0;
  __int128_t *b_addr_1  = waveAddr1 + 64 * 4;
  __int128_t *b_addr_2  = waveAddr1 + 64 * 8;
  __int128_t *b_addr_3  = waveAddr1 + 64 * 12;
  __int128_t *b_addr_4  = waveAddr1 + 64 * 16;
  __int128_t *b_addr_5  = waveAddr1 + 64 * 20;
  __int128_t *b_addr_6  = waveAddr1 + 64 * 24;
  __int128_t *b_addr_7  = waveAddr1 + 64 * 28;

  asm volatile("s_waitcnt vmcnt(7)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(b_addr_0), "v"(data[0])
               : "memory");

  asm volatile("s_waitcnt vmcnt(7)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(b_addr_1), "v"(data[1])
               : "memory");

  asm volatile("s_waitcnt vmcnt(7)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
              :
              : "v"(b_addr_2), "v"(data[2])
              : "memory");

  asm volatile("s_waitcnt vmcnt(7)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
              :
              : "v"(b_addr_3), "v"(data[3])
              : "memory");

  asm volatile("s_waitcnt vmcnt(7)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
              :
              : "v"(b_addr_4), "v"(data[4])
              : "memory");
 
   asm volatile("s_waitcnt vmcnt(7)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
                :
                : "v"(b_addr_5), "v"(data[5])
                : "memory");
 
   asm volatile("s_waitcnt vmcnt(7)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(b_addr_6), "v"(data[6])
               : "memory");
 
   asm volatile("s_waitcnt vmcnt(7)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(b_addr_7), "v"(data[7])
               : "memory");

}

#if HIP_ARCH >= 936
__global__ void memory_copyx4_x8_wave_continue_bps_4wave_kernel(__int128_t *array, __int128_t *b) {
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
  __int128_t *waveAddr1 = b + waveId * 1024/16 + threadId;

  __int128_t *b_addr_0  = waveAddr1 + 64 * 0;
  __int128_t *b_addr_1  = waveAddr1 + 64 * 4;
  __int128_t *b_addr_2  = waveAddr1 + 64 * 8;
  __int128_t *b_addr_3  = waveAddr1 + 64 * 12;
  __int128_t *b_addr_4  = waveAddr1 + 64 * 16;
  __int128_t *b_addr_5  = waveAddr1 + 64 * 20;
  __int128_t *b_addr_6  = waveAddr1 + 64 * 24;
  __int128_t *b_addr_7  = waveAddr1 + 64 * 28;

  asm volatile("s_waitcnt lgkmcnt(7)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(b_addr_0), "v"(data[0])
               : "memory");

  asm volatile("s_waitcnt lgkmcnt(6)\n"
                "s_waitcnt vmcnt(0)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(b_addr_1), "v"(data[1])
               : "memory");

  asm volatile("s_waitcnt lgkmcnt(5)\n"
                "s_waitcnt vmcnt(0)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
              :
              : "v"(b_addr_2), "v"(data[2])
              : "memory");

  asm volatile("s_waitcnt lgkmcnt(4)\n"
                "s_waitcnt vmcnt(0)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
              :
              : "v"(b_addr_3), "v"(data[3])
              : "memory");

  asm volatile("s_waitcnt lgkmcnt(3)\n"
                "s_waitcnt vmcnt(0)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
              :
              : "v"(b_addr_4), "v"(data[4])
              : "memory");
 
   asm volatile("s_waitcnt lgkmcnt(2)\n"
                "s_waitcnt vmcnt(0)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
                :
                : "v"(b_addr_5), "v"(data[5])
                : "memory");
 
   asm volatile("s_waitcnt lgkmcnt(1)\n"
                "s_waitcnt vmcnt(0)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(b_addr_6), "v"(data[6])
               : "memory");
 
   asm volatile("s_waitcnt lgkmcnt(0)\n"
                "s_waitcnt vmcnt(0)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(b_addr_7), "v"(data[7])
               : "memory");



}

__global__ void memory_copyx4_x8_wave_continue_bps_8wave_kernel(__int128_t *array, __int128_t *b) {

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
  __int128_t *waveAddr1 = b + waveId * 1024/16 + threadId;

  __int128_t *b_addr_0  = waveAddr1 + 64 * 0;
  __int128_t *b_addr_1  = waveAddr1 + 64 * 8;
  __int128_t *b_addr_2  = waveAddr1 + 64 * 16;
  __int128_t *b_addr_3  = waveAddr1 + 64 * 24;
  __int128_t *b_addr_4  = waveAddr1 + 64 * 32;
  __int128_t *b_addr_5  = waveAddr1 + 64 * 40;
  __int128_t *b_addr_6  = waveAddr1 + 64 * 48;
  __int128_t *b_addr_7  = waveAddr1 + 64 * 56;

  asm volatile("s_waitcnt lgkmcnt(7)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(b_addr_0), "v"(data[0])
               : "memory");

  asm volatile("s_waitcnt lgkmcnt(6)\n"
                "s_waitcnt vmcnt(0)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(b_addr_1), "v"(data[1])
               : "memory");

  asm volatile("s_waitcnt lgkmcnt(5)\n"
                "s_waitcnt vmcnt(0)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
              :
              : "v"(b_addr_2), "v"(data[2])
              : "memory");

  asm volatile("s_waitcnt lgkmcnt(4)\n"
                "s_waitcnt vmcnt(0)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
              :
              : "v"(b_addr_3), "v"(data[3])
              : "memory");

  asm volatile("s_waitcnt lgkmcnt(3)\n"
                "s_waitcnt vmcnt(0)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
              :
              : "v"(b_addr_4), "v"(data[4])
              : "memory");
 
   asm volatile("s_waitcnt lgkmcnt(2)\n"
                "s_waitcnt vmcnt(0)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
                :
                : "v"(b_addr_5), "v"(data[5])
                : "memory");
 
   asm volatile("s_waitcnt lgkmcnt(1)\n"
                "s_waitcnt vmcnt(0)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(b_addr_6), "v"(data[6])
               : "memory");
 
   asm volatile("s_waitcnt lgkmcnt(0)\n"
                "s_waitcnt vmcnt(0)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(b_addr_7), "v"(data[7])
               : "memory");



}

__global__ void memory_copyx4_x8_wave_continue_bps_16wave_kernel(__int128_t *array, __int128_t *b) {
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
  __int128_t *waveAddr1 = b + waveId * 1024/16 + threadId;

  __int128_t *b_addr_0  = waveAddr1 + 64 * 0;
  __int128_t *b_addr_1  = waveAddr1 + 64 * 16;
  __int128_t *b_addr_2  = waveAddr1 + 64 * 32;
  __int128_t *b_addr_3  = waveAddr1 + 64 * 48;
  __int128_t *b_addr_4  = waveAddr1 + 64 * 64;
  __int128_t *b_addr_5  = waveAddr1 + 64 * 80;
  __int128_t *b_addr_6  = waveAddr1 + 64 * 96;
  __int128_t *b_addr_7  = waveAddr1 + 64 * 112;

  asm volatile("s_waitcnt lgkmcnt(7)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(b_addr_0), "v"(data[0])
               : "memory");

  asm volatile("s_waitcnt lgkmcnt(6)\n"
                "s_waitcnt vmcnt(0)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(b_addr_1), "v"(data[1])
               : "memory");

  asm volatile("s_waitcnt lgkmcnt(5)\n"
                "s_waitcnt vmcnt(0)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
              :
              : "v"(b_addr_2), "v"(data[2])
              : "memory");

  asm volatile("s_waitcnt lgkmcnt(4)\n"
                "s_waitcnt vmcnt(0)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
              :
              : "v"(b_addr_3), "v"(data[3])
              : "memory");

  asm volatile("s_waitcnt lgkmcnt(3)\n"
                "s_waitcnt vmcnt(0)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
              :
              : "v"(b_addr_4), "v"(data[4])
              : "memory");
 
   asm volatile("s_waitcnt lgkmcnt(2)\n"
                "s_waitcnt vmcnt(0)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
                :
                : "v"(b_addr_5), "v"(data[5])
                : "memory");
 
   asm volatile("s_waitcnt lgkmcnt(1)\n"
                "s_waitcnt vmcnt(0)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(b_addr_6), "v"(data[6])
               : "memory");
 
   asm volatile("s_waitcnt lgkmcnt(0)\n"
                "s_waitcnt vmcnt(0)\n"
                "global_store_dwordx4 %0, %1, off glc slc\n"
               :
               : "v"(b_addr_7), "v"(data[7])
               : "memory");



}
#endif