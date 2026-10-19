#pragma once
#include "hip/hip_runtime.h"



__global__ void moe_triton_1_kernel(const __int128_t *a, __int128_t *b, int weight, int height, int w, int h) {
  // int block_h = height / h;
  int block_w = weight / w;
  int block_in_h = blockIdx.x / block_w;
  int block_in_w = blockIdx.x % block_w;
  int piece_id_on_h_in_block = threadIdx.x / 64;
  int piece_id_on_w_in_block = threadIdx.x % 64;
  // int test = blockIdx.x;
  const int i = block_in_h * block_w * w * h + block_in_w * w + piece_id_on_h_in_block * weight + piece_id_on_w_in_block;

  
  
  __int128_t value = a[i];
  __int128_t value1 = a[i + weight * 4];
  __int128_t value2 = a[i + weight * 8];
  __int128_t value3 = a[i + weight * 12];

  if (value == 123456) {
    b[i] = value;
  }
  if (value1 == 123456) {
    b[i+1] = value1;
  }
  if (value2 == 123456) {
    b[i+2] = value2;
  }
  if (value3 == 123456) {
    b[i+3] = value3;
  }


}


__global__ void moe_triton_2_kernel(const __int128_t *a, __int128_t *b, int weight, int height, int w, int h) {
  // int block_h = height / h;
  int block_w = weight / w;
  int block_in_h = blockIdx.x / block_w;
  int block_in_w = blockIdx.x % block_w;
  int piece_id_on_h_in_block = threadIdx.x / 16; // w
  int piece_id_on_w_in_block = threadIdx.x % 16;
  const int i = block_in_h * block_w * w * h + block_in_w * w + piece_id_on_h_in_block * weight + piece_id_on_w_in_block;

  
  
  __int128_t value = a[i];

  if (value == 123456) {
    b[i] = value;
  }

}

// // cpp -> not correct
// __global__ void moe_cpp_kernel(const int *a, int *b, int weight, int height, int w, int h) {
//   int block_h = height / h; // blocknum_h each col
//   int block_w = weight / w; // blocknum_w each row
//   int block_in_h = blockIdx.x / block_w;  // block_id in h dimension 
//   int block_in_w = blockIdx.x % block_w;   // block_id in w dimension
//   int piece_id_on_h_in_block = threadIdx.x / 32; // piece h in blcok
//   int piece_id_on_w_in_block = threadIdx.x % 32; // piece w in block
//   // caculate global memory
//   const int i = block_in_h * block_w * w * h + block_in_w * w + piece_id_on_h_in_block * weight + piece_id_on_w_in_block;
//   const int stride  = 32;
  
//   int value0;
//   int value1;
//   int value2;
//   int value3;
  
//   for (int loop = 0; loop < 2; loop++) {
//     value0 = a[i + loop *stride];
//     value1 = a[i + loop *stride + 8 * weight];
//     value2 = a[i + loop *stride + 16 * weight];
//     value3 = a[i + loop *stride + 24 * weight];

//     if (value0  == 123456) {
//       b[i] = value0;
//     }
//     if (value1  == 123456) {
//       b[i+1] = value1;
//     }
//     if (value2  == 123456) {
//       b[i+2] = value2;
//     }
//     if (value3 == 123456) {
//       b[i+3]  = value3;
//     }
//   }


// }

#if HIP_ARCH >= 936
__global__ void moe_cpp_kernel(int *a, int weight, int height, int w, int h) {

  int data[8];
  int lds_offset;

  // int block_h = height / h; // blocknum_h each col
  int block_w = weight / w; // blocknum_w each row
  int block_in_h = blockIdx.x / block_w;  // block_id in h dimension 
  int block_in_w = blockIdx.x % block_w;   // block_id in w dimension
  int piece_id_on_h_in_block = threadIdx.x / 32; // piece h in blcok
  int piece_id_on_w_in_block = threadIdx.x % 32; // piece w in block
  // caculate global memory
  const int i = block_in_h * block_w * w * h + block_in_w * w + piece_id_on_h_in_block * weight + piece_id_on_w_in_block;
  const int stride  = 32;
  // const int j = piece_id_on_h_in_block * 32 + piece_id_on_w_in_block;
  int *addr0_loop0 = a + i, *addr0_loop1 = a + i + stride;
  int *addr1_loop0 = a + i + 8 * weight, *addr1_loop1 = a + i + stride + 8 * weight;
  int *addr2_loop0 = a + i + 16 * weight, *addr2_loop1 = a + i + stride + 16 * weight;
  int *addr3_loop0 = a + i + 24 * weight, *addr3_loop1 = a + i + stride + 24 * weight;
  asm volatile("v_readfirstlane_b32 %0, %1 \n"
               "s_mov_b32 m0, %0 \n"
               : "=s"(lds_offset)
               : "v"(threadIdx.x * 4 * 4) //wave 偏移 4xdwordx4
               :);

  asm volatile(
  "global_load_dword %0, %1, off lds \n"
  "s_add_u32 m0, m0, 0x400 \n"
  : "=v"(data[0])
  : "v"(addr0_loop0)
  :
  );
  
  asm volatile(
  "global_load_dword %0, %1, off lds\n"
  "s_add_u32 m0, m0, 0x400 \n"
    : "=v"(data[1])
    : "v"(addr1_loop0)
    :
    );
  asm volatile(
    "global_load_dword %0, %1, off lds\n"
    "s_add_u32 m0, m0, 0x400 \n"
    : "=v"(data[2])
    : "v"(addr2_loop0)
    :
  );
  asm volatile(
    "global_load_dword %0, %1, off lds\n"
    "s_sub_u32 m0, m0, 0x1000 \n"
    : "=v"(data[3])
    : "v"(addr3_loop0)
    :
  );
  // load another 32x32 dword block
  asm volatile(
    "global_load_dword %0, %1, off lds\n"
    "s_add_u32 m0, m0, 0x400 \n"
    : "=v"(data[4])
    : "v"(addr0_loop1)
    :
  );
  asm volatile(
    "global_load_dword %0, %1, off lds\n"
    "s_add_u32 m0, m0, 0x400 \n"
    : "=v"(data[5])
    : "v"(addr1_loop1)
    :
  );
  asm volatile(
    "global_load_dword %0, %1, off lds\n"
    "s_add_u32 m0, m0, 0x400 \n"
    : "=v"(data[6])
    : "v"(addr2_loop1)
    :
  );
  asm volatile(
    "global_load_dword %0, %1, off lds\n"
    : "=v"(data[7])
    : "v"(addr3_loop1)
    :
  );


}
#endif