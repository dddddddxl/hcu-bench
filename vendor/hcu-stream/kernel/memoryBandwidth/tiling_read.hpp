#pragma once
#include "hip/hip_runtime.h"

__global__ void tiling_read_kernel(const __int128_t *a, __int128_t *b,
                                   uint32_t block_num_in_h,
                                   uint32_t block_num_in_w, uint32_t c,
                                   uint32_t h, uint32_t w, const int mt_n,
                                   const int mt_h, const int mt_w) {
  int batch_id_in_block = threadIdx.x / (mt_h * mt_w);
  int channel_id = blockIdx.y;
  int piece_id_on_h_in_block = threadIdx.x % (mt_h * mt_w) / mt_w;
  int piece_id_on_w_in_block = threadIdx.x % (mt_h * mt_w) % mt_w;
  int block_idx_on_n = blockIdx.x / (block_num_in_h * block_num_in_w);
  int block_idx_on_h =
      blockIdx.x % (block_num_in_h * block_num_in_w) / block_num_in_w;
  int block_idx_on_w =
      blockIdx.x % (block_num_in_h * block_num_in_w) % block_num_in_w;

  int batch_id = block_idx_on_n * mt_n + batch_id_in_block;
  int h_id = block_idx_on_h * mt_h * 4 + piece_id_on_h_in_block * 4;
  int w_id = block_idx_on_w * mt_w + piece_id_on_w_in_block;

  const int i = batch_id * c * h * w + channel_id * h * w + h_id * w + w_id;
  __int128_t value1 = a[i];
  __int128_t value2 = a[i + w];
  __int128_t value3 = a[i + 2 * w];
  __int128_t value4 = a[i + 3 * w];
  // 此处并不会发生实际写出，仅为了不让编译器优化掉读指令
  if (value1 == 123456 || value2 == 123456 || value3 == 123456 ||
      value4 == 123456) {
    b[i] = value1;
    b[i + w] = value2;
    b[i + 2 * w] = value3;
    b[i + 3 * w] = value4;
  }
}
__global__ void tiling_read_W_kernel(const __int128_t *a, __int128_t *b,
                                   uint32_t block_num_in_h,
                                   uint32_t block_num_in_w, uint32_t c,
                                   uint32_t h, uint32_t w, const int mt_n,
                                   const int mt_h, const int mt_w) {
  int batch_id_in_block = threadIdx.x / (mt_h * mt_w);
  int channel_id = blockIdx.y;
  int piece_id_on_h_in_block = threadIdx.x % (mt_h * mt_w) / mt_w;
  int piece_id_on_w_in_block = threadIdx.x % (mt_h * mt_w) % mt_w;
  int block_idx_on_n = blockIdx.x / (block_num_in_h * block_num_in_w);
  int block_idx_on_h =
      blockIdx.x % (block_num_in_h * block_num_in_w) / block_num_in_w;
  int block_idx_on_w =
      blockIdx.x % (block_num_in_h * block_num_in_w) % block_num_in_w;

  int batch_id = block_idx_on_n * mt_n + batch_id_in_block;
  int h_id = block_idx_on_h * mt_h + piece_id_on_h_in_block;
  int w_id = block_idx_on_w * mt_w * 4 + piece_id_on_w_in_block * 4;

  const int i = batch_id * c * h * w + channel_id * h * w + h_id * w + w_id;
  __int128_t value1 = a[i];
  __int128_t value2 = a[i + 1];
  __int128_t value3 = a[i + 2];
  __int128_t value4 = a[i + 3];
  // 此处并不会发生实际写出，仅为了不让编译器优化掉读指令
  if (value1 == 123456 || value2 == 123456 || value3 == 123456 ||
      value4 == 123456) {
    b[i] = value1;
    b[i + 1] = value2;
    b[i + 2] = value3;
    b[i + 3] = value4;
  }
}

__global__ void tiling_read_kernel_2W_inst(const __int128_t* a, __int128_t* b,
                                          uint32_t block_num_in_h, uint32_t block_num_in_w,
                                          uint32_t c, uint32_t h, uint32_t w,
                                          const int mt_n, const int mt_h, const int mt_w) {
    int batch_id_in_block = threadIdx.x / (mt_h * mt_w);
    int channel_id = blockIdx.y;
    int piece_id_on_h_in_block = threadIdx.x % (mt_h * mt_w) / mt_w;
    int piece_id_on_w_in_block = threadIdx.x % (mt_h * mt_w) % mt_w;
    int block_idx_on_n = blockIdx.x / (block_num_in_h * block_num_in_w);
    int block_idx_on_h = blockIdx.x % (block_num_in_h * block_num_in_w) / block_num_in_w;
    int block_idx_on_w = blockIdx.x % (block_num_in_h * block_num_in_w) % block_num_in_w;

    int batch_id = block_idx_on_n * mt_n + batch_id_in_block;
    int h_id = block_idx_on_h * mt_h * 4 + piece_id_on_h_in_block * 4;
    int w_id = block_idx_on_w * mt_w + piece_id_on_w_in_block;

    const int i = batch_id * c * h * w + channel_id * h * w + h_id * w;
    int i_1 = i + w_id * 2;
    int i_2 = i_1 + 1;
    __int128_t value1 = a[i_1];
    __int128_t value2 = a[i_1 + w];
    __int128_t value3 = a[i_1 + 2 * w];
    __int128_t value4 = a[i_1 + 3 * w];
    __int128_t value5 = a[i_2];
    __int128_t value6 = a[i_2 + w];
    __int128_t value7 = a[i_2 + 2 * w];
    __int128_t value8 = a[i_2 + 3 * w];
    // 此处并不会发生实际写出，仅为了不让编译器优化掉读指令
    if (value1 == 123456 || value2 == 123456 || value3 == 123456 || value4 == 123456
      || value5 == 123456 || value6 == 123456 || value7 == 123456 || value8 == 123456  ) {
       b[i] = value1;
       b[i + w] = value2;
       b[i + 2 * w] = value3;
       b[i + 3 * w] = value4;
       b[i + 4 * w] = value5;
       b[i + 5 * w] = value6;
       b[i + 6 * w] = value7;
       b[i + 7 * w] = value8;
     }

    // const int i = batch_id * c * h * w + channel_id * h * w + h_id * w + w_id;
    // __int128_t value1 = a[i];
    // __int128_t value2 = a[i + w];
    // __int128_t value3 = a[i + 2 * w];
    // __int128_t value4 = a[i + 3 * w];
    // __int128_t value5 = a[i + w / 2];
    // __int128_t value6 = a[i + w / 2 + w];
    // __int128_t value7 = a[i + w / 2 + 2 * w];
    // __int128_t value8 = a[i + w / 2 + 3 * w];
    // // 此处并不会发生实际写出，仅为了不让编译器优化掉读指令
    // if (value1 == 123456 || value2 == 123456 || value3 == 123456 || value4 == 123456
    //   || value5 == 123456 || value6 == 123456 || value7 == 123456 || value8 == 123456  ) {
    //    b[i] = value1;
    //    b[i + w] = value2;
    //    b[i + 2 * w] = value3;
    //    b[i + 3 * w] = value4;
    //    b[i + 4 * w] = value5;
    //    b[i + 5 * w] = value6;
    //    b[i + 6 * w] = value7;
    //    b[i + 7 * w] = value8;
    //  }
}

__global__ void tiling_read_kernel_2H_inst(const __int128_t* a, __int128_t* b,
      uint32_t block_num_in_h, uint32_t block_num_in_w,
      uint32_t c, uint32_t h, uint32_t w,
      const int mt_n, const int mt_h, const int mt_w) {
    int batch_id_in_block = threadIdx.x / (mt_h * mt_w);
    int channel_id = blockIdx.y;
    int piece_id_on_h_in_block = threadIdx.x % (mt_h * mt_w) / mt_w;
    int piece_id_on_w_in_block = threadIdx.x % (mt_h * mt_w) % mt_w;
    int block_idx_on_n = blockIdx.x / (block_num_in_h * block_num_in_w);
    int block_idx_on_h = blockIdx.x % (block_num_in_h * block_num_in_w) / block_num_in_w;
    int block_idx_on_w = blockIdx.x % (block_num_in_h * block_num_in_w) % block_num_in_w;

    int batch_id = block_idx_on_n * mt_n + batch_id_in_block;
    int h_id = block_idx_on_h * mt_h * 8 + piece_id_on_h_in_block * 8;
    int w_id = block_idx_on_w * mt_w + piece_id_on_w_in_block;

    const int i = batch_id * c * h * w + channel_id * h * w + h_id * w + w_id;
    __int128_t value1 = a[i];
    __int128_t value2 = a[i + w];
    __int128_t value3 = a[i + 2 * w];
    __int128_t value4 = a[i + 3 * w];
    __int128_t value5 = a[i + 4 * w];
    __int128_t value6 = a[i + 5 * w];
    __int128_t value7 = a[i + 6 * w];
    __int128_t value8 = a[i + 7 * w];
    // 此处并不会发生实际写出，仅为了不让编译器优化掉读指令
    if (value1 == 123456 || value2 == 123456 || value3 == 123456 || value4 == 123456
     || value5 == 123456 || value6 == 123456 || value7 == 123456 || value8 == 123456  ) {
      b[i] = value1;
      b[i + w] = value2;
      b[i + 2 * w] = value3;
      b[i + 3 * w] = value4;
      b[i + 4 * w] = value5;
      b[i + 5 * w] = value6;
      b[i + 6 * w] = value7;
      b[i + 7 * w] = value8;
    }
}
__global__ void winograd_read_kernel(const __int128_t *a, __int128_t *b,
                                     uint32_t block_num_in_h,
                                     uint32_t block_num_in_w, uint32_t c,
                                     uint32_t h, uint32_t w, const int mt_n,
                                     const int mt_h, const int mt_w) {
  int batch_id_in_block = threadIdx.x / (mt_h * mt_w);
  int channel_id = blockIdx.y;
  // piece中h和w的id
  int piece_id_on_h_in_block = threadIdx.x % (mt_h * mt_w) / mt_w;
  int piece_id_on_w_in_block = threadIdx.x % (mt_h * mt_w) % mt_w;
  // 一共有 1023 * 1023
  int block_idx_on_n = blockIdx.x / ((h - 31) * (w - 7));
  int block_idx_on_h = blockIdx.x % ((h - 31) * (w - 7)) / (w - 7);
  int block_idx_on_w = blockIdx.x % ((h - 31) * (w - 7)) % (w - 7);

  int batch_id = block_idx_on_n * mt_n + batch_id_in_block;
  int h_id = block_idx_on_h + piece_id_on_h_in_block * 4;
  int w_id = block_idx_on_w + piece_id_on_w_in_block;

  const int i = batch_id * c * h * w + channel_id * h * w + h_id * w + w_id;
  __int128_t value1 = a[i];
  __int128_t value2 = a[i + w];
  __int128_t value3 = a[i + 2 * w];
  __int128_t value4 = a[i + 3 * w];
  // 此处并不会发生实际写出，仅为了不让编译器优化掉读指令
  if (value1 == 123456 || value2 == 123456 || value3 == 123456 ||
      value4 == 123456) {
    b[i] = value1;
    b[i + w] = value2;
    b[i + 2 * w] = value3;
    b[i + 3 * w] = value4;
  }
}


__global__ void tiling_read_nhwc_kernel(const __int128_t *a, __int128_t *b, 
                                uint32_t block_num_in_c, 
                                uint32_t block_num_in_whn, 
                                uint32_t C, uint32_t H, uint32_t W, uint32_t N,
                                const int mt_c, const int mt_whn) {


  int piece_id_on_c_in_block = threadIdx.x % (mt_c / 4);
  int piece_id_on_w_in_block = threadIdx.x / (mt_c / 4);

  int block_idx_on_c = blockIdx.x  / block_num_in_whn;
  int block_idx_on_whn = blockIdx.x % block_num_in_whn;
  int channel_id = block_idx_on_c * mt_c + piece_id_on_c_in_block * 4;
  int whn_id = block_idx_on_whn * mt_whn + piece_id_on_w_in_block;

  const int i = whn_id * C + channel_id;
  __int128_t value1 = a[i];
  __int128_t value2 = a[i + 1];
  __int128_t value3 = a[i + 2];
  __int128_t value4 = a[i + 3];
  // 此处并不会发生实际写出，仅为了不让编译器优化掉读指令
  if (value1 == 123456 || value2 == 123456 || value3 == 123456 ||
  value4 == 123456) {
    b[i] = value1;
    b[i + 1] = value2;
    b[i + 2] = value3;
    b[i + 3] = value4;
  }
}

__global__ void tiling_read_nchw_kernel(const __int128_t *a, __int128_t *b,
                                uint32_t block_num_in_h,
                                uint32_t block_num_in_w, uint32_t c,
                                uint32_t h, uint32_t w, const int mt_n,
                                const int mt_h, const int mt_w) {
  int batch_id_in_block = threadIdx.x / (mt_h * mt_w);
  int channel_id = blockIdx.y;
  int piece_id_on_h_in_block = threadIdx.x % (mt_h * mt_w) / mt_w;
  int piece_id_on_w_in_block = threadIdx.x % (mt_h * mt_w) % mt_w;
  int block_idx_on_n = blockIdx.x / (block_num_in_h * block_num_in_w);
  int block_idx_on_h =
  blockIdx.x % (block_num_in_h * block_num_in_w) / block_num_in_w;
  int block_idx_on_w =
  blockIdx.x % (block_num_in_h * block_num_in_w) % block_num_in_w;

  int batch_id = block_idx_on_n * mt_n + batch_id_in_block;
  int h_id = block_idx_on_h * mt_h * 4 + piece_id_on_h_in_block * 4;
  int w_id = block_idx_on_w * mt_w + piece_id_on_w_in_block;

  const int i = batch_id * c * h * w + channel_id * h * w + h_id * w + w_id;
  __int128_t value1 = a[i];
  __int128_t value2 = a[i + w];
  __int128_t value3 = a[i + 2 * w];
  __int128_t value4 = a[i + 3 * w];
  // 此处并不会发生实际写出，仅为了不让编译器优化掉读指令
  if (value1 == 123456 || value2 == 123456 || value3 == 123456 ||
  value4 == 123456) {
    b[i] = value1;
    b[i + w] = value2;
    b[i + 2 * w] = value3;
    b[i + 3 * w] = value4;
  }
}