#pragma once

#define WIDTH 10240
#define HEIGHT 10240
#define ARRAY_SIZE (WIDTH * HEIGHT)

#define LAUNCH_TIMES 100
#define WAVE_SIZE 64


// for lds_read allocate
#define LDS_TOTAL_SIZE (2 * 1024)
#define lds_read_iter 2048
#define FIX_ITERS 300
#define FIX_BLOCK_NUM (120 * 4 * 8 * 10)
#define WARP_SIZE 64

// for L2CacheRW
#define L2_BLOCK_SIZE 256

// for TCC_stride read
#define TCC_BLOCK_DIM_X 64
#define TCC_BLOCK_DIM_Y 4
#define TCC_WIDTH 256
#define isFisrt false


// VGPR safe
#define busy_loops 200000
