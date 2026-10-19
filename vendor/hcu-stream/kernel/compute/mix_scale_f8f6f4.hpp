#include "hip/hip_runtime.h"
#include "type/DataTypes_FP4.hpp"
#include "type/DataTypes_FP8_Scale.hpp"
#include "gemmHelper/peak.hpp"

using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int32_t)))) int32_t;
using floatx4_type = __attribute__((__vector_size__(4 * sizeof(float)))) float;

#if HIP_ARCH == 946
extern "C"
__global__ __attribute__((hcu_scale_buffer_size(1))) void v_mmac_scale_f32_16x16x32_fp8_fp6(float8_t *A, float8_t *B, float *C, float *D, 
                                                    scale_t *ScaleA, scale_t *ScaleB, uint64_t *start, uint64_t *end)
{
  int block_id = blockIdx.y * gridDim.x + blockIdx.x;
  int thread_id = block_id * blockDim.x + threadIdx.x;
  uint64_t clock_start = 0;
  uint64_t clock_end = 0;
  // each thread is responsible for data of 2 dword, just 2 * 4 bytes in total
  constexpr uint32_t BYTES_PER_THREAD = 8;
  constexpr uint32_t NUM_PER_THREAD   = BYTES_PER_THREAD / sizeof(bfloat8_t);

  // define const variables when test ISA v_mmac_16x16x32_fp8*
  constexpr uint32_t WAVE_SIZE_L = 64;

  // compute wave counts in a block
  uint32_t A_WAVES = blockDim.x / WAVE_SIZE_L;

  // acquire thread id
  uint32_t tx = threadIdx.x;

  // compute the wave id and lane id of this thread
  uint32_t wave_id = tx / WAVE_SIZE_L;
  uint32_t lane_id = tx % WAVE_SIZE_L;

  // Load ScaleA/ScaleB
  __shared__ scale_t LdsScale[32]; 
  int ldsAddr = reinterpret_cast<unsigned long>(&(LdsScale[0])) + threadIdx.x;

  if (tx / 16 == 0) {
      asm volatile(
          "ds_write_b8 %0 ,%1\n"
          "s_waitcnt lgkmcnt(0)\n"
          :
          : "v"(ldsAddr), "v"(ScaleA[tx])
          :);
  }
  if (tx / 16 == 1) {
      asm volatile(
          "ds_write_b8 %0 ,%1\n"
          "s_waitcnt lgkmcnt(0)\n"
          :
          : "v"(ldsAddr), "v"(ScaleB[tx - 16])
          :);
  }
  __syncthreads();

  int scaleAddr0 = 0, scaleAddr1 = 8;

  // copy scale data from lds to scale buffer
  if (tx / 16 == 0) {
      asm volatile(
          "ds_scale_copy_ds2buf %0, %2 op_ctrl0:0 offset0:0 offset1:0 \n"
          "ds_scale_copy_ds2buf %1, %3 op_ctrl0:0 offset0:0 offset1:0 \n"
          "s_waitcnt lgkmcnt(0)\n"
          :
          : "v"(ldsAddr), "v"(ldsAddr + 16), "s"(scaleAddr0), "s"(scaleAddr1)
          :);
  }
  __syncthreads();

  /* for A, compute the start position of this wave in this block

                      |<------- 32 ---- --->|
              --------------------------------------
                      | ----> row major     |    ^
                      |                     |    |
                      |                     |    |
                      |                     |    |
              block 0 |                     | 16 * A_WAVES
                      |                     |    |
                      |                     |    |
                      |                     |    V
              --------|---------------------|-------
                      |                     |    ^
                      |                     |    |
                      |                     |    |
                      |                     |    |
              block 1 |                     | 16 * A_WAVES
                      |                     |    |
                      |                     |    |
                      |                     |    V
              --------|---------------------|-------
                      |                     |    ^
                      |                     |    |
                      |       wave 0        |    16
                      |                     |    |
                      |                     |    V
                      |---------------------|-------
                      |                     |
                      |                     |
                      |       wave 1        |
                      |                     |
                      |                     |
              block 2  |---------------------|
                      |                     |
                      |                     |
                      |       wave 2        |
                      |                     |
                      |                     |
                      |---------------------|
                      |                     |
                      |                     |
                      |       wave 3        |
                      |                     |
                      |                     |
              --------|---------------------|-------
  */
  uint32_t A_block_offset = blockIdx.y * (16 * 32) * A_WAVES + wave_id * (16 * 32);

  /* for B, compute the start position of this wave in this block

                      |<--- blockIdx.x * (32 * 16) --->|
              -----------------------------------------------------
                  ^   |          |          |          |          |
                  |   |          |          |          |          |
                  |   |          |          |          |          |
                  |   |          |          |          |          |   |
                  |   |          |          |          |          |   |
                  |   |  wave 0  |  wave 1  |  wave 2  |  wave 3  |   |
                  32   |(block 0) |(block 1) |(block 2) |(block 3) |   V
                  |   |          |          |          |          | col major
                  |   |          |          |          |          |
                  |   |          |          |          |          |
                  |   |          |          |          |          |
                  V   |          |          |          |          |
              -----------------------------------------------------
                      |<---16--->|
  */
  uint32_t B_block_offset = blockIdx.x * (32 * 16);

  /* compute the offset of this thread in corresponding wave, in wave !!

      for A:
                      |<---- lane_col ---->|
              ------------------------------------------------------------------------
                  ^   |                    |    -------> row major                   |
                  |   |                    |                                         |
                  |   |                    |                                         |
            lane_row  |                    |                                         |
                  |   |                    |                                         |
                  |   |                    |                                         | 16
                  V   |                    |                                         |
              --------|--------------------*                                         |
                      |                                                              |
                      |                                                              |
                      |                                                              |
                      |                                                              |
                      ----------------------------------------------------------------
                                                    32
      for B:
                      |<---- lane_row ---->|
              -------------------------------------
                  ^   |                    |      |
                  |   |                    |      |
                  |   |                    |      |
            lane_col  |                    |      |
                  |   |                    |      |
                  |   |                    |      |
                  V   |                    |      |
              --------|--------------------*      |
                      |                           |
                      |  |                        |
                      |  |                        |
                      |  |                        |  32
                      |  |                        |
                      |  V                        |
                      | col major                 |
                      |                           |
                      |                           |
                      |                           |
                      |                           |
                      |                           |
                      |                           |
                      |                           |
                      |                           |
                      -----------------------------
                                    16
  */
  uint32_t lane_row    = lane_id % 16;
  uint32_t lane_col    = lane_id / 16;
  uint32_t lane_offset = lane_row * 32 + lane_col * NUM_PER_THREAD;

  // A: global -> vgpr
  bit8x8_t a;
#pragma unroll
  for (uint32_t i = 0; i < NUM_PER_THREAD; ++i)
  {
      a[i] = A[A_block_offset + lane_offset + i].uchar_value();
  }

  // B: global -> vgpr
  bit8x8_t b;
#pragma unroll
  for (uint32_t i = 0; i < NUM_PER_THREAD; ++i)
  {
      b[i] = B[B_block_offset + lane_offset + i].uchar_value();
  }

    /* compute the start pos of the load/write index in C/D of this thread

                                                                    |<------- 16 ---- --->|
        ------------------------------------------------------------------------------------------------
            ^                                             ^         | ----> row major     |    ^
            |                                             |         |                     |    |
            |                                             |         |                     |    |
            |                                             |         |                     |    |
            |                                             |         |                     | 16 * A_WAVES
            |                                             |         |                     |    |
            |                                             |         |                     |    |
            |                                             |         |                     |    V
            |                                    blockIdx.y blocks  |---------------------|-------
            |                                             |         |                     |    ^
            |                                             |         |                     |    |
            |                                             |         |                     |    |
            |                                             |         |                     |    |
            |                                             |         |                     | 16 * A_WAVES
            |                                             |         |                     |    |
    gridDim.y blocks in Vertical                          |         |                     |    |
           thus                                           V         |                     |    V
    gridDim.y * A_WAVES * 16 * 16                       ------------|---------------------|-------
            |                                             ^         |                     |    ^
            |                                             |         |                     |    |
            |                                             |         |       wave 0        |    16
            |                                             |         |                     |    |
            |                                             |         |                     |    V
            |                                      wave_id * 16     |---------------------|-------
            |                                             |         |                     |
            |                                             |         |                     |
            |                                             |         |       wave 1        |
            |                                             |         |                     |
            |                                             V         |                     |
            |                                           ------------|---------------------|
            |                                             |         |            |        |
            |                                          lane_row     |            |        |
            |                                             |         |            |        |
            |                     |                     ------------|------------|        |
            |                     |                                 |<-lane_col->         |
            |                     |                                 |---------------------|
            |                     V                                 |                     |
            |                  col major                            |                     |
            |                                                       |       wave 3        |
            |                                                       |                     |
            V                                                       |                     |
        ------------------------------------------------------------|---------------------|-------
            |<-------- blockIdx.x blocks in Horizontal ------------>|
    */
  uint32_t load_write_idx = blockIdx.x * gridDim.y * A_WAVES * 16 * 16 /* offset of blocks before*/
                          + blockIdx.y * 16 * A_WAVES + wave_id * 16 + lane_row /* vertical offset in current block */
                          + lane_col * (gridDim.y * A_WAVES * 16); /* horizontal offste in current block */

  // C: init with 0
  float4_type c;
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i)
  {
    c[i] = 0.0f;
    // c[i] = C[offset + 64 * i];
  }

    // execute v_mmac_scale_f32_16x16x32* instructions
  int m0 = 8 << 16;
  asm volatile(
    "s_mov_b32 m0, %0\n"
    "s_nop 0\n"
    : 
    : "s"(m0)
    :);
      __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_memtime %0\n"
                "s_waitcnt lgkmcnt(0)\n"
                : "={s[6:7]}"(clock_start)
                :);

  __builtin_amdgcn_sched_barrier(0);
  for (int i = 0; i < 800; ++i)
    for (int j = 0; j < 64; ++j)
      c = __builtin_hcu_mmac_scale_f32_16x16x32_f8f6f4_lit_lts(a, b, c, /*RealABTypeId*/2, /*lit=*/false, /*lts=*/false, /*scale_a_offset*/0, /*scale_b_offset*/0);

  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_memtime %0\n"
                "s_waitcnt lgkmcnt(0)\n"
                : "={s[8:9]}"(clock_end)
                :);

  __builtin_amdgcn_sched_barrier(0);
  // D: vgpr -> global
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i)
  {
      D[load_write_idx + A_WAVES * (gridDim.y * WAVE_SIZE_L * i)] = c[i];
  }
  start[thread_id] = clock_start;
  end[thread_id] = clock_end;
}


void run_mix_scale_f8f6f4gemm(int iter, int trans_type) {
  g_device_props = DeviceInit();
  dim3 dimGrid = dim3(g_device_props.multiProcessorCount / 4, 4);
  dim3 dimBlock = dim3(256);
  int m_threads =
      g_device_props.multiProcessorCount * 4 * 64; // 4waves on each wave
  params.M = 32 * dimGrid.x;
  params.N = 32 * dimGrid.y;
  params.K = 32;
  if (trans_type == 1 /*NT*/) {
    params.transA = false;
    params.transB = true;
  } else if (trans_type == 2 /*TN*/) {
    params.transA = true;
    params.transB = false;
  } else if (trans_type == 3 /*TT*/) {
    params.transA = true;
    params.transB = true;
  } else if (trans_type == 0 /*NN*/) {
    params.transA = false;
    params.transB = false;
  }


  shared_ptr<GemmMat<float8_t>> m_hA;
  shared_ptr<GemmMat<float8_t>> m_hB;
  std::vector<shared_ptr<GemmMat<float8_t>>> m_dA;
  std::vector<shared_ptr<GemmMat<float8_t>>> m_dB;
  shared_ptr<GemmMat<float>> m_hC;
  shared_ptr<GemmMat<scale_t>> m_scalehA, m_scalehB;
  std::vector<shared_ptr<GemmMat<scale_t>>> m_scaledA, m_scaledB;
  std::vector<shared_ptr<GemmMat<float>>> m_dC, m_hD;

  InitmatrixAB_scale<float8_t, float, float>(m_hA, m_hB, m_scalehA, m_scalehB,
                                               m_dA, m_dB, m_scaledA, m_scaledB);
  InitmatrixC<float8_t, float, float>(m_hC, m_dC);
  InitmatrixD<float8_t, float, float>(m_hD);

  std::vector<std::vector<double>> TFLOPS(g_device_num,
                                          std::vector<double>(iter, 0.0));
  for (int i = 0; i < iter; i++) {
    for (int j = 0; j < g_device_num; j++) {
      std::cout << "HCU" << j << " ";
      CHECK_HIP_ERROR(hipSetDevice(j));
      float *dataC = (float *)m_dC[j]->data();
      float *dataD = (float *)m_dC[j]->data();
      float8_t *dataA = m_dA[j]->data();
      float8_t *dataB = m_dB[j]->data();
      scale_t *dataScaleA = m_scaledA[j]->data();
      scale_t *dataScaleB = m_scaledB[j]->data();
      auto m_clock_start = new uint64_t[m_threads];
      auto m_clock_end = new uint64_t[m_threads];

      for (int i = 0; i < m_threads; i++) {
        m_clock_start[i] = 0;
        m_clock_end[i] = 0;
      }
      uint64_t *clockStart;
      uint64_t *clockEnd;
      CHECK_HIP_ERROR(
          hipMalloc((void **)&clockStart, m_threads * sizeof(uint64_t)));
      CHECK_HIP_ERROR(
          hipMalloc((void **)&clockEnd, m_threads * sizeof(uint64_t)));

      CHECK_HIP_ERROR(hipMemcpy(clockStart, m_clock_start,
                                m_threads * sizeof(uint64_t),
                                hipMemcpyHostToDevice));
      CHECK_HIP_ERROR(hipMemcpy(clockEnd, m_clock_end,
                                m_threads * sizeof(uint64_t),
                                hipMemcpyHostToDevice));

      hipEvent_t start_event_ = nullptr;
      hipEvent_t stop_event_ = nullptr;

      CHECK_HIP_ERROR(hipEventCreate(&start_event_));
      CHECK_HIP_ERROR(hipEventCreate(&stop_event_));
      std::cout<<"launch kernel"<<std::endl;
      hipLaunchKernelGGL(v_mmac_scale_f32_16x16x32_fp8_fp6, dimGrid, dimBlock, 0, 0,
                         dataA, dataB, dataC, dataD, dataScaleA, dataScaleB, clockStart, clockEnd);
      std::cout<<"finsh kernel"<<std::endl;
      CHECK_HIP_ERROR(hipMemcpy(m_clock_start, clockStart,
                                m_threads * sizeof(uint64_t),
                                hipMemcpyDeviceToHost));
      CHECK_HIP_ERROR(hipMemcpy(m_clock_end, clockEnd,
                                m_threads * sizeof(uint64_t),
                                hipMemcpyDeviceToHost));

      TFLOPS[j][i] = getTflops(m_clock_start, m_clock_end, m_threads / 64, 9);
      CHECK_HIP_ERROR(hipFree(clockStart));
      CHECK_HIP_ERROR(hipFree(clockEnd));
      delete[] m_clock_start;
      delete[] m_clock_end;
    }
    for (int j = 0; j < g_device_num; j++) {
      CHECK_HIP_ERROR(hipSetDevice(j));
      CHECK_HIP_ERROR(hipDeviceSynchronize());
    }
    // std::cout << "tflops" << setw(3) << TFLOPS[0][i] << ": " << std::endl;
    InitmatrixC<float8_t, float, float>(m_hC, m_dC);

    for (int j = 0; j < g_device_num; j++) {
      CHECK_HIP_ERROR(hipSetDevice(j));
      CHECK_HIP_ERROR(hipDeviceSynchronize());
    }
  }
  PrintStatistics(TFLOPS);

}
#endif