#include "hip/hip_runtime.h"
#include "gemmHelper/peak.hpp"
#include "gemmHelper/utils.hpp"
#include "type/DataTypes_FP8_Scale.hpp"
#include "type/DataTypes_FP4.hpp"
using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int32_t)))) int32_t;
using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int32_t)))) int32_t;
using uint8x4_t = __attribute__((__vector_size__(4 * sizeof(uint8_t)))) uint8_t;
using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;

#if HIP_ARCH == 946

extern "C"
__global__ __attribute__((hcu_scale_buffer_size(1))) void v_mmac_scale_f32_16x16x64_fp4(float4x2_t *A, float4x2_t *B, float *C, float *D, 
                                                  scale_t *ScaleA, scale_t *ScaleB, uint64_t *start, uint64_t *end)
{
  int block_id = blockIdx.y * gridDim.x + blockIdx.x;
  int thread_id = block_id * blockDim.x + threadIdx.x;
  uint64_t clock_start = 0;
  uint64_t clock_end = 0;

  constexpr uint32_t WAVE_SIZE_L = 64;
  uint32_t A_WAVES = blockDim.x / WAVE_SIZE_L;
  uint32_t tx = threadIdx.x;
  uint32_t wave_id = tx / WAVE_SIZE_L;
  uint32_t lane_id = tx % WAVE_SIZE_L;

  int row = threadIdx.x % 16;
  int col = threadIdx.x / 16;
  int offset;
  int32x2_t a, b;
  floatx4_t c;

  // Load ScaleA/ScaleB
  __shared__ scale_t LdsScale[64];
  int ldsAddr = reinterpret_cast<unsigned long>(&(LdsScale[0])) + threadIdx.x;

  if (threadIdx.x / 32 == 0) {
      asm volatile(
          "ds_write_b8 %0 ,%1\n"
          "s_waitcnt lgkmcnt(0)\n"
          :
          : "v"(ldsAddr), "v"(ScaleA[threadIdx.x])
          :);
  }
  if (threadIdx.x / 32 == 1) {
      asm volatile(
          "ds_write_b8 %0 ,%1\n"
          "s_waitcnt lgkmcnt(0)\n"
          :
          : "v"(ldsAddr), "v"(ScaleB[threadIdx.x - 32])
          :);
  }
  __syncthreads();

  int scaleAddr0 = 0, scaleAddr1 = 8;
  if (threadIdx.x / 16 == 0) {
      asm volatile(
          "ds_scale_copy_ds2buf %0, %4 op_ctrl0:0 offset0:0 offset1:0 \n"
          "ds_scale_copy_ds2buf %1, %4 op_ctrl0:0 offset0:1 offset1:0 \n"
          "ds_scale_copy_ds2buf %2, %5 op_ctrl0:0 offset0:0 offset1:0 \n"
          "ds_scale_copy_ds2buf %3, %5 op_ctrl0:0 offset0:1 offset1:0 \n"
          "s_waitcnt lgkmcnt(0)\n"
          :
          : "v"(ldsAddr), "v"(ldsAddr + 16), "v"(ldsAddr * 2 + 32), "v"(ldsAddr * 2 + 1 + 32), "s"(scaleAddr0), "s"(scaleAddr1)
          :);
  }
  __syncthreads();

  // load A/B
  offset = (row * 64 + col * 16) / 16;
  a = reinterpret_cast<int32x2_t*>(A)[offset];
  b = reinterpret_cast<int32x2_t*>(B)[offset];

  // D write offset for this lane inside a 16x16 tile
  offset = row + col * 16;

#pragma unroll
  for (int i = 0; i < 4; i++)
    c[i] = C[offset + 64 * i];

  int m0 = 8 << 16;
  asm volatile(
    "s_mov_b32 m0, %0\n"
    "s_nop 0\n"
    :
    :"s"(m0)
  :);

  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_memtime %0\n"
                "s_waitcnt lgkmcnt(0)\n"
                : "={s[6:7]}"(clock_start)
                :);

  __builtin_amdgcn_sched_barrier(0);
  for (int i = 0; i < 800; ++i)
    for (int j = 0; j < 64; ++j)
      c = __builtin_hcu_mmac_scale_f32_16x16x64_fp4_lit_lts(a, b, c, /*lit=*/false, /*lts=*/false, /*scale_a_offset*/0, /*scale_b_offset*/0);

  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_memtime %0\n"
                "s_waitcnt lgkmcnt(0)\n"
                : "={s[8:9]}"(clock_end)
                :);

  __builtin_amdgcn_sched_barrier(0);
  // execute v_mmac_scale_f32_16x16x64_fp4 instruction
  

  // D: vgpr -> global
#pragma unroll
  for (int i = 0; i < 4; i++)
    D[offset + 64 * i] = c[i];

  start[thread_id] = clock_start;
  end[thread_id] = clock_end;
}

void run_scale_fp4gemm(int iter, int trans_type) {
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


  shared_ptr<GemmMat<float4x2_t>> m_hA;
  shared_ptr<GemmMat<float4x2_t>> m_hB;
  std::vector<shared_ptr<GemmMat<float4x2_t>>> m_dA;
  std::vector<shared_ptr<GemmMat<float4x2_t>>> m_dB;
  shared_ptr<GemmMat<float>> m_hC;
  shared_ptr<GemmMat<scale_t>> m_scalehA, m_scalehB;
  std::vector<shared_ptr<GemmMat<scale_t>>> m_scaledA, m_scaledB;
  std::vector<shared_ptr<GemmMat<float>>> m_dC, m_hD;

  InitmatrixAB_scale<float4x2_t, float, float>(m_hA, m_hB, m_scalehA, m_scalehB,
                                               m_dA, m_dB, m_scaledA, m_scaledB);
  InitmatrixC<float4x2_t, float, float>(m_hC, m_dC);
  InitmatrixD<float4x2_t, float, float>(m_hD);

  std::vector<std::vector<double>> TFLOPS(g_device_num,
                                          std::vector<double>(iter, 0.0));
  for (int i = 0; i < iter; i++) {
    for (int j = 0; j < g_device_num; j++) {
      std::cout << "HCU" << j << " ";
      CHECK_HIP_ERROR(hipSetDevice(j));
      float *dataC = (float *)m_dC[j]->data();
      float *dataD = (float *)m_dC[j]->data();
      float4x2_t *dataA = m_dA[j]->data();
      float4x2_t *dataB = m_dB[j]->data();
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
      std::cout<<"launch "<<std::endl;
      hipLaunchKernelGGL(v_mmac_scale_f32_16x16x64_fp4, dimGrid, dimBlock, 0, 0,
                         dataA, dataB, dataC, dataD, dataScaleA, dataScaleB, clockStart, clockEnd);
      std:cout<<"finsh"<<std::endl;
      CHECK_HIP_ERROR(hipMemcpy(m_clock_start, clockStart,
                                m_threads * sizeof(uint64_t),
                                hipMemcpyDeviceToHost));
      CHECK_HIP_ERROR(hipMemcpy(m_clock_end, clockEnd,
                                m_threads * sizeof(uint64_t),
                                hipMemcpyDeviceToHost));

      TFLOPS[j][i] = getTflops(m_clock_start, m_clock_end, m_threads / 64, 8);
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
    InitmatrixC<float4x2_t, float, float>(m_hC, m_dC);

    for (int j = 0; j < g_device_num; j++) {
      CHECK_HIP_ERROR(hipSetDevice(j));
      CHECK_HIP_ERROR(hipDeviceSynchronize());
    }
  }
  PrintStatistics(TFLOPS);


}
#endif