#include "hip/hip_runtime.h"
#include "gemmHelper/peak.hpp"

using half_t = _Float16;
using float4_type = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using half4_t = __attribute__((__vector_size__(4 * sizeof(half_t)))) half_t;
#if HIP_ARCH >= 928
extern "C" __global__ void v_mmac_f32_16x16x16_f16(half_t *A, half_t *B, float *C, float *D,
                                 uint64_t *start, uint64_t *end) {
  int block_id = blockIdx.y * gridDim.x + blockIdx.x;
  int thread_id = block_id * blockDim.x + threadIdx.x;
  uint64_t clock_start = 0;
  uint64_t clock_end = 0;
  int row = threadIdx.x % 16;
  int col = threadIdx.x / 16;
  int offset;
  half4_t a, b;
  float4_type c;

  offset = row * 16 + col * 4;
  a[0] = A[offset];
  a[1] = A[offset + 1];
  a[2] = A[offset + 2];
  a[3] = A[offset + 3];
  b[0] = B[offset];
  b[1] = B[offset + 1];
  b[2] = B[offset + 2];
  b[3] = B[offset + 3];

  offset = row + col * 16;
#pragma unroll
  for (int i = 0; i < 4; i++)
    c[i] = C[offset + 64 * i];

  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_memtime %0\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "={s[6:7]}"(clock_start)
               :);

  __builtin_amdgcn_sched_barrier(0);
  for (int i = 0; i < 800; ++i)
    for (int j = 0; j < 64; ++j)
      c = __builtin_hcu_mmac_f32_16x16x16_f16(a, b, c);

  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_memtime %0\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "={s[8:9]}"(clock_end)
               :);

  __builtin_amdgcn_sched_barrier(0);
#pragma unroll
  for (int i = 0; i < 4; i++)
    D[offset + 64 * i] = c[i];

  start[thread_id] = clock_start;
  end[thread_id] = clock_end;
}

void run_f16gemm(int iter, int trans_type) {
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

  shared_ptr<GemmMat<_Float16>> m_hA, m_hB;
  std::vector<shared_ptr<GemmMat<_Float16>>> m_dA, m_dB;
  shared_ptr<GemmMat<_Float16>> m_hC;
  std::vector<shared_ptr<GemmMat<_Float16>>> m_dC, m_hD;

  InitmatrixAB<_Float16, _Float16, float>(m_hA, m_hB, m_dA, m_dB);
  InitmatrixC<_Float16, _Float16, float>(m_hC, m_dC);
  InitmatrixD<_Float16, _Float16, float>(m_hD);

  std::vector<std::vector<double>> TFLOPS(g_device_num,
                                          std::vector<double>(iter, 0.0));
  for (int i = 0; i < iter; i++) {
    for (int j = 0; j < g_device_num; j++) {
      // std::cout << "HCU" << j << " ";
      CHECK_HIP_ERROR(hipSetDevice(j));
      float *dataD = (float *)m_dC[j]->data();
      float *dataC = (float *)m_dC[j]->data();
      _Float16 *dataA = m_dA[j]->data();
      _Float16 *dataB = m_dB[j]->data();
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
      hipLaunchKernelGGL(v_mmac_f32_16x16x16_f16, dimGrid, dimBlock, 0, 0, dataA, dataB, dataC,
                         dataD, clockStart, clockEnd);

      CHECK_HIP_ERROR(hipMemcpy(m_clock_start, clockStart,
                                m_threads * sizeof(uint64_t),
                                hipMemcpyDeviceToHost));
      CHECK_HIP_ERROR(hipMemcpy(m_clock_end, clockEnd,
                                m_threads * sizeof(uint64_t),
                                hipMemcpyDeviceToHost));

      TFLOPS[j][i] = getTflops(m_clock_start, m_clock_end, m_threads / 64, 0);
    }
    for (int j = 0; j < g_device_num; j++) {
      CHECK_HIP_ERROR(hipSetDevice(j));
      CHECK_HIP_ERROR(hipDeviceSynchronize());
    }
    // std::cout << "tflops" << setw(3) << TFLOPS[0][i] << ": " << std::endl;
    InitmatrixC<_Float16, _Float16, float>(m_hC, m_dC);

    for (int j = 0; j < g_device_num; j++) {
      CHECK_HIP_ERROR(hipSetDevice(j));
      CHECK_HIP_ERROR(hipDeviceSynchronize());
    }
  }
  PrintStatistics(TFLOPS);
}
#endif