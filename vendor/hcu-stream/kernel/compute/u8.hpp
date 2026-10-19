#include "hip/hip_runtime.h"
#include "gemmHelper/peak.hpp"

using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int32_t)))) int32_t;
using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int32_t)))) int32_t;
using uint8x4_t = __attribute__((__vector_size__(4 * sizeof(uint8_t)))) uint8_t;

#if HIP_ARCH >= 928

extern "C" __global__ void v_mmac_i32_16x16x32_u8(uint8_t *A, uint8_t *B, int32_t *C,
                                  int32_t *D, uint64_t *start, uint64_t *end) {
  int block_id = blockIdx.y * gridDim.x + blockIdx.x;
  int thread_id = block_id * blockDim.x + threadIdx.x;
  uint64_t clock_start = 0;
  uint64_t clock_end = 0;
  int row = threadIdx.x % 16;
  int col = threadIdx.x / 16;
  int offset;
  int32x2_t a, b;
  int32x4_t c;

  offset = row * 32 + col * 8;
  a[0] = reinterpret_cast<int32_t>(
      uint8x4_t{A[offset], A[offset + 1], A[offset + 2], A[offset + 3]});
  a[1] = reinterpret_cast<int32_t>(
      uint8x4_t{A[offset + 4], A[offset + 5], A[offset + 6], A[offset + 7]});
  b[0] = reinterpret_cast<int32_t>(
      uint8x4_t{B[offset], B[offset + 1], B[offset + 2], B[offset + 3]});
  b[1] = reinterpret_cast<int32_t>(
      uint8x4_t{B[offset + 4], B[offset + 5], B[offset + 6], B[offset + 7]});

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
      c = __builtin_hcu_mmac_i32_16x16x32_u8(a, b, c);

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

void run_u8gemm(int iter, int trans_type) {
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

  shared_ptr<GemmMat<uint8_t>> m_hA, m_hB;
  std::vector<shared_ptr<GemmMat<uint8_t>>> m_dA, m_dB;
  shared_ptr<GemmMat<int32_t>> m_hC;
  std::vector<shared_ptr<GemmMat<int32_t>>> m_dC, m_hD;

  InitmatrixAB<uint8_t, int32_t, int32_t>(m_hA, m_hB, m_dA, m_dB);
  InitmatrixC<uint8_t, int32_t, int32_t>(m_hC, m_dC);
  InitmatrixD<uint8_t, int32_t, int32_t>(m_hD);

  std::vector<std::vector<double>> TFLOPS(g_device_num,
                                          std::vector<double>(iter, 0.0));
  for (int i = 0; i < iter; i++) {
    for (int j = 0; j < g_device_num; j++) {
      // std::cout << "HCU" << j << " ";
      CHECK_HIP_ERROR(hipSetDevice(j));
      int32_t *dataD = m_dC[j]->data();
      int32_t *dataC = m_dC[j]->data();
      uint8_t *dataA = m_dA[j]->data();
      uint8_t *dataB = m_dB[j]->data();
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
      hipLaunchKernelGGL(v_mmac_i32_16x16x32_u8, dimGrid, dimBlock, 0, 0, dataA, dataB, dataC,
                         dataD, clockStart, clockEnd);

      CHECK_HIP_ERROR(hipMemcpy(m_clock_start, clockStart,
                                m_threads * sizeof(uint64_t),
                                hipMemcpyDeviceToHost));
      CHECK_HIP_ERROR(hipMemcpy(m_clock_end, clockEnd,
                                m_threads * sizeof(uint64_t),
                                hipMemcpyDeviceToHost));

      TFLOPS[j][i] = getTflops(m_clock_start, m_clock_end, m_threads / 64, 5);
    }
    for (int j = 0; j < g_device_num; j++) {
      CHECK_HIP_ERROR(hipSetDevice(j));
      CHECK_HIP_ERROR(hipDeviceSynchronize());
    }
    // std::cout << "tflops" << setw(3) << TFLOPS[0][i] << ": " << std::endl;
    InitmatrixC<uint8_t, int32_t, int32_t>(m_hC, m_dC);

    for (int j = 0; j < g_device_num; j++) {
      CHECK_HIP_ERROR(hipSetDevice(j));
      CHECK_HIP_ERROR(hipDeviceSynchronize());
    }
  }
  PrintStatistics(TFLOPS);
}
#endif