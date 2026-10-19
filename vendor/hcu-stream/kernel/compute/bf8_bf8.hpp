#include "type/DataTypes_FP8.hpp"
#include "hip/hip_runtime.h"
#include "gemmHelper/peak.hpp"

using bit8x8_t = __attribute__((__vector_size__(8 * sizeof(uint8_t)))) uint8_t;
using float4_type = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using v2i = __attribute__((__vector_size__(2 * sizeof(int)))) int32_t;

#if HIP_ARCH >= 938
/*
    A: row major
    B: col major
    C: col major
    D: col major
*/
extern "C" __global__ void
v_mmac_f32_16x16x32_bf8_bf8(bfloat8_t *A, bfloat8_t *B, float *C, float *D,
                            uint64_t *start, uint64_t *end) {
  int block_id = blockIdx.y * gridDim.x + blockIdx.x;
  int thread_id = block_id * blockDim.x + threadIdx.x;
  uint64_t clock_start = 0;
  uint64_t clock_end = 0;
  // each thread is responsible for data of 2 dword, just 2 * 4 bytes in total
  constexpr uint32_t BYTES_PER_THREAD = 8;
  constexpr uint32_t NUM_PER_THREAD = BYTES_PER_THREAD / sizeof(bfloat8_t);

  // define const variables when test ISA v_mmac_16x16x32_fp8*
  constexpr uint32_t WAVE_SIZE_L = 64;

  // compute wave counts in a block
  uint32_t A_WAVES = blockDim.x / WAVE_SIZE_L;

  // acquire thread id
  uint32_t tx = threadIdx.x;

  // compute the wave id and lane id of this thread
  uint32_t wave_id = tx / WAVE_SIZE_L;
  uint32_t lane_id = tx % WAVE_SIZE_L;

  uint32_t A_block_offset =
      blockIdx.y * (16 * 32) * A_WAVES + wave_id * (16 * 32);
  uint32_t B_block_offset = blockIdx.x * (32 * 16);

  uint32_t lane_row = lane_id % 16;
  uint32_t lane_col = lane_id / 16;
  uint32_t lane_offset = lane_row * 32 + lane_col * NUM_PER_THREAD;
  bit8x8_t a;
#pragma unroll
  for (uint32_t i = 0; i < NUM_PER_THREAD; ++i) {
    a[i] = A[A_block_offset + lane_offset + i].uchar_value();
  }

  // B: global -> vgpr
  bit8x8_t b;
#pragma unroll
  for (uint32_t i = 0; i < NUM_PER_THREAD; ++i) {
    b[i] = B[B_block_offset + lane_offset + i].uchar_value();
  }
  uint32_t load_write_idx = blockIdx.x * gridDim.y * A_WAVES * 16 * 16 +
                            blockIdx.y * 16 * A_WAVES + wave_id * 16 +
                            lane_row + lane_col * (gridDim.y * A_WAVES * 16);

  // C: global -> vgpr
  float4_type c;
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    // each data has a interval of 64 threads
    c[i] = C[load_write_idx + A_WAVES * (gridDim.y * WAVE_SIZE_L * i)];
  }

  // execute v_mmac_16x16x32_fp8* instructions
  v2i *_a = reinterpret_cast<v2i *>(&a);
  v2i *_b = reinterpret_cast<v2i *>(&b);

  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_memtime %0\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "={s[6:7]}"(clock_start)
               :);

  __builtin_amdgcn_sched_barrier(0);
  for (int i = 0; i < 800; ++i)
    for (int j = 0; j < 64; ++j)
      c = __builtin_hcu_mmac_f32_16x16x32_bf8_bf8(*_a, *_b, c);

  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_memtime %0\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "={s[8:9]}"(clock_end)
               :);

  __builtin_amdgcn_sched_barrier(0);

  // D: vgpr -> global
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    D[load_write_idx + A_WAVES * (gridDim.y * WAVE_SIZE_L * i)] = c[i];
  }
  start[thread_id] = clock_start;
  end[thread_id] = clock_end;
}

void run_bf8_bf8gemm(int iter, int trans_type) {

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

  shared_ptr<GemmMat<bfloat8_t>> m_hA;
  shared_ptr<GemmMat<bfloat8_t>> m_hB;
  std::vector<shared_ptr<GemmMat<bfloat8_t>>> m_dA;
  std::vector<shared_ptr<GemmMat<bfloat8_t>>> m_dB;
  shared_ptr<GemmMat<float>> m_hC;
  std::vector<shared_ptr<GemmMat<float>>> m_dC, m_hD;

  InitmatrixAB_ex3<bfloat8_t, bfloat8_t, float, float>(m_hA, m_hB, m_dA, m_dB);
  InitmatrixC<bfloat8_t, float, float>(m_hC, m_dC);
  InitmatrixD<bfloat8_t, float, float>(m_hD);

  std::vector<std::vector<double>> TFLOPS(g_device_num,
                                          std::vector<double>(iter, 0.0));
  for (int i = 0; i < iter; i++) {
    for (int j = 0; j < g_device_num; j++) {
      // std::cout << "HCU" << j << " ";
      CHECK_HIP_ERROR(hipSetDevice(j));
      float *dataD = (float *)m_dC[j]->data();
      float *dataC = (float *)m_dC[j]->data();
      bfloat8_t *dataA = m_dA[j]->data();
      bfloat8_t *dataB = m_dB[j]->data();
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
      hipLaunchKernelGGL(v_mmac_f32_16x16x32_bf8_bf8, dimGrid, dimBlock, 0, 0, dataA, dataB, dataC,
                         dataD, clockStart, clockEnd);

      CHECK_HIP_ERROR(hipMemcpy(m_clock_start, clockStart,
                                m_threads * sizeof(uint64_t),
                                hipMemcpyDeviceToHost));
      CHECK_HIP_ERROR(hipMemcpy(m_clock_end, clockEnd,
                                m_threads * sizeof(uint64_t),
                                hipMemcpyDeviceToHost));

      TFLOPS[j][i] = getTflops(m_clock_start, m_clock_end, m_threads / 64, 9);
    }
    for (int j = 0; j < g_device_num; j++) {
      CHECK_HIP_ERROR(hipSetDevice(j));
      CHECK_HIP_ERROR(hipDeviceSynchronize());
    }
    // std::cout << "tflops" << setw(3) << TFLOPS[0][i] << ": " << std::endl;
    InitmatrixC<bfloat8_t, float, float>(m_hC, m_dC);

    for (int j = 0; j < g_device_num; j++) {
      CHECK_HIP_ERROR(hipSetDevice(j));
      CHECK_HIP_ERROR(hipDeviceSynchronize());
    }
  }
  PrintStatistics(TFLOPS);
}
#endif