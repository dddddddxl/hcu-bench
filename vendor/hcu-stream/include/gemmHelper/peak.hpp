#ifndef __PEAK_H__
#define __PEAK_H__
#include "gemmHelper/matrix.hpp"
#include "gemmHelper/utils.hpp"

template <typename SRC_T, typename DST_T, typename ACCUM_T>
void InitmatrixAB(shared_ptr<GemmMat<SRC_T>> m_hA,
                  shared_ptr<GemmMat<SRC_T>> m_hB,
                  std::vector<shared_ptr<GemmMat<SRC_T>>> &m_dA,
                  std::vector<shared_ptr<GemmMat<SRC_T>>> &m_dB);

template <typename SRC_TA, typename SRC_TB, typename DST_T, typename ACCUM_T>
void InitmatrixAB_ex3(shared_ptr<GemmMat<SRC_TA>> m_hA,
                      shared_ptr<GemmMat<SRC_TB>> m_hB,
                      std::vector<shared_ptr<GemmMat<SRC_TA>>> &m_dA,
                      std::vector<shared_ptr<GemmMat<SRC_TB>>> &m_dB);

template <typename SRC_T, typename DST_T, typename ACCUM_T>
void InitmatrixAB_scale(shared_ptr<GemmMat<SRC_T>> m_hA,
                        shared_ptr<GemmMat<SRC_T>> m_hB,
                        shared_ptr<GemmMat<scale_t>> m_hScaleA,
                        shared_ptr<GemmMat<scale_t>> m_hScaleB,
                        std::vector<shared_ptr<GemmMat<SRC_T>>> &m_dA,
                        std::vector<shared_ptr<GemmMat<SRC_T>>> &m_dB,
                        std::vector<shared_ptr<GemmMat<scale_t>>> &m_dScaleA,
                        std::vector<shared_ptr<GemmMat<scale_t>>> &m_dScaleB);

template <typename SRC_T, typename DST_T, typename ACCUM_T>
void InitmatrixC(shared_ptr<GemmMat<DST_T>> m_hC,
                 std::vector<shared_ptr<GemmMat<DST_T>>> &m_dC);

template <typename SRC_T, typename DST_T, typename ACCUM_T>
void InitmatrixD(std::vector<shared_ptr<GemmMat<DST_T>>> &m_hD);

template <typename SRC_T, typename DST_T, typename ACCUM_T>
double HelperLaunchKerneltemplate(std::vector<shared_ptr<GemmMat<SRC_T>>> &m_dA,
                                  std::vector<shared_ptr<GemmMat<SRC_T>>> &m_dB,
                                  std::vector<shared_ptr<GemmMat<DST_T>>> &m_dC,
                                  int index, int gemm_type);

void PrintStatistics(std::vector<std::vector<double>> &m_tflops);

hipDeviceProp_t g_device_props;
int g_device_num;
hipDeviceProp_t DeviceInit(void) {
  int device_id = 0;
  hipDeviceProp_t g_props;
  CHECK_HIP_ERROR(hipGetDeviceCount(&g_device_num));
  CHECK_HIP_ERROR(hipGetDevice(&device_id));
  CHECK_HIP_ERROR(hipGetDeviceProperties(&(g_props), device_id));

  return g_props;
}
param params;

void PrintStatistics(std::vector<std::vector<double>> &m_tflops) {
  for (int i = 0; i < g_device_num; i++) {
    vector<double>::iterator min =
        min_element(m_tflops[i].begin(), m_tflops[i].end());
    vector<double>::iterator max =
        max_element(m_tflops[i].begin(), m_tflops[i].end());
    double sum = accumulate(m_tflops[i].begin(), m_tflops[i].end(), 0.0);
    double mean = sum / m_tflops[i].size();
    double accum = 0.0;
    for_each(m_tflops[i].begin(), m_tflops[i].end(), [&](const double tflops) {
      accum += (tflops - mean) * (tflops - mean);
    });
    double stdev = sqrt(accum / m_tflops[i].size());

    std::cout << "HCU" << i << ": min: " << *min << "\tmax: " << *max;
    std::cout << "\tmean: " << mean << "\tstdev: " << stdev << std::endl;
  }
}

template <typename SRC_T, typename DST_T, typename ACCUM_T>
void InitmatrixAB(shared_ptr<GemmMat<SRC_T>> m_hA,
                  shared_ptr<GemmMat<SRC_T>> m_hB,
                  std::vector<shared_ptr<GemmMat<SRC_T>>> &m_dA,
                  std::vector<shared_ptr<GemmMat<SRC_T>>> &m_dB) {
  Range<SRC_T> range =
      ::RandRange<SRC_T, DST_T, ACCUM_T>(params.alpha, params.beta, params.K)
          .first;
  Rng<SRC_T> rng(range.min, range.max);
  bool m_true_rand = false;
  srand(m_true_rand == true ? (unsigned)time(NULL) : 1);
  // 根据条件来决定是否用当前时间来初始化随机数生成器。
  m_hA = std::make_shared<GemmMat<SRC_T>>(params.M, params.K, params.lda,
                                          params.stridea, params.batch_count,
                                          params.transA, CPU, rng, false);
  m_hB = std::make_shared<GemmMat<SRC_T>>(params.K, params.N, params.ldb,
                                          params.strideb, params.batch_count,
                                          params.transB, CPU, rng, false);
  // std::cout << "A = " << m_hA->get(0,1,0) << std::endl;
  // std::cout << "B = " << m_hB->get(0,1,0) << std::endl;
  for (int i = 0; i < g_device_num; i++) {
    CHECK_HIP_ERROR(hipSetDevice(i));
    m_dA.push_back(std::make_shared<GemmMat<SRC_T>>(
        params.M, params.K, params.lda, params.stridea, params.batch_count,
        params.transA, GPU0));
    m_dB.push_back(std::make_shared<GemmMat<SRC_T>>(
        params.K, params.N, params.ldb, params.strideb, params.batch_count,
        params.transB, GPU0));
    m_dA[i]->Copy(m_hA);
    m_dB[i]->Copy(m_hB);
  }
}

template <typename SRC_TA, typename SRC_TB, typename DST_T, typename ACCUM_T>
void InitmatrixAB_ex3(
    shared_ptr<GemmMat<SRC_TA>> m_hA, shared_ptr<GemmMat<SRC_TB>> m_hB,
    std::vector<shared_ptr<GemmMat<SRC_TA>>> &m_dA,
    std::vector<shared_ptr<GemmMat<SRC_TB>>> &m_dB) {
  Range<SRC_TA> range_A =
      ::RandRange<SRC_TA, DST_T, ACCUM_T>(params.alpha, params.beta, params.K).first;
  Rng<SRC_TA> rng_A(range_A.min, range_A.max);
  Range<SRC_TB> range_B =
      ::RandRange<SRC_TB, DST_T, ACCUM_T>(params.alpha, params.beta, params.K).first;
  Rng<SRC_TB> rng_B(range_B.min, range_B.max);

  bool m_true_rand = false;
  srand(m_true_rand == true ? (unsigned)time(NULL) : 1);
  // 根据条件来决定是否用当前时间来初始化随机数生成器。
  m_hA = std::make_shared<GemmMat<SRC_TA>>(
    params.M, params.K, params.lda, params.stridea, params.batch_count, params.transA, CPU, rng_A, false);
  m_hB = std::make_shared<GemmMat<SRC_TB>>(
    params.K, params.N, params.ldb, params.strideb, params.batch_count, params.transB, CPU, rng_B, false);
  // std::cout << "A = " << m_hA->get(0,1,0) << std::endl;
  // std::cout << "B = " << m_hB->get(0,1,0) << std::endl;
  for (int i = 0; i < g_device_num; i++) {
    CHECK_HIP_ERROR(hipSetDevice(i));
    m_dA.push_back(std::make_shared<GemmMat<SRC_TA>>(
      params.M, params.K, params.lda, params.stridea, params.batch_count, params.transA, GPU0));
    m_dB.push_back(std::make_shared<GemmMat<SRC_TB>>(
      params.K, params.N, params.ldb, params.strideb, params.batch_count, params.transB, GPU0));
    m_dA[i]->Copy(m_hA);
    m_dB[i]->Copy(m_hB);
  }
}

template <typename SRC_T, typename DST_T, typename ACCUM_T>
void InitmatrixAB_scale(shared_ptr<GemmMat<SRC_T>> m_hA,
                        shared_ptr<GemmMat<SRC_T>> m_hB,
                        shared_ptr<GemmMat<scale_t>> m_hScaleA,
                        shared_ptr<GemmMat<scale_t>> m_hScaleB,
                        std::vector<shared_ptr<GemmMat<SRC_T>>> &m_dA,
                        std::vector<shared_ptr<GemmMat<SRC_T>>> &m_dB,
                        std::vector<shared_ptr<GemmMat<scale_t>>> &m_dScaleA,
                        std::vector<shared_ptr<GemmMat<scale_t>>> &m_dScaleB) {
  // static_assert(std::is_same<SRC_T, float4x2_t>::value,
  //               "InitmatrixAB_scale currently supports SRC_T=float4x2_t only.");

  
  Range<SRC_T> range =
      ::RandRange<SRC_T, DST_T, ACCUM_T>(params.alpha, params.beta, params.K)
      .first;
  Rng<SRC_T> rng(range.min, range.max);

  // Scale values are 8-bit exponents (see DataTypes_FP8_Scale.hpp). Keep them
  // away from extremes; also keep small enough to avoid overflow.
  Rng<scale_t> rng_scale(static_cast<scale_t>(100), static_cast<scale_t>(180));

  bool m_true_rand = false;
  srand(m_true_rand == true ? (unsigned)time(NULL) : 1);

  m_hA = std::make_shared<GemmMat<SRC_T>>(params.M, params.K, params.lda,
                                         params.stridea, params.batch_count,
                                         params.transA, CPU, rng, false);
  m_hB = std::make_shared<GemmMat<SRC_T>>(params.K, params.N, params.ldb,
                                         params.strideb, params.batch_count,
                                         params.transB, CPU, rng, false);

  // Scale vectors: each block uses 32 bytes for ScaleA and 32 bytes for ScaleB.
  // (The kernel indexes only the first 32 of each.)
  m_hScaleA = std::make_shared<GemmMat<scale_t>>(1, 32, 0, 0, 1, false, CPU,
                                                rng_scale, false);
  m_hScaleB = std::make_shared<GemmMat<scale_t>>(1, 32, 0, 0, 1, false, CPU,
                                                rng_scale, false);

  for (int i = 0; i < g_device_num; i++) {
    CHECK_HIP_ERROR(hipSetDevice(i));
    m_dA.push_back(std::make_shared<GemmMat<SRC_T>>(
        params.M, params.K, params.lda, params.stridea, params.batch_count,
        params.transA, GPU0));
    m_dB.push_back(std::make_shared<GemmMat<SRC_T>>(
        params.K, params.N, params.ldb, params.strideb, params.batch_count,
        params.transB, GPU0));
    m_dScaleA.push_back(std::make_shared<GemmMat<scale_t>>(1, 32, 0, 0, 1, false, GPU0));
    m_dScaleB.push_back(std::make_shared<GemmMat<scale_t>>(1, 32, 0, 0, 1, false, GPU0));

    m_dA[i]->Copy(m_hA);
    m_dB[i]->Copy(m_hB);
    m_dScaleA[i]->Copy(m_hScaleA);
    m_dScaleB[i]->Copy(m_hScaleB);
  }
}
template <typename SRC_T, typename DST_T, typename ACCUM_T>
void InitmatrixC(shared_ptr<GemmMat<DST_T>> m_hC,
                 std::vector<shared_ptr<GemmMat<DST_T>>> &m_dC) {
  Range<DST_T> range =
      ::RandRange<SRC_T, DST_T, ACCUM_T>(params.alpha, params.beta, params.K)
          .second;
  Rng<DST_T> rng(range.min, range.max);
  bool m_true_rand = false;
  srand(m_true_rand == true ? (unsigned)time(NULL) : 1);
  m_hC = std::make_shared<GemmMat<DST_T>>(params.M, params.N, params.ldc,
                                          params.stridec, params.batch_count,
                                          false, CPU, rng, false);
  for (int i = 0; i < g_device_num; i++) {
    CHECK_HIP_ERROR(hipSetDevice(i));
    m_dC.push_back(std::make_shared<GemmMat<DST_T>>(
        params.M, params.N, params.ldc, params.stridec, params.batch_count,
        false, GPU0));
    m_dC[i]->Copy(m_hC);
  }
}

template <typename SRC_T, typename DST_T, typename ACCUM_T>
void InitmatrixD(std::vector<shared_ptr<GemmMat<DST_T>>> &m_hD) {
  for (int i = 0; i < g_device_num; i++) {
    m_hD.push_back(std::make_shared<GemmMat<DST_T>>(
        params.M, params.N, params.ldc, params.stridec, params.batch_count,
        false, CPU));
    // std::cout << "D = " << m_hD[i].get()->size() << std::endl;
  }

  for (int i = 0; i < g_device_num; i++) {
    CHECK_HIP_ERROR(hipSetDevice(i));
    CHECK_HIP_ERROR(hipDeviceSynchronize());
  }
}
#endif