#ifndef __UTILS_H__
#define __UTILS_H__
#include "hip/hip_runtime.h"
#include <iomanip>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
using namespace std;

#define HIP_ASSERT(x) (assert((x) == hipSuccess))
using float4_type = __attribute__((__vector_size__(4 * sizeof(float)))) float;
typedef float float4_ __attribute__((ext_vector_type(4)));
typedef int int4_ __attribute__((ext_vector_type(4)));

#define CHECK_HIP_ERROR(error)                                                 \
  if (error != hipSuccess) {                                                   \
    fprintf(stderr, "Hip error: '%s'(%d) at %s:%d\n",                          \
            hipGetErrorString(error), error, __FILE__, __LINE__);              \
    exit(EXIT_FAILURE);                                                        \
  }

std::string getNum(const std::string &line) {
  size_t end = line.find("Mhz *");
  if (end != std::string::npos) {
    size_t start = end;
    while (std::isdigit(line[start - 1]))
      --start;

    return line.substr(start, end - start);
  }

  return "";
}
double get_sclk() {
  if (std::getenv("MODEL")) {
    char *str = getenv("MODEL");
    std::string model(str);
    int sclk = std::stoi(model);
    if (sclk == 1)
      return 1400.0;
    return (double)sclk;
  } else {
    int deviceNum, deviceId;
    CHECK_HIP_ERROR(hipGetDeviceCount(&deviceNum));
    CHECK_HIP_ERROR(hipGetDevice(&deviceId));
    int maxCard = -1;
    double sclk;
    std::vector<std::string> lines;
    std::string line;
    std::string cardName;
    std::string sclkPath;

    for (int i = 0; i < 50; i++) {
      cardName = "card" + std::to_string(i);
      sclkPath = "/sys/class/drm/" + cardName + "/device/pp_dpm_sclk";
      std::ifstream file(sclkPath.c_str());
      if (file.good())
        maxCard = i;
    }
    if (maxCard == -1) {
      assert(0 && "no pp_dpm_sclk in /sys/class/drm/");
    }
    // std::cout << "maxCard: " << maxCard << "; deviceNum: " << deviceNum << ";
    // deviceId " << deviceId << std::endl;
    int cardId = maxCard - deviceNum + deviceId + 1;
    cardName = "card" + std::to_string(cardId);
    sclkPath = "/sys/class/drm/" + cardName + "/device/pp_dpm_sclk";
    std::ifstream file(sclkPath.c_str());
    if (!file.is_open())
      throw std::runtime_error("can not open: " + sclkPath);
    while (std::getline(file, line)) {
      lines.push_back(line);
    }
    for (const auto &line : lines) {
      std::string str = getNum(line);
      if (!str.empty())
        sclk = (double)std::stoi(str);
    }

    return sclk;
  }
}

int getInstK(int gemm_type) {
  int K = 0;
  switch (gemm_type) {
  case 0:
  case 1:
    K = 16;
    break;
  case 2:
  case 3:
    K = 8;
    break;
  case 4:
  case 5:
    K = 32;
    break;
  case 6:
    K = 4;
    break;
  case 7:
  case 8:
    K = 64;
    break;
  case 9:
    K = 32;
    break;
  default:
    std::string error_code = "Fatal: gemm type setting error!";
    throw std::runtime_error(error_code);
  }
  return K;
}

double getTflops(uint64_t *m_clock_start, uint64_t *m_clock_end, int m_wave_num,
                 int gemm_type) {
  double *total_t = new double[m_wave_num];
  double tflops = 0;
  double sclk = get_sclk() * 1e-3;
  for (uint i = 0; i < m_wave_num; ++i) {
    total_t[i] = (double)m_clock_end[64 * i] - (double)m_clock_start[64 * i];
    double wave_tflop =
        16 * 16 * getInstK(gemm_type) * 2 * 51200 * 1e-3 / (total_t[i] / sclk);
    tflops += wave_tflop;
  }
  // double t = (*std::max_element(total_t, total_t + 8)) / (sclk * 1e9);
  delete[] total_t;
  return tflops;
}
int StatFile(const char *file_path, std::string &msg, bool exit) {
  struct stat st;
  int stat_status = stat(file_path, &st);
  if (stat_status == -1) {
    if (exit)
      throw std::runtime_error(msg);
    else
      return stat_status;
  }
  return stat_status;
}
std::string GetKernelName(int gemm_type) {
  std::string kernel_name;
  switch (gemm_type) {
  case 0:
    kernel_name = "hgemm";
    break;
  case 1:
    kernel_name = "bf16gemm";
    break;
  case 2:
    kernel_name = "sgemm";
    break;
  case 3:
    kernel_name = "tf32gemm";
    break;
  case 4:
    kernel_name = "i8gemm";
    break;
  case 5:
    kernel_name = "u8gemm";
    break;
  case 6:
    kernel_name = "dgemm";
    break;
  case 7:
    kernel_name = "i4gemm";
    break;
  case 8:
    kernel_name = "u4gemm";
    break;
  case 9:
    kernel_name = "fp8_fp8gemm";
    break;
  case 10:
    kernel_name = "fp8_bf8gemm";
    break;
  case 11:
    kernel_name = "bf8_fp8gemm";
    break;
  case 12:
    kernel_name = "bf8_bf8gemm";
  default:
    std::string error_code = "Fatal: gemm type setting error!";
    throw std::runtime_error(error_code);
  }
  return kernel_name;
}

#endif