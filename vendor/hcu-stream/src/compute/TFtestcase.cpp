#include "testcase.hpp"

#include "compute/u8.hpp"
#include "compute/i8.hpp"
#include "compute/bf16.hpp"
#include "compute/f16.hpp"
#include "compute/f32x4.hpp"
#include "compute/f32x8.hpp"
#include "compute/tf32.hpp"
#include "compute/f64.hpp"
#include "compute/i4.hpp"
#include "compute/u4.hpp"
#include "compute/fp8_fp8.hpp"
#include "compute/bf8_bf8.hpp"
#include "compute/fp8_bf8.hpp"
#include "compute/bf8_fp8.hpp"
#include "compute/mix_f8f6f4.hpp"
#include "compute/mix_scale_f8f6f4.hpp"
#include "compute/fp4.hpp"
#include "compute/scale_fp4.hpp"

static std::unordered_map<std::string, void (TFLOPsCase::*)()> functionMap = {



#if HIP_ARCH >= 926 && HIP_ARCH != 928 && HIP_ARCH != 9210
  {"f64", &TFLOPsCase::run_f64},
#endif

#if HIP_ARCH >= 926
  {"f32x4", &TFLOPsCase::run_f32x4},
#endif

#if HIP_ARCH >= 926 && HIP_ARCH <= 938
  {"f32x8", &TFLOPsCase::run_f32x8},
#endif

#if HIP_ARCH >= 928
    {"u8", &TFLOPsCase::run_u8},
    {"i8", &TFLOPsCase::run_i8},
    {"bf16", &TFLOPsCase::run_bf16},
    {"f16", &TFLOPsCase::run_f16},
    {"tf32", &TFLOPsCase::run_tf32},
#endif
#if HIP_ARCH >= 936 && HIP_ARCH != 9210
    {"i4", &TFLOPsCase::run_i4},
    {"u4", &TFLOPsCase::run_u4},
#endif

#if HIP_ARCH >= 938 
    {"fp8_fp8", &TFLOPsCase::run_fp8_fp8},
    {"bf8_bf8", &TFLOPsCase::run_bf8_bf8},
    {"fp8_bf8", &TFLOPsCase::run_fp8_bf8},
    {"bf8_fp8", &TFLOPsCase::run_bf8_fp8},
#endif

#if HIP_ARCH >= 946  // include 92a(9210)
    {"mix_f8f6f4", &TFLOPsCase::run_mix_f8f6f4},
#endif

#if HIP_ARCH == 946
    {"mix_scale_f8f6f4", &TFLOPsCase::run_mix_scale_f8f6f4},
    {"fp4",  &TFLOPsCase::run_fp4},
    {"scale_fp4",  &TFLOPsCase::run_scale_fp4},
#endif

};

struct Record {
  std::string DCU_arch;
  std::string INST_name;
  int cycle;
  int M;
  int N;
  int K;
};

const std::vector<Record> records = {
    // KM
    {"926", "v_mmac_16x16x4_f32", 64/*cycle*/, 16, 16, 4},
    {"926", "v_mmac_16x16x4_f64", 32/*cycle*/, 16, 16, 4 },
    // ZDE cycle
    {"928", "v_mmac_16x16x4_f32", 64/*cycle*/, 16, 16, 4},
    {"928", "v_mmac_16x16x8_f32", 64/*cycle*/, 16, 16, 8},
    {"928", "v_mmac_f32_16x16x8_tf32", 32, 16, 16, 8},
    {"928", "v_mmac_f32_16x16x16_f16", 32, 16, 16, 16 },
    {"928", "v_mmac_f32_16x16x16_bf16", 32, 16, 16, 16 },
    {"928", "v_mmac_i32_16x16x32_i8", 32, 16, 16, 32 },
    {"928", "v_mmac_i32_16x16x32_u8", 32, 16, 16, 32 },
    // BMZ cycle
    {"936", "v_mmac_16x16x4_f64", 32/*cycle*/, 16, 16, 4 },
    {"936", "v_mmac_16x16x4_f32", 32/*cycle*/, 16, 16, 4 },
    {"936", "v_mmac_16x16x8_f32", 32, 16, 16, 8 },
    {"936", "v_mmac_f32_16x16x8_tf32", 8, 16, 16, 8 },
    {"936", "v_mmac_f32_16x16x16_f16", 8, 16, 16, 16 },
    {"936", "v_mmac_f32_16x16x16_bf16", 8, 16, 16, 16 },
    {"936", "v_mmac_i32_16x16x32_i8", 8, 16, 16, 32 },
    {"936", "v_mmac_i32_16x16x32_u8", 8, 16, 16, 32  },
    {"936", "v_mmac_i32_16x16x64_i4", 8, 16, 16, 64 },
    {"936", "v_mmac_i32_16x16x64_u4", 8, 16, 16, 64 },
    // NMZ cycle
    {"938", "v_mmac_16x16x4_f64", 32/*cycle*/, 16, 16, 4},
    {"938", "v_mmac_16x16x4_f32", 32/*cycle*/, 16, 16, 4},
    {"938", "v_mmac_16x16x8_f32", 32, 16, 16, 8},
    {"938", "v_mmac_f32_16x16x8_tf32", 8, 16, 16, 8},
    {"938", "v_mmac_f32_16x16x16_f16", 8, 16, 16, 16},
    {"938", "v_mmac_f32_16x16x16_bf16", 8, 16, 16, 16},
    {"938", "v_mmac_i32_16x16x32_i8", 8, 16, 16, 32},
    {"938", "v_mmac_i32_16x16x32_u8", 8, 16, 16, 32},
    {"938", "v_mmac_i32_16x16x64_i4", 16, 16, 16, 64},
    {"938", "v_mmac_i32_16x16x64_u4", 16/*cycle*/, 16, 16, 64},
    {"938", "v_mmac_f32_16x16x32_fp8_fp8", 8, 16, 16, 32},
    {"938", "v_mmac_f32_16x16x32_bf8_bf8", 8, 16, 16, 32},
    {"938", "v_mmac_f32_16x16x32_fp8_bf8", 8, 16, 16, 32},
    {"938", "v_mmac_f32_16x16x32_bf8_fp8", 8, 16, 16, 32},
    // YY
    // {"9210", "v_mmac_16x16x4_f64", 32/*cycle*/, 16, 16, 4},
    {"9210", "v_mmac_16x16x4_f32", 64/*cycle*/, 16, 16, 4},
    // {"9210", "v_mmac_16x16x8_f32", 32, 16, 16, 8},
    {"9210", "v_mmac_f32_16x16x8_tf32", 32, 16, 16, 8},
    {"9210", "v_mmac_f32_16x16x16_f16", 32, 16, 16, 16},
    {"9210", "v_mmac_f32_16x16x16_bf16", 32, 16, 16, 16},
    {"9210", "v_mmac_i32_16x16x32_i8", 32, 16, 16, 32},
    {"9210", "v_mmac_i32_16x16x32_u8", 32, 16, 16, 32},
    // {"9210", "v_mmac_i32_16x16x64_i4", 16, 16, 16, 64},
    // {"9210", "v_mmac_i32_16x16x64_u4", 16/*cycle*/, 16, 16, 64},
    {"9210", "v_mmac_f32_16x16x32_fp8_fp8", 32, 16, 16, 32},
    {"9210", "v_mmac_f32_16x16x32_bf8_bf8", 32, 16, 16, 32},
    {"9210", "v_mmac_f32_16x16x32_fp8_bf8", 32, 16, 16, 32},
    {"9210", "v_mmac_f32_16x16x32_bf8_fp8", 32, 16, 16, 32},
    {"9210", "v_mmac_f32_16X16X32_f8f6f4_f8f6f4", 32, 16, 16, 32},

    // SHB
    {"946", "v_mmac_16x16x4_f64", 256/*cycle*/, 16, 16, 4},
    {"946", "v_mmac_16x16x4_f32", 32/*cycle*/, 16, 16, 4},
    // {"946", "v_mmac_16x16x8_f32", 32, 16, 16, 8},
    {"946", "v_mmac_f32_16x16x8_tf32", 8, 16, 16, 8},
    {"946", "v_mmac_f32_16x16x16_f16", 8, 16, 16, 16},
    {"946", "v_mmac_f32_16x16x16_bf16", 8, 16, 16, 16},
    {"946", "v_mmac_i32_16x16x32_i8", 8, 16, 16, 32},
    {"946", "v_mmac_i32_16x16x32_u8", 8, 16, 16, 32},
    {"946", "v_mmac_i32_16x16x64_i4", 16, 16, 16, 64},
    {"946", "v_mmac_i32_16x16x64_u4", 16/*cycle*/, 16, 16, 64},
    {"946", "v_mmac_f32_16x16x32_fp8_fp8", 8, 16, 16, 32},
    {"946", "v_mmac_f32_16x16x32_bf8_bf8", 8, 16, 16, 32},
    {"946", "v_mmac_f32_16x16x32_fp8_bf8", 8, 16, 16, 32},
    {"946", "v_mmac_f32_16x16x32_bf8_fp8", 8, 16, 16, 32},
    {"946", "v_mmac_f32_16X16X32_f8f6f4_f8f6f4", 8, 16, 16, 32},
    {"946", "v_mmac_f32_16X16X64_fp4", 8, 16, 16, 64},
    {"946", "v_mmac_scale_f32_16x16x32_f8f6f4_f8f6f4", 8, 16, 16, 32},
    {"946", "v_mmac_scale_f32_16x16x64_fp4", 8, 16, 16, 64},

};

TFLOPsCase::TFLOPsCase(std::vector<std::string> case_list, int wave, double crop,
                       Output *output, Device *device)
    : Testcase(case_list, wave, crop, output, device) {}

void TFLOPsCase::run() {
  // 查找并调用对应的函数
  for (auto name : case_list) {
    auto it = functionMap.find(name);
    if (it != functionMap.end()) {
      (this->*(it->second))();
    } else {
      std::cout << "Unknown TFLOPs case name!" << std::endl;
    }
  }
}

void TFLOPsCase::getList() {
  std::cout << std::left << std::setw(30) << "compute case" << std::endl;
  std::cout << std::left << std::setw(30) << "Usage: ./run tflops bf16"
            << std::endl;
  std::cout << "----------------------" << std::endl;
  for (const auto &key : functionMap) {
    std::cout << std::left << std::setw(20) << key.first << std::endl;
  }
}

double TFLOPsCase::getTheory(const std::string &inst, int cuNum) {
  string arch;
#if HIP_ARCH == 928
  arch = "928";
#elif HIP_ARCH == 936
  arch = "936";
#elif HIP_ARCH == 938
  arch = "938";
#elif HIP_ARCH == 9210
  arch = "9210";
#elif HIP_ARCH == 946
  arch = "946";
#else
  assert(0 && "unsupport arch !");
#endif

  double sclk = device->get_sclk();
  double theory = 0.0;
  for (const auto &record : records) {
    if (record.DCU_arch == arch && record.INST_name == inst) {
      theory = double(record.M * record.N * record.K * 2 * cuNum * 4 * sclk) /
               double(record.cycle * 1000 * 1000);
      return theory;
    }
  }
  assert(0 && "not support Inst for arch");
  return 0.0;
}


#if HIP_ARCH >= 926 && HIP_ARCH != 928 && HIP_ARCH != 9210
void TFLOPsCase::run_f64() {
  run_f64gemm(10, 1);
}
#endif

// gfx926 support f32x4 and f64(928 not support f64)
#if HIP_ARCH >= 926 
void TFLOPsCase::run_f32x4() {
  run_f32x4gemm(10, 1);
}
#endif

#if HIP_ARCH >= 926 && HIP_ARCH <= 938
void TFLOPsCase::run_f32x8() {
  run_f32x8gemm(10, 1);
}
#endif

#if HIP_ARCH >= 928
void TFLOPsCase::run_u8() {
  run_u8gemm(10, 1);
}
void TFLOPsCase::run_i8() {
  run_i8gemm(10, 1);
}
void TFLOPsCase::run_bf16() {
  run_bf16gemm(10, 1);
}
void TFLOPsCase::run_f16() {
  run_f16gemm(10, 1);
}
void TFLOPsCase::run_tf32() {
  run_tf32gemm(10, 1);
}
#endif

#if HIP_ARCH >= 936 && HIP_ARCH != 9210
void TFLOPsCase::run_i4() {
  run_i4gemm(10, 1);
}
void TFLOPsCase::run_u4() {
  run_u4gemm(10, 1);
}
#endif

#if HIP_ARCH >= 938
void TFLOPsCase::run_fp8_fp8() {
  run_fp8_fp8gemm(10, 1);
}
void TFLOPsCase::run_bf8_bf8() {
  run_bf8_bf8gemm(10, 1);
}
void TFLOPsCase::run_fp8_bf8() {
  run_fp8_bf8gemm(10, 1);
}
void TFLOPsCase::run_bf8_fp8() {
  run_bf8_fp8gemm(10, 1);
}
#endif

#if HIP_ARCH >= 946  // include 92a(9210)
  void TFLOPsCase::run_mix_f8f6f4() {
    run_mix_f8f6f4gemm(10, 1);
  }
#endif

#if HIP_ARCH == 946
  void TFLOPsCase::run_mix_scale_f8f6f4() {
    run_mix_scale_f8f6f4gemm(10, 1);
  }
  void TFLOPsCase::run_fp4() {
    run_fp4gemm(10, 1);
  }
  void TFLOPsCase::run_scale_fp4() {
    run_scale_fp4gemm(10, 1);
  }
#endif