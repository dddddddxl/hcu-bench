#include "testcase.hpp"
#include "hip/hip_runtime.h"

static std::unordered_map<std::string, void (KernelDispatch::*)()> functionMap =
    {
        {"dispatch_fix_grid", &KernelDispatch::dispatch_fix_grid},
        {"dispatch_fix_block_512", &KernelDispatch::dispatch_fix_block_512},
        {"dispatch_fix_block_1024", &KernelDispatch::dispatch_fix_block_1024},
};

// 用于dispatch测试的空kernel
__global__ void kernel_dispatch() {}

KernelDispatch::KernelDispatch(std::vector<std::string> case_list, Output *output, Device *device, bool clean)
    : Testcase(case_list, 1, 1, output, device) { use_clean = clean; }

void KernelDispatch::run() {
  // 查找并调用对应的函数
  for (auto name : case_list) {
    auto it = functionMap.find(name);
    if (it != functionMap.end()) {
      (this->*(it->second))();
    } else {
      std::cout << "Unknown kernel dispatch case name!" << std::endl;
    }
  }
}

#define MAX_GRID 131072
#define MAX_BLOCK 1024
#define GRID_STEP 2048
#define BLOCK_STEP 64

void KernelDispatch::dispatch_fix_block_1024() {
  int end = MAX_GRID / GRID_STEP;
  int i = 1;
  double avgCyclesPerTG = 0.0;
  double firstElapsedTime = 0.0;
  std::string funcName = "dispatch_fix_block_1024";
  for (; i <= end; ++i) {
    // warm up
    kernel_dispatch<<<GRID_STEP * i, 1024>>>();

    hipEvent_t start, stop;
    CHECK_HIP_ERROR(hipEventCreate(&start));
    CHECK_HIP_ERROR(hipEventCreate(&stop));

    CHECK_HIP_ERROR(hipEventRecord(start));
    kernel_dispatch<<<GRID_STEP * i, 1024>>>();
    CHECK_HIP_ERROR(hipEventRecord(stop));
    CHECK_HIP_ERROR(hipEventSynchronize(stop));
    float elapsedTime = 0.0f;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTime, start, stop));
    CHECK_HIP_ERROR(hipPeekAtLastError());
    output->add_dispatch_record(funcName, GRID_STEP * i, 1024, GRID_STEP * 1024 * i / 64, 1024 / 64, (double)elapsedTime);
    if (i == 1)
      firstElapsedTime = elapsedTime;
    else
      avgCyclesPerTG += (elapsedTime - firstElapsedTime) * device->get_sclk() * 1000.0 / (GRID_STEP * (i - 1));
  }
  avgCyclesPerTG /= end - 1;
  output->add_dispatch_cycles_per_tg(funcName, avgCyclesPerTG);
}

void KernelDispatch::dispatch_fix_block_512() {
  int end = MAX_GRID / GRID_STEP;
  int i = 1;
  double avgCyclesPerTG = 0.0;
  double firstElapsedTime = 0.0;
  std::string funcName = "dispatch_fix_block_512";
  for (; i <= end; ++i) {
    // warm up
    kernel_dispatch<<<GRID_STEP * i, 512>>>();

    hipEvent_t start, stop;
    CHECK_HIP_ERROR(hipEventCreate(&start));
    CHECK_HIP_ERROR(hipEventCreate(&stop));

    CHECK_HIP_ERROR(hipEventRecord(start));
    kernel_dispatch<<<GRID_STEP * i, 512>>>();
    CHECK_HIP_ERROR(hipEventRecord(stop));
    CHECK_HIP_ERROR(hipEventSynchronize(stop));
    float elapsedTime = 0.0f;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTime, start, stop));
    CHECK_HIP_ERROR(hipPeekAtLastError());
    output->add_dispatch_record(funcName, GRID_STEP * i, 512, GRID_STEP * 512 * i / 64, 512 / 64, (double)elapsedTime);
    if (i == 1)
      firstElapsedTime = elapsedTime;
    else
      avgCyclesPerTG += (elapsedTime - firstElapsedTime) * device->get_sclk() * 1000.0 / (GRID_STEP * (i - 1));
  }
  avgCyclesPerTG /= end - 1;
  output->add_dispatch_cycles_per_tg(funcName, avgCyclesPerTG);
}

void KernelDispatch::dispatch_fix_grid() {
  int end = MAX_BLOCK / BLOCK_STEP;
  int i = 1;
  std::string funcName = "dispatch_fix_grid";
  for (; i <= end; ++i) {
    // warm up
    kernel_dispatch<<<MAX_GRID, BLOCK_STEP * i>>>();

    hipEvent_t start, stop;
    CHECK_HIP_ERROR(hipEventCreate(&start));
    CHECK_HIP_ERROR(hipEventCreate(&stop));

    CHECK_HIP_ERROR(hipEventRecord(start));
    kernel_dispatch<<<MAX_GRID, BLOCK_STEP * i>>>();
    CHECK_HIP_ERROR(hipEventRecord(stop));
    CHECK_HIP_ERROR(hipEventSynchronize(stop));
    float elapsedTime = 0.0f;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTime, start, stop));
    CHECK_HIP_ERROR(hipPeekAtLastError());
    output->add_dispatch_record(funcName, MAX_GRID, BLOCK_STEP * i, MAX_GRID * BLOCK_STEP * i / 64, BLOCK_STEP * i / 64, (double)elapsedTime);
  }
}

void KernelDispatch::print() {
  if (use_clean)
    output->print_clean();
  else
    output->print_record();
}

void KernelDispatch::getList() {
  std::cout << std::left << std::setw(30) << "Kernel Dispatch case" << std::endl;
  std::cout << std::left << std::setw(30) << "Usage: ./run kernel_dispatch dispatch_fix_block_1024"
            << std::endl;
  std::cout << "----------------------" << std::endl;
  for (const auto &key : functionMap) {
    std::cout << std::left << std::setw(20) << key.first << std::endl;
  }
}