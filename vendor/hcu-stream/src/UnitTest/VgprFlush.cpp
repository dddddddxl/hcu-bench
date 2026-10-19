#include "testcase.hpp"
#include "UnitTest/safe.hpp"
#include "UnitTest/unsafe.hpp"
#include "hip/hip_runtime.h"

constexpr int FIRST_V = 16;
constexpr int LAST_V  = 255;
constexpr int VGPR_COUNT (LAST_V - FIRST_V + 1);

static std::unordered_map<std::string, void (VGPRNoFlush::*)()> functionMap =
    {
        {"safe", &VGPRNoFlush::safe_kernel},
        {"unsafe", &VGPRNoFlush::unsafe_kernel},
};

VGPRNoFlush::VGPRNoFlush(std::vector<std::string> case_list, Output *output, Device *device, bool clean, int _blocks)
: Testcase(case_list, 4, 1, output, device) {
    blocks = _blocks;
    use_clean = clean;
}

void VGPRNoFlush::run() {
  // 查找并调用对应的函数
  for (auto name : case_list) {
    auto it = functionMap.find(name);
    if (it != functionMap.end()) {
      (this->*(it->second))();
    } else {
      std::cout << "Unknown kernel VGPR_Flush case name!" << std::endl;
    }
  }
}

void VGPRNoFlush::safe_kernel() {
    printf("VGPR inline asm test: kernel1 write v64..v255, kernel2 read and dump to host\n");

    const int threads = wave * 64;
    uint32_t *host_buf1 = (uint32_t *)malloc(VGPR_COUNT * sizeof(uint32_t));
    uint32_t *host_buf2 = (uint32_t *)malloc(VGPR_COUNT * 64 * sizeof(uint32_t));
    for (int i = 0; i < VGPR_COUNT * 64; i++){
        host_buf1[i % VGPR_COUNT] = 4;
        host_buf2[i] = 4;
    }

    uint32_t *device_buf1 = nullptr, *device_buf2 = nullptr;    
    hipMalloc(&device_buf1, VGPR_COUNT * sizeof(uint32_t));
    hipMalloc(&device_buf2, 64 * VGPR_COUNT * sizeof(uint32_t));
    hipMemcpy(device_buf1, host_buf1, VGPR_COUNT * sizeof(uint32_t), hipMemcpyHostToDevice);

    // Launch kernel1
    printf("Launching kernel1 (blocks=%d threads=%d) ...\n", blocks, threads);
    hipLaunchKernelGGL(kernel1_write_vgprs_and_busy_safe, dim3(blocks), dim3(threads), 0, 0, busy_loops, device_buf1, VGPR_COUNT);
    hipDeviceSynchronize();
   
    hipError_t e =hipMemcpy(host_buf1, device_buf1, VGPR_COUNT * sizeof(uint32_t), hipMemcpyDeviceToHost);
    hipMemcpy(device_buf2, host_buf2, 64 * VGPR_COUNT * sizeof(uint32_t), hipMemcpyHostToDevice);

    // Launch kernel2 (single wave: 1 block of 64 threads)
    printf("Launching kernel2 to read VGPRs ...\n");
    hipLaunchKernelGGL(kernel2_read_vgprs_to_mem_safe, dim3(1), dim3(64), 0, 0, device_buf2, VGPR_COUNT);
    hipDeviceSynchronize();
    e =hipMemcpy(host_buf2, device_buf2, 64 * VGPR_COUNT * sizeof(uint32_t), hipMemcpyDeviceToHost);
    if (e != hipSuccess) {
        printf("hipMemcpyFromSymbol failed: %d\n", e);
        return;
    }

    std::vector<uint32_t> vgpr_results;
    for (int i = 48; i < VGPR_COUNT; ++i) {
        uint32_t value = host_buf2[0 * VGPR_COUNT + i];
        vgpr_results.push_back(value);
    }
    
    // 存储结果到 output
    VGPRSafe_Output *vgpr_output = dynamic_cast<VGPRSafe_Output*>(output);
    if (vgpr_output) {
        vgpr_output->add_vgpr_record("safe", vgpr_results);
    }
}

void VGPRNoFlush::unsafe_kernel() {
    printf("VGPR inline asm test: kernel1 write v16..v255, kernel2 read and dump to host\n");

    const int threads = wave * 64;
    uint32_t *host_buf1 = (uint32_t *)malloc(VGPR_COUNT * sizeof(uint32_t));
    uint32_t *host_buf2 = (uint32_t *)malloc(VGPR_COUNT * 64 * sizeof(uint32_t));
    for (int i = 0; i < VGPR_COUNT * 64; i++){
	host_buf1[i % VGPR_COUNT] = 4;
	host_buf2[i] = 4;
    }

    uint32_t *device_buf1 = nullptr, *device_buf2 = nullptr;    
    hipMalloc(&device_buf1, VGPR_COUNT * sizeof(uint32_t));
    hipMalloc(&device_buf2, 64 * VGPR_COUNT * sizeof(uint32_t));
    hipMemcpy(device_buf1, host_buf1, VGPR_COUNT * sizeof(uint32_t), hipMemcpyHostToDevice);

    // Launch kernel1
    printf("Launching kernel1 (blocks=%d threads=%d) ...\n", blocks, threads);
    hipLaunchKernelGGL(kernel1_write_vgprs_and_busy_unsafe, dim3(blocks), dim3(threads), 0, 0, busy_loops, device_buf1, VGPR_COUNT);
    hipDeviceSynchronize();
    hipError_t e =hipMemcpy(host_buf1, device_buf1, VGPR_COUNT * sizeof(uint32_t), hipMemcpyDeviceToHost);
    hipMemcpy(device_buf2, host_buf2, 64 * VGPR_COUNT * sizeof(uint32_t), hipMemcpyHostToDevice);

    // Launch kernel2 (single wave: 1 block of 64 threads)
    printf("Launching kernel2 to read VGPRs ...\n");
    hipLaunchKernelGGL(kernel2_read_vgprs_to_mem_unsafe, dim3(1), dim3(64), 0, 0, device_buf2, VGPR_COUNT);
    hipDeviceSynchronize();
    e =hipMemcpy(host_buf2, device_buf2, 64 * VGPR_COUNT * sizeof(uint32_t), hipMemcpyDeviceToHost);
    if (e != hipSuccess) {
        printf("hipMemcpyFromSymbol failed: %d\n", e);
        return;
    }

    std::vector<uint32_t> vgpr_results;
    for (int i = 48; i < VGPR_COUNT; ++i) {
        uint32_t value = host_buf2[0 * VGPR_COUNT + i];
        vgpr_results.push_back(value);
    }
    
    // 存储结果到 output
    VGPRSafe_Output *vgpr_output = dynamic_cast<VGPRSafe_Output*>(output);
    if (vgpr_output) {
        vgpr_output->add_vgpr_record("unsafe", vgpr_results);
    }

}


void VGPRNoFlush::print() {
  VGPRSafe_Output *vgpr_output = dynamic_cast<VGPRSafe_Output*>(output);
  if (vgpr_output) {
    if (use_clean)
      vgpr_output->print_clean();
    else
      vgpr_output->print_record();
  }
}

void VGPRNoFlush::getList() {
  std::cout << std::left << std::setw(30) << "Kernel VGPR_NO_FLUSH case" << std::endl;
  std::cout << std::left << std::setw(30) << "Usage: ./run VGPR safe"
            << std::endl;
  std::cout << "----------------------" << std::endl;
  for (const auto &key : functionMap) {
    std::cout << std::left << std::setw(20) << key.first << std::endl;
  }
}