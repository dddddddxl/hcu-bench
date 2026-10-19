
#include "memoryBandwidth/L2_cache_rw.hpp"
#include "testcase.hpp"

L2CacheRW::L2CacheRW(Output *output, Device *device)
    : MemoryBandwidth({"L2CacheRW"}, 4, 1, output, device, 0, false, 0, false,
                      false) {}

void L2CacheRW::run() {
  // 查找并调用对应的函数
  // std::cout<< "debug1"<<std::endl;
  cache_rw();
}

void L2CacheRW::getList() {
  std::cout << std::left << std::setw(30) << "L2CacheRW case: only one case"
            << std::endl;
  std::cout << std::left << std::setw(30) << "Usage: ./run L2cache"
            << std::endl;
}

void L2CacheRW::print() { output->print_record(); }

std::vector<double> L2CacheRW::L2_read_rw(int iter, int parts) {
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1000>> duration;
  std::vector<double> timings;
  for (uint32_t k = 0; k < 100; k++) {
    t1 = std::chrono::high_resolution_clock::now();
    L2_read_kernel<<<FIX_BLOCK_NUM, L2_BLOCK_SIZE /*256*/, 0, 0>>>(src, dst,
                                                                   iter, parts);
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    timings.push_back(duration.count());
  }
  return timings;
}

void L2CacheRW::cache_rw() {
  int iter = 64;
  int idx = 0;
  double trans_size = iter * L2_BLOCK_SIZE * FIX_BLOCK_NUM * sizeof(__int128_t);
  for (int i = 3; i < 10000; i += max(2.0, i * 0.1)) {
    uint32_t data_size = iter * L2_BLOCK_SIZE * i * sizeof(__int128_t);
    CHECK_HIP_ERROR(hipMalloc((void **)&src, data_size));
    CHECK_HIP_ERROR(hipMalloc((void **)&dst, data_size));
    auto timings = L2_read_rw(iter, i);
    std::string func_name = "L2Read_" + std::to_string(idx++);
    auto minmax = std::minmax_element(timings.begin() + 1, timings.end());
    // Calculate average; ignore the first result
    double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                     (double)(timings.size() - 1);
    // Display results
    double average_speed = pow(2.0, -20.0) * (double)trans_size / average;
    std::cout << "run " << func_name << "........" << std::endl;
    output->add_L2_record(func_name, average_speed /*bandwidth*/,
                          average /*average time*/, data_size, trans_size,
                          *minmax.first, *minmax.second);
    CHECK_HIP_ERROR(hipFree(src));
    CHECK_HIP_ERROR(hipFree(dst));
  }
}