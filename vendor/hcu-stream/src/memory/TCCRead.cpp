
#include "memoryBandwidth/init_kernel.hpp"
#include "memoryBandwidth/tcc_read.hpp"
#include "testcase.hpp"

TCCRead::TCCRead(Output *output, Device *device)
    : MemoryBandwidth({"L2CacheRW"}, 4, 1, output, device, 0, false, 0, false,
                      false) {}

void TCCRead::run() { run_tcc_read(); }

void TCCRead::getList() {
  std::cout << std::left << std::setw(30) << "TCCRead case: only one case"
            << std::endl;
  std::cout << std::left << std::setw(30) << "Usage: ./run tcc_read"
            << std::endl;
}

void TCCRead::print() { output->print_record(); }

std::vector<double> TCCRead::run_kernel(int height, uint32_t width) {

  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1000>> duration;
  std::vector<double> timings;
  for (uint32_t k = 0; k < 100; k++) {
    t1 = std::chrono::high_resolution_clock::now();
    dim3 grid_size(height / TCC_BLOCK_DIM_Y, 1, 1);
    dim3 block_size(TCC_BLOCK_DIM_X, TCC_BLOCK_DIM_Y, 1);
    stride_read_kernel<<<grid_size, block_size, 0, 0>>>(src, dst, width);
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    timings.push_back(duration.count());
  }
  return timings;
}

void TCCRead::run_tcc_read() {
  const int BLOCK_SIZE = 256;
  int height = 204800;
  for (uint32_t i = TCC_BLOCK_DIM_X * sizeof(__int128_t) / TCC_WIDTH; i < 64;
       i++) {
    uint32_t data_size = height * i * TCC_WIDTH;
    CHECK_HIP_ERROR(hipMalloc((void **)&src, data_size));
    CHECK_HIP_ERROR(hipMalloc((void **)&dst, data_size));
    __int128_t init_value = 1;
    uint32_t data_count = data_size / sizeof(__int128_t);
    init_kernel<<<dim3(data_count / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
        src, init_value);
    init_kernel<<<dim3(data_count / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
        dst, init_value);

    std::string func_name = "StrideRead_" + std::to_string(i) + " TCC";
    uint32_t width = i * TCC_WIDTH / sizeof(__int128_t);
    auto timings = run_kernel(height, width);
    double data = height * TCC_BLOCK_DIM_X * sizeof(__int128_t);
    // auto minmax = std::minmax_element(timings.begin() + 1, timings.end());
    // Calculate average; ignore the first result
    double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                     (double)(timings.size() - 1);
    // Display results
    double average_speed = pow(2.0, -20.0) * data / average;

    output->add_record(func_name, 4, data * pow(2.0, -20.0), average_speed,
                       average, 0, 0, device->device_index);
    CHECK_HIP_ERROR(hipFree(src));
    CHECK_HIP_ERROR(hipFree(dst));
  }
}