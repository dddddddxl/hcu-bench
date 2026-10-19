#include "gemmMemory.hpp"
#include "testcase.hpp"

gemmMemory::gemmMemory(Output *output, Device *device, bool clean, int lds,
                       int height, int width)
    : MemoryBandwidth({"gemm"}, /*wave*/ 4, /*crop*/ 1, output, device,
                      /*lds_size*/ lds, clean, 0, false, false) {
  this->h = height;
  this->w = width;
}

void gemmMemory::runAll() {
  std::vector<int> _height = {3072,  4096,  4992,  6144,  8192,  9984,  12288,
                             13312, 17792, 19008, 19968, 24576, 35584, 38016};
  std::vector<int> _width = {1024, 1664, 2048, 3072,  3328,  4096,  4448,
                            6144, 8192, 8896, 11008, 12288, 13312, 17792};

  for (int j = 0; j < _width.size(); j++) {
    std::cout << "\nwidth: " << _width[j] << std::endl;
    for (int i = 0; i < _height.size(); i++) {
      run_Gemmbenchmark(_height[i], _width[j]);
    }
  }
}
void gemmMemory::run() {
  run_Gemmbenchmark(h, w);
}

void gemmMemory::getList() {
  std::cout << std::left << std::setw(30) << "gemmMemory case" << std::endl;
  std::cout << std::left << std::setw(30) << "Usage: ./run gemm" << std::endl;
  std::cout << "----------------------" << std::endl;
}

void gemmMemory::print() { output->print_record(); }
void gemmMemory::run_Gemmbenchmark(int height, int width) {
  int THREADS_PER_BLOCK = wave * 64;
  __int128_t *inArray;
  uint64_t MEM_ARRAY_SIZE = width * height;
  inArray = new __int128_t[MEM_ARRAY_SIZE]; // 1.0GB

  __int128_t *inArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int128_t);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
#if HIP_ARCH >= 936
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    t1 = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(gemmTest, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK),
                       dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g, height,
                       width);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    t = duration.count();
    if (i == 0) {
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    // std::cout<<i<<" : "<<bw<<std::endl;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
#endif
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));
  if (use_clean) {
    std::cout << avg_bw / (LAUNCH_TIMES - 1) << std::endl;
  } else {
    output->add_record(to_string(height) + "x" + to_string(width), wave,
                       data * pow(10.0, -6.0), avg_bw / (LAUNCH_TIMES - 1),
                       avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
  }
}
