#include "paMemory.hpp"
#include "testcase.hpp"

paMemory::paMemory(Output *output, Device *device, bool clean, int pad,
                   int height, int width)
    : MemoryBandwidth({"pa"}, /*wave*/ 4, /*crop*/ 1, output, device,
                      /*lds_size*/ 4, clean, 0 /*arraysize*/, false, false) {
  this->pad = pad;
  this->h = height;
  this->w = width;
}

void paMemory::run() {
    run_pabenchmark(h, w);
}
void paMemory::runAll() {
  std::vector<int> _height = {4096, 8192};
  int _width = 128;

  // std::cout << "\nwidth: " << width << std::endl;
  for (int i = 0; i < _height.size(); i++) {
    run_pabenchmark(_height[i], _width);
  }
}

void paMemory::getList() {
  std::cout << std::left << std::setw(30) << "paMemory case" << std::endl;
  std::cout << std::left << std::setw(30) << "Usage: ./run pa" << std::endl;
  std::cout << "----------------------" << std::endl;
}

void paMemory::print() { output->print_record(); }
void paMemory::run_pabenchmark(int height, int width) {
  int THREADS_PER_BLOCK = wave * 64;
  int *inArray;
  uint64_t MEM_ARRAY_SIZE = (width + pad) * height;
  uint64_t real_size = width * height;
  inArray = new int[MEM_ARRAY_SIZE]; // 1.0GB

  int *inArray_g;
  HIP_ASSERT(hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(int)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(int),
                       hipMemcpyHostToDevice));

  uint64_t data = real_size * sizeof(int);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
#if HIP_ARCH >= 936
  // warm up
  for (int i = 0; i < 20; ++i) {
    hipLaunchKernelGGL(paTest, dim3(real_size / THREADS_PER_BLOCK),
                       dim3(THREADS_PER_BLOCK), 4 * 1024, 0, inArray_g, width,
                       height, pad);
    HIP_ASSERT(hipDeviceSynchronize());
  }
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    t1 = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(paTest, dim3(real_size / THREADS_PER_BLOCK),
                       dim3(THREADS_PER_BLOCK), 4 * 1024, 0, inArray_g, height,
                       width, pad);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    t = duration.count();
    if (i == 0) {
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    // std::cout << i << " : " << bw << std::endl;
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
                       avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), 4, device->device_index);
  }
}
