#include "memoryBandwidth/init_kernel.hpp"
#include "memoryBandwidth/lds_l1_l2.hpp"
#include "memoryBandwidth/lds_to_memory.hpp"
#include "memoryBandwidth/mapping_matrix_write.hpp"
#include "memoryBandwidth/matrix_pattern.hpp"
#include "memoryBandwidth/memory_copy.hpp"
#include "memoryBandwidth/memory_read.hpp"
#include "memoryBandwidth/memory_to_lds.hpp"
#include "memoryBandwidth/memory_write.hpp"
#include "memoryBandwidth/power.hpp"
#include "memoryBandwidth/tiling_read.hpp"
#include "moe.hpp"
#include "testcase.hpp"

static std::unordered_map<std::string, void (MemoryBandwidth::*)()>
    functionMap = {
#if HIP_ARCH == 936
        {"HBM_GCpower", &MemoryBandwidth::HBM_GCpower},
        {"HBMpower", &MemoryBandwidth::HBMpower},
        {"GCpower", &MemoryBandwidth::GCpower},
#endif
        {"memory_readx1_x1", &MemoryBandwidth::memory_readx1_x1},
        {"memory_readx1_x4", &MemoryBandwidth::memory_readx1_x4},
        {"memory_readx1_x8192", &MemoryBandwidth::memory_readx1_x8192},
        {"memory_readx1_x8192_80cu",
         &MemoryBandwidth::memory_readx1_x8192_80cu},

        {"memory_readx2_x1", &MemoryBandwidth::memory_readx2_x1},
        {"memory_readx2_x2", &MemoryBandwidth::memory_readx2_x2},
        {"memory_readx2_x4", &MemoryBandwidth::memory_readx2_x4},
        {"memory_readx2_x4096", &MemoryBandwidth::memory_readx2_x4096},
        {"memory_readx2_x4096_80cu",
         &MemoryBandwidth::memory_readx2_x4096_80cu},

        {"memory_readx4_x1", &MemoryBandwidth::memory_readx4_x1},
        {"memory_readx4_x2", &MemoryBandwidth::memory_readx4_x2},
        {"memory_readx4_x4", &MemoryBandwidth::memory_readx4_x4},
        {"memory_readx4_x8_wave_continue", &MemoryBandwidth::memory_readx4_x8_wave_continue},
        {"memory_readx4_x8_wave_continue_8wave", &MemoryBandwidth::memory_readx4_x8_wave_continue_8wave},
        {"memory_readx4_x8_wave_continue_16wave", &MemoryBandwidth::memory_readx4_x8_wave_continue_16wave},
#if HIP_ARCH >= 936
        {"memory_readx4_x8_wave_continue_bps", &MemoryBandwidth::memory_readx4_x8_wave_continue_bps},
        {"memory_readx4_x8_wave_continue_8wave_bps", &MemoryBandwidth::memory_readx4_x8_wave_continue_8wave_bps},
        {"memory_readx4_x8_wave_continue_16wave_bps", &MemoryBandwidth::memory_readx4_x8_wave_continue_16wave_bps},
#endif
        {"memory_readx4_x8_inst_continue", &MemoryBandwidth::memory_readx4_x8_inst_continue},
        {"memory_readx4_x16", &MemoryBandwidth::memory_readx4_x16},
        {"memory_readx4_x4", &MemoryBandwidth::memory_readx4_x4},
        {"memory_readx4_x64", &MemoryBandwidth::memory_readx4_x64},
        {"memory_readx4_x2048", &MemoryBandwidth::memory_readx4_x2048},
        {"memory_readx4_x2048_80cu",
         &MemoryBandwidth::memory_readx4_x2048_80cu},

        {"memory_write", &MemoryBandwidth::memory_write},
        {"memory_writex1_x1", &MemoryBandwidth::memory_writex1_x1},
        {"memory_writex1_x4", &MemoryBandwidth::memory_writex1_x4},
        {"memory_writex2_x1", &MemoryBandwidth::memory_writex2_x1},
        {"memory_writex2_x2", &MemoryBandwidth::memory_writex2_x2},

        {"memory_writex2_x4", &MemoryBandwidth::memory_writex2_x4},
        {"memory_writex4_x2", &MemoryBandwidth::memory_writex4_x2},
        {"memory_writex4_x4", &MemoryBandwidth::memory_writex4_x4},
        {"memory_writex4_x8_wave_continue", &MemoryBandwidth::memory_writex4_x8_wave_continue},
        {"memory_writex4_x8_wave_continue_8wave", &MemoryBandwidth::memory_writex4_x8_wave_continue_8wave},
        {"memory_writex4_x8_wave_continue_16wave", &MemoryBandwidth::memory_writex4_x8_wave_continue_16wave},
        {"memory_writex4_x8_inst_continue", &MemoryBandwidth::memory_writex4_x8_inst_continue},

        {"memory_copy", &MemoryBandwidth::memory_copy},
        {"memory_copyx2_x1", &MemoryBandwidth::memory_copyx2_x1},
        {"memory_copy_x2", &MemoryBandwidth::memory_copy_x2},
        {"memory_copyx4_x4", &MemoryBandwidth::memory_copyx4_x4},
        {"memory_copyx4_x4_v2", &MemoryBandwidth::memory_copyx4_x4_wave_continue},
        {"memory_copyx4_x8_wave_continue", &MemoryBandwidth::memory_copyx4_x8_wave_continue},
        {"memory_copyx4_x8_wave_asm_continue", &MemoryBandwidth::memory_copyx4_x8_wave_continue_asm},
        {"memory_copyx4_x8_wave_continue_8wave", &MemoryBandwidth::memory_copyx4_x8_wave_continue_8wave},
        {"memory_copyx4_x8_wave_continue_16wave", &MemoryBandwidth::memory_copyx4_x8_wave_continue_16wave},
        {"memory_copyx4_x8_inst_continue", &MemoryBandwidth::memory_copyx4_x8_inst_continue},
#if HIP_ARCH >= 936
        {"memory_copyx4_x8_wave_continue_4wave_bps", &MemoryBandwidth::memory_copyx4_x8_wave_continue_bps_4wave},
        {"memory_copyx4_x8_wave_continue_8wave_bps", &MemoryBandwidth::memory_copyx4_x8_wave_continue_bps_8wave},
        {"memory_copyx4_x8_wave_continue_16wave_bps", &MemoryBandwidth::memory_copyx4_x8_wave_continue_bps_16wave},
#endif
        {"matrix_read", &MemoryBandwidth::matrix_read},
        {"matrix_read_x4", &MemoryBandwidth::matrix_read_x4},
        {"matrix_write", &MemoryBandwidth::matrix_write},
        {"matrix_write_x4", &MemoryBandwidth::matrix_write_x4},
        {"matrix_copy", &MemoryBandwidth::matrix_copy},
        {"matrix_copy_x4", &MemoryBandwidth::matrix_copy_x4},

        {"matrix_readx4_x2", &MemoryBandwidth::matrix_readx4_x2},
        {"matrix_writex4_x2", &MemoryBandwidth::matrix_writex4_x2},
        {"matrix_readx4_x4", &MemoryBandwidth::matrix_readx4_x4},
        {"matrix_writex4_x4", &MemoryBandwidth::matrix_writex4_x4},
        {"matrix_copyx4_x4", &MemoryBandwidth::matrix_copyx4_x4},

        {"matrix_readx4_multi", &MemoryBandwidth::matrix_readx4_multi},
        {"matrix_writex4_multi", &MemoryBandwidth::matrix_writex4_multi},
        {"matrix_copyx4_multi", &MemoryBandwidth::matrix_copyx4_multi},

        {"mapping_matrix_write", &MemoryBandwidth::mapping_matrix_write},
        {"mapping_matrix_writex4_x4", &MemoryBandwidth::mapping_matrix_writex4_x4},
        {"tiling_read", &MemoryBandwidth::tiling_read},
        {"tiling_read_nchw", &MemoryBandwidth::tiling_read_nchw_16t},
        {"tiling_read_nhwc", &MemoryBandwidth::tiling_read_nhwc},
        {"tiling_read_2w", &MemoryBandwidth::tiling_read_2w},
        {"tiling_read_2h", &MemoryBandwidth::tiling_read_2h},
        {"tiling_read_w", &MemoryBandwidth::tiling_read_w},
        {"winograd_read", &MemoryBandwidth::winograd_read},
        {"memory_to_ldsx1", &MemoryBandwidth::memory_to_ldsx1},
#if HIP_ARCH >= 936
        {"memory_to_ldsx4", &MemoryBandwidth::memory_to_ldsx4},
        {"memory_to_ldsx4_x4", &MemoryBandwidth::memory_to_ldsx4_x4},
#endif
        {"lds_to_memory", &MemoryBandwidth::lds_to_memory},
        {"lds_to_memory_x4", &MemoryBandwidth::lds_to_memory_x4},
        {"l1_read", &MemoryBandwidth::l1_read},
        {"l2_read", &MemoryBandwidth::l2_read},
        {"l2_write", &MemoryBandwidth::l2_write},
        {"lds_read", &MemoryBandwidth::lds_read},
        {"lds_write", &MemoryBandwidth::lds_write},
        {"lds_add_u32", &MemoryBandwidth::lds_add_u32},
        {"lds_read4x", &MemoryBandwidth::lds_read4x},
        {"lds_write4x", &MemoryBandwidth::lds_write4x},
        {"moe_triton_1", &MemoryBandwidth::moe_triton_1},
        {"moe_triton_2", &MemoryBandwidth::moe_triton_2},
#if HIP_ARCH >= 936
        {"moe_cpp", &MemoryBandwidth::moe_cpp},
#endif
#if HIP_ARCH >= 928
        {"lds_b32_matrix_read", &MemoryBandwidth::lds_b32_matrix_read},
        {"lds_b16_matrix_read", &MemoryBandwidth::lds_b16_matrix_read},
        {"lds_b16_alt_matrix_read", &MemoryBandwidth::lds_b16_alt_matrix_read},
        {"lds_b8_matrix_read", &MemoryBandwidth::lds_b8_matrix_read},
#endif

};

Testcase::Testcase(std::vector<std::string> case_list, int wave, double crop,
                   Output *output, Device *device)
    : case_list(case_list), crop(crop), wave(wave), output(output),
      device(device) {}

MemoryBandwidth::MemoryBandwidth(std::vector<std::string> case_list, int wave,
                                 double crop, Output *output, Device *device,
                                 int lds_size, bool clean, int array_size,
                                 bool _useEvent, bool _printEach)
    : Testcase(case_list, wave, crop, output, device) {
  lds = lds_size;
  use_clean = clean;
  arraysize = array_size;
  useEvent = _useEvent;
  printEach = _printEach;
}

void MemoryBandwidth::run() {
  // 查找并调用对应的函数
  for (auto name : case_list) {
    auto it = functionMap.find(name);
    if (it != functionMap.end()) {
      if (device->run_all) {
        for (int i = 0; i < device->device_count; i++) {
          CHECK_HIP_ERROR(hipSetDevice(i));
          device->device_index = i;
          (this->*(it->second))();
        }
      }
      else {
        (this->*(it->second))();
      }
    } else {
      std::cout << "Unknown bandwidth case name!" << std::endl;
    }
  }
}

double MemoryBandwidth::getLdsCacheTheory(const std::string &name) {
  if (name == "L1") {
    double sclk = device->get_sclk() / 1000;
    return 64 * device->cuNum * sclk;
  } else if (name == "L2 read") {
    double sclk = device->get_sclk() / 1000;
    double mul = 64.0;
#if HIP_ARCH >= 936
    mul = 128.0;
#endif
    return mul * 32 * sclk;
  } else if (name == "L2 write") {
    double sclk = device->get_sclk() / 1000;
    return 64 * 32 * sclk;
  } else if (name == "lds") {
    double sclk = device->get_sclk() / 1000;
    return 128 * device->cuNum * sclk;
  } else {
    return 0.0;
  }
}

void MemoryBandwidth::getList() {
  std::cout << std::left << std::setw(30) << "MemoryBadwidth case" << std::endl;
  std::cout << std::left << std::setw(30)
            << "Usage: ./run bandwidth memory_read" << std::endl;
  std::cout << "----------------------" << std::endl;
  for (const auto &key : functionMap) {
    std::cout << std::left << std::setw(30) << key.first << std::endl;
  }
}

void MemoryBandwidth::print() {
  if (use_clean)
    output->print_clean();
  else
    output->print_record();
}

void MemoryBandwidth::memory_readx1_x1() {
  int THREADS_PER_BLOCK = wave * 64;
  int *inArray;
  uint MEM_ARRAY_SIZE = arraysize * 4;
  inArray = new int[MEM_ARRAY_SIZE]; // 1.562GB

  int *inArray_g;
  HIP_ASSERT(hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(int)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(int),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(int);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(memory_readx1_x1_kernel,
                            dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK),
                            dim3(THREADS_PER_BLOCK), lds * 1024, 0,
                            start_event_, stop_event_, 0, inArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(memory_readx1_x1_kernel,
                         dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK),
                         dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));

  output->add_record("memory_read_dwordx1_x1", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::memory_readx1_x4() {
  int THREADS_PER_BLOCK = wave * 64;
  int *inArray;
  uint MEM_ARRAY_SIZE = arraysize * 4;
  inArray = new int[MEM_ARRAY_SIZE]; // 1.562GB

  int *inArray_g;
  HIP_ASSERT(hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(int)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(int),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(int);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0;
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(memory_readx1_x1_kernel,
                            dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 4),
                            dim3(THREADS_PER_BLOCK), lds * 1024, 0,
                            start_event_, stop_event_, 0, inArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(memory_readx1_x4_kernel,
                         dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 4),
                         dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));

  output->add_record("memory_read_dwordx1_x4", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}

// not support gpu time
void MemoryBandwidth::memory_readx1_x8192() {
  int THREADS_PER_BLOCK = 4 /*wave*/ * 64;
  // only wave = 4;
  int *inArray;
  uint MEM_ARRAY_SIZE = 256 * 1024 * 1024;
  inArray = new int[MEM_ARRAY_SIZE]; // 1.562GB

  int *inArray_g;
  HIP_ASSERT(hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(int)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(int),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(int);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    t1 = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(memory_readx1_x8192_kernel, dim3(128),
                       dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    t = duration.count();
    if (i == 0) {
      if (isFisrt) {
        std::cout << (double)data * pow(10.0, -9.0) / t << std::endl;
        return;
      }
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));

  output->add_record("memory_read_dwordx1_x8192", 4 /*wave*/,
                     data * pow(10.0, -6.0), avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::memory_readx1_x8192_80cu() {
  int THREADS_PER_BLOCK = 4 /*wave*/ * 64;
  // only wave = 4;
  int *inArray;
  uint MEM_ARRAY_SIZE = 160 * 1024 * 1024;
  inArray = new int[MEM_ARRAY_SIZE]; // 1.562GB

  int *inArray_g;
  HIP_ASSERT(hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(int)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(int),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(int);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    t1 = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(memory_readx1_x8192_80cu_kernel, dim3(80),
                       dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    t = duration.count();
    if (i == 0) {
      if (isFisrt) {
        std::cout << (double)data * pow(10.0, -9.0) / t << std::endl;
        return;
      }
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));

  output->add_record("memory_read_dwordx1_x8192_80cu", 4 /*wave*/,
                     data * pow(10.0, -6.0), avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}

// global_dwordx2
void MemoryBandwidth::memory_readx2_x1() {
  int THREADS_PER_BLOCK = wave * 64;
  __int64_t *inArray;
  uint MEM_ARRAY_SIZE = arraysize * 2;
  inArray = new __int64_t[MEM_ARRAY_SIZE]; // 1.562GB

  __int64_t *inArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(__int64_t)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(__int64_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int64_t);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(memory_readx2_x1_kernel,
                            dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK),
                            dim3(THREADS_PER_BLOCK), lds * 1024, 0,
                            start_event_, stop_event_, 0, inArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(memory_readx2_x1_kernel,
                         dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK),
                         dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));
  output->add_record("memory_read_dwordx2_x1", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::memory_readx2_x2() {
  int THREADS_PER_BLOCK = wave * 64;
  __int64_t *inArray;
  uint MEM_ARRAY_SIZE = arraysize * 2;
  inArray = new __int64_t[MEM_ARRAY_SIZE]; // 1.562GB

  __int64_t *inArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(__int64_t)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(__int64_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int64_t);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(memory_readx2_x2_kernel,
                            dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 2),
                            dim3(THREADS_PER_BLOCK), lds * 1024, 0,
                            start_event_, stop_event_, 0, inArray_g, wave);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(memory_readx2_x2_kernel,
                         dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 2),
                         dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g, wave);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));
  output->add_record("memory_read_dwordx2_x2", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::memory_readx2_x4() {
  int THREADS_PER_BLOCK = wave * 64;
  __int64_t *inArray;
  uint MEM_ARRAY_SIZE = arraysize * 2;
  inArray = new __int64_t[MEM_ARRAY_SIZE]; // 1.562GB

  __int64_t *inArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(__int64_t)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(__int64_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int64_t);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(memory_readx2_x4_kernel,
                            dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 4),
                            dim3(THREADS_PER_BLOCK), lds * 1024, 0,
                            start_event_, stop_event_, 0, inArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(memory_readx2_x4_kernel,
                         dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 4),
                         dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));
  output->add_record("memory_read_dwordx2_x4", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}

void MemoryBandwidth::memory_readx2_x4096() {
  int THREADS_PER_BLOCK = 4 /*wave*/ * 64;
  __int64_t *inArray;
  uint MEM_ARRAY_SIZE = 128 * 1024 * 1024; // 1G / 8
  inArray = new __int64_t[MEM_ARRAY_SIZE]; // 1.0GB

  __int64_t *inArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(__int64_t)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(__int64_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int64_t);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    t1 = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(memory_readx2_x4096_kernel, dim3(128),
                       dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    t = duration.count();
    if (i == 0) {
      if (isFisrt) {
        std::cout << (double)data * pow(10.0, -9.0) / t << std::endl;
        return;
      }
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));
  output->add_record("memory_readx2_x4096", 4 /*wave*/, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::memory_readx2_x4096_80cu() {
  int THREADS_PER_BLOCK = 4 /*wave*/ * 64;
  __int64_t *inArray;
  uint MEM_ARRAY_SIZE = 80 * 1024 * 1024;  // 1G / 8
  inArray = new __int64_t[MEM_ARRAY_SIZE]; // 1.0GB

  __int64_t *inArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(__int64_t)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(__int64_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int64_t);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    t1 = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(memory_readx2_x4096_80cu_kernel, dim3(80),
                       dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    t = duration.count();
    if (i == 0) {
      if (isFisrt) {
        std::cout << (double)data * pow(10.0, -9.0) / t << std::endl;
        return;
      }
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));
  output->add_record("memory_readx2_x4096_80cu", 4 /*wave*/,
                     data * pow(10.0, -6.0), avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}

// global_dwordx4
void MemoryBandwidth::memory_readx4_x1() {
  int THREADS_PER_BLOCK = wave * 64;
  __int128_t *inArray;
  uint MEM_ARRAY_SIZE = arraysize;
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
  float time_ms = 0.0f;
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(memory_readx4_x1_kernel,
                            dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK),
                            dim3(THREADS_PER_BLOCK), lds * 1024, 0,
                            start_event_, stop_event_, 0, inArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(memory_readx4_x1_kernel,
                         dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK),
                         dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }
    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }
    if (i == 0) {
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));
  output->add_record("memory_readx4_x1", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::memory_readx4_x2() {
  int THREADS_PER_BLOCK = wave * 64;
  __int128_t *inArray;
  uint MEM_ARRAY_SIZE = arraysize;
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
  float time_ms = 0.0f; // hip event
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(memory_readx4_x2_kernel,
                            dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 2),
                            dim3(THREADS_PER_BLOCK), lds * 1024, 0,
                            start_event_, stop_event_, 0, inArray_g, wave);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(memory_readx4_x2_kernel,
                         dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 2),
                         dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g, wave);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));
  output->add_record("memory_readx4_x2", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::memory_readx4_x4() {
  int THREADS_PER_BLOCK = wave * 64;
  uint MEM_ARRAY_SIZE = arraysize;
  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int128_t);

  __int128_t *inArray = new __int128_t[MEM_ARRAY_SIZE];
  __int128_t *inArray_g;
  HIP_ASSERT(hipMalloc((void **)&inArray_g, data));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, data, hipMemcpyHostToDevice));

  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  double t = 0.0;       // 当前循环耗时（秒）
  float time_ms = 0.0f; // HIP Event计时用

  for (int i = 0; i < LAUNCH_TIMES; ++i) {

    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(memory_readx4_x4_kernel,
                            dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 4),
                            dim3(THREADS_PER_BLOCK), 32 * 1024, 0, start_event_,
                            stop_event_, 0, inArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      // -------------------- std::chrono 模式 --------------------
      auto t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(memory_readx4_x4_kernel,
                         dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 4),
                         dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      auto t2 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration = t2 - t1;
      t = duration.count(); // seconds
    }

    // -------------------- 可选打印每次执行时间 --------------------
    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    // 忽略第一次启动（冷启动）
    if (i == 0) {
      continue;
    }

    double bw = (double)data * 1e-9 / t; // GB/s
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }

  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));

  output->add_record("memory_readx4_x4", wave,
                     data * 1e-6,                        // data size (MB)
                     avg_bw / (LAUNCH_TIMES - 1),        // avg BW
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), // avg latency (ms)
                     getBWtheory(), lds, device->device_index);
}

void MemoryBandwidth::memory_readx4_x8_inst_continue() {
  int THREADS_PER_BLOCK = wave * 64;
  __int128_t *inArray;
  uint MEM_ARRAY_SIZE = arraysize;
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
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    t1 = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(memory_readx4_x8_inst_continue_kernel,
                       dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
                       dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    t = duration.count();
    if (i == 0) {
      if (isFisrt) {
        std::cout << (double)data * pow(10.0, -9.0) / t << std::endl;
        return;
      }
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));
  output->add_record("memory_readx4_x8_inst_con", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::memory_readx4_x8_wave_continue() {
  int THREADS_PER_BLOCK = wave * 64;
  __int128_t *inArray;
  uint MEM_ARRAY_SIZE = arraysize;
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
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    t1 = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(memory_readx4_x8_wave_continue_kernel,
                       dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
                       dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    t = duration.count();
    if (i == 0) {
      if (isFisrt) {
        std::cout << (double)data * pow(10.0, -9.0) / t << std::endl;
        return;
      }
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));
  output->add_record("memory_readx4_x8_wave_con", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::memory_readx4_x8_wave_continue_8wave() {
  wave = 8;
  int THREADS_PER_BLOCK = wave * 64;
  __int128_t *inArray;
  uint MEM_ARRAY_SIZE = arraysize;
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
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    t1 = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(memory_readx4_x8_wave_continue_8wave_kernel,
                       dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
                       dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    t = duration.count();
    if (i == 0) {
      if (isFisrt) {
        std::cout << (double)data * pow(10.0, -9.0) / t << std::endl;
        return;
      }
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));
  output->add_record("memory_readx4_x8_wave_con", 8, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::memory_readx4_x8_wave_continue_16wave() {
  wave = 16;
  int THREADS_PER_BLOCK = wave * 64;
  __int128_t *inArray;
  uint MEM_ARRAY_SIZE = arraysize;
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
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    t1 = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(memory_readx4_x8_wave_continue_16wave_kernel,
                       dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
                       dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    t = duration.count();
    if (i == 0) {
      if (isFisrt) {
        std::cout << (double)data * pow(10.0, -9.0) / t << std::endl;
        return;
      }
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));
  output->add_record("memory_readx4_x8_wave_con", 16, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}

// bps 
#if HIP_ARCH >= 936
void MemoryBandwidth::memory_readx4_x8_wave_continue_bps() {
  int THREADS_PER_BLOCK = wave * 64;
  __int128_t *inArray;
  uint MEM_ARRAY_SIZE = arraysize;
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
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    t1 = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(memory_readx4_x8_wave_continue_bps_kernel,
                       dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
                       dim3(THREADS_PER_BLOCK), 32 * 1024, 0, inArray_g);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    t = duration.count();
    if (i == 0) {
      if (isFisrt) {
        std::cout << (double)data * pow(10.0, -9.0) / t << std::endl;
        return;
      }
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));
  output->add_record("memory_readx4_x8_wave_con_bps", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), 32, device->device_index);
}
void MemoryBandwidth::memory_readx4_x8_wave_continue_8wave_bps() {
  wave = 8;
  int THREADS_PER_BLOCK = wave * 64;
  __int128_t *inArray;
  uint MEM_ARRAY_SIZE = arraysize;
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
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    t1 = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(memory_readx4_x8_wave_continue_8wave_bps_kernel,
                       dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
                       dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    t = duration.count();
    if (i == 0) {
      if (isFisrt) {
        std::cout << (double)data * pow(10.0, -9.0) / t << std::endl;
        return;
      }
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));
  output->add_record("memory_readx4_x8_wave_con_bps", 8, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), 32, device->device_index);
}
void MemoryBandwidth::memory_readx4_x8_wave_continue_16wave_bps() {
  wave = 16;
  int THREADS_PER_BLOCK = wave * 64;
  __int128_t *inArray;
  uint MEM_ARRAY_SIZE = arraysize;
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
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    t1 = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(memory_readx4_x8_wave_continue_16wave_bps_kernel,
                       dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
                       dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    t = duration.count();
    if (i == 0) {
      if (isFisrt) {
        std::cout << (double)data * pow(10.0, -9.0) / t << std::endl;
        return;
      }
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));
  output->add_record("memory_readx4_x8_wave_con_bps", 16, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), 32, device->device_index);
}
#endif

void MemoryBandwidth::memory_readx4_x16() {
  int THREADS_PER_BLOCK = wave * 64;
  __int128_t *inArray;
  uint MEM_ARRAY_SIZE = arraysize;
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
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    t1 = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(memory_readx4_x16_kernel,
                       dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 16),
                       dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    t = duration.count();
    if (i == 0) {
      if (isFisrt) {
        std::cout << (double)data * pow(10.0, -9.0) / t << std::endl;
        return;
      }
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));
  output->add_record("memory_readx4_x16", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::memory_readx4_x64() {
  int THREADS_PER_BLOCK = wave * 64;
  __int128_t *inArray;
  uint MEM_ARRAY_SIZE = arraysize;
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
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    t1 = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(memory_readx4_x64_kernel,
                       dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 64),
                       dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    t = duration.count();
    if (i == 0) {
      if (isFisrt) {
        std::cout << (double)data * pow(10.0, -9.0) / t << std::endl;
        return;
      }
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));
  output->add_record("memory_readx4_x64", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::memory_readx4_x2048() {
  int THREADS_PER_BLOCK = 4 /*wave*/ * 64;
  __int128_t *inArray;
  uint MEM_ARRAY_SIZE = (128 * 1024 * 4 * 2048) / 16;
  inArray = new __int128_t[MEM_ARRAY_SIZE]; // 1.562GB

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
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    t1 = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(memory_readx4_x2048_kernel, dim3(128),
                       dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    t = duration.count();
    if (i == 0) {
      if (isFisrt) {
        std::cout << (double)data * pow(10.0, -9.0) / t << std::endl;
        return;
      }
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));
  output->add_record("memory_readx4_x2048", 4 /*wave*/, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::memory_readx4_x2048_80cu() {
  int THREADS_PER_BLOCK = 4 /*wave*/ * 64;
  __int128_t *inArray;
  uint MEM_ARRAY_SIZE = (80 * 1024 * 4 * 2048) / 16;
  inArray = new __int128_t[MEM_ARRAY_SIZE]; // 1.562GB

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
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    t1 = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(memory_readx4_x2048_80cu_kernel, dim3(80),
                       dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    t = duration.count();
    if (i == 0) {
      if (isFisrt) {
        std::cout << (double)data * pow(10.0, -9.0) / t << std::endl;
        return;
      }
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));
  output->add_record("memory_readx4_x2048_80cu", 4 /*wave*/,
                     data * pow(10.0, -6.0), avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}

void MemoryBandwidth::memory_write() {
  int THREADS_PER_BLOCK = wave * 64;
  __int128_t *outArray;
  uint MEM_ARRAY_SIZE = arraysize;
  outArray = new __int128_t[MEM_ARRAY_SIZE]; // 1.562GB

  __int128_t *outArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray,
                       MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int128_t);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  float peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (uint i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(memory_write_kernel,
                            dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK),
                            dim3(THREADS_PER_BLOCK), lds * 1024, 0,
                            start_event_, stop_event_, 0, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(memory_write_kernel,
                         dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK),
                         dim3(THREADS_PER_BLOCK), lds * 1024, 0, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    float bw = (float)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] outArray;
  output->add_record("memory_write", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::memory_writex1_x1() {
  int THREADS_PER_BLOCK = wave * 64;
  int *outArray;
  uint MEM_ARRAY_SIZE = arraysize * 4;
  outArray = new int[MEM_ARRAY_SIZE]; // 1.562GB

  int *outArray_g;
  HIP_ASSERT(hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(int)));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray, MEM_ARRAY_SIZE * sizeof(int),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(int);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  float peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (uint i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(memory_writex1_x1_kernel,
                            dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK),
                            dim3(THREADS_PER_BLOCK), lds * 1024, 0,
                            start_event_, stop_event_, 0, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(memory_writex1_x1_kernel,
                         dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK),
                         dim3(THREADS_PER_BLOCK), lds * 1024, 0, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    float bw = (float)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] outArray;
  output->add_record("memory_writex1_x1", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::memory_writex2_x1() {
  int THREADS_PER_BLOCK = wave * 64;
  __int64_t *outArray;
  uint MEM_ARRAY_SIZE = arraysize * 2;
  outArray = new __int64_t[MEM_ARRAY_SIZE]; // 1.562GB

  __int64_t *outArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(__int64_t)));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray, MEM_ARRAY_SIZE * sizeof(__int64_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int64_t);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  float peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (uint i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(memory_writex2_x1_kernel,
                            dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK),
                            dim3(THREADS_PER_BLOCK), lds * 1024, 0,
                            start_event_, stop_event_, 0, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(memory_writex2_x1_kernel,
                         dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK),
                         dim3(THREADS_PER_BLOCK), lds * 1024, 0, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    float bw = (float)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] outArray;
  output->add_record("memory_writex2_x1", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}

void MemoryBandwidth::memory_writex4_x2() {
  int THREADS_PER_BLOCK = wave * 64;
  __int128_t *outArray;
  uint MEM_ARRAY_SIZE = arraysize;
  outArray = new __int128_t[MEM_ARRAY_SIZE]; // 1.0GB

  __int128_t *outArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray,
                       MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int128_t);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(memory_writex4_x2_kernel,
                            dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 2),
                            dim3(THREADS_PER_BLOCK), lds * 1024, 0,
                            start_event_, stop_event_, 0, outArray_g, wave);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(memory_writex4_x2_kernel,
                         dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 2),
                         dim3(THREADS_PER_BLOCK), lds * 1024, 0, outArray_g, wave);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] outArray;
  HIP_ASSERT(hipFree(outArray_g));
  output->add_record("memory_writex4_x2", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}

void MemoryBandwidth::memory_writex4_x4() {
  int THREADS_PER_BLOCK = wave * 64;
  __int128_t *outArray;
  uint MEM_ARRAY_SIZE = arraysize;
  outArray = new __int128_t[MEM_ARRAY_SIZE]; // 1.0GB

  __int128_t *outArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray,
                       MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int128_t);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(memory_writex4_x4_kernel,
                            dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 4),
                            dim3(THREADS_PER_BLOCK), lds * 1024, 0,
                            start_event_, stop_event_, 0, outArray_g, wave);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(memory_writex4_x4_kernel,
                         dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 4),
                         dim3(THREADS_PER_BLOCK), lds * 1024, 0, outArray_g, wave);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] outArray;
  HIP_ASSERT(hipFree(outArray_g));
  output->add_record("memory_writex4_x4", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}

void MemoryBandwidth::memory_writex4_x8_inst_continue() {
  int THREADS_PER_BLOCK = wave * 64;
  __int128_t *outArray;
  uint MEM_ARRAY_SIZE = arraysize;
  outArray = new __int128_t[MEM_ARRAY_SIZE]; // 1.0GB

  __int128_t *outArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray,
                       MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int128_t);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(memory_writex4_x8_inst_continue_kernel,
                            dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
                            dim3(THREADS_PER_BLOCK), lds * 1024, 0,
                            start_event_, stop_event_, 0, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(memory_writex4_x8_inst_continue_kernel,
                         dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
                         dim3(THREADS_PER_BLOCK), lds * 1024, 0, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] outArray;
  HIP_ASSERT(hipFree(outArray_g));
  output->add_record("memory_writex4_x8_inst_con", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::memory_writex4_x8_wave_continue() {
  int THREADS_PER_BLOCK = wave * 64;
  __int128_t *outArray;
  uint MEM_ARRAY_SIZE = arraysize;
  outArray = new __int128_t[MEM_ARRAY_SIZE]; // 1.0GB

  __int128_t *outArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray,
                       MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int128_t);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(memory_writex4_x8_wave_continue_kernel,
                            dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
                            dim3(THREADS_PER_BLOCK), lds * 1024, 0,
                            start_event_, stop_event_, 0, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(memory_writex4_x8_wave_continue_kernel,
                         dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
                         dim3(THREADS_PER_BLOCK), lds * 1024, 0, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] outArray;
  HIP_ASSERT(hipFree(outArray_g));
  output->add_record("memory_writex4_x8_wave_con", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::memory_writex4_x8_wave_continue_8wave() {
  wave = 8;
  int THREADS_PER_BLOCK = wave * 64;
  __int128_t *outArray;
  uint MEM_ARRAY_SIZE = arraysize;
  outArray = new __int128_t[MEM_ARRAY_SIZE]; // 1.0GB

  __int128_t *outArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray,
                       MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int128_t);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(memory_writex4_x8_wave_continue_8wave_kernel,
                            dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
                            dim3(THREADS_PER_BLOCK), lds * 1024, 0,
                            start_event_, stop_event_, 0, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(memory_writex4_x8_wave_continue_8wave_kernel,
                         dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
                         dim3(THREADS_PER_BLOCK), lds * 1024, 0, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] outArray;
  HIP_ASSERT(hipFree(outArray_g));
  output->add_record("memory_writex4_x8_wave_con", 8, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::memory_writex4_x8_wave_continue_16wave() {
  wave = 16;
  int THREADS_PER_BLOCK = wave * 64;
  __int128_t *outArray;
  uint MEM_ARRAY_SIZE = arraysize;
  outArray = new __int128_t[MEM_ARRAY_SIZE]; // 1.0GB

  __int128_t *outArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray,
                       MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int128_t);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(memory_writex4_x8_wave_continue_16wave_kernel,
                            dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
                            dim3(THREADS_PER_BLOCK), lds * 1024, 0,
                            start_event_, stop_event_, 0, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(memory_writex4_x8_wave_continue_16wave_kernel,
                         dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
                         dim3(THREADS_PER_BLOCK), lds * 1024, 0, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] outArray;
  HIP_ASSERT(hipFree(outArray_g));
  output->add_record("memory_writex4_x8_wave_con", 16, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}

void MemoryBandwidth::memory_writex1_x4() {
  int THREADS_PER_BLOCK = wave * 64;
  int *outArray;
  uint MEM_ARRAY_SIZE = arraysize * 4;
  outArray = new int[MEM_ARRAY_SIZE]; // 1.562GB

  int *outArray_g;
  HIP_ASSERT(hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(int)));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray, MEM_ARRAY_SIZE * sizeof(int),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(int);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  float peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (uint i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(memory_writex1_x4_kernel,
                            dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 4),
                            dim3(THREADS_PER_BLOCK), lds * 1024, 0,
                            start_event_, stop_event_, 0, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(memory_writex1_x4_kernel,
                         dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 4),
                         dim3(THREADS_PER_BLOCK), lds * 1024, 0, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    float bw = (float)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] outArray;
  output->add_record("memory_writex1_x4", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}

void MemoryBandwidth::memory_writex2_x2() {
  int THREADS_PER_BLOCK = wave * 64;
  __int64_t *outArray;
  uint MEM_ARRAY_SIZE = arraysize * 2;
  outArray = new __int64_t[MEM_ARRAY_SIZE]; // 1.562GB

  __int64_t *outArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(__int64_t)));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray, MEM_ARRAY_SIZE * sizeof(__int64_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int64_t);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  float peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (uint i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(memory_writex2_x2_kernel,
                            dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 2),
                            dim3(THREADS_PER_BLOCK), lds * 1024, 0,
                            start_event_, stop_event_, 0, outArray_g, wave);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(memory_writex2_x2_kernel,
                         dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 2),
                         dim3(THREADS_PER_BLOCK), lds * 1024, 0, outArray_g, wave);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    float bw = (float)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] outArray;
  output->add_record("memory_writex2_x2", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}

void MemoryBandwidth::memory_writex2_x4() {
  int THREADS_PER_BLOCK = wave * 64;
  __int64_t *outArray;
  uint MEM_ARRAY_SIZE = arraysize * 2;
  outArray = new __int64_t[MEM_ARRAY_SIZE]; // 1.562GB

  __int64_t *outArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(__int64_t)));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray, MEM_ARRAY_SIZE * sizeof(__int64_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int64_t);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  float peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (uint i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(memory_writex2_x4_kernel,
                            dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 4),
                            dim3(THREADS_PER_BLOCK), lds * 1024, 0,
                            start_event_, stop_event_, 0, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(memory_writex2_x4_kernel,
                         dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 4),
                         dim3(THREADS_PER_BLOCK), lds * 1024, 0, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    float bw = (float)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] outArray;
  output->add_record("memory_writex2_x4", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}

void MemoryBandwidth::memory_copy() {
  int THREADS_PER_BLOCK = wave * 64;
  uint MEM_ARRAY_SIZE = arraysize;
  __int128_t *inArray;
  __int128_t *outArray;
  inArray = new __int128_t[MEM_ARRAY_SIZE];
  outArray = new __int128_t[MEM_ARRAY_SIZE];

  __int128_t *inArray_g;
  __int128_t *outArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(
      hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray,
                       MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int128_t) * 2;
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  float peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (uint i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(
          memory_copy_kernel, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK),
          dim3(THREADS_PER_BLOCK), lds * 1024, 0, start_event_, stop_event_, 0,
          inArray_g, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(
          memory_copy_kernel, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK),
          dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    float bw = (float)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  delete[] outArray;
  HIP_ASSERT(hipFree(inArray_g));
  HIP_ASSERT(hipFree(outArray_g));
  output->add_record("memory_copy", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}

void MemoryBandwidth::memory_copyx2_x1() {
  int THREADS_PER_BLOCK = wave * 64;
  uint MEM_ARRAY_SIZE = arraysize;
  double *inArray;
  double *outArray;
  inArray = new double[MEM_ARRAY_SIZE];
  outArray = new double[MEM_ARRAY_SIZE];

  double *inArray_g;
  double *outArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(double)));
  HIP_ASSERT(
      hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(double)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(double),
                       hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray,
                       MEM_ARRAY_SIZE * sizeof(double),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(double) * 2;
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  float peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (uint i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(
          memory_copyx2_x1_kernel, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK),
          dim3(THREADS_PER_BLOCK), lds * 1024, 0, start_event_, stop_event_, 0,
          inArray_g, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(
          memory_copyx2_x1_kernel, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK),
          dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    float bw = (float)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  delete[] outArray;
  HIP_ASSERT(hipFree(inArray_g));
  HIP_ASSERT(hipFree(outArray_g));
  output->add_record("memory_copyx2_x1", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}

void MemoryBandwidth::memory_copyx4_x4() {
  int THREADS_PER_BLOCK = wave * 64;
  uint MEM_ARRAY_SIZE = arraysize;
  __int128_t *inArray;
  __int128_t *outArray;
  inArray = new __int128_t[MEM_ARRAY_SIZE];
  outArray = new __int128_t[MEM_ARRAY_SIZE];

  __int128_t *inArray_g;
  __int128_t *outArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(
      hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray,
                       MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int128_t) * 2;
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  float peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (uint i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(
          memory_copyx4_x4_kernel, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 4),
          dim3(THREADS_PER_BLOCK), lds * 1024, 0, start_event_, stop_event_, 0,
          inArray_g, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(
          memory_copyx4_x4_kernel, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 4),
          dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    float bw = (float)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  delete[] outArray;
  HIP_ASSERT(hipFree(inArray_g));
  HIP_ASSERT(hipFree(outArray_g));
  output->add_record("memory_copyx4_x4", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}

// wave continue
void MemoryBandwidth::memory_copyx4_x4_wave_continue() {
  int THREADS_PER_BLOCK = wave * 64;
  uint MEM_ARRAY_SIZE = arraysize;
  __int128_t *inArray;
  __int128_t *outArray;
  inArray = new __int128_t[MEM_ARRAY_SIZE];
  outArray = new __int128_t[MEM_ARRAY_SIZE];

  __int128_t *inArray_g;
  __int128_t *outArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(
      hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(__int128_t),
                        hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray,
                        MEM_ARRAY_SIZE * sizeof(__int128_t),
                        hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int128_t) * 2;
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  float peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (uint i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(
        memory_copyx4_x4_wave_continue_kernel, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 4),
          dim3(THREADS_PER_BLOCK), lds * 1024, 0, start_event_, stop_event_, 0,
          inArray_g, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(
          memory_copyx4_x4_wave_continue_kernel, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 4),
          dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    float bw = (float)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  delete[] outArray;
  HIP_ASSERT(hipFree(inArray_g));
  HIP_ASSERT(hipFree(outArray_g));
  output->add_record("memory_copyx4_x4_wave_continue_kernel", wave, data * pow(10.0, -6.0),
                      avg_bw / (LAUNCH_TIMES - 1),
                      avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}

void MemoryBandwidth::memory_copyx4_x8_wave_continue() {
  int THREADS_PER_BLOCK = wave * 64;
  uint MEM_ARRAY_SIZE = arraysize;
  __int128_t *inArray;
  __int128_t *outArray;
  inArray = new __int128_t[MEM_ARRAY_SIZE];
  outArray = new __int128_t[MEM_ARRAY_SIZE];

  __int128_t *inArray_g;
  __int128_t *outArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(
      hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray,
                       MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int128_t) * 2;
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  float peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (uint i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(
        memory_copyx4_x8_wave_continue_kernel, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
          dim3(THREADS_PER_BLOCK), lds * 1024, 0, start_event_, stop_event_, 0,
          inArray_g, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(
        memory_copyx4_x8_wave_continue_kernel, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
          dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    float bw = (float)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  delete[] outArray;
  HIP_ASSERT(hipFree(inArray_g));
  HIP_ASSERT(hipFree(outArray_g));
  output->add_record("memory_copyx4_x8_wave_con", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::memory_copyx4_x8_wave_continue_asm() {
  int THREADS_PER_BLOCK = wave * 64;
  uint MEM_ARRAY_SIZE = arraysize;
  __int128_t *inArray;
  __int128_t *outArray;
  inArray = new __int128_t[MEM_ARRAY_SIZE];
  outArray = new __int128_t[MEM_ARRAY_SIZE];

  __int128_t *inArray_g;
  __int128_t *outArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(
      hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray,
                       MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int128_t) * 2;
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  float peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (uint i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(
        memory_copyx4_x8_wave_continue_asm_kernel, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
          dim3(THREADS_PER_BLOCK), lds * 1024, 0, start_event_, stop_event_, 0,
          inArray_g, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(
        memory_copyx4_x8_wave_continue_asm_kernel, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
          dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    float bw = (float)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  delete[] outArray;
  HIP_ASSERT(hipFree(inArray_g));
  HIP_ASSERT(hipFree(outArray_g));
  output->add_record("memory_copyx4_x8_wave_con_asm", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}


void MemoryBandwidth::memory_copyx4_x8_wave_continue_8wave() {
  wave = 8;
  int THREADS_PER_BLOCK = wave * 64;
  uint MEM_ARRAY_SIZE = arraysize;
  __int128_t *inArray;
  __int128_t *outArray;
  inArray = new __int128_t[MEM_ARRAY_SIZE];
  outArray = new __int128_t[MEM_ARRAY_SIZE];

  __int128_t *inArray_g;
  __int128_t *outArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(
      hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray,
                       MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int128_t) * 2;
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  float peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (uint i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(
        memory_copyx4_x8_wave_continue_8wave_kernel, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
          dim3(THREADS_PER_BLOCK), lds * 1024, 0, start_event_, stop_event_, 0,
          inArray_g, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(
        memory_copyx4_x8_wave_continue_8wave_kernel, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
          dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    float bw = (float)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  delete[] outArray;
  HIP_ASSERT(hipFree(inArray_g));
  HIP_ASSERT(hipFree(outArray_g));
  output->add_record("memory_copyx4_x8_wave_con", 8, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::memory_copyx4_x8_wave_continue_16wave() {
  wave = 16;
  int THREADS_PER_BLOCK = wave * 64;
  uint MEM_ARRAY_SIZE = arraysize;
  __int128_t *inArray;
  __int128_t *outArray;
  inArray = new __int128_t[MEM_ARRAY_SIZE];
  outArray = new __int128_t[MEM_ARRAY_SIZE];

  __int128_t *inArray_g;
  __int128_t *outArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(
      hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray,
                       MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int128_t) * 2;
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  float peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (uint i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(
        memory_copyx4_x8_wave_continue_16wave_kernel, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
          dim3(THREADS_PER_BLOCK), lds * 1024, 0, start_event_, stop_event_, 0,
          inArray_g, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(
        memory_copyx4_x8_wave_continue_16wave_kernel, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
          dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    float bw = (float)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  delete[] outArray;
  HIP_ASSERT(hipFree(inArray_g));
  HIP_ASSERT(hipFree(outArray_g));
  output->add_record("memory_copyx4_x8_wave_con", 16, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::memory_copyx4_x8_inst_continue() {
  int THREADS_PER_BLOCK = wave * 64;
  uint MEM_ARRAY_SIZE = arraysize;
  __int128_t *inArray;
  __int128_t *outArray;
  inArray = new __int128_t[MEM_ARRAY_SIZE];
  outArray = new __int128_t[MEM_ARRAY_SIZE];

  __int128_t *inArray_g;
  __int128_t *outArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(
      hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray,
                       MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int128_t) * 2;
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  float peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (uint i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(
        memory_copyx4_x8_inst_continue_kernel, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
          dim3(THREADS_PER_BLOCK), lds * 1024, 0, start_event_, stop_event_, 0,
          inArray_g, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(
        memory_copyx4_x8_inst_continue_kernel, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
          dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    float bw = (float)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  delete[] outArray;
  HIP_ASSERT(hipFree(inArray_g));
  HIP_ASSERT(hipFree(outArray_g));
  output->add_record("memory_copyx4_x8_inst_con", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
#if HIP_ARCH >= 936
void MemoryBandwidth::memory_copyx4_x8_wave_continue_bps_4wave() {
  wave = 4;
  int THREADS_PER_BLOCK = wave * 64;
  uint MEM_ARRAY_SIZE = arraysize;
  __int128_t *inArray;
  __int128_t *outArray;
  inArray = new __int128_t[MEM_ARRAY_SIZE];
  outArray = new __int128_t[MEM_ARRAY_SIZE];

  __int128_t *inArray_g;
  __int128_t *outArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(
      hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray,
                       MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int128_t) * 2;
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  float peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (uint i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(
        memory_copyx4_x8_wave_continue_bps_4wave_kernel, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
          dim3(THREADS_PER_BLOCK), 32 * 1024, 0, start_event_, stop_event_, 0,
          inArray_g, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(
        memory_copyx4_x8_wave_continue_bps_4wave_kernel, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
          dim3(THREADS_PER_BLOCK), 32 * 1024, 0, inArray_g, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    float bw = (float)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  delete[] outArray;
  HIP_ASSERT(hipFree(inArray_g));
  HIP_ASSERT(hipFree(outArray_g));
  output->add_record("memory_copyx4_x8_inst_con_bps", 4, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), 32, device->device_index);
}
void MemoryBandwidth::memory_copyx4_x8_wave_continue_bps_8wave() {
  wave = 8;
  int THREADS_PER_BLOCK = wave * 64;
  uint MEM_ARRAY_SIZE = arraysize;
  __int128_t *inArray;
  __int128_t *outArray;
  inArray = new __int128_t[MEM_ARRAY_SIZE];
  outArray = new __int128_t[MEM_ARRAY_SIZE];

  __int128_t *inArray_g;
  __int128_t *outArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(
      hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray,
                       MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int128_t) * 2;
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  float peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (uint i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(
        memory_copyx4_x8_wave_continue_bps_8wave_kernel, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
          dim3(THREADS_PER_BLOCK), 32 * 1024, 0, start_event_, stop_event_, 0,
          inArray_g, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(
        memory_copyx4_x8_wave_continue_bps_8wave_kernel, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
          dim3(THREADS_PER_BLOCK), 32 * 1024, 0, inArray_g, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    float bw = (float)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  delete[] outArray;
  HIP_ASSERT(hipFree(inArray_g));
  HIP_ASSERT(hipFree(outArray_g));
  output->add_record("memory_copyx4_x8_inst_con_bps", 8, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), 32, device->device_index);
}
void MemoryBandwidth::memory_copyx4_x8_wave_continue_bps_16wave() {
  wave = 16;
  int THREADS_PER_BLOCK = wave * 64;
  uint MEM_ARRAY_SIZE = arraysize;
  __int128_t *inArray;
  __int128_t *outArray;
  inArray = new __int128_t[MEM_ARRAY_SIZE];
  outArray = new __int128_t[MEM_ARRAY_SIZE];

  __int128_t *inArray_g;
  __int128_t *outArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(
      hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray,
                       MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int128_t) * 2;
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  float peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (uint i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(
        memory_copyx4_x8_wave_continue_bps_16wave_kernel, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
          dim3(THREADS_PER_BLOCK), 32 * 1024, 0, start_event_, stop_event_, 0,
          inArray_g, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(
        memory_copyx4_x8_wave_continue_bps_16wave_kernel, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
          dim3(THREADS_PER_BLOCK), 32 * 1024, 0, inArray_g, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    float bw = (float)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  delete[] outArray;
  HIP_ASSERT(hipFree(inArray_g));
  HIP_ASSERT(hipFree(outArray_g));
  output->add_record("memory_copyx4_x8_inst_con_bps", 16, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), 32, device->device_index);
}
#endif

void MemoryBandwidth::memory_copy_x2() {
  int THREADS_PER_BLOCK = wave * 64;
  uint MEM_ARRAY_SIZE = arraysize;
  __int128_t *inArray;
  __int128_t *outArray;
  inArray = new __int128_t[MEM_ARRAY_SIZE];
  outArray = new __int128_t[MEM_ARRAY_SIZE];

  __int128_t *inArray_g;
  __int128_t *outArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(
      hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray,
                       MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int128_t) * 2;
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  float peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (uint i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(
          memory_copy_x2_kernel, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 2),
          dim3(THREADS_PER_BLOCK), lds * 1024, 0, start_event_, stop_event_, 0,
          inArray_g, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(
          memory_copy_x2_kernel, dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 2),
          dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    float bw = (float)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  delete[] outArray;
  HIP_ASSERT(hipFree(inArray_g));
  HIP_ASSERT(hipFree(outArray_g));
  output->add_record("memory_copy_x2", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}

void MemoryBandwidth::matrix_read() {
  const int _WIDTH = WIDTH / crop;
  const int _HEIGHT = HEIGHT / crop;
  uint MEM_ARRAY_SIZE = _WIDTH * _HEIGHT;
  // const int BLOCK_DIM_X = 64;
  // const int BLOCK_DIM_Y = wave;
  // use 1 dim block
  const int BLOCK_DIM_X = 64;
  const int BLOCK_DIM_Y = wave;
  const int BLOCK_SIZE = (BLOCK_DIM_X * BLOCK_DIM_Y);

  assert((wave == 4 || wave == 8 || wave == 16) &&
         "matrix_read 只支持4/8/16 wave带宽测试");
  assert(_WIDTH % BLOCK_DIM_X == 0 &&
         "matrix_read要求：WIDTH 必须是 BLOCK_DIM_X 的倍数");
  assert(_HEIGHT % BLOCK_DIM_Y == 0 &&
         "matrix_read要求：HEIGHT 必须是 BLOCK_DIM_Y 的倍数");

  // Create device buffers
  __int128_t *d_a;
  __int128_t *d_b;

  HIP_ASSERT(hipMalloc((void **)&d_a, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMalloc((void **)&d_b, MEM_ARRAY_SIZE * sizeof(__int128_t)));

  // init
  __int128_t init_value = 1;
  init_kernel<<<dim3(MEM_ARRAY_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_a, init_value);
  init_kernel<<<dim3(MEM_ARRAY_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_b, init_value);

  std::vector<double> timings;
  double t = 0.0;
  float time_ms = 0.0f; // HIP Event
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double> duration;

  for (int i = 0; i < LAUNCH_TIMES; ++i) {

    dim3 grid_size(_WIDTH / BLOCK_DIM_X, _HEIGHT / BLOCK_DIM_Y, 1);
    dim3 block_size(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    if (useEvent) {
      // HIP Event
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(matrix_read_kernel, grid_size, block_size,
                            lds * 1024, 0, start_event_, stop_event_, 0, d_a,
                            d_b, _WIDTH);

      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // ms -> s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      // std::chrono
      t1 = std::chrono::high_resolution_clock::now();
      matrix_read_kernel<<<grid_size, block_size, lds * 1024, 0>>>(d_a, d_b,
                                                                   _WIDTH);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count(); // s
    }

    if (printEach) {
      std::cout << "Launch " << i << (useEvent ? " [Event]" : " [Chrono]")
                << ": " << std::fixed << std::setprecision(6) << t * 1e3
                << " ms" << std::endl;
    }

    timings.push_back(t);
  }

  HIP_ASSERT(hipFree(d_a));
  HIP_ASSERT(hipFree(d_b));

  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double data_size = MEM_ARRAY_SIZE * sizeof(__int128_t);
  double average_bw = pow(10.0, -9.0) * data_size / (average);

  output->add_record("matrix_read", wave, data_size * pow(10.0, -6.0),
                     average_bw, average * 1e3, getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::matrix_read_x4() {
  const int _WIDTH = WIDTH / crop;
  const int _HEIGHT = HEIGHT / crop;
  uint MEM_ARRAY_SIZE = _WIDTH * _HEIGHT;
  // const int BLOCK_DIM_X = 64;
  // const int BLOCK_DIM_Y = wave;
  // use 1 dim block
  const int BLOCK_DIM_X = 64;
  const int BLOCK_DIM_Y = wave;
  const int BLOCK_SIZE = (BLOCK_DIM_X * BLOCK_DIM_Y);

  assert((wave == 4 || wave == 8 || wave == 16) &&
         "matrix_readx4 只支持4/8/16 wave带宽测试");
  assert(_WIDTH % BLOCK_DIM_X == 0 &&
         "matrix_read要求：WIDTH 必须是 BLOCK_DIM_X 的倍数");
  assert(_HEIGHT % BLOCK_DIM_Y == 0 &&
         "matrix_read要求：HEIGHT 必须是 BLOCK_DIM_Y 的倍数");

  // Create device buffers
  __int128_t *d_a;
  __int128_t *d_b;

  HIP_ASSERT(hipMalloc((void **)&d_a, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMalloc((void **)&d_b, MEM_ARRAY_SIZE * sizeof(__int128_t)));

  // init
  __int128_t init_value = 1;
  init_kernel<<<dim3(MEM_ARRAY_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_a, init_value);
  init_kernel<<<dim3(MEM_ARRAY_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_b, init_value);

  std::vector<double> timings;
  double t = 0.0;
  float time_ms = 0.0f; // HIP Event
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double> duration;

  for (int i = 0; i < LAUNCH_TIMES; ++i) {

    dim3 grid_size(_WIDTH / BLOCK_DIM_X / 4, _HEIGHT / BLOCK_DIM_Y, 1);
    dim3 block_size(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    if (useEvent) {
      // HIP Event
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(matrix_read_x4_kernel, grid_size, block_size,
                            lds * 1024, 0, start_event_, stop_event_, 0, d_a,
                            d_b, _WIDTH);

      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // ms -> s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      // std::chrono
      t1 = std::chrono::high_resolution_clock::now();
      matrix_read_x4_kernel<<<grid_size, block_size, lds * 1024, 0>>>(d_a, d_b,
                                                                      _WIDTH);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count(); // s
    }

    if (printEach) {
      std::cout << "Launch " << i << (useEvent ? " [Event]" : " [Chrono]")
                << ": " << std::fixed << std::setprecision(6) << t * 1e3
                << " ms" << std::endl;
    }

    timings.push_back(t);
  }

  HIP_ASSERT(hipFree(d_a));
  HIP_ASSERT(hipFree(d_b));

  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double data_size = MEM_ARRAY_SIZE * sizeof(__int128_t);
  double average_bw = pow(10.0, -9.0) * data_size / (average);

  output->add_record("matrix_read_x4", wave, data_size * pow(10.0, -6.0),
                     average_bw, average * 1e3, getBWtheory(), lds, device->device_index);
}

void MemoryBandwidth::matrix_readx4_x2() {
  const int _WIDTH = WIDTH / crop;
  const int _HEIGHT = HEIGHT / crop;
  uint MEM_ARRAY_SIZE = _WIDTH * _HEIGHT;
  // const int BLOCK_DIM_X = 64;
  // const int BLOCK_DIM_Y = wave;
  // use 1 dim block
  const int BLOCK_DIM_X = 64;
  const int BLOCK_DIM_Y = wave;
  const int BLOCK_SIZE = (BLOCK_DIM_X * BLOCK_DIM_Y);

  assert((wave == 4 || wave == 8 || wave == 16) &&
         "matrix_readx4 只支持4/8/16 wave带宽测试");
  assert(_WIDTH % BLOCK_DIM_X == 0 &&
         "matrix_read要求：WIDTH 必须是 BLOCK_DIM_X 的倍数");
  assert(_HEIGHT % BLOCK_DIM_Y == 0 &&
         "matrix_read要求：HEIGHT 必须是 BLOCK_DIM_Y 的倍数");

  // Create device buffers
  __int128_t *d_a;
  __int128_t *d_b;

  HIP_ASSERT(hipMalloc((void **)&d_a, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMalloc((void **)&d_b, MEM_ARRAY_SIZE * sizeof(__int128_t)));

  // init
  __int128_t init_value = 1;
  init_kernel<<<dim3(MEM_ARRAY_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_a, init_value);
  init_kernel<<<dim3(MEM_ARRAY_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_b, init_value);

  std::vector<double> timings;
  double t = 0.0;
  float time_ms = 0.0f; // HIP Event
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double> duration;

  for (int i = 0; i < LAUNCH_TIMES; ++i) {

    dim3 grid_size((_WIDTH + 2 * BLOCK_DIM_X - 1) / (BLOCK_DIM_X * 2),
                   _HEIGHT / BLOCK_DIM_Y, 1);
    dim3 block_size(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

    if (useEvent) {
      // HIP Event
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(matrix_readx4_x2_kernel, grid_size, block_size,
                            lds * 1024, 0, start_event_, stop_event_, 0, d_a,
                            d_b, _WIDTH);

      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // ms -> s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      // std::chrono
      t1 = std::chrono::high_resolution_clock::now();
      matrix_readx4_x2_kernel<<<grid_size, block_size, lds * 1024, 0>>>(
          d_a, d_b, _WIDTH);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count(); // s
    }

    if (printEach) {
      std::cout << "Launch " << i << (useEvent ? " [Event]" : " [Chrono]")
                << ": " << std::fixed << std::setprecision(6) << t * 1e3
                << " ms" << std::endl;
    }

    timings.push_back(t);
  }

  HIP_ASSERT(hipFree(d_a));
  HIP_ASSERT(hipFree(d_b));

  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double data_size = MEM_ARRAY_SIZE * sizeof(__int128_t);
  double average_bw = pow(10.0, -9.0) * data_size / (average);

  output->add_record("matrix_readx4_x2", wave, data_size * pow(10.0, -6.0),
                     average_bw, average * 1e3, getBWtheory(), lds, device->device_index);
}


void MemoryBandwidth::matrix_readx4_x4() {
  const int _WIDTH = WIDTH / crop;
  const int _HEIGHT = HEIGHT / crop;
  uint MEM_ARRAY_SIZE = _WIDTH * _HEIGHT;
  // const int BLOCK_DIM_X = 64;
  // const int BLOCK_DIM_Y = wave;
  // use 1 dim block
  const int BLOCK_DIM_X = 64;
  const int BLOCK_DIM_Y = wave;
  const int BLOCK_SIZE = (BLOCK_DIM_X * BLOCK_DIM_Y);

  assert((wave == 4 || wave == 8 || wave == 16) &&
         "matrix_readx4 只支持4/8/16 wave带宽测试");
  assert(_WIDTH % BLOCK_DIM_X == 0 &&
         "matrix_read要求：WIDTH 必须是 BLOCK_DIM_X 的倍数");
  assert(_HEIGHT % BLOCK_DIM_Y == 0 &&
         "matrix_read要求：HEIGHT 必须是 BLOCK_DIM_Y 的倍数");

  // Create device buffers
  __int128_t *d_a;
  __int128_t *d_b;

  HIP_ASSERT(hipMalloc((void **)&d_a, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMalloc((void **)&d_b, MEM_ARRAY_SIZE * sizeof(__int128_t)));

  // init
  __int128_t init_value = 1;
  init_kernel<<<dim3(MEM_ARRAY_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_a, init_value);
  init_kernel<<<dim3(MEM_ARRAY_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_b, init_value);

  std::vector<double> timings;
  double t = 0.0;
  float time_ms = 0.0f; // HIP Event
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double> duration;

  for (int i = 0; i < LAUNCH_TIMES; ++i) {

    dim3 grid_size((_WIDTH + 4 * BLOCK_DIM_X - 1) / (BLOCK_DIM_X * 4),
                   _HEIGHT / BLOCK_DIM_Y, 1);
    dim3 block_size(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

    if (useEvent) {
      // HIP Event
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(matrix_readx4_x4_kernel, grid_size, block_size,
                            lds * 1024, 0, start_event_, stop_event_, 0, d_a,
                            d_b, _WIDTH);

      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // ms -> s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      // std::chrono
      t1 = std::chrono::high_resolution_clock::now();
      matrix_readx4_x4_kernel<<<grid_size, block_size, lds * 1024, 0>>>(
          d_a, d_b, _WIDTH);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count(); // s
    }

    if (printEach) {
      std::cout << "Launch " << i << (useEvent ? " [Event]" : " [Chrono]")
                << ": " << std::fixed << std::setprecision(6) << t * 1e3
                << " ms" << std::endl;
    }

    timings.push_back(t);
  }

  HIP_ASSERT(hipFree(d_a));
  HIP_ASSERT(hipFree(d_b));

  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double data_size = MEM_ARRAY_SIZE * sizeof(__int128_t);
  double average_bw = pow(10.0, -9.0) * data_size / (average);

  output->add_record("matrix_readx4_x4", wave, data_size * pow(10.0, -6.0),
                     average_bw, average * 1e3, getBWtheory(), lds, device->device_index);
}

void MemoryBandwidth::matrix_readx4_multi() {
  const int _WIDTH = WIDTH / crop;
  const int _HEIGHT = HEIGHT / crop;
  uint MEM_ARRAY_SIZE = _WIDTH * _HEIGHT;
  // const int BLOCK_DIM_X = 64;
  // const int BLOCK_DIM_Y = wave;
  // use 1 dim block
  const int BLOCK_DIM_X = 64;
  const int BLOCK_DIM_Y = wave;
  const int BLOCK_SIZE = (BLOCK_DIM_X * BLOCK_DIM_Y);

  assert((wave == 4 || wave == 8 || wave == 16) &&
         "matrix_readx4 只支持4/8/16 wave带宽测试");
  assert(_WIDTH % BLOCK_DIM_X == 0 &&
         "matrix_read要求：WIDTH 必须是 BLOCK_DIM_X 的倍数");
  assert(_HEIGHT % BLOCK_DIM_Y == 0 &&
         "matrix_read要求：HEIGHT 必须是 BLOCK_DIM_Y 的倍数");

  // Create device buffers
  __int128_t *d_a;
  __int128_t *d_b;

  HIP_ASSERT(hipMalloc((void **)&d_a, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMalloc((void **)&d_b, MEM_ARRAY_SIZE * sizeof(__int128_t)));

  // init
  __int128_t init_value = 1;
  init_kernel<<<dim3(MEM_ARRAY_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_a, init_value);
  init_kernel<<<dim3(MEM_ARRAY_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_b, init_value);

  std::vector<double> timings;
  double t = 0.0;
  float time_ms = 0.0f; // HIP Event
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double> duration;

  for (int i = 0; i < LAUNCH_TIMES; ++i) {

    dim3 grid_size((_WIDTH + 8 * BLOCK_DIM_X - 1) / (BLOCK_DIM_X * 8),
                   _HEIGHT / BLOCK_DIM_Y, 1);
    dim3 block_size(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

    if (useEvent) {
      // HIP Event
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(matrix_readx4_multi_kernel, grid_size, block_size,
                            lds * 1024, 0, start_event_, stop_event_, 0, d_a,
                            d_b, _WIDTH, 8);

      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // ms -> s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      // std::chrono
      t1 = std::chrono::high_resolution_clock::now();
      matrix_readx4_multi_kernel<<<grid_size, block_size, lds * 1024, 0>>>(
          d_a, d_b, _WIDTH, 8);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count(); // s
    }

    if (printEach) {
      std::cout << "Launch " << i << (useEvent ? " [Event]" : " [Chrono]")
                << ": " << std::fixed << std::setprecision(6) << t * 1e3
                << " ms" << std::endl;
    }

    timings.push_back(t);
  }

  HIP_ASSERT(hipFree(d_a));
  HIP_ASSERT(hipFree(d_b));

  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double data_size = MEM_ARRAY_SIZE * sizeof(__int128_t);
  double average_bw = pow(10.0, -9.0) * data_size / (average);

  output->add_record("matrix_readx4_multi", wave, data_size * pow(10.0, -6.0),
                     average_bw, average * 1e3, getBWtheory(), lds, device->device_index);
}

void MemoryBandwidth::matrix_write() {
  const int _WIDTH = WIDTH / crop;
  const int _HEIGHT = HEIGHT / crop;
  uint MEM_ARRAY_SIZE = _WIDTH * _HEIGHT;

  // use 1 dim block
  const int BLOCK_DIM_X = 64;
  const int BLOCK_DIM_Y = wave;
  const int BLOCK_SIZE = (BLOCK_DIM_X * BLOCK_DIM_Y);

  assert((wave == 4 || wave == 8 || wave == 16) &&
         "matrix_write 只支持4/8/16 wave带宽测试");
  assert(_WIDTH % BLOCK_DIM_X == 0 &&
         "matrix_write要求：WIDTH 必须是 BLOCK_DIM_X 的倍数");
  assert(_HEIGHT % BLOCK_DIM_Y == 0 &&
         "matrix_write要求：HEIGHT 必须是 BLOCK_DIM_Y 的倍数");

  // Create device buffers
  __int128_t *d_a;

  HIP_ASSERT(hipMalloc((void **)&d_a, MEM_ARRAY_SIZE * sizeof(__int128_t)));

  // init
  __int128_t init_value = 1;
  init_kernel<<<dim3(MEM_ARRAY_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_a, init_value);
  std::vector<double> timings;
  double t = 0.0;
  float time_ms = 0.0f; // HIP Event
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double> duration;

  for (int i = 0; i < LAUNCH_TIMES; ++i) {

    dim3 grid_size(_WIDTH / BLOCK_DIM_X, _HEIGHT / BLOCK_DIM_Y, 1);
    dim3 block_size(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

    if (useEvent) {
      // HIP Event
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(matrix_write_kernel, grid_size, block_size,
                            lds * 1024, 0, start_event_, stop_event_, 0, d_a, 1,
                            _WIDTH);

      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // ms -> s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      // std::chrono
      t1 = std::chrono::high_resolution_clock::now();
      matrix_write_kernel<<<grid_size, block_size, lds * 1024, 0>>>(d_a, 1,
                                                                    _WIDTH);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count(); // s
    }

    if (printEach) {
      std::cout << "Launch " << i << (useEvent ? " [Event]" : " [Chrono]")
                << ": " << std::fixed << std::setprecision(6) << t * 1e3
                << " ms" << std::endl;
    }

    timings.push_back(t);
  }

  HIP_ASSERT(hipFree(d_a));

  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double data_size = MEM_ARRAY_SIZE * sizeof(__int128_t);
  double average_bw = pow(10.0, -9.0) * data_size / (average);
  output->add_record("matrix_write", wave, data_size * pow(10.0, -6.0),
                     average_bw, average * 1e3, getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::matrix_write_x4() {
  const int _WIDTH = WIDTH / crop;
  const int _HEIGHT = HEIGHT / crop;
  uint MEM_ARRAY_SIZE = _WIDTH * _HEIGHT;

  // const int BLOCK_DIM_X = 64;
  // const int BLOCK_DIM_Y = wave;
  // use 1 dim block
  const int BLOCK_DIM_X = 64;
  const int BLOCK_DIM_Y = wave;
  const int BLOCK_SIZE = (BLOCK_DIM_X * BLOCK_DIM_Y);
  assert((wave == 4 || wave == 8 || wave == 16) &&
         "matrix_write_x4 只支持4/8/16 wave带宽测试");
  assert(_WIDTH % BLOCK_DIM_X == 0 &&
         "matrix_write要求：WIDTH 必须是 BLOCK_DIM_X 的倍数");
  assert(_HEIGHT % BLOCK_DIM_Y == 0 &&
         "matrix_write要求：HEIGHT 必须是 BLOCK_DIM_Y 的倍数");

  // Create device buffers
  __int128_t *d_a;

  HIP_ASSERT(hipMalloc((void **)&d_a, MEM_ARRAY_SIZE * sizeof(__int128_t)));

  // init
  __int128_t init_value = 1;
  init_kernel<<<dim3(MEM_ARRAY_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_a, init_value);
  std::vector<double> timings;
  double t = 0.0;
  float time_ms = 0.0f; // HIP Event
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double> duration;

  for (int i = 0; i < LAUNCH_TIMES; ++i) {

    dim3 grid_size(_WIDTH / BLOCK_DIM_X / 4, _HEIGHT / BLOCK_DIM_Y, 1);
    dim3 block_size(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

    if (useEvent) {
      // HIP Event
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(matrix_write_x4_kernel, grid_size, block_size,
                            lds * 1024, 0, start_event_, stop_event_, 0, d_a, 1,
                            _WIDTH);

      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // ms -> s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      // std::chrono
      t1 = std::chrono::high_resolution_clock::now();
      matrix_write_x4_kernel<<<grid_size, block_size, lds * 1024, 0>>>(d_a, 1,
                                                                       _WIDTH);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count(); // s
    }

    if (printEach) {
      std::cout << "Launch " << i << (useEvent ? " [Event]" : " [Chrono]")
                << ": " << std::fixed << std::setprecision(6) << t * 1e3
                << " ms" << std::endl;
    }

    timings.push_back(t);
  }

  HIP_ASSERT(hipFree(d_a));

  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double data_size = MEM_ARRAY_SIZE * sizeof(__int128_t);
  double average_bw = pow(10.0, -9.0) * data_size / (average);
  output->add_record("matrix_write_x4", wave, data_size * pow(10.0, -6.0),
                     average_bw, average * 1e3, getBWtheory(), lds, device->device_index);
}

void MemoryBandwidth::matrix_writex4_x2() {
  const int _WIDTH = WIDTH / crop;
  const int _HEIGHT = HEIGHT / crop;
  uint MEM_ARRAY_SIZE = _WIDTH * _HEIGHT;

  // const int BLOCK_DIM_X = 64;
  // const int BLOCK_DIM_Y = wave;
  // use 1 dim block
  const int BLOCK_DIM_X = 64;
  const int BLOCK_DIM_Y = wave;
  const int BLOCK_SIZE = (BLOCK_DIM_X * BLOCK_DIM_Y);
  assert((wave == 4 || wave == 8 || wave == 16) &&
         "matrix_write_x4 只支持4/8/16 wave带宽测试");
  assert(_WIDTH % BLOCK_DIM_X == 0 &&
         "matrix_write要求：WIDTH 必须是 BLOCK_DIM_X 的倍数");
  assert(_HEIGHT % BLOCK_DIM_Y == 0 &&
         "matrix_write要求：HEIGHT 必须是 BLOCK_DIM_Y 的倍数");

  // Create device buffers
  __int128_t *d_a;

  HIP_ASSERT(hipMalloc((void **)&d_a, MEM_ARRAY_SIZE * sizeof(__int128_t)));

  // init
  __int128_t init_value = 1;
  init_kernel<<<dim3(MEM_ARRAY_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_a, init_value);

  std::vector<double> timings;
  double t = 0.0;
  float time_ms = 0.0f; // HIP Event
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double> duration;

  for (int i = 0; i < LAUNCH_TIMES; ++i) {

    dim3 grid_size((_WIDTH + 2 * BLOCK_DIM_X - 1) / (BLOCK_DIM_X * 2),
                   _HEIGHT / BLOCK_DIM_Y, 1);
    dim3 block_size(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

    if (useEvent) {
      // HIP Event
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(matrix_writex4_x2_kernel, grid_size, block_size,
                            lds * 1024, 0, start_event_, stop_event_, 0, d_a, 1,
                            _WIDTH);

      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // ms -> s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      // std::chrono
      t1 = std::chrono::high_resolution_clock::now();
      matrix_writex4_x2_kernel<<<grid_size, block_size, lds * 1024, 0>>>(
          d_a, 1, _WIDTH);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count(); // s
    }

    if (printEach) {
      std::cout << "Launch " << i << (useEvent ? " [Event]" : " [Chrono]")
                << ": " << std::fixed << std::setprecision(6) << t * 1e3
                << " ms" << std::endl;
    }

    timings.push_back(t);
  }

  HIP_ASSERT(hipFree(d_a));

  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double data_size = MEM_ARRAY_SIZE * sizeof(__int128_t);
  double average_bw = pow(10.0, -9.0) * data_size / (average);
  output->add_record("matrix_writex4_x2", wave, data_size * pow(10.0, -6.0),
                     average_bw, average * 1e3, getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::matrix_writex4_x4() {
  const int _WIDTH = WIDTH / crop;
  const int _HEIGHT = HEIGHT / crop;
  uint MEM_ARRAY_SIZE = _WIDTH * _HEIGHT;

  // const int BLOCK_DIM_X = 64;
  // const int BLOCK_DIM_Y = wave;
  // use 1 dim block
  const int BLOCK_DIM_X = 64;
  const int BLOCK_DIM_Y = wave;
  const int BLOCK_SIZE = (BLOCK_DIM_X * BLOCK_DIM_Y);
  assert((wave == 4 || wave == 8 || wave == 16) &&
         "matrix_write_x4 只支持4/8/16 wave带宽测试");
  assert(_WIDTH % BLOCK_DIM_X == 0 &&
         "matrix_write要求：WIDTH 必须是 BLOCK_DIM_X 的倍数");
  assert(_HEIGHT % BLOCK_DIM_Y == 0 &&
         "matrix_write要求：HEIGHT 必须是 BLOCK_DIM_Y 的倍数");

  // Create device buffers
  __int128_t *d_a;

  HIP_ASSERT(hipMalloc((void **)&d_a, MEM_ARRAY_SIZE * sizeof(__int128_t)));

  // init
  __int128_t init_value = 1;
  init_kernel<<<dim3(MEM_ARRAY_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_a, init_value);

  std::vector<double> timings;
  double t = 0.0;
  float time_ms = 0.0f; // HIP Event
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double> duration;

  for (int i = 0; i < LAUNCH_TIMES; ++i) {

    dim3 grid_size((_WIDTH + 4 * BLOCK_DIM_X - 1) / (BLOCK_DIM_X * 4),
                   _HEIGHT / BLOCK_DIM_Y, 1);
    dim3 block_size(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

    if (useEvent) {
      // HIP Event
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(matrix_writex4_x4_kernel, grid_size, block_size,
                            lds * 1024, 0, start_event_, stop_event_, 0, d_a, 1,
                            _WIDTH);

      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // ms -> s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      // std::chrono
      t1 = std::chrono::high_resolution_clock::now();
      matrix_writex4_x4_kernel<<<grid_size, block_size, lds * 1024, 0>>>(
          d_a, 1, _WIDTH);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count(); // s
    }

    if (printEach) {
      std::cout << "Launch " << i << (useEvent ? " [Event]" : " [Chrono]")
                << ": " << std::fixed << std::setprecision(6) << t * 1e3
                << " ms" << std::endl;
    }

    timings.push_back(t);
  }

  HIP_ASSERT(hipFree(d_a));

  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double data_size = MEM_ARRAY_SIZE * sizeof(__int128_t);
  double average_bw = pow(10.0, -9.0) * data_size / (average);
  output->add_record("matrix_writex4_x4", wave, data_size * pow(10.0, -6.0),
                     average_bw, average * 1e3, getBWtheory(), lds, device->device_index);
}

void MemoryBandwidth::matrix_writex4_multi() {
  const int _WIDTH = WIDTH / crop;
  const int _HEIGHT = HEIGHT / crop;
  uint MEM_ARRAY_SIZE = _WIDTH * _HEIGHT;

  // const int BLOCK_DIM_X = 64;
  // const int BLOCK_DIM_Y = wave;
  // use 1 dim block
  const int BLOCK_DIM_X = 64;
  const int BLOCK_DIM_Y = wave;
  const int BLOCK_SIZE = (BLOCK_DIM_X * BLOCK_DIM_Y);
  assert((wave == 4 || wave == 8 || wave == 16) &&
         "matrix_write_x4 只支持4/8/16 wave带宽测试");
  assert(_WIDTH % BLOCK_DIM_X == 0 &&
         "matrix_write要求：WIDTH 必须是 BLOCK_DIM_X 的倍数");
  assert(_HEIGHT % BLOCK_DIM_Y == 0 &&
         "matrix_write要求：HEIGHT 必须是 BLOCK_DIM_Y 的倍数");

  // Create device buffers
  __int128_t *d_a;

  HIP_ASSERT(hipMalloc((void **)&d_a, MEM_ARRAY_SIZE * sizeof(__int128_t)));

  // init
  __int128_t init_value = 1;
  init_kernel<<<dim3(MEM_ARRAY_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_a, init_value);

  std::vector<double> timings;
  double t = 0.0;
  float time_ms = 0.0f; // HIP Event
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double> duration;

  for (int i = 0; i < LAUNCH_TIMES; ++i) {

    dim3 grid_size((_WIDTH + 8 * BLOCK_DIM_X - 1) / (BLOCK_DIM_X * 8),
                   _HEIGHT / BLOCK_DIM_Y, 1);
    dim3 block_size(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

    if (useEvent) {
      // HIP Event
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(matrix_writex4_multi_kernel, grid_size, block_size,
                            lds * 1024, 0, start_event_, stop_event_, 0, d_a, 1,
                            _WIDTH, 8);

      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // ms -> s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      // std::chrono
      t1 = std::chrono::high_resolution_clock::now();
      matrix_writex4_multi_kernel<<<grid_size, block_size, lds * 1024, 0>>>(
          d_a, 1, _WIDTH, 8);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count(); // s
    }

    if (printEach) {
      std::cout << "Launch " << i << (useEvent ? " [Event]" : " [Chrono]")
                << ": " << std::fixed << std::setprecision(6) << t * 1e3
                << " ms" << std::endl;
    }

    timings.push_back(t);
  }

  HIP_ASSERT(hipFree(d_a));

  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double data_size = MEM_ARRAY_SIZE * sizeof(__int128_t);
  double average_bw = pow(10.0, -9.0) * data_size / (average);
  output->add_record("matrix_writex4_multi", wave, data_size * pow(10.0, -6.0),
                     average_bw, average * 1e3, getBWtheory(), lds, device->device_index);
}

void MemoryBandwidth::matrix_copy() {
  const int _WIDTH = WIDTH / crop;
  const int _HEIGHT = HEIGHT / crop;
  uint MEM_ARRAY_SIZE = _WIDTH * _HEIGHT;

  // const int BLOCK_DIM_X = 64;
  // const int BLOCK_DIM_Y = wave;
  // use 1 dim block
  const int BLOCK_DIM_X = 64;
  const int BLOCK_DIM_Y = wave;
  const int BLOCK_SIZE = (BLOCK_DIM_X * BLOCK_DIM_Y);

  assert((wave == 4 || wave == 8 || wave == 16) &&
         "matrix_copy 只支持4/8/16 wave带宽测试");
  assert(_WIDTH % BLOCK_DIM_X == 0 &&
         "matrix_copy要求：WIDTH 必须是 BLOCK_DIM_X 的倍数");
  assert(_HEIGHT % BLOCK_DIM_Y == 0 &&
         "matrix_copy要求：HEIGHT 必须是 BLOCK_DIM_Y 的倍数");

  // Create device buffers
  __int128_t *d_a;
  __int128_t *d_b;

  HIP_ASSERT(hipMalloc((void **)&d_a, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMalloc((void **)&d_b, MEM_ARRAY_SIZE * sizeof(__int128_t)));

  // init
  __int128_t init_value = 1;
  init_kernel<<<dim3(MEM_ARRAY_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_a, init_value);
  init_kernel<<<dim3(MEM_ARRAY_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_b, init_value);

  std::vector<double> timings;
  double t = 0.0;
  float time_ms = 0.0f; // HIP Event
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double> duration;

  for (int i = 0; i < LAUNCH_TIMES; ++i) {

    dim3 grid_size(_WIDTH / BLOCK_DIM_X, _HEIGHT / BLOCK_DIM_Y, 1);
    dim3 block_size(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

    if (useEvent) {
      // HIP Event
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(matrix_copy_kernel, grid_size, block_size,
                            lds * 1024, 0, start_event_, stop_event_, 0, d_a,
                            d_b, _WIDTH);

      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // ms -> s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      // std::chrono
      t1 = std::chrono::high_resolution_clock::now();
      matrix_copy_kernel<<<grid_size, block_size, lds * 1024, 0>>>(d_a, d_b,
                                                                   _WIDTH);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count(); // s
    }

    if (printEach) {
      std::cout << "Launch " << i << (useEvent ? " [Event]" : " [Chrono]")
                << ": " << std::fixed << std::setprecision(6) << t * 1e3
                << " ms" << std::endl;
    }

    timings.push_back(t);
  }

  HIP_ASSERT(hipFree(d_a));
  HIP_ASSERT(hipFree(d_b));

  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double data_size = 2 * MEM_ARRAY_SIZE * sizeof(__int128_t);
  double average_bw = pow(10.0, -9.0) * data_size / (average);
  output->add_record("matrix_copy", wave, data_size * pow(10.0, -6.0),
                     average_bw, average * 1e3, getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::matrix_copy_x4() {
  const int _WIDTH = WIDTH / crop;
  const int _HEIGHT = HEIGHT / crop;
  uint MEM_ARRAY_SIZE = _WIDTH * _HEIGHT;

  // const int BLOCK_DIM_X = 64;
  // const int BLOCK_DIM_Y = wave;
  // use 1 dim block
  const int BLOCK_DIM_X = 64;
  const int BLOCK_DIM_Y = wave;
  const int BLOCK_SIZE = (BLOCK_DIM_X * BLOCK_DIM_Y);
  assert((wave == 4 || wave == 8 || wave == 16) &&
         "matrix_copy_x4 只支持4/8/16 wave带宽测试");
  assert(_WIDTH % BLOCK_DIM_X == 0 &&
         "matrix_copy_x4要求：WIDTH 必须是 BLOCK_DIM_X 的倍数");
  assert(_HEIGHT % BLOCK_DIM_Y == 0 &&
         "matrix_copy要求：HEIGHT 必须是 BLOCK_DIM_Y 的倍数");

  // Create device buffers
  __int128_t *d_a;
  __int128_t *d_b;

  HIP_ASSERT(hipMalloc((void **)&d_a, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMalloc((void **)&d_b, MEM_ARRAY_SIZE * sizeof(__int128_t)));

  // init
  __int128_t init_value = 1;
  init_kernel<<<dim3(MEM_ARRAY_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_a, init_value);
  init_kernel<<<dim3(MEM_ARRAY_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_b, init_value);

  std::vector<double> timings;
  double t = 0.0;
  float time_ms = 0.0f; // HIP Event
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double> duration;

  for (int i = 0; i < LAUNCH_TIMES; ++i) {

    dim3 grid_size(_WIDTH / BLOCK_DIM_X / 4, _HEIGHT / BLOCK_DIM_Y, 1);
    dim3 block_size(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

    if (useEvent) {
      // HIP Event
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(matrix_copy_x4_kernel, grid_size, block_size,
                            lds * 1024, 0, start_event_, stop_event_, 0, d_a,
                            d_b, _WIDTH);

      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // ms -> s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      // std::chrono
      t1 = std::chrono::high_resolution_clock::now();
      matrix_copy_x4_kernel<<<grid_size, block_size, lds * 1024, 0>>>(d_a, d_b,
                                                                      _WIDTH);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count(); // s
    }

    if (printEach) {
      std::cout << "Launch " << i << (useEvent ? " [Event]" : " [Chrono]")
                << ": " << std::fixed << std::setprecision(6) << t * 1e3
                << " ms" << std::endl;
    }

    timings.push_back(t);
  }

  HIP_ASSERT(hipFree(d_a));
  HIP_ASSERT(hipFree(d_b));

  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double data_size = 2 * MEM_ARRAY_SIZE * sizeof(__int128_t);
  double average_bw = pow(10.0, -9.0) * data_size / (average);
  output->add_record("matrix_copy_x4", wave, data_size * pow(10.0, -6.0),
                     average_bw, average * 1e3, getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::matrix_copyx4_x4() {
  const int _WIDTH = WIDTH / crop;
  const int _HEIGHT = HEIGHT / crop;
  uint MEM_ARRAY_SIZE = _WIDTH * _HEIGHT;

  // const int BLOCK_DIM_X = 64;
  // const int BLOCK_DIM_Y = wave;
  // use 1 dim block
  const int BLOCK_DIM_X = 64;
  const int BLOCK_DIM_Y = wave;
  const int BLOCK_SIZE = (BLOCK_DIM_X * BLOCK_DIM_Y);
  assert((wave == 4 || wave == 8 || wave == 16) &&
         "matrix_copy_x4 只支持4/8/16 wave带宽测试");
  assert(_WIDTH % BLOCK_DIM_X == 0 &&
         "matrix_copy_x4要求：WIDTH 必须是 BLOCK_DIM_X 的倍数");
  assert(_HEIGHT % BLOCK_DIM_Y == 0 &&
         "matrix_copy要求：HEIGHT 必须是 BLOCK_DIM_Y 的倍数");

  // Create device buffers
  __int128_t *d_a;
  __int128_t *d_b;

  HIP_ASSERT(hipMalloc((void **)&d_a, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMalloc((void **)&d_b, MEM_ARRAY_SIZE * sizeof(__int128_t)));

  // init
  __int128_t init_value = 1;
  init_kernel<<<dim3(MEM_ARRAY_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_a, init_value);
  init_kernel<<<dim3(MEM_ARRAY_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_b, init_value);

  std::vector<double> timings;
  double t = 0.0;
  float time_ms = 0.0f; // HIP Event
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double> duration;

  for (int i = 0; i < LAUNCH_TIMES; ++i) {

    dim3 grid_size((_WIDTH + 4 * BLOCK_DIM_X - 1) / (BLOCK_DIM_X * 4),
                   _HEIGHT / BLOCK_DIM_Y, 1);
    dim3 block_size(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

    if (useEvent) {
      // HIP Event
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(matrix_copyx4_x4_kernel, grid_size, block_size,
                            lds * 1024, 0, start_event_, stop_event_, 0, d_a,
                            d_b, _WIDTH);

      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // ms -> s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      // std::chrono
      t1 = std::chrono::high_resolution_clock::now();
      matrix_copyx4_x4_kernel<<<grid_size, block_size, lds * 1024, 0>>>(
          d_a, d_b, _WIDTH);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count(); // s
    }

    if (printEach) {
      std::cout << "Launch " << i << (useEvent ? " [Event]" : " [Chrono]")
                << ": " << std::fixed << std::setprecision(6) << t * 1e3
                << " ms" << std::endl;
    }

    timings.push_back(t);
  }

  HIP_ASSERT(hipFree(d_a));
  HIP_ASSERT(hipFree(d_b));

  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double data_size = 2 * MEM_ARRAY_SIZE * sizeof(__int128_t);
  double average_bw = pow(10.0, -9.0) * data_size / (average);
  output->add_record("matrix_copyx4_x4", wave, data_size * pow(10.0, -6.0),
                     average_bw, average * 1e3, getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::matrix_copyx4_multi() {
  const int _WIDTH = WIDTH / crop;
  const int _HEIGHT = HEIGHT / crop;
  uint MEM_ARRAY_SIZE = _WIDTH * _HEIGHT;

  // const int BLOCK_DIM_X = 64;
  // const int BLOCK_DIM_Y = wave;
  // use 1 dim block
  const int BLOCK_DIM_X = 64;
  const int BLOCK_DIM_Y = wave;
  const int BLOCK_SIZE = (BLOCK_DIM_X * BLOCK_DIM_Y);
  assert((wave == 4 || wave == 8 || wave == 16) &&
         "matrix_copy_x4 只支持4/8/16 wave带宽测试");
  assert(_WIDTH % BLOCK_DIM_X == 0 &&
         "matrix_copy_x4要求：WIDTH 必须是 BLOCK_DIM_X 的倍数");
  assert(_HEIGHT % BLOCK_DIM_Y == 0 &&
         "matrix_copy要求：HEIGHT 必须是 BLOCK_DIM_Y 的倍数");

  // Create device buffers
  __int128_t *d_a;
  __int128_t *d_b;

  HIP_ASSERT(hipMalloc((void **)&d_a, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMalloc((void **)&d_b, MEM_ARRAY_SIZE * sizeof(__int128_t)));

  // init
  __int128_t init_value = 1;
  init_kernel<<<dim3(MEM_ARRAY_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_a, init_value);
  init_kernel<<<dim3(MEM_ARRAY_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_b, init_value);

  std::vector<double> timings;
  double t = 0.0;
  float time_ms = 0.0f; // HIP Event
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double> duration;

  for (int i = 0; i < LAUNCH_TIMES; ++i) {

    dim3 grid_size((_WIDTH + 8 * BLOCK_DIM_X - 1) / (BLOCK_DIM_X * 8),
                   _HEIGHT / BLOCK_DIM_Y, 1);
    dim3 block_size(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

    if (useEvent) {
      // HIP Event
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(matrix_copyx4_multi_kernel, grid_size, block_size,
                            lds * 1024, 0, start_event_, stop_event_, 0, d_a,
                            d_b, _WIDTH, 8);

      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // ms -> s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      // std::chrono
      t1 = std::chrono::high_resolution_clock::now();
      matrix_copyx4_multi_kernel<<<grid_size, block_size, lds * 1024, 0>>>(
          d_a, d_b, _WIDTH, 8);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count(); // s
    }

    if (printEach) {
      std::cout << "Launch " << i << (useEvent ? " [Event]" : " [Chrono]")
                << ": " << std::fixed << std::setprecision(6) << t * 1e3
                << " ms" << std::endl;
    }

    timings.push_back(t);
  }

  HIP_ASSERT(hipFree(d_a));
  HIP_ASSERT(hipFree(d_b));

  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double data_size = 2 * MEM_ARRAY_SIZE * sizeof(__int128_t);
  double average_bw = pow(10.0, -9.0) * data_size / (average);
  output->add_record("matrix_copyx4_multi", wave, data_size * pow(10.0, -6.0),
                     average_bw, average * 1e3, getBWtheory(), lds, device->device_index);
}


void MemoryBandwidth::mapping_matrix_write() {
  const int _WIDTH = WIDTH / crop;
  const int _HEIGHT = HEIGHT / crop;
  uint MEM_ARRAY_SIZE = _WIDTH * _HEIGHT;

  const int BLOCK_DIM_X = 64;
  const int BLOCK_DIM_Y = 4;
  const int BLOCK_SIZE = (BLOCK_DIM_X * BLOCK_DIM_Y);

  assert(_WIDTH % BLOCK_DIM_X == 0 &&
         "mapping_matrix_write要求：WIDTH 必须是 BLOCK_DIM_X 的倍数");
  assert(
      _HEIGHT % BLOCK_DIM_Y == 0 &&
      "mapping_matrix_write要求：WIDTH要求：HEIGHT 必须是 BLOCK_DIM_Y 的倍数");

  // Create device buffers
  __int128_t *d_a;

  HIP_ASSERT(hipMalloc((void **)&d_a, MEM_ARRAY_SIZE * sizeof(__int128_t)));

  // init
  __int128_t init_value = 1;
  init_kernel<<<dim3(MEM_ARRAY_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), lds * 1024,
                0>>>(d_a, init_value);

  std::vector<double> timings;
  double t = 0.0;
  float time_ms = 0.0f; // HIP Event
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double> duration;

  for (int i = 0; i < LAUNCH_TIMES; ++i) {

    dim3 grid_size(_WIDTH / BLOCK_DIM_X, _HEIGHT / BLOCK_DIM_Y, 1);
    dim3 block_size(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

    if (useEvent) {
      // HIP Event
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(mapping_matrix_write_kernel, grid_size, block_size,
                            lds * 1024, 0, start_event_, stop_event_, 0, d_a, 1,
                            _WIDTH);

      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // ms -> s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      // std::chrono
      t1 = std::chrono::high_resolution_clock::now();
      mapping_matrix_write_kernel<<<grid_size, block_size, 0, 0>>>(d_a, 1,
                                                                   _WIDTH);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count(); // s
    }

    if (printEach) {
      std::cout << "Launch " << i << (useEvent ? " [Event]" : " [Chrono]")
                << ": " << std::fixed << std::setprecision(6) << t * 1e3
                << " ms" << std::endl;
    }

    timings.push_back(t);
  }

  HIP_ASSERT(hipFree(d_a));

  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double data_size = MEM_ARRAY_SIZE * sizeof(__int128_t);
  double average_bw = pow(10.0, -9.0) * data_size / (average);

  output->add_record("mapping_write", wave, data_size * pow(10.0, -6.0),
                     average_bw, average * 1e3, getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::mapping_matrix_writex4_x4() {
  const int _WIDTH = WIDTH / crop;
  const int _HEIGHT = HEIGHT / crop;
  uint MEM_ARRAY_SIZE = _WIDTH * _HEIGHT;

  const int BLOCK_DIM_X = 64;
  const int BLOCK_DIM_Y = 4;
  const int BLOCK_SIZE = (BLOCK_DIM_X * BLOCK_DIM_Y);

  assert(_WIDTH % BLOCK_DIM_X == 0 &&
         "mapping_matrix_write要求：WIDTH 必须是 BLOCK_DIM_X 的倍数");
  assert(
      _HEIGHT % BLOCK_DIM_Y == 0 &&
      "mapping_matrix_write要求：WIDTH要求：HEIGHT 必须是 BLOCK_DIM_Y 的倍数");

  // Create device buffers
  __int128_t *d_a;

  HIP_ASSERT(hipMalloc((void **)&d_a, MEM_ARRAY_SIZE * sizeof(__int128_t)));

  // init
  __int128_t init_value = 1;
  init_kernel<<<dim3(MEM_ARRAY_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), lds * 1024,
                0>>>(d_a, init_value);

  std::vector<double> timings;
  double t = 0.0;
  float time_ms = 0.0f; // HIP Event
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double> duration;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {

    dim3 grid_size(_WIDTH / BLOCK_DIM_X / 4, _HEIGHT / BLOCK_DIM_Y, 1);
    dim3 block_size(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    if (useEvent) {
      // HIP Event
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(mapping_matrix_write_kernelx4_x4, grid_size, block_size,
                            lds * 1024, 0, start_event_, stop_event_, 0, d_a, 1,
                            _WIDTH);

      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // ms -> s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      // std::chrono
      t1 = std::chrono::high_resolution_clock::now();
      mapping_matrix_write_kernelx4_x4<<<grid_size, block_size, 0, 0>>>(d_a, 1,
                                                                   _WIDTH);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count(); // s
    }

    if (printEach) {
      std::cout << "Launch " << i << (useEvent ? " [Event]" : " [Chrono]")
                << ": " << std::fixed << std::setprecision(6) << t * 1e3
                << " ms" << std::endl;
    }

    timings.push_back(t);
  }
  HIP_ASSERT(hipFree(d_a));
  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double data_size = MEM_ARRAY_SIZE * sizeof(__int128_t);
  double average_bw = pow(10.0, -9.0) * data_size / (average);

  output->add_record("mapping_writex4_x4", wave, data_size * pow(10.0, -6.0),
                     average_bw, average * 1e3, getBWtheory(), lds, device->device_index);
}

void MemoryBandwidth::tiling_read() {
  // conv shape
  const int N = 4;
  const int C = 16;
  const int H = 1024;
  const int W = 1024;
  const int CONV_SIZE = (N * C * H * W);

  const int MT_N = 4;
  const int MT_H = 8;
  const int MT_W = 8;

  const int BLOCK_DIM_X = 64;
  const int BLOCK_DIM_Y = 4;
  const int BLOCK_SIZE = (BLOCK_DIM_X * BLOCK_DIM_Y);

  assert(N % MT_N == 0 && "tiling_read要求：N 必须是 MT_N 的倍数");
  assert(H % (MT_H * 4) == 0 && "tiling_read要求：H 必须是 MT_H * 4 的倍数");
  assert(W % MT_W == 0 && "tiling_read要求：W 必须是 MT_W 的倍数");

  // Create device buffers
  __int128_t *d_a;
  __int128_t *d_b;

  HIP_ASSERT(hipMalloc((void **)&d_a, CONV_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMalloc((void **)&d_b, CONV_SIZE * sizeof(__int128_t)));

  // init
  __int128_t init_value = 1;
  init_kernel<<<dim3(CONV_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_a, init_value);
  init_kernel<<<dim3(CONV_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_b, init_value);

  std::vector<double> timings;
  double t = 0.0;
  float time_ms = 0.0f; // HIP Event
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double> duration;

  for (int i = 0; i < LAUNCH_TIMES; ++i) {

    uint32_t block_num_in_n = ceil(N * 1.0 / MT_N);
    uint32_t block_num_in_h = ceil(H * 1.0 / (MT_H * 4));
    uint32_t block_num_in_w = ceil(W * 1.0 / MT_W);
    uint32_t X_block_num = block_num_in_n * block_num_in_h * block_num_in_w;
    dim3 grid_size(X_block_num, C, 1);
    dim3 block_size(256, 1, 1);

    if (useEvent) {
      // HIP Event
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(tiling_read_kernel, grid_size, block_size,
                            lds * 1024, 0, start_event_, stop_event_, 0, d_a,
                            d_b, block_num_in_h, block_num_in_w, C, H, W, MT_N,
                            MT_H, MT_W);

      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // ms -> s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      // std::chrono
      t1 = std::chrono::high_resolution_clock::now();
      tiling_read_kernel<<<grid_size, block_size, lds * 1024, 0>>>(
          d_a, d_b, block_num_in_h, block_num_in_w, C, H, W, MT_N, MT_H, MT_W);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count(); // s
    }

    if (printEach) {
      std::cout << "Launch " << i << (useEvent ? " [Event]" : " [Chrono]")
                << ": " << std::fixed << std::setprecision(6) << t * 1e3
                << " ms" << std::endl;
    }

    timings.push_back(t);
  }

  HIP_ASSERT(hipFree(d_a));
  HIP_ASSERT(hipFree(d_b));

  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double data_size = CONV_SIZE * sizeof(__int128_t);
  double average_bw = pow(10.0, -9.0) * data_size / (average);
  output->add_record("tiling_read", wave, data_size * pow(10.0, -6.0),
                     average_bw, average * 1e3, getBWtheory(), lds, device->device_index);
}

void MemoryBandwidth::tiling_read_nhwc() {
  // conv shape
  const int N = 512;
  const int H = 14;
  const int W = 14;
  const int C = 1024;
  const int CONV_SIZE = (N * C * H * W);

  // const int MT_N = 1; // batch size
  // const int MT_H = 8;
  // const int MT_W = 8;
  const int MT_C = 4 * 16; // C channel read 4 dwordx4 per thread * 16
  const int MT_WHN = 16; // 4wave 16

  const int BLOCK_DIM_X = 64;
  const int BLOCK_DIM_Y = 4;
  const int BLOCK_SIZE = (BLOCK_DIM_X * BLOCK_DIM_Y);

  // assert(N % MT_N == 0 && "tiling_read要求：N 必须是 MT_N 的倍数");
  // assert(H % (MT_H) == 0 && "tiling_read要求：H 必须是 MT_H 的倍数");
  // assert(C % MT_W == 0 && "tiling_read要求：W 必须是 MT_W 的倍数");

  // Create device buffers
  __int128_t *d_a;
  __int128_t *d_b;

  HIP_ASSERT(hipMalloc((void **)&d_a, CONV_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMalloc((void **)&d_b, CONV_SIZE * sizeof(__int128_t)));

  // init
  __int128_t init_value = 1;
  init_kernel<<<dim3(CONV_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_a, init_value);
  init_kernel<<<dim3(CONV_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_b, init_value);

  std::vector<double> timings;
  double t = 0.0;
  float time_ms = 0.0f; // HIP Event
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double> duration;

  for (int i = 0; i < LAUNCH_TIMES; ++i) {

    uint32_t block_num_in_c = ceil(C * 1.0 / MT_C);
    uint32_t block_num_in_whn = ceil(W * H * N * 1.0 / MT_WHN);

    // uint32_t block_num_in_n = ceil(N * 1.0 / MT_N);
    // uint32_t block_num_in_h = ceil(H * 1.0 / (MT_H * 4));
    // uint32_t block_num_in_w = ceil(W * 1.0 / MT_W);
    uint32_t X_block_num = block_num_in_c * block_num_in_whn;
    dim3 grid_size(X_block_num, 1, 1);
    dim3 block_size(256, 1, 1);

    if (useEvent) {
      // HIP Event
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(tiling_read_nhwc_kernel, grid_size, block_size,
                            lds * 1024, 0, start_event_, stop_event_, 0, d_a,
                            d_b, block_num_in_c, block_num_in_whn, C, H, W, N,
                            MT_C, MT_WHN);

      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // ms -> s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      // std::chrono
      t1 = std::chrono::high_resolution_clock::now();
      tiling_read_nhwc_kernel<<<grid_size, block_size, lds * 1024, 0>>>(
          d_a, d_b, block_num_in_c, block_num_in_whn, C, H, W, N, MT_C, MT_WHN);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count(); // s
    }

    if (printEach) {
      std::cout << "Launch " << i << (useEvent ? " [Event]" : " [Chrono]")
                << ": " << std::fixed << std::setprecision(6) << t * 1e3
                << " ms" << std::endl;
    }

    timings.push_back(t);
  }

  HIP_ASSERT(hipFree(d_a));
  HIP_ASSERT(hipFree(d_b));

  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double data_size = CONV_SIZE * sizeof(__int128_t);
  double average_bw = pow(10.0, -9.0) * data_size / (average);
  output->add_record("tiling_read_nhwc", wave, data_size * pow(10.0, -6.0),
                     average_bw, average * 1e3, getBWtheory(), lds, device->device_index);
}

void MemoryBandwidth::tiling_read_nchw_16t() {
  // conv shape
  const int N = 4;
  const int C = 16;
  const int H = 1024;
  const int W = 1024;
  const int CONV_SIZE = (N * C * H * W);

  const int MT_N = 4;
  const int MT_H = 4;
  const int MT_W = 16;

  const int BLOCK_DIM_X = 64;
  const int BLOCK_DIM_Y = 4;
  const int BLOCK_SIZE = (BLOCK_DIM_X * BLOCK_DIM_Y);

  assert(N % MT_N == 0 && "tiling_read要求：N 必须是 MT_N 的倍数");
  assert(H % (MT_H * 4) == 0 && "tiling_read要求：H 必须是 MT_H * 4 的倍数");
  assert(W % MT_W == 0 && "tiling_read要求：W 必须是 MT_W 的倍数");

  // Create device buffers
  __int128_t *d_a;
  __int128_t *d_b;

  HIP_ASSERT(hipMalloc((void **)&d_a, CONV_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMalloc((void **)&d_b, CONV_SIZE * sizeof(__int128_t)));

  // init
  __int128_t init_value = 1;
  init_kernel<<<dim3(CONV_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_a, init_value);
  init_kernel<<<dim3(CONV_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_b, init_value);

  std::vector<double> timings;
  double t = 0.0;
  float time_ms = 0.0f; // HIP Event
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double> duration;

  for (int i = 0; i < LAUNCH_TIMES; ++i) {

    uint32_t block_num_in_n = ceil(N * 1.0 / MT_N);
    uint32_t block_num_in_h = ceil(H * 1.0 / (MT_H * 4));
    uint32_t block_num_in_w = ceil(W * 1.0 / MT_W);
    uint32_t X_block_num = block_num_in_n * block_num_in_h * block_num_in_w;
    dim3 grid_size(X_block_num, C, 1);
    dim3 block_size(256, 1, 1);

    if (useEvent) {
      // HIP Event
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(tiling_read_nchw_kernel, grid_size, block_size,
                            lds * 1024, 0, start_event_, stop_event_, 0, d_a,
                            d_b, block_num_in_h, block_num_in_w, C, H, W, MT_N,
                            MT_H, MT_W);

      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // ms -> s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      // std::chrono
      t1 = std::chrono::high_resolution_clock::now();
      tiling_read_nchw_kernel<<<grid_size, block_size, lds * 1024, 0>>>(
          d_a, d_b, block_num_in_h, block_num_in_w, C, H, W, MT_N, MT_H, MT_W);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count(); // s
    }

    if (printEach) {
      std::cout << "Launch " << i << (useEvent ? " [Event]" : " [Chrono]")
                << ": " << std::fixed << std::setprecision(6) << t * 1e3
                << " ms" << std::endl;
    }

    timings.push_back(t);
  }

  HIP_ASSERT(hipFree(d_a));
  HIP_ASSERT(hipFree(d_b));

  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double data_size = CONV_SIZE * sizeof(__int128_t);
  double average_bw = pow(10.0, -9.0) * data_size / (average);
  output->add_record("tiling_read_nchw", wave, data_size * pow(10.0, -6.0),
                     average_bw, average * 1e3, getBWtheory(), lds, device->device_index);
}

void MemoryBandwidth::tiling_read_w() {
  // conv shape
  const int N = 4;
  const int C = 16;
  const int H = 1024;
  const int W = 1024;
  const int CONV_SIZE = (N * C * H * W);

  const int MT_N = 4;
  const int MT_H = 8;
  const int MT_W = 8;

  const int BLOCK_DIM_X = 64;
  const int BLOCK_DIM_Y = 4;
  const int BLOCK_SIZE = (BLOCK_DIM_X * BLOCK_DIM_Y);

  assert(N % MT_N == 0 && "tiling_read要求：N 必须是 MT_N 的倍数");
  assert(H % (MT_H * 4) == 0 && "tiling_read要求：H 必须是 MT_H * 4 的倍数");
  assert(W % MT_W == 0 && "tiling_read要求：W 必须是 MT_W 的倍数");

  // Create device buffers
  __int128_t *d_a;
  __int128_t *d_b;

  HIP_ASSERT(hipMalloc((void **)&d_a, CONV_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMalloc((void **)&d_b, CONV_SIZE * sizeof(__int128_t)));

  // init
  __int128_t init_value = 1;
  init_kernel<<<dim3(CONV_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_a, init_value);
  init_kernel<<<dim3(CONV_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_b, init_value);

  std::vector<double> timings;
  double t = 0.0;
  float time_ms = 0.0f; // HIP Event
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double> duration;

  for (int i = 0; i < LAUNCH_TIMES; ++i) {

    uint32_t block_num_in_n = ceil(N * 1.0 / MT_N);
    uint32_t block_num_in_h = ceil(H * 1.0 / (MT_H));
    uint32_t block_num_in_w = ceil(W * 1.0 / (MT_W * 4));
    uint32_t X_block_num = block_num_in_n * block_num_in_h * block_num_in_w;
    dim3 grid_size(X_block_num, C, 1);
    dim3 block_size(256, 1, 1);

    if (useEvent) {
      // HIP Event
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(tiling_read_W_kernel, grid_size, block_size,
                            lds * 1024, 0, start_event_, stop_event_, 0, d_a,
                            d_b, block_num_in_h, block_num_in_w, C, H, W, MT_N,
                            MT_H, MT_W);

      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // ms -> s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      // std::chrono
      t1 = std::chrono::high_resolution_clock::now();
      tiling_read_W_kernel<<<grid_size, block_size, lds * 1024, 0>>>(
          d_a, d_b, block_num_in_h, block_num_in_w, C, H, W, MT_N, MT_H, MT_W);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count(); // s
    }

    if (printEach) {
      std::cout << "Launch " << i << (useEvent ? " [Event]" : " [Chrono]")
                << ": " << std::fixed << std::setprecision(6) << t * 1e3
                << " ms" << std::endl;
    }

    timings.push_back(t);
  }

  HIP_ASSERT(hipFree(d_a));
  HIP_ASSERT(hipFree(d_b));

  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double data_size = CONV_SIZE * sizeof(__int128_t);
  double average_bw = pow(10.0, -9.0) * data_size / (average);
  output->add_record("tiling_read", wave, data_size * pow(10.0, -6.0),
                     average_bw, average * 1e3, getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::tiling_read_2h() {
  // conv shape
  const int N = 4;
  const int C = 16;
  const int H = 1024;
  const int W = 1024;
  const int CONV_SIZE = (N * C * H * W);

  const int MT_N = 4;
  const int MT_H = 8;
  const int MT_W = 8;

  const int BLOCK_DIM_X = 64;
  const int BLOCK_DIM_Y = 4;
  const int BLOCK_SIZE = (BLOCK_DIM_X * BLOCK_DIM_Y);

  assert(N % MT_N == 0 && "tiling_read要求：N 必须是 MT_N 的倍数");
  assert(H % (MT_H * 4) == 0 && "tiling_read要求：H 必须是 MT_H * 4 的倍数");
  assert(W % MT_W == 0 && "tiling_read要求：W 必须是 MT_W 的倍数");

  // Create device buffers
  __int128_t *d_a;
  __int128_t *d_b;

  HIP_ASSERT(hipMalloc((void **)&d_a, CONV_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMalloc((void **)&d_b, CONV_SIZE * sizeof(__int128_t)));

  // init
  __int128_t init_value = 1;
  init_kernel<<<dim3(CONV_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_a, init_value);
  init_kernel<<<dim3(CONV_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_b, init_value);

  std::vector<double> timings;
  double t = 0.0;
  float time_ms = 0.0f; // HIP Event
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double> duration;

  for (int i = 0; i < LAUNCH_TIMES; ++i) {

    uint32_t block_num_in_n = ceil(N * 1.0 / MT_N);
    uint32_t block_num_in_h =
        ceil(H * 1.0 / (MT_H * 8)); // MT_H方向 指令数目x2，从4变成8
    uint32_t block_num_in_w = ceil(W * 1.0 / MT_W);
    uint32_t X_block_num = block_num_in_n * block_num_in_h * block_num_in_w;
    dim3 grid_size(X_block_num, C, 1);
    dim3 block_size(256, 1, 1);

    if (useEvent) {
      // HIP Event
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(tiling_read_kernel_2H_inst, grid_size, block_size,
                            lds * 1024, 0, start_event_, stop_event_, 0, d_a,
                            d_b, block_num_in_h, block_num_in_w, C, H, W, MT_N,
                            MT_H, MT_W);

      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // ms -> s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      // std::chrono
      t1 = std::chrono::high_resolution_clock::now();
      tiling_read_kernel_2H_inst<<<grid_size, block_size, lds * 1024, 0>>>(
          d_a, d_b, block_num_in_h, block_num_in_w, C, H, W, MT_N, MT_H, MT_W);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count(); // s
    }

    if (printEach) {
      std::cout << "Launch " << i << (useEvent ? " [Event]" : " [Chrono]")
                << ": " << std::fixed << std::setprecision(6) << t * 1e3
                << " ms" << std::endl;
    }

    timings.push_back(t);
  }

  HIP_ASSERT(hipFree(d_a));
  HIP_ASSERT(hipFree(d_b));

  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double data_size = CONV_SIZE * sizeof(__int128_t);
  double average_bw = pow(10.0, -9.0) * data_size / (average);
  output->add_record("tiling_read_kernel_2H_inst", wave,
                     data_size * pow(10.0, -6.0), average_bw, average * 1e3,
                     getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::tiling_read_2w() {
  // conv shape
  const int N = 4;
  const int C = 16;
  const int H = 1024;
  const int W = 1024;
  const int CONV_SIZE = (N * C * H * W);

  const int MT_N = 4;
  const int MT_H = 8;
  const int MT_W = 8;

  const int BLOCK_DIM_X = 64;
  const int BLOCK_DIM_Y = 4;
  const int BLOCK_SIZE = (BLOCK_DIM_X * BLOCK_DIM_Y);

  assert(N % MT_N == 0 && "tiling_read要求：N 必须是 MT_N 的倍数");
  assert(H % (MT_H * 4) == 0 && "tiling_read要求：H 必须是 MT_H * 4 的倍数");
  assert(W % MT_W == 0 && "tiling_read要求：W 必须是 MT_W 的倍数");

  // Create device buffers
  __int128_t *d_a;
  __int128_t *d_b;

  HIP_ASSERT(hipMalloc((void **)&d_a, CONV_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMalloc((void **)&d_b, CONV_SIZE * sizeof(__int128_t)));

  // init
  __int128_t init_value = 1;
  init_kernel<<<dim3(CONV_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_a, init_value);
  init_kernel<<<dim3(CONV_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_b, init_value);

  std::vector<double> timings;
  double t = 0.0;
  float time_ms = 0.0f; // HIP Event
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double> duration;

  for (int i = 0; i < LAUNCH_TIMES; ++i) {

    uint32_t block_num_in_n = ceil(N * 1.0 / MT_N);
    uint32_t block_num_in_h = ceil(H * 1.0 / (MT_H * 4));
    uint32_t block_num_in_w =
        ceil(W * 1.0 / (MT_W * 2)); // MT_H方向 指令数目x2，从4变成8
    uint32_t X_block_num = block_num_in_n * block_num_in_h * block_num_in_w;
    dim3 grid_size(X_block_num, C, 1);
    dim3 block_size(256, 1, 1);

    if (useEvent) {
      // HIP Event
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(tiling_read_kernel_2W_inst, grid_size, block_size,
                            lds * 1024, 0, start_event_, stop_event_, 0, d_a,
                            d_b, block_num_in_h, block_num_in_w, C, H, W, MT_N,
                            MT_H, MT_W);

      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // ms -> s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      // std::chrono
      t1 = std::chrono::high_resolution_clock::now();
      tiling_read_kernel_2W_inst<<<grid_size, block_size, lds * 1024, 0>>>(
          d_a, d_b, block_num_in_h, block_num_in_w, C, H, W, MT_N, MT_H, MT_W);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count(); // s
    }

    if (printEach) {
      std::cout << "Launch " << i << (useEvent ? " [Event]" : " [Chrono]")
                << ": " << std::fixed << std::setprecision(6) << t * 1e3
                << " ms" << std::endl;
    }

    timings.push_back(t);
  }

  HIP_ASSERT(hipFree(d_a));
  HIP_ASSERT(hipFree(d_b));

  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double data_size = CONV_SIZE * sizeof(__int128_t);
  double average_bw = pow(10.0, -9.0) * data_size / (average);
  output->add_record("tiling_read_kernel_2W_inst", wave,
                     data_size * pow(10.0, -6.0), average_bw, average * 1e3,
                     getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::winograd_read() {
  // conv shape
  const int N = 4;
  const int C = 16;
  const int H = 64;
  const int W = 64;
  const int CONV_SIZE = (N * C * H * W);

  const int MT_N = 4;
  const int MT_H = 8;
  const int MT_W = 8;

  const int BLOCK_DIM_X = 64;
  const int BLOCK_DIM_Y = 4;
  const int BLOCK_SIZE = (BLOCK_DIM_X * BLOCK_DIM_Y);

  assert(N % MT_N == 0 && "tiling_read要求：N 必须是 MT_N 的倍数");
  assert(H % (MT_H * 4) == 0 && "tiling_read要求：H 必须是 MT_H * 4 的倍数");
  assert(W % MT_W == 0 && "tiling_read要求：W 必须是 MT_W 的倍数");

  // Create device buffers
  __int128_t *d_a;
  __int128_t *d_b;

  HIP_ASSERT(hipMalloc((void **)&d_a, CONV_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMalloc((void **)&d_b, CONV_SIZE * sizeof(__int128_t)));

  // init
  __int128_t init_value = 1;
  init_kernel<<<dim3(CONV_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_a, init_value);
  init_kernel<<<dim3(CONV_SIZE / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_b, init_value);

  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1000>> duration;
  std::vector<double> timings;

  for (int k = 0; k < 1000; ++k) {
    t1 = std::chrono::high_resolution_clock::now();
    uint32_t block_num_in_n = ceil(N * 1.0 / MT_N);
    uint32_t block_num_in_h = ceil(H * 1.0 / (MT_H * 4));
    uint32_t block_num_in_w = ceil(W * 1.0 / MT_W);
    uint32_t X_block_num = block_num_in_n * (H - 31) * (W - 7);
    // 一共有 1023 * 1023 * 4 个block
    dim3 grid_size(X_block_num, C, 1);
    dim3 block_size(256, 1, 1);
    winograd_read_kernel<<<grid_size, block_size, lds * 1024, 0>>>(
        d_a, d_b, block_num_in_h, block_num_in_w, C, H, W, MT_N, MT_H, MT_W);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    timings.push_back(duration.count());
  }

  HIP_ASSERT(hipFree(d_a));
  HIP_ASSERT(hipFree(d_b));

  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double data_size =
      (N * C * (H - 31) * (W - 7) * 4 * 32 * 8) * sizeof(__int128_t);
  double average_bw = pow(10.0, -6.0) * data_size / (average);
  output->add_record("winograd_read", wave, data_size * pow(10.0, -6.0),
                     average_bw, average, getBWtheory(), lds, device->device_index);
}

void MemoryBandwidth::memory_to_ldsx1() {
  int THREADS_PER_BLOCK = wave * 64;
  uint MEM_ARRAY_SIZE = arraysize;
  int *inArray;
  int size = MEM_ARRAY_SIZE * 4;
  inArray = new int[size]; // 1.562GB

  int *inArray_g;
  HIP_ASSERT(hipMalloc((void **)&inArray_g, size * sizeof(int)));
  HIP_ASSERT(
      hipMemcpy(inArray_g, inArray, size * sizeof(int), hipMemcpyHostToDevice));

  uint64_t data = size * sizeof(int);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  float peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (uint i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(memory_to_ldsx1_kernel,
                            dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK),
                            dim3(THREADS_PER_BLOCK), lds * 1024, 0,
                            start_event_, stop_event_, 0, inArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(memory_to_ldsx1_kernel,
                         dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK),
                         dim3(THREADS_PER_BLOCK), 1024 * lds, 0, inArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    float bw = (float)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));
  output->add_record("memory_to_ldsx1", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), 4, device->device_index);
}
#if HIP_ARCH >= 936
void MemoryBandwidth::memory_to_ldsx4() {

  int THREADS_PER_BLOCK = wave * 64;
  uint MEM_ARRAY_SIZE = arraysize;
  __int128_t *inArray;
  inArray = new __int128_t[MEM_ARRAY_SIZE]; // 1.562GB

  __int128_t *inArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int128_t);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(memory_to_ldsx4_kernel,
                            dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK),
                            dim3(THREADS_PER_BLOCK), lds * 1024, 0,
                            start_event_, stop_event_, 0, inArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(memory_to_ldsx4_kernel,
                         dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK),
                         dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));
  output->add_record("memory_to_ldsx4", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), 4, device->device_index);
}
void MemoryBandwidth::memory_to_ldsx4_x4() {
  int THREADS_PER_BLOCK = wave * 64;
  uint MEM_ARRAY_SIZE = ARRAY_SIZE;
  __int128_t *inArray;
  inArray = new __int128_t[MEM_ARRAY_SIZE]; // 1.562GB
  // for (int i = 0; i < MEM_ARRAY_SIZE; ++i) {
  //   h_A[i] = rand() % 256 - 128;  // 随机初始化 int8_t 数据 (-128 到 127)
  // }

  __int128_t *inArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int128_t);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(memory_to_ldsx4_x4_kernel,
                            dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 4),
                            dim3(THREADS_PER_BLOCK), lds * 1024, 0,
                            start_event_, stop_event_, 0, inArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(memory_to_ldsx4_x4_kernel,
                         dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 4),
                         dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    // std::cout<<bw<<std::endl;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));
  output->add_record("memory_to_ldsx4_x4", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), 4, device->device_index);
}
#endif
void MemoryBandwidth::lds_to_memory() {
  int THREADS_PER_BLOCK = wave * 64;
  uint MEM_ARRAY_SIZE = arraysize;
  __int128_t *outArray;
  outArray = new __int128_t[MEM_ARRAY_SIZE];

  __int128_t *outArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray,
                       MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int128_t);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  float peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (uint i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(lds_to_memory_kernel,
                            dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK),
                            dim3(THREADS_PER_BLOCK), lds * 1024, 0,
                            start_event_, stop_event_, 0, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(lds_to_memory_kernel,
                         dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK),
                         dim3(THREADS_PER_BLOCK), lds * 1024, 0, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    float bw = (float)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] outArray;
  HIP_ASSERT(hipFree(outArray_g));
  output->add_record("lds_to_memory", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::lds_to_memory_x4() {
  int THREADS_PER_BLOCK = wave * 64;
  uint MEM_ARRAY_SIZE = ARRAY_SIZE;
  __int128_t *outArray;
  outArray = new __int128_t[MEM_ARRAY_SIZE];

  __int128_t *outArray_g;
  HIP_ASSERT(
      hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray,
                       MEM_ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int128_t);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  float time_ms = 0.0f;
  float peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (uint i = 0; i < LAUNCH_TIMES; ++i) {
    if (useEvent) {
      // -------------------- HIP EVENT 模式 --------------------
      hipEvent_t start_event_, stop_event_;
      HIP_ASSERT(hipEventCreate(&start_event_));
      HIP_ASSERT(hipEventCreate(&stop_event_));

      hipExtLaunchKernelGGL(lds_to_memory_x4_kernel,
                            dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 4),
                            dim3(THREADS_PER_BLOCK), lds * 1024, 0,
                            start_event_, stop_event_, 0, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      HIP_ASSERT(hipEventElapsedTime(&time_ms, start_event_, stop_event_));
      t = time_ms / 1e3; // convert ms → s

      HIP_ASSERT(hipEventDestroy(start_event_));
      HIP_ASSERT(hipEventDestroy(stop_event_));
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(lds_to_memory_x4_kernel,
                         dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 4),
                         dim3(THREADS_PER_BLOCK), lds * 1024, 0, outArray_g);
      HIP_ASSERT(hipDeviceSynchronize());
      t2 = std::chrono::high_resolution_clock::now();
      duration = t2 - t1;
      t = duration.count();
    }

    if (printEach) {
      std::cout << "Launch " << i << ": " << std::fixed << std::setprecision(6)
                << t * 1e3 << " ms" << "  BW: " << (double)data * 1e-9 / t
                << "GB/s " << std::endl;
    }

    if (i == 0) {
      continue;
    } // ignore the first launch
    float bw = (float)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] outArray;
  HIP_ASSERT(hipFree(outArray_g));
  output->add_record("lds_to_memory_x4", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::l1_read() {
  const int BLOCKS_NUM = device->cuNum;
  const int THREADS_PER_BLOCK = 1024;
  double SCLK = device->get_sclk();
  const int TOTAL_THREADS = THREADS_PER_BLOCK * device->cuNum;
  const int MEM_ARRAY_SIZE = TOTAL_THREADS * 4;
  const int LOAD_TIMES = 1008;
  const int WAVE_NUM = TOTAL_THREADS / WAVE_SIZE;
  uint64_t startClk[TOTAL_THREADS];
  uint64_t stopClk[TOTAL_THREADS];
  int inArray[MEM_ARRAY_SIZE], outArray[MEM_ARRAY_SIZE], golden[MEM_ARRAY_SIZE];

  for (uint32_t i = 0; i < MEM_ARRAY_SIZE; i++) {
    golden[i] = inArray[i] = i;
    outArray[i] = 3;
  }

  uint64_t *startClk_g;
  uint64_t *stopClk_g;
  int *inArray_g;
  int *outArray_g;

  HIP_ASSERT(hipMalloc((void **)&startClk_g, TOTAL_THREADS * sizeof(uint64_t)));
  HIP_ASSERT(hipMalloc((void **)&stopClk_g, TOTAL_THREADS * sizeof(uint64_t)));
  HIP_ASSERT(hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(int)));
  HIP_ASSERT(hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(int)));

  HIP_ASSERT(hipMemcpy(startClk_g, startClk, TOTAL_THREADS * sizeof(uint64_t),
                       hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(stopClk_g, stopClk, TOTAL_THREADS * sizeof(uint64_t),
                       hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(int),
                       hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray, MEM_ARRAY_SIZE * sizeof(int),
                       hipMemcpyHostToDevice));

  /* repeat launch kernel to get average */
  double max_clk = 0.0;
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  double data = MEM_ARRAY_SIZE * LOAD_TIMES * sizeof(int);
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    hipLaunchKernelGGL(l1_read_kernel, dim3(BLOCKS_NUM),
                       dim3(THREADS_PER_BLOCK), lds * 1024, 0, startClk_g,
                       stopClk_g, inArray_g, outArray_g, LOAD_TIMES);

    HIP_ASSERT(hipMemcpy(startClk, startClk_g, TOTAL_THREADS * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));
    HIP_ASSERT(hipMemcpy(stopClk, stopClk_g, TOTAL_THREADS * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));
    HIP_ASSERT(hipMemcpy(inArray, inArray_g, MEM_ARRAY_SIZE * sizeof(int),
                         hipMemcpyDeviceToHost));
    HIP_ASSERT(hipMemcpy(outArray, outArray_g, MEM_ARRAY_SIZE * sizeof(int),
                         hipMemcpyDeviceToHost));
    max_clk = 0.0;
    double bw = 0.0; //, avg_clk = 0.0;
    for (int i = 0; i < WAVE_NUM; i++) {
      double wave_clk = stopClk[WAVE_SIZE * i] - startClk[WAVE_SIZE * i];
      if (max_clk < wave_clk)
        max_clk = wave_clk;
      // avg_clk += stopClk[WAVE_SIZE * i] - startClk[WAVE_SIZE * i];
    }
    // avg_clk /= WAVE_NUM;
    bw = data / max_clk;
    avg_lat += max_clk;
    // bw = data / avg_clk;
    // avg_lat += avg_clk;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  HIP_ASSERT(hipFree(startClk_g));
  HIP_ASSERT(hipFree(stopClk_g));
  HIP_ASSERT(hipFree(inArray_g));
  HIP_ASSERT(hipFree(outArray_g));

  output->add_record("l1_read", 16 /*wave*/, data * pow(10.0, -6.0),
                     (avg_bw / LAUNCH_TIMES) * pow(10.0, -9.0) * 1e6 * SCLK,
                     avg_lat / LAUNCH_TIMES, getLdsCacheTheory("L1"), lds, device->device_index);
}
void MemoryBandwidth::l2_read() {
  double SCLK = device->get_sclk();
  const int BLOCKS_NUM = device->cuNum;
  const int THREADS_PER_BLOCK = 1024;
  const int TOTAL_THREADS = BLOCKS_NUM * THREADS_PER_BLOCK;
  const int MEM_ARRAY_SIZE = TOTAL_THREADS;
  const uint WAVE_NUM = TOTAL_THREADS / WAVE_SIZE;
  const uint LOAD_TIMES = 1008;

  // host malloc and init
  uint64_t startClk[TOTAL_THREADS], stopClk[TOTAL_THREADS];
  float4_type inArray[MEM_ARRAY_SIZE];
  memset(startClk, 0, sizeof(startClk));
  memset(stopClk, 0, sizeof(stopClk));
  memset(inArray, 0.0, sizeof(inArray));
  for (int i = 0; i < MEM_ARRAY_SIZE; ++i) {
    for (int j = 0; j < 4; ++j) {
      inArray[i][j] = (float)j;
    }
  }

  uint64_t *startClk_g;
  uint64_t *stopClk_g;
  float4_type *inArray_g;

  HIP_ASSERT(hipMalloc((void **)&startClk_g, TOTAL_THREADS * sizeof(uint64_t)));
  HIP_ASSERT(hipMalloc((void **)&stopClk_g, TOTAL_THREADS * sizeof(uint64_t)));
  HIP_ASSERT(hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(float4_type)));

  HIP_ASSERT(hipMemcpy(startClk_g, startClk, TOTAL_THREADS * sizeof(uint64_t),
                       hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(stopClk_g, stopClk, TOTAL_THREADS * sizeof(uint64_t),
                       hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(float4_type),
                       hipMemcpyHostToDevice));

  double data = MEM_ARRAY_SIZE * LOAD_TIMES * sizeof(float4_type);
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  double max_clk = 0.0;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {

    hipLaunchKernelGGL(l2_read_kernel, dim3(BLOCKS_NUM),
                       dim3(THREADS_PER_BLOCK), lds * 1024, 0, startClk_g,
                       stopClk_g, inArray_g, LOAD_TIMES);
    HIP_ASSERT(hipMemcpy(startClk, startClk_g, TOTAL_THREADS * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));
    HIP_ASSERT(hipMemcpy(stopClk, stopClk_g, TOTAL_THREADS * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));
    max_clk = 0.0;
    // double avg_clk = 0.0;
    for (int i = 0; i < WAVE_NUM; ++i) {
      // avg_clk += stopClk[WAVE_SIZE * i] - startClk[WAVE_SIZE * i];
      double wave_clk = stopClk[WAVE_SIZE * i] - startClk[WAVE_SIZE * i];
      if (max_clk < wave_clk)
        max_clk = wave_clk;
    }
    // avg_clk /= WAVE_NUM;
    // double bw = data / avg_clk;
    // avg_lat += avg_clk;
    double bw = data / max_clk;
    avg_lat += max_clk;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  HIP_ASSERT(hipFree(startClk_g));
  HIP_ASSERT(hipFree(stopClk_g));
  HIP_ASSERT(hipFree(inArray_g));
  output->add_record("l2_read", 8 /*wave*/, data * pow(10.0, -6.0),
                     (avg_bw / LAUNCH_TIMES) * pow(10.0, -9.0) * 1e6 * SCLK,
                     avg_lat / LAUNCH_TIMES, getLdsCacheTheory("L2 read"), lds, device->device_index);
}
void MemoryBandwidth::l2_write() {
  double SCLK = device->get_sclk();
  const int THREADS_PER_BLOCK = 512;
  const int BLOCKS_NUM = device->cuNum;
  const int TOTAL_THREADS = BLOCKS_NUM * THREADS_PER_BLOCK;
  const int MEM_ARRAY_SIZE = TOTAL_THREADS;
  const uint WAVE_NUM = TOTAL_THREADS / WAVE_SIZE;
  const uint LOAD_TIMES = 1000;
  uint64_t startClk[TOTAL_THREADS];
  uint64_t stopClk[TOTAL_THREADS];
  float4_type outArray[MEM_ARRAY_SIZE];

  uint64_t *startClk_g;
  uint64_t *stopClk_g;
  float4_type *outArray_g;

  HIP_ASSERT(hipMalloc((void **)&startClk_g, TOTAL_THREADS * sizeof(uint64_t)));
  HIP_ASSERT(hipMalloc((void **)&stopClk_g, TOTAL_THREADS * sizeof(uint64_t)));
  HIP_ASSERT(
      hipMalloc((void **)&outArray_g, MEM_ARRAY_SIZE * sizeof(float4_type)));

  HIP_ASSERT(hipMemcpy(startClk_g, startClk, TOTAL_THREADS * sizeof(uint64_t),
                       hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(stopClk_g, stopClk, TOTAL_THREADS * sizeof(uint64_t),
                       hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(outArray_g, outArray, MEM_ARRAY_SIZE * sizeof(float4_type),
                       hipMemcpyHostToDevice));

  double max_clk = 0.0;
  double data = MEM_ARRAY_SIZE * LOAD_TIMES * sizeof(float4_type);
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    hipLaunchKernelGGL(l2_write_kernel, dim3(BLOCKS_NUM),
                       dim3(THREADS_PER_BLOCK), lds * 1024, 0, startClk_g,
                       stopClk_g, outArray_g, LOAD_TIMES);
    HIP_ASSERT(hipMemcpy(startClk, startClk_g, TOTAL_THREADS * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));
    HIP_ASSERT(hipMemcpy(stopClk, stopClk_g, TOTAL_THREADS * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));

    max_clk = 0.0;
    // double avg_clk = 0.0;
    for (int i = 0; i < WAVE_NUM; i++) {
      // avg_clk += stopClk[WAVE_SIZE * i] - startClk[WAVE_SIZE * i];
      double wave_clk = stopClk[WAVE_SIZE * i] - startClk[WAVE_SIZE * i];
      if (max_clk < wave_clk)
        max_clk = wave_clk;
    }
    // avg_clk /= WAVE_NUM;
    // double bw = data / avg_clk;
    // avg_lat += avg_clk;
    double bw = data / max_clk;
    avg_lat += max_clk;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  HIP_ASSERT(hipFree(startClk_g));
  HIP_ASSERT(hipFree(stopClk_g));
  HIP_ASSERT(hipFree(outArray_g));
  output->add_record("l2_write", 8 /*wave*/, data * pow(10.0, -6.0),
                     (avg_bw / LAUNCH_TIMES) * pow(10.0, -9.0) * 1e6 * SCLK,
                     avg_lat / LAUNCH_TIMES, getLdsCacheTheory("L2 write"),
                     lds, device->device_index);
}
void MemoryBandwidth::lds_read() {
  double SCLK = device->get_sclk();
  const int THREADS_PER_BLOCK = 1024;
  const int BLOCKS_NUM = device->cuNum;

  const int TOTAL_THREADS = BLOCKS_NUM * THREADS_PER_BLOCK;
  const int MEM_ARRAY_SIZE = TOTAL_THREADS * 16;
  const int WAVE_NUM = TOTAL_THREADS / WAVE_SIZE;
  const int LOAD_TIMES = 1000;
  uint64_t startClk[TOTAL_THREADS];
  uint64_t stopClk[TOTAL_THREADS];

  uint64_t *startClk_g;
  uint64_t *stopClk_g;
  HIP_ASSERT(hipMalloc((void **)&startClk_g, TOTAL_THREADS * sizeof(uint64_t)));
  HIP_ASSERT(hipMalloc((void **)&stopClk_g, TOTAL_THREADS * sizeof(uint64_t)));
  HIP_ASSERT(hipMemcpy(startClk_g, startClk, TOTAL_THREADS * sizeof(uint64_t),
                       hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(stopClk_g, stopClk, TOTAL_THREADS * sizeof(uint64_t),
                       hipMemcpyHostToDevice));

  double data = MEM_ARRAY_SIZE * LOAD_TIMES * sizeof(int);
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    hipLaunchKernelGGL(lds_read_kernel, dim3(BLOCKS_NUM),
                       dim3(THREADS_PER_BLOCK), 65536, 0, startClk_g, stopClk_g,
                       LOAD_TIMES);
    HIP_ASSERT(hipMemcpy(startClk, startClk_g, TOTAL_THREADS * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));
    HIP_ASSERT(hipMemcpy(stopClk, stopClk_g, TOTAL_THREADS * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));

    double avg_clk = 0;
    for (int i = 0; i < WAVE_NUM; i++) {
      avg_clk += stopClk[WAVE_SIZE * i] - startClk[WAVE_SIZE * i];
    }
    avg_clk /= WAVE_NUM;
    double bw = data / avg_clk;
    avg_lat += avg_clk;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  HIP_ASSERT(hipFree(startClk_g));
  HIP_ASSERT(hipFree(stopClk_g));
  output->add_record("lds_read", wave, data * pow(10.0, -6.0),
                     (avg_bw / LAUNCH_TIMES) * pow(10.0, -9.0) * 1e6 * SCLK,
                     avg_lat / LAUNCH_TIMES, getLdsCacheTheory("lds"), 64, device->device_index);
}

void MemoryBandwidth::lds_add_u32() {
  double SCLK = device->get_sclk();
  const int THREADS_PER_BLOCK = 1024;
  const int BLOCKS_NUM = device->cuNum;

  const int TOTAL_THREADS = BLOCKS_NUM * THREADS_PER_BLOCK;
  const int MEM_ARRAY_SIZE = TOTAL_THREADS * 16;
  const int WAVE_NUM = TOTAL_THREADS / WAVE_SIZE;
  const int LOAD_TIMES = 1000;
  uint64_t startClk[TOTAL_THREADS];
  uint64_t stopClk[TOTAL_THREADS];

  uint64_t *startClk_g;
  uint64_t *stopClk_g;
  HIP_ASSERT(hipMalloc((void **)&startClk_g, TOTAL_THREADS * sizeof(uint64_t)));
  HIP_ASSERT(hipMalloc((void **)&stopClk_g, TOTAL_THREADS * sizeof(uint64_t)));
  HIP_ASSERT(hipMemcpy(startClk_g, startClk, TOTAL_THREADS * sizeof(uint64_t),
                       hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(stopClk_g, stopClk, TOTAL_THREADS * sizeof(uint64_t),
                       hipMemcpyHostToDevice));

  double data = MEM_ARRAY_SIZE * LOAD_TIMES * sizeof(int);
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    hipLaunchKernelGGL(lds_add_u32_kernel, dim3(BLOCKS_NUM),
                       dim3(THREADS_PER_BLOCK), 65536, 0, startClk_g, stopClk_g,
                       LOAD_TIMES);
    HIP_ASSERT(hipMemcpy(startClk, startClk_g, TOTAL_THREADS * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));
    HIP_ASSERT(hipMemcpy(stopClk, stopClk_g, TOTAL_THREADS * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));

    double avg_clk = 0;
    for (int i = 0; i < WAVE_NUM; i++) {
      avg_clk += stopClk[WAVE_SIZE * i] - startClk[WAVE_SIZE * i];
    }
    avg_clk /= WAVE_NUM;
    double bw = data / avg_clk;
    avg_lat += avg_clk;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  HIP_ASSERT(hipFree(startClk_g));
  HIP_ASSERT(hipFree(stopClk_g));
  output->add_record("lds_add_u32", wave, data * pow(10.0, -6.0),
                     (avg_bw / LAUNCH_TIMES) * pow(10.0, -9.0) * 1e6 * SCLK,
                     avg_lat / LAUNCH_TIMES, getLdsCacheTheory("lds") / 2, 64, device->device_index);
  // lds_add_u32 theory is only half of lds_read
}

void MemoryBandwidth::lds_write() {
  double SCLK = device->get_sclk();
  const int THREADS_PER_BLOCK = 1024;
  const int BLOCKS_NUM = device->cuNum;
  const int TOTAL_THREADS = BLOCKS_NUM * THREADS_PER_BLOCK;
  const int MEM_ARRAY_SIZE = TOTAL_THREADS * 16;
  const int WAVE_NUM = TOTAL_THREADS / WAVE_SIZE;
  const int LOAD_TIMES = 1000;
  uint64_t startClk[TOTAL_THREADS];
  uint64_t stopClk[TOTAL_THREADS];

  uint64_t *startClk_g;
  uint64_t *stopClk_g;
  HIP_ASSERT(hipMalloc((void **)&startClk_g, TOTAL_THREADS * sizeof(uint64_t)));
  HIP_ASSERT(hipMalloc((void **)&stopClk_g, TOTAL_THREADS * sizeof(uint64_t)));
  HIP_ASSERT(hipMemcpy(startClk_g, startClk, TOTAL_THREADS * sizeof(uint64_t),
                       hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(stopClk_g, stopClk, TOTAL_THREADS * sizeof(uint64_t),
                       hipMemcpyHostToDevice));

  double data = MEM_ARRAY_SIZE * LOAD_TIMES * sizeof(int);
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    hipLaunchKernelGGL(lds_write_write, dim3(BLOCKS_NUM),
                       dim3(THREADS_PER_BLOCK), 65536, 0, startClk_g, stopClk_g,
                       LOAD_TIMES);
    HIP_ASSERT(hipMemcpy(startClk, startClk_g, TOTAL_THREADS * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));
    HIP_ASSERT(hipMemcpy(stopClk, stopClk_g, TOTAL_THREADS * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));
    double avg_clk = 0;
    for (int i = 0; i < WAVE_NUM; i++) {
      avg_clk += stopClk[WAVE_SIZE * i] - startClk[WAVE_SIZE * i];
    }
    avg_clk /= WAVE_NUM;
    double bw = data / avg_clk;
    avg_lat += avg_clk;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  HIP_ASSERT(hipFree(startClk_g));
  HIP_ASSERT(hipFree(stopClk_g));
  output->add_record("lds_write", wave, data * pow(10.0, -6.0),
                     (avg_bw / LAUNCH_TIMES) * pow(10.0, -9.0) * 1e6 * SCLK,
                     avg_lat / LAUNCH_TIMES, getLdsCacheTheory("lds"), 64, device->device_index);
}
void MemoryBandwidth::lds_read4x() {

  double SCLK = device->get_sclk();
  __int128_t *dst;

  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1000>> duration;
  std::vector<double> timings;

  for (int k = 0; k < LAUNCH_TIMES; ++k) {
    t1 = std::chrono::high_resolution_clock::now();
    constexpr int total_item = LDS_TOTAL_SIZE / sizeof(__int128_t);
    constexpr int iters = total_item / WARP_SIZE;
    lds_read4x_kernel<<<FIX_BLOCK_NUM, WARP_SIZE, LDS_TOTAL_SIZE, 0>>>(
        dst, iters, FIX_ITERS, WARP_SIZE);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    timings.push_back(duration.count());
  }

  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double average_speed = pow(10.0, -9.0) * LDS_TOTAL_SIZE * FIX_ITERS *
                         FIX_BLOCK_NUM / (average * 1e-3);

  double data = pow(10.0, -6.0) * LDS_TOTAL_SIZE * FIX_ITERS * FIX_BLOCK_NUM;

  output->add_record("lds_read4x", 1 /*wave*/, data, average_speed, average,
                     getLdsCacheTheory("lds"), 2, device->device_index);
}
void MemoryBandwidth::lds_write4x() {

  double SCLK = device->get_sclk();
  __int128_t *dst;

  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1000>> duration;
  std::vector<double> timings;

  for (int k = 0; k < LAUNCH_TIMES; ++k) {
    t1 = std::chrono::high_resolution_clock::now();
    constexpr int total_item = LDS_TOTAL_SIZE / sizeof(__int128_t);
    constexpr int iters = total_item / WARP_SIZE;
    lds_write4x_kernel<<<FIX_BLOCK_NUM, WARP_SIZE, LDS_TOTAL_SIZE, 0>>>(
        dst, iters, FIX_ITERS, WARP_SIZE);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    timings.push_back(duration.count());
  }

  auto minmax = std::minmax_element(timings.begin() + 1, timings.end());
  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double average_speed = pow(10.0, -9.0) * LDS_TOTAL_SIZE * FIX_ITERS *
                         FIX_BLOCK_NUM / (average * 1e-3);

  double data = pow(10.0, -6.0) * LDS_TOTAL_SIZE * FIX_ITERS * FIX_BLOCK_NUM;
  output->add_record("lds_write4x", 1 /*wave*/, data, average_speed, average,
                     getLdsCacheTheory("lds"), 2, device->device_index);
}
#if HIP_ARCH >= 928
void MemoryBandwidth::lds_b32_matrix_read() {
  float *dst;
  HIP_ASSERT(hipMalloc((void **)&dst, 128 * sizeof(float)));
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1000>> duration;
  std::vector<double> timings;
  timings.reserve(500);

  for (int k = 0; k < LAUNCH_TIMES; ++k) {
    t1 = std::chrono::high_resolution_clock::now();
    constexpr int total_item = LDS_TOTAL_SIZE / sizeof(float);
    constexpr int iters = total_item / (32 * 8);
    lds_b32_matrix_read_kernel<<<FIX_BLOCK_NUM, WARP_SIZE, LDS_TOTAL_SIZE, 0>>>(
        dst, iters, FIX_ITERS);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    timings.push_back(duration.count());
  }

  unsigned long data_size = LDS_TOTAL_SIZE * FIX_ITERS * FIX_BLOCK_NUM;
  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double average_speed =
      pow(2.0, -20.0) * LDS_TOTAL_SIZE * FIX_ITERS * FIX_BLOCK_NUM / average;
  output->add_record("b32_lds_matrix_read" /*function name*/, wave /*wave*/,
                     data_size * pow(10.0, -6.0) /*datasize*/,
                     average_speed /*bandwidth*/, average /*latency*/,
                     getLdsCacheTheory("lds") /*theory*/, 4 /*lds_size*/, device->device_index);
  HIP_ASSERT(hipFree(dst));
}
void MemoryBandwidth::lds_b16_matrix_read() {
  float *dst;
  HIP_ASSERT(hipMalloc((void **)&dst, 128 * sizeof(float)));
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1000>> duration;
  std::vector<double> timings;

  for (int k = 0; k < LAUNCH_TIMES; ++k) {
    t1 = std::chrono::high_resolution_clock::now();
    constexpr int total_item = LDS_TOTAL_SIZE / sizeof(_Float16);
    constexpr int iters = total_item / (32 * 16);
    lds_b16_matrix_read_kernel<<<FIX_BLOCK_NUM, WARP_SIZE, LDS_TOTAL_SIZE, 0>>>(
        dst, iters, FIX_ITERS);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    timings.push_back(duration.count());
  }

  unsigned long data_size = LDS_TOTAL_SIZE * FIX_ITERS * FIX_BLOCK_NUM;
  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double average_speed =
      pow(2.0, -20.0) * LDS_TOTAL_SIZE * FIX_ITERS * FIX_BLOCK_NUM / average;

  output->add_record("b16_lds_matrix_read" /*function name*/, wave /*wave*/,
                     data_size * pow(10.0, -6.0) /*datasize*/,
                     average_speed /*bandwidth*/, average /*latency*/,
                     getLdsCacheTheory("lds") /*theory*/, 4 /*lds_size*/, device->device_index);
  HIP_ASSERT(hipFree(dst));
}
void MemoryBandwidth::lds_b16_alt_matrix_read() {
  __int128_t *dst;
  HIP_ASSERT(hipMalloc((void **)&dst, 128 * sizeof(__int128_t)));
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1000>> duration;
  std::vector<double> timings;

  for (int k = 0; k < LAUNCH_TIMES; ++k) {
    t1 = std::chrono::high_resolution_clock::now();
    constexpr int total_item = LDS_TOTAL_SIZE / sizeof(float);
    constexpr int iters = total_item / (32 * 8);
    lds_b16_alt_matrix_read_kernel<<<FIX_BLOCK_NUM, WARP_SIZE, LDS_TOTAL_SIZE,
                                     0>>>(reinterpret_cast<float *>(dst), iters,
                                          FIX_ITERS);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    timings.push_back(duration.count());
  }

  unsigned long data_size = LDS_TOTAL_SIZE * FIX_ITERS * FIX_BLOCK_NUM;
  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double average_speed =
      pow(2.0, -20.0) * LDS_TOTAL_SIZE * FIX_ITERS * FIX_BLOCK_NUM / average;

  output->add_record("b16_alt_lds_matrix_read" /*function name*/, wave /*wave*/,
                     data_size * pow(10.0, -6.0) /*datasize*/,
                     average_speed /*bandwidth*/, average /*latency*/,
                     getLdsCacheTheory("lds") /*theory*/, 4 /*lds_size*/, device->device_index);
  HIP_ASSERT(hipFree(dst));
}
void MemoryBandwidth::lds_b8_matrix_read() {
  __int128_t *dst;
  HIP_ASSERT(hipMalloc((void **)&dst, 128 * sizeof(__int128_t)));
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1000>> duration;
  std::vector<double> timings;

  for (int k = 0; k < LAUNCH_TIMES; ++k) {
    t1 = std::chrono::high_resolution_clock::now();
    constexpr int total_item = LDS_TOTAL_SIZE / sizeof(float);
    constexpr int iters = total_item / (32 * 8);
    lds_b8_matrix_read_kernel<<<FIX_BLOCK_NUM, WARP_SIZE, LDS_TOTAL_SIZE, 0>>>(
        reinterpret_cast<float *>(dst), iters, FIX_ITERS);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    timings.push_back(duration.count());
  }

  unsigned long data_size = LDS_TOTAL_SIZE * FIX_ITERS * FIX_BLOCK_NUM;
  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double average_speed =
      pow(2.0, -20.0) * LDS_TOTAL_SIZE * FIX_ITERS * FIX_BLOCK_NUM / average;

  output->add_record("b8_lds_matrix_read" /*function name*/, wave /*wave*/,
                     data_size * pow(10.0, -6.0) /*datasize*/,
                     average_speed /*bandwidth*/, average /*latency*/,
                     getLdsCacheTheory("lds") /*theory*/, 4 /*lds_size*/, device->device_index);
  HIP_ASSERT(hipFree(dst));
}
#endif

void MemoryBandwidth::moe_triton_1() {
  // conv shape
  const int num_exports = 128;
  const int N = 256;
  const int K = 7168 / 8; // use bf16 type, convert __int128_t(pack)
  uint size = num_exports * N * K;

  const int block_h = 16;      // each block read 16 __int128_t element in h
  const int block_w = 512 / 8; // each block read 64 __int128_t element in w
  const int BLOCK_SIZE = 256;
  // Create device buffers
  __int128_t *d_a;
  __int128_t *d_b;

  HIP_ASSERT(hipMalloc((void **)&d_a, size * sizeof(__int128_t)));
  HIP_ASSERT(hipMalloc((void **)&d_b, size * sizeof(__int128_t)));

  // init
  __int128_t init_value = 1;
  init_kernel<<<dim3(size / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(d_a,
                                                                   init_value);
  init_kernel<<<dim3(size / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(d_b,
                                                                   init_value);

  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1000>> duration;
  std::vector<double> timings;

  for (int k = 0; k < LAUNCH_TIMES; ++k) {
    t1 = std::chrono::high_resolution_clock::now();
    dim3 grid_size(size / 256 / 4, 1, 1);
    dim3 block_size(256, 1, 1);
    moe_triton_1_kernel<<<grid_size, block_size, lds * 1024, 0>>>(
        d_a, d_b, K, num_exports * N, block_w, block_h);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    timings.push_back(duration.count());
  }

  HIP_ASSERT(hipFree(d_a));
  HIP_ASSERT(hipFree(d_b));

  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double data_size = size * sizeof(__int128_t);
  double average_bw = pow(10.0, -6.0) * data_size / (average);
  output->add_record("moe_triton_1", wave, data_size * pow(10.0, -6.0),
                     average_bw, average, getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::moe_triton_2() {
  // conv shape
  const int num_exports = 128;
  const int N = 256;
  const int K = 7168 / 8; // use bf16 type, convert __int128_t(pack)
  uint size = num_exports * N * K;

  const int block_h = 16;      // each block read 16 __int128_t element in h
  const int block_w = 128 / 8; // each block read 64 __int128_t element in w
  const int BLOCK_SIZE = 256;
  // Create device buffers
  __int128_t *d_a;
  __int128_t *d_b;

  HIP_ASSERT(hipMalloc((void **)&d_a, size * sizeof(__int128_t)));
  HIP_ASSERT(hipMalloc((void **)&d_b, size * sizeof(__int128_t)));

  // init
  __int128_t init_value = 1;
  init_kernel<<<dim3(size / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(d_a,
                                                                   init_value);
  init_kernel<<<dim3(size / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(d_b,
                                                                   init_value);

  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1000>> duration;
  std::vector<double> timings;

  for (int k = 0; k < LAUNCH_TIMES; ++k) {
    t1 = std::chrono::high_resolution_clock::now();
    dim3 grid_size(size / 256, 1, 1);
    dim3 block_size(256, 1, 1);
    moe_triton_2_kernel<<<grid_size, block_size, lds * 1024, 0>>>(
        d_a, d_b, K, num_exports * N, block_w, block_h);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    timings.push_back(duration.count());
  }

  HIP_ASSERT(hipFree(d_a));
  HIP_ASSERT(hipFree(d_b));

  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double data_size = size * sizeof(__int128_t);
  double average_bw = pow(10.0, -6.0) * data_size / (average);
  output->add_record("moe_triton_2", wave, data_size * pow(10.0, -6.0),
                     average_bw, average, getBWtheory(), lds, device->device_index);
}
/*
void MemoryBandwidth::moe_cpp() {
  // conv shape
  const int num_exports = 128;
  const int N = 512;
  const int K = 7168 / 2; // use bf16 type, convert __int128_t(pack)
  uint size = num_exports * N * K;

  const int block_h = 32;  // each block read 16 __int128_t element in h
  const int block_w = 128 / 2;  // each block read 64 __int128_t element in w
  const int BLOCK_SIZE = 256;
  // Create device buffers
  int *d_a;
  int *d_b;

  HIP_ASSERT(hipMalloc((void **)&d_a, size * sizeof(int)));
  HIP_ASSERT(hipMalloc((void **)&d_b, size * sizeof(int)));

  // init
  int init_value = 1;
  init_kernel<<<dim3(size / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_a, init_value);
  init_kernel<<<dim3(size / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(
      d_b, init_value);

  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1000>> duration;
  std::vector<double> timings;

  for (int k = 0; k < 3; ++k) {
    t1 = std::chrono::high_resolution_clock::now();
    dim3 grid_size(size / 256 / 8, 1, 1);
    dim3 block_size(256, 1, 1);
    // moe_cpp_kernel<<<grid_size, block_size, 9 * 1024, 0>>>(
    //     d_a, d_b, K, num_exports * N, block_w, block_h);
    hipLaunchKernelGGL(moe_cpp_kernel,
                       dim3(size / 256 / 8),
                       dim3(256), 1024 * lds, 0, d_a, d_b, K, num_exports * N,
block_w, block_h); HIP_ASSERT(hipDeviceSynchronize()); t2 =
std::chrono::high_resolution_clock::now(); duration = t2 - t1;
    timings.push_back(duration.count());
  }

  HIP_ASSERT(hipFree(d_a));
  HIP_ASSERT(hipFree(d_b));

  double average = std::accumulate(timings.begin() + 1, timings.end(), 0.0) /
                   (double)(timings.size() - 1);
  double data_size = size * sizeof(int);
  double average_bw = pow(10.0, -6.0) * data_size / (average);
  output->add_record("moe_cpp", wave, data_size * pow(10.0, -6.0),
                     average_bw, average, getBWtheory(), 9);
}
*/
#if HIP_ARCH >= 936
void MemoryBandwidth::moe_cpp() {

  int THREADS_PER_BLOCK = wave * 64;
  const int num_exports = 128;
  const int N = 512;
  const int K = 7168 / 2; // use bf16 type, convert dword(pack)
  uint MEM_ARRAY_SIZE = num_exports * N * K;
  const int block_h = 32;      // each block read 16 __int128_t element in h
  const int block_w = 128 / 2; // each block read 64 __int128_t element in w
  int *inArray;
  inArray = new int[MEM_ARRAY_SIZE];

  int *inArray_g;
  HIP_ASSERT(hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(int)));
  HIP_ASSERT(hipMemcpy(inArray_g, inArray, MEM_ARRAY_SIZE * sizeof(int),
                       hipMemcpyHostToDevice));

  uint64_t data = MEM_ARRAY_SIZE * sizeof(int);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    t1 = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(moe_cpp_kernel,
                       dim3(MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 8),
                       dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g, K,
                       num_exports * N, block_w, block_h);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    t = duration.count();
    if (i == 0) {
      if (isFisrt) {
        std::cout << (double)data * pow(10.0, -9.0) / t << std::endl;
        return;
      }
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    // std::cout<< "bw:"<<bw <<std::endl;
    peak_bw = std::max(peak_bw, bw);
  }
  delete[] inArray;
  HIP_ASSERT(hipFree(inArray_g));
  output->add_record("moe_cpp", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), 4, device->device_index);
}
#endif

#if HIP_ARCH == 936
void MemoryBandwidth::HBM_GCpower() {
  int THREADS_PER_BLOCK = wave * 64;
  // size_t free_size = device->free_mem;
  // size_t total_size = device->total_mem;
  // printf("Total (GC)HBM Memory: %lu bytes, Free HBM Memory: %lu bytes\n",
  //        total_size, free_size);
  int num = 102858 / 2;
  num = (num / 8) * 8;

  uint64_t MEM_ARRAY_SIZE = 80 * 256 * num;
  std::cout << "num : " << num << std::endl;
  __int128_t *inArray_g = nullptr;

  HIP_ASSERT(
      hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)) * 2);
  std::cout << "num : " << num << std::endl;

  hiprandState_t *d_states;
  hipMalloc(&d_states, 80 * 256 * sizeof(hiprandState_t));

  int M = 32 * 80;
  int N = 32;
  int K = 32;
  size_t sizeA = M * K * sizeof(int8_t);
  size_t sizeB = K * N * sizeof(int8_t);
  size_t sizeC = M * N * sizeof(int);
  size_t sizeD = M * N * sizeof(int);

  // 分配 Host 端内存
  int8_t *h_A = (int8_t *)malloc(sizeA);
  int8_t *h_B = (int8_t *)malloc(sizeB);
  int *h_C = (int *)malloc(sizeC);
  int *h_D = (int *)malloc(sizeD);

  // 初始化数据
  for (int i = 0; i < M * K; ++i) {
    h_A[i] = rand() % 256 - 128; // 随机初始化 int8_t 数据 (-128 到 127)
  }

  for (int i = 0; i < K * N; ++i) {
    h_B[i] = rand() % 256 - 128; // 随机初始化 int8_t 数据 (-128 到 127)
  }

  for (int i = 0; i < K * N; ++i) {
    h_C[i] = rand();
  }

  // 申请 GPU 端内存
  int8_t *d_A, *d_B;
  int *d_C, *d_D;
  hipMalloc(&d_A, sizeA);
  hipMalloc(&d_B, sizeB);
  hipMalloc(&d_C, sizeC);
  hipMalloc(&d_D, sizeD);
  // 拷贝数据到 GPU
  hipMemcpy(d_A, h_A, sizeA, hipMemcpyHostToDevice);
  hipMemcpy(d_B, h_B, sizeB, hipMemcpyHostToDevice);
  hipMemcpy(d_C, h_C, sizeC, hipMemcpyHostToDevice);

  uint64_t data = 0; // MEM_ARRAY_SIZE * sizeof(__int128_t);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (int i = 0; i < 2; ++i) {
    t1 = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(powerTest_GC_HBM, dim3(160), dim3(THREADS_PER_BLOCK),
                       lds * 1024, 0, d_A, d_B, d_C, d_D, inArray_g, num,
                       d_states, -1);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    t = duration.count();
    if (i == 0) {
      if (isFisrt) {
        std::cout << (double)data * pow(10.0, -9.0) / t << std::endl;
        return;
      }
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  HIP_ASSERT(hipFree(inArray_g));
  output->add_record("powerTest_GC_HBM", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::HBMpower() {
  int THREADS_PER_BLOCK = wave * 64;
  __int128_t *inArray;
  // size = 160TG * 64k  240TG x 512(8M)
  size_t free_size = device->free_mem;
  size_t total_size = device->total_mem;
  printf("Total HBM Memory: %lu bytes, Free HBM Memory: %lu bytes\n",
         total_size, free_size);
  int num = 102858 / 2;
  num = (num / 8) * 8;

  // uint64_t MEM_ARRAY_SIZE = 160 * 256 * num;
  uint64_t MEM_ARRAY_SIZE = 80 * 256 * num;
  std::cout << "num : " << num << std::endl;
  __int128_t *inArray_g = nullptr;

  HIP_ASSERT(
      hipMalloc((void **)&inArray_g, MEM_ARRAY_SIZE * sizeof(__int128_t)) * 2);

  uint64_t data = MEM_ARRAY_SIZE * sizeof(__int128_t);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  hiprandState_t *d_states;
  hipMalloc(&d_states, 80 * 256 * sizeof(hiprandState_t));
  // hipMalloc(&d_states, 160 * 256 * sizeof(hiprandState_t));
  for (int i = 0; i < 2; ++i) {
    t1 = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(powerTest_HBM,
                       dim3(160), // MEM_ARRAY_SIZE / THREADS_PER_BLOCK / 16
                       dim3(THREADS_PER_BLOCK), lds * 1024, 0, inArray_g, num,
                       d_states);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    t = duration.count();
    if (i == 0) {
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  // delete[] inArray;
  // HIP_ASSERT(hipFree(inArray_g));
  output->add_record("powerTest_HBM", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
void MemoryBandwidth::GCpower() {
  int THREADS_PER_BLOCK = wave * 64;

  int M = 32 * 80;
  int N = 32;
  int K = 32;
  size_t sizeA = M * K * sizeof(int8_t);
  size_t sizeB = K * N * sizeof(int8_t);
  size_t sizeC = M * N * sizeof(int);
  size_t sizeD = M * N * sizeof(int);

  // 分配 Host 端内存
  int8_t *h_A = (int8_t *)malloc(sizeA);
  int8_t *h_B = (int8_t *)malloc(sizeB);
  int *h_C = (int *)malloc(sizeC);
  int *h_D = (int *)malloc(sizeD);

  // 初始化数据
  for (int i = 0; i < M * K; ++i) {
    h_A[i] = rand() % 256 - 128; // 随机初始化 int8_t 数据 (-128 到 127)
  }

  for (int i = 0; i < K * N; ++i) {
    h_B[i] = rand() % 256 - 128; // 随机初始化 int8_t 数据 (-128 到 127)
  }

  for (int i = 0; i < K * N; ++i) {
    h_C[i] = rand();
  }

  // 申请 GPU 端内存
  int8_t *d_A, *d_B;
  int *d_C, *d_D;
  hipMalloc(&d_A, sizeA);
  hipMalloc(&d_B, sizeB);
  hipMalloc(&d_C, sizeC);
  hipMalloc(&d_D, sizeD);
  // 拷贝数据到 GPU
  hipMemcpy(d_A, h_A, sizeA, hipMemcpyHostToDevice);
  hipMemcpy(d_B, h_B, sizeB, hipMemcpyHostToDevice);
  hipMemcpy(d_C, h_C, sizeC, hipMemcpyHostToDevice);

  uint64_t data = 0; // MEM_ARRAY_SIZE * sizeof(__int128_t);
  std::chrono::high_resolution_clock::time_point t1, t2;
  std::chrono::duration<double, std::ratio<1, 1>> duration;
  double t = 0.0;
  double peak_bw = 0.0, avg_bw = 0.0, avg_lat = 0.0;
  for (int i = 0; i < 2; ++i) {
    t1 = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(powerTest_GC, dim3(80), dim3(THREADS_PER_BLOCK),
                       lds * 1024, 0, d_A, d_B, d_C, d_D);
    HIP_ASSERT(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    t = duration.count();
    if (i == 0) {
      if (isFisrt) {
        std::cout << (double)data * pow(10.0, -9.0) / t << std::endl;
        return;
      }
      continue;
    } // ignore the first launch
    double bw = (double)data * pow(10.0, -9.0) / t;
    avg_lat += t;
    avg_bw += bw;
    peak_bw = std::max(peak_bw, bw);
  }
  output->add_record("powerTest_GC", wave, data * pow(10.0, -6.0),
                     avg_bw / (LAUNCH_TIMES - 1),
                     avg_lat * 1e3 / (LAUNCH_TIMES - 1), getBWtheory(), lds, device->device_index);
}
#endif

