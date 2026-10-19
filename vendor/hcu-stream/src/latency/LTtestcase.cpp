#include "memoryLatency/cache.hpp"
#include "memoryLatency/lds.hpp"
#include "memoryLatency/memory.hpp"
#include "testcase.hpp"
static std::unordered_map<std::string, void (MemoryLatency::*)()> functionMap =
    {
        {"memory_read", &MemoryLatency::latency_memory_read},
        {"memory_write", &MemoryLatency::latency_memory_write},
        {"memory_to_ldsx1", &MemoryLatency::latency_memory_to_ldsx1},
#if HIP_ARCH >= 936
        {"memory_to_ldsx4", &MemoryLatency::latency_memory_to_ldsx4},
#endif
        {"lds_to_memory", &MemoryLatency::latency_lds_to_memory},

        {"l1_read", &MemoryLatency::latency_l1_read},
        {"l2_read", &MemoryLatency::latency_l2_read},
        {"l2_write", &MemoryLatency::latency_l2_write},
        {"lds_read_b16", &MemoryLatency::latency_lds_read_b16},
        {"lds_read_b32", &MemoryLatency::latency_lds_read_b32},
        {"lds_read_b64", &MemoryLatency::latency_lds_read_b64},
        {"lds_read_b128", &MemoryLatency::latency_lds_read_b128},
        {"lds_read", &MemoryLatency::latency_lds_read},
        {"lds_write", &MemoryLatency::latency_lds_write},
        {"global_load_dwordx1", &MemoryLatency::latency_global_load_dwordx1},
        {"global_load_dwordx2", &MemoryLatency::latency_global_load_dwordx2},
        {"global_load_dwordx4", &MemoryLatency::latency_global_load_dwordx4},
};
struct LatencyTheory {
  std::string gfxArch;
  std::string func;
  std::string theory;
};

const std::vector<LatencyTheory> latTheoryMap = {
    // ZDE cycle
    {"928", "l1_read", "106~114 cycle"},
    {"928", "l2_read", "150~232 cycle"},
    {"928", "l2_write", "143~229 cycle"},
    {"928", "lds_read", "38~50 cycle"},
    {"928", "lds_write", "42~62 cycle"},
    {"928", "lds_read_b16", "- cycle"},
    {"928", "lds_read_b32", "- cycle"},
    {"928", "lds_read_b64", "- cycle"},
    {"928", "lds_read_b128", "- cycle"},
    {"928", "memory_read", "732(500+ 232)"},
    {"928", "memory_write", "359(70+ 289)"},
    {"928", "lds_to_memory", "359(70+ 289)"},
    {"928", "memory_to_ldsx1", "732(500+ 232)"},
    {"928", "global_load_dwordx4", "732(500+ 232)"},
    {"928", "global_load_dwordx2", "732(500+ 232)"},
    {"928", "global_load_dwordx1", "732(500+ 232)"},
    // BMZ cycle
    {"936", "l1_read", "114~122 cycle"},
    {"936", "l2_read", "150~222 cycle"},
    {"936", "l2_write", "178~309cycle"},
    {"936", "lds_read_b16", "38±4 cycle"},
    {"936", "lds_read_b32", "38±4 cycle"},
    {"936", "lds_read", "38±4 cycle"},
    {"936", "lds_read_b64", "42±4 cycle"},
    {"936", "lds_read_b128", "50±4 cycle"},
    {"936", "lds_write", "60±4 cycle"},
    {"936", "memory_read", "774(596 + 178)"},
    {"936", "memory_write", "456(330 + 126)"},
    {"936", "lds_to_memory", "979(853 + 126)"},
    {"936", "memory_to_ldsx1", "758(580 + 178)"},
    {"936", "memory_to_ldsx4", "758(580 + 178)"},
    {"936", "global_load_dwordx4", "774(596 + 178)"},
    {"936", "global_load_dwordx2", "774(596 + 178)"},
    {"936", "global_load_dwordx1", "774(596 + 178)"},
};

MemoryLatency::MemoryLatency(std::vector<std::string> case_list, int wave,
                             double crop, Output *output, Device *device)
    : Testcase(case_list, wave, crop, output, device) {}

void MemoryLatency::run() {
  // 查找并调用对应的函数
  for (auto name : case_list) {
    auto it = functionMap.find(name);
    if (it != functionMap.end()) {
      (this->*(it->second))();
    } else {
      std::cout << "Unknown latency case name!" << std::endl;
    }
  }
}

void MemoryLatency::getList() {
  std::cout << std::left << std::setw(30) << "Latency case" << std::endl;
  std::cout << std::left << std::setw(30) << "Usage: ./run latency l1_read"
            << std::endl;
  std::cout << "----------------------" << std::endl;
  for (const auto &key : functionMap) {
    std::cout << std::left << std::setw(20) << key.first << std::endl;
  }
}

void MemoryLatency::latency_l1_read() {
  int N = WAVE_SIZE;
  int load_times = 10000;
  uint32_t sharedMemBytes = 0;
  uint64_t clock_c[N];
  uint32_t array_c[N * 4];
  uint64_t *clock_g;
  uint32_t *array_g;

  memset(clock_c, 0xFF, N * sizeof(uint64_t));
  memset(array_c, 0xFF, N * sizeof(uint32_t) * 4);
  for (int i = 0; i < N; ++i) {
    array_c[i] = i;
  }

  HIP_ASSERT(hipMalloc((void **)&clock_g, N * sizeof(uint64_t)));
  HIP_ASSERT(hipMalloc((void **)&array_g, N * sizeof(uint32_t) * 4));
  HIP_ASSERT(
      hipMemcpy(clock_g, clock_c, N * sizeof(uint64_t), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(array_g, array_c, N * sizeof(uint32_t) * 4,
                       hipMemcpyHostToDevice));

  float avg_latency = 0.f;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    hipLaunchKernelGGL(latency_l1_read_kernel, dim3(1), dim3(N), sharedMemBytes,
                       0, clock_g, array_g);

    HIP_ASSERT(hipMemcpy(clock_c, clock_g, N * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));
    avg_latency += (float)*clock_c;
  }
  output->add_record_lt("latecny_l1_read",
                        avg_latency / LAUNCH_TIMES / load_times,
                        getLatencyTheory("l1_read"));
}
void MemoryLatency::latency_l2_read() {
  int N = WAVE_SIZE;
  int load_times = 10000;
  uint32_t sharedMemBytes = 0;
  uint64_t clock_c[N];
  uint32_t array_c[N * 4];
  uint64_t *clock_g;
  uint32_t *array_g;

  memset(clock_c, 0xFF, N * sizeof(uint64_t));
  memset(array_c, 0xFF, N * sizeof(uint32_t) * 4);
  for (int i = 0; i < N; ++i) {
    array_c[i] = i;
  }

  HIP_ASSERT(hipMalloc((void **)&clock_g, N * sizeof(uint64_t)));
  HIP_ASSERT(hipMalloc((void **)&array_g, N * sizeof(uint32_t) * 4));
  HIP_ASSERT(
      hipMemcpy(clock_g, clock_c, N * sizeof(uint64_t), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(array_g, array_c, N * sizeof(uint32_t) * 4,
                       hipMemcpyHostToDevice));

  float avg_latency = 0.f;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    hipLaunchKernelGGL(latency_l2_read_kernel, dim3(1), dim3(N), sharedMemBytes,
                       0, clock_g, array_g);

    HIP_ASSERT(hipMemcpy(clock_c, clock_g, N * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));
    avg_latency += (float)*clock_c;
  }

  output->add_record_lt("latecny_l2_read",
                        avg_latency / LAUNCH_TIMES / load_times,
                        getLatencyTheory("l2_read"));
}
void MemoryLatency::latency_l2_write() {
  int N = WAVE_SIZE;
  int load_times = 10000;
  uint32_t sharedMemBytes = 0;
  uint64_t clock_c[N];
  uint32_t array_c[N * 4];
  uint64_t *clock_g;
  uint32_t *array_g;

  memset(clock_c, 0xFF, N * sizeof(uint64_t));
  memset(array_c, 0xFF, N * sizeof(uint32_t) * 4);
  for (int i = 0; i < N; ++i) {
    array_c[i] = i;
  }

  HIP_ASSERT(hipMalloc((void **)&clock_g, N * sizeof(uint64_t)));
  HIP_ASSERT(hipMalloc((void **)&array_g, N * sizeof(uint32_t) * 4));
  HIP_ASSERT(
      hipMemcpy(clock_g, clock_c, N * sizeof(uint64_t), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(array_g, array_c, N * sizeof(uint32_t) * 4,
                       hipMemcpyHostToDevice));

  float avg_latency = 0.f;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    hipLaunchKernelGGL(latency_l2_write_kernel, dim3(1), dim3(N),
                       sharedMemBytes, 0, clock_g, array_g);

    HIP_ASSERT(hipMemcpy(clock_c, clock_g, N * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));
    avg_latency += (float)*clock_c;
  }

  output->add_record_lt("latecny_l2_write",
                        avg_latency / LAUNCH_TIMES / load_times,
                        getLatencyTheory("l2_write"));
}
void MemoryLatency::latency_memory_read() {
  int N = WAVE_SIZE;
  // int load_times = 10000;
  uint32_t sharedMemBytes = 0;
  uint64_t clock_c[N];
  uint32_t array_c[N * 4];
  uint64_t *clock_g;
  uint32_t *array_g;

  memset(clock_c, 0xFF, N * sizeof(uint64_t));
  memset(array_c, 0xFF, N * sizeof(uint32_t) * 4);
  for (int i = 0; i < N; ++i) {
    array_c[i] = i;
  }

  HIP_ASSERT(hipMalloc((void **)&clock_g, N * sizeof(uint64_t)));
  HIP_ASSERT(hipMalloc((void **)&array_g, N * sizeof(uint32_t) * 4));
  HIP_ASSERT(
      hipMemcpy(clock_g, clock_c, N * sizeof(uint64_t), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(array_g, array_c, N * sizeof(uint32_t) * 4,
                       hipMemcpyHostToDevice));

  float avg_latency = 0.f;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    hipLaunchKernelGGL(latency_memory_read_kernel, dim3(1), dim3(N),
                       sharedMemBytes, 0, clock_g, array_g);
    HIP_ASSERT(hipMemcpy(clock_c, clock_g, N * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));
    avg_latency += (float)*clock_c;
  }

  output->add_record_lt("memory_read", avg_latency / LAUNCH_TIMES,
                        getLatencyTheory("memory_read"));
}
void MemoryLatency::latency_memory_write() {
  int N = WAVE_SIZE;
  // int load_times = 10000;
  uint32_t sharedMemBytes = 0;
  uint64_t clock_c[N];
  uint32_t array_c[N * 4];
  uint64_t *clock_g;
  uint32_t *array_g;

  memset(clock_c, 0xFF, N * sizeof(uint64_t));
  memset(array_c, 0xFF, N * sizeof(uint32_t) * 4);
  for (int i = 0; i < N; ++i) {
    array_c[i] = i;
  }

  HIP_ASSERT(hipMalloc((void **)&clock_g, N * sizeof(uint64_t)));
  HIP_ASSERT(hipMalloc((void **)&array_g, N * sizeof(uint32_t) * 4));
  HIP_ASSERT(
      hipMemcpy(clock_g, clock_c, N * sizeof(uint64_t), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(array_g, array_c, N * sizeof(uint32_t) * 4,
                       hipMemcpyHostToDevice));

  float avg_latency = 0.f;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    hipLaunchKernelGGL(latency_memory_write_kernel, dim3(1), dim3(N),
                       sharedMemBytes, 0, clock_g, array_g);

    HIP_ASSERT(hipMemcpy(clock_c, clock_g, N * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));
    avg_latency += (float)*clock_c;
  }

  output->add_record_lt("latecny_memory_write", avg_latency / LAUNCH_TIMES,
                        getLatencyTheory("memory_write"));
}
void MemoryLatency::latency_memory_to_ldsx1() {
  int N = WAVE_SIZE;
  // int load_times = 10000;
  uint32_t sharedMemBytes = N * sizeof(uint32_t) * 4;
  uint64_t clock_c[N];
  uint32_t array_c[N * 4];
  uint64_t *clock_g;
  uint32_t *array_g;

  memset(clock_c, 0xFF, N * sizeof(uint64_t));
  memset(array_c, 0xFF, N * sizeof(uint32_t) * 4);
  for (int i = 0; i < N; ++i) {
    array_c[i] = i;
  }

  HIP_ASSERT(hipMalloc((void **)&clock_g, N * sizeof(uint64_t)));
  HIP_ASSERT(hipMalloc((void **)&array_g, N * sizeof(uint32_t) * 4));
  HIP_ASSERT(
      hipMemcpy(clock_g, clock_c, N * sizeof(uint64_t), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(array_g, array_c, N * sizeof(uint32_t) * 4,
                       hipMemcpyHostToDevice));

  float avg_latency = 0.f;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    hipLaunchKernelGGL(latency_memory_to_ldsx1_kernel, dim3(1), dim3(N),
                       sharedMemBytes, 0, clock_g, array_g);
    HIP_ASSERT(hipMemcpy(clock_c, clock_g, N * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));
    avg_latency += (float)*clock_c;
  }
  output->add_record_lt("latecny_memory_to_ldsx1", avg_latency / LAUNCH_TIMES,
                        getLatencyTheory("memory_to_ldsx1"));
}
#if HIP_ARCH >= 936
void MemoryLatency::latency_memory_to_ldsx4() {
  int N = WAVE_SIZE;
  // int load_times = 10000;
  uint32_t sharedMemBytes = N * sizeof(uint32_t) * 4;
  uint64_t clock_c[N];
  uint32_t array_c[N * 4];
  uint64_t *clock_g;
  uint32_t *array_g;

  memset(clock_c, 0xFF, N * sizeof(uint64_t));
  memset(array_c, 0xFF, N * sizeof(uint32_t) * 4);
  for (int i = 0; i < N; ++i) {
    array_c[i] = i;
  }

  HIP_ASSERT(hipMalloc((void **)&clock_g, N * sizeof(uint64_t)));
  HIP_ASSERT(hipMalloc((void **)&array_g, N * sizeof(uint32_t) * 4));
  HIP_ASSERT(
      hipMemcpy(clock_g, clock_c, N * sizeof(uint64_t), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(array_g, array_c, N * sizeof(uint32_t) * 4,
                       hipMemcpyHostToDevice));

  float avg_latency = 0.f;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    hipLaunchKernelGGL(latency_memory_to_ldsx4_kernel, dim3(1), dim3(N),
                       sharedMemBytes, 0, clock_g, array_g);
    HIP_ASSERT(hipMemcpy(clock_c, clock_g, N * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));
    avg_latency += (float)*clock_c;
  }
  output->add_record_lt("latecny_memory_to_ldsx4", avg_latency / LAUNCH_TIMES,
                        getLatencyTheory("memory_to_ldsx4"));
}
#endif
void MemoryLatency::latency_lds_to_memory() {
  int N = WAVE_SIZE;
  // int load_times = 10000;
  uint32_t sharedMemBytes = N * sizeof(uint32_t) * 4;
  uint64_t clock_c[N];
  uint32_t array_c[N * 4];
  uint64_t *clock_g;
  uint32_t *array_g;

  memset(clock_c, 0xFF, N * sizeof(uint64_t));
  memset(array_c, 0xFF, N * sizeof(uint32_t) * 4);
  for (int i = 0; i < N; ++i) {
    array_c[i] = i;
  }

  HIP_ASSERT(hipMalloc((void **)&clock_g, N * sizeof(uint64_t)));
  HIP_ASSERT(hipMalloc((void **)&array_g, N * sizeof(uint32_t) * 4));
  HIP_ASSERT(
      hipMemcpy(clock_g, clock_c, N * sizeof(uint64_t), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(array_g, array_c, N * sizeof(uint32_t) * 4,
                       hipMemcpyHostToDevice));

  float avg_latency = 0.f;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    hipLaunchKernelGGL(latency_lds_to_memory_kernel, dim3(1), dim3(N),
                       sharedMemBytes, 0, clock_g, array_g);

    HIP_ASSERT(hipMemcpy(clock_c, clock_g, N * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));
    avg_latency += (float)*clock_c;
  }

  output->add_record_lt("latecny_lds_to_memory", avg_latency / LAUNCH_TIMES,
                        getLatencyTheory("lds_to_memory"));
}
void MemoryLatency::latency_lds_read() {
  int N = WAVE_SIZE;
  uint32_t sharedMemBytes = N * sizeof(uint32_t) * 4;
  uint64_t clock_c[N];
  uint64_t *clock_g;

  memset(clock_c, 0xFF, N * sizeof(uint64_t));

  HIP_ASSERT(hipMalloc((void **)&clock_g, N * sizeof(uint64_t)));
  HIP_ASSERT(
      hipMemcpy(clock_g, clock_c, N * sizeof(uint64_t), hipMemcpyHostToDevice));

  float avg_latency = 0.f;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    hipLaunchKernelGGL(latency_lds_read_kernel, dim3(1), dim3(N),
                       sharedMemBytes, 0, clock_g);

    HIP_ASSERT(hipMemcpy(clock_c, clock_g, N * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));
    avg_latency += (float)*clock_c;
  }

  output->add_record_lt("latecny_lds_read",
                        avg_latency / LAUNCH_TIMES / lds_read_iter,
                        getLatencyTheory("lds_read"));
}
void MemoryLatency::latency_lds_read_b16() {
  int N = WAVE_SIZE;
  // uint32_t sharedMemBytes = N * sizeof(uint16_t) * 4;
  uint64_t clock_c[N];
  uint64_t *clock_g;

  memset(clock_c, 0xFF, N * sizeof(uint64_t));

  HIP_ASSERT(hipMalloc((void **)&clock_g, N * sizeof(uint64_t)));
  HIP_ASSERT(
      hipMemcpy(clock_g, clock_c, N * sizeof(uint64_t), hipMemcpyHostToDevice));

  float avg_latency = 0.f;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    hipLaunchKernelGGL(latency_lds_read_b16_kernel, dim3(1), dim3(N), 0, 0,
                       clock_g);

    HIP_ASSERT(hipMemcpy(clock_c, clock_g, N * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));
    avg_latency += (float)*clock_c;
  }

  output->add_record_lt("latecny_lds_read_b16",
                        avg_latency / LAUNCH_TIMES / LDS_TOTAL_SIZE,
                        getLatencyTheory("lds_read_b16"));
}
void MemoryLatency::latency_lds_read_b32() {
  int N = WAVE_SIZE;
  // uint32_t sharedMemBytes = N * sizeof(uint16_t) * 4;
  uint64_t clock_c[N];
  uint64_t *clock_g;

  memset(clock_c, 0xFF, N * sizeof(uint64_t));

  HIP_ASSERT(hipMalloc((void **)&clock_g, N * sizeof(uint64_t)));
  HIP_ASSERT(
      hipMemcpy(clock_g, clock_c, N * sizeof(uint64_t), hipMemcpyHostToDevice));

  float avg_latency = 0.f;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    hipLaunchKernelGGL(latency_lds_read_b32_kernel, dim3(1), dim3(N), 0, 0,
                       clock_g);

    HIP_ASSERT(hipMemcpy(clock_c, clock_g, N * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));
    avg_latency += (float)*clock_c;
  }

  output->add_record_lt("latecny_lds_read_b32",
                        avg_latency / LAUNCH_TIMES / LDS_TOTAL_SIZE,
                        getLatencyTheory("lds_read"));
}
void MemoryLatency::latency_lds_read_b64() {
  int N = WAVE_SIZE;
  // uint32_t sharedMemBytes = N * sizeof(uint64_t) * 4;
  uint64_t clock_c[N];
  uint64_t *clock_g;

  memset(clock_c, 0xFF, N * sizeof(uint64_t));

  HIP_ASSERT(hipMalloc((void **)&clock_g, N * sizeof(uint64_t)));
  HIP_ASSERT(
      hipMemcpy(clock_g, clock_c, N * sizeof(uint64_t), hipMemcpyHostToDevice));

  float avg_latency = 0.f;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    hipLaunchKernelGGL(latency_lds_read_b64_kernel, dim3(1), dim3(N), 0, 0,
                       clock_g);

    HIP_ASSERT(hipMemcpy(clock_c, clock_g, N * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));
    avg_latency += (float)*clock_c;
  }

  output->add_record_lt("latecny_lds_read_b64",
                        avg_latency / LAUNCH_TIMES / LDS_TOTAL_SIZE,
                        getLatencyTheory("lds_read_b64"));
}
void MemoryLatency::latency_lds_read_b128() {
  int N = WAVE_SIZE;
  // uint32_t sharedMemBytes = N * sizeof(__int128_t) * 4;
  uint64_t clock_c[N];
  uint64_t *clock_g;

  memset(clock_c, 0xFF, N * sizeof(uint64_t));

  HIP_ASSERT(hipMalloc((void **)&clock_g, N * sizeof(uint64_t)));
  HIP_ASSERT(
      hipMemcpy(clock_g, clock_c, N * sizeof(uint64_t), hipMemcpyHostToDevice));

  float avg_latency = 0.f;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    hipLaunchKernelGGL(latency_lds_read_b128_kernel, dim3(1), dim3(N), 0, 0,
                       clock_g);

    HIP_ASSERT(hipMemcpy(clock_c, clock_g, N * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));
    avg_latency += (float)*clock_c;
  }

  output->add_record_lt("latecny_lds_read_b128",
                        avg_latency / LAUNCH_TIMES / LDS_TOTAL_SIZE,
                        getLatencyTheory("lds_read_b128"));
}
void MemoryLatency::latency_lds_write() {
  int N = WAVE_SIZE;
  int load_times = 10000;
  uint32_t sharedMemBytes = N * sizeof(uint32_t) * 4;
  uint64_t clock_c[N];
  uint64_t *clock_g;

  memset(clock_c, 0xFF, N * sizeof(uint64_t));
  HIP_ASSERT(hipMalloc((void **)&clock_g, N * sizeof(uint64_t)));
  HIP_ASSERT(
      hipMemcpy(clock_g, clock_c, N * sizeof(uint64_t), hipMemcpyHostToDevice));

  float avg_latency = 0.f;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    hipLaunchKernelGGL(latency_lds_write_kernel, dim3(1), dim3(N),
                       sharedMemBytes, 0, clock_g);

    HIP_ASSERT(hipMemcpy(clock_c, clock_g, N * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));
    avg_latency += (float)*clock_c;
  }

  output->add_record_lt("latecny_lds_write",
                        avg_latency / LAUNCH_TIMES / load_times,
                        getLatencyTheory("lds_write"));
}

void MemoryLatency::latency_global_load_dwordx1() {
  int N = WAVE_SIZE;
  uint64_t clock_c[N];
  uint32_t *array_c;
  array_c = new uint32_t[ARRAY_SIZE];
  uint64_t *clock_g;
  uint32_t *array_g;
  memset(clock_c, 0xFF, N * sizeof(uint64_t));

  for (int i = 0; i < ARRAY_SIZE; ++i) {
    array_c[i] = i;
  }

  HIP_ASSERT(hipMalloc((void **)&clock_g, N * sizeof(uint64_t)));
  HIP_ASSERT(hipMalloc((void **)&array_g, ARRAY_SIZE * sizeof(uint32_t)));
  HIP_ASSERT(
      hipMemcpy(clock_g, clock_c, N * sizeof(uint64_t), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(array_g, array_c, ARRAY_SIZE * sizeof(uint32_t),
                       hipMemcpyHostToDevice));

  float avg_latency = 0.f;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    hipLaunchKernelGGL(latency_global_load_dwordx1_kernel, dim3(1), dim3(N), 0,
                       0, reinterpret_cast<uint32_t *>(array_g), clock_g);
    HIP_ASSERT(hipMemcpy(clock_c, clock_g, N * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));
    avg_latency += (float)*clock_c;
  }

  output->add_record_lt("global_load_dwordx1",
                        avg_latency / LAUNCH_TIMES / HEIGHT,
                        getLatencyTheory("global_load_dwordx1"));
}

void MemoryLatency::latency_global_load_dwordx2() {
  int N = WAVE_SIZE;
  uint64_t clock_c[N];
  uint64_t *array_c;
  array_c = new uint64_t[ARRAY_SIZE];
  uint64_t *clock_g;
  uint64_t *array_g;

  memset(clock_c, 0xFF, N * sizeof(uint64_t));
  memset(array_c, 0xFF, ARRAY_SIZE * sizeof(uint64_t));
  for (int i = 0; i < N; ++i) {
    array_c[i] = i;
  }

  HIP_ASSERT(hipMalloc((void **)&clock_g, N * sizeof(uint64_t)));
  HIP_ASSERT(hipMalloc((void **)&array_g, ARRAY_SIZE * sizeof(uint64_t)));
  HIP_ASSERT(
      hipMemcpy(clock_g, clock_c, N * sizeof(uint64_t), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(array_g, array_c, ARRAY_SIZE * sizeof(uint64_t),
                       hipMemcpyHostToDevice));

  float avg_latency = 0.f;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    hipLaunchKernelGGL(latency_global_load_dwordx2_kernel, dim3(1), dim3(1), 0,
                       0, array_g, clock_g);
    HIP_ASSERT(hipMemcpy(clock_c, clock_g, N * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));
    avg_latency += (float)*clock_c;
  }

  output->add_record_lt("global_load_dwordx2",
                        avg_latency / LAUNCH_TIMES / HEIGHT,
                        getLatencyTheory("global_load_dwordx2"));
}

void MemoryLatency::latency_global_load_dwordx4() {
  int N = WAVE_SIZE;
  uint64_t clock_c[N];
  __int128_t *array_c;
  array_c = new __int128_t[ARRAY_SIZE];
  uint64_t *clock_g;
  __int128_t *array_g;

  memset(clock_c, 0xFF, N * sizeof(uint64_t));
  // memset(array_c, 0xFF, ARRAY_SIZE * sizeof(__int128_t));
  // for (int i = 0; i < N; ++i) {
  //   array_c[i] = i;
  // }

  HIP_ASSERT(hipMalloc((void **)&clock_g, N * sizeof(uint64_t)));
  HIP_ASSERT(hipMalloc((void **)&array_g, ARRAY_SIZE * sizeof(__int128_t)));
  HIP_ASSERT(
      hipMemcpy(clock_g, clock_c, N * sizeof(uint64_t), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(array_g, array_c, ARRAY_SIZE * sizeof(__int128_t),
                       hipMemcpyHostToDevice));

  float avg_latency = 0.f;
  for (int i = 0; i < LAUNCH_TIMES; ++i) {
    hipLaunchKernelGGL(latency_global_load_dwordx4_kernel, dim3(1), dim3(1), 0,
                       0, array_g, clock_g);
    HIP_ASSERT(hipMemcpy(clock_c, clock_g, N * sizeof(uint64_t),
                         hipMemcpyDeviceToHost));
    avg_latency += (float)*clock_c;
  }

  output->add_record_lt("global_load_dwordx4",
                        avg_latency / LAUNCH_TIMES / HEIGHT,
                        getLatencyTheory("global_load_dwordx4"));
}

void MemoryLatency::print() { output->print_record(); }

std::string MemoryLatency::getLatencyTheory(std::string func) {
  std::string arch;
#if HIP_ARCH == 928
  arch = "928";
#elif HIP_ARCH == 936
  arch = "936";
#elif HIP_ARCH == 938
  arch = "938";
#else
  assert(0 && "unsupport arch !");
#endif

  for (const auto &record : latTheoryMap) {
    if (record.gfxArch == arch && record.func == func) {
      return record.theory;
    }
  }
  assert(0 && "not support Inst for arch or not add theory value!");
  return "Not exit";
}
