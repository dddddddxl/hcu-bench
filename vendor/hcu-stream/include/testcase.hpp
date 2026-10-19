
#ifndef TESTCASE_H
#define TESTCASE_H
#include "hip/hip_runtime.h"
#include <hip/hip_ext.h>
#include <numeric>
#include <hsa/hsa.h>

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>

#include "device.hpp"
#include "gemmHelper/global.hpp"
#include "gemmHelper/matrix.hpp"
#include "gemmHelper/rand.hpp"
#include "output.hpp"
#include "params.h"
#include "hipapi.hpp"
using namespace std;

class Testcase {
protected:
  std::vector<std::string> case_list;
  // std::string description;
  double crop;
  int wave;
  int lds;
  Output *output;
  Device *device;

public:
  // Testcase(std::string name, int crop);
  Testcase(std::vector<std::string> case_list, int wave, double crop,
           Output *output, Device *device);
  virtual ~Testcase() {
    std::cout << "Testcase constructor called" << std::endl;
  }
  virtual void run() = 0;
  virtual void print() = 0;
  virtual void getList() = 0;
  virtual void runAll() = 0;
  void printDeviceinfo();
};

class MemoryBandwidth : public Testcase {
public:
  int lds_size;
  bool use_clean;
  int arraysize;
  bool useEvent;
  bool printEach;

public:
  MemoryBandwidth(std::vector<std::string> case_list, int wave, double crop,
                  Output *output, Device *device, int lds, bool clean,
                  int array_size, bool useEvent, bool printEach);
  virtual ~MemoryBandwidth() {
    std::cout << "MemoryBandwidth destructor called" << std::endl;
  }
  void run();
  void print();
  void getList();
  void runAll() {}
  double getBWtheory() { return 1.024 * device->get_uclk(); }
  double getLdsCacheTheory(const std::string &name);

public:
  // global_dowrdx1
  void memory_readx1_x1();
  void memory_readx1_x4();
  void memory_readx1_x8192();
  void memory_readx1_x8192_80cu();
  // global_dwordx2
  void memory_readx2_x1();
  void memory_readx2_x2();
  void memory_readx2_x4();
  void memory_readx2_x4096();
  void memory_readx2_x4096_80cu();
  // global_dwordx4
  void memory_readx4_x1();
  void memory_readx4_x2();
  void memory_readx4_x4();
  void memory_readx4_x8();
  void memory_readx4_x16();
  void memory_readx4_x64();
  void memory_readx4_x2048();
  void memory_readx4_x2048_80cu();

  void memory_readx4_x8_wave_continue();
  void memory_readx4_x8_wave_continue_8wave();
  void memory_readx4_x8_wave_continue_16wave();
#if HIP_ARCH >= 936
  void memory_readx4_x8_wave_continue_bps();
  void memory_readx4_x8_wave_continue_8wave_bps();
  void memory_readx4_x8_wave_continue_16wave_bps();
#endif
  void memory_readx4_x8_inst_continue();

  // other
  void memory_write();
  void memory_writex1_x1();
  void memory_writex2_x1();
  void memory_writex2_x2();

  void memory_writex4_x2();
  void memory_writex4_x4();
  void memory_writex2_x4();
  void memory_writex1_x4();
  void memory_writex4_x8_wave_continue();
  void memory_writex4_x8_wave_continue_8wave();
  void memory_writex4_x8_wave_continue_16wave();
  void memory_writex4_x8_inst_continue();

  void memory_copy();
  void memory_copyx2_x1();
  void memory_copy_x2();
  void memory_copyx4_x4();
  void memory_copyx4_x8_wave_continue();
  void memory_copyx4_x8_wave_continue_asm();
  void memory_copyx4_x8_wave_continue_8wave();
  void memory_copyx4_x8_wave_continue_16wave();
  void memory_copyx4_x8_inst_continue();
  void memory_copyx4_x4_wave_continue();

#if HIP_ARCH >= 936
  void memory_copyx4_x8_wave_continue_bps_4wave();
  void memory_copyx4_x8_wave_continue_bps_8wave();
  void memory_copyx4_x8_wave_continue_bps_16wave();
#endif

  void matrix_read();
  void matrix_read_x4();
  void matrix_write();
  void matrix_write_x4();
  void matrix_copy();
  void matrix_copy_x4();
  void matrix_readx4_x2();
  void matrix_writex4_x2();
  void matrix_readx4_x4();
  void matrix_writex4_x4();
  void matrix_copyx4_x4();

  void matrix_readx4_multi();
  void matrix_writex4_multi();
  void matrix_copyx4_multi();

  void mapping_matrix_write();
  void mapping_matrix_writex4_x4();
  void tiling_read();
  void tiling_read_nhwc();
  void tiling_read_nchw_16t();
  void tiling_read_w();
  void tiling_read_2h();
  void tiling_read_2w();
  void winograd_read();
  void memory_to_ldsx1();

  void moe_triton_1();
  void moe_triton_2();
#if HIP_ARCH >= 936
  void moe_cpp();
#endif

#if HIP_ARCH == 936
  void HBMpower();
  void GCpower();
  void HBM_GCpower();
#endif
#if HIP_ARCH >= 936
  void memory_to_ldsx4();
  void memory_to_ldsx4_x4();
#endif
  void lds_to_memory();
  void lds_to_memory_x4();
  // L1/L2 cache
  void l1_read();
  void l2_read();
  void l2_write();

  // // LDS read/write
  void lds_read();
  void lds_add_u32();
  void lds_write();
  void lds_read4x();
  void lds_write4x();
#if HIP_ARCH >= 928
  void lds_b32_matrix_read();
  void lds_b16_matrix_read();
  void lds_b16_alt_matrix_read();
  void lds_b8_matrix_read();
#endif
};

class L2CacheRW : public MemoryBandwidth {
public:
  L2CacheRW(Output *output, Device *device);
  virtual ~L2CacheRW() {
    std::cout << "L2CacheRW destructor called" << std::endl;
  }

public:
  // L2 cache rw
  __int128_t *src = nullptr;
  __int128_t *dst = nullptr;
  void cache_rw();
  std::vector<double> L2_read_rw(int iter, int parts);
  void run();
  void print();
  void getList();
};

class TCCRead : public MemoryBandwidth {
public:
  TCCRead(Output *output, Device *device);
  virtual ~TCCRead() { std::cout << "TCCRead destructor called" << std::endl; }

public:
  __int128_t *src = nullptr;
  __int128_t *dst = nullptr;
  void run_tcc_read();
  std::vector<double> run_kernel(int height, uint32_t width);
  void run();
  void print();
  void getList();
};

class ElementWise : public MemoryBandwidth {
public:
  ElementWise(Output *output, Device *device, bool use_clean);
  virtual ~ElementWise() {std::cout << "elementWise test finish!" << std::endl; }
public:
  int data_shape;
  void run();
  void print();
  void getList();
  void runAll();
  void test_elementwise_modern();
  void test_elementwise_legacy();
  template <typename T>
  void run_benchmark(void (*kernel)(T *, T *, T *, int), const char *tag,
                     int elenum, bool runAll = false);
  template <typename DT, int BLOCK_SZ, int LOOP>
  void run_benchmark_legacy(int vec_size, bool runAll = false);
};

class gemmMemory : public MemoryBandwidth {
private:
  int h;
  int w;
public:
  gemmMemory(Output *output, Device *device, bool use_clean, int lds, int height, int width);
  virtual ~gemmMemory() { std::cout << "gemmMemory test finish!" << std::endl; }

public:
  int data_shape;
  void run_Gemmbenchmark(int height, int width);
  void run();
  void runAll();
  void print();
  void getList();
};

class faMemory : public MemoryBandwidth {
private:
  int h;
  int w;
public:
  faMemory(Output *output, Device *device, bool use_clean, int lds, int weight, int width);
  virtual ~faMemory() { std::cout << "faMemory test finish!" << std::endl; }

public:
  void run_fabenchmark(int heigth, int width);
  void run();
  void runAll();
  void print();
  void getList();
};

class paMemory : public MemoryBandwidth {
private:
  int pad;
  int h;
  int w;

public:
  paMemory(Output *output, Device *device, bool use_clean, int pad, int weight, int width);
  virtual ~paMemory() { std::cout << "paMemory test finish!" << std::endl; }

public:
  void run_pabenchmark(int height, int width);
  void run();
  void print();
  void runAll();
  void getList();
};


class MemoryLatency : public Testcase {
public:
  MemoryLatency(std::vector<std::string> case_list, int wave, double crop,
                Output *output, Device *device);
  virtual ~MemoryLatency() {
    std::cout << "MemoryLatency destructor called" << std::endl;
  }
  void run();
  void print();
  void getList();
  void runAll(){}
  std::string getLatencyTheory(std::string func);

public:
  void latency_l1_read();
  void latency_l2_read();
  void latency_l2_write();
  void latency_memory_read();
  void latency_memory_write();
  void latency_memory_to_ldsx1();
#if HIP_ARCH >= 936
  void latency_memory_to_ldsx4();
#endif
  void latency_lds_to_memory();
  void latency_lds_read();
  void latency_lds_read_b16();
  void latency_lds_read_b32();
  void latency_lds_read_b64();
  void latency_lds_read_b128();
  void latency_lds_write();
  void latency_global_load_dwordx1();
  void latency_global_load_dwordx2();
  void latency_global_load_dwordx4();
};

class TFLOPsCase : public Testcase {
public:
  TFLOPsCase(std::vector<std::string> case_list, int wave, double crop,
             Output *output, Device *device);
  virtual ~TFLOPsCase() {
    std::cout << "MemoryLatency destructor called" << std::endl;
  }
  void run();
  void print() {}
  void getList();
  void runAll() {}

public:

  double getTheory(const std::string &inst, int cuNum);
#if HIP_ARCH >= 926 && HIP_ARCH != 928 && HIP_ARCH != 9210
  void run_f64();  // gfx926 support
#endif

#if HIP_ARCH >= 926
  void run_f32x4();
#endif

#if HIP_ARCH >= 926 && HIP_ARCH <= 938
  void run_f32x8();
#endif

  // case_list
#if HIP_ARCH >= 928
  void run_u8();
  void run_i8();
  void run_bf16();
  void run_f16();
  void run_tf32();
#endif

#if HIP_ARCH >= 936 && HIP_ARCH != 9210
  void run_i4();
  void run_u4();
#endif

#if HIP_ARCH >= 938
  void run_fp8_fp8();
  void run_bf8_bf8();
  void run_fp8_bf8();
  void run_bf8_fp8();
#endif

#if HIP_ARCH >= 946  // include 92a(9210)
  void run_mix_f8f6f4();
#endif

#if HIP_ARCH == 946
  void  run_mix_scale_f8f6f4();
  void run_fp4();
  void run_scale_fp4();
#endif

};

class KernelDispatch : public Testcase {
private:
  bool use_clean;
public:
  KernelDispatch(std::vector<std::string> case_list, Output *output, Device *device, bool clean = true);
  virtual ~KernelDispatch() {
    std::cout << "KernelDispatch destructor called" << std::endl;
  }

public:
  void dispatch_fix_grid();
  void dispatch_fix_block_1024();
  void dispatch_fix_block_512();
  void run();
  void runAll() {}
  void print();
  void getList();
};

class VGPRNoFlush : public Testcase {
private:
  int blocks;
  bool use_clean;
public:
  VGPRNoFlush(std::vector<std::string> case_list, Output *output, Device *device, bool clean, int _blocks);
  virtual ~VGPRNoFlush() {
    std::cout << "VGPRNoFlush destructor called" << std::endl;
  }
  void run();
  void print();
  void getList();
  void runAll() {}
  void safe_kernel();
  void unsafe_kernel();
};

#endif
