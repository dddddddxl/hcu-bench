#ifndef DEVICE_H
#define DEVICE_H

#include "hip/hip_runtime.h"
#include "hipapi.hpp"
#include <iomanip>
#include <string>
#include <vector>

class Device {
public:
  int cuNum;
  int device_index;
  int device_count;
  double clockRate; // MHz
  int memory_bus_width;
  int driver;
  int compute_major;
  int compute_minor;
  std::string device_name;
  size_t free_mem;
  size_t total_mem;
  bool run_all;

public:
  Device(int device, bool print, bool runAll);
  ~Device();

  double get_sclk();
  double get_uclk();
  void getNum();
  std::string getNum(const std::string &line);
  hipDeviceProp_t getProp() { return prop; }

private:
  hipDeviceProp_t prop;
};

#endif