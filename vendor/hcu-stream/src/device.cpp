#include "device.hpp"
#include <iostream>
using namespace std;

Device::Device(int device, bool print, bool runAll = false) {
  int count = 0;
  run_all = runAll;
  CHECK_HIP_ERROR(hipGetDeviceCount(&count));
  if (device >= count) {
    throw std::runtime_error("Invalid device index");
  }
  CHECK_HIP_ERROR(hipSetDevice(device));
  int _driver;
  CHECK_HIP_ERROR(hipDriverGetVersion(&_driver));
  CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, device));
  cuNum = prop.multiProcessorCount;
  device_index = device;
  device_count = count;
  clockRate = prop.clockRate;
  memory_bus_width = prop.memoryBusWidth;
  driver = _driver;
  double global_mem_mb = static_cast<double>(prop.totalGlobalMem) / 1048576.0;
  device_name = prop.name;
  compute_major = prop.major;
  compute_minor = prop.minor;

  hipMemGetInfo(&free_mem, &total_mem);
  // 61
  if (print) {
    std::cout << ".-----------------------------------------------------------------------------. \n";
    std::cout << "|----------------.------------------------------------------------------------|\n";
    std::cout << "| Device ID      | " << device << std::setw(60) << "|\n";
    std::cout << "| Device Name    | " << prop.name << std::setw(61 - std::string(prop.name).length()) << "|\n";
    std::cout << "| Device Vendor  | " << "C-3000, Inc." << std::setw(49) << "|\n";
    std::cout << "| Device Driver  | " << _driver << " (HSA1.1,LC) (Linux)" << std::setw(33) << "|\n";
    std::cout << "| Clang Version  | ------------                                               |\n";
    std::cout << "| Compute Units  | " << prop.multiProcessorCount << " at "
              << prop.clockRate / 1000.0 << " MHz (" << prop.multiProcessorCount
              << " coumpute Units )" << std::fixed
              << std::setw(26) << "|\n";
    std::cout << "| Memory, Cache  | " << total_mem / (1024 * 1024)
              << " MB VRAM, "  << prop.sharedMemPerBlock / 1024 << " KB local "<< std::setw(34)<<"|\n";
    std::cout << "|----------------'------------------------------------------------------------|\n";
  }
}

double Device::get_sclk() {
  if (std::getenv("MODEL")) {
    char *str = getenv("MODEL");
    std::string model(str);
    int sclk = std::stoi(model);
    if (sclk == 1)
      return 1400.0;
    return (double)sclk;
  } else {
    int deviceNum, deviceId;
    CHECK_HIP_ERROR(hipGetDeviceCount(&deviceNum));
    CHECK_HIP_ERROR(hipGetDevice(&deviceId));
    int maxCard = -1;
    double sclk;
    std::vector<std::string> lines;
    std::string line;
    std::string cardName;
    std::string sclkPath;

    for (int i = 0; i < 50; i++) {
      cardName = "card" + std::to_string(i);
      sclkPath = "/sys/class/drm/" + cardName + "/device/pp_dpm_sclk";
      std::ifstream file(sclkPath.c_str());
      if (file.good())
        maxCard = i;
    }
    if (maxCard == -1) {
      assert(0 && "no pp_dpm_sclk in /sys/class/drm/");
    }
    // std::cout << "maxCard: " << maxCard << "; deviceNum: " << deviceNum << ";
    // deviceId " << deviceId << std::endl;
    int cardId = maxCard - deviceNum + deviceId + 1;
    cardName = "card" + std::to_string(cardId);
    sclkPath = "/sys/class/drm/" + cardName + "/device/pp_dpm_sclk";
    std::ifstream file(sclkPath.c_str());
    if (!file.is_open())
      throw std::runtime_error("can not open: " + sclkPath);
    while (std::getline(file, line)) {
      lines.push_back(line);
    }
    for (const auto &line : lines) {
      std::string str = getNum(line);
      if (!str.empty())
        sclk = (double)std::stoi(str);
    }

    return sclk;
  }
}

double Device::get_uclk() {
  if (std::getenv("UCLK")) {
    char *str = getenv("UCLK");
    std::string UCLK(str);
    int _uclk = std::stoi(UCLK);
    return (double)_uclk;
  } else {
    return 1400.0f;
  }
}

std::string Device::getNum(const std::string &line) {
  size_t end = line.find("Mhz *");
  if (end != std::string::npos) {
    size_t start = end;
    while (std::isdigit(line[start - 1]))
      --start;

    return line.substr(start, end - start);
  }

  return "";
}
