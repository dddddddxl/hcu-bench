#include "output.hpp"
#include <iostream>
#include <cstdint>
#include <vector>
using namespace std;

Output::Output() {
  func.clear();
  wave.clear();
  data.clear();
  avg_bw.clear();
  latency.clear();
}

BW_Output::BW_Output() : Output() {
  theory.clear();
  lds.clear();
  deviceId.clear();
  // std::cout << "print Bandwidth:....\n";
}

// 析构函数
BW_Output::~BW_Output() { std::cout << "BW_Output instance destroyed.\n"; }

// 添加记录
void BW_Output::add_record(const std::string &name, int _wave, double data_val,
                           double bw, double lat, double theory_val, int _lds, int _id) {
  lds.push_back(_lds);
  func.push_back(name);
  data.push_back(data_val);
  wave.push_back(_wave);
  avg_bw.push_back(bw);
  latency.push_back(lat);
  theory.push_back(theory_val);
  deviceId.push_back(_id);
  // std::cout << "Record added: " << name << "\n";
}

void BW_Output::print_clean() {
  if (func.empty()) {
    // std::cout << "No records to display.\n";
    return;
  }

  for (size_t i = 0; i < func.size(); ++i) {
    std::cout << avg_bw[i] <<"\n";
  }
}

// 打印记录
void BW_Output::print_record() {
  if (func.empty()) {
    std::cout << "No records to display.\n";
    return;
  }

  // 打印表头
  std::cout << std::left << std::setw(6) << "HCU" << std::setw(32) << "Function" << std::setw(10)
            << "lds(size)" << setw(8) << "Wave" << std::setw(15)
            << "DataSize(MB)" << std::setw(15) << "Avg BW(GB/s)"
            << std::setw(15) << "Latency(ms)"
            << "\n";

  std::cout << std::string(100, '-') << "\n";

  // 打印记录
  for (size_t i = 0; i < func.size(); ++i) {
    std::cout << std::left << std::setw(6) << deviceId[i] 
              << std::setw(32) << func[i] << std::setw(10)
              << lds[i] << std::setw(8) << wave[i] << std::setw(15) << data[i]
              << std::setw(15) << avg_bw[i] << std::setw(15) << latency[i]
              << "\n";
  }
}

// 清除所有记录
void BW_Output::clear_records() {
  func.clear();
  data.clear();
  avg_bw.clear();
  latency.clear();
  theory.clear();
  lds.clear();
  std::cout << "All records cleared.\n";
}

LT_Output::LT_Output() : Output() {
  theory.clear();
  std::cout << "LT_Output instance created.\n";
}

// 析构函数
LT_Output::~LT_Output() { std::cout << "LT_Output instance destroyed.\n"; }

// 添加记录
void LT_Output::add_record_lt(const std::string &name, double lat,
                              std::string theory_val) {
  func.push_back(name);
  latency.push_back(lat);
  theory.push_back(theory_val);
  // std::cout << "Record added: " << name << "\n";
}

// 打印记录
void LT_Output::print_record() {
  if (func.empty()) {
    std::cout << "No records to display.\n";
    return;
  }

  // 打印表头
  std::cout << std::left << std::setw(40) << "Function" << std::setw(15)
            << "Latency(ms)" << std::setw(30) << "Theory(MB/s)"
            << "\n";

  std::cout << std::string(85, '-') << "\n";

  // 打印记录
  for (size_t i = 0; i < func.size(); ++i) {
    std::cout << std::left << std::setw(40) << func[i] << std::setw(15)
              << latency[i] << std::setw(30) << theory[i] << "\n";
  }
}

// 清除所有记录
void LT_Output::clear_records() {
  func.clear();
  data.clear();
  avg_bw.clear();
  latency.clear();
  theory.clear();
  std::cout << "All records cleared.\n";
}

TF_Output::TF_Output() : Output() {
  std::cout << "TF_Output instance created.\n";
}

// 析构函数
TF_Output::~TF_Output() { std::cout << "TF_Output instance destroyed.\n"; }

L2_Output::L2_Output() : Output() {
  std::cout << "L2_Output instance created.\n";
}

// 析构函数
L2_Output::~L2_Output() {
  min.clear();
  max.clear();
  trans_size.clear();
  std::cout << "L2_Output instance destroyed.\n";
}
void L2_Output::add_L2_record(std::string &name, double _avg_bw,
                              double avg_time, double datasize,
                              double _trans_size, double _min, double _max) {
  func.push_back(name);
  data.push_back(datasize);
  avg_bw.push_back(_avg_bw);
  latency.push_back(avg_time);
  min.push_back(_min);
  max.push_back(_max);
  trans_size.push_back(_trans_size);
}
void L2_Output::print_record() {
  if (func.empty()) {
    std::cout << "No records to display.\n";
    return;
  }

  // 打印表头
  std::cout << std::left << std::setw(20) << "Function" << std::left
            << std::setw(20) << "DataSize(MB)" << std::left << std::setw(20)
            << "TransSize(GB)" << std::left << std::setw(20) << "Speed(GB/s)"
            << std::left << std::setw(20) << "Min (ms)" << std::left
            << std::setw(20) << "Max (ms)" << std::left << std::setw(20)
            << "Average (ms)" << std::endl
            << std::fixed;

  std::cout << std::string(85, '-') << "\n";

  // 打印记录
  for (size_t i = 0; i < func.size(); ++i) {
    std::cout << std::left << std::setw(20) << func[i] << std::left
              << std::setw(20) << std::setprecision(3)
              << data[i] * pow(2.0, -20.0) << std::left << std::setw(20)
              << std::setprecision(3) << trans_size[i] * pow(2.0, -30.0)
              << std::left << std::setw(20) << std::setprecision(3) << avg_bw[i]
              << std::left << std::setw(20) << std::setprecision(5) << min[i]
              << std::left << std::setw(20) << std::setprecision(5) << max[i]
              << std::left << std::setw(20) << std::setprecision(5)
              << latency[i] << std::endl;
  }
}

Dispatch_Output::Dispatch_Output() : Output() {
  grids.clear();
  blocks.clear();
  total_waves.clear();
  waves_per_tg.clear();
  latencies.clear();
  // std::cout << "Dispatch_Output instance created.\n";
}

// 析构函数
Dispatch_Output::~Dispatch_Output() {
  grids.clear();
  blocks.clear();
  total_waves.clear();
  waves_per_tg.clear();
  latencies.clear();
  std::cout << "Dispatch_Output instance destroyed.\n";
}
void Dispatch_Output::add_dispatch_record(std::string &name, int grid, int block, int total_wave, int _waves_per_tg, double lat) {
  if (std::find(func.begin(), func.end(), name) == func.end()) {
    func.push_back(name);
    grids.resize(func.size());
    blocks.resize(func.size());
    total_waves.resize(func.size());
    waves_per_tg.resize(func.size());
    latencies.resize(func.size());
  }
  int current = func.size() - 1;
  grids[current].push_back(grid);
  blocks[current].push_back(block);
  total_waves[current].push_back(total_wave);
  waves_per_tg[current].push_back(_waves_per_tg);
  latencies[current].push_back(lat);
}
void Dispatch_Output::print_record_per_func(int pos) {
  if (latencies[pos].empty()) {
    std::cout << "No records to display.\n";
    return;
  }

  // 打印表头
  std::cout << std::left << std::setw(20) << "Grid" << std::left
            << std::setw(20) << "Block" << std::left << std::setw(20)
            << "Total_Waves" << std::left << std::setw(20) << "Waves_Per_TG"
            << std::left << std::setw(20) << "Latencies (ms)" << std::endl
            << std::fixed;

  std::cout << std::string(85, '-') << "\n";

  // 打印记录
  for (size_t i = 0; i < latencies[pos].size(); ++i) {
    std::cout << std::left << std::setw(20) << grids[pos][i]
              << std::left << std::setw(20) << blocks[pos][i]
              << std::left << std::setw(20) << total_waves[pos][i]
              << std::left << std::setw(20) << waves_per_tg[pos][i]
              << std::left << std::setw(20) << std::setprecision(3) << latencies[pos][i]
              << std::endl;
  }
  if (dispatch_cycles_per_tg.count(func[pos]))
    std::cout << "Dispatch cycles per tg = " << std::setprecision(3) << dispatch_cycles_per_tg[func[pos]] << " cycles/tg" << std::endl;
}
void Dispatch_Output::print_record() {
  for (size_t i = 0; i < func.size(); ++i) {
    std::cout << std::left << std::setw(20) << "Function" << std::left << std::setw(20) << func[i] << std::endl;
    print_record_per_func(i);
  }

}
void Dispatch_Output::print_clean() {
  if (func.empty())
    return;
  for (size_t i = 0; i < func.size(); ++i) {
    if (dispatch_cycles_per_tg.count(func[i]))
      std::cout << std::setprecision(3) << dispatch_cycles_per_tg[func[i]] << " cycles/tg" << std::endl;
  }
}

VGPRSafe_Output::VGPRSafe_Output() : Output() {
  test_types.clear();
  vgpr_values.clear();
}

VGPRSafe_Output::~VGPRSafe_Output() {
  test_types.clear();
  vgpr_values.clear();
}

void VGPRSafe_Output::add_vgpr_record(const std::string &test_type, const std::vector<uint32_t> &values) {
  test_types.push_back(test_type);
  vgpr_values.push_back(values);
}

void VGPRSafe_Output::print_clean() {
  if (test_types.empty()) {
    return;
  }
  
  for (size_t i = 0; i < test_types.size(); ++i) {
    const std::string &type = test_types[i];
    const std::vector<uint32_t> &values = vgpr_values[i];
    
    uint32_t expected_value = (type == "safe") ? 0 : 2;
    bool all_match = true;
    
    for (size_t j = 0; j < values.size(); ++j) {
      if (values[j] != expected_value) {
        all_match = false;
        break;
      }
    }
    
    if (all_match) {
      std::cout << "PASS" << std::endl;
    } else {
      std::cout << "FAIL" << std::endl;
    }
  }
}

void VGPRSafe_Output::print_record() {
  if (test_types.empty()) {
    std::cout << "No records to display.\n";
    return;
  }
  
  // 打印表头
  std::cout << std::left << std::setw(20) << "Test Type" << std::setw(15) << "VGPR Index" << std::setw(15) << "Value" << "\n";
  std::cout << std::string(50, '-') << "\n";
  
  // 打印记录
  for (size_t i = 0; i < test_types.size(); ++i) {
    const std::string &type = test_types[i];
    const std::vector<uint32_t> &values = vgpr_values[i];
    
    // FIRST_V = 16, 打印从 i=48 开始，即 v64 开始 (16 + 48 = 64)
    const int FIRST_V = 16;
    const int start_index = 48;
    
    for (size_t j = 0; j < values.size(); ++j) {
      int vgpr_index = FIRST_V + start_index + static_cast<int>(j);
      std::cout << std::left << std::setw(20) << type 
                << std::setw(15) << ("v" + std::to_string(vgpr_index))
                << std::setw(15) << values[j] << "\n";
      
      // 如果达到或超过 v255，省略剩余
      if (vgpr_index >= 255) {
        if (j + 1 < values.size()) {
          std::cout << std::left << std::setw(20) << type 
                    << std::setw(15) << "..."
                    << std::setw(15) << "(omitted remaining)" << "\n";
        }
        break;
      }
    }
  }
}
