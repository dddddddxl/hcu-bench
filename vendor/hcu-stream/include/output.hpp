
#ifndef OUTPUT_H
#define OUTPUT_H

#include <iomanip>
#include <string>
#include <unordered_map>
#include <vector>

class Output {
public:
  Output();
  virtual ~Output() {}
  std::vector<std::string> func;
  std::vector<int> wave;
  std::vector<double> data;
  std::vector<double> avg_bw;
  std::vector<double> latency;
  virtual void add_record(const std::string &name, int _wave, double data_val,
                          double bw, double lat, double theory_val, int lds, int id) {}
  virtual void add_record_lt(const std::string &name, double lat,
                             std::string theory_val) {}
  virtual void add_L2_record(std::string &name, double avg_bw, double avg_time,
                             double datasize, double trans_size, double min,
                             double max) {}
  virtual void add_dispatch_record(std::string &name, int grid, int block, int total_wave, int _waves_per_tg, double lat) {}
  virtual void add_dispatch_cycles_per_tg(std::string &name, double _dispatch_cycles_per_tg) {}
  virtual void print_record() = 0;
  virtual void print_clean(){};
};

class BW_Output : public Output {
public:
  BW_Output();
  ~BW_Output();
  std::vector<int> lds;
  std::vector<int> deviceId;
  std::vector<double> theory;
  void add_record(const std::string &name, int wave, double data_val, double bw,
                  double lat, double theory_val, int _lds, int _id);
  void print_record();
  void print_clean();
  void clear_records();
};
class LT_Output : public Output {
public:
  LT_Output();
  ~LT_Output();
  std::vector<std::string> theory;
  void add_record_lt(const std::string &name, double lat,
                     std::string theory_val);
  void print_record();
  void clear_records();
};

class TF_Output : public Output {
public:
  TF_Output();
  ~TF_Output();
  void print_record() {}
};
// extern Output *output;

class L2_Output : public Output {
public:
  L2_Output();
  ~L2_Output();
  std::vector<double> min;
  std::vector<double> max;
  std::vector<double> trans_size;
  void add_L2_record(std::string &name, double avg_bw, double avg_time,
                     double datasize, double trans_size, double min,
                     double max);
  void print_record();
};

class Dispatch_Output : public Output {
public:
  Dispatch_Output();
  ~Dispatch_Output();
  std::vector<std::vector<int>> grids;
  std::vector<std::vector<int>> blocks;
  std::vector<std::vector<int>> total_waves;
  std::vector<std::vector<int>> waves_per_tg;
  std::vector<std::vector<double>> latencies;
  std::unordered_map<std::string, double> dispatch_cycles_per_tg;
  void add_dispatch_record(std::string &name, int grid, int block, int total_wave, int _waves_per_tg, double lat);
  void add_dispatch_cycles_per_tg(std::string &name, double _dispatch_cycles_per_tg) {
    dispatch_cycles_per_tg.insert({name, _dispatch_cycles_per_tg});
  }
  void print_record_per_func(int pos);
  void print_record();
  void print_clean();
};


class VGPRSafe_Output : public Output {
public:
  VGPRSafe_Output();
  ~VGPRSafe_Output();
  std::vector<std::string> test_types;  // "safe" or "unsafe"
  std::vector<std::vector<uint32_t>> vgpr_values;  // 存储每个测试的 VGPR 值
  void add_vgpr_record(const std::string &test_type, const std::vector<uint32_t> &values);
  void print_record();
  void print_clean();
};

#endif // output
