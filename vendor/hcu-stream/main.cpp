#include "output.hpp"
#include "testcase.hpp"
#include <argparse/argparse.hpp>
#include <iostream>
#define BUILD_TIMESTAMP __DATE__ " " __TIME__

// https://github.com/p-ranav/argparse?tab=readme-ov-file#printing-help
int main(int argc, char **argv) {
  // 数据缩减量除数
  double crop_data = 1;
  // 默认kernel 为4wave
  int wave = 4;
  // 默认使用GPU device 0
  int device_index = 0;
  // pa pad 设置
  int pad = 1024;
  int height = 0;
  int width = 0;
  int lds_size = 0;
  int array_size = 10240 * 10240;
  int blocks = 192; // VGPR_FLUSH_NMZ
  std::vector<std::string> testcaseToRun;
  argparse::ArgumentParser program("run");

  // ./run bandwidth ......
  argparse::ArgumentParser bw_command("bandwidth");
  bw_command.add_description("memory bandwidth test");
  bw_command.add_argument("name")
      .metavar("case name")
      .nargs(argparse::nargs_pattern::any)
      .help("case name, eg: ./run bandwidth name");
  bw_command.add_argument("-d", "--div")
      .store_into(crop_data)
      .help("specify the crop (matrix case)datasize, eg: width / '-d', height / '-d' ")
      .metavar("1,2,4,....");
  bw_command.add_argument("-s", "--arraysize")
      .store_into(array_size)
      .help("specify the arraysize for linear case")
      .metavar("8388608~128MB, 16777216~256MB 33554432~512MB 67108864~1GB 134217728~2GB 268435456~4GB");
  bw_command.add_argument("-l", "--list", "bw_list")
      .help("List available testcases")
      .default_value(false)
      .implicit_value(true);
  bw_command.add_argument("-w", "--wave")
      .store_into(wave)
      .help("specify the case wavefroent")
      .metavar("4,8,16....");
  bw_command.add_argument("--lds")
      .store_into(lds_size)
      .help("allocate LDS size:~KB")
      .metavar("1,2,3....N KB");
  bw_command.add_argument("--clean")
      .help("run bandwidth testcase clean!")
      .default_value(false)
      .implicit_value(true);
  bw_command.add_argument("--useEvent")
      .help("use gpu time else cpu time!")
      .default_value(false)
      .implicit_value(true);
  bw_command.add_argument("--printEach")
      .help("print each result time(ms)")
      .default_value(false)
      .implicit_value(true);
  bw_command.add_argument("-d", "--device")
      .store_into(device_index)
      .help("set the GPU index")
      .metavar("0,1,2,3....");
  bw_command.add_argument("--allHCU")
      .help("print all HCU result")
      .default_value(false)
      .implicit_value(true);
  argparse::ArgumentParser lt_command("latency");
  lt_command.add_description("memory latency test");
  lt_command.add_argument("name")
      .metavar("case name")
      .nargs(argparse::nargs_pattern::any)
      .help("case name, eg: ./run latency name");
  lt_command.add_argument("-d", "--div")
      .store_into(crop_data)
      .help("specify the crop datasize")
      .metavar("0.5,1,2....");
  lt_command.add_argument("-l", "--list", "lt_list")
      .help("List available testcases")
      .default_value(false)
      .implicit_value(true);
  lt_command.add_argument("-w", "--wave")
      .store_into(wave)
      .help("specify the case wavefroent")
      .metavar("4,8,16....");

  argparse::ArgumentParser tf_command("tflops");
  tf_command.add_description("gemm peak inst test");
  tf_command.add_argument("name")
      .metavar("case name")
      .nargs(argparse::nargs_pattern::any)
      .help("case name, eg: ./run tflops name");
  tf_command.add_argument("-l", "--list", "tf_list")
      .help("List available testcases")
      .default_value(false)
      .implicit_value(true);
  tf_command.add_argument("-w", "--wave")
      .store_into(wave)
      .help("specify the case wavefroent")
      .metavar("4,8,16....");
  argparse::ArgumentParser L2Cache("l2_rw");
  L2Cache.add_description("test L2 cache rw with different stride");
  L2Cache.add_argument("-l", "--list")
      .help("List available testcases")
      .default_value(false)
      .implicit_value(true);
  ;
  argparse::ArgumentParser _TCCRead("tcc_read");
  _TCCRead.add_description("Test the bandwidth of TCC Band conflict");
  argparse::ArgumentParser elewise("elewise");
  elewise.add_description("Test the bandwidth of element wise memory: ./run elewise");
  elewise.add_argument("--clean")
      .help("run element clean, easy to record!")
      .default_value(false)
      .implicit_value(true);
  elewise.add_argument("--runAll")
      .help("run all elewise case")
      .default_value(false)
      .implicit_value(true);
  argparse::ArgumentParser fa("fa");
  fa.add_description("Test the bandwidth of fa memory read/write: ./run fa");
  fa.add_argument("--clean")
      .help("run faCase clean, easy to record!")
      .default_value(false)
      .implicit_value(true);
  fa.add_argument("--lds")
      .store_into(lds_size)
      .help("allocate LDS size:~KB")
      .metavar("1,2,3....N KB");
  fa.add_argument("--height")
      .store_into(height)
      .help("set fa height");
  fa.add_argument("--width")
      .store_into(width)
      .help("set fa width");
  fa.add_argument("--runAll")
      .help("run all fa case")
      .default_value(false)
      .implicit_value(true);
  argparse::ArgumentParser pa("pa");
  pa.add_description("Test the bandwidth of pa memory read/write: ./run pa");
  pa.add_argument("--clean")
      .help("run paCase clean, easy to record!")
      .default_value(false)
      .implicit_value(true);
  pa.add_argument("--pad").store_into(pad).help("alloc pad size!!");
  pa.add_argument("--height")
      .store_into(height)
      .help("set pa height");
  pa.add_argument("--width")
      .store_into(width)
      .help("set pa width");
  pa.add_argument("--runAll")
      .help("run all pa case")
      .default_value(false)
      .implicit_value(true);
  argparse::ArgumentParser gemm("gemm");
  gemm.add_description("Test the bandwidth of gemm memory read/write: ./run gemm");
  gemm.add_argument("--clean")
      .help("run element clean, easy to record!")
      .default_value(false)
      .implicit_value(true);
  gemm.add_argument("--lds")
      .store_into(lds_size)
      .help("allocate LDS size:~KB")
      .metavar("1,2,3....N KB");
  gemm.add_argument("--height")
      .store_into(height)
      .help("set gemm height");
  gemm.add_argument("--width")
      .store_into(width)
      .help("set gemm width");
  gemm.add_argument("--runAll")
      .help("run all gemm case")
      .default_value(false)
      .implicit_value(true);
  argparse::ArgumentParser kernel_dispatch("kernel_dispatch");
  kernel_dispatch.add_description("test kernel dispatch latencies");
  kernel_dispatch.add_argument("name")
      .metavar("case name")
      .nargs(argparse::nargs_pattern::any)
      .help("case name, eg: ./run kernel_dispatch name");
  kernel_dispatch.add_argument("-l", "--list")
      .help("List available testcases")
      .default_value(false)
      .implicit_value(true);
  kernel_dispatch.add_argument("--clean")
      .help("run kernel dispatch testcase clean!")
      .default_value(false)
      .implicit_value(true);
  argparse::ArgumentParser VGPR_safe("VGPR_TEST");
  VGPR_safe.add_description("verify vgpr safe/unsafe case!");
  VGPR_safe.add_argument("name")
      .metavar("case name")
      .nargs(argparse::nargs_pattern::any)
      .help("case name, eg: ./run VGPR_UNIT_TEST name");
  VGPR_safe.add_argument("-l", "--list")
      .help("List available testcases")
      .default_value(false)
      .implicit_value(true);
  VGPR_safe.add_argument("--clean")
      .help("run kernel VGPR_safe testcase clean!")
      .default_value(false)
      .implicit_value(true);
  VGPR_safe.add_argument("--blocks")
      .store_into(blocks)
      .help("set launch block, default 192(3block) nmz 64CU");
          

  program.add_subparser(L2Cache);
  program.add_subparser(_TCCRead);
  program.add_subparser(bw_command);
  program.add_subparser(lt_command);
  program.add_subparser(tf_command);
  program.add_subparser(elewise);
  program.add_subparser(fa);
  program.add_subparser(pa);
  program.add_subparser(gemm);
  program.add_subparser(kernel_dispatch);
  program.add_subparser(VGPR_safe);

  program.add_argument("-o", "--output")
      .default_value(std::string("-"))
      .required()
      .help("specify the output file. temply unsupport")
      .metavar("/path/");
  program.add_argument("--verbose")
      .default_value(false)
      .implicit_value(true)
      .help("display detail infomation");
  program.add_argument("-v", "--version")
      .help("显示版本信息（编译时间）")
      .default_value(false)
      .implicit_value(true);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }
  if (program["--version"] == true) {
    std::cout << "Version: " << BUILD_TIMESTAMP
               << std::endl;
    return 0;
  }
  bool printALL = bw_command["--allHCU"] == true;
  Device *device = new Device(device_index, false, printALL);
  if(program.is_subcommand_used(bw_command)) {
    if (bw_command.is_used("name"))
      testcaseToRun = bw_command.get<std::vector<std::string>>("name");
    Output *output = new BW_Output();
    bool clean = bw_command["--clean"] == true;
    bool useE = bw_command["--useEvent"] == true;
    bool printE = bw_command["--printEach"] == true;
    Testcase *test = new MemoryBandwidth(testcaseToRun, wave, crop_data, output,
                                         device, lds_size, clean /*--clean*/,
                                         array_size, useE, printE);
    if (bw_command["-l"] == true){
      test->getList();
      return 0;
    }
    test->run();
    test->print();
  }
  else if(program.is_subcommand_used(lt_command)) {
    if (lt_command.is_used("name"))
      testcaseToRun = lt_command.get<std::vector<std::string>>("name");
    Output *output = new LT_Output();
    Testcase *test =
        new MemoryLatency(testcaseToRun, wave, crop_data, output, device);
    if (lt_command["-l"] == true){
      test->getList();
      return 0;
    }
    test->run();
    test->print();
  }
  else if(program.is_subcommand_used(tf_command)) {
    if (tf_command.is_used("name"))
      testcaseToRun = tf_command.get<std::vector<std::string>>("name");
    Output *output = new TF_Output();
    Testcase *test =
        new TFLOPsCase(testcaseToRun, wave, crop_data, output, device);
    if (tf_command["-l"] == true){
      test->getList();
      return 0;
    }
    test->run();
    test->print();
  } else if (program.is_subcommand_used(L2Cache)) {

    Output *output = new L2_Output();
    Testcase *test = new L2CacheRW(output, device);
    if (L2Cache["-l"] == true) {
      test->getList();
      return 0;
    }
    test->run();
    test->print();
  } else if (program.is_subcommand_used(_TCCRead)) {
    Output *output = new BW_Output();
    Testcase *test = new TCCRead(output, device);
    test->run();
    test->print();
  } else if (program.is_subcommand_used(elewise)) {
    Output *output = new L2_Output();
    bool clean = elewise["--clean"] == true;
    Testcase *test = new ElementWise(output, device, clean /*--clean*/);
    if (elewise["--runAll"] == true) {
      test->runAll();
      test->print();
      return 0;
    }
    test->run();
    std::cout<< "element wise"<< std::endl;
    test->print();
  } else if (program.is_subcommand_used(fa)) {
    Output *output = new BW_Output();
    bool clean = fa["--clean"] == true;
    Testcase *test = new faMemory(output, device, clean /*--clean*/, lds_size, height, width);
    if (fa["--runAll"] == true) {
      test->runAll();
      test->print();
      return 0;
    }
    test->run();
    test->print();
  } else if (program.is_subcommand_used(pa)) {
    Output *output = new BW_Output();
    bool clean = pa["--clean"] == true;
    Testcase *test = new paMemory(output, device, clean /*--clean*/, pad, height, width);
    if (pa["--runAll"] == true) {
      test->runAll();
      test->print();
      return 0;
    }
    test->run();
    test->print();
  } else if (program.is_subcommand_used(gemm)) {
    Output *output = new BW_Output();
    bool clean = gemm["--clean"] == true;
    Testcase *test =
        new gemmMemory(output, device, clean /*--clean*/, lds_size, height, width);
    if (gemm["--runAll"] == true) {
      test->runAll();
      test->print();
      return 0;
    }
    std::cout << "gemmMemory case:" << std::endl;
    test->run();
    test->print();
  } else if (program.is_subcommand_used(kernel_dispatch)) {
    if (kernel_dispatch.is_used("name"))
      testcaseToRun = kernel_dispatch.get<std::vector<std::string>>("name");
    bool clean = kernel_dispatch["--clean"] == true;
    Output *output = new Dispatch_Output();
    Testcase *test = new KernelDispatch(testcaseToRun, output, device, clean);
    if (kernel_dispatch["-l"] == true) {
      test->getList();
      return 0;
    }
    test->run();
    test->print();
  } else if (program.is_subcommand_used(VGPR_safe)) {
    if (VGPR_safe.is_used("name"))
      testcaseToRun = VGPR_safe.get<std::vector<std::string>>("name");
    bool clean = VGPR_safe["--clean"] == true;
    Output *output = new VGPRSafe_Output();
    Testcase *test = new VGPRNoFlush(testcaseToRun, output, device, clean, blocks);
    if (VGPR_safe["-l"] == true) {
      test->getList();
      return 0;
    }
    test->run();
    test->print();
  }
}
