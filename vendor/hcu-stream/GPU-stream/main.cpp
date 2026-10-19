
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <limits>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <unordered_map>

#define VERSION_STRING "3.4"

#include "Stream.h"

#if defined(CUDA)
#include "CUDAStream.h"
#elif defined(HIP)
#include "HIPStream.h"
#elif defined(HC)
#include "HCStream.h"
#elif defined(OCL)
#include "OCLStream.h"
#elif defined(USE_RAJA)
#include "RAJAStream.hpp"
#elif defined(KOKKOS)
#include "KokkosStream.hpp"
#elif defined(ACC)
#include "ACCStream.h"
#elif defined(SYCL)
#include "SYCLStream.h"
#elif defined(OMP)
#include "OMPStream.h"
#endif

#ifdef HIP_DISABLE_FP64
    #define PRECISION_LMT 1
#else
    #define PRECISION_LMT 2
#endif

typedef enum {
  COPY,
  MUL,
  ADD,
  TRIAD,
  DOT,
  WRITE,
  READ,
  NONETEST
}TestCase;

TestCase getTestName(char* c)
{
  if (strcmp(c, "copy") == 0)
    return TestCase::COPY;
  else if (strcmp(c, "mul") == 0)
    return TestCase::MUL;
  else if (strcmp(c, "add") == 0)
    return TestCase::ADD;
  else if (strcmp(c, "triad") == 0)
    return TestCase::TRIAD;
  else if (strcmp(c, "dot") == 0)
    return TestCase::DOT;
  else if (strcmp(c, "write") == 0)
    return TestCase::WRITE;
  else if (strcmp(c, "read") == 0)
    return TestCase::READ;
  else
    return TestCase::NONETEST;
}
TestCase test_case = TestCase::NONETEST;

// Default size of 2^29
unsigned int ARRAY_SIZE = 536870912;
unsigned int num_times = 800;
unsigned int deviceIndex = 0;
unsigned int data_precision = 0;
unsigned int gddr6_rate = 0;
bool output_as_csv = false;
bool mibibytes = false;
std::string csv_separator = ",";
float startA, startB, startC;
bool use_rand = false;
bool hcu = false;

template <typename T>
void check_solution(const unsigned int ntimes, std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, T& sum);

template <typename T>
void run();

void parseArguments(int argc, char *argv[]);

int main(int argc, char *argv[])
{

  parseArguments(argc, argv);
  if (test_case == TestCase::NONETEST)
  {
    std::cout << "  -t  --testcase   testcase  Run one of copy mul add triad dot read write" << std::endl;;
    return 0;
  }

  if (!output_as_csv && !hcu)
  {
    std::cout
      << "BabelStream" << std::endl
      << "Version: " << VERSION_STRING << std::endl
      << "Implementation: " << IMPLEMENTATION_STRING << std::endl;
  }

  // gen rand value
  if (use_rand)
  {
    srand(time(NULL));
    startA = (float)rand() / (float)RAND_MAX;
    startB = (float)rand() / (float)RAND_MAX;
    startC = (float)rand() / (float)RAND_MAX;
  }
  else
  {
    if (data_precision == 0){
        startA = startA_def*10;
        startB = startB_def*10;
        startC = startC_def*10;
    }
    else {
        startA = startA_def;
        startB = startB_def;
        startC = startC_def;
    }
  }


    if (data_precision == 0){
        run<long>();
    }else if (data_precision == 1){
        run<float>();
    }
#ifndef HIP_DISABLE_FP64
    else if (data_precision == 2){
        run<double>();
    }
#endif
    else {
        std::cerr << "Invalid: --precision " << data_precision << std::endl;
        exit(EXIT_FAILURE);
    }
}


template <typename T>
T get_startScalar(void)
{
  return startScalar;
}
template <>
long get_startScalar<long>(void)
{
  return startScalarLong;
}

template <typename T>
void print_typename(void)
{
    if (std::is_same<T, float>::value)
    {
        std::cout << "Precision: float" << std::endl;
    }
    else if (std::is_same<T, double>::value)
    {
        std::cout << "Precision: double" << std::endl;
    }
    else if (std::is_same<T, long>::value)
    {
        std::cout << "Precision: long" << std::endl;
    }
    else
    {
        std::cout << "Precision: Unknown" << std::endl;
    }
}

template <typename T>
void run()
{
  std::unordered_map<TestCase, std::string> cuda_result = {
    {TestCase::READ,  "H100:3231.05GB/s  A800:1844.18GB/s"},
    {TestCase::WRITE, "H100:3295.38GB/s  A800:1869.50GB/s"},
    {TestCase::COPY,  "H100:2756.14GB/s  A800:1743.24GB/s"},
    {TestCase::DOT,   "H100:3177.72GB/s  A800:1552.64GB/s"},
    {TestCase::MUL,   "H100:2752.16GB/s  A800:1742.45GB/s"},
    {TestCase::ADD,   "H100:3083.46GB/s  A800:1769.98GB/s"},
    {TestCase::TRIAD, "H100:3079.96GB/s  A800:1769.82GB/s"}
  };
  std::string labels[7] = {"Copy", "Mul", "Add", "Triad", "Dot", "Write", "Read"};
  size_t sizes[7] = {
    2 * sizeof(T) * ARRAY_SIZE,
    2 * sizeof(T) * ARRAY_SIZE,
    3 * sizeof(T) * ARRAY_SIZE,
    3 * sizeof(T) * ARRAY_SIZE,
    2 * sizeof(T) * ARRAY_SIZE,
    3 * sizeof(T) * ARRAY_SIZE,
    2 * sizeof(T) * ARRAY_SIZE
  };

#ifdef MODEL
  num_times = 1;
#endif

  std::streamsize ss = std::cout.precision();

  if (!output_as_csv && !hcu)
  {
    std::cout << "Running kernels " << num_times << " times" << std::endl;
    print_typename<T>();

    if (mibibytes)
    {
      // MiB = 2^20
      std::cout << std::setprecision(1) << std::fixed
                << "Array size: " << ARRAY_SIZE*sizeof(T)*pow(2.0, -20.0) << " MiB"
                << " (=" << ARRAY_SIZE*sizeof(T)*pow(2.0, -30.0) << " GiB)" << std::endl;
      std::cout << "Total size: " << sizes[test_case]*pow(2.0, -20.0) << " MiB"
                << " (=" << sizes[test_case]*pow(2.0, -30.0) << " GiB)" << std::endl;
    }
    else
    {
      // MB = 10^6
      std::cout << std::setprecision(1) << std::fixed
                << "Array size: " << ARRAY_SIZE*sizeof(T)*1.0E-6 << " MB"
                << " (=" << ARRAY_SIZE*sizeof(T)*1.0E-9 << " GB)" << std::endl;
      std::cout << "Total size: " << sizes[test_case]*1.0E-6 << " MB"
                << " (=" << sizes[test_case]*1.0E-9 << " GB)" << std::endl;
    }
    std::cout.precision(ss);

  }

  // Create host vectors
  std::vector<T> a(ARRAY_SIZE);
  std::vector<T> b(ARRAY_SIZE);
  std::vector<T> c(ARRAY_SIZE);

  // Result of the Dot kernel
  T sum, initA, initB, initC, initScalar;

  Stream<T> *stream;

#if defined(CUDA)
  // Use the CUDA implementation
  stream = new CUDAStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(HIP)
  // Use the HIP implementation
  stream = new HIPStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(HC)
  // Use the HC implementation
  stream = new HCStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(OCL)
  // Use the OpenCL implementation
  stream = new OCLStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(USE_RAJA)
  // Use the RAJA implementation
  stream = new RAJAStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(KOKKOS)
  // Use the Kokkos implementation
  stream = new KokkosStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(ACC)
  // Use the OpenACC implementation
  stream = new ACCStream<T>(ARRAY_SIZE, a.data(), b.data(), c.data(), deviceIndex);

#elif defined(SYCL)
  // Use the SYCL implementation
  stream = new SYCLStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(OMP)
  // Use the OpenMP implementation
  stream = new OMPStream<T>(ARRAY_SIZE, a.data(), b.data(), c.data(), deviceIndex);

#endif

  initA = (T)startA;
  initB = (T)startB;
  initC = (T)startC;
  initScalar = get_startScalar<T>();
  if (!hcu) {
    std::cout <<"initA:"<<initA<<", initB:"<<initB<<", initC:"<<initC<<", initScalar:"<<initScalar<< std::endl;
  }

  // List of times
  std::vector<double> timings;

  // Declare timers
  float milliseconds = 0;

  // Only Write
  for (unsigned int k = 1; k < (num_times+1); k++)
  {
    switch (test_case)
    {
      case TestCase::WRITE:
        // Execute Write
        stream->init_arrays(initA+(T)k, initB+(T)k, initC+(T)k, milliseconds);
        timings.push_back( milliseconds * 1.0E-3);
        break;

      case TestCase::READ:
        // Execute read
        stream->read(milliseconds);
        timings.push_back( milliseconds * 1.0E-3);
        break;

      default:
        break;
    }
  }
  // Main loop
  for (unsigned int k = 0; k < num_times; k++)
  {
    switch (test_case)
    {
    case TestCase::COPY:
      // Execute Copy
      stream->copy(milliseconds);
      timings.push_back( milliseconds * 1.0E-3);
      break;
    
    case TestCase::MUL:
      // Execute Mul
      stream->mul(milliseconds);
      timings.push_back( milliseconds * 1.0E-3);
      break;

    case TestCase::ADD:
      // Execute Add
      stream->add(milliseconds);
      timings.push_back( milliseconds * 1.0E-3);
      break;

    case TestCase::TRIAD:
      // Execute Triad
      stream->triad(milliseconds);
      timings.push_back( milliseconds * 1.0E-3);
      break;

    case TestCase::DOT:
      // Execute Dot
      sum = stream->dot(milliseconds);
      timings.push_back( milliseconds * 1.0E-3);
      break;

    default:
      break;
    }
  }

  // Check solutions
  // stream->read_arrays(a, b, c);
  // check_solution<T>(num_times, a, b, c, sum);

  // theoretical bandwidth
  double theoretical_bw = (double)gddr6_rate * 6400;

  // Display timing results
  if (!hcu) {
    if (output_as_csv)
    {
      std::cout
        << "function" << csv_separator
        << "num_times" << csv_separator
        << "n_elements" << csv_separator
        << "sizeof" << csv_separator
        << ((mibibytes) ? "max_gibytes_per_sec" : "max_gbytes_per_sec") << csv_separator
        << "min_runtime" << csv_separator
        << "max_runtime" << csv_separator
        << "avg_runtime" << std::endl;
    }
    else
    {
      std::cout
        << std::left << std::setw(12) << "Function"
        << std::left << std::setw(12) << ((mibibytes) ? "GiB/s(mean)" : "GB/s(mean)")
        << std::left << std::setw(12) << "Min (sec)"
        << std::left << std::setw(12) << "Max"
        << std::left << std::setw(12) << "Average"
        << std::left << "Rate("<<std::setprecision(1)<<theoretical_bw<<")"
        << std::endl
        << std::fixed;
    }
  }

#ifdef MODEL
   auto minmax = std::minmax_element(timings.begin(), timings.end());

  // Calculate average; ignore the first result
  double average = std::accumulate(timings.begin(), timings.end(), 0.0) / (double)(num_times);
#else
  // Get min/max; ignore the first result
  auto minmax = std::minmax_element(timings.begin()+1, timings.end());

  // Calculate average; ignore the first result
  double average = std::accumulate(timings.begin()+1, timings.end(), 0.0) / (double)(num_times - 1);
#endif


  // Display results
  if (!hcu) {
    if (output_as_csv)
    {
      std::cout
        << labels[test_case] << csv_separator
        << num_times << csv_separator
        << ARRAY_SIZE << csv_separator
        << sizeof(T) << csv_separator
        << ((mibibytes) ? pow(2.0, -30.0) : 1.0E-9) * sizes[test_case] / (*minmax.first) << csv_separator
        << *minmax.first << csv_separator
        << *minmax.second << csv_separator
        << average
        << std::endl;
    }
    else
    {
      double average_bw = ((mibibytes) ? pow(2.0, -30.0) : 1.0E-9) * sizes[test_case] / (average);
      std::cout
        << std::left << std::setw(12) << labels[test_case]
        << std::left << std::setw(12) << std::setprecision(3) << average_bw
        << std::left << std::setw(12) << std::setprecision(5) << *minmax.first
        << std::left << std::setw(12) << std::setprecision(5) << *minmax.second
        << std::left << std::setw(12) << std::setprecision(5) << average;
      if (theoretical_bw > 1){
        std::cout << std::setprecision(2) << average_bw*100.0/theoretical_bw << "%";
      }
      std::cout << std::endl;
    }
  }
  else {
    double average_bw = ((mibibytes) ? pow(2.0, -30.0) : 1.0E-9) * sizes[test_case] / (average);
    std::cout <<labels[test_case] <<"  " << cuda_result[test_case] << " HCU:"
        <<  average_bw <<"GB/s";
    std::cout << std::endl;
  }
  delete stream;

}

template <typename T>
T temp_abs(T x)
{
    if (x < 0){
        return -x;
    }
    return x;
}

template <typename T>
void check_solution(const unsigned int ntimes, std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, T& sum)
{
  // Generate correct solution
  T goldA = startA;
  T goldB = startB;
  T goldC = startC;
  T goldSum = 0.0;

  const T scalar = get_startScalar<T>();

  for (unsigned int i = 0; i < ntimes; i++)
  {
    goldC = goldA;
    goldB = scalar * goldC;
    goldC = goldA + goldB;
    goldA = goldB + scalar * goldC;
  }

  // Do the reduction
  goldSum = goldA * goldB * ARRAY_SIZE;

  // Calculate the average error
  T errA = std::accumulate(a.begin(), a.end(), 0.0, [&](T sum, const T val){ return sum + temp_abs(val - goldA); });
  errA /= a.size();
  T errB = std::accumulate(b.begin(), b.end(), 0.0, [&](T sum, const T val){ return sum + temp_abs(val - goldB); });
  errB /= b.size();
  T errC = std::accumulate(c.begin(), c.end(), 0.0, [&](T sum, const T val){ return sum + temp_abs(val - goldC); });
  errC /= c.size();
  T errSum = temp_abs(sum - goldSum);

  double epsi = std::numeric_limits<T>::epsilon() * 100.0;

  if ((double)errA > epsi)
    std::cerr
      << "Validation failed on a[]. Average error " << errA
      << std::endl;
  if ((double)errB > epsi)
    std::cerr
      << "Validation failed on b[]. Average error " << errB
      << std::endl;
  if ((double)errC > epsi)
    std::cerr
      << "Validation failed on c[]. Average error " << errC
      << std::endl;
  // Check sum to 8 decimal places
  if ((double)errSum > 1.0E-8)
    std::cerr
      << "Validation failed on sum. Error " << errSum
      << std::endl << std::setprecision(15)
      << "Sum was " << sum << " but should be " << goldSum
      << std::endl;

}

int parseUInt(const char *str, unsigned int *output)
{
  char *next;
  *output = strtoul(str, &next, 10);
  return !strlen(next);
}

void parseArguments(int argc, char *argv[])
{
  for (int i = 1; i < argc; i++)
  {
    if (!std::string("--list").compare(argv[i]))
    {
      listDevices();
      exit(EXIT_SUCCESS);
    }
    else if (!std::string("--device").compare(argv[i]))
    {
      if (++i >= argc || !parseUInt(argv[i], &deviceIndex))
      {
        std::cerr << "Invalid device index." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--arraysize").compare(argv[i]) ||
             !std::string("-s").compare(argv[i]))
    {
      if (++i >= argc || !parseUInt(argv[i], &ARRAY_SIZE))
      {
        std::cerr << "Invalid array size." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--numtimes").compare(argv[i]) ||
             !std::string("-n").compare(argv[i]))
    {
      if (++i >= argc || !parseUInt(argv[i], &num_times))
      {
        std::cerr << "Invalid number of times." << std::endl;
        exit(EXIT_FAILURE);
      }
      if (num_times < 2)
      {
        std::cerr << "Number of times must be 2 or more" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--precision").compare(argv[i]))
    {
      if (++i >= argc || !parseUInt(argv[i], &data_precision))
      {
        std::cerr << "Invalid --precision." << std::endl;
        exit(EXIT_FAILURE);
      }
      if (data_precision > PRECISION_LMT)
      {
        std::cerr << "Unknow: --precision " << data_precision << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--gddr6_rate").compare(argv[i]))
    {
      if (++i >= argc || !parseUInt(argv[i], &gddr6_rate))
      {
        std::cerr << "Invalid --gddr6_rate." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--csv").compare(argv[i]))
    {
      output_as_csv = true;
    }
    else if (!std::string("--mibibytes").compare(argv[i]))
    {
      mibibytes = true;
    }
    else if (!std::string("--random").compare(argv[i]))
    {
      use_rand = true;
    }
    else if (!std::string("--testcase").compare(argv[i]) ||
        !std::string("-t").compare(argv[i]))
    {
      if (++i >= argc || (getTestName(argv[i]) == TestCase::NONETEST))
      {
        std::cerr << "Invalid testcase: " << argv[i] << std::endl;
        exit(EXIT_FAILURE);
      }
      test_case = getTestName(argv[i]);
    }

    else if (!std::string("--hcu").compare(argv[i]))
    {
      hcu = true;
    }
    else if (!std::string("--help").compare(argv[i]) ||
             !std::string("-h").compare(argv[i]))
    {
      std::cout << std::endl;
      std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "      --hcu                  print format for hcu" << std::endl;
      std::cout << "  -h  --help                 Print the message" << std::endl;
      std::cout << "      --list                 List available devices" << std::endl;
      std::cout << "      --device     INDEX     Select device at INDEX" << std::endl;
      std::cout << "  -s  --arraysize  SIZE      Use SIZE elements in the array" << std::endl;
      std::cout << "  -n  --numtimes   NUM       Run the test NUM times (NUM >= 2)" << std::endl;
      std::cout << "      --precision  NUM       The data type of the array(0:long, 1:float, 2:double)" << std::endl;
      std::cout << "      --csv                  Output as csv table" << std::endl;
      std::cout << "      --mibibytes            Use MiB=2^20 for bandwidth calculation (default MB=10^6)" << std::endl;
      std::cout << "      --random               Use random value init the array" << std::endl;
      std::cout << "  -t  --testcase   testcase  Run one of copy mul add triad dot read write" << std::endl;
    //   std::cout << "      --gddr6_rate         Used to calculate the theoretical bandwidth(100~160:10~16Gbps)" << std::endl;
      std::cout << std::endl;
      exit(EXIT_SUCCESS);
    }
    else
    {
      std::cerr << "Unrecognized argument '" << argv[i] << "' (try '--help')"
                << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}


