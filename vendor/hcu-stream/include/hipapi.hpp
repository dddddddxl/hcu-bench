#pragma once
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
using namespace std;

#define HIP_ASSERT(x) (assert((x) == hipSuccess))
using float4_type = __attribute__((__vector_size__(4 * sizeof(float)))) float;
typedef float float4_ __attribute__((ext_vector_type(4)));
typedef int int4_ __attribute__((ext_vector_type(4)));

#define CHECK_HIP_ERROR(error)                                                 \
  if (error != hipSuccess) {                                                   \
    fprintf(stderr, "Hip error: '%s'(%d) at %s:%d\n",                          \
            hipGetErrorString(error), error, __FILE__, __LINE__);              \
    exit(EXIT_FAILURE);                                                        \
  }
