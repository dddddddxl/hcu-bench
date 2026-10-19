#pragma once

#include <array>
#include <cstdint>
#include <assert.h>
#include <hip/hip_runtime.h>

#ifndef C10_HOST_DEVICE
#define C10_HOST_DEVICE __host__ __device__
#endif

constexpr int MAX_DIMS = 16;


template <typename Value>
struct DivMod {
  Value div, mod;
  C10_HOST_DEVICE DivMod(Value div, Value mod) : div(div), mod(mod) { }
};

template <typename Value>
struct IntDivider {
  IntDivider() = default;
  IntDivider(Value d) : divisor(d) { }

  C10_HOST_DEVICE inline Value div(Value n) const { return n / divisor; }
  C10_HOST_DEVICE inline Value mod(Value n) const { return n % divisor; }
  C10_HOST_DEVICE inline DivMod<Value> divmod(Value n) const {
    return DivMod<Value>(n / divisor, n % divisor);
  }

  Value divisor;
};

// Implement fast integer division.
template <>
struct IntDivider<unsigned int> {
  static_assert(sizeof(unsigned int) == 4, "Assumes 32-bit unsigned int.");

  IntDivider() = default;

  IntDivider(unsigned int d) : divisor(d) {
    assert(divisor >= 1 && divisor <= INT32_MAX);

    // TODO: gcc/clang has __builtin_clz() but it's not portable.
    for (shift = 0; shift < 32; shift++) if ((1U << shift) >= divisor) break;

    uint64_t one = 1;
    uint64_t magic = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;
    m1 = magic;
    assert(m1 > 0 && m1 == magic);  // m1 must fit in 32 bits.
  }

  C10_HOST_DEVICE inline unsigned int div(unsigned int n) const {
    // 't' is the higher 32-bits of unsigned 32-bit multiplication of 'n' and
    // 'm1'.
    unsigned int t = __umulhi(n, m1);
    return (t + n) >> shift;
  }

  C10_HOST_DEVICE inline unsigned int mod(unsigned int n) const {
    return n - div(n) * divisor;
  }

  C10_HOST_DEVICE inline DivMod<unsigned int> divmod(unsigned int n) const {
    unsigned int q = div(n);
    return DivMod<unsigned int>(q, n - q * divisor);
  }

  unsigned int divisor;  // d above.
  unsigned int m1;  // Magic number: m' above.
  unsigned int shift;  // Shift amounts.
};


template <typename index_t = uint32_t, bool signed_strides = false>
struct OffsetCalculator {
  using stride_t = std::conditional_t<signed_strides,
                                      std::make_signed_t<index_t>,
                                      index_t>;
  using offset_type = stride_t;
  OffsetCalculator(int dims, const int* sizes, const uint64_t* strides) : dims(dims) {
    for (int i=0; i < dims; i++){
        sizes_[i] = IntDivider<index_t>(sizes[i]);
        strides_[i] = (stride_t)strides[i];
        // printf("sizes[%d] = %d, strides_[%d] = %d\n", i, sizes[i], i, strides_[i]);
    }
  }

  C10_HOST_DEVICE stride_t get(index_t linear_idx) const {
    stride_t offset = 0;

    #pragma unroll
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim == dims) {
        break;
      }
      auto divmod = sizes_[dim].divmod(linear_idx);
      // printf("get function: sizes[%d] = %d, lineear_idx = %d, result = [%d]\n", dim, sizes[dim], linear_idx, divmod);
      linear_idx = divmod.div;
      // printf("[%d] = %d, lineear_idx = %d, result = [%d]\n", dim, sizes[dim], linear_idx, divmod);
      offset += divmod.mod * strides_[dim];

    }
    return offset;
  }

  int dims;
  IntDivider<index_t> sizes_[MAX_DIMS];
  stride_t strides_[MAX_DIMS];
};

// template <typename index_t = uint32_t>
// struct TrivialOffsetCalculator {

//   C10_HOST_DEVICE index_t get(index_t linear_idx) const {
//     return linear_idx;
//   }
// };