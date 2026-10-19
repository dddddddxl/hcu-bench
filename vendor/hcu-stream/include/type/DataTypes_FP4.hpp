// On DCU:
//    the 8-bit storage format of fp4 is: {2'b0, fp4, 2'b0}
//    the 8-bit storage format of fp4x2 is: {fp4, fp4}

#pragma once

#include "type/DataTypes_FP6.hpp"

#define __FP4_HOST_DEVICE__ __host__ __device__
#define __FP4_HOST_DEVICE_STATIC__ __FP4_HOST_DEVICE__ static inline

typedef fp8_storage_t fp4_storage_t;

enum fp4_interpretation_t {
  E2M1 = 0,
};

__FP4_HOST_DEVICE_STATIC__ fp4_storage_t cvt_double_to_fp4(
    const double x, const fp4_interpretation_t /* fp4_interpretation */,
    const enum hipRoundMode /* rounding */) {
  union {
    uint32_t ui32;
    fp4_storage_t fp4[4];
  } u{0};

  u.ui32 = fcbx::from_float<float, fcbx::Encoding::E2M1, true>(float(x),
                                                               0x7f /* scale */);

  return u.fp4[0] << 2;
}

template <typename T>
__FP4_HOST_DEVICE_STATIC__ fp4_storage_t cvt_float_to_fp4(
    const T x, const fp4_interpretation_t /* fp4_interpretation */,
    const enum hipRoundMode rounding,
    uint8_t scale = 0x7f) {
  union {
    uint32_t ui32;
    fp4_storage_t fp4[4];
  } u{0};

  u.ui32 = fcbx::from_float<T, fcbx::Encoding::E2M1, true>(x, scale, rounding);

  return u.fp4[0] << 2;
}

template <typename T>
__FP4_HOST_DEVICE_STATIC__ fp4_storage_t cvt_float_to_fp4_sr(const T x, uint32_t seed, uint8_t scale = 0x7) {
  union {
    uint32_t ui32;
    fp4_storage_t fp4[4];
  } u{0};

  u.ui32 = fcbx::from_float_sr<T, fcbx::Encoding::E2M1, true>(x, seed, scale);

  return u.fp4[0] << 2;
}

struct fp4_e2m1 {
  fp4_storage_t __x;

public:
  __FP4_HOST_DEVICE__ fp4_e2m1() = default;

  __FP4_HOST_DEVICE__
  explicit fp4_e2m1(const double f)
      : __x(cvt_double_to_fp4(f, E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit fp4_e2m1(const float f)
      : __x(cvt_float_to_fp4<float>(f, E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit fp4_e2m1(const int val)
      : __x(cvt_float_to_fp4<float>(float(val), E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit fp4_e2m1(const long int val)
      : __x(cvt_float_to_fp4<float>(float(val), E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit fp4_e2m1(const long long int val)
      : __x(cvt_float_to_fp4<float>(float(val), E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit fp4_e2m1(const short int val)
      : __x(cvt_float_to_fp4<float>(float(val), E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit fp4_e2m1(const unsigned int val)
      : __x(cvt_float_to_fp4<float>(float(val), E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit fp4_e2m1(const unsigned long int val)
      : __x(cvt_float_to_fp4<float>(float(val), E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit fp4_e2m1(const unsigned long long int val)
      : __x(cvt_float_to_fp4<float>(float(val), E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit fp4_e2m1(const unsigned short int val)
      : __x(cvt_float_to_fp4<float>(float(val), E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ operator float() const {
    using namespace fcbx;
    floatx2_storage_t ret{
        to_float<float, Encoding::E2M1, true>((__x >> 2) & 0xFu, 0x7f),
        to_float<float, Encoding::E2M1, true>((__x >> 2) >> 4, 0x7f)};
    return ret[0];
  }

  __FP4_HOST_DEVICE__ operator double() const { return double(float(*this)); }

  __FP4_HOST_DEVICE__ operator float16_t() const {
    return float16_t(float(*this));
  }

  __FP4_HOST_DEVICE__ operator bfloat16_t() const {
    return bfloat16_t(float(*this));
  }

  __FP4_HOST_DEVICE__ bool isinf() const {
    return false; // e2m1 has no infinity
  }

  __FP4_HOST_DEVICE__ bool isnan() const {
    return false; // e2m1 has no NaN
  }

  __FP4_HOST_DEVICE__ bool iszero() const {
    return this->__x == 0x0 || this->__x == 0x80;
  }
};

using float4_t = fp4_e2m1;

inline std::ostream &operator<<(std::ostream &os, float4_t src) {
  os << float(src);
  return os;
}

// operator+
inline float operator+(float4_t a, float4_t b) { return float(a) + float(b); }
inline float operator+(float4_t a, float b) { return float(a) + b; }
inline float operator+(float a, float4_t b) { return a + float(b); }
inline float operator+(float4_t a, float16_t b) { return float(a) + float(b); }
inline float operator+(float16_t a, float4_t b) { return float(a) + float(b); }
inline float operator+(float4_t a, bfloat16_t b) { return float(a) + float(b); }
inline float operator+(bfloat16_t a, float4_t b) { return float(a) + float(b); }
inline float operator+(float4_t a, float8_t b) { return float(a) + float(b); }
inline float operator+(float8_t a, float4_t b) { return float(a) + float(b); }
inline float operator+(float4_t a, bfloat8_t b) { return float(a) + float(b); }
inline float operator+(bfloat8_t a, float4_t b) { return float(a) + float(b); }
inline float operator+(float4_t a, float6_t b) { return float(a) + float(b); }
inline float operator+(float6_t a, float4_t b) { return float(a) + float(b); }
inline float operator+(float4_t a, bfloat6_t b) { return float(a) + float(b); }
inline float operator+(bfloat6_t a, float4_t b) { return float(a) + float(b); }

// operator-
inline float operator-(float4_t a, float4_t b) { return float(a) - float(b); }
inline float operator-(float4_t a, float b) { return float(a) - b; }
inline float operator-(float a, float4_t b) { return a - float(b); }
inline float operator-(float4_t a, float16_t b) { return float(a) - float(b); }
inline float operator-(float16_t a, float4_t b) { return float(a) - float(b); }
inline float operator-(float4_t a, bfloat16_t b) { return float(a) - float(b); }
inline float operator-(bfloat16_t a, float4_t b) { return float(a) - float(b); }
inline float operator-(float4_t a, float8_t b) { return float(a) - float(b); }
inline float operator-(float8_t a, float4_t b) { return float(a) - float(b); }
inline float operator-(float4_t a, bfloat8_t b) { return float(a) - float(b); }
inline float operator-(bfloat8_t a, float4_t b) { return float(a) - float(b); }
inline float operator-(float4_t a, float6_t b) { return float(a) - float(b); }
inline float operator-(float6_t a, float4_t b) { return float(a) - float(b); }
inline float operator-(float4_t a, bfloat6_t b) { return float(a) - float(b); }
inline float operator-(bfloat6_t a, float4_t b) { return float(a) - float(b); }

// operator*
inline float operator*(float4_t a, float4_t b) { return float(a) * float(b); }
inline float operator*(float4_t a, float b) { return float(a) * b; }
inline float operator*(float a, float4_t b) { return a * float(b); }
inline float operator*(float4_t a, float16_t b) { return float(a) * float(b); }
inline float operator*(float16_t a, float4_t b) { return float(a) * float(b); }
inline float operator*(float4_t a, bfloat16_t b) { return float(a) * float(b); }
inline float operator*(bfloat16_t a, float4_t b) { return float(a) * float(b); }
inline float operator*(float4_t a, float8_t b) { return float(a) * float(b); }
inline float operator*(float8_t a, float4_t b) { return float(a) * float(b); }
inline float operator*(float4_t a, bfloat8_t b) { return float(a) * float(b); }
inline float operator*(bfloat8_t a, float4_t b) { return float(a) * float(b); }
inline float operator*(float4_t a, float6_t b) { return float(a) * float(b); }
inline float operator*(float6_t a, float4_t b) { return float(a) * float(b); }
inline float operator*(float4_t a, bfloat6_t b) { return float(a) * float(b); }
inline float operator*(bfloat6_t a, float4_t b) { return float(a) * float(b); }

// operator/
inline float operator/(float4_t a, float4_t b) { return float(a) / float(b); }
inline float operator/(float4_t a, float b) { return float(a) / b; }
inline float operator/(float a, float4_t b) { return a / float(b); }
inline float operator/(float4_t a, float16_t b) { return float(a) / float(b); }
inline float operator/(float16_t a, float4_t b) { return float(a) / float(b); }
inline float operator/(float4_t a, bfloat16_t b) { return float(a) / float(b); }
inline float operator/(bfloat16_t a, float4_t b) { return float(a) / float(b); }
inline float operator/(float4_t a, float8_t b) { return float(a) / float(b); }
inline float operator/(float8_t a, float4_t b) { return float(a) / float(b); }
inline float operator/(float4_t a, bfloat8_t b) { return float(a) / float(b); }
inline float operator/(bfloat8_t a, float4_t b) { return float(a) / float(b); }
inline float operator/(float4_t a, float6_t b) { return float(a) / float(b); }
inline float operator/(float6_t a, float4_t b) { return float(a) / float(b); }
inline float operator/(float4_t a, bfloat6_t b) { return float(a) / float(b); }
inline float operator/(bfloat6_t a, float4_t b) { return float(a) / float(b); }

// operator+=
inline float operator+=(float4_t a, float4_t b) { return float(a) + float(b); }
inline float operator+=(float4_t a, float b) { return float(a) + b; }
inline float operator+=(float a, float4_t b) { return a + float(b); }
inline float operator+=(float4_t a, float16_t b) { return float(a) + float(b); }
inline float operator+=(float16_t a, float4_t b) { return float(a) + float(b); }
inline float operator+=(float4_t a, bfloat16_t b) {
  return float(a) + float(b);
}
inline float operator+=(bfloat16_t a, float4_t b) {
  return float(a) + float(b);
}
inline float operator+=(float4_t a, float8_t b) { return float(a) + float(b); }
inline float operator+=(float8_t a, float4_t b) { return float(a) + float(b); }
inline float operator+=(float4_t a, bfloat8_t b) { return float(a) + float(b); }
inline float operator+=(bfloat8_t a, float4_t b) { return float(a) + float(b); }
inline float operator+=(float4_t a, float6_t b) { return float(a) + float(b); }
inline float operator+=(float6_t a, float4_t b) { return float(a) + float(b); }
inline float operator+=(float4_t a, bfloat6_t b) { return float(a) + float(b); }
inline float operator+=(bfloat6_t a, float4_t b) { return float(a) + float(b); }

// operator-=
inline float operator-=(float4_t a, float4_t b) { return float(a) - float(b); }
inline float operator-=(float4_t a, float b) { return float(a) - b; }
inline float operator-=(float a, float4_t b) { return a - float(b); }
inline float operator-=(float4_t a, float16_t b) { return float(a) - float(b); }
inline float operator-=(float16_t a, float4_t b) { return float(a) - float(b); }
inline float operator-=(float4_t a, bfloat16_t b) {
  return float(a) - float(b);
}
inline float operator-=(bfloat16_t a, float4_t b) {
  return float(a) - float(b);
}
inline float operator-=(float4_t a, float8_t b) { return float(a) - float(b); }
inline float operator-=(float8_t a, float4_t b) { return float(a) - float(b); }
inline float operator-=(float4_t a, bfloat8_t b) { return float(a) - float(b); }
inline float operator-=(bfloat8_t a, float4_t b) { return float(a) - float(b); }
inline float operator-=(float4_t a, float6_t b) { return float(a) - float(b); }
inline float operator-=(float6_t a, float4_t b) { return float(a) - float(b); }
inline float operator-=(float4_t a, bfloat6_t b) { return float(a) - float(b); }
inline float operator-=(bfloat6_t a, float4_t b) { return float(a) - float(b); }

// operator*=
inline float operator*=(float4_t a, float4_t b) { return float(a) * float(b); }
inline float operator*=(float4_t a, float b) { return float(a) * b; }
inline float operator*=(float a, float4_t b) { return a * float(b); }
inline float operator*=(float4_t a, float16_t b) { return float(a) * float(b); }
inline float operator*=(float16_t a, float4_t b) { return float(a) * float(b); }
inline float operator*=(float4_t a, bfloat16_t b) {
  return float(a) * float(b);
}
inline float operator*=(bfloat16_t a, float4_t b) {
  return float(a) * float(b);
}
inline float operator*=(float4_t a, float8_t b) { return float(a) * float(b); }
inline float operator*=(float8_t a, float4_t b) { return float(a) * float(b); }
inline float operator*=(float4_t a, bfloat8_t b) { return float(a) * float(b); }
inline float operator*=(bfloat8_t a, float4_t b) { return float(a) * float(b); }
inline float operator*=(float4_t a, float6_t b) { return float(a) * float(b); }
inline float operator*=(float6_t a, float4_t b) { return float(a) * float(b); }
inline float operator*=(float4_t a, bfloat6_t b) { return float(a) * float(b); }
inline float operator*=(bfloat6_t a, float4_t b) { return float(a) * float(b); }

// operator/=
inline float operator/=(float4_t a, float4_t b) { return float(a) / float(b); }
inline float operator/=(float4_t a, float b) { return float(a) / b; }
inline float operator/=(float a, float4_t b) { return a / float(b); }
inline float operator/=(float4_t a, float16_t b) { return float(a) / float(b); }
inline float operator/=(float16_t a, float4_t b) { return float(a) / float(b); }
inline float operator/=(float4_t a, bfloat16_t b) {
  return float(a) / float(b);
}
inline float operator/=(bfloat16_t a, float4_t b) {
  return float(a) / float(b);
}
inline float operator/=(float4_t a, float8_t b) { return float(a) / float(b); }
inline float operator/=(float8_t a, float4_t b) { return float(a) / float(b); }
inline float operator/=(float4_t a, bfloat8_t b) { return float(a) / float(b); }
inline float operator/=(bfloat8_t a, float4_t b) { return float(a) / float(b); }
inline float operator/=(float4_t a, float6_t b) { return float(a) / float(b); }
inline float operator/=(float6_t a, float4_t b) { return float(a) / float(b); }
inline float operator/=(float4_t a, bfloat6_t b) { return float(a) / float(b); }
inline float operator/=(bfloat6_t a, float4_t b) { return float(a) / float(b); }

// operator>
inline float operator>(float4_t a, float4_t b) { return float(a) > float(b); }
inline float operator>(float4_t a, float b) { return float(a) > b; }
inline float operator>(float a, float4_t b) { return a > float(b); }
inline float operator>(float4_t a, float16_t b) { return float(a) > float(b); }
inline float operator>(float16_t a, float4_t b) { return float(a) > float(b); }
inline float operator>(float4_t a, bfloat16_t b) { return float(a) > float(b); }
inline float operator>(bfloat16_t a, float4_t b) { return float(a) > float(b); }
inline float operator>(float4_t a, float8_t b) { return float(a) > float(b); }
inline float operator>(float8_t a, float4_t b) { return float(a) > float(b); }
inline float operator>(float4_t a, bfloat8_t b) { return float(a) > float(b); }
inline float operator>(bfloat8_t a, float4_t b) { return float(a) > float(b); }
inline float operator>(float4_t a, float6_t b) { return float(a) > float(b); }
inline float operator>(float6_t a, float4_t b) { return float(a) > float(b); }
inline float operator>(float4_t a, bfloat6_t b) { return float(a) > float(b); }
inline float operator>(bfloat6_t a, float4_t b) { return float(a) > float(b); }

// operator>=
inline float operator>=(float4_t a, float4_t b) { return float(a) >= float(b); }
inline float operator>=(float4_t a, float b) { return float(a) >= b; }
inline float operator>=(float a, float4_t b) { return a >= float(b); }
inline float operator>=(float4_t a, float16_t b) {
  return float(a) >= float(b);
}
inline float operator>=(float16_t a, float4_t b) {
  return float(a) >= float(b);
}
inline float operator>=(float4_t a, bfloat16_t b) {
  return float(a) >= float(b);
}
inline float operator>=(bfloat16_t a, float4_t b) {
  return float(a) >= float(b);
}
inline float operator>=(float4_t a, float8_t b) { return float(a) >= float(b); }
inline float operator>=(float8_t a, float4_t b) { return float(a) >= float(b); }
inline float operator>=(float4_t a, bfloat8_t b) {
  return float(a) >= float(b);
}
inline float operator>=(bfloat8_t a, float4_t b) {
  return float(a) >= float(b);
}
inline float operator>=(float4_t a, float6_t b) { return float(a) >= float(b); }
inline float operator>=(float6_t a, float4_t b) { return float(a) >= float(b); }
inline float operator>=(float4_t a, bfloat6_t b) {
  return float(a) >= float(b);
}
inline float operator>=(bfloat6_t a, float4_t b) {
  return float(a) >= float(b);
}

// operator<
inline float operator<(float4_t a, float4_t b) { return float(a) < float(b); }
inline float operator<(float4_t a, float b) { return float(a) < b; }
inline float operator<(float a, float4_t b) { return a < float(b); }
inline float operator<(float4_t a, float16_t b) { return float(a) < float(b); }
inline float operator<(float16_t a, float4_t b) { return float(a) < float(b); }
inline float operator<(float4_t a, bfloat16_t b) { return float(a) < float(b); }
inline float operator<(bfloat16_t a, float4_t b) { return float(a) < float(b); }
inline float operator<(float4_t a, float8_t b) { return float(a) < float(b); }
inline float operator<(float8_t a, float4_t b) { return float(a) < float(b); }
inline float operator<(float4_t a, bfloat8_t b) { return float(a) < float(b); }
inline float operator<(bfloat8_t a, float4_t b) { return float(a) < float(b); }
inline float operator<(float4_t a, float6_t b) { return float(a) < float(b); }
inline float operator<(float6_t a, float4_t b) { return float(a) < float(b); }
inline float operator<(float4_t a, bfloat6_t b) { return float(a) < float(b); }
inline float operator<(bfloat6_t a, float4_t b) { return float(a) < float(b); }

// operator<=
inline float operator<=(float4_t a, float4_t b) { return float(a) <= float(b); }
inline float operator<=(float4_t a, float b) { return float(a) <= b; }
inline float operator<=(float a, float4_t b) { return a <= float(b); }
inline float operator<=(float4_t a, float16_t b) {
  return float(a) <= float(b);
}
inline float operator<=(float16_t a, float4_t b) {
  return float(a) <= float(b);
}
inline float operator<=(float4_t a, bfloat16_t b) {
  return float(a) <= float(b);
}
inline float operator<=(bfloat16_t a, float4_t b) {
  return float(a) <= float(b);
}
inline float operator<=(float4_t a, float8_t b) { return float(a) <= float(b); }
inline float operator<=(float8_t a, float4_t b) { return float(a) <= float(b); }
inline float operator<=(float4_t a, bfloat8_t b) {
  return float(a) <= float(b);
}
inline float operator<=(bfloat8_t a, float4_t b) {
  return float(a) <= float(b);
}
inline float operator<=(float4_t a, float6_t b) { return float(a) <= float(b); }
inline float operator<=(float6_t a, float4_t b) { return float(a) <= float(b); }
inline float operator<=(float4_t a, bfloat6_t b) {
  return float(a) <= float(b);
}
inline float operator<=(bfloat6_t a, float4_t b) {
  return float(a) <= float(b);
}

// operator!=
inline float operator!=(float4_t a, float4_t b) { return float(a) != float(b); }
inline float operator!=(float4_t a, float b) { return float(a) != b; }
inline float operator!=(float a, float4_t b) { return a != float(b); }
inline float operator!=(float4_t a, float16_t b) {
  return float(a) != float(b);
}
inline float operator!=(float16_t a, float4_t b) {
  return float(a) != float(b);
}
inline float operator!=(float4_t a, bfloat16_t b) {
  return float(a) != float(b);
}
inline float operator!=(bfloat16_t a, float4_t b) {
  return float(a) != float(b);
}
inline float operator!=(float4_t a, float8_t b) { return float(a) != float(b); }
inline float operator!=(float8_t a, float4_t b) { return float(a) != float(b); }
inline float operator!=(float4_t a, bfloat8_t b) {
  return float(a) != float(b);
}
inline float operator!=(bfloat8_t a, float4_t b) {
  return float(a) != float(b);
}
inline float operator!=(float4_t a, float6_t b) { return float(a) != float(b); }
inline float operator!=(float6_t a, float4_t b) { return float(a) != float(b); }
inline float operator!=(float4_t a, bfloat6_t b) {
  return float(a) != float(b);
}
inline float operator!=(bfloat6_t a, float4_t b) {
  return float(a) != float(b);
}

// operator==
inline float operator==(float4_t a, float4_t b) { return float(a) == float(b); }
inline float operator==(float4_t a, float b) { return float(a) == b; }
inline float operator==(float a, float4_t b) { return a == float(b); }
inline float operator==(float4_t a, float16_t b) {
  return float(a) == float(b);
}
inline float operator==(float16_t a, float4_t b) {
  return float(a) == float(b);
}
inline float operator==(float4_t a, bfloat16_t b) {
  return float(a) == float(b);
}
inline float operator==(bfloat16_t a, float4_t b) {
  return float(a) == float(b);
}
inline float operator==(float4_t a, float8_t b) { return float(a) == float(b); }
inline float operator==(float8_t a, float4_t b) { return float(a) == float(b); }
inline float operator==(float4_t a, bfloat8_t b) {
  return float(a) == float(b);
}
inline float operator==(bfloat8_t a, float4_t b) {
  return float(a) == float(b);
}
inline float operator==(float4_t a, float6_t b) { return float(a) == float(b); }
inline float operator==(float6_t a, float4_t b) { return float(a) == float(b); }
inline float operator==(float4_t a, bfloat6_t b) {
  return float(a) == float(b);
}
inline float operator==(bfloat6_t a, float4_t b) {
  return float(a) == float(b);
}

// operator++
inline float operator++(float4_t a) { return float(a) + float(1); }

// operator--
inline float operator--(float4_t a) { return float(a) - float(1); }

__FP6_HOST_DEVICE_STATIC__ void condense_fp4_to_b4(void *fp4, void *b4, int fp4_size, int b4_size) {
  assert(fp4_size % 2 == 0 && fp4_size / 2 == b4_size);
  // condense each 2 fp4 into 1 b8
  for (size_t i = 0; i < fp4_size; i += 2) {
    uint8_t a = *((uint8_t*)fp4 + i);
    uint8_t b = *((uint8_t*)fp4 + i + 1);
    *((uint8_t*)b4 + i / 2) = a >> 2 | b << 2;
  }
}

struct fp4x2_e2m1 {
  fp4x2_storage_t __x;

public:
  __FP4_HOST_DEVICE__ fp4x2_e2m1() = default;

  __FP4_HOST_DEVICE__
  explicit fp4x2_e2m1(const float4_t a, const float4_t b) {
    __x = fp4x2_storage_t(a.__x >> 2 | b.__x << 2);
  }

  __FP4_HOST_DEVICE__
  explicit fp4x2_e2m1(const float a, const float b) {
    __x = fp4x2_storage_t(float4_t(a).__x >> 2 | float4_t(b).__x << 2);
  }

  __FP4_HOST_DEVICE__
  explicit fp4x2_e2m1(void *data) { condense_fp4_to_b4(data, &__x, 2, 1); }

  __FP4_HOST_DEVICE__ explicit fp4x2_e2m1(const uint8_t val) : __x(val) {}

  __FP4_HOST_DEVICE__ explicit fp4x2_e2m1(const int val) : __x(uint8_t(val)) {}

  __FP4_HOST_DEVICE__ float4_t const operator[](uint32_t i) {
    float4_t res;
    res.__x = (__x >> (i % 2 * 4) & 0x0f) << 2;
    return res;
  }
};

using float4x2_t = fp4x2_e2m1;

inline std::ostream &operator<<(std::ostream &os, const float4x2_t &lhs) {
  return os << (uint32_t)lhs.__x;
}

inline float operator*(float4x2_t a, float4x2_t b) {
  return a[0] * b[0] + a[1] * b[1];
}

inline float operator==(float4x2_t a, float4x2_t b) { return a.__x == b.__x; }

// TODO: If necessary, add other operator overloads for float4x2_t.