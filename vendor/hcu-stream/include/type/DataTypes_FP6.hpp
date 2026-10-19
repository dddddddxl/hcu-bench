// On DCU, the 8-bit storage format of fp6/bf6 is: {2'b0, fp6/bf6}

#pragma once

#include "type/DataTypes_FP8.hpp"

#include <vector>

#define __FP6_HOST_DEVICE__ __host__ __device__
#define __FP6_HOST_DEVICE_STATIC__ __FP6_HOST_DEVICE__ static inline

typedef fp8_storage_t fp6_storage_t;

static_assert(sizeof(fp6_storage_t[4]) == sizeof(uint32_t));

enum fp6_interpretation_t {
  E3M2 = 0, /**< FP6 E3M2 Type*/
  E2M3 = 1, /**< FP6 E2M3 Type */
};

__FP6_HOST_DEVICE_STATIC__ fp6_storage_t cvt_double_to_fp6(
    const double x, const fp6_interpretation_t fp6_interpretation_t,
    const enum hipRoundMode /* rounding */) {
  union {
    uint32_t ui32;
    fp6_storage_t fp6[4];
  } u{0};

  if (fp6_interpretation_t == E2M3) {
    u.ui32 = fcbx::from_float<float, fcbx::Encoding::E2M3, true>(float(x), 0x7f);
  } else if (fp6_interpretation_t == E3M2) {
    u.ui32 = fcbx::from_float<float, fcbx::Encoding::E3M2, true>(float(x), 0x7f);
  }

  return u.fp6[0];
}

template <typename T>
__FP6_HOST_DEVICE_STATIC__ fp6_storage_t
cvt_float_to_fp6(const T x, const fp6_interpretation_t fp6_interpretation_t,
                 const enum hipRoundMode  rounding,
                 uint8_t scale = 0x7f) {
  union {
    uint32_t ui32;
    fp6_storage_t fp6[4];
  } u{0};

  if (fp6_interpretation_t == E2M3)
    u.ui32 = fcbx::from_float<T, fcbx::Encoding::E2M3, true>(x, 0x7f, rounding);
  else if (fp6_interpretation_t == E3M2)
    u.ui32 = fcbx::from_float<T, fcbx::Encoding::E3M2, true>(x, 0x7f, rounding);

  return u.fp6[0];
}

template <typename T>
__FP6_HOST_DEVICE_STATIC__ fp6_storage_t
cvt_float_to_fp6_sr(T x, const fp6_interpretation_t fp6_interpretation_t, uint32_t seed, uint8_t scale = 0) {
  union {
    uint32_t ui32;
    fp6_storage_t fp6[4];
  } u{0};

  if (fp6_interpretation_t == E2M3)
    u.ui32 = fcbx::from_float_sr<T, fcbx::Encoding::E2M3, true>(x, seed, scale);
  else if (fp6_interpretation_t == E3M2)
    u.ui32 = fcbx::from_float_sr<T, fcbx::Encoding::E3M2, true>(x, seed, scale);

  return u.fp6[0];
}

struct fp6_e2m3 {
  fp6_storage_t __x;

  __FP6_HOST_DEVICE__ fp6_e2m3() = default;
  __FP6_HOST_DEVICE__ inline explicit fp6_e2m3(const double f)
      : __x(cvt_double_to_fp6(f, E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit fp6_e2m3(const float f)
      : __x(cvt_float_to_fp6<float>(f, E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit fp6_e2m3(const int val)
      : __x(cvt_float_to_fp6<float>(float(val), E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit fp6_e2m3(const long int val)
      : __x(cvt_float_to_fp6<float>(float(val), E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit fp6_e2m3(const long long int val)
      : __x(cvt_float_to_fp6<float>(float(val), E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit fp6_e2m3(const short int val)
      : __x(cvt_float_to_fp6<float>(float(val), E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit fp6_e2m3(const unsigned int val)
      : __x(cvt_float_to_fp6<float>(float(val), E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit fp6_e2m3(const unsigned long int val)
      : __x(cvt_float_to_fp6<float>(float(val), E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit fp6_e2m3(const unsigned long long int val)
      : __x(cvt_float_to_fp6<float>(float(val), E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit fp6_e2m3(const unsigned short int val)
      : __x(cvt_float_to_fp6<float>(float(val), E2M3, hipRoundNearest)) {}

  __FP6_HOST_DEVICE__ operator float() const {
    using namespace fcbx;
    float ret{to_float<float, Encoding::E2M3, true>(__x, 0x7f)};
    return ret;
  }

  __FP6_HOST_DEVICE__ operator double() const { return double(float(*this)); }

  __FP6_HOST_DEVICE__ operator float16_t() const {
    return float16_t(float(*this));
  }

  __FP6_HOST_DEVICE__ operator bfloat16_t() const {
    return bfloat16_t(float(*this));
  }

  __FP6_HOST_DEVICE__ bool isinf() const {
    return false; // e2m3 has no infinity
  }

  __FP6_HOST_DEVICE__ bool isnan() const {
    return false; // e2m3 has no NaN
  }

  __FP6_HOST_DEVICE__ bool iszero() const {
    return this->__x == 0x0 || this->__x == 0x80;
  }
};

using float6_t = fp6_e2m3;

inline std::ostream &operator<<(std::ostream &os, float6_t src) {
  os << float(src);
  return os;
}

// operator+
inline float operator+(float6_t a, float6_t b) { return float(a) + float(b); }
inline float operator+(float6_t a, float b) { return float(a) + b; }
inline float operator+(float a, float6_t b) { return a + float(b); }
inline float operator+(float6_t a, float16_t b) { return float(a) + float(b); }
inline float operator+(float16_t a, float6_t b) { return float(a) + float(b); }
inline float operator+(float6_t a, bfloat16_t b) { return float(a) + float(b); }
inline float operator+(bfloat16_t a, float6_t b) { return float(a) + float(b); }
inline float operator+(float6_t a, float8_t b) { return float(a) + float(b); }
inline float operator+(float8_t a, float6_t b) { return float(a) + float(b); }
inline float operator+(float6_t a, bfloat8_t b) { return float(a) + float(b); }
inline float operator+(bfloat8_t a, float6_t b) { return float(a) + float(b); }

// operator-
inline float operator-(float6_t a, float6_t b) { return float(a) - float(b); }
inline float operator-(float6_t a, float b) { return float(a) - b; }
inline float operator-(float a, float6_t b) { return a - float(b); }
inline float operator-(float6_t a, float16_t b) { return float(a) - float(b); }
inline float operator-(float16_t a, float6_t b) { return float(a) - float(b); }
inline float operator-(float6_t a, bfloat16_t b) { return float(a) - float(b); }
inline float operator-(bfloat16_t a, float6_t b) { return float(a) - float(b); }
inline float operator-(float6_t a, float8_t b) { return float(a) - float(b); }
inline float operator-(float8_t a, float6_t b) { return float(a) - float(b); }
inline float operator-(float6_t a, bfloat8_t b) { return float(a) - float(b); }
inline float operator-(bfloat8_t a, float6_t b) { return float(a) - float(b); }

// operator*
inline float operator*(float6_t a, float6_t b) { return float(a) * float(b); }
inline float operator*(float6_t a, float b) { return float(a) * b; }
inline float operator*(float a, float6_t b) { return a * float(b); }
inline float operator*(float6_t a, float16_t b) { return float(a) * float(b); }
inline float operator*(float16_t a, float6_t b) { return float(a) * float(b); }
inline float operator*(float6_t a, bfloat16_t b) { return float(a) * float(b); }
inline float operator*(bfloat16_t a, float6_t b) { return float(a) * float(b); }
inline float operator*(float6_t a, float8_t b) { return float(a) * float(b); }
inline float operator*(float8_t a, float6_t b) { return float(a) * float(b); }
inline float operator*(float6_t a, bfloat8_t b) { return float(a) * float(b); }
inline float operator*(bfloat8_t a, float6_t b) { return float(a) * float(b); }

// operator/
inline float operator/(float6_t a, float6_t b) { return float(a) / float(b); }
inline float operator/(float6_t a, float b) { return float(a) / b; }
inline float operator/(float a, float6_t b) { return a / float(b); }
inline float operator/(float6_t a, float16_t b) { return float(a) / float(b); }
inline float operator/(float16_t a, float6_t b) { return float(a) / float(b); }
inline float operator/(float6_t a, bfloat16_t b) { return float(a) / float(b); }
inline float operator/(bfloat16_t a, float6_t b) { return float(a) / float(b); }
inline float operator/(float6_t a, float8_t b) { return float(a) / float(b); }
inline float operator/(float8_t a, float6_t b) { return float(a) / float(b); }
inline float operator/(float6_t a, bfloat8_t b) { return float(a) / float(b); }
inline float operator/(bfloat8_t a, float6_t b) { return float(a) / float(b); }

// operator+=
inline float operator+=(float6_t a, float6_t b) { return float(a) + float(b); }
inline float operator+=(float6_t a, float b) { return float(a) + b; }
inline float operator+=(float a, float6_t b) { return a + float(b); }
inline float operator+=(float6_t a, float16_t b) { return float(a) + float(b); }
inline float operator+=(float16_t a, float6_t b) { return float(a) + float(b); }
inline float operator+=(float6_t a, bfloat16_t b) {
  return float(a) + float(b);
}
inline float operator+=(bfloat16_t a, float6_t b) {
  return float(a) + float(b);
}
inline float operator+=(float6_t a, float8_t b) { return float(a) + float(b); }
inline float operator+=(float8_t a, float6_t b) { return float(a) + float(b); }
inline float operator+=(float6_t a, bfloat8_t b) { return float(a) + float(b); }
inline float operator+=(bfloat8_t a, float6_t b) { return float(a) + float(b); }

// operator-=
inline float operator-=(float6_t a, float6_t b) { return float(a) - float(b); }
inline float operator-=(float6_t a, float b) { return float(a) - b; }
inline float operator-=(float a, float6_t b) { return a - float(b); }
inline float operator-=(float6_t a, float16_t b) { return float(a) - float(b); }
inline float operator-=(float16_t a, float6_t b) { return float(a) - float(b); }
inline float operator-=(float6_t a, bfloat16_t b) {
  return float(a) - float(b);
}
inline float operator-=(bfloat16_t a, float6_t b) {
  return float(a) - float(b);
}
inline float operator-=(float6_t a, float8_t b) { return float(a) - float(b); }
inline float operator-=(float8_t a, float6_t b) { return float(a) - float(b); }
inline float operator-=(float6_t a, bfloat8_t b) { return float(a) - float(b); }
inline float operator-=(bfloat8_t a, float6_t b) { return float(a) - float(b); }

// operator*=
inline float operator*=(float6_t a, float6_t b) { return float(a) * float(b); }
inline float operator*=(float6_t a, float b) { return float(a) * b; }
inline float operator*=(float a, float6_t b) { return a * float(b); }
inline float operator*=(float6_t a, float16_t b) { return float(a) * float(b); }
inline float operator*=(float16_t a, float6_t b) { return float(a) * float(b); }
inline float operator*=(float6_t a, bfloat16_t b) {
  return float(a) * float(b);
}
inline float operator*=(bfloat16_t a, float6_t b) {
  return float(a) * float(b);
}
inline float operator*=(float6_t a, float8_t b) { return float(a) * float(b); }
inline float operator*=(float8_t a, float6_t b) { return float(a) * float(b); }
inline float operator*=(float6_t a, bfloat8_t b) { return float(a) * float(b); }
inline float operator*=(bfloat8_t a, float6_t b) { return float(a) * float(b); }

// operator/=
inline float operator/=(float6_t a, float6_t b) { return float(a) / float(b); }
inline float operator/=(float6_t a, float b) { return float(a) / b; }
inline float operator/=(float a, float6_t b) { return a / float(b); }
inline float operator/=(float6_t a, float16_t b) { return float(a) / float(b); }
inline float operator/=(float16_t a, float6_t b) { return float(a) / float(b); }
inline float operator/=(float6_t a, bfloat16_t b) {
  return float(a) / float(b);
}
inline float operator/=(bfloat16_t a, float6_t b) {
  return float(a) / float(b);
}
inline float operator/=(float6_t a, float8_t b) { return float(a) / float(b); }
inline float operator/=(float8_t a, float6_t b) { return float(a) / float(b); }
inline float operator/=(float6_t a, bfloat8_t b) { return float(a) / float(b); }
inline float operator/=(bfloat8_t a, float6_t b) { return float(a) / float(b); }

// operator>
inline float operator>(float6_t a, float6_t b) { return float(a) > float(b); }
inline float operator>(float6_t a, float b) { return float(a) > b; }
inline float operator>(float a, float6_t b) { return a > float(b); }
inline float operator>(float6_t a, float16_t b) { return float(a) > float(b); }
inline float operator>(float16_t a, float6_t b) { return float(a) > float(b); }
inline float operator>(float6_t a, bfloat16_t b) { return float(a) > float(b); }
inline float operator>(bfloat16_t a, float6_t b) { return float(a) > float(b); }
inline float operator>(float6_t a, float8_t b) { return float(a) > float(b); }
inline float operator>(float8_t a, float6_t b) { return float(a) > float(b); }
inline float operator>(float6_t a, bfloat8_t b) { return float(a) > float(b); }
inline float operator>(bfloat8_t a, float6_t b) { return float(a) > float(b); }

// operator>=
inline float operator>=(float6_t a, float6_t b) { return float(a) >= float(b); }
inline float operator>=(float6_t a, float b) { return float(a) >= b; }
inline float operator>=(float a, float6_t b) { return a >= float(b); }
inline float operator>=(float6_t a, float16_t b) {
  return float(a) >= float(b);
}
inline float operator>=(float16_t a, float6_t b) {
  return float(a) >= float(b);
}
inline float operator>=(float6_t a, bfloat16_t b) {
  return float(a) >= float(b);
}
inline float operator>=(bfloat16_t a, float6_t b) {
  return float(a) >= float(b);
}
inline float operator>=(float6_t a, float8_t b) { return float(a) >= float(b); }
inline float operator>=(float8_t a, float6_t b) { return float(a) >= float(b); }
inline float operator>=(float6_t a, bfloat8_t b) {
  return float(a) >= float(b);
}
inline float operator>=(bfloat8_t a, float6_t b) {
  return float(a) >= float(b);
}

// operator<
inline float operator<(float6_t a, float6_t b) { return float(a) < float(b); }
inline float operator<(float6_t a, float b) { return float(a) < b; }
inline float operator<(float a, float6_t b) { return a < float(b); }
inline float operator<(float6_t a, float16_t b) { return float(a) < float(b); }
inline float operator<(float16_t a, float6_t b) { return float(a) < float(b); }
inline float operator<(float6_t a, bfloat16_t b) { return float(a) < float(b); }
inline float operator<(bfloat16_t a, float6_t b) { return float(a) < float(b); }
inline float operator<(float6_t a, float8_t b) { return float(a) < float(b); }
inline float operator<(float8_t a, float6_t b) { return float(a) < float(b); }
inline float operator<(float6_t a, bfloat8_t b) { return float(a) < float(b); }
inline float operator<(bfloat8_t a, float6_t b) { return float(a) < float(b); }

// operator<=
inline float operator<=(float6_t a, float6_t b) { return float(a) <= float(b); }
inline float operator<=(float6_t a, float b) { return float(a) <= b; }
inline float operator<=(float a, float6_t b) { return a <= float(b); }
inline float operator<=(float6_t a, float16_t b) {
  return float(a) <= float(b);
}
inline float operator<=(float16_t a, float6_t b) {
  return float(a) <= float(b);
}
inline float operator<=(float6_t a, bfloat16_t b) {
  return float(a) <= float(b);
}
inline float operator<=(bfloat16_t a, float6_t b) {
  return float(a) <= float(b);
}
inline float operator<=(float6_t a, float8_t b) { return float(a) <= float(b); }
inline float operator<=(float8_t a, float6_t b) { return float(a) <= float(b); }
inline float operator<=(float6_t a, bfloat8_t b) {
  return float(a) <= float(b);
}
inline float operator<=(bfloat8_t a, float6_t b) {
  return float(a) <= float(b);
}

// operator!=
inline float operator!=(float6_t a, float6_t b) { return float(a) != float(b); }
inline float operator!=(float6_t a, float b) { return float(a) != b; }
inline float operator!=(float a, float6_t b) { return a != float(b); }
inline float operator!=(float6_t a, float16_t b) {
  return float(a) != float(b);
}
inline float operator!=(float16_t a, float6_t b) {
  return float(a) != float(b);
}
inline float operator!=(float6_t a, bfloat16_t b) {
  return float(a) != float(b);
}
inline float operator!=(bfloat16_t a, float6_t b) {
  return float(a) != float(b);
}
inline float operator!=(float6_t a, float8_t b) { return float(a) != float(b); }
inline float operator!=(float8_t a, float6_t b) { return float(a) != float(b); }
inline float operator!=(float6_t a, bfloat8_t b) {
  return float(a) != float(b);
}
inline float operator!=(bfloat8_t a, float6_t b) {
  return float(a) != float(b);
}

// operator==
inline float operator==(float6_t a, float6_t b) { return float(a) == float(b); }
inline float operator==(float6_t a, float b) { return float(a) == b; }
inline float operator==(float a, float6_t b) { return a == float(b); }
inline float operator==(float6_t a, float16_t b) {
  return float(a) == float(b);
}
inline float operator==(float16_t a, float6_t b) {
  return float(a) == float(b);
}
inline float operator==(float6_t a, bfloat16_t b) {
  return float(a) == float(b);
}
inline float operator==(bfloat16_t a, float6_t b) {
  return float(a) == float(b);
}
inline float operator==(float6_t a, float8_t b) { return float(a) == float(b); }
inline float operator==(float8_t a, float6_t b) { return float(a) == float(b); }
inline float operator==(float6_t a, bfloat8_t b) {
  return float(a) == float(b);
}
inline float operator==(bfloat8_t a, float6_t b) {
  return float(a) == float(b);
}

// operator++
inline float operator++(float6_t a) { return float(a) + float(1); }

// operator--
inline float operator--(float6_t a) { return float(a) - float(1); }

struct fp6_e3m2 {
  fp6_storage_t __x;

  __FP6_HOST_DEVICE__ fp6_e3m2() = default;
  __FP6_HOST_DEVICE__ inline explicit fp6_e3m2(const double f)
      : __x(cvt_double_to_fp6(f, E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit fp6_e3m2(const float f)
      : __x(cvt_float_to_fp6<float>(f, E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit fp6_e3m2(const int val)
      : __x(cvt_float_to_fp6<float>(float(val), E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit fp6_e3m2(const long int val)
      : __x(cvt_float_to_fp6<float>(float(val), E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit fp6_e3m2(const long long int val)
      : __x(cvt_float_to_fp6<float>(float(val), E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit fp6_e3m2(const short int val)
      : __x(cvt_float_to_fp6<float>(float(val), E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit fp6_e3m2(const unsigned int val)
      : __x(cvt_float_to_fp6<float>(float(val), E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit fp6_e3m2(const unsigned long int val)
      : __x(cvt_float_to_fp6<float>(float(val), E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit fp6_e3m2(const unsigned long long int val)
      : __x(cvt_float_to_fp6<float>(float(val), E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit fp6_e3m2(const unsigned short int val)
      : __x(cvt_float_to_fp6<float>(float(val), E3M2, hipRoundNearest)) {}

  __FP6_HOST_DEVICE__ operator float() const {
    using namespace fcbx;
    float ret{to_float<float, Encoding::E3M2, true>(__x, 0x7f)};
    return ret;
  }

  __FP6_HOST_DEVICE__ operator double() const { return double(float(*this)); }

  __FP6_HOST_DEVICE__ operator float16_t() const {
    return float16_t(float(*this));
  }

  __FP6_HOST_DEVICE__ operator bfloat16_t() const {
    return bfloat16_t(float(*this));
  }

  __FP6_HOST_DEVICE__ bool isinf() const {
    return false; // e3m2 has no infinity
  }

  __FP6_HOST_DEVICE__ bool isnan() const {
    return false; // e3m2 has no NaN
  }

  __FP6_HOST_DEVICE__ bool iszero() const {
    return this->__x == 0x0 || this->__x == 0x80;
  }
};

using bfloat6_t = fp6_e3m2;

inline std::ostream &operator<<(std::ostream &os, bfloat6_t src) {
  os << float(src);
  return os;
}

// operator+
inline float operator+(bfloat6_t a, bfloat6_t b) { return float(a) + float(b); }
inline float operator+(bfloat6_t a, float b) { return float(a) + b; }
inline float operator+(float a, bfloat6_t b) { return a + float(b); }
inline float operator+(bfloat6_t a, float16_t b) { return float(a) + float(b); }
inline float operator+(float16_t a, bfloat6_t b) { return float(a) + float(b); }
inline float operator+(bfloat6_t a, bfloat16_t b) {
  return float(a) + float(b);
}
inline float operator+(bfloat16_t a, bfloat6_t b) {
  return float(a) + float(b);
}
inline float operator+(bfloat6_t a, float8_t b) { return float(a) + float(b); }
inline float operator+(float8_t a, bfloat6_t b) { return float(a) + float(b); }
inline float operator+(bfloat6_t a, bfloat8_t b) { return float(a) + float(b); }
inline float operator+(bfloat8_t a, bfloat6_t b) { return float(a) + float(b); }
inline float operator+(bfloat6_t a, float6_t b) { return float(a) + float(b); }
inline float operator+(float6_t a, bfloat6_t b) { return float(a) + float(b); }

// operator-
inline float operator-(bfloat6_t a, bfloat6_t b) { return float(a) - float(b); }
inline float operator-(bfloat6_t a, float b) { return float(a) - b; }
inline float operator-(float a, bfloat6_t b) { return a - float(b); }
inline float operator-(bfloat6_t a, float16_t b) { return float(a) - float(b); }
inline float operator-(float16_t a, bfloat6_t b) { return float(a) - float(b); }
inline float operator-(bfloat6_t a, bfloat16_t b) {
  return float(a) - float(b);
}
inline float operator-(bfloat16_t a, bfloat6_t b) {
  return float(a) - float(b);
}
inline float operator-(bfloat6_t a, float8_t b) { return float(a) - float(b); }
inline float operator-(float8_t a, bfloat6_t b) { return float(a) - float(b); }
inline float operator-(bfloat6_t a, bfloat8_t b) { return float(a) - float(b); }
inline float operator-(bfloat8_t a, bfloat6_t b) { return float(a) - float(b); }
inline float operator-(bfloat6_t a, float6_t b) { return float(a) - float(b); }
inline float operator-(float6_t a, bfloat6_t b) { return float(a) - float(b); }

// operator*
inline float operator*(bfloat6_t a, bfloat6_t b) { return float(a) * float(b); }
inline float operator*(bfloat6_t a, float b) { return float(a) * b; }
inline float operator*(float a, bfloat6_t b) { return a * float(b); }
inline float operator*(bfloat6_t a, float16_t b) { return float(a) * float(b); }
inline float operator*(float16_t a, bfloat6_t b) { return float(a) * float(b); }
inline float operator*(bfloat6_t a, bfloat16_t b) {
  return float(a) * float(b);
}
inline float operator*(bfloat16_t a, bfloat6_t b) {
  return float(a) * float(b);
}
inline float operator*(bfloat6_t a, float8_t b) { return float(a) * float(b); }
inline float operator*(float8_t a, bfloat6_t b) { return float(a) * float(b); }
inline float operator*(bfloat6_t a, bfloat8_t b) { return float(a) * float(b); }
inline float operator*(bfloat8_t a, bfloat6_t b) { return float(a) * float(b); }
inline float operator*(bfloat6_t a, float6_t b) { return float(a) * float(b); }
inline float operator*(float6_t a, bfloat6_t b) { return float(a) * float(b); }

// operator/
inline float operator/(bfloat6_t a, bfloat6_t b) { return float(a) / float(b); }
inline float operator/(bfloat6_t a, float b) { return float(a) / b; }
inline float operator/(float a, bfloat6_t b) { return a / float(b); }
inline float operator/(bfloat6_t a, float16_t b) { return float(a) / float(b); }
inline float operator/(float16_t a, bfloat6_t b) { return float(a) / float(b); }
inline float operator/(bfloat6_t a, bfloat16_t b) {
  return float(a) / float(b);
}
inline float operator/(bfloat16_t a, bfloat6_t b) {
  return float(a) / float(b);
}
inline float operator/(bfloat6_t a, float8_t b) { return float(a) / float(b); }
inline float operator/(float8_t a, bfloat6_t b) { return float(a) / float(b); }
inline float operator/(bfloat6_t a, bfloat8_t b) { return float(a) / float(b); }
inline float operator/(bfloat8_t a, bfloat6_t b) { return float(a) / float(b); }
inline float operator/(bfloat6_t a, float6_t b) { return float(a) / float(b); }
inline float operator/(float6_t a, bfloat6_t b) { return float(a) / float(b); }

// operator+=
inline float operator+=(bfloat6_t a, bfloat6_t b) {
  return float(a) + float(b);
}
inline float operator+=(bfloat6_t a, float b) { return float(a) + b; }
inline float operator+=(float a, bfloat6_t b) { return a + float(b); }
inline float operator+=(bfloat6_t a, float16_t b) {
  return float(a) + float(b);
}
inline float operator+=(float16_t a, bfloat6_t b) {
  return float(a) + float(b);
}
inline float operator+=(bfloat6_t a, bfloat16_t b) {
  return float(a) + float(b);
}
inline float operator+=(bfloat16_t a, bfloat6_t b) {
  return float(a) + float(b);
}
inline float operator+=(bfloat6_t a, float8_t b) { return float(a) + float(b); }
inline float operator+=(float8_t a, bfloat6_t b) { return float(a) + float(b); }
inline float operator+=(bfloat6_t a, bfloat8_t b) {
  return float(a) + float(b);
}
inline float operator+=(bfloat8_t a, bfloat6_t b) {
  return float(a) + float(b);
}
inline float operator+=(bfloat6_t a, float6_t b) { return float(a) + float(b); }
inline float operator+=(float6_t a, bfloat6_t b) { return float(a) + float(b); }

// operator-=
inline float operator-=(bfloat6_t a, bfloat6_t b) {
  return float(a) - float(b);
}
inline float operator-=(bfloat6_t a, float b) { return float(a) - b; }
inline float operator-=(float a, bfloat6_t b) { return a - float(b); }
inline float operator-=(bfloat6_t a, float16_t b) {
  return float(a) - float(b);
}
inline float operator-=(float16_t a, bfloat6_t b) {
  return float(a) - float(b);
}
inline float operator-=(bfloat6_t a, bfloat16_t b) {
  return float(a) - float(b);
}
inline float operator-=(bfloat16_t a, bfloat6_t b) {
  return float(a) - float(b);
}
inline float operator-=(bfloat6_t a, float8_t b) { return float(a) - float(b); }
inline float operator-=(float8_t a, bfloat6_t b) { return float(a) - float(b); }
inline float operator-=(bfloat6_t a, bfloat8_t b) {
  return float(a) - float(b);
}
inline float operator-=(bfloat8_t a, bfloat6_t b) {
  return float(a) - float(b);
}
inline float operator-=(bfloat6_t a, float6_t b) { return float(a) - float(b); }
inline float operator-=(float6_t a, bfloat6_t b) { return float(a) - float(b); }

// operator*=
inline float operator*=(bfloat6_t a, bfloat6_t b) {
  return float(a) * float(b);
}
inline float operator*=(bfloat6_t a, float b) { return float(a) * b; }
inline float operator*=(float a, bfloat6_t b) { return a * float(b); }
inline float operator*=(bfloat6_t a, float16_t b) {
  return float(a) * float(b);
}
inline float operator*=(float16_t a, bfloat6_t b) {
  return float(a) * float(b);
}
inline float operator*=(bfloat6_t a, bfloat16_t b) {
  return float(a) * float(b);
}
inline float operator*=(bfloat16_t a, bfloat6_t b) {
  return float(a) * float(b);
}
inline float operator*=(bfloat6_t a, float8_t b) { return float(a) * float(b); }
inline float operator*=(float8_t a, bfloat6_t b) { return float(a) * float(b); }
inline float operator*=(bfloat6_t a, bfloat8_t b) {
  return float(a) * float(b);
}
inline float operator*=(bfloat8_t a, bfloat6_t b) {
  return float(a) * float(b);
}
inline float operator*=(bfloat6_t a, float6_t b) { return float(a) * float(b); }
inline float operator*=(float6_t a, bfloat6_t b) { return float(a) * float(b); }

// operator/=
inline float operator/=(bfloat6_t a, bfloat6_t b) {
  return float(a) / float(b);
}
inline float operator/=(bfloat6_t a, float b) { return float(a) / b; }
inline float operator/=(float a, bfloat6_t b) { return a / float(b); }
inline float operator/=(bfloat6_t a, float16_t b) {
  return float(a) / float(b);
}
inline float operator/=(float16_t a, bfloat6_t b) {
  return float(a) / float(b);
}
inline float operator/=(bfloat6_t a, bfloat16_t b) {
  return float(a) / float(b);
}
inline float operator/=(bfloat16_t a, bfloat6_t b) {
  return float(a) / float(b);
}
inline float operator/=(bfloat6_t a, float8_t b) { return float(a) / float(b); }
inline float operator/=(float8_t a, bfloat6_t b) { return float(a) / float(b); }
inline float operator/=(bfloat6_t a, bfloat8_t b) {
  return float(a) / float(b);
}
inline float operator/=(bfloat8_t a, bfloat6_t b) {
  return float(a) / float(b);
}
inline float operator/=(bfloat6_t a, float6_t b) { return float(a) / float(b); }
inline float operator/=(float6_t a, bfloat6_t b) { return float(a) / float(b); }

// operator>
inline float operator>(bfloat6_t a, bfloat6_t b) { return float(a) > float(b); }
inline float operator>(bfloat6_t a, float b) { return float(a) > b; }
inline float operator>(float a, bfloat6_t b) { return a > float(b); }
inline float operator>(bfloat6_t a, float16_t b) { return float(a) > float(b); }
inline float operator>(float16_t a, bfloat6_t b) { return float(a) > float(b); }
inline float operator>(bfloat6_t a, bfloat16_t b) {
  return float(a) > float(b);
}
inline float operator>(bfloat16_t a, bfloat6_t b) {
  return float(a) > float(b);
}
inline float operator>(bfloat6_t a, float8_t b) { return float(a) > float(b); }
inline float operator>(float8_t a, bfloat6_t b) { return float(a) > float(b); }
inline float operator>(bfloat6_t a, bfloat8_t b) { return float(a) > float(b); }
inline float operator>(bfloat8_t a, bfloat6_t b) { return float(a) > float(b); }
inline float operator>(bfloat6_t a, float6_t b) { return float(a) > float(b); }
inline float operator>(float6_t a, bfloat6_t b) { return float(a) > float(b); }

// operator>=
inline float operator>=(bfloat6_t a, bfloat6_t b) {
  return float(a) >= float(b);
}
inline float operator>=(bfloat6_t a, float b) { return float(a) >= b; }
inline float operator>=(float a, bfloat6_t b) { return a >= float(b); }
inline float operator>=(bfloat6_t a, float16_t b) {
  return float(a) >= float(b);
}
inline float operator>=(float16_t a, bfloat6_t b) {
  return float(a) >= float(b);
}
inline float operator>=(bfloat6_t a, bfloat16_t b) {
  return float(a) >= float(b);
}
inline float operator>=(bfloat16_t a, bfloat6_t b) {
  return float(a) >= float(b);
}
inline float operator>=(bfloat6_t a, float8_t b) {
  return float(a) >= float(b);
}
inline float operator>=(float8_t a, bfloat6_t b) {
  return float(a) >= float(b);
}
inline float operator>=(bfloat6_t a, bfloat8_t b) {
  return float(a) >= float(b);
}
inline float operator>=(bfloat8_t a, bfloat6_t b) {
  return float(a) >= float(b);
}
inline float operator>=(bfloat6_t a, float6_t b) {
  return float(a) >= float(b);
}
inline float operator>=(float6_t a, bfloat6_t b) {
  return float(a) >= float(b);
}

// operator<
inline float operator<(bfloat6_t a, bfloat6_t b) { return float(a) < float(b); }
inline float operator<(bfloat6_t a, float b) { return float(a) < b; }
inline float operator<(float a, bfloat6_t b) { return a < float(b); }
inline float operator<(bfloat6_t a, float16_t b) { return float(a) < float(b); }
inline float operator<(float16_t a, bfloat6_t b) { return float(a) < float(b); }
inline float operator<(bfloat6_t a, bfloat16_t b) {
  return float(a) < float(b);
}
inline float operator<(bfloat16_t a, bfloat6_t b) {
  return float(a) < float(b);
}
inline float operator<(bfloat6_t a, float8_t b) { return float(a) < float(b); }
inline float operator<(float8_t a, bfloat6_t b) { return float(a) < float(b); }
inline float operator<(bfloat6_t a, bfloat8_t b) { return float(a) < float(b); }
inline float operator<(bfloat8_t a, bfloat6_t b) { return float(a) < float(b); }
inline float operator<(bfloat6_t a, float6_t b) { return float(a) < float(b); }
inline float operator<(float6_t a, bfloat6_t b) { return float(a) < float(b); }

// operator<=
inline float operator<=(bfloat6_t a, bfloat6_t b) {
  return float(a) <= float(b);
}
inline float operator<=(bfloat6_t a, float b) { return float(a) <= b; }
inline float operator<=(float a, bfloat6_t b) { return a <= float(b); }
inline float operator<=(bfloat6_t a, float16_t b) {
  return float(a) <= float(b);
}
inline float operator<=(float16_t a, bfloat6_t b) {
  return float(a) <= float(b);
}
inline float operator<=(bfloat6_t a, bfloat16_t b) {
  return float(a) <= float(b);
}
inline float operator<=(bfloat16_t a, bfloat6_t b) {
  return float(a) <= float(b);
}
inline float operator<=(bfloat6_t a, float8_t b) {
  return float(a) <= float(b);
}
inline float operator<=(float8_t a, bfloat6_t b) {
  return float(a) <= float(b);
}
inline float operator<=(bfloat6_t a, bfloat8_t b) {
  return float(a) <= float(b);
}
inline float operator<=(bfloat8_t a, bfloat6_t b) {
  return float(a) <= float(b);
}
inline float operator<=(bfloat6_t a, float6_t b) {
  return float(a) <= float(b);
}
inline float operator<=(float6_t a, bfloat6_t b) {
  return float(a) <= float(b);
}

// operator!=
inline float operator!=(bfloat6_t a, bfloat6_t b) {
  return float(a) != float(b);
}
inline float operator!=(bfloat6_t a, float b) { return float(a) != b; }
inline float operator!=(float a, bfloat6_t b) { return a != float(b); }
inline float operator!=(bfloat6_t a, float16_t b) {
  return float(a) != float(b);
}
inline float operator!=(float16_t a, bfloat6_t b) {
  return float(a) != float(b);
}
inline float operator!=(bfloat6_t a, bfloat16_t b) {
  return float(a) != float(b);
}
inline float operator!=(bfloat16_t a, bfloat6_t b) {
  return float(a) != float(b);
}
inline float operator!=(bfloat6_t a, float8_t b) {
  return float(a) != float(b);
}
inline float operator!=(float8_t a, bfloat6_t b) {
  return float(a) != float(b);
}
inline float operator!=(bfloat6_t a, bfloat8_t b) {
  return float(a) != float(b);
}
inline float operator!=(bfloat8_t a, bfloat6_t b) {
  return float(a) != float(b);
}
inline float operator!=(bfloat6_t a, float6_t b) {
  return float(a) != float(b);
}
inline float operator!=(float6_t a, bfloat6_t b) {
  return float(a) != float(b);
}

// operator==
inline float operator==(bfloat6_t a, bfloat6_t b) {
  return float(a) == float(b);
}
inline float operator==(bfloat6_t a, float b) { return float(a) == b; }
inline float operator==(float a, bfloat6_t b) { return a == float(b); }
inline float operator==(bfloat6_t a, float16_t b) {
  return float(a) == float(b);
}
inline float operator==(float16_t a, bfloat6_t b) {
  return float(a) == float(b);
}
inline float operator==(bfloat6_t a, bfloat16_t b) {
  return float(a) == float(b);
}
inline float operator==(bfloat16_t a, bfloat6_t b) {
  return float(a) == float(b);
}
inline float operator==(bfloat6_t a, float8_t b) {
  return float(a) == float(b);
}
inline float operator==(float8_t a, bfloat6_t b) {
  return float(a) == float(b);
}
inline float operator==(bfloat6_t a, bfloat8_t b) {
  return float(a) == float(b);
}
inline float operator==(bfloat8_t a, bfloat6_t b) {
  return float(a) == float(b);
}
inline float operator==(bfloat6_t a, float6_t b) {
  return float(a) == float(b);
}
inline float operator==(float6_t a, bfloat6_t b) {
  return float(a) == float(b);
}

// operator++
inline float operator++(bfloat6_t a) { return float(a) + float(1); }

// operator--
inline float operator--(bfloat6_t a) { return float(a) - float(1); }

__FP6_HOST_DEVICE_STATIC__ void condense_fp6_to_b6(const void *fp6, void *b6, int fp6_size, int b6_size) {
  assert((fp6_size * 6) % 8 == 0 && (fp6_size * 6) / 8 == b6_size);
  // condense each 4 fp6 into 3 b8
  fp6_storage_t valid_data[4] = {0, 0, 0, 0};
  for (size_t i = 0; i < fp6_size; i += 4) {
    for (size_t j = 0; j < 4; ++j)
      valid_data[j] = (*((uint8_t*)fp6 + i + j)) & 0x3f;
    *((uint8_t*)b6 + i / 4 * 3) = valid_data[0] | ((valid_data[1] & 0x3) << 6);
    *((uint8_t*)b6 + i / 4 * 3 + 1) = (valid_data[1] >> 2) | ((valid_data[2] & 0xf) << 4);
    *((uint8_t*)b6 + i / 4 * 3 + 2) = (valid_data[2] >> 4) | (valid_data[3] << 2);
  }
}

struct __fp6x4_e2m3_condense_t {
  fp6_storage_t storage[3];

  __FP6_HOST_DEVICE__ __fp6x4_e2m3_condense_t(float6_t data[4]) {
    condense_fp6_to_b6(&data[0], &storage[0], 4, 3);
  }
  __host__ __fp6x4_e2m3_condense_t(const std::vector<float6_t> &data) {
    condense_fp6_to_b6(data.data(), &storage[0], 4, 3);
  }
  __FP6_HOST_DEVICE__ __fp6x4_e2m3_condense_t(void *data) {
    condense_fp6_to_b6(data, &storage[0], 4, 3);
  }

  __FP6_HOST_DEVICE__ float6_t const operator[](uint32_t i) {
    float6_t res;
    uint32_t lo = i == 0 ? 0 : i - 1, hi = i == 3 ? 2 : i;
    res.__x = /*low bits*/(storage[lo] >> ((4 - i) * 2)) |
              /*high bits*/((storage[hi] & ((1 << (3 - i) * 2) - 1)) << (i * 2));
    return res;
  }
};

// print condensed bits
inline std::ostream &operator<<(std::ostream &os, __fp6x4_e2m3_condense_t src) {
  os << (uint32_t)src.storage[0] << " " << (uint32_t)src.storage[1] << " " << (uint32_t)src.storage[2];
  return os;
}

struct __fp6x4_e3m2_condense_t {
  fp6_storage_t storage[3];

  __FP6_HOST_DEVICE__ __fp6x4_e3m2_condense_t(bfloat6_t data[4]) {
    condense_fp6_to_b6(&data[0], &storage[0], 4, 3);
  }
  __host__ __fp6x4_e3m2_condense_t(const std::vector<bfloat6_t> &data) {
    condense_fp6_to_b6(data.data(), &storage[0], 4, 3);
  }
  __FP6_HOST_DEVICE__ __fp6x4_e3m2_condense_t(void *data) {
    condense_fp6_to_b6(data, &storage[0], 4, 3);
  }

  __FP6_HOST_DEVICE__ bfloat6_t const operator[](uint32_t i) {
    bfloat6_t res;
    uint32_t lo = i == 0 ? 0 : i - 1, hi = i == 3 ? 2 : i;
    res.__x = /*low bits*/(storage[lo] >> ((4 - i) * 2)) |
              /*high bits*/((storage[hi] & ((1 << (3 - i) * 2) - 1)) << (i * 2));
    return res;
  }
};

inline std::ostream &operator<<(std::ostream &os, __fp6x4_e3m2_condense_t src) {
  os << (uint32_t)src.storage[0] << " " << (uint32_t)src.storage[1] << " " << (uint32_t)src.storage[2];
  return os;
}