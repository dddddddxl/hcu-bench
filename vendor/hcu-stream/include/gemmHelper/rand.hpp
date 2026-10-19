#ifndef __RAND_HPP__
#define __RAND_HPP__

#include "gemmHelper/global.hpp"
#include <stdlib.h>

template <typename T> class Rng {

#define RNG_DEFAULT 0
#define RNG_CONST 1
#define RNG_RANGE 2

public:
  Rng() : m_mode(RNG_DEFAULT){};
  Rng(T val) : m_mode(RNG_CONST), m_val(val){};
  Rng(T min, T max) : m_mode(RNG_RANGE), m_min(min), m_max(max){};
  T operator()() {
    if (m_mode == RNG_DEFAULT) {
      if constexpr (std::is_same<T, int8_t>::value ||
                    std::is_same<T, int16_t>::value ||
                    std::is_same<T, int32_t>::value) {
        return static_cast<T>(rand() % 255 - 128);
      } else if constexpr (std::is_same<T, Int4x2>::value) {
        return static_cast<T>(rand() % 255 - 128);
      } else if constexpr (std::is_same<T, uint8_t>::value ||
                           std::is_same<T, Uint4x2>::value) {
        return static_cast<T>(rand() % 255);
      } /*else if constexpr (std::is_same<T, fp8>::value ||
                           std::is_same<T, bf8>::value) {
        // model not support fp8 denormal
        float tmp = rand() * 1.0 / RAND_MAX;
        if (tmp < 0 && tmp > -0.015625f) {
          tmp -= 0.015625f;
        }
        if (tmp > 0 && tmp < 0.015625f) {
          tmp += 0.015625f;
        }
        return static_cast<T>(tmp);
      }*/
      else if constexpr (std::is_same<T, float8_t>::value ||
                         std::is_same<T, bfloat8_t>::value) {
        float tmp = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * (1.0f - 0.0f) + 0.0f;
        return static_cast<T>(tmp);
      } 
      else if constexpr (std::is_same<T, float6_t>::value) {
        float tmp = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * (1.0f - 0.0f) + 0.0f;
        return float6_t(tmp);
      }
      else if constexpr (std::is_same<T, bfloat6_t>::value) {
        float tmp = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * (1.0f - 0.0f) + 0.0f;
        return bfloat6_t(tmp);
      }
      else if constexpr (std::is_same<T, float4_t>::value) {
        float tmp = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * (1.0f - 0.0f) + 0.0f;
        return float4_t(tmp);
      }
      else if constexpr (std::is_same<T, float4x2_t>::value) {
        float tmp = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * (1.0f - 0.0f) + 0.0f;
        float tmp2 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * (1.0f - 0.0f) + 0.0f;
        return float4x2_t(tmp, tmp2);
      }
      else {
        return static_cast<T>(rand() * 1.0 / RAND_MAX);
      }
    } else if (m_mode == RNG_CONST) {
      return m_val;
    } else {
      if constexpr (std::is_same<T, Int4x2>::value ||
                    std::is_same<T, Uint4x2>::value) {
        return T(rand() % (m_max[0] - m_min[0] + 1) + m_min[0],
                 rand() % (m_max[1] - m_min[1] + 1) + m_min[0]);
      } else if constexpr (std::is_same<T, float8_t>::value) {
        float tmp =
            rand() * 1.0 / RAND_MAX * (float)(1.0f - 0.0f) + (float)m_min;
        // model not support fp8 denormal
        if (tmp < 0 && tmp > -0.015625f) {
          tmp -= 0.015625f;
        }
        if (tmp > 0 && tmp < 0.015625f) {
          tmp += 0.015625f;
        }
        return static_cast<T>(tmp);
      } else if constexpr (std::is_same<T, bfloat8_t>::value) {
        float tmp =
            rand() * 1.0 / RAND_MAX * (float)(1.0f - 0.0f) + (float)m_min;
        // model not support bf8 denormal
        if (tmp < 0 && tmp > -0.000061035f) {
          tmp -= 0.000061035f;
        }
        if (tmp > 0 && tmp < 0.000061035f) {
          tmp += 0.000061035f;
        }
        return static_cast<T>(tmp);
      }
      else if constexpr (std::is_same<T, float6_t>::value) {
        float tmp = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * (1.0f - 0.0f) + 0.0f;
        return float6_t(tmp);
      }
      else if constexpr (std::is_same<T, bfloat6_t>::value) {
        float tmp = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * (1.0f - 0.0f) + 0.0f;
        return bfloat6_t(tmp);
      }
      else if constexpr (std::is_same<T, float4_t>::value) {
        float tmp = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * (1.0f - 0.0f) + 0.0f;
        return float4_t(tmp);
      }
      else if constexpr (std::is_same<T, float4x2_t>::value) {
        float tmp = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * (1.0f - 0.0f) + 0.0f;
        float tmp2 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * (1.0f - 0.0f) + 0.0f;
        return float4x2_t(tmp, tmp2);
      }
      else {
        return static_cast<T>(rand() * 1.0 / RAND_MAX * (float)(m_max - m_min) +
                              (float)m_min);
      }
    }
  }

private:
  int m_mode;
  T m_val;
  T m_min;
  T m_max;
};

template <typename T> struct Range {
  T min;
  T max;
};

/* solve ax^2 + bx + c = 0 */
inline double SolveQuadraticEquation(double a, double b, double c) {
  return (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
}

/*
 * random int32 value range for A/B/C to avoid D value overflow
 * D = alpha * A * B + beta * C
 *
 * requirements:
 * alpha * K * x_max * x_max + beta * x_max <= 2^31 - 1
 * alpha * K * x_min * x_max + beta * x_min >= -2^31
 */
inline Range<int32_t> Int32RandomValueRange(int alpha, int beta, size_t K) {
  double x_max = SolveQuadraticEquation(static_cast<double>(alpha * K),
                                        static_cast<double>(beta),
                                        static_cast<double>(-INT32_MAX));
  int max = static_cast<int>(floor(x_max));

  double x_min = static_cast<double>(INT32_MIN) / (alpha * K * max + beta);

  int min = static_cast<int>(ceil(x_min));

  return Range<int32_t>{min, max};
}

template <typename SRC_T, typename DST_T, typename ACCUM_T>
std::pair<Range<SRC_T>, Range<DST_T>> RandRange(ACCUM_T alpha, ACCUM_T beta,
                                                size_t K) {
  if constexpr (std::is_same<SRC_T, int8_t>::value) {
    Range<int32_t> range_i32 = Int32RandomValueRange(alpha, beta, K);
    Range<int8_t> range_i8;

    range_i8.min = std::max(range_i32.min, INT8_MIN);
    range_i8.max = std::min(range_i32.max, INT8_MAX);

    return std::make_pair(range_i8, range_i32);
  } else if constexpr (std::is_same<SRC_T, Int4x2>::value) {
    Range<int32_t> range_i32 = Int32RandomValueRange(alpha, beta, K);
    Range<Int4x2> range_i4x2;

    range_i4x2.min =
        Int4x2(std::max(range_i32.min, -8), std::max(range_i32.min, -8));
    range_i4x2.max =
        Int4x2(std::min(range_i32.max, 7), std::min(range_i32.max, 7));

    return std::make_pair(range_i4x2, range_i32);
  } else if constexpr (std::is_same<SRC_T, int16_t>::value) {
    Range<int32_t> range_i32 = Int32RandomValueRange(alpha, beta, K);
    Range<int16_t> range_i16;

    range_i16.min = std::max(range_i32.min, INT16_MIN);
    range_i16.max = std::min(range_i32.max, INT16_MAX);

    return std::make_pair(range_i16, range_i32);
  } else if constexpr (std::is_same<SRC_T, int32_t>::value) {
    Range<int32_t> range_i32 = Int32RandomValueRange(alpha, beta, K);

    return std::make_pair(range_i32, range_i32);
  } else if constexpr (std::is_same<SRC_T, Uint4x2>::value) {
    Range<int32_t> range_i32 = Int32RandomValueRange(alpha, beta, K);
    Range<Uint4x2> range_u4x2;

    range_u4x2.min = Uint4x2(0, 0);
    range_u4x2.max = Uint4x2(std::min(range_i32.max, (int32_t)15),
                             std::min(range_i32.max, (int32_t)15));

    return std::make_pair(range_u4x2, range_i32);
  } else if constexpr (std::is_same<SRC_T, uint8_t>::value) {
    Range<int32_t> range_i32 = Int32RandomValueRange(alpha, beta, K);
    Range<uint8_t> range_u8;

    range_u8.min = 0;
    range_u8.max = std::min(range_i32.max, (int32_t)UINT8_MAX);

    return std::make_pair(range_u8, range_i32);
  } else if constexpr (std::is_same<SRC_T, float8_t>::value) {
    return std::make_pair(Range<SRC_T>{SRC_T(-0.5f), SRC_T(0.5f)},
                          Range<DST_T>{DST_T(-0.5f), DST_T(0.5f)});
  } else if constexpr (std::is_same<SRC_T, bfloat8_t>::value) {
    return std::make_pair(Range<SRC_T>{SRC_T(-0.5f), SRC_T(0.5f)},
                          Range<DST_T>{DST_T(-0.5f), DST_T(0.5f)});
  } else if constexpr (std::is_same<SRC_T, float4x2_t>::value) {
    return std::make_pair(Range<SRC_T>{SRC_T(0.0f, 1.0f), SRC_T(0.0f, 1.0f)},
                          Range<DST_T>{DST_T(0.0f), DST_T(1.0f)});
  } else {
    return std::make_pair(Range<SRC_T>{SRC_T(0.0f), SRC_T(1.0f)},
                          Range<DST_T>{DST_T(0.0f), DST_T(1.0f)});
  }
}

#endif /* __RAND_HPP__ */
