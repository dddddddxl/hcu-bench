/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2021 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <cinttypes>
#include <cmath>
#include <iostream>

#define TENSILE_USE_TF32

#ifndef __BYTE_ORDER__
#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
#endif

#define TFLOAT32_Q_NAN_VALUE 0xFFC10000

namespace Tensile {
/**
 * \ingroup DataTypes
 * @{
 */

struct TFloat32 {
  TFloat32() : data(TFLOAT32_ZERO_VALUE) {}

  TFloat32(TFloat32 const &other) = default;

  template <typename T, typename = typename std::enable_if<
                            (!std::is_same<T, TFloat32>::value) &&
                            std::is_convertible<T, float>::value>::type>
  explicit TFloat32(T const &value)
      : data(float_to_tfloat32(static_cast<float>(value)).data) {}

  explicit operator float() const { return tfloat32_to_float(*this); }

  explicit operator double() const { return static_cast<double>(float(*this)); }

  explicit operator int() const { return static_cast<int>(float(*this)); }

  explicit operator uint32_t() const {
    return static_cast<uint32_t>(float(*this));
  }

  uint32_t data;

private:
  static const int16_t TFLOAT32_ZERO_VALUE = 0x00;

  static float tfloat32_to_float(const TFloat32 v) {
    union {
      float fp32;
      uint32_t i32;
    };
    i32 = v.data;
    return fp32;
  }

  static TFloat32 float_to_tfloat32(const float v) {
    TFloat32 tf32;
    if (std::isnan(v)) {
      tf32.data = TFLOAT32_Q_NAN_VALUE;
      return tf32;
    }
    union {
      float fp32;
      uint32_t i32;
    };
    fp32 = v;
    i32 &= 0xFFFFE000;

    tf32.data = i32;
    return tf32;
  }
};

inline std::ostream &operator<<(std::ostream &os, const TFloat32 &tf32) {
  os << static_cast<float>(tf32);
  return os;
}

inline TFloat32 operator+(TFloat32 a, TFloat32 b) {
  return static_cast<TFloat32>(static_cast<float>(a) + static_cast<float>(b));
}
inline TFloat32 operator+(int a, TFloat32 b) {
  return static_cast<TFloat32>(static_cast<float>(a) + static_cast<float>(b));
}
inline TFloat32 operator+(TFloat32 a, int b) {
  return static_cast<TFloat32>(static_cast<float>(a) + static_cast<float>(b));
}
inline TFloat32 operator-(TFloat32 a, TFloat32 b) {
  return static_cast<TFloat32>(static_cast<float>(a) - static_cast<float>(b));
}
inline TFloat32 operator*(TFloat32 a, TFloat32 b) {
  return static_cast<TFloat32>(static_cast<float>(a) * static_cast<float>(b));
}
inline TFloat32 operator/(TFloat32 a, TFloat32 b) {
  return static_cast<TFloat32>(static_cast<float>(a) / static_cast<float>(b));
}

inline bool operator<(TFloat32 a, TFloat32 b) {
  return static_cast<float>(a) < static_cast<float>(b);
}
inline bool operator<=(TFloat32 a, TFloat32 b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}
inline bool operator==(TFloat32 a, TFloat32 b) {
  return static_cast<float>(a) == static_cast<float>(b);
}
inline bool operator!=(TFloat32 a, TFloat32 b) {
  return static_cast<float>(a) != static_cast<float>(b);
}
inline bool operator>(TFloat32 a, TFloat32 b) {
  return static_cast<float>(a) > static_cast<float>(b);
}
inline bool operator>=(TFloat32 a, TFloat32 b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}

inline TFloat32 &operator+=(TFloat32 &a, TFloat32 b) {
  a = a + b;
  return a;
}
inline TFloat32 &operator-=(TFloat32 &a, TFloat32 b) {
  a = a - b;
  return a;
}
inline TFloat32 &operator*=(TFloat32 &a, TFloat32 b) {
  a = a * b;
  return a;
}
inline TFloat32 &operator/=(TFloat32 &a, TFloat32 b) {
  a = a / b;
  return a;
}

inline TFloat32 operator++(TFloat32 &a) {
  a += TFloat32(1);
  return a;
}
inline TFloat32 operator++(TFloat32 &a, int) {
  TFloat32 original_value = a;
  ++a;
  return original_value;
}

/**
 * @}
 */
} // namespace Tensile

namespace std {
inline bool isinf(const Tensile::TFloat32 &a) {
  return std::isinf(static_cast<float>(a));
}
inline bool isnan(const Tensile::TFloat32 &a) {
  return std::isnan(static_cast<float>(a));
}
inline bool iszero(const Tensile::TFloat32 &a) {
  return (a.data & 0x7FFFFFFF) == 0;
}

inline Tensile::TFloat32 abs(const Tensile::TFloat32 &a) {
  return static_cast<Tensile::TFloat32>(std::abs(static_cast<float>(a)));
}
inline Tensile::TFloat32 sin(const Tensile::TFloat32 &a) {
  return static_cast<Tensile::TFloat32>(std::sin(static_cast<float>(a)));
}
inline Tensile::TFloat32 cos(const Tensile::TFloat32 &a) {
  return static_cast<Tensile::TFloat32>(std::cos(static_cast<float>(a)));
}
} // namespace std
