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
#include <cstring>
#include <iostream>

#define TENSILE_USE_BF16

#ifndef __BYTE_ORDER__
#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
#endif

#define BFLOAT16_Q_NAN_VALUE 0xFFC1
#define MAX_BF16 0x7F7F
#define NEG_MAX_BF16 0xFF7F
namespace Tensile {
/**
 * \ingroup DataTypes
 * @{
 */
// Floating-point type with 8 bits of exponent and 7 bits of mantissa.
struct alignas(2) BFloat16 {

  //
  // Data members
  //

  // data type
  uint16_t data;

  //
  // Methods
  //

  // Constructs from an unsigned short
  static BFloat16 bitcast(uint16_t x) {
    BFloat16 h;
    h.data = x;
    return h;
  }

private:
  struct from_32_bit_integer_t {};
  static constexpr from_32_bit_integer_t from_32_bit_integer{};

  template <class T> explicit BFloat16(from_32_bit_integer_t, T x) {
    float flt = static_cast<float>(x);
    uint32_t bits;
    std::memcpy(&bits, &flt, sizeof(bits));
    data = uint16_t(bits >> 16);
  }

public:
  // Default constructor
  BFloat16() = default;

  BFloat16(BFloat16 const &other) = default;

  // Floating-point conversion - round toward nearest
  explicit BFloat16(float x) {

    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    if ((bits & 0x7f800000) != 0x7f800000) { // exclude nan and inf
      bool mantissa_bit = ((bits & (1 << 16)) != 0);
      bool round_bit = ((bits & (1 << 15)) != 0);
      bool sticky_bit = ((bits & ((1 << 15) - 1)) != 0);
      if ((round_bit && sticky_bit) || (round_bit && mantissa_bit)) {
        bits += uint32_t(1 << 16);
      }
    }
    data = uint16_t((bits >> 16) & 0xffff);
  }

  // Floating-point conversion - round toward nearest
  explicit BFloat16(double x) : BFloat16(float(x)) {}

  // Integer conversion - round toward nearest
  explicit BFloat16(int x) : BFloat16(from_32_bit_integer, x) {}

  explicit BFloat16(uint32_t x) : BFloat16(from_32_bit_integer, x) {}

  // Converts to float
  operator float() const {
    unsigned bits = (unsigned(data) << 16);
    float flt;
    std::memcpy(&flt, &bits, sizeof(flt));
    return flt;
  }

  // Converts to double
  explicit operator double() const { return double(float(*this)); }

  // Converts to int
  explicit operator int() const { return int(float(*this)); }

  // Casts to bool
  explicit operator bool() const { return (float(*this) != 0.0f); }

  // Obtains raw bits
  uint16_t raw() const { return data; }
  // Returns the sign bit
  bool signbit() const { return ((raw() & 0x8000) != 0); }

  // Returns the biased exponent
  int exponent_biased() const { return int((raw() >> 7) & 0x0ff); }

  // Returns the unbiased exponent
  int exponent() const { return exponent_biased() - 127; }

  // Returns the mantissa
  int mantissa() const { return int(raw() & 0x7f); }
};

inline std::ostream &operator<<(std::ostream &os, const BFloat16 &bf16) {
  os << static_cast<float>(bf16);
  return os;
}

inline BFloat16 operator+(BFloat16 a, BFloat16 b) {
  return static_cast<BFloat16>(static_cast<float>(a) + static_cast<float>(b));
}
inline BFloat16 operator+(int a, BFloat16 b) {
  return static_cast<BFloat16>(static_cast<float>(a) + static_cast<float>(b));
}
inline BFloat16 operator+(BFloat16 a, int b) {
  return static_cast<BFloat16>(static_cast<float>(a) + static_cast<float>(b));
}
inline BFloat16 operator-(BFloat16 a, BFloat16 b) {
  return static_cast<BFloat16>(static_cast<float>(a) - static_cast<float>(b));
}
inline BFloat16 operator*(BFloat16 a, BFloat16 b) {
  return static_cast<BFloat16>(static_cast<float>(a) * static_cast<float>(b));
}
inline BFloat16 operator/(BFloat16 a, BFloat16 b) {
  return static_cast<BFloat16>(static_cast<float>(a) / static_cast<float>(b));
}

inline bool operator<(BFloat16 a, BFloat16 b) {
  return static_cast<float>(a) < static_cast<float>(b);
}
inline bool operator<=(BFloat16 a, BFloat16 b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}
inline bool operator==(BFloat16 a, BFloat16 b) {
  return static_cast<float>(a) == static_cast<float>(b);
}
inline bool operator!=(BFloat16 a, BFloat16 b) {
  return static_cast<float>(a) != static_cast<float>(b);
}
inline bool operator>(BFloat16 a, BFloat16 b) {
  return static_cast<float>(a) > static_cast<float>(b);
}
inline bool operator>=(BFloat16 a, BFloat16 b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}

inline BFloat16 &operator+=(BFloat16 &a, BFloat16 b) {
  a = a + b;
  return a;
}
inline BFloat16 &operator-=(BFloat16 &a, BFloat16 b) {
  a = a - b;
  return a;
}
inline BFloat16 &operator*=(BFloat16 &a, BFloat16 b) {
  a = a * b;
  return a;
}
inline BFloat16 &operator/=(BFloat16 &a, BFloat16 b) {
  a = a / b;
  return a;
}

inline BFloat16 operator++(BFloat16 &a) {
  a += BFloat16(1);
  return a;
}
inline BFloat16 operator++(BFloat16 &a, int) {
  BFloat16 original_value = a;
  ++a;
  return original_value;
}

/**
 * @}
 */
} // namespace Tensile

namespace std {
inline bool isinf(const Tensile::BFloat16 &a) {
  return std::isinf(static_cast<float>(a));
}
inline bool isnan(const Tensile::BFloat16 &a) {
  return std::isnan(static_cast<float>(a));
}
inline bool iszero(const Tensile::BFloat16 &a) {
  return (a.data & 0x7FFF) == 0;
}

inline Tensile::BFloat16 abs(const Tensile::BFloat16 &a) {
  return static_cast<Tensile::BFloat16>(std::abs(static_cast<float>(a)));
}
inline Tensile::BFloat16 sin(const Tensile::BFloat16 &a) {
  return static_cast<Tensile::BFloat16>(std::sin(static_cast<float>(a)));
}
inline Tensile::BFloat16 cos(const Tensile::BFloat16 &a) {
  return static_cast<Tensile::BFloat16>(std::cos(static_cast<float>(a)));
}
} // namespace std
