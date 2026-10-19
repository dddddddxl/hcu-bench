#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>

#include "type/DataTypes_BFloat16.hpp"
#include "type/half.hpp"

using float16_t = half_float::half;
using bfloat16_t = Tensile::BFloat16;

#ifndef __host__
#define __host__
#define __device__
#endif

#define __OCP_FP_HOST__ __host__
#define __OCP_FP_DEVICE__ __device__
#define __OCP_FP_HOST_DEVICE__ __OCP_FP_HOST__ __OCP_FP_DEVICE__
#define __OCP_FP_DEVICE_STATIC__                                               \
  __OCP_FP_DEVICE__ static __inline__ __attribute__((always_inline))
#define __OCP_FP_HOST_DEVICE_STATIC__ __OCP_FP_HOST_DEVICE__ static

static_assert(sizeof(unsigned int) == 4);
static_assert(sizeof(float) == 4);
static_assert(sizeof(unsigned short) == 2);

// Although we do have some abstractions of half and bfloat16, since this will
// be a standalone header which will act as a base abstraction, and will be
// maintained in the future, it makes sense to keep these vector types separate
// from existing implementations. We can add conversion functions in a different
// header using these functions.
typedef uint8_t fp8_storage_t;
typedef uint16_t fp8x2_storage_t;
typedef uint8_t fp4x2_storage_t;
typedef uint32_t fp4x8_storage_t;
typedef __bf16 bf16_storage_t;
typedef _Float16 fp16_storage_t;
typedef uint8_t scale_t;

typedef unsigned int __attribute__((ext_vector_type(2))) uintx2_storage_t;
typedef uint8_t __attribute__((ext_vector_type(8))) fp8x8_storage_t;
typedef __bf16 __attribute__((ext_vector_type(2))) bf16x2_storage_t;
typedef __bf16 __attribute__((ext_vector_type(8))) bf16x8_storage_t;
typedef __bf16 __attribute__((ext_vector_type(32))) bf16x32_storage_t;
typedef float __attribute__((ext_vector_type(2))) floatx2_storage_t;
typedef float __attribute__((ext_vector_type(8))) floatx8_storage_t;
typedef float __attribute__((ext_vector_type(16))) floatx16_storage_t;
typedef float __attribute__((ext_vector_type(32))) floatx32_storage_t;
typedef _Float16 __attribute__((ext_vector_type(2))) fp16x2_storage_t;
typedef _Float16 __attribute__((ext_vector_type(8))) fp16x8_storage_t;
typedef _Float16 __attribute__((ext_vector_type(32))) fp16x32_storage_t;
typedef uint32_t __attribute__((ext_vector_type(6))) fp6x32_storage_t;
typedef short __attribute__((ext_vector_type(2))) shortx2_storage_t;

static_assert(sizeof(uintx2_storage_t) == sizeof(fp8x8_storage_t));

enum hipRoundMode {
  hipRoundNearest = 0,
  hipRoundZero = 1,
  hipRoundPosInf = 2,
  hipRoundMinInf = 3,
  hipRoundStochastic = 4,
};

namespace fcbx {
constexpr uint8_t OCP_SCALE_EXP_NAN = 0xff;

enum class Encoding : size_t {
  E2M1 = 0,
  E2M3,
  E3M2,
  E4M3,
  E4M3Mx,
  E4M3Nanoo,
  E5M2,
  E5M2Mx,
  E5M2Nanoo,

  E5M10, // FP16
  E8M7,  // BF16

  IEEE754,

  // Keep this one last
  NumEncodings,
};
enum fp16 : uint16_t {};
enum bf16 : uint16_t {};

struct Float {
  int32_t ExpBias;
  uint32_t ExpBits;
  uint32_t ExpMask;
  uint32_t ManBits;
  uint32_t ManMask;
  int32_t MaxExp;
  int32_t MinExp;
  bool MxScale;
  bool HasNaN;
  bool HasInf;
};

static const float ieee754_nan = std::numeric_limits<float>::quiet_NaN();
static const float ieee754_inf = std::numeric_limits<float>::infinity();

__OCP_FP_HOST_DEVICE_STATIC__ uint32_t U32(float f) {
  static_assert(sizeof(uint32_t) == sizeof(float));
  union {
    float f32;
    uint32_t ui32;
  } u{f};
  return u.ui32;
}

__OCP_FP_HOST_DEVICE_STATIC__ float F32(uint32_t u32) {
  static_assert(sizeof(uint32_t) == sizeof(float));
  union {
    uint32_t ui32;
    float f32;
  } u{u32};
  return u.f32;
}

constexpr __OCP_FP_HOST_DEVICE_STATIC__ uint32_t bitmask(uint32_t bits) {
  if (bits < 1)
    return 0;
  return ((uint32_t)1 << bits) - 1;
}

constexpr std::array<Float, (size_t)Encoding::NumEncodings> init() {
  std::array<Float, (size_t)Encoding::NumEncodings> a{};

  a[(size_t)Encoding::E2M1] = {
      .ExpBias = 1,
      .ExpBits = 2,
      .ExpMask = bitmask(2),
      .ManBits = 1,
      .ManMask = bitmask(1),
      .MaxExp = 2,
      .MinExp = 0,
      .MxScale = true,
      .HasNaN = false,
      .HasInf = false,
  };

  a[(size_t)Encoding::E2M3] = {
      .ExpBias = 1,
      .ExpBits = 2,
      .ExpMask = bitmask(2),
      .ManBits = 3,
      .ManMask = bitmask(3),
      .MaxExp = 2,
      .MinExp = 0,
      .MxScale = true,
      .HasNaN = false,
      .HasInf = false,
  };

  a[(size_t)Encoding::E3M2] = {
      .ExpBias = 3,
      .ExpBits = 3,
      .ExpMask = bitmask(3),
      .ManBits = 2,
      .ManMask = bitmask(2),
      .MaxExp = 4,
      .MinExp = -2,
      .MxScale = true,
      .HasNaN = false,
      .HasInf = false,
  };

  a[(size_t)Encoding::E4M3] = {
      .ExpBias = 7,
      .ExpBits = 4,
      .ExpMask = bitmask(4),
      .ManBits = 3,
      .ManMask = bitmask(3),
      .MaxExp = 8,
      .MinExp = -6,
      .MxScale = false,
      .HasNaN = true,
      .HasInf = false,
  };

  a[(size_t)Encoding::E4M3Mx] = {
      .ExpBias = 7,
      .ExpBits = 4,
      .ExpMask = bitmask(4),
      .ManBits = 3,
      .ManMask = bitmask(3),
      .MaxExp = 8,
      .MinExp = -6,
      .MxScale = true,
      .HasNaN = true,
      .HasInf = false,
  };

  a[(size_t)Encoding::E4M3Nanoo] = {
      .ExpBias = 8,
      .ExpBits = 4,
      .ExpMask = bitmask(4),
      .ManBits = 3,
      .ManMask = bitmask(3),
      .MaxExp = 7,
      .MinExp = -7,
      .MxScale = false,
      .HasNaN = true,
      .HasInf = false,
  };

  a[(size_t)Encoding::E5M2] = {
      .ExpBias = 15,
      .ExpBits = 5,
      .ExpMask = bitmask(5),
      .ManBits = 2,
      .ManMask = bitmask(2),
      .MaxExp = 15,
      .MinExp = -14,
      .MxScale = false,
      .HasNaN = true,
      .HasInf = true,
  };

  a[(size_t)Encoding::E5M2Mx] = {
      .ExpBias = 15,
      .ExpBits = 5,
      .ExpMask = bitmask(5),
      .ManBits = 2,
      .ManMask = bitmask(2),
      .MaxExp = 15,
      .MinExp = -14,
      .MxScale = true,
      .HasNaN = true,
      .HasInf = true,
  };

  a[(size_t)Encoding::E5M2Nanoo] = {
      .ExpBias = 16,
      .ExpBits = 5,
      .ExpMask = bitmask(5),
      .ManBits = 2,
      .ManMask = bitmask(2),
      .MaxExp = 15,
      .MinExp = -15,
      .MxScale = false,
      .HasNaN = true,
      .HasInf = true,
  };

  a[(size_t)Encoding::E5M10] = {
      .ExpBias = 15,
      .ExpBits = 5,
      .ExpMask = bitmask(5),
      .ManBits = 10,
      .ManMask = bitmask(10),
      .MaxExp = 15,
      .MinExp = -14,
      .MxScale = false,
      .HasNaN = true,
      .HasInf = true,
  };

  a[(size_t)Encoding::E8M7] = {
      .ExpBias = 127,
      .ExpBits = 8,
      .ExpMask = bitmask(8),
      .ManBits = 7,
      .ManMask = bitmask(7),
      .MaxExp = 127,
      .MinExp = -126,
      .MxScale = false,
      .HasNaN = true,
      .HasInf = true,
  };

  a[(size_t)Encoding::IEEE754] = {
      .ExpBias = 127,
      .ExpBits = 8,
      .ExpMask = bitmask(8),
      .ManBits = 23,
      .ManMask = bitmask(23),
      .MaxExp = 127,
      .MinExp = -126,
      .MxScale = false,
      .HasNaN = true,
      .HasInf = true,
  };

  return a;
}

static constexpr auto encodings = init();

template <Encoding E, bool sat>
__OCP_FP_HOST_DEVICE_STATIC__ uint32_t exponentbits(uint32_t val) {
  const auto &enc = encodings[(size_t)E];
  return (val >> enc.ManBits) & enc.ExpMask;
}

template <Encoding E, bool sat>
__OCP_FP_HOST_DEVICE_STATIC__ uint32_t mantissa(uint32_t val) {
  const auto &enc = encodings[(size_t)E];
  return val & enc.ManMask;
}

template <Encoding E, bool sat>
__OCP_FP_HOST_DEVICE_STATIC__ bool issubnorm(uint32_t val) {
  switch (E) {
  default:
    return exponentbits<E, sat>(val) == 0 && mantissa<E, sat>(val) != 0;
  }

  __builtin_trap();
  // Unreachable
  return false;
}

template <Encoding E, bool sat>
__OCP_FP_HOST_DEVICE_STATIC__ int32_t exponent(uint32_t val) {
  const auto &enc = encodings[(size_t)E];
  auto unbiased_exp = exponentbits<E, sat>(val);
  unbiased_exp = issubnorm<E, sat>(val) ? 1 : unbiased_exp;
  return (int32_t)unbiased_exp - enc.ExpBias;
}

template <Encoding E, bool sat>
__OCP_FP_HOST_DEVICE_STATIC__ uint32_t signbit(uint32_t val) {
  const auto &enc = encodings[(size_t)E];
  return (val >> (enc.ExpBits + enc.ManBits)) & 1;
}

template <Encoding E, bool sat>
__OCP_FP_HOST_DEVICE_STATIC__ uint32_t nan(uint32_t sign) {
  const auto &enc = encodings[(size_t)E];

  switch (E) {
  case Encoding::E2M1:
    return (sign << (enc.ExpBits + enc.ManBits)) | 0b0111;
  case Encoding::E2M3:
  case Encoding::E3M2:
    return (sign << (enc.ExpBits + enc.ManBits)) | 0b011111;
  case Encoding::E4M3:
  case Encoding::E4M3Mx:
    return (sign << (enc.ExpBits + enc.ManBits)) | 0x7f;
  case Encoding::E5M2:
  case Encoding::E5M2Mx:
    return (sign << (enc.ExpBits + enc.ManBits)) | 0x7e;
  case Encoding::E4M3Nanoo:
  case Encoding::E5M2Nanoo:
    return 0b10000000;
  case Encoding::E5M10:
  case Encoding::E8M7:
    return (sign << (enc.ExpBits + enc.ManBits)) |
           (enc.ExpMask << enc.ManBits) | enc.ManMask;
  case Encoding::IEEE754:
    return U32(sign ? std::copysign(ieee754_nan, -1.0F) : ieee754_nan);
  default:
    __builtin_trap();
    return 0;
  }
}

template <Encoding E, bool sat>
__OCP_FP_HOST_DEVICE_STATIC__ uint32_t zero(uint32_t sign) {
  const auto &enc = encodings[(size_t)E];

  switch (E) {
  case Encoding::E2M1:
  case Encoding::E2M3:
  case Encoding::E3M2:
  case Encoding::E4M3:
  case Encoding::E4M3Mx:
  case Encoding::E5M2:
  case Encoding::E5M2Mx:
  case Encoding::E5M10:
  case Encoding::E8M7:
    return (sign << (enc.ExpBits + enc.ManBits)) | 0;
  case Encoding::E4M3Nanoo:
  case Encoding::E5M2Nanoo:
    return 0;
  case Encoding::IEEE754:
    return U32(sign ? std::copysign(0.0F, -1.0F) : 0.0F);
  default:
    __builtin_trap();
    return 0;
  }
}

template <Encoding E, bool sat>
__OCP_FP_HOST_DEVICE_STATIC__ uint32_t inf(uint32_t sign) {
  const auto &enc = encodings[(size_t)E];

  switch (E) {
  case Encoding::E2M1:
  case Encoding::E2M3:
  case Encoding::E3M2:
    return nan<E, sat>(sign);
  case Encoding::E4M3:
  case Encoding::E4M3Mx:
  case Encoding::E4M3Nanoo:
  case Encoding::E5M2Nanoo:
    if constexpr (sat) {
      sign <<= enc.ExpBits + enc.ManBits;
      return sign | 0b01111111;
    }

    return nan<E, sat>(sign);
  case Encoding::E5M2:
  case Encoding::E5M2Mx:
    sign <<= enc.ExpBits + enc.ManBits;
    if constexpr (sat) {
      return sign | 0b01111011;
    }

    return sign | 0b01111100;
  case Encoding::E5M10:
  case Encoding::E8M7:
    sign <<= enc.ExpBits + enc.ManBits;
    return sign | (enc.ExpMask << enc.ManBits) | 0;
  case Encoding::IEEE754:
    return U32(sign ? std::copysign(ieee754_inf, -1.0F) : ieee754_inf);
  default:
    __builtin_trap();
    return 0;
  }
}

template <Encoding E, bool sat>
__OCP_FP_HOST_DEVICE_STATIC__ bool isnan(uint32_t val) {
  const auto &enc = encodings[(size_t)E];
  if (!enc.HasNaN)
    return false;

  if constexpr (E == Encoding::E4M3Mx || E == Encoding::E4M3 ||
                E == Encoding::E4M3Nanoo || E == Encoding::E5M2Nanoo)
    return nan<E, sat>(signbit<E, sat>(val)) == val;

  return exponentbits<E, sat>(val) == enc.ExpMask && mantissa<E, sat>(val) != 0;
}

template <Encoding E, bool sat>
__OCP_FP_HOST_DEVICE_STATIC__ bool isinf(uint32_t val) {
  const auto &enc = encodings[(size_t)E];
  if (!enc.HasInf)
    return false;

  if constexpr (E == Encoding::E5M10 || E == Encoding::E8M7) {
    return exponentbits<E, sat>(val) == enc.ExpMask &&
           mantissa<E, sat>(val) == 0;
  }

  return inf<E, sat>(signbit<E, sat>(val)) == val;
}

template <Encoding E, bool sat>
__OCP_FP_HOST_DEVICE_STATIC__ bool iszero(uint32_t val) {
  return zero<E, sat>(signbit<E, sat>(val)) == val;
}

template <Encoding E, bool sat>
__OCP_FP_HOST_DEVICE_STATIC__ bool inrange(uint32_t val) {
  return !(isnan<E, sat>(val) || isinf<E, sat>(val));
}

template <typename T>
__OCP_FP_HOST_DEVICE_STATIC__ T makenan(Encoding E, uint32_t sign) {
  switch (E) {
  // case Encoding::E5M10:
  //   //return (T)nan<Encoding::E5M10, false>(sign);
  //   if constexpr (std::is_same_v<T, __bf16>)
  //     return static_cast<T>(__float2bfloat16(
  //       static_cast<float>(nan<Encoding::E5M10, false>(sign))));
  //   else
  //     return (T)nan<Encoding::E5M10, false>(sign);
  // case Encoding::E8M7:
  //   //return (T)nan<Encoding::E8M7, false>(sign);
  //   if constexpr (std::is_same_v<T, __bf16>)
  //     return static_cast<T>(__float2bfloat16(
  //       static_cast<float>(nan<Encoding::E8M7, false>(sign))));
  //   else
  //     return (T)nan<Encoding::E8M7, false>(sign);
  case Encoding::IEEE754:
    return (T)F32(nan<Encoding::IEEE754, false>(sign));
  default:
    __builtin_trap();
    // Unreachable
    return T();
  }
}

template <typename T>
__OCP_FP_HOST_DEVICE_STATIC__ T makeinf(Encoding E, uint32_t sign) {
  switch (E) {
  // case Encoding::E5M10:
  //   //return (T)inf<Encoding::E5M10, false>(sign);
  //   if constexpr (std::is_same_v<T, __bf16>)
  //     return static_cast<T>(__float2bfloat16(
  //       static_cast<float>(inf<Encoding::E5M10, false>(sign))));
  //   else
  //     return (T)inf<Encoding::E5M10, false>(sign);
  // case Encoding::E8M7:
  //   //return (T)inf<Encoding::E8M7, false>(sign);
  //   if constexpr (std::is_same_v<T, __bf16>)
  //     return static_cast<T>(__float2bfloat16(
  //       static_cast<float>(inf<Encoding::E8M7, false>(sign))));
  //   else
  //     return (T)inf<Encoding::E8M7, false>(sign);
  case Encoding::IEEE754:
    return (T)F32(inf<Encoding::IEEE754, false>(sign));
  default:
    __builtin_trap();
    // Unreachable
    return T();
  }
}

template <typename T>
__OCP_FP_HOST_DEVICE_STATIC__ T makezero(Encoding E, uint32_t sign) {
  switch (E) {
  // case Encoding::E5M10:
  //   //return (T)zero<Encoding::E5M10, false>(sign);
  //   if constexpr (std::is_same_v<T, __bf16>)
  //     return static_cast<T>(__float2bfloat16(
  //       static_cast<float>(zero<Encoding::E5M10, false>(sign))));
  //   else
  //     return (T)zero<Encoding::E5M10, false>(sign);
  // case Encoding::E8M7:
  //   //return (T)zero<Encoding::E8M7, false>(sign);
  //   if constexpr (std::is_same_v<T, __bf16>)
  //     return static_cast<T>(__float2bfloat16(
  //       static_cast<float>(zero<Encoding::E8M7, false>(sign))));
  //   else
  //     return (T)zero<Encoding::E8M7, false>(sign);
  case Encoding::IEEE754:
    return (T)F32(zero<Encoding::IEEE754, false>(sign));
  default:
    __builtin_trap();
    // Unreachable
    return T();
  }
}

template <typename T, Encoding E, bool sat>
__OCP_FP_HOST_DEVICE_STATIC__ T to_float(uint32_t u32, uint8_t scale_exp) {
  // We do not support bf16/fp16 <-> float
  static_assert(E != Encoding::IEEE754 && E != Encoding::E5M10 &&
                E != Encoding::E8M7);

  const auto &enc = encodings[(size_t)E];
  const auto dstE = []() -> Encoding {
    if constexpr (std::is_same<T, float>())
      return Encoding::IEEE754;
    else if constexpr (std::is_same<T, fp16_storage_t>())
      return Encoding::E5M10;
    else if constexpr (std::is_same<T, bf16_storage_t>())
      return Encoding::E8M7;
    else
      __builtin_trap();
  }();
  const auto &dstEnc = encodings[(size_t)dstE];

  if (isnan<E, sat>(u32) || (enc.MxScale && scale_exp == OCP_SCALE_EXP_NAN))
    return makenan<T>(dstE, signbit<E, sat>(u32));

  if (isinf<E, sat>(u32))
    return makeinf<T>(dstE, signbit<E, sat>(u32));

  if (iszero<E, sat>(u32))
    return makezero<T>(dstE, signbit<E, sat>(u32));

  auto dstMan = mantissa<E, sat>(u32) << (dstEnc.ManBits - enc.ManBits);
  auto dstExp = (uint32_t)(exponent<E, sat>(u32) + dstEnc.ExpBias);
  dstExp &= dstEnc.ExpMask;

  if (issubnorm<E, sat>(u32)) {
    auto leadbit = (uint32_t)1 << dstEnc.ManBits;
    while ((dstMan & leadbit) == 0) {
      dstMan <<= 1;
      dstExp -= 1;
    }

    dstMan &= dstEnc.ManMask;
  }

  auto sign = signbit<E, sat>(u32) << (dstEnc.ExpBits + dstEnc.ManBits);

  if (enc.MxScale) {
    int32_t exp = dstExp - dstEnc.ExpBias;
    int32_t tmp = exp + (int32_t)(scale_exp - 0x7f);
    size_t diff = abs(tmp - dstEnc.MinExp);

    if (tmp < dstEnc.MinExp) {
      // if (diff > dstEnc.ManBits + 1) return makezero<T>(dstE, signbit<E,
      // sat>(u32));

      dstExp = 0; // Subnormal
      dstMan |= (uint32_t)1 << dstEnc.ManBits;

      auto roundBitShift = diff - 1;
      auto roundBit = (dstMan & ((uint32_t)1 << roundBitShift)) != 0;
      auto stickyMask = ((uint32_t)1 << roundBitShift) - 1;
      auto stickyBits = dstMan & stickyMask;
      auto odd = (dstMan & ((uint32_t)1 << diff)) != 0;

      dstMan >>= diff;

      if ((roundBit && stickyBits != 0) || (roundBit && odd)) {
        ++dstMan;
        if ((dstMan & ((uint32_t)1 << dstEnc.ManBits)) != 0)
          ++dstExp;
      }

      dstMan &= dstEnc.ManMask;
    } else {
      dstExp = (uint32_t)(exp + (scale_exp - 0x7f) + dstEnc.ExpBias);

      // Overflow: return infinity (gfx950 HW behavior)
      if (dstExp >= dstEnc.ExpMask)
        return makeinf<T>(dstE, signbit<E, sat>(u32));

      dstExp &= dstEnc.ExpMask;
    }
  }

  auto dst = sign | (dstExp << dstEnc.ManBits) | dstMan;

  union {
    float f32;
    fp16_storage_t fp16[2];
    bf16_storage_t bf16[2];
    uint32_t u32;
  } u;
  u.u32 = dst;
  if constexpr (std::is_same<T, float>())
    return u.f32;
  else if constexpr (std::is_same<T, fp16_storage_t>())
    return u.fp16[0];
  else if constexpr (std::is_same<T, bf16_storage_t>())
    return u.bf16[0];
  else
    __builtin_trap();
}

template <typename T, Encoding E, bool sat>
__OCP_FP_HOST_DEVICE_STATIC__ uint32_t from_float_sr(T f,
                                                     uint32_t seed,
                                                     uint8_t scale_exp = 0x7f) {
  // We do not support bf16/fp16 <-> float
  static_assert(E != Encoding::IEEE754 && E != Encoding::E5M10 &&
                E != Encoding::E8M7);
  static_assert(sizeof(fp16_storage_t[2]) == sizeof(float));
  static_assert(sizeof(bf16_storage_t[2]) == sizeof(float));
  union {
    float f32;
    uint16_t f16[2];
    uint32_t u32;
  } u;

  if constexpr (std::is_same<T, float>())
    u.f32 = f;
  else if constexpr (std::is_same<T, float16_t>() ||
                     std::is_same<T, bfloat16_t>())
    u.f16[0] = *(uint16_t*)(&f);
  else
    __builtin_trap();

  const auto &enc = encodings[(size_t)E];
  const auto srcE = []() -> Encoding {
    if constexpr (std::is_same<T, float>())
      return Encoding::IEEE754;
    else if constexpr (std::is_same<T, float16_t>())
      return Encoding::E5M10;
    else if constexpr (std::is_same<T, bfloat16_t>())
      return Encoding::E8M7;
    else
      __builtin_trap();
  }();
  const auto &srcEnc = encodings[(size_t)srcE];

  auto srcU32 = u.u32; // (srcE == Encoding::IEEE754) ? U32(f) : (uint32_t)f;
  auto signBit = signbit<srcE, false>(srcU32);
  auto sign = signBit << (enc.ExpBits + enc.ManBits);

  if (isnan<srcE, sat>(srcU32) ||
      (enc.MxScale && scale_exp == OCP_SCALE_EXP_NAN))
    return nan<E, sat>(signBit);

  if (isinf<srcE, sat>(srcU32))
    return inf<E, sat>(signBit);

  if (iszero<srcE, sat>(srcU32))
    return zero<E, sat>(signBit);

  auto srcMan = mantissa<srcE, false>(srcU32);
  auto srcExp = exponent<srcE, false>(srcU32);
  if (enc.MxScale) {
    if (issubnorm<srcE, false>(srcU32)) {
      auto leadbit = (uint32_t)1 << srcEnc.ManBits;
      while ((srcMan & leadbit) == 0) {
        srcMan <<= 1;
        srcExp -= 1;
      }

      srcMan &= srcEnc.ManMask;
    }

    // for HCU, scale_exp is biased, need non-biased scale_exp here
    srcExp -= (scale_exp - 0x7f);
  }

  auto exp = srcExp;
  auto man = srcMan;
  bool subnorm = false;

  if (exp > enc.MaxExp) {
    return inf<E, sat>(signBit);
  } else if (exp >= enc.MinExp) {
    man = srcMan;
  } else if (exp < enc.MinExp) {
    subnorm = true;
    exp = 0;

    auto diff = (uint32_t)(enc.MinExp - srcExp);
    if (diff >= 32) {
      man = 0;
      srcMan = 0;
    } else {
      srcMan |= (uint32_t)1 << srcEnc.ManBits;
      srcMan >>= diff;
    }

    man = srcMan;
  }

  // For stochastic-rounding we add the aligned random value to the
  // mantissa, and then truncate (RTZ).
  uint32_t manMask = (1 << (srcEnc.ManBits - enc.ManBits)) - 1;
  man += (seed & manMask);

  man >>= (srcEnc.ManBits - enc.ManBits);

  // Increment exponent when mantissa overflows due to rounding
  if (man >= (uint32_t)1 << enc.ManBits)
    ++exp;
  man &= enc.ManMask;

  if (exp > enc.MaxExp)
    return inf<E, sat>(signBit);

  auto biasedExp = (uint32_t)exp;
  if (!subnorm)
    biasedExp = (uint32_t)(exp + enc.ExpBias);
  biasedExp &= enc.ExpMask;

  auto val = sign | biasedExp << enc.ManBits | man;
  if (inrange<E, sat>(val))
    return val;
  else if (man == 0 && exp == 0)
    return zero<E, sat>(signBit);
  else
    return inf<E, sat>(signBit);
}

template <typename T, Encoding E, bool sat>
__OCP_FP_HOST_DEVICE_STATIC__ uint32_t from_float(T f,
                                                  uint8_t scale_exp = 0x7f,
                                                  hipRoundMode rounding = hipRoundNearest) {
  // We do not support bf16/fp16 <-> float
  static_assert(E != Encoding::IEEE754 && E != Encoding::E5M10 &&
                E != Encoding::E8M7);
  static_assert(sizeof(fp16_storage_t[2]) == sizeof(float));
  static_assert(sizeof(bf16_storage_t[2]) == sizeof(float));
  union {
    float f32;
    uint16_t f16[2];
    uint32_t u32;
  } u;

  if constexpr (std::is_same<T, float>())
    u.f32 = f;
  else if constexpr (std::is_same<T, float16_t>() ||
                     std::is_same<T, bfloat16_t>())
    u.f16[0] = *(uint16_t*)(&f);
  else
    __builtin_trap();

  const auto &enc = encodings[(size_t)E];
  const auto srcE = []() -> Encoding {
    if constexpr (std::is_same<T, float>())
      return Encoding::IEEE754;
    else if constexpr (std::is_same<T, fp16_storage_t>())
      return Encoding::E5M10;
    else if constexpr (std::is_same<T, bf16_storage_t>())
      return Encoding::E8M7;
    else
      __builtin_trap();
  }();
  const auto &srcEnc = encodings[(size_t)srcE];

  auto srcU32 = u.u32; // (srcE == Encoding::IEEE754) ? U32(f) : (uint32_t)f;
  auto signBit = signbit<srcE, false>(srcU32);
  auto sign = signBit << (enc.ExpBits + enc.ManBits);

  if (isnan<srcE, sat>(srcU32) ||
      (enc.MxScale && scale_exp == OCP_SCALE_EXP_NAN))
    return nan<E, sat>(signBit);

  if (isinf<srcE, sat>(srcU32))
    return inf<E, sat>(signBit);

  if (iszero<srcE, sat>(srcU32))
    return zero<E, sat>(signBit);

  auto srcMan = mantissa<srcE, false>(srcU32);
  auto srcExp = exponent<srcE, false>(srcU32);
  if (enc.MxScale) {
    if (issubnorm<srcE, false>(srcU32)) {
      auto leadbit = (uint32_t)1 << srcEnc.ManBits;
      while ((srcMan & leadbit) == 0) {
        srcMan <<= 1;
        srcExp -= 1;
      }

      srcMan &= srcEnc.ManMask;
    }

    // for HCU, scale_exp is biased, need non-biased scale_exp here
    srcExp -= (scale_exp - 0x7f);
  }

  auto exp = srcExp;
  auto man = srcMan;
  uint32_t stickyBits = 0;
  bool subnorm = false;

  if (exp > enc.MaxExp) {
    return inf<E, sat>(signBit);
  } else if (exp >= enc.MinExp) {
    man >>= srcEnc.ManBits - enc.ManBits;
  } else if (exp < enc.MinExp) {
    subnorm = true;
    exp = 0;

    auto diff = (uint32_t)(enc.MinExp - srcExp);
    if (diff >= 32) {
      man = 0;
      srcMan = 0;
    } else {
      srcMan |= (uint32_t)1 << srcEnc.ManBits;
      stickyBits = srcMan & (((uint32_t)1 << diff) - (uint32_t)1);
      srcMan >>= diff;

      man = srcMan;
      man >>= srcEnc.ManBits - enc.ManBits;
      man &= enc.ManMask;
    }
  }

  auto roundBitShift = srcEnc.ManBits - (enc.ManBits + 1);
  auto roundBit = ((srcMan >> roundBitShift) & 1) != 0;
  stickyBits |= srcMan & (((uint32_t)1 << roundBitShift) - 1);
  auto odd = (man & 1) != 0;

  bool need_round_up = false;

  switch (rounding) {
    case hipRoundNearest:
      need_round_up = (roundBit && stickyBits != 0) || (roundBit && odd);
      break;

    case hipRoundZero:
      need_round_up = false;
      break;

    default:
    __builtin_trap();
      break;
  }

  if (need_round_up) {
    ++man;
    if ((man & ((uint32_t)1 << enc.ManBits)) != 0) {
      ++exp;
      man &= enc.ManMask;
    }
  }

  if (exp > enc.MaxExp)
    return inf<E, sat>(signBit);

  auto biasedExp = (uint32_t)exp;
  if (!subnorm)
    biasedExp = (uint32_t)(exp + enc.ExpBias);
  biasedExp &= enc.ExpMask;

  auto val = sign | biasedExp << enc.ManBits | man;
  if (inrange<E, sat>(val))
    return val;
  else if (man == 0 && exp == 0)
    return zero<E, sat>(signBit);
  else
    return inf<E, sat>(signBit);
}

template <typename InType, typename OutType, typename float_base_t,
          Encoding in_encode, Encoding out_encode, bool sr = false>
__OCP_FP_HOST_DEVICE_STATIC__ OutType fp6_cvt_packedx32(InType in,
                                                        uint8_t scale = 0x7f,
                                                        uint32_t seed = 0) {
  // This is tightly coupled with the definitions of the amd_ocp_types
  constexpr bool in_float = std::is_same<InType, floatx32_storage_t>::value ||
                            std::is_same<InType, fp16x32_storage_t>::value ||
                            std::is_same<InType, bf16x32_storage_t>::value;
  constexpr bool out_float = std::is_same<OutType, floatx32_storage_t>::value ||
                             std::is_same<OutType, fp16x32_storage_t>::value ||
                             std::is_same<OutType, bf16x32_storage_t>::value;
  using other_type = typename std::conditional<in_float, OutType, InType>::type;

  struct fp6x32_packed {
    uint8_t val1 : 6;
    uint8_t val2 : 6;
    uint8_t val3 : 6;
    uint8_t val4 : 6;
    uint8_t val5 : 6;
    uint8_t val6 : 6;
    uint8_t val7 : 6;
    uint8_t val8 : 6;
    uint8_t val9 : 6;
    uint8_t val10 : 6;
    uint8_t val11 : 6;
    uint8_t val12 : 6;
    uint8_t val13 : 6;
    uint8_t val14 : 6;
    uint8_t val15 : 6;
    uint8_t val16 : 6;
    uint8_t val17 : 6;
    uint8_t val18 : 6;
    uint8_t val19 : 6;
    uint8_t val20 : 6;
    uint8_t val21 : 6;
    uint8_t val22 : 6;
    uint8_t val23 : 6;
    uint8_t val24 : 6;
    uint8_t val25 : 6;
    uint8_t val26 : 6;
    uint8_t val27 : 6;
    uint8_t val28 : 6;
    uint8_t val29 : 6;
    uint8_t val30 : 6;
    uint8_t val31 : 6;
    uint8_t val32 : 6;
    unsigned long long padded;
  } __attribute__((packed));

  static_assert(sizeof(other_type) == sizeof(fp6x32_packed));
  union {
    other_type o;
    fp6x32_packed fp6;
  } u;

  // TODO maybe make it simpler
  if constexpr (in_float) {
    if constexpr (sr) {
      u.fp6.val1 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[0], seed, scale));
      u.fp6.val2 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[1], seed, scale));
      u.fp6.val3 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[2], seed, scale));
      u.fp6.val4 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[3], seed, scale));
      u.fp6.val5 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[4], seed, scale));
      u.fp6.val6 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[5], seed, scale));
      u.fp6.val7 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[6], seed, scale));
      u.fp6.val8 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[7], seed, scale));
      u.fp6.val9 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[8], seed, scale));
      u.fp6.val10 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[9], seed, scale));
      u.fp6.val11 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[10], seed, scale));
      u.fp6.val12 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[11], seed, scale));
      u.fp6.val13 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[12], seed, scale));
      u.fp6.val14 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[13], seed, scale));
      u.fp6.val15 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[14], seed, scale));
      u.fp6.val16 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[15], seed, scale));
      u.fp6.val17 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[16], seed, scale));
      u.fp6.val18 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[17], seed, scale));
      u.fp6.val19 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[18], seed, scale));
      u.fp6.val20 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[19], seed, scale));
      u.fp6.val21 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[20], seed, scale));
      u.fp6.val22 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[21], seed, scale));
      u.fp6.val23 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[22], seed, scale));
      u.fp6.val24 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[23], seed, scale));
      u.fp6.val25 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[24], seed, scale));
      u.fp6.val26 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[25], seed, scale));
      u.fp6.val27 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[26], seed, scale));
      u.fp6.val28 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[27], seed, scale));
      u.fp6.val29 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[28], seed, scale));
      u.fp6.val30 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[29], seed, scale));
      u.fp6.val31 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[30], seed, scale));
      u.fp6.val32 = static_cast<uint8_t>(
          from_float_sr<float_base_t, out_encode, true>(in[31], seed, scale));
    } else {
      u.fp6.val1 = from_float<float_base_t, out_encode, true>(in[0], scale);
      u.fp6.val2 = from_float<float_base_t, out_encode, true>(in[1], scale);
      u.fp6.val3 = from_float<float_base_t, out_encode, true>(in[2], scale);
      u.fp6.val4 = from_float<float_base_t, out_encode, true>(in[3], scale);
      u.fp6.val5 = from_float<float_base_t, out_encode, true>(in[4], scale);
      u.fp6.val6 = from_float<float_base_t, out_encode, true>(in[5], scale);
      u.fp6.val7 = from_float<float_base_t, out_encode, true>(in[6], scale);
      u.fp6.val8 = from_float<float_base_t, out_encode, true>(in[7], scale);
      u.fp6.val9 = from_float<float_base_t, out_encode, true>(in[8], scale);
      u.fp6.val10 = from_float<float_base_t, out_encode, true>(in[9], scale);
      u.fp6.val11 = from_float<float_base_t, out_encode, true>(in[10], scale);
      u.fp6.val12 = from_float<float_base_t, out_encode, true>(in[11], scale);
      u.fp6.val13 = from_float<float_base_t, out_encode, true>(in[12], scale);
      u.fp6.val14 = from_float<float_base_t, out_encode, true>(in[13], scale);
      u.fp6.val15 = from_float<float_base_t, out_encode, true>(in[14], scale);
      u.fp6.val16 = from_float<float_base_t, out_encode, true>(in[15], scale);
      u.fp6.val17 = from_float<float_base_t, out_encode, true>(in[16], scale);
      u.fp6.val18 = from_float<float_base_t, out_encode, true>(in[17], scale);
      u.fp6.val19 = from_float<float_base_t, out_encode, true>(in[18], scale);
      u.fp6.val20 = from_float<float_base_t, out_encode, true>(in[19], scale);
      u.fp6.val21 = from_float<float_base_t, out_encode, true>(in[20], scale);
      u.fp6.val22 = from_float<float_base_t, out_encode, true>(in[21], scale);
      u.fp6.val23 = from_float<float_base_t, out_encode, true>(in[22], scale);
      u.fp6.val24 = from_float<float_base_t, out_encode, true>(in[23], scale);
      u.fp6.val25 = from_float<float_base_t, out_encode, true>(in[24], scale);
      u.fp6.val26 = from_float<float_base_t, out_encode, true>(in[25], scale);
      u.fp6.val27 = from_float<float_base_t, out_encode, true>(in[26], scale);
      u.fp6.val28 = from_float<float_base_t, out_encode, true>(in[27], scale);
      u.fp6.val29 = from_float<float_base_t, out_encode, true>(in[28], scale);
      u.fp6.val30 = from_float<float_base_t, out_encode, true>(in[29], scale);
      u.fp6.val31 = from_float<float_base_t, out_encode, true>(in[30], scale);
      u.fp6.val32 = from_float<float_base_t, out_encode, true>(in[31], scale);
    }
    return u.o;
  } else {
    OutType ret;
    u.o = in;
    ret[0] = to_float<float_base_t, in_encode, true>(u.fp6.val1, scale);
    ret[1] = to_float<float_base_t, in_encode, true>(u.fp6.val2, scale);
    ret[2] = to_float<float_base_t, in_encode, true>(u.fp6.val3, scale);
    ret[3] = to_float<float_base_t, in_encode, true>(u.fp6.val4, scale);
    ret[4] = to_float<float_base_t, in_encode, true>(u.fp6.val5, scale);
    ret[5] = to_float<float_base_t, in_encode, true>(u.fp6.val6, scale);
    ret[6] = to_float<float_base_t, in_encode, true>(u.fp6.val7, scale);
    ret[7] = to_float<float_base_t, in_encode, true>(u.fp6.val8, scale);
    ret[8] = to_float<float_base_t, in_encode, true>(u.fp6.val9, scale);
    ret[9] = to_float<float_base_t, in_encode, true>(u.fp6.val10, scale);
    ret[10] = to_float<float_base_t, in_encode, true>(u.fp6.val11, scale);
    ret[11] = to_float<float_base_t, in_encode, true>(u.fp6.val12, scale);
    ret[12] = to_float<float_base_t, in_encode, true>(u.fp6.val13, scale);
    ret[13] = to_float<float_base_t, in_encode, true>(u.fp6.val14, scale);
    ret[14] = to_float<float_base_t, in_encode, true>(u.fp6.val15, scale);
    ret[15] = to_float<float_base_t, in_encode, true>(u.fp6.val16, scale);
    ret[16] = to_float<float_base_t, in_encode, true>(u.fp6.val17, scale);
    ret[17] = to_float<float_base_t, in_encode, true>(u.fp6.val18, scale);
    ret[18] = to_float<float_base_t, in_encode, true>(u.fp6.val19, scale);
    ret[19] = to_float<float_base_t, in_encode, true>(u.fp6.val20, scale);
    ret[20] = to_float<float_base_t, in_encode, true>(u.fp6.val21, scale);
    ret[21] = to_float<float_base_t, in_encode, true>(u.fp6.val22, scale);
    ret[22] = to_float<float_base_t, in_encode, true>(u.fp6.val23, scale);
    ret[23] = to_float<float_base_t, in_encode, true>(u.fp6.val24, scale);
    ret[24] = to_float<float_base_t, in_encode, true>(u.fp6.val25, scale);
    ret[25] = to_float<float_base_t, in_encode, true>(u.fp6.val26, scale);
    ret[26] = to_float<float_base_t, in_encode, true>(u.fp6.val27, scale);
    ret[27] = to_float<float_base_t, in_encode, true>(u.fp6.val28, scale);
    ret[28] = to_float<float_base_t, in_encode, true>(u.fp6.val29, scale);
    ret[29] = to_float<float_base_t, in_encode, true>(u.fp6.val30, scale);
    ret[30] = to_float<float_base_t, in_encode, true>(u.fp6.val31, scale);
    ret[31] = to_float<float_base_t, in_encode, true>(u.fp6.val32, scale);
    return ret;
  }
}
} // namespace fcbx