#pragma once

#include "type/OCP_Common.h"

#define __HOST_DEVICE__ __host__ __device__
#define __HOST_DEVICE_STATIC__ __HOST_DEVICE__ static inline

/*
  On HCU:
    The "scale" operand is encoded in the exponent bits of an IEEE-754 float: f32_exp = scale.
    The range of 8-bit "scale" is [0:255]
*/
__HOST_DEVICE_STATIC__ float scale_to_float(const scale_t scale_exp) {
  const size_t SCALE_EXP_SHIFT = 23; // IEEE-754 "float" mantissa bits

  if (scale_exp == fcbx::OCP_SCALE_EXP_NAN)
    return fcbx::ieee754_nan;

  return fcbx::F32((scale_exp & 0xff) << SCALE_EXP_SHIFT);
}

__HOST_DEVICE_STATIC__ scale_t float_to_scale(const float value, hipRoundMode rounding) {
  const size_t SCALE_EXP_SHIFT = 23; // IEEE-754 "float" mantissa bits

  if (value == fcbx::ieee754_nan)
    return fcbx::OCP_SCALE_EXP_NAN;

  uint8_t res = uint8_t((*(uint32_t*)&value & 0x7f800000) >> SCALE_EXP_SHIFT);
  if (rounding == hipRoundMode::hipRoundPosInf && (*(uint32_t*)&value & 0x7fffff) != 0)
    res += 1;

  return res;
}