#ifndef TYPERNG_H
#define TYPERNG_H

#pragma once

#include "type/DataTypes_BFloat16.hpp"
#include "type/DataTypes_FP4.hpp"
#include "type/DataTypes_FP6.hpp"
#include "type/DataTypes_FP8.hpp"
#include "type/DataTypes_FP8_Scale.hpp"
#include "type/DataTypes_Int4x2.hpp"
#include "type/DataTypes_TFloat32.hpp"
#include "type/DataTypes_Uint4x2.hpp"

#include "type/half.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

#include <hip/hip_runtime.h>



float rng(float min, float max);
int rng(int min, int max);
template <typename T>
T rng() { return static_cast<T>(rng(0.0f, 1.0f)); }
template <> int32_t rng();
template <> uint32_t rng();
template <> int8_t rng();
template <> uint8_t rng();
template <> Int4x2 rng();
template <> Uint4x2 rng();
template <> float8_t rng();
template <> bfloat8_t rng();
template <> float6_t rng();
template <> bfloat6_t rng();
template <> float4_t rng();
template <> float4x2_t rng();
template <> int16_t rng();
template <> uint16_t rng();



/* Random Number Generator */
float rng(float min, float max) {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * (max - min) + min;
}
int rng(int min, int max) {
    return min+rand()%(max-min+1);
}

template <> int32_t rng() { return rng(-100, 100); }
template <> uint32_t rng() { return rng(0, 200); }
template <> int8_t rng() { return rng(-128, 127); }
template <> uint8_t rng() { return rng(0, 255); }
template <> Int4x2 rng() { return Int4x2(rng(-8, 7), rng(-8, 7)); }
template <> Uint4x2 rng() { return Uint4x2(rng(0, 15), rng(0, 15)); }
template <> float8_t rng() {
    return rng(0.f, 1.f);
}
template <> bfloat8_t rng() {
    return rng(0.f, 1.f);
}
template <> float6_t rng() { return float6_t(rng(0.f, 1.f)); }
template <> bfloat6_t rng() { return bfloat6_t(rng(0.f, 1.f)); }
template <> float4_t rng() { return float4_t(rng(0.f, 1.f)); }
template <> float4x2_t rng() {
    return float4x2_t(rng(0.f, 1.f), rng(0.f, 1.f));
}
template <> int16_t rng() { return rng(-100, 100); }
template <> uint16_t rng() { return rng(0, 200); }

#endif