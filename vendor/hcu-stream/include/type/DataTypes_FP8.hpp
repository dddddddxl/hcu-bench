#pragma once

#include <cmath>
#include <cfloat>
#include <limits>
#include <random>
#include <cassert>
#include <iostream>

#include "OCP_Common.h"

#ifndef __host__
#define __host__
#define __device__
#endif

/*

DEFINITION:
    E4M3:
        float8_t
        FP8_E4M3
        float_e4m3_t
    E5M2:
        bfloat8_t
        FP8_E5M2
        float_e5m2_t

TODO:
    1. [√] exception handling, sNAN and qNAN...
    2. [√] stochastic rounding
    2. [ ] (optional)cvt between half and fp8, inplemented in ck and cutlass(in cutlass, fp8 -> float -> half and half -> float -> fp8)
    3. [ ] (optional)fp8 mmac inplementation
    4. [√] debug: fp32 -> fp8/bf8, norm -> denorm needs shift and thus discard mantissas and cause the output tend to be smaller
    5. [√] debug: some numbers exist in theory but doesn't exist in IEEE fp32 formats, such as 57344.001

TEST:
    http://172.18.38.11/display/DW/FP8+test
*/


// some const variables and func using when type conversion
namespace {

// fp8_e5m2
constexpr int32_t e5m2_exp_bits  = 5;
constexpr int32_t e5m2_mant_bits = 2;
constexpr int32_t e5m2_bits      = 8;
constexpr int32_t e5m2_bias      = (1 << (e5m2_exp_bits - 1)) - 1;

// fp8_e4m3
constexpr int32_t e4m3_exp_bits  = 4;
constexpr int32_t e4m3_mant_bits = 3;
constexpr int32_t e4m3_bits      = 8;
constexpr int32_t e4m3_bias      = (1 << (e4m3_exp_bits - 1)) - 1;

// fp16
constexpr int32_t fp16_exp_bits  = 5;
constexpr int32_t fp16_mant_bits = 10;
constexpr int32_t fp16_bits      = 16;
constexpr int32_t fp16_bias      = (1 << (fp16_exp_bits - 1)) - 1;

// fp32
constexpr int32_t fp32_exp_bits  = 8;
constexpr int32_t fp32_mant_bits = 23;
constexpr int32_t fp32_bits      = 32;
constexpr int32_t fp32_bias      = (1 << (fp32_exp_bits - 1)) - 1;

// functions to debug data with 8bits or 16bits
static constexpr uint32_t BITS_TEMPLATE[32] = {
    0x80000000, 0x40000000, 0x20000000, 0x10000000,
    0x8000000,  0x4000000,  0x2000000,  0x1000000,
    0x800000,   0x400000,   0x200000,   0x100000,
    0x80000,    0x40000,    0x20000,    0x10000,
    0x8000,     0x4000,     0x2000,     0x1000,
    0x800,      0x400,      0x200,      0x100,
    0x80,       0x40,       0x20,       0x10,
    0x8,        0x4,        0x2,        0x1
};

std::string fp8_bits(const uint8_t src) {
    std::string res("");
    for (uint32_t i = 0; i < 8; ++i) {
        res += src & BITS_TEMPLATE[24 + i] ? "1": "0";
    }
    return res;
}

}

// layout definition
enum class FP8_LAYOUT {
    E5M2,
    E4M3
};


// some settings to influence the behavior of conversion
enum class FP8_CVT_CONFIGS_ITEM {
    /* inst.clamp is off defaultly */
    INST_CLAMP_ON,
    INST_CLAMP_OFF,
    /* mode.dx10_clamp, may change the exception flags even if inst.clamp is off */
    DX10_CLAMP_ON,
    DX10_CLAMP_OFF,
    /* dx10_clamp + inst.clamp */
    DX10_INST_CLAMP_ON,
    /* mode2.{fp,bf}8_ovfl, controls the saturate mode, default on */
    OVERFLOW_ON,
    OVERFLOW_OFF,
    /* rounding mode, only "rounding to nearest even" and "stochastic rounding" is supported yet */
    ROUND_NEAREST,
    ROUND_STOCHASTIC,
    ROUND_ZERO,
    /* denorm mode is on defaultly*/
    DENORM_INPUT_ON,
    DENORM_INPUT_OFF,
    F32_DENORM_OUTPUT_ON,
    F32_DENORM_OUTPUT_OFF,
    /* ieee mode, for bf8, can be turned on; for fp8, off */
    IEEE_MODE_ON,
    IEEE_MODE_OFF,
    // omod
    OUTPUT_MODIFIER_OFF,
    OUTPUT_MODIFIER_MUL2,
    OUTPUT_MODIFIER_MUL4,
    OUTPUT_MODIFIER_DIV2,
    // input modifier
    INPUT_MODIFIER_OFF,
    INPUT_MODIFIER_ABS,
    INPUT_MODIFIER_NEG,
};
namespace {
    using CFG = FP8_CVT_CONFIGS_ITEM;
}

// struct to carry all possible settings while pass parameters
struct FP8_CVT_CONFIGS {
    CFG clamp_mode;
    CFG saturate_mode;
    CFG round_mode;
    CFG denorm_mode;
    CFG f32_denorm_mode;
    CFG ieee_mode;
    CFG output_modifier;
    CFG input_modifier;
    uint32_t random_value;
    FP8_CVT_CONFIGS(
        CFG a = CFG::INST_CLAMP_OFF,
        CFG b = CFG::OVERFLOW_ON,
        CFG c = CFG::ROUND_NEAREST,
        CFG d = CFG::DENORM_INPUT_ON,
        CFG e = CFG::F32_DENORM_OUTPUT_ON,
        CFG f = CFG::IEEE_MODE_ON,
        CFG g = CFG::OUTPUT_MODIFIER_OFF,
        CFG h = CFG::INPUT_MODIFIER_OFF)
    : clamp_mode(a), saturate_mode(b), round_mode(c), denorm_mode(d), f32_denorm_mode(e),
          ieee_mode(f), output_modifier(g), input_modifier(h) {}
};
namespace {  //namespace>
    FP8_CVT_CONFIGS default_config;
}

// add a global configuration for saturation, to better handle inf
__attribute__((weak)) bool __forbidden_inf = true;
__attribute__((weak)) bool forbidden_inf() {
    return __forbidden_inf;
}
struct ALLOW_FP8_INF {
    ALLOW_FP8_INF() {
        __forbidden_inf = false;
        printf("\x1b[31mallow inf of FP8\033[0m\n");
    }
    ~ALLOW_FP8_INF() {
        __forbidden_inf = true;
        printf("\x1b[32mforbidden inf of FP8\033[0m\n");
    }
};


/* conversion: float -> fp8

    reference:
        1. https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1
        2. DCU BWb0 SP Delta Spec v0.7
        3. (optional)https://github.com/ROCm/composable_kernel/blob/develop/include/ck/utility/type_convert.hpp

    detail:
            1. support saturation
            2. support denormals
            3. support output clamp
            4. support input modifier

    concepts:
        1. modifier: -, ||, sdwa, op_sel and so on
        2. saturation: clamp output to between +-MAX
        3. denormals: input can be denormal, but output denorms is decided by "dmode_input" signal, for fp8/bf8, cvt instructions all support denormals, cannot be changed, refered to 66th page in spec
        4. output clamp:
            ①inst.clamp: for floats data, clamp data to [0, 1], except for NAN
            ②mode.dx10_clamp: bit field in mode registers, works when inst.clamp is on, which can cvt NAN -> 0
            especially, inst.clamp has higher priority than {fp,bf}8_overflow
        5. exception flags: output as the output of hardware modules to report exceptions, and thus generate some bit fields of [5:0], such as overflow、precision loss、input denomals、NAN and so on
        6. rounding mode: in fp8, only "rounding to nearest even" is supported
        7. IEEE mode: cvt instructions of bf8 support IEEE, since fp8 has no inf
        8. qNAN: quiet NAN, used for undefined arithmetic results(i.e., div 0), highest bit in mantissa is 1
        9. sNAN: signaling NAN, used for un-initilized value, highest bit in mantissa is 0
*/
template<FP8_LAYOUT LAYOUT>
__host__ __device__
uint8_t __float2fp8(const float src, FP8_CVT_CONFIGS &config=default_config, uint32_t random_value = 0) {

    // initialize ret value
    uint8_t result = 0x0;
    uint32_t __src;
    __src = *(unsigned int*)&src;
    // input modifier
    if (config.input_modifier == CFG::INPUT_MODIFIER_ABS)
        __src = *(unsigned int*)&src & 0x7fffffff;
    if (config.input_modifier == CFG::INPUT_MODIFIER_NEG)
        __src = *(unsigned int*)&src ^ 0x80000000;

    // fetch sign bits
    uint8_t sign_bits = (__src & 0x80000000) >> (fp32_bits - e5m2_bits);
    // fetch exponent bitss
    uint32_t exp_bits  = __src & 0x7f800000;
    // fetch mantissa bits
    uint32_t mant_bits = __src & 0x007fffff;
    // fetch absolute value
    uint32_t data_scale = __src & 0x7fffffff;

    bool use_clamp = (config.clamp_mode == CFG::INST_CLAMP_ON or config.clamp_mode == CFG::DX10_INST_CLAMP_ON);

    // categorical discussions
    if constexpr (LAYOUT == FP8_LAYOUT::E5M2) {
        /*
            ①. input: sNAN or qNAN,         output: +0(dx10_clamp+inst.clamp) or qNAN
            ②. input: inf,                  output: 1/+0(inst.clamp) or MAX(saturate) or inf
            ③. input: greater than MAX,     output: 1(inst.clamp, >MAX), 0(inst.clamp, <-MAX) or MAX(saturate) or inf
            ④. input: less than denorm MIN, output: 0(dmode_input off) or min denorm(use rounding to nearest even) or +/-0
            ⑤. input: less than norm MIN,   output: denorm bf8
            ⑥. input: norm,                 output: norm bf8
            ⑦. input: denorm,               output: denorm of fp32 is to small, talked in ④
        */
        /* NAN */
        if (exp_bits == 0x7f800000 and mant_bits > 0x0) {
            if (config.clamp_mode == CFG::DX10_INST_CLAMP_ON) {
                result = 0x0;              // output +0
            }
            else {
                result = sign_bits | 0x7c | mant_bits >> (fp32_mant_bits - e5m2_mant_bits) | 0x2; // sNAN -> qNAN, first MSB is set 1
            }
        }
        /* inf or greater than MAX value of E5M2*/
        else if ((exp_bits == 0x7f800000 and mant_bits == 0x0) or (data_scale > 0x47600000)) {
            if (use_clamp) {
                result = sign_bits ? 0x0: 0x3c; // clamp output into [0,1]
            }
            else if (config.saturate_mode == CFG::OVERFLOW_ON and forbidden_inf()) {
                result = sign_bits | 0x7b; // output MAX
            }
            else {
                result = sign_bits | 0x7c; // output inf
            }
        }
        /* less than MIN of denorm */
        else if (exp_bits <= 0x37000000) {
            if (config.denorm_mode == CFG::DENORM_INPUT_OFF and (exp_bits == 0x0) and mant_bits) {
                result = 0x0;              // flush input denormals to 0, including sign bits
            }
            else if (exp_bits == 0x37000000 and mant_bits > 0x0) {
                result = sign_bits | 0x1;  // output the min of denorm, essentially rounding to nearest even
                result = (use_clamp and (sign_bits > 0x0)) ? 0x0: result; // once inst.clamp, -denorm output +0, else keep -min denorm
            }
            else {
                result = use_clamp ? 0x0: sign_bits; // output +/-0
            }
        }
        /* others */
        else {
            /* norm f32 can be represented as denorm bf8 */
            if (exp_bits < 0x38800000) {
                uint32_t shift = (0x38800000 - exp_bits) >> fp32_mant_bits;
                mant_bits = (0x800000 | mant_bits) >> shift;
                exp_bits  = 0x0;
            }
            /* norm f32 can be represented as norm bf8 */
            else {
                exp_bits = (exp_bits >> fp32_mant_bits) - (fp32_bias - e5m2_bias);
                exp_bits = exp_bits << e5m2_mant_bits;
            }
            // get discard bits, 21bits for bf8
            uint32_t discard_bits = mant_bits & 0x1fffff;
            // record if the following rounding will result in a carry
            bool carry_a_bit = false;
            // rounding
            switch (config.round_mode) {
            case CFG::ROUND_NEAREST:
                if (discard_bits > 0x100000 or (discard_bits == 0x100000 and (mant_bits & 0x200000))) {
                    carry_a_bit = true;
                }
                break;
            case CFG::ROUND_STOCHASTIC:
                //config.random_value = prand_generator(src, config.seed);
                discard_bits += (config.random_value & 0x1fffff); // fetch 21bits of random value to add
                discard_bits &= 0x002fffff;          // fetch valid bits, 21 bits to discard and 1 bit to decide whether carry
                if (discard_bits & 0x200000) {       // according to yang yu, rounding to zero
                    carry_a_bit = true;              // carry 1
                }
                break;
            case CFG::ROUND_ZERO:
                carry_a_bit = false;
                break;
            default:
                throw "\e[1;31mNot Inplemented Round Mode!\e[0m";
                break;
            }
            mant_bits = (mant_bits & 0x600000) >> (fp32_mant_bits - e5m2_mant_bits);
            if (carry_a_bit and mant_bits == 0x3 and exp_bits == 0x78) {
                /* in this case carry will make a inf */
            } else if (carry_a_bit) {
                ++mant_bits;
            }
            result = sign_bits + exp_bits + mant_bits; // + rather than |, since mant may carry a bit to exp

            if (use_clamp) {
                if (sign_bits > 0x0)    result = 0x0;
                else if (result > 0x3c) result = 0x3c;
            }
        }
    }
    else if constexpr (LAYOUT == FP8_LAYOUT::E4M3) {
        /*
            ①. input: sNAN or qNAN,         output: +0(dx10_clamp+inst.clamp) or qNAN
            ②. input: inf,                  output: 1/+0(inst.clamp) or MAX(saturate) or qNAN(fp8 has nor inf)
            ③. input: greater than MAX,     output: 1(inst.clamp, >MAX), 0(inst.clamp, <-MAX) or MAX(saturate) or inf
            ④. input: less than denorm MIN, output: 0(dmode_input off) or min denorm(use rounding to nearest even) or +/-0
            ⑤. input: less than norm MIN,   output: denorm bf8
            ⑥. input: norm,                 output: norm bf8
            ⑦. input: denorm,               output: denorm of fp32 is to small, talked in ④  //is to small???
        */
        /* NAN */
        if (exp_bits == 0x7f800000 and mant_bits > 0x0) {
            if (config.clamp_mode == CFG::DX10_INST_CLAMP_ON) {
                result = 0x0;              // output +0
            }
            else {
                result = sign_bits | 0x7f; // output qNAN
            }
        }
        /* inf or greater than MAX value of E4M3 */
        else if ((exp_bits == 0x7f800000 and mant_bits == 0x0) or (data_scale > 0x43e00000)) {
            if (use_clamp) {
                result = sign_bits ? 0x0: 0x38; // clamp output into [0,1]
            }
            else if (config.saturate_mode == CFG::OVERFLOW_ON and forbidden_inf()) {
                result = sign_bits | 0x7e; // output MAX
            }
            else {
                result = sign_bits | 0x7f; // output inf
            }
        }
        /* less than MIN of denorm */
        else if (exp_bits <= 0x3a800000) {
            if (config.denorm_mode == CFG::DENORM_INPUT_OFF and (exp_bits == 0x0) and mant_bits) {
                result = 0x0;              // flush input denormals to 0 (how about the order for these branches)
            }
            else if (exp_bits == 0x3a800000 and mant_bits > 0x0) {
                result = sign_bits | 0x1;  // output min of denorm, essentially rounding to nearest even
                result = (use_clamp and (sign_bits > 0x0)) ? 0x0: result; // once inst.clamp, -denorm output +0.
            }
            else {
                result = use_clamp ? 0x0: sign_bits; // output +/-0
            }
        }
        /* others */
        else {
            /* norm fp32 can be represented as denorm fp8 */
            if (exp_bits < 0x3c800000) {
                uint32_t shift = (0x3c800000 - exp_bits) >> fp32_mant_bits;
                mant_bits = (0x800000 | mant_bits) >> shift;
                exp_bits  = 0x0;
            }
            /* norm fp32 can be represented as norm fp8 */
            else {
                exp_bits = (exp_bits >> fp32_mant_bits) - (fp32_bias - e4m3_bias);
                exp_bits = exp_bits << e4m3_mant_bits;
            }
            // get discard bits
            uint32_t discard_bits = mant_bits & 0xfffff;
            // record whether the following rounding will result in a carry
            bool carry_a_bit = false;
            // rounding
            switch (config.round_mode) {
            case CFG::ROUND_NEAREST:
                if (discard_bits > 0x80000 or (discard_bits == 0x80000 and (mant_bits & 0x100000))) {
                    carry_a_bit = true;
                }
                break;
            case CFG::ROUND_STOCHASTIC:
                // config.random_value = prand_generator(src, config.seed);
                discard_bits += (config.random_value & 0xfffff); // fetch 20bits of random value to add
                discard_bits &= 0x001fffff;          // fetch valid bits, 21 bits to discard and 1 bit to decide whether carry
                if (discard_bits & 0x100000) {       // according to yang yu, rounding to zero
                    carry_a_bit = true;              // carry 1
                }
                break;
            case CFG::ROUND_ZERO:
                carry_a_bit = false;
                break;
            default:
                throw "\e[1;31mNot Inplemented Round Mode!\e[0m";
                break;
            }
            mant_bits = (mant_bits & 0x700000) >> (fp32_mant_bits - e4m3_mant_bits);
            if (false) {
                /* carry will make a inf, but e4m3 has no inf, and thus S.1111.000 is allowed */
            } else if (carry_a_bit) {
                ++mant_bits;
            }
            result = sign_bits + exp_bits + mant_bits; // + rather than |, since mant may carry a bit to exp
            // decide the output once inst.clamp is on
            if (use_clamp) {
                if (sign_bits > 0x0)    result = 0x0;
                else if (result > 0x38) result = 0x38;
            }
        }
    }
    return result;
}


/* conversion: fp8 -> float

    reference:
        1. https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1
        2. DCU BWb0 SP Delta Spec v0.7
        3. (optional)https://github.com/ROCm/composable_kernel/blob/develop/include/ck/utility/type_convert.hpp

    detail:
            1. support IEEE mode
            2. support denormals
            3. support output clamp
            4. support output modifier
*/
template<FP8_LAYOUT LAYOUT, typename SRC_TYPE=uint8_t>
__host__ __device__
float __fp82float(const SRC_TYPE src, FP8_CVT_CONFIGS &config=default_config) {
    // initialize ret value
    float result = 0.f;
    // conversion from float to unsigned int(32 bits) for convience to fetching each bit
    uint8_t __src = *(uint8_t*)&src;
    bool is_nan = false;
    bool use_clamp = (config.clamp_mode == CFG::INST_CLAMP_ON or config.clamp_mode == CFG::DX10_INST_CLAMP_ON);

    // categorical discussions
    if constexpr (LAYOUT == FP8_LAYOUT::E5M2) {
        // fetch sign bits
        uint32_t sign_bits = (__src & 0x80) << (fp32_bits - e5m2_bits);
        // fetch exponent bits
        uint32_t exp_bits  = (__src & 0x7c) >> e5m2_mant_bits;
        // fetch mantissa bits
        uint32_t mant_bits = __src & 0x3;
        /*
            ①. input: inf,                  output: 1/0(inst.clamp) or inf
            ②. input: NAN,                  output: 0(dx10_inst_clamp) or qNAN(IEEE mode: sNaN->qNaN) or sNaN
            ③. input: greater than MAX,     impossible
            ④. input: less than denorm MIN, impossible
            ⑤. input: less than norm MIN,   impossible
            ⑥. input: norm,                 output: norm fp32
            ⑦. input: denorm,               output: norm fp32
        */
        // inf
        if (exp_bits == 0x1f and mant_bits == 0x0) {
            sign_bits = sign_bits | 0x7f800000;
            result = *(float*)&sign_bits;  // output inf
        }
        // NaN
        else if (exp_bits == 0x1f and mant_bits > 0x0) {
            is_nan = true;
            uint32_t tmp;
            if (config.clamp_mode == CFG::DX10_INST_CLAMP_ON) {
                tmp = 0x0;   // output +0
            }
            // qNAN
            else if (mant_bits >= 0x2) {
                tmp = sign_bits | 0x7f800000 | (mant_bits << (fp32_mant_bits - e5m2_mant_bits));
            }
            // sNAN
            else if (mant_bits == 0x1) {
                if (config.ieee_mode == CFG::IEEE_MODE_ON) { // output qNAN when IEEE mode is open
                    tmp = sign_bits | 0x7fc00000 | (mant_bits << (fp32_mant_bits - e5m2_mant_bits));
                } else {                                     // output sNAN
                    tmp = sign_bits | 0x7f800000 | (mant_bits << (fp32_mant_bits - e5m2_mant_bits));
                }
            }
            result = *(float*)&tmp;
        }
        // 0
        else if (exp_bits == 0x0 and mant_bits == 0x0) {
            result = (sign_bits == 0x0) ? 0.f: -0.f;
        }
        // denorm
        else if (exp_bits == 0x0 and mant_bits > 0x0) {
            result = pow(2.f, 1 - e5m2_bias) *
                   ( 0.5f * bool(mant_bits & 0x2)
                   + 0.25f * bool(mant_bits & 0x1));
            result = sign_bits > 0 ? -result: result;
        }
        // norm
        else {
            uint32_t tmp = sign_bits | ((exp_bits - e5m2_bias + fp32_bias) << fp32_mant_bits)
                            | (mant_bits << (fp32_mant_bits - e5m2_mant_bits));
            result = *(float*)&tmp;
        }
    }
    else if constexpr (LAYOUT == FP8_LAYOUT::E4M3) {
        // fetch sign bits
        uint32_t sign_bits = __src & 0x80;
        // fetch exponent bits
        uint32_t exp_bits  = (__src & 0x78) >> e4m3_mant_bits;
        // fetch mantissa bits
        uint32_t mant_bits = __src & 0x7;
        /*
            ①. input: inf,                  impossible, e4m3 has no inf
            ②. input: qNAN,                 output: +0(dx10_inst_clamp) or qNAN
            ②. input: sNAN,                 impossible, e4m3 has no sNAN
            ③. input: greater than MAX,     impossible
            ④. input: less than denorm MIN, impossible
            ⑤. input: less than norm MIN,   impossible
            ⑥. input: norm,                 output: norm fp32
            ⑦. input: denorm,               output: norm fp32
        */
        // NaN
        if (exp_bits == 0xf and mant_bits == 0x7) {
            is_nan = true;
            uint32_t tmp;
            if (config.clamp_mode == CFG::DX10_INST_CLAMP_ON) {
                tmp = 0x0;  // output +0.f
            } else {
                tmp = sign_bits ? 0xfff00000: 0x7ff00000; // fp8 only has qNAN
            }
            result = *(float*)&tmp;
        }
        // 0
        else if (exp_bits == 0x0 and mant_bits == 0x0) {
            result = (sign_bits == 0x0)? 0.f: -0.f;
        }
        // denorm
        else if (exp_bits == 0x0 and mant_bits > 0x0) {
            // attention, for denorm, exponents is 1
            result = pow(2.f, 1 - e4m3_bias) *
                   ( 0.5f * bool(mant_bits & 0x4)
                   + 0.25f * bool(mant_bits & 0x2)
                   + 0.125f * bool(mant_bits & 0x1));
            result = sign_bits > 0 ? -result: result;
        }
        // norm
        else {
            uint32_t tmp = (sign_bits << (fp32_bits - e4m3_bits)) | ((exp_bits - e4m3_bias + fp32_bias) << fp32_mant_bits) | (mant_bits << (fp32_mant_bits - e4m3_mant_bits));
            result = *(float*)&tmp;
        }
    }

    /* some features work on output: */
    // except nan because nan is dealt with in the above
    if (!is_nan) {
        // output modifier
        if (config.output_modifier != CFG::OUTPUT_MODIFIER_OFF and
            config.ieee_mode == CFG::IEEE_MODE_OFF and
            config.f32_denorm_mode == CFG::F32_DENORM_OUTPUT_OFF) {
            if (config.output_modifier == CFG::OUTPUT_MODIFIER_MUL2)
                result = result * 2;
            else if (config.output_modifier == CFG::OUTPUT_MODIFIER_MUL4)
                result = result * 4;
            else if (config.output_modifier == CFG::OUTPUT_MODIFIER_DIV2)
                result = result / 2;
        }
        // clamp
        if (use_clamp) {
            result = (result > 1.f) ? 1.f : (*(uint32_t*)(&result) >> 31 & 1) == 1 ? +0.f : result;
        }
    }

    return result;
}



// type and function for FP8
template<FP8_LAYOUT LAYOUT>
struct alignas(1) FP8 {
    static constexpr bool IS_E4M3 = (LAYOUT == FP8_LAYOUT::E4M3);
    static constexpr bool IS_E5M2 = (LAYOUT == FP8_LAYOUT::E5M2);
public:
    __host__ __device__
    FP8(): data(0)
    {
    }

    // copy constructor from a fp8
    __host__ __device__
    FP8(const FP8<LAYOUT>& value)
        : data(value.data)
    {
    }

    // copy constructor from a uint8_t value
    __host__ __device__
    FP8(const uint8_t value)
        : data(value)
    {
    }

    // copy constructor from a float value and some configs
    __host__ __device__
    FP8(const float value,
        FP8_CVT_CONFIGS &config=default_config)
        : data(__float2fp8<LAYOUT>(value, config))
    {
    }

    // copy constructor from a int value( Failed Yet !)
    __host__ __device__
    FP8(const uint32_t value)
        : data(__float2fp8<LAYOUT>(*(float*)&value))
    {
    }

    // copy constructor from a int value( Failed Yet !)
    __host__ __device__
    FP8(const int value)
        : data(__float2fp8<LAYOUT>(static_cast<float>(value)))
    {
    }

    // copy assign constructor from a fp8
    __host__ __device__
    FP8<LAYOUT>& operator=(const FP8<LAYOUT>& value)
    {
        this->data = value.data;
        return *this;
    }

    // acquire float value
    __host__ __device__
    float float_value(FP8_CVT_CONFIGS &config=default_config) const
    {
        return __fp82float<LAYOUT>(this->data, config);
    }

    // acquire double value
    __host__ __device__
    double double_value(FP8_CVT_CONFIGS &config=default_config) const
    {
        return static_cast<double>(this->float_value(config));
    }

    // acquire data
    __host__ __device__
    uint8_t uchar_value() const
    {
        return this->data;
    }

    // acquire string of bits
    std::string bits_string() const {
        return fp8_bits(this->data);
    }

    // set value by uchar
    __host__ __device__
    void from_uchar(const uint8_t value)
    {
        this->data = value;
    }

    // set value by uint32_t
    __host__ __device__
    void from_uint32(const uint32_t value, FP8_CVT_CONFIGS &config=default_config, uint32_t random_value = 0)
    {
        config.random_value = random_value;
        this->data = __float2fp8<LAYOUT>(*(float*)&value, config);
    }

    // set value by float
    __host__ __device__
    void from_float(const float value, FP8_CVT_CONFIGS &config=default_config, uint32_t random_value = 0)
    {
        config.random_value = random_value;
        this->data = __float2fp8<LAYOUT>(value, config);
    }

    // check inf
    __host__ __device__
    bool isinf() const
    {
        // Specially, E4M3 has no infinity
        return IS_E5M2 ? bool((this->data & 0x7c) == 0x7c): false;
    }

    // check nan
    __host__ __device__
    bool isnan() const
    {
        if constexpr (IS_E4M3) {
            return (this->data & 0x7f) == 0x7f;
        }
        else if constexpr (IS_E5M2) {
            return ((this->data & 0x7c) == 0x7c) && ((this->data & 0x3) != 0);
        }
        return false;
    }

    // check zero
    __host__ __device__
    bool iszero() const
    {
        return this->data == 0x0 or this->data == 0x80;
    }

    // acquire absolute value
    __host__ __device__
    FP8<LAYOUT> abs() const
    {
        return FP8<LAYOUT>(static_cast<uint8_t>(this->data & 0x7f));
    }

    // acquire reverse value
    __host__ __device__
    FP8<LAYOUT> reverse() const
    {
        return FP8<LAYOUT>(static_cast<uint8_t>((this->data & 0x7f) | (this->data ^ 0x80)));
    }

    // make self reverse sign bits
    __host__ __device__
    void reverse()
    {
        this->data ^= 0x80;
    }

    // float op
    __host__ __device__
    explicit operator float() const
    {
        return this->float_value();
    }

    // double op
    __host__ __device__
    explicit operator double() const
    {
        return this->double_value();
    }

    // judge whether negative
    __host__ __device__
    bool is_negative()
    {
        return this->data & 0x80;
    }

    // judge whether negative
    __host__ __device__
    bool is_positive()
    {
        return !this->is_negative();
    }

    // compare <
    __host__ __device__
    bool operator<(const FP8<LAYOUT>& rhs)  const
    {
        return this->float_value() < rhs.float_value();
    }

    // compare >
    __host__ __device__
    bool operator>(const FP8<LAYOUT>& rhs)  const
    {
        return this->float_value() > rhs.float_value();
    }

    // compare ==
    __host__ __device__
    bool operator==(const FP8<LAYOUT>& rhs) const
    {
        return this->data == rhs.data;
    }

    // compare == uchar
    __host__ __device__
    bool operator==(const uint8_t rhs) const {
        return this->data == rhs;
    }

    // compare == float
    __host__
    bool is(const float rhs, FP8_CVT_CONFIGS &config=default_config) const
    {
        return std::abs(this->float_value(config) - rhs) <= 1e-10;
    }

    // compare !=
    __host__ __device__
    bool operator!=(const FP8<LAYOUT>& rhs) const
    {
        return this->data not_eq rhs.data;
    }

    // compare != uchar
    __host__ __device__
    bool operator!=(const uint8_t rhs) const {
        return this->data not_eq rhs;
    }

    // get value
    __host__ __device__
    float operator*() const
    {
        return this->float_value();
    }

    // check 8bits of fp8
    void check() const
    {
        printf("%s\n", this->bits_string().c_str());
    }

    // acquire limits of norm
    static float norm_max()
    {
        if constexpr (IS_E5M2)
            return FP8<LAYOUT>(static_cast<uint8_t>(0x7b)).float_value();
        else if constexpr (IS_E4M3)
            return FP8<LAYOUT>(static_cast<uint8_t>(0x7e)).float_value();
        return 0;
    }

    static float norm_min()
    {
        if constexpr (IS_E5M2)
            return FP8<LAYOUT>(static_cast<uint8_t>(0x4)).float_value();
        else if constexpr (IS_E4M3)
            return FP8<LAYOUT>(static_cast<uint8_t>(0x8)).float_value();
        return 0;
    }

    static float subnorm_max()
    {
        if constexpr (IS_E5M2)
            return FP8<LAYOUT>(static_cast<uint8_t>(0x3)).float_value();
        else if constexpr (IS_E4M3)
            return FP8<LAYOUT>(static_cast<uint8_t>(0x7)).float_value();
        return 0;
    }

    static float subnorm_min()
    {
        return FP8<LAYOUT>(static_cast<uint8_t>(0x1)).float_value();
    }

protected:
    /* according to /opt/rocm/include/hip/amd_detail/host_defines.h:43:typedef unsigned char uint8_t; */
    uint8_t data;
}; // FP8


// distinct types
using float8_t     = FP8<FP8_LAYOUT::E4M3>;
using FP8_E4M3     = FP8<FP8_LAYOUT::E4M3>;
using float_e4m3_t = FP8<FP8_LAYOUT::E4M3>;

using bfloat8_t    = FP8<FP8_LAYOUT::E5M2>;
using FP8_E5M2     = FP8<FP8_LAYOUT::E5M2>;
using float_e5m2_t = FP8<FP8_LAYOUT::E5M2>;

// output in ostream
template<FP8_LAYOUT T>
inline std::ostream& operator<<(std::ostream& os, const FP8<T> src)
{
    os << src.float_value();
    return os;
}

// fp8 * fp8  |  bf8 * bf8
template<FP8_LAYOUT T>
inline float operator*(const FP8<T> lhs, const FP8<T> rhs)
{
    return lhs.float_value() * rhs.float_value();
}

// fp8 * bf8  |  bf8 * fp8
template<FP8_LAYOUT T1, FP8_LAYOUT T2>
inline float operator*(const FP8<T1> lhs, const FP8<T2> rhs)
{
    return lhs.float_value() * rhs.float_value();
}

// fp8 + fp8
template<FP8_LAYOUT T>
inline float operator+(const FP8<T> lhs, const FP8<T> rhs)
{
    return lhs.float_value() + rhs.float_value();
}

// float += fp8
template<FP8_LAYOUT T>
inline void operator+=(float& lhs, const FP8<T> rhs)
{
    lhs = lhs + rhs.float_value();
}

// fp8 += float
template<FP8_LAYOUT T>
inline void operator+=(FP8<T>& lhs, const float rhs)
{
    float tmp = lhs.float_value() + rhs;
    lhs.set(tmp);
}

// float + fp8
template<FP8_LAYOUT T>
inline float operator+(const float lhs, const FP8<T> rhs)
{
    return lhs + rhs.float_value();
}

// fp8 + float
template<FP8_LAYOUT T>
inline float operator+(const FP8<T> lhs, const float rhs)
{
    return lhs.float_value() + rhs;
}

// float -= fp8
template<FP8_LAYOUT T>
inline void operator-=(float& lhs, const FP8<T> rhs)
{
    lhs = lhs - rhs.float_value();
}

// fp8 -= float
template<FP8_LAYOUT T>
inline void operator-=(FP8<T>& lhs, const float rhs)
{
    float tmp = lhs.float_value() - rhs;
    lhs.set(tmp);
}

// fp8 - float
template<FP8_LAYOUT T>
inline float operator-(const FP8<T> lhs, const float rhs)
{
    return lhs.float_value() - rhs;
}

// float - fp8
template<FP8_LAYOUT T>
inline float operator-(const float lhs, const FP8<T> rhs)
{
    return lhs - rhs.float_value();
}

// fp8 & uint8_t
template<FP8_LAYOUT T>
__host__ __device__
inline float operator&(const FP8<T> lhs, const uint8_t rhs)
{
    return lhs.uchar_value() & rhs;
}

// -fp8 | -bf8
template<FP8_LAYOUT T>
__host__ __device__
inline FP8<T> operator-(const FP8<T> lhs)
{
    return lhs.reverse();
}

// ~fp8 | ~bf8
template<FP8_LAYOUT T>
__host__ __device__
inline FP8<T> operator~(const FP8<T> lhs)
{
    return lhs.reverse();
}

// abs
// this doesn't conflict with std::abs even if "using namespace std;" in other files
template<FP8_LAYOUT T>
__host__ __device__
inline FP8<T> abs(const FP8<T> lhs)
{
    return lhs.abs();
}


// functions in std
namespace std {

    template<FP8_LAYOUT T>
    inline bool isinf(const FP8<T>& src)
    {
        return src.isinf();
    }

    template<FP8_LAYOUT T>
    inline bool isnan(const FP8<T>& src)
    {
        return src.isnan();
    }

    template<FP8_LAYOUT T>
    inline bool iszero(const FP8<T>& src)
    {
        return src.iszero();
    }

    template<FP8_LAYOUT T>
    inline FP8<T> abs(const FP8<T>& src)
    {
        return src.abs();
    }

} // namespace std

// ===================================== OCP FP8 Convert ===================================== //

#define __FP8_HOST_DEVICE__ __host__ __device__
#define __FP8_HOST_DEVICE_STATIC__ __FP8_HOST_DEVICE__ static inline

template <typename T>
__FP8_HOST_DEVICE_STATIC__ fp8_storage_t
cvt_float_to_fp8(const T x, const FP8_LAYOUT layout,
                 const enum hipRoundMode  rounding,
                 uint8_t scale = 0x7f) {
  union {
    uint32_t ui32;
    fp8_storage_t fp8[4];
  } u{0};

  if (layout == FP8_LAYOUT::E4M3)
    u.ui32 = fcbx::from_float<T, fcbx::Encoding::E4M3Mx, true>(x, scale, rounding);
  else if (layout == FP8_LAYOUT::E5M2)
    u.ui32 = fcbx::from_float<T, fcbx::Encoding::E5M2Mx, true>(x, scale, rounding);

  return u.fp8[0];
}

template <typename T>
__FP8_HOST_DEVICE_STATIC__ fp8_storage_t
cvt_float_to_fp8_sr(T x, const FP8_LAYOUT layout, uint32_t seed, uint8_t scale = 0) {
  union {
    uint32_t ui32;
    fp8_storage_t fp8[4];
  } u{0};

  if (layout == FP8_LAYOUT::E4M3)
    u.ui32 = fcbx::from_float_sr<T, fcbx::Encoding::E4M3Mx, true>(x, seed, scale);
  else if (layout == FP8_LAYOUT::E5M2)
    u.ui32 = fcbx::from_float_sr<T, fcbx::Encoding::E5M2Mx, true>(x, seed, scale);

  return u.fp8[0];
}

