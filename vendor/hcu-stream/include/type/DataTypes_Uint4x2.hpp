#pragma once
#if defined(__clang__) && defined(__HIP__)
#define TENSILE_HOST_DEVICE __host__ __device__
#else
#define TENSILE_HOST_DEVICE
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Uint4x2 definition for 4-bit signed integer, 8 lanes datatype.

struct Uint4x2 {
  /// The scalar type.
  typedef Uint4x2 Scalar;

  /// The aligned storage to make sure we have good alignment.
  // AlignedStruct<kVectorSize> aligned_;
  /// The data in registers.
  uint8_t registers;
  /// int8_t data members.
  // int8_t a, b;

  /// Default Constructor
  TENSILE_HOST_DEVICE
  Uint4x2() {}

  /// Constructor to convert from uint8_t type
  TENSILE_HOST_DEVICE
  Uint4x2(uint8_t value) { registers = value; }

  TENSILE_HOST_DEVICE
  Uint4x2(uint8_t xa, uint8_t xb) { registers = (xa & 0xf) | (xb << 4); } // ignore waring

  /// Accessor to the ith lane.
  TENSILE_HOST_DEVICE
  int get_scalar(uint8_t i, int32_t value) const {
    return value >> (i % 2 * 4) & 0x0f;
  }

  TENSILE_HOST_DEVICE
  operator uint8_t() const { return registers; }

  /// Accessor to the ith lane.
  TENSILE_HOST_DEVICE
  int const operator[](uint32_t i) const {
    return registers >> (i % 2 * 4) & 0x0f;
  }
};
