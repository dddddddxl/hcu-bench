#pragma once
#if defined(__clang__) && defined(__HIP__)
#define TENSILE_HOST_DEVICE __host__ __device__
#else
#define TENSILE_HOST_DEVICE
#endif

#define BIT_STR(x, offsets) ((x >> offsets) & 0x0f)
#define BIT_STR_OP(x, y, op)                                                   \
  (BIT_STR(x, 0) op BIT_STR(y, 0)) << 0 | (BIT_STR(x, 4) op BIT_STR(y, 4)) << 4

#define BIT_STR_NEG(x) (BIT_STR(x, 0) ^ 0x8) << 0 | (BIT_STR(x, 4) ^ 0x8) << 4

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Int4x2 definition for 4-bit signed integer, 8 lanes datatype.

struct Int4x2 {
  /// The scalar type.
  typedef Int4x2 Scalar;

  /// The aligned storage to make sure we have good alignment.
  // AlignedStruct<kVectorSize> aligned_;
  /// The data in registers.
  uint8_t registers;
  /// int8_t data members.
  // int8_t a, b;

  /// Default Constructor
  TENSILE_HOST_DEVICE
  Int4x2() {}

  /// Constructor to convert from uint8_t type
  TENSILE_HOST_DEVICE
  Int4x2(uint8_t value) { registers = value; }

  TENSILE_HOST_DEVICE
  Int4x2(int8_t xa, int8_t xb) {
    registers = (uint8_t)(xa << 4) >> 4 | (uint8_t)(xb << 4);
  }

  /// Accessor to the ith lane.
  TENSILE_HOST_DEVICE
  int get_scalar(uint8_t i, int32_t value) const {
    return (value >> (i % 2 * 4) & 0x0f) -
           16 * (value >> (i % 2 * 4 + 3) & 0x01);
  }

  TENSILE_HOST_DEVICE
  operator int8_t() const { return registers; }

  /// Accessor to the ith lane.
  TENSILE_HOST_DEVICE
  int const operator[](uint32_t i) const {
    return (registers >> (i % 2 * 4) & 0x0f) -
           16 * (registers >> (i % 2 * 4 + 3) & 0x01);
  }
};

inline std::ostream &operator<<(std::ostream &os, const Int4x2 &lhs) {
  return os << lhs.registers;
}

inline TENSILE_HOST_DEVICE Int4x2 operator+(Int4x2 a) { return a; }

inline TENSILE_HOST_DEVICE Int4x2 operator-(Int4x2 a) {
  return Int4x2(BIT_STR_NEG(a));
}

inline TENSILE_HOST_DEVICE Int4x2 operator+(Int4x2 a, Int4x2 b) {
  return Int4x2(BIT_STR_OP(a, b, +));
}

inline TENSILE_HOST_DEVICE Int4x2 &operator+=(Int4x2 &a, Int4x2 b) {
  a = Int4x2(BIT_STR_OP(a, b, +));
  return a;
}

inline TENSILE_HOST_DEVICE Int4x2 operator+(int a, Int4x2 b) {
  return Int4x2(BIT_STR_OP(a, b, +));
}

inline TENSILE_HOST_DEVICE Int4x2 operator+(Int4x2 a, int b) {
  return Int4x2(BIT_STR_OP(a, b, +));
}

inline TENSILE_HOST_DEVICE Int4x2 operator-(Int4x2 a, Int4x2 b) {
  return Int4x2(BIT_STR_OP(a, b, -));
}

inline TENSILE_HOST_DEVICE int32_t operator*(Int4x2 a, Int4x2 b) {
  return (a[0] * b[0] + a[1] * b[1]);
}

inline TENSILE_HOST_DEVICE Int4x2 operator/(Int4x2 a, Int4x2 b) {
  return Int4x2(BIT_STR_OP(a, b, /));
}

inline TENSILE_HOST_DEVICE bool operator<(Int4x2 a, Int4x2 b) {
  return a.registers < b.registers;
}

inline TENSILE_HOST_DEVICE bool operator==(Int4x2 a, Int4x2 b) {
  return a.registers == b.registers;
}

inline TENSILE_HOST_DEVICE bool operator>(Int4x2 a, Int4x2 b) {
  return a.registers > b.registers;
}

inline TENSILE_HOST_DEVICE bool operator<=(Int4x2 a, Int4x2 b) {
  return a.registers <= b.registers;
}

inline TENSILE_HOST_DEVICE bool operator!=(Int4x2 a, Int4x2 b) {
  return a.registers != b.registers;
}

inline TENSILE_HOST_DEVICE bool operator>=(Int4x2 a, Int4x2 b) {
  return a.registers >= b.registers;
}
