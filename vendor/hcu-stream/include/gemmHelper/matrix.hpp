#ifndef __MATRIX_H__
#define __MATRIX_H__

#include "gemmHelper/global.hpp"
#include "hip/hip_runtime.h"
#include "gemmHelper/rand.hpp"

#define CPU -1
#define GPU0 0

typedef struct param {
  size_t M = 0;
  size_t N = 0;
  size_t K = 0;
  size_t lda = 0;
  size_t ldb = 0;
  size_t ldc = 0;
  size_t stridea = 0;
  size_t strideb = 0;
  size_t stridec = 0;
  size_t strided = 0;
  size_t batch_count = 1;
  bool transA = false;
  bool transB = true;
  float alpha = 1.0f;
  float beta = 0.0f;
  int test_cnt = 1;
} param;

template <typename T> class GemmMat {
public:
  GemmMat(size_t row, size_t col, size_t ld, size_t stride, size_t batch_count,
          bool trans, int dev);
  GemmMat(size_t row, size_t col, size_t ld, size_t stride, size_t batch_count,
          bool trans, int dev, Rng<T> rng, bool rand_int = false);
  ~GemmMat();
  void Print();
  void Print(std::string);
  size_t row();
  size_t col();
  size_t ld();
  size_t stride1();
  size_t stride2();
  size_t stride();
  size_t batch_count();
  bool trans();
  size_t size();
  size_t byte_size();
  int dev();
  T *data();
  GemmMat<T> *to(int dev);
  void Copy(shared_ptr<GemmMat<T>> mat);

  template <typename U = T, typename = typename std::enable_if<
                                !std::is_same<U, Int4x2>{} &&
                                !std::is_same<U, Uint4x2>{} && 
                                !std::is_same<U, float4x2_t>{}>::type>  // float4x2_t
  T &get(size_t row, size_t col, size_t batch_count) {
    size_t idx = row * m_stride1 + col * m_stride2 + batch_count * m_stride;
    return m_data[idx];
  }
  template <typename U = T,
            typename = typename std::enable_if<
                std::is_same<U, Int4x2>{} || std::is_same<U, Uint4x2>{}>::type>
  int32_t get(size_t row, size_t col, size_t batch_count) {
    size_t idx = row * m_stride1 + col * m_stride2 + batch_count * m_stride;
    return m_data[idx / 2][idx];
  } 

  // m_data设置
  template <typename U = T,
            typename = typename std::enable_if<
                std::is_same<U, float4x2_t>{}>::type>
  float4_t get(size_t row, size_t col, size_t batch_count) {
    size_t idx = row * m_stride1 + col * m_stride2 + batch_count * m_stride;
    // fp4 is packed into fp4x2 bytes (2 fp4 per byte)
    return m_data[idx / 2][idx];
  }


private:
  size_t m_row;
  size_t m_col;
  size_t m_ld;
  size_t m_stride1;
  size_t m_stride2;
  size_t m_stride;
  size_t m_batch_count;
  bool m_trans;
  size_t m_size;
  size_t m_byte_size;
  int m_dev;
  T *m_data;
  T *m_shadow;
};

template <typename T>
GemmMat<T>::GemmMat(size_t row, size_t col, size_t ld, size_t stride,
                    size_t batch_count, bool trans, int dev)
    : m_row(row), m_col(col), m_stride(stride), m_batch_count(batch_count),
      m_trans(trans), m_dev(dev), m_data(nullptr), m_shadow(nullptr) {
  if (trans == true) {
    m_ld = (ld == 0) ? col : ld;
    m_stride1 = m_ld;
    m_stride2 = 1;
    m_size = row * m_ld;
    m_stride = (stride == 0) ? m_size : stride;
    m_size = (m_size == m_stride) ? m_size : row * m_stride;
    m_size = m_size * m_batch_count;
  } else {
    m_ld = (ld == 0) ? row : ld;
    m_stride1 = 1;
    m_stride2 = m_ld;
    m_size = m_ld * col;
    m_stride = (stride == 0) ? m_size : stride;
    m_size = (m_size == m_stride) ? m_size : m_stride * col;
    m_size = m_size * m_batch_count;
  }

  if constexpr (std::is_same<T, Int4x2>::value ||
                std::is_same<T, Uint4x2>::value ||
                std::is_same<T, float4x2_t>::value) {
    m_byte_size = (m_size + 1) / 2;
  } else {
    m_byte_size = m_size * sizeof(T);
  }

  if (dev == CPU) {
    // std::cout<<"init cpu data!"<<std::endl;
    m_data = reinterpret_cast<T *>(malloc(m_byte_size));
    assert(m_data);
  } else {
    CHECK_HIP_ERROR(hipMalloc((void **)&m_data, m_byte_size));
    // std::cout<<"init gpu data!"<< std::endl;
  }
}

template <typename T>
GemmMat<T>::GemmMat(size_t row, size_t col, size_t ld, size_t stride,
                    size_t batch_count, bool trans, int dev, Rng<T> rng,
                    bool rand_int)
    : GemmMat<T>::GemmMat(row, col, ld, stride, batch_count, trans, dev) {
  if (dev == CPU) {
    if constexpr (std::is_same<T, Int4x2>::value ||
                  std::is_same<T, Uint4x2>::value ||
                  std::is_same<T, float4x2_t>::value) {
      for (size_t i = 0; i < ((m_size + 1) / 2); i++)
        m_data[i] = rng();
    } else if constexpr (std::is_same<T, double>::value ||
                         std::is_same<T, float>::value ||
                         std::is_same<T, _Float16>::value) {
      if (rand_int) {
        Rng<int32_t> rng_int(0, 1);
        for (size_t i = 0; i < m_size; ++i)
          m_data[i] = rng_int();
      } else {
        for (size_t i = 0; i < m_size; ++i)
          m_data[i] = rng();
      }
    } else {
      for (size_t i = 0; i < m_size; ++i)
        m_data[i] = rng();
    }
    // std::cout<<"init cpu data with rng!"<<std::endl;
  } else {
    m_shadow = reinterpret_cast<T *>(malloc(m_byte_size));

    if constexpr (std::is_same<T, Int4x2>::value ||
                  std::is_same<T, Uint4x2>::value ||
                  std::is_same<T, float4x2_t>::value) {
      for (size_t i = 0; i < ((m_size + 1) / 2); i++)
        m_shadow[i] = rng();
    } else if constexpr (std::is_same<T, double>::value ||
                         std::is_same<T, float>::value ||
                         std::is_same<T, _Float16>::value) {
      if (rand_int) {
        Rng<int32_t> rng_int(0, 1);
        for (size_t i = 0; i < m_size; ++i)
          m_shadow[i] = rng_int();
      } else {
        for (size_t i = 0; i < m_size; ++i)
          m_shadow[i] = rng();
      }
    } else {
      for (size_t i = 0; i < m_size; ++i)
        m_shadow[i] = rng();
    }

    CHECK_HIP_ERROR(
        hipMemcpy(m_data, m_shadow, m_byte_size, hipMemcpyHostToDevice));
    // std::cout<<"init gpu data with rng!"<<std::endl;
    free(m_shadow);
  }
}

template <typename T> GemmMat<T>::~GemmMat() {
  if (m_dev == CPU) {
    free(m_data);
  } else {
    CHECK_HIP_ERROR(hipFree(m_data));
  }
}

template <typename T> void GemmMat<T>::Print() {
  for (size_t k = 0; k < m_batch_count; k++) {
    for (size_t i = 0; i < m_row; i++) {
      for (size_t j = 0; j < m_col; j++) {
        size_t idx = m_trans ? IDX2T(i, j, m_ld) : IDX2N(i, j, m_ld);
        idx = idx + k * m_stride;
        if constexpr (std::is_same<T, Int4x2>{} || std::is_same<T, Uint4x2>{})
          std::cout << m_data[idx / 2][idx] << ", ";
        else if constexpr (std::is_same<T, int8_t>{})
          std::cout << static_cast<int>(m_data[idx]) << ", ";
        else if constexpr (std::is_same<T, uint8_t>{})
          std::cout << static_cast<unsigned int>(m_data[idx]) << ", ";
        else {
          std::cout << m_data[idx] << ", ";
        }
      }
      std::cout << std::endl;
    }
  }
}

template <typename T> void GemmMat<T>::Print(std::string filename) {
  size_t pos = 0;
  std::string fatal_msg;
  struct stat st;
  pos = filename.find_last_of("/");
  if (pos == std::string::npos) {
    fatal_msg = "\n*** Error output file name !";
    ExitWithMsg(-1, fatal_msg, true);
  }
  std::string fileDir = filename.substr(0, pos).c_str();
  if (stat(fileDir.c_str(), &st) == -1) {
    mkdir(fileDir.c_str(), 0775);
  }
  std::ofstream ofs(filename, std::ofstream::binary | std::ofstream::trunc);
  // if (g_device_props.arch.hasDp4x && std::is_same<T, double>{}) {
  //   char c = '\0';
  //   for (size_t i = 0; i < m_byte_size; i++)
  //     ofs.write(&c, sizeof(c));
  // }

  for (size_t k = 0; k < m_batch_count; k++) {
    for (size_t j = 0; j < m_col; j++) {
      for (size_t i = 0; i < m_row; i++) {
        size_t idx = m_trans ? IDX2T(i, j, m_ld) : IDX2N(i, j, m_ld);
        idx = idx + k * m_stride;
        T val = m_data[idx];
        ofs.write(reinterpret_cast<char *>(&val), sizeof(val));
      }
    }
  }

  ofs.close();
}

template <typename T> GemmMat<T> *GemmMat<T>::to(int dev) {
  GemmMat<T> *mat =
      new GemmMat<T>(m_row, m_col, m_ld, m_stride, m_batch_count, m_trans, dev);

  assert(m_dev != dev);

  if (m_dev == CPU) {
    CHECK_HIP_ERROR(
        hipMemcpy(mat->m_data, m_data, m_byte_size, hipMemcpyHostToDevice));
  } else {
    CHECK_HIP_ERROR(
        hipMemcpy(mat->m_data, m_data, m_byte_size, hipMemcpyDeviceToHost));
  }

  return mat;
}

template <typename T> void GemmMat<T>::Copy(shared_ptr<GemmMat<T>> mat) {
  if (m_dev == CPU) {
    if (mat->dev() == CPU) {
      memcpy(m_data, mat->m_data, m_byte_size);
      // std::cout<<"cpu -> cpu"<<std::endl;
    } else {
      // std::cout<<"gpu -> cpu"<<std::endl;
      CHECK_HIP_ERROR(
          hipMemcpy(m_data, mat->m_data, m_byte_size, hipMemcpyDeviceToHost));
    }
  } else {
    if (mat->dev() == CPU) {
      CHECK_HIP_ERROR(
          hipMemcpy(m_data, mat->m_data, m_byte_size, hipMemcpyHostToDevice));
      // std::cout<<"cpu -> gpu"<<std::endl;
    } else {
      assert(false);
    }
  }
}

template <typename T> size_t GemmMat<T>::size(void) { return m_size; }

template <typename T> size_t GemmMat<T>::byte_size(void) { return m_byte_size; }

template <typename T> size_t GemmMat<T>::stride1(void) { return m_stride1; }

template <typename T> size_t GemmMat<T>::stride2(void) { return m_stride2; }

template <typename T> bool GemmMat<T>::trans(void) { return m_trans; }

template <typename T> size_t GemmMat<T>::row(void) { return m_row; }

template <typename T> size_t GemmMat<T>::col(void) { return m_col; }

template <typename T> size_t GemmMat<T>::ld(void) { return m_ld; }

template <typename T> size_t GemmMat<T>::stride(void) { return m_stride; }

template <typename T> size_t GemmMat<T>::batch_count(void) {
  return m_batch_count;
}

template <typename T> int GemmMat<T>::dev(void) { return m_dev; }

template <typename T> T *GemmMat<T>::data(void) { return m_data; }
#endif /* __MATRIX_H__ */