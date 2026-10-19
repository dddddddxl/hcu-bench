#ifndef DATA_DEFINE_H
#define DATA_DEFINE_H
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#define HIP_CHECK(call)                                    \
    do                                                     \
    {                                                      \
        const hipError_t error_code = call;                \
        if (error_code != hipSuccess)                      \
        {                                                  \
            printf("HIP Error:\n");                        \
            printf("    File:       %s\n", __FILE__);      \
            printf("    Line:       %d\n", __LINE__);      \
            printf("    Error code: %d\n", error_code);    \
            printf("    Error text: %s\n",                 \
                hipGetErrorString(error_code));            \
            exit(1);                                       \
        }                                                  \
    } while (0)

    
#ifdef __cplusplus
extern "C" {
#endif

    enum DataDevice
    {
        CPU = 0,
        DCU = 1,
        CUDA = 2
    };

    enum DataType
    {
        FLOAT32 = 0,
        FLOAT16 = 1,
        BFLOAT16 = 2,
        INT64 = 3,
        INT32 = 4,
        INT16 = 5,
        INT8 = 6,
        INT4 = 7,
        INT2 = 8,
        UINT64 = 9,
        UINT32 = 10,
        UINT16 = 11,
        UINT8 = 12,
        BIT = 13,
        INT4_NOZERO = 14,
        INT32PARAM = 100
    };

    typedef struct __align__(16) {
        half2 x;
        half2 y;
        half2 z;
        half2 w;
     } half8;
     
     typedef struct __align__(8) {
        half2 x;
        half2 y;
     } half4;

#ifdef __cplusplus
}
#endif

#endif