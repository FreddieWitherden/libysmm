#ifndef LIBYSMM_COMMON_H
#define LIBYSMM_COMMON_H

typedef enum libysmm_support_level
{
    LIBYSMM_SUPPORT_LEVEL_NONE = 1,
    LIBYSMM_SUPPORT_LEVEL_BASIC = 2,
    LIBYSMM_SUPPORT_LEVEL_TUNED = 3,
    LIBYSMM_SUPPORT_LEVEL_NATIVE = 4,
} libysmm_support_level_t;

typedef enum libysmm_dtype
{
    LIBYSMM_DTYPE_FP32 = 1,
    LIBYSMM_DTYPE_FP64 = 2,
} libysmm_dtype_t;

typedef enum libysmm_layout
{
    LIBYSMM_LAYOUT_COL_MAJOR = 1,
    LIBYSMM_LAYOUT_ROW_MAJOR = 2,
} libysmm_layout_t;

typedef enum libysmm_transpose
{
    LIBYSMM_TRANSPOSE_NN = 1,
    LIBYSMM_TRANSPOSE_NT = 2,
    LIBYSMM_TRANSPOSE_TT = 3,
} libysmm_transpose_t;

typedef struct libysmm_smm
{
    libysmm_dtype_t dtype;
    libysmm_layout_t layout;
    libysmm_transpose_t transpose;

    int m;
    int n;
    int k;

    int lda;
    int ldb;
    int ldc;

    double alpha;
    double beta;

    void *a;

    int flags;
} libysmm_smm_t;

#endif
