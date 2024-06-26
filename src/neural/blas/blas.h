#pragma once

#ifdef USE_OPENBLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#ifdef USE_EIGEN
#include <Eigen/Dense>
#endif

#include "neural/blas/sgemm.h"

class Blas {
public:
    // This is interface for convolution GEMM, not the normal GEMM. Some parameters
    // will be invalid.
    static void ConvolutionSgemm(const int M, const int N, const int K,
                                 const float alpha,
                                 const float *A, const int lda,
                                 const float *B, const int ldb,
                                 const float beta,
                                 float *C, const int ldc);


    // This is interface for Winograd GEMM, not the normal GEMM. Some parameters
    // will be invalid.
    static void WinogradSgemm(const int offset_u, const int offset_v, const int offset_m,
                              const int M, const int N, const int K,
                              const float alpha,
                              const float *A, const int lda,
                              const float *B, const int ldb,
                              const float beta,
                              float *C, const int ldc);


    // This is interface for fullyconnet GEMM, not the normal GEMM. Some parameters
    // will be invalid.
    static void DenseSgemm(const int inputs,
                           const int outputs,
                           const int batch_size,
                           const float *input,
                           const float *kernel,
                           float *output);
};
