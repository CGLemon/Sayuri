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
    // This is interface for convolution. It is not the real general
    // matrix multiply. Some parameters will invalid.
    static void ConvolutionSgemm(const int M, const int N, const int K,
                                 const float alpha, 
                                 const float *A, const int lda,
                                 const float *B, const int ldb,
                                 const float beta,
                                 float *C, const int ldc);


    // This is interface for Winograd. It is not the real general
    // matrix multiply. Some parameters will invalid.
    static void WinogradSgemm(const int set_U, const int set_V, const int set_M,
                              const int M, const int N, const int K,
                              const float alpha,
                              const float *A, const int lda,
                              const float *B, const int ldb,
                              const float beta,
                              float *C, const int ldc);


    // This is interface for fullyconnet. It is not the real general
    // matrix multiply. Some parameters will invalid.
    static void DenseSgemm(const int inputs,
                           const int outputs,
                           const int batch_size,
                           const float *input,
                           const float *kernel,
                           float *output);
};
