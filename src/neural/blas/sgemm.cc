#include "neural/blas/sgemm.h"

void sgemm_nn(int M, int N, int K, float alpha, const float *A, int lda,
              const float *B, int ldb, float *C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float A_PART = alpha * A[i * lda + k];
            for (int j = 0; j < N; ++j) {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

void sgemm_nt(int M, int N, int K, float alpha, const float *A, int lda,
              const float *B, int ldb, float *C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
          float sum = 0;
          for (int k = 0; k < K; ++k) {
              sum += alpha * A[i * lda + k] * B[j * ldb + k];
          }
          C[i * ldc + j] += sum;
       }
    }
}

void sgemm_tn(int M, int N, int K, float alpha, const float *A, int lda,
              const float *B, int ldb, float *C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float A_PART = alpha * A[k * lda + i];
            for (int j = 0; j < N; ++j) {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

void sgemm_tt(int M, int N, int K, float alpha, const float *A, int lda,
              const float *B, int ldb, float *C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += alpha * A[i + k * lda] * B[k + j * ldb];
            }
            C[i * ldc + j] += sum;
        }
    }
}

#define INITIALIZE_SGEMM(M, N, beta)  \
    for (int i = 0; i < M; ++i) {     \
        for (int j = 0; j < N; ++j) { \
            C[i * ldc + j] *= beta;   \
        }                             \
    }



template <>
void Sgemm<false, false>::apply(int M, int N, int K,
                                float alpha,
                                const float *A, int lda,
                                const float *B, int ldb,
                                float beta,
                                float *C, int ldc) {
    INITIALIZE_SGEMM(M, N, beta);
    sgemm_nn(M, N, K, alpha, A, lda, B, ldb, C, ldc);
}

template <>
void Sgemm<true, false>::apply(int M, int N, int K,
                               float alpha,
                               const float *A, int lda,
                               const float *B, int ldb,
                               float beta,
                               float *C, int ldc) {
    INITIALIZE_SGEMM(M, N, beta);
    sgemm_tn(M, N, K, alpha, A, lda, B, ldb, C, ldc);
}

template <>
void Sgemm<false, true>::apply(int M, int N, int K,
                               float alpha,
                               const float *A, int lda,
                               const float *B, int ldb,
                               float beta,
                               float *C, int ldc) {
    INITIALIZE_SGEMM(M, N, beta);
    sgemm_nt(M, N, K, alpha, A, lda, B, ldb, C, ldc);
}

template <>
void Sgemm<true, true>::apply(int M, int N, int K,
                              float alpha,
                              const float *A, int lda,
                              const float *B, int ldb,
                              float beta,
                              float *C, int ldc) {
    INITIALIZE_SGEMM(M, N, beta);
    sgemm_tt(M, N, K, alpha, A, lda, B, ldb, C, ldc);
}

