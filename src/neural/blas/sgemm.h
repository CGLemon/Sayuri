#pragma once

template <bool TA, bool TB>
class Sgemm {
public:
    static void apply(int M, int N, int K,
                      float alpha,
                      const float *A, int lda,
                      const float *B, int ldb,
                      float beta,
                      float *C, int ldc);
};
