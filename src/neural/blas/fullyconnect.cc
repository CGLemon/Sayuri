#include "neural/blas/blas.h"
#include "neural/blas/fullyconnect.h"
#include "neural/blas/biases.h"

#include "neural/activation.h"

void FullyConnect::Forward(const size_t input_size,
                           const size_t output_size,
                           const std::vector<float> &input,
                           const std::vector<float> &weights,
                           const std::vector<float> &biases,
                           std::vector<float> &output,
                           const Activation act) {
    static constexpr int batch = 1;
    Blas::DenseSgemm((int)input_size,
                     (int)output_size,
                     batch,
                     input.data(),
                     weights.data(),
                     output.data());

    AddVectorBiases::Forward(output_size, output, biases, act);
}
