#include "neural/blas/blas.h"
#include "neural/blas/fullyconnect.h"
#include "neural/blas/biases.h"

void FullyConnect::Forward(const size_t input_size,
                           const size_t output_size,
                           const std::vector<float> &input,
                           const std::vector<float> &weights,
                           const std::vector<float> &biases,
                           std::vector<float> &output, bool ReLU) {
    static constexpr int batch = 1;
    Blas::DenseSgemm((int)input_size,
                     (int)output_size,
                     batch, 
                     input.data(),
                     weights.data(),
                     output.data());

    AddVectorBiases::Forward(output_size, output, biases, ReLU);
}

std::vector<float> FullyConnect::Innerproduct(const size_t input_size,
                                              const size_t output_size,
                                              const std::vector<float> &input,
                                              const std::vector<float> &weights,
                                              const std::vector<float> &biases,
                                              bool ReLU) {
    auto output = std::vector<float>(output_size);
    output.reserve(output_size);

    static constexpr int batch = 1;
    Blas::DenseSgemm((int)input_size,
                     (int)output_size,
                     batch, 
                     input.data(),
                     weights.data(),
                     output.data());
  
    AddVectorBiases::Forward(output_size, output, biases, ReLU);

    return output;
}
