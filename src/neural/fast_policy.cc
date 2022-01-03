#include "neural/fast_policy.h"
#include "neural/blas/convolution.h"
#include "neural/blas/biases.h"
#include "neural/fast_policy_model"
#include "utils/log.h"


#include <fstream>
#include <iostream>
#include <sstream>

FastPolicy &FastPolicy::Get() {
    static FastPolicy fp;
    return fp;
}

void FastPolicy::LoadWeights() {
    auto line = std::string{};
    auto weights = std::stringstream(kFastPolicyWeights);

    std::getline(weights, line);

    assert(line == "0"); // version

    size_t cnt = 0;
    while(std::getline(weights, line)) {
        FillWeights(line, ++cnt);
    }
}

void FastPolicy::FillWeights(std::string &weights_line, size_t cnt) {
    std::stringstream buffer(weights_line);
    double weight;
    std::vector<float> weights;
    while(buffer >> weight) {
        weights.emplace_back(weight);
    }

    assert(cnt >= 1 && cnt <= 8);

    if (cnt == 1) {
        conv_weights_1 = weights;
    } else if (cnt == 2) {
        conv_biases_1 = weights;
    } else if (cnt == 3) {
        conv_weights_2 = weights;
    } else if (cnt == 4) {
        conv_biases_2 = weights;
    } else if (cnt == 5) {
        conv_weights_3 = weights;
    } else if (cnt == 6) {
        conv_biases_3 = weights;
    } else if (cnt == 7) {
        conv_weights_4 = weights;
    } else if (cnt == 8) {
        conv_biases_4 = weights;
    }
}

std::vector<float> FastPolicy::Softmax(std::vector<float> &input) {
    auto output = std::vector<float>{};
    output.reserve(input.size());

    const auto alpha = *std::max_element(std::begin(input), std::end(input));
    auto denom = 0.0f;

    for (const auto in_val : input) {
        auto val = std::exp((in_val - alpha));
        denom += val;
        output.emplace_back(val);
    }

    for (auto &out : output) {
        out /= denom;
    }

    return output;
}

std::vector<float> FastPolicy::GetPlanes(const GameState &state) {
    const auto boardsize = state.GetBoardSize();
    const auto num_intersections = state.GetNumIntersections();
    const auto to_move =  state.GetToMove();
    auto planes = std::vector<float>(3 * num_intersections);

    for (int idx = 0; idx < num_intersections; ++idx) {
        const auto x = idx % boardsize;
        const auto y = idx / boardsize;
        const auto vtx = state.GetVertex(x, y);
        const auto color = state.GetState(vtx);

        if (color == to_move) {
            planes[0 * num_intersections + idx] = static_cast<float>(true);
        } else if (color == !to_move) {
            planes[1 * num_intersections + idx] = static_cast<float>(true);
        }
    }

    for (int idx = 0; idx < num_intersections; ++idx) {
        planes[2 * num_intersections + idx] = static_cast<float>(true);
    }

    return planes;
}

std::vector<float> FastPolicy::Forward(const GameState &state) {
    const auto board_size = state.GetBoardSize();
    const auto num_intersections = board_size * board_size;
    auto workspace = std::vector<float>(Convolution<5>::GetWorkspaceSize(board_size, 32));

    auto planes = GetPlanes(state);

    auto conv_buf_1 = std::vector<float>(num_intersections * 32);
    auto conv_buf_2 = std::vector<float>(num_intersections * 32);
    auto conv_out = std::vector<float>(num_intersections);

    Convolution<5>::Forward(board_size, 3, 32, planes, conv_weights_1, workspace, conv_buf_1);
    AddSpatialBiases::Forward(board_size, 32,  conv_buf_1, conv_biases_1, true);


    Convolution<3>::Forward(board_size, 32, 8, conv_buf_1, conv_weights_2, workspace, conv_buf_2);
    AddSpatialBiases::Forward(board_size, 8,  conv_buf_2, conv_biases_2, true);


    Convolution<3>::Forward(board_size, 8, 8, conv_buf_2, conv_weights_3, workspace, conv_buf_1);
    AddSpatialBiases::Forward(board_size, 8,  conv_buf_1, conv_biases_3, true);


    Convolution<3>::Forward(board_size, 8, 1, conv_buf_1, conv_weights_4, workspace, conv_buf_2);
    AddSpatialBiases::Forward(board_size, 1,  conv_buf_2, conv_biases_4, false);

    for (int i = 0; i < num_intersections; ++i) {
        conv_out[i] = conv_buf_2[i];
    }

    return Softmax(conv_out);
}
