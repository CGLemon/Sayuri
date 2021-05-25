#include "neural/network_basic.h"
#include "neural/description.h"
#include "game/game_state.h"
#include "utils/cache.h"

#include "fast_policy_model"

#include <vector>
#include <algorithm>
#include <string>

class FastPolicy {
public:
    static FastPolicy &Get();

    void LoaderFile();

    std::vector<float> Forward(const GameState &state);

private:

    std::vector<float> GetPlanes(const GameState &state);

    void FillWeights(std::string &weights, size_t cnt);

    static std::vector<float> Softmax(std::vector<float> &input);


    std::vector<float> conv_weights_1;
    std::vector<float> conv_biases_1;

    std::vector<float> conv_weights_2;
    std::vector<float> conv_biases_2;

    std::vector<float> conv_weights_3;
    std::vector<float> conv_biases_3;

    std::vector<float> conv_weights_4;
    std::vector<float> conv_biases_4;

    std::vector<float> conv_weights_5;
    std::vector<float> conv_biases_5;
};

