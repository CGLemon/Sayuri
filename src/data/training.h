#pragma once

#include <vector>
#include <iostream>

struct TrainingBuffer {
    int version;

    int mode;

    int board_size;

    float komi;

    int side_to_move;

    std::vector<float> planes;

    int probabilities_index{-1};

    std::vector<float> probabilities;

    int auxiliary_probabilities_index{-1};

    std::vector<float> auxiliary_probabilities;

    std::vector<int> ownership;

    int result;

    float final_score;

    void StreamOut(std::ostream &out) const;
};
