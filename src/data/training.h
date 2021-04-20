#pragma once

#include <vector>
#include <iostream>

struct TrainingBuffer {
    int version;

    int mode;

    int board_size;

    int side_to_move;

    float komi;

    std::vector<float> planes;

    std::vector<float> probabilities;

    std::vector<float> auxiliary_probabilities;

    std::vector<float> ownership;

    int winner;

    float final_score;

    void StreamOut(std::ostream &out);

    friend std::ostream &operator<<(std::ostream& out, TrainingBuffer& buf);
};
