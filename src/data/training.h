#pragma once

#include <vector>
#include <iostream>

struct Training {
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

    float q_value;

    float final_score;
 /*
    Output format is here. EEvery data package are 45 lines .

    ------- claiming -------
     L1       : Version
     L2       : Mode
     
     ------- Inputs data -------
     L3       : Board size
     L4       : Komi
     L5  - L38: Binary features
     L39      : Current Player
    
     ------- Prediction data -------
     L40      : Probabilities
     L41      : Auxiliary probabilities
     L42      : Ownership
     L43      : Result
     L44      : Q value
     L45      : Final score
  */
    void StreamOut(std::ostream &out) const;
};

int GetTrainigVersion();

int GetTrainigMode();
