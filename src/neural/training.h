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

    std::vector<float> probabilities;

    std::vector<float> auxiliary_probabilities;

    std::vector<int> ownership;

    int result;

    float q_value;

    float final_score;

    bool discard{false};

 /*
    Output format is here. Every data package is 45 lines.

    ------- Version -------
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

int GetTrainingVersion();

int GetTrainingMode();
