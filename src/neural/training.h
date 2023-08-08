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
    std::vector<int> expected_values;

    int result;

    float q_value;
    float avg_q_value;
    float short_avg_q, middle_avg_q, long_avg_q;

    float final_score;
    float score_lead;
    float avg_score_lead;
    float short_avg_score, middle_avg_score, long_avg_score;

    float q_stddev, score_stddev;

    float kld;
    float rule;
    float wave;

    bool discard{false};

 /*
    Output format is here. Every v2 data package is 54 lines.

    ------- Version -------
     L1        : Version
     L2        : Mode
     
     ------- Inputs data -------
     L3        : Board Size
     L4        : Komi
     L5        : Rule
     L6        : Wave
     L7  - L43 : Binary Features
     L44       : Current Player
    
     ------- Prediction data -------
     L45       : Probabilities
     L46       : Auxiliary Probabilities
     L47       : Expected Values
     L48       : Ownership
     L49       : Result
     L50       : Average Q Value, Short, Middel, Long
     L51       : Final Score
     L52       : Average Score Lead, Short, Middel, Long

     ------- Misc data -------
     L53       : Q Stddev, Score Stddev
     L54       : KLD

  */
    void StreamOut(std::ostream &out) const;
};

int GetTrainingVersion();

int GetTrainingMode();
