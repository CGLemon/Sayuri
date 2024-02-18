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
    Output format is here. Every v2 data package is 53 lines.

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
     L47       : Ownership
     L48       : Result
     L49       : Average Q Value, Short, Middel, Long
     L50       : Final Score
     L51       : Average Score Lead, Short, Middel, Long

     ------- Misc data -------
     L52       : Q Stddev, Score Stddev
     L53       : KLD

  */
    void StreamOut(std::ostream &out) const;
};

int GetTrainingVersion();

int GetTrainingMode();
