#pragma once

#include <unordered_map>
#include <vector>

#include "game/game_state.h"
#include "pattern/gammas_dict.h"

class PatternsScan {
public:
    static PatternsScan& Get();

    void MMTraining(std::string sgf_name, std::string filename) const;

private:
    void CollectPatterns(std::string sgfstring, GammasDict &dict) const;

    void CollectGammas(std::string sgfstring,
                           GammasDict &dict,
                           std::ostream &out) const;
};
