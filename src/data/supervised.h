#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

class Supervised {
public:
    static Supervised &Get();

    void FromSgf(std::string sgf_name,
                     std::string out_name_prefix,
                     float cutoff_games_prob,
                     float cutoff_moves_rate) const;

private:
    bool SgfProcess(std::string &sgfstring,
                        std::ostream &out_file,
                        bool cut_off,
                        float cutoff_moves_rate) const;

};
