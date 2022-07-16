#include "pattern/pattern.h"
#include "pattern/gammas_dict.h"
#include "pattern/pattern_gammas.h"
#include "game/types.h"

#include <iostream>
#include <sstream>
#include <string>

GammasDict& GammasDict::Get() {
    static GammasDict dict;
    return dict;
}

void GammasDict::Initialize() {
    PtcoordsInit();
    PatternHashInit();

    std::istringstream iss{kPatternGammas};
    std::string line;

    val_3x3_avg_ = 0;

    while (std::getline(iss, line)) {
        if (line.empty()) {
            continue;
        }

        if (line[0] == '#') {
            continue;
        }

        std::istringstream data{line};

        float gamma;
        int dist;
        std::string spat;

        data >> gamma >> dist >> spat;

        for (int symm = 0; symm < 8; ++symm) {
            for (int c = 0; c < 2; ++c) {
                std::uint64_t hash = PatternHash[0][kInvalid][0];
                int color_map[2][4] = {
                    {kBlack, kWhite, kEmpty, kInvalid},
                    {kWhite, kBlack, kEmpty, kInvalid}
                };

                for (int i = kPointIndex[2]; i < kPointIndex[dist + 1]; ++i) {
                    int color = CharToColor(spat[i]);
                    color = color_map[c][color];

                    if (color != kInvalid) {
                         hash ^= PatternHash[symm][color][i];
                    }
                }
                Insert(hash, gamma);

                if (dist == 3) {
                    Insert3x3(hash, gamma);
                }
            }
        }
    }
    val_3x3_avg_ /= dict_3x3_.size();
    val_3x3_avg_ *= 0.75f;
}

bool GammasDict::Hit3x3(std::uint64_t hash, float *val) {
    auto it = dict_3x3_.find(hash);
    if (it == std::end(dict_3x3_)) {
        return false;
    }
    if (it->second < val_3x3_avg_) {
        return false;
    }

    if (val) {
        *val = it->second;
    }
    return true;
}

bool GammasDict::Insert3x3(std::uint64_t hash, float val) {
    if (Hit3x3(hash, nullptr)) {
        return false;
    }
    dict_3x3_.insert({hash, val});
    val_3x3_avg_ += val;
    return true;
}

bool GammasDict::Insert(std::uint64_t hash, float val) {
    float g;
    if (Probe(hash, g)) {
        return false;
    }
    dict_.insert({hash, val});
    return true;
}

bool GammasDict::Probe(std::uint64_t hash, float &val) const {
    auto it = dict_.find(hash);
    if (it == std::end(dict_)) {
        return false;
    }
    val = it->second;
    return true;
}
