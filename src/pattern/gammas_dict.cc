#include "pattern/pattern.h"
#include "pattern/gammas_dict.h"
#include "game/types.h"

#include <fstream>
#include <sstream>
#include <string>

GammasDict& GammasDict::Get() {
    static GammasDict dict;
    return dict;
}

void GammasDict::Initialize(std::string filename) {
    std::stringstream iss;
    std::string line;

    if (filename.empty()) {
        return;
    }

    auto file = std::ifstream{};
    file.open(filename.c_str());

    if (file.is_open()) {
        iss << file.rdbuf();
        file.close();
    } else {
        return;
    }

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

        data >> gamma >> dist;

        if (dist == 0) {
            // featurn
            std::uint64_t hash;
            data >> hash;

            InsertFeature(hash, gamma);
        } else if (dist > 0) {
            // pattern
            std::string spat;
            data >> spat;

            for (int symm = 0; symm < 8; ++symm) {
                std::uint64_t hash = PatternHash[0][kInvalid][0];

                for (int i = kPointIndex[2]; i < kPointIndex[dist + 1]; ++i) {
                    int color = CharToColor(spat[i]);
                    if (color != kInvalid) {
                         hash ^= PatternHash[symm][color][i];
                    }
                }
                InsertPattern(hash, gamma);
            }
        }
    }
}

bool GammasDict::ProbePattern(std::uint64_t hash, float &val) const {
    auto it = pattern_dict_.find(hash);
    if (it == std::end(pattern_dict_)) {
        return false;
    }
    val = it->second;
    return true;
}

bool GammasDict::ProbeFeature(std::uint64_t hash, float &val) const {
    auto it = feature_dict_.find(hash);
    if (it == std::end(feature_dict_)) {
        return false;
    }
    val = it->second;
    return true;
}

bool GammasDict::InsertPattern(std::uint64_t hash, float val) {
    float g;
    if (ProbePattern(hash, g)) {
        return false;
    }
    pattern_dict_.insert({hash, val});
    return true;
}

bool GammasDict::InsertFeature(std::uint64_t hash, float val) {
    float g;
    if (ProbeFeature(hash, g)) {
        return false;
    }
    feature_dict_.insert({hash, val});
    return true;
}
