#include "pattern/pattern.h"
#include "pattern/gammas_dict.h"
#include "game/types.h"
#include "utils/log.h"
#include "utils/format.h"

#include <stdexcept>
#include <fstream>
#include <sstream>
#include <string>

GammasDict& GammasDict::Get() {
    static GammasDict dict;
    return dict;
}

void GammasDict::LoadPatterns(std::string filename) {
    if (filename.empty()) {
        return;
    }

    try {
        std::stringstream iss;
        std::string line;

        auto file = std::ifstream{};
        file.open(filename.c_str());

        if (file.is_open()) {
            iss << file.rdbuf();
            file.close();
        } else {
            throw std::runtime_error{"Cann't open the file."};
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
                // feature
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
        LOGGING << GetInformation();
    } catch (const std::exception& e) {
        LOGGING << "Fail to load the patterns file: " << filename << '!' << std::endl;
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

std::string GammasDict::GetInformation() const {
    auto oss = std::ostringstream();
    oss << Format("The patterns file contains %zu patterns and %zu features.\n",
                      pattern_dict_.size(), feature_dict_.size());
    return oss.str();
}

bool GammasDict::Valid() const {
    return pattern_dict_.size() + feature_dict_.size() > 0;
}
