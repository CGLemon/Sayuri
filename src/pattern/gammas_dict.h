#pragma once

#include <unordered_map>
#include <vector>
#include <string>

#include "pattern/pattern.h"

class GammasDict {
public:
    static GammasDict& Get();

    void LoadPatternsGammas(std::string filename);

    bool InsertPattern(Pattern pattern);

    int Size() const;

    bool ProbeGammas(std::uint64_t hash, float &val) const;

    int GetIndex(std::uint64_t hash) const;

    Pattern GetPattern(int idx) const;

private:
    std::unordered_map<std::uint64_t, int> index_dict_; // pattern hash -> index

    std::unordered_map<std::uint64_t, float> gammas_dict_; // pattern hash -> gammas

    std::vector<Pattern> order_; // index -> pattern
};
