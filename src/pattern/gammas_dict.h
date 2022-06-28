#pragma once

#include <unordered_map>
#include <vector>
#include <string>

#include "pattern/pattern.h"

class GammasDict {
public:
    static GammasDict& Get();

    void LoadPatternsGammas(std::string filename);

    bool InsertPattern(LocPattern pattern);

    int Size() const;

    bool ProbeGammas(std::uint64_t hash, float &val) const;

    bool ProbeGammas(std::vector<LocPattern> &plist, float &val) const;

    int GetIndex(std::uint64_t hash) const;

    LocPattern GetPattern(int idx) const;

private:
    std::unordered_map<std::uint64_t, int> index_dict_; // pattern hash -> index

    std::unordered_map<std::uint64_t, float> gammas_dict_; // pattern hash -> gammas

    std::vector<LocPattern> order_; // index -> pattern
};
