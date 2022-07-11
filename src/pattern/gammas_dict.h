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

    // Get the number of patterns.
    int Size() const;

    // Get the Gamma value from a pattern.
    bool ProbeGammas(std::uint64_t hash, float &val) const;

    // Get the Gamma value from patterns list.
    bool ProbeGammas(std::vector<LocPattern> &plist, float &val) const;

    int GetIndex(std::uint64_t hash) const;

    // Return all features list.
    std::vector<std::uint32_t> GetAllFeatures() const;

    // Return the number of features which we collected.
    int GetNumFeatures(std::uint32_t f) const;

    // Sort the patterns by features order.
    void Sort();

    LocPattern GetPattern(int idx) const;

private:
    std::unordered_map<std::uint64_t, int> index_dict_; // pattern hash -> index

    std::unordered_map<std::uint64_t, float> gammas_dict_; // pattern hash -> gammas

    std::vector<LocPattern> order_; // index -> pattern
};
