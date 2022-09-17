#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "pattern/mm.h"

class MmTrainer {
public:
    static MmTrainer& Get();

    void Run(std::string sgf_name);

private:
    using FeatureSpatDict = std::unordered_map<std::uint64_t, std::string>;
    using FeatureOrder = std::vector<std::uint64_t>;
    using FeatureOrderDict = std::unordered_map<std::uint64_t, int>;

    void FillPatterns(std::string sgfstring);
    void FillMmParticipant(std::string sgfstring);
    void SaveResult(std::string filename);

    std::vector<FeatureSpatDict> feature_spat_dicts_;   // hash -> string
    std::vector<FeatureOrder> feature_orders_;          // index -> hash
    std::vector<FeatureOrderDict> feature_order_dicts_; // hash -> index

    MinorizationMaximization mm_;
};
