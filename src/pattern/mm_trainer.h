#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "game/simple_board.h"
#include "pattern/pattern.h"
#include "pattern/mm.h"

class MmTrainer {
public:
    static MmTrainer& Get();

    void Run(std::string sgf_name, std::string out_name);
    void Test();

private:
    using FeatureSpatDict = std::unordered_map<std::uint64_t, std::string>;
    using FeatureOrder = std::vector<std::uint64_t>;
    using FeatureOrderDict = std::unordered_map<std::uint64_t, int>;
    using FeatureConuter = std::vector<int>;

    bool PatternMatch(const SimpleBoard& board,
                          int feature, int dist,
                          int vertex, std::uint64_t &mhash) const;

    void FillPatterns(std::string sgfstring);
    void FillMmParticipant(std::string sgfstring);
    void InitMm();
    void FilterPatterns();

    void SaveResult(std::string filename);

    std::vector<FeatureSpatDict> feature_spat_dicts_;   // hash  -> string
    std::vector<FeatureOrder> feature_orders_;          // index -> hash
    std::vector<FeatureOrderDict> feature_order_dicts_; // hash  -> index
    std::vector<FeatureConuter> feature_counters_;      // index -> conut

    MinorizationMaximization mm_;
    int num_patterns_;

    static constexpr int kMmMaxPatternDist = kMaxPatternDist;
    static constexpr int kMmMinPatternDist = 3;
};
