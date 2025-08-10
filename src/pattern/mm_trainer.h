#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

#include "game/board.h"
#include "pattern/pattern.h"
#include "pattern/mm.h"

class MmTrainer {
public:
    static MmTrainer& Get();

    void Run(std::string sgf_name, std::string out_name, int min_count);
    void Test();

private:
    using FeatureSpatDict = std::unordered_map<std::uint64_t, std::string>;
    using FeatureOrder = std::vector<std::uint64_t>;
    using FeatureOrderDict = std::unordered_map<std::uint64_t, int>;
    using FeatureConuter = std::vector<int>;

    bool PatternMatch(const Board& board,
                      int feature, int dist,
                      int vertex, int color, std::uint64_t &mhash) const;

    bool FillPatterns(std::string sgfstring);
    void FillMmParticipant(std::string sgfstring);
    void InitMm();
    void FilterPatterns(int min_count);

    void SaveResult(std::string filename);

    std::vector<FeatureSpatDict> feature_spat_dicts_;   // hash  -> string
    std::vector<FeatureOrder> feature_orders_;          // index -> hash
    std::vector<FeatureOrderDict> feature_order_dicts_; // hash  -> index
    std::vector<FeatureConuter> feature_counters_;      // index -> conut

    std::unique_ptr<MinorizationMaximization> mm_;
    int num_patterns_;

    static constexpr int kMmMaxPatternDist = kMaxPatternDist;
    static constexpr int kMmMinPatternDist = 3;
    static constexpr int kMaxSgfGames = 2000;
};
