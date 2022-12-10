#pragma once

#include <unordered_map>
#include <string>
#include <cstdint>


class GammasDict {
public:
    static GammasDict& Get();

    void Initialize(std::string filename);

    bool ProbePattern(std::uint64_t hash, float &val) const;
    bool ProbeFeature(std::uint64_t hash, float &val) const;

private:
    bool InsertPattern(std::uint64_t hash, float val);
    bool InsertFeature(std::uint64_t hash, float val);

    std::unordered_map<std::uint64_t, float> pattern_dict_;
    std::unordered_map<std::uint64_t, float> feature_dict_;
};
