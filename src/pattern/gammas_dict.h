#pragma once

#include <unordered_map>

class GammasDict {
public:
    static GammasDict& Get();

    void Initialize(std::string filename);

    bool Probe(std::uint64_t hash, float &val) const;

    bool Hit3x3(std::uint64_t hash, float *val, bool skip_bad);

private:
    float val_3x3_avg_;

    bool Insert3x3(std::uint64_t hash, float val);

    bool Insert(std::uint64_t hash, float val);

    std::unordered_map<std::uint64_t, float> dict_;

    std::unordered_map<std::uint64_t, float> dict_3x3_;
};
