#pragma once

#include <unordered_map>

class GammasDict {
public:
    static GammasDict& Get();

    void Initialize();

    bool Probe(std::uint64_t hash, float &val) const;

    bool Insert(std::uint64_t hash, float val);

private:
    std::unordered_map<std::uint64_t, float> dict_;
};
