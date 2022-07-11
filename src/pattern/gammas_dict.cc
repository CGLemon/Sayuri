#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

#include "gammas_dict.h"

GammasDict& GammasDict::Get() {
    static GammasDict dict_;
    return dict_;
}

bool GammasDict::InsertPattern(LocPattern pattern) {
    if (pattern.feature == LocPattern::kNoFeature) {
        return false;
    }

    if (index_dict_.find(pattern()) == std::end(index_dict_)) {
        int idx = order_.size();
        index_dict_.insert({pattern(), idx});
        order_.emplace_back(pattern);
        return true;
    }
    return false;
}

int GammasDict::Size() const{
    return order_.size();
}

bool GammasDict::ProbeGammas(std::uint64_t hash, float &val) const {
    auto it = gammas_dict_.find(hash);
    if (it == std::end(gammas_dict_)) {
        return false;
    }
    val = it->second;
    return true;
}

bool GammasDict::ProbeGammas(std::vector<LocPattern> &plist, float &val) const {
    float gamma = 1.f;
    int cnt = 0;
    for (auto& p : plist) {
        float pval;
        if (ProbeGammas(p(), pval)) {
            cnt++;
            gamma *= pval;
        }
    }
    if (cnt != 0) {
        val = gamma;
        return true;
    }
    return false;
}

int GammasDict::GetIndex(std::uint64_t hash) const {
    auto it = index_dict_.find(hash);
    if (it == std::end(index_dict_)) {
        return -1;
    }
    return it->second;
}

LocPattern GammasDict::GetPattern(int idx) const {
    return order_[idx];
}

void GammasDict::LoadPatternsGammas(std::string filename) {
    std::ifstream file;
    file.open(filename);

    if (!file.is_open()) return;

    index_dict_.clear();
    gammas_dict_.clear();
    order_.clear();

    std::string buf;

    while(std::getline(file, buf)) {

        std::uint64_t hash;
        float gammas;

        std::istringstream iss{buf};

        iss >> hash >> gammas;

        if (InsertPattern(LocPattern::FromHash(hash))) {
            gammas_dict_.insert({hash, gammas});
        }
    }

    file.close();
}

std::vector<std::uint32_t> GammasDict::GetAllFeatures() const {
    auto flist = std::vector<std::uint32_t>{};
    for (auto &p : order_) {
        auto it = std::find(std::begin(flist), std::end(flist), p.feature);
        if (it == std::end(flist)) {
            flist.emplace_back(p.feature);
        }
    }

    std::sort(std::begin(flist), std::end(flist));
    return flist;
}

int GammasDict::GetNumFeatures(std::uint32_t feature) const {
    int num = 0;
    for (auto &p : order_) {
        if (p.feature == feature) {
            num++;
        }
    }
    return num;
}

void GammasDict::Sort() {
    auto order_buf = order_;
    auto flist = GetAllFeatures();

    index_dict_.clear();
    order_.clear();

    for (auto feature : flist) {
        for (auto p : order_buf) {
            if (p.feature == feature) {
                int idx = order_.size();
                index_dict_.insert({p(), idx});
                order_.emplace_back(p);
            }
        }
    }
}
