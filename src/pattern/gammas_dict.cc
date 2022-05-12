#include <fstream>
#include <iostream>
#include <sstream>

#include "gammas_dict.h"

GammasDict& GammasDict::Get() {
    static GammasDict dict_;
    return dict_;
}

bool GammasDict::InsertPattern(LocPattern pattern) {
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
