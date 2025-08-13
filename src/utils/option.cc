#include "utils/option.h"

#include <set>

void Option::Adjust() {
    HandleInvalid();
    for (auto &sval : val_list_) {
        if (type_ == Type::kInteger)  {
            auto ival = std::stoi(sval);
            if (use_max_) {
                ival = std::min(ival, std::stoi(max_));
            }
            if (use_min_) {
                ival = std::max(ival, std::stoi(min_));
            }
            sval = std::to_string(ival);
        } else if (type_ == Type::kFloating) {
            auto fval = std::stof(sval);
            if (use_max_) {
                fval = std::min(fval, std::stof(max_));
            }
            if (use_min_) {
                fval = std::max(fval, std::stof(min_));
            }
            sval = std::to_string(fval);
        } else if (type_ == Type::kDouble) {
            auto dval = std::stod(sval);
            if (use_max_) {
                dval = std::min(dval, std::stod(max_));
            }
            if (use_min_) {
                dval = std::max(dval, std::stod(min_));
            }
            sval = std::to_string(dval);
        }
    }
}

std::string Option::GetCurrentVal() const {
    return *std::rbegin(val_list_);
}

std::string Option::TypeToString(Type type) const {
    if (type == Type::kString) {
        return "kString";
    }
    if (type == Type::kBoolean) {
        return "kBoolean";
    }
    if (type == Type::kInteger) {
        return "kInteger";
    }
    if (type == Type::kFloating) {
        return "kFloating";
    }
    if (type == Type::kDouble) {
        return "kDouble";
    }
    return "kInvalid";
}

void Option::HandleInvalid() const {
    if (use_max_ && use_min_) {
        if (std::stof(max_) < std::stof(min_)) {
            auto out = std::ostringstream{};
            out << " Option Error :";
            out << " Max : " << max_ << " |";
            out << " Min : " << min_ << " |";
            out << " Minimal is bigger than maximal.";
            out << " It is not accepted.";
            throw std::runtime_error(out.str());
        }
    }

    if (type_ == Type::kInvalid) {
        auto out = std::ostringstream{};
        out << " Option Error :";
        out << " Please initialize first.";
        throw std::runtime_error(out.str());
    }
};

void Option::Unique() {
    auto dest_list = std::vector<std::string>{};
    auto check_set = std::set<std::string>{};
    for (auto &val : val_list_) {
        if (check_set.count(val) == 0) {
            dest_list.emplace_back(val);
        }
        check_set.emplace(val);
    }
    std::swap(val_list_, dest_list);
}

std::string Option::ToString() const {
    auto out = std::ostringstream{};
    out << GetCurrentVal();
    if (use_max_) {
        out << ", Max: " << max_;
    }
    if (use_min_) {
        out << ", Min: " << min_;
    }
    out << ", " << TypeToString(type_);
    return out.str();
}

std::unordered_map<std::string, Option> kOptionsMap;
