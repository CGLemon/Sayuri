#pragma once

#include <iostream>
#include <sstream>
#include <utility>
#include <string>
#include <vector>
#include <cassert>
#include <unordered_map>
#include <type_traits>
#include <stdexcept>

#define IS_SAME_ALL \
    std::is_same<T, int>::value || \
    std::is_same<T, float>::value || \
    std::is_same<T, double>::value || \
    std::is_same<T, bool>::value || \
    std::is_same<T, std::string>::value

#define IS_SAME_NUMERIC \
    std::is_same<T, int>::value || \
    std::is_same<T, float>::value || \
    std::is_same<T, double>::value

#define IS_SAME_EXCEPT_STRING \
    std::is_same<T, int>::value || \
    std::is_same<T, float>::value || \
    std::is_same<T, double>::value || \
    std::is_same<T, bool>::value

class Option {
private:
    enum class Type {
        kInvalid,
        kString,   // std::string
        kBoolean,  // bool
        kInteger,  // int
        kFloating, // single-precision floating-point
        kDouble    // double-precision floating-point
    };

    Type type_{Type::kInvalid};
    std::vector<std::string> val_list_{};

    bool use_max_{false};
    std::string max_{};

    bool use_min_{false};
    std::string min_{};

    bool default_{false};

    void Adjust();

    std::string GetCurrentVal() const;
    std::string TypeToString(Type type) const;

    template<
        typename T,
        typename = std::enable_if_t<IS_SAME_ALL>
    >
    Option(Type t, T val,
           bool use_max, std::string max,
           bool use_min, std::string min) :
               type_(t),
               use_max_(use_max), max_(max),
               use_min_(use_min), min_(min) {
                   FancyPush(val);
               }

    template<
        typename T,
        typename = std::enable_if_t<IS_SAME_ALL>
    >
    Option(Type t, T val) : type_(t) {
        FancyPush(val);
    }

    operator int() const {
        if (type_ != Type::kInteger) {
            auto err = std::ostringstream{};
            err << "Incorrect option type for \""
                    << GetCurrentVal() << "\"."
                    << " Expect " << TypeToString(Type::kInteger)
                    << " but option is " << TypeToString(type_) << ".";
            throw std::runtime_error(err.str());
        }
        return std::stoi(GetCurrentVal());
    }

    operator bool() const {
        if (type_ != Type::kBoolean) {
            auto err = std::ostringstream{};
            err << "Incorrect option type for \""
                    << GetCurrentVal() << "\"."
                    << " Expect " << TypeToString(Type::kBoolean)
                    << " but option is " << TypeToString(type_) << ".";
            throw std::runtime_error(err.str());
        }
        return (GetCurrentVal() == "true");
    }

    operator float() const {
        if (type_ != Type::kFloating) {
            auto err = std::ostringstream{};
            err << "Incorrect option type for \""
                    << GetCurrentVal() << "\"."
                    << " Expect " << TypeToString(Type::kFloating)
                    << " but option is " << TypeToString(type_) << ".";
            throw std::runtime_error(err.str());
        }
        return std::stof(GetCurrentVal());
    }

    operator double() const {
        if (type_ != Type::kDouble) {
            auto err = std::ostringstream{};
            err << "Incorrect option type for \""
                    << GetCurrentVal() << "\"."
                    << " Expect " << TypeToString(Type::kDouble)
                    << " but option is " << TypeToString(type_) << ".";
            throw std::runtime_error(err.str());
        }
        return std::stod(GetCurrentVal());
    }

    operator std::string() const {
        if (type_ != Type::kString) {
            auto err = std::ostringstream{};
            err << "Incorrect option type for \""
                    << GetCurrentVal() << "\"."
                    << " Expect " << TypeToString(Type::kString)
                    << " but option is " << TypeToString(type_) << ".";
            throw std::runtime_error(err.str());
        }
        return GetCurrentVal();
    }

    template<
        typename T,
        typename = std::enable_if_t<IS_SAME_ALL>
    >
    T FancyGet(int idx = -1) {
        std::string val;
        if (idx >= (int)val_list_.size()) {
            throw std::runtime_error("FancyGet() is overflow.");
        }
        if (idx >= 0) {
            val = val_list_[idx];
        } else {
            val = GetCurrentVal();
        }
        Option res(type_, val);
        return (T)res;
    }

    template<
        typename T,
        typename = std::enable_if_t<IS_SAME_EXCEPT_STRING>
    >
    std::string FancyCast(T val) {
        auto out = std::string{};
        if (std::is_same<T, int>::value ||
                std::is_same<T, float>::value ||
                std::is_same<T, double>::value) {
            out = std::to_string(val);
        } else if (std::is_same<T, bool>::value) {
            if (val) {
                out = std::string{"true"};
            } else {
                out = std::string{"false"};
            }
        }
        return out;
    }
    std::string FancyCast(std::string val) {
        return val;
    }
    std::string FancyCast(const char* val) {
        return std::string{val};
    }

    template<
        typename T,
        typename = std::enable_if_t<IS_SAME_ALL>
    >
    void FancyPush(T val) {
        if (default_) {
            val_list_.clear();
            default_ = false;
        }

        val_list_.emplace_back(FancyCast(val));

        HandleInvalid();
        Adjust();
    }

    void HandleInvalid() const;

public:
    Option() = default;

    void operator<<(const Option &&o) { *this = std::forward<decltype(o)>(o); }

    template<typename T>
    T Get(int idx = -1) {
        return FancyGet<T>(idx);
    }

    template<typename T>
    void Set(T val) {
        FancyPush(val);
    }

    bool IsDefault() const {
        return default_;
    };

    void SetAsDefault(bool v = true) {
        default_ = v;
    }

    int Count() const {
        return val_list_.size();
    }

    void Unique();

    std::string ToString() const;

    // Get Option object.
    template<
        typename T,
        typename = std::enable_if_t<IS_SAME_ALL>
    >
    static Option::Type GetOptionType(T /* val */) {
        if (std::is_same<T, int>::value) {
            return Type::kInteger;
        }
        if (std::is_same<T, float>::value) {
            return Type::kFloating;
        }
        if (std::is_same<T, double>::value) {
            return Type::kDouble;
        }
        if (std::is_same<T, bool>::value) {
            return Type::kBoolean;
        }
        if (std::is_same<T, std::string>::value) {
            return Type::kString;
        }
        return Type::kInvalid;
    }

    template<
        typename T,
        typename = std::enable_if_t<IS_SAME_NUMERIC>
    >
    static Option SetOption(T val, T max, T min) {
        auto out = Option(GetOptionType(val), val,
                              true, std::to_string(max),
                              true, std::to_string(min));
        out.SetAsDefault();
        return out;
    }

    template<
        typename T,
        typename = std::enable_if_t<IS_SAME_ALL>
    >
    static Option SetOption(T val) {
        auto out = Option(GetOptionType(val), val,
                              false, std::string{},
                              false, std::string{});
        out.SetAsDefault();
        return out;
    }
};

extern std::unordered_map<std::string, Option> kOptionsMap;

template<
    typename T,
    typename = std::enable_if_t<IS_SAME_ALL>
>
inline T GetOption(std::string key, int idx=-1) {
    auto it = kOptionsMap.find(key);
    T val = it->second.Get<T>(idx);
    return val;
}

template<
    typename T,
    typename = std::enable_if_t<IS_SAME_EXCEPT_STRING>
>
inline bool SetOption(std::string key, T val, bool as_default=false) {
    auto it = kOptionsMap.find(key);
    if (it != std::end(kOptionsMap)) {
        it->second.Set<T>(val);
        if (as_default) {
            it->second.SetAsDefault();
        }
        return true;
    }
    return false;
}

inline bool SetOption(std::string key, std::string val, bool as_default=false) {
    auto it = kOptionsMap.find(key);
    if (it != std::end(kOptionsMap)) {
        it->second.Set<std::string>(val);
        if (as_default) {
            it->second.SetAsDefault();
        }
        return true;
    }
    return false;
}

inline int GetOptionCount(std::string key) {
    auto it = kOptionsMap.find(key);
    return it->second.Count();
}

inline bool IsOptionDefault(std::string key) {
    auto it = kOptionsMap.find(key);
    return it->second.IsDefault();
}

inline void UniqueOption(std::string key) {
    auto it = kOptionsMap.find(key);
    it->second.Unique();
}

inline std::string OptionsToString() {
    auto out = std::ostringstream{};
    for (auto &it: kOptionsMap) {
        const auto &name = it.first;
        const auto &option = it.second;
        out << name << ": " << option.ToString() << "\n";
    }
    return out.str();
}
