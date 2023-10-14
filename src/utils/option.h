#pragma once

#include <utility>
#include <string>
#include <vector>
#include <cassert>
#include <unordered_map>
#include <type_traits>
#include <stdexcept>

class Option {
private:
    enum class Type {
        kInvalid,
        kString,  // std::string
        kBoolean, // bool
        kInteger, // int
        kFloating // float
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

    template<
        typename T,
        typename = std::enable_if_t<
                       std::is_same<T, int>::value ||
                       std::is_same<T, float>::value ||
                       std::is_same<T, bool>::value ||
                       std::is_same<T, std::string>::value
                   >
    >
    Option(Type t, T val,
           bool use_max, std::string max,
           bool use_min, std::string min) :
               type_(t),
               use_max_(use_max), max_(max),
               use_min_(use_min), min_(min) {
                   FancyPush(val);
                   Adjust();
               }

    Option(Type t, std::string val)
               : type_(t)
                 {
                     val_list_.emplace_back(val);
                 }

    operator int() const {
        if (type_ != Type::kInteger) {
            throw std::runtime_error("Not the correct casting option type.");
        }
        return std::stoi(GetCurrentVal());
    }

    operator bool() const {
        if (type_ != Type::kBoolean) {
            throw std::runtime_error("Not the correct casting option type.");
        }
        return (GetCurrentVal() == "true");
    }

    operator float() const {
        if (type_ != Type::kFloating) {
            throw std::runtime_error("Not the correct casting option type.");
        }
        return std::stof(GetCurrentVal());
    }

    operator std::string() const {
        if (type_ != Type::kString) {
            throw std::runtime_error("Not the correct casting option type.");
        }
        return GetCurrentVal();
    }

    template<
        typename T,
        typename = std::enable_if_t<
                       std::is_same<T, int>::value ||
                       std::is_same<T, float>::value ||
                       std::is_same<T, bool>::value ||
                       std::is_same<T, std::string>::value
                   >
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
        typename = std::enable_if_t<
                       std::is_same<T, int>::value ||
                       std::is_same<T, float>::value ||
                       std::is_same<T, bool>::value
                   >
    >
    std::string FancyCasting(T val) {
        auto out = std::string{};
        if (std::is_same<T, int>::value ||
                std::is_same<T, float>::value) {
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
    std::string FancyCasting(std::string val) {
        return val;
    }
    std::string FancyCasting(const char* val) {
        return std::string{val};
    }

    template<
        typename T,
        typename = std::enable_if_t<
                       std::is_same<T, int>::value ||
                       std::is_same<T, float>::value ||
                       std::is_same<T, bool>::value ||
                       std::is_same<T, std::string>::value
                   >
    >
    void FancyPush(T val) {
        if (default_) {
            val_list_.clear();
            default_ = false;
        }

        val_list_.emplace_back(FancyCasting(val));

        HandleInvalid();
        Adjust();

        (T)(*this);
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

    // Get Option object.
    template<
        typename T,
        typename = std::enable_if_t<
                       std::is_same<T, int>::value ||
                       std::is_same<T, float>::value
                   >
    >
    static Option SetOption(T val, T max, T min) {
        auto t = Type::kInvalid;

        if (std::is_same<T, int>::value) {
            t = Type::kInteger;
        } else if (std::is_same<T, float>::value) {
            t = Type::kFloating;
        }
        auto out = Option(t, val,
                              true, std::to_string(max),
                              true, std::to_string(min));
        out.SetAsDefault();
        return out;
    }

    template<
        typename T,
        typename = std::enable_if_t<
                       std::is_same<T, int>::value ||
                       std::is_same<T, float>::value ||
                       std::is_same<T, bool>::value ||
                       std::is_same<T, std::string>::value
                   >
    >
    static Option SetOption(T val) {
        auto t = Type::kInvalid;

        if (std::is_same<T, int>::value) {
            t = Type::kInteger;
        } else if (std::is_same<T, float>::value) {
            t = Type::kFloating;
        } else if (std::is_same<T, bool>::value) {
            t = Type::kBoolean;
        } else if (std::is_same<T, std::string>::value) {
            t = Type::kString;
        }

        auto out = Option(t, val,
                              false, std::string{},
                              false, std::string{});
        out.SetAsDefault();
        return out;
    }
};

extern std::unordered_map<std::string, Option> kOptionsMap;

template<
    typename T,
    typename = std::enable_if_t<
                   std::is_same<T, int>::value ||
                   std::is_same<T, float>::value ||
                   std::is_same<T, bool>::value ||
                   std::is_same<T, std::string>::value
               >
>
inline T GetOption(std::string key, int idx=-1) {
    auto it = kOptionsMap.find(key);
    T val = it->second.Get<T>(idx);
    return val;
}


template<
    typename T,
    typename = std::enable_if_t<
                   std::is_same<T, int>::value ||
                   std::is_same<T, float>::value ||
                   std::is_same<T, bool>::value
               >
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
