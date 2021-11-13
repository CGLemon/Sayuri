#pragma once

#include <utility>
#include <string>
#include <cassert>

#include <unordered_map>

class Option {
private:
    enum class Type {
        kInvalid,
        kString,
        kBool,
        kInteger,
        kFloat,
        kChar
    };

    Type type_{Type::kInvalid};
    std::string value_{};
    int max_{0};
    int min_{0};

    Option(Type t, std::string val, int max, int min) :
               type_(t), value_(val), max_(max), min_(min) {}

    operator int() const {
        assert(type_ == Type::kInteger);
        return std::stoi(value_);
    }

    operator bool() const {
        assert(type_ == Type::kBool);
        return (value_ == "true");
    }

    operator float() const {
        assert(type_ == Type::kFloat);
        return std::stof(value_);
    }

    operator char() const {
        assert(type_ == Type::kChar);
        return char(value_[0]);
    }

    operator std::string() const {
        assert(type_ == Type::kString);
        return value_;
    }

    template<typename T>
    void Adjust();

    // Return true if the max and min are valid.
    bool BoundaryValid() const;

    // Throw error if the max and min are not accept.
    void OptionHandle() const;

public:
    Option() = default;

    void operator<<(const Option &&o) { *this = std::forward<decltype(o)>(o); }

    // Get Option object.
    template<typename T>
    static Option setoption(T val, int max = 0, int min = 0);

    // Get the value. We need to assign type.
    template<typename T>
    T Get() const;

    // Set the value.
    template<typename T>
    void Set(T value);
};

// Adjust the value. Be sure the value is not bigger 
// than maximal and smaller than minimal.
template<typename T>
void Option::Adjust() {
    if (!BoundaryValid()) {
        return;
    }

    const T upper = static_cast<T>(max_);
    const T lower = static_cast<T>(min_);
    const T val = static_cast<T>(*this);

    if (val > upper) {
        Set<T>(upper);
    } else if (val < lower) {
        Set<T>(lower);
    }
}
