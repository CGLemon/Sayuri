#include "utils/option.h"

#include <sstream>

bool Option::BoundaryValid() const {
    OptionHandle();
    return !(max_ == 0 && min_ == 0);
}

template<>
Option Option::setoption<std::string>(std::string val, int /*max*/, int /*min*/) {
    return Option{Type::kString, val, 0, 0};
}

template<>
Option Option::setoption<const char *>(const char *val, int /*max*/, int /*min*/) {
    return Option{Type::kString, std::string{val}, 0, 0};
}

template<>
Option Option::setoption<bool>(bool val, int /*max*/, int /*min*/) {
    if (val) {
        return Option{Type::kBool, "true", 0, 0};
    }
    return Option{Type::kBool, "false", 0, 0};
}

template<>
Option Option::setoption<int>(int val, int max, int min) {
    auto op = Option{Type::kInteger, std::to_string(val), max, min};
    op.Adjust<int>();
    return op;
}

template<>
Option Option::setoption<float>(float val, int max, int min) {
    auto op = Option{Type::kFloat, std::to_string(val), max, min};
    op.Adjust<float>();
    return op;
}

template<>
Option Option::setoption<char>(char val, int /*max*/, int /*min*/) {
    return Option{Type::kChar, std::string{val}, 0, 0};
}

#define OPTION_EXPASSION(T) \
template<>                  \
T Option::Get<T>() const {  \
    return (T)*this;        \
}

OPTION_EXPASSION(std::string)
OPTION_EXPASSION(bool)
OPTION_EXPASSION(float)
OPTION_EXPASSION(int)
OPTION_EXPASSION(char)

template<>
const char* Option::Get<const char*>() const {
    return value_.c_str();
}

#undef OPTION_EXPASSION

template<>
void Option::Set<std::string>(std::string value) {
    OptionHandle();
    value_ = value;
}

template<>
void Option::Set<bool>(bool value) {
    OptionHandle();
    if (value) {
        value_ = std::string{"true"};
    } else {
        value_ = std::string{"false"};
    }
}

template<>
void Option::Set<int>(int value) {
    OptionHandle();
    value_ = std::to_string(value);
    Adjust<int>();
}

template<>
void Option::Set<float>(float value) {
    OptionHandle();
    value_ = std::to_string(value);
    Adjust<float>();
}

template<>
void Option::Set<char>(char value) {
    OptionHandle();
    value_ = std::string{value};
}

void Option::OptionHandle() const {
    if (max_ < min_) {
        auto out = std::ostringstream{};
        out << " Option Error :";
        out << " Max : " << max_ << " |";
        out << " Min : " << min_ << " |";
        out << " Minimal is bigger than maximal.";
        out << " It is not accepted.";
        throw std::runtime_error(out.str());
    }

    if (type_ == Type::kInvalid) {
        auto out = std::ostringstream{};
        out << " Option Error :";
        out << " Please initialize first.";
        throw std::runtime_error(out.str());
    }
};
