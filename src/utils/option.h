#pragma once

#include <algorithm>
#include <any>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace option_detail {

template <typename T> inline constexpr bool IsEnum() {
    return std::is_enum_v<T>;
}

template <typename T>
inline constexpr bool kIsSupported =
    std::is_same_v<T, int> || std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, bool> || std::is_same_v<T, std::string> || IsEnum<T>();

template <typename T>
using OptionValueTypeT = std::conditional_t<std::is_same_v<std::decay_t<T>, char*> ||
                                                std::is_same_v<std::decay_t<T>, const char*>,
                                            std::string,
                                            std::decay_t<T>>;

template <typename T>
inline constexpr bool kIsNumeric =
    std::is_same_v<T, int> || std::is_same_v<T, float> || std::is_same_v<T, double>;

inline bool IsNumeric(std::type_index t) {
    return t == typeid(int) || t == typeid(float) || t == typeid(double);
}

inline bool IsBoolean(std::type_index t) {
    return t == typeid(bool);
}

inline bool IsString(std::type_index t) {
    return t == typeid(std::string);
}

std::string TypeName(std::type_index t);
std::string AnyToString(const std::any& a);
std::any ParseFromString(std::type_index t, std::string_view raw);

} // namespace option_detail

class Option {
public:
    using SetterFn = std::function<void(Option&, std::string_view)>;

    Option() = default;

    template <typename T> static Option Make(T value) {
        using ValueT = option_detail::OptionValueTypeT<T>;
        static_assert(option_detail::kIsSupported<ValueT>, "Unsupported option type.");
        Option o;
        o.type_ = typeid(ValueT);
        o.is_enum_ = std::is_enum_v<ValueT>;
        o.PushDefault(std::any(ValueT(std::move(value))));
        return o;
    }

    template <typename T> T Get(int idx = -1) const {
        static_assert(option_detail::kIsSupported<T>, "Unsupported option type.");
        RequireType<T>();
        if (history_.empty()) {
            throw std::runtime_error("Option has no value.");
        }
        if (idx >= static_cast<int>(history_.size())) {
            throw std::runtime_error("Option::Get index out of range.");
        }
        const std::any& slot = (idx < 0) ? history_.back() : history_[idx];
        return std::any_cast<T>(slot);
    }

    template <typename T> void Set(T value) {
        using ValueT = option_detail::OptionValueTypeT<T>;
        static_assert(option_detail::kIsSupported<ValueT>, "Unsupported option type.");
        RequireType<ValueT>();
        Push(std::any(ValueT(std::move(value))));
    }

    template <typename T> void SetRange(T min, T max) {
        static_assert(option_detail::kIsNumeric<T>, "Bounds only valid for numeric types.");
        RequireInitialized();
        if (has_choices_) {
            throw std::runtime_error("Option Error: range cannot be combined with choices.");
        }
        if (type_ != typeid(T)) {
            throw std::runtime_error("Option Error: bounds type does not match option value type.");
        }
        if (max < min) {
            throw std::runtime_error("Option Error: max < min.");
        }
        min_ = std::any(std::move(min));
        max_ = std::any(std::move(max));
        has_range_ = true;
        ClampLast();
    }

    template <typename T>
    void SetChoices(std::initializer_list<std::string> names, std::initializer_list<T> values) {
        static_assert(option_detail::kIsSupported<T>, "Unsupported choice type.");
        RequireInitialized();
        if (has_range_) {
            throw std::runtime_error("Option Error: choices cannot be combined with range.");
        }
        if (type_ != typeid(T)) {
            throw std::runtime_error(
                "Option Error: choices type does not match option value type.");
        }
        if (names.size() != values.size()) {
            throw std::runtime_error("Option Error: choices names/values size mismatch.");
        }
        if (names.size() == 0) {
            throw std::runtime_error("Option Error: choices list is empty.");
        }

        std::vector<std::string> name_vec(std::begin(names), std::end(names));
        for (size_t i = 0; i < name_vec.size(); ++i) {
            for (size_t j = i + 1; j < name_vec.size(); ++j) {
                if (name_vec[i] == name_vec[j]) {
                    throw std::runtime_error("Option Error: duplicate choice name: " + name_vec[i]);
                }
            }
        }

        std::vector<std::any> value_vec;
        value_vec.reserve(values.size());
        for (const auto& v : values) {
            value_vec.emplace_back(std::any(v));
        }

        choice_names_ = std::move(name_vec);
        choice_values_ = std::move(value_vec);
        choice_equals_ = [](const std::any& a, const std::any& b) {
            return std::any_cast<T>(a) == std::any_cast<T>(b);
        };
        choice_to_string_ = [](const std::any& a) -> std::string {
            if constexpr (std::is_enum_v<T>) {
                return std::to_string(static_cast<long long>(std::any_cast<T>(a)));
            } else {
                return option_detail::AnyToString(a);
            }
        };
        has_choices_ = true;

        for (const auto& v : history_) {
            EnsureValueHasChoice(v);
        }
    }

    void SetFromString(std::string_view raw);

    void SetSetter(SetterFn fn) {
        setter_ = std::move(fn);
    }
    bool HasSetter() const {
        return static_cast<bool>(setter_);
    }
    void SetNoValue(bool v = true) {
        no_value_ = v;
    }
    bool AllowsNoValue() const {
        return no_value_;
    }

    void SetHelper(std::string helper) {
        helper_ = std::move(helper);
    }
    void SetGroup(std::string group) {
        group_ = std::move(group);
    }
    const std::string& Helper() const {
        return helper_;
    }
    const std::string& Group() const {
        return group_;
    }

    const std::vector<std::string>& ChoiceNames() const {
        return choice_names_;
    }
    bool HasChoices() const {
        return has_choices_;
    }
    bool IsDefault() const {
        return is_default_;
    }
    bool IsBoolean() const {
        return option_detail::IsBoolean(type_);
    }
    void SetCurrentAsDefault() {
        if (history_.empty()) {
            throw std::runtime_error(
                "Option Error: cannot set current value as default on empty history.");
        }

        is_default_ = true;
        default_ = history_.back();
        history_.clear();
        history_.emplace_back(default_);
    }
    int Count() const {
        return static_cast<int>(history_.size());
    }

    void Unique();
    std::string ToDebugString() const;
    std::string HelpMetadata() const;

private:
    std::vector<std::any> history_;
    std::any default_;

    std::any min_;
    std::any max_;

    std::vector<std::string> choice_names_;
    std::vector<std::any> choice_values_;
    std::function<bool(const std::any&, const std::any&)> choice_equals_;
    std::function<std::string(const std::any&)> choice_to_string_;

    SetterFn setter_;

    std::string helper_;
    std::string group_;

    std::type_index type_ = typeid(void);
    bool is_enum_ = false;
    bool has_range_ = false;
    bool has_choices_ = false;
    bool is_default_ = false;
    bool no_value_ = false;

    static std::string HelpTypePlaceholder(std::type_index t, bool is_enum);
    void EnsureValueHasChoice(const std::any& v) const;
    void Push(std::any v);
    void PushDefault(std::any v);

    template <typename T> bool TryClampLastAs() {
        if (type_ != typeid(T)) {
            return false;
        }
        std::any& slot = history_.back();
        T value = std::any_cast<T>(slot);
        if (has_range_) {
            value = std::max(value, std::any_cast<T>(min_));
            value = std::min(value, std::any_cast<T>(max_));
        }
        slot = value;
        return true;
    }

    void ClampLast();
    void RequireInitialized() const;

    template <typename T> void RequireType() const {
        RequireInitialized();
        if (type_ != typeid(T)) {
            std::ostringstream err;
            err << "Incorrect option type. Expect " << option_detail::TypeName(typeid(T))
                << " but option is " << option_detail::TypeName(type_) << ".";
            throw std::runtime_error(err.str());
        }
    }
};

struct OptionRegistration {
    std::vector<std::string> flags;
    std::string key;
    Option option;

    template <typename T> OptionRegistration&& Range(T min, T max) && {
        option.SetRange(std::move(min), std::move(max));
        return std::move(*this);
    }

    template <typename T>
    OptionRegistration&& Choices(std::initializer_list<std::string> names,
                                 std::initializer_list<T> values) && {
        option.SetChoices<T>(names, values);
        return std::move(*this);
    }

    OptionRegistration&& Helper(std::string helper) && {
        option.SetHelper(std::move(helper));
        return std::move(*this);
    }

    OptionRegistration&& Group(std::string group) && {
        option.SetGroup(std::move(group));
        return std::move(*this);
    }

    OptionRegistration&& Setter(Option::SetterFn fn) && {
        option.SetSetter(std::move(fn));
        return std::move(*this);
    }

    OptionRegistration&& NoValue() && {
        option.SetNoValue();
        return std::move(*this);
    }
};

template <typename T>
inline OptionRegistration
RegisterOption(std::initializer_list<std::string> flags, std::string key, T default_val) {
    return OptionRegistration{
        std::vector<std::string>(flags), std::move(key), Option::Make<T>(std::move(default_val))};
}

inline OptionRegistration
RegisterOption(std::initializer_list<std::string> flags, std::string key, const char* default_val) {
    return RegisterOption<std::string>(flags, std::move(key), std::string(default_val));
}

class OptionsMap {
public:
    explicit OptionsMap(std::string initial_profile = "default");

    OptionsMap& operator<<(OptionRegistration reg);
    const std::string& GetProfile() const {
        return current_profile_;
    }
    const std::string& GetDefaultProfile() const;
    std::vector<std::string> GetProfiles() const;
    void SetProfile(std::string name);
    void AddProfile(std::string name);
    void RemoveProfile(const std::string& name);

    Option& operator[](const std::string& key) {
        return Current()[key];
    }
    const Option& at(const std::string& key) const {
        return Current().at(key);
    }
    Option& at(const std::string& key) {
        return Current().at(key);
    }
    auto find(const std::string& key) {
        return Current().find(key);
    }
    auto find(const std::string& key) const {
        return Current().find(key);
    }
    auto begin() {
        return Current().begin();
    }
    auto begin() const {
        return Current().begin();
    }
    auto end() {
        return Current().end();
    }
    auto end() const {
        return Current().end();
    }
    void clear();

    void ParseArgs(int argc, char** argv);
    std::string HelpersToString(std::string_view group = "") const;

private:
    using ProfileMap = std::unordered_map<std::string, Option>;
    struct ParsedArg {
        std::string base_flag;
        std::string profile_name;
        std::string value;
    };

    std::vector<ParsedArg> TokenizeArgs(int argc, char** argv) const;
    std::vector<std::string> FlagsOf(const std::string& key) const;

    ProfileMap& Current() {
        return profiles_.at(current_profile_);
    }
    const ProfileMap& Current() const {
        return profiles_.at(current_profile_);
    }

    std::unordered_map<std::string, ProfileMap> profiles_;
    std::string base_profile_;
    std::string current_profile_;
    std::vector<std::string> seen_profiles_;
    std::vector<std::string> insertion_order_;
    std::unordered_map<std::string, std::string> flags_;
};

extern OptionsMap kOptionsMap;

template <typename T> inline T GetOption(const std::string& key, int idx = -1) {
    auto it = kOptionsMap.find(key);
    if (it == std::end(kOptionsMap)) {
        throw std::runtime_error("Unknown option: " + key);
    }
    return it->second.Get<T>(idx);
}

template <typename T>
inline bool SetOption(const std::string& key, T value, bool as_default = false) {
    auto it = kOptionsMap.find(key);
    if (it == std::end(kOptionsMap)) {
        return false;
    }
    it->second.Set<T>(std::move(value));
    if (as_default) {
        it->second.SetCurrentAsDefault();
    }
    return true;
}

inline bool SetOption(const std::string& key, const char* value, bool as_default = false) {
    return SetOption<std::string>(key, std::string(value), as_default);
}

inline int GetOptionCount(const std::string& key) {
    return kOptionsMap.at(key).Count();
}

inline bool IsOptionDefault(const std::string& key) {
    return kOptionsMap.at(key).IsDefault();
}

inline void UniqueOption(const std::string& key) {
    kOptionsMap.at(key).Unique();
}

inline std::string OptionHelpersToString(std::string_view group = "") {
    return kOptionsMap.HelpersToString(group);
}

inline void ParseArgs(int argc, char** argv) {
    kOptionsMap.ParseArgs(argc, argv);
}
