#include "utils/option.h"

#include <algorithm>
#include <any>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace option_detail {

std::string TypeName(std::type_index t) {
    if (t == typeid(int)) {
        return "kInteger";
    }
    if (t == typeid(float)) {
        return "kFloating";
    }
    if (t == typeid(double)) {
        return "kDouble";
    }
    if (IsBoolean(t)) {
        return "kBoolean";
    }
    if (IsString(t)) {
        return "kString";
    }
    return "kInvalid";
}

std::string AnyToString(const std::any& a) {
    std::ostringstream out;
    const auto t = std::type_index(a.type());
    if (t == typeid(int)) {
        return std::to_string(std::any_cast<int>(a));
    }
    if (t == typeid(float)) {
        out.precision(std::numeric_limits<float>::max_digits10);
        out << std::any_cast<float>(a);
        return out.str();
    }
    if (t == typeid(double)) {
        out.precision(std::numeric_limits<double>::max_digits10);
        out << std::any_cast<double>(a);
        return out.str();
    }
    if (IsBoolean(t)) {
        return std::any_cast<bool>(a) ? "true" : "false";
    }
    if (IsString(t)) {
        return std::any_cast<std::string>(a);
    }
    return "<invalid>";
}

std::any ParseFromString(std::type_index t, std::string_view raw) {
    const std::string s(raw);
    auto ParseNumeric = [&]() -> std::any {
        std::size_t pos = 0;
        std::any out;
        if (t == typeid(int)) {
            out = std::any(std::stoi(s, &pos));
        } else if (t == typeid(float)) {
            out = std::any(std::stof(s, &pos));
        } else if (t == typeid(double)) {
            out = std::any(std::stod(s, &pos));
        }
        if (pos != s.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return out;
    };

    try {
        if (IsNumeric(t)) {
            return ParseNumeric();
        }
        if (IsBoolean(t)) {
            if (s == "true") {
                return std::any(true);
            }
            if (s == "false") {
                return std::any(false);
            }
            throw std::runtime_error("Cannot parse bool from \"" + s + "\".");
        }
        if (IsString(t)) {
            return std::any(s);
        }
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("Cannot parse " + TypeName(t) + " from \"" + s + "\".");
    } catch (const std::out_of_range&) {
        throw std::runtime_error("Value out of range for " + TypeName(t) + ": \"" + s + "\".");
    }
    throw std::runtime_error("Unsupported option type during parse.");
}

} // namespace option_detail

std::string Option::HelpTypePlaceholder(std::type_index t, bool is_enum) {
    if (t == typeid(int)) {
        return "<int>";
    }
    if (t == typeid(float)) {
        return "<float>";
    }
    if (t == typeid(double)) {
        return "<double>";
    }
    if (option_detail::IsBoolean(t)) {
        return {};
    }
    if (option_detail::IsString(t)) {
        return "<string>";
    }
    if (is_enum) {
        return "<enum>";
    }
    return {};
}

void Option::SetFromString(std::string_view raw) {
    RequireInitialized();
    if (setter_) {
        setter_(*this, raw);
        return;
    }
    if (has_choices_) {
        for (size_t i = 0; i < choice_names_.size(); ++i) {
            if (choice_names_[i] == raw) {
                Push(choice_values_[i]);
                return;
            }
        }
        std::ostringstream err;
        err << "Invalid value \"" << raw << "\". Expected one of:";
        for (const auto& n : choice_names_) {
            err << " " << n;
        }
        throw std::runtime_error(err.str());
    }
    Push(option_detail::ParseFromString(type_, raw));
}

void Option::Unique() {
    std::vector<std::any> dest;
    std::vector<std::string> seen;
    for (auto it = std::rbegin(history_); it != std::rend(history_); ++it) {
        std::string s;
        if (has_choices_) {
            s = choice_to_string_(*it);
        } else {
            s = option_detail::AnyToString(*it);
        }
        if (std::find(std::begin(seen), std::end(seen), s) == std::end(seen)) {
            seen.emplace_back(std::move(s));
            dest.emplace_back(*it);
        }
    }
    std::reverse(std::begin(dest), std::end(dest));
    history_ = std::move(dest);
}

std::string Option::ToString() const {
    std::ostringstream out;
    if (history_.empty()) {
        out << "<empty>";
    } else if (has_choices_) {
        std::string name = ChoiceNameOfCurrent();
        if (name.empty()) {
            name = "<unmatched>";
        }
        out << name;
    } else {
        out << option_detail::AnyToString(history_.back());
    }
    if (has_range_) {
        out << ", Max: " << option_detail::AnyToString(max_);
        out << ", Min: " << option_detail::AnyToString(min_);
    }
    if (has_choices_) {
        out << ", Choices: {";
        for (size_t i = 0; i < choice_names_.size(); ++i) {
            if (i != 0) {
                out << ", ";
            }
            out << choice_names_[i];
        }
        out << "}";
    } else {
        out << ", " << option_detail::TypeName(type_);
    }
    return out.str();
}

std::string Option::HelpMetadata() const {
    std::ostringstream out;
    bool has_content = false;
    const auto Sep = [&]() {
        if (has_content) {
            out << " ";
        }
        has_content = true;
    };

    std::string placeholder = HelpTypePlaceholder(type_, is_enum_);
    if (has_choices_) {
        placeholder.clear();
    }
    if (!placeholder.empty()) {
        Sep();
        out << placeholder;
    }
    if (has_range_) {
        Sep();
        out << "[" << option_detail::AnyToString(min_) << ", " << option_detail::AnyToString(max_)
            << "]";
    }
    if (has_choices_) {
        Sep();
        out << "{";
        for (size_t i = 0; i < choice_names_.size(); ++i) {
            if (i != 0) {
                out << ", ";
            }
            out << choice_names_[i];
        }
        out << "}";
    }

    std::any d = default_;
    if (!d.has_value() && is_default_ && !history_.empty()) {
        d = history_.back();
    }
    if (d.has_value()) {
        std::string label;
        if (has_choices_) {
            for (size_t i = 0; i < choice_values_.size(); ++i) {
                if (choice_equals_(d, choice_values_[i])) {
                    label = choice_names_[i];
                    break;
                }
            }
        } else {
            label = option_detail::AnyToString(d);
        }
        if (!label.empty()) {
            Sep();
            out << "(default: " << label << ")";
        }
    }
    return out.str();
}

std::string Option::ChoiceNameOfCurrent() const {
    if (!has_choices_ || history_.empty()) {
        return {};
    }
    for (size_t i = 0; i < choice_values_.size(); ++i) {
        if (choice_equals_(history_.back(), choice_values_[i])) {
            return choice_names_[i];
        }
    }
    return {};
}

void Option::EnsureValueHasChoice(const std::any& v) const {
    if (!has_choices_) {
        return;
    }
    for (const auto& cv : choice_values_) {
        if (choice_equals_(v, cv)) {
            return;
        }
    }
    std::ostringstream err;
    err << "Option Error: value is not one of the registered choices:";
    for (const auto& n : choice_names_) {
        err << " " << n;
    }
    throw std::runtime_error(err.str());
}

void Option::Push(std::any v) {
    EnsureValueHasChoice(v);
    if (is_default_) {
        if (!history_.empty()) {
            default_ = history_.back();
        }
        history_.clear();
        is_default_ = false;
    }
    history_.emplace_back(std::move(v));
    ClampLast();
}

void Option::ClampLast() {
    if (history_.empty() || !has_range_) {
        return;
    }
    (void)(TryClampLastAs<int>() || TryClampLastAs<float>() || TryClampLastAs<double>());
}

void Option::RequireInitialized() const {
    if (type_ == typeid(void)) {
        throw std::runtime_error("Option Error: not initialized.");
    }
}

OptionsMap kOptionsMap;

OptionsMap::OptionsMap(std::string initial_profile)
    : base_profile_(std::move(initial_profile)), current_profile_(base_profile_) {
    profiles_[base_profile_] = {};
}

void OptionsMap::clear() {
    profiles_.clear();
    flags_.clear();
    seen_profiles_.clear();
    insertion_order_.clear();
    profiles_[base_profile_] = {};
    profiles_[current_profile_] = {};
}

OptionsMap& OptionsMap::operator<<(OptionRegistration reg) {
    if (Current().find(reg.key) != std::end(Current())) {
        throw std::runtime_error("Option Error: duplicate option key \"" + reg.key + "\".");
    }
    for (const auto& f : reg.flags) {
        if (f.empty() || f.front() != '-') {
            throw std::runtime_error("Option Error: flag \"" + f + "\" must start with '-'.");
        }
        if (auto it = flags_.find(f); it != std::end(flags_)) {
            throw std::runtime_error("Option Error: flag \"" + f + "\" already bound to key \"" +
                                     it->second + "\".");
        }
    }
    for (auto& [profile_name, options] : profiles_) {
        (void)profile_name;
        options[reg.key] = reg.option;
    }
    for (const auto& f : reg.flags) {
        flags_[f] = reg.key;
    }
    insertion_order_.emplace_back(reg.key);
    return *this;
}

const std::string& OptionsMap::GetDefaultProfile() const {
    return seen_profiles_.empty() ? base_profile_ : seen_profiles_.front();
}

std::vector<std::string> OptionsMap::GetProfiles() const {
    return seen_profiles_.empty() ? std::vector<std::string>{base_profile_} : seen_profiles_;
}

void OptionsMap::SetProfile(std::string name) {
    if (name == current_profile_) {
        return;
    }
    if (profiles_.find(name) == std::end(profiles_)) {
        throw std::runtime_error("Option Error: profile \"" + name + "\" does not exist.");
    }
    current_profile_ = std::move(name);
}

void OptionsMap::AddProfile(std::string name) {
    if (profiles_.find(name) != std::end(profiles_)) {
        throw std::runtime_error("Option Error: profile \"" + name + "\" already exists.");
    }
    seen_profiles_.emplace_back(name);
    profiles_[std::move(name)] = profiles_.at(base_profile_);
}

void OptionsMap::RemoveProfile(const std::string& name) {
    if (profiles_.find(name) == std::end(profiles_)) {
        throw std::runtime_error("Option Error: profile \"" + name + "\" does not exist.");
    }
    profiles_.erase(name);
    seen_profiles_.erase(std::remove(std::begin(seen_profiles_), std::end(seen_profiles_), name),
                         std::end(seen_profiles_));
}

std::vector<OptionsMap::ParsedArg> OptionsMap::TokenizeArgs(int argc, char** argv) const {
    std::vector<ParsedArg> records;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        std::string flag_part = arg;
        std::string value;
        bool has_eq_value = false;
        if (auto eq = arg.find('='); eq != std::string::npos) {
            flag_part = arg.substr(0, eq);
            value = arg.substr(eq + 1);
            has_eq_value = true;
        }

        std::string base_flag = flag_part;
        std::string profile_name;
        if (auto at_pos = flag_part.find('@'); at_pos != std::string::npos) {
            base_flag = flag_part.substr(0, at_pos);
            profile_name = flag_part.substr(at_pos + 1);
            if (profile_name.empty()) {
                throw std::runtime_error("Empty profile name after '@' in flag: " + flag_part);
            }
        }

        if (base_flag.empty()) {
            if (has_eq_value) {
                throw std::runtime_error("Profile-only token cannot carry a value: " + arg);
            }
            records.emplace_back(ParsedArg{"", std::move(profile_name), ""});
            continue;
        }

        auto it = flags_.find(base_flag);
        if (it == std::end(flags_)) {
            throw std::runtime_error("Unknown command-line flag: " + base_flag);
        }
        const std::string& key = it->second;
        const Option& option = profiles_.at(base_profile_).at(key);
        const bool can_omit_value = option.IsBoolean() || option.AllowsNoValue();

        if (!has_eq_value) {
            const bool next_looks_like_flag =
                (i + 1 >= argc) || (argv[i + 1][0] == '-' && argv[i + 1][1] != '\0');

            if (can_omit_value && next_looks_like_flag) {
                value = "true";
            } else if (!next_looks_like_flag) {
                value = argv[++i];
            } else {
                throw std::runtime_error("Missing value for command-line flag: " + flag_part);
            }
        }

        records.emplace_back(
            ParsedArg{std::move(base_flag), std::move(profile_name), std::move(value)});
    }
    return records;
}

void OptionsMap::ParseArgs(int argc, char** argv) {
    seen_profiles_.clear();
    if (profiles_.find(base_profile_) == std::end(profiles_)) {
        profiles_[base_profile_] = profiles_.at(current_profile_);
    }

    const std::vector<ParsedArg> records = TokenizeArgs(argc, argv);

    for (const auto& r : records) {
        if (r.profile_name.empty()) {
            continue;
        }
        if (std::find(std::begin(seen_profiles_), std::end(seen_profiles_), r.profile_name) ==
            std::end(seen_profiles_)) {
            seen_profiles_.emplace_back(r.profile_name);
        }
    }
    if (seen_profiles_.empty()) {
        for (const auto& r : records) {
            if (r.profile_name.empty() && !r.base_flag.empty()) {
                seen_profiles_.emplace_back(base_profile_);
                break;
            }
        }
    }

    for (const auto& name : seen_profiles_) {
        if (profiles_.find(name) == std::end(profiles_)) {
            profiles_[name] = profiles_.at(base_profile_);
        }
    }

    for (const auto& r : records) {
        if (r.base_flag.empty()) {
            continue;
        }
        const std::string& key = flags_.at(r.base_flag);
        if (r.profile_name.empty()) {
            for (auto& [profile_name, options] : profiles_) {
                (void)profile_name;
                options.at(key).SetFromString(r.value);
            }
        } else {
            profiles_.at(r.profile_name).at(key).SetFromString(r.value);
        }
    }

    if (!seen_profiles_.empty()) {
        current_profile_ = seen_profiles_.front();
    }
}

std::vector<std::string> OptionsMap::FlagsOf(const std::string& key) const {
    std::vector<std::string> out;
    for (const auto& [flag, k] : flags_) {
        if (k == key) {
            out.emplace_back(flag);
        }
    }
    std::sort(std::begin(out), std::end(out), [](const std::string& a, const std::string& b) {
        if (a.size() != b.size()) {
            return a.size() < b.size();
        }
        return a < b;
    });
    return out;
}

std::string OptionsMap::HelpersToString(std::string_view group) const {
    const auto JoinFlags = [](const std::vector<std::string>& flags) -> std::string {
        std::ostringstream out;
        for (size_t i = 0; i < flags.size(); ++i) {
            if (i != 0) {
                out << ", ";
            }
            out << flags[i];
        }
        return out.str();
    };

    struct Entry {
        std::string header;
        std::string flags;
        std::string metadata;
        std::string helper;
    };

    std::vector<Entry> entries;
    std::vector<std::string> group_order;

    const auto& current = Current();
    for (const auto& key : insertion_order_) {
        auto it = current.find(key);
        if (it == std::end(current)) {
            continue;
        }
        const Option& opt = it->second;
        const std::string& g = opt.Group();
        if (!group.empty() && g != group) {
            continue;
        }
        auto flags = FlagsOf(key);
        if (flags.empty()) {
            continue;
        }
        std::string header = g.empty() ? "default" : g;
        if (std::find(std::begin(group_order), std::end(group_order), header) ==
            std::end(group_order)) {
            group_order.emplace_back(header);
        }
        entries.emplace_back(
            Entry{std::move(header), JoinFlags(flags), opt.HelpMetadata(), opt.Helper()});
    }

    size_t flags_w = 0;
    size_t metadata_w = 0;
    for (const auto& e : entries) {
        flags_w = std::max(flags_w, e.flags.size());
        metadata_w = std::max(metadata_w, e.metadata.size());
    }

    std::ostringstream out;
    bool first = true;
    for (const auto& header : group_order) {
        if (!first) {
            out << "\n";
        }
        first = false;
        out << "[" << header << "]\n";
        for (const auto& e : entries) {
            if (e.header != header) {
                continue;
            }
            out << "  " << std::left << std::setw(static_cast<int>(flags_w)) << e.flags << " "
                << std::setw(static_cast<int>(metadata_w)) << e.metadata << " : " << e.helper
                << "\n";
        }
    }
    return out.str();
}
