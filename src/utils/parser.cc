#include "utils/parser.h"

constexpr size_t CommandParser::kMaxBufferSize;

CommandParser::CommandParser(std::string &input) {
    Parser(std::forward<std::string>(input), kMaxBufferSize);
}

CommandParser::CommandParser(std::string &input, const size_t max) {
    Parser(std::forward<std::string>(input), std::min(max, kMaxBufferSize));
}

CommandParser::CommandParser(int argc, char** argv) {
    auto out = std::ostringstream{};
    for (int i = 0; i < argc; ++i) {
        out << argv[i] << " ";
    }
    Parser(std::forward<std::string>(out.str()), kMaxBufferSize);
}

bool CommandParser::Valid() const {
    return count_ != 0;
}

void CommandParser::Parser(std::string &input, const size_t max) {
    count_ = 0;
    auto stream = std::istringstream{input};
    auto in = std::string{};
    while (stream >> in) {
        commands_buffer_.emplace_back(std::make_shared<std::string>(in));
        count_++;
        if (count_ >= max) break;
    }
}

void CommandParser::Parser(std::string &&input, const size_t max) {
    count_ = 0;
    auto stream = std::istringstream{input};
    auto in = std::string{};
    while (stream >> in) {
        commands_buffer_.emplace_back(std::make_shared<std::string>(in));
        count_++;
        if (count_ >= max) break;
    }
}

size_t CommandParser::GetCount() const {
    return count_;
}

std::shared_ptr<CommandParser::Reuslt> CommandParser::GetCommand(size_t id) const {
    if (!Valid() || id > count_) {
        return nullptr;
    }
    return std::make_shared<Reuslt>(Reuslt(*commands_buffer_[id], (int)id));
}

std::shared_ptr<CommandParser::Reuslt> CommandParser::GetCommands(size_t b) const {
    return GetSlice(b, count_);
}

std::shared_ptr<CommandParser::Reuslt> CommandParser::GetSlice(size_t b, size_t e) const {
     if (!Valid() || b >= count_ || e > count_ || b >= e) {
         return nullptr;
     }

     auto out = std::ostringstream{};
     auto begin = std::next(std::begin(commands_buffer_), b);
     auto end = std::next(std::begin(commands_buffer_), e);
     auto stop = std::prev(end, 1);

     if (begin != end) {
         std::for_each(begin, stop, [&](auto in)
                                        {  out << *in << " "; });
     }

     out << **stop;
     return std::make_shared<Reuslt>(Reuslt(out.str(), -1));
}

std::shared_ptr<CommandParser::Reuslt> CommandParser::Find(const std::string input, int id) const {
    if (!Valid()) {
        return nullptr;
    }

    if (id < 0) {
        for (auto i = size_t{0}; i < GetCount(); ++i) {
            const auto res = GetCommand((size_t)i);
            if (res->str_ == input) {
                return res;
            }
        }
    } else {
        if (const auto res = GetCommand((size_t)id)) {
            return res->str_ == input ? res : nullptr;
        }
    }
    return nullptr;
}

std::shared_ptr<CommandParser::Reuslt> CommandParser::Find(const std::initializer_list<std::string> inputs, int id) const {
    for (const auto &in : inputs) {
        if (const auto res = Find(in, id)) {
            return res;
        }
    }
    return nullptr;
}

std::shared_ptr<CommandParser::Reuslt> CommandParser::FindNext(const std::string input) const {
    const auto res = Find(input);

    if (!res || res->idx_+1 > (int)GetCount()) {
        return nullptr;
    }
    return GetCommand(res->idx_+1);
}

std::shared_ptr<CommandParser::Reuslt> CommandParser::FindNext(const std::initializer_list<std::string> inputs) const {
    for (const auto &in : inputs) {
        if (const auto res = FindNext(in)) {
            return res;
        }
    }
    return nullptr;
}

std::shared_ptr<CommandParser::Reuslt> CommandParser::RemoveCommand(size_t id) {
    if (id > GetCount()) {
        return nullptr;
    }

    const auto str_ = *commands_buffer_[id];
    commands_buffer_.erase(std::begin(commands_buffer_)+id);
    count_--;

    return std::make_shared<Reuslt>(Reuslt(str_, -1));
}

std::shared_ptr<CommandParser::Reuslt> CommandParser::RemoveSlice(size_t b, size_t e) {
    if (b > GetCount() || e > GetCount() || b > e) {
        return nullptr;
    }
    if (b == e) {
        return RemoveCommand(e);
    }
    auto out = GetSlice(b, e);
    commands_buffer_.erase(std::begin(commands_buffer_)+b, std::begin(commands_buffer_)+e);
    count_ -= (e-b);
    return out;
}

template<>
std::string CommandParser::Reuslt::Get<std::string>() const {
    return str_;
}

template<>
int CommandParser::Reuslt::Get<int>() const {
    return std::stoi(str_);
}

template<>
float CommandParser::Reuslt::Get<float>() const{
    return std::stof(str_);
}

template<>
char CommandParser::Reuslt::Get<char>() const{
    return str_[0];
}

template<>
const char* CommandParser::Reuslt::Get<const char*>() const{
    return str_.c_str();
}
