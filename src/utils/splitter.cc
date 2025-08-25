#include "utils/splitter.h"

constexpr size_t Splitter::kMaxBufferSize;

Splitter::Splitter(std::string &input) {
    Parse(std::forward<std::string>(input), kMaxBufferSize);
}

Splitter::Splitter(std::string &input, const size_t max) {
    Parse(std::forward<std::string>(input), std::min(max, kMaxBufferSize));
}

Splitter::Splitter(int argc, char** argv) {
    Parse(argc, argv, kMaxBufferSize);
}

bool Splitter::Valid() const {
    return count_ != 0;
}

void Splitter::Parse(std::string &input, const size_t max) {
    count_ = 0;
    auto stream = std::istringstream{input};
    auto in = std::string{};
    while (stream >> in) {
        buffer_.emplace_back(std::make_shared<std::string>(in));
        count_++;
        if (count_ >= max) break;
    }
}

void Splitter::Parse(std::string &&input, const size_t max) {
    count_ = 0;
    auto stream = std::istringstream{input};
    auto in = std::string{};
    while (stream >> in) {
        buffer_.emplace_back(std::make_shared<std::string>(in));
        count_++;
        if (count_ >= max) break;
    }
}

void Splitter::Parse(int argc, char** argv, const size_t max) {
    count_ = std::min((size_t)argc, max);
    for (auto i = size_t{0}; i < count_; ++i) {
        buffer_.emplace_back(std::make_shared<std::string>(argv[i]));
    }
}

size_t Splitter::GetCount() const {
    return count_;
}

std::shared_ptr<Splitter::Reuslt> Splitter::GetWord(size_t id) const {
    if (!Valid() || id >= count_) {
        return nullptr;
    }
    return std::make_shared<Reuslt>(*buffer_[id], (int)id);
}

std::shared_ptr<Splitter::Reuslt> Splitter::GetSlice(size_t b) const {
    return GetSlice(b, count_);
}

std::shared_ptr<Splitter::Reuslt> Splitter::GetSlice(size_t b, size_t e) const {
     if (!Valid() ||
             b >= count_ ||
             e > count_ ||
             b >= e) {
         return nullptr;
     }

     auto out = std::ostringstream{};
     auto begin = std::next(std::begin(buffer_), b);
     auto end = std::next(std::begin(buffer_), e);
     auto stop = std::prev(end, 1);

     if (begin != end) {
         std::for_each(begin, stop,
             [&](auto in) { out << *in << " "; });
     }

     out << **stop;
     return std::make_shared<Reuslt>(out.str(), -1);
}

std::shared_ptr<Splitter::Reuslt> Splitter::Find(const std::string input, int id) const {
    if (!Valid()) {
        return nullptr;
    }

    if (id < 0) {
        // Only return first match result.
        for (auto i = size_t{0}; i < count_; ++i) {
            const auto res = GetWord(i);
            if (res->str_ == input) {
                return res;
            }
        }
    } else {
        // Only check the specific string.
        if (const auto res = GetWord(id)) {
            return res->str_ == input ? res : nullptr;
        }
    }
    // Don't match any string.
    return nullptr;
}

std::shared_ptr<Splitter::Reuslt> Splitter::Find(const std::initializer_list<std::string> inputs, int id) const {
    for (const auto &in : inputs) {
        // Only return first match result.
        if (const auto res = Find(in, id)) {
            return res;
        }
    }
    return nullptr;
}

std::shared_ptr<Splitter::Reuslt> Splitter::FindNext(const std::string input) const {
    const auto res = Find(input);

    if (!res || res->idx_+1 > (int)count_) {
        return nullptr;
    }
    return GetWord(res->idx_+1);
}

std::shared_ptr<Splitter::Reuslt> Splitter::FindNext(const std::initializer_list<std::string> inputs) const {
    for (const auto &in : inputs) {
        if (const auto res = FindNext(in)) {
            return res;
        }
    }
    return nullptr;
}

std::shared_ptr<Splitter::Reuslt> Splitter::FindDigit(int id) const {
    if (!Valid()) {
        return nullptr;
    }

    if (id < 0) {
        // Only return first match result.
        for (auto i = size_t{0}; i < count_; ++i) {
            const auto res = GetWord(i);
            if (res->IsDigit()) {
                return res;
            }
        }
    } else {
        // Only check the specific string.
        if (const auto res = GetWord(id)) {
            return res->IsDigit() ? res : nullptr;
        }
    }
    // Don't match any string.
    return nullptr;
}

std::shared_ptr<Splitter::Reuslt> Splitter::RemoveWord(size_t id) {
    if (id > count_) {
        return nullptr;
    }

    const auto str_ = *buffer_[id];
    buffer_.erase(std::begin(buffer_)+id);
    count_--;

    return std::make_shared<Reuslt>(str_, -1);
}

std::shared_ptr<Splitter::Reuslt> Splitter::RemoveSlice(size_t b, size_t e) {
    if (b > count_ || e > count_ || b > e) {
        return nullptr;
    }
    if (b == e) {
        return RemoveWord(e);
    }
    auto out = GetSlice(b, e);
    buffer_.erase(std::begin(buffer_)+b, std::begin(buffer_)+e);
    count_ -= (e-b);
    return out;
}

std::string Splitter::Reuslt::Upper() const {
    auto upper = str_;
    for (auto & c: upper) {
        c = std::toupper(c);
    }
    return upper;
}

std::string Splitter::Reuslt::Lower() const {
    auto lower = str_;
    for (auto & c: lower) {
        c = std::tolower(c);
    }
    return lower;
}

bool Splitter::Reuslt::IsDigit() const {
    bool is_digit = true;

    for (char c : str_) {
        is_digit &= (bool)std::isdigit(c);
    }
    return is_digit;
}

template<>
std::string Splitter::Reuslt::Get<std::string>() const {
    return str_;
}

template<>
int Splitter::Reuslt::Get<int>() const {
    return std::stoi(str_);
}

template<>
float Splitter::Reuslt::Get<float>() const {
    return std::stof(str_);
}

template<>
double Splitter::Reuslt::Get<double>() const {
    return std::stod(str_);
}

template<>
char Splitter::Reuslt::Get<char>() const {
    return str_[0];
}

template<>
const char* Splitter::Reuslt::Get<const char*>() const {
    return str_.c_str();
}

int Splitter::Reuslt::Index() const {
    return idx_;
}
