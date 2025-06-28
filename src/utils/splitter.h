#pragma once

#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <sstream>
#include <stdexcept>

class Splitter {
public:
    class Reuslt {
    public:
        Reuslt() = delete;
        Reuslt(const std::string &s, const int i) : str_(s), idx_(i) {};

        Reuslt(const std::string &&s, const int i) :
            str_(std::forward<decltype(s)>(s)), idx_(i) {};

        std::string Upper() const; // Return the upper case string.
        std::string Lower() const; // Return the lower case string.
        bool IsDigit() const;

        template<typename T=std::string> T Get() const;
        template<typename T=std::string> T Get(T default_val, bool &error) const {
            error = false;
            try {
                return Get<T>();
            } catch (const std::exception& e) {
                error = true;
            }
            return default_val;
        }
        template<typename T=std::string> T Get(T default_val) const {
            bool error;
            return Get<T>(default_val, error);
        }

        int Index() const;

    private:
        std::string str_;
        int idx_;

        friend class Splitter;
    };

    static constexpr size_t kMaxBufferSize = 1024 * 1024 * 1024;

    Splitter() = delete;
    Splitter(std::string &input);
    Splitter(std::string &input, const size_t max);
    Splitter(int argc, char** argv);

    bool Valid() const;
    size_t GetCount() const;

    std::shared_ptr<Reuslt> GetWord(size_t id) const;
    std::shared_ptr<Reuslt> GetSlice(size_t begin = 0) const;
    std::shared_ptr<Reuslt> GetSlice(size_t begin, size_t end) const;
    std::shared_ptr<Reuslt> Find(const std::string input, int id = -1) const;
    std::shared_ptr<Reuslt> Find(const std::initializer_list<std::string> inputs, int id = -1) const;
    std::shared_ptr<Reuslt> FindNext(const std::string input) const;
    std::shared_ptr<Reuslt> FindNext(const std::initializer_list<std::string> inputs) const;
    std::shared_ptr<Reuslt> FindDigit(int id = -1) const;
    std::shared_ptr<Reuslt> RemoveWord(size_t id);
    std::shared_ptr<Reuslt> RemoveSlice(size_t begin, size_t end);

private:
    std::vector<std::shared_ptr<const std::string>> buffer_;
    size_t count_;

    void Parse(std::string &input, const size_t max);
    void Parse(std::string &&input, const size_t max);
    void Parse(int argc, char** argv, const size_t max);
};
