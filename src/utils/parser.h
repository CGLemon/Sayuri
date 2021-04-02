#ifndef UTILS_PARSER_H_INCLUDE
#define UTILS_PARSER_H_INCLUDE

#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <sstream>

/**
 * Split the string to words.
 */
class CommandParser {
public:
    class Reuslt {
    public:
        Reuslt(const std::string &s, const int i) : str_(s), idx_(i) {};

        Reuslt(const std::string &&s, const int i) :
            str_(std::forward<decltype(s)>(s)), idx_(i) {};

        template<typename T> T Get() const;

    private:
        std::string str_;
        int idx_;

        friend class CommandParser;
    };

    static constexpr size_t kMaxBufferSize = 1024 * 1024 * 1024;

    CommandParser() = delete;
    CommandParser(std::string &input);
    CommandParser(std::string &input, const size_t max);
    CommandParser(int argc, char** argv);

    bool Valid() const;
    size_t GetCount() const;

    std::shared_ptr<Reuslt> GetCommand(size_t id) const;
    std::shared_ptr<Reuslt> GetCommands(size_t begin = 0) const;
    std::shared_ptr<Reuslt> GetSlice(size_t begin, size_t end) const;
    std::shared_ptr<Reuslt> Find(const std::string input, int id = -1) const;
    std::shared_ptr<Reuslt> Find(const std::initializer_list<std::string> inputs, int id = -1) const;
    std::shared_ptr<Reuslt> FindNext(const std::string input) const;
    std::shared_ptr<Reuslt> FindNext(const std::initializer_list<std::string> inputs) const;
    std::shared_ptr<Reuslt> RemoveCommand(size_t id);
    std::shared_ptr<Reuslt> RemoveSlice(size_t begin, size_t end);

private:
    std::vector<std::shared_ptr<const std::string>> commands_buffer_;
    size_t count_;

    void Parser(std::string &input, const size_t max);
    void Parser(std::string &&input, const size_t max);
};


#endif
