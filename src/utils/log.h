#ifndef UTILS_LOG_H_INCLUDE
#define UTILS_LOG_H_INCLUDE

#include <sstream>
#include <fstream>
#include <string>
#include <deque>

#include "mutex.h"
#include "utils/mutex.h"

class LogWriter {
public:
    static LogWriter& Get();

    void SetFilename(std::string filename);

private:
    void WriteString(std::string data);

    Mutex mutex_;

    std::string filename_;
    std::ofstream file_;
    std::deque<std::string> buffer_;

    friend class Logging;
    friend class StandError;
};


class Logging : public std::ostringstream{
public:
    Logging(const char* file, int line, bool write_only);
    ~Logging();

private:
    bool write_only_;
    std::string file_;
    int line_;
};

class StandError : public std::ostringstream{
public:
    StandError(const char* file, int line);
    ~StandError();

private:
    std::string file_;
    int line_;
};

#define LOGGING (Utils::Logging(__FILE__, __LINE__, false))
#define WRITING (Utils::Logging(__FILE__, __LINE__, true))
#define ERROR (Utils::StandError(__FILE__, __LINE__))

#endif
