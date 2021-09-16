#pragma once

#include <sstream>
#include <fstream>
#include <string>
#include <deque>
#include <mutex>

class LogWriter {
public:
    static LogWriter& Get();

    void SetFilename(std::string filename);

private:
    void WriteString(std::string data);

    std::mutex mutex_;

    std::string filename_{};
    std::ofstream file_;

    friend class Logging;
    friend class StandError;
};

class LogOptions {
public:
    static LogOptions& Get();

    void SetQuiet(bool q);

private:
    bool quiet_;

    friend class Logging;
    friend class StandError;
};

class Logging : public std::ostringstream {
public:
    Logging(const char* file, int line, bool write_only);
    ~Logging();

private:
    bool write_only_;
    std::string file_;
    int line_;
};

class StandError : public std::ostringstream {
public:
    StandError(const char* file, int line);
    ~StandError();

private:
    std::string file_;
    int line_;
};

#define LOGGING (::Logging(__FILE__, __LINE__, false))
#define WRITING (::Logging(__FILE__, __LINE__, true))
#define ERROR (::StandError(__FILE__, __LINE__))
