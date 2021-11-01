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
    Logging(const char* file, int line, bool write_only, bool use_options);
    ~Logging();

private:
    bool write_only_;
    bool use_options_;
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

#define LOGGING (::Logging(__FILE__, __LINE__, false, true))
#define WRITING (::Logging(__FILE__, __LINE__, true, true))
#define DUMPING (::Logging(__FILE__, __LINE__, false, false))
#define ERROR (::StandError(__FILE__, __LINE__))
