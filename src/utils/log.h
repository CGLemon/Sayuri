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
};

class LogOptions {
public:
    static LogOptions& Get();

    void SetQuiet(bool q);

private:
    bool quiet_;

    friend class Logging;
};

class Logging : public std::ostringstream {
public:
    Logging(const char* file, int line, bool err, bool write_only, bool use_options);
    ~Logging();

private:
    bool write_only_;
    bool use_options_;
    bool err_;
    std::string file_;
    int line_;
};

#define LOGGING (::Logging(__FILE__, __LINE__, true,  false, true))
#define WRITING (::Logging(__FILE__, __LINE__, false, true,  true))
#define DUMPING (::Logging(__FILE__, __LINE__, false, false, false))
