#include <iostream>
#include <sstream>

#include "utils/log.h"

LogWriter& LogWriter::Get() {
    static LogWriter writer;
    return writer;
}

void LogWriter::SetFilename(std::string filename) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (filename_ == filename) {
        return;
    }
    filename_ = filename;

    if (filename.empty() && file_.is_open()) {
        file_.close();
        return;
    }

    file_.open(filename_, std::ios_base::app);
}

void LogWriter::WriteString(std::string data) {
    std::lock_guard<std::mutex> lk(mutex_);

    auto stm = std::istringstream(data);
    auto line = std::string{};

    if (!filename_.empty()) {
        // TODO: Print more verbose for every lines.
        while (std::getline(stm, line)) {
            file_ << line << std::endl;
        }
    }
}

LogOptions& LogOptions::Get() {
    static LogOptions lo;
    return lo;
}

void LogOptions::SetQuiet(bool q) {
    quiet_ = q;
}

Logging::Logging(const char* file, int line, bool write_only, bool use_options) {
    file_ = std::string{file};
    line_ = line;
    write_only_ = write_only;
    use_options_ = use_options;
}

Logging::~Logging() {
    if (!write_only_ && (!use_options_ || !LogOptions::Get().quiet_)) {
        std::cout << str() << std::flush;
    }
    LogWriter::Get().WriteString(str());
}

StandError::StandError(const char* file, int line) {
    file_ = std::string{file};
    line_ = line;
}

StandError::~StandError() {
    std::cerr << str() << std::flush;
    LogWriter::Get().WriteString(str());
}
