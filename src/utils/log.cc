#include <iostream>
#include <sstream>

#include "utils/log.h"

LogWriter& LogWriter::Get() {
    static LogWriter writer;
    return writer;
}

void LogWriter::SetFilename(std::string filename) {
    Mutex::Lock lock_(mutex_);
    if (filename_ == filename) {
        return;
    }
    filename_ = filename;

    if (filename.empty()) {
        file_.close();
        return;
    }

    file_.open(filename_, std::ios_base::app);

    for (const auto& line : buffer_) {
        file_ << line << std::endl;
    }
    buffer_.clear();
}

void LogWriter::WriteString(std::string data) {
    Mutex::Lock lock_(mutex_);
    auto stm = std::istringstream(data);
    auto line = std::string{};

    if (filename_.empty()) {
        while (std::getline(stm, line)) {
            buffer_.emplace_back(line);
            if (buffer_.size()  >= kMaxBufferLines) {
                buffer_.pop_front();
            }
        }
    } else {
        while (std::getline(stm, line)) {
            file_ << line << std::endl;
        }
    }
}

Logging::Logging(const char* file, int line, bool write_only) {
    file_ = std::string{file};
    line_ = line;
    write_only_ = write_only;
}

Logging::~Logging() {
    if (!write_only_) {
        std::cerr << str() << std::flush;
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
