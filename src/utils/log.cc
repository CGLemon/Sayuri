#include <iostream>

#include "utils/log.h"
#include "utils/parser.h"

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
    auto parser = CommandParser(data);
    auto cnt = size_t{0};

    if (filename_.empty()) {
        while (const auto line = parser.GetCommand(cnt++)) {
            buffer_.emplace_back(line->Get<std::string>());
            if (parser.GetCount() == cnt) {
                break;
            }
        }
    } else {
        while (const auto line = parser.GetCommand(cnt++)) {
            file_ << line->Get<std::string>() << std::endl;
            if (parser.GetCount() == cnt) {
                break;
            }
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
