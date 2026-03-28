#pragma once

#include <string>
#include <unordered_map>

#include "utils/splitter.h"

class ArgsParser {
public:
    ArgsParser() = delete;
    ArgsParser(int argc, char** argv);

private:
    void Parse(Splitter& splitter);

    void DumpHelper() const;
    void DumpWarning() const;
    void InitBasicParameters() const;
    void InitOptionsMap() const;

    std::string inputs_;
};
