#pragma once

#include "utils/parser.h"
#include <string>
#include <unordered_map>

template<typename T>
T GetOption(std::string name);

template<typename T>
bool SetOption(std::string name, T val);

class ArgsParser {
public:
    ArgsParser() = delete;
    ArgsParser(int argc, char** argv);

private:
    void Parse(CommandParser &parser);

    void DumpHelper() const;
    void DumpWarning() const;
    void InitBasicParameters() const;
    void InitOptionsMap() const;

    bool init_fpu_root_{false};
    std::string inputs_;
};
