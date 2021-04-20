#pragma once

#include <string>
#include <unordered_map>

const std::string kProgram = "NA";

const std::string kVersion = "pre-alpha"; 

const std::string kNoWeightsFile = "NO_WEIGHT_FILE";

const std::string kNologFile = "NO_LOG_FILE";

template<typename T>
T GetOption(std::string name);

template<typename T>
bool SetOption(std::string name, T val);

void InitBasicParameters();

class ArgsParser {
public:
    ArgsParser() = delete;
    ArgsParser(int argc, char** argv);

    void Dump() const;

private:
    void Helper() const;
};

