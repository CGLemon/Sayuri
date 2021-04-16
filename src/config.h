#pragma once

#include <string>
#include <unordered_map>

const std::string kProgram = "NA";

const std::string kVersion = "pre-alpha"; 

const std::string kNoWeightsFile = "NO_WEIGHT_FILE";

const std::string kNologFile = "NO_LOG_FILE";

template<typename T>
T option(std::string name);

template<typename T>
bool set_option(std::string name, T val);

void init_basic_parameters();

class ArgsParser {
public:
    ArgsParser() = delete;
    ArgsParser(int argc, char** argv);

    void dump() const;

private:
    void help() const;
};

