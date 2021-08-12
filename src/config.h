#pragma once

#include <string>
#include <unordered_map>

const std::string kProgram = "Sayuri";

const std::string kVersion = "pre-alpha";

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

