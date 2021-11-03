#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>

template<typename ... Args>
std::string Format(const char *fmt, Args ... args) {
    int size = std::snprintf(nullptr, 0, fmt, args ...);
    if(size < 0){
        throw std::runtime_error("Error during formatting.");
    }

    auto buf = std::vector<char>(size+1);
    std::snprintf(buf.data(), buf.size(), fmt, args ...);

    return std::string(buf.data(), buf.data() + size);
}
