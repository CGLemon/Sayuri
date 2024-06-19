#pragma once

#include <string>
#include <vector>
#include <chrono>

// Get current date/time, format is YYYY-MM-DD-HH:mm:ss
const std::string CurrentDateTime();

// Reture the time hash base on localtime().
std::uint64_t GetTimeHash();

class Timer {
public:
    Timer();
    void Clock();

    int GetDurationSeconds() const;
    int GetDurationMilliseconds() const;
    int GetDurationMicroseconds() const;

    float GetDuration() const;

private:
    std::chrono::steady_clock::time_point clock_time_;
};
