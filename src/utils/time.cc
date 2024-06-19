#include <iostream>
#include <cstdio>
#include <ctime>
#include <cstring>
#include "utils/time.h"

const std::string CurrentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d-%X", &tstruct);

    return buf;
}

std::uint64_t GetTimeHash() {
    time_t     now = time(0);
    struct tm  tstruct;
    tstruct = *localtime(&now);

    size_t bufsize = sizeof(struct tm);
    std::vector<char> buf(bufsize);
    std::memcpy(buf.data(), &tstruct, bufsize);
    std::string timebuf(buf.data(), buf.data() + bufsize);

    return std::hash<std::string>()(timebuf);
}

Timer::Timer() {
    Clock();
}

void Timer::Clock() {
    clock_time_ = std::chrono::steady_clock::now();
}

int Timer::GetDurationSeconds() const {
    const auto end_time = std::chrono::steady_clock::now();
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(end_time - clock_time_).count();
    return seconds;
}

int Timer::GetDurationMilliseconds() const {
    const auto end_time = std::chrono::steady_clock::now();
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - clock_time_).count();
    return milliseconds;
}

int Timer::GetDurationMicroseconds() const {
    const auto end_time = std::chrono::steady_clock::now();
    const auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time - clock_time_).count();
    return microseconds;
}

float Timer::GetDuration() const {
    const auto seconds = GetDurationSeconds();
    const auto milliseconds = GetDurationMilliseconds();
    if (seconds == (milliseconds/1000)) {
        return static_cast<float>(milliseconds) / 1000.f;
    } else {
        return static_cast<float>(seconds);
    }
}
