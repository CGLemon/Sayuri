#pragma once

#include <array>
#include <string>
#include <iostream>

#include "utils/time.h"

class TimeControl {
public:
    TimeControl();

    void TimeSettings(const int main_time,
                      const int byo_yomi_time,
                      const int byo_yomi_stones);

    void TimeLeft(const int color, const int time, const int stones);

    void SetLagBuffer(int lag_buffer);

    float GetThinkingTime(int color, int boardsize, int move_num) const;

    void Clock();
    void TookTime(int color);

    bool IsTimeOver(int color) const;

    void TimeStream(std::ostream &out) const;
    void TimeStream(std::ostream &out, int color) const;
    std::string ToString() const;

private:
    void Reset();
    void CheckInByo();

    int main_time_;
    int byo_time_;
    int byo_stones_;
    int byo_periods_;

    std::array<float, 2> maintime_left_;
    std::array<float, 2> byotime_left_;
    std::array<int,   2> stones_left_;
    std::array<int,   2> periods_left_;
    std::array<bool,  2> in_byo_;

    float lag_buffer_cs_{0};

    Timer timer_;

    bool IsInfiniteTime() const;
    int EstimateMovesExpected(int boardsize, int move_num, int div_delta) const;
};
