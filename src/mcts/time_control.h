#pragma once

#include <array>
#include <string>

#include "utils/time.h"

class TimeControl {
public:
    TimeControl();

    void TimeSettings(const int main_time,
                          const int byo_yomi_time,
                          const int byo_yomi_stones,
                          const int byo_yomi_periods);

    void TimeLeft(const int color, const int time, const int stones);

    void SetLagBuffer(int lag_buffer);

    float GetThinkingTime(int color, int boardsize, int move_num) const;

    void Clock();
    void TookTime(int color);

    bool IsTimeOver(int color) const;

    void TimeStream(std::ostream &out) const;
    void TimeStream(std::ostream &out, int color) const;
    std::string ToString() const;

    bool IsInfiniteTime(int color) const;

private:
    void Reset();
    void CheckInByo();

    int main_time_;
    int byo_time_;
    int byo_stones_;
    int byo_periods_;

    std::array<int, 2> maintime_left_; // centiseconds
    std::array<int, 2> byotime_left_;  // centiseconds
    std::array<int, 2> stones_left_;
    std::array<int, 2> periods_left_;

    std::array<bool,  2> in_byo_;

    int lag_buffer_; // centiseconds

    Timer timer_;

    int EstimateMovesExpected(int boardsize, int move_num, int div_delta) const;
};
