#include "mcts/time_control.h"
#include "game/types.h"

#include <iomanip>
#include <cassert>
#include <algorithm>
#include <sstream>
#include <iostream>

static_assert(sizeof(int) == 4, "");

TimeControl::TimeControl() {
    SetLagBuffer(0);
    TimeSettings(0, 0, 0, 0);
}

void TimeControl::TimeSettings(const int main_time,
                                   const int byo_yomi_time,
                                   const int byo_yomi_stones,
                                   const int byo_yomi_periods) {
    // We store the time as centisecond. If the input seconds is greater
    // than 248 days, we set infinite time.
    int max_value = 248 * 24 * 60 * 60;
    if (main_time > max_value ||
            byo_yomi_time > max_value) {
        TimeSettings(0, 0, 0, 0);
        return;
    }

    main_time_ = 100 * main_time; // max time about 248 days
    byo_time_ = 100 * byo_yomi_time; // max time about 248 days

    byo_stones_ = byo_yomi_stones;
    byo_periods_ = byo_yomi_periods;

    if (main_time_ <= 0) {
        main_time_ = 0;
    }

    if ((byo_stones_ <= 0 && byo_periods_ <= 0) ||
            (byo_stones_ > 0 && byo_periods_ > 0)) {
        // The byo_yomi_stones and byo_periods should not be greater or
        // smaller than zero at the same time.
        byo_time_ = byo_periods_ = 0;
    }

    if (byo_time_ <= 0 && byo_periods_ == 0) {
        byo_stones_ = 0;
    }

    Reset();
}

void TimeControl::TimeLeft(const int color, const int time, const int stones) {
    if (time <= 0 && stones <= 0) {
        // From pachi: some GTP things send 0 0 at the end of main time
        byotime_left_[color] = byo_time_;
        byotime_left_[color] = byo_stones_;
        stones_left_[color] = byo_periods_;
    } else if (stones <= 0) {
        maintime_left_[color] = 100 * time; // second to centisecond
    } else {
        maintime_left_[color] = 0; // no time
        byotime_left_[color] = 100 * time; // second to centisecond

        if (byo_periods_) {
            periods_left_[color] = stones;
            stones_left_[color] = 0;
        } else if (byo_stones_) {
            periods_left_[color] = 0;
            stones_left_[color] = stones;
        }
    }

    CheckInByo();
}

void TimeControl::Clock() {
    timer_.Clock();
}

void TimeControl::TookTime(int color) {
    assert(color == kBlack || color == kWhite);

    if (IsInfiniteTime(color)) {
        return;
    }

    assert(!IsTimeOver(color));
    int remaining_took_time = timer_.GetDurationMilliseconds()/10;

    if (!in_byo_[color]) {
        if (maintime_left_[color] >= remaining_took_time) {
            maintime_left_[color] -= remaining_took_time;
            remaining_took_time = 0;
        } else {
            remaining_took_time -= maintime_left_[color];
            maintime_left_[color] = 0;
            in_byo_[color] = true;
        }
    }

    if (in_byo_[color] && remaining_took_time > 0) {
        byotime_left_[color] -= remaining_took_time;

        if (byo_periods_) {
            // Byo-Yomi type
            if (byotime_left_[color] < 0) {
                periods_left_[color]--;
            }

            if (periods_left_[color] > 0) {
                byotime_left_[color] = byo_time_;
            }
        } else if (byo_stones_) {
            // Canadian type
            stones_left_[color]--;
            if (stones_left_[color] == 0) {
                if (byotime_left_[color] > 0) {
                    byotime_left_[color] = byo_time_;
                    stones_left_[color] = byo_stones_;
                }
            }
        }
    }
}

void TimeControl::SetLagBuffer(int lag_buffer) {
    constexpr int kMinLag = 25; // 0.25 second is big enough for CPU
                                // forward pipe hiccupping.

    lag_buffer *= 100;
    lag_buffer_ = lag_buffer < kMinLag ? kMinLag : lag_buffer;

}

void TimeControl::Reset() {
    maintime_left_.fill(main_time_);
    byotime_left_.fill(byo_time_);
    stones_left_.fill(byo_stones_);
    periods_left_.fill(byo_periods_);

    CheckInByo();
}

void TimeControl::CheckInByo() {
    in_byo_[kBlack] = (maintime_left_[kBlack] <= 0);
    in_byo_[kWhite] = (maintime_left_[kWhite] <= 0);
}

std::string TimeControl::ToString() const {
    auto out = std::ostringstream{};
    TimeStream(out);
    return out.str();
}

void TimeControl::TimeStream(std::ostream &out) const {
    TimeStream(out, kBlack);
    out << " | ";
    TimeStream(out, kWhite);
    out << std::endl;
}

void TimeControl::TimeStream(std::ostream &out, int color) const {
    assert(color == kBlack || kWhite);

    if (color == kBlack) {
        out << "Black time: ";
    } else {
        out << "White time: ";
    }

    if (IsInfiniteTime(color)) {
        out << "infinite";
    } else if (!in_byo_[color]) {
       const int remaining = maintime_left_[color]/100; // centisecond to second
       const int hours = remaining / 3600;
       const int minutes = (remaining % 3600) / 60;
       const int seconds = remaining % 60;
       out << std::setw(2) << hours << ":";
       out << std::setw(2) << std::setfill('0') << minutes << ":";
       out << std::setw(2) << std::setfill('0') << seconds;
    } else {
       const int remaining = byotime_left_[color]/100; // centisecond to second
       const int hours = remaining / 3600;
       const int minutes = (remaining % 3600) / 60;
       const int seconds = remaining % 60;

       out << std::setw(2) << hours << ":";
       out << std::setw(2) << std::setfill('0') << minutes << ":";
       out << std::setw(2) << std::setfill('0') << seconds << ", ";

        if (byo_periods_) {
            // Byo-Yomi type
            out << "Periods left: " << periods_left_[color];
        } else if (byo_stones_) {
            // Canadian type
            out << "Stones left: " << stones_left_[color];
        }
    }
    out << std::setfill(' ');
}

float TimeControl::GetThinkingTime(int color, int boardsize, int move_num) const {
    assert(color == kBlack || color == kWhite);

    if(IsInfiniteTime(color)) {
        return 31 * 24 * 60 * 60;
    }

    if(IsTimeOver(color)) {
        // no time to use
        return 0;
    }

    int time_remaining = 0;
    int moves_remaining = 0;
    int extra_time_per_move = 0;

    if (in_byo_[color]) {
        if (byo_periods_) {
            // just use the byo time
            extra_time_per_move = byo_time_;
        } else if (byo_stones_) {
            time_remaining = byotime_left_[color];
            moves_remaining = stones_left_[color];
        }
    } else {
        int byo_extra = 0;

        if (byo_periods_) {
            byo_extra = byo_time_ * (periods_left_[color] - 1);
            extra_time_per_move = byo_time_;
        } else if (byo_stones_) {
            byo_extra = byotime_left_[color] / stones_left_[color];
            extra_time_per_move = byo_extra;
        }

        moves_remaining = EstimateMovesExpected(boardsize, move_num, 0);
        time_remaining = maintime_left_[color] + byo_extra;
    }

    int base_time = std::max(time_remaining - lag_buffer_, 0) / std::max(moves_remaining, 1);
    int inc_time = std::max(extra_time_per_move - lag_buffer_, 0);

    // Output value may loss littel precision. It is OK that most case
    // the losing of precision is smaller than one second. If the losing
    // value greater than one second, output value is very large. We don't 
    // care the losing in that case.
    return (float)(base_time + inc_time) / 100.f; // centisecond to second
}


bool TimeControl::IsTimeOver(int color) const {
    if (maintime_left_[color] > 0 ||
            byotime_left_[color] > 0) {
        return false;
    }
    return true;
}

bool TimeControl::IsInfiniteTime(int /* color */) const {
    return main_time_ == 0 &&
               byo_time_ == 0 &&
               byo_stones_ == 0 &&
               byo_periods_ == 0;
}

int TimeControl::EstimateMovesExpected(int boardsize, int move_num, int div_delta) const {
    const int num_intersections = boardsize * boardsize;

    const int side_move_num = move_num/2;
    const int base_remaining = num_intersections / (3 + div_delta);
    const int fast_moves = num_intersections / (9 + div_delta);
    const int moves_buffer = num_intersections / (2 * boardsize + div_delta);

    int estimate_moves = 0;

    if (side_move_num < fast_moves) {
        estimate_moves = base_remaining + fast_moves - side_move_num;
    } else {
        estimate_moves = base_remaining - side_move_num;
    }

    return std::max(estimate_moves, moves_buffer);
}
