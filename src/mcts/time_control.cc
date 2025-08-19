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
        // The byo yomi stones and byo periods should not be greater or
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
            // Japanese byo yomi style type
            if (byotime_left_[color] < 0) {
                periods_left_[color]--;
            }

            if (periods_left_[color] > 0) {
                byotime_left_[color] = byo_time_;
            }
        } else if (byo_stones_) {
            // Canadian style type
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

void TimeControl::SetLagBuffer(float lag_buffer_sec) {
    lag_buffer_ = static_cast<int>(lag_buffer_sec * 100.f); // second to centisecond
    lag_buffer_ = std::max(lag_buffer_, 0);
}

float TimeControl::GetLagBuffer() const {
    return static_cast<double>(lag_buffer_) / 100.f;
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
            // Japanese byo yomi style type
            out << "Periods left: " << periods_left_[color];
        } else if (byo_stones_) {
            // Canadian style type
            out << "Stones left: " << stones_left_[color];
        }
    }
    out << std::setfill(' ');
}

float TimeControl::GetBufferEffect(int color, int boardsize, int move_num) const {
    if (IsInfiniteTime(color)) {
        return 0.f;
    }
    float t1 = GetThinkingTime(
                   color, boardsize, move_num, true);
    float t2 = GetThinkingTime(
                   color, boardsize, move_num, false);
    return std::max(t2 - t1, 0.f);
}

float TimeControl::GetThinkingTime(int color, int boardsize,
                                   int move_num, bool use_lag_buffer) const {
    assert(color == kBlack || color == kWhite);

    if (IsInfiniteTime(color)) {
        return GetInfiniteTime();
    }

    if (IsTimeOver(color)) {
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

        moves_remaining = EstimateMovesExpected(boardsize, move_num);
        time_remaining = maintime_left_[color] + byo_extra;
    }

    int lag_buffer_cs = use_lag_buffer ?
                            lag_buffer_ : 0;
    int base_time = std::max(time_remaining - lag_buffer_cs, 0) /
                        std::max(moves_remaining, 1);
    int inc_time = std::max(extra_time_per_move - lag_buffer_cs, 0);

    return static_cast<double>(base_time + inc_time) / 100.f; // centisecond to second
}

bool TimeControl::CanAccumulateTime(int color) const {
    // Returns true if we are in a time control where we
    // can save up time. If not, we should not move quickly
    // even if certain of our move, but plough ahead.

    if (in_byo_[color]) {
        // Cannot accumulate in Japanese byo yomi
        if (byo_periods_) {
            return false;
        }

        // Cannot accumulate in Canadese style with
        // one move remaining in the period
        if (byo_stones_ && stones_left_[color] == 1) {
            return false;
        }
    } else {
        // If there is a base time, we should expect
        // to be able to accumulate. This may be somewhat
        // of an illusion if the base time is tiny and byo
        // yomi time is big.
    }

    return true;
}

bool TimeControl::InByo(int color) const {
    return in_byo_[color];
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

float TimeControl::GetInfiniteTime() const {
    return 31 * 24 * 60 * 60;
}

float TimeControl::GetDuration() const {
    return timer_.GetDuration();
}

int TimeControl::EstimateMovesExpected(int boardsize, int move_num) const {
    const int num_intersections = boardsize * boardsize;
    const int side_move_num = move_num/2;

    // The 'base_move_num' is 153 on 19x19.
    // The 'base_move_num' is  71 on 13x13.
    // The 'base_move_num' is  32 on 9x9.
    const int base_move_num = (0.8f * num_intersections + 1.75f * (boardsize - 9))/2;
    const int base_remaining = base_move_num- side_move_num;
    const int opening_move_num = (0.2f * num_intersections) / 2;

    // The formula is base from this
    //     https://www.remi-coulom.fr/Publications/TimeManagement.pdf
    //
    // We should reduce the time used in the opening stage. And leave
    // more time for middle game. Because the engine should spend more
    // time on complicated and undecided semeai and life-and-death
    // conditions.
    float opening_factor = 2.5f;
    int estimated_moves =
        base_remaining +
        opening_factor * std::max(opening_move_num - side_move_num, 0);

    // Be sure that the moves left is not too low.
    // Minimal moves left is 43 on 19x19.
    // Minimal moves left is 21 on 13x13.
    // Minimal moves left is 15 on 9x9.
    estimated_moves = std::max(estimated_moves,
                          std::max((int)(0.3f * base_move_num), 15));
    return estimated_moves;
}
