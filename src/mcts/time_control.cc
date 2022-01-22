#include "mcts/time_control.h"
#include "game/types.h"

#include <iomanip>
#include <cassert>
#include <algorithm>
#include <sstream>

TimeControl::TimeControl() {
    TimeSettings(0, 0, 0);
}

void TimeControl::TimeSettings(const int main_time,
                               const int byo_yomi_time,
                               const int byo_yomi_stones) {
    main_time_ = main_time;
    byo_time_ = byo_yomi_time;
    byo_stones_ = byo_yomi_stones;
    byo_periods_ = 1;

    if (main_time_ <= 0) {
        main_time_ = 0;
    }

    if (byo_stones_ <= 0 || byo_time_ <= 0) {
        byo_time_ = 0;
        byo_stones_ = 0;
    }

    Reset();
}

void TimeControl::TimeLeft(const int color, const int time, const int stones) {
    assert(color == kBlack || color == kWhite);
    if (stones <= 0) {
        maintime_left_[color] = static_cast<float>(time);
        byotime_left_[color] = 0.0f;
        stones_left_[color] = 0;
    } else {
        maintime_left_[color] = 0.0f;
        byotime_left_[color] = static_cast<float>(time);
        stones_left_[color] = stones;
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
    float remaining_took_time = timer_.GetDuration();

    if (!in_byo_[color]) {
        if (maintime_left_[color] >= remaining_took_time) {
            maintime_left_[color] -= remaining_took_time;
            remaining_took_time = 0.0f;
        } else {
            remaining_took_time -= maintime_left_[color];
            maintime_left_[color] = 0.0f;
            in_byo_[color] = true;
        }
    }

    if (in_byo_[color] && remaining_took_time > 0.0f) {
        byotime_left_[color] -= remaining_took_time;
        stones_left_[color] --;

        if (stones_left_[color] == 0) {
            if (byotime_left_[color] > 0.0f) {
                byotime_left_[color] = byo_time_;
                stones_left_[color] = byo_stones_;
            }
        }
    }

    CheckInByo();
}

void TimeControl::SetLagBuffer(int lag_buffer) {
    lag_buffer_ = lag_buffer < 0 ? 0.0f : (float)lag_buffer;
}

void TimeControl::Reset() {
    maintime_left_ = {(float)main_time_, (float)main_time_};
    byotime_left_ = {(float)byo_time_, (float)byo_time_};
    stones_left_ = {byo_stones_, byo_stones_};
    periods_left_ = {byo_periods_, byo_periods_};

    CheckInByo();
}

void TimeControl::CheckInByo() {
    in_byo_[kBlack] = (maintime_left_[kBlack] <= 0.f);
    in_byo_[kWhite] = (maintime_left_[kWhite] <= 0.f);
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
       const int remaining = static_cast<int>(maintime_left_[color]);
       const int hours = remaining / 3600;
       const int minutes = (remaining % 3600) / 60;
       const int seconds = remaining % 60;
       out << std::setw(2) << hours << ":";
       out << std::setw(2) << std::setfill('0') << minutes << ":";
       out << std::setw(2) << std::setfill('0') << seconds;
    } else {
       const int remaining = static_cast<int>(byotime_left_[color]);
       const int stones_left = stones_left_[color];
       const int hours = remaining / 3600;
       const int minutes = (remaining % 3600) / 60;
       const int seconds = remaining % 60;

       out << std::setw(2) << hours << ":";
       out << std::setw(2) << std::setfill('0') << minutes << ":";
       out << std::setw(2) << std::setfill('0') << seconds << ", ";;
       out << "Stones left: " << stones_left;
    }
    out << std::setfill(' ');
}

float TimeControl::GetThinkingTime(int color, int boardsize, int move_num) const {
    assert(color == kBlack || color == kWhite);

    if(IsInfiniteTime(color)) {
        return 31 * 24 * 60 * 60 * 100;
    }

    if(IsTimeOver(color)) {
        // No time to use;
        return 0.f;
    }

    auto time_remaining = 0.f;
    auto moves_remaining = 0;
    auto extra_time_per_move = 0.f;

    if (in_byo_[color]) {
        time_remaining = byotime_left_[color];
        moves_remaining = stones_left_[color];
    } else {
        float byo_extra = 0;
        if (stones_left_[color] > 0) {
            byo_extra = byotime_left_[color] / stones_left_[color];
        }

        time_remaining = maintime_left_[color] + byo_extra;
        moves_remaining = EstimateMovesExpected(boardsize, move_num, 0);
        extra_time_per_move = byo_extra;
    }

    auto base_time = std::max(time_remaining - lag_buffer_, 0.f) /
                         std::max(moves_remaining, 1);
    auto inc_time = std::max(extra_time_per_move - lag_buffer_, 0.f);

    return base_time + inc_time;
}


bool TimeControl::IsTimeOver(int color) const {
    if (maintime_left_[color] > 0.0f) {
        return false;
    }
    if (byotime_left_[color] > 0.0f) {
        return false;
    }
    return true;
}

bool TimeControl::IsInfiniteTime(int color) const {
    return maintime_left_[color] == 0 &&
               byotime_left_[color] == 0 &&
               stones_left_[color] == 0 &&
               main_time_ == 0 && byo_stones_ == 0 && byo_time_ == 0;
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

