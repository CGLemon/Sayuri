#pragma once

#include "game/types.h"

#include <array>
#include <cmath>
#include <vector>

// Imported from the: https://github.com/lightvector/KataGo/blob/master/cpp/neuralnet/nninputs.cpp
class ScoreValue {
public:
    ScoreValue() = default;

    static ScoreValue& Get();

    void Initialize();
    float ExpectedScoreValue(float mean, float stddev, float center,
                             float scale, float board_size);
private:
    static constexpr float kTwoOverPi = 0.63661977236758134308f;
    static constexpr int kExtraRadius = 60;
    static constexpr int kSvTableAssumedBSize = kBoardSize;
    static constexpr int kSvTableMeanRadius = kSvTableAssumedBSize * kSvTableAssumedBSize + kExtraRadius;
    static constexpr int kSvTableMeanLen = kSvTableMeanRadius * 2;
    static constexpr int kSvTableStddevLen = kSvTableAssumedBSize * kSvTableAssumedBSize + kExtraRadius;

    static constexpr int kEntrySize = kSvTableMeanLen * kSvTableStddevLen;
    std::array<float, kEntrySize> expected_lookup_table_;

    float ScoreSmoothAdjust(float mean, float center,
                            float scale, float board_size);
};

inline ScoreValue& ScoreValue::Get() {
    static ScoreValue sv;
    return sv;
}

inline void ScoreValue::Initialize() {
    // Precompute normal PDF

    // Must be divisible by 2. This is both the number
    // of segments that we divide points into, and that
    // we divide stddevs into
    const int steps_per_unit = 10;

    const int bound_stddevs = 5;
    const int bound_stddev_steps_abs = bound_stddevs * steps_per_unit;
    const int min_stddev_steps = -bound_stddev_steps_abs;
    const int max_stddev_steps = bound_stddev_steps_abs;

    auto normal_pdf = std::vector<float>(max_stddev_steps - min_stddev_steps +1);
    for(int i = min_stddev_steps; i <= max_stddev_steps; i++) {
        float x_in_stddevs = static_cast<float>(i) / steps_per_unit;
        float w = std::exp(-0.5f * x_in_stddevs * x_in_stddevs);
        normal_pdf[i - min_stddev_steps] = w;
    }
    // Precompute scorevalue at increments of 1/steps_per_unit points
    const int sv_steps_abs = kSvTableMeanRadius * steps_per_unit +
                                 steps_per_unit/2 +
                                 bound_stddevs * kSvTableStddevLen * steps_per_unit;
    const int min_sv_steps = -sv_steps_abs;
    const int max_sv_steps = sv_steps_abs;

    auto sv_precomp = std::vector<float>(max_sv_steps - min_sv_steps + 1);
    for(int i = min_sv_steps; i <= max_sv_steps; i++) {
        float mean = static_cast<float>(i) / steps_per_unit;
        float sv = ScoreSmoothAdjust(mean, 0.0, 1.0, kSvTableAssumedBSize);
        sv_precomp[i - min_sv_steps] = sv;
    }

    // Perform numeric integration
    for(int mean_idx = 0; mean_idx < kSvTableMeanLen; mean_idx++) {
        int mean_steps = (mean_idx - kSvTableMeanRadius) * steps_per_unit - steps_per_unit/2;
        for(int stddev_idx = 0; stddev_idx < kSvTableStddevLen; stddev_idx++) {
            float w_sum = 0.0;
            float w_sv_sum = 0.0;
            for(int i = min_stddev_steps; i <= max_stddev_steps; i++) {
                int x_steps = mean_steps + stddev_idx * i;
                float w = normal_pdf[i-min_stddev_steps];
                float sv = sv_precomp[x_steps - min_sv_steps];
                w_sum += w;
                w_sv_sum += w*sv;
            }
            expected_lookup_table_[mean_idx * kSvTableStddevLen + stddev_idx] = w_sv_sum / w_sum;
        }
    }
}

inline float ScoreValue::ScoreSmoothAdjust(float final_score, float center,
                                           float scale, float board_size) {
    float adjusted_score = final_score - center;
    return std::atan(adjusted_score / (scale * board_size)) * kTwoOverPi;
}

inline float ScoreValue::ExpectedScoreValue(float mean, float stddev, float center,
                                            float scale, float board_size) {
    float scale_factor = static_cast<float>(kSvTableAssumedBSize) / (scale * board_size);

    float mean_scaled = (mean - center) * scale_factor;
    float stddev_scaled = stddev * scale_factor;

    float mean_rounded = std::round(mean_scaled);
    float stddev_floored = std::floor(stddev_scaled);
    int mean_idx_0 = static_cast<int>(mean_rounded) + kSvTableMeanRadius;
    int stddev_idx_0 = static_cast<int>(stddev_floored);
    int mean_idx_1 = mean_idx_0 + 1;
    int stddev_idx_1 = stddev_idx_0 + 1;

    if (mean_idx_0 < 0) {
        mean_idx_0 = 0;
        mean_idx_1 = 0;
    }
    if (mean_idx_1 >= kSvTableMeanLen) {
        mean_idx_0 = kSvTableMeanLen - 1;
        mean_idx_1 = kSvTableMeanLen - 1;
    }
    if (stddev_idx_1 >= kSvTableStddevLen) {
        stddev_idx_0 = kSvTableStddevLen - 1;
        stddev_idx_1 = kSvTableStddevLen - 1;
    }

    float lambda_mean = mean_scaled - mean_rounded + 0.5;
    float lambda_stddev = stddev_scaled - stddev_floored;

    float a00 = expected_lookup_table_[mean_idx_0 * kSvTableStddevLen + stddev_idx_0];
    float a01 = expected_lookup_table_[mean_idx_0 * kSvTableStddevLen + stddev_idx_1];
    float a10 = expected_lookup_table_[mean_idx_1 * kSvTableStddevLen + stddev_idx_0];
    float a11 = expected_lookup_table_[mean_idx_1 * kSvTableStddevLen + stddev_idx_1];

    float b0 = a00 + lambda_stddev * (a01 - a00);
    float b1 = a10 + lambda_stddev * (a11 - a10);
    return b0 + lambda_mean * (b1 - b0);
}

