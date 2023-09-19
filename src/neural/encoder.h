#pragma once

#include <vector>
#include <array>
#include "game/symmetry.h"
#include "game/game_state.h"
#include "neural/network_basic.h"

class Encoder {
public:
    static constexpr int kPlaneChannels = kInputChannels;
    static constexpr int kHistoryMoves = 8;
    static constexpr int kNumBinaryFeatures = 13;
    static constexpr int kNumMiscFeatures = 6;
    static constexpr int kNumFeatures = kNumBinaryFeatures + kNumMiscFeatures;
    static_assert(kPlaneChannels == 3*kHistoryMoves + kNumFeatures, "");

    static Encoder& Get();

    // Get the Network input datas.
    InputData GetInputs(const GameState &state, int symmetry = Symmetry::kIdentitySymmetry) const;

    /*
     * Get the v3 Network input planes.
     *
     * planes 1 -24 : last 8 history moves
     * plane     25 :
     * planes 26-29 :
     * planes 30-33 :
     * planes 34-37 :
     * plane     38 : rule, not used now
     * plane     39 : wave
     * plane     40 : komi/20
     * plane     41 : -komi/20
     * plane     42 : intersections/361
     * plane     43 : fill ones
     *
     */
    std::vector<float> GetPlanes(const GameState &state, int symmetry = Symmetry::kIdentitySymmetry) const;

    std::string GetPlanesString(const GameState &state, int symmetry = Symmetry::kIdentitySymmetry) const;

private:
    void SymmetryPlanes(const GameState &state, std::vector<float> &planes, int symmetry) const;

    void FillColorStones(const Board* board,
                         std::vector<float>::iterator black_it,
                         std::vector<float>::iterator white_it) const;

    void FillMove(const Board* board,
                  std::vector<float>::iterator move_it) const;

    void EncoderHistoryMove(const GameState &state,
                            int counter,
                            std::vector<float>::iterator it) const;

    void FillFeatures(const Board* board,
                      const int to_move,
                      std::vector<float>::iterator feat_it) const;

    void FillMisc(const Board* board,
                  const int color,
                  float rule, float wave, float komi,
                  std::vector<float>::iterator color_it) const;

    void EncoderFeatures(const GameState &state,
                         std::vector<float>::iterator it) const;

};
