#pragma once

#include <vector>
#include <array>
#include "game/types.h"
#include "game/symmetry.h"
#include "game/game_state.h"
#include "neural/network_basic.h"

class Encoder {
public:
    static Encoder& Get();

    // Get the Network input datas.
    InputData GetInputs(const GameState &state,
                        const int symmetry = Symmetry::kIdentitySymmetry,
                        const int weights_version = -1) const;

    /*
     *
     * Get the Network input planes.
     *
     * v1~v2 weights -> v1 encoder:
     * planes 1 -24 : last 8 history moves, for each three planes
     *                    1. current player's stones on board
     *                    2. opponent player's stones on board
     *                    3. last move
     * plane     25 : ko move
     * plane     26 : pass-alive and pass-dead area
     * planes 27-30 : strings with 1, 2, 3 and 4 liberties 
     * planes 31-34 : ladder features
     * plane     35 : komi/20
     * plane     36 : -komi/20
     * plane     37 : intersections/361
     * plane     38 : fill ones
     *
     *
     * v3~v4 weights -> v2 encoder:
     * planes 1 -24 : last 8 history moves, for each three planes
     *                    1. current player's stones on board
     *                    2. opponent player's stones on board
     *                    3. last move
     * plane     25 : ko move
     * planes 26-29 : pass-alive and pass-dead area
     * planes 30-33 : strings with 1, 2, 3 and 4 liberties
     * planes 34-37 : ladder features
     * plane     38 : scoring rule (area or territory)
     * plane     39 : wave
     * plane     40 : komi/20
     * plane     41 : -komi/20
     * plane     42 : intersections/361
     * plane     43 : fill ones
     *
     */
    std::vector<float> GetPlanes(const GameState &state,
                                 const int symmetry = Symmetry::kIdentitySymmetry,
                                 const int weights_version = -1) const;

    std::string GetPlanesString(const GameState &state,
                                const int symmetry = Symmetry::kIdentitySymmetry,
                                const int weights_version = -1) const;

    constexpr static int GetEncoderVersion(const int weights_version = -1) {
        switch (weights_version) {
            case 1: 
            case 2: return 1;
            case 3: 
            case 4: return 2;
            default: break;
        };
        return 2;
    }
    constexpr static int GetHistoryMoves(const int weights_version = - 1) {
        const int encoder_version = GetEncoderVersion(weights_version);
        if (encoder_version == 1 || encoder_version == 2) {
            return 8;
        }
        return 0;
    }
    constexpr static int GetNumBinaryFeatures(const int weights_version = - 1) {
        const int encoder_version = GetEncoderVersion(weights_version);
        if (encoder_version == 1) {
            return 10;
        }
        if (encoder_version == 2) {
            return 13;
        }
        return 0;
    }
    constexpr static int GetNumMiscFeatures(const int weights_version = - 1) {
        const int encoder_version = GetEncoderVersion(weights_version);
        if (encoder_version == 1) {
            return 4;
        }
        if (encoder_version == 2) {
            return 6;
        }
        return 0;
    }
    constexpr static int GetNumFeatures(const int weights_version = - 1) {
        return GetNumBinaryFeatures(weights_version) + GetNumMiscFeatures(weights_version);
    }
    constexpr static int GetInputChannels(const int weights_version = -1) {
        return 3 * GetHistoryMoves(weights_version) + GetNumFeatures(weights_version);
    }


private:
    void SymmetryPlanes(const GameState &state, std::vector<float> &planes,
                        const int symmetry, const int weights_version) const;

    void FillColorStones(const Board* board,
                         std::vector<float>::iterator black_it,
                         std::vector<float>::iterator white_it) const;

    void FillMove(const Board* board,
                  std::vector<float>::iterator move_it) const;

    void EncoderHistoryMove(const GameState &state,
                            std::vector<float>::iterator it,
                            const int weights_version) const;

    void FillKoMove(const Board* board,
                    std::vector<float>::iterator ko_it) const;

    void FillArea(const Board* board,
                  const int to_move,
                  const int scoring,
                  std::vector<float>::iterator area_it,
                  const int weights_version) const;

    void FillLiberties(const Board* board,
                       std::vector<float>::iterator liberties_it) const;

    void FillLadder(const Board* board,
                    std::vector<float>::iterator ladder_it) const;

    void FillMisc(const Board* board,
                  const int to_move,
                  const int scoring, float wave, float komi,
                  std::vector<float>::iterator misc_it,
                  const int weights_version) const;

    void EncoderFeatures(const GameState &state,
                         std::vector<float>::iterator it,
                         const int weights_version) const;
};
