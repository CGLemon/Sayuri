#pragma once

#include <array>
#include <vector>
#include <string>
#include <cassert>
#include <functional>
#include <algorithm>

#include "game/types.h"
#include "game/strings.h"
#include "game/types.h"
#include "game/zobrist.h"

class Board {
public:
    void Reset(const int boardsize);
    void ResetBoard();
    void ResetBasicData();

    void SetBoardSize(int boardsize);
    void SetToMove(int color);
    void SetMoveNumber(int number);
    void SetLastMove(int first_vtx, int second_vtx);

    float GetKomi() const;
    int GetMoveNumber() const;
    int GetBoardSize() const;
    int GetLetterBoxSize() const;
    int GetNumVertices() const;
    int GetNumIntersections() const;
    int GetToMove() const;
    int GetLastMove() const;
    int GetPasses() const;
    std::uint64_t GetHash() const;
    int GetState(const int vtx) const;
    int GetState(const int x, const int y) const;

    int GetX(const int vtx) const;
    int GetY(const int vtx) const;

    int GetEmptyCount() const;
    int GetEmpty(const int idx) const;

    // Get the vertex board move.
    int GetVertex(const int x, const int y) const;

    // Get the index board.
    int GetIndex(const int x, const int y) const;

    int IndexToVertex(int idx) const;
    int VertexToIndex(int vtx) const;

    // Reture true if the move is legal.
    bool IsLegalForPass(const int color) const;
    bool IsLegalMove(const int vertex, const int color) const;
    bool IsLegalMove(const int vertex, const int color,
                     std::function<bool(int, int)> AvoidToMove) const;
    // Play the move assume the move is legal.
    void PlayMoveAssumeLegal(const int vtx, const int color);

    // The GTP showboard.
    std::string GetBoardString(const int last_move, bool is_sgf) const;

    // Compute the symmetry Zobrist hashing.
    std::uint64_t ComputeSymmetryHash(int symmetry) const;

    void ExchangeToMove();

    // Reture the zobrist hash value for this move.
    std::uint64_t GetMoveHash(const int vtx, const int color) const;

    bool IsEdge(const int vtx) const;
    bool IsCorner(const int vtx) const;
    bool IsThreatPass(const int vtx, const int color) const;

    std::string GetMoveTypesString(int vtx, int color) const;

    // Compute final score with Tromp Taylor rule.
    float ComputeFinalScore(float komi) const;

    // Compute score on board(without komi) with Tromp Taylor rule.
    int ComputeScoreOnBoard() const;

    // Compute score area with Tromp Taylor rule.
    void ComputeScoreArea(std::vector<int> &result) const;

    bool IsCaptureMove(const int vtx, const int color) const;

    // For patterns...
    static void InitPattern3();
    std::uint16_t GetPattern3Hash(const int vtx) const;
    std::uint16_t GetSymmetryPattern3Hash(const int vtx,
                                              const int color,
                                              const int symmetry) const;
    bool MatchPattern3(const int vtx) const;
    std::string GetPatternSpat(const int vtx, const int color, const int dist) const;
    std::uint64_t GetPatternHash(const int vtx, const int color, const int dist) const;
    std::uint64_t GetSymmetryPatternHash(const int vtx, const int color,
                                             const int dist, const int symmetry) const;
    std::uint64_t GetSurroundPatternHash(std::uint64_t hash,
                                             const int vtx,
                                             const int color,
                                             const int dist) const;

    // For board features...
    bool GetBorderLevel(const int vtx, const int color, std::uint64_t &hash) const;
    bool GetDistLevel(const int vtx, const int color, std::uint64_t &hash) const;
    bool GetDistLevel2(const int vtx, const int color, std::uint64_t &hash) const;
    bool GetCapureLevel(const int vtx, const int color, std::uint64_t &hash) const;
    bool GetAtariLevel(const int vtx, const int color, std::uint64_t &hash) const;
    bool GetSelfAtariLevel(const int vtx, const int color, std::uint64_t &hash) const;
    bool GetFeatureWrapper(const int f, const int vtx, const int color, std::uint64_t &hash) const;
    static int GetMaxFeatures();

private:
    // Compute the Zobrist hashing.
    std::uint64_t ComputeHash() const;

    // The movement directions eight way.
    std::array<int, 8> directions_;

    // The board contents.
    std::array<VertexType, kNumVertices> state_;

    // The counts of neighboring stones.
    std::array<std::uint16_t, kNumVertices> neighbours_;

    // The Zobrist hash of board position.
    std::uint64_t hash_;

    // The board size.
    int board_size_;

    // The letter box size.
    int letter_box_size_;

    // The vertices number.
    int num_vertices_;

    // The intersections number.
    int num_intersections_;

    // The next player to play move.
    int to_move_;

    // The last played move.
    int last_move_;

    // My last played move.
    int last_move_2_;

    // The passes number.
    int passes_;

    // The move number.
    int move_number_;

    int ComputeStoneCount(const int color) const;

    // The Generally function compute the Zobrist hashing.
    std::uint64_t ComputeHash(std::function<int(int)> transform) const;

    // About to display the board information.
    bool IsStar(const int x, const int y) const;

    std::string GetStoneCountString() const;
    std::string GetStateString(const VertexType color, bool is_star) const;
    std::string GetSpcacesString(const int times) const;
    std::string GetColumnsString(const int bsize) const;
    std::string GetHashingString() const;

    // Update the board after do a move.
    void UpdateBoard(const int vtx, const int color);
    void UpdateStone(const int vtx, const int color);

    void SetPasses(int val);
    void IncrementPasses();

    // Update Zobrist key for board position.
    void UpdateZobrist(const int vtx, const int new_color, const int old_color);

    // Update Zobrist key for to move.
    void UpdateZobristToMove(const int new_color, const int old_color);


    // Update Zobrist key for pass move.
    void UpdateZobristPass(const int new_pass, const int old_pass);
};


inline int Board::GetX(const int vtx) const {
    const int x = (vtx % letter_box_size_) - 1;
    assert(x >= 0 && x < board_size_);
    return x;
}

inline int Board::GetY(const int vtx) const {
    const int y = (vtx / letter_box_size_) - 1;
    assert(y >= 0 && y < board_size_);
    return y;
}

inline int Board::GetVertex(const int x, const int y) const {
    assert(x >= 0 || x < board_size_);
    assert(y >= 0 || y < board_size_);
    return (y + 1) * letter_box_size_ + (x + 1);
}

inline int Board::GetIndex(const int x, const int y) const {
    assert(x >= 0 || x < board_size_);
    assert(y >= 0 || y < board_size_);
    return y * board_size_ + x;
}

inline int Board::IndexToVertex(int idx) const {
    int x = idx % board_size_;
    int y = idx / board_size_;

    assert(x >= 0 || x < board_size_);
    assert(y >= 0 || y < board_size_);
    return GetVertex(x, y);
}

inline int Board::VertexToIndex(int vtx) const {
    int x = GetX(vtx);
    int y = GetY(vtx);
    return GetIndex(x, y);
}

inline void Board::UpdateZobrist(const int vtx,
                                 const int new_color,
                                 const int old_color) {
    hash_ ^= Zobrist::kState[old_color][vtx];
    hash_ ^= Zobrist::kState[new_color][vtx];
}

inline void Board::UpdateZobristToMove(const int new_color,
                                       const int old_color) {
    if (old_color != new_color) {
        hash_ ^= Zobrist::kBlackToMove;
    }
}

inline void Board::UpdateZobristPass(const int new_pass,
                                     const int old_pass) {
    hash_ ^= Zobrist::KPass[old_pass];
    hash_ ^= Zobrist::KPass[new_pass];
}

inline int Board::GetBoardSize() const {
    return board_size_;
}

inline int Board::GetLetterBoxSize() const {
    return letter_box_size_;
}

inline int Board::GetNumVertices() const {
    return num_vertices_;
}

inline int Board::GetNumIntersections() const {
    return num_intersections_;
}

inline int Board::GetToMove() const {
    return to_move_;
}

inline int Board::GetLastMove() const {
    return last_move_;
}

inline int Board::GetPasses() const {
    return passes_;
}

inline std::uint64_t Board::GetHash() const {
    return hash_;
}

inline int Board::GetState(const int vtx) const {
    return state_[vtx];
}

inline int Board::GetState(const int x, const int y) const {
    return GetState(GetVertex(x,y));
}

inline std::uint64_t Board::GetMoveHash(const int vtx, const int color) const {
    std::uint64_t hash = Zobrist::kState[color][vtx];
    if (color == to_move_) {
        hash ^= Zobrist::kBlackToMove; 
    }
    return hash;
}
