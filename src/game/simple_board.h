#pragma once

#include <array>
#include <vector>
#include <string>
#include <cassert>
#include <functional>
#include <algorithm>

#include "game/strings.h"
#include "game/types.h"
#include "game/zobrist.h"

class SimpleBoard {
public:
    void Reset(const int boardsize);
    void ResetBoard();
    void ResetBasicData();

    void SetBoardSize(int boardsize);
    void SetToMove(int color);
    void SetLastMove(int vertex);

    // Get the current board size.
    int GetBoardSize() const;

    // Get the current letter box size (board size + 2).
    int GetLetterBoxSize() const;

    // Get the current number of vertices (letter box).
    int GetNumVertices() const;

    // Get the current number of intersections.
    int GetNumIntersections() const;

    // Get the side to move color.
    int GetToMove() const;

    // Get the last move.
    int GetLastMove() const;

    // Get ko move if last move is ko move. Will reture null vertex
    // if last move is not ko move.
    int GetKoMove() const;

    // Number of accumulated pass move played.  Will reture 0 if last
    // move is not pass move.
    int GetPasses() const;

    // Get ko hash.
    std::uint64_t GetKoHash() const;

    // Get zobrist hash.
    std::uint64_t GetHash() const;

    // Get number of captured stones.
    int GetPrisoner(const int color) const;

    // Get board state on this vertex position
    int GetState(const int vtx) const;
    int GetState(const int x, const int y) const;

    // Get the number liberties of string.
    int GetLiberties(const int vtx) const;

    // Get the number stones of string.
    int GetStones(const int vtx) const;

    // Get the number of empty points. 
    int GetEmptyCount() const;

    // Empty index to vertex.
    int GetEmpty(const int idx) const;

    // Get all vertices of a string.
    std::vector<int> GetStringList(const int vtx) const;

    // The vertex to x.
    int GetX(const int vtx) const;

    // The vertex to y.
    int GetY(const int vtx) const;

    // Get the vertex position.
    int GetVertex(const int x, const int y) const;

    // Get the index board.
    int GetIndex(const int x, const int y) const;

    // Reture true if the move is legal.
    bool IsLegalMove(const int vertex, const int color) const;
    bool IsLegalMove(const int vertex, const int color,
                         std::function<bool(int, int)> AvoidToMove) const;

    // Return true if the move is self-atari. Notice that
    // it is not full implement. We do not consider the 
    // all libs from capture.
    bool IsSelfAtariMove(const int vtx, const int color) const;

    // Reture true if the move can atari others.
    bool IsAtariMove(const int vtx, const int color) const;

    // Reture true if the move can capture others.
    bool IsCaptureMove(const int vtx, const int color) const;

    bool IsEscapeMove(const int vtx, const int color) const;

    // Reture true if the two vertex are Neighbor.
    bool IsNeighbor(const int vtx, const int avtx) const;

    // Reture true if the string is ladder.
    bool IsLadder(const int vtx, std::vector<int> &vital_moves) const;

    // Reture true if the point is real eye.
    bool IsRealEye(const int vtx, const int color) const;

    // Reture true if the point is eye shape.
    bool IsSimpleEye(const int vtx, const int color) const;

    // Reture true if the point is seki.
    bool IsSeki(const int vtx) const;

    // Play the move assume the move is legal.
    void PlayMoveAssumeLegal(const int vtx, const int color);

    // Get the current board states and informations, for GTP showboard.
    std::string GetBoardString(const int last_move, bool y_invert) const;

    // Compute the symmetry Zobrist hashing.
    std::uint64_t ComputeSymmetryHash(int komove, int symmetry) const;

    // Compute the symmetry Zobrist ko hashing.
    std::uint64_t ComputeKoHash(int symmetry) const;

    std::uint64_t GetMoveHash(const int vtx, const int color) const;

    int ComputeReachGroup(int start_vertex, int spread_color,
                              std::vector<bool> &buf,
                              std::function<int(int)> Peek) const;
    int ComputeReachGroup(int start_vertex, int spread_color, std::vector<bool> &buf) const;


    // Remove the marked strings from board.
    void RemoveMarkedStrings(std::vector<int> &marked);


    // for patterns...
    bool MatchPattern3(const int vtx, const int color) const;
    std::uint64_t GetPatternHash(const int vtx, const int color, const int dist) const;
    std::uint64_t GetSurroundPatternHash(std::uint64_t hash,
                                             const int vtx,
                                             const int color,
                                             const int dist) const;
    bool GetBorderLevel(const int vtx, int &dist) const;
    bool GetDistLevel(const int vtx, int &dist) const;

protected:
    // Compute the Zobrist hashing.
    std::uint64_t ComputeHash(int komove = kNullVertex) const;

    // Compute the Zobrist ko hashing.
    std::uint64_t ComputeKoHash() const;

    int ComputeReachColor(int color) const;
    int ComputeReachColor(int color, int spread_color,
                          std::vector<bool> &buf,
                          std::function<int(int)> Peek) const;


    // The movement directions eight way.
    std::array<int, 8> directions_;

    // The board contents.
    std::array<VertexType, kNumVertices> state_;

    // The counts of neighboring stones.
    std::array<std::uint16_t, kNumVertices> neighbours_;

    // The empty intersections.
    std::array<std::uint16_t, kNumVertices> empty_;

    // The empty intersection indices.
    std::array<std::uint16_t, kNumVertices> empty_idx_;

    // The board strings.
    Strings strings_;

    // The Prisoners per color
    std::array<int, 2> prisoners_;

    // The Zobrist hash of board position.
    std::uint64_t hash_;

    // The Zobrist ko hash of board position.
    std::uint64_t ko_hash_;

    // The board size.
    int board_size_;

    // The letter box size (board size + 2).
    int letter_box_size_;

    // The vertices number.
    int num_vertices_;

    // The intersections number.
    int num_intersections_;

    // The next player to play move.
    int to_move_;

    // The last played move.
    int last_move_; 

    // Can't play this move if the last move is ko move.
    int ko_move_;

    // The count of empties.
    int empty_cnt_;

    // The accumulated passes number.
    int passes_;

    // The move number.
    int move_number_;

    // Reture true if the move will kill itself.
    bool IsSuicide(const int vtx, const int color) const;

    // Compute the empty count of the point neighbor.
    int CountPliberties(const int vtx) const;

    bool IsBoundary(const int vtx) const;
    bool IsKillableSekiEyeShape(const int vtx,
                                    const int eye_size,
                                    const std::vector<int> &eye_next) const;

    void FindStringSurround(const int vtx,
                                const int color,
                                std::vector<int>& lib_buf,
                                std::vector<int>& index_buf) const;

    // Find the liberties of string.
    int FindStringLiberties(const int vtx, std::vector<int>& buf) const;

    // Find what move can gain liberty by Capturing.
    int FindStringLibertiesGainingCaptures(const int vtx, std::vector<int>& buf) const;

    // Get the possible lowest and most liberties. 
    std::pair<int, int> GetLadderLiberties(const int vtx, const int color) const;

    // Find Prey's possible move for ladder searching.
    LadderType PreySelections(const int prey_color,
                              const int ladder_vtx,
                              std::vector<int>& selections, bool think_ko) const;

    // Find Hunter's possible move for ladder searching.
    LadderType HunterSelections(const int prey_color,
                                const int ladder_vtx, std::vector<int>& selections) const;

    // Prey do move to try to escape from hunter.
    LadderType PreyMove(SimpleBoard* board,
                        const int hunter_vtx, const int prey_color,
                        const int ladder_vtx, size_t& ladder_nodes, bool fork) const;

    // Hunter do move to try to capture the prey.
    LadderType HunterMove(SimpleBoard* board,
                          const int prey_vtx, const int prey_color,
                          const int ladder_vtx, size_t& ladder_nodes, bool fork) const;

private:
    // The Generally function compute the Zobrist hashing.
    std::uint64_t ComputeHash(int komove, std::function<int(int)> transform) const;

    // The Generally function compute the Zobrist ko hashing.
    std::uint64_t ComputeKoHash(std::function<int(int)> transform) const;

    // About to display the board information.
    bool IsStar(const int x, const int y) const;
    std::string GetStateString(const VertexType color, bool is_star) const;
    std::string GetSpacesString(const int times) const;
    std::string GetColumnsString(const int bsize) const;
    std::string GetPrisonersString() const;
    std::string GetHashingString() const;

    // Update the to move.
    void ExchangeToMove();

    // Add a stone to board.
    void AddStone(const int vtx, const int color);

    // Remove a stone from board.
    void RemoveStone(const int vtx, const int color);

    // Merge two strings.
    void MergeStrings(const int ip, const int aip);

    // Remove a string from board.
    int RemoveString(const int ip);

    void IncrementPrisoner(const int color, const int val);

    // Update the board after doing a legal move.
    int UpdateBoard(const int vtx, const int color);

    void SetPasses(int val);
    void IncrementPasses();

    // Update Zobrist key for board position.
    void UpdateZobrist(const int vtx, const int new_color, const int old_color);

    // Update Zobrist key for prisoner.
    void UpdateZobristPrisoner(const int color, const int new_pris,
                               const int old_pris);

    // Update Zobrist key for to move.
    void UpdateZobristToMove(const int new_color, const int old_color);

    // Update Zobrist key for ko move.
    void UpdateZobristKo(const int new_komove, const int old_komove);

    // Update Zobrist key for pass move.
    void UpdateZobristPass(const int new_pass, const int old_pass);
};

inline int SimpleBoard::GetX(const int vtx) const {
    const int x = (vtx % letter_box_size_) - 1;
    assert(x >= 0 && x < board_size_);
    return x;
}

inline int SimpleBoard::GetY(const int vtx) const {
    const int y = (vtx / letter_box_size_) - 1;
    assert(y >= 0 && y < board_size_);
    return y;
}

inline int SimpleBoard::GetVertex(const int x, const int y) const {
    assert(x >= 0 || x < board_size_);
    assert(y >= 0 || y < board_size_);
    return (y + 1) * letter_box_size_ + (x + 1);
}

inline int SimpleBoard::GetIndex(const int x, const int y) const {
    assert(x >= 0 || x < board_size_);
    assert(y >= 0 || y < board_size_);
    return y * board_size_ + x;
}

inline void SimpleBoard::UpdateZobrist(const int vtx,
                                           const int new_color,
                                           const int old_color) {
    hash_ ^= Zobrist::kState[old_color][vtx];
    hash_ ^= Zobrist::kState[new_color][vtx];
    ko_hash_ ^= Zobrist::kState[old_color][vtx];
    ko_hash_ ^= Zobrist::kState[new_color][vtx];
}

inline void SimpleBoard::UpdateZobristPrisoner(const int color,
                                                   const int new_pris,
                                                   const int old_pris) {
    hash_ ^= Zobrist::kPrisoner[color][old_pris];
    hash_ ^= Zobrist::kPrisoner[color][new_pris];
}

inline void SimpleBoard::UpdateZobristToMove(const int new_color,
                                                 const int old_color) {
    if (old_color != new_color) {
        hash_ ^= Zobrist::kBlackToMove;
    }
}

inline void SimpleBoard::UpdateZobristKo(const int new_komove,
                                             const int old_komove) {
    hash_ ^= Zobrist::kKoMove[old_komove];
    hash_ ^= Zobrist::kKoMove[new_komove];
}

inline void SimpleBoard::UpdateZobristPass(const int new_pass,
                                               const int old_pass) {
    hash_ ^= Zobrist::KPass[old_pass];
    hash_ ^= Zobrist::KPass[new_pass];
}

inline int SimpleBoard::GetPrisoner(const int color) const {
    return prisoners_[color];
}

inline int SimpleBoard::GetBoardSize() const {
    return board_size_;
}

inline int SimpleBoard::GetLetterBoxSize() const {
    return letter_box_size_;
}

inline int SimpleBoard::GetNumVertices() const {
    return num_vertices_;
}

inline int SimpleBoard::GetNumIntersections() const {
    return num_intersections_;
}

inline int SimpleBoard::GetToMove() const {
    return to_move_;
}

inline int SimpleBoard::GetLastMove() const {
    return last_move_;
}

inline int SimpleBoard::GetKoMove() const {
    return ko_move_;
}

inline int SimpleBoard::GetPasses() const {
    return passes_;
}

inline std::uint64_t SimpleBoard::GetKoHash() const {
    return ko_hash_;
}

inline std::uint64_t SimpleBoard::GetHash() const {
    return hash_;
}

inline std::uint64_t SimpleBoard::GetMoveHash(const int vtx, const int color) const {
    std::uint64_t hash = Zobrist::kState[color][vtx];
    if (color == to_move_) {
        hash ^= Zobrist::kBlackToMove; 
    }
    return hash;
}

inline int SimpleBoard::GetState(const int vtx) const {
    return state_[vtx];
}

inline int SimpleBoard::GetState(const int x, const int y) const {
    return GetState(GetVertex(x,y));
}

inline int SimpleBoard::GetLiberties(const int vtx) const {
    return strings_.GetLiberty(strings_.GetParent(vtx));
}

inline int SimpleBoard::GetStones(const int vtx) const {
    return strings_.GetStones(strings_.GetParent(vtx));
}

inline int SimpleBoard::GetEmptyCount() const {
    return empty_cnt_;
}

inline int SimpleBoard::GetEmpty(const int idx) const {
    return empty_[idx];
}
