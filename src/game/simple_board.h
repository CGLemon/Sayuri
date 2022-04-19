#pragma once

#include <array>
#include <vector>
#include <string>
#include <cassert>
#include <memory>
#include <functional>
#include <algorithm>

#include "game/strings.h"
#include "game/types.h"
#include "game/zobrist.h"
#include "pattern/pattern.h"

class SimpleBoard {
public:
    void Reset(const int boardsize);
    void ResetBoard();
    void ResetBasicData();

    void SetBoardSize(int boardsize);
    void SetToMove(int color);
    void SetLastMove(int vertex);

    int GetBoardSize() const;
    int GetLetterBoxSize() const;
    int GetNumVertices() const;
    int GetNumIntersections() const;
    int GetToMove() const;
    int GetLastMove() const;
    int GetKoMove() const;
    int GetPasses() const;
    std::uint64_t GetKoHash() const;
    std::uint64_t GetHash() const;
    int GetPrisoner(const int color) const;
    int GetState(const int vtx) const;
    int GetState(const int x, const int y) const;
    int GetLiberties(const int vtx) const;
    int GetStones(const int vtx) const;
    int GetEmptyCount() const;
    int GetEmpty(const int idx) const;
    std::vector<int> GetStringList(const int vtx) const;

    int GetX(const int vtx) const;
    int GetY(const int vtx) const;

    // Get the vertex board move.
    int GetVertex(const int x, const int y) const;

    // Get the index board.
    int GetIndex(const int x, const int y) const;

    // Reture true if the move is legal.
    bool IsLegalMove(const int vertex, const int color) const;
    bool IsLegalMove(const int vertex, const int color,
                     std::function<bool(int, int)> AvoidToMove) const;

    // Reture true if the move can atari others.
    bool IsAtariMove(const int vtx, const int color) const;

    // Reture true if the move can capture others.
    bool IsCaptureMove(const int vtx, const int color) const;

    bool IsEscapeMove(const int vtx, const int color) const;

    // Reture true if the two vertex are Neighbor.
    bool IsNeighbor(const int vtx, const int avtx) const;

    // Reture true if the string is ladder.
    bool IsLadder(const int vtx, std::vector<int> &vital_moves) const;

    // Reture true if the point is read eye.
    bool IsRealEye(const int vtx, const int color) const;

    // Reture true if the point is eye shape.
    bool IsSimpleEye(const int vtx, const int color) const;

    // Reture true if the point is seki.
    bool IsSeki(const int vtx) const;

    // Play the move assume the move is legal.
    void PlayMoveAssumeLegal(const int vtx, const int color);

    // The GTP showboard.
    std::string GetBoardString(const int last_move, bool is_sgf) const;

    Pattern GetPattern3x3(const int vtx, const int color) const;

    // Compute the symmetry Zobrist hashing.
    std::uint64_t ComputeSymmetryHash(int komove, int symmetry) const;

    // Compute the symmetry Zobrist ko hashing.
    std::uint64_t ComputeKoHash(int symmetry) const;

    int ComputeReachGroup(int start_vertex, int spread_color,
                          std::vector<bool> &buf,
                          std::function<int(int)> Peek) const;
    int ComputeReachGroup(int start_vertex, int spread_color, std::vector<bool> &buf) const;


    void RemoveMarkedStrings(std::vector<int> &marked);

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

    // Can't play this move if the last move is ko move.
    int ko_move_;

    // The count of empties.
    int empty_cnt_;

    // The passes number.
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
    LadderType PreyMove(std::shared_ptr<SimpleBoard> board,
                        const int hunter_vtx, const int prey_color,
                        const int ladder_vtx, size_t& ladder_nodes, bool fork) const;

    // Hunter do move to try to capture the prey.
    LadderType HunterMove(std::shared_ptr<SimpleBoard> board,
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
    std::string GetSpcacesString(const int times) const;
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

    // Update the board after do a move.
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
