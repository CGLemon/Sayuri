#ifndef GAME_SIMPLE_BOARD_H_INCLUDE
#define GAME_SIMPLE_BOARD_H_INCLUDE

#include <array>
#include <vector>
#include <string>
#include <cassert>
#include <functional>

#include "game/string.h"
#include "game/types.h"
#include "game/zobrist.h"

class SimpleBoard {
public:
    void Reset(const int boardsize);
    void ResetBoard();
    void ResetBasicData();

    void SetBoardSize(int boardsize);
    void SetToMove(int color);
    void SetMoveNumber(int number);
    void SetLastMove(int vertex);

    float GetKomi() const;
    int GetMoveNumber() const;
    int GetBoardSize() const;
    int GetLetterBoxSize() const;
    int GetNumVertices() const;
    int GetNumIntersections() const;
    int GetToMove() const;
    int GetLastMove() const;
    int GetKoMove() const;
    int GetPasses() const;
    int GetKoHash() const;
    int GetHash() const;
    int GetPrisoner(const int color) const;
    int GetState(const int vtx) const;
    int GetState(const int x, const int y) const;

    // Get the vertex board.
    int GetVertex(const int x, const int y) const;

    // Get the index board.
    int GetIndex(const int x, const int y) const;

    // Reture true if the move is legal.
    bool IsLegalMove(const int vertex, const int color) const;
    bool IsLegalMove(const int vertex, const int color,
                     std::function<bool(int, int)> AvoidToMove) const;

    // Play the move assume the move is legal.
    void PlayMoveAssumeLegal(const int vtx, const int color);

    // The GTP showboard.
    std::string GetBoardString(const int last_move, bool is_sgf) const;

    // Compute the symmetry Zobrist hashing.
    std::uint64_t ComputeSymmetryHash(int komove, int symmetry) const;

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
    String strings_;

    // The Prisoners per color
    std::array<int, 2> prisoners_;

    // The Zobrist hash of board.
    std::uint64_t hash_;

    // The Zobrist ko hash of board.
    std::uint64_t ko_hash_;

    int board_size_; 
    int letter_box_size_;
    int num_vertices_;
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

    bool IsSimpleEye(const int vtx, const int color) const;
    bool IsSuicide(const int vtx, const int color) const;
    int CountPliberties(const int vtx) const;

private:
    // The Generally function compute the Zobrist hashing.
    std::uint64_t ComputeHash(int komove, std::function<int(int)> transform) const;

    // The Generally function compute the Zobrist ko hashing.
    std::uint64_t ComputeKoHash(std::function<int(int)> transform) const;

    // About Board infomation.
    bool IsStar(const int x, const int y) const;
    std::string GetStateString(const VertexType color, bool is_star) const;
    std::string GetSpcacesString(const int times) const;
    std::string GetColumnsString(const int bsize) const;
    std::string GetPrisonersString() const;

    // About to update the board.
    void ExchangeToMove();
    void AddStone(const int vtx, const int color);
    void RemoveStone(const int vtx, const int color);
    void MergeStrings(const int ip, const int aip);
    int RemoveString(const int ip);
    int UpdateBoard(const int vtx, const int color);
    void SetPasses(int val);
    void IncrementPasses();

    // About to update the Zobrist hashing.
    void UpdateZobrist(const int vtx, const int new_color, const int old_color);
    void UpdateZobristPrisoner(const int color, const int new_pris,
                               const int old_pris);
    void UpdateZobristToMove(const int new_color, const int old_color);
    void UpdateZobristKo(const int new_komove, const int old_komove);
    void UpdateZobristPass(const int new_pass, const int old_pass);
};

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
#endif
