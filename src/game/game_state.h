#pragma once

#include <vector>
#include <memory>
#include <string>
#include <cstdint>

#include "game/board.h"

class GameState {
public:
    Board board_;

    void Reset(const int boardsize, const float komi);

    // GTP interface to clear the board.
    void ClearBoard();

    bool PlayMove(const int vtx);

    bool PlayMove(const int vtx, const int color);

    bool UndoMove();

    void SetKomi(float komi);

    void SetToMove(const int color);

    void SetWinner(GameResult result);

    void SetFinalScore(float score);

    int TextToVertex(std::string text);

    std::string VertexToSgf(const int vtx);

    std::string VertexToText(const int vtx);

    // GTP interface to play move.
    bool PlayTextMove(std::string input);

    // GTP interface to show board.
    void ShowBoard() const;

    // GTP interface to set fixed handicap.
    bool SetFixdHandicap(int handicap);

    // GTP interface to set free handicap.
    bool SetFreeHandicap(std::vector<std::string> movelist);

    // GTP interface to place free handicap.
    std::vector<int> PlaceFreeHandicap(int handicap);

    void SetHandicap(int handicap);

    // Compute final score by Tromp Taylor
    float GetFinalScore(float bonus = 0) const;

    std::vector<bool> GetStrictSafeArea() const;

    bool IsGameOver() const;
    bool IsSuperko() const;
    bool IsLegalMove(const int vertex) const;
    bool IsLegalMove(const int vertex, const int color) const;
    bool IsLegalMove(const int vertex, const int color,
                     std::function<bool(int, int)> AvoidToMove) const;

    // Compute ownership by Tromp Taylor.
    std::vector<int> GetOwnership() const;

    std::vector<int> GetOwnershipAndRemovedDeadStrings(int playouts) const;
    std::vector<int> MarKDeadStrings(int playouts) const;
    void RemoveDeadStrings(int playouts);

    int GetVertex(const int x, const int y) const;
    int GetIndex(const int x, const int y) const;
    int GetX(const int vtx) const;
    int GetY(const int vtx) const;

    float GetKomi() const;
    int GetWinner() const;
    int GetHandicap() const;
    int GetMoveNumber() const;
    int GetBoardSize() const;
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
    int GetLiberties(const int vtx) const;
    std::vector<int> GetStringList(const int vtx) const;

    std::shared_ptr<const Board> GetPastBoard(unsigned int p) const;
    const std::vector<std::shared_ptr<const Board>>& GetHistory() const;

    float black_score_;

private:
    void PlayMoveFast(const int vtx, const int color);
    void FillRandomMove();
    void PlayRandomMove();

    std::string GetStateString() const;

    std::vector<std::shared_ptr<const Board>> game_history_;

    std::vector<std::uint64_t> ko_hash_history_;

    // The board handicap.
    int handicap_;

    // The komi integer part.
    int komi_integer_;

    // The half komi part.
    bool komi_half_;

    bool komi_negative_;

    std::uint64_t komi_hash_;

    int winner_;
};
