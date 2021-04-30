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

    void SetColor(const int color);

    void SetWinner(GameResult result);

    int TextToVertex(std::string text);

    std::string VertexToSgf(const int vtx);

    // GTP interface to play move.
    bool PlayTextMove(std::string input);

    // GTP interface to show board.
    void ShowBoard() const;

    // GTP interface to set fixed handicap.
    bool SetFixdHandicap(int handicap);

    // GTP interface to set free handicap.
    bool SetFreeHandicap(std::vector<std::string> movelist);

    float GetFinalScore(float bonus = 0) const;

    bool IsSuperko() const;
    bool IsLegalMove(const int vertex, const int color) const;
    bool IsLegalMove(const int vertex, const int color,
                     std::function<bool(int, int)> AvoidToMove) const;

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

    std::shared_ptr<const Board> GetPastBoard(unsigned int p) const;
    const std::vector<std::shared_ptr<const Board>>& GetHistory() const;

private:
    void SetHandicap(int handicap);

    std::string GetStateString() const;

    std::vector<std::shared_ptr<const Board>> game_history_;

    std::vector<std::uint64_t> ko_hash_history_;

    // The board handicap.
    int handicap_;

    // The komi integer part.
    int komi_integer_;

    // The komi floating-point part.
    float komi_float_;

    int winner_;
};
