#ifndef GAME_GAME_STATE_H_INCLUDE
#define GAME_GAME_STATE_H_INCLUDE

#include <vector>
#include <memory>
#include <string>

#include "game/board.h"


class GameState {
public:
    Board board_;

    void Reset(const int boardsize, const float komi);

    void ClearBoard();

    bool PlayMove(const int vtx, const int color);

    bool UndoMove();

    void SetKomi(float komi);

    int TextToVertex(std::string text);

    // GTP interface to play move.
    bool PlayTextMove(std::string input);

    // GTP interface to show board.
    void ShowBoard() const;

    // GTP interface to set fixed handicap.
    bool SetFixdHandicap(int handicap);

    // GTP interface to set free handicap.
    bool SetFreeHandicap(std::vector<std::string> movelist);

    std::vector<int> GetSimpleOwnership() const;
    float GetFinalScore(float bonus = 0) const;

    float GetKomi() const;
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

private:
    void SetHandicap(int handicap);

    std::string GetStateString() const;

    std::vector<std::shared_ptr<const Board>> game_history_;

    // The board handicap.
    int handicap_;

    // The komi integer part.
    int komi_integer_;

    // The komi floating-point part.
    float komi_float_;

    int winner_;
};


#endif
