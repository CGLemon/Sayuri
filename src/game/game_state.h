#pragma once

#include <vector>
#include <memory>
#include <string>
#include <cstdint>

#include "game/board.h"

class GameState {
public:
    Board board_;

    void Reset(const int boardsize,
               const float komi,
               const int scoring);

    // GTP interface to reset the boardsize.
    void SetBoardSize(const int boardsize);

    // GTP interface to clear the board.
    void ClearBoard();

    bool AppendMove(const int vtx, const int color);

    bool PlayMove(const int vtx);

    bool PlayMove(const int vtx, const int color);

    bool UndoMove();

    void SetKomi(float komi);

    void SetToMove(const int color);

    void SetWinner(GameResult result);

    void SetRule(const int scoring);

    void SetTerritoryHelper(const std::vector<int> &ownership);

    int TextToVertex(std::string text) const;

    int TextToColor(std::string text) const;

    std::string VertexToSgf(const int vtx) const;

    std::string VertexToText(const int vtx) const;

    // For debug...
    void ShowMoveTypes(int vtx, int color) const;

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

    // Place the handicap stone one the board. All Handicap() functions
    // should play the stones via this function.
    bool PlayHandicapStones(std::vector<int> movelist_vertex,
                            bool kata_like_handicap_style);

    // Compute final score based on Tromp Taylor rule.
    float GetFinalScore(const int color) const;
    float GetFinalScore(const int color,
                        const std::vector<int> &territory_helper);

    // The safe area means both players do not need to play move in
    // it. It can be efficiently to end the a game if someone refuses
    // to play pass (or resign).
    std::vector<bool> GetStrictSafeArea() const;

    bool IsGameOver() const;
    bool IsSuperko() const;
    bool IsLegalMove(const int vertex) const;
    bool IsLegalMove(const int vertex, const int color) const;
    bool IsLegalMove(const int vertex, const int color,
                     std::function<bool(int, int)> AvoidToMove) const;
    bool IsNeighborColor(const int vtx, const int color) const;

    // Compute ownership based on Tromp Taylor rule.
    std::vector<int> GetOwnership() const;

    // Compute ownership based on Tromp Taylor rule but without
    // pass-dead and pass-alive area.
    std::vector<int> GetRawOwnership() const;

    // Remove the strings which is in the list. It will remove whole string
    // if one vertex is in the dead list.
    void RemoveDeadStrings(std::vector<int> &dead_list);

    int GetVertex(const int x, const int y) const;
    int GetIndex(const int x, const int y) const;
    int GetX(const int vtx) const;
    int GetY(const int vtx) const;
    int IndexToVertex(int idx) const;
    int VertexToIndex(int vtx) const;

    float GetKomiWithPenalty() const;
    float GetPenalty() const;
    float GetKomi() const;
    int GetWinner() const;
    int GetHandicap() const;
    int GetMoveNumber() const;
    int GetBoardSize() const;
    int GetNumIntersections() const;
    int GetNumVertices() const;
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
    std::vector<int> GetStringList(const int vtx) const;
    int GetScoringRule() const;

    std::uint64_t ComputeSymmetryHash(const int symm) const;
    std::uint64_t ComputeSymmetryKoHash(const int symm) const;

    std::vector<int> GetAppendMoves(int color) const;
    std::shared_ptr<const Board> GetPastBoard(unsigned int p) const;
    const std::vector<std::shared_ptr<const Board>>& GetHistory() const;

    void PlayRandomMove();
    float GetGammaValue(const int vtx, const int color) const;
    std::vector<float> GetGammasPolicy(const int color) const;

    std::uint64_t GetMoveHash(const int vtx, const int color) const;

    void SetComment(std::string c);
    void RewriteComment(std::string c, size_t i);
    std::string GetComment(size_t i) const;

    float GetWave() const;
    std::string GetRuleString() const;

private:
    using VertexColor = std::pair<int, int>;

    // Play the move without pushing current board to the history.
    void PlayMoveFast(const int vtx, const int color);

    void PushComment();

    std::string GetStateString() const;

    std::vector<std::shared_ptr<const Board>> game_history_;

    std::vector<std::uint64_t> ko_hash_history_;

    std::vector<VertexColor> append_moves_;

    std::vector<std::string> comments_;

    std::vector<int> territory_helper_;

    ScoringRuleType scoring_rule_;

    // Comment for next move.
    std::string last_comment_;

    // The board handicap.
    int handicap_;

    // The komi integer part.
    int komi_integer_;

    // The half komi part.
    bool komi_half_;

    // True if the current komi is negtive.
    bool komi_negative_;

    int move_number_;

    std::uint64_t komi_hash_;

    std::uint64_t scoring_hash_;

    int winner_;
};
