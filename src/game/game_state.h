#pragma once

#include <vector>
#include <memory>
#include <string>
#include <cstdint>

#include "game/board.h"

class GameState {
public:
    Board board_;

    // Reset all data and clear the board.
    void Reset(const int boardsize,
               const float komi,
               const int scoring);

    // GTP interface to reset the boardsize.
    void SetBoardSize(const int boardsize);

    // GTP interface to clear the board.
    void ClearBoard();

    // Play a move on the board but use current game state instead
    // of first game state in the history buffer. Will return false
    // if the move is illegal.
    bool AppendMove(const int vtx, const int color);

    // Same as PlayMove(vtx, color) but color is always tomove.
    bool PlayMove(const int vtx);

    // Play a move on board and push current game state to the history
    // buffer. Will return false if the move is illegal.
    bool PlayMove(const int vtx, const int color);

    // Play undo move and remove last game state of the history buffer.
    bool UndoMove();

    // Set komi.
    void SetKomi(float komi);

    // Set side to move color.
    void SetToMove(const int color);

    // Set winner.
    void SetWinner(GameResult result);

    // Set scoring rule.
    void SetRule(const int scoring);

    // Set the territory helper. The element of vector should be one
    // of kBlack/kWhite/kEmpty.
    void SetTerritoryHelper(const std::vector<int> &ownership);

    // Transfer GTP vertex string to vertex numeric.
    int TextToVertex(std::string text) const;

    // Transfer GTP color string to color numeric.
    int TextToColor(std::string text) const;

    // Transfer vertex numeric to SGF verterx string.
    std::string VertexToSgf(const int vtx) const;

    // Transfer vertex numeric to  GTP vertex string.
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

    // Set number of handicap stones.
    void SetHandicap(int handicap);

    // Place the handicap stone one the board. All Handicap() functions
    // should play the stones via this function.
    bool PlayHandicapStones(std::vector<int> movelist_vertex,
                            bool kata_like_handicap_style);

    // Compute final score based on current scoring rule.
    float GetFinalScore(const int color) const;
    float GetFinalScore(const int color,
                        const std::vector<int> &territory_helper) const;

    // The safe area means both players do not need to play move in
    // it. It can be efficiently to end the a game if someone refuses
    // to play pass (or resign).
    std::vector<bool> GetStrictSafeArea() const;

    // Return true if the game is finished.
    bool IsGameOver() const;

    // Return true if current position is super ko, repetition position.
    bool IsSuperko() const;

    // Return true if the move is legal.
    bool IsLegalMove(const int vertex) const;
    bool IsLegalMove(const int vertex, const int color) const;
    bool IsLegalMove(const int vertex, const int color,
                     std::function<bool(int, int)> AvoidToMove) const;

    // Reture true if the specified color is adjacent to this vertex.
    bool IsNeighborColor(const int vtx, const int color) const;

    // Reture true if the move is in the seki.
    bool IsSeki(const int vtx) const;

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

    // Return the scoring penalty for black side.
    float GetPenalty() const;
    float GetPenalty(int scoring_rule) const;

    // Return the offset penalty if we want to change the scoring rule in any
    // time. We hope the final score of new rule is as same as old.
    float GetPenaltyOffset(int new_scoring_rule, int old_scoring_rule) const;

    // Return komi + black penalty (bonus) points.
    float GetKomiWithPenalty() const;

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

    // Reture the zobrist hash value only for this move.
    std::uint64_t GetMoveHash(const int vtx, const int color) const;

    void SetComment(std::string c);
    void RewriteComment(std::string c, size_t i);
    std::string GetComment(size_t i) const;

    float GetWave() const;
    std::string GetRuleString() const;

private:
    using VertexColor = std::pair<int, int>;

    // Play a move without pushing current game state to the history buffer.
    void PlayMoveFast(const int vtx, const int color);

    void PushComment();

    std::string GetStateString() const;

    std::vector<std::shared_ptr<const Board>> game_history_;

    std::vector<std::uint64_t> ko_hash_history_;

    std::vector<VertexColor> append_moves_;

    std::vector<std::string> comments_;

    // The territory helper helps us to remove the dead stones for scoring.
    // final position.
    std::vector<int> territory_helper_;

    // Current scoring rule.
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
