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
    // Reset the board size and clear the board.
    void Reset(const int boardsize);

    // Set the board size. Will adjust it if the setting board size
    // is greater than max size.
    void SetBoardSize(int boardsize);

    // Set to move color.
    void SetToMove(int color);

    // Set last move.
    void SetLastMove(int first_vtx, int second_vtx);

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
    //
    // TODO: There are some bug in this function. The program can not
    //       find ladder in following game.
    // 
    // (;GM[1]FF[4]CA[UTF-8]AP[Sabaki:0.52.0]KM[7.5]SZ[19]DT[2022-09-24]PB[Sayuri-test]PW[leela-0.11]
    // SBKV[44.48];
    // B[qd];W[dd]SBKV[44.3];B[pq];W[dp]SBKV[45.19];B[nc];W[po]SBKV[44.63];B[qo];W[qn]SBKV[44.16];
    // B[qp];W[pm]SBKV[44.9];B[on];W[pn]SBKV[45.42];B[np];W[qi]SBKV[45.73];B[cc];W[cd]SBKV[44.92];B[dc];W[ec]SBKV[45.8];
    // B[eb];W[fc]SBKV[47.24];B[fb];W[gc]SBKV[48.25];B[cq];W[cp]SBKV[48.32];B[dq];W[eq]SBKV[49.01];B[er];W[fq]SBKV[51.16];
    // B[fr];W[gq]SBKV[51.93];B[dj];W[cl]SBKV[52.56];B[gb];W[hc]SBKV[53.24];B[gr];W[hq]SBKV[53.94];B[dg];W[bd]SBKV[51.55];
    // B[bc];W[fg]SBKV[52.63];B[fi];W[qf]SBKV[55.19];B[dl];W[dm]SBKV[56.23];B[em];W[dn]SBKV[59.82];B[ef];W[ff]SBKV[70.28];
    // B[ee];W[ed]SBKV[72.56];B[fe];W[ge]SBKV[73.33];B[gf];W[hf]SBKV[79.56];B[gg];W[gh]SBKV[79.29];B[hg];W[ig]SBKV[87.79];
    // B[hh];W[hi]SBKV[93.19];B[ih];W[jh]SBKV[93.45];B[ii];W[ij]SBKV[96.62];B[ji];W[ki]SBKV[97.36];B[jj];W[jk]SBKV[98.04];
    // B[kj];W[lj]SBKV[98.52];B[kk];W[kl]SBKV[49.1];B[lk];W[mk]SBKV[19.13];B[ll];W[lm]SBKV[8.06];B[ml];W[nl]SBKV[6.03];
    // B[mm];W[mn]SBKV[3.2];B[nm];W[om]SBKV[1.35];B[nn];W[no]SBKV[1.21];B[bl];W[oo]SBKV[0.14];B[ck];W[oq]SBKV[0.15];B[or])
    bool IsLadder(const int vtx, std::vector<int> &vital_moves) const;

    // Reture true if the point is real eye.
    bool IsRealEye(const int vtx, const int color) const;

    // Reture true if the point is eye shape.
    bool IsSimpleEye(const int vtx, const int color) const;

    // Reture true if the empty point is in the seki. This function is not always
    // correct. Some cases, like Hanezeki, will be missed. You can watch the webside
    // to get the more seki information.
    // https://senseis.xmp.net/?Seki
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


    // Compute final score with Tromp Taylor rule.
    float ComputeFinalScore(float komi) const;

    // Compute score on board(without komi) with Tromp Taylor rule.
    int ComputeScoreOnBoard() const;

    // Compute score area with Tromp Taylor rule.
    void ComputeScoreArea(std::vector<int> &result) const;

    // Get the ladder type map.
    // LadderType::kLadderDeath means that the ladder string is already death.
    // LadderType::kLadderEscapable means that the ladder string has a chance to escape.
    // LadderType::kLadderTake means that someone can take the opponent's ladder strings.
    // LadderType::kLadderAtari means that someone can atari opponent's ladder string.
    std::vector<LadderType> GetLadderMap() const;

    // Compute all safe area which both players don't need to play the move. Mark
    // all empty points seki if 'mark_seki' is true.
    void ComputeSafeArea(std::vector<bool> &result, bool mark_seki) const;

    // Compute the empty area in the Seki.
    void ComputeSekiPoints(std::vector<bool> &result) const;

    std::string GetMoveTypesString(int vtx, int color) const;

    void GenerateCandidateMoves(std::vector<int> &moves_set, int color) const;

    // For patterns...
    bool MatchPattern3(const int vtx, const int color) const;
    std::string GetPatternSpat(const int vtx, const int color, const int dist) const;
    std::uint64_t GetPatternHash(const int vtx, const int color, const int dist) const;
    std::uint64_t GetSymmetryPatternHash(const int vtx, const int color,
                                             const int dist, const int symmetry) const;
    std::uint64_t GetSurroundPatternHash(std::uint64_t hash,
                                             const int vtx,
                                             const int color,
                                             const int dist) const;

    // For board features...
    bool GetDummyLevel(const int vtx, std::uint64_t &hash) const;
    bool GetBorderLevel(const int vtx, std::uint64_t &hash) const;
    bool GetDistLevel(const int vtx, std::uint64_t &hash) const;
    bool GetDistLevel2(const int vtx, std::uint64_t &hash) const;
    bool GetFeatureWrapper(const int f, const int vtx, std::uint64_t &hash) const;
    static int GetMaxFeatures();

private:
    void ResetBoard();
    void ResetBasicData();

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

    // My last played move.
    int last_move_2_; 

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

    // Return true if the vertex surrounds border.
    bool IsBorder(const int vtx) const;

    // Return true if it is killable eyeshape in the seki.
    bool IsKillableSekiEyeShape(const int vtx,
                                    const int eye_size,
                                    const std::vector<int> &eye_next) const;

    // Find the liberties of the string and the surrounded strings.
    void FindStringSurround(const int vtx,
                                const int color,
                                std::vector<int>& lib_buf,
                                std::vector<int>& index_buf) const;

    // Find the liberties of the string.
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
    LadderType PreyMove(Board* board,
                        const int hunter_vtx, const int prey_color,
                        const int ladder_vtx, size_t& ladder_nodes, bool fork) const;

    // Hunter do move to try to capture the prey.
    LadderType HunterMove(Board* board,
                          const int prey_vtx, const int prey_color,
                          const int ladder_vtx, size_t& ladder_nodes, bool fork) const;

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

    // Compute all pass-alive string.
    // Mark all vital regions of pass-alive string if 'mark_vitals' is true.
    // Mark all pass-dead regions if 'mark_pass_dead' is true.
    void ComputePassAliveArea(std::vector<bool> &result,
                                  const int color,
                                  bool mark_vitals,
                                  bool mark_pass_dead) const;

    // Compute black area and white area.
    void ComputeReachArea(std::vector<int> &result) const;

    // Reture true if string is pass-alive (unconditional life).
    bool IsPassAliveString(const int vertex,
                               bool allow_sucide,
                               const std::vector<bool> &vitals,
                               const std::vector<int> &features,
                               const std::vector<int> &regions_index,
                               const std::vector<int> &regions_next,
                               const std::vector<int> &strings_index,
                               const std::vector<int> &strings_next) const;

    // Reture true if string is pass-dead (unconditional death).
    bool IsPassDeadRegion(const int vertex,
                               const int color,
                               bool allow_sucide,
                               std::vector<int> &features,
                               const std::vector<int> &regions_next) const;

    // Gather the vertex if it is true on the buffer.
    std::vector<int> GatherVertices(std::vector<bool> &buf) const;

    // The 'target' is the string type. We will split all strings (groups) then
    // store the string (group) index in the 'regions_index' and store next
    // vertex postion in the 'regions_next'. Becare that the string (group) index
    // is from 1.
    std::vector<int> ClassifyGroups(const int target,
                                        std::vector<int> &features,
                                        std::vector<int> &regions_index,
                                        std::vector<int> &regions_next) const;

    void ComputeInnerRegions(const int vtx,
                                 const int color,
                                 const std::vector<int> &regions_next,
                                 std::vector<bool> &inner_regions) const;
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

inline void Board::UpdateZobrist(const int vtx,
                                     const int new_color,
                                     const int old_color) {
    hash_ ^= Zobrist::kState[old_color][vtx];
    hash_ ^= Zobrist::kState[new_color][vtx];
    ko_hash_ ^= Zobrist::kState[old_color][vtx];
    ko_hash_ ^= Zobrist::kState[new_color][vtx];
}

inline void Board::UpdateZobristPrisoner(const int color,
                                             const int new_pris,
                                             const int old_pris) {
    hash_ ^= Zobrist::kPrisoner[color][old_pris];
    hash_ ^= Zobrist::kPrisoner[color][new_pris];
}

inline void Board::UpdateZobristToMove(const int new_color,
                                           const int old_color) {
    if (old_color != new_color) {
        hash_ ^= Zobrist::kBlackToMove;
    }
}

inline void Board::UpdateZobristKo(const int new_komove,
                                       const int old_komove) {
    hash_ ^= Zobrist::kKoMove[old_komove];
    hash_ ^= Zobrist::kKoMove[new_komove];
}

inline void Board::UpdateZobristPass(const int new_pass,
                                         const int old_pass) {
    hash_ ^= Zobrist::KPass[old_pass];
    hash_ ^= Zobrist::KPass[new_pass];
}

inline int Board::GetPrisoner(const int color) const {
    return prisoners_[color];
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

inline int Board::GetKoMove() const {
    return ko_move_;
}

inline int Board::GetPasses() const {
    return passes_;
}

inline std::uint64_t Board::GetKoHash() const {
    return ko_hash_;
}

inline std::uint64_t Board::GetHash() const {
    return hash_;
}

inline std::uint64_t Board::GetMoveHash(const int vtx, const int color) const {
    std::uint64_t hash = Zobrist::kState[color][vtx];
    if (color == to_move_) {
        hash ^= Zobrist::kBlackToMove; 
    }
    return hash;
}

inline int Board::GetState(const int vtx) const {
    return state_[vtx];
}

inline int Board::GetState(const int x, const int y) const {
    return GetState(GetVertex(x,y));
}

inline int Board::GetLiberties(const int vtx) const {
    return strings_.GetLiberty(strings_.GetParent(vtx));
}

inline int Board::GetStones(const int vtx) const {
    return strings_.GetStones(strings_.GetParent(vtx));
}

inline int Board::GetEmptyCount() const {
    return empty_cnt_;
}

inline int Board::GetEmpty(const int idx) const {
    return empty_[idx];
}
