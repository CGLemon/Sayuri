#pragma once

#include <unordered_map>
#include <string>
#include <vector>
#include <utility>

#include "game/game_state.h"

class Book {
public:
    static Book &Get();

    // Generate the new opening book.
    void GenerateBook(std::string sgf_name, std::string filename) const;

    // Load the opening book.
    void LoadBook(std::string book_name);

    // Try to find the opening moves. Return false if there is
    // no move in the book.
    bool Probe(const GameState &state, int &book_move) const;

    std::vector<std::pair<float, int>> GetCandidateMoves(const GameState &state) const;

    std::string GetVerbose() const;

private:
    using VertexFrequencyList = std::vector<std::pair<int ,int>>;
    using VertexProbabilityList = std::vector<std::pair<int ,float>>;

    template <typename T>
    using BookMap = std::unordered_map<std::uint64_t, T>;

    BookMap<VertexProbabilityList> data_;

    void BookDataProcess(std::string sgfstring,
                         Book::BookMap<VertexFrequencyList> &book_data) const;

    static constexpr int kBookBoardSize = 19;
    static constexpr int kMaxBookMoves = 30;
    static constexpr int kFilterThreshold = 25;
    static constexpr int kMaxSgfGames = 100000;
};
