#pragma once

#include <unordered_map>
#include <string>
#include <vector>
#include <utility>

#include "game/game_state.h"

class Book {
public:
    static Book &Get();

    // Generate the new opening book from SGF data.
    void GenerateBook(std::string sgf_name, std::string filename) const;

    // Load the opening book from a file.
    void LoadBook(std::string book_name);

    // Search for an opening move in the book given the current
    // game state. Return true if a move was found, false otherwise.
    bool Probe(const GameState &state, int &book_move) const;

    // Retrieve candidate moves from the book for the given game state.
    std::vector<std::pair<float, int>> GetCandidateMoves(const GameState &state) const;

    // Get a human-readable description of the current book state.
    std::string GetVerbose() const;

private:
    using VertexFrequencyList = std::vector<std::pair<int ,int>>;
    using VertexProbabilityList = std::vector<std::pair<int ,float>>;

    template <typename T>
    using BookMap = std::unordered_map<std::uint64_t, T>;

    BookMap<VertexProbabilityList> data_;

    bool BookDataProcess(std::string sgfstring,
                         Book::BookMap<VertexFrequencyList> &book_data) const;

    static constexpr int kBookBoardSize = 19;
    static constexpr int kMaxBookMoves = 30;
    static constexpr int kFilterFreqThreshold = 5;
    static constexpr float kFilterProbThreshold = 0.001f;
    static constexpr int kMaxSgfGames = 100000;
};
