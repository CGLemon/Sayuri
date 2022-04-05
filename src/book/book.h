#pragma once

#include <unordered_map>
#include <string>

#include "game/game_state.h"

class Book {
public:
    static Book &Get();

    void GenerateBook(std::string sgf_name, std::string filename) const;

    void LoadBook(std::string book_name);

    int Probe(const GameState &state) const;

private:
    void BookDataProcess(std::string sgfstring,
                             std::unordered_map<std::uint64_t, int> &book_data) const;

    std::unordered_map<std::uint64_t, int> data_;

    static constexpr int kBookBoardSize = 19;
    static constexpr int kMaxBookMoves = 30;
    static constexpr int kFilterThreshold = 25;
};
