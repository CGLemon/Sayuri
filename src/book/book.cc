
#include <fstream>
#include <sstream>
#include <utility>
#include <algorithm>

#include "book/book.h"
#include "utils/log.h"
#include "utils/random.h"
#include "utils/format.h"
#include "game/sgf.h"
#include "game/types.h"
#include "game/symmetry.h"

Book &Book::Get() {
    static Book book;
    return book;
}

void Book::GenerateBook(std::string sgf_name, std::string filename) const {
    auto sgfs = SgfParser::Get().ChopAll(sgf_name);

    std::unordered_map<std::uint64_t, int> book_data;

    int games = 0;
    for (const auto &sgf: sgfs) {
        BookDataProcess(sgf, book_data);
        if (++games % 1000 == 0) {
            LOGGING << Format("parsed %d games\n", games);
        }
    }

    auto file = std::ofstream{};

    file.open(filename, std::ios_base::app);
    if (!file.is_open()) {
        ERROR << "Fail to create the file: " << filename << '!' << std::endl; 
        return;
    }

    std::unordered_map<std::uint64_t, int> filtered_book_data;

    for (const auto &it: book_data) {
        if (it.second >= kFilterThreshold) {
            filtered_book_data.insert(it);
        }
    }

    int idx = 0;
    for (const auto &it: filtered_book_data) {
        if (idx++ != 0) {
            file << '\n';
        }
        file << it.first << ' ' <<  it.second;
    }

    file.close();
}

void Book::BookDataProcess(std::string sgfstring,
                               std::unordered_map<std::uint64_t, int> &book_data) const {
    GameState state;
    try {
        state = Sgf::Get().FromString(sgfstring, kMaxBookMoves);
    } catch (const char *err) {
        ERROR << "Fail to load the SGF file! Discard it." << std::endl
                  << Format("\tCause: %s.", err) << std::endl;
        return;
    }

    if (state.GetBoardSize() != kBookBoardSize) {
        return;
    }

    auto history = state.GetHistory();
    auto movelist = std::vector<int>{};

    for (const auto &board : history) {
        auto vtx = board->GetLastMove();
        if (vtx != kNullVertex) {
            movelist.emplace_back(vtx);
        }
    }

    GameState main_state;
    main_state.Reset(state.GetBoardSize(), state.GetKomi());

    int book_move_num = std::min(kMaxBookMoves, (int)movelist.size());

    for (int i = 0; i < book_move_num; ++i) {
        main_state.PlayMove(movelist[i]);

        for (int symm = 0; symm < Symmetry::kNumSymmetris; ++symm) { 
            auto hash = main_state.ComputeSymmetryKoHash(symm);
            auto it = book_data.find(hash);

            if (it == std::end(book_data)) {
                book_data.insert({hash, 1});
            } else {
                it->second++;
            }
        }
    }
}

void Book::LoadBook(std::string book_name) {
    if (book_name.empty()) return;

    std::ifstream file;
    file.open(book_name);
    if (!file.is_open()) {
        ERROR << "Fail to load the file: " << book_name << '!' << std::endl; 
        return;
    }

    data_.clear();

    auto line = std::string{};
    while(std::getline(file, line)) {
        if (line.empty()) break;

        std::istringstream iss{line};
        std::uint64_t hash;
        int count;

        iss >> hash >> count;
        data_.insert({hash, count});
    }
    file.close();
}

int Book::Probe(const GameState &state) const {
    if (data_.empty() ||
            state.GetBoardSize() != kBookBoardSize ||
            state.GetMoveNumber() > kMaxBookMoves) {
        return kPass;
    }

    const auto board_size = state.GetBoardSize();
    const auto num_intersections = state.GetNumIntersections();

    auto acc_score = 0;
    auto candidate_moves = std::vector<std::pair<int, int>>{};

    for (int idx = 0; idx < num_intersections; ++idx) {
        const auto x = idx % board_size;
        const auto y = idx / board_size;
        const auto vtx = state.GetVertex(x, y);
        if (state.IsLegalMove(vtx)) {
            auto current_state = state;
            current_state.PlayMove(vtx);

            auto hash = current_state.GetKoHash();
            auto it = data_.find(hash);

            if (it != std::end(data_)) {
                candidate_moves.emplace_back(it->second, vtx);
                acc_score += it->second;
            }
        }
    }

    if (candidate_moves.empty()) return kPass;

    std::stable_sort(std::rbegin(candidate_moves), std::rend(candidate_moves));

    const auto rand = Random<kXoroShiro128Plus>::Get().Generate() % acc_score;
    int choice;
    acc_score = 0;

    for (int i = 0; i < (int)candidate_moves.size(); ++i) {
        acc_score +=  candidate_moves[i].first;
        if (rand < acc_score) {
            choice = i;
            break;
        }
    }

    return candidate_moves[choice].second;
}
