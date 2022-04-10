
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
    std::unordered_map<std::uint64_t, VertexFrequencyList> book_data;

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

    int idx = 0;

    for (const auto &it: book_data) {
        VertexProbabilityList filtered_vprob_list;
        VertexFrequencyList filtered_vfreq_list;

        int acc = 0;
        auto &vfreq_list = it.second;

        for (const auto &vfreq: vfreq_list) {
            if (vfreq.second > kFilterThreshold) {
                filtered_vfreq_list.emplace_back(vfreq);
                acc += vfreq.second;
            }
        }

        if (acc != 0) {
            if (idx++ != 0) {
                file << '\n';
            }
            file << it.first << ' ';

            for (const auto &vfreq: filtered_vfreq_list) {
                int vertex = vfreq.first;
                float prob = (float)vfreq.second / acc;
                file << vertex << ' ' << prob << ' ';
            }
        }
    }


    file.close();
}

void Book::BookDataProcess(std::string sgfstring,
                               std::unordered_map<std::uint64_t, VertexFrequencyList> &book_data) const {

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
        int vertex = movelist[i];

        for (int symm = 0; symm < Symmetry::kNumSymmetris; ++symm) { 
            auto hash = main_state.ComputeSymmetryKoHash(symm);
            auto it = book_data.find(hash);

            if (it == std::end(book_data)) {
                // Insert new hash state in the book, also insert the 
                // new move.

                VertexFrequencyList vfreq;
                vfreq.emplace_back(vertex, 1);

                book_data.insert({hash,  vfreq});
            } else {
                auto &vfreq_list = it->second;
                auto vfreq_it = std::find_if(std::begin(vfreq_list), std::end(vfreq_list),
                                                 [vertex](auto &element) { return element.first == vertex; });
                if (vfreq_it == std::end(vfreq_list)) {
                    // Insert new move in the book.
                    vfreq_list.emplace_back(vertex, 1);
                } else {
                    vfreq_it->second++;
                }
            }
        }

        main_state.PlayMove(vertex);
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
        int vertex;
        float prob;
        VertexProbabilityList vprob;

        iss >> hash;
        while (iss >> vertex) {
            iss >> prob;
            vprob.emplace_back(vertex, prob);
        }

        data_.insert({hash, vprob});
    }
    file.close();
}

int Book::Probe(const GameState &state) const {
    if (data_.empty() ||
            state.GetBoardSize() != kBookBoardSize ||
            state.GetMoveNumber() > kMaxBookMoves) {
        return kPass;
    }

    auto acc_score = 0;
    auto candidate_moves = std::vector<std::pair<int, int>>{};


    auto hash = state.GetKoHash();
    auto it = data_.find(hash);

    if (it != std::end(data_)) {
        auto &vprob_list = it->second;

        for (auto &vprob : vprob_list) {
            int vtx = vprob.first;
            int score = (int)(vprob.second * 10000);

            candidate_moves.emplace_back(score, vtx);
            acc_score += score;
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
