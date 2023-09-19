#include <fstream>
#include <sstream>
#include <utility>
#include <algorithm>

#include "utils/log.h"
#include "utils/random.h"
#include "utils/format.h"
#include "game/sgf.h"
#include "game/types.h"
#include "game/book.h"
#include "game/symmetry.h"
#include "game/iterator.h"

Book &Book::Get() {
    static Book book;
    return book;
}

void Book::GenerateBook(std::string sgf_name, std::string filename) const {

    auto sgfs = SgfParser::Get().ChopAll(sgf_name);
    BookMap<VertexFrequencyList> book_data;

    std::shuffle(std::begin(sgfs),
        std::end(sgfs), Random<kXoroShiro128Plus>::Get());

    int games = 0;
    for (const auto &sgf: sgfs) {
        BookDataProcess(sgf, book_data);
        if (++games % 1000 == 0) {
            LOGGING << Format("Parsed %d games\n", games);
        }
        if (games > kMaxSgfGames) {
            LOGGING << "Too many games. Cut off remaining games.\n";
            break;
        }
    }

    auto file = std::ofstream{};

    file.open(filename);
    if (!file.is_open()) {
        LOGGING << "Fail to create the file: " << filename << '!' << std::endl; 
        return;
    }

    int idx = 0;

    for (const auto &it: book_data) {
        VertexProbabilityList filtered_vprob_list;
        VertexFrequencyList filtered_vfreq_list;

        int accm = 0;
        auto &vfreq_list = it.second;

        for (const auto &vfreq: vfreq_list) {
            if (vfreq.second > kFilterThreshold) {
                filtered_vfreq_list.emplace_back(vfreq);
                accm += vfreq.second;
            }
        }

        if (accm != 0) {
            if (idx++ != 0) {
                file << '\n';
            }
            file << it.first;

            for (const auto &vfreq: filtered_vfreq_list) {
                int vertex = vfreq.first;
                float prob = (float)vfreq.second / accm;
                file << ' ' <<  vertex << ' ' << prob;
            }
        }
    }

    file.close();
}

void Book::BookDataProcess(std::string sgfstring,
                           Book::BookMap<VertexFrequencyList> &book_data) const {

    GameState state;
    try {
        state = Sgf::Get().FromString(sgfstring, kMaxBookMoves);
    } catch (const char *err) {
        LOGGING << "Fail to load the SGF file! Discard it." << std::endl
                    << Format("\tCause: %s.", err) << std::endl;
        return;
    }

    if (state.GetBoardSize() != kBookBoardSize) {
        return;
    }

    auto game_ite = GameStateIterator(state);
    int book_move_num = std::min(kMaxBookMoves, (int)game_ite.MaxMoveNumber());

    // TODO: Same positions may have variant paths. We should 
    //       consider it. But it will use too many memory and process
    //       is slow. Try to find a better algorithm to deal with it. 
    int i = 0;
    do {
        if (i++ >= book_move_num) {
            break;
        }

        const auto vertex = game_ite.GetVertex();
        GameState& main_state = game_ite.GetState();

        for (int symm = 0; symm < Symmetry::kNumSymmetris; ++symm) { 
            auto hash = main_state.ComputeSymmetryKoHash(symm);
            auto it = book_data.find(hash);

            const int symm_vtx = Symmetry::Get().TransformVertex(state.GetBoardSize(), symm, vertex);

            if (it == std::end(book_data)) {
                // Insert new hash state in the book, also insert the 
                // new move.

                VertexFrequencyList vfreq;
                vfreq.emplace_back(symm_vtx, 1);

                book_data.insert({hash,  vfreq});
            } else {
                auto &vfreq_list = it->second;
                auto vfreq_it = std::find_if(std::begin(vfreq_list), std::end(vfreq_list),
                                                 [symm_vtx](auto &element) { return element.first == symm_vtx; });
                if (vfreq_it == std::end(vfreq_list)) {
                    // Insert new move in the book.
                    vfreq_list.emplace_back(symm_vtx, 1);
                } else {
                    vfreq_it->second++;
                }
            }
        }
    } while (game_ite.Next());
}

void Book::LoadBook(std::string book_name) {
    if (book_name.empty()) return;

    std::ifstream file;
    file.open(book_name);
    if (!file.is_open()) {
        LOGGING << "Fail to load the file: " << book_name << '!' << std::endl; 
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
    LOGGING << GetVerbose();
}

bool Book::Probe(const GameState &state, int &book_move) const {
    if (data_.empty() ||
            state.GetBoardSize() != kBookBoardSize ||
            state.GetMoveNumber() > kMaxBookMoves) {
        return false;
    }

    auto accm_score = 0;
    auto candidate_moves = std::vector<std::pair<int, int>>{};

    auto hash = state.GetKoHash();
    auto it = data_.find(hash);

    if (it != std::end(data_)) {
        auto &vprob_list = it->second;

        for (auto &vprob : vprob_list) {
            int vtx = vprob.first;
            int score = (int)(vprob.second * 10000);

            candidate_moves.emplace_back(score, vtx);
            accm_score += score;
        }
    }

    if (candidate_moves.empty()) return false;

    std::sort(std::rbegin(candidate_moves), std::rend(candidate_moves));

    const auto rand = Random<kXoroShiro128Plus>::Get().Generate() % accm_score;
    int choice = 0;
    accm_score = 0;

    for (int i = 0; i < (int)candidate_moves.size(); ++i) {
        accm_score +=  candidate_moves[i].first;
        if (rand < (unsigned)accm_score) {
            choice = i;
            break;
        }
    }

    book_move = candidate_moves[choice].second;
    return true;
}

std::vector<std::pair<float, int>> Book::GetCandidateMoves(const GameState &state) const {
    auto candidate_moves = std::vector<std::pair<float, int>>{};
    auto hash = state.GetKoHash();
    auto it = data_.find(hash);

    if (it != std::end(data_)) {
        auto &vprob_list = it->second;

        for (auto &vprob : vprob_list) {
            int vtx = vprob.first;
            float score = vprob.second;

            candidate_moves.emplace_back(score, vtx);
        }
    }

    std::sort(std::rbegin(candidate_moves), std::rend(candidate_moves));
    return candidate_moves;
}

std::string Book::GetVerbose() const {
    auto oss = std::ostringstream();

    int positions = 0;
    int moves = 0;
    for (auto &it : data_) {
        positions += 1;
        moves += it.second.size();
    }
    oss << Format("The Book contains %d positions and %d candidate moves\n", positions, moves);
    return oss.str();
}
