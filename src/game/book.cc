#include <algorithm>
#include <fstream>
#include <sstream>
#include <utility>
#include <stdexcept>
#include <numeric>

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
    BookMap<VertexFrequencyList> book_data_freq;

    std::shuffle(std::begin(sgfs),
        std::end(sgfs), Random<kXoroShiro128Plus>::Get());

    int games = 0;
    for (const auto &sgf: sgfs) {
        if (games >= kMaxSgfGames) {
            LOGGING << "Too many games. Cut off remaining games.\n";
            break;
        }
        if (!BookDataProcess(sgf, book_data_freq)) {
            continue;
        }
        if (++games % 1000 == 0) {
            LOGGING << Format("Parsed %d games\n", games);
        }
    }

    auto file = std::ofstream{};
    file.open(filename);
    if (!file.is_open()) {
        LOGGING << "Fail to create the file: " << filename << '!' << std::endl;
        return;
    }

    int idx = 0;

    for (const auto &it: book_data_freq) {
        auto filtered_vfreq_list = VertexFrequencyList{};
        auto filtered_vprob_list = VertexProbabilityList{};

        const auto hash = it.first;
        const auto &vfreq_list = it.second;

        for (const auto &vfreq: vfreq_list) {
            if (vfreq.second >= kFilterFreqThreshold) {
                filtered_vfreq_list.emplace_back(vfreq);
            }
        }
        if (!filtered_vfreq_list.empty()) {
            const int accum_freq = std::accumulate(
                std::begin(filtered_vfreq_list),
                std::end(filtered_vfreq_list),
                0,
                [](int sum, const auto &vfreq) { return sum + vfreq.second; }
            );
            float accum_prob = 0.0f;
            for (const auto &vfreq: filtered_vfreq_list) {
                float prob = static_cast<float>(vfreq.second)/accum_freq;
                if (prob >= kFilterProbThreshold) {
                    filtered_vprob_list.emplace_back(vfreq.first, prob);
                    accum_prob += prob;
                }
            }
            for (auto &vprob: filtered_vprob_list) {
                vprob.second /= accum_prob;
            }
        }

        if (!filtered_vprob_list.empty()) {
            if (idx++ != 0) {
                file << '\n';
            }
            file << hash;

            for (const auto &vprob: filtered_vprob_list) {
                int vertex = vprob.first;
                float prob = vprob.second;
                file << ' ' <<  vertex << ' ' << prob;
            }
        }
    }

    file.close();
}

bool Book::BookDataProcess(std::string sgfstring,
                           Book::BookMap<VertexFrequencyList> &book_data) const {
    GameState state;
    try {
        state = Sgf::Get().FromString(sgfstring, kMaxBookMoves);
    } catch (const std::exception& e) {
        LOGGING << "Fail to load the SGF file! Discard it." << std::endl
                    << Format("\tCause: %s.", e.what()) << std::endl;
        return false;
    }

    if (state.GetBoardSize() != kBookBoardSize) {
        LOGGING << "Rejected: Board size (" << state.GetBoardSize()
                << ") does not match expected size (" << kBookBoardSize << ")." << std::endl;
        return false;
    }
    if (state.GetHandicap() != 0) {
        LOGGING << "Rejected: Handicap (" << state.GetHandicap()
                << ") is not supported." << std::endl;
        return false;
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
    return true;
}

void Book::LoadBook(std::string book_name) {
    if (book_name.empty()) return;

    try {
        std::ifstream file;
        file.open(book_name);
        if (!file.is_open()) {
            throw std::runtime_error{"Cann't open the file."};
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
        LOGGING << GetInformation();
    } catch (const std::exception& e) {
        LOGGING << "Fail to load the opening file: " << book_name << '!' << std::endl;
    }
}

bool Book::Probe(const GameState &state, int &book_move) const {
    if (data_.empty() ||
            state.GetBoardSize() != kBookBoardSize ||
            state.GetMoveNumber() > kMaxBookMoves) {
        return false;
    }

    auto candidate_moves = GetCandidateMoves(state);
    if (candidate_moves.empty()) {
        return false;
    }

    auto accum_score = std::accumulate(
        std::begin(candidate_moves), std::end(candidate_moves), 0.0f,
        [](float sum, const auto &candidate) { return sum + candidate.first; }
    );

    const unsigned N_base = 1000000;
    const auto rand = Random<kXoroShiro128Plus>::Get().Generate() %
                          static_cast<unsigned>(N_base * accum_score);
    int choice = 0;
    accum_score = 0.0f;

    for (int i = 0; i < (int)candidate_moves.size(); ++i) {
        accum_score +=  candidate_moves[i].first;
        if (rand < static_cast<unsigned>(N_base * accum_score)) {
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
            if (!state.IsLegalMove(vtx)) {
                continue;
            }
            candidate_moves.emplace_back(score, vtx);
        }
    }

    std::sort(std::rbegin(candidate_moves), std::rend(candidate_moves));
    return candidate_moves;
}

std::string Book::GetInformation() const {
    auto oss = std::ostringstream();

    int positions = 0;
    int moves = 0;
    for (auto &it : data_) {
        positions += 1;
        moves += it.second.size();
    }
    oss << Format("The Book contains %d positions and %d candidate moves.\n", positions, moves);
    return oss.str();
}
