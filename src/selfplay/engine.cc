#include "selfplay/engine.h"
#include "utils/threadpool.h"
#include "utils/random.h"
#include "utils/komi.h"
#include "utils/filesystem.h"
#include "game/sgf.h"
#include "config.h"

#include <sstream>
#include <algorithm>
#include <cmath>

void Engine::Initialize() {
    default_playouts_ = GetOption<int>("playouts");
    komi_stddev_ = GetOption<float>("komi_stddev");
    komi_big_stddev_ = GetOption<float>("komi_big_stddev");
    komi_big_stddev_prob_ = GetOption<float>("komi_big_stddev_prob");
    handicap_fair_komi_prob_ = GetOption<float>("handicap_fair_komi_prob");
    random_opening_prob_ = GetOption<float>("random_opening_prob");
    random_moves_factor_ = GetOption<float>("random_moves_factor");
    parallel_games_ = GetOption<int>("parallel_games");
    last_net_accm_queries_ = 0;

    if (!network_) {
        network_ = std::make_unique<Network>();
    }
    network_->Initialize(SelectWeights());

    game_pool_.clear();
    for (int i = 0; i < parallel_games_; ++i) {
        game_pool_.emplace_back(GameState{});
        game_pool_[i].Reset(GetOption<int>("defualt_boardsize"),
                                GetOption<float>("defualt_komi"));
    }

    search_pool_.clear();
    for (int i = 0; i < parallel_games_; ++i) {
        search_pool_.emplace_back(std::make_unique<Search>(game_pool_[i], *network_));
    }

    ThreadPool::Get(GetOption<int>("threads") * parallel_games_);

    ParseQueries();
}

std::string Engine::SelectWeights() const {
    // default weights
    auto select_weights = GetOption<std::string>("weights_file");
    if (!select_weights.empty()) {
        return select_weights;
    }

    auto weights_dir = GetOption<std::string>("weights_dir");
    auto weights_list = GetFileList(weights_dir);

    if (!weights_list.empty()) {
        // Seletet the last weights in this directory.
        std::sort(std::begin(weights_list), std::end(weights_list),
                      [weights_dir](std::string a, std::string b) {
                          auto time_a = GetFileTime(ConcatPath(weights_dir, a));
                          auto time_b = GetFileTime(ConcatPath(weights_dir, b));
                          return difftime(time_a, time_b) > 0.f;
                      });
        select_weights = ConcatPath(weights_dir, weights_list[0]);
    }

    return select_weights;
}

void Engine::ParseQueries() {
    auto queries = GetOption<std::string>("selfplay_query");
    std::istringstream iss{queries};
    std::string query;

    float bq_acc_prob = 0.f;

    while (iss >> query) {
        for (char &c : query) {
            if (c == ':') {
                c = ' ';
            }
        }

        std::istringstream tiss{query};
        std::string token;
        std::vector<std::string> tokens;

        while (tiss >> token) {
            tokens.emplace_back(token);
        }

        if (tokens[0] == "bkp" && tokens.size() == 4) {
            // boardsize-komi-probabilities
            // "bkp:19:7.5:0.2"

            // Assume the query is valid.
            BoardQuery q {
                .board_size = std::stoi(tokens[1]),
                .komi       = std::stof(tokens[2]),
                .prob       = std::stof(tokens[3])};
            board_queries_.emplace_back(q);
            bq_acc_prob += q.prob;
        } else if (tokens[0] == "bhp" && tokens.size() == 4) {
            // boardsize-handicaps-probabilities
            // "bhp:9:2:0.1"

            HandicapQuery q {
                .board_size    = std::stoi(tokens[1]),
                .handicaps     = std::stoi(tokens[2]),
                .probabilities = std::stof(tokens[3])};
            if (q.handicaps >= 2) {
                handicap_queries_.emplace_back(q);
            }
        }
    }

    int max_bsize = -1;
    if (board_queries_.empty()) {
        BoardQuery q {
            .board_size = GetOption<int>("defualt_boardsize"),
            .komi = GetOption<float>("defualt_komi"),
            .prob = 1.f };
        board_queries_.emplace_back(q);
        max_bsize = q.board_size;
    } else {
        for (auto &q : board_queries_) {
            q.prob /= bq_acc_prob;
            max_bsize = std::max(q.board_size, max_bsize);
        }
    }

    // Adjust the matched NN size.
    network_->Reload(max_bsize);
}

void Engine::SaveSgf(std::string filename, int g) {
    Handel(g);
    Sgf::Get().ToFile(filename, game_pool_[g]);
}

void Engine::GatherTrainingData(std::vector<Training> &chunk, int g) {
    Handel(g);
    search_pool_[g]->GatherTrainingBuffer(chunk, game_pool_[g]);
}

void Engine::PrepareGame(int g) {
    Handel(g);
    auto &state = game_pool_[g];

    state.ClearBoard();

    constexpr std::uint32_t kRange = 1000000;
    std::uint32_t rand = Random<>::Get().RandFix<kRange>();

    float acc_prob = 0.f;
    int select_bk_idx = 0;

    for (int i = 0; i < (int)board_queries_.size(); ++i) {
        acc_prob += board_queries_[i].prob;
        if (rand <= kRange * acc_prob) {
            select_bk_idx = i;
            break;
        }
    }
    int query_boardsize = board_queries_[select_bk_idx].board_size;
    float query_komi = board_queries_[select_bk_idx].komi;
    state.Reset(query_boardsize, query_komi);

    const int h = GetHandicaps(g);
    if (h > 0) {
        SetHandicapGame(g, h);
    } else {
        SetNormalGame(g);
    }
}

void Engine::Selfplay(int g) {
    Handel(g);
    auto &state = game_pool_[g];
    while (!state.IsGameOver()) {
        state.PlayMove(search_pool_[g]->GetSelfPlayMove());
    }
}

void Engine::SetNormalGame(int g) {
    Handel(g);
    if (Random<>::Get().Roulette<10000>(random_opening_prob_)) {
        SetRandomOpeningGame(g);
    }
    SetUnfairKomi(g);
}

void Engine::SetHandicapGame(int g, int handicaps) {
    Handel(g);
    auto &state = game_pool_[g];

    for (int i = 0; i < handicaps-1; ++i) {
        state.SetToMove(kBlack);
        int random_move = network_->GetVertexWithPolicy(state, 0.75f, false);
        state.AppendMove(random_move, kBlack);
    }
    state.SetHandicap(handicaps);
    SetFairKomi(g);

    if (Random<>::Get().Roulette<10000>(random_opening_prob_)) {
        SetRandomOpeningGame(g);
    }
    if (!Random<>::Get().Roulette<10000>(handicap_fair_komi_prob_)) {
        SetUnfairKomi(g);
    }
}

void Engine::SetRandomOpeningGame(int g) {
    Handel(g);
    auto &state = game_pool_[g];

    const int board_size = state.GetBoardSize();
    const int random_moves_cnt =
        random_moves_factor_ * state.GetNumIntersections();
    auto dist = std::normal_distribution<float>(0.f, (float)board_size/4);
    const int remainig_random_moves =
        std::max(
            int(dist(Random<>::Get())) + random_moves_cnt - state.GetMoveNumber(),
            0
        );

    const float lambda = 0.69314718056f/board_size;
    const float init_temp = 0.8f;
    int times = 0;
    for (int i = 0; i < remainig_random_moves; ++i) {
        if (state.GetPasses() >= 2) {
            break;
        }
        float curr_temp = std::max(
            init_temp * std::exp(-(lambda * times)), 0.2f);

        int random_move = network_->GetVertexWithPolicy(state, curr_temp, false);
        state.PlayMove(random_move);
        times += 1;
    }
    SetFairKomi(g);
}

void Engine::SetUnfairKomi(int g) {
    Handel(g);
    auto &state = game_pool_[g];
    float komi = state.GetKomi();

    float stddev = komi_stddev_;
    if (Random<>::Get().Roulette<10000>(komi_big_stddev_prob_)) {
        stddev = komi_big_stddev_;
    }

    auto dist = std::normal_distribution<float>(0.f, stddev);
    float bonus = dist(Random<>::Get());

    state.SetKomi(AdjustKomi<float>(komi + bonus));
}

void Engine::SetFairKomi(int g) {
    Handel(g);
    auto &state = game_pool_[g];

    auto result = search_pool_[g]->Computation(
                      default_playouts_, Search::kNoExploring);
    auto komi = state.GetKomi();
    auto score_lead = result.root_score_lead;

    if (state.GetToMove() == kWhite) {
        score_lead = 0.0f - score_lead;
    }

    state.SetKomi(AdjustKomi<float>(komi + score_lead));
}

int Engine::GetHandicaps(int g) {
    Handel(g);
    auto &state = game_pool_[g];

    for (auto &q : handicap_queries_) {
        if (state.GetBoardSize() == q.board_size) {
            if (Random<>::Get().Roulette<10000>(q.probabilities)) {
                return Random<>::Get().Generate() % (q.handicaps-1) + 2;
            }
        }
    }
    return 0;
}

int Engine::GetParallelGames() const {
    return parallel_games_;
}

size_t Engine::GetNetReportQueries() {
    size_t curr_net_accm_queries = network_->GetNumQueries();
    const auto report_queries =
        curr_net_accm_queries - last_net_accm_queries_;
    last_net_accm_queries_ = curr_net_accm_queries;
    return report_queries;
}

void Engine::Handel(int g) {
    if (g < 0 || g >= parallel_games_) {
        throw std::runtime_error("Selection is out of array.");
    }
}
