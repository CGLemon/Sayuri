#include "selfplay/engine.h"
#include "utils/threadpool.h"
#include "utils/random.h"
#include "utils/komi.h"
#include "game/sgf.h"
#include "config.h"

#include <sstream>

void Engine::Initialize() {
    parallel_games_ = GetOption<int>("parallel_games");

    if (!network_) {
        network_ = std::make_unique<Network>();
    }
    network_->Initialize(GetOption<std::string>("weights_file"));

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
            // boardsize-handicaps-rate
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
    auto &state = game_pool_[g];

    int boardsize = state.GetBoardSize();
    float komi = state.GetKomi();

    float variance = GetOption<float>("komi_variance");
    auto dist = std::normal_distribution<float>(0.f, variance);
    float bonus = dist(Random<>::Get());

    game_pool_[g].Reset(boardsize, AdjustKomi<float>(komi + bonus));
}

void Engine::SetHandicapGame(int g, int handicaps) {
    auto &state = game_pool_[g];

    for (int i = 0; i < handicaps-1; ++i) {
        auto result = search_pool_[g]->
                          Computation(16, Search::kNullTag);
        state.AppendMove(result.random_move, kBlack);
    }
    state.SetHandicap(handicaps);
    SetFairKomi(g);
}

void Engine::SetFairKomi(int g) {
    auto &state = game_pool_[g];
    auto result = search_pool_[g]->Computation(400, Search::kNullTag);
    auto komi = state.GetKomi();
    auto score_lead = result.root_final_score;

    if (state.GetToMove() == kWhite) {
        score_lead = 0.0f - score_lead;
    }

    state.SetKomi(AdjustKomi<int>(komi + score_lead));
}

int Engine::GetHandicaps(int g) {

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

void Engine::Handel(int g) {
    if (g < 0 || g >= parallel_games_) {
        throw std::runtime_error("Selection is out of array.");
    }
}
