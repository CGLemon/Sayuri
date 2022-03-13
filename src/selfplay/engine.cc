#include "selfplay/engine.h"
#include "utils/threadpool.h"
#include "utils/random.h"
#include "utils/komi.h"
#include "game/sgf.h"
#include "config.h"

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

    ThreadPool::Get((GetOption<int>("threads") - 1) * parallel_games_); // one thread is main thread
}

void Engine::SetCacheSize(int playout) {
    network_->SetCacheSize(parallel_games_ * playout);
}

void Engine::SaveSgf(std::string filename, int g) {
    Handel(g);
    std::lock_guard<std::mutex> lock(io_mtx_);
    Sgf::Get().ToFile(filename, game_pool_[g]);
}

void Engine::SaveTrainingData(std::string filename, int g) {
    Handel(g);
    std::lock_guard<std::mutex> lock(io_mtx_);
    search_pool_[g]->SaveTrainingBuffer(filename, game_pool_[g]);
}

void Engine::PrepareGame(int g) {
    Handel(g);
    auto &state = game_pool_[g];

    state.ClearBoard();

    SetNormalGame(g);
}

void Engine::Selfplay(int g) {
    Handel(g);
    auto &state = game_pool_[g];
    while (!state.IsGameOver()) {
        state.PlayMove(search_pool_[g]->GetSelfPlayMove());
    }
}

void Engine::SetNormalGame(int g) {
    auto komi = GetOption<float>("defualt_komi");
    auto mean = GetOption<float>("komi_mean");
    auto variant = GetOption<float>("komi_variant");
    auto dis = std::normal_distribution<float>(mean, variant);

    auto bonus = dis(Random<kXoroShiro128Plus>::Get());

    game_pool_[g].SetKomi(AdjustKomi<float>(komi + bonus));
}

void Engine::SetHandicapGame(int g) {
    auto &state = game_pool_[g];

    int handicap = Random<kXoroShiro128Plus>::Get().RandFix<4>() + 1;
    state.SetFixdHandicap(handicap);

    SetFairKomi(g);
}

void Engine::SetFairKomi(int g) {
    auto &state = game_pool_[g];
    auto result = search_pool_[g]->Computation(400, 0, Search::kNullTag);
    auto komi = state.GetKomi();
    auto final_score = result.root_final_score;

    if (state.GetToMove() == kBlack) {
        final_score = 0.0f - final_score;
    }

    state.SetKomi(AdjustKomi<int>(final_score + komi));
}

int Engine::GetParallelGames() const {
    return parallel_games_;
}

void Engine::Handel(int g) {
    if (g < 0 || g >= parallel_games_) {
        throw std::runtime_error("Selection is out of array.");
    }
}
