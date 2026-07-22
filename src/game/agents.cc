#include "game/agents.h"

#include <algorithm>
#include <stdexcept>
#include <thread>

#include "game/book.h"
#include "game/game_state.h"
#include "mcts/parameters.h"
#include "mcts/search.h"
#include "pattern/gammas_dict.h"
#include "utils/filesystem.h"
#include "utils/option.h"
#include "utils/threadpool.h"

AgentContext::AgentContext(Network* n) {
    state = std::make_unique<GameState>();
    state->Reset(GetOption<int>("default_boardsize"),
                 GetOption<float>("default_komi"),
                 GetOption<int>("scoring_rule"));
    network = n;
    search = std::make_unique<Search>(*state, *network);
}

GameState& AgentContext::GetState() {
    return *state;
}

Network& AgentContext::GetNetwork() {
    return *network;
}

Search& AgentContext::GetSearch() {
    return *search;
}

Agents::Agents() = default;

Agents::~Agents() {
    Shutdown();
}

AgentContext& Agents::GetContext(int idx) {
    if (idx < 0 || static_cast<std::size_t>(idx) >= GetNumContexts()) {
        throw std::out_of_range("invalid context index");
    }
    return *contexts_[idx];
}

Network& Agents::GetNetwork() {
    return *network_;
}

std::string Agents::SelectWeights() const {
    auto select_weights = GetOption<std::string>("weights_file");
    if (!select_weights.empty()) {
        return select_weights;
    }

    auto weights_dir = GetOption<std::string>("weights_dir");
    auto weights_list = GetFileList(weights_dir);

    if (!weights_list.empty()) {
        // Seletet the last weights in this directory.
        std::sort(std::begin(weights_list),
                  std::end(weights_list),
                  [weights_dir](std::string a, std::string b) {
                      auto time_a = GetFileTime(ConcatPath(weights_dir, a));
                      auto time_b = GetFileTime(ConcatPath(weights_dir, b));
                      return difftime(time_a, time_b) > 0.f;
                  });
        select_weights = ConcatPath(weights_dir, weights_list[0]);
    }

    return select_weights;
}

void Agents::InitializeNetwork() {
    if (network_ != nullptr) {
        return;
    }
    weights_name_ = SelectWeights();
    network_ = std::make_unique<Network>();
    network_->Initialize(weights_name_);
}

void Agents::InitializeContexts(int num_contexts) {
    contexts_.reserve(static_cast<std::size_t>(num_contexts));
    for (int i = 0; i < num_contexts; ++i) {
        contexts_.emplace_back(std::make_unique<AgentContext>(network_.get()));
    }
    SetThreads(GetOption<int>("threads"));

    Book::Get().LoadBook(GetOption<std::string>("book_file"));
    GammasDict::Get().LoadPatterns(GetOption<std::string>("patterns_file"));
}

void Agents::Initialize(int num_contexts) {
    Initialize(num_contexts, false);
}

void Agents::Initialize(int num_contexts, bool reduce_for_random_network) {
    if (num_contexts <= 0) {
        throw std::invalid_argument("num_contexts must be positive");
    }
    InitializeNetwork();
    if (reduce_for_random_network) {
        if (GetNetwork().GetName().find("random") != std::string::npos) {
            // Will be CPU-bound so reducing number of threads.
            num_contexts =
                std::min(static_cast<int>(std::thread::hardware_concurrency()) - 1, num_contexts);
            num_contexts = std::max(1, num_contexts);
        }
    }
    InitializeContexts(num_contexts);
}

void Agents::Shutdown() {
    contexts_.clear();
    if (network_) {
        network_->Destroy();
        network_.reset();
    }
}

void Agents::SetBoardSize(int board_size) {
    for (auto& ctx : contexts_) {
        ctx->GetState().SetBoardSize(board_size);
    }
}

void Agents::SetBatchSize(int batch_size) {
    for (auto& ctx : contexts_) {
        Parameters* param = ctx->GetSearch().GetParams();
        param->batch_size = batch_size;
    }
    network_->Reconstruct(Network::Option::Get().SetBatchSize(batch_size));
}

void Agents::SetThreads(int threads) {
    for (auto& ctx : contexts_) {
        Parameters* param = ctx->GetSearch().GetParams();
        param->threads = threads;
    }
    ThreadPool::Get("search", GetNumContexts() * threads);
}

std::size_t Agents::GetNumContexts() const {
    return contexts_.size();
}

GptAgent::GptAgent() {
    Initialize(GptAgent::kNumContext, false);
}

GameState& GptAgent::GetState() {
    return GetContext().GetState();
}

Search& GptAgent::GetSearch() {
    return GetContext().GetSearch();
}

void GptAgent::SetBoardSize(int board_size) {
    Agents::SetBoardSize(board_size);
    GetNetwork().Reconstruct(Network::Option::Get().SetBoardSize(board_size));
}

AgentContext& GptAgent::GetContext() {
    return Agents::GetContext(GptAgent::kGptContextIdx);
}

ParallelAgents::ParallelAgents(int num_contexts) {
    // Reduce contexts when running with a random/no-weights network.
    Initialize(num_contexts, true);
}

bool ParallelAgents::ShouldHalt() const {
    return weights_name_ != SelectWeights();
}