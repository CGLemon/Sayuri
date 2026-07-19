#include "game/agents.h"

#include <stdexcept>

#include "game/book.h"
#include "game/game_state.h"
#include "mcts/parameters.h"
#include "mcts/search.h"
#include "pattern/gammas_dict.h"
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

Agents::Agents(int num_contexts) {
    Initialize(num_contexts);
}

Agents::~Agents() {
    Shutdown();
}

AgentContext& Agents::GetContext(int idx) {
    if (idx < 0 || static_cast<std::size_t>(idx) >= GetNumContexts()) {
        throw std::out_of_range("invalid context index");
    }
    return *contexts_[idx];
}

void Agents::Initialize(int num_contexts) {
    if (num_contexts <= 0) {
        throw std::invalid_argument("num_contexts must be positive");
    }
    if (network_ != nullptr) {
        return;
    }
    network_ = std::make_unique<Network>();
    network_->Initialize(GetOption<std::string>("weights_file"));

    contexts_.reserve(static_cast<std::size_t>(num_contexts));
    for (int i = 0; i < num_contexts; ++i) {
        contexts_.emplace_back(std::make_unique<AgentContext>(network_.get()));
    }
    SetThreads(GetOption<int>("threads"));

    Book::Get().LoadBook(GetOption<std::string>("book_file"));
    GammasDict::Get().LoadPatterns(GetOption<std::string>("patterns_file"));
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

GptAgent::GptAgent() : Agents(GptAgent::kNumContext) {}

GameState& GptAgent::GetState() {
    return GetContext().GetState();
}

Network& GptAgent::GetNetwork() {
    return GetContext().GetNetwork();
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