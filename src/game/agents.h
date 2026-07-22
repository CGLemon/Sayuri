#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "neural/network.h"

class GameState;
class Search;

struct AgentContext {
public:
    explicit AgentContext(Network* net);

    GameState& GetState();
    Network& GetNetwork();
    Search& GetSearch();

private:
    std::unique_ptr<GameState> state{nullptr};
    std::unique_ptr<Search> search{nullptr};
    Network* network{nullptr};
};

class Agents {
public:
    Agents();
    ~Agents();

    void Initialize(int num_contexts);
    AgentContext& GetContext(int idx);
    Network& GetNetwork();
    void Shutdown();

    void SetBoardSize(int board_size);
    void SetBatchSize(int batch_size);
    void SetThreads(int threads);

    std::size_t GetNumContexts() const;

protected:
    std::string SelectWeights() const;
    void InitializeNetwork();
    void InitializeContexts(int num_contexts);
    void Initialize(int num_contexts, bool reduce_for_random_network);

    std::string weights_name_;

private:
    std::unique_ptr<Network> network_{nullptr};
    std::vector<std::unique_ptr<AgentContext>> contexts_;
};

class GptAgent : public Agents {
public:
    static constexpr int kNumContext = 1;
    static constexpr int kGptContextIdx = 0;

    GptAgent();
    AgentContext& GetContext();

    GameState& GetState();
    Search& GetSearch();

    void SetBoardSize(int board_size);
};

class ParallelAgents : public Agents {
public:
    explicit ParallelAgents(int num_contexts);

    bool ShouldHalt() const;
};
