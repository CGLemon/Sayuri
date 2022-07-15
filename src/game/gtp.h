#pragma once

#include "game/game_state.h"
#include "game/types.h"
#include "config.h"
#include "neural/fast_policy.h"
#include "mcts/search.h"
#include "utils/parser.h"
#include "utils/threadpool.h"
#include "book/book.h"
#include "pattern/gammas_dict.h"

#include <memory>

class Search;

class GtpLoop {
public:
    class Agent {
    public:
        GameState& GetState() {
            return main_state_;
        }

        Network& GetNetwork() {
            return network_;
        }

        Search& GetSearch() {
            return *search_;
        }

        void ApplySearch() {
            search_ = std::make_unique<Search>(main_state_, network_);
        }

        void Quit() {
            network_.Destroy();
            if (search_) {
                search_.reset();
            }
        }

    private:
        std::unique_ptr<Search> search_;

        GameState main_state_;
        Network network_;
    };

    GtpLoop() {
        agent_ = std::make_unique<Agent>();
        agent_->GetState().Reset(GetOption<int>("defualt_boardsize"),
                                     GetOption<float>("defualt_komi"));
        agent_->GetNetwork().Initialize(GetOption<std::string>("weights_file"));
        agent_->ApplySearch();

        ThreadPool::Get(GetOption<int>("threads"));
        FastPolicy::Get().LoadWeights();

        Book::Get().LoadBook(GetOption<std::string>("book_file"));
        GammasDict::Get().Initialize();

        Loop();
    }

    ~GtpLoop() {}

private:
    void Loop();

    std::string GTPSuccess(std::string);
    std::string GTPFail(std::string);

    std::string Execute(CommandParser &parser, bool &try_ponder);

    std::unique_ptr<Agent> agent_{nullptr};
};
