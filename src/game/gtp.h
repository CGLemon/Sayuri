#pragma once

#include "game/game_state.h"
#include "game/types.h"
#include "game/book.h"
#include "config.h"
#include "mcts/search.h"
#include "mcts/node.h"
#include "utils/splitter.h"
#include "utils/threadpool.h"
#include "utils/format.h"
#include "pattern/gammas_dict.h"
#include "version.h"

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

        void Apply() {
            if (search_) {
                return;
            }
            main_state_.Reset(GetOption<int>("defualt_boardsize"),
                                 GetOption<float>("defualt_komi"),
                                 GetOption<int>("scoring_rule"));
            network_.Initialize(GetOption<std::string>("weights_file"));
            search_ = std::make_unique<Search>(main_state_, network_);
        }

        void Quit() {
            network_.Destroy();
            if (search_) {
                search_.reset();
            }
        }

        void SetBoardSize(int board_size) {
            main_state_.SetBoardSize(board_size);
            network_.Reconstruct(
                Network::Parameters::Get().SetBoardSize(board_size));
        }
        void SetBatchSize(int batch_size) {
            Parameters * param = search_->GetParams();
            param->batch_size = batch_size;
            network_.Reconstruct(
                Network::Parameters::Get().SetBatchSize(param->batch_size));
        }
        void SetThreads(int threads) {
            Parameters * param = search_->GetParams();
            param->threads = threads;
            ThreadPool::Get("search", param->threads);
        }

    private:
        std::unique_ptr<Search> search_{nullptr};
        GameState main_state_;
        Network network_;
    };

    GtpLoop() {
        agent_ = std::make_unique<Agent>();
        agent_->Apply();

        ThreadPool::Get("search", GetOption<int>("threads"));

        Book::Get().LoadBook(GetOption<std::string>("book_file"));
        GammasDict::Get().Initialize(GetOption<std::string>("patterns_file"));

        auto kgs_hint = GetOption<std::string>("kgs_hint");
        if (kgs_hint.empty()) {
            version_verbose_ = Format("%s (%s)",
                                          GetProgramVersion().c_str(),
                                          GetVersionName().c_str()
                                     );
        } else {
            version_verbose_ = Format("%s (%s). %s",
                                          GetProgramVersion().c_str(),
                                          GetVersionName().c_str(),
                                          kgs_hint.c_str()
                                     );
        }
        curr_id_ = -1;
        prev_pondering_ = false;

        Loop();
    }

    ~GtpLoop() {}

private:
    void Loop();

    std::string GtpSuccess(std::string);
    std::string GtpFail(std::string);

    AnalysisConfig ParseAnalysisConfig(Splitter &spt, int &color);
    bool ParseOption(Splitter &spt, std::string &rep);
    bool NetBench(Splitter &spt, std::string &rep);

    std::string Execute(Splitter &spt, bool &try_ponder);

    std::unique_ptr<Agent> agent_{nullptr};

    int curr_id_;
    bool prev_pondering_;
    std::string version_verbose_;
};
