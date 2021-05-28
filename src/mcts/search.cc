#include <numeric>
#include <algorithm>
#include <sstream>
#include <iomanip>

#include "mcts/search.h"
#include "utils/log.h"

Search::~Search() {
    group_->WaitToJoin();
}

void Search::Initialize() {
    param_ = std::make_shared<Parameters>();
    param_->Reset();

    threads_ = param_->threads;

    group_ = std::make_unique<ThreadGroup<void>>(&ThreadPool::Get(threads_ - 1));

    max_playouts_ = param_->playouts;
    playouts_.store(0);
}

void Search::PlaySimulation(GameState &currstate, Node *const node,
                            Node *const root_node, SearchResult &search_result) {
    node->IncrementThreads();
    if (node->Expandable()) {
        if (currstate.GetPasses() >= 2) {
            search_result.FromGameover(currstate);
            node->ApplyEvals(search_result.GetEvals());
        } else {
            const bool have_children = node->HaveChildren();
            const bool success = node->ExpendChildren(network_, currstate, false);
            if (!have_children && success) {
                search_result.FromNetEvals(node->GetNodeEvals());
            }
        }
    }
    if (node->HaveChildren() && !search_result.IsValid()) {
        auto color = currstate.GetToMove();
        auto next = node->UctSelectChild(color, node == root_node);
        auto vtx = next->GetVertex();

        currstate.PlayMove(vtx, color);
        PlaySimulation(currstate, next, root_node, search_result);
    }
    if (search_result.IsValid()) {
        node->Update(search_result.GetEvals());
    }
    node->DecrementThreads();
}

void Search::PrepareRootNode() {
    auto data = std::make_shared<NodeData>();
    node_stats_ = std::make_shared<NodeStats>();

    data->parameters = param_;
    data->node_stats = node_stats_;
    root_node_ = std::make_shared<Node>(data);

    playouts_.store(0);
    running_.store(true);

    root_node_->PrepareRootNode(network_, root_state_);
    const auto evals = root_node_->GetNodeEvals();
    root_node_->Update(std::make_shared<NodeEvals>(evals));

    const auto color = root_state_.GetToMove();
    const auto winloss = color == kBlack ? evals.black_wl : 1 - evals.black_wl;
    const auto final_score = color == kBlack ? evals.black_final_score :
                                                   -evals.black_final_score;

    if (GetOption<bool>("analysis_verbose")) {
        LOGGING << "Raw NN output:" << std::endl
                    << std::fixed << std::setprecision(2)
                    << std::setw(7) << "eval:" << ' ' << winloss * 100.f << "%" << std::endl
                    << std::setw(7) << "draw:" << ' ' << evals.draw * 100.f << "%" << std::endl
                    << std::setw(7) << "final score:" << ' ' << final_score << std::endl;
    }
}

void Search::ClearNodes() {
    if (root_node_) {
        root_node_.reset();
        root_node_ = nullptr;
    }
    if (node_stats_) {
        assert(node_stats_->nodes.load() == 0);
        assert(node_stats_->edges.load() == 0);

        node_stats_.reset();
        node_stats_ = nullptr;
    }
}

void Search::TimeSettings(const int main_time,
                          const int byo_yomi_time,
                          const int byo_yomi_stones) {
    time_control_.TimeSettings(main_time, byo_yomi_time, byo_yomi_stones);
}

void Search::TimeLeft(const int color, const int time, const int stones) {
    time_control_.TimeLeft(color, time, stones);
}

ComputationResult Search::Computation(int playours) {
    auto result = ComputationResult{};

    if (root_state_.IsGameOver()) {
        return result;
    }

    const auto Worker = [this]() -> void {
        while(running_.load()) {
            auto currstate = std::make_unique<GameState>(root_state_);
            auto result = SearchResult{};
            PlaySimulation(*currstate, root_node_.get(), root_node_.get(), result);
            if (result.IsValid()) {
                playouts_.fetch_add(1);
            }
        };
    };

    const auto color = root_state_.GetToMove();
    const auto board_size = root_state_.GetBoardSize();
    const auto move_num = root_state_.GetMoveNumber();

    Timer timer;

    time_control_.Clock();
    timer.Clock();

    auto thinking_time = time_control_.GetThinkingTime(color, board_size, move_num);

    PrepareRootNode();

    if (thinking_time < timer.GetDuration()) {
        running_.store(false);
    }

    if (GetOption<bool>("analysis_verbose")) {
        LOGGING << "Thinking time: " << thinking_time << "(sec)" << std::endl;
    }

    for (int t = 0; t < param_->threads-1; ++t) {
        group_->AddTask(Worker);
    }

    auto keep_running = true;
    while (running_.load()) {
        auto currstate = std::make_unique<GameState>(root_state_);
        auto result = SearchResult{};

        PlaySimulation(*currstate, root_node_.get(), root_node_.get(), result);
        if (result.IsValid()) {
            playouts_.fetch_add(1);
        }
        const auto elapsed = timer.GetDuration();

        keep_running &= (elapsed < thinking_time);
        keep_running &= (playouts_.load() < playours);
        keep_running &= running_.load();
        running_.store(keep_running);
    }

    group_->WaitToJoin();

    result.best_move = root_node_->GetBestMove();
    result.root_eval = root_node_->GetEval(color, false);
    result.final_score = root_node_->GetFinalScore(color);

    auto num_intersections = root_state_.GetNumIntersections();
    auto ownership = root_node_->GetOwnership(color);
    result.ownership.resize(num_intersections);
    std::copy(std::begin(ownership), 
                  std::begin(ownership) + num_intersections,
                  std::begin(result.ownership));


    time_control_.TookTime(color);
    if (GetOption<bool>("analysis_verbose")) {
        LOGGING << root_node_->ToString(root_state_);
        LOGGING << time_control_.ToString();
    }

    ClearNodes();

    return result;
}

int Search::ThinkBestMove() {
    auto result = Computation(max_playouts_);
    if (result.root_eval < param_->resign_threshold) {
        return kResign;
    }

    return result.best_move;
}
