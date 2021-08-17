#include <numeric>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <fstream>

#include "mcts/search.h"
#include "neural/encoder.h"
#include "utils/log.h"

Search::~Search() {
    group_->WaitToJoin();
}

void Search::Initialize() {
    param_ = std::make_shared<Parameters>();
    param_->Reset();

    threads_ = param_->threads;

    group_ = std::make_unique<ThreadGroup<void>>(&ThreadPool::Get());

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
        Node *next = nullptr;
        if (playouts_.load() < param_->cap_playouts) {
            next = node->ProbSelectChild();
        } else {
            next = node->UctSelectChild(color, node == root_node);
        }
        auto vtx = next->GetVertex();

        currstate.PlayMove(vtx, color);
        PlaySimulation(currstate, next, root_node, search_result);
    }
    if (search_result.IsValid()) {
        node->Update(search_result.GetEvals());
    }
    node->DecrementThreads();
}

std::vector<float> Search::PrepareRootNode() {
    auto data = std::make_shared<NodeData>();
    node_stats_ = std::make_shared<NodeStats>();

    data->parameters = param_;
    data->node_stats = node_stats_;
    root_node_ = std::make_shared<Node>(data);

    playouts_.store(0);
    running_.store(true);

    auto root_noise = std::vector<float>{};

    root_node_->PrepareRootNode(network_, root_state_, root_noise);
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

    return root_noise;
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
    auto computation_result = ComputationResult{};

    if (root_state_.IsGameOver()) {
        computation_result.best_move = kPass;
        return computation_result;
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

    computation_result.board_size = board_size;
    computation_result.komi = root_state_.GetKomi();
    computation_result.movenum = root_state_.GetMoveNumber();

    Timer timer;

    time_control_.Clock();
    timer.Clock();

    const auto thinking_time = time_control_.GetThinkingTime(color, board_size, move_num);
    const auto root_noise = PrepareRootNode();

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

    time_control_.TookTime(color);
    if (GetOption<bool>("analysis_verbose")) {
        LOGGING << root_node_->ToString(root_state_);
        LOGGING << time_control_.ToString();
    }

    // Fill side to move, moves, root eval and score.
    computation_result.to_move = static_cast<VertexType>(color);
    computation_result.best_move = root_node_->GetBestMove();
    computation_result.random_move = root_node_->RandomizeFirstProportionally(1);
    computation_result.root_eval = root_node_->GetEval(color, false);
    computation_result.root_final_score = root_node_->GetFinalScore(color);

    auto num_intersections = root_state_.GetNumIntersections();

    // Resize the childern status buffer.
    computation_result.root_ownership.resize(num_intersections);
    computation_result.root_probabilities.resize(num_intersections+1);
    computation_result.root_target_probabilities.resize(num_intersections+1);
    computation_result.root_policy.resize(num_intersections+1);
    computation_result.root_noise.resize(num_intersections+1);
    computation_result.root_visits.resize(num_intersections+1);

    // Fill noise.
    std::copy(std::begin(root_noise), 
                  std::end(root_noise),
                  std::begin(computation_result.root_noise));

    // Fill ownership.
    auto ownership = root_node_->GetOwnership(color);
    std::copy(std::begin(ownership), 
                  std::begin(ownership) + num_intersections,
                  std::begin(computation_result.root_ownership));

    // Fill visits.
    auto parentvisits = 0;
    const auto &children = root_node_->GetChildren();
    for (const auto &child : children) {
        const auto node = child->Get();
        const auto visits = node->GetVisits();
        const auto vertex = node->GetVertex();
        const auto policy = node->GetPolicy();
        parentvisits += visits;
        if (vertex == kPass) {
            computation_result.root_visits[num_intersections] = visits;
            computation_result.root_policy[num_intersections] = policy;
            continue;
        }

        const auto x = root_state_.GetX(vertex);
        const auto y = root_state_.GetY(vertex);
        const auto index = root_state_.GetIndex(x, y);

        computation_result.root_visits[index] = visits;
        computation_result.root_policy[index] = policy;
    }

    // Fill probabilities.
    for (int idx = 0; idx < num_intersections+1; ++idx) {
        float prob = (float) computation_result.root_visits[idx]/ (float) parentvisits;
        computation_result.root_probabilities[idx] = prob;
    }

    // Fill target probabilities.
    // root_node_->PolicyTargetPruning();
    float tot_target_policy = 0.0f;
    for (int idx = 0; idx < num_intersections+1; ++idx) {
        const auto x = idx % board_size;
        const auto y = idx / board_size;
        int vertex;

        if (idx == num_intersections) {
            vertex = kPass;
        } else {
            vertex = root_state_.GetVertex(x, y);
        }

        auto node = root_node_->GetChild(vertex);
        if (node && !node->IsPruned()) {
            const auto prob = computation_result.root_probabilities[idx];
            computation_result.root_target_probabilities[idx] = prob;
            tot_target_policy += prob;
        } else {
            computation_result.root_target_probabilities[idx] = 0.0f;
        }
    }

    for (auto &prob : computation_result.root_target_probabilities) {
        prob /= tot_target_policy;
    }

    // Push the data to buffer.
    GatherData(root_state_, computation_result);

    ClearNodes();

    return computation_result;
}

int Search::ThinkBestMove() {
    auto result = Computation(max_playouts_);
    if (result.root_eval < param_->resign_threshold) {
        return kResign;
    }

    return result.best_move;
}

int Search::GetSelfPlayMove() {
    auto result = Computation(max_playouts_);
    int move = result.best_move;
    if (param_->random_moves_cnt > result.movenum) {
        move = result.random_move;
    }

    return move;
}

void Search::SaveTrainingBuffer(std::string filename, GameState &end_state) {
    auto file = std::ofstream{};
    file.open(filename, std::ios_base::app);

    if (!file.is_open()) {
        ERROR << "Fail to create the file: " << filename << '!' << std::endl; 
        return;
    }

    auto num_intersections = end_state.GetNumIntersections();
    auto winner = end_state.GetWinner();
    auto black_final_score = 0.f;
    auto ownership = end_state.GetOwnership(200);

    for (const auto owner : ownership) {
        if (owner == kBlack) {
            black_final_score += 1;
        } else if (owner == kWhite) {
            black_final_score -= 1;
        }
    }

    black_final_score -= end_state.GetKomi();

    for (auto &buf : training_buffer_) {
        assert(winner != kUndecide);
        if (winner == kDraw) {
            buf.final_score = 0;
            buf.result = 0;
        } else {
            buf.result = (int)winner == (int)buf.side_to_move ? 1 : -1;
            buf.final_score = buf.side_to_move == kBlack ? black_final_score : -black_final_score;
        }

        buf.ownership.resize(num_intersections);
        for (int idx = 0; idx < num_intersections; ++idx) {
            const auto owner = ownership[idx];
            if (buf.side_to_move == owner) {
                buf.ownership[idx] = 1;
            } else if ((!buf.side_to_move) == owner) {
                buf.ownership[idx] = -1;
            } else {
                buf.ownership[idx] = 0;
            }
        }
    }

    auto aux_prob = std::vector<float>(num_intersections+1, 0);
    aux_prob[num_intersections] = 1.f;
    for (int i = training_buffer_.size()-1; i >= 0; --i) {
        auto &buf = training_buffer_[i];

        buf.auxiliary_probabilities_index = -1;
        buf.auxiliary_probabilities = aux_prob;

        buf.probabilities_index = -1;
        aux_prob = buf.probabilities;
    }

    for (auto &buf : training_buffer_) {
        buf.StreamOut(file);
    }

    training_buffer_.clear();
}

void Search::GatherData(const GameState &state, ComputationResult &result) {
    if (training_buffer_.size() > 999) {
        // To many data in the buffer.
        return;
    }

    auto data = Training{};
    data.version = GetTrainigVersion();
    data.mode = GetTrainigMode();

    data.board_size = result.board_size;
    data.komi = result.komi;
    data.side_to_move = result.to_move;

    data.planes = Encoder::Get().GetPlanes(state);
    data.probabilities = result.root_target_probabilities;

    training_buffer_.emplace_back(data);
}
