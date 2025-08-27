#include <numeric>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <stack>
#include <random>
#include <cmath>
#include <chrono>

#include "mcts/search.h"
#include "neural/encoder.h"
#include "utils/log.h"
#include "utils/format.h"
#include "utils/random.h"
#include "utils/kldivergence.h"
#include "game/book.h"

#ifdef WIN32
#include <windows.h>
#else
#include <sys/select.h>
#endif

constexpr int Search::kMaxPlayouts;

Search::~Search() {
    ReleaseTree();
    group_->WaitToJoin();
}

void Search::Initialize() {
    param_ = std::make_unique<Parameters>();
    param_->Reset();

    no_exploring_param_ = std::make_unique<Parameters>();
    no_exploring_param_->Reset();
    no_exploring_param_->gumbel = false;
    no_exploring_param_->dirichlet_noise = false;
    no_exploring_param_->root_policy_temp = 1.f;
    no_exploring_param_->forced_playouts_k = 0.f;
    no_exploring_param_->no_exploring_phase = true;
    no_exploring_param_->kldgain_per_node = 0.0;
    no_exploring_param_->kldgain_interval = 0;

    analysis_config_.Clear();
    last_state_ = root_state_;
    root_node_.reset(nullptr);

    group_ = std::make_unique<ThreadGroup<void>>(&ThreadPool::Get());
    playouts_.store(0, std::memory_order_relaxed);
    ThreadPool::Get("tree-destruction", 1);
}

void Search::PlaySinglePlayout() {
    auto currstate = std::make_unique<GameState>(root_state_);
    auto result = SearchResult{};
    PlaySimulation(*currstate, root_node_.get(), 0, result);
    if (result.IsValid()) {
        playouts_.fetch_add(1, std::memory_order_relaxed);
    }
}

void Search::PlaySimulation(GameState &currstate, Node *const node,
                            const int depth, SearchResult &search_result) {
    node->IncrementThreads();

    const bool end_by_passes = currstate.GetPasses() >= 2;
    const auto scoring = currstate.GetScoringRule();
    if (end_by_passes) {
        if (scoring == kArea) {
            search_result.FromGameOver(network_, currstate);
        } else if (scoring == kTerritory) {
            // The scoring area should be easier rule for network,
            // so we switch the rule into territory and keep playing. 
            while (currstate.GetPasses() >= 1) {
                currstate.UndoMove();
            }
            const auto komi = currstate.GetKomi();
            const auto offset = currstate.GetPenaltyOffset(kArea, kTerritory);

            currstate.SetRule(kArea);
            currstate.SetKomi(komi + offset);
        }
    }

    // Terminated node, try to expand it.
    if (node->Expandable()) {
        const auto last_move = currstate.GetLastMove();

        if (end_by_passes && scoring == kArea) {
            if (node->SetTerminal() &&
                    search_result.IsValid()) {
                // The game is over, setting the game result value.
                node->ApplyEvals(search_result.GetEvals());
            }
        } else if (last_move != kPass &&
                       currstate.IsSuperko()) {
            // Prune all superko nodes.
            node->Invalidate();
        } else {
            const bool has_children = node->HasChildren();

            // If we cannot expand the node, it means another thread is currently
            // working on it. In that case, skip the simulation stage for now.
            // However, the node may still be eligible for PUCT evaluation.
            auto node_evals = NodeEvals{};
            const bool success = node->ExpandChildren(
                network_, currstate, node_evals, analysis_config_, false);

            if (!has_children && success) {
                search_result.FromNetEvals(node_evals);
            }
        }
    }

    // Not the terminated node, search the next node.
    if (node->HasChildren() && !search_result.IsValid()) {
        auto color = currstate.GetToMove();
        Node *next = nullptr;

        // Go to the next node.
        next = node->DescentSelectChild(color, depth == 0);
        currstate.PlayMove(next->GetVertex(), color);

        // Recursive calls function.
        PlaySimulation(currstate, next, depth+1, search_result);
    }

    // Now Update this node if it valid.
    if (search_result.IsValid()) {
        node->Update(search_result.GetEvals());
    }
    node->DecrementThreads();
}

void Search::PrepareRootNode(ComputationResult &result, Search::OptionTag tag) {
    bool reused = AdvanceToNewRootState(tag);

    if (!reused) {
        // Try release whole trees.
        ReleaseTree();

        // Do not reuse the tree, allocate new root node.
        root_node_ = std::make_unique<Node>(param_.get(), kPass, 1.0f);
    }

    playouts_.store(0, std::memory_order_relaxed);
    running_.store(true, std::memory_order_relaxed);

    root_evals_ = NodeEvals{};
    const bool success = root_node_->PrepareRootNode(
                             network_, root_state_, root_evals_, analysis_config_);

    if (!reused && success) {
        root_node_->Update(&root_evals_);
    }

    // We should retrieve the root policy from the NN cache. Since the
    // softmax temperature of 'root_evals_' may not be 1.0, we need to
    // recompute it.
    auto netlist = network_.GetOutput(root_state_, Network::kRandom);
    auto num_intersections = root_state_.GetNumIntersections();
    root_raw_probabilities_.resize(num_intersections+1);

    std::copy(std::begin(netlist.probabilities),
                  std::begin(netlist.probabilities) + num_intersections,
                  std::begin(root_raw_probabilities_));
    root_raw_probabilities_[num_intersections] = netlist.pass_probability;

    UpdateComputationResult(result);
    prev_kldgain_visits_ = result.visits;
    prev_kldgain_target_policy_.resize(num_intersections+1);
    std::copy(std::begin(result.target_policy_dist),
                  std::begin(result.target_policy_dist) + (num_intersections+1),
                  std::begin(prev_kldgain_target_policy_));
}

void Search::ReleaseTree() {
    if (root_node_) {
        auto p = root_node_.release();
        group_->AddTask([p](){ delete p; });
    }
}

void Search::TimeSettings(const int main_time,
                          const int byo_yomi_time,
                          const int byo_yomi_stones,
                          const int byo_yomi_periods) {
    time_control_.TimeSettings(main_time, byo_yomi_time,
                                   byo_yomi_stones, byo_yomi_periods);
    time_control_.SetLagBuffer(param_->lag_buffer);
}

void Search::TimeLeft(const int color, const int time, const int stones) {
    time_control_.TimeLeft(color, time, stones);
}

bool Search::InputPending(Search::OptionTag tag) const {
    if (!(tag & kPonder)) {
        return false;
    }
#ifdef WIN32
    static int init = 0, pipe;
    static HANDLE inh;
    DWORD dw;

    if (!init) {
        init = 1;
        inh = GetStdHandle(STD_INPUT_HANDLE);
        pipe = !GetConsoleMode(inh, &dw);
        if (!pipe) {
            SetConsoleMode(inh, dw & ~(ENABLE_MOUSE_INPUT | ENABLE_WINDOW_INPUT));
            FlushConsoleInputBuffer(inh);
        }
    }

    if (pipe) {
        if (!PeekNamedPipe(inh, nullptr, 0, nullptr, &dw, nullptr)) {
            exit(EXIT_FAILURE);
        }

        return dw;
    } else {
        if (!GetNumberOfConsoleInputEvents(inh, &dw)) {
            exit(EXIT_FAILURE);
        }

        return dw > 1;
    }
    return false;
#else
    fd_set read_fds;
    FD_ZERO(&read_fds);
    FD_SET(0,&read_fds);
    struct timeval timeout{0,0};
    select(1,&read_fds,nullptr,nullptr,&timeout);
    return FD_ISSET(0, &read_fds);
#endif
}

ComputationResult Search::Computation(int playouts, Search::OptionTag tag) {
    auto computation_result = ComputationResult{};
    playouts = std::min(playouts, kMaxPlayouts);

    // Remove all pass moves if we don't want to stop
    // the search.
    int num_removed_passes = 0;
    if (tag & kForced) {
        while (root_state_.GetPasses() >= 2) {
            // Remove double pass move.
            root_state_.UndoMove();
            root_state_.UndoMove();
            num_removed_passes += 2;
        }
    }

    // Disable any exploring setting.
    if (tag & kNoExploring) {
        std::swap(param_, no_exploring_param_);
    }

    // Prepare some basic information.
    const auto color = root_state_.GetToMove();
    const auto board_size = root_state_.GetBoardSize();
    const auto move_num = root_state_.GetMoveNumber();

    computation_result.to_move = static_cast<VertexType>(color);
    computation_result.board_size = board_size;
    computation_result.komi = root_state_.GetKomi();
    computation_result.movenum = root_state_.GetMoveNumber();
    computation_result.visits = root_node_ ? root_node_->GetVisits() : 0;
    computation_result.playouts = 0;
    computation_result.elapsed = 0.f;
    computation_result.threads = param_->threads;
    computation_result.batch_size = param_->batch_size;

    if (root_state_.IsGameOver()) {
        // Always reture pass move if the passese number is greater than two.
        computation_result.high_priority_move = kPass;
        return computation_result;
    }

    if (tag & kThinking) {
        auto book_move = kNullVertex;
        if (Book::Get().Probe(root_state_, book_move)) {
            // Current game state is found in book.
            computation_result.high_priority_move = book_move;
            return computation_result;
        }
    }

    Timer analysis_timer, verbose_timer;
    analysis_timer.Clock();
    verbose_timer.Clock();

    // Set the time control.
    time_control_.Clock();
    time_control_.SetLagBuffer(
        std::max(param_->lag_buffer, time_control_.GetLagBuffer()));

    // Compute the max thinking time. The bound time is
    // max const time if we already set it.
    const float bound_time = (param_->const_time > 0 &&
                                 time_control_.IsInfiniteTime(color)) ?
                                     param_->const_time : std::numeric_limits<float>::max();
    const float thinking_time = !(tag & kThinking) ?
                                    time_control_.GetInfiniteTime() :
                                    std::min(
                                        bound_time,
                                        time_control_.GetThinkingTime(
                                            color, board_size, move_num));

    // The buffer_effect means how much thinking time is actually
    // added when a lag buffer is applied.
    const float buffer_effect = time_control_.GetBufferEffect(
                                    color, board_size, move_num);

    PrepareRootNode(computation_result, tag);

    if (param_->analysis_verbose) {
        LOGGING << Format("Reuse %d nodes.\n", root_node_->GetVisits()-1);
        LOGGING << Format("Using %d threads for search, and %d as the network's batch size.\n",
                              computation_result.threads, computation_result.batch_size);
        LOGGING << Format("Thinking at most %.2f seconds.\n", thinking_time);
        LOGGING << Format("Remaining %d playouts left.\n", GetPlayoutsLeft(playouts, tag));
    }

    if (thinking_time < time_control_.GetDuration() || AchieveCap(playouts, tag)) {
        // Prepare the root node spent little time. Disable the
        // tree search if the time is up.
        running_.store(false, std::memory_order_relaxed);
    }

    for (int t = 0; t < param_->threads; ++t) {
        group_->AddTask(
            [this, playouts, tag]() -> void {
                while (running_.load(std::memory_order_relaxed)) {
                    PlaySinglePlayout();
                    if (AchieveCap(playouts, tag)) {
                        running_.store(false, std::memory_order_release);
                    }
                }
            }
        );
    }
    
    auto keep_running = running_.load(std::memory_order_relaxed);
    while (!InputPending(tag) && keep_running) {
        UpdateComputationResultFast(computation_result);

        if ((tag & kAnalysis) &&
                analysis_config_.interval > 0 &&
                analysis_config_.interval * 10 <
                    analysis_timer.GetDurationMilliseconds()) {
            // Output the analysis string for GTP interface, like sabaki...
            analysis_timer.Clock();
            if (computation_result.visits > 1) {
                DUMPING << root_node_->ToAnalysisString(
                                           root_state_, color, analysis_config_);
            }
        }
        if (param_->analysis_verbose &&
                verbose_timer.GetDuration() > 2.5f) {
            if (computation_result.visits > 1) {
                LOGGING << Format("Playouts: %d, Win: %5.2f%%, PV: %s\n",
                                    computation_result.playouts,
                                    root_node_->GetWL(color) * 100.0f,
                                    root_node_->GetPvString(root_state_).c_str());
            }
            verbose_timer.Clock();
        }

        if (tag & kThinking) {
            keep_running &= (computation_result.elapsed < thinking_time);
        }
        keep_running &= HaveAlternateMoves(computation_result.elapsed, thinking_time, playouts, tag);
        keep_running &= !AchieveCap(playouts, tag);
        keep_running &= !StoppedByKldGain(computation_result, tag);
        keep_running &= running_.load(std::memory_order_relaxed);
        if (keep_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    };

    // Stop the search.
    running_.store(false, std::memory_order_release);

    // Wait for all threads to join the main thread.
    group_->WaitToJoin();

    if (tag & kThinking) {
        time_control_.TookTime(color);
        UpdateLagBuffer(thinking_time, buffer_effect);
    }
    if (tag & kAnalysis) {
        // Output the last analysis verbose because the MCTS may
        // be finished in the short time. It can help the low playouts
        // MCTS to show current analysis graph in GUI.
        if (root_node_->GetVisits() > 1) {
            DUMPING << root_node_->ToAnalysisString(
                           root_state_, color, analysis_config_);
        }
    }

    // Make sure the least one is correct.
    UpdateComputationResult(computation_result);

    if (param_->analysis_verbose) {
        const auto space = 14;
        LOGGING << root_node_->ToVerboseString(root_state_, color)
                    << " * Time Status:\n"
                    << "   " << time_control_.ToString()
                    << std::setw(space) << "elapsed:" << ' ' << computation_result.elapsed << " (sec)\n"
                    << std::setw(space) << "speed:" << ' ' << (float)computation_result.playouts /
                                               std::max(1e-4f, computation_result.elapsed) << " (p/sec)\n"
                    << std::setw(space) << "playouts:" << ' ' << computation_result.playouts << "\n"
                    << std::setw(space) << "visits:" << ' ' << computation_result.visits << "\n";

    }

    // Save the last game state.
    last_state_ = root_state_;

    // Recover the search status.
    if (tag & kForced) {
        for (int i = 0; i < num_removed_passes; ++i) {
            root_state_.PlayMove(kPass);
        }
    }
    if (tag & kNoExploring) {
        std::swap(param_, no_exploring_param_);
    }

    return computation_result;
}

void Search::UpdateLagBuffer(float thinking_time, float buffer_effect) {
    // Try to adjust the lag buffer. Avoid to be time-out for
    // the last move.
    float curr_lag_buf = time_control_.GetLagBuffer();
    const auto elapsed = time_control_.GetDuration();

    // Compute a conservative thinking time with lag buffer.
    const auto thinking_time_with_lag =
                    thinking_time + std::max(
                                        0.75f * buffer_effect,
                                        buffer_effect - 1.0f);

    if (elapsed > thinking_time_with_lag) {
        const auto diff = elapsed - thinking_time_with_lag;

        // Give it a more conservative time buffer.
        curr_lag_buf = curr_lag_buf + std::min(
                                            1.5f * diff,
                                            1.0f + diff);
        time_control_.SetLagBuffer(
            std::max(param_->lag_buffer, curr_lag_buf));
    }
}

void Search::UpdateComputationResultFast(ComputationResult &result) const {
    result.elapsed = time_control_.GetDuration();
    result.visits = root_node_->GetVisits();
    result.playouts = playouts_.load(std::memory_order_relaxed);
}

void Search::UpdateComputationResult(ComputationResult &result) const {
    const auto color = root_state_.GetToMove();
    const auto num_intersections = root_state_.GetNumIntersections();

    UpdateComputationResultFast(result);

    // Fill best moves, root eval and score.
    result.best_move = root_node_->GetBestMove(true);
    result.best_no_pass_move = root_node_->GetBestMove(false);
    if (param_->gumbel || param_->no_exploring_phase) {
        // RandomMoveWithLogitsQ() will help to prune the low
        // visits moves. However, if there is only few candidate
        // moves, it will make the probability too sharp. So we
        // only enable this function for Sequential Halving
        // process.
        // During the fast search phase (no exploring phase), the
        // sharp probability may reduce the randomization. It helps
        // the to optimize the the strengh and improve some diversity.
        result.random_move = root_node_->
                                 GetRandomMoveWithLogitsQ(
                                     root_state_, 1.f);
    } else {
        // According to "On Strength Adjustment for MCTS-Based Programs",
        // pruning the low visits moves. It can help to avoid to play the
        // blunder move. The Elo range should be between around 0 ~ 1000.
        result.random_move = root_node_->
                                 GetRandomMoveProportionally(
                                     param_->random_moves_temp,
                                     param_->random_min_ratio,
                                     param_->random_min_visits);
    }
    result.gumbel_move = root_node_->GetGumbelMove(true);
    result.gumbel_no_pass_move = root_node_->GetGumbelMove(false);
    result.root_score_lead = root_node_->GetFinalScore(color);
    result.root_eval = root_node_->GetWL(color, false);
    result.root_score_stddev = root_node_->GetScoreStddev();
    result.root_eval_stddev = root_node_->GetWLStddev();
    {
        auto best_node = root_node_->GetChild(result.best_move);
        if (best_node->GetVisits() >= 1) {
           result.best_eval = best_node->GetWL(color, false);
        } else {
           result.best_eval = result.root_eval;
        }
    }
    result.side_resign = result.root_eval < param_->resign_threshold ||
                             result.root_eval > (1.f - param_->resign_threshold);

    // Here we gather the part of training target data.
    result.root_ownership.resize(num_intersections, 0);
    result.root_searched_visits.resize(num_intersections+1, 0);
    result.root_estimated_q.resize(num_intersections+1, 0);
    result.root_visits_dist.resize(num_intersections+1, 0);
    result.target_policy_dist.resize(num_intersections+1, 0);

    std::fill(std::begin(result.root_ownership), std::end(result.root_ownership), 0);
    std::fill(std::begin(result.root_searched_visits), std::end(result.root_searched_visits), 0);
    std::fill(std::begin(result.root_estimated_q), std::end(result.root_estimated_q), 0);
    std::fill(std::begin(result.root_visits_dist), std::end(result.root_visits_dist), 0);
    std::fill(std::begin(result.target_policy_dist), std::end(result.target_policy_dist), 0);

    // Fill ownership.
    auto ownership = root_node_->GetOwnership(color);
    std::copy(std::begin(ownership),
                  std::begin(ownership) + num_intersections,
                  std::begin(result.root_ownership));

    // First loop we gather some root's information.
    auto children_visits = 0;
    auto total_visited_policy = 0.0f;
    const auto &children = root_node_->GetChildren();
    for (const auto &child : children) {
        const auto node = child.GetPointer();
        const auto visits = node->GetVisits();

        children_visits += visits;
        if (visits > 0) {
            total_visited_policy += node->GetPolicy();
        }
    }

    // Fill root visits and estimated Q.
    for (const auto &child : children) {
        const auto node = child.GetPointer();
        const auto visits = node->GetVisits();
        const auto vertex = node->GetVertex();

        // Fill root visits for each child.
        const auto index = root_state_.VertexToIndexIncludingPass(vertex);
        result.root_searched_visits[index] = visits;

        // Fill estimated Q value for each child. If the child is
        // unvisited, set the FPU value.
        const auto parent_score = result.root_score_lead;
        const auto q_value = visits == 0 ?
                                 node->GetFpu(color, total_visited_policy, true) :
                                 node->GetWL(color, false) +
                                     node->GetScoreEval(color, parent_score);
        result.root_estimated_q[index] = q_value;
    }

    // Fill raw visits distribution.
    if (children_visits == 0) {
        // uniform distribution
        for (int idx = 0; idx < num_intersections+1; ++idx) {
            result.root_visits_dist[idx] = 1.f/(num_intersections+1);
        }
    } else {
        // Normalize the distribution. Be sure the sum is 1.
        for (int idx = 0; idx < num_intersections+1; ++idx) {
            result.root_visits_dist[idx] =
                (float)(result.root_searched_visits[idx]) / children_visits;
        }
    }

    // Fill policy target distribution. The completed Q policy improve
    // robust in the low playouts case.
    auto prob_with_completed_q =
        root_node_->GetProbLogitsCompletedQ(root_state_);

    if (children_visits == 0) {
        // No useful distribution. Apply uniform distribution.
        result.target_policy_dist = result.root_visits_dist;
    } else if (root_node_->ShouldApplyGumbel() ||
                   param_->always_completed_q_policy) {
        // Apply Gumbel target policy.
        result.target_policy_dist = prob_with_completed_q;
    } else {
        // Merge completed Q policy and raw visits distribution policy.
        auto damping = 800.f;
        auto target_dist_buf = result.root_visits_dist;
        for (int idx = 0; idx < num_intersections+1; ++idx) {
            float factor = std::min(std::min(children_visits, (int)damping) / damping, 1.0f);
            target_dist_buf[idx] = factor * target_dist_buf[idx] + (1.0f - factor) * prob_with_completed_q[idx];
        }

        // Find out the max distribution. We assume it is the best move.
        int best_index = 0;
        for (int idx = 0; idx < num_intersections+1; ++idx) {
            if (target_dist_buf[idx] > target_dist_buf[best_index]) {
                best_index = idx;
            }
        }

        // Prune the noise visits based one "Policy Target Pruning". We think
        // normal PUCT search distribution should be optimal solution. The redundant
        // visits is noise. Please see here for detail, https://arxiv.org/abs/1902.10565v2
        const int virtual_visits = std::max(3200, children_visits);
        float cpuct = root_node_->GetCpuct(virtual_visits);
        float accum_target_policy = 0.0f;
        for (int idx = 0; idx < num_intersections+1; ++idx) {
            if (idx != best_index) {
                const float prob = root_raw_probabilities_[idx];
                const float puct_scaling = cpuct * prob * virtual_visits;
                const float value_diff = result.root_estimated_q[best_index] -
                                             result.root_estimated_q[idx];
                if (value_diff > 0) {
                    // The 'wanted_visits' is the bound of optimal solution. Be sure
                    // the puct value is greater or eqaul to best child.
                    const int wanted_visits = std::max(0,
                        (int)std::round(puct_scaling / value_diff) - 1);
                    const float wanted_prob = (float)wanted_visits / virtual_visits;
                    target_dist_buf[idx] = std::min(wanted_prob, target_dist_buf[idx]);
                }
            }
            accum_target_policy += target_dist_buf[idx];
        }

        if (accum_target_policy < 1e-4f) {
            // All moves are pruned. We directly use the raw
            // distribution.
            result.target_policy_dist = result.root_visits_dist;
        } else {
            // Normalize the distribution. Be sure the sum is 1.
            for (int idx = 0; idx < num_intersections+1; ++idx) {
                target_dist_buf[idx] /= accum_target_policy;
            }
            result.target_policy_dist = target_dist_buf;
        }
    }

    // Finally, compute the target policy KLD, aka policy surpise. All training
    // target data are collected done.
    result.policy_kld = GetKlDivergence(
        result.target_policy_dist, root_raw_probabilities_);

    // Fill the dead strings and live strings.
    constexpr float kOwnershipThreshold = 0.75f; // ~87.5%

    auto safe_ownership = root_state_.GetOwnership();
    auto safe_area = root_state_.GetStrictSafeArea();

    auto alive = std::vector<std::vector<int>>{};
    auto dead = std::vector<std::vector<int>>{};

    for (int idx = 0; idx < num_intersections; ++idx) {
        const auto vtx = root_state_.IndexToVertex(idx);

        // owner value, 1 is mine, -1 is opp's.
        const auto owner = safe_area[idx] == true ?
                               2 * (float)(safe_ownership[idx] == color) - 1 :
                               result.root_ownership[idx];
        const auto state = root_state_.GetState(vtx);

        if (owner > kOwnershipThreshold) {
            // It is my territory.
            if (color == state) {
                alive.emplace_back(root_state_.GetStringList(vtx));
            } else if ((!color) == state) {
                dead.emplace_back(root_state_.GetStringList(vtx));
            }
        } else if (owner < -kOwnershipThreshold) {
            // It is opp's territory.
            if ((!color) == state) {
                alive.emplace_back(root_state_.GetStringList(vtx));
            } else if (color == state) {
                dead.emplace_back(root_state_.GetStringList(vtx));
            }
        }
    }

    // Remove multiple mentions of the same string
    // unique reorders and returns new iterator, erase actually deletes
    std::sort(std::begin(alive), std::end(alive));
    alive.erase(std::unique(std::begin(alive), std::end(alive)),
                std::end(alive));

    std::sort(std::begin(dead), std::end(dead));
    dead.erase(std::unique(std::begin(dead), std::end(dead)),
               std::end(dead));

    result.alive_strings = alive;
    result.dead_strings = dead;

    if (param_->capture_all_dead) {
        // Generate the capture all dead move.
        auto fill_moves = std::vector<int>{};
        auto raw_ownership = root_state_.GetRawOwnership();

        for (int idx = 0; idx < num_intersections; ++idx) {
            const auto vtx = root_state_.IndexToVertex(idx);
            const auto raw_owner = raw_ownership[idx];

            // owner value, 1 is mine, -1 is opp's.
            const auto owner = safe_area[idx] == true ?
                                2 * (float)(safe_ownership[idx] == color) - 1 :
                                result.root_ownership[idx];
            if (owner > kOwnershipThreshold &&
                    root_state_.IsLegalMove(vtx, color)) {
                if (raw_owner == kEmpty && root_state_.IsNeighborColor(vtx, color)) {
                    // adjacent my string
                    fill_moves.emplace_back(vtx);
                }
                if (raw_owner == (!color)) {
                    // in the opp's eye
                    fill_moves.emplace_back(vtx);
                }
            }
        }
        if (!fill_moves.empty()) {
            // Randomize the remove the dead move list.
            std::shuffle(std::begin(fill_moves),
                            std::end(fill_moves),
                            Random<>::Get());

            // The capture move will be the first move.
            std::sort(std::begin(fill_moves), std::end(fill_moves),
                        [this, color](const int &v0, const int &v1) {
                            int v0_cap = root_state_.board_.IsCaptureMove(v0, color);
                            int v1_cap = root_state_.board_.IsCaptureMove(v1, color);
                            return v0_cap > v1_cap;
                        });

            for (int move : fill_moves) {
                auto fork_state = root_state_;
                fork_state.PlayMove(move, color);
                if (!fork_state.IsSuperko()) {
                    // Find the first non-superko move.
                    result.capture_all_dead_move = move;
                    break;
                }
            }
        }
    }
}

bool ShouldResign(GameState &state, ComputationResult &result, Parameters *param) {
    const auto movenum = state.GetMoveNumber();
    const auto num_intersections = state.GetNumIntersections();
    const auto board_size = state.GetBoardSize();
    auto resign_threshold = param->resign_threshold;

    if (resign_threshold <= 0.0f ||
            movenum <= num_intersections / 4 ||
            state.IsGameOver()) {
        // one of these cases should not allow resign
        //
        // case 1. threshold is zero
        // case 2. too early in game to resign
        // case 3. game is finished
        return false;
    }

    // Seem the 7 is the fair komi for most board size.
    const auto virtual_fair_komi = 7.0f;
    const auto komi_diff = state.GetKomi() - virtual_fair_komi;
    const auto to_move = state.GetToMove();
    if ((komi_diff > 0.f && to_move == kBlack) ||
            (komi_diff < 0.f && to_move == kWhite)) {
        auto blend_ratio = std::min(1.0f, movenum / (0.6f * num_intersections));
        auto blended_resign_threshold =
            blend_ratio * resign_threshold +
            (1.0f - blend_ratio) * resign_threshold/
                std::max(1.f, std::abs(5.f * komi_diff/board_size));

        // Shift the resign threshold by komi. Compensate for
        // komi disadvantages.
        resign_threshold = std::min(blended_resign_threshold, resign_threshold);
    }

    const auto handicap = state.GetHandicap();
    if (handicap > 0 && to_move == kWhite) {
        const auto handicap_resign_threshold =
                       (resign_threshold-1.f) * handicap/20.f;
        auto blend_ratio = std::min(1.0f, movenum / (0.6f * num_intersections));
        auto blended_resign_threshold = blend_ratio * resign_threshold +
                                            (1.0f - blend_ratio) * handicap_resign_threshold;

        // Allow lower eval for white in handicap games
        // where opp may fumble.
        resign_threshold = std::min(blended_resign_threshold, resign_threshold);
    }

    return result.best_eval < resign_threshold;
}

bool ShouldPass(GameState &state, ComputationResult &result, Parameters *param) {
    if (!(param->friendly_pass) ||
            state.GetLastMove() != kPass ||
            state.GetScoringRule() != kArea) {
        return false;
    }
    const auto num_intersections = state.GetNumIntersections();
    const auto move_threshold = num_intersections / 3;
    if (state.GetMoveNumber() <= move_threshold) {
        // Too early to pass.
        return false;
    }

    auto dead_list = std::vector<int>{};
    auto fork_state = state;

    for (const auto &string : result.dead_strings) {
        for (const auto vtx: string) {
            dead_list.emplace_back(vtx);
        }
    }

    // Remove the dead strings predicted by NN.
    fork_state.RemoveDeadStrings(dead_list);
    int num_dame = 0;

    for (int idx = 0; idx < num_intersections; ++idx) {
        const auto vtx = fork_state.IndexToVertex(idx);
        if (fork_state.GetState(vtx) != kEmpty &&
                fork_state.GetLiberties(vtx) == 1) {
            // At least one live string in atari, the game
            // is not over yet.
            return false;
        } else if (fork_state.GetState(vtx) == kEmpty) {
            // This empty point does not belong to any
            // side. It should be the dame.
            num_dame += 1;
        }
    }
    (void) num_dame;

    // Compute the final score.
    const auto score = fork_state.GetFinalScore(state.GetToMove());

    if (score > 0.1f) {
        // We already won the game. We will play the pass move.
        return true;
    }

    // The game result is unknown. We will keep playing.
    return false;
}

int Search::GetBestMove(int playouts, OptionTag tag) {
    auto result = Computation(playouts, tag);

    if (ShouldResign(root_state_, result, param_.get())) {
        return kResign;
    }
    if (result.high_priority_move != kNullVertex) {
        return result.high_priority_move;
    }

    // In early game, apply some randomness to improve exploration.
    int best_move = result.best_move;
    const int random_moves_cnt = param_->random_moves_factor *
                                     root_state_.GetNumIntersections();
    if (root_state_.GetMoveNumber() < random_moves_cnt) {
        best_move = result.random_move;
    }
    // If we are clearly winning, consider passing early.
    if (ShouldPass(root_state_, result, param_.get())) {
         // If the current win rate is high enough, play a pass move
         // to finish the game.
        best_move = kPass;
    }

    // Under area scoring rules, if we’re passing but there are still
    // capturable dead stones, play a move to capture them before passing.
    if (param_->capture_all_dead &&
            best_move == kPass &&
            root_state_.GetScoringRule() == kArea &&
            result.capture_all_dead_move != kNullVertex) {
        // Avoid passing if there are still dead stones to capture.
        best_move = result.capture_all_dead_move;
    }
    return best_move;
}

int Search::ThinkBestMove() {
    auto tag = param_->reuse_tree ? kThinking : (kThinking | kUnreused);
    int best_move = GetBestMove(param_->playouts, tag);
    return best_move;
}

bool ShouldForbidPass(GameState &state,
                      ComputationResult &result,
                      NodeEvals &root_evals) {
    // We never resign in self-play because some target training datas
    // are generated from final position. In order to improve the quality
    // of the training data, we forbid pass when the game is not complete.

    const auto num_intersections = state.GetNumIntersections();
    const auto move_threshold = num_intersections / 6;
    if (state.GetMoveNumber() <= move_threshold) {
        // Too early to pass.
        return true;
    }
    if (state.GetScoringRule() == kTerritory) {
        // Scoring territory accepts for pass in any time.
        return false;
    }

    if (state.GetScoringRule() == kArea) {
        // We need to capture all dead stones under the Tromp-Taylor rules and
        // fill dame to make sure final position is clearly. Then, forbid pass
        // if any dead stone is still on the board except stones in the pass-dead
        // area, or if empty area is too large.
        int to_move = result.to_move;
        auto safe_ownership = state.GetOwnership();

        // Scan the dead strings/stons by root evaluation of
        // MCTS.
        for (const auto &string : result.dead_strings) {
            // All vertices in a string should be same color.
            const auto vtx = string[0];
            const auto idx = state.VertexToIndex(vtx);

            // Some opp's strings are death. Forbid the pass
            // move. Keep to eat all opp's dead strings.
            if (state.GetState(vtx) == (!to_move) &&
                    safe_ownership[idx] != to_move) {
                return true;
            }
        }

        constexpr float kRawOwnershipThreshold = 0.8f; // ~90%

        // Scan the dead strings/stons by raw NN evaluation,
        for (int idx = 0; idx < num_intersections; ++idx) {
            float owner = root_evals.black_ownership[idx];
            if (to_move == kWhite) {
                owner = 0.f - owner;
            }

            // Some opp's stones are not really alive. Keep to
            // eat these stones.
            if (owner >= kRawOwnershipThreshold &&
                    safe_ownership[idx] != to_move) {
                return true;
            }
        }

        constexpr int kMaxEmptyGroupThreshold = 8;
        auto &board = state.board_;
        auto buf = std::vector<bool>(state.GetNumVertices(), false);

        for (int idx = 0; idx < num_intersections; ++idx) {
            const auto vtx = state.IndexToVertex(idx);

            if (safe_ownership[idx] == kEmpty && !buf[vtx]) {
                int group_size = board.ComputeReachGroup(vtx, kEmpty, buf);

                // TODO: The empty group size threshold should be smaller.
                if (group_size >= kMaxEmptyGroupThreshold) {
                    // Too large empty group on the board. We think the area
                    // is not stable so keeping to play untill the area is filled
                    // or belongs to someone.
                    return true;
                }
            }
        }
    }

    return false;
}

int Search::GetSelfPlayMove(OptionTag tag) {
    // We always reuse the sub-tree at fast search phase. The
    // kUnreused option doesn't mean discarding the sub-tree.
    // It means visit cap (The search result is as same as
    // "playout cap" + "discard the tree"). The default is playouts
    // cap. If the reuse tag is true, it is visit cap oscillation.
    // Every visit of fast search phase may be different. If the
    // reuse tag is false, it is playout cap oscillation which
    // is used by KataGo. Please see here, https://arxiv.org/abs/1902.10565v2
    if (!(param_->reuse_tree)) {
        tag = tag | kUnreused;
    }

    bool already_lost = training_data_buffer_.empty() ?
                            false : std::rbegin(training_data_buffer_)->accum_resign_cnt > 0;
    const int random_moves_cnt = param_->random_moves_factor *
                                     root_state_.GetNumIntersections();
    bool is_opening_random = root_state_.GetMoveNumber() < random_moves_cnt;

    // Decide the playouts number first. Default is max
    // playouts. May use the lower playouts instead of it.
    int playouts = param_->playouts;

    float fast_search_prob = param_->fastsearch_playouts_prob;
    if (already_lost) {
        // Someone already won the game. Do not record this kind
        // of positions too much to avoid introducing pathological
        // biases in the training data
        float record_prob = (1.0f - fast_search_prob) * (1.0f - param_->resign_discard_prob);
        fast_search_prob = 1.0f - record_prob;
    }

    if (param_->fastsearch_playouts > 0 &&
            param_->fastsearch_playouts < param_->playouts &&
            Random<>::Get().Roulette<10000>(fast_search_prob)) {

        // The reduce playouts must be smaller than default
        // playouts. It is fast search phase so we also disable
        // all exploring settings.
        playouts = std::min(playouts, param_->fastsearch_playouts);

        if (already_lost) {
            // If someone already won the game, the Q value was not very effective
            // in the MCTS. Low playouts with policy network is good enough. 
            playouts = std::min(playouts, param_->resign_playouts);
        }
        tag = tag | kNoExploring;
    }

    if (!network_.Valid()) {
        // The network is dummy backend. The playout path is
        // random, so we only use one tenth playouts in order
        // to reduce time.
        playouts /= 10;
    }

    // The playouts should be at least one for the self-play move
    // because some move select functions need at least one.
    playouts = std::max(1, playouts);

    // Now start the MCTS.
    auto result = Computation(playouts, tag);
    const bool is_gumbel = root_node_->ShouldApplyGumbel() && !(tag & kNoExploring);

    // Default is the best move or the Gumbel-Top-k trick move. May use
    // another move instead of it later.
    int move = is_gumbel ? result.gumbel_move : result.best_move;

    // The game is not end. Don't play the pass move.
    bool forbid_pass = ShouldForbidPass(root_state_, result, root_evals_);
    if (forbid_pass) {
        move = is_gumbel ?
                   result.gumbel_no_pass_move :
                   result.best_no_pass_move;
    }

    // Do we lose the the game?
    float root_eval = result.root_eval;
    float root_score = result.root_score_lead;

    // Do the random move in the opening stage in order to improve the
    // game state diversity. we thought Gumbel noise may be good enough.
    // So Don't play the random move if 'is_gumbel' is true. We also play
    // random move when fast search phase. It may litte improve exploring.
    if ((is_opening_random && !is_gumbel) ||
            (!already_lost &&
                 (tag & kNoExploring) &&
                 Random<>::Get().Roulette<10000>(param_->random_fastsearch_prob))) {
        if (!(forbid_pass && result.random_move == kPass)) {
            move = result.random_move;
        }
    }

    // If the 'discard_it' is true, we will discard the current training
    // data. It is because that the quality of current data is bad. To
    // discard it can improve the network performance.
    bool discard_it = false;
    if (tag & kNoExploring) {
        // It is fast search of "Playout Cap Randomization". Do
        // not record the low quality datas.
        discard_it = true;
    }
    if (result.to_move == kWhite) {
        // Always record the black's view point.
        root_eval = 1.0f - root_eval;
        root_score = 0.f - root_score;
    }

    const float record_kld = result.policy_kld;
    const char discard_char = discard_it ? 'F' : 'T';

    // Save the evaluation information comment in the SGF file.
    root_state_.SetComment(
        Format("%d, %d, %.2f, %.2f, %.2f, %c",
            result.playouts, result.visits,
            root_eval, root_score, record_kld, discard_char));

    // Push the data to buffer.
    if (!(tag & kNoBuffer)) {
        GatherData(root_state_, result, discard_it);
    }
    return move;
}

void Search::TryPonder() {
    if (param_->ponder) {
        // The ponder mode always reuses the tree.
        Computation(GetPonderPlayouts(), kPonder);
    }
}

int Search::Analyze(bool ponder, AnalysisConfig &analysis_config) {
    // Set the current analysis config.
    analysis_config_ = analysis_config;

    auto analysis_tag = kAnalysis;
    auto reuse_tag = param_->reuse_tree ?
                         kNullTag : kUnreused;
    if (analysis_config_.use_reuse_label) {
        // Cover the reuse tree tag, only this time.
        reuse_tag = analysis_config_.reuse_tree ?
                        kNullTag : kUnreused;
    }

    auto ponder_tag = ponder ? kPonder : kThinking;

    auto tag = reuse_tag | ponder_tag | analysis_tag;

    // The tree shape may be different with last move.
    if (analysis_config_.MoveRestrictions()) {
        ReleaseTree();
    }

    int playouts = ponder == true ? GetPonderPlayouts()
                                      : param_->playouts;
    if (analysis_config_.use_playouts_label) {
        // Cover the playouts, only this time.
        playouts = analysis_config_.playouts;
    }

    int best_move = GetBestMove(playouts, tag);

    // Disable to reuse the tree for next move.
    if (analysis_config_.MoveRestrictions()) {
        ReleaseTree();
    }

    // Clear config after finishing the search.
    analysis_config_.Clear();

    return best_move;
}

void Search::ClearTrainingBuffer() {
    training_data_buffer_.clear();
}

void Search::SaveTrainingBuffer(std::string filename) {
    auto file = std::ofstream{};
    file.open(filename, std::ios_base::app);

    if (!file.is_open()) {
        LOGGING << "Fail to create the file: " << filename << '!' << std::endl;
        return;
    }

    auto chunk = std::vector<TrainingData>{};
    GatherTrainingBuffer(chunk);

    for (auto &buf : chunk) {
        buf.StreamOut(file);
    }
    file.close();
}

void Search::UpdateTerritoryHelper() {
    auto temp_state = root_state_; // copy

    if (root_state_.GetScoringRule() == kTerritory) {
        // Keep playing until all dead strings are removed.
        while (root_state_.GetLastMove() == kPass) {
            root_state_.UndoMove();
        }

        const auto komi = root_state_.GetKomi();
        const auto offset = root_state_.GetPenaltyOffset(kArea, kTerritory);

        root_state_.SetRule(kArea);
        root_state_.SetKomi(komi + offset);

        while (!root_state_.IsGameOver()) {
            auto tag = Search::kNoExploring | Search::kNoBuffer;
            root_state_.PlayMove(GetSelfPlayMove(tag));
        }
    }
    auto end_state = root_state_;
    root_state_ = temp_state;
    root_state_.SetTerritoryHelper(end_state.GetOwnership());
}

Parameters *Search::GetParams(bool no_exploring_param) {
    Parameters * out = no_exploring_param ?
                           no_exploring_param_.get() : param_.get();
    return out;
}

void Search::GatherTrainingBuffer(std::vector<TrainingData> &chunk) {

    // Compute the final status positions.
    auto ownership = root_state_.GetOwnership();
    auto num_intersections = root_state_.GetNumIntersections();
    auto winner = kUndecide;
    auto black_final_score = root_state_.GetFinalScore(kBlack);

    // Get the player who won the game.
    if (std::abs(black_final_score) < 1e-4f) {
        winner = kDraw;
    } else if (black_final_score > 0) {
        winner = kBlackWon;
    } else if (black_final_score < 0) {
        winner = kWhiteWon;
    }

    // Set the most buffer values.
    const int buf_size = training_data_buffer_.size();
    for (int i = 0; i < buf_size; ++i) {
        assert(winner != kUndecide);

        auto &buf = training_data_buffer_[i];
        if (winner == kDraw) {
            buf.final_score = 0;
            buf.result = 0;
        } else {
            buf.result = (int)winner == (int)buf.side_to_move ? 1 : -1;
            buf.final_score = buf.side_to_move == kBlack ? black_final_score : -black_final_score;
        }

        buf.ownership.resize(num_intersections, 0);
        for (int idx = 0; idx < num_intersections; ++idx) {
            const auto owner = ownership[idx];
            if (owner == buf.side_to_move) {
                buf.ownership[idx] = 1;
            } else if (owner == !buf.side_to_move) {
                buf.ownership[idx] = -1;
            } else {
                buf.ownership[idx] = 0;
            }
        }

        // Compute average Q/score value.
        const int window_half_size =
            std::max(3, (int)(buf.board_size/2)); // full size is (1 + 2*window_half_size)
        float window_q_sum = 0.f;
        float window_score_sum = 0.f;
        int window_size = 0;
        for (int w = -window_half_size; w <= window_half_size; ++w) {
            int w_index = i+w;
            if (w_index < 0 || w_index >= buf_size) {
                continue;
            }
            auto &w_buf = training_data_buffer_[w_index];

            if (w_buf.side_to_move == buf.side_to_move) {
                window_q_sum += w_buf.q_value;
                window_score_sum += w_buf.score_lead;
            } else {
                window_q_sum -= w_buf.q_value;
                window_score_sum -= w_buf.score_lead;
            }
            window_size += 1;
        }
        buf.avg_q_value = window_q_sum / window_size;
        buf.avg_score_lead = window_score_sum / window_size;
    }

    // Compute the short, middle and long term average values
    for (int i = 0; i < buf_size; ++i) {
        auto &buf = training_data_buffer_[i];

        // Please see here:
        // https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md#short-term-value-and-score-targets
        double short_term_q = 0;
        double middle_term_q = 0;
        double long_term_q = 0;

        double short_term_score = 0;
        double middle_term_score = 0;
        double long_term_score = 0;

        // Coefficient from here:
        // https://github.com/lightvector/KataGo/blob/master/cpp/dataio/trainingwrite.cpp
        const double short_term_lambda = 1.0/(1.0 + num_intersections * 0.18);
        const double middle_term_lambda = 1.0/(1.0 + num_intersections * 0.06);
        const double long_term_lambda = 1.0/(1.0 + num_intersections * 0.02);

        double short_term_gamma = 1.;
        double middle_term_gamma = 1.;
        double long_term_gamma = 1.;

        for (int h = 0; h < buf_size; ++h) {
            const auto buf_idx = std::min(i + h, buf_size-1);
            const auto &curr_buf = training_data_buffer_[buf_idx];
            double sign = curr_buf.side_to_move == buf.side_to_move ? 1 : -1;
            const double avg_q = curr_buf.avg_q_value;
            const double avg_s = curr_buf.avg_score_lead;

            short_term_q += (1. - short_term_lambda) * sign * (short_term_gamma * avg_q);
            short_term_score += (1. - short_term_lambda) * sign * (short_term_gamma * avg_s);
            short_term_gamma *= short_term_lambda;

            middle_term_q += (1. - middle_term_lambda) * sign * (middle_term_gamma * avg_q);
            middle_term_score += (1. - middle_term_lambda) * sign * (middle_term_gamma * avg_s);
            middle_term_gamma *= middle_term_lambda;

            long_term_q += (1. - long_term_lambda) * sign * (long_term_gamma * avg_q);
            long_term_score += (1. - long_term_lambda) * sign * (long_term_gamma * avg_s);
            long_term_gamma *= long_term_lambda;
        }

        buf.short_avg_q = short_term_q;
        buf.middle_avg_q = middle_term_q;
        buf.long_avg_q = long_term_q;
        buf.short_avg_score = short_term_score;
        buf.middle_avg_score = middle_term_score;
        buf.long_avg_score = long_term_score;
    }

    // Set the auxiliary probablility in the buffer.
    auto aux_prob = std::vector<float>(num_intersections+1, 0);

    // Force the last aux policy is pass move.
    aux_prob[num_intersections] = 1.f;

    for (int i = buf_size-1; i >= 0; --i) {
        auto &buf = training_data_buffer_[i];
        buf.auxiliary_probabilities = aux_prob;
        aux_prob = buf.probabilities;
    }

    // output the the data.
    for (auto &buf : training_data_buffer_) {
        chunk.emplace_back(buf);
    }

    // Release the buffer.
    training_data_buffer_.clear();
}

void Search::GatherData(const GameState &state,
                        ComputationResult &result,
                        bool discard) {
    if (training_data_buffer_.size() > 9999) {
        // To many data in the buffer.
        return;
    }

    auto data = TrainingData{};
    data.version = GetTrainingVersion();
    data.mode = GetTrainingMode();
    data.discard = discard;

    data.board_size = result.board_size;
    data.komi = state.GetKomiWithPenalty();
    data.side_to_move = result.to_move;

    // Map the root eval from [0 ~ 1] to [-1 ~ 1]
    data.q_value = 2 * result.root_eval - 1.f;
    data.score_lead = result.root_score_lead;
    data.score_stddev = result.root_score_stddev;
    data.q_stddev = result.root_eval_stddev;
    data.planes = Encoder::Get().GetPlanes(state);
    data.probabilities = result.target_policy_dist;
    data.wave = state.GetWave();
    data.rule = state.GetScoringRule() == kArea ? 0.f : 1.f;
    data.kld = result.policy_kld;

    // Fill resign count
    data.accum_resign_cnt = training_data_buffer_.empty() || !result.side_resign ?
                                0 : std::rbegin(training_data_buffer_)->accum_resign_cnt + 1;

    training_data_buffer_.emplace_back(data);
}

bool Search::AdvanceToNewRootState(Search::OptionTag tag) {
    if (!root_node_) {
        return false;
    }

    const auto depth =
        int(root_state_.GetMoveNumber() - last_state_.GetMoveNumber());

    if (depth < 0) {
        return false;
    }

    auto move_list = std::stack<int>{};
    auto test = root_state_;
    for (auto i = 0; i < depth; ++i) {
        move_list.emplace(test.GetLastMove());
        test.UndoMove();
    }

    if (test.GetHash() != last_state_.GetHash() ||
           test.GetBoardSize() != last_state_.GetBoardSize()) {
        return false;
    }

    while (!move_list.empty()) {
        int vtx = move_list.top();

        auto next_node = root_node_->PopChild(vtx);
        auto p = root_node_.release();

        // Lazy tree destruction. May save a little of time when
        // dealing with large trees. We will collect these future
        // resuls after the search finished.
        group_->AddTask([p](){ delete p; });

        if (next_node) {
            root_node_.reset(next_node);
        } else {
            return false;
        }

        last_state_.PlayMove(vtx);
        move_list.pop();
    }

    if (root_state_.GetHash() != last_state_.GetHash()) {
        return false;
    }

    if (!root_node_->HasChildren()) {
        // If the root node does not have children, that means
        // it is equal to edge. We discard it.
        return false;
    }

    if ((tag & kUnreused) && param_->gumbel) {
        // Gumbel search will make tree shape weird. We need to build
        // the tree from scratch if the remaining playouts is not enough.
        // The dirichlet noise also make tree shape weird. But it should
        // be OK for most case. For example, Leela Zero reuse the tree
        // during the self-play. Look like it is no negative effect.

        int remaining_playouts = param_->playouts -
                                     (root_node_->GetVisits() - 1);
        int gumbel_thres = param_->gumbel_playouts_threshold;
        if (remaining_playouts < gumbel_thres) {
            return false;
        }
    }

    return true;
}

bool Search::HaveAlternateMoves(const float elapsed, const float limit,
                                const int cap, Search::OptionTag tag) {
    const auto &children = root_node_->GetChildren();
    const auto color = root_state_.GetToMove();

    size_t valid_cnt = 0;
    int topvisits = 0;
    float toplcb = 0.0f;
    for (const auto &child : children) {
        const auto node = child.GetPointer();

        if (!node->IsActive()) {
            continue;
        }
        ++valid_cnt;
        topvisits = std::max(topvisits, node->GetVisits());
        toplcb = std::max(toplcb, node->GetLcb(color));
    }

    if (valid_cnt == 1) {
        if (param_->analysis_verbose) {
            LOGGING << "Only one valid move, stopping early.\n";
        }
        return false;
    }
    if (param_->timemanage == TimeControl::TimeManagement::kOff ||
            !(tag & kThinking)) {
        return true;
    }

    int playouts = playouts_.load(std::memory_order_relaxed);
    int estimated_playouts = GetPlayoutsLeft(cap, tag);
    if (elapsed >= 1.0f && playouts >= 100) {
        // Wait some time. Besure that we can estimate playout rate
        // correctly.
        const double remaining = std::max(static_cast<double>(limit) - elapsed, 0.);
        const double playouts_per_sec = static_cast<double>(playouts)/elapsed;
        estimated_playouts = std::min(
            estimated_playouts,
            static_cast<int>(std::round(remaining * playouts_per_sec)));
    }

    size_t bad_cnt = 0;
    for (const auto &child : children) {
        const auto node = child.GetPointer();
        const auto visits = node->GetVisits();

        if (!node->IsActive()) {
            continue;
        }
        bool has_enough_visits =
            visits + estimated_playouts >= topvisits;
        bool has_high_winrate =
            visits > 0 ? node->GetWL(color, false) >= toplcb : false;
        if (!(has_enough_visits || has_high_winrate)) {
            ++bad_cnt;
        }
    }
    if (bad_cnt != valid_cnt - 1) {
        // We have two or above reasonable moves.
        return true;
    }

    if (param_->timemanage == TimeControl::TimeManagement::kOn) {
        // Mode "on": Will save up the current thinking time for next
        //            move if time controller doesn't reset the thinking
        //            time every move.
        if (param_->const_time > 0 || !time_control_.CanAccumulateTime(color)) {
            return true;
        }
    } else if (param_->timemanage == TimeControl::TimeManagement::kFast) {
        // Mode "fast": Always save up the current thinking time. 
    } else if (param_->timemanage == TimeControl::TimeManagement::kKeep) {
        // Mode "keep": Only save up the current thinking time in byo
        //              phase.
        if (param_->const_time > 0 || !time_control_.InByo(color)) {
            return true;
        }
    }

    if (param_->analysis_verbose) {
        LOGGING << Format("Remaining %.1f(sec) left, stopping early.\n",
                              limit - elapsed);
    }
    return false;
}

bool Search::AchieveCap(const int cap, Search::OptionTag tag) {
    // No playouts left, stopping search.
    return GetPlayoutsLeft(cap, tag) == 0;
}

int Search::GetPlayoutsLeft(const int cap, Search::OptionTag tag) {

    // Defalut we use playouts cap.
    int accumulation = playouts_.load(std::memory_order_relaxed);

    // In this condition, We disable the reuse-tree. But we use visit
    // cap instead of discarding the sub-tree. They should be equal.
    if (tag & kUnreused) {
        // The visits number is greater or equal to 1 because it
        // always includes the root visit. We should reduce it.
        accumulation = root_node_->GetVisits() - 1;
    }

    // Compute how many playouts we need.
    const auto remaining = std::max(cap - accumulation, 0);

    return remaining;
}

bool Search::StoppedByKldGain(ComputationResult &result, Search::OptionTag tag) {
    int visits_diff = result.visits - prev_kldgain_visits_;
    if (param_->kldgain_interval <= 0 ||
            visits_diff < param_->kldgain_interval) {
        return false;
    }
    UpdateComputationResult(result);

    const auto num_intersections = root_state_.GetNumIntersections();
    auto curr_target_policy = std::vector<double>(num_intersections+1);
    std::copy(std::begin(result.target_policy_dist),
                  std::begin(result.target_policy_dist) + (num_intersections+1),
                  std::begin(curr_target_policy));

    const auto kldgain = GetKlDivergence(curr_target_policy, prev_kldgain_target_policy_);
    bool should_stop = kldgain / visits_diff < param_->kldgain_per_node;

    prev_kldgain_visits_ = result.visits;
    prev_kldgain_target_policy_ = curr_target_policy;

    if (param_->fastsearch_playouts > 0 &&
            param_->fastsearch_playouts_prob > 0.0 &&
            !AchieveCap(param_->fastsearch_playouts, tag)) {
        should_stop = false;
    }
    return should_stop;
}

int Search::GetPonderPlayouts() const {
    // The factor means 'ponder_playouts = playouts * div_factor'.
    const int div_factor = std::max(1, param_->ponder_factor);

    // TODO: We should consider tree memory limit. Avoid to use
    //       too many system memory.
    const int ponder_playouts_base = std::min(param_->playouts,
                                                  kMaxPlayouts/div_factor);
    const int ponder_playouts =  ponder_playouts_base * div_factor;

    return ponder_playouts;
}

std::string Search::GetDebugMoves(std::vector<int> moves) {
    return root_node_->GetPathVerboseString(
               root_state_, root_state_.GetToMove(), moves);
}
