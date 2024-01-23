#include <numeric>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <stack>
#include <random>
#include <cmath>

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

    analysis_config_.Clear();
    last_state_ = root_state_;
    root_node_.reset(nullptr);

    group_ = std::make_unique<ThreadGroup<void>>(&ThreadPool::Get());

    max_playouts_ = param_->playouts;
    playouts_.store(0, std::memory_order_relaxed);
}

void Search::PlaySimulation(GameState &currstate, Node *const node,
                            const int depth, SearchResult &search_result) {
    node->IncrementThreads();

    const bool end_by_passes = currstate.GetPasses() >= 2;
    if (end_by_passes) {
        search_result.FromGameOver(currstate);
    }

    // Terminated node, try to expand it.
    if (node->Expandable()) {
        const auto last_move = currstate.GetLastMove();

        if (end_by_passes) {
            if (node->SetTerminal() &&
                    search_result.IsValid()) {
                // The game is over, setting the game result value.
                node->ApplyEvals(search_result.GetEvals());
            }
        } else if (last_move != kPass &&
                       currstate.IsSuperko()) {
            // Prune this superko move.
            node->Invalidate();
        } else {
            const bool has_children = node->HasChildren();

            // If we can not expand the node, it means that another thread
            // is under this node. Skip the simulation stage this time. However,
            // it still has a chance do PUCT/UCT.
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

        // Go to the next node by PUCT algoritim.
        next = node->PuctSelectChild(color, depth == 0);

        auto vtx = next->GetVertex();
        currstate.PlayMove(vtx, color);

        // Recursive calls.
        PlaySimulation(currstate, next, depth+1, search_result);
    }

    // Now Update this node if it valid.
    if (search_result.IsValid()) {
        node->Update(search_result.GetEvals());
    }
    node->DecrementThreads();
}

void Search::PrepareRootNode(Search::OptionTag tag) {
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

    // Compute the current root visits distribution. We can compute
    // the KLD gaining from this distribution.
    int visits;
    last_root_dist_ = GetRootDistribution(visits);

    // We should get this root policy from the NN cache. The softmax
    // temperature of 'root_evals_' may be not 1 so we need to
    // compute it again.
    auto netlist = network_.GetOutput(root_state_, Network::kRandom, 1);
    auto num_intersections = root_state_.GetNumIntersections();
    root_raw_probabilities_.resize(num_intersections+1);

    std::copy(std::begin(netlist.probabilities),
                  std::begin(netlist.probabilities) + num_intersections,
                  std::begin(root_raw_probabilities_));
    root_raw_probabilities_[num_intersections] = netlist.pass_probability;
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
    int num_passes = 0;
    if (tag & kForced) {
        while (root_state_.GetPasses() >= 2) {
            // Remove double pass move.
            root_state_.UndoMove();
            root_state_.UndoMove();
            num_passes+=2;
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
    computation_result.playouts = 0;
    computation_result.seconds = 0.f;
    computation_result.threads = param_->threads;
    computation_result.batch_size = param_->batch_size;

    if (root_state_.IsGameOver()) {
        // Always reture pass move if the passese number is greater than two.
        computation_result.best_move = kPass;
        return computation_result;
    }

    if (tag & kThinking) {
        auto book_move = kNullVertex;
        if (Book::Get().Probe(root_state_, book_move)) {
            // Current game state is found in book.
            computation_result.best_move = book_move;
            return computation_result;
        }
    }

    // The SMP workers run on every threads except for the main thread.
    const auto Worker = [this]() -> void {
        while(running_.load(std::memory_order_relaxed)) {
            auto currstate = std::make_unique<GameState>(root_state_);
            auto result = SearchResult{};
            PlaySimulation(*currstate, root_node_.get(), 0, result);
            if (result.IsValid()) {
                playouts_.fetch_add(1, std::memory_order_relaxed);
            }
        };
    };

    Timer timer; // main timer
    Timer analysis_timer; // for analysis

    // Set the time control.
    time_control_.Clock();
    time_control_.SetLagBuffer(
        std::max(param_->lag_buffer, time_control_.GetLagBuffer()));

    // Clean the timer.
    timer.Clock();
    analysis_timer.Clock();

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

    // Will be zero if time mananger is invalid.
    const float buffer_effect = time_control_.GetBufferEffect(
                                    color, board_size, move_num);

    PrepareRootNode(tag);

    if (param_->analysis_verbose) {
        LOGGING << Format("Reuse %d nodes\n", root_node_->GetVisits()-1);
        LOGGING << Format("Use %d threads for search\n", param_->threads);
        LOGGING << Format("Max thinking time: %.2f(sec)\n", thinking_time);
        LOGGING << Format("Max playouts number: %d\n", playouts);
    }

    if (thinking_time < timer.GetDuration() || AchieveCap(playouts, tag)) {
        // Prepare the root node spent little time. Disable the
        // tree search if the time is up.
        running_.store(false, std::memory_order_relaxed);
    }

    for (int t = 1; t < param_->threads; ++t) {
        // SMP thread is running.
        group_->AddTask(Worker);
    }

    // Main thread is running.
    auto last_updating_visits = root_node_->GetVisits();
    auto keep_running = running_.load(std::memory_order_relaxed);

    while (!InputPending(tag) && keep_running) {
        auto currstate = std::make_unique<GameState>(root_state_);
        auto result = SearchResult{};

        PlaySimulation(*currstate, root_node_.get(), 0, result);
        if (result.IsValid()) {
            playouts_.fetch_add(1, std::memory_order_relaxed);
        }

        const auto root_visits = root_node_->GetVisits();
        const auto elapsed = timer.GetDuration();

        if ((tag & kAnalysis) &&
                analysis_config_.interval > 0 &&
                analysis_config_.interval * 10 <
                    analysis_timer.GetDurationMilliseconds()) {
            // Output the analysis string for GTP interface, like sabaki...
            analysis_timer.Clock();
            if (root_visits > 1) {
                DUMPING << root_node_->ToAnalysisString(
                                           root_state_, color, analysis_config_);
            }
        }
        if (param_->resign_playouts > 0 &&
                AchieveCap(param_->resign_playouts, tag)) {
            // If someone already won the game, the Q value was not very effective
            // in the MCTS. Low playouts with policy network is good enough. Just
            // simply stop the tree search.
            float wl = root_node_->GetWL(color, false);
            keep_running &= !(wl < param_->resign_threshold ||
                                wl > (1.f-param_->resign_threshold));
        }
        if (tag & kThinking) {
            keep_running &= (elapsed < thinking_time);

            const int check_freq = 100;
            if (root_visits - last_updating_visits >= check_freq) {
                last_updating_visits = root_visits;
                keep_running &= HaveAlternateMoves(elapsed, thinking_time);
            }
        }
        keep_running &= !AchieveCap(playouts, tag);
        keep_running &= running_.load(std::memory_order_relaxed);
    };

    running_.store(false, std::memory_order_release);

    // Wait for all threads to join the main thread.
    group_->WaitToJoin();

    const auto played_playouts =
                   playouts_.load(std::memory_order_relaxed);

    if (tag & kThinking) {
        time_control_.TookTime(color);

        // Try to adjust the lag buffer. Avoid to be time-out for
        // the last move.
        float curr_lag_buf = time_control_.GetLagBuffer();
        const auto elapsed = timer.GetDuration();

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
    if (tag & kAnalysis) {
        // Output the last analysis verbose because the MCTS may
        // be finished in the short time. It can help the low playouts
        // MCTS to show current analysis graph in GUI.
        if (root_node_->GetVisits() > 1) {
            DUMPING << root_node_->ToAnalysisString(
                           root_state_, color, analysis_config_);
        }
    }

    if (param_->analysis_verbose) {
        LOGGING << root_node_->ToVerboseString(root_state_, color);
        LOGGING << " * Time Status:\n";
        LOGGING << "  " << time_control_.ToString();
        LOGGING << "  spent: " << timer.GetDuration() << "(sec)\n";
        LOGGING << "  speed: " << (float)played_playouts /
                                      timer.GetDuration() << "(p/sec)\n";
        LOGGING << "  playouts: " << played_playouts << "\n";
    }

    // Record perfomance infomation.
    computation_result.seconds = timer.GetDuration();
    computation_result.playouts = played_playouts;

    // Gather computation information and training data.
    GatherComputationResult(computation_result);

    // Save the last game state.
    last_state_ = root_state_;

    // Recover the search status.
    if (tag & kForced) {
        for (int i = 0; i < num_passes; ++i) {
            root_state_.PlayMove(kPass);
        }
    }
    if (tag & kNoExploring) {
        std::swap(param_, no_exploring_param_);
    }

    return computation_result;
}

void Search::GatherComputationResult(ComputationResult &result) const {
    const auto color = root_state_.GetToMove();
    const auto num_intersections = root_state_.GetNumIntersections();

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
                                 RandomMoveWithLogitsQ(
                                     root_state_, 1.f);
    } else {
        // According to "On Strength Adjustment for MCTS-Based Programs",
        // pruning the low visits moves. It can help to avoid to play the
        // blunder move. The Elo range should be between around 0 ~ 1000.
        result.random_move = root_node_->
                                 RandomMoveProportionally(
                                     param_->random_moves_temp,
                                     param_->random_q_decay,
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

    // Resize the childern status buffer.
    result.root_ownership.resize(num_intersections, 0);
    result.root_playouts_dist.resize(num_intersections+1, 0);
    result.root_visits.resize(num_intersections+1, 0);
    result.target_playouts_dist.resize(num_intersections+1, 0);

    // Fill ownership.
    auto ownership = root_node_->GetOwnership(color);
    std::copy(std::begin(ownership),
                  std::begin(ownership) + num_intersections,
                  std::begin(result.root_ownership));

    // Fill root visits.
    auto parentvisits = 0;
    const auto &children = root_node_->GetChildren();
    for (const auto &child : children) {
        const auto node = child.Get();
        const auto visits = node->GetVisits();
        const auto vertex = node->GetVertex();

        parentvisits += visits;
        if (vertex == kPass) {
            result.root_visits[num_intersections] = visits;
            continue;
        }

        const auto x = root_state_.GetX(vertex);
        const auto y = root_state_.GetY(vertex);
        const auto index = root_state_.GetIndex(x, y);

        result.root_visits[index] = visits;
    }

    // Fill raw probabilities.
    if (parentvisits != 0) {
        for (int idx = 0; idx < num_intersections+1; ++idx) {
            result.root_playouts_dist[idx] =
                (float)(result.root_visits[idx]) / parentvisits;
        }
    } else {
        for (int idx = 0; idx < num_intersections+1; ++idx) {
            result.root_playouts_dist[idx] = 1.f/(num_intersections+1);
        }
    }

    // Fill target distribution.
    if (parentvisits != 0 &&
            (root_node_->ShouldApplyGumbel() ||
             param_->always_completed_q_policy)) {
        result.target_playouts_dist =
            root_node_->GetProbLogitsCompletedQ(root_state_);
    } else {
        float accm_target_policy = 0.0f;
        size_t target_cnt = 0;
        for (int idx = 0; idx < num_intersections+1; ++idx) {
            int vertex;

            if (idx == num_intersections) {
                vertex = kPass;
            } else {
                vertex = root_state_.IndexToVertex(idx);
            }

            auto node = root_node_->GetChild(vertex);

            // TODO: Prune more bad children in order to get better
            //       target playouts distribution.
            if (node != nullptr &&
                    node->GetVisits() > 1 &&
                    node->IsActive()) {
                const auto prob = result.root_playouts_dist[idx];
                result.target_playouts_dist[idx] = prob;
                accm_target_policy += prob;
                target_cnt += 1;
            } else {
                result.target_playouts_dist[idx] = 0.0f;
            }
        }

        if (target_cnt == 0) {
            // All moves are pruned. We directly use the raw
            // distribution.
            result.target_playouts_dist = result.root_playouts_dist;
        } else {
            for (auto &prob : result.target_playouts_dist) {
                prob /= accm_target_policy;
            }
        }
    }

    result.policy_kld = GetKlDivergence(
        result.target_playouts_dist, root_raw_probabilities_);

    // Fill the dead strings and live strings.
    constexpr float kOwnshipThreshold = 0.75f;

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


        if (owner > kOwnshipThreshold) {
            // It is my territory.
            if (color == state) {
                alive.emplace_back(root_state_.GetStringList(vtx));
            } else if ((!color) == state) {
                dead.emplace_back(root_state_.GetStringList(vtx));
            }
        } else if (owner < -kOwnshipThreshold) {
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

    // Select the capture all dead move.
    auto fill_moves = std::vector<int>{};
    auto raw_ownership = root_state_.GetRawOwnership();

    for (int idx = 0; idx < num_intersections; ++idx) {
        const auto vtx = root_state_.IndexToVertex(idx);
        const auto raw_owner = raw_ownership[idx];

        // owner value, 1 is mine, -1 is opp's.
        const auto owner = safe_area[idx] == true ?
                               2 * (float)(safe_ownership[idx] == color) - 1 :
                               result.root_ownership[idx];
        if (owner > kOwnshipThreshold &&
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
        result.capture_all_dead_move = *std::begin(fill_moves);
    }
}

bool ShouldResign(GameState &state, ComputationResult &result, Parameters *param) {
    const auto movenum = state.GetMoveNumber();
    const auto num_intersections = state.GetNumIntersections();
    const auto board_size = state.GetBoardSize();

    const auto move_threshold = num_intersections / 4;
    if (movenum <= move_threshold) {
        // Too early to resign.
        return false;
    }

    if (state.IsGameOver()) {
        return false;
    }

    auto resign_threshold = param->resign_threshold;

    // Seem the 7 is the fair komi for most board size.
    const auto virtual_fair_komi = 7.0f;
    const auto komi_diff = state.GetKomi() - virtual_fair_komi;
    const auto to_move = state.GetToMove();
    if ((komi_diff > 0.f && to_move == kBlack) ||
            (komi_diff < 0.f && to_move == kWhite)) {
        // Shift the resign threshold by komi. Compensate for
        // komi disadvantages.
        auto blend_ratio = std::min(1.0f, movenum / (0.6f * num_intersections));
        resign_threshold =
            blend_ratio * resign_threshold +
            (1.0f - blend_ratio) * resign_threshold/
                std::max(1.f, std::abs(5.f * komi_diff/board_size));
    }

    const auto handicap = state.GetHandicap();
    if (handicap > 0 && to_move == kWhite) {
        const auto handicap_resign_threshold =
                       (resign_threshold-1.f) * handicap/20.f;
        auto blend_ratio = std::min(1.0f, movenum / (0.6f * num_intersections));
        auto blended_resign_threshold = blend_ratio * resign_threshold +
                                            (1.0f - blend_ratio) * handicap_resign_threshold;
        if (result.best_eval > blended_resign_threshold) {
            // Allow lower eval for white in handicap games
            // where opp may fumble.
            return false;
        }
    }

    if (result.best_eval > resign_threshold) {
        return false;
    }

    return true;
}

bool ShouldPass(GameState &state, ComputationResult &result, Parameters *param) {
    if (!(param->friendly_pass) || state.GetLastMove() != kPass) {
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
            // At least one string in atari, the game
            // is not over yet.
            return false;
        } else if (fork_state.GetState(vtx) == kEmpty) {
            // This empty point does not belong to any
            // side. It is the dame.
            num_dame += 1;
        }
    }
    (void) num_dame;

    // Compute the final score.
    float score = fork_state.GetFinalScore();

    if (state.GetToMove() == kWhite) {
        score = 0 - score;
    }

    if (score > 0.1f) {
        // We already win the game. We will play the pass move.
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

    int best_move = result.best_move;
    const int random_moves_cnt = param_->random_moves_factor *
                                     root_state_.GetNumIntersections();
    if (root_state_.GetMoveNumber() < random_moves_cnt) {
        // TODO: It is possible the pass move. Should we prune it?
        best_move = result.random_move;
    }
    if (ShouldPass(root_state_, result, param_.get())) {
        // Quickly play the move if we have already won the
        // game.
        best_move = kPass;
    }
    if (param_->capture_all_dead &&
            best_move == kPass &&
            result.capture_all_dead_move != kNullVertex) {
        // Refuse playing the pass move until all dead stones
        // are removed.
        best_move = result.capture_all_dead_move;
    }
    return best_move;
}

int Search::ThinkBestMove() {
    auto tag = param_->reuse_tree ? kThinking : (kThinking | kUnreused);
    int best_move = GetBestMove(max_playouts_, tag);
    return best_move;
}

bool ShouldForbidPass(GameState &state,
                      ComputationResult &result,
                      NodeEvals &root_evals) {

    const auto num_intersections = state.GetNumIntersections();
    const auto move_threshold = num_intersections / 3;
    if (state.GetMoveNumber() <= move_threshold) {
        // Too early to pass.
        return true;
    }

    int to_move = result.to_move;
    auto safe_ownership = state.GetOwnership();

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

    constexpr float kRawOwnshipThreshold = 0.8f; // ~90%

    for (int idx = 0; idx < num_intersections; ++idx) {
        float owner = root_evals.black_ownership[idx];
        if (to_move == kWhite) {
            owner = 0.f - owner;
        }

        // Some opp's stone are not really alive. Keep to
        // eat these stones.
        if (owner >= kRawOwnshipThreshold &&
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
                // Too large empty group.
                return true;
            }
        }
    }

    return false;
}

int Search::GetSelfPlayMove() {
    // We always reuse the sub-tree at fast search phase. The
    // kUnreused option doesn't mean discarding the sub-tree.
    // It means visit cap (The search result is as same as
    // "playout cap" + "discard the tree"). The default is playouts
    // cap. If the reuse tag is true, it is visit cap oscillation.
    // Every visit of fast search phase may be different. If the
    // reuse tag is false, it is playout cap oscillation which
    // is used by KataGo. Please see here, https://arxiv.org/abs/1902.10565v2
    auto tag = param_->reuse_tree ?
                   kNullTag : kUnreused;

    const int random_moves_cnt = param_->random_moves_factor *
                                     root_state_.GetNumIntersections();
    bool is_opening_random = root_state_.GetMoveNumber() < random_moves_cnt;

    // Decide the playouts number first. Default is max
    // playouts. May use the lower playouts instead of it.
    int playouts = max_playouts_;

    if (param_->reduce_playouts > 0 &&
            param_->reduce_playouts < max_playouts_ &&
            Random<>::Get().Roulette<10000>(param_->reduce_playouts_prob)) {

        // The reduce playouts must be smaller than default
        // playouts. It is fast search phase so we also disable
        // all exploring settings.
        playouts = std::min(
            playouts, param_->reduce_playouts);
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

    // Do the random move in the opening stage in order to improve the
    // game state diversity. The Gumbel noise may be good enough. Don't
    // play the random move if 'is_gumbel' is true. Or do random move
    // when fast search phase.
    if ((is_opening_random && !is_gumbel) ||
            ((tag & kNoExploring) &&
                 Random<>::Get().Roulette<10000>(param_->random_fastsearch_prob))) {
        if (!(forbid_pass && result.random_move == kPass)) {
            move = result.random_move;
        }
    }

    // If the 'discard_it' is true, we will discard the current training
    // data. It is because that the quality of current data is bad. To
    // discard it can improve the network performance.
    bool discard_it = false;
    float root_eval = result.root_eval;
    float root_score = result.root_score_lead;
    if (tag & kNoExploring) {
        // It is fast search of "Playout Cap Randomization". Do
        // not record the low quality datas.
        discard_it = true;
    }
    if (root_eval < param_->resign_threshold ||
            root_eval > (1.f-param_->resign_threshold)) {
        // Someone already won the game. Do not record this kind
        // of positions too much to avoid introducing pathological
        // biases in the training data
        if (Random<>::Get().Roulette<10000>(param_->resign_discard_prob)) {
            discard_it = true;
        }
    }
    if (result.to_move == kWhite) {
        // Always record the black's view point.
        root_eval = 1.0f - root_eval;
        root_score = 0.f - root_score;
    }

    // TODO: Use "Policy Surprise Weighting" instead of
    //       "Playout Cap Randomization". See the document
    //       here, https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md
    const float record_kld = result.policy_kld;
    const char discard_char = discard_it ? 'F' : 'T';
    const int root_visits = root_node_->GetVisits();

    // Save the move comment in the SGF file.
    root_state_.SetComment(
        Format("%d, %d, %.2f, %.2f, %.2f, %c",
            result.playouts, root_visits,
            root_eval, root_score, record_kld, discard_char));

    // Push the data to buffer.
    GatherData(root_state_, result, discard_it);

    return move;
}

void Search::TryPonder() {
    if (param_->ponder) {
        // The ponder mode always reuses the tree.
        Computation(GetPonderPlayouts(), kPonder);
    }
}

int Search::Analyze(bool ponder, AnalysisConfig &analysis_config) {
    auto analysis_tag = kAnalysis;
    auto reuse_tag = param_->reuse_tree ?
                         kNullTag : kUnreused;
    auto ponder_tag = ponder ? kPonder : kThinking;

    auto tag = reuse_tag | ponder_tag | analysis_tag;

    // Set the current analysis config.
    analysis_config_ = analysis_config;

    // The tree shape may be different with last move.
    if (analysis_config_.MoveRestrictions()) {
        ReleaseTree();
    }

    int playouts = ponder == true ? GetPonderPlayouts()
                                      : max_playouts_;
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
    training_buffer_.clear();
}

void Search::SaveTrainingBuffer(std::string filename, GameState &end_state) {
    auto file = std::ofstream{};
    file.open(filename, std::ios_base::app);

    if (!file.is_open()) {
        LOGGING << "Fail to create the file: " << filename << '!' << std::endl;
        return;
    }

    auto chunk = std::vector<Training>{};
    GatherTrainingBuffer(chunk, end_state);

    for (auto &buf : chunk) {
        buf.StreamOut(file);
    }
    file.close();
}

void Search::GatherTrainingBuffer(std::vector<Training> &chunk, GameState &end_state) {

    // Compute the final status positions. We do not remove the dead stones.
    auto ownership = end_state.GetOwnership();

    auto num_intersections = end_state.GetNumIntersections();
    auto winner = kUndecide;
    auto black_final_score = 0.f;

    // Compute the final score.
    for (const auto owner : ownership) {
        if (owner == kBlack) {
            black_final_score += 1;
        } else if (owner == kWhite) {
            black_final_score -= 1;
        }
    }
    black_final_score -= end_state.GetKomi();

    // Get the player who won the game.
    if (std::abs(black_final_score) < 1e-4f) {
        winner = kDraw;
    } else if (black_final_score > 0) {
        winner = kBlackWon;
    } else if (black_final_score < 0) {
        winner = kWhiteWon;
    }

    // Set the most buffer values.
    const int buf_size = training_buffer_.size();
    for (int i = 0; i < buf_size; ++i) {
        assert(winner != kUndecide);

        auto &buf = training_buffer_[i];
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
            auto &w_buf = training_buffer_[w_index];

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
        auto &buf = training_buffer_[i];
        const auto board_size = buf.board_size;

        // Please see here,
        // https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md#short-term-value-and-score-targets
        double short_term_q = 0;
        double middle_term_q = 0;
        double long_term_q = 0;

        double short_term_score = 0;
        double middle_term_score = 0;
        double long_term_score = 0;

        const int short_time_horizon = 0.4f * board_size;
        const int middle_time_horizon = 1 * board_size;
        const int long_time_horizon = 3 * board_size;

        const double lambda = 1. - 1./std::max(2, short_time_horizon);
        double gamma = 1.;

        for (int h = 0; h < long_time_horizon; ++h) {
            int buf_idx = std::min(i+h, buf_size-1);
            auto &curr_buf = training_buffer_[buf_idx];

            if (curr_buf.side_to_move == buf.side_to_move) {
                if (h < short_time_horizon) {
                    short_term_q += (gamma * curr_buf.avg_q_value);
                    short_term_score += (gamma * curr_buf.avg_score_lead);
                }
                if (h < middle_time_horizon) {
                    middle_term_q += (gamma * curr_buf.avg_q_value);
                    middle_term_score += (gamma * curr_buf.avg_score_lead);
                }
                long_term_q += (gamma * curr_buf.avg_q_value);
                long_term_score += (gamma * curr_buf.avg_score_lead);
            } else {
                if (h < short_time_horizon) {
                    short_term_q -= (gamma * curr_buf.avg_q_value);
                    short_term_score -= (gamma * curr_buf.avg_score_lead);
                }
                if (h < middle_time_horizon) {
                    middle_term_q -= (gamma * curr_buf.avg_q_value);
                    middle_term_score -= (gamma * curr_buf.avg_score_lead);
                }
                long_term_q -= (gamma * curr_buf.avg_q_value);
                long_term_score -= (gamma * curr_buf.avg_score_lead);
            }
            gamma *= lambda;
        }

        buf.short_avg_q = short_term_q * (1. - lambda);
        buf.middle_avg_q = middle_term_q * (1. - lambda);
        buf.long_avg_q = long_term_q * (1. - lambda);
        buf.short_avg_score = short_term_score * (1. - lambda);
        buf.middle_avg_score = middle_term_score * (1. - lambda);
        buf.long_avg_score = long_term_score * (1. - lambda);
    }

    // Set the auxiliary probablility in the buffer.
    auto aux_prob = std::vector<float>(num_intersections+1, 0);

    // Force the last aux policy is pass move.
    aux_prob[num_intersections] = 1.f;

    for (int i = buf_size-1; i >= 0; --i) {
        auto &buf = training_buffer_[i];
        buf.auxiliary_probabilities = aux_prob;
        aux_prob = buf.probabilities;
    }

    // output the the data.
    for (auto &buf : training_buffer_) {
        chunk.emplace_back(buf);
    }

    // Release the buffer.
    training_buffer_.clear();
}

void Search::GatherData(const GameState &state,
                        ComputationResult &result,
                        bool discard) {
    if (training_buffer_.size() > 9999) {
        // To many data in the buffer.
        return;
    }

    auto data = Training{};
    data.version = GetTrainingVersion();
    data.mode = GetTrainingMode();
    data.discard = discard;

    data.board_size = result.board_size;
    data.komi = result.komi;
    data.side_to_move = result.to_move;

    // Map the root eval from [0 ~ 1] to [-1 ~ 1]
    data.q_value = 2 * result.root_eval - 1.f;
    data.score_lead = result.root_score_lead;
    data.score_stddev = result.root_score_stddev;
    data.q_stddev = result.root_eval_stddev;
    data.planes = Encoder::Get().GetPlanes(state);
    data.probabilities = result.target_playouts_dist;
    data.wave = state.GetWave();
    data.rule = state.GetRule();
    data.kld = result.policy_kld;

    training_buffer_.emplace_back(data);
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

    if (param_->relative_rank >= 0) {
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

        int remaining_playouts = max_playouts_ -
                                     (root_node_->GetVisits() - 1);
        int gumbel_thres = param_->gumbel_playouts_threshold;
        if (remaining_playouts < gumbel_thres) {
            return false;
        }
    }

    return true;
}

bool Search::HaveAlternateMoves(float elapsed, float limit) {
    const auto &children = root_node_->GetChildren();
    if (children.size() == 1) {
        // Only one (pass) move.
        return false;
    }

    int visits;
    last_root_dist_ = GetRootDistribution(visits);

    if (last_root_dist_.size() <= 1) {
        // Be sure that there are at least two nodes.
        return true;
    }
    if (elapsed <= 1.0f || param_->timemanage == "off") {
        // A second for estimating playouts may be more precision.
        return true;
    }

    auto sorted_dist = last_root_dist_;
    std::sort(std::rbegin(sorted_dist), std::rend(sorted_dist));

    const double remaining = limit - elapsed;
    const double playouts = playouts_.load(std::memory_order_relaxed);
    const double playouts_per_sec = playouts/elapsed;
    const double estimated_playouts = remaining * playouts_per_sec;

    double top_visits = visits * sorted_dist[0];
    double sec_visits = visits * sorted_dist[1];
    double thres = 0.5;

    if (sorted_dist[0] > 0.9) {
        // Current distribution is too sharp. We simply stop the
        // search for saving thinking time.
        double cap_visits = sec_visits + estimated_playouts;
        if (thres * top_visits > cap_visits) {
            return false;
        }
    }
    return true;
}

bool Search::AchieveCap(const int cap, Search::OptionTag tag) {
    const auto playouts = playouts_.load(std::memory_order_relaxed);
    const auto visits = root_node_->GetVisits();
    bool should_stop = false;

    if (tag & kUnreused) {
        // Disable the reuse-tree mode. But we use visit cap instead of
        // discarding the sub-tree. They should be equal.
        if (visits - 1 >= cap) {
            // The visits number is greater or equal to 1 because the
            // it always includes the root visit. We should reduce it.
            should_stop |= true;
        }
    }
    if (playouts >= cap) {
        // It is playout cap. In the most case, we should be more easy
        // to achieve visit cap.
        should_stop |= true;
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

std::vector<double> Search::GetRootDistribution(int &parentvisits) {
    auto result = std::vector<double>{};
    const auto &children = root_node_->GetChildren();

    parentvisits = 0;
    for (const auto &child : children) {
        auto node = child.Get();
        int visits = 0;
        if (node && node->IsActive()) {
            visits = child.GetVisits();
        }
        parentvisits += visits;
        result.emplace_back(visits);
    }
    if (parentvisits == 0) {
        return std::vector<double>{};
    }

    for (auto &v : result) {
        v /= parentvisits;
    }
    return result;
}
