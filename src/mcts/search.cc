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
            const auto visits = node->GetVisits();
            if (param_->no_dcnn &&
                    visits < GetExpandThreshold(currstate)) {
                // Do the rollout only if the visits is below threshold
                // in the no dcnn mode.
                search_result.FromRollout(currstate);
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
    }

    // Not the terminated node, search the next node.
    if (node->HasChildren() && !search_result.IsValid()) {
        auto color = currstate.GetToMove();
        Node *next = nullptr;

        // Go to the next node by PUCT/UCT algoritim.
        if (param_->no_dcnn) {
            next = node->UctSelectChild(color, depth == 0, currstate);
        } else {
            next = node->PuctSelectChild(color, depth == 0);
        }
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

void Search::PrepareRootNode() {
    bool reused = AdvanceToNewRootState();

    if (!reused) {
        // Try release whole trees.
        ReleaseTree();

        // Do not reuse the tree, allocate new root node.
        root_node_ = std::make_unique<Node>(param_.get(), kPass, 1.0f);
    }

    playouts_.store(0, std::memory_order_relaxed);
    running_.store(true, std::memory_order_relaxed);

    auto node_evals = NodeEvals{};
    const bool success = root_node_->PrepareRootNode(
                             network_, root_state_, node_evals, analysis_config_);

    if (!reused && success) {
        root_node_->Update(&node_evals);
    }
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

    // Disable any noise if we forbid them.
    const bool gumbel = param_->gumbel;
    const bool dirichlet_noise = param_->dirichlet_noise;
    if (tag & kNoNoise) {
        param_->gumbel =
            param_->dirichlet_noise = false;
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
    const auto thinking_time = std::min(
                                   bound_time,
                                   time_control_.GetThinkingTime(
                                       color, board_size, move_num));

    // Will be zero if time mananger is invalid.
    const float buffer_effect = time_control_.GetBufferEffect(
                                    color, board_size, move_num);

    PrepareRootNode();

    if (param_->analysis_verbose) {
        if (param_->no_dcnn) {
            LOGGING << "Disable DCNN forwarding pipe\n";
        }
        LOGGING << Format("Reuse %d nodes\n", root_node_->GetVisits()-1);
        LOGGING << Format("Use %d threads for search\n", param_->threads);
        LOGGING << Format("Max thinking time: %.2f(sec)\n", thinking_time);
        LOGGING << Format("Max playouts number: %d\n", playouts);
    }

    if (thinking_time < timer.GetDuration() || playouts == 0) {
        // Prepare the root node will spent little time. So disable
        // the tree search if the time is up.
        running_.store(false, std::memory_order_relaxed);
    }

    for (int t = 1; t < param_->threads; ++t) {
        // SMP thread is running.
        group_->AddTask(Worker);
    }

    // Main thread is running.
    auto keep_running = running_.load(std::memory_order_relaxed);

    while (!InputPending(tag) && keep_running) {
        auto currstate = std::make_unique<GameState>(root_state_);
        auto result = SearchResult{};

        PlaySimulation(*currstate, root_node_.get(), 0, result);
        if (result.IsValid()) {
            playouts_.fetch_add(1, std::memory_order_relaxed);
        }

        if ((tag & kAnalysis) &&
                analysis_config_.interval > 0 &&
                analysis_config_.interval * 10 <
                    analysis_timer.GetDurationMilliseconds()) {
            // Output the analysis string for GTP interface, like sabaki...
            analysis_timer.Clock();
            if (root_node_->GetVisits() > 1) {
                DUMPING << root_node_->ToAnalysisString(
                                           root_state_, color, analysis_config_);
            }
        }

        const auto elapsed = (tag & kThinking) ?
                                 timer.GetDuration() : std::numeric_limits<float>::lowest();

        // TODO: Stop running when there is no alternate move.
        if (tag & kUnreused) {
            // We simply limit the root visits instead of unreuse the tree. It is
            // because that limiting the root node visits is equal to unreuse tree.
            // Notice that the visits of root node start from one. We need to
            // reduce it.
            keep_running &= (root_node_->GetVisits() - 1 < playouts);
        }
        if (param_->resign_playouts > 0 &&
                param_->resign_playouts <=
                    playouts_.load(std::memory_order_relaxed)) {
            // If someone already won the game, the Q value was not very effective
            // in the MCTS. Low playouts with policy network is good enough. Just
            // simply stop the tree search.
            float wl = root_node_->GetWL(color, false);
            keep_running &= !(wl < param_->resign_threshold ||
                                wl > (1.f-param_->resign_threshold));
        }
        keep_running &= (elapsed < thinking_time);
        keep_running &= (playouts_.load(std::memory_order_relaxed) < playouts);
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

    // Gather computation infomation and training data.
    GatherComputationResult(computation_result);

    // Save the last game state.
    last_state_ = root_state_;

    // Recover the search status.
    if (tag & kForced) {
        for (int i = 0; i < num_passes; ++i) {
            root_state_.PlayMove(kPass);
        }
    }
    if (tag & kNoNoise) {
        param_->gumbel = gumbel;
        param_->dirichlet_noise = dirichlet_noise;
    } 

    return computation_result;
}

void Search::GatherComputationResult(ComputationResult &result) const {
    const auto color = root_state_.GetToMove();
    const auto num_intersections = root_state_.GetNumIntersections();
    const auto board_size = root_state_.GetBoardSize(); 

    // Fill best moves, root eval and score.
    result.best_move = root_node_->GetBestMove(true);
    result.best_no_pass_move = root_node_->GetBestMove(false);
    result.random_move = root_node_->
                             RandomizeMoveWithGumbel(
                                 root_state_,
                                 1, param_->random_min_visits);
    result.gumbel_move = root_node_->GetGumbelMove();
    result.root_final_score = root_node_->GetFinalScore(color);
    result.root_eval = root_node_->GetWL(color, false);
    {
        auto best_node = root_node_->GetChild(result.best_move);
        if (best_node->GetVisits() >= 1) {
           result.best_eval = best_node->GetWL(color, false);
        } else {
           result.best_eval = result.root_eval;
        }
    }

    // Resize the childern status buffer.
    result.root_ownership.resize(num_intersections);
    result.root_playouts_dist.resize(num_intersections+1);
    result.root_visits.resize(num_intersections+1);
    result.target_playouts_dist.resize(num_intersections+1);

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
        float acc_target_policy = 0.0f;
        size_t target_cnt = 0;
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

            // TODO: Prune more bad children in order to get better
            //       target playouts distribution.
            if (node != nullptr &&
                    node->GetVisits() > 1 &&
                    node->IsActive()) {
                const auto prob = result.root_playouts_dist[idx];
                result.target_playouts_dist[idx] = prob;
                acc_target_policy += prob;
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
                prob /= acc_target_policy;
            }
        }
    }

    // Fill the dead strings and live strings.
    constexpr float kOwnshipThreshold = 0.75f;

    auto safe_ownership = root_state_.GetOwnership();
    auto safe_area = root_state_.GetStrictSafeArea();

    auto alive = std::vector<std::vector<int>>{};
    auto dead = std::vector<std::vector<int>>{};

    for (int idx = 0; idx < num_intersections; ++idx) {
        const auto x = idx % board_size;
        const auto y = idx / board_size;

        const auto vtx = root_state_.GetVertex(x,y);

        // owner value, 1 is mine, -1 is opp's.
        const auto owner = safe_area[idx] == true ?
                               2 * (float)(safe_ownership[idx] == color) - 1 : result.root_ownership[idx];
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
}

bool ShouldResign(GameState &state, ComputationResult &result, float resign_threshold) {
    const auto handicap = state.GetHandicap();
    const auto movenum = state.GetMoveNumber();
    const auto num_intersections = state.GetNumIntersections();

    const auto move_threshold = num_intersections / 4;
    if (movenum <= move_threshold) {
        // Too early to resign.
        return false;
    }

    if (state.IsGameOver()) {
        return false;
    }

    // TODO: Blend the dynamic komi resign threshold.

    if (handicap > 0 && state.GetToMove() == kWhite) {
        const auto handicap_resign_threshold =
                       resign_threshold / (1 + 2 * handicap);

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

bool ShouldPass(GameState &state, ComputationResult &result, bool friendly_pass) {
    if (!friendly_pass || state.GetLastMove() != kPass) {
        return false;
    }

    const auto move_threshold = state.GetNumIntersections() / 3;
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

    const auto board_size = fork_state.GetBoardSize();
    for (int y = 0; y < board_size; ++y) {
        for (int x = 0; x < board_size; ++x) {
            const auto vtx = fork_state.GetVertex(x, y);
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

int Search::ThinkBestMove() {
    auto tag = param_->reuse_tree ? kThinking : (kThinking | kUnreused);
    auto result = Computation(max_playouts_, tag);

    if (ShouldResign(root_state_, result, param_->resign_threshold)) {
        return kResign;
    }

    if (ShouldPass(root_state_, result, param_->friendly_pass)) {
        return kPass;
    }

    return result.best_move;
}

bool ShouldForbidPass(GameState &state, ComputationResult &result) {
    int to_move = result.to_move;
    auto safe_ownership = state.GetOwnership();

    for (const auto &string : result.dead_strings) {
        // All vertex in a string should be same color.
        const auto vtx = string[0];
        const auto x = state.GetX(vtx);
        const auto y = state.GetY(vtx);
        const auto idx = state.GetIndex(x, y);

        // Some opp's strings are death. Forbid the pass
        // move. Keep to eat all opp's dead strings.
        if (state.GetState(vtx) == (!to_move) &&
                safe_ownership[idx] != to_move) {
            return true;
        }
    }
    return false;
}

int Search::GetSelfPlayMove() {
    auto tag = param_->reuse_tree ? kThinking : (kThinking | kUnreused);

    int playouts = max_playouts_;
    int reduce_playouts = param_->reduce_playouts;
    float prob = param_->reduce_playouts_prob;
    if (reduce_playouts > 0 &&
            reduce_playouts < max_playouts_ &&
            Random<>::Get().Roulette<10000>(prob)) {

        const auto diff = max_playouts_ - reduce_playouts;
        auto geometric = std::geometric_distribution<>(
                             1.f/(std::sqrt((float)diff)));
        const auto v = std::min(geometric(Random<>::Get()), diff);

        // The reduce playouts must be smaller than playouts.
        playouts = std::min(
                       playouts,
                       reduce_playouts + v);
        tag = tag | kNoNoise;
    }

    if (!network_.Valid()) {
        // The network is dummy backend. The playing move is 
        // random, so we only use one tenth playouts in order
        // to reduce time.
        playouts /= 10;
    }

    // There is at least one playout for the self-play move
    // because some move select functions need at least one.
    playouts = std::max(1, playouts);

    auto result = Computation(playouts, tag);
    int move = result.best_move;

    // The game is not end. Don't play the pass move.
    if (ShouldForbidPass(root_state_, result)) {
        move = result.best_no_pass_move;
    }

    int random_moves_cnt = param_->random_moves_factor *
                               result.board_size * result.board_size;

    // Do the random move in the opening step in order to improve the
    // game state diversity.
    if (random_moves_cnt > result.movenum) {
        move = result.random_move;
    }

    // The Gumbel-Top-k trick holds more information, so we use it instead
    // of random move.
    if (root_node_->ShouldApplyGumbel()) {
        move = result.gumbel_move;
    }

    // Save the move comment.
    float root_eval = result.root_eval;
    float root_score = result.root_final_score;

    if (result.to_move == kWhite) {
        root_eval = 1.0f - root_eval;
        root_score = 0.f - root_score;
    }
    root_state_.SetComment(
        Format("%d, %.2f, %.2f",
            result.playouts,
            root_eval, root_score));

    // Push the data to buffer.
    bool record_it = !(tag & kNoNoise);
    if (record_it) {
        GatherData(root_state_, result);
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
    auto analysis_tag = kAnalysis;
    auto reuse_tag = param_->reuse_tree ?
                         kNullTag : kUnreused;
    auto ponder_tag = ponder ? kPonder : kThinking;

    auto tag = reuse_tag | ponder_tag | analysis_tag;

    // Set the current analysis config.
    analysis_config_ = analysis_config;

    int playouts = ponder == true ? GetPonderPlayouts()
                                      : max_playouts_;
    auto result = Computation(playouts, tag);

    // Disable to reuse the tree for next move.
    if (analysis_config_.MoveRestrictions()) {
        ReleaseTree();
    }

    // Clear config after finishing the search.
    analysis_config_.Clear();

    if (ShouldResign(root_state_, result, param_->resign_threshold)) {
        return kResign;
    }
    if (ShouldPass(root_state_, result, param_->friendly_pass)) {
        return kPass;
    }

    return result.best_move;
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

    // Set the all buffer values except for auxiliary probablility.
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
            if (owner == buf.side_to_move) {
                buf.ownership[idx] = 1;
            } else if (owner == !buf.side_to_move) {
                buf.ownership[idx] = -1;
            } else {
                buf.ownership[idx] = 0;
            }
        }
    }

    // Set the auxiliary probablility in the buffer.
    auto aux_prob = std::vector<float>(num_intersections+1, 0);

    // Force the last aux policy is pass move.
    aux_prob[num_intersections] = 1.f;

    for (int i = training_buffer_.size()-1; i >= 0; --i) {
        auto &buf = training_buffer_[i];

        buf.auxiliary_probabilities_index = -1;
        buf.auxiliary_probabilities = aux_prob;

        buf.probabilities_index = -1;
        aux_prob = buf.probabilities;
    }

    // output the the data.
    for (auto &buf : training_buffer_) {
        chunk.emplace_back(buf);
    }

    // Release the buffer.
    training_buffer_.clear();
}

void Search::GatherData(const GameState &state, ComputationResult &result) {
    if (training_buffer_.size() > 9999) {
        // To many data in the buffer.
        return;
    }

    auto data = Training{};
    data.version = GetTrainingVersion();
    data.mode = GetTrainingMode();

    data.board_size = result.board_size;
    data.komi = result.komi;
    data.side_to_move = result.to_move;

    // Map the root eval from [0 ~ 1] to [-1 ~ 1]
    data.q_value = 2 * result.root_eval - 1.f;
    data.planes = Encoder::Get().GetPlanes(state);
    data.probabilities = result.target_playouts_dist;

    training_buffer_.emplace_back(data);
}

bool Search::AdvanceToNewRootState() {
    if (!root_node_) {
        return false;
    }

    if (param_->gumbel ||
            param_->dirichlet_noise ||
            param_->root_dcnn) {
        // Need to re-build the trees if we apply noise or Gumbel. Reuse
        // the tree will ignore them. The root_dcnn option only apply
        // the network at root. The tree shape of root is different
        // from children.
        return false;
    }

    const auto temp_diff = param_->policy_temp -
                               param_->root_policy_temp;
    if (std::abs(temp_diff) > 1e-4f) {
        // The tree shape is different if the temperature is different.
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
    return true;
}

int Search::GetPonderPlayouts() const {
    // We don't need to consider the NN cache size to set number
    // of ponder playouts that because we apply lazy tree destruction
    // and reuse the tree. They can efficiently use the large tree.
    // Set the greatest number as we can.

    // The factor means 'ponder_playouts = playouts * div_factor'.
    const int div_factor = std::max(1, param_->ponder_factor);

    // TODO: We should consider tree memory limit. Avoid to use
    //       too many system memory.
    const int ponder_playouts_base = std::min(param_->playouts,
                                                  kMaxPlayouts/div_factor);
    const int ponder_playouts =  ponder_playouts_base * div_factor;

    return ponder_playouts;
}

int Search::GetExpandThreshold(GameState &state) const {
    const auto board_size = state.GetBoardSize();

    if (param_->expand_threshold >= 0) {
        return param_->expand_threshold;
    }

    // We tend to select the large 'Expand Threshold' in order
    // to converge the average winrate. The other engine may
    // select the little value becuase they apply RAVE method.
    return std::max(20 + 2 * (board_size-9), 20);
}

std::string Search::GetDebugMoves(std::vector<int> moves) {
    return root_node_->GetPathVerboseString(
               root_state_, root_state_.GetToMove(), moves);
}
