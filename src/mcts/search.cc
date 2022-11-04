#include <numeric>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <stack>

#include "mcts/search.h"
#include "neural/encoder.h"
#include "utils/log.h"
#include "utils/format.h"
#include "book/book.h"

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
                            Node *const root_node, SearchResult &search_result) {
    node->IncrementThreads();

    const bool end_by_passes = currstate.GetPasses() >= 2;
    if (end_by_passes) {
        search_result.FromGameOver(currstate);
    }

    // Terminated node, try to expand it. 
    if (node->Expandable()) {
        if (end_by_passes) {
            if (node->SetTerminal() &&
                    search_result.IsValid()) {
                // The game is over, setting the game result value.
                node->ApplyEvals(search_result.GetEvals());
            }
        } else {
            const auto visits = node->GetVisits();
            if (param_->no_dcnn &&
                    visits < GetExpandThreshold(currstate)) {
                // Do the rollout only if we do not expand this child.
                search_result.FromRollout(currstate);
            } else {
                const bool have_children = node->HaveChildren();

                // If we can not expand the node, it means that another thread
                // is expanding this node. Skip the simulation this time.
                auto node_evals = NodeEvals{};
                const bool success = node->ExpandChildren(network_, currstate,
                                                              node_evals, analysis_config_, false);

                if (!have_children && success) {
                    search_result.FromNetEvals(node_evals);
                }
            }
        }
        if (search_result.IsValid() &&
                param_->first_pass_bonus) {
            search_result.AddPassBouns(currstate);
        }
    }

    // Not the terminate node, search the next node.
    if (node->HaveChildren() && !search_result.IsValid()) {
        auto color = currstate.GetToMove();
        Node *next = nullptr;
        if (playouts_.load(std::memory_order_relaxed) < param_->cap_playouts) {
            // Go to the next node by best polcy.
            next = node->ProbSelectChild();
        } else {
            // Go to the next node by PUCT/UCT algoritim.
            if (param_->no_dcnn) {
                next = node->UctSelectChild(color, node == root_node, currstate);
            } else {
                next = node->PuctSelectChild(color, node == root_node);
            }
        }
        auto vtx = next->GetVertex();

        currstate.PlayMove(vtx, color);
        PlaySimulation(currstate, next, root_node, search_result);
    }

    // Now Update this node.
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
        root_node_ = std::make_unique<Node>(kPass, 1.0f);
        root_node_->SetParameters(param_.get());
    }

    playouts_.store(0, std::memory_order_relaxed);
    running_.store(true, std::memory_order_relaxed);

    auto node_evals = NodeEvals{};
    auto root_noise = std::vector<float>{}; // unused
    const bool success = root_node_->PrepareRootNode(
                             network_, root_state_, node_evals, analysis_config_, root_noise);

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
        while (root_state_.GetLastMove() == kPass) {
            root_state_.UndoMove();
            num_passes++;
        }
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
        auto book_move = Book::Get().Probe(root_state_);
        if (book_move != kPass) {
            // Current game state is found in book.
            computation_result.best_move = book_move;
            return computation_result;
        }
    }

    // The SMP worker runs on every threads except the main thread.
    const auto Worker = [this]() -> void {
        while(running_.load(std::memory_order_relaxed)) {
            auto currstate = std::make_unique<GameState>(root_state_);
            auto result = SearchResult{};
            PlaySimulation(*currstate, root_node_.get(), root_node_.get(), result);
            if (result.IsValid()) {
                playouts_.fetch_add(1, std::memory_order_relaxed);
            }
        };
    };

    Timer timer;
    Timer analyze_timer; // for analyzing

    // Set the time control.
    time_control_.SetLagBuffer(param_->lag_buffer);
    time_control_.Clock();
    timer.Clock();
    analyze_timer.Clock();

    // Compute max thinking time.
    const float bound_time = (param_->const_time > 0 &&
                                 time_control_.IsInfiniteTime(color)) ?
                                     param_->const_time : std::numeric_limits<float>::max();

    const auto thinking_time = std::min(
                                   bound_time,
                                   time_control_.GetThinkingTime(color, board_size, move_num));
    PrepareRootNode();

    if (param_->analysis_verbose) {
        if (param_->no_dcnn) {
            LOGGING << "Disable DCNN forwarding pipe\n";
        }
        LOGGING << Format("Reuse %d nodes\n", root_node_->GetVisits()-1);
        LOGGING << Format("Use %d threads for search\n", param_->threads);
        LOGGING << Format("Max thinking time: %.0f(sec)\n", thinking_time);
        LOGGING << Format("Max playouts number: %d\n", playouts);
    }

    if (thinking_time < timer.GetDuration() || playouts == 0) {
        // Prepare the root node will spent little time. So disable
        // to do tree search if the time is up.
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

        PlaySimulation(*currstate, root_node_.get(), root_node_.get(), result);
        if (result.IsValid()) {
            playouts_.fetch_add(1, std::memory_order_relaxed);
        }

        if ((tag & kAnalyze) &&
                analyze_timer.GetDurationMilliseconds() > analysis_config_.interval * 10) {
            // Output analysis verbose for GTP interface, like sabaki...
            analyze_timer.Clock();
            DUMPING << root_node_->ToAnalysisString(root_state_, color, analysis_config_);
        }

        const auto elapsed = (tag & kThinking) ?
                                 timer.GetDuration() : std::numeric_limits<float>::lowest();

        // TODO: Stop running when there are no alternate move.
        if (tag & kUnreused) {
            // We simply limit the root visits instead of unreuse the tree. It is
            // because that limiting the root node visits is equal to unreuse tree.
            // Notice that the visits of root node start from one. We need to
            // reduce it.
            keep_running &= (root_node_->GetVisits() - 1 < playouts);
        }
        keep_running &= (elapsed < thinking_time);
        keep_running &= (playouts_.load(std::memory_order_relaxed) < playouts);
        keep_running &= running_.load(std::memory_order_relaxed);
    };

    running_.store(false, std::memory_order_release);

    // Wait for all threads to join the main thread.
    group_->WaitToJoin();

    if (tag & kThinking) {
        time_control_.TookTime(color);
    }
    if (tag & kAnalyze) {
        DUMPING << root_node_->ToAnalysisString(root_state_, color, analysis_config_);
    }
    if (param_->analysis_verbose) {
        LOGGING << root_node_->ToVerboseString(root_state_, color);
        LOGGING << " * Time Status:\n";
        LOGGING << "  " << time_control_.ToString();
        LOGGING << "  spent: " << timer.GetDuration() << "(sec)\n";
        LOGGING << "  speed: " << (float)playouts_.load(std::memory_order_relaxed) /
                                      timer.GetDuration() << "(p/sec)\n";
    }

    // Record perfomance infomation.
    computation_result.seconds = timer.GetDuration();
    computation_result.playouts = playouts_.load(std::memory_order_relaxed);

    // Gather computation infomation and training data.
    GatherComputationResult(computation_result);

    // Save the last game state.
    last_state_ = root_state_;

    if (tag & kForced) {
        for (int i = 0; i < num_passes; ++i) {
            root_state_.PlayMove(kPass);
        }
    }

    return computation_result;
}

void Search::GatherComputationResult(ComputationResult &result) const {
    const auto color = root_state_.GetToMove();
    const auto num_intersections = root_state_.GetNumIntersections();
    const auto board_size = root_state_.GetBoardSize(); 

    // Fill best moves, root eval and score.
    result.best_move = root_node_->GetBestMove();
    result.random_move = root_node_->RandomizeFirstProportionally(1);
    result.root_eval = root_node_->GetWL(color, false);
    result.root_final_score = root_node_->GetFinalScore(color);

    // Resize the childern status buffer.
    result.root_ownership.resize(num_intersections);
    result.root_probabilities.resize(num_intersections+1);
    result.root_visits.resize(num_intersections+1);
    result.target_probabilities.resize(num_intersections+1);

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
    for (int idx = 0; idx < num_intersections+1; ++idx) {
        float prob = (float) result.root_visits[idx]/ (float) parentvisits;
        result.root_probabilities[idx] = prob;
    }

    // Fill target probabilities.

    // TODO: Prune some bad children in order get better
    //       target probabilities.
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
        if (node && node->IsActive()) {
            const auto prob = result.root_probabilities[idx];
            result.target_probabilities[idx] = prob;
            tot_target_policy += prob;
        } else {
            result.target_probabilities[idx] = 0.0f;
        }
    }

    for (auto &prob : result.target_probabilities) {
        prob /= tot_target_policy;
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

    // remove multiple mentions of the same string
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

    if (handicap > 0 && state.GetToMove() == kWhite) {
        const auto handicap_resign_threshold =
                       resign_threshold / (1 + handicap);

        auto blend_ratio = std::min(1.0f, movenum / (0.6f * num_intersections));
        auto blended_resign_threshold = blend_ratio * resign_threshold +
                                            (1 - blend_ratio) * handicap_resign_threshold;
        if (result.root_eval > blended_resign_threshold) {
            // Allow lower eval for white in handicap games
            // where opp may fumble.
            return false;
        }
    }

    if (result.root_eval > resign_threshold) {
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
                // side. It is dame.
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
        // We already win the game. I will play the pass move.
        return true;
    }

    // The game result is unknown. I will keep playing.
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

int Search::GetSelfPlayMove() {
    auto tag = param_->reuse_tree ? kThinking : (kThinking | kUnreused);
    auto result = Computation(max_playouts_, tag);

    int move = result.best_move;
    if (param_->random_moves_cnt > result.movenum) {
        move = result.random_move;
    }

    // Push the data to buffer.
    GatherData(root_state_, result);

    return move;
}

void Search::TryPonder() {
    if (param_->ponder) {
        Computation(GetPonderPlayouts(), kPonder);
    }
}

int Search::Analyze(bool ponder, AnalysisConfig &analysis_config) {
    // Ponder mode always reuse the tree.
    auto reuse_tag = (param_->reuse_tree || ponder) ? kNullTag : kUnreused;
    auto ponder_tag = ponder ? (kAnalyze | kPonder) : (kAnalyze | kThinking);

    auto tag = reuse_tag | ponder_tag;

    // Set the current analysis config.
    analysis_config_ = analysis_config;

    int playouts = ponder == true ? GetPonderPlayouts()
                                      : max_playouts_;
    auto result = Computation(playouts, tag);

    // Clear confing after finishing the search.
    analysis_config_.Clear();

    // Disable the reuse the tree.
    if (analysis_config_.MoveRestrictions()) {
        ReleaseTree();
    }

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

    auto fork_state = end_state;

    // Now, compute the final status. remove the dead strings
    // first.
    fork_state.RemoveDeadStrings(200);

    auto num_intersections = fork_state.GetNumIntersections();
    auto winner = kUndecide;
    auto black_final_score = 0.f;
    auto ownership = fork_state.GetOwnership();

    // Compute the final score.
    for (const auto owner : ownership) {
        if (owner == kBlack) {
            black_final_score += 1;
        } else if (owner == kWhite) {
            black_final_score -= 1;
        }
    }
    black_final_score -= fork_state.GetKomi();

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
            if (buf.side_to_move == owner) {
                buf.ownership[idx] = 1;
            } else if (buf.side_to_move == !owner) {
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

    // Save the data.
    for (auto &buf : training_buffer_) {
        buf.StreamOut(file);
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

    // map the root eval from [0 ~ 1] to [-1 ~ 1]
    data.q_value = 2 * result.root_eval - 1.f;
    data.planes = Encoder::Get().GetPlanes(state);
    data.probabilities = result.target_probabilities;

    training_buffer_.emplace_back(data);
}

bool Search::AdvanceToNewRootState() {
    if (!root_node_) {
        return false;
    }

    if (param_->dirichlet_noise || param_->root_dcnn) {
        // Need to re-build the trees if we apply noise. Reuse the
        // tree will ignore the noise. The root_dcnn option only
        // apply the network at root. The tree shape of root is different
        // from children.
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

    if (!root_node_->HaveChildren()) {
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
    const int ponder_playouts = std::max(4 * 1024,
                                             ponder_playouts_base * div_factor);
    return ponder_playouts;
}

int Search::GetExpandThreshold(GameState &state) const {
    const auto board_size = state.GetBoardSize();

    if (param_->expand_threshold >= 0) {
        return param_->expand_threshold;
    }

    return std::min(20 + 2 * (board_size-9), 60);
}
