#include "data/supervised.h"
#include "data/training.h"
#include "game/game_state.h"
#include "game/sgf.h"
#include "game/types.h"
#include "game/iterator.h"
#include "neural/encoder.h"
#include "utils/log.h"
#include "utils/format.h"
#include "utils/random.h"
#include "utils/threadpool.h"

#include "config.h"

#include <fstream>
#include <cassert>
#include <cmath>
#include <thread>

Supervised &Supervised::Get() {
    static Supervised supervised;
    return supervised;
}

void Supervised::FromSgfs(bool general,
                              std::string sgf_name,
                              std::string out_name_prefix) {
    file_cnt_.store(0);
    worker_cnt_.store(0);
    tot_games_.store(0);
    running_.store(true);

    auto sgfs = SgfParser::Get().ChopAll(sgf_name);

    auto Worker = [this, general, out_name_prefix]() -> void {
        auto file = std::ofstream{};
        bool closed = true;
        int games = 0;
        int worker_cnt = worker_cnt_.fetch_add(1);

        LOGGING << Format("Thread %d is ready", worker_cnt+1) << std::endl;

        while (true) {
            if (!running_.load(std::memory_order_relaxed) && tasks_.empty()) {
                LOGGING << Format("Thread %d is terminate, totally parsed %d games.",
                                          worker_cnt+1, tot_games_.load(std::memory_order_relaxed)) << std::endl;
                break;
            }

            auto sgf = std::string{};

            {
                std::lock_guard<std::mutex> lk(mtx_);
                if (!tasks_.empty()) {
                    sgf = tasks_.front();
                    tasks_.pop();
                }
            }

            if (sgf.empty()) {
                std::this_thread::yield();
                continue;
            }

            if (closed) {
                auto out_name = Format("%s_%d.txt", out_name_prefix.c_str(), file_cnt_.fetch_add(1));
                file.open(out_name, std::ios_base::app);

                if (!file.is_open()) {
                    LOGGING << "Fail to create the file: " << out_name << '!' << std::endl; 
                    return;
                }
                closed = false;
            }

            constexpr int kChopPerGames = 200;

            bool success = false;
            if (general) {
                success = GeneralSgfProcess(sgf, file);
            } else {
                success = SgfProcess(sgf, file);
            }

            if (success) {
                games += 1;
                tot_games_.fetch_add(1, std::memory_order_relaxed);
                if (games % kChopPerGames == 0) {
                    if (!closed) {
                        file.close();
                        closed = true;
                    }
                    LOGGING << Format("Thread %d parsed %d games, totally parsed %d games.",
                                          worker_cnt+1, games, tot_games_.load(std::memory_order_relaxed)) << std::endl;
                }
            }
        }

        if (!closed) {
            file.close();
        }
    };

    auto threads = GetOption<int>("threads");
    auto group =  ThreadGroup<void>(&ThreadPool::Get(threads));

    for (int t = 0; t < threads; ++t) {
        group.AddTask(Worker);
    }

    while (!sgfs.empty()) {
        auto sgf = sgfs.back();
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if ((int)tasks_.size() <= threads * 4) {
                tasks_.push(sgf);
                sgfs.pop_back();
            } else {
                std::this_thread::yield();
            }
        }
    }

    running_.store(false, std::memory_order_relaxed);
    group.WaitToJoin();
}

bool Supervised::GeneralSgfProcess(std::string &sgfstring,
                                       std::ostream &out_file) const {
    GameState state;

    try {
        state = Sgf::Get().FromString(sgfstring, 9999);
    } catch (const char *err) {
        LOGGING << "Fail to load the SGF file! Discard it." << std::endl
                    << Format("\tCause: %s.", err) << std::endl;
        return false;
    }

    const auto board_size = state.GetBoardSize();
    const auto num_intersections = state.GetNumIntersections();
    const auto komi = state.GetKomi();
    const auto winner = state.GetWinner();

    if (winner == kUndecide) {
        LOGGING << "The SGF file is no reulst! Discard it." << std::endl;
        return false;
    }

    const auto zero_ownership = std::vector<float>(num_intersections, 0.f);
    const auto zero_final_score = 0.f;

    auto game_ite = GameStateIterator(state);
    auto train_datas = std::vector<Training>{};

    const auto VertexToIndex = [](GameState &state, int vertex) -> int {
        if (vertex == kPass) {
            return state.GetNumIntersections();
        }

        auto x = state.GetX(vertex);
        auto y = state.GetY(vertex);
        return state.GetIndex(x, y);
    };

    if (game_ite.MaxMoveNumber() == 0) {
        return false;
    }

    do {
        auto vtx = game_ite.GetVertex();
        auto aux_vtx = game_ite.GetNextVertex();
        GameState& main_state = game_ite.GetState();

        auto buf = Training{};

        buf.version = GetTrainingVersion();
        buf.mode = GetTrainingMode();
        buf.board_size = board_size;
        buf.komi = komi;
        buf.side_to_move = main_state.GetToMove();

        buf.planes = Encoder::Get().GetPlanes(main_state);

        buf.probabilities = std::vector<float>(num_intersections+1, 0);
        buf.auxiliary_probabilities = std::vector<float>(num_intersections+1, 0);
        buf.ownership = std::vector<int>(num_intersections, 0);

        buf.probabilities_index = VertexToIndex(main_state, vtx);
        buf.probabilities[VertexToIndex(main_state, vtx)] = 1.0f;

        buf.auxiliary_probabilities_index = VertexToIndex(main_state, aux_vtx);
        buf.auxiliary_probabilities[VertexToIndex(main_state, aux_vtx)] = 1.0f;

        for (int idx = 0; idx < num_intersections; ++idx) {
            if (zero_ownership[idx] == buf.side_to_move) {
                buf.ownership[idx] = 1; 
            } else if (zero_ownership[idx] == !buf.side_to_move) {
                buf.ownership[idx] = -1;
            }
        }

        assert(winner != kUndecide);
        if (winner == kDraw) {
            buf.final_score = 0;
            buf.q_value = 0;
            buf.result = 0;
        } else {
            buf.result = (int)winner == (int)buf.side_to_move ? 1 : -1;
            buf.q_value = buf.result;
            buf.final_score = zero_final_score;
        }

        train_datas.emplace_back(buf);
    } while (game_ite.Next());

    for (const auto &buf : train_datas) {
        buf.StreamOut(out_file);
    }

    return true;
}

bool Supervised::SgfProcess(std::string &sgfstring,
                                std::ostream &out_file) const {
    GameState state;

    try {
        state = Sgf::Get().FromString(sgfstring, 9999);
    } catch (const char *err) {
        LOGGING << "Fail to load the SGF file! Discard it." << std::endl
                    << Format("\tCause: %s.", err) << std::endl;
        return false;
    }

    auto ownership = state.GetOwnershipAndRemovedDeadStrings(200);

    auto success = true;
    auto black_score_on_board = 0;
    const auto board_size = state.GetBoardSize();

    for (int y = 0; y < board_size; ++y) {
        for (int x = 0; x < board_size; ++x) {
            auto index = state.GetIndex(x, board_size-y-1);
            auto owner = ownership[index];

            if (owner == kBlack) {
                black_score_on_board += 1;
            } else if (owner == kWhite) {
                black_score_on_board -= 1;
            } else if (owner == kInvalid) {
                success = false;
            }
        }
    }

    const auto black_final_score = (float)black_score_on_board - state.GetKomi();
    auto game_winner = kUndecide;

    if (std::abs(black_final_score) < 1e-4f) {
        game_winner = kDraw;
    } else if (black_final_score > 0) {
        game_winner = kBlackWon;
    } else if (black_final_score < 0) {
        game_winner = kWhiteWon;
    }

    (void) success;


    const auto num_intersections = state.GetNumIntersections();
    const auto komi = state.GetKomi();
    const auto winner = game_winner;

    auto game_ite = GameStateIterator(state);
    auto train_datas = std::vector<Training>{};

    const auto VertexToIndex = [](GameState &state, int vertex) -> int {
        if (vertex == kPass) {
            return state.GetNumIntersections();
        }

        auto x = state.GetX(vertex);
        auto y = state.GetY(vertex);
        return state.GetIndex(x, y);
    };

    if (game_ite.MaxMoveNumber() == 0) {
        return false;
    }

    do {
        auto vtx = game_ite.GetVertex();
        auto aux_vtx = game_ite.GetNextVertex();
        GameState& main_state = game_ite.GetState();

        auto buf = Training{};

        buf.version = GetTrainingVersion();
        buf.mode = GetTrainingMode();
        buf.board_size = board_size;
        buf.komi = komi;
        buf.side_to_move = main_state.GetToMove();

        buf.planes = Encoder::Get().GetPlanes(main_state);

        buf.probabilities = std::vector<float>(num_intersections+1, 0);
        buf.auxiliary_probabilities = std::vector<float>(num_intersections+1, 0);
        buf.ownership = std::vector<int>(num_intersections, 0);

        buf.probabilities_index = VertexToIndex(main_state, vtx);
        buf.probabilities[VertexToIndex(main_state, vtx)] = 1.0f;

        buf.auxiliary_probabilities_index = VertexToIndex(main_state, aux_vtx);
        buf.auxiliary_probabilities[VertexToIndex(main_state, aux_vtx)] = 1.0f;

        for (int idx = 0; idx < num_intersections; ++idx) {
            if (ownership[idx] == buf.side_to_move) {
                buf.ownership[idx] = 1; 
            } else if (ownership[idx] == !buf.side_to_move) {
                buf.ownership[idx] = -1;
            }
        }

        assert(winner != kUndecide);
        if (winner == kDraw) {
            buf.final_score = 0;
            buf.q_value = 0;
            buf.result = 0;
        } else {
            buf.result = (int)winner == (int)buf.side_to_move ? 1 : -1;
            buf.q_value = buf.result;
            buf.final_score = buf.side_to_move == kBlack ? black_final_score : -black_final_score;
        }

        train_datas.emplace_back(buf);
    } while (game_ite.Next());

    for (const auto &buf : train_datas) {
        buf.StreamOut(out_file);
    }

    return true;
}
