#include "game/game_state.h"
#include "game/sgf.h"
#include "game/types.h"
#include "game/iterator.h"
#include "neural/supervised.h"
#include "neural/encoder.h"
#include "utils/log.h"
#include "utils/format.h"
#include "utils/random.h"
#include "utils/threadpool.h"
#include "utils/time.h"
#include "utils/gzip_helper.h"
#include "utils/option.h"

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
    // Init all status.
    file_cnt_.store(0, std::memory_order_relaxed);
    worker_cnt_.store(0, std::memory_order_relaxed);
    tot_games_.store(0, std::memory_order_relaxed);
    running_threads_.store(0, std::memory_order_relaxed);
    running_.store(true, std::memory_order_relaxed);

    auto sgfs = SgfParser::Get().ChopAll(sgf_name);

    auto Worker = [this, general, out_name_prefix]() -> void {
        int games = 0;
        auto chunk = std::vector<Training>{};

        running_threads_.fetch_add(1, std::memory_order_relaxed);
        int worker_cnt = worker_cnt_.fetch_add(1, std::memory_order_relaxed);

        LOGGING << Format("[%s] Thread %d is ready\n",
                              CurrentDateTime().c_str(), worker_cnt+1);

        while (true) {
            if (!running_.load(std::memory_order_relaxed) && tasks_.empty()) {
                break;
            }

            // Get the SGF string from the queue.
            auto sgf = std::string{};

            {
                std::lock_guard<std::mutex> lk(mtx_);
                if (!tasks_.empty()) {
                    sgf = tasks_.front();
                    tasks_.pop();
                }
            }

            if (sgf.empty()) {
                // Fail to get the SGF string from the queue.
                std::this_thread::yield();
                continue;
            }

            constexpr int kGamesPerChunk = 25;

            // Parse the SGF string.
            bool success = false;
            if (general) {
                success = GeneralSgfProcess(sgf, chunk);
            } else {
                success = SgfProcess(sgf, chunk);
            }

            if (success) {
                games += 1;
                tot_games_.fetch_add(1, std::memory_order_relaxed);
                if (games % kGamesPerChunk == 0) {

                    if (!SaveChunk(out_name_prefix, chunk)) {
                        break;
                    }

                    LOGGING << Format("[%s] Thread %d parsed %d games, totally parsed %d games.\n",
                                          CurrentDateTime().c_str(),
                                          worker_cnt+1, games, tot_games_.load(std::memory_order_relaxed)
                                     );
                }
            }
        }

        // Save the remaining training data.
        if (!chunk.empty()) {
            SaveChunk(out_name_prefix, chunk);
        }

        running_threads_.fetch_sub(1, std::memory_order_relaxed);
        LOGGING << Format("[%s] Thread %d is terminate, totally parsed %d games.\n",
                                  CurrentDateTime().c_str(),
                                  worker_cnt+1, tot_games_.load(std::memory_order_relaxed)
                         );
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
            if (running_threads_.load(std::memory_order_relaxed) < 0) {
                // Can not open the storage file, stop running.
                break;
            }
        }
    }

    running_.store(false, std::memory_order_relaxed);
    group.WaitToJoin();
}

bool Supervised::SaveChunk(std::string out_name_prefix,
                           std::vector<Training> &chunk) {
    auto out_name = Format("%s_%d.txt", 
                               out_name_prefix.c_str(),
                               file_cnt_.fetch_add(1));

    auto oss = std::ostringstream{};
    for (auto &data : chunk) {
        data.StreamOut(oss);
    }

    bool is_open = true;

    try {
        auto buf = oss.str();
        SaveGzip(out_name, buf);
    } catch (const char *err) {
        auto file = std::ofstream{};
        file.open(out_name, std::ios_base::app);
        if (!file.is_open()) {
            LOGGING << "Fail to create the file: " << out_name << '!' << std::endl; 
            running_threads_.store(-1, std::memory_order_relaxed);
            is_open = false;
        } else {
            for (auto &data : chunk) {
                data.StreamOut(file);
            }
            file.close();
        }
    }
    chunk.clear();

    return is_open;
}

bool Supervised::GeneralSgfProcess(std::string &sgfstring,
                                   std::vector<Training> &chunk) const {
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

    const auto zero_ownership = std::vector<int>(num_intersections, 0.f);
    const auto zero_final_score = 0.f;

    auto game_ite = GameStateIterator(state);

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

    // Remove the double pass moves in the middle.
    game_ite.RemoveUnusedDoublePass();

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
            buf.ownership[idx] = zero_ownership[idx]; 
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

        chunk.emplace_back(buf);
    } while (game_ite.Next());


    return true;
}

bool Supervised::SgfProcess(std::string &sgfstring,
                            std::vector<Training> &chunk) const {
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

    // Remove the double pass moves in the middle.
    game_ite.RemoveUnusedDoublePass();

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

        chunk.emplace_back(buf);
    } while (game_ite.Next());

    return true;
}
