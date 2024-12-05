#include "game/sgf.h"
#include "game/iterator.h"
#include "neural/encoder.h"
#include "neural/supervised_writer.h"
#include "utils/filesystem.h"
#include "utils/log.h"
#include "utils/format.h"
#include "utils/option.h"
#include "utils/random.h"
#include "utils/gzip_helper.h"

#include <fstream>

SupervisedWriter::SupervisedWriter() {
    Initialize();
    Loop();
}

void SupervisedWriter::Initialize() {
    threads_ = GetOption<int>("threads");
    input_sgf_directory_ = GetOption<std::string>("input_sgf_directory");
    target_directory_ = GetOption<std::string>("target_directory");

    if (target_directory_.empty()) {
        target_directory_ = std::string{"data"};
    }
    if (!IsDirectoryExist(target_directory_)) {
        TryCreateDirectory(target_directory_);
    }

    if (!network_) {
        network_ = std::make_unique<Network>();
    }
    network_->Initialize(GetOption<std::string>("weights_file"));

    auto sgf_files = GetFileList(input_sgf_directory_);
    std::shuffle(std::begin(sgf_files), std::end(sgf_files), Random<>::Get());
    for (auto sgf_file : sgf_files) {
        sgf_files_queue_.emplace_back(sgf_file);
    }
}

std::string SupervisedWriter::TryGetSgfFilename() {
    std::lock_guard<std::mutex> lock(sgf_file_mutex_);
    if (sgf_files_queue_.empty()) {
        return std::string{};
    }

    auto sgf_file = sgf_files_queue_.front();
    sgf_files_queue_.pop_front();
    return ConcatPath(input_sgf_directory_, sgf_file);
}

void SupervisedWriter::PushSgfString(std::vector<std::string> &sgfs) {
    std::lock_guard<std::mutex> lock(sgf_string_mutex_);
    for (auto &sgf_string : sgfs) {
        sgf_strings_queue_.emplace_back(sgf_string);
    }
}

std::string SupervisedWriter::TryGetSgfString() {
    std::lock_guard<std::mutex> lock(sgf_string_mutex_);
    if (sgf_strings_queue_.empty()) {
        return std::string{};
    }

    auto sgf_string = sgf_strings_queue_.front();
    sgf_strings_queue_.pop_front();
    return sgf_string;
}

bool SupervisedWriter::SaveChunk(const int out_id,
                                 std::vector<TrainingData> &chunk) {
    const auto SaveFile = [](std::string out_name, std::ostringstream& oss) {
        bool is_open = true;
        auto buf = oss.str();
        try {
            SaveGzip(out_name, buf);
        } catch (const std::exception& e) {
            auto file = std::ofstream{};
            file.open(out_name, std::ios_base::app);
            is_open = file.is_open();

            if (!is_open) {
                LOGGING << "Fail to create the file: " << out_name << '!' << std::endl;
            } else {
                file << buf;
                file.close();
            }
        }
        return is_open;
    };
    auto out_name = ConcatPath(
                        target_directory_,
                        "game_" +
                            std::to_string(out_id) +
                            ".txt");
    auto data_oss = std::ostringstream{};
    for (auto &data : chunk) {
        data.StreamOut(data_oss);
    }

    bool is_open = SaveFile(out_name, data_oss);
    chunk.clear();
    return is_open;
}

void SupervisedWriter::GenenrateDataAndWrite(std::string &sgfstring) {
    GameState state;
    try {
        state = Sgf::Get().FromString(sgfstring, 9999);
    } catch (const std::exception& e) {
        LOGGING << "Fail to load the SGF file! Discard it." << std::endl
                    << Format("\tCause: %s.", e.what()) << std::endl;
        return;
    }

    auto game_ite = GameStateIterator(state);
    if (game_ite.MaxMoveNumber() == 0) {
        return;
    }

    auto chunk = std::vector<TrainingData>{};
    do {
        TrainingData data;
        const auto vertex = game_ite.GetVertex();
        const auto next_vertex = game_ite.GetNextVertex();
        GameState& main_state = game_ite.GetState();
        const auto num_intersections = main_state.GetNumIntersections();

        data.version = GetDefaultVersion();
        data.mode = GetDefaultMode();
        data.board_size = main_state.GetBoardSize();
        data.komi = main_state.GetKomiWithPenalty();
        data.side_to_move = main_state.GetToMove();

        data.probabilities.resize(num_intersections + 1, 0.0f);
        if (vertex == kPass) {
            data.probabilities[num_intersections] = 1.0f;
        } else {
            data.probabilities[main_state.VertexToIndex(vertex)] = 1.0f;
        }

        data.auxiliary_probabilities.resize(num_intersections + 1);
        if (next_vertex == kPass) {
            data.auxiliary_probabilities[num_intersections] = 1.0f;
        } else {
            data.auxiliary_probabilities[main_state.VertexToIndex(next_vertex)] = 1.0f;
        }

        data.planes = Encoder::Get().GetPlanes(main_state);
        data.wave = main_state.GetWave();
        data.rule = main_state.GetScoringRule() == kArea ? 0.f : 1.f;
        data.kld = 1.0f;
        AssignNetworkResult(data, main_state);
        chunk.emplace_back(data);
    } while (game_ite.Next());

    const auto game_id = num_sgf_games_.fetch_add(1, std::memory_order_relaxed);
    SaveChunk(game_id, chunk);
}

void SupervisedWriter::AssignNetworkResult(TrainingData &data, GameState &state) {
    auto convertWinrateToQ = [](float winrate) -> float {
        return 2 * winrate - 1.f;
    };

    auto net_list = network_->GetOutput(
        state, Network::kRandom, Network::Query::Get().SetCache(false));
    const auto num_intersections = state.GetNumIntersections();
    data.ownership.resize(num_intersections, 0);

    for (int idx = 0; idx < num_intersections; ++idx) {
        if (net_list.ownership[idx] > 0.15f) {
            data.ownership[idx] = 1;
        } else if (net_list.ownership[idx] < -0.15f) {
            data.ownership[idx] = -1;
        } else {
            data.ownership[idx] = 0;
        }
    }
    data.result = convertWinrateToQ(net_list.wdl_winrate) > 0.f ? 1 : -1;
    data.q_value =
        data.avg_q_value =
        data.short_avg_q =
        data.middle_avg_q =
        data.long_avg_q = convertWinrateToQ(net_list.stm_winrate);
    data.final_score =
        data.score_lead =
        data.avg_score_lead =
        data.short_avg_score =
        data.middle_avg_score =
        data.long_avg_score = net_list.final_score;
    data.q_stddev = net_list.q_error;
    data.score_stddev = net_list.score_error;
}

void SupervisedWriter::AssignWorkers() {
    // assign SGF reader
    workers_.emplace_back(
        [this]() -> void {
            int games = 0;
            while (true) {
                auto sgf_name = TryGetSgfFilename();
                if (sgf_name.empty()) {
                    break;
                }
                auto sgfs = SgfParser::Get().ChopAll(sgf_name);
                std::shuffle(std::begin(sgfs), std::end(sgfs), Random<>::Get());
                PushSgfString(sgfs);
                games += sgfs.size();
            }
            reader_done_.store(true, std::memory_order_relaxed);
            LOGGING << Format("Already read all SGF games. Read %d games.", games)
                        << std::endl;
        }
    );

    // assign SGF writer
    for (int t = 0; t < threads_; ++t) { 
        workers_.emplace_back(
            [this, t]() -> void {
                while (true) {
                    auto sgfstring = TryGetSgfString();
                    if (sgfstring.empty()) {
                        if (reader_done_.load(std::memory_order_relaxed)) {
                            break;
                        }
                        std::this_thread::yield();
                        continue;
                    }
                    GenenrateDataAndWrite(sgfstring);

                    if (t == 0) {
                        LOGGING << Format("Saved %d games.", num_sgf_games_.load(std::memory_order_relaxed)) << std::endl;
                    } 
                }
                LOGGING << Format("Thread %d is finished.", t) << std::endl;
            }
        );
    }
}

void SupervisedWriter::WaitForWorkers() {
    for (auto &t : workers_) {
        t.join();
    }
}

void SupervisedWriter::Loop() {
    AssignWorkers();
    WaitForWorkers();
}
