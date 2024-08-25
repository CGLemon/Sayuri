#include "selfplay/pipe.h"
#include "mcts/time_control.h"
#include "utils/random.h"
#include "utils/filesystem.h"
#include "utils/log.h"
#include "utils/gzip_helper.h"
#include "utils/time.h"
#include "config.h"

SelfPlayPipe::SelfPlayPipe() {
    Initialize();
    Loop();
}

void SelfPlayPipe::Initialize() {
    // Close search verbose.
    SetOption("analysis_verbose", false);

    // For each game has only one thread.
    SetOption("threads", 1);

    // Disable time management.
    SetOption("timemanage", (int)TimeControl::TimeManagement::kOff);

    // Wait for engine ready.
    engine_.Initialize();

    // Reset number conuter.
    max_games_ = GetOption<int>("num_games");
    accmulate_games_.store(0, std::memory_order_relaxed);
    played_games_.store(0, std::memory_order_relaxed);
    running_threads_.store(0, std::memory_order_relaxed);
    chunk_games_ = 0;

    // Generate the data files name. If the target file already existed, we re-generate
    // new file name.
    target_directory_ = GetOption<std::string>("target_directory");
    tdata_directory_ = ConcatPath(target_directory_, "tdata");
    vdata_directory_ = ConcatPath(target_directory_, "vdata");
    sgf_directory_ = ConcatPath(target_directory_, "sgf");
    queies_directory_ = ConcatPath(target_directory_, "net_queries");

    filename_hash_ = engine_.GetNetSha256();
    tdata_directory_hash_ = ConcatPath(tdata_directory_, filename_hash_);
    vdata_directory_hash_ = ConcatPath(vdata_directory_, filename_hash_);
}

void SelfPlayPipe::CreateWorkspace() {
    const auto SafeMakeDir = [](std::string dir_name) {
        if (!IsDirectoryExist(dir_name)) {
            TryCreateDirectory(dir_name);
        }
    };
    SafeMakeDir(tdata_directory_);
    SafeMakeDir(tdata_directory_hash_);
    SafeMakeDir(vdata_directory_);
    SafeMakeDir(vdata_directory_hash_);
    SafeMakeDir(sgf_directory_);
    SafeMakeDir(queies_directory_);
}

bool SelfPlayPipe::SaveChunk(const int out_id,
                             float vdata_prob,
                             std::vector<TrainingData> &chunk) {
    const auto SaveFile = [](std::string out_name, std::ostringstream& oss) {
        bool is_open = true;
        auto buf = oss.str();
        try {
            SaveGzip(out_name, buf);
        } catch (const char *err) {
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
    auto tdata_out_name = ConcatPath(
                              tdata_directory_hash_,
                              filename_hash_ +
                                  "_" +
                                  std::to_string(out_id) +
                                  ".txt");
    auto vdata_out_name = ConcatPath(
                              vdata_directory_hash_,
                              filename_hash_ +
                                  "_" +
                                  std::to_string(out_id) +
                                  ".txt");

    auto tdata_oss = std::ostringstream{};
    auto vdata_oss = std::ostringstream{};
    vdata_prob = std::max(std::min(vdata_prob, 1.0f), 0.0f);

    for (auto &data : chunk) {
        if (Random<>::Get().Roulette<10000>(1.0f - vdata_prob)) {
            data.StreamOut(tdata_oss);
        } else {
            data.StreamOut(vdata_oss);
        }
    }

    bool is_open = true;
    is_open &= SaveFile(tdata_out_name, tdata_oss);
    is_open &= SaveFile(vdata_out_name, vdata_oss);

    chunk.clear();
    return is_open;
}

bool SelfPlayPipe::SaveNetQueries(const size_t queries) {
    auto out_name = ConcatPath(
                        queies_directory_,
                        filename_hash_ + ".txt");

    auto file = std::ofstream{};
    file.open(out_name, std::ios_base::app);

    bool is_open = file.is_open();
    if (is_open) {
        file << queries << std::endl;
    } else {
        LOGGING << "Fail to create the file: " << out_name << '!' << std::endl;
    }

    return is_open;
}

void SelfPlayPipe::Loop() {
    // Be sure that all data are ready.
    if (target_directory_.size() == 0) {
        LOGGING << "Please give the target directory name." << std::endl;
        return;
    }
    if (!IsDirectoryExist(target_directory_)) {
        LOGGING << "Target directory do not exist." << std::endl;
        return;
    }
    if (max_games_ == 0) {
        LOGGING << "The number of self-play games must be greater than one." << std::endl;
        return;
    }
    if (!IsGzipValid()) {
        LOGGING << "WARNING: There is no gzip tool. The output chunks may be very large." << std::endl;
    }
    CreateWorkspace();

    // Dump some infomations.
    LOGGING << "============================================" << std::endl;
    LOGGING << "Hash value: " << filename_hash_ << std::endl;
    LOGGING << "Target self-play games: " << max_games_ << std::endl;
    LOGGING << "Directory for saving: " << target_directory_  << std::endl;
    LOGGING << "Starting time is: " << CurrentDateTime()  << std::endl;

    for (int g = 0; g < engine_.GetParallelGames(); ++g) {
        workers_.emplace_back(
            [this, g]() -> void {
                constexpr int kGamesPerChunk = 25;
                auto sgf_filename = ConcatPath(
                                        sgf_directory_, filename_hash_ + ".sgf");
                running_threads_.fetch_add(1, std::memory_order_relaxed);

                while (accmulate_games_.fetch_add(1) < max_games_) {
                    engine_.PrepareGame(g);
                    engine_.Selfplay(g);

                    {
                        // Save the current chunk.
                        std::lock_guard<std::mutex> lock(data_mutex_);

                        engine_.GatherTrainingData(chunk_, g);

                        if ((chunk_games_+1) % kGamesPerChunk == 0) {
                            if (!SaveChunk(chunk_games_/kGamesPerChunk, 0.1f, chunk_)) {
                                break;
                            }
                        }
                        engine_.SaveSgf(sgf_filename, g);
                        chunk_games_ += 1;
                    }

                    played_games_.fetch_add(1);
                    auto played_games = played_games_.load(std::memory_order_relaxed);

                    if (played_games % 100 == 0) {
                        std::lock_guard<std::mutex> lock(log_mutex_);
                        LOGGING << '[' << CurrentDateTime() << ']' << " Played " << played_games << " games." << std::endl;
                        SaveNetQueries(engine_.GetNetReportQueries());
                    }
                }

                {
                    std::lock_guard<std::mutex> lock(data_mutex_);
                    running_threads_.fetch_sub(1, std::memory_order_relaxed);

                    // The last thread saves the remaining training data.
                    if (!chunk_.empty() &&
                            running_threads_.load(std::memory_order_relaxed) == 0) {
                        SaveChunk(chunk_games_/kGamesPerChunk, 0.1f, chunk_);
                        chunk_games_ += 1;
                    }
                }
            }
        );
    }

    for (auto &t : workers_) {
        t.join();
    }
    LOGGING << '[' << CurrentDateTime() << ']'
                << " Finish the self-play loop. Totally played "
                << played_games_.load(std::memory_order_relaxed) << " games." << std::endl;
}
