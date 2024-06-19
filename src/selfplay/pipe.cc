#include "selfplay/pipe.h"
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

    engine_.Initialize();

    target_directory_ = GetOption<std::string>("target_directory");
    max_games_ = GetOption<int>("num_games");
    accmulate_games_.store(0, std::memory_order_relaxed);
    played_games_.store(0, std::memory_order_relaxed);
    running_threads_.store(0, std::memory_order_relaxed);

    chunk_games_ = 0;

    while (true) {
        auto ss = std::ostringstream();
        ss << std::hex << std::uppercase
               << GetTimeHash() << std::dec;

        filename_hash_ = ss.str();
        sgf_directory_ = ConcatPath(target_directory_, "sgf");
        data_directory_ = ConcatPath(target_directory_, "data");
        data_directory_hash_ = ConcatPath(data_directory_, filename_hash_);

        bool not_existence = true;
        if (IsDirectoryExist(data_directory_hash_)) {
            not_existence = false;
        }

        for (auto sgf_name : GetFileList(sgf_directory_)) {
            if ((filename_hash_ + ".sgf") == sgf_name) {
                not_existence = false;
                break;
            }
        }

        if (not_existence) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1)); // next time hash
    }
}

bool SelfPlayPipe::SaveChunk(const int out_id,
                             std::vector<TrainingData> &chunk) {
    auto out_name = ConcatPath(
                        data_directory_hash_,
                        filename_hash_ +
                            "_" +
                            std::to_string(out_id) +
                            ".txt");

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
            is_open = false;
            LOGGING << "Fail to create the file: " << out_name << '!' << std::endl;
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

bool SelfPlayPipe::SaveNetQueries(const size_t queries) {
    auto out_name = ConcatPath(
                        target_directory_,
                        "net_queries.txt");

    auto file = std::ofstream{};
    file.open(out_name, std::ios_base::app);

    bool is_open = true;

    if (file.is_open()) {
        is_open = true;
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

    // Dump some infomations.
    LOGGING << "============================================" << std::endl;
    LOGGING << "Hash value: " << filename_hash_ << std::endl;
    LOGGING << "Target self-play games: " << max_games_ << std::endl;
    LOGGING << "Directory for saving: " << target_directory_  << std::endl;
    LOGGING << "Starting time is: " << CurrentDateTime()  << std::endl;

    if (!IsDirectoryExist(data_directory_)) {
        TryCreateDirectory(data_directory_);
    }
    if (!IsDirectoryExist(data_directory_hash_)) {
        TryCreateDirectory(data_directory_hash_);
    }
    if (!IsDirectoryExist(sgf_directory_)) {
        TryCreateDirectory(sgf_directory_);
    }

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
                            if (!SaveChunk(chunk_games_/kGamesPerChunk, chunk_)) {
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
                        SaveChunk(chunk_games_/kGamesPerChunk, chunk_);
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
