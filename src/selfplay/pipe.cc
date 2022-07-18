#include "selfplay/pipe.h"
#include "utils/random.h"
#include "utils/filesystem.h"
#include "utils/log.h"
#include "config.h"

SelfPlayPipe::SelfPlayPipe() {
    Initialize();
    Loop();
}

void SelfPlayPipe::Initialize() {
    // Close search verbose.
    SetOption("analysis_verbose", false);

    // Force that one game use one thread.
    SetOption("threads", 1);

    if (GetOption<int>("playouts") == 0) {
        SetOption("playouts", 200);
    }

    engine_.Initialize();
    // TODO: Re-compute the NN cache size.

    target_directory_ = GetOption<std::string>("target_directory");
    max_games_ = GetOption<int>("num_games");
    accmulate_games_.store(0, std::memory_order_relaxed);
    played_games_.store(0, std::memory_order_relaxed);

    auto ss = std::ostringstream();
    ss << std::hex << Random<kXoroShiro128Plus>::Get().Generate() << std::dec;

    filename_hash_ = ss.str();
    sgf_filename_ = ConnectPath(target_directory_, filename_hash_ + ".sgf");
    data_directory_ = ConnectPath(target_directory_, "data");
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

    // Dump some infomations.
    LOGGING << "============================================" << std::endl;
    LOGGING << "Hash value: " << filename_hash_ << std::endl;
    LOGGING << "Target self-play games: " << max_games_ << std::endl;
    LOGGING << "Directory for saving: " << target_directory_  << std::endl;
    LOGGING << "Starting time is: " << CurrentDateTime()  << std::endl;

    if (!IsDirectoryExist(data_directory_)) {
        CreateDirectory(data_directory_);
    }

    for (int g = 0; g < engine_.GetParallelGames(); ++g) {
        workers_.emplace_back(
            [this, g]() -> void {
                while (accmulate_games_.load(std::memory_order_relaxed) < max_games_) {
                    int curr_games = accmulate_games_.fetch_add(1);

                    engine_.PrepareGame(g);
                    engine_.Selfplay(g);

                    auto data_filename = ConnectPath(data_directory_,
                                                         filename_hash_ +
                                                             "_" +
                                                             std::to_string(curr_games/100) + // chop per 100 games
                                                             ".dat");
                    engine_.SaveTrainingData(data_filename, g);
                    engine_.SaveSgf(sgf_filename_, g);

                    played_games_.fetch_add(1);
                    auto played_games = played_games_.load(std::memory_order_relaxed);

                    if (played_games % 100 == 0) {
                        std::lock_guard<std::mutex> lock(log_mutex_);
                        LOGGING << '[' << CurrentDateTime() << ']' << "Played " << played_games << " games." << std::endl;
                    }
                }
            }
        );
    }

    for (auto &t : workers_) {
        t.join();
    }
    LOGGING << '[' << CurrentDateTime() << ']'
                << "Finish the self-play loop. Total Played "
                << played_games_.load(std::memory_order_relaxed) << " games." << std::endl;
}
