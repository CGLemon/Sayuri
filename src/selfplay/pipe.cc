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
    // Close all verbose.
    SetOption("analysis_verbose", false);

    // Force that one game use one thread.
    SetOption("threads", 1);

    engine_.Initialize();
    // TODO: Re-compute the NN cache size.

    target_directory_ = GetOption<std::string>("target_directory");
    max_games_ = GetOption<int>("num_games");
    accmulate_games_.store(0);
    played_games_.store(0);

    auto ss = std::ostringstream();
    ss << std::hex << Random<kXoroShiro128Plus>::Get().Generate() << std::dec;

    filename_hash_ = ss.str();
    sgf_filename_ = ConnectPath(target_directory_, filename_hash_ + ".sgf");
    data_filename_ = ConnectPath(target_directory_, filename_hash_ + ".data");
}

void SelfPlayPipe::Loop() {
    // Check all data are ready.
    if (target_directory_.size() == 0) {
        LOGGING << "Please give the target directory name." << std::endl;
        return;
    }
    if (max_games_ == 0) {
        LOGGING << "The number of self-play games must be greater than one." << std::endl;
        return;
    }

    // Dump some infomations.
    LOGGING << "Hash value: " << filename_hash_ << std::endl;
    LOGGING << "Target self-play games: " << max_games_ << std::endl;
    LOGGING << "Directory for saving: " << target_directory_  << std::endl;
    LOGGING << "Start time is: " << CurrentDateTime()  << std::endl;

    // If the directory didn't exist, creating a new one.
    if (!IsDirectoryExist(target_directory_)) {
        CreateDirectory(target_directory_);
    }

    for (int g = 0; g < engine_.GetParallelGames(); ++g) {
        workers_.emplace_back(
            [this, g]() -> void {
                while (accmulate_games_.load() < max_games_) {
                    accmulate_games_.fetch_add(1);

                    engine_.PrepareGame(g);
                    engine_.Selfplay(g);
                    engine_.SaveTrainingData(data_filename_, g);
                    engine_.SaveSgf(sgf_filename_, g);

                    played_games_.fetch_add(1);
                    auto played_games = played_games_.load();

                    if (played_games % 10 == 0) {
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
                << played_games_.load() << " games." << std::endl;
}
