#include "selfplay/pipe.h"
#include "mcts/time_control.h"
#include "utils/random.h"
#include "utils/filesystem.h"
#include "utils/log.h"
#include "utils/gzip_helper.h"
#include "utils/time.h"
#include "config.h"

#include <algorithm>
#include <stdexcept>

SelfPlayPipe::SelfPlayPipe() {
    Initialize();
    Loop();
    Finish();
}

void SelfPlayPipe::Initialize() {
    // Close search verbose.
    SetOption("analysis_verbose", false);

    // Disable time management.
    SetOption("timemanage", (int)TimeControl::TimeManagement::kOff);

    // Wait for engine ready.
    engine_.Initialize();

    // Group for self-play workers and data writer worker.
    group_ = std::make_unique<ThreadGroup<void>>(&ThreadPool::Get());

    // Reset number conuter.
    max_games_ = GetOption<int>("num_games");
    accumulation_games_.store(0, std::memory_order_relaxed);
    played_games_.store(0, std::memory_order_relaxed);
    running_threads_.store(0, std::memory_order_relaxed);
    writing_worker_running_.store(true, std::memory_order_relaxed);
    num_saved_chunks_ = 0;

    // Generate the data files name. If the target file already existed, we re-generate
    // new file name.
    target_directory_ = GetOption<std::string>("target_directory");
    tdata_directory_ = ConcatPath(target_directory_, "tdata");
    vdata_directory_ = ConcatPath(target_directory_, "vdata");
    sgf_directory_ = ConcatPath(target_directory_, "sgf");
    queies_directory_ = ConcatPath(target_directory_, "net_queries");
    while (true) {
        auto ss = std::ostringstream();
        ss << std::hex << std::uppercase
               << GetTimeHash() << std::dec;

        filename_hash_ = ss.str();
        tdata_directory_hash_ = ConcatPath(tdata_directory_, filename_hash_);
        vdata_directory_hash_ = ConcatPath(vdata_directory_, filename_hash_);

        bool collision = false;

        if (IsDirectoryExist(tdata_directory_hash_)) {
            collision = true;
        }
        if (IsDirectoryExist(vdata_directory_hash_)) {
            collision = true;
        }
        for (auto sgf_name : GetFileList(sgf_directory_)) {
            if ((filename_hash_ + ".sgf") == sgf_name) {
                collision = true;
                break;
            }
        }
        for (auto queies_name : GetFileList(queies_directory_)) {
            if ((filename_hash_ + ".txt") == queies_name) {
                collision = true;
                break;
            }
        }
        if (!collision) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1)); // next time hash
    }
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

bool SelfPlayPipe::SaveSgf(std::string &sgfstring) {
    const auto SaveFile = [](std::string out_name, std::string &buf) {
        auto file = std::ofstream{};
        file.open(out_name, std::ios_base::app);
        bool is_open = file.is_open();
        if (!is_open) {
            LOGGING << "Fail to create the file: " << out_name << '!' << std::endl;
        } else {
            file << buf;
            file.close();
        }
        return is_open;
    };

    auto sgf_filename = ConcatPath(
                            sgf_directory_, filename_hash_ + ".sgf");
    bool is_open = SaveFile(sgf_filename, sgfstring);
    return is_open;
}

bool SelfPlayPipe::SaveChunk(const int out_id,
                             float vdata_prob,
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
    auto tdata_out_name = ConcatPath(
                              tdata_directory_hash_,
                              "game_" +
                                  std::to_string(out_id) +
                                  ".txt");
    auto vdata_out_name = ConcatPath(
                              vdata_directory_hash_,
                              "game_" +
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

bool SelfPlayPipe::SaveNetQueries(int games, std::string net_queries) {
    auto out_name = ConcatPath(
                        queies_directory_,
                        filename_hash_ + ".txt");

    auto file = std::ofstream{};
    file.open(out_name, std::ios_base::app);

    bool is_open = file.is_open();
    if (is_open) {
        file << games
                 << " "
                 << net_queries << std::endl;
    } else {
        LOGGING << "Fail to create the file: " << out_name << '!' << std::endl;
    }

    return is_open;
}

int SelfPlayPipe::FancyCeil(int val, int step) const {
    return step * (val / step + static_cast<bool>(val % step));
}

void SelfPlayPipe::AssignDataWorker() {
    ThreadPool::Get("data-writer", 1);
    group_->AddTask(
        [this]() -> void {
            constexpr float kValidationRatio = 0.1f;

            const int games = engine_.GetParallelGames();
            bool keep_running = writing_worker_running_.load(std::memory_order_relaxed);
            auto data_buffer = std::vector<std::shared_ptr<DataSgfPair>>{};
            auto queries_buffer = std::list<GamesQueriesPair>{};

            while (keep_running) {
                std::this_thread::yield();
                keep_running = writing_worker_running_.load(std::memory_order_relaxed);
                {
                    // Gather the item to the local buffer.
                    std::lock_guard<std::mutex> lock(data_mutex_);
                    while (!data_sgf_buffer_.empty()) {
                        data_buffer.emplace_back(data_sgf_buffer_.front());
                        data_sgf_buffer_.pop_front();
                        keep_running |= true;
                    }
                }
                {
                    // Gather the item to the local buffer.
                    std::lock_guard<std::mutex> lock(log_mutex_);
                    while (!games_queries_buffer_.empty()) {
                        queries_buffer.emplace_back(games_queries_buffer_.front());
                        games_queries_buffer_.pop_front();
                        keep_running |= true;
                    }
                }

                // Write the data from the local buffer.
                const int games_buffer_size =
                    writing_worker_running_.load(std::memory_order_relaxed) ? games : 1;
                while ((int)data_buffer.size() >= games_buffer_size) {
                    std::shuffle(std::begin(data_buffer), std::end(data_buffer), Random<>::Get());

                    auto data_sgf_pair = data_buffer.back();
                    data_buffer.pop_back();
                    if (SaveChunk(num_saved_chunks_, kValidationRatio, data_sgf_pair->first)) {
                        num_saved_chunks_ += 1;
                    }
                    SaveSgf(data_sgf_pair->second);
                }
                while (!queries_buffer.empty()) {
                    auto queries_pair = queries_buffer.front();
                    queries_buffer.pop_front();
                    SaveNetQueries(queries_pair.first, queries_pair.second);
                }
            }
        }
    );
}

void SelfPlayPipe::AssignSelfplayWorkers() {
    ThreadPool::Get("selfplay", engine_.GetParallelGames());
    for (int g = 0; g < engine_.GetParallelGames(); ++g) {
        group_->AddTask(
            [this, g]() -> void {
                constexpr int kMainThreadIdx = 0;
                constexpr int kBufferGames = 25;
                constexpr int kVerboseGames = 100;
                bool should_halt = false;
                running_threads_.fetch_add(1, std::memory_order_relaxed);

                while (accumulation_games_.fetch_add(1) < max_games_) {
                    if (g == kMainThreadIdx && !should_halt) {
                        if (engine_.ShouldHalt()) {
                            int accum_games = std::max(
                                engine_.GetParallelGames(),
                                accumulation_games_.load(std::memory_order_relaxed));
                            // Assume no data racing so we do not lock max_games_.
                            max_games_ = std::min(
                                max_games_,
                                FancyCeil(accum_games + kBufferGames, kBufferGames));
                            should_halt = true;
                            LOGGING << '[' << CurrentDateTime() << ']'
                                        << " Will halt the self-play loop after playing "
                                        << max_games_ << " games." << std::endl;
                        }
                    }

                    // Engine start self-play and saving the data into queue after
                    // game is finished.
                    auto data_sgf_pair = std::make_shared<DataSgfPair>();
                    engine_.PrepareGame(g);
                    engine_.Selfplay(g);
                    engine_.GatherTrainingData(data_sgf_pair->first, g);
                    engine_.GatherSgfString(data_sgf_pair->second, g);
                    {
                        std::lock_guard<std::mutex> lock(data_mutex_);
                        data_sgf_buffer_.emplace_back(data_sgf_pair);
                    }
                    auto played_games = played_games_.fetch_add(1) + 1;

                    {
                        // Dump number of games verbose.
                        std::lock_guard<std::mutex> lock(log_mutex_);
                        if (played_games % kVerboseGames == 0) {
                            LOGGING << '[' << CurrentDateTime() << ']'
                                        << " Played " << played_games << " games." << std::endl;
                        }
                        // The accumulated value throughout the process may be inaccurate,
                        // but the final value is correct.
                        games_queries_buffer_.emplace_back(
                            played_games, engine_.GetNetReportQueries());
                    }
                }
                running_threads_.fetch_sub(1, std::memory_order_relaxed);

                if (g == kMainThreadIdx) {
                    while (running_threads_.load(std::memory_order_relaxed) != 0) {
                        std::this_thread::yield();
                    }
                    writing_worker_running_.store(false, std::memory_order_relaxed);
                }
            }
        );
    }
}

void SelfPlayPipe::Loop() {
    if (target_directory_.size() == 0) {
        LOGGING << "ABORT: Please give the target directory name." << std::endl;
        return;
    }
    if (!IsDirectoryExist(target_directory_)) {
        LOGGING << "ABORT: Target directory do not exist." << std::endl;
        return;
    }
    if (max_games_ == 0) {
        LOGGING << "ABORT: The number of self-play games must be greater than one." << std::endl;
        return;
    }
    if (max_games_ < engine_.GetParallelGames()) {
        max_games_ = engine_.GetParallelGames();
        LOGGING << "WARNING: The number of self-play games must be greater than parallel games. New value is "
                    << max_games_ << "." << std::endl;
    }
    if (!IsGzipValid()) {
        LOGGING << "WARNING: There is no gzip tool. The output chunks may be very large." << std::endl;
    }
    CreateWorkspace();

    LOGGING << "============================================" << std::endl
                << "Hash value: " << filename_hash_ << std::endl
                << "Number of parallel games: " << engine_.GetParallelGames() << std::endl
                << "Target self-play games: " << max_games_ << std::endl
                << "Directory for saving: " << target_directory_  << std::endl
                << "Starting time is: " << CurrentDateTime()  << std::endl;

    AssignDataWorker();
    AssignSelfplayWorkers();

    group_->WaitToJoin();

    LOGGING << '[' << CurrentDateTime() << ']'
                << " Finish the self-play loop. Totally played "
                << played_games_.load(std::memory_order_relaxed) << " games." << std::endl;
}

void SelfPlayPipe::Finish() {
    engine_.Abort();
}
