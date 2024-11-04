#include "selfplay/pipe.h"
#include "mcts/time_control.h"
#include "utils/random.h"
#include "utils/filesystem.h"
#include "utils/log.h"
#include "utils/gzip_helper.h"
#include "utils/time.h"
#include "config.h"

#include <algorithm>

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
    accumulation_games_.store(0, std::memory_order_relaxed);
    played_games_.store(0, std::memory_order_relaxed);
    running_threads_.store(0, std::memory_order_relaxed);
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

bool SelfPlayPipe::GatherChunkFromBuffer(int games, std::vector<TrainingData> &chunk) {
    if ((int)game_chunk_buffer_.size() < games) {
        return false;
    }
    std::shuffle(std::begin(game_chunk_buffer_),
        std::end(game_chunk_buffer_), Random<>::Get());
    for (int i = 0; i < games; ++i) {
        auto game_data = game_chunk_buffer_.back();
        game_chunk_buffer_.pop_back();
        for (auto &data : *game_data) {
            chunk.emplace_back(data);
        }
    }
    return true;
}

int SelfPlayPipe::FancyCeil(int val, int step) const {
    return step * (val / step + static_cast<bool>(val % step));
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
    if (max_games_ < engine_.GetParallelGames()) {
        max_games_ = FancyCeil(engine_.GetParallelGames(), 25);
        LOGGING << "The number of self-play games must be greater than parallel games. New value is "
                    << max_games_ << "." << std::endl;
    }
    if (max_games_ % 25 != 0) {
        max_games_ = FancyCeil(max_games_, 25);
        LOGGING << "The number of self-play games must be divided by 25. New value is "
                    << max_games_ << "." << std::endl;
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
                constexpr int kMainThreadIdx = 0;
                constexpr int kGamesPerChunk = 25;
                bool should_halt = false;
                auto sgf_filename = ConcatPath(
                                        sgf_directory_, filename_hash_ + ".sgf");
                running_threads_.fetch_add(1, std::memory_order_relaxed);

                while (accumulation_games_.fetch_add(1) < max_games_) {
                    if (g == kMainThreadIdx && !should_halt) {
                        if (engine_.ShouldHalt()) {
                            int accm_games = std::max(
                                engine_.GetParallelGames(),
                                accumulation_games_.load(std::memory_order_relaxed));
                            // Assume no data racing so we do not lock max_games_.
                            max_games_ = std::min(
                                max_games_,
                                FancyCeil(accm_games + kGamesPerChunk, kGamesPerChunk));
                            should_halt = true;
                            LOGGING << '[' << CurrentDateTime() << ']'
                                        << " Will halt the self-play loop after playing "
                                        << max_games_ << " games." << std::endl;
                        }
                    }

                    auto game_data = std::make_shared<GameTrainingData>();
                    engine_.PrepareGame(g);
                    engine_.Selfplay(g);
                    engine_.GatherTrainingData(*game_data, g);
                    {
                        // Save the current chunk.
                        std::lock_guard<std::mutex> lock(data_mutex_);
                        game_chunk_buffer_.emplace_back(game_data);
                        int num_games_in_buffer = game_chunk_buffer_.size();
                        if (num_games_in_buffer >= kGamesPerChunk &&
                                num_games_in_buffer >= engine_.GetParallelGames()) {
                            std::vector<TrainingData> chunk_buffer;
                            GatherChunkFromBuffer(kGamesPerChunk, chunk_buffer);
                            if (!SaveChunk(num_saved_chunks_, 0.1f, chunk_buffer)) {
                                break;
                            }
                            num_saved_chunks_ += 1;
                        }
                        engine_.SaveSgf(sgf_filename, g);
                    }

                    played_games_.fetch_add(1);
                    auto played_games = played_games_.load(std::memory_order_relaxed);

                    {
                        // Save some verbose.
                        std::lock_guard<std::mutex> lock(log_mutex_);
                        if (played_games % 10 == 0) {
                            LOGGING << '[' << CurrentDateTime() << ']'
                                        << " Played " << played_games << " games." << std::endl;
                        }
                        if (played_games % kGamesPerChunk == 0) {
                            // Not a precision value but the final accumulation value is
                            // correct.
                            SaveNetQueries(
                                played_games, engine_.GetNetReportQueries());
                        }
                    }
                    
                }
                running_threads_.fetch_sub(1, std::memory_order_relaxed);

                if (g == kMainThreadIdx) {
                    while (running_threads_.load(std::memory_order_relaxed) != 0) {
                        std::this_thread::yield();
                    }

                    // Save remaining chunks.
                    std::vector<TrainingData> chunk_buffer;
                    while(GatherChunkFromBuffer(kGamesPerChunk, chunk_buffer)) {
                        if (!SaveChunk(num_saved_chunks_, 0.1f, chunk_buffer)) {
                            break;
                        }
                        num_saved_chunks_ += 1;
                    }

                    // Still possible remain some chunks in the buffer because we do not
                    // lock variable, max_games_. We may play too many games. Simply discard
                    // them be sure every chunk has 25 games training data.
                    if (!game_chunk_buffer_.empty()) {
                        LOGGING << '[' << CurrentDateTime() << ']'
                                    << " Discard "
                                    << game_chunk_buffer_.size()
                                    << " games training data." << std::endl;
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
