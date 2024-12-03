#include "neural/supervised_writer.h"
#include "utils/filesystem.h"
#include "utils/log.h"
#include "utils/option.h"

SupervisedWriter::SupervisedWriter() {
    Initialize();
    Loop();
}

void SupervisedWriter::Initialize() {
    input_sgf_directory_ = GetOption<std::string>("input_sgf_directory");
    sgf_files_ = GetFileList(GetOption<std::string>("input_sgf_directory"));
    if (!network_) {
        network_ = std::make_unique<Network>();
    }
    network_->Initialize(GetOption<std::string>("weights_file"));
}

std::string SupervisedWriter::TryGetSgfFilename() {
    std::lock_guard<std::mutex> lock(sgf_mutex_);
    if (sgf_files_.empty()) {
        return std::string{};
    }

    auto sgf_file = sgf_files_.back();
    sgf_files_.pop_back();
    return ConcatPath(input_sgf_directory_, sgf_file);
}

void SupervisedWriter::AssignWorkers() {
    workers_.emplace_back(
        [this]() -> void {
            while (true) {
                auto sgf_filename = TryGetSgfFilename();
                if (sgf_filename.empty()) {
                    break;
                }
                LOGGING << " Parse the SGF file from "
                            << sgf_filename << std::endl;
            }
        }
    );
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

