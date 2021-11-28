#include <iostream>
#include <memory>

#include "include.h"

static void StartGtpLoop() {
    auto gtp_loop = std::make_unique<GtpLoop>();
}

static void StartSelfplayLoop() {
    auto selfplay_loop = std::make_unique<SelfPlayPipe>();
}

int main(int argc, char **argv) {
    ArgsParser(argc, argv);

    ThreadPool::Get(0);

    if (GetOption<std::string>("mode") == "gtp") {
        StartGtpLoop();
    } else if (GetOption<std::string>("mode") == "selfplay") {
        StartSelfplayLoop();
    }
    return 0;
}
