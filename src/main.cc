#include <iostream>
#include <memory>

#include "include.h"

static void StartGtpLoop() {
    auto gtp_loop = std::make_shared<GtpLoop>();
}

int main(int argc, char **argv) {
    ArgsParser(argc, argv);

    ThreadPool::Get(0);

    StartGtpLoop();

    return 0;
}
