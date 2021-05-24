#include <iostream>
#include <memory>

#include "utils/threadpool.h"
#include "include.h"

static void StartGtpLoop() {
    auto gtploop = std::make_shared<GtpLoop>();
}

int main(int argc, char **argv) {
    ArgsParser(argc, argv);

    ThreadPool::Get(1);

    StartGtpLoop();

    return 0;
}
