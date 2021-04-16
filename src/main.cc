#include <iostream>
#include <memory>

#include "include.h"

static void StartGtpLoop() {
    auto gtploop = std::make_shared<GtpLoop>();
}

int main(int argc, char **argv) {
    ArgsParser(argc, argv);

    StartGtpLoop();

    return 0;
}
