#include <iostream>
#include <memory>

#include "include.h"

static void StartGtpLoop() {
    auto gtploop = std::make_shared<GtpLoop>();
}

int main(int argc, char **argv) {
    StartGtpLoop();
    return 0;
}
