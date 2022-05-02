#include <iostream>
#include <memory>

#include "include.h"

void DumpLicense() {
    LOGGING
            << "    " << kProgram << " " << kVersion << "  " << "Copyright (C) 2021-2022  Hung Zhe Lin\n"
            << "    This program comes with ABSOLUTELY NO WARRANTY.\n"
            << "    This is free software, and you are welcome to redistribute it\n"
            << "    under certain conditions; see the COPYING file for details.\n"
            ;
}

void StartGtpLoop() {
    auto gtp_loop = std::make_unique<GtpLoop>();
}

void StartSelfplayLoop() {
    auto selfplay_loop = std::make_unique<SelfPlayPipe>();
}

int main(int argc, char **argv) {
    ArgsParser(argc, argv);

    DumpLicense();

    ThreadPool::Get(0);

    if (GetOption<std::string>("mode") == "gtp") {
        StartGtpLoop();
    } else if (GetOption<std::string>("mode") == "selfplay") {
        StartSelfplayLoop();
    }
    return 0;
}
