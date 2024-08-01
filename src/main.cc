#include <memory>

#include "include.h"

void DumpLicense() {
    auto name_ver = Format("%s %s (%s)",
                               GetProgramName().c_str(),
                               GetProgramVersion().c_str(),
                               GetVersionName().c_str());
    LOGGING
            << "    " << name_ver << "  " << "Copyright (C) 2021-2024  Hung Tse Lin\n"
            << "    This program comes with ABSOLUTELY NO WARRANTY.\n"
            << "    This is free software, and you are welcome to redistribute it\n"
            << "    under certain conditions; see the COPYING file for details.\n"
            ;
}

void StartGtpLoop() {
    try {
        auto loop = std::make_unique<GtpLoop>();
    } catch (const std::exception& e) {
        LOGGING << Format(
            "Get the exception during the GTP loop. Exception: %s.\n", e.what());
    }
}

void StartSelfplayLoop() {
    try {
        auto loop = std::make_unique<SelfPlayPipe>();
    } catch (const std::exception& e) {
        LOGGING << Format(
            "Get the exception during the self-play loop. Exception: %s.\n", e.what());
    }
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
