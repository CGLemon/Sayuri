#include <memory>

#include "include.h"

void DumpLicense() {
    auto name_ver = Format("%s %s(%s)",
                               GetProgramName().c_str(),
                               GetProgramVersion().c_str(),
                               GetVersionName().c_str());
    LOGGING
            << "    " << name_ver << "  " << "Copyright (C) 2021-2022  Hung Zhe Lin\n"
            << "    This program comes with ABSOLUTELY NO WARRANTY.\n"
            << "    This is free software, and you are welcome to redistribute it\n"
            << "    under certain conditions; see the COPYING file for details.\n"
            ;
}

void StartGtpLoop() {
    auto loop = std::make_unique<GtpLoop>();
}

void StartSelfplayLoop() {
    auto loop = std::make_unique<SelfPlayPipe>();
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
