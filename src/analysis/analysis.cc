#include "analysis/analysis.h"
#include "utils/simple_json.h"
#include "config.h"

#include <string>
#include <iostream>

Analysis::Analysis() {
    // Close search verbose.
    SetOption("analysis_verbose", false);

    Loop();
}

void Analysis::Loop() {
    auto line = std::string{};
    while (true) {
        Json query;
        auto input = std::string{};
        if (std::getline(std::cin, input)) {
            query.Parse(input);
            if (query.Find("action", Json::kString) &&
                    query["action"].ToString() == "quit") {
                break;
            }
        }
    }
}
