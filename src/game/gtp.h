#pragma once

#include <memory>

#include "config.h"
#include "game/agents.h"
#include "game/game_state.h"
#include "game/types.h"
#include "mcts/node.h"
#include "mcts/search.h"
#include "utils/format.h"
#include "utils/splitter.h"
#include "version.h"

class GtpLoop {
public:
    GtpLoop() {
        agent_ = std::make_unique<GptAgent>();

        auto kgs_hint = GetOption<std::string>("kgs_hint");
        if (kgs_hint.empty()) {
            version_verbose_ =
                Format("%s (%s)", GetProgramVersion().c_str(), GetVersionName().c_str());
        } else {
            version_verbose_ = Format("%s (%s). %s",
                                      GetProgramVersion().c_str(),
                                      GetVersionName().c_str(),
                                      kgs_hint.c_str());
        }
        curr_id_ = -1;
        prev_pondering_ = false;

        Loop();
    }

    ~GtpLoop() {}

private:
    void Loop();

    std::string GtpSuccess(std::string);
    std::string GtpFail(std::string);

    AnalysisConfig ParseAnalysisConfig(Splitter& spt, int& color);
    bool ParseOption(Splitter& spt, std::string& rep);
    bool NetBench(Splitter& spt, std::string& rep);

    std::string Execute(Splitter& spt, bool& try_ponder);

    std::unique_ptr<GptAgent> agent_{nullptr};

    int curr_id_;
    bool prev_pondering_;
    std::string version_verbose_;
};
