#pragma once

#include "game/game_state.h"
#include "game/types.h"
#include "utils/parser.h"

#include <memory>

class GtpLoop {
public:
    class Agent {
    public:
        GameState& GetState() {
            return main_state_;
        }

    private:
        GameState main_state_;
    };

    GtpLoop() {
        agent_ = std::make_unique<Agent>();
        agent_->GetState().Reset(kDefaultBoardSize, kDefaultKomi);
        Loop();
    }

    ~GtpLoop() {}

private:
    void Loop();

    std::string GTPSuccess(std::string);
    std::string GTPFail(std::string);

    std::string Execute(CommandParser &parser);

    std::unique_ptr<Agent> agent_{nullptr};
};
