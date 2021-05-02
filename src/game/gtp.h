#pragma once

#include "game/game_state.h"
#include "game/types.h"
#include "config.h"
#include "utils/parser.h"
#include "neural/network.h"
#include <memory>

class GtpLoop {
public:
    class Agent {
    public:
        GameState& GetState() {
            return main_state_;
        }

        Network& GetNetwork() {
            return network_;
        }

    private:
        GameState main_state_;
        Network network_;
    };

    GtpLoop() {
        agent_ = std::make_unique<Agent>();
        agent_->GetState().Reset(GetOption<int>("defualt_boardsize"),
                                     GetOption<float>("defualt_komi"));
        agent_->GetNetwork().Initialize(GetOption<std::string>("weights_file"));

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
