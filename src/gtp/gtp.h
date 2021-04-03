#ifndef GTP_GTP_H_INCLUDE
#define GTP_GTP_H_INCLUDE

#include "game/game_state.h"
#include "game/types.h"
#include "utils/parser.h"

#include <memory>

class GTP {
public:
    class Agent {
    public:
        GameState& GetState() {
            return main_state_;
        }

    private:
        GameState main_state_;
    };

    GTP() {
        agent_ = std::make_unique<Agent>();
        agent_->GetState().Reset(kDefaultBoardSize, kDefaultKomi);
    }

    void Loop();

private:
    std::string GTPSuccess(std::string);
    std::string GTPFail(std::string);

    std::string Execute(CommandParser &parser);

    std::unique_ptr<Agent> agent_{nullptr};
};

#endif
