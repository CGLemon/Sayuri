#pragma once

#include "selfplay/tournament.h"
#include "selfplay/engine.h"
#include <vector>
#include <memory>

class Tournament {
public:
    void Initialize();

private:
    std::vector<std::unique_ptr<Engine>> engines_;

};