#pragma once

#include "analysis/group.h"

class Analysis {
public:
    Analysis();

private:
    void Loop();

    int parallel_games_;
    Group group_;
};
