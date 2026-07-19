#include <cstddef>

#include "game/agents.h"
#include "tests/unitest.h"

UNITEST(Agents, ConstructorCreatesFixedContexts) {
    Agents agents(3);

    EXPECT_EQ(agents.GetNumContexts(), static_cast<std::size_t>(3));
}

UNITEST(Agents, ContextCanBeAccessedByIndex) {
    Agents agents(3);

    EXPECT_NE(&agents.GetContext(0), &agents.GetContext(1));
    EXPECT_NO_THROW(agents.GetContext(2));
    EXPECT_THROW(agents.GetContext(-1));
    EXPECT_THROW(agents.GetContext(3));
}

UNITEST(Agents, ConstructorRejectsNonPositiveCount) {
    EXPECT_THROW(Agents(0));
    EXPECT_THROW(Agents(-1));
}

UNITEST(Agents, ShutdownCanBeCalledRepeatedly) {
    Agents agents(1);

    EXPECT_NO_THROW(agents.Shutdown());
    EXPECT_NO_THROW(agents.Shutdown());
}
