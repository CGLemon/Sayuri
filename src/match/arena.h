#pragma once

#include "match/match_player.h"
#include "mcts/search.h"
#include "game/game_state.h"

#include <memory>

class Arena {
public:
    Arena();

private:
    struct PlayerWrapper {
        PlayersPool::Player *player;
        Network network;
        std::unique_ptr<Search> search;

        void Apply(GameState &game_state) {
            network.Initialize(player->name);
            search = std::make_unique<Search>(game_state, network);
        }
    };

    void Run();
    void MatchGames(int games);


    PlayersPool pool_;
};
