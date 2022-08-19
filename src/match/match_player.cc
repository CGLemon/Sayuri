#include "match/match_player.h"
#include "utils/random.h"
#include "utils/log.h"
#include "config.h"

#include <sstream>

void PlayersPool::Initialize() {
    auto match_weights = GetOption<std::string>("match_weights");
    auto iss = std::istringstream{match_weights};
    auto weights_name = std::string{};

    while (iss >> weights_name) {

        LOGGING << "assigned " << weights_name << '\n';
 
        int idx = players_.size();

        players_.emplace_back(std::make_unique<Player>());
        Player * player = players_[idx].get();

        player->elo = 1000;
        player->name = weights_name;
    }
}

bool PlayersPool::RandomMatch(PlayersPool::Player **a, PlayersPool::Player **b) {
    if (players_.empty()) {
        *a = nullptr;
        *b = nullptr;
        return false;
    }

    int rand0 = Random<kXoroShiro128Plus>::Get().Generate() % players_.size();
    int rand1 = Random<kXoroShiro128Plus>::Get().Generate() % players_.size();
    while (rand1 == rand0 && players_.size() != 1) {
        rand1 = Random<kXoroShiro128Plus>::Get().Generate() % players_.size();
    }

    *a = players_[rand0].get();
    *b = players_[rand1].get();

    return true;
}
